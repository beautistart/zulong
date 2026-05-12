# File: zulong/l2/orchestrator_graph.py
"""
外层编排器 (Orchestrator Graph)

Plan-Execute-Reflect 循环，基于 LangGraph StateGraph 实现。
5 个节点: plan → schedule → execute → reflect → synthesize
内层 FC 循环（fc_graph.py）完全不动，7 层安全网原封不动。

设计原则：
- plan / execute 通过 run_fc_loop 调用 LLM（保留全部安全网）
- schedule 是纯代码逻辑（拓扑排序 + 注意力切换 + 检查点）
- reflect 是轻量 LLM 调用（质量评估 → CONTINUE/REDO/REPLAN）
- synthesize 支持分级汇总（大项目按 Tier 分层总结）
- 依赖产出注册到 AttentionWindow（不暴力注入），由 BFS 扩散决定保留

✅ v2.0 升级：编译为 LangGraph StateGraph + Checkpointer + Streaming
"""

import logging
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# ✅ 导入 LangGraph
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.sqlite import SqliteSaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("⚠️ LangGraph 未安装，将使用传统状态机模式")

# 编排器阶段名常量
PHASE_PLAN = "plan"
PHASE_SCHEDULE = "schedule"
PHASE_EXECUTE = "execute"
PHASE_REFLECT = "reflect"
PHASE_SYNTHESIZE = "synthesize"


# ═══════════════════════════════════════════════════════════════
# 编排器状态
# ═══════════════════════════════════════════════════════════════

class OrchestratorState(TypedDict, total=False):
    """外层编排器的全局状态"""
    phase: str                            # 当前阶段
    plan_version: int                     # 规划版本号（replan 时递增）
    replan_count: int                     # 已 replan 次数
    current_subtask_id: Optional[str]     # 当前执行的子任务节点 ID
    completed_results: Dict[str, str]     # node_id → result 摘要（供反思和汇总用）
    messages: List[Dict]                  # 对话消息列表
    vllm_model_id: str                    # 远程模型 ID
    tool_definitions: List[Dict]          # 完整工具定义列表
    user_input_text: str                  # 用户原始输入
    total_fc_turns: int                   # 全局已消耗 FC 步数
    max_total_fc_turns: int               # 全局 FC 步数上限
    subtask_reflection_count: Dict[str, int]  # node_id → 该节点反思次数
    should_terminate: str                 # 终止原因（空字符串表示继续）
    response: Optional[str]              # 最终回复


# ═══════════════════════════════════════════════════════════════
# 工具集过滤
# ═══════════════════════════════════════════════════════════════

# Plan 阶段工具集：只建图不执行
_PLAN_TOOLS = {
    "task_create_plan", "task_add_node", "task_add_dependency",
    "task_remove_node", "task_update_node",
    "task_view_overview", "task_get_detail",
    "recall_memory", "discover_related",
}

# Execute 阶段工具集：执行+读写节点
_EXECUTE_TOOLS = {
    "task_mark_status", "task_get_detail", "task_update_node",
    "task_view_overview",
    "exec_write_file", "exec_run_command",
    "recall_memory", "read_memory_node", "discover_related",
    "navigate_attention", "search_experience",
}


def _filter_tool_definitions(
    all_tools: List[Dict], allowed_names: set
) -> List[Dict]:
    """从完整工具列表中过滤出指定阶段允许的工具"""
    return [
        t for t in all_tools
        if t.get("function", {}).get("name", "") in allowed_names
    ]


# ═══════════════════════════════════════════════════════════════
# 条件路由函数（LangGraph StateGraph 用）
# ═══════════════════════════════════════════════════════════════

def route_after_plan(state: OrchestratorState) -> str:
    """Plan 节点后的路由决策"""
    if state.get("should_terminate"):
        return "end"
    # Plan 完成后总是进入 Schedule
    return "schedule"


def route_after_reflect(state: OrchestratorState) -> str:
    """Reflect 节点后的路由决策
    
    Returns:
        "schedule" - 继续下一个子任务
        "execute" - 重新执行当前子任务 (REDO)
        "plan" - 重新规划 (REPLAN)
        "synthesize" - 所有子任务完成，进入汇总
    """
    phase = state.get("phase", "")
    
    if state.get("should_terminate"):
        return "end"
    
    # 根据 reflect_node 设置的 phase 决定下一步
    if phase == PHASE_SCHEDULE:
        return "schedule"
    elif phase == PHASE_EXECUTE:
        return "execute"
    elif phase == PHASE_PLAN:
        return "plan"
    elif phase == PHASE_SYNTHESIZE:
        return "synthesize"
    else:
        # 默认继续调度
        return "schedule"


def should_enter_synthesize(state: OrchestratorState) -> bool:
    """检查是否应该进入 synthesize 阶段
    
    判断条件：
    1. should_terminate 被设置
    2. 所有叶子节点都已完成
    """
    if state.get("should_terminate"):
        return True
    
    from zulong.tools.task_tools import get_active_task_graph
    tg = get_active_task_graph()
    if tg is None:
        return True
    
    leaves = tg.get_leaf_nodes()
    if not leaves:
        return True
    
    all_done = all(n.status in ("completed", "skipped") for n in leaves)
    return all_done


# ═══════════════════════════════════════════════════════════════
# 5 个编排器节点
# ═══════════════════════════════════════════════════════════════

def plan_node(state: OrchestratorState, engine) -> OrchestratorState:
    """Plan 节点：通过 FC 循环让模型建立/修改任务图

    模型可用工具限定为规划工具集，不允许执行类工具。
    """
    from zulong.l2.unified_fc_runner import run_fc_loop
    from zulong.tools.task_tools import get_active_task_graph

    logger.info("[Orchestrator] >>> 进入 PLAN 阶段")

    # 切换注意力到全局模式
    if engine._attn_window:
        engine._attn_window.set_phase(PHASE_PLAN)

    # 过滤工具集
    plan_tools = _filter_tool_definitions(state["tool_definitions"], _PLAN_TOOLS)

    # 🔥 [Fix-8] 检查是否有活跃任务图谱
    current_tg = get_active_task_graph()
    has_active_graph = current_tg is not None

    # 构建规划提示
    is_replan = state.get("replan_count", 0) > 0
    plan_hint = ""
    if is_replan:
        plan_hint = (
            "\n你正在重新规划（Replan）。请保留已完成的节点，"
            "只修改/增删 pending 状态的节点。"
            "先用 task_view_overview 查看当前状态。\n"
        )
    elif not has_active_graph:
        # 🔥 [Fix-8] 没有活跃图谱时，必须先创建新图谱
        plan_hint = (
            "\n【重要】当前没有活跃任务图谱，请首先调用 task_create_plan 创建新图谱：\n"
            "1. 调用 task_create_plan(title='任务标题') 创建新图谱和根节点\n"
            "2. 然后用 task_add_node 添加子任务节点（parent_id='req'）\n"
            "3. 用 task_add_dependency 建立依赖关系\n"
            "4. 完成后不要执行，只需要完成规划即可\n"
        )
    else:
        plan_hint = (
            "\n请为用户需求创建任务规划：\n"
            "1. 用 task_add_node 添加子任务节点\n"
            "2. 用 task_add_dependency 建立依赖关系\n"
            "3. 完成后不要执行，只需要完成规划即可\n"
        )

    # 注入规划指令到消息
    messages = list(state["messages"])
    messages.append({
        "role": "system",
        "content": f"[编排器-规划阶段] {plan_hint}",
    })

    response, fc_turn = run_fc_loop(
        engine=engine,
        messages=messages,
        tool_definitions=plan_tools,
        vllm_model_id=state["vllm_model_id"],
        force_first_tool=False,
        user_input=state["user_input_text"],
        intent_max_tokens=2048,
    )

    state["total_fc_turns"] = state.get("total_fc_turns", 0) + fc_turn
    state["plan_version"] = state.get("plan_version", 0) + 1
    state["phase"] = PHASE_SCHEDULE

    # 保存检查点
    _save_checkpoint(state, f"plan_v{state['plan_version']}")

    logger.info(
        f"[Orchestrator] PLAN 完成: version={state['plan_version']}, "
        f"fc_turns_used={fc_turn}, total={state['total_fc_turns']}"
    )
    return state


def schedule_node(state: OrchestratorState, engine) -> OrchestratorState:
    """Schedule 节点：纯代码逻辑，拓扑排序找下一个可执行节点

    不调用 LLM，零 token 消耗。
    """
    from zulong.tools.task_tools import get_active_task_graph
    from zulong.l2.task_graph import TaskScheduler

    logger.info("[Orchestrator] >>> 进入 SCHEDULE 阶段")

    tg = get_active_task_graph()
    if tg is None:
        logger.error("[Orchestrator] Schedule: 无活跃任务图")
        state["should_terminate"] = "no_task_graph"
        state["phase"] = PHASE_SYNTHESIZE
        return state

    # 检查全局 FC 步数限制 — 软着陆：存检查点+进度报告+自动继续
    max_turns = state.get("max_total_fc_turns", 100)
    total_used = state.get("total_fc_turns", 0)
    if total_used >= max_turns:
        _save_checkpoint(state, f"budget_pause_{total_used}")
        report = _generate_progress_report(state, tg)
        # 将进度报告存入 state 供前端/日志消费
        if "progress_reports" not in state:
            state["progress_reports"] = []
        state["progress_reports"].append(report)
        logger.warning(
            f"[Orchestrator] 全局 FC 预算已用完 ({total_used}/{max_turns})，"
            f"已存检查点，自动扩展预算继续执行"
        )
        # 自动扩展预算（每次增加 50%）而非强制终止
        extension = max(max_turns // 2, 20)
        state["max_total_fc_turns"] = max_turns + extension
        logger.info(
            f"[Orchestrator] 预算扩展: {max_turns} → {state['max_total_fc_turns']}"
        )

    scheduler = TaskScheduler(tg)
    next_executable = scheduler.get_next_executable()

    if not next_executable:
        # 所有叶节点都已完成 → synthesize
        leaves = tg.get_leaf_nodes()
        all_done = all(n.status in ("completed", "skipped") for n in leaves)
        if all_done:
            logger.info("[Orchestrator] 所有子任务已完成，进入 synthesize")
            state["phase"] = PHASE_SYNTHESIZE
            return state
        else:
            # 有未完成但不可执行的节点（被阻塞）
            blocked = [n for n in leaves if n.status not in ("completed", "skipped")]
            logger.warning(
                f"[Orchestrator] 无可执行节点，{len(blocked)} 个阻塞: "
                f"{[n.id for n in blocked[:5]]}"
            )
            state["should_terminate"] = "all_blocked"
            state["phase"] = PHASE_SYNTHESIZE
            return state

    # P2-16: 选择可执行节点（支持同层并行）
    # 当有多个无依赖的叶节点时，记录全部以支持并行执行
    subtask_id = next_executable[0]
    state["current_subtask_id"] = subtask_id
    state["parallel_subtask_ids"] = next_executable  # 同层可并行节点列表

    # 切换注意力到 FOCUS 模式
    if engine._attn_window:
        engine._attn_window.set_phase(PHASE_EXECUTE, subtask_id=subtask_id)

    state["phase"] = PHASE_EXECUTE

    logger.info(
        f"[Orchestrator] SCHEDULE: 选择子任务 {subtask_id}, "
        f"可执行队列: {next_executable}"
    )
    return state


def execute_node(state: OrchestratorState, engine) -> OrchestratorState:
    """Execute 节点：通过 FC 循环执行单个子任务

    依赖产出注册到 AttentionWindow（不暴力注入），
    由 BFS 激活扩散 + 评分驱逐机制自动决定保留。
    """
    from zulong.l2.unified_fc_runner import run_fc_loop
    from zulong.tools.task_tools import get_active_task_graph

    subtask_id = state.get("current_subtask_id")
    logger.info(f"[Orchestrator] >>> 进入 EXECUTE 阶段: {subtask_id}")

    tg = get_active_task_graph()
    if tg is None or subtask_id is None:
        state["phase"] = PHASE_SYNTHESIZE
        state["should_terminate"] = "no_task_graph"
        return state

    node = tg.get_node(subtask_id)
    if node is None:
        logger.warning(f"[Orchestrator] 子任务 {subtask_id} 不存在，跳过")
        state["phase"] = PHASE_SCHEDULE
        return state

    # ── 注册依赖产出到 AttentionWindow（核心：不暴力注入）──
    if engine._attn_window:
        dep_ids = tg.get_dependencies(subtask_id)
        current_turn = engine._attn_window._current_turn
        for dep_id in dep_ids:
            dep_node = tg.get_node(dep_id)
            if dep_node and dep_node.status == "completed" and dep_node.result:
                dep_msg = {
                    "role": "user",
                    "content": (
                        f"[依赖产出 {dep_node.label}] "
                        f"{dep_node.result[:1500]}"
                    ),
                }
                # 注册为非钉住消息，让 AttentionWindow 评分决定保留
                engine._attn_window.register_message(
                    dep_msg,
                    turn=current_turn,
                    node_id=f"task:{tg.id}/{dep_id}",
                    pinned=False,
                )
                logger.debug(
                    f"[Orchestrator] 注册依赖产出: {dep_id} → AttentionWindow"
                )

    # 过滤工具集
    exec_tools = _filter_tool_definitions(state["tool_definitions"], _EXECUTE_TOOLS)

    # 构建执行提示
    messages = list(state["messages"])
    messages.append({
        "role": "user",
        "content": (
            f"[编排器-执行阶段] 当前子任务: {subtask_id}（{node.label}）\n"
            f"描述: {node.desc}\n"
            f"请执行此子任务。完成后用 task_mark_status 标记为 completed 并填写 result。"
        ),
    })

    # 计算子任务 FC 预算
    subtask_budget = state.get("subtask_fc_budget", 30)
    remaining_global = state.get("max_total_fc_turns", 100) - state.get("total_fc_turns", 0)
    effective_budget = min(subtask_budget, remaining_global)

    response, fc_turn = run_fc_loop(
        engine=engine,
        messages=messages,
        tool_definitions=exec_tools,
        vllm_model_id=state["vllm_model_id"],
        force_first_tool=False,
        user_input=f"执行子任务: {node.label}",
        intent_max_tokens=4096,
    )

    state["total_fc_turns"] = state.get("total_fc_turns", 0) + fc_turn

    # 记录完成结果
    updated_node = tg.get_node(subtask_id)
    if updated_node and updated_node.result:
        if "completed_results" not in state:
            state["completed_results"] = {}
        state["completed_results"][subtask_id] = updated_node.result[:300]

    state["phase"] = PHASE_REFLECT

    # 保存检查点
    _save_checkpoint(state, f"exec_{subtask_id}")

    logger.info(
        f"[Orchestrator] EXECUTE 完成: {subtask_id}, "
        f"fc_turns_used={fc_turn}, total={state['total_fc_turns']}"
    )
    return state


def reflect_node(state: OrchestratorState, engine) -> OrchestratorState:
    """Reflect 节点：轻量 LLM 评估子任务质量

    决策：CONTINUE / REDO / REPLAN
    """
    from zulong.tools.task_tools import get_active_task_graph

    subtask_id = state.get("current_subtask_id")
    logger.info(f"[Orchestrator] >>> 进入 REFLECT 阶段: {subtask_id}")

    # 切换注意力回全局
    if engine._attn_window:
        engine._attn_window.set_phase(PHASE_REFLECT)

    tg = get_active_task_graph()
    if tg is None:
        state["phase"] = PHASE_SYNTHESIZE
        return state

    node = tg.get_node(subtask_id) if subtask_id else None

    # API 错误检测：如果上一轮 FC 因 API 错误终止，跳过 REDO 避免请求风暴
    _last_reason = getattr(engine, '_last_fc_terminate_reason', '')
    if _last_reason in ("api_error", "backup_model"):
        _api_fail_count = state.get("_consecutive_api_failures", 0) + 1
        state["_consecutive_api_failures"] = _api_fail_count
        if _api_fail_count >= 2:
            logger.warning(
                f"[Orchestrator] REFLECT: 连续 {_api_fail_count} 次 API 错误，"
                f"终止编排避免请求风暴"
            )
            state["should_terminate"] = "api_error"
            state["phase"] = PHASE_SYNTHESIZE
            return state
        logger.warning(
            f"[Orchestrator] REFLECT: API 错误 ({_last_reason})，跳过 REDO"
        )
        # 强制跳过当前子任务，进入下一个
        if node and node.status != "completed":
            tg.update_node_status(subtask_id, "skipped", result="API 错误，自动跳过")
        state["phase"] = PHASE_SCHEDULE
        return state
    else:
        # API 成功时重置计数
        state["_consecutive_api_failures"] = 0

    # 初始化反思计数
    if "subtask_reflection_count" not in state:
        state["subtask_reflection_count"] = {}

    # 简单规则反思（不消耗 LLM token，零成本）
    decision = "CONTINUE"

    if node:
        if node.status != "completed":
            # 子任务未完成 → redo
            redo_count = state["subtask_reflection_count"].get(subtask_id, 0)
            max_redo = state.get("max_redo_per_subtask", 5)
            if redo_count < max_redo:
                decision = "REDO"
                state["subtask_reflection_count"][subtask_id] = redo_count + 1
                logger.info(
                    f"[Orchestrator] REFLECT: {subtask_id} 未完成，"
                    f"REDO ({redo_count + 1}/{max_redo})"
                )
            else:
                # redo 达上限 → 强制跳过并继续
                tg.update_node_status(subtask_id, "skipped", result="redo 达上限，自动跳过")
                decision = "CONTINUE"
                logger.warning(
                    f"[Orchestrator] REFLECT: {subtask_id} redo 达上限，强制 CONTINUE"
                )
        elif node.result and len(node.result.strip()) < 20:
            # 结果过短，可能质量不足
            redo_count = state["subtask_reflection_count"].get(subtask_id, 0)
            if redo_count < 1:
                decision = "REDO"
                state["subtask_reflection_count"][subtask_id] = redo_count + 1
                logger.info(
                    f"[Orchestrator] REFLECT: {subtask_id} 结果过短，REDO"
                )

    # 执行决策
    if decision == "CONTINUE":
        state["phase"] = PHASE_SCHEDULE
    elif decision == "REDO":
        state["phase"] = PHASE_EXECUTE
    elif decision == "REPLAN":
        replan_count = state.get("replan_count", 0)
        max_replan = state.get("max_replan_count", 3)
        if replan_count < max_replan:
            state["replan_count"] = replan_count + 1
            state["phase"] = PHASE_PLAN
            logger.info(
                f"[Orchestrator] REFLECT: REPLAN ({replan_count + 1}/{max_replan})"
            )
        else:
            state["phase"] = PHASE_SCHEDULE
            logger.warning("[Orchestrator] REFLECT: replan 达上限，强制 CONTINUE")

    # 保存检查点
    _save_checkpoint(state, f"reflect_{subtask_id}")

    logger.info(f"[Orchestrator] REFLECT 决策: {decision} → {state['phase']}")
    return state


def synthesize_node(state: OrchestratorState, engine) -> OrchestratorState:
    """Synthesize 节点：分级汇总所有子任务产出

    小项目（≤ threshold）：直接 FC 循环汇总
    大项目（> threshold）：按 Tier 分层总结，再汇合
    """
    from zulong.l2.unified_fc_runner import run_fc_loop
    from zulong.tools.task_tools import get_active_task_graph
    from zulong.l2.task_graph import TaskScheduler

    logger.info("[Orchestrator] >>> 进入 SYNTHESIZE 阶段")

    # 切换注意力到全局
    if engine._attn_window:
        engine._attn_window.set_phase(PHASE_SYNTHESIZE)

    tg = get_active_task_graph()
    if tg is None:
        state["response"] = "任务已完成，但无法获取任务图数据。"
        state["should_terminate"] = "done"
        return state

    leaves = tg.get_leaf_nodes()
    completed_leaves = [n for n in leaves if n.status in ("completed", "skipped")]
    threshold = state.get("large_project_threshold", 10)

    if len(completed_leaves) <= threshold:
        # ── 小项目：直接汇总 ──
        result_lines = []
        for n in completed_leaves:
            result_brief = n.result[:300] if n.result else "(无结果)"
            result_lines.append(f"- {n.label}: {result_brief}")
        results_text = "\n".join(result_lines)

        messages = list(state["messages"])
        messages.append({
            "role": "system",
            "content": (
                f"[编排器-汇总阶段] 所有子任务已完成。\n"
                f"用户需求: {state.get('user_input_text', '')}\n\n"
                f"各子任务成果:\n{results_text}\n\n"
                f"请生成完整的最终报告回复给用户。"
            ),
        })

        # 汇总阶段不限制工具
        response, fc_turn = run_fc_loop(
            engine=engine,
            messages=messages,
            tool_definitions=state["tool_definitions"],
            vllm_model_id=state["vllm_model_id"],
            force_first_tool=False,
            user_input=state.get("user_input_text", ""),
            intent_max_tokens=4096,
        )
        state["total_fc_turns"] = state.get("total_fc_turns", 0) + fc_turn
        state["response"] = response

    else:
        # ── 大项目：分级汇总 ──
        scheduler = TaskScheduler(tg)
        tiers = scheduler.compute_execution_tiers()

        tier_summaries = []
        for tier_idx, tier_ids in enumerate(tiers):
            tier_nodes = [tg.get_node(nid) for nid in tier_ids if tg.get_node(nid)]
            tier_content = "\n".join(
                f"- {n.label}: {(n.result[:300] if n.result else '(无结果)')}"
                for n in tier_nodes
                if n.status in ("completed", "skipped")
            )
            if not tier_content:
                continue

            # 每个 Tier 生成简短摘要（通过 FC 循环）
            tier_messages = list(state["messages"])
            tier_messages.append({
                "role": "system",
                "content": (
                    f"请用 2-3 句话总结以下第 {tier_idx + 1} 阶段的工作成果:\n"
                    f"{tier_content}"
                ),
            })

            tier_response, tier_turns = run_fc_loop(
                engine=engine,
                messages=tier_messages,
                tool_definitions=[],  # 汇总不需要工具
                vllm_model_id=state["vllm_model_id"],
                force_first_tool=False,
                user_input="",
                intent_max_tokens=512,
            )
            state["total_fc_turns"] = state.get("total_fc_turns", 0) + tier_turns
            tier_summaries.append(
                f"阶段 {tier_idx + 1}: {tier_response or '(汇总失败)'}"
            )

        # 汇合所有 Tier 摘要
        final_input = "\n".join(tier_summaries)
        messages = list(state["messages"])
        messages.append({
            "role": "system",
            "content": (
                f"[编排器-最终汇总] 用户需求: {state.get('user_input_text', '')}\n\n"
                f"各阶段成果:\n{final_input}\n\n"
                f"请生成结构化的最终报告回复给用户。"
            ),
        })

        response, fc_turn = run_fc_loop(
            engine=engine,
            messages=messages,
            tool_definitions=state["tool_definitions"],
            vllm_model_id=state["vllm_model_id"],
            force_first_tool=False,
            user_input=state.get("user_input_text", ""),
            intent_max_tokens=4096,
        )
        state["total_fc_turns"] = state.get("total_fc_turns", 0) + fc_turn
        state["response"] = response

    state["should_terminate"] = "done"
    logger.info(
        f"[Orchestrator] SYNTHESIZE 完成: total_fc_turns={state['total_fc_turns']}"
    )
    return state


# ═══════════════════════════════════════════════════════════════
# 检查点管理（轻量内联，避免过度抽象）
# ═══════════════════════════════════════════════════════════════

def _save_checkpoint(state: OrchestratorState, label: str):
    """保存编排器状态检查点到磁盘"""
    try:
        from zulong.config.config_manager import get_l2_inference_config
        config = get_l2_inference_config()
        orch_config = config.get("orchestrator", {})
        checkpoint_dir = orch_config.get("checkpoint_dir", "./data/orchestrator_checkpoints")

        import os
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "label": label,
            "timestamp": time.time(),
            "phase": state.get("phase", ""),
            "plan_version": state.get("plan_version", 0),
            "replan_count": state.get("replan_count", 0),
            "current_subtask_id": state.get("current_subtask_id"),
            "total_fc_turns": state.get("total_fc_turns", 0),
            "completed_results": state.get("completed_results", {}),
            "subtask_reflection_count": state.get("subtask_reflection_count", {}),
        }

        filepath = os.path.join(checkpoint_dir, f"{label}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        logger.debug(f"[Orchestrator] 检查点已保存: {filepath}")

    except Exception as e:
        logger.debug(f"[Orchestrator] 检查点保存跳过: {e}")


def _generate_progress_report(state, tg) -> str:
    """生成进度报告：汇总已完成/进行中/待办子任务

    用于预算耗尽或软限制时向用户/LLM 提供可视化进度。

    Args:
        state: OrchestratorState
        tg: TaskGraph 实例

    Returns:
        Markdown 格式的进度报告字符串
    """
    total_turns = state.get("total_fc_turns", 0)
    max_turns = state.get("max_total_fc_turns", 100)
    completed_results = state.get("completed_results", {})

    lines = [
        f"## 进度报告 (FC 步数: {total_turns}/{max_turns})",
        "",
    ]

    if tg is not None:
        leaves = tg.get_leaf_nodes()
        done = [n for n in leaves if n.status in ("completed", "skipped")]
        wip = [n for n in leaves if n.status == "in_progress"]
        todo = [n for n in leaves if n.status in ("pending", "not_started")]
        blocked = [n for n in leaves if n.status == "blocked"]

        lines.append(f"**完成**: {len(done)}/{len(leaves)} 个子任务")
        if done:
            for n in done:
                brief = (n.result or "")[:80]
                lines.append(f"  - [x] {n.label}: {brief}")
        if wip:
            lines.append(f"**进行中**: {len(wip)}")
            for n in wip:
                lines.append(f"  - [ ] {n.label}")
        if todo:
            lines.append(f"**待办**: {len(todo)}")
            for n in todo:
                lines.append(f"  - [ ] {n.label}")
        if blocked:
            lines.append(f"**阻塞**: {len(blocked)}")
            for n in blocked:
                lines.append(f"  - [!] {n.label}")
    else:
        lines.append("(无活跃任务图)")

    report = "\n".join(lines)
    logger.info("[Orchestrator] 进度报告已生成:\n%s", report)
    return report


def load_latest_checkpoint() -> Optional[Dict]:
    """加载最近的检查点（供 RESUME 场景使用）"""
    try:
        from zulong.config.config_manager import get_l2_inference_config
        config = get_l2_inference_config()
        orch_config = config.get("orchestrator", {})
        checkpoint_dir = orch_config.get("checkpoint_dir", "./data/orchestrator_checkpoints")

        if not os.path.isdir(checkpoint_dir):
            return None

        files = [
            f for f in os.listdir(checkpoint_dir) if f.endswith(".json")
        ]
        if not files:
            return None

        # 按修改时间排序，取最新
        files.sort(
            key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)),
            reverse=True,
        )
        filepath = os.path.join(checkpoint_dir, files[0])
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        logger.debug(f"[Orchestrator] 加载检查点失败: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# LangGraph StateGraph 构建（✅ v2.0 新增）
# ═══════════════════════════════════════════════════════════════

def build_orchestrator_graph(engine) -> Any:
    """构建外层编排器 LangGraph StateGraph
    
    Args:
        engine: InferenceEngine 实例
        
    Returns:
        CompiledStateGraph (如果 LangGraph 可用)
        None (如果 LangGraph 不可用)
    """
    if not LANGGRAPH_AVAILABLE:
        logger.warning("[Orchestrator] LangGraph 不可用，无法构建 StateGraph")
        return None
    
    try:
        workflow = StateGraph(OrchestratorState)
        
        # 添加 5 个阶段节点
        workflow.add_node("plan", lambda s: plan_node(s, engine))
        workflow.add_node("schedule", lambda s: schedule_node(s, engine))
        workflow.add_node("execute", lambda s: execute_node(s, engine))
        workflow.add_node("reflect", lambda s: reflect_node(s, engine))
        workflow.add_node("synthesize", lambda s: synthesize_node(s, engine))
        
        # 设置入口点
        workflow.set_entry_point("plan")
        
        # 条件路由：plan → schedule 或 end
        workflow.add_conditional_edges(
            "plan",
            route_after_plan,
            {"schedule": "schedule", "end": END}
        )
        
        # 固定边：schedule → execute
        workflow.add_edge("schedule", "execute")
        
        # 条件路由：reflect → schedule/execute/plan/synthesize
        workflow.add_conditional_edges(
            "reflect",
            route_after_reflect,
            {
                "schedule": "schedule",
                "execute": "execute",
                "plan": "plan",
                "synthesize": "synthesize",
                "end": END,
            }
        )
        
        # synthesize → end
        workflow.add_edge("synthesize", END)
        
        # ✅ 配置 Checkpointer（如果启用）
        from zulong.config.config_manager import get_l2_inference_config
        config = get_l2_inference_config()
        orch_config = config.get("orchestrator", {})
        use_checkpointer = orch_config.get("use_langgraph_checkpointer", False)
        
        if use_checkpointer:
            db_path = orch_config.get(
                "checkpointer_db_path",
                "./data/orchestrator_checkpoints.db"
            )
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            checkpointer = SqliteSaver.from_conn_string(db_path)
            logger.info(f"[Orchestrator] ✅ Checkpointer 已启用: {db_path}")
            compiled = workflow.compile(checkpointer=checkpointer)
        else:
            logger.info("[Orchestrator] ⚠️ Checkpointer 未启用，使用内存模式")
            compiled = workflow.compile()
        
        logger.info("[Orchestrator] ✅ LangGraph StateGraph 构建成功")
        return compiled
        
    except Exception as e:
        logger.error(f"[Orchestrator] LangGraph StateGraph 构建失败: {e}", exc_info=True)
        return None


class OrchestratorWithLangGraph:
    """✅ v2.0 新增：支持 LangGraph 的编排器包装类
    
    提供断点续传、流式输出等高级功能
    """
    
    def __init__(self, engine):
        self.engine = engine
        self.graph = build_orchestrator_graph(engine)
        self.use_langgraph = LANGGRAPH_AVAILABLE and self.graph is not None
        
        if self.use_langgraph:
            logger.info("[OrchestratorWithLangGraph] ✅ LangGraph 模式已启用")
        else:
            logger.info("[OrchestratorWithLangGraph] ⚠️ 降级为传统状态机模式")
    
    async def run(
        self,
        user_input: str,
        messages: List[Dict],
        tool_definitions: List[Dict],
        vllm_model_id: str,
        thread_id: str = None,
        is_resume: bool = False,
    ) -> Tuple[Optional[str], int, str]:
        """运行编排器，支持断点续传
        
        Args:
            user_input: 用户原始输入
            messages: 初始对话消息列表
            tool_definitions: 完整工具定义列表
            vllm_model_id: 远程模型 ID
            thread_id: 线程ID（用于断点续传）
            is_resume: 是否为任务恢复
            
        Returns:
            (response, total_fc_turns, thread_id)
        """
        if not self.use_langgraph:
            # 降级为传统模式
            logger.warning("[OrchestratorWithLangGraph] 降级为传统状态机模式")
            response, total_turns = run_orchestrator(
                engine=self.engine,
                messages=messages,
                tool_definitions=tool_definitions,
                vllm_model_id=vllm_model_id,
                user_input=user_input,
                is_resume=is_resume,
            )
            return response, total_turns, ""
        
        # ✅ LangGraph 模式
        import uuid
        if not thread_id:
            thread_id = f"task_{uuid.uuid4().hex[:12]}"
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # 初始化状态
        from zulong.config.config_manager import get_l2_inference_config
        config_mgr = get_l2_inference_config()
        orch_config = config_mgr.get("orchestrator", {})
        
        initial_state: OrchestratorState = {
            "phase": PHASE_SCHEDULE if is_resume else PHASE_PLAN,
            "plan_version": 1 if is_resume else 0,
            "replan_count": 0,
            "current_subtask_id": None,
            "completed_results": {},
            "messages": messages,
            "vllm_model_id": vllm_model_id,
            "tool_definitions": tool_definitions,
            "user_input_text": user_input,
            "total_fc_turns": 0,
            "max_total_fc_turns": orch_config.get("max_total_fc_turns", 100),
            "subtask_reflection_count": {},
            "should_terminate": "",
            "response": None,
            "max_redo_per_subtask": orch_config.get("max_redo_per_subtask", 3),
            "max_replan_count": orch_config.get("max_replan_count", 2),
            "subtask_fc_budget": orch_config.get("subtask_fc_budget", 30),
            "large_project_threshold": orch_config.get("large_project_threshold", 10),
        }
        
        logger.info(
            f"[OrchestratorWithLangGraph] 启动 LangGraph 编排器: "
            f"thread_id={thread_id}, is_resume={is_resume}"
        )
        
        # ✅ 执行图（自动保存检查点）
        final_state = await self.graph.ainvoke(initial_state, config=config)
        
        response = final_state.get("response")
        total_turns = final_state.get("total_fc_turns", 0)
        
        logger.info(
            f"[OrchestratorWithLangGraph] 编排器完成: "
            f"thread_id={thread_id}, total_fc_turns={total_turns}"
        )
        
        return response, total_turns, thread_id
    
    async def resume(self, thread_id: str) -> Tuple[Optional[str], int, str]:
        """从检查点恢复执行
        
        Args:
            thread_id: 之前保存的线程ID
            
        Returns:
            (response, total_fc_turns, thread_id)
        """
        if not self.use_langgraph:
            logger.error("[OrchestratorWithLangGraph] LangGraph 未启用，无法恢复")
            return None, 0, ""
        
        config = {"configurable": {"thread_id": thread_id}}
        
        logger.info(f"[OrchestratorWithLangGraph] 从检查点恢复: thread_id={thread_id}")
        
        # ✅ LangGraph 自动从最后一个检查点恢复
        final_state = await self.graph.ainvoke(None, config=config)
        
        response = final_state.get("response")
        total_turns = final_state.get("total_fc_turns", 0)
        
        return response, total_turns, thread_id
    
    async def stream_run(
        self,
        user_input: str,
        messages: List[Dict],
        tool_definitions: List[Dict],
        vllm_model_id: str,
        thread_id: str = None,
    ):
        """流式运行编排器，实时推送进度
        
        Yields:
            dict: {"node": node_name, "update": update_data}
        """
        if not self.use_langgraph:
            logger.warning("[OrchestratorWithLangGraph] LangGraph 未启用，无法流式输出")
            return
        
        import uuid
        if not thread_id:
            thread_id = f"task_{uuid.uuid4().hex[:12]}"
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # 初始化状态（同 run 方法）
        from zulong.config.config_manager import get_l2_inference_config
        config_mgr = get_l2_inference_config()
        orch_config = config_mgr.get("orchestrator", {})
        
        initial_state: OrchestratorState = {
            "phase": PHASE_PLAN,
            "plan_version": 0,
            "replan_count": 0,
            "current_subtask_id": None,
            "completed_results": {},
            "messages": messages,
            "vllm_model_id": vllm_model_id,
            "tool_definitions": tool_definitions,
            "user_input_text": user_input,
            "total_fc_turns": 0,
            "max_total_fc_turns": orch_config.get("max_total_fc_turns", 100),
            "subtask_reflection_count": {},
            "should_terminate": "",
            "response": None,
            "max_redo_per_subtask": orch_config.get("max_redo_per_subtask", 3),
            "max_replan_count": orch_config.get("max_replan_count", 2),
            "subtask_fc_budget": orch_config.get("subtask_fc_budget", 30),
            "large_project_threshold": orch_config.get("large_project_threshold", 10),
        }
        
        logger.info(f"[OrchestratorWithLangGraph] 开始流式执行: thread_id={thread_id}")
        
        # ✅ 使用 stream_mode="updates" 获取每个节点的输出
        async for event in self.graph.astream(
            initial_state,
            config=config,
            stream_mode="updates"
        ):
            # event格式: {"node_name": {"field": "value"}}
            node_name, update = list(event.items())[0]
            
            yield {
                "type": "progress",
                "node": node_name,
                "update": update,
                "timestamp": time.time(),
                "thread_id": thread_id,
            }
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """列出所有可用的检查点
        
        Returns:
            检查点列表，每个检查点包含 thread_id、创建时间等信息
        """
        if not self.use_langgraph:
            logger.warning("[OrchestratorWithLangGraph] LangGraph 未启用，无法列出检查点")
            return []
        
        try:
            # 获取 checkpointer 实例
            if not hasattr(self.graph, 'checkpointer') or self.graph.checkpointer is None:
                logger.warning("[OrchestratorWithLangGraph] Checkpointer 未配置")
                return []
            
            checkpointer = self.graph.checkpointer
            
            # 列出所有线程的检查点
            checkpoints = []
            
            # SqliteSaver 使用 SQLite 数据库存储检查点
            # 我们需要直接查询数据库
            import sqlite3
            from zulong.config.config_manager import get_l2_inference_config
            config = get_l2_inference_config()
            orch_config = config.get("orchestrator", {})
            db_path = orch_config.get("checkpointer_db_path", "./data/orchestrator_checkpoints.db")
            
            if not os.path.exists(db_path):
                logger.info(f"[OrchestratorWithLangGraph] 检查点数据库不存在: {db_path}")
                return []
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 查询所有不同的 thread_id
            cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
            rows = cursor.fetchall()
            
            for (thread_id,) in rows:
                # 获取每个线程的最新检查点
                cursor.execute(
                    "SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint "
                    "FROM checkpoints "
                    "WHERE thread_id = ? "
                    "ORDER BY checkpoint_id DESC "
                    "LIMIT 1",
                    (thread_id,)
                )
                checkpoint_row = cursor.fetchone()
                
                if checkpoint_row:
                    checkpoint_id, parent_id, checkpoint_type, checkpoint_data = checkpoint_row
                    
                    # 解析检查点数据
                    try:
                        checkpoint_info = {
                            "thread_id": thread_id,
                            "checkpoint_id": checkpoint_id,
                            "parent_checkpoint_id": parent_id,
                            "type": checkpoint_type,
                            "created_at": None,  # SqliteSaver 不直接存储时间戳
                        }
                        
                        # 尝试从 checkpoint_data 中提取更多信息
                        if checkpoint_data:
                            import pickle
                            try:
                                state = pickle.loads(checkpoint_data)
                                # 提取状态中的关键信息
                                if isinstance(state, dict):
                                    channel_values = state.get('channel_values', {})
                                    checkpoint_info['phase'] = channel_values.get('phase', 'unknown')
                                    checkpoint_info['plan_version'] = channel_values.get('plan_version', 0)
                                    checkpoint_info['replan_count'] = channel_values.get('replan_count', 0)
                            except Exception as e:
                                logger.debug(f"[OrchestratorWithLangGraph] 解析检查点数据失败: {e}")
                        
                        checkpoints.append(checkpoint_info)
                    except Exception as e:
                        logger.error(f"[OrchestratorWithLangGraph] 处理检查点失败: {e}")
            
            conn.close()
            
            logger.info(f"[OrchestratorWithLangGraph] 找到 {len(checkpoints)} 个检查点")
            return checkpoints
            
        except Exception as e:
            logger.error(f"[OrchestratorWithLangGraph] 列出检查点失败: {e}", exc_info=True)
            return []
    
    async def restore(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Tuple[Optional[str], int, str]:
        """恢复到指定的检查点
        
        Args:
            thread_id: 线程ID
            checkpoint_id: 检查点ID（可选，如果不指定则恢复到最新的检查点）
            
        Returns:
            (response, total_fc_turns, thread_id)
        """
        if not self.use_langgraph:
            logger.error("[OrchestratorWithLangGraph] LangGraph 未启用，无法恢复")
            return None, 0, ""
        
        config = {"configurable": {"thread_id": thread_id}}
        
        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id
            logger.info(
                f"[OrchestratorWithLangGraph] 恢复到指定检查点: "
                f"thread_id={thread_id}, checkpoint_id={checkpoint_id}"
            )
        else:
            logger.info(
                f"[OrchestratorWithLangGraph] 恢复到最新检查点: thread_id={thread_id}"
            )
        
        try:
            # ✅ LangGraph 自动从指定检查点恢复
            final_state = await self.graph.ainvoke(None, config=config)
            
            response = final_state.get("response")
            total_turns = final_state.get("total_fc_turns", 0)
            
            logger.info(
                f"[OrchestratorWithLangGraph] 恢复完成: "
                f"thread_id={thread_id}, total_fc_turns={total_turns}"
            )
            
            return response, total_turns, thread_id
            
        except Exception as e:
            logger.error(
                f"[OrchestratorWithLangGraph] 恢复失败: thread_id={thread_id}, error={e}",
                exc_info=True
            )
            return None, 0, thread_id


# ═══════════════════════════════════════════════════════════════
# 编排器入口
# ═══════════════════════════════════════════════════════════════

def run_orchestrator(
    engine,
    messages: List[Dict],
    tool_definitions: List[Dict],
    vllm_model_id: str,
    user_input: str = "",
    is_resume: bool = False,
) -> Tuple[Optional[str], int]:
    """执行编排器循环，返回 (response, total_fc_turns)

    Args:
        engine: InferenceEngine 实例
        messages: 初始对话消息列表
        tool_definitions: 完整工具定义列表
        vllm_model_id: 远程模型 ID
        user_input: 用户原始输入
        is_resume: 是否为任务恢复

    Returns:
        (response, total_fc_turns)
    """
    from zulong.config.config_manager import get_l2_inference_config
    config = get_l2_inference_config()
    orch_config = config.get("orchestrator", {})

    # 初始化状态
    state: OrchestratorState = {
        "phase": PHASE_PLAN,
        "plan_version": 0,
        "replan_count": 0,
        "current_subtask_id": None,
        "completed_results": {},
        "messages": messages,
        "vllm_model_id": vllm_model_id,
        "tool_definitions": tool_definitions,
        "user_input_text": user_input,
        "total_fc_turns": 0,
        "max_total_fc_turns": orch_config.get("max_total_fc_turns", 100),
        "subtask_reflection_count": {},
        "should_terminate": "",
        "response": None,
        "max_redo_per_subtask": orch_config.get("max_redo_per_subtask", 3),
        "max_replan_count": orch_config.get("max_replan_count", 2),
        "subtask_fc_budget": orch_config.get("subtask_fc_budget", 30),
        "large_project_threshold": orch_config.get("large_project_threshold", 10),
    }

    # RESUME 场景：跳过 plan，直接进入 schedule
    if is_resume:
        state["phase"] = PHASE_SCHEDULE
        state["plan_version"] = 1  # 已有规划

    logger.info(
        f"[Orchestrator] 启动编排器: phase={state['phase']}, "
        f"is_resume={is_resume}, max_fc_turns={state['max_total_fc_turns']}"
    )

    # 状态机循环（安全上限防止无限循环）
    max_iterations = 200
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        phase = state["phase"]

        if state.get("should_terminate"):
            break

        if phase == PHASE_PLAN:
            state = plan_node(state, engine)
        elif phase == PHASE_SCHEDULE:
            state = schedule_node(state, engine)
        elif phase == PHASE_EXECUTE:
            state = execute_node(state, engine)
        elif phase == PHASE_REFLECT:
            state = reflect_node(state, engine)
        elif phase == PHASE_SYNTHESIZE:
            state = synthesize_node(state, engine)
            break
        else:
            logger.error(f"[Orchestrator] 未知阶段: {phase}")
            break

    if iteration >= max_iterations:
        logger.error("[Orchestrator] 达到最大迭代次数，强制终止")

    response = state.get("response")
    total_turns = state.get("total_fc_turns", 0)

    logger.info(
        f"[Orchestrator] 编排器完成: iterations={iteration}, "
        f"total_fc_turns={total_turns}, terminate={state.get('should_terminate')}"
    )

    return (response, total_turns)
