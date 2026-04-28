"""
FC Loop 的 LangGraph StateGraph 实现

将 inference_engine.py 中的 while FC 循环替换为 4 节点有向图：
  check → call_model → exec_tools / eval_response → (循环回 check 或 END)

步数作为主要收敛控制，时间信号降级为 Circuit Breaker 的辅助信号。
"""

import concurrent.futures
import json as _json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, TypedDict, TYPE_CHECKING

from langgraph.graph import StateGraph, END

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from zulong.l2.inference_engine import InferenceEngine

from zulong.l2.circuit_breaker import CircuitBreakerState
from zulong.l2.attention_window import MAX_TOOL_RESULT_CHARS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. FCLoopState — 图节点间传递的状态袋
# ---------------------------------------------------------------------------

class FCLoopState(TypedDict, total=False):
    """FC Loop 的图状态（TypedDict 确保 LangGraph 正确合并部分更新）。

    total=False 使所有字段可选，节点只需返回要更新的字段，
    未返回的字段由 LangGraph 自动保留上一轮的值。
    messages 列表通过引用共享，节点内直接 append 即可。
    """
    messages: List[Dict]
    fc_turn: int
    response: Optional[str]
    tool_results_buffer: List[Dict]
    cb_force_no_tools: bool
    gap_continue_count: int
    should_terminate: str
    tool_calls_data: Optional[List[Dict]]
    response_content: Optional[str]
    force_first_tool: bool
    vllm_model_id: str
    tool_definitions: List[Dict]
    user_input_text: str
    is_resume: bool
    resume_automark_count: int
    null_response_count: int  # 连续 response=None 的拦截次数
    api_timeout_count: int  # API 连续超时次数


# ---------------------------------------------------------------------------
# 2. 节点工厂函数（闭包捕获 engine 实例）
# ---------------------------------------------------------------------------

def _make_check_node(engine: "InferenceEngine"):
    """节点 1: 前置检查与步数递增"""

    def check_node(state: dict) -> dict:
        fc_turn = state["fc_turn"] + 1
        state["fc_turn"] = fc_turn

        # 进度监控
        if fc_turn % engine._warning_interval == 0:
            logger.info(
                f"[FC][Graph] 进度: {fc_turn}/{engine._max_fc_turns} 步，"
                f"已执行 {len(state['tool_results_buffer'])} 次工具调用"
            )

        if fc_turn > engine._soft_limit:
            logger.warning(f"[FC][Graph] ⚠️ 已超过软限制 ({engine._soft_limit} 步)，继续执行...")

        # 硬限制检查
        if fc_turn >= engine._hard_limit:
            logger.error(f"[FC][Graph] 🚨 达到硬限制 ({engine._hard_limit} 步)，强制终止")
            return {"fc_turn": fc_turn, "should_terminate": "hard_limit"}

        # 中断信号检查
        with engine._lock:
            interrupted = engine._interrupt_flag
        if interrupted:
            logger.info(f"[FC][Graph] Turn {fc_turn}: 检测到中断信号，终止 FC 循环")
            # 保留最后一个有效回复而非空字符串
            if not state.get("response"):
                # 从消息历史中提取最后的 assistant 回复
                last_reply = ""
                for msg in reversed(state.get("messages", [])):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if content and len(content) > 10:
                            last_reply = content
                            break
                state["response"] = last_reply
            return {"fc_turn": fc_turn, "should_terminate": "interrupt"}

        return {"fc_turn": fc_turn, "should_terminate": ""}

    return check_node


def _make_call_model_node(engine: "InferenceEngine"):
    """节点 2: LLM API 调用"""

    def call_model_node(state: dict) -> dict:
        fc_turn = state["fc_turn"]
        messages = state["messages"]
        cb_force_no_tools = state.get("cb_force_no_tools", False)
        tool_definitions = state["tool_definitions"]
        vllm_model_id = state["vllm_model_id"]
        force_first_tool = state.get("force_first_tool", False)

        # 构建 API 调用参数（使用注意力窗口裁剪后的消息）
        windowed_messages = (
            engine._attn_window.apply_window()
            if engine._attn_window else messages
        )
        api_kwargs: Dict[str, Any] = {
            "model": vllm_model_id,
            "messages": windowed_messages,
            "max_tokens": 1024,
            "temperature": 0.3,
            "top_p": 0.85,
            "stream": False,
            **engine._get_llm_extra_kwargs(),
        }

        # 传入工具定义
        if cb_force_no_tools:
            logger.info("[FC][Graph][CB] Circuit Breaker RED: 强制文本回复，移除所有工具定义")
        elif tool_definitions:
            api_kwargs["tools"] = tool_definitions
            if force_first_tool and fc_turn == 1:
                api_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": "task_view_overview"},
                }
                logger.info("[FC][Graph] RESUME 第一轮：强制调用 task_view_overview")
            else:
                api_kwargs["tool_choice"] = "auto"

        # API 调用（含超时）
        def _call(kwargs=api_kwargs):
            return engine.vllm_client.chat.completions.create(**kwargs)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_call)
            api_response = future.result(timeout=engine._fc_loop_timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(
                f"⚠️ [FC][Graph] Turn {fc_turn} 超时 (>{engine._fc_loop_timeout}s)，继续尝试..."
            )
            api_timeout_count = state.get("api_timeout_count", 0) + 1
            _MAX_API_TIMEOUTS = 2

            if fc_turn >= engine._hard_limit:
                logger.error("[FC][Graph] 🚨 超时且达到硬限制，使用降级回复")
                return {
                    "response": engine._get_fallback_response(state.get("user_input_text", "")),
                    "should_terminate": "api_error",
                    "tool_calls_data": None,
                    "response_content": None,
                    "api_timeout_count": api_timeout_count,
                }
            if api_timeout_count >= _MAX_API_TIMEOUTS:
                logger.warning(
                    f"[FC][Graph] 连续 {api_timeout_count} 次超时，尝试备用模型"
                )
                try:
                    from zulong.models.container import LLM_MODEL_ID_BACKUP
                    if engine.backup_client and LLM_MODEL_ID_BACKUP:
                        backup_resp = engine.backup_client.chat.completions.create(
                            model=LLM_MODEL_ID_BACKUP,
                            messages=messages,
                            max_tokens=1024,
                            temperature=0.3,
                            stream=False,
                            **engine._get_llm_extra_kwargs(),
                        )
                        return {
                            "response": backup_resp.choices[0].message.content or "",
                            "should_terminate": "backup_model",
                            "tool_calls_data": None,
                            "response_content": None,
                            "api_timeout_count": api_timeout_count,
                        }
                except Exception as backup_err:
                    logger.warning(f"[FC][Graph] 备用模型也失败: {backup_err}")
                return {
                    "response": engine._get_fallback_response(state.get("user_input_text", "")),
                    "should_terminate": "api_error",
                    "tool_calls_data": None,
                    "response_content": None,
                    "api_timeout_count": api_timeout_count,
                }
            # 超时重试 → 路由回 check
            return {
                "tool_calls_data": None,
                "response_content": None,
                "should_terminate": "",
                "api_timeout_count": api_timeout_count,
            }
        except Exception as api_err:
            logger.error(f"🚨 [FC][Graph] Turn {fc_turn} API 调用失败: {api_err}")
            # 尝试备用模型
            try:
                from zulong.models.container import LLM_MODEL_ID_BACKUP
                if engine.backup_client and LLM_MODEL_ID_BACKUP:
                    logger.info(f"🔄 [FC][Graph] 切换备用模型: {LLM_MODEL_ID_BACKUP}")
                    backup_resp = engine.backup_client.chat.completions.create(
                        model=LLM_MODEL_ID_BACKUP,
                        messages=messages,
                        max_tokens=1024,
                        temperature=0.3,
                        stream=False,
                        **engine._get_llm_extra_kwargs(),
                    )
                    return {
                        "response": backup_resp.choices[0].message.content or "",
                        "should_terminate": "api_error",
                        "tool_calls_data": None,
                        "response_content": None,
                    }
                else:
                    return {
                        "response": engine._get_fallback_response(state.get("user_input_text", "")),
                        "should_terminate": "api_error",
                        "tool_calls_data": None,
                        "response_content": None,
                    }
            except Exception as backup_err:
                logger.warning(f"🚨 [FC][Graph] 备用模型也失败: {backup_err}")
                return {
                    "response": engine._get_fallback_response(state.get("user_input_text", "")),
                    "should_terminate": "api_error",
                    "tool_calls_data": None,
                    "response_content": None,
                }
        finally:
            executor.shutdown(wait=False)

        choice = api_response.choices[0]
        msg = choice.message

        # 拆解 API 返回（避免将 OpenAI 对象存入 state）
        tool_calls_data = None
        response_content = msg.content or ""

        if msg.tool_calls:
            logger.info(
                f"[FC][Graph] Turn {fc_turn}: 模型请求 {len(msg.tool_calls)} 个工具调用"
            )
            tool_calls_data = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        else:
            logger.info(
                f"[FC][Graph] Turn {fc_turn}: 模型直接回复，长度 {len(response_content)}"
            )

        return {
            "tool_calls_data": tool_calls_data,
            "response_content": response_content,
            "should_terminate": "",
        }

    return call_model_node


def _make_exec_tools_node(engine: "InferenceEngine"):
    """节点 3: 工具执行"""

    def exec_tools_node(state: dict) -> dict:
        fc_turn = state["fc_turn"]
        messages = state["messages"]
        tool_calls_data = state["tool_calls_data"]
        tool_results_buffer = state["tool_results_buffer"]
        response_content = state.get("response_content", "")

        # 将 assistant 消息（含 tool_calls）追加到上下文
        assistant_msg = {
            "role": "assistant",
            "content": response_content or "",
            "tool_calls": tool_calls_data,
        }
        grp = engine._attn_window.new_tool_group() if engine._attn_window else None
        messages.append(assistant_msg)
        if engine._attn_window:
            engine._attn_window.register_message(
                assistant_msg, turn=fc_turn, group_id=grp,
            )

        # 执行每个工具调用
        for tc_data in tool_calls_data:
            # 中断检查
            with engine._lock:
                interrupted = engine._interrupt_flag
            if interrupted:
                logger.info("[FC][Graph] 工具执行中检测到中断，跳过剩余工具")
                break

            tool_name = tc_data["function"]["name"]
            tool_args = {}
            try:
                tool_args = _json.loads(tc_data["function"]["arguments"] or "{}")
            except Exception:
                pass

            # 注意力窗口：观察工具调用
            if engine._attn_window:
                engine._attn_window.observe_tool_call(tool_name, tool_args)
                if tool_name == "navigate_attention":
                    engine._attn_window.on_navigate_attention(
                        direction=tool_args.get("direction", ""),
                        target_node_id=tool_args.get("target_node_id"),
                    )

            # 构造类 tool_call 对象供 _execute_tool_call 使用
            class _ToolCallProxy:
                """轻量代理，模拟 OpenAI ChatCompletionMessageToolCall 接口"""
                def __init__(self, data):
                    self.id = data["id"]
                    self.type = data["type"]
                    self.function = type("F", (), {
                        "name": data["function"]["name"],
                        "arguments": data["function"]["arguments"],
                    })()

            tc_proxy = _ToolCallProxy(tc_data)
            result_text = engine._execute_tool_call(tc_proxy)

            # Circuit Breaker: 记录工具调用
            engine._circuit_breaker.record_call(tool_name, tool_args, result_text)
            if tool_name in ("task_create_plan", "start_task_plan"):
                engine._circuit_breaker.escalate_for_planning()

            # 工具结果截断保护
            if len(result_text) > MAX_TOOL_RESULT_CHARS:
                orig_len = len(result_text)
                result_text = (
                    result_text[:MAX_TOOL_RESULT_CHARS]
                    + f"\n...(已截断，原始长度 {orig_len} 字符)"
                )
                logger.debug(
                    f"[FC][Graph] 工具 {tool_name} 结果截断: {orig_len} → {MAX_TOOL_RESULT_CHARS}"
                )

            tool_msg = {
                "role": "tool",
                "tool_call_id": tc_data["id"],
                "content": result_text,
            }
            messages.append(tool_msg)
            if engine._attn_window:
                engine._attn_window.register_message(
                    tool_msg, turn=fc_turn,
                    tool_name=tool_name,
                    node_id=tool_args.get("node_id") or tool_args.get("target_node_id"),
                    group_id=grp,
                )

            logger.info(f"[FC][Graph] 工具 {tool_name} 结果: {result_text[:200]}")
            tool_results_buffer.append({
                "tool_name": tool_name,
                "result": result_text,
            })

            # 发布任务图谱更新事件
            engine._publish_task_graph_event(
                "agent_tool_call", fc_turn, tool_name, result_text,
            )

        # Circuit Breaker: 本轮所有工具执行完毕，评估状态
        cb_state, cb_reason = engine._circuit_breaker.evaluate(fc_turn, messages)

        cb_force_no_tools = False
        if cb_state == CircuitBreakerState.RED:
            logger.warning(f"[FC][Graph][CB] RED 触发 (turn={fc_turn}): {cb_reason}")
            cb_force_no_tools = True
            cb_convergence = {
                "role": "system",
                "content": (
                    f"[Circuit Breaker 强制收敛] {cb_reason}\n"
                    "你必须立刻基于已有信息生成最终回复，不允许再调用任何工具。"
                ),
            }
            messages.append(cb_convergence)
            if engine._attn_window:
                engine._attn_window.register_message(cb_convergence, turn=fc_turn)

        elif cb_state == CircuitBreakerState.YELLOW:
            logger.warning(f"[FC][Graph][CB] YELLOW 警告 (turn={fc_turn}): {cb_reason}")
            cb_hint = {
                "role": "system",
                "content": (
                    f"[Circuit Breaker 警告] {cb_reason}\n"
                    "请尽快总结当前信息并回复用户，避免继续调用更多工具。"
                ),
            }
            messages.append(cb_hint)
            if engine._attn_window:
                engine._attn_window.register_message(cb_hint, turn=fc_turn)

        # ── MemoryGraph: BFS 扩散激活（以当前活跃节点为种子）──
        try:
            from zulong.memory.memory_graph import get_memory_graph
            from zulong.tools.task_tools import get_active_task_graph
            _mg = get_memory_graph()
            _tg = get_active_task_graph()
            logger.info(
                f"[FC][Graph] BFS 前置检查: mg={'有' if _mg else '无'}, "
                f"tg={'有' if _tg else '无'}"
            )
            if _mg and _tg:
                # 确保 TaskGraph 节点已投射到 MemoryGraph（首次创建时未同步）
                try:
                    from zulong.memory.graph_adapters import TaskGraphAdapter
                    _tga = TaskGraphAdapter()
                    _synced = _tga.sync(_mg, _tg)
                    if _synced:
                        logger.info(f"[FC][Graph] TaskGraph→MemoryGraph 同步: {_synced} 个节点")
                except Exception as _sync_err:
                    logger.info(f"[FC][Graph] TaskGraph 同步跳过: {_sync_err}")

                _in_progress = _tg.get_nodes_by_status("in_progress")
                if _in_progress:
                    _seed_ids = [f"task:{_tg.id}/{n.id}" for n in _in_progress]
                    # 过滤掉 MemoryGraph 中不存在的种子 ID
                    _valid_seeds = [s for s in _seed_ids if _mg.has_node(s)]
                    logger.info(
                        f"[FC][Graph] BFS 种子: seed_ids={_seed_ids}, "
                        f"valid={len(_valid_seeds)}/{len(_seed_ids)}"
                    )
                    if _valid_seeds:
                        _activations = _mg.compute_activations(
                            _valid_seeds, max_depth=3, decay=0.5,
                        )
                        logger.info(
                            f"[FC][Graph] BFS 激活扩散完成: "
                            f"seeds={[n.id for n in _in_progress]}"
                        )

                        # ── 思维导航: 自动焦点漂移 ──
                        # 如果最高激活节点不在当前种子中，自动迁移焦点
                        if _activations:
                            _focus_ctx = _mg.get_last_focus_context()
                            _current_focus = (
                                _focus_ctx.get("focused_task_node_id", "")
                                if _focus_ctx else ""
                            )
                            _top_node = max(
                                _activations, key=_activations.get,
                            )
                            # 仅当最高激活节点不是当前焦点且激活值 > 0.6 时漂移
                            if (_top_node != _current_focus
                                    and _activations[_top_node] > 0.6
                                    and _top_node not in _valid_seeds):
                                _mg.update_focus_to_node(_top_node)
                                # 联动 AttentionWindow 模式切换
                                if engine._attn_window:
                                    engine._attn_window.on_navigate_attention(
                                        direction="jump",
                                        target_node_id=_top_node,
                                    )
                                logger.info(
                                    f"[FC][Graph] 焦点自动漂移: "
                                    f"{_current_focus} → {_top_node} "
                                    f"(activation={_activations[_top_node]:.2f})"
                                )
        except Exception as _mg_err:
            logger.info(f"[FC][Graph] MemoryGraph 激活扩散跳过: {_mg_err}")

        return {
            "cb_force_no_tools": cb_force_no_tools,
            "tool_calls_data": None,
            "response_content": None,
            "should_terminate": "",
        }

    return exec_tools_node


def _make_eval_response_node(engine: "InferenceEngine"):
    """节点 4: 文本回复评估"""

    def eval_response_node(state: dict) -> dict:
        fc_turn = state["fc_turn"]
        messages = state["messages"]
        response_content = state.get("response_content", "") or ""
        cb_force_no_tools = state.get("cb_force_no_tools", False)
        tool_results_buffer = state["tool_results_buffer"]
        gap_continue_count = state.get("gap_continue_count", 0)

        response = response_content

        # CB 强制收敛：接受回复，跳过 Rule A 和 InfoGap
        if cb_force_no_tools:
            # 空回复保护：CB RED 后模型可能返回空字符串，此时使用降级回复
            if not response or len(response.strip()) < 10:
                logger.warning(
                    f"[FC][Graph][CB] Circuit Breaker 强制回复为空(len={len(response)})，生成降级回复"
                )
                # 优先利用工具结果缓冲区中的内容生成回复
                fallback = ""
                if tool_results_buffer:
                    useful_results = [
                        r["result"][:300] for r in tool_results_buffer
                        if r.get("result") and len(r.get("result", "")) > 20
                        and "error" not in r.get("result", "").lower()[:50]
                        # 过滤 JSON 结构化工具输出（task_view_overview 等返回的 JSON）
                        and not r.get("result", "").lstrip().startswith(("{", "["))
                    ]
                    if useful_results:
                        fallback = "根据已收集的信息：\n" + "\n".join(useful_results[:3])
                # 其次利用任务图生成进度报告
                if not fallback:
                    try:
                        from zulong.tools.task_tools import get_active_task_graph as _get_tg_fb
                        _fb_tg = _get_tg_fb()
                        if _fb_tg:
                            _fb_title = _fb_tg.title or "当前任务"
                            _fb_leaves = _fb_tg.get_leaf_nodes()
                            _fb_completed = [n for n in _fb_leaves if n.status == "completed"]
                            _fb_uncompleted = [n for n in _fb_leaves if n.status not in ("completed", "skipped")]
                            fallback = f"任务「{_fb_title}」进度：{len(_fb_completed)}/{len(_fb_leaves)} 完成。"
                            if _fb_uncompleted:
                                fallback += f"\n下一步需要执行：{_fb_uncompleted[0].label}。"
                            fallback += "\n请说「继续」来推进任务执行。"
                    except Exception:
                        pass
                # 最后使用引擎降级
                if not fallback:
                    fallback = engine._get_fallback_response(
                        state.get("user_input_text", "")
                    )
                response = fallback
            # CB 路径下也执行 Backfill：回填任务图节点内容
            # 质量检查：如果回复主要是 JSON/结构化工具输出，跳过 Backfill 防止数据污染
            _json_chars = sum(1 for c in response if c in '{}[]":,')
            _is_structured = (_json_chars / max(len(response), 1)) > 0.12
            if response and len(response) > 100 and not _is_structured:
                try:
                    from zulong.tools.task_tools import get_active_task_graph as _get_tg_cb
                    from zulong.tools.task_tools import _save_active_backup
                    _cb_tg = _get_tg_cb()
                    if _cb_tg:
                        _cb_leaves = _cb_tg.get_leaf_nodes()
                        _cb_uncompleted = [
                            n for n in _cb_leaves
                            if n.status not in ("completed", "skipped")
                        ]
                        if _cb_uncompleted:
                            _cb_filled = 0
                            for _cb_node in _cb_uncompleted:
                                if _has_content_match(response, _cb_node.label):
                                    _cb_content = _extract_node_content(
                                        response, _cb_node.label, max_len=500,
                                    )
                                    _cb_tg.update_node_status(
                                        _cb_node.id, "completed", result=_cb_content,
                                    )
                                    _cb_filled += 1
                            if _cb_filled > 0:
                                try:
                                    _save_active_backup()
                                except Exception:
                                    pass
                                logger.info(
                                    f"[FC][Graph][CB][Backfill] CB 路径回填: "
                                    f"{_cb_filled}/{len(_cb_uncompleted)} 节点已完成"
                                )
                                engine._publish_task_graph_event(
                                    "agent_tool_call", fc_turn, "task_backfill",
                                    f'{{"backfilled":{_cb_filled},"total_leaf":{len(_cb_leaves)}}}',
                                )
                except Exception as cb_bf_err:
                    logger.warning(f"[FC][Graph][CB][Backfill] 异常: {cb_bf_err}")

            logger.info(
                f"[FC][Graph][CB] Circuit Breaker 强制文本回复已接受，长度={len(response)}"
            )
            return {
                "response": response,
                "cb_force_no_tools": False,
                "should_terminate": "done",
            }

        # Rule A: 过早完成声明拦截（使用 engine 级别持久化实例，保留 retry_count）
        should_block = False
        try:
            rule_guardian = engine._rule_guardian
            from zulong.tools.task_tools import get_active_task_graph as _get_tg
            block, block_reason = rule_guardian.check_premature_completion(
                response, _get_tg()
            )
            if block:
                correction = {
                    "role": "system",
                    "content": (
                        f"[规则守护] {block_reason}\n"
                        "请调用 task_view_overview 查看任务图，"
                        "然后继续执行未完成的任务。不要直接回复用户。"
                    ),
                }
                messages.append({"role": "assistant", "content": response})
                messages.append(correction)
                if engine._attn_window:
                    engine._attn_window.register_message(
                        {"role": "assistant", "content": response}, turn=fc_turn,
                    )
                    engine._attn_window.register_message(correction, turn=fc_turn)
                should_block = True
        except Exception as guard_err:
            logger.warning(f"[FC][Graph][RuleGuardian] 检查异常: {guard_err}")

        if should_block:
            new_null_count = state.get("null_response_count", 0) + 1
            result = {
                "response": None,
                "should_terminate": "",
                "null_response_count": new_null_count,
            }
            # 拦截次数达到阈值时，注入 CB 强制收敛信号
            # 确保下一轮 call_model 不提供工具，模型必须生成最终文本回复
            if new_null_count >= 2:
                convergence_msg = {
                    "role": "system",
                    "content": (
                        "[强制收敛] 多次拦截检测到任务图有未完成节点，"
                        "但模型持续尝试直接回复。请立即基于已有信息生成最终回复。"
                    ),
                }
                messages.append(convergence_msg)
                if engine._attn_window:
                    engine._attn_window.register_message(convergence_msg, turn=fc_turn)
                result["cb_force_no_tools"] = True
                logger.info(
                    f"[FC][Graph] 拦截次数达 {new_null_count}，注入 CB 强制收敛"
                )
            return result

        # 构建子任务上下文（供 InfoGapDetector 结构化依赖检查）
        subtask_ctx = None
        try:
            from zulong.tools.task_tools import get_active_task_graph
            active_tg = get_active_task_graph()
            if active_tg:
                in_progress = active_tg.get_nodes_by_status("in_progress")
                if in_progress:
                    cur_node = in_progress[0]
                    deps = active_tg.get_dependencies(cur_node.id)
                    available = {}
                    for dep_id in deps:
                        dep_node = active_tg.get_node(dep_id)
                        if dep_node and dep_node.status == "completed":
                            available[dep_id] = dep_node.result or ""
                    subtask_ctx = {
                        "current_subtask": cur_node.id,
                        "dependencies": deps,
                        "available_results": available,
                    }
        except Exception:
            pass

        # 信息缺口检测
        _MAX_GAP_CONTINUES = 5
        should_continue = False
        try:
            from zulong.l2.info_gap_detector import InfoGapType
            gap_type, gap_desc, gap_conf = engine._info_gap_detector.detect(
                llm_output=response,
                tool_results=tool_results_buffer if tool_results_buffer else None,
                subtask_context=subtask_ctx,
            )
            if gap_type == InfoGapType.NEED_USER_INPUT and gap_conf >= 0.6:
                logger.info(
                    f"[FC][Graph][InfoGap] 需要用户输入: {gap_desc} (置信度={gap_conf:.2f})"
                )
                # 模型已在回复中向用户提问，直接接受
            elif gap_type == InfoGapType.NEED_SUBTASK_RESULT and gap_conf >= 0.6:
                logger.info(
                    f"[FC][Graph][InfoGap] 需要子任务结果: {gap_desc} "
                    f"(置信度={gap_conf:.2f}), 重试={gap_continue_count}/{_MAX_GAP_CONTINUES}"
                )
                # RESUME 模式下模型正在为子任务产生内容，不应被 InfoGap 拦截
                _is_resume = state.get("is_resume", False)
                if _is_resume and len(response) > 100:
                    logger.info(
                        "[FC][Graph][InfoGap] RESUME 模式且回复充实，跳过子任务结果拦截"
                    )
                elif gap_continue_count >= _MAX_GAP_CONTINUES:
                    logger.warning(
                        f"[FC][Graph][InfoGap] 闭环已达重试上限 ({_MAX_GAP_CONTINUES})，"
                        f"标记任务为 blocked 并放行"
                    )
                    # 修复：达到上限后标记当前节点为 blocked，而非静默放行
                    try:
                        from zulong.tools.task_tools import get_active_task_graph
                        _tg = get_active_task_graph()
                        if _tg:
                            in_prog = _tg.get_nodes_by_status("in_progress")
                            for _n in in_prog:
                                _tg.update_node_status(
                                    _n.id, "blocked",
                                    result=f"信息缺口: {gap_desc}"
                                )
                                logger.info(
                                    f"[FC][Graph][InfoGap] 节点 {_n.id} 标记为 blocked"
                                )
                    except Exception:
                        pass
                else:
                    gap_hint = {
                        "role": "system",
                        "content": (
                            f"[信息缺口提示] 当前子任务缺少前置结果: {gap_desc}\n"
                            "请先用 task_view_overview 查看任务图，找到并执行未完成的前置子任务，"
                            "或用 task_mark_status 更新进度后继续。"
                        ),
                    }
                    messages.append({"role": "assistant", "content": response})
                    messages.append(gap_hint)
                    if engine._attn_window:
                        engine._attn_window.register_message(
                            {"role": "assistant", "content": response}, turn=fc_turn,
                        )
                        engine._attn_window.register_message(gap_hint, turn=fc_turn)
                    should_continue = True
                    new_null_count = state.get("null_response_count", 0) + 1
                    result = {
                        "response": None,
                        "gap_continue_count": gap_continue_count + 1,
                        "should_terminate": "",
                        "null_response_count": new_null_count,
                    }
                    if new_null_count >= 2:
                        result["cb_force_no_tools"] = True
                        logger.info(
                            f"[FC][Graph][InfoGap] 拦截次数达 {new_null_count}，注入 CB 强制收敛"
                        )
                    return result
            else:
                logger.debug(
                    f"[FC][Graph][InfoGap] 信息充足 (type={gap_type.value}, conf={gap_conf:.2f})"
                )
        except Exception as e:
            logger.warning(f"[FC][Graph][InfoGap] 检测异常，跳过: {e}")

        if should_continue:
            return {
                "response": None,
                "should_terminate": "",
            }

        # ── RESUME Auto-Mark 安全网 ──────────────────────────────
        # 4B 模型在 RESUME 流程中常常生成实质内容但忘记调用 task_mark_status，
        # 这里在接受回复前自动补标节点，避免任务图永远不更新。
        _MAX_RESUME_AUTOMARKS = 5
        is_resume = state.get("is_resume", False)
        resume_automark_count = state.get("resume_automark_count", 0)

        if (
            is_resume
            and len(response) > 100
            and resume_automark_count < _MAX_RESUME_AUTOMARKS
            and not response.rstrip().endswith(("?", "\uff1f"))
            and not _is_filler_content(response)
        ):
            try:
                from zulong.tools.task_tools import get_active_task_graph as _get_tg_am
                from zulong.tools.task_tools import _save_active_backup
                tg = _get_tg_am()
                if tg:
                    leaf_nodes = tg.get_leaf_nodes()
                    uncompleted = [n for n in leaf_nodes if n.status != "completed"]
                    if uncompleted:
                        # 优先选 in_progress 节点，其次 pending
                        target = None
                        for n in uncompleted:
                            if n.status == "in_progress":
                                target = n
                                break
                        if not target:
                            target = uncompleted[0]

                        # 自动标记完成
                        result_text = response[:500]
                        tg.update_node_status(target.id, "completed", result=result_text)
                        try:
                            _save_active_backup()
                        except Exception:
                            pass
                        logger.info(
                            f"[FC][RESUME][AutoMark] 自动完成节点 {target.id}"
                            f" ({target.label}), result_len={len(result_text)}"
                        )
                        engine._publish_task_graph_event(
                            "agent_tool_call", fc_turn, "task_mark_status",
                            f'{{"node_id":"{target.id}","status":"completed","auto_mark":true}}',
                        )

                        # 检查剩余未完成节点
                        remaining = [n for n in tg.get_leaf_nodes() if n.status != "completed"]
                        if remaining:
                            next_node = remaining[0]
                            tg.update_node_status(next_node.id, "in_progress")
                            try:
                                _save_active_backup()
                            except Exception:
                                pass
                            logger.info(
                                f"[FC][RESUME][AutoMark] 下一节点: {next_node.id}"
                                f" ({next_node.label}), 标记 in_progress"
                            )
                            continuation = {
                                "role": "system",
                                "content": (
                                    f"[自动进度更新] 节点 {target.id}（{target.label}）已完成。\n"
                                    f"请继续执行节点 {next_node.id}（{next_node.label}）"
                                    f"：{next_node.desc or next_node.label}\n"
                                    f"完成后请用 task_mark_status(node_id='{next_node.id}',"
                                    f" status='completed', result='结果') 提交。"
                                ),
                            }
                            messages.append({"role": "assistant", "content": response})
                            messages.append(continuation)
                            if engine._attn_window:
                                engine._attn_window.register_message(
                                    {"role": "assistant", "content": response}, turn=fc_turn,
                                )
                                engine._attn_window.register_message(continuation, turn=fc_turn)
                            new_null_count = state.get("null_response_count", 0) + 1
                            result = {
                                "response": None,
                                "resume_automark_count": resume_automark_count + 1,
                                "should_terminate": "",
                                "null_response_count": new_null_count,
                            }
                            if new_null_count >= 2:
                                result["cb_force_no_tools"] = True
                                logger.info(
                                    f"[FC][RESUME][AutoMark] 拦截次数达 {new_null_count}，注入 CB 强制收敛"
                                )
                            return result
                        else:
                            logger.info(
                                "[FC][RESUME][AutoMark] 所有节点已完成，接受回复"
                            )
            except Exception as am_err:
                logger.warning(f"[FC][RESUME][AutoMark] 异常: {am_err}")
        # ── END RESUME Auto-Mark ─────────────────────────────────

        # ── COMPLEX 首次执行：任务图节点内容回填 ──────────────────
        # 4B 模型在 COMPLEX 首次执行时常见行为：
        #   1. 调用 task_create_plan + task_add_node 创建任务图骨架
        #   2. 直接生成完整回复内容，跳过逐节点 task_mark_status
        # 导致任务图节点全部为空但回复内容完整。
        # 此安全网在接受回复前自动回填节点内容，确保前端任务图可视化正确。
        if (
            not is_resume
            and len(response) > 100
            and not response.rstrip().endswith(("?", "\uff1f"))
            and not _is_filler_content(response)
        ):
            try:
                from zulong.tools.task_tools import get_active_task_graph as _get_tg_bf
                from zulong.tools.task_tools import _save_active_backup
                tg = _get_tg_bf()
                if tg:
                    leaf_nodes = tg.get_leaf_nodes()
                    uncompleted = [
                        n for n in leaf_nodes
                        if n.status not in ("completed", "skipped")
                    ]
                    if uncompleted:
                        backfill_count = 0
                        skipped_count = 0
                        for node in uncompleted:
                            # 尝试从回复中提取与节点相关的内容片段
                            node_content = _extract_node_content(
                                response, node.label, max_len=500,
                            )
                            # 只有在回复中确实找到匹配内容时才标记完成
                            # _extract_node_content 兜底返回的是 response[:max_len]
                            # 与节点无关的兜底内容不应视为完成证据
                            if _has_content_match(response, node.label):
                                tg.update_node_status(
                                    node.id, "completed", result=node_content,
                                )
                                backfill_count += 1
                            else:
                                skipped_count += 1
                                logger.debug(
                                    f"[FC][Backfill] 跳过节点 {node.id}（{node.label}）：回复中未找到匹配内容"
                                )
                        try:
                            _save_active_backup()
                        except Exception:
                            pass
                        logger.info(
                            f"[FC][Backfill] COMPLEX 首次执行回填: "
                            f"已填充 {backfill_count} 个节点，跳过 {skipped_count} 个无匹配节点"
                        )
                        engine._publish_task_graph_event(
                            "agent_tool_call", fc_turn, "task_backfill",
                            f'{{"backfilled":{backfill_count},"skipped":{skipped_count},"total_leaf":{len(leaf_nodes)}}}',
                        )
            except Exception as bf_err:
                logger.warning(f"[FC][Backfill] 异常: {bf_err}")
        # ── END COMPLEX Backfill ─────────────────────────────────

        # ── 空/短回复 + 未完成任务图 安全网 ──────────────────────
        # 4B 模型常在创建任务图骨架后返回空字符串就停止，
        # 此时任务图有未完成节点，不应接受空回复。
        if not response or len(response.strip()) < 10:
            try:
                from zulong.tools.task_tools import get_active_task_graph as _get_tg_empty
                _tg_empty = _get_tg_empty()
                if _tg_empty:
                    _leaves = _tg_empty.get_leaf_nodes()
                    _uncompleted = [
                        n for n in _leaves
                        if n.status not in ("completed", "skipped")
                    ]
                    if _uncompleted:
                        new_null_count = state.get("null_response_count", 0) + 1
                        # 找到下一个需要执行的节点
                        _next = None
                        for _n in _uncompleted:
                            if _n.status == "in_progress":
                                _next = _n
                                break
                        if not _next:
                            _next = _uncompleted[0]
                            _tg_empty.update_node_status(_next.id, "in_progress")

                        # 构建提示：告知当前任务图状态和下一步
                        _completed_count = len([
                            n for n in _leaves if n.status == "completed"
                        ])
                        nudge = {
                            "role": "system",
                            "content": (
                                f"[空回复拦截] 你的回复为空，但任务图还有 "
                                f"{len(_uncompleted)}/{len(_leaves)} 个未完成节点。\n"
                                f"当前进度：{_completed_count}/{len(_leaves)} 完成。\n"
                                f"请立即开始执行节点 {_next.id}（{_next.label}）"
                                f"：{_next.desc or _next.label}\n"
                                f"生成该节点的详细内容，完成后调用 "
                                f"task_mark_status(node_id='{_next.id}', "
                                f"status='completed', result='你的结果')。"
                            ),
                        }
                        messages.append(nudge)
                        if engine._attn_window:
                            engine._attn_window.register_message(nudge, turn=fc_turn)
                        logger.info(
                            f"[FC][Graph][EmptyGuard] 空回复拦截，"
                            f"未完成节点 {len(_uncompleted)}/{len(_leaves)}，"
                            f"提示执行 {_next.id}（{_next.label}），"
                            f"null_count={new_null_count}"
                        )
                        result = {
                            "response": None,
                            "should_terminate": "",
                            "null_response_count": new_null_count,
                        }
                        # 多次空回复后强制收敛，避免无限循环
                        if new_null_count >= 2:
                            result["cb_force_no_tools"] = True
                            logger.info(
                                f"[FC][Graph][EmptyGuard] 空回复拦截次数达 "
                                f"{new_null_count}，注入 CB 强制收敛"
                            )
                        return result
            except Exception as eg_err:
                logger.warning(f"[FC][Graph][EmptyGuard] 异常: {eg_err}")
        # ── END 空/短回复安全网 ───────────────────────────────────

        # ── 任务全完成但回复为空 → 从 TaskGraph 合成摘要 ─────────
        if not response or len(response.strip()) < 10:
            try:
                from zulong.tools.task_tools import get_active_task_graph as _get_tg_synth
                _tg_synth = _get_tg_synth()
                if _tg_synth:
                    _synth_leaves = _tg_synth.get_leaf_nodes()
                    _synth_completed = [
                        n for n in _synth_leaves if n.status == "completed"
                    ]
                    if _synth_completed and len(_synth_completed) == len(_synth_leaves):
                        # 所有任务已完成 → 从节点 result 合成摘要回复
                        parts = [f"## {_tg_synth.title}\n"]
                        for node in _synth_completed:
                            result_text = getattr(node, 'result', '') or ''
                            if result_text:
                                parts.append(f"### {node.label}\n{result_text}\n")
                            else:
                                parts.append(f"### {node.label}\n（已完成）\n")
                        response = "\n".join(parts)
                        logger.info(
                            f"[FC][Graph][Synthesize] 任务全部完成但模型返回空响应，"
                            f"从 {len(_synth_completed)} 个已完成节点合成摘要回复"
                        )
            except Exception as synth_err:
                logger.warning(f"[FC][Graph][Synthesize] 合成异常: {synth_err}")
        # ── END 任务全完成合成 ──────────────────────────────────

        # 所有检查通过，接受回复
        return {
            "response": response,
            "should_terminate": "done",
        }

    return eval_response_node


# ---------------------------------------------------------------------------
# 3. 辅助函数
# ---------------------------------------------------------------------------

_FILLER_PATTERNS = [
    "我正在思考", "让我继续", "我来继续", "让我想想", "接下来我",
    "我正在处理", "正在分析", "正在执行", "稍等", "我需要",
    "但我需要", "不过我需要", "还需要进一步", "需要更多信息",
]


def _is_filler_content(text: str) -> bool:
    """检测回复是否为填充性内容（无实质任务成果）"""
    stripped = text.strip()
    if len(stripped) < 50:
        return True
    filler_count = sum(1 for p in _FILLER_PATTERNS if p in stripped)
    if filler_count >= 2:
        return True
    return False


def _has_content_match(response: str, node_label: str) -> bool:
    """检查回复中是否包含与节点标签匹配的内容。

    用于 Backfill 前置判断：只有回复中确实存在与节点相关的内容时，
    才能将该节点标记为 completed。避免将兜底截取的内容误判为完成。

    策略：
    1. 精确匹配节点标签
    2. 节点标签中的中文关键词（>=2字）至少匹配到一个
    """
    if not response or not node_label:
        return False

    # 策略1: 精确匹配
    if node_label in response:
        return True

    # 策略2: 2字符滑动窗口匹配（中文无空格分词，取连续2字符子串）
    # 要求至少命中 2 个不同 bigram，减少短文本单个常用词误匹配
    import re as _re
    cjk_runs = _re.findall("[一-鿿]{2,}", node_label)
    matched_bigrams = set()
    for run in cjk_runs:
        for i in range(len(run) - 1):
            bigram = run[i:i+2]
            if bigram in response:
                matched_bigrams.add(bigram)
    if len(matched_bigrams) >= 2:
        return True

    return False


def _extract_node_content(response: str, node_label: str, max_len: int = 500) -> str:
    """从回复中提取与节点标签相关的内容片段。

    策略：
    1. 在回复中搜索节点标签（或关键词），提取标签后的 max_len 字符
    2. 如果找不到标签，用节点标签中的核心关键词模糊匹配
    3. 兜底：返回回复前 max_len 字符
    """
    if not response or not node_label:
        return response[:max_len] if response else ""

    # 策略1: 精确匹配节点标签
    idx = response.find(node_label)
    if idx >= 0:
        start = idx
        end = min(len(response), start + max_len)
        # 向后扩展到段落边界（双换行）
        next_section = response.find("\n\n", start + len(node_label))
        if 0 < next_section - start <= max_len:
            end = next_section
        return response[start:end].strip()

    # 策略2: 提取节点标签中的核心关键词进行模糊匹配
    import re as _re
    keywords = _re.findall("[一-鿿]{2,}", node_label)
    for kw in keywords:
        idx = response.find(kw)
        if idx >= 0:
            start = max(0, idx - 20)
            end = min(len(response), idx + max_len)
            return response[start:end].strip()

    # 策略3: 兜底 — 返回回复前 max_len 字符
    return response[:max_len]


# ---------------------------------------------------------------------------
# 4. 条件路由函数
# ---------------------------------------------------------------------------

def _route_after_check(state: dict) -> str:
    """check 节点后的路由"""
    if state.get("should_terminate"):
        return "end"
    return "call_model"


def _route_after_call(state: dict) -> str:
    """call_model 节点后的路由"""
    if state.get("should_terminate"):
        return "end"
    # 超时重试（tool_calls_data 和 response_content 都为 None）
    if state.get("tool_calls_data") is None and state.get("response_content") is None:
        return "check"
    # 有工具调用
    if state.get("tool_calls_data"):
        return "exec_tools"
    # 纯文本回复
    return "eval_response"


def _route_after_eval(state: dict) -> str:
    """eval_response 节点后的路由"""
    if state.get("should_terminate"):
        return "end"
    # response 被拦截（Rule A / InfoGap / AutoMark），回到 check 继续
    if state.get("response") is None:
        # 安全上限：连续拦截次数超过阈值时强制终止，防止无限循环
        # 从 8 降到 3：三层拦截机制最多各触发一次，避免 80-240s 额外延迟
        _MAX_NULL_RESPONSES = 3
        null_count = state.get("null_response_count", 0)
        if null_count >= _MAX_NULL_RESPONSES:
            logger.warning(
                f"[FC][Graph] 连续 {null_count} 次 response=None 拦截，"
                f"超过安全上限 ({_MAX_NULL_RESPONSES})，强制终止"
            )
            return "end"
        return "check"
    return "end"


# ---------------------------------------------------------------------------
# 4. 图构建与执行
# ---------------------------------------------------------------------------

def build_fc_graph(engine: "InferenceEngine") -> "CompiledStateGraph":
    """构建 FC Loop 的 LangGraph StateGraph。

    Args:
        engine: InferenceEngine 实例（通过闭包注入到节点函数中）

    Returns:
        编译后的 CompiledStateGraph，可多次 invoke。
    """
    graph = StateGraph(FCLoopState)

    # 添加节点
    graph.add_node("check", _make_check_node(engine))
    graph.add_node("call_model", _make_call_model_node(engine))
    graph.add_node("exec_tools", _make_exec_tools_node(engine))
    graph.add_node("eval_response", _make_eval_response_node(engine))

    # 设置入口
    graph.set_entry_point("check")

    # 条件边: check →
    graph.add_conditional_edges(
        "check",
        _route_after_check,
        {"call_model": "call_model", "end": END},
    )

    # 条件边: call_model →
    graph.add_conditional_edges(
        "call_model",
        _route_after_call,
        {
            "end": END,
            "check": "check",
            "exec_tools": "exec_tools",
            "eval_response": "eval_response",
        },
    )

    # 固定边: exec_tools → check（对应原 while 循环的 continue）
    graph.add_edge("exec_tools", "check")

    # 条件边: eval_response →
    graph.add_conditional_edges(
        "eval_response",
        _route_after_eval,
        {"end": END, "check": "check"},
    )

    compiled = graph.compile()
    logger.info("[FC][Graph] LangGraph FC Loop 图已编译")
    return compiled


def run_fc_loop(
    engine: "InferenceEngine",
    messages: List[Dict],
    tool_definitions: List[Dict],
    vllm_model_id: str,
    force_first_tool: bool = False,
    user_input: str = "",
    is_resume: bool = None,
) -> Tuple[Optional[str], int]:
    """执行 LangGraph FC Loop，返回 (response, fc_turn)。

    Args:
        engine: InferenceEngine 实例
        messages: OpenAI 格式对话消息列表（可变引用，节点内直接 append）
        tool_definitions: 工具定义列表
        vllm_model_id: 远程模型 ID
        force_first_tool: RESUME 场景第一轮是否强制 task_view_overview
        user_input: 用户原始输入（用于降级回复）
        is_resume: 是否为任务恢复场景（默认 None 时回退到 force_first_tool）

    Returns:
        (response, fc_turn) — response 可能为 None（需要调用方降级处理）
    """
    compiled_graph = build_fc_graph(engine)

    # 每次 FC 循环开始前重置 RuleGuardian 计数器
    if hasattr(engine, '_rule_guardian'):
        engine._rule_guardian.reset()

    # 组装初始状态
    initial_state: FCLoopState = {
        "messages": messages,
        "fc_turn": 0,
        "response": None,
        "tool_results_buffer": [],
        "cb_force_no_tools": False,
        "gap_continue_count": 0,
        "should_terminate": "",
        "tool_calls_data": None,
        "response_content": None,
        "force_first_tool": force_first_tool,
        "vllm_model_id": vllm_model_id,
        "tool_definitions": tool_definitions,
        "user_input_text": user_input,
        "is_resume": is_resume if is_resume is not None else force_first_tool,
        "resume_automark_count": 0,
        "null_response_count": 0,
        "api_timeout_count": 0,
    }

    # 执行图（recursion_limit 作为安全网）
    recursion_limit = engine._hard_limit + 10
    try:
        final_state = compiled_graph.invoke(
            initial_state,
            config={"recursion_limit": recursion_limit},
        )
    except Exception as e:
        # GraphRecursionError 或其他异常
        err_name = type(e).__name__
        logger.error(
            f"[FC][Graph] 图执行异常 ({err_name}): {e}"
        )
        # 尝试从当前 messages 中提取最后一条 assistant 回复
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content and len(content) > 10:
                    logger.info(
                        f"[FC][Graph] 从 messages 中恢复最后回复，长度={len(content)}"
                    )
                    return content, engine._hard_limit
        return engine._get_fallback_response(user_input), engine._hard_limit

    response = final_state.get("response")
    fc_turn = final_state.get("fc_turn", 0)

    logger.info(f"[FC][Graph] 图执行完成，共 {fc_turn} 轮，response={'有' if response else '无'}")
    return response, fc_turn
