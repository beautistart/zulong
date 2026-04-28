# File: zulong/l2/intent_prompt_builder.py
# 两阶段 FC 意图分类 — 提示词构建器与工具过滤
#
# Round 1: 极简分类提示 + start_session 工具 → 模型被强制输出意图标签
# Round 2: 场景化提示 + 过滤后的工具集 → 模型在受控范围内自由行动
#
# 设计原则：固化骨架 + 自由决策
# - 基础设施操作由代码确定性执行（创建图谱 / 恢复任务 / 过滤工具）
# - 内容决策由模型自由生成（如何规划 / 执行什么 / 怎么回答）

import logging
import asyncio
from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# 意图枚举
# ──────────────────────────────────────────────────────────

class IntentType(Enum):
    """用户意图分类"""
    CHAT = "chat"          # 日常闲聊、简单问答、知识查询
    COMPLEX = "complex"    # 多步骤复杂任务（开发、设计、写报告等）
    RESUME = "resume"      # 恢复/继续之前挂起的任务


# ──────────────────────────────────────────────────────────
# Round 1: 分类阶段
# ──────────────────────────────────────────────────────────

def build_round1_system_prompt() -> str:
    """构建 Round 1 分类提示词（~300 token）

    此提示词极度精简，仅包含身份声明和分类指令。
    不包含任何任务管理规则、模板或示例。
    当有活跃任务图时，注入上下文帮助 LLM 正确分类。

    Returns:
        系统提示词文本
    """
    now = datetime.now()
    current_time_str = now.strftime("%Y-%m-%d %H:%M")

    # 检测活跃任务图，为分类器提供上下文
    active_graph_hint = ""
    try:
        from zulong.tools.task_tools import get_active_task_graph
        tg = get_active_task_graph()
        if tg is not None:
            root = tg.get_node("req")
            title = root.label if root else tg.title
            leaves = tg.get_leaf_nodes()
            uncompleted = [n for n in leaves if n.status not in ("completed", "skipped")]
            if uncompleted:
                active_graph_hint = (
                    f"\n⚠️ 重要上下文：当前有一个进行中的任务「{title}」，"
                    f"还有 {len(uncompleted)} 个未完成的子任务。"
                    f"如果用户的输入看起来是在回答问题、补充信息、或与该任务相关，"
                    f"请分类为 complex（而非 chat）。\n"
                )
            elif leaves:
                # 所有叶子节点已完成 → 注入已完成任务上下文
                active_graph_hint = (
                    f"\n⚠️ 重要上下文：刚刚完成了一个任务「{title}」。"
                    f"如果用户的输入是关于这个已完成任务的后续提问"
                    f"（如怎么运行、怎么部署、怎么使用、查看结果等），"
                    f"请分类为 chat（不是 complex），因为这不需要创建新任务。\n"
                )
    except Exception:
        pass

    return (
        "你是祖龙（ZULONG），一个智能助手。\n"
        f"当前时间：{current_time_str}\n"
        f"{active_graph_hint}"
        "\n"
        "你的任务：分析用户输入的意图，然后调用 start_session 工具进行分类。\n"
        "\n"
        "意图分类定义：\n"
        "- chat: 简短闲聊、单一问题、打招呼。例：\"你好\"、\"今天天气怎么样\"、\"Python是什么\"\n"
        "- complex: 需要多个步骤或产出多个部分的任务。包括但不限于：\n"
        "  * 开发/编码/设计/写报告/做游戏等创作任务\n"
        "  * 分析对比、列出多项内容、归纳总结等结构化输出\n"
        "  * 包含\"并\"、\"同时\"、\"另外\"等连接词的多部分请求\n"
        "  * 包含\"分析\"、\"对比\"、\"列出\"、\"设计\"、\"帮我做\"等动词的任务\n"
        "  * 用户为正在进行的任务补充信息（如回答出发日期、预算等问题）\n"
        "- resume: 用户想要继续/恢复/接着做之前暂停或挂起的任务\n"
        "\n"
        "分类要点：\n"
        "- 包含\"继续\"、\"接着做\"、\"恢复\"、\"上次那个\"等恢复意图时 → resume\n"
        "- 用户请求中包含2个或以上子任务/子要求时 → complex\n"
        "- 用户请求需要结构化输出（列表、对比表、多段分析）时 → complex\n"
        "- 仅在明确是简短闲聊或单一事实查询时 → chat\n"
        "\n"
        "⚠️ 语言规则：task_description 必须使用与用户输入相同的语言。"
        "用户用中文提问，task_description 必须用中文。\n"
    )


def get_round1_tools() -> List[Dict[str, Any]]:
    """获取 Round 1 的工具定义（仅 start_session）

    返回硬编码的 OpenAI FC 格式 schema，不依赖 ToolRegistry。

    Returns:
        包含单个工具定义的列表
    """
    return [{
        "type": "function",
        "function": {
            "name": "start_session",
            "description": "对用户输入进行意图分类。必须在每次对话开始时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["chat", "complex", "resume"],
                        "description": "用户意图分类：chat（闲聊/问答）、complex（复杂多步骤任务）、resume（恢复之前的任务）",
                    },
                    "reason": {
                        "type": "string",
                        "description": "分类理由的简短说明",
                    },
                    "task_description": {
                        "type": "string",
                        "description": "对于 complex/resume 意图，提取任务的简短描述（用于创建任务图或匹配挂起任务）。必须使用与用户输入相同的语言。",
                    },
                },
                "required": ["intent", "reason"],
            },
        },
    }]


# ──────────────────────────────────────────────────────────
# Round 2: 执行阶段 — 工具过滤
# ──────────────────────────────────────────────────────────

# 各场景允许的工具集（COMPLEX 返回 None 表示不过滤）
_CHAT_TOOLS = {
    "recall_memory", "read_memory_node", "save_memory_note",
    "discover_related", "navigate_attention", "search_experience",
    "openclaw_search", "search_tools",
}

_RESUME_TOOLS = {
    # 任务管理
    "task_view_overview", "task_mark_status", "task_suspend",
    "task_list_suspended",
    # 执行类（RESUME 必须能实际执行任务内容）
    "exec_write_file", "exec_run_command",
    # 记忆/检索
    "recall_memory", "read_memory_node", "discover_related",
    "search_experience",
    # 注意力导航
    "navigate_attention",
}
# 关键：RESUME 场景物理排除 task_create_plan 和 task_add_node（防止重复创建图）


def get_round2_tool_names(intent_type: IntentType) -> Optional[set]:
    """获取 Round 2 允许的工具名集合

    Args:
        intent_type: 意图分类结果

    Returns:
        允许的工具名集合。None 表示不过滤（使用全部工具）。
    """
    if intent_type == IntentType.CHAT:
        return _CHAT_TOOLS.copy()
    elif intent_type == IntentType.RESUME:
        return _RESUME_TOOLS.copy()
    else:
        # COMPLEX: 使用全部工具
        return None


# ──────────────────────────────────────────────────────────
# Round 2: 执行阶段 — 场景化提示词构建
# ──────────────────────────────────────────────────────────

def _build_time_header() -> str:
    """构建时间和身份头部（所有场景共用）"""
    now = datetime.now()
    hour = now.hour
    current_time_str = now.strftime("%Y-%m-%d %H:%M")

    if 5 <= hour < 11:
        time_period = "早晨"
    elif 11 <= hour < 14:
        time_period = "中午"
    elif 14 <= hour < 18:
        time_period = "下午"
    elif 18 <= hour < 22:
        time_period = "晚上"
    else:
        time_period = "深夜"

    return (
        "**重要身份认知**：\n"
        "- 你的名字叫 \"祖龙 (ZULONG)\"\n"
        f"\n当前时间：{current_time_str} ({time_period})。\n"
        "\n【人称代词】\n"
        "- \"我\" 指的是你自己（祖龙）\n"
        "- \"你\" 指的是用户\n"
        "- 当用户说\"我家\"、\"我叫\"时，指的是用户\n"
    )


def _inject_memory_context(system_parts: list, user_input: str, rag_manager=None,
                           attn_stats: dict = None):
    """注入记忆上下文（MemoryGraph 检索 + 思维导航 + 容量感知）

    Args:
        system_parts: 系统提示词片段列表
        user_input: 用户输入（用于记忆检索）
        rag_manager: RAGManager 实例
        attn_stats: AttentionWindowManager.stats 字典（可选）
    """
    # 思维导航注入
    try:
        from zulong.memory.memory_graph import get_memory_graph as _get_mg_nav
        _mg_nav = _get_mg_nav()
        if _mg_nav:
            focus_summary = _mg_nav.get_focus_path_summary()
            if focus_summary:
                system_parts.append(f"\n{focus_summary}\n")
                logger.debug(f"[思维导航] 已注入焦点路径 ({len(focus_summary)} chars)")
    except Exception as e:
        logger.debug(f"[思维导航] 注入跳过: {e}")

    # MemoryGraph 记忆检索
    try:
        from zulong.memory.memory_graph import get_memory_graph
        _mg = get_memory_graph()
        if _mg:
            if not getattr(_mg, '_rag_manager', None) and rag_manager:
                _mg.set_rag_manager(rag_manager)

            def _run_async_bridge(coro):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None and loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        return pool.submit(asyncio.run, coro).result(timeout=30)
                else:
                    return asyncio.run(coro)

            # 动态 top_k：根据注意力窗口剩余预算调整检索量
            _top_k = 8  # 默认值
            if attn_stats:
                _ratio = attn_stats.get("usage_ratio", 0)
                if _ratio > 0.8:
                    _top_k = 3  # 容量紧张，减少检索
                elif _ratio > 0.6:
                    _top_k = 5
                # else: 保持 8

            mg_results = _run_async_bridge(
                _mg.retrieve_context(user_input, top_k=_top_k, session_id="")
            )
            if mg_results:
                memory_sections = []
                for r in mg_results:
                    ntype = r.get("node_type", "")
                    content = r.get("content", "")
                    label = r.get("label", "")
                    if not content:
                        continue
                    if ntype == "experience":
                        continue  # EXPERIENCE 由 search_experience FC 工具按需获取
                    elif ntype == "dialogue":
                        memory_sections.append(f"【历史对话】{content[:200]}")
                    elif ntype == "task":
                        status = r.get("metadata", {}).get("status", "")
                        memory_sections.append(
                            f"【相关任务】{label}" + (f"（状态：{status}）" if status else "")
                        )
                    elif ntype == "knowledge":
                        memory_sections.append(f"【知识参考】{content[:300]}")
                    elif ntype == "episode":
                        memory_sections.append(f"【历史摘要】{content[:200]}")
                    elif ntype in ("person", "concept"):
                        memory_sections.append(f"【知识参考】{label}: {content[:200]}")
                    else:
                        memory_sections.append(f"【参考】{content[:200]}")

                if memory_sections:
                    system_parts.append(
                        "\n【记忆上下文】\n" + "\n".join(memory_sections) + "\n"
                    )
                    logger.info(f"[MemoryGraph] 注入 {len(memory_sections)} 条记忆到上下文")
    except Exception as e:
        logger.warning(f"[MemoryGraph] 记忆检索失败，降级跳过: {e}")

    # 注意力状态 + 容量仪表盘
    _mem_count = sum(1 for p in system_parts if "【记忆上下文】" in p or "【历史对话】" in p)
    _has_memory = _mem_count > 0
    _attn_lines = ["\n【注意力状态】"]

    # 容量仪表盘（当 attn_stats 可用时注入）
    if attn_stats:
        _mode = attn_stats.get("mode", "global")
        _ratio = attn_stats.get("usage_ratio", 0)
        _remaining = attn_stats.get("remaining_tokens", 0)
        _budget = attn_stats.get("budget", 0)
        # 使用比率条 ████░░░░░░
        _bar_len = 10
        _filled = int(_ratio * _bar_len)
        _bar = "\u2588" * _filled + "\u2591" * (_bar_len - _filled)
        _attn_lines.append(f"容量: [{_bar}] {_ratio:.0%}  剩余≈{_remaining}tok  模式={_mode}")
        if _ratio >= 0.85:
            _attn_lines.append("容量紧张，请精简输出，避免冗长工具调用。")
    else:
        _attn_lines.append("容量: 未启用窗口管理")

    if _has_memory:
        _attn_lines.append(f"已注入 {_mem_count} 段记忆/上下文。")
        _attn_lines.append("如果这些信息不足以回答用户问题，请主动调用 recall_memory 工具检索更多相关记忆。")
    else:
        _attn_lines.append("当前对话未注入任何记忆上下文。")
        _attn_lines.append("如果用户的问题涉及历史信息或个人偏好，请主动调用 recall_memory 工具进行检索。")
    _attn_lines.append("如果需要用户补充信息才能继续，请直接用自然语言向用户提问。\n")
    system_parts.append("\n".join(_attn_lines))


def _build_chat_prompt(user_input: str, rag_context: Optional[str],
                       visual_context: Optional[str], rag_manager=None,
                       attn_stats: dict = None) -> list:
    """构建 CHAT 场景的 messages（~800 token）

    不包含任何任务管理规则。仅保留身份、交流风格、RAG、记忆。
    """
    system_parts = [_build_time_header()]

    system_parts.append(
        "\n【交流风格】\n"
        "用自然、友好的口语和用户对话，就像朋友聊天一样。\n"
        "⚠️ 必须使用用户输入的语言回复。用户用中文提问就用中文回答，用英文就用英文回答。\n"
    )

    if visual_context:
        is_simple_greeting = any(kw in visual_context for kw in ["挥手", "注视", "走近"])
        if is_simple_greeting:
            system_parts.append(
                "\n【回应风格】\n"
                "用户正在和你互动！请用简短、活泼、口语化的方式回应。\n"
                "打招呼回复控制在 40 字以内，像真人对话一样自然。\n"
                f"\n【视觉观察】\n{visual_context}\n"
            )
        else:
            system_parts.append(
                "\n【回答建议】\n"
                "1. 直接基于视觉观察回答用户问题\n"
                "2. 如果视觉信息不足，诚实告知用户\n"
                "3. 使用自然口语，50-150 字\n"
                f"\n【视觉观察】\n{visual_context}\n"
            )
    else:
        system_parts.append(
            "\n【回答建议】\n"
            "1. 友好、专业地回答用户问题\n"
            "2. 如果信息不足，诚实告知用户\n"
            "3. 使用自然流畅的口语，50-150 字\n"
        )

    if rag_context:
        system_parts.append(f"\n【参考知识】\n{rag_context}\n")

    # 已完成任务上下文注入：让 CHAT 模式能回答关于已完成任务的后续提问
    try:
        from zulong.tools.task_tools import get_active_task_graph
        _chat_tg = get_active_task_graph()
        if _chat_tg is not None:
            _chat_root = _chat_tg.get_node("req")
            _chat_title = _chat_root.label if _chat_root else _chat_tg.title
            _chat_leaves = _chat_tg.get_leaf_nodes()
            _chat_uncompleted = [
                n for n in _chat_leaves
                if n.status not in ("completed", "skipped")
            ]
            if not _chat_uncompleted and _chat_leaves:
                _completed_lines = [
                    f"\n【已完成任务上下文】\n刚刚完成了任务「{_chat_title}」："
                ]
                for _cn in _chat_leaves[:8]:
                    _r_brief = _cn.result[:100] if _cn.result else ""
                    _completed_lines.append(
                        f"- {_cn.label}"
                        + (f"：{_r_brief}" if _r_brief else "")
                    )
                _user_req = _chat_tg.metadata.get("user_requirement", "")
                if _user_req:
                    _completed_lines.append(
                        f"\n用户的原始需求：{_user_req[:200]}"
                    )
                _completed_lines.append(
                    "\n如果用户在询问与这个任务相关的问题，请基于以上信息回答。"
                )
                system_parts.append("\n".join(_completed_lines))
    except Exception:
        pass

    _inject_memory_context(system_parts, user_input, rag_manager, attn_stats)

    system_parts.append(
        "\n⚠️ 语言要求：必须使用与用户输入相同的语言回复。\n"
        "\n请开始回答用户的问题："
    )
    system_prompt = "".join(system_parts)

    logger.info(f"[IntentPrompt] CHAT 系统提示词: {len(system_prompt)} chars")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(user_input)},
    ]


def _build_complex_prompt(user_input: str, rag_context: Optional[str],
                          visual_context: Optional[str], scaffold_data: dict,
                          rag_manager=None, attn_stats: dict = None) -> list:
    """构建 COMPLEX 场景的 messages（~1500 token）

    包含任务创建规则（但不包含恢复规则）。scaffold_data 中包含已创建的任务图信息。
    """
    system_parts = [_build_time_header()]

    system_parts.append(
        "\n【交流风格】\n"
        "用自然、友好的口语和用户对话，就像朋友聊天一样。\n"
        "⚠️ 必须使用用户输入的语言回复。用户用中文提问就用中文回答，用英文就用英文回答。\n"
    )

    # 任务管理规则（仅创建部分，不包含恢复规则）
    system_parts.append(
        "\n【任务管理规则】\n"
        "当前已进入任务规划模式。系统已自动创建任务图骨架。\n"
        "\n"
        "⚠️ 核心原则：不要反问用户！直接根据已有信息开始规划和执行。\n"
        "即使信息不完整，也要基于合理假设直接输出完整方案。\n"
        "\n"
        "你需要做的：\n"
        "1. 用 task_add_node 向任务图添加子任务节点（parent_id='req' 挂到根节点下）\n"
        "   - 每个子节点代表一个独立的工作模块/步骤\n"
        "   - 先搭建完整大纲再执行，不要边做边加\n"
        "2. 用 task_view_overview 查看一次任务概览确认结构（只需查看一次）\n"
        "3. 按顺序逐个执行每个子任务：\n"
        "   a) 调用 task_mark_status(node_id='节点ID', status='in_progress')\n"
        "   b) 生成该子任务的完整内容/结果\n"
        "   c) 调用 task_mark_status(node_id='节点ID', status='completed', result='结果摘要')\n"
        "   d) 立即开始下一个子任务\n"
        "\n"
        "重要规则：\n"
        "- 不需要调用 task_create_plan（任务图已自动创建）\n"
        "- 所有子节点必须通过 parent_id 正确挂到父节点下\n"
        "- task_view_overview 只需要调用一次，不要重复调用\n"
        "- 每完成一个子任务必须调用 task_mark_status 标记为 completed\n"
        "- ⚠️ 在还有未完成的子任务时，绝对不能声称任务已全部完成\n"
        "- ⚠️ 绝对不要向用户反问或要求补充信息，直接基于合理假设开始执行\n"
        "- ⚠️ 工具调用中的 label、desc、result 等字段必须使用与用户相同的语言\n"
    )

    # 信息缺口感知
    system_parts.append(
        "\n【信息视角】\n"
        "执行每个子任务前，检查是否有足够的信息：\n"
        "- 如果需要前置子任务的结果（如数据、分析、接口定义），先确认该子任务已完成\n"
        "- 如果信息不完整，基于常见情况和合理假设直接执行，不要向用户追问\n"
        "- 如果信息充足，直接执行\n"
    )

    # 注入 scaffold 信息
    graph_id = scaffold_data.get("graph_id", "")
    title = scaffold_data.get("title", "")
    graph_lost = scaffold_data.get("graph_lost", False)

    if graph_lost:
        # 任务图数据丢失 → 通知模型需要重新创建
        lost_graph_id = scaffold_data.get("lost_graph_id", "")
        system_parts.append(
            f"\n【重要：任务图数据丢失】\n"
            f"用户引用了之前的任务图（{lost_graph_id}），但该任务图的数据已丢失。\n"
            f"请先告知用户：之前的任务记录已丢失，需要重新创建任务计划。\n"
            f"然后根据用户的描述，重新用 task_create_plan 创建任务图，"
            f"用 task_add_node 添加子任务节点。\n"
        )
    elif scaffold_data.get("already_exists") and graph_id:
        # 任务图已存在（用户在同一任务上追加请求）→ 先查看再修改
        system_parts.append(
            f"\n【当前任务图（已有）】\n"
            f"图谱ID: {graph_id}\n"
            f"任务: {title}\n"
            f"这是一个已经存在的任务图，用户正在此基础上提出新的要求。\n"
            f"请先用 task_view_overview 查看当前任务图的完整状态，\n"
            f"然后根据用户的新需求：\n"
            f"- 用 task_add_node 添加新的子任务节点\n"
            f"- 用 task_mark_status 完成未完成的任务\n"
            f"- 不需要重新创建任务图\n"
        )
    elif graph_id:
        system_parts.append(
            f"\n【当前任务图】\n"
            f"图谱ID: {graph_id}\n"
            f"根节点: req ({title})\n"
            f"请用 task_add_node 开始添加子任务节点。\n"
        )

    if visual_context:
        system_parts.append(f"\n【视觉观察】\n{visual_context}\n")

    if rag_context:
        system_parts.append(f"\n【参考知识】\n{rag_context}\n")

    _inject_memory_context(system_parts, user_input, rag_manager, attn_stats)

    system_parts.append(
        "\n⚠️ 语言要求：用户的输入是中文还是英文，你的所有输出就必须用同一种语言，"
        "包括回复正文、task_add_node 的 label 和 desc、task_mark_status 的 result、"
        "以及 exec_write_file 的内容。绝不可以混用语言。\n"
        "\n请开始规划和执行用户的任务："
    )
    system_prompt = "".join(system_parts)

    logger.info(f"[IntentPrompt] COMPLEX 系统提示词: {len(system_prompt)} chars")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(user_input)},
    ]


def _build_resume_prompt(user_input: str, rag_context: Optional[str],
                         visual_context: Optional[str], scaffold_data: dict,
                         rag_manager=None, attn_stats: dict = None) -> list:
    """构建 RESUME 场景的 messages（~1000 token）

    包含恢复指令和已恢复的任务信息。物理排除 task_create_plan 和 task_add_node。
    """
    system_parts = [_build_time_header()]

    system_parts.append(
        "\n【交流风格】\n"
        "用自然、友好的口语和用户对话，就像朋友聊天一样。\n"
        "⚠️ 必须使用用户输入的语言回复。用户用中文提问就用中文回答，用英文就用英文回答。\n"
    )

    # 任务恢复规则
    task_desc = scaffold_data.get("description", "")
    task_id = scaffold_data.get("task_id", "")
    has_graph = scaffold_data.get("has_task_graph", False)
    already_active = scaffold_data.get("already_active", False)

    title = scaffold_data.get("title", task_desc)
    system_parts.append(
        "\n【任务恢复模式】\n"
        f"系统已自动恢复之前挂起的任务" + (f"「{title}」" if title else "") + "。\n"
    )

    if already_active:
        system_parts.append(
            "⚠️ 当前内存中已有活跃任务图，不要调用 task_list_suspended 搜索挂起任务。\n"
            "直接从 task_view_overview 开始查看进度，然后执行未完成的子任务。\n"
        )

    if has_graph:
        # 直接注入任务进度表，让模型立刻看到哪些没完成
        _progress_summary = ""
        _user_requirement = ""
        _uncompleted = []
        try:
            from zulong.tools.task_tools import get_active_task_graph
            _tg = get_active_task_graph()
            if _tg:
                # 提取用户原始需求（Rule B 在创建时存储的）
                _user_requirement = getattr(_tg, 'metadata', {}).get("user_requirement", "")

                _leaf_nodes = _tg.get_leaf_nodes()
                _completed = [n for n in _leaf_nodes if n.status == "completed"]
                _uncompleted = [n for n in _leaf_nodes if n.status != "completed"]
                _total = len(_leaf_nodes)
                _done = len(_completed)

                _progress_lines = [f"进度: {_done}/{_total} 个工作项已完成。"]

                # 已完成的任务（附带结果摘要）
                if _completed:
                    _progress_lines.append("✅ 已完成：")
                    for _n in _completed:
                        _result_brief = ""
                        if _n.result:
                            _result_brief = f" → {_n.result[:80]}"
                        _progress_lines.append(f"  - {_n.id}: {_n.label}{_result_brief}")

                # 未完成的任务（附带描述信息）
                if _uncompleted:
                    _progress_lines.append("❌ 未完成的任务：")
                    for _n in _uncompleted:
                        _status_text = {"pending": "待开始", "not_started": "待开始",
                                        "in_progress": "进行中", "blocked": "阻塞"}.get(_n.status, _n.status)
                        _desc_brief = ""
                        if _n.desc:
                            _desc_brief = f"\n    描述: {_n.desc[:120]}"
                        _progress_lines.append(f"  - {_n.id}: {_n.label} ({_status_text}){_desc_brief}")
                    _progress_lines.append("")
                    _progress_lines.append(f"⚠️ 请从第一个未完成任务 {_uncompleted[0].id}（{_uncompleted[0].label}）开始执行。")
                else:
                    _progress_lines.append("✅ 所有工作项已完成。")
                _progress_summary = "\n".join(_progress_lines)
        except Exception as _e:
            logger.warning(f"[RESUME] 获取任务进度失败: {_e}")

        system_parts.append(
            "任务图已加载到内存中。\n"
        )

        # 注入用户原始需求（让模型理解任务背景）
        if _user_requirement:
            system_parts.append(
                f"\n【用户原始需求】\n{_user_requirement[:500]}\n"
            )

        if _progress_summary:
            system_parts.append(f"\n【当前任务进度】\n{_progress_summary}\n")

        # 精简指令：4B 模型更易遵循短指令
        _first_uncompleted_id = ""
        _first_uncompleted_label = ""
        if _uncompleted:
            _first_uncompleted_id = _uncompleted[0].id
            _first_uncompleted_label = _uncompleted[0].label

        system_parts.append(
            "\n【执行规则】\n"
            "对每个未完成节点：先调用 task_mark_status(node_id, 'in_progress') 开始，"
            "完成后调用 task_mark_status(node_id, 'completed', result='结果摘要')。\n"
        )
        if _first_uncompleted_id:
            system_parts.append(
                f"第一步：请立即调用 task_mark_status(node_id='{_first_uncompleted_id}', status='in_progress')\n"
            )
        system_parts.append(
            "⚠️ 未完成子任务时，不能声称任务已全部完成。\n"
        )
    else:
        system_parts.append(
            "注意：该任务没有关联的任务图，请根据上下文继续处理。\n"
        )

    if visual_context:
        system_parts.append(f"\n【视觉观察】\n{visual_context}\n")

    if rag_context:
        system_parts.append(f"\n【参考知识】\n{rag_context}\n")

    _inject_memory_context(system_parts, user_input, rag_manager, attn_stats)

    system_parts.append(
        "\n⚠️ 语言要求：必须使用与用户输入相同的语言回复，包括工具调用中的所有字段。\n"
        "\n请继续执行之前的任务："
    )
    system_prompt = "".join(system_parts)

    logger.info(f"[IntentPrompt] RESUME 系统提示词: {len(system_prompt)} chars")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(user_input)},
    ]


def build_round2_system_prompt(intent_type: IntentType, user_input: str,
                               rag_context: Optional[str],
                               visual_context: Optional[str],
                               scaffold_data: dict,
                               rag_manager=None,
                               attn_stats: dict = None) -> list:
    """构建 Round 2 场景化系统提示词

    根据 Round 1 分类结果构建对应场景的 messages 列表。

    Args:
        intent_type: Round 1 分类结果
        user_input: 用户原始输入
        rag_context: RAG 检索到的上下文
        visual_context: 视觉上下文
        scaffold_data: Round 1 scaffold 操作的返回数据
        rag_manager: RAGManager 实例（用于 MemoryGraph 注入）
        attn_stats: AttentionWindowManager.stats 字典（可选，用于容量仪表盘）

    Returns:
        构建好的 messages 列表 [{"role": "system", ...}, {"role": "user", ...}]
    """
    if intent_type == IntentType.CHAT:
        return _build_chat_prompt(user_input, rag_context, visual_context, rag_manager, attn_stats)
    elif intent_type == IntentType.COMPLEX:
        return _build_complex_prompt(user_input, rag_context, visual_context, scaffold_data, rag_manager, attn_stats)
    elif intent_type == IntentType.RESUME:
        return _build_resume_prompt(user_input, rag_context, visual_context, scaffold_data, rag_manager, attn_stats)
    else:
        logger.warning(f"[IntentPrompt] 未知意图类型 {intent_type}，降级为 CHAT")
        return _build_chat_prompt(user_input, rag_context, visual_context, rag_manager, attn_stats)
