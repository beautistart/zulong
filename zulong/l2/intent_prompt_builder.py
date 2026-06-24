# File: zulong/l2/intent_prompt_builder.py
# 统一主链提示词构建器
#
# 当前 TSD 主链:
# 用户事件 -> L1-B（BFS/记忆预检索 + 输出模态 + ALBERT辅助 + 工具预判 + 上下文打包）-> L2（单次决策）
#
# 本模块只消费 L1-B 注入的工具包、任务图策略和上下文信号。
# L2 据此决定直接回答、调用已有工具，或在工具增强轮次中自主补充工具。

import logging
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def _build_time_header() -> str:
    """构建时间和身份头部。"""
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


def _build_environment_header(runtime_context: Optional[Dict[str, Any]] = None) -> str:
    try:
        from zulong.utils.runtime_env import get_runtime_context
        detected_context = get_runtime_context()
    except Exception:
        detected_context = {}

    runtime_context = {**detected_context, **(runtime_context or {})}
    workspace_root = runtime_context.get("workspace_root") or "未提供工作区"
    shell_name = runtime_context.get("shell") or os.environ.get("SHELL") or os.environ.get("COMSPEC") or "unknown"
    os_name = runtime_context.get("os_name") or ("Windows" if os.name == "nt" else os.name)
    preferred_commands = runtime_context.get("preferred_commands") or ["rg", "python", "npm", "git"]
    command_guidance = runtime_context.get("command_guidance") or "运行命令必须符合当前操作系统和 Shell。"
    return (
        "\n【运行环境】\n"
        f"- 操作系统: {os_name}\n"
        f"- Shell: {shell_name}\n"
        f"- 工作区根目录: {workspace_root}\n"
        f"- 推荐命令: {', '.join(preferred_commands)}\n"
        f"- 命令风格: {command_guidance}\n"
        "- 代码阅读/架构分析默认可直接使用只读文件/代码工具；只有 LLM 判断本轮确属复杂、持续、多步骤任务时，才调用 task_create_plan 创建/绑定任务图根节点。\n"
        "- 若已经决定创建代码/任务图谱，可用 task_add_node 添加代码分析子节点，再调用 index_project(root_dir=...)。\n"
        "- 已有活跃任务图时，在该图下调用 index_project；之后用 search_code_symbols(query=...)、zulong_code_query(file_path=...) 继续分析。\n"
        "- 如果结构化代码工具因解析器/索引限制失败，优先改用 read_file(file_path=...) 只读读取源码，再继续分析。\n"
        "- 若需要运行命令，再调用 exec_run_command，且命令必须符合当前 shell。\n"
    )


def _inject_memory_context(
    system_parts: list,
    user_input: str,
    rag_manager=None,
    attn_stats: Optional[dict] = None,
    pre_retrieved_memory: Optional[str] = None,
) -> None:
    """只消费 L1-B 预检索记忆，不在 L2 侧主动检索 MemoryGraph。"""
    if pre_retrieved_memory:
        logger.info("[MemoryGraph] 使用 L1-B 预检索记忆 (%d 字符)", len(pre_retrieved_memory))
        system_parts.append("\n【记忆上下文】\n" + pre_retrieved_memory + "\n")
    else:
        logger.debug("[MemoryGraph] L1-B 未提供预检索记忆；L2 不主动检索")

    memory_count = sum(1 for part in system_parts if "【记忆上下文】" in part or "【历史对话】" in part)
    attn_lines = ["\n【注意力状态】"]
    if attn_stats:
        mode = attn_stats.get("mode", "global")
        ratio = attn_stats.get("usage_ratio", 0)
        remaining = attn_stats.get("remaining_tokens", 0)
        bar_len = 10
        filled = int(ratio * bar_len)
        bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
        attn_lines.append(f"容量: [{bar}] {ratio:.0%}  剩余≈{remaining}tok  模式={mode}")
        if ratio >= 0.85:
            attn_lines.append("容量紧张，请精简输出，避免冗长工具调用。")
    else:
        attn_lines.append("容量: 未启用窗口管理")

    if memory_count:
        attn_lines.append(f"已注入 {memory_count} 段记忆/上下文。")
        attn_lines.append("请优先基于已注入的记忆/上下文直接回答，不要为了闲聊继续检索。")
    else:
        attn_lines.append("当前对话未注入任何记忆上下文。")
        attn_lines.append("如果信息不足，请直接说明或向用户追问。")
    attn_lines.append("如果需要用户补充信息才能继续，请直接用自然语言向用户提问。\n")
    system_parts.append("\n".join(attn_lines))


def _append_completed_task_context(system_parts: list) -> None:
    """注入刚完成任务的摘要，供后续追问使用。"""
    try:
        from zulong.tools.task_tools import get_active_task_graph

        task_graph = get_active_task_graph()
        if task_graph is None:
            return
        root = task_graph.get_node("req")
        title = root.label if root else task_graph.title
        leaves = task_graph.get_leaf_nodes()
        uncompleted = [n for n in leaves if n.status not in ("completed", "skipped")]
        if uncompleted or not leaves:
            return

        lines = [f"\n【已完成任务上下文】\n刚刚完成了任务「{title}」："]
        for node in leaves[:8]:
            result = node.result[:100] if node.result else ""
            lines.append(f"- {node.label}" + (f"：{result}" if result else ""))
        user_requirement = task_graph.metadata.get("user_requirement", "")
        if user_requirement:
            lines.append(f"\n用户的原始需求：{user_requirement[:200]}")
        lines.append("\n如果用户在询问与这个任务相关的问题，请基于以上信息回答。")
        system_parts.append("\n".join(lines))
    except Exception:
        pass


def _append_active_task_context(system_parts: list, task_graph_policy: str) -> None:
    """按 L1-B 任务图策略注入任务图执行提示。"""
    if task_graph_policy in ("", "none", None):
        return

    try:
        from zulong.tools.task_tools import get_active_task_graph

        task_graph = get_active_task_graph()
    except Exception:
        task_graph = None

    system_parts.append(
        "\n【任务图策略】\n"
        f"- L1-B 建议策略: {task_graph_policy}\n"
        "- 这是执行策略信号，不是用户意图分类标签。\n"
        "- 如果需要理解现有任务图，优先调用 task_view_overview。\n"
        "- 如果需要创建/扩展任务图，先确保确有多步骤工作，再调用 task_* 工具。\n"
        "- 如果工具包不足，请调用 request_tool_supplement 申请补充工具。\n"
    )

    if not task_graph:
        system_parts.append(
            "\n当前未加载活跃任务图。若用户请求确实需要持续执行，"
            "可由 LLM 自主选择 task_create_plan 创建任务图，再用 task_add_node 分解工作；"
            "普通问答/原因分析/规则解释不要创建任务图。\n"
        )
        return

    try:
        root = task_graph.get_node("req")
        title = root.label if root else task_graph.title
        leaves = task_graph.get_leaf_nodes()
        completed = [n for n in leaves if n.status == "completed"]
        uncompleted = [n for n in leaves if n.status != "completed"]
        lines = [
            "\n【当前任务图摘要】",
            f"任务: {title}",
            f"进度: {len(completed)}/{len(leaves)} 个工作项已完成。",
        ]
        if uncompleted:
            lines.append("未完成工作项：")
            for node in uncompleted[:8]:
                status_text = {
                    "pending": "待开始",
                    "not_started": "待开始",
                    "in_progress": "进行中",
                    "blocked": "阻塞",
                }.get(node.status, node.status)
                lines.append(f"- {node.id}: {node.label} ({status_text})")
            lines.append("仍有未完成工作项时，不能声称任务已全部完成。")
        elif leaves:
            lines.append("所有工作项已完成；如果用户提出新需求，请判断是追问、扩展，还是全新任务。")
        system_parts.append("\n".join(lines) + "\n")
    except Exception as exc:
        logger.debug("[Prompt] 任务图摘要注入失败: %s", exc)


def build_unified_system_prompt(
    user_input: str,
    rag_context: Optional[str],
    visual_context: Optional[str],
    scaffold_data: Optional[Dict[str, Any]] = None,
    rag_manager=None,
    attn_stats: Optional[dict] = None,
    voice_mode: str = "TEXT_ONLY",
    pre_retrieved_memory: Optional[str] = None,
    runtime_context: Optional[Dict[str, Any]] = None,
    tool_prediction: Optional[Dict[str, Any]] = None,
    tool_bundle: Optional[List[str]] = None,
    task_graph_policy: Optional[str] = None,
) -> list:
    """构建统一主链系统提示词。

    该函数只消费 L1-B 打包结果，不执行或表达会话意图分类。
    """
    scaffold_data = scaffold_data or {}
    tool_prediction = tool_prediction or {}
    tool_bundle = list(tool_bundle or scaffold_data.get("tools") or [])
    context_bundle = (
        scaffold_data.get("context_bundle")
        or tool_prediction.get("context_bundle")
        or {}
    )
    task_graph_policy = task_graph_policy or scaffold_data.get("policy") or "none"
    turn_shape = context_bundle.get("turn_shape", "")
    l1b_context_pack = scaffold_data.get("l1b_context_pack") or ""
    enable_voice_hint = voice_mode in ("AUTO_TTS", "FORCED_TTS")

    system_parts = [_build_time_header()]
    system_parts.append(_build_environment_header(runtime_context))

    system_parts.append(
        "\n【统一主链】\n"
        "- 本轮已经由 L1-B 同步完成 BFS 会话自恢复、MemoryGraph 预检索、输出模态辅助、ALBERT 辅助分类、工具预判和上下文打包。\n"
        "- 你是 L2 推理层：直接基于 L1-B 任务包进行第一次模型决策。\n"
        "- 只有当上下文不足或工具不足时，才通过已暴露工具进行增量补充；不要在决策前自行构建上下文。\n"
        "- 不要输出内部路由标签或状态标签。\n"
        "- 主回答必须由模型生成，不能依赖固定回复模板。\n"
    )

    system_parts.append(
        "\n【交流风格】\n"
        "用自然、友好的口语和用户对话。\n"
        "必须使用用户输入的语言回复。用户用中文提问就用中文回答，用英文就用英文回答。\n"
    )

    if enable_voice_hint:
        system_parts.append(
            "\n【语音功能】\n"
            "你拥有 TTS 语音合成功能。系统会根据 L1-B 的 voice_mode 将文字回复转换为语音。\n"
            "如果用户明确要求语音回复，可以自然回应并继续生成正常内容。\n"
        )

    system_parts.append(
        "\n【执行规则】\n"
        "1. 普通问答、确认、简短说明可以直接自然语言回复。\n"
        "2. 需要读取项目代码时，优先使用代码/文件工具，不要先用 shell 穷举目录。\n"
        "3. 需要多步骤持续执行时，再使用 task_* 工具创建、复用或推进任务图。\n"
        "4. 运行命令必须符合【运行环境】中声明的当前操作系统和 Shell。\n"
        "5. 工具调用中的 label、desc、result 等字段必须使用与用户相同的语言。\n"
        "6. 内容型子任务写入工作目录时，优先使用相对 file_path。\n"
        "7. 如信息不足，先调用记忆/任务/工具补充类工具增量补齐；确实无法继续时再向用户追问。\n"
        "8. ⚠️ **先说再做**: 准备调用工具执行任务时，必须先给用户一句普通可见的步骤说明——只说本步将做什么，不输出推理过程，不写\"我在思考/我会分析\"。如果工具调用格式允许 assistant.content，请把这句话放在 assistant.content；如果当前模型/Provider 不保留 tool_calls 同轮 content，请先调用 announce_step(message=...) 再调用实际工具。\n"
    )

    if l1b_context_pack:
        system_parts.append("\n" + l1b_context_pack + "\n")

    if scaffold_data.get("graph_lost"):
        lost_graph_id = scaffold_data.get("lost_graph_id", "")
        system_parts.append(
            "\n【任务图缺失】\n"
            f"用户引用的任务图 {lost_graph_id} 未能从内存、挂起任务或备份中恢复。\n"
            "请说明该任务记录不可用，并根据用户当前描述重新建立可执行计划。\n"
        )

    if visual_context:
        system_parts.append(f"\n【视觉观察】\n{visual_context}\n")

    if rag_context:
        system_parts.append(f"\n【参考知识】\n{rag_context}\n")

    if not l1b_context_pack:
        _inject_memory_context(
            system_parts,
            user_input,
            rag_manager=rag_manager,
            attn_stats=attn_stats,
            pre_retrieved_memory=pre_retrieved_memory,
        )

    system_parts.append(
        "\n⚠️ 语言要求：必须使用与用户输入相同的语言回复。\n"
        "\n请根据以上上下文开始回答或执行："
    )
    system_prompt = "".join(system_parts)

    logger.info(
        "[UnifiedPrompt] 系统提示词: %d chars, tools=%d, policy=%s, turn_shape=%s",
        len(system_prompt),
        len(tool_bundle),
        task_graph_policy,
        turn_shape or "unknown",
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(user_input)},
    ]
