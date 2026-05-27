# File: zulong/l2/intent_prompt_builder.py
# 统一主链提示词构建器
#
# 当前 TSD 主链:
# 用户事件 -> L1-B（工具预判 + 上下文/记忆检索 + 工具包打包）-> L2（推理）
#
# 本模块只消费 L1-B 注入的工具包、任务图策略和上下文信号。
# L2 据此决定直接回答、调用已有工具，或在工具增强轮次中自主补充工具。

import asyncio
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
    runtime_context = runtime_context or {}
    workspace_root = runtime_context.get("workspace_root") or "未提供工作区"
    shell_name = runtime_context.get("shell") or os.environ.get("SHELL") or "PowerShell"
    os_name = runtime_context.get("os_name") or ("Windows" if os.name == "nt" else os.name)
    preferred_commands = runtime_context.get("preferred_commands") or [
        "Get-ChildItem",
        "Select-String",
        "Get-Content",
        "rg",
        "python",
        "npm",
        "git",
    ]
    return (
        "\n【运行环境】\n"
        f"- 操作系统: {os_name}\n"
        f"- Shell: {shell_name}\n"
        f"- 工作区根目录: {workspace_root}\n"
        f"- 推荐命令: {', '.join(preferred_commands)}\n"
        "- 当前环境优先使用 Windows/PowerShell 命令，不要生成 Unix 专属命令（如 `find / -name`、`ls -la`、`pwd && ...`、`2>/dev/null`）。\n"
        "- 代码阅读/架构分析优先使用代码工具：index_project(root_dir=...)、search_code_symbols(query=...)、zulong_code_query(file_path=...)。\n"
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
    """注入 MemoryGraph 上下文和注意力状态。"""
    try:
        from zulong.memory.memory_graph import get_memory_graph as _get_mg_nav

        _mg_nav = _get_mg_nav()
        if _mg_nav:
            focus_summary = _mg_nav.get_focus_path_summary()
            if focus_summary:
                system_parts.append(f"\n{focus_summary}\n")
                logger.debug("[Prompt] 已注入焦点路径 (%d chars)", len(focus_summary))
    except Exception as e:
        logger.debug("[Prompt] 焦点路径注入跳过: %s", e)

    if pre_retrieved_memory:
        logger.info("[MemoryGraph] 使用 L1-B 预检索记忆 (%d 字符)", len(pre_retrieved_memory))
        system_parts.append("\n【记忆上下文】\n" + pre_retrieved_memory + "\n")
    else:
        try:
            from zulong.memory.memory_graph import get_memory_graph

            _mg = get_memory_graph()
            if _mg:
                if not getattr(_mg, "_rag_manager", None) and rag_manager:
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
                    return asyncio.run(coro)

                _usage_ratio = 0.0
                _context_window_size = 131072
                if attn_stats:
                    _usage_ratio = attn_stats.get("usage_ratio", 0.0)
                    _context_window_size = attn_stats.get("context_window_size", 131072)

                if hasattr(_mg, "retrieve_context_dynamic"):
                    mg_results = _run_async_bridge(
                        _mg.retrieve_context_dynamic(
                            user_input,
                            context_window_size=_context_window_size,
                            usage_ratio=_usage_ratio,
                            session_id="",
                        )
                    )
                else:
                    _top_k = 3 if _usage_ratio > 0.8 else 5 if _usage_ratio > 0.6 else 8
                    mg_results = _run_async_bridge(
                        _mg.retrieve_context(user_input, top_k=_top_k, session_id="")
                    )

                if mg_results:
                    remaining_ratio = max(0.0, 1.0 - _usage_ratio)
                    content_limit = max(100, min(500, int(500 * remaining_ratio)))
                    memory_sections = []
                    for result in mg_results:
                        node_type = result.get("node_type", "")
                        content = result.get("content", "")
                        label = result.get("label", "")
                        if not content:
                            continue
                        if node_type == "experience":
                            continue
                        if node_type == "dialogue":
                            memory_sections.append(f"【历史对话】{content[:content_limit]}")
                        elif node_type == "task":
                            status = result.get("metadata", {}).get("status", "")
                            memory_sections.append(
                                f"【相关任务】{label}" + (f"（状态：{status}）" if status else "")
                            )
                        elif node_type == "knowledge":
                            memory_sections.append(f"【知识参考】{content[:content_limit]}")
                        elif node_type == "episode":
                            memory_sections.append(f"【历史摘要】{content[:content_limit]}")
                        elif node_type in ("person", "concept"):
                            memory_sections.append(f"【知识参考】{label}: {content[:content_limit]}")
                        else:
                            memory_sections.append(f"【参考】{content[:content_limit]}")

                    if memory_sections:
                        system_parts.append("\n【记忆上下文】\n" + "\n".join(memory_sections) + "\n")
                        logger.info("[MemoryGraph] 注入 %d 条记忆到上下文", len(memory_sections))
        except Exception as e:
            logger.warning("[MemoryGraph] 记忆检索失败，降级跳过: %s", e)

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
            "请用 task_create_plan 创建任务图，再用 task_add_node 分解工作。\n"
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
    simple_social = turn_shape == "simple_social" and not tool_bundle and task_graph_policy in ("", "none", None)
    enable_voice_hint = voice_mode in ("AUTO_TTS", "FORCED_TTS")

    system_parts = [_build_time_header()]
    if not simple_social:
        system_parts.append(_build_environment_header(runtime_context))

    if simple_social:
        system_parts.append(
            "\n【统一主链】\n"
            "- 本轮已经由 L1-B 完成工具预判、上下文检索和任务打包。\n"
            "- L1-B 判定当前轮次无需工具，你只需要进行自然语言推理并生成回复。\n"
            "- 不要输出内部路由标签或状态标签。\n"
            "- 主回答必须由模型生成，不能依赖固定回复模板。\n"
        )
    else:
        system_parts.append(
            "\n【统一主链】\n"
            "- 本轮已经由 L1-B 完成工具预判、上下文检索和工具包打包。\n"
            "- 你是 L2 推理层：根据当前消息、上下文和工具包自主决定直接回答、调用工具或请求补充工具。\n"
            "- 不要输出内部路由标签或状态标签。\n"
            "- 主回答必须由模型生成，不能依赖固定回复模板。\n"
        )

    system_parts.append(
        "\n【交流风格】\n"
        "用自然、友好的口语和用户对话。\n"
        "必须使用用户输入的语言回复。用户用中文提问就用中文回答，用英文就用英文回答。\n"
    )

    if simple_social:
        system_parts.append(
            "\n【轻量寒暄约束】\n"
            "用户只是打招呼或寒暄时，直接自然回应即可。\n"
            "不要要求用户提供具体任务，不要把寒暄改写成任务需求，不要调用工具。\n"
            "回复保持简短、自然、有人味。\n"
        )

    if enable_voice_hint:
        system_parts.append(
            "\n【语音功能】\n"
            "你拥有 TTS 语音合成功能。系统会根据 L1-B 的 voice_mode 将文字回复转换为语音。\n"
            "如果用户明确要求语音回复，可以自然回应并继续生成正常内容。\n"
        )

    if tool_bundle:
        system_parts.append(
            "\n【L1-B 工具包】\n"
            f"- 建议工具: {', '.join(tool_bundle)}\n"
            "- 这些工具只是预判结果，不代表必须全部调用。\n"
            "- 如果要完成用户请求还缺工具，调用 request_tool_supplement 补充；如果无需工具，直接回答。\n"
        )
    elif not simple_social:
        system_parts.append(
            "\n【工具使用】\n"
            "L1-B 未建议具体工具。若当前消息可直接回答，请直接回答；不要为了形式调用工具。\n"
        )

    if not simple_social:
        system_parts.append(
            "\n【执行规则】\n"
            "1. 普通问答、确认、简短说明可以直接自然语言回复。\n"
            "2. 需要读取项目代码时，优先使用代码/文件工具，不要先用 shell 穷举目录。\n"
            "3. 需要多步骤持续执行时，再使用 task_* 工具创建、复用或推进任务图。\n"
            "4. 运行命令必须符合当前 Windows/PowerShell 环境。\n"
            "5. 工具调用中的 label、desc、result 等字段必须使用与用户相同的语言。\n"
            "6. 内容型子任务写入工作目录时，优先使用相对 file_path。\n"
            "7. 如信息不足，先用已有上下文和工具补齐；确实无法继续时再向用户追问。\n"
        )

    if not simple_social:
        _append_active_task_context(system_parts, task_graph_policy or "none")
        _append_completed_task_context(system_parts)

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

    if simple_social:
        system_parts.append(
            "\n【注意力状态】\n"
            "轻量寒暄轮次不注入历史记忆，避免旧任务或旧回复污染当前问候。\n"
        )
    else:
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
