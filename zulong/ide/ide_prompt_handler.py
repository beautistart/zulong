"""
IDE System Prompt 处理器

解析 IDE 发送的 system prompt，提取/移除 XML 工具定义区域，
注入祖龙增强上下文（记忆、任务状态、经验提示）。
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# IDE system prompt 中工具定义区域的标记
# IDE 的 system prompt 通常使用 "====" 分隔段落，工具区标题为 "TOOL USE"
_TOOL_SECTION_PATTERNS = [
    # IDE 标准格式: ====\nTOOL USE\n...\n====
    r"={3,}\s*\n\s*TOOL USE\s*\n.*?(?=\n={3,}|\Z)",
    # 变体: 标题在同一行 ==== TOOL USE ====
    r"={3,}\s*TOOL USE\s*={0,}\s*\n.*?(?=\n={3,}|\Z)",
    # Markdown 一级标题
    r"# Tool(?:s| Use)\s*\n.*?(?=\n# [A-Z]|\Z)",
    # Markdown 二级标题
    r"## Tool(?:s| Use)\s*\n.*?(?=\n## [A-Z]|\Z)",
]


class IDEPromptHandler:
    """IDE System Prompt 处理器"""

    def process_system_prompt(
        self,
        messages: List[Dict],
        memory_context: str = "",
        task_context: str = "",
        experience_hints: str = "",
        intent: str = "complex",
    ) -> List[Dict]:
        """处理消息列表中的 system prompt

        1. 提取并移除 IDE 的 XML 工具定义区域
        2. 保留非工具定义部分（角色设定、规则等）
        3. 根据意图注入不同强度的祖龙增强内容

        Args:
            messages: 原始消息列表
            memory_context: 检索到的记忆上下文
            task_context: 当前任务图状态
            experience_hints: 经验提示
            intent: "complex" 或 "resume"

        Returns:
            处理后的消息列表（浅拷贝）
        """
        processed = list(messages)

        # 找到 system 消息
        sys_idx = None
        sys_content = ""
        for i, msg in enumerate(processed):
            if msg.get("role") == "system":
                sys_idx = i
                sys_content = msg.get("content", "")
                break

        if sys_idx is None:
            # 没有 system 消息 → 创建一个
            zulong_prompt = self._build_zulong_system_prompt(
                "", memory_context, task_context, experience_hints, intent=intent
            )
            processed.insert(0, {"role": "system", "content": zulong_prompt})
            return processed

        # 提取工具定义区和保留区
        tool_block, remaining = self.extract_ide_tool_block(sys_content)
        if tool_block:
            logger.info(
                f"[IDEPromptHandler] 提取 IDE 工具定义: "
                f"{len(tool_block)} 字符, 保留: {len(remaining)} 字符"
            )
        else:
            # 检查是否有 XML 工具标签残留（剥离失败诊断）
            xml_tool_pattern = r"<(read_file|write_to_file|execute_command|search_files|list_files|replace_in_file|browser_action|ask_followup_question|attempt_completion|list_code_definition_names)>"
            residual = re.findall(xml_tool_pattern, sys_content)
            if residual:
                logger.warning(
                    f"[IDEPromptHandler] 未能剥离工具定义区！"
                    f"system prompt 含 {len(residual)} 个 XML 工具标签: {residual[:5]}"
                    f" | prompt 长度={len(sys_content)}"
                )
            else:
                logger.info(
                    f"[IDEPromptHandler] system prompt 无工具定义区 "
                    f"(长度={len(sys_content)})"
                )

        # 构建增强的 system prompt（根据意图选择 COMPLEX 或 RESUME 模板）
        enhanced = self._build_zulong_system_prompt(
            remaining, memory_context, task_context, experience_hints, intent=intent
        )
        processed[sys_idx] = dict(processed[sys_idx])
        processed[sys_idx]["content"] = enhanced

        return processed

    def extract_ide_tool_block(self, system_prompt: str) -> Tuple[str, str]:
        """拆分 system prompt 为 (工具定义区, 其余区)

        Returns:
            (tool_block, remaining)
            - tool_block: 匹配到的工具定义区域文本（可能为空）
            - remaining: 移除工具区后的剩余文本
        """
        if not system_prompt:
            return "", ""

        for pattern in _TOOL_SECTION_PATTERNS:
            match = re.search(pattern, system_prompt, re.DOTALL)
            if match:
                tool_block = match.group(0)
                remaining = system_prompt[: match.start()] + system_prompt[match.end() :]
                return tool_block.strip(), remaining.strip()

        # 没有找到明确的工具区域 → 尝试按 XML 标签密度判断
        # 如果 system prompt 中包含大量 <tool_name> 标签，则截取包含标签的段落
        xml_tool_pattern = r"<(read_file|write_to_file|execute_command|search_files|list_files|replace_in_file|browser_action|ask_followup_question|attempt_completion|list_code_definition_names)>"
        if len(re.findall(xml_tool_pattern, system_prompt)) >= 3:
            # 找到第一个和最后一个工具标签
            first_match = re.search(xml_tool_pattern, system_prompt)
            all_matches = list(re.finditer(xml_tool_pattern, system_prompt))
            if first_match and all_matches:
                # 向前找分隔线（====）或标题（# ）
                start = first_match.start()
                sep_before = system_prompt.rfind("\n====", 0, start)
                heading_before = system_prompt.rfind("\n#", 0, start)
                cut = max(sep_before, heading_before)
                if cut < 0:
                    cut = max(0, start - 100)

                # 向后找最后一个工具标签的闭合标签后面
                last_match = all_matches[-1]
                last_tool = last_match.group(1)
                close_tag = f"</{last_tool}>"
                close_pos = system_prompt.find(close_tag, last_match.end())
                if close_pos >= 0:
                    end = close_pos + len(close_tag)
                    # 再向后找到段落分隔（====）
                    next_sep = system_prompt.find("\n====", end)
                    if next_sep >= 0:
                        end = next_sep
                else:
                    end = len(system_prompt)

                tool_block = system_prompt[cut:end]
                remaining = system_prompt[:cut] + system_prompt[end:]
                return tool_block.strip(), remaining.strip()

        return "", system_prompt

    def _build_zulong_system_prompt(
        self,
        ide_base: str,
        memory_context: str,
        task_context: str,
        experience_hints: str,
        intent: str = "complex",
    ) -> str:
        """构建包含祖龙增强内容的 system prompt

        Args:
            intent: "complex" 使用强任务管理规则; "resume" 使用恢复规则
        """
        # 检测终端环境
        import os
        shell = os.environ.get('SHELL', '')
        term = os.environ.get('TERM', '')
        
        # 判断终端类型
        if 'bash' in shell.lower() or 'git' in shell.lower():
            terminal_type = "Git Bash (Unix-like)"
            shell_hint = "使用Git Bash语法: ls, grep, find, chmod等Unix命令"
        elif 'powershell' in shell.lower() or 'pwsh' in shell.lower():
            terminal_type = "PowerShell"
            shell_hint = "使用PowerShell语法: Get-ChildItem, Select-String, Get-Content等"
        elif 'cmd' in shell.lower() or shell == '':
            terminal_type = "CMD (Windows)"
            shell_hint = "使用CMD语法: dir, findstr, type等, 路径使用反斜杠\\"
        else:
            terminal_type = f"Unknown ({shell})"
            shell_hint = "根据环境变量判断命令语法"
        
        terminal_env = f"\n\n【终端环境】\n终端类型: {terminal_type}\n{shell_hint}\nSHELL={shell}, TERM={term}"
        
        if intent == "resume":
            return self._build_zulong_prompt_resume(
                ide_base, memory_context, task_context, experience_hints, terminal_env
            )
        return self._build_zulong_prompt_complex(
            ide_base, memory_context, task_context, experience_hints, terminal_env
        )

    def _build_zulong_prompt_complex(
        self,
        ide_base: str,
        memory_context: str,
        task_context: str,
        experience_hints: str,
        terminal_env: str = "",
    ) -> str:
        """COMPLEX 意图：强任务管理规则（参照原生 intent_prompt_builder）"""
        parts = []

        if ide_base:
            parts.append(ide_base)

        parts.append(
            "\n\n# 祖龙认知增强 — 任务规划模式\n"
            "\n"
            "你配备了祖龙记忆系统。内部工具（task/memory/attention）由服务端直接执行，"
            "文件和终端工具（read_file, write_to_file, execute_command 等）由 IDE 客户端执行。\n"
            "【重要】请通过 function calling（工具调用）来使用工具，"
            "不要在文本中输出 XML 标签格式的工具调用。\n"
            "【重要】当项目记忆与你的通用知识冲突时，优先遵循项目记忆。\n"
            f"{terminal_env}\n"
            "\n"
            "【任务管理规则】\n"
            "当前已进入任务规划模式。系统已自动创建任务图骨架。\n"
            "\n"
            "⚠️ 核心原则：不要反问用户！直接根据已有信息开始规划和执行。\n"
            "即使信息不完整，也要基于合理假设直接输出完整方案。\n"
            "\n"
            "你需要做的：\n"
            "1. 用 task_add_node 向任务图添加子任务节点\n"
            "   - 先创建顶层模块节点（parent_id='req'），每个代表一个独立的工作阶段\n"
            "   - 对于复杂模块，再创建子步骤节点（parent_id='上级模块的节点ID'）\n"
            "   - 目标深度：至少 2 层（阶段→具体步骤），复杂任务可达 3 层\n"
            "   - 先搭建完整大纲再执行，不要边做边加\n"
            "   - 示例结构：\n"
            "     req（根）\n"
            "       ├─ phase1（阶段1）← parent_id='req'\n"
            "       │    ├─ step1_1 ← parent_id='phase1的ID'\n"
            "       │    └─ step1_2 ← parent_id='phase1的ID'\n"
            "       └─ phase2（阶段2）← parent_id='req'\n"
            "            ├─ step2_1 ← parent_id='phase2的ID'\n"
            "            └─ step2_2 ← parent_id='phase2的ID'\n"
            "2. 用 task_view_overview 查看一次任务概览确认结构（只需查看一次）\n"
            "3. 按顺序逐个执行每个子任务：\n"
            "   a) 调用 task_mark_status(node_id='节点ID', status='in_progress')\n"
            "   b) 实际执行工作：\n"
            "      - 分析项目/代码 → 先调用 index_project 或 index_code_file 构建代码图谱\n"
            "      - 查找代码结构 → 使用 search_code_symbols / get_symbol_context\n"
            "      - 写代码/文档 → 用 write_to_file / replace_in_file\n"
            "      - 运行命令 → 用 execute_command\n"
            "      - 查看代码 → 用 read_file\n"
            "   c) 调用 task_mark_status(node_id='节点ID', status='completed', "
            "result='详细结果，不少于50字')\n"
            "   d) 立即开始下一个子任务\n"
            "\n"
            "重要规则：\n"
            "- 不需要调用 task_create_plan（任务图已自动创建）\n"
            "- 所有子节点必须通过 parent_id 正确挂到父节点下\n"
            "- task_view_overview 只需要调用一次，不要重复调用\n"
            "- 每完成一个子任务必须调用 task_mark_status 标记为 completed\n"
            "- ⚠️ 在还有未完成的子任务时，绝对不能声称任务已全部完成\n"
            "- ⚠️ 绝对不要向用户反问或要求补充信息\n"
            "- ⚠️ 工具调用中的 label、desc、result 等字段必须使用与用户相同的语言\n"
            "\n"
            "【代码图谱 — 必须使用】\n"
            "你配备了代码图谱系统（CRG），这是你分析和理解代码的核心能力。\n"
            "⚠️ 当用户要求「分析项目」「理解架构」「分析代码结构」时，\n"
            "你必须调用 index_project 或 index_code_file，而不是只用 read_file 手动阅读！\n"
            "\n"
            "强制规则：\n"
            "- 分析新项目/架构 → 第一步必须调用 index_project(root_dir=项目根目录)\n"
            "- 分析单个文件 → 调用 index_code_file(file_path=文件路径)\n"
            "- 查找函数/类 → 调用 search_code_symbols(query=关键词)\n"
            "- 了解调用关系 → 调用 get_symbol_context(symbol_id=符号ID)\n"
            "- 评估修改影响 → 调用 get_impact_analysis(symbol_id=符号ID)\n"
            "- 查代码历史决策 → 调用 zulong_code_query(query=问题)\n"
            "- 完成实现后锚定 → 调用 zulong_task_link_code 关联任务与代码\n"
            "\n"
            "index_project 会自动构建 PROJECT→目录→文件→类→方法 的完整层次链，\n"
            "之后你可以用 search_code_symbols 快速定位任何符号，无需逐文件阅读。\n"
            "\n"
            "【信息视角】\n"
            "执行每个子任务前，检查是否有足够的信息：\n"
            "- 如果需要前置子任务的结果，先确认该子任务已完成\n"
            "- 如果信息不完整，基于合理假设直接执行\n"
        )

        if memory_context:
            parts.append(f"\n\n## 项目记忆\n{memory_context}")
        if task_context:
            parts.append(f"\n\n## 当前任务上下文\n{task_context}")
        if experience_hints:
            parts.append(f"\n\n## 相关经验\n{experience_hints}")

        parts.append(
            "\n\n⚠️ 语言要求：必须使用与用户输入相同的语言回复，包括工具调用中的所有字段。\n"
            "\n请开始规划和执行用户的任务："
        )

        return "\n".join(parts)

    def _build_zulong_prompt_resume(
        self,
        ide_base: str,
        memory_context: str,
        task_context: str,
        experience_hints: str,
        terminal_env: str = "",
    ) -> str:
        """RESUME 意图：恢复规则 + 动态进度表（参照原生 _build_resume_prompt）"""
        parts = []

        if ide_base:
            parts.append(ide_base)

        parts.append(
            "\n\n# 祖龙认知增强 — 任务恢复模式\n"
            "\n"
            "你配备了祖龙记忆系统。内部工具（task/memory/attention）由服务端直接执行，"
            "文件和终端工具（read_file, write_to_file, execute_command 等）由 IDE 客户端执行。\n"
            "【重要】请通过 function calling（工具调用）来使用工具，"
            "不要在文本中输出 XML 标签格式的工具调用。\n"
            f"{terminal_env}\n"
            "\n"
            "【任务恢复模式】\n"
            "系统已自动恢复之前挂起的任务。任务图已加载到内存中。\n"
        )

        # 动态注入任务进度表
        progress_summary, first_uncompleted_id = self._build_progress_table()
        if progress_summary:
            parts.append(f"\n【当前任务进度】\n{progress_summary}\n")

        parts.append(
            "\n【执行规则】\n"
            "对每个未完成节点：先调用 task_mark_status(node_id, 'in_progress') 开始，"
            "完成后调用 task_mark_status(node_id, 'completed', result='结果摘要')。\n"
        )
        if first_uncompleted_id:
            parts.append(
                f"第一步：请立即调用 task_mark_status(node_id='{first_uncompleted_id}', "
                f"status='in_progress')\n"
            )
        
        # 检查是否只有根节点（需要创建任务结构）
        from zulong.tools.task_tools import get_active_task_graph
        tg = get_active_task_graph()
        has_only_root = tg and len(tg.nodes) == 1 and 'req' in tg.nodes
        
        if has_only_root:
            parts.append(
                "\n⚠️ 任务图当前只有根节点，需要先创建任务结构：\n"
                "✓ 请使用 task_add_node 添加子任务节点（parent_id='req'）\n"
                "✓ 创建至少2层结构（阶段→具体步骤）\n"
                "✓ 添加完成后调用 task_view_overview 确认结构\n"
                "✓ 然后按顺序执行每个子任务\n"
            )
        else:
            parts.append(
                "\n⚠️ 恢复后的执行规则：\n"
                "✗ 禁止调用 task_create_plan — 这会创建全新图谱，丢弃已恢复的进度\n"
                "✗ 禁止调用 task_add_node — 节点已经在恢复的图谱中了\n"
                "✓ 只使用 task_mark_status 更新现有节点状态，然后继续执行\n"
                "⚠️ 未完成子任务时，不能声称任务已全部完成。\n"
            )

        if memory_context:
            parts.append(f"\n\n## 项目记忆\n{memory_context}")
        if task_context:
            parts.append(f"\n\n## 当前任务上下文\n{task_context}")
        if experience_hints:
            parts.append(f"\n\n## 相关经验\n{experience_hints}")

        parts.append(
            "\n\n⚠️ 语言要求：必须使用与用户输入相同的语言回复。\n"
            "\n请继续执行之前的任务："
        )

        return "\n".join(parts)

    @staticmethod
    def _build_progress_table() -> tuple:
        """构建任务进度表（用于 RESUME 提示词注入）

        Returns:
            (progress_summary: str, first_uncompleted_id: str)
        """
        try:
            from zulong.tools.task_tools import get_active_task_graph
            tg = get_active_task_graph()
            if not tg:
                return "", ""
            leaves = tg.get_leaf_nodes()
            completed = [n for n in leaves if n.status == "completed"]
            uncompleted = [n for n in leaves if n.status != "completed"]
            total = len(leaves)
            done = len(completed)

            lines = [f"进度: {done}/{total} 个工作项已完成。"]
            if completed:
                lines.append("已完成：")
                for n in completed:
                    brief = f" -> {n.result[:80]}" if n.result else ""
                    lines.append(f"  - {n.id}: {n.label}{brief}")
            if uncompleted:
                lines.append("未完成的任务：")
                status_map = {
                    "pending": "待开始", "not_started": "待开始",
                    "in_progress": "进行中", "blocked": "阻塞",
                }
                for n in uncompleted:
                    st = status_map.get(n.status, n.status)
                    desc = f"\n    描述: {n.desc[:120]}" if n.desc else ""
                    lines.append(f"  - {n.id}: {n.label} ({st}){desc}")
                lines.append(
                    f"\n请从第一个未完成任务 {uncompleted[0].id}"
                    f"（{uncompleted[0].label}）开始执行。"
                )
                return "\n".join(lines), uncompleted[0].id
            else:
                lines.append("所有工作项已完成。")
                return "\n".join(lines), ""
        except Exception:
            return "", ""

    def retrieve_zulong_context(self, user_message: str) -> Tuple[str, str, str]:
        """检索祖龙上下文（记忆、任务、经验）

        Args:
            user_message: 用户最新消息

        Returns:
            (memory_context, task_context, experience_hints)
        """
        memory_context = ""
        task_context = ""
        experience_hints = ""

        # 1. 记忆检索
        try:
            from zulong.memory.memory_graph import get_memory_graph
            import asyncio

            mg = get_memory_graph()
            if mg:
                loop = asyncio.new_event_loop()
                try:
                    results = loop.run_until_complete(
                        mg.retrieve_context(user_message[:200], top_k=5)
                    )
                finally:
                    loop.close()

                if results:
                    sections = []
                    for r in results[:5]:
                        node_type = r.get("node_type", "")
                        label = r.get("label", "")
                        content = (r.get("content", "") or "")[:200]
                        sections.append(f"- [{node_type}] {label}: {content}")
                    memory_context = "\n".join(sections)
        except Exception as e:
            logger.info(f"[IDEPromptHandler] 记忆检索跳过: {e}")

        # 2. 任务图状态
        try:
            from zulong.tools.task_tools import get_active_task_graph

            tg = get_active_task_graph()
            if tg:
                table = tg.to_focused_planning_table()
                if table:
                    task_context = table[:500]
        except Exception as e:
            logger.info(f"[IDEPromptHandler] 任务图检索跳过: {e}")

        # 3. 经验检索
        try:
            from zulong.memory.rag_manager import RAGManager, RAGConfig

            rag_config = RAGConfig(
                vector_dimension=512,
                vector_store_type="faiss",
                base_path="./data/rag",
            )
            rag = RAGManager(rag_config)
            exp_results = rag.search_all(user_message[:200], top_k=2)
            parts = []
            for lib_name, docs in exp_results.items():
                for doc in docs[:2]:
                    parts.append(f"- [{lib_name}] {doc.content[:150]}")
            if parts:
                experience_hints = "\n".join(parts)
        except Exception as e:
            logger.info(f"[IDEPromptHandler] 经验检索跳过: {e}")

        return memory_context, task_context, experience_hints
