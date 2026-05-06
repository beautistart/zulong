"""
IDE 格式解析工具

祖龙 FC 循环使用 OpenAI 原生 tool_calls 格式。
本模块提供：
- XML → FC：当 LLM 在文本中输出 XML 格式工具调用时，回退解析为 FC 格式
- 工具结果解析：从 IDE 返回的消息中提取工具执行结果
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def _unescape_closing_tag(value: str, *tag_names: str) -> str:
    """反向还原闭合标签转义。"""
    for tag in tag_names:
        if not tag:
            continue
        value = value.replace(f"<\\/{tag}>", f"</{tag}>")
    return value


class IDEFormatTranslator:
    """IDE 格式解析工具"""

    # ── IDE 消息 → 工具执行结果 ────────────────────────────

    @staticmethod
    def parse_ide_tool_results(
        messages: List[Dict],
        pending_call_ids: List[str],
    ) -> List[Dict]:
        """从 IDE 的后续消息中提取工具执行结果

        IDE 执行工具后，将结果作为新消息发回：
        - 工具成功：user 消息包含执行结果
        - 工具失败：user 消息包含错误信息
        - 通常还附带 [environment_details] 块

        Args:
            messages: IDE 发送的消息列表
            pending_call_ids: 上一轮暂停时等待的 tool_call IDs

        Returns:
            工具结果列表: [{"tool_call_id": "call_xxx", "content": "结果文本"}]
        """
        results = []

        # 查找最后的 user 消息（IDE 工具结果通常在最新的 user 消息中）
        user_messages = []
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_messages.append(msg)
                # IDE 可能发送多条 user 消息（一个工具结果一条）
                # 但通常只有一条包含所有工具结果
                break

        if not user_messages:
            logger.warning("[IDEFormat] 未找到 user 消息，无法提取工具结果")
            return results

        # 提取结果文本（去除 environment_details 块）
        for user_msg in user_messages:
            content = user_msg.get("content", "")
            if isinstance(content, list):
                # 多模态消息：提取文本部分
                text_parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                content = "\n".join(text_parts)

            # 清理 environment_details
            cleaned = IDEFormatTranslator._strip_environment_details(content)

            # 为每个 pending_call_id 分配结果
            # IDE 通常一次只执行一个工具，结果在单条消息中
            if pending_call_ids:
                # 第一个 pending call 获取主结果
                results.append({
                    "tool_call_id": pending_call_ids[0],
                    "content": cleaned.strip() or "(工具执行完成，无输出)",
                })
                # 如果有多个 pending calls，后续的标记为无输出
                for call_id in pending_call_ids[1:]:
                    results.append({
                        "tool_call_id": call_id,
                        "content": "(等待执行)",
                    })

        return results

    @staticmethod
    def _strip_environment_details(text: str) -> str:
        """从 IDE 消息中移除 [environment_details] 块"""
        # IDE 的 environment_details 格式:
        # <environment_details>
        # ...
        # </environment_details>
        pattern = r"<environment_details>.*?</environment_details>"
        cleaned = re.sub(pattern, "", text, flags=re.DOTALL)
        return cleaned.strip()

    # ── XML 文本 → OpenAI tool_calls 解析 ────────────────────

    @staticmethod
    def parse_xml_tool_calls(text: str) -> List[Dict]:
        """从 LLM 文本输出中解析 XML 格式的工具调用

        当 LLM 输出包含 IDE XML 工具标签时（偶尔发生），
        解析为 OpenAI tool_calls 格式供 FC 循环处理。

        支持三类格式：
        1. IDE 标准: <list_files><path>...</path></list_files>
        2. 通用包装: <tool_call><function name="list_files">
                      <parameter name="path">...</parameter></function></tool_call>
        3. JSON 包装: <tool_call>{"name":"list_files","arguments":{...}}</tool_call>

        Args:
            text: LLM 输出的文本

        Returns:
            解析出的 tool_calls 列表（OpenAI 格式）
        """
        from zulong.ide.ide_tool_registry import IDE_REMOTE_TOOLS
        import uuid

        # 预处理: 移除 <thinking>/<think> 块（模型思考过程，不属于工具调用）
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # 扩展工具名单：远程工具 + 内部代码智能工具（模型可能以 XML 形式调用）
        _INTERNAL_XML_PARSEABLE = {
            "index_project", "index_code_file", "search_code_symbols",
            "get_symbol_context", "get_impact_analysis", "analyze_module",
            "task_create_plan", "task_add_node", "task_mark_status",
            "task_view_overview", "recall_memory", "save_memory_note",
            "navigate_attention", "search_experience", "search_tools",
            "zulong_code_query", "zulong_task_link_code",
        }
        ALL_PARSEABLE_TOOLS = IDE_REMOTE_TOOLS | _INTERNAL_XML_PARSEABLE

        tool_calls = []

        # ── 格式 1a: IDE 标准 <tool_name>...</tool_name> (有闭合标签) ──
        for tool_name in ALL_PARSEABLE_TOOLS:
            pattern = rf"<{re.escape(tool_name)}>(.*?)</{re.escape(tool_name)}>"
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                args = IDEFormatTranslator._parse_xml_args(match)
                if args:  # 仅在成功解析出参数时才添加
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    })

        if tool_calls:
            return tool_calls

        # ── 格式 1b: 无闭合标签回退 <tool_name>...(到下一个工具标签或文本末尾) ──
        # 部分模型不输出 </tool_name> 闭合标签
        _tool_open_re = "|".join(re.escape(t) for t in ALL_PARSEABLE_TOOLS)
        for tool_name in ALL_PARSEABLE_TOOLS:
            unclosed_pat = (
                rf"<{re.escape(tool_name)}>"
                rf"(.*?)"
                rf"(?=<(?:{_tool_open_re})>|\Z)"
            )
            matches = re.findall(unclosed_pat, text, re.DOTALL)
            for match in matches:
                args = IDEFormatTranslator._parse_xml_args(match)
                if args:
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    })

        if tool_calls:
            return tool_calls

        # ── 格式 2: <tool_call> 包装格式 ──
        # 匹配 <tool_call>...</tool_call> 或 <function_call>...</function_call>
        tc_blocks = re.findall(
            r"<(?:tool_call|function_call)>(.*?)</(?:tool_call|function_call)>",
            text, re.DOTALL)
        for block in tc_blocks:
            parsed = IDEFormatTranslator._parse_tool_call_block(
                block, IDE_REMOTE_TOOLS)
            if parsed:
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": parsed,
                })

        return tool_calls

    @staticmethod
    def _parse_tool_call_block(
        block: str, known_tools: set
    ) -> Optional[Dict[str, str]]:
        """解析 <tool_call> 块内部，提取工具名和参数

        支持多种 LLM 输出变体：
        - <function name="list_files"><parameter name="path">v</parameter></function>
        - <name>list_files</name><parameter>v</parameter>
        - JSON: {"name":"list_files","arguments":{"path":"v"}}
        """
        block = block.strip()

        # ── 变体 A: 内嵌 JSON ──
        if block.lstrip().startswith("{"):
            try:
                obj = json.loads(block)
                name = obj.get("name", "")
                if name in known_tools:
                    raw_args = obj.get("arguments", {})
                    if isinstance(raw_args, str):
                        return {"name": name, "arguments": raw_args}
                    return {"name": name,
                            "arguments": json.dumps(raw_args, ensure_ascii=False)}
            except (json.JSONDecodeError, TypeError):
                pass

        # ── 变体 B: <function name="..."> 属性格式 ──
        fn_attr = re.search(
            r'<function\s+name=["\'](\w+)["\']>(.*?)</function>',
            block, re.DOTALL)
        if fn_attr:
            name = fn_attr.group(1)
            if name in known_tools:
                args = IDEFormatTranslator._parse_parameter_tags(
                    fn_attr.group(2))
                return {"name": name,
                        "arguments": json.dumps(args, ensure_ascii=False)}

        # ── 变体 C: <name>tool_name</name> + <parameter> 标签 ──
        name_m = re.search(r"<name>(\w+)</name>", block)
        if name_m and name_m.group(1) in known_tools:
            args = IDEFormatTranslator._parse_parameter_tags(block)
            return {"name": name_m.group(1),
                    "arguments": json.dumps(args, ensure_ascii=False)}

        # ── 变体 D: <function>tool_name</function> + <parameter> ──
        func_m = re.search(r"<function>(\w+)</function>", block)
        if func_m and func_m.group(1) in known_tools:
            args = IDEFormatTranslator._parse_parameter_tags(block)
            return {"name": func_m.group(1),
                    "arguments": json.dumps(args, ensure_ascii=False)}

        return None

    @staticmethod
    def _parse_parameter_tags(block: str) -> Dict[str, str]:
        """从 XML 块中提取 <parameter name="key">value</parameter> 格式参数"""
        args = {}
        # 格式 1: <parameter name="key">value</parameter>
        for m in re.finditer(
                r'<parameter\s+name=["\'](\w+)["\']>(.*?)</parameter>',
                block, re.DOTALL):
            args[m.group(1)] = m.group(2).strip()
        if args:
            return args
        # 格式 2: <param_name>value</param_name>（回退到标准解析）
        fallback = IDEFormatTranslator._parse_xml_args(block)
        # 排除已知结构标签（不是参数名）
        _STRUCTURAL_TAGS = {
            "name", "function", "tool_call", "function_call",
            "arguments", "tool", "invoke", "tool_use",
        }
        return {k: v for k, v in fallback.items()
                if k not in _STRUCTURAL_TAGS}

    @staticmethod
    def _parse_xml_args(xml_content: str) -> Dict[str, str]:
        """从 XML 内容块中解析参数标签

        支持三种格式：
        1. 标准格式: <param>value</param>
        2. IDE 属性格式 (有闭合): <parameter=param>value</parameter>
        3. IDE 原生格式 (无闭合): <parameter=param>\nvalue
        """
        args = {}
        # 格式 1: 标准 <param>value</param>
        pattern = r"<(\w+)>(.*?)</\1>"
        for match in re.finditer(pattern, xml_content, re.DOTALL):
            param_name = match.group(1)
            # 跳过 "parameter" 标签名（属于格式 2）
            if param_name == "parameter":
                continue
            param_value = match.group(2).strip()
            # 反向还原闭合标签转义
            param_value = _unescape_closing_tag(param_value, param_name)
            args[param_name] = param_value
        # 格式 2: IDE 属性格式 <parameter=name>value</parameter> (有闭合标签)
        attr_pattern = r"<parameter=(\w+)>(.*?)</parameter>"
        for match in re.finditer(attr_pattern, xml_content, re.DOTALL):
            param_name = match.group(1)
            param_value = match.group(2).strip()
            param_value = _unescape_closing_tag(param_value, param_name)
            if param_name not in args:  # 标准格式优先
                args[param_name] = param_value
        # 格式 3: IDE 原生无闭合标签: <parameter=name>\nvalue
        # 值延续到下一个 <parameter= 或内容末尾
        # 常见于 DeepSeek 等模型遵循 IDE 系统提示词输出
        no_close_pattern = r"<parameter=(\w+)>\s*(.*?)(?=<parameter=|\Z)"
        for match in re.finditer(no_close_pattern, xml_content, re.DOTALL):
            param_name = match.group(1)
            param_value = match.group(2).strip()
            if param_name not in args and param_value:
                param_value = _unescape_closing_tag(param_value, param_name)
                args[param_name] = param_value
        return args
