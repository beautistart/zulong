"""Shared coarse tool capability inference for L2 execution policies.

This module keeps policy decisions generic: recovery and quality gates can ask
for "attention_switch" or "note_anchor" capabilities instead of binding to one
specific tool name. Explicit tool schema annotations are preferred; name and
parameter heuristics are only compatibility fallbacks for legacy schemas.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set


def tool_capabilities(tool_definition: Dict[str, Any]) -> Set[str]:
    """Infer coarse capabilities from a tool schema.

    Supported capability labels intentionally describe behavior, not concrete
    tool names:
    - attention_switch
    - note_anchor
    - memory_persist
    - tag_anchor
    - file_write
    - verification
    """

    raw_caps = (
        tool_definition.get("x_zulong_capabilities")
        or tool_definition.get("capabilities")
        or tool_definition.get("function", {}).get("x_zulong_capabilities")
        or tool_definition.get("function", {}).get("capabilities")
        or []
    )
    if isinstance(raw_caps, str):
        raw_caps = [raw_caps]
    aliases = {
        "memory_write": "memory_persist",
        "memory_landing": "memory_persist",
        "persist_memory": "memory_persist",
        "tag": "tag_anchor",
        "label_anchor": "tag_anchor",
        "note": "note_anchor",
        "attention": "attention_switch",
    }
    caps: Set[str] = {
        aliases.get(str(cap).strip().lower(), str(cap).strip().lower())
        for cap in raw_caps
        if str(cap or "").strip()
    }
    if caps:
        return caps

    fn = str(tool_definition.get("function", {}).get("name", "") or "").strip().lower()
    desc = str(tool_definition.get("function", {}).get("description", "") or "").lower()
    text = f"{fn} {desc}"
    cat = str(tool_definition.get("category", "") or "").strip().lower()
    params = tool_definition.get("function", {}).get("parameters", {}) or {}
    props = params.get("properties", {}) if isinstance(params, dict) else {}
    prop_names = {str(k).lower() for k in props.keys()} if isinstance(props, dict) else set()

    if (
        "attention" in text
        or "注意力" in text
        or (
            bool({"mode", "direction", "target_node_id"} & prop_names)
            and ("global" in text or "focus" in text or "single_chain" in text)
        )
    ):
        caps.add("attention_switch")

    if (
        "note" in text
        or "memory" in text
        or "便签" in text
        or "笔记" in text
        or "记忆" in text
        or (
            bool({"content", "label"} & prop_names)
            and ("anchor" in text or "关联" in text or "长期" in text)
        )
    ):
        caps.add("note_anchor")

    if (
        "memory" in text
        or "记忆" in text
        or "落盘" in text
        or "持久" in text
        or "保存笔记" in text
        or "保存记忆" in text
        or bool({"importance", "entries"} & prop_names)
    ):
        caps.add("memory_persist")

    if (
        "tag" in text
        or "标签" in text
        or "重要性" in text
        or "importance" in text
        or bool({"tag", "tags", "importance"} & prop_names)
    ):
        # 标签能力只表示给已经落盘/待落盘信息附加检索或重要性标签；
        # 是否允许进入 RED 受限恢复由调用方再结合 memory/note 能力筛选。
        caps.add("tag_anchor")

    if (
        "write" in text
        or "replace" in text
        or "create file" in text
        or "写入" in text
        or "修改文件" in text
        or "创建文件" in text
        or cat in {"file", "code"}
        or (
            bool({"path", "file_path", "target_path"} & prop_names)
            and bool({"content", "diff", "replacement"} & prop_names)
        )
    ):
        caps.add("file_write")

    if (
        "command" in text
        or "execute" in text
        or "run" in text
        or "browser" in text
        or "read" in text
        or "verify" in text
        or "test" in text
        or "命令" in text
        or "执行" in text
        or "读取" in text
        or "验证" in text
        or "测试" in text
        or bool({"command", "regex", "query", "url"} & prop_names)
    ):
        caps.add("verification")

    return caps


def tool_capability_map(
    tool_definitions: Optional[Iterable[Dict[str, Any]]],
) -> Dict[str, Set[str]]:
    capability_by_name: Dict[str, Set[str]] = {}
    for td in tool_definitions or []:
        fn = str(td.get("function", {}).get("name", "") or "").strip()
        if fn:
            capability_by_name[fn] = tool_capabilities(td)
    return capability_by_name


def tool_has_capability(tool_definition: Dict[str, Any], capability: str) -> bool:
    return capability in tool_capabilities(tool_definition)


def filter_tools_by_capabilities(
    tool_definitions: Optional[Iterable[Dict[str, Any]]],
    capabilities: Iterable[str],
) -> List[Dict[str, Any]]:
    wanted = {str(cap or "").strip() for cap in capabilities if str(cap or "").strip()}
    if not wanted:
        return []
    return [
        td for td in (tool_definitions or [])
        if tool_capabilities(td) & wanted
    ]
