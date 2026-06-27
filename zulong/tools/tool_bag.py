"""Unified tool bag and L1-B tool prediction helpers.

This module is intentionally deterministic.  It does not classify a whole
conversation turn into legacy session categories; it only predicts which concrete
tools and context hints are likely useful for the next L2 step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from zulong.tools.base import BaseTool, ToolCategory


WRITE_TOOL_NAMES = {
    "exec_write_file",
    "ide_write_file",
    "write_to_file",
    "replace_in_file",
    "delete_file",
    "task_remove_node",
    "delete_memory_node",
    "delete_memory_edge",
}

TERMINAL_TOOL_NAMES = {
    "exec_run_command",
    "execute_command",
    "system_command",
}

IDE_TOOL_NAMES = {
    "ide_open_workspace",
    "ide_write_file",
    # VS Code 完整控制工具 (TSD v2.7 工具袋扩充)
    "vscode_run_command",
    "get_diagnostics",
    "ask_user_input",
    "ask_user_select_file",
    "vscode_manage_extension",
    "open_settings",
    "open_problems",
}

NETWORK_TOOL_NAMES = {
    "web_search",
    "network",
}

MEMORY_TOOL_NAMES = {
    "recall_memory",
    "read_memory_node",
    "save_memory_note",
    "discover_related",
    "activate_memory_network",
    "list_memory",
    "search_experience",
}

MEMORY_SAVE_CUES = (
    "记住", "保存记忆", "存储记忆", "记忆存储", "写入记忆",
    "保存到记忆", "保存到长期记忆", "存到记忆", "记录到记忆",
    "写入长期记忆", "长期记忆写入", "保存这个事实", "记录这个事实",
    "备忘", "笔记", "remember", "save memory", "store memory",
)

MEMORY_RECALL_CUES = (
    "记得", "回忆", "检索记忆", "读取记忆", "从长期记忆", "长期记忆检索",
    "之前", "上次", "刚才", "历史", "经验",
    "recall", "previous", "last time",
)

TASK_TOOL_NAMES = {
    "task_create_plan",
    "task_add_node",
    "task_mark_status",
    "task_view_overview",
    "task_suspend",
    "task_list_suspended",
    "task_add_dependency",
    "task_get_detail",
    "task_update_node",
    "task_remove_node",
    "task_update_content",
    "task_attach_file",
    "submit_final_answer",
    "task_resume_by_address",
    "task_revise_node",
}

CODE_TOOL_NAMES = {
    "read_file",
    "zulong_memory_write_with_code",
    "zulong_code_query",
    "zulong_task_link_code",
    "search_code_symbols",
    "get_symbol_context",
    "get_impact_analysis",
    "index_code_file",
    "index_project",
    "analyze_module",
}

PROJECT_READ_TOOL_NAMES = {
    "read_file",
    "zulong_code_query",
    "search_code_symbols",
    "get_symbol_context",
    "get_impact_analysis",
    "index_code_file",
    "index_project",
    "analyze_module",
}

# ===== VS Code 完整控制工具集 (TSD v2.7 第23章 工具袋扩充) =====

VSCODE_COMMAND_TOOL_NAMES = {
    "vscode_run_command",
}

VSCODE_DIAGNOSTIC_TOOL_NAMES = {
    "get_diagnostics",
}

VSCODE_INTERACTION_TOOL_NAMES = {
    "ask_user_input",
    "ask_user_select_file",
}

VSCODE_EXTENSION_TOOL_NAMES = {
    "vscode_manage_extension",
}

VSCODE_UI_TOOL_NAMES = {
    "open_settings",
    "open_problems",
}

ALWAYS_AVAILABLE_TOOLS = {
    "request_tool_supplement",
    "search_tools",
}

_EMBEDDING_TOOL_CACHE: Dict[Tuple[Any, ...], Tuple[List[str], List[str], Any]] = {}


@dataclass
class ToolBagEntry:
    name: str
    category: str
    description: str
    inputs: List[str] = field(default_factory=list)
    risk: str = "low"
    executor: str = "backend"
    requires_approval: bool = False
    examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "inputs": self.inputs,
            "risk": self.risk,
            "executor": self.executor,
            "requires_approval": self.requires_approval,
            "examples": self.examples,
        }


@dataclass
class ToolPrediction:
    predicted_tools: List[str]
    context_bundle: Dict[str, Any]
    reasons: List[str]
    risk_notes: List[str]
    task_graph_policy: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_tools": self.predicted_tools,
            "context_bundle": self.context_bundle,
            "reasons": self.reasons,
            "risk_notes": self.risk_notes,
            "task_graph_policy": self.task_graph_policy,
        }


def build_tool_bag(registry) -> Dict[str, ToolBagEntry]:
    """Build a clear user/model-facing catalog from the active ToolRegistry."""
    bag: Dict[str, ToolBagEntry] = {}
    tools = getattr(registry, "tools", {}) or {}
    for name, tool in tools.items():
        if not getattr(tool, "enabled", True):
            continue
        bag[name] = describe_tool(tool)
    return bag


def describe_tool(tool: BaseTool) -> ToolBagEntry:
    name = tool.name
    schema = _safe_schema(tool)
    props = (
        schema.get("function", {})
        .get("parameters", {})
        .get("properties", {})
    )
    category = _category_for(name, tool.category)
    risk = _risk_for(name, tool.category)
    return ToolBagEntry(
        name=name,
        category=category,
        description=(getattr(tool, "description", "") or "").strip() or f"{name} 工具",
        inputs=list(props.keys()),
        risk=risk,
        executor=_executor_for(name, tool.category),
        requires_approval=(risk == "high"),
        examples=_examples_for(name, category),
    )


def predict_tools_for_turn(
    text: str,
    *,
    registry=None,
    intent_result: Optional[Dict[str, Any]] = None,
    referenced_nodes: Optional[Sequence[Any]] = None,
    has_task_graph: bool = False,
    max_prompt_tools: Optional[int] = None,
    embedding_provider: Optional[Any] = None,
) -> ToolPrediction:
    """Predict concrete tools/context for L2 without hard-routing the turn."""
    text = (text or "").strip()
    lower = text.lower()
    predicted: List[str] = []
    reasons: List[str] = []
    risk_notes: List[str] = []
    context_bundle: Dict[str, Any] = {}
    task_graph_policy = "none"

    def add(names: Iterable[str], reason: str) -> None:
        added = []
        for name in names:
            if name not in predicted:
                predicted.append(name)
                added.append(name)
        if added and reason not in reasons:
            reasons.append(reason)

    if _is_simple_social(text):
        context_bundle["turn_shape"] = "simple_social"
        predicted.clear()
        return _finalize_prediction(predicted, context_bundle, reasons, risk_notes, task_graph_policy, registry, max_prompt_tools)

    if _is_presentation_only_request(text):
        context_bundle["turn_shape"] = "presentation_only"
        predicted.clear()
        task_graph_policy = "none"
        reasons.append("用户只是在约束回复格式/排版，不需要任务图或工具执行")
        return _finalize_prediction(predicted, context_bundle, reasons, risk_notes, task_graph_policy, registry, max_prompt_tools)

    if _needs_realtime(lower):
        add(["web_search"], "问题涉及实时/最新互联网信息")
        context_bundle["needs_realtime"] = True

    if _needs_memory_save(lower):
        add(["save_memory_note", "recall_memory", "read_memory_node", "discover_related", "search_experience"], "用户明确要求写入长期记忆")
        context_bundle["needs_memory"] = True
        context_bundle["needs_memory_write"] = True
        task_graph_policy = "none"
    elif _needs_memory(lower, intent_result):
        add(["recall_memory", "read_memory_node", "discover_related", "search_experience"], "需要结合祖龙图记忆或历史经验")
        context_bundle["needs_memory"] = True

    if referenced_nodes:
        add(["read_memory_node", "task_get_detail", "task_view_overview"], "用户引用了图节点或任务上下文")
        context_bundle["referenced_nodes"] = list(referenced_nodes)
        task_graph_policy = "inspect"

    if _needs_project_read(lower):
        requires_code_graph_anchor = _needs_code_graph_task_anchor(lower)
        if requires_code_graph_anchor:
            add([
                "task_create_plan",
                "task_add_node",
                "task_view_overview",
                "task_mark_status",
                "submit_final_answer",
            ], "创建代码图谱前需要先创建/绑定任务图节点，便于 CRG 挂载到任务图")
            context_bundle["code_graph_requires_task_graph"] = True
            task_graph_policy = "inspect_or_create"
        add([
            "read_file",
            "zulong_code_query",
            "search_code_symbols",
            "get_symbol_context",
            "get_impact_analysis",
            "index_code_file",
            "index_project",
            "analyze_module",
        ], "需要读取或分析项目代码")
        context_bundle["needs_project_context"] = True

    if _needs_ide_workspace_open(lower):
        add(["ide_open_workspace"], "用户要求打开或切换 VS Code 项目文件夹")
        context_bundle["needs_ide_workspace"] = True

    if _needs_task_graph(lower, intent_result, has_task_graph):
        add([
            "search_experience",
            "task_view_overview",
            "task_get_detail",
            "task_create_plan",
            "task_add_node",
            "task_mark_status",
            "task_update_node",
            "task_list_suspended",
            "task_resume_by_address",
            "task_suspend",
            "submit_final_answer",
        ], "可能需要任务图跟踪、创建或恢复复杂任务，并参考历史执行经验")
        # L1-B 只声明需要任务图能力；是否恢复旧任务由 L2/LLM
        # 通过 task_list_suspended / task_resume_by_address 等工具语义判断。
        task_graph_policy = "inspect_or_create"

    if _needs_write(lower) and not context_bundle.get("needs_memory_write"):
        # 统一用 exec_write_file：内部自动检测 VS Code 桥状态路由
        # （桥可用→编辑器实时通道；桥不可用→本地静默写入）
        add(["exec_write_file"], "用户请求创建或修改文件")
        if _needs_directory_create(lower):
            context_bundle["needs_directory_create"] = True
        add(["task_attach_file", "zulong_memory_write_with_code"], "用户请求创建或修改文件/代码")
        risk_notes.append("涉及文件写入，应走 diff、审批和 checkpoint。")
        task_graph_policy = "inspect_or_create"

    if _needs_terminal(lower):
        add(["exec_run_command"], "用户请求运行、测试、构建或诊断命令")
        risk_notes.append("涉及终端命令，应审批后执行。")

    # ===== VS Code 完整控制预判 (TSD v2.7 工具袋扩充) =====

    if _needs_vscode_command(lower):
        add(["vscode_run_command"], "用户要求执行 VS Code 命令（格式化/重构/Git/测试运行）")
        context_bundle["needs_vscode_command"] = True
        risk_notes.append("vscode_run_command 可执行 VS Code 命令，高风险命令需审批。")

    if _needs_diagnostics(lower):
        add(["get_diagnostics", "open_problems"], "用户询问代码错误或需要检查 lint/编译状态")
        context_bundle["needs_diagnostics"] = True

    if _needs_extension_management(lower):
        add(["vscode_manage_extension"], "用户要求管理 VS Code 扩展")
        risk_notes.append("扩展安装/卸载涉及系统权限，需用户审批。")

    if registry is not None:
        rule_predicted = list(predicted)
        semantic_tools, semantic_reasons = _predict_tools_with_embeddings(
            text,
            registry=registry,
            context_bundle=context_bundle,
            task_graph_policy=task_graph_policy,
            intent_result=intent_result,
            max_results=max_prompt_tools,
            embedding_provider=embedding_provider,
        )
        if semantic_tools:
            predicted = []
            add(semantic_tools, semantic_reasons[0] if semantic_reasons else "基于 embedding 语义检索匹配工具")
            for reason in semantic_reasons[1:]:
                if reason not in reasons:
                    reasons.append(reason)
            _add_rule_guards(
                predicted,
                rule_predicted,
                context_bundle=context_bundle,
                task_graph_policy=task_graph_policy,
            )
            context_bundle["tool_prediction_source"] = "embedding"
        else:
            context_bundle["tool_prediction_source"] = "rule_fallback"

    return _finalize_prediction(predicted, context_bundle, reasons, risk_notes, task_graph_policy, registry, max_prompt_tools)


def supplement_tools(
    registry,
    *,
    missing_capability: str = "",
    reason: str = "",
    suggested_tools: Optional[Sequence[str]] = None,
    max_results: Optional[int] = None,
    list_all_tools: bool = False,
) -> Dict[str, Any]:
    """Return matching tool bag entries and schemas for an L2 supplement request."""
    bag = build_tool_bag(registry)
    if list_all_tools:
        candidates = sorted(bag.keys())
        schemas = _schemas_for(registry, candidates)
        return {
            "message": f"工具袋共有 {len(candidates)} 个工具",
            "tools": [bag[name].to_dict() for name in candidates],
            "supplemented_tools": candidates,
            "supplemented_tool_schemas": schemas,
            "_discovered_schemas": schemas,
            "list_all_tools": True,
        }

    query = " ".join(
        part for part in [missing_capability, reason, " ".join(suggested_tools or [])]
        if part
    ).lower()

    candidates: List[str] = []
    for name in suggested_tools or []:
        if name in bag and name not in candidates:
            candidates.append(name)

    if query:
        scored = []
        query_tokens = set(_tokens(query))
        for name, entry in bag.items():
            haystack = " ".join([
                name,
                entry.category,
                entry.description,
                " ".join(entry.inputs),
                " ".join(entry.examples),
            ]).lower()
            score = sum(1 for t in query_tokens if t and t in haystack)
            score += _keyword_score(query, name, entry)
            if score > 0:
                scored.append((score, name))
        for _, name in sorted(scored, key=lambda item: (-item[0], item[1])):
            if name not in candidates:
                candidates.append(name)
            if max_results and len(candidates) >= max_results:
                break

    if not candidates:
        candidates = [name for name in ALWAYS_AVAILABLE_TOOLS if name in bag]

    candidates = [name for name in candidates if name in bag]
    if max_results:
        candidates = candidates[:max(1, max_results)]
    schemas = _schemas_for(registry, candidates)

    return {
        "message": f"已补充 {len(candidates)} 个工具",
        "tools": [bag[name].to_dict() for name in candidates],
        "supplemented_tools": candidates,
        "supplemented_tool_schemas": schemas,
        "_discovered_schemas": schemas,
    }


def _schemas_for(registry, names: Sequence[str]) -> List[Dict[str, Any]]:
    schemas: List[Dict[str, Any]] = []
    for name in names:
        tool = registry.get(name) if hasattr(registry, "get") else getattr(registry, "tools", {}).get(name)
        if not tool:
            continue
        try:
            schemas.append(tool.get_function_schema())
        except Exception:
            continue
    return schemas


def summarize_tool_bundle(prediction: Dict[str, Any], *, limit: int = 1600) -> str:
    names = prediction.get("predicted_tools") or []
    reasons = prediction.get("reasons") or []
    risks = prediction.get("risk_notes") or []
    policy = prediction.get("task_graph_policy") or "none"
    lines = [
        "【L1-B 工具预判】",
        f"- 建议工具: {', '.join(names) if names else '无'}",
        f"- TaskGraph 策略: {policy}",
    ]
    if reasons:
        lines.append("- 预判理由:")
        lines.extend(f"  - {r}" for r in reasons[:6])
    if risks:
        lines.append("- 风险提示:")
        lines.extend(f"  - {r}" for r in risks[:4])
    ctx = prediction.get("context_bundle") or {}
    if ctx.get("needs_ide_file_write"):
        if policy == "inspect_or_create":
            lines.append(
                "- 文件写入提示: 用户给出了宿主机绝对路径。若任务需要新建项目/多步开发，"
                "先建立 TaskGraph/workspace 绑定，再调用 ide_write_file 写具体文件；"
                "长文件按 800-1200 字符分片，第一片 mode='overwrite'，后续 mode='append'；"
                "不要只用自然语言声称创建成功。"
            )
        else:
            lines.append(
                "- 文件写入提示: 用户给出了宿主机绝对路径，优先调用 ide_write_file，"
                "长文件按 800-1200 字符分片，第一片 mode='overwrite'，后续 mode='append'；"
                "不要只用自然语言声称创建成功。"
            )
    if ctx.get("needs_directory_create"):
        if policy == "inspect_or_create":
            lines.append(
                "- 目录创建提示: 若这是新项目/复杂任务的根目录，先检索历史经验并调用 "
                "task_create_plan 创建绑定 workspace_dir；ide_write_file(create_directory=true) "
                "只用于已有工作区内的普通子目录创建。"
            )
        else:
            lines.append(
                "- 目录创建提示: 用户要求创建文件夹/目录，调用 ide_write_file 时必须设置 create_directory=true。"
            )
    if ctx.get("code_graph_requires_task_graph"):
        lines.append(
            "- ????????: ?? LLM ????????????????? task_create_plan ??/????????????????????"
            "??????? task_add_node ?????????????? index_project?"
        )
    if prediction.get("task_graph_policy") in {"reuse", "inspect", "continue"} or any(
        name in {"task_list_suspended", "task_resume_by_address"} for name in prediction.get("predicted_tools", [])
    ):
        lines.append(
            "- 任务恢复提示: 是否恢复旧任务必须由你结合用户语义判断；需要恢复时先调用 task_list_suspended 或 task_resume_by_address，不要新建任务图。"
        )
    if ctx.get("needs_vscode_command"):
        lines.append(
            "- VS Code 命令提示: 可调用 vscode_run_command 执行格式化、重构、Git 等。"
        )
    if ctx.get("needs_diagnostics"):
        lines.append(
            "- 诊断提示: 调用 get_diagnostics 检查代码错误，open_problems 打开问题面板。"
        )
    lines.append(
        "- 如果这些工具不够，请调用 request_tool_supplement，说明缺少什么能力。"
    )
    text = "\n".join(lines)
    return text[:limit]


def _finalize_prediction(
    predicted: List[str],
    context_bundle: Dict[str, Any],
    reasons: List[str],
    risk_notes: List[str],
    task_graph_policy: str,
    registry,
    max_prompt_tools: Optional[int],
) -> ToolPrediction:
    available = set(getattr(registry, "tools", {}).keys()) if registry is not None else None
    names = []
    for name in predicted:
        if available is not None and name not in available:
            continue
        if name not in names:
            names.append(name)
    # L1-B只做工具预判，不做工具优先级裁剪。常驻工具由L2注入。
    return ToolPrediction(
        predicted_tools=names,
        context_bundle=context_bundle,
        reasons=reasons,
        risk_notes=risk_notes,
        task_graph_policy=task_graph_policy,
    )


def _predict_tools_with_embeddings(
    text: str,
    *,
    registry,
    context_bundle: Dict[str, Any],
    task_graph_policy: str,
    intent_result: Optional[Dict[str, Any]],
    max_results: Optional[int],
    embedding_provider: Optional[Any],
) -> Tuple[List[str], List[str]]:
    """Rank active tools by embedding similarity against the current turn."""
    try:
        import numpy as np
    except Exception:
        return [], ["embedding 依赖不可用，使用规则兜底"]

    bag = build_tool_bag(registry)
    if not bag:
        return [], ["工具袋为空，使用规则兜底"]

    provider = embedding_provider or _load_default_tool_embedding_provider()
    if provider is None:
        return [], ["embedding 模型不可用，使用规则兜底"]

    names = sorted(bag.keys())
    catalog_signature = _tool_catalog_signature(bag, names)
    cache_key = (id(provider), catalog_signature)
    cached = _EMBEDDING_TOOL_CACHE.get(cache_key)
    if cached:
        cached_names, docs, doc_vectors = cached
    else:
        cached_names = names
        docs = [_tool_embedding_text(bag[name]) for name in cached_names]
        try:
            doc_vectors = _encode_documents(provider, docs)
        except Exception:
            return [], ["工具 embedding 编码失败，使用规则兜底"]
        _EMBEDDING_TOOL_CACHE[cache_key] = (cached_names, docs, doc_vectors)

    query_text = _build_tool_embedding_query(
        text,
        context_bundle=context_bundle,
        task_graph_policy=task_graph_policy,
        intent_result=intent_result,
    )
    try:
        query_vector = _encode_query(provider, query_text)
    except Exception:
        return [], ["查询 embedding 编码失败，使用规则兜底"]

    doc_matrix = np.asarray(doc_vectors, dtype="float32")
    query = np.asarray(query_vector, dtype="float32").reshape(-1)
    if doc_matrix.ndim != 2 or query.size == 0:
        return [], ["embedding 向量形状异常，使用规则兜底"]
    if doc_matrix.shape[1] != query.shape[0]:
        return [], ["embedding 维度不匹配，使用规则兜底"]

    denom = (np.linalg.norm(doc_matrix, axis=1) * np.linalg.norm(query)) + 1e-8
    scores = (doc_matrix @ query) / denom
    ranked_indices = list(np.argsort(-scores))
    limit = max_results or _default_embedding_tool_limit(context_bundle, task_graph_policy)

    selected: List[str] = []
    score_rows: List[Dict[str, Any]] = []
    for idx in ranked_indices:
        name = cached_names[int(idx)]
        score = float(scores[int(idx)])
        if not _semantic_tool_allowed(name, context_bundle, text):
            continue
        if name not in selected:
            selected.append(name)
            score_rows.append({"name": name, "score": round(score, 4)})
        if len(selected) >= limit:
            break

    if not selected:
        return [], ["embedding 未选出可用工具，使用规则兜底"]

    context_bundle["embedding_top_tools"] = score_rows[:8]
    return selected, ["基于 embedding 语义相似度匹配首轮工具"]


def _load_default_tool_embedding_provider() -> Optional[Any]:
    try:
        from zulong.models.embedding_model import embedding_model

        if getattr(embedding_model, "model", None) is None:
            embedding_model.load()
        return embedding_model
    except Exception:
        return None


def _tool_catalog_signature(bag: Dict[str, ToolBagEntry], names: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(
        (
            name,
            bag[name].category,
            bag[name].description[:240],
            tuple(bag[name].inputs),
        )
        for name in names
    )


def _tool_embedding_text(entry: ToolBagEntry) -> str:
    return "\n".join(
        [
            f"tool_name: {entry.name}",
            f"category: {entry.category}",
            f"description: {entry.description}",
            f"inputs: {', '.join(entry.inputs) if entry.inputs else 'none'}",
            f"risk: {entry.risk}",
            f"examples: {'; '.join(entry.examples)}",
        ]
    )


def _build_tool_embedding_query(
    text: str,
    *,
    context_bundle: Dict[str, Any],
    task_graph_policy: str,
    intent_result: Optional[Dict[str, Any]],
) -> str:
    parts = [text or ""]
    if intent_result:
        intent = intent_result.get("intent") or intent_result.get("label")
        if intent:
            parts.append(f"intent: {intent}")
    if context_bundle.get("needs_realtime"):
        parts.append("need realtime web search current latest online information")
    if context_bundle.get("needs_memory_write"):
        parts.append("need save long term memory note recall memory node related memories")
    elif context_bundle.get("needs_memory"):
        parts.append("need recall memory read memory node discover related experience")
    if context_bundle.get("needs_project_context"):
        parts.append("need read analyze project code files symbols impact module index")
    if context_bundle.get("needs_ide_file_write"):
        parts.append("need write file through ide bridge vscode workspace approval")
    if context_bundle.get("needs_ide_workspace"):
        parts.append("need open switch vscode ide workspace folder")
    if context_bundle.get("needs_vscode_command"):
        parts.append("need vscode command format refactor git test task")
    if context_bundle.get("needs_diagnostics"):
        parts.append("need diagnostics lint compile errors problems warnings")
    if task_graph_policy != "none":
        parts.append("need task graph plan progress resume node status final answer")
    return "\n".join(part for part in parts if part)


def _encode_query(provider: Any, query: str):
    if hasattr(provider, "encode_query"):
        return provider.encode_query(query)
    if hasattr(provider, "encode"):
        values = provider.encode([query])
        return values[0]
    raise TypeError("embedding provider lacks encode_query/encode")


def _encode_documents(provider: Any, documents: List[str]):
    if hasattr(provider, "encode_documents"):
        return provider.encode_documents(documents)
    if hasattr(provider, "encode"):
        return provider.encode(documents)
    raise TypeError("embedding provider lacks encode_documents/encode")


def _default_embedding_tool_limit(
    context_bundle: Dict[str, Any],
    task_graph_policy: str,
) -> int:
    if context_bundle.get("needs_project_context") or task_graph_policy != "none":
        return 18
    if context_bundle.get("needs_memory") or context_bundle.get("needs_memory_write"):
        return 8
    return 10


def _semantic_tool_allowed(name: str, context_bundle: Dict[str, Any], text: str) -> bool:
    if _has_read_only_constraint(text) and name in WRITE_TOOL_NAMES:
        return False
    if context_bundle.get("needs_memory_write"):
        blocked = (WRITE_TOOL_NAMES | CODE_TOOL_NAMES | TERMINAL_TOOL_NAMES) - MEMORY_TOOL_NAMES
        if name in blocked:
            return False
    return True


def _add_rule_guards(
    predicted: List[str],
    rule_predicted: Sequence[str],
    *,
    context_bundle: Dict[str, Any],
    task_graph_policy: str,
) -> None:
    """Add minimal deterministic guard tools after semantic ranking."""

    def has_any(names: Set[str]) -> bool:
        return any(name in names for name in predicted)

    def add_names(names: Iterable[str]) -> None:
        available = set(rule_predicted) if rule_predicted else set()
        for name in names:
            if available and name not in available:
                continue
            if name not in predicted:
                predicted.append(name)

    if context_bundle.get("needs_memory_write"):
        add_names(["save_memory_note", "recall_memory", "read_memory_node", "discover_related", "search_experience"])
    elif context_bundle.get("needs_memory") and not has_any(MEMORY_TOOL_NAMES):
        add_names(["recall_memory", "read_memory_node", "discover_related", "search_experience"])

    if context_bundle.get("needs_project_context") and not has_any(PROJECT_READ_TOOL_NAMES):
        add_names([
            "read_file",
            "zulong_code_query",
            "search_code_symbols",
            "get_symbol_context",
            "get_impact_analysis",
            "index_code_file",
            "index_project",
            "analyze_module",
        ])

    if context_bundle.get("needs_ide_file_write") and not has_any(WRITE_TOOL_NAMES):
        add_names(["ide_write_file", "task_attach_file", "zulong_memory_write_with_code"])
    elif any(name in WRITE_TOOL_NAMES for name in rule_predicted) and not has_any(WRITE_TOOL_NAMES):
        add_names([name for name in rule_predicted if name in WRITE_TOOL_NAMES])

    if any(name in TERMINAL_TOOL_NAMES for name in rule_predicted) and not has_any(TERMINAL_TOOL_NAMES):
        add_names(["exec_run_command"])

    if task_graph_policy != "none" and not has_any(TASK_TOOL_NAMES):
        add_names([
            "search_experience",
            "task_view_overview",
            "task_get_detail",
            "task_create_plan",
            "task_add_node",
            "task_mark_status",
            "task_update_node",
            "task_list_suspended",
            "task_resume_by_address",
            "task_suspend",
            "submit_final_answer",
        ])


def _safe_schema(tool: BaseTool) -> Dict[str, Any]:
    try:
        return tool.get_function_schema()
    except Exception:
        return {}


def _category_for(name: str, category: ToolCategory) -> str:
    if name in MEMORY_TOOL_NAMES:
        return "memory"
    if name in TASK_TOOL_NAMES:
        return "task_graph"
    if name in CODE_TOOL_NAMES:
        return "project_code"
    if name in WRITE_TOOL_NAMES:
        return "file_write"
    if name in TERMINAL_TOOL_NAMES:
        return "terminal"
    # VS Code 完整控制 — 细粒度分类（优先于泛型 IDE_TOOL_NAMES）
    if name in VSCODE_COMMAND_TOOL_NAMES:
        return "vscode_command"
    if name in VSCODE_DIAGNOSTIC_TOOL_NAMES:
        return "vscode_diagnostic"
    if name in VSCODE_INTERACTION_TOOL_NAMES:
        return "vscode_interaction"
    if name in VSCODE_EXTENSION_TOOL_NAMES:
        return "vscode_extension"
    if name in VSCODE_UI_TOOL_NAMES:
        return "vscode_ui"
    if name in IDE_TOOL_NAMES:
        return "ide_bridge"
    if name in NETWORK_TOOL_NAMES or category == ToolCategory.NETWORK:
        return "network"
    if category == ToolCategory.ROBOT:
        return "robot"
    if category == ToolCategory.SYSTEM:
        return "system"
    return "utility"


def _risk_for(name: str, category: ToolCategory) -> str:
    if name in WRITE_TOOL_NAMES or name in TERMINAL_TOOL_NAMES:
        return "high"
    # VS Code 命令和扩展管理为高风险
    if name in VSCODE_COMMAND_TOOL_NAMES or name in VSCODE_EXTENSION_TOOL_NAMES:
        return "high"
    # VS Code 诊断/交互/UI 为低风险（优先于 IDE_TOOL_NAMES 的 medium）
    if name in VSCODE_DIAGNOSTIC_TOOL_NAMES or name in VSCODE_INTERACTION_TOOL_NAMES or name in VSCODE_UI_TOOL_NAMES:
        return "low"
    if name in IDE_TOOL_NAMES:
        return "medium"
    if name in {"set_importance", "save_memory_note", "task_update_node", "task_mark_status", "task_add_node"}:
        return "medium"
    if category == ToolCategory.ROBOT:
        return "medium"
    return "low"


def _executor_for(name: str, category: ToolCategory) -> str:
    if name in IDE_TOOL_NAMES:
        return "vscode_bridge"
    if category == ToolCategory.NETWORK:
        return "external_service"
    return "backend"


def _examples_for(name: str, category: str) -> List[str]:
    examples = {
        "network": ["查询实时天气", "查找最新新闻", "检索在线文档"],
        "memory": ["回忆历史任务", "读取记忆节点", "保存用户偏好"],
        "project_code": ["分析项目结构", "查找符号定义", "评估改动影响"],
        "file_write": ["生成代码文件", "修改配置", "写入文档"],
        "terminal": ["运行测试", "安装依赖", "执行构建命令"],
        "ide_bridge": ["打开项目文件夹", "切换 VS Code 工作区", "启动 IDE 后台桥"],
        "task_graph": ["创建任务计划", "查看任务进度", "标记节点完成"],
        "vscode_command": ["格式化代码", "整理导入", "运行测试任务", "Git 提交"],
        "vscode_diagnostic": ["检查 lint 错误", "查看编译警告", "获取所有文件诊断"],
        "vscode_interaction": ["弹出输入框收集信息", "让用户选择文件路径"],
        "vscode_extension": ["安装扩展", "卸载扩展", "查看已安装扩展"],
        "vscode_ui": ["打开设置面板", "打开问题面板"],
    }
    return examples.get(category, [f"使用 {name} 完成相关操作"])


def _tokens(text: str) -> List[str]:
    return re.findall(r"[\w\-\./]+|[\u4e00-\u9fff]{2,}", text.lower())


def _keyword_score(query: str, name: str, entry: ToolBagEntry) -> int:
    score = 0
    groups = [
        (("天气", "新闻", "实时", "最新", "搜索", "联网", "weather", "search"), {"web_search"}),
        (("文件", "写入", "创建", "新建", "命名", "修改", "代码", "保存"), WRITE_TOOL_NAMES | CODE_TOOL_NAMES),
        (("运行", "测试", "构建", "命令", "终端", "npm", "python"), TERMINAL_TOOL_NAMES),
        (("打开", "启动", "切换", "项目", "工作区", "文件夹", "目录", "vscode", "ide", "workspace", "folder"), IDE_TOOL_NAMES),
        (("任务", "计划", "进度", "继续", "恢复"), TASK_TOOL_NAMES),
        (("记忆", "历史", "经验", "回忆", "记住", "备忘", "笔记"), MEMORY_TOOL_NAMES),
    ]
    is_memory_save = _needs_memory_save(query)
    if name == "save_memory_note" and is_memory_save:
        score += 12
    if name in WRITE_TOOL_NAMES | CODE_TOOL_NAMES and is_memory_save:
        score -= 4
    for words, names in groups:
        if name in names and any(w in query for w in words):
            score += 4
    if entry.category in query:
        score += 2
    if name == "read_file" and any(
        phrase in query
        for phrase in (
            "读取项目文件内容",
            "读取文件内容",
            "源码内容",
            "源代码",
            "read file",
            "file content",
            "source code",
            "read source",
        )
    ):
        score += 10
    if name == "read_file" and any(
        phrase in query
        for phrase in (
            "只读",
            "read only",
            "read-only",
            "readonly",
            "不要修改",
            "do not modify",
            "without modifying",
        )
    ):
        score += 6
    return score


# ===== VS Code 完整控制需求检测 (TSD v2.7 工具袋扩充) =====

def _needs_vscode_command(text: str) -> bool:
    """检测是否需要 VS Code 命令能力"""
    indicators = [
        "格式化", "format", "整理导入", "organize", "prettier", "lint",
        "重构", "重命名", "rename", "跳转",
        "git", "提交", "commit", "推送", "push", "拉取", "pull",
        "分支", "branch", "merge", "合并", "暂存", "stage",
        "测试任务", "test task", "build task", "构建任务",
        "运行测试", "run test", "调试", "debug",
        "命令面板", "command palette",
    ]
    lowered = text.lower()
    return any(indicator.lower() in lowered for indicator in indicators)


def _needs_diagnostics(text: str) -> bool:
    """检测是否需要诊断信息"""
    indicators = [
        "错误", "报错", "error", "warning", "警告",
        "诊断", "diagnostic", "lint", "类型错误", "type error",
        "编译", "compile", "问题", "problem",
    ]
    lowered = text.lower()
    return any(indicator.lower() in lowered for indicator in indicators)


def _needs_extension_management(text: str) -> bool:
    """检测是否需要扩展管理"""
    indicators = [
        "扩展", "extension", "插件", "plugin",
        "安装扩展", "install extension", "卸载扩展", "uninstall",
        "禁用", "disable", "启用", "enable",
    ]
    lowered = text.lower()
    return any(indicator.lower() in lowered for indicator in indicators)


def _has_read_only_constraint(text: str) -> bool:
    lowered = (text or "").lower()
    literal_cues = (
        "不要修改", "不修改", "无需修改", "不需要修改",
        "不要改", "别改", "不要动", "不要动文件", "不要改文件",
        "不要写", "不写", "不要生成文件",
        "不要保存", "不保存", "只读", "仅分析", "只分析",
        "read only", "read-only", "readonly",
        "do not modify", "don't modify", "without modifying",
        "do not edit", "don't edit", "no edit", "no edits",
        "analyze only", "inspect only", "no file changes", "no changes",
    )
    return any(cue in lowered for cue in literal_cues)


def _is_simple_social(text: str) -> bool:
    normalized = text.lower().strip(" \t\r\n。！？!?.,，~～")
    return len(text) <= 30 and normalized in {
        "你好", "您好", "hi", "hello", "hey",
        "早上好", "下午好", "晚上好",
        "谢谢", "感谢", "多谢", "thanks", "thank you",
        "你好呀", "嗨", "哈喽",
    }


def _is_presentation_only_request(text: str) -> bool:
    lowered = (text or "").lower()
    if _needs_project_read(lowered):
        return False
    lightweight_write_cues = (
        "创建文件", "新建文件", "修改文件", "写入文件", "保存到文件",
        "exec_write_file", "ide_write_file",
    )
    if any(cue in lowered for cue in lightweight_write_cues):
        return False
    if _needs_terminal(lowered) or _needs_task_graph(lowered, None, False):
        return False
    presentation_cues = [
        "markdown",
        "二级标题",
        "一级标题",
        "标题",
        "列表项",
        "列表",
        "代码块",
        "表格",
        "只回复",
        "不要写别的",
        "只要回复",
        "用两句话",
        "一句话",
        "用一句话",
        "用两句话",
        "格式回复",
        "按这个格式",
        "只输出",
    ]
    has_presentation = any(cue in lowered for cue in presentation_cues)
    if not has_presentation:
        return False
    content_action_cues = [
        "创建", "新建", "实现", "开发", "运行", "执行", "分析项目", "分析代码",
        "读取文件", "查看文件", "打开文件", "工作区", "终端", "命令", "任务", "计划", "步骤",
    ]
    if any(cue in lowered for cue in content_action_cues):
        return False
    return True


def _needs_realtime(text: str) -> bool:
    if any(k in text for k in ("运行状态", "系统状态", "工作状态", "当前状态")):
        return False
    if any(k in text for k in (
        "当前项目", "当前工程", "当前工作区", "当前代码", "当前文件",
        "当前目录", "当前仓库", "current project", "current workspace",
    )):
        return False
    return any(k in text for k in (
        "天气", "气温", "降雨", "下雨", "空气质量", "台风",
        "今天", "明天", "后天", "现在", "当前", "实时", "最新",
        "新闻", "热搜", "股价", "汇率", "油价", "票房",
        "航班", "火车", "高铁", "路况", "限行",
        "weather", "forecast", "latest", "today", "tomorrow",
    ))


def _needs_memory(text: str, intent_result: Optional[Dict[str, Any]]) -> bool:
    intent = (intent_result or {}).get("intent", "").lower()
    return intent in {"chat_memory", "task_resume"} or any(k in text for k in MEMORY_SAVE_CUES + MEMORY_RECALL_CUES)


def _needs_memory_save(text: str) -> bool:
    return any(k in text for k in MEMORY_SAVE_CUES)


def _needs_project_read(text: str) -> bool:
    if _needs_memory_save(text) and not any(k in text for k in (
        "读取", "查看", "分析项目", "分析代码", "当前项目", "当前工作区",
        "源码", "模块", "函数", "类", "仓库",
        "read file", "source code", "current project", "current workspace",
    )):
        return False
    return any(k in text for k in (
        "代码", "项目", "文件", "模块", "函数", "类", "报错", "bug",
        "日志", "源码", "编译", "依赖", "仓库", "实现", "重构",
        "code", "project", "file", "function", "class", "bug",
    )) or bool(re.search(r"\b[\w\-./\\]+\.(py|ts|tsx|js|jsx|json|yaml|yml|md|html|css|go|rs|java|cpp|c|cs)\b", text))


def _needs_code_graph_task_anchor(text: str) -> bool:
    explicit_code_graph = any(k in text for k in (
        "代码图谱", "代码结构图", "创建图谱", "建立图谱", "生成图谱",
        "code graph", "crg", "index_project",
    ))
    project_scope = any(k in text for k in (
        "项目", "系统", "架构", "源码", "代码库", "仓库", "工程",
        "project", "architecture", "codebase", "repository",
    ))
    analysis_scope = any(k in text for k in (
        "分析", "理解", "梳理", "扫描", "索引", "结构", "深度",
        "analyze", "analysis", "understand", "inspect", "index",
    ))
    return explicit_code_graph or (project_scope and analysis_scope)


def _needs_ide_workspace_open(text: str) -> bool:
    has_open = any(k in text for k in (
        "打开", "启动", "切换", "进入", "载入", "加载",
        "open", "switch", "load",
    ))
    has_workspace = any(k in text for k in (
        "vscode", "vs code", "ide", "项目", "工程", "工作区", "文件夹", "目录",
        "workspace", "project", "folder", "directory",
    ))
    has_path = bool(re.search(r"(?:[a-zA-Z]:[\\/]|\\\\|/)[^\s，。；;]+", text))
    return has_open and (has_workspace or has_path)


def _needs_task_graph(text: str, intent_result: Optional[Dict[str, Any]], has_task_graph: bool) -> bool:
    if _has_read_only_constraint(text) and any(k in text for k in (
        "inspect", "analysis", "analyze", "compare", "检查", "分析", "对比",
    )):
        return has_task_graph
    explicit_task_cues = any(k in text for k in (
        "任务", "计划", "步骤", "复杂", "项目开发",
        "实现一个", "做一个", "开发", "重构", "编写",
        "implement", "create", "build", "develop", "write", "refactor",
        "next step", "plan",
    ))
    if explicit_task_cues:
        return True
    complex_creation_cues = any(k in text for k in (
        "创建", "新建", "生成", "写", "实现", "制作", "搭建",
        "web端", "web 端", "网页", "小游戏", "游戏", "应用", "项目",
    ))
    if complex_creation_cues and (_needs_directory_create(text) or _has_explicit_host_path(text)):
        return True
    if not has_task_graph:
        return False
    followup_cues = any(k in text for k in (
        "下一步", "然后呢", "做到哪了", "进度", "未完成",
    ))
    return followup_cues


def _needs_write(text: str) -> bool:
    if _needs_memory_save(text):
        return False
    if _needs_memory(text, None) and any(k in text for k in ("回忆", "检索", "读取", "召回", "回答", "复述")):
        return False
    if _has_read_only_constraint(text):
        return False
    if _is_presentation_only_request(text):
        return False
    if any(k in text for k in (
        "写", "创建", "新建", "命名为", "命名成", "生成", "修改", "修复",
        "实现", "删除", "替换", "保存到", "新增", "改一下", "补齐", "重构",
    )):
        return True
    return bool(re.search(
        r"\b(write|create|modify|fix|implement|delete|refactor|develop|build)\b",
        text,
    ))


def _needs_directory_create(text: str) -> bool:
    return any(k in text for k in ("文件夹", "目录", "folder", "directory"))


def _has_explicit_host_path(text: str) -> bool:
    return bool(re.search(r"(?:[a-zA-Z]:[\\/]|\\\\|/)[^\s，。；;]+", text or ""))


def _needs_terminal(text: str) -> bool:
    if any(k in text for k in ("运行状态", "当前状态", "系统状态", "工作状态")):
        return False
    if any(k in text for k in (
        "运行测试", "执行命令", "运行命令", "终端", "控制台",
        "安装依赖", "启动服务", "启动项目", "编译项目", "执行脚本",
        "报错", "npm", "pip", "pytest", "tsc",
    )):
        return True
    return bool(re.search(
        r"\b(run tests?|execute command|run command|terminal|shell|console|"
        r"install dependencies|start (?:server|service|dev server|project)|"
        r"compile project|debug|npm|pip|pytest|tsc)\b",
        text,
    ))
