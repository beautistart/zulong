"""Dynamic attention context retrieval and rendering.

TSD v2.9.24 defines L2 dynamic attention as graph-position-aware
context retrieval and reconstruction.  This module is intentionally a thin
orchestration layer: it builds one per-call AttentionRetrievalPlan, reads
TaskGraph / MemoryGraph / BFS evidence, renders a compact active context, and
never stores a backing pool or performs global de-duplication.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AttentionRetrievalPlan:
    """One-call retrieval plan; not a global context pool."""

    mode: str = "GLOBAL"
    query_text: str = ""
    focus_node_id: str = ""
    focus_path: List[str] = field(default_factory=list)
    focus_depth: int = 0
    focus_node_address: str = ""
    focus_path_addresses: List[str] = field(default_factory=list)
    seed_addresses: List[str] = field(default_factory=list)
    graph_memory_ids: List[str] = field(default_factory=list)
    memory_addresses: List[Dict[str, Any]] = field(default_factory=list)
    source_node_ids: List[str] = field(default_factory=list)
    bfs_depth: int = 1
    target_tokens: int = 1200
    max_items: int = 8
    trigger_reason: str = "model_call"
    pressure_percent: float = 0.0
    include_navigation_map: bool = False
    navigation_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "query_text": self.query_text,
            "focus_node_id": self.focus_node_id,
            "focus_path": list(self.focus_path),
            "focus_depth": self.focus_depth,
            "focus_node_address": self.focus_node_address,
            "focus_path_addresses": list(self.focus_path_addresses),
            "seed_addresses": list(self.seed_addresses),
            "graph_memory_ids": list(self.graph_memory_ids),
            "memory_addresses": list(self.memory_addresses),
            "source_node_ids": list(self.source_node_ids),
            "bfs_depth": self.bfs_depth,
            "target_tokens": self.target_tokens,
            "max_items": self.max_items,
            "trigger_reason": self.trigger_reason,
            "pressure_percent": round(float(self.pressure_percent or 0.0), 1),
            "include_navigation_map": self.include_navigation_map,
            "navigation_reason": self.navigation_reason,
        }


@dataclass
class AttentionContextItem:
    """Renderable context item with source address."""

    text: str
    source: str = ""
    score: float = 0.0
    kind: str = "memory"
    graph_memory_id: str = ""
    shard_id: str = ""
    memory_address: Dict[str, Any] = field(default_factory=dict)
    source_node_ids: List[str] = field(default_factory=list)
    node_id: str = ""

    def address_line(self) -> str:
        if self.source:
            return self.source
        parts: List[str] = []
        if self.graph_memory_id:
            parts.append(f"graph_memory_id={self.graph_memory_id}")
        if self.shard_id:
            parts.append(f"shard_id={self.shard_id}")
        mem_addr = self.memory_address or {}
        full_path = mem_addr.get("full_path") or ""
        if full_path:
            parts.append(f"full_path={full_path}")
        if mem_addr:
            parts.append(f"memory_address={_compact_memory_address(mem_addr)}")
        if parts:
            return ", ".join(parts)
        if self.node_id:
            return str(self.node_id)
        return "unknown"


@dataclass
class AttentionContextBundle:
    plan: AttentionRetrievalPlan
    focus_summary: str = ""
    navigation_map: str = ""
    task_items: List[AttentionContextItem] = field(default_factory=list)
    memory_items: List[AttentionContextItem] = field(default_factory=list)
    bfs_items: List[AttentionContextItem] = field(default_factory=list)
    telemetry: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_items(self) -> List[AttentionContextItem]:
        return list(self.task_items) + list(self.memory_items) + list(self.bfs_items)


def normalize_attention_mode(mode: Any) -> str:
    raw = getattr(mode, "value", mode)
    text = str(raw or "global").strip().upper()
    aliases = {
        "GLOBAL": "GLOBAL",
        "FOCUS": "FOCUS",
        "SINGLE_CHAIN": "SINGLE_CHAIN",
        "SINGLE-CHAIN": "SINGLE_CHAIN",
        "SINGLE CHAIN": "SINGLE_CHAIN",
        "GLOBAL_ATTENTION": "GLOBAL",
        "LOCAL": "FOCUS",
    }
    text = aliases.get(text, aliases.get(text.replace("-", "_"), text.replace("-", "_")))
    if text not in {"GLOBAL", "FOCUS", "SINGLE_CHAIN"}:
        return "GLOBAL"
    return text


def estimate_text_tokens(text: str) -> int:
    """Cheap local estimate used only to cap active-context rendering."""
    if not text:
        return 0
    return max(1, len(str(text)) // 4 + 1)


def resolve_focus_position(
    *,
    attention_window: Any = None,
    task_graph: Any = None,
    memory_graph: Any = None,
    explicit_focus: str = "",
) -> Dict[str, Any]:
    """Resolve current TaskGraph / MemoryGraph position without failing the call."""
    focus_node_id = str(explicit_focus or "").strip()
    if not focus_node_id and attention_window is not None:
        focus_node_id = str(getattr(attention_window, "_current_node_id", "") or "")

    if task_graph is not None:
        focus_node_id = _resolve_task_focus(task_graph, focus_node_id)

    focus_node_address = ""
    focus_depth = 0
    focus_path: List[str] = []
    focus_path_addresses: List[str] = []
    if task_graph is not None and focus_node_id:
        try:
            focus_node_address = task_graph.get_node_address(focus_node_id)
        except Exception:
            focus_node_address = f"tg:{getattr(task_graph, 'id', 'unknown')}/{focus_node_id}"
        try:
            focus_depth = int(task_graph.get_node_depth(focus_node_id) or 0)
        except Exception:
            focus_depth = 0
        focus_path = _task_focus_path(task_graph, focus_node_id)
        focus_path_addresses = [_safe_task_address(task_graph, nid) for nid in focus_path]

    memory_focus = _get_memory_focus(memory_graph)
    mg_focus_path = [str(x) for x in (memory_focus.get("focus_path") or []) if x]
    graph_memory_ids = list(mg_focus_path)
    if not graph_memory_ids and focus_node_id:
        graph_memory_ids.append(focus_node_id)

    focus_summary = _memory_focus_summary(memory_graph, memory_focus)
    seed_addresses = [addr for addr in focus_path_addresses if addr]
    seed_addresses.extend([f"graph_memory_id={gid}" for gid in graph_memory_ids if gid])

    return {
        "focus_node_id": focus_node_id,
        "focus_node_address": focus_node_address,
        "focus_path": focus_path,
        "focus_path_addresses": focus_path_addresses,
        "focus_depth": focus_depth or max(0, len(focus_path) - 1),
        "memory_focus": memory_focus,
        "focus_summary": focus_summary,
        "graph_memory_ids": graph_memory_ids,
        "seed_addresses": seed_addresses,
    }


def build_attention_retrieval_plan(
    *,
    mode: Any,
    query_text: str,
    pressure_percent: float = 0.0,
    trigger_reason: str = "model_call",
    focus: Optional[Dict[str, Any]] = None,
    threshold_budget_tokens: int = 0,
    include_navigation_map: bool = False,
    navigation_reason: str = "",
) -> AttentionRetrievalPlan:
    normalized_mode = normalize_attention_mode(mode)
    focus = focus or {}
    if normalized_mode == "GLOBAL":
        target_tokens = int(threshold_budget_tokens * 0.12) if threshold_budget_tokens else 1800
        max_items = 10
        bfs_depth = 2
    elif normalized_mode == "SINGLE_CHAIN":
        target_tokens = int(threshold_budget_tokens * 0.10) if threshold_budget_tokens else 1400
        max_items = 8
        bfs_depth = 2
    else:
        target_tokens = int(threshold_budget_tokens * 0.07) if threshold_budget_tokens else 900
        max_items = 5
        bfs_depth = 1
    target_tokens = max(450, min(target_tokens, 2600))

    return AttentionRetrievalPlan(
        mode=normalized_mode,
        query_text=str(query_text or "")[:1200],
        focus_node_id=str(focus.get("focus_node_id") or ""),
        focus_path=list(focus.get("focus_path") or []),
        focus_depth=int(focus.get("focus_depth") or 0),
        focus_node_address=str(focus.get("focus_node_address") or ""),
        focus_path_addresses=list(focus.get("focus_path_addresses") or []),
        seed_addresses=list(focus.get("seed_addresses") or []),
        graph_memory_ids=list(focus.get("graph_memory_ids") or []),
        bfs_depth=bfs_depth,
        target_tokens=target_tokens,
        max_items=max_items,
        trigger_reason=trigger_reason,
        pressure_percent=float(pressure_percent or 0.0),
        include_navigation_map=bool(include_navigation_map),
        navigation_reason=navigation_reason or (trigger_reason if include_navigation_map else ""),
    )


def build_navigation_map(
    task_graph: Any,
    plan: AttentionRetrievalPlan,
    uncovered_node_ids: Optional[Sequence[str]] = None,
) -> str:
    if not plan.include_navigation_map or task_graph is None:
        return ""
    if not hasattr(task_graph, "render_navigator_map"):
        return ""
    try:
        return str(task_graph.render_navigator_map(
            plan.focus_node_id or "",
            uncovered_node_ids=list(uncovered_node_ids or []),
        ) or "")
    except Exception as exc:
        logger.debug("[AttentionContext] 导航地图构建跳过: %s", exc)
        return ""


def retrieve_attention_context(memory_graph: Any, plan: AttentionRetrievalPlan) -> Tuple[List[AttentionContextItem], Dict[str, Any]]:
    if memory_graph is None or not hasattr(memory_graph, "retrieve_context"):
        return [], {"retrieved_memory_count": 0, "retrieve_error": "memory_graph_unavailable"}
    try:
        raw_results = _run_maybe_async(memory_graph.retrieve_context(
            plan.query_text or plan.focus_node_address or "当前任务上下文",
            top_k=max(plan.max_items, 3),
        ))
    except Exception as exc:
        logger.debug("[AttentionContext] MemoryGraph 检索失败: %s", exc)
        return [], {"retrieved_memory_count": 0, "retrieve_error": str(exc)[:160]}

    items: List[AttentionContextItem] = []
    for raw in list(raw_results or [])[: plan.max_items * 2]:
        item = _memory_result_to_item(raw)
        if item.text:
            items.append(item)
        if item.graph_memory_id and item.graph_memory_id not in plan.graph_memory_ids:
            plan.graph_memory_ids.append(item.graph_memory_id)
        if item.memory_address:
            plan.memory_addresses.append(item.memory_address)
        for sid in item.source_node_ids:
            if sid and sid not in plan.source_node_ids:
                plan.source_node_ids.append(sid)
    items.sort(key=lambda it: float(it.score or 0.0), reverse=True)
    return items[: plan.max_items], {"retrieved_memory_count": len(items)}


def run_bfs_expansion(memory_graph: Any, plan: AttentionRetrievalPlan) -> Tuple[List[AttentionContextItem], Dict[str, Any]]:
    if memory_graph is None:
        return [], {"bfs_seed_count": 0, "bfs_activated_count": 0}
    seeds = _derive_bfs_seed_ids(plan, memory_graph)
    if not seeds:
        return [], {"bfs_seed_count": 0, "bfs_activated_count": 0}
    try:
        if hasattr(memory_graph, "compute_activations_dynamic"):
            acts = memory_graph.compute_activations_dynamic(seeds, usage_ratio=min(max(plan.pressure_percent / 100.0, 0.0), 2.0))
        elif hasattr(memory_graph, "compute_activations"):
            acts = memory_graph.compute_activations(seeds, max_depth=plan.bfs_depth, decay=0.5, min_activation=0.01)
        else:
            acts = {}
    except Exception as exc:
        logger.debug("[AttentionContext] BFS 扩散失败: %s", exc)
        return [], {"bfs_seed_count": len(seeds), "bfs_activated_count": 0, "bfs_error": str(exc)[:160]}

    items: List[AttentionContextItem] = []
    for node_id, score in sorted((acts or {}).items(), key=lambda kv: -float(kv[1] or 0.0)):
        node_id = str(node_id)
        if node_id in seeds:
            continue
        text = _memory_node_preview(memory_graph, node_id)
        item = AttentionContextItem(
            text=text or f"BFS 激活候选节点 {node_id}",
            source=f"graph_memory_id={node_id}",
            score=float(score or 0.0),
            kind="bfs",
            graph_memory_id=node_id,
            node_id=node_id,
        )
        items.append(item)
        if len(items) >= plan.max_items:
            break
    return items, {"bfs_seed_count": len(seeds), "bfs_activated_count": len(acts or {})}


def build_attention_context_bundle(
    *,
    mode: Any,
    query_text: str,
    attention_window: Any = None,
    task_graph: Any = None,
    memory_graph: Any = None,
    pressure_percent: float = 0.0,
    trigger_reason: str = "model_call",
    include_navigation_map: bool = False,
    navigation_reason: str = "",
    uncovered_node_ids: Optional[Sequence[str]] = None,
) -> AttentionContextBundle:
    focus = resolve_focus_position(
        attention_window=attention_window,
        task_graph=task_graph,
        memory_graph=memory_graph,
    )
    threshold_budget_tokens = int(getattr(attention_window, "threshold_budget_tokens", 0) or 0)
    plan = build_attention_retrieval_plan(
        mode=mode,
        query_text=query_text,
        pressure_percent=pressure_percent,
        trigger_reason=trigger_reason,
        focus=focus,
        threshold_budget_tokens=threshold_budget_tokens,
        include_navigation_map=include_navigation_map,
        navigation_reason=navigation_reason,
    )
    task_items = _task_context_items(task_graph, plan)
    navigation_map = build_navigation_map(task_graph, plan, uncovered_node_ids=uncovered_node_ids)
    memory_items, retrieval_meta = retrieve_attention_context(memory_graph, plan)
    bfs_items, bfs_meta = run_bfs_expansion(memory_graph, plan)

    telemetry = {
        **retrieval_meta,
        **bfs_meta,
        "mode": plan.mode,
        "focus_address": plan.focus_node_address,
        "focus_path": list(plan.focus_path),
        "navigation_map_injected": bool(navigation_map),
        "navigation_map_reason": plan.navigation_reason if navigation_map else "skipped",
    }
    bundle = AttentionContextBundle(
        plan=plan,
        focus_summary=focus.get("focus_summary", ""),
        navigation_map=navigation_map,
        task_items=task_items,
        memory_items=memory_items,
        bfs_items=bfs_items,
        telemetry=telemetry,
    )
    rendered = render_attention_context(bundle)
    telemetry["active_context_token_estimate"] = estimate_text_tokens(rendered)
    telemetry["active_context_item_count"] = len(bundle.all_items)
    return bundle


def render_attention_context(bundle: AttentionContextBundle) -> str:
    plan = bundle.plan
    lines: List[str] = []
    focus = plan.focus_node_address or plan.focus_node_id or "unknown"
    lines.append("【注意力计划】")
    lines.append(
        f"mode={plan.mode}; reason={plan.trigger_reason}; pressure={plan.pressure_percent:.1f}%; focus={focus}"
    )
    lines.append("动态注意力=图位置感知的本轮上下文检索与重组；不是压缩器，不维护 backing_pool。")
    if bundle.focus_summary:
        lines.append("【思维导航】")
        lines.extend(_limit_lines(str(bundle.focus_summary), 5))
    elif plan.focus_path_addresses:
        lines.append("【思维导航】")
        lines.append(" › ".join(plan.focus_path_addresses[-5:]))

    if bundle.navigation_map:
        lines.append("【导航地图】")
        lines.extend(_limit_lines(bundle.navigation_map, 15))

    rendered_items: List[str] = []
    token_budget = max(300, plan.target_tokens)
    used_tokens = estimate_text_tokens("\n".join(lines))
    for idx, item in enumerate(_rank_context_items(bundle), 1):
        line = _render_context_item(idx, item)
        line_tokens = estimate_text_tokens(line)
        if rendered_items and used_tokens + line_tokens > token_budget:
            break
        rendered_items.append(line)
        used_tokens += line_tokens
        if len(rendered_items) >= plan.max_items:
            break
    if rendered_items:
        lines.append("【当前必要上下文】")
        lines.extend(rendered_items)

    jump_lines = _address_jump_lines(bundle)
    if jump_lines:
        lines.append("【地址回跳】")
        lines.extend(jump_lines[:8])

    return "\n".join(line for line in lines if line is not None).strip()


def render_attention_context_message(bundle: AttentionContextBundle) -> Dict[str, str]:
    return {"role": "system", "content": render_attention_context(bundle)}


def _resolve_task_focus(task_graph: Any, focus_node_id: str) -> str:
    try:
        if focus_node_id and task_graph.get_node(focus_node_id):
            return focus_node_id
    except Exception:
        pass
    try:
        for status in ("in_progress", "needs_adjust", "pending"):
            nodes = [n for n in task_graph.get_nodes_by_status(status) if not str(getattr(n, "id", "")).startswith("crg_")]
            if nodes:
                return str(nodes[0].id)
    except Exception:
        pass
    try:
        if task_graph.get_node("req"):
            return "req"
    except Exception:
        pass
    try:
        nodes = list(getattr(task_graph, "_nodes", {}).keys())
        if nodes:
            return str(nodes[0])
    except Exception:
        pass
    return focus_node_id or ""


def _task_focus_path(task_graph: Any, focus_node_id: str) -> List[str]:
    if not task_graph or not focus_node_id:
        return []
    path: List[str] = []
    try:
        ancestors = list(task_graph.get_ancestor_chain(focus_node_id) or [])
        path = [str(getattr(n, "id", n)) for n in reversed(ancestors)]
    except Exception:
        path = []
    if focus_node_id:
        path.append(str(focus_node_id))
    return [p for p in path if p]


def _safe_task_address(task_graph: Any, node_id: str) -> str:
    if not node_id:
        return ""
    try:
        return str(task_graph.get_node_address(node_id))
    except Exception:
        return f"tg:{getattr(task_graph, 'id', 'unknown')}/{node_id}"


def _get_memory_focus(memory_graph: Any) -> Dict[str, Any]:
    if memory_graph is None or not hasattr(memory_graph, "get_last_focus_context"):
        return {}
    try:
        return dict(memory_graph.get_last_focus_context() or {})
    except Exception:
        return {}


def _memory_focus_summary(memory_graph: Any, memory_focus: Dict[str, Any]) -> str:
    if memory_graph is not None and hasattr(memory_graph, "get_focus_path_summary"):
        try:
            summary = memory_graph.get_focus_path_summary()
            if summary:
                return str(summary)
        except Exception:
            pass
    path = memory_focus.get("focus_path") or []
    if path:
        return "MemoryGraph focus_path: " + " › ".join(str(x) for x in path[-6:])
    return ""


def _task_context_items(task_graph: Any, plan: AttentionRetrievalPlan) -> List[AttentionContextItem]:
    if task_graph is None or not plan.focus_node_id:
        return []
    items: List[AttentionContextItem] = []
    try:
        focus_node = task_graph.get_node(plan.focus_node_id)
    except Exception:
        focus_node = None
    if focus_node:
        label = getattr(focus_node, "label", "") or getattr(focus_node, "id", plan.focus_node_id)
        desc = getattr(focus_node, "desc", "") or getattr(focus_node, "description", "") or ""
        result = getattr(focus_node, "result", "") or ""
        status = getattr(focus_node, "status", "") or ""
        body = f"当前焦点 {label} status={status}"
        if desc:
            body += f"；说明={_compact_text(desc, 180)}"
        if result:
            body += f"；结果={_compact_text(result, 180)}"
        items.append(AttentionContextItem(
            text=body,
            source=plan.focus_node_address or _safe_task_address(task_graph, plan.focus_node_id),
            score=3.0,
            kind="task",
            node_id=plan.focus_node_id,
        ))
    try:
        for child in list(task_graph.get_children(plan.focus_node_id) or [])[:4]:
            if str(getattr(child, "id", "")).startswith("crg_"):
                continue
            items.append(AttentionContextItem(
                text=f"子节点 {getattr(child, 'id', '')}: {getattr(child, 'label', '')} status={getattr(child, 'status', '')}",
                source=_safe_task_address(task_graph, getattr(child, "id", "")),
                score=1.8,
                kind="task",
                node_id=str(getattr(child, "id", "")),
            ))
    except Exception:
        pass
    return items


def _memory_result_to_item(raw: Any) -> AttentionContextItem:
    if not isinstance(raw, dict):
        return AttentionContextItem(text=str(raw or ""), kind="memory")
    addr = raw.get("memory_address") or {}
    metadata = raw.get("metadata") or {}
    graph_memory_id = str(raw.get("graph_memory_id") or raw.get("node_id") or addr.get("graph_memory_id") or "")
    shard_id = str(raw.get("shard_id") or addr.get("shard_id") or "")
    source_node_ids = raw.get("source_node_ids") or addr.get("source_node_ids") or metadata.get("source_node_ids") or []
    if isinstance(source_node_ids, str):
        source_node_ids = [source_node_ids]
    summary = raw.get("summary") or raw.get("content_summary") or metadata.get("content_summary") or ""
    content = raw.get("content") or raw.get("text") or ""
    label = raw.get("label") or raw.get("full_path") or graph_memory_id
    text = _compact_text(summary or content or label, 320)
    if summary and content and summary != content:
        text = _compact_text(summary, 220) + "；详情线索=" + _compact_text(content, 120)
    source_parts: List[str] = []
    if graph_memory_id:
        source_parts.append(f"graph_memory_id={graph_memory_id}")
    if shard_id:
        source_parts.append(f"shard_id={shard_id}")
    if addr.get("full_path") or raw.get("full_path"):
        source_parts.append(f"full_path={addr.get('full_path') or raw.get('full_path')}")
    return AttentionContextItem(
        text=text,
        source=", ".join(source_parts),
        score=float(raw.get("score", raw.get("relevance", 0.0)) or 0.0),
        kind=str(raw.get("source") or raw.get("node_type") or "memory"),
        graph_memory_id=graph_memory_id,
        shard_id=shard_id,
        memory_address=dict(addr) if isinstance(addr, dict) else {},
        source_node_ids=[str(x) for x in source_node_ids if x],
        node_id=str(raw.get("node_id") or graph_memory_id),
    )


def _derive_bfs_seed_ids(plan: AttentionRetrievalPlan, memory_graph: Any) -> List[str]:
    seeds: List[str] = []
    for gid in plan.graph_memory_ids:
        if gid and gid not in seeds:
            seeds.append(gid)
    for sid in plan.source_node_ids:
        if sid and sid not in seeds:
            seeds.append(sid)
    for addr in plan.memory_addresses:
        if isinstance(addr, dict):
            gid = addr.get("graph_memory_id") or addr.get("node_id")
            if gid and gid not in seeds:
                seeds.append(str(gid))
    # Keep only existing nodes when the graph exposes has_node; otherwise keep seeds.
    if hasattr(memory_graph, "has_node"):
        filtered = []
        for seed in seeds:
            try:
                if memory_graph.has_node(seed):
                    filtered.append(seed)
            except Exception:
                pass
        return filtered
    return seeds


def _memory_node_preview(memory_graph: Any, node_id: str) -> str:
    node = None
    for method_name in ("get_node", "get_node_properties"):
        method = getattr(memory_graph, method_name, None)
        if not method:
            continue
        try:
            node = method(node_id)
            if node:
                break
        except Exception:
            continue
    if not node:
        return ""
    metadata = getattr(node, "metadata", {}) or {}
    content = (
        metadata.get("content_summary")
        or metadata.get("summary")
        or metadata.get("content")
        or getattr(node, "content", "")
        or getattr(node, "label", "")
        or node_id
    )
    return _compact_text(str(content), 260)


def _rank_context_items(bundle: AttentionContextBundle) -> List[AttentionContextItem]:
    return sorted(bundle.all_items, key=lambda item: (float(item.score or 0.0), item.kind == "task"), reverse=True)


def _render_context_item(index: int, item: AttentionContextItem) -> str:
    return f"{index}. [来源: {item.address_line()}] {_compact_text(item.text, 360)}"


def _address_jump_lines(bundle: AttentionContextBundle) -> List[str]:
    plan = bundle.plan
    lines: List[str] = []
    if plan.focus_node_address:
        lines.append(plan.focus_node_address)
    for addr in plan.focus_path_addresses[-5:]:
        if addr and addr not in lines:
            lines.append(addr)
    for item in bundle.all_items:
        if item.graph_memory_id:
            line = f"graph_memory_id={item.graph_memory_id}"
            if item.shard_id:
                line += f"; shard_id={item.shard_id}"
            if line not in lines:
                lines.append(line)
        if item.memory_address:
            addr_line = _compact_memory_address(item.memory_address)
            if addr_line and addr_line not in lines:
                lines.append(addr_line)
    return lines


def _run_maybe_async(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    # _call_model runs in the FC worker thread, but if this helper is ever called
    # inside an existing loop, use a private loop in a temporary thread-safe way.
    import threading
    result: Dict[str, Any] = {}
    error: Dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(value)
        except BaseException as exc:  # pragma: no cover - defensive bridge
            error["error"] = exc

    t = threading.Thread(target=_runner, name="attention-context-async-bridge", daemon=True)
    t.start()
    t.join(timeout=10)
    if "error" in error:
        raise error["error"]
    return result.get("value", [])


def _limit_lines(text: str, max_lines: int) -> List[str]:
    lines = [ln.rstrip() for ln in str(text or "").splitlines() if ln.strip()]
    if len(lines) <= max_lines:
        return lines
    return lines[: max_lines - 1] + [f"… 省略 {len(lines) - max_lines + 1} 行"]


def _compact_text(text: Any, max_chars: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def _compact_memory_address(addr: Dict[str, Any]) -> str:
    if not isinstance(addr, dict):
        return ""
    parts = []
    for key in ("graph_memory_id", "node_id", "shard_id", "full_path", "summary_ref_id", "source"):
        val = addr.get(key)
        if val:
            parts.append(f"{key}={val}")
    src = addr.get("source_node_ids")
    if src:
        if isinstance(src, (list, tuple)):
            parts.append("source_node_ids=" + ",".join(str(x) for x in src[:4]))
        else:
            parts.append(f"source_node_ids={src}")
    return "; ".join(parts)
