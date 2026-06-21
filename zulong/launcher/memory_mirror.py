"""Mirror Web conversation events into MemoryGraph.

MemoryGraph is the authoritative long-term store for conversations, tasks,
tools, approvals, and other interaction events. InteractionStore is only a
small SQLite cache/ledger for fast Web session restoration and audit queries.
This module makes sure every user-facing interaction is represented in the
graph even when the L1-B/L2 route changes.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

_LAST_ROUND_BY_CONVERSATION: Dict[str, str] = {}
_LAST_TOOL_CALL_BY_ROUND: Dict[str, str] = {}


def _compact_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value or ""))


def _edge_exists(graph: Any, source: str, target: str) -> bool:
    if hasattr(graph, "has_edge"):
        try:
            return bool(graph.has_edge(source, target))
        except Exception:
            pass
    try:
        return bool(getattr(graph, "_graph").has_edge(source, target))
    except Exception:
        return False


def _persist_node_update(graph: Any, node: Any) -> None:
    if hasattr(graph, "update_node"):
        try:
            graph.update_node(node)
        except Exception:
            pass


def _metadata_of(node: Any) -> Dict[str, Any]:
    return getattr(node, "metadata", {}) or {}


def _node_type_value(node: Any) -> str:
    node_type = getattr(node, "node_type", "")
    return getattr(node_type, "value", node_type) or ""


def _node_id_of(node: Any) -> str:
    return getattr(node, "node_id", "") or ""


def _node_type_for_delta(node_type: Any) -> str:
    return str(getattr(node_type, "value", node_type) or "")


def _node_delta_payload(
    *,
    node_id: str,
    node_type: Any,
    label: str,
    metadata: Optional[Dict[str, Any]] = None,
    backend_ref: str = "",
    activation: float = 0.0,
) -> Dict[str, Any]:
    metadata = dict(metadata or {})
    return {
        "id": node_id,
        "type": _node_type_for_delta(node_type),
        "label": label,
        "activation": activation,
        "metadata": metadata,
        "backend_ref": backend_ref,
        "children_count": 0,
        "graph_memory_id": node_id,
        "full_path": metadata.get("full_path") or metadata.get("graph_address") or node_id,
    }


def _edge_delta_payload(
    source: str,
    target: str,
    edge_type: Any,
    *,
    weight: float = 1.0,
    protected: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "type": _node_type_for_delta(edge_type),
        "weight": weight,
        "protected": protected,
        "metadata": dict(metadata or {}),
    }


def _get_payload_interaction(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    interaction = payload.get("interaction")
    if isinstance(interaction, dict):
        return interaction
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("interaction"), dict):
        return data["interaction"]
    return {}


def _payload_source_event_id(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    return (
        str(payload.get("source_event_id") or "")
        or str(payload.get("event_id") or "")
        or str(payload.get("id") or "")
    )


def _payload_task_graph_id(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(
        payload.get("task_graph_id")
        or payload.get("graph_id")
        or payload.get("task_id")
        or ""
    )


def _payload_files(payload: Dict[str, Any], interaction: Dict[str, Any]) -> List[str]:
    files: List[str] = []
    candidates: Iterable[Any] = (
        payload.get("files") if isinstance(payload, dict) else None,
        payload.get("affected_files") if isinstance(payload, dict) else None,
        interaction.get("affected_files") if isinstance(interaction, dict) else None,
        payload.get("path") if isinstance(payload, dict) else None,
        payload.get("file_path") if isinstance(payload, dict) else None,
    )
    for item in candidates:
        if not item:
            continue
        if isinstance(item, (list, tuple, set)):
            for value in item:
                if value:
                    files.append(str(value))
        elif isinstance(item, str):
            files.append(item)
    if isinstance(payload, dict):
        for result in payload.get("results") or []:
            if isinstance(result, dict):
                for key in ("path", "file_path"):
                    if result.get(key):
                        files.append(str(result[key]))
    seen = set()
    deduped = []
    for path in files:
        norm = path.strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(norm)
    return deduped


def _event_node_type(event_type: str, interaction: Dict[str, Any], payload: Dict[str, Any]):
    try:
        from zulong.memory.memory_graph import NodeType
    except Exception:
        return None
    kind = str(interaction.get("kind") or "").lower()
    event = str(event_type or "").lower()
    if kind == "approval" or "approval" in event:
        return NodeType.APPROVAL
    if kind == "action":
        return NodeType.TOOL_CALL
    if kind == "observation":
        if (
            interaction.get("tool_name")
            or payload.get("tool_name")
            or payload.get("results")
            or "tool" in event
        ):
            return NodeType.TOOL_RESULT
    if event in {"tool_call", "ide_tool_request", "ide_tool_exec"}:
        return NodeType.TOOL_CALL
    if event in {"ide_tool_result"} or event.endswith("tool_finished"):
        return NodeType.TOOL_RESULT
    return None


def _stable_execution_node_id(
    round_id: str,
    node_type: Any,
    event_type: str,
    interaction: Dict[str, Any],
    payload: Dict[str, Any],
) -> str:
    node_type_value = getattr(node_type, "value", str(node_type))
    pair_id = (
        interaction.get("pair_id")
        or interaction.get("interaction_id")
        or payload.get("pair_id")
        or payload.get("call_id")
        or payload.get("tool_call_id")
        or payload.get("approval_id")
        or payload.get("approvalId")
        or _payload_source_event_id(payload)
        or event_type
    )
    phase = interaction.get("phase") or payload.get("phase") or event_type
    if node_type_value == "approval":
        suffix = f"{pair_id}_{phase}_{interaction.get('status') or payload.get('approved', '')}"
    else:
        suffix = str(pair_id)
    return f"{round_id}/{node_type_value}_{_compact_id(suffix)}"


def _find_execution_node_by_pair(mg: Any, node_type: Any, round_id: str, pair_id: str) -> str:
    if not pair_id:
        return ""
    try:
        for node in mg.get_nodes_by_type(node_type):
            meta = _metadata_of(node)
            if meta.get("parent_round") == round_id and str(meta.get("pair_id") or "") == str(pair_id):
                return _node_id_of(node)
    except Exception:
        pass
    return ""


def _add_file_references(
    mg: Any,
    source_node_id: str,
    files: List[str],
    metadata: Dict[str, Any],
    record_node_change: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    record_edge_change: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> None:
    if not files:
        return
    try:
        from zulong.memory.memory_graph import EdgeType, GraphNode, NodeType
    except Exception:
        return
    for path in files[:20]:
        safe_path = path.replace("\\", "/")
        file_id = f"file:{_compact_id(safe_path)}"
        if not mg.has_node(file_id):
            file_metadata = {
                "path": path,
                "content": path,
                "source": "execution_event",
            }
            try:
                mg.add_node(GraphNode(
                    node_id=file_id,
                    node_type=NodeType.FILE,
                    label=os.path.basename(path) or path[-80:],
                    backend_ref=f"file:{path}",
                    metadata=file_metadata,
                ))
                if record_node_change:
                    record_node_change("add_node", _node_delta_payload(
                        node_id=file_id,
                        node_type=NodeType.FILE,
                        label=os.path.basename(path) or path[-80:],
                        backend_ref=f"file:{path}",
                        metadata=file_metadata,
                    ))
            except Exception:
                continue
        if mg.has_node(file_id) and not _edge_exists(mg, source_node_id, file_id):
            try:
                edge_metadata = {**metadata, "link_type": "execution_file_reference"}
                if mg.add_edge(
                    source_node_id,
                    file_id,
                    EdgeType.REFERENCE,
                    weight=0.8,
                    protected=True,
                    metadata=edge_metadata,
                ) and record_edge_change:
                    record_edge_change("add_edge", _edge_delta_payload(
                        source_node_id,
                        file_id,
                        EdgeType.REFERENCE,
                        weight=0.8,
                        protected=True,
                        metadata=edge_metadata,
                    ))
            except Exception:
                pass


def mirror_interaction_to_memory_graph(
    *,
    conversation_id: Optional[str],
    turn_id: Optional[str],
    role: str,
    text: str,
    event_type: str = "message",
    source: str = "web_chat",
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Best-effort MemoryGraph mirror for Web conversation events."""
    if not conversation_id or not turn_id or not text:
        return
    try:
        from zulong.memory.memory_graph import (
            EdgeType,
            GraphNode,
            Importance,
            NodeType,
            get_memory_graph,
        )

        mg = get_memory_graph()
        if mg is None:
            return

        now = time.time()
        safe_conversation_id = _compact_id(conversation_id)
        safe_turn_id = _compact_id(turn_id)
        safe_role = _compact_id(role or "system")
        safe_event_type = _compact_id(event_type or "message")
        payload = payload or {}
        explicit_session_id = str(
            payload.get("session_node_id")
            or payload.get("dialogue_session_id")
            or ""
        ).strip()
        if explicit_session_id.startswith("dialogue:session_") and "/" not in explicit_session_id:
            session_id = explicit_session_id
        else:
            session_id = f"dialogue:session_{safe_conversation_id}"
        round_id = f"{session_id}/round_{safe_turn_id}"
        message_id = f"{round_id}/{safe_role}_{safe_event_type}"
        changes: List[Dict[str, Any]] = []
        changed_node_ids: Set[str] = set()
        changed_edge_ids: Set[str] = set()

        def add_node_change(action: str, data: Dict[str, Any]) -> None:
            node_id = str(data.get("id") or "")
            if node_id:
                changed_node_ids.add(node_id)
            changes.append({"action": action, "data": data})

        def add_edge_change(action: str, data: Dict[str, Any]) -> None:
            source = str(data.get("source") or "")
            target = str(data.get("target") or "")
            edge_type = str(data.get("type") or "")
            if source or target:
                changed_edge_ids.add(f"{source}|{target}|{edge_type}")
            changes.append({"action": action, "data": data})

        if not mg.has_node(session_id):
            session_metadata = {
                "sub_type": "session",
                "conversation_id": conversation_id,
                "source": source,
                "full_path": session_id,
                "topic_summary": text[:200],
                "bound_window_id": conversation_id,
                "window_binding": "web_chat",
                "is_root_node": True,
                "graph_level": 0,
                "node_role": "session_root",
                "round_count": 0,
            }
            mg.add_node(GraphNode(
                node_id=session_id,
                node_type=NodeType.DIALOGUE,
                label=f"Web 会话 {conversation_id[-8:]}",
                backend_ref=f"interaction:{conversation_id}",
                metadata=session_metadata,
            ))
            add_node_change("add_node", _node_delta_payload(
                node_id=session_id,
                node_type=NodeType.DIALOGUE,
                label=f"Web 会话 {conversation_id[-8:]}",
                backend_ref=f"interaction:{conversation_id}",
                metadata=session_metadata,
            ))
            # TSD §23.11.6: 回写 session_node_id 到 InteractionStore
            try:
                from zulong.launcher.interaction_store import get_interaction_store
                _store = get_interaction_store()
                _existing = _store.get_conversation(conversation_id)
                if _existing and not _existing.get("session_node_id"):
                    _store.upsert_conversation(
                        conversation_id,
                        session_node_id=session_id,
                    )
            except Exception:
                pass  # best-effort, 不阻塞主流程

        round_exists = mg.has_node(round_id)
        prev_round_id = _LAST_ROUND_BY_CONVERSATION.get(conversation_id) or ""
        if not round_exists:
            if not prev_round_id:
                prev_round_id = _find_latest_round_for_conversation(mg, conversation_id)
            round_metadata = {
                "sub_type": "round",
                "conversation_id": conversation_id,
                "request_id": turn_id,
                "goal": text[:500],
                "user_text": text if role == "user" else "",
                "created_from": "interaction_store_mirror",
                "full_path": round_id,
                "parent_session": session_id,
                "session_id": session_id,
                "prev_round_id": prev_round_id or "",
                "is_root_node": False,
                "graph_level": 1,
                "node_role": "dialogue_round",
            }
            mg.add_node(GraphNode(
                node_id=round_id,
                node_type=NodeType.DIALOGUE,
                label=text[:80],
                backend_ref=f"interaction:{conversation_id}/{turn_id}",
                metadata=round_metadata,
            ))
            add_node_change("add_node", _node_delta_payload(
                node_id=round_id,
                node_type=NodeType.DIALOGUE,
                label=text[:80],
                backend_ref=f"interaction:{conversation_id}/{turn_id}",
                metadata=round_metadata,
            ))
        if not _edge_exists(mg, session_id, round_id):
            if mg.add_edge(session_id, round_id, EdgeType.HIERARCHY, weight=1.0, protected=True):
                add_edge_change("add_edge", _edge_delta_payload(
                    session_id,
                    round_id,
                    EdgeType.HIERARCHY,
                    weight=1.0,
                    protected=True,
                ))
        if prev_round_id and mg.has_node(prev_round_id) and not _edge_exists(mg, prev_round_id, round_id):
            if mg.add_edge(prev_round_id, round_id, EdgeType.TEMPORAL, weight=1.0, protected=True):
                add_edge_change("add_edge", _edge_delta_payload(
                    prev_round_id,
                    round_id,
                    EdgeType.TEMPORAL,
                    weight=1.0,
                    protected=True,
                ))
        sess_node = mg.get_node(session_id)
        if sess_node:
            sess_node.metadata["round_count"] = _count_round_children(mg, session_id)
            sess_node.metadata["last_round_id"] = round_id
            sess_node.metadata["last_active_at"] = now
            _persist_node_update(mg, sess_node)
            add_node_change("update_node", _node_delta_payload(
                node_id=session_id,
                node_type=NodeType.DIALOGUE,
                label=getattr(sess_node, "label", f"Web 会话 {conversation_id[-8:]}"),
                backend_ref=getattr(sess_node, "backend_ref", f"interaction:{conversation_id}"),
                metadata=getattr(sess_node, "metadata", {}) or {},
                activation=float(getattr(sess_node, "activation", 0.0) or 0.0),
            ))
        elif role == "user":
            node = mg.get_node(round_id)
            if node:
                node.metadata["goal"] = text[:500]
                node.metadata["user_text"] = text
                node.metadata["session_id"] = session_id
                node.metadata["parent_session"] = session_id
                node.metadata["full_path"] = round_id
                node.metadata["node_role"] = "dialogue_round"
                node.metadata["graph_level"] = 1
                _persist_node_update(mg, node)
                add_node_change("update_node", _node_delta_payload(
                    node_id=round_id,
                    node_type=NodeType.DIALOGUE,
                    label=getattr(node, "label", text[:80]),
                    backend_ref=getattr(node, "backend_ref", f"interaction:{conversation_id}/{turn_id}"),
                    metadata=getattr(node, "metadata", {}) or {},
                    activation=float(getattr(node, "activation", 0.0) or 0.0),
                ))

        _LAST_ROUND_BY_CONVERSATION[conversation_id] = round_id

        message_metadata = {
            "sub_type": "agent_turn" if role != "user" else "user_turn",
            "conversation_id": conversation_id,
            "request_id": turn_id,
            "event_type": event_type,
            "role": role,
            "source": source,
            "content": text,
            "payload": payload,
            "created_at": now,
            "parent_round": round_id,
            "parent_session": session_id,
            "full_path": message_id,
            "is_root_node": False,
            "graph_level": 2,
            "node_role": "message",
        }
        message_exists = mg.has_node(message_id)
        mg.add_node(GraphNode(
            node_id=message_id,
            node_type=NodeType.DIALOGUE,
            label=(role or "message")[:20],
            backend_ref=f"interaction:{conversation_id}/{turn_id}/{event_type}/{role}",
            metadata=message_metadata,
        ))
        add_node_change("update_node" if message_exists else "add_node", _node_delta_payload(
            node_id=message_id,
            node_type=NodeType.DIALOGUE,
            label=(role or "message")[:20],
            backend_ref=f"interaction:{conversation_id}/{turn_id}/{event_type}/{role}",
            metadata=message_metadata,
        ))
        if not _edge_exists(mg, round_id, message_id):
            if mg.add_edge(round_id, message_id, EdgeType.HIERARCHY, weight=1.0, protected=True):
                add_edge_change("add_edge", _edge_delta_payload(
                    round_id,
                    message_id,
                    EdgeType.HIERARCHY,
                    weight=1.0,
                    protected=True,
                ))
        if role == "assistant":
            node = mg.get_node(round_id)
            if node:
                node.metadata["bot_text"] = text
                node.metadata["status"] = "completed"
                node.metadata["completed_at"] = now
                _persist_node_update(mg, node)
                add_node_change("update_node", _node_delta_payload(
                    node_id=round_id,
                    node_type=NodeType.DIALOGUE,
                    label=getattr(node, "label", text[:80]),
                    backend_ref=getattr(node, "backend_ref", f"interaction:{conversation_id}/{turn_id}"),
                    metadata=getattr(node, "metadata", {}) or {},
                    activation=float(getattr(node, "activation", 0.0) or 0.0),
                ))
            try:
                mg.index_summary(round_id, f"{node.metadata.get('goal', '') if node else ''} {text}"[:500])
            except Exception:
                pass
        execution_node_id = _mirror_execution_event(
            mg=mg,
            session_id=session_id,
            round_id=round_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            role=role,
            text=text,
            event_type=event_type,
            source=source,
            payload=payload or {},
            record_node_change=add_node_change,
            record_edge_change=add_edge_change,
        )
        if event_type in {"approval_required", "checkpoint_created", "ide:file_changed"}:
            try:
                mg.set_importance(message_id, Importance.IMPORTANT)
            except Exception:
                pass
        if execution_node_id:
            try:
                mg.set_importance(execution_node_id, Importance.IMPORTANT)
            except Exception:
                pass
        try:
            if role == "assistant":
                round_node = mg.get_node(round_id)
                if round_node:
                    round_node.metadata["content"] = (
                        f"{round_node.metadata.get('goal', '')}\n{text}"
                    ).strip()[:1000]
                    _persist_node_update(mg, round_node)
                    add_node_change("update_node", _node_delta_payload(
                        node_id=round_id,
                        node_type=NodeType.DIALOGUE,
                        label=getattr(round_node, "label", text[:80]),
                        backend_ref=getattr(round_node, "backend_ref", f"interaction:{conversation_id}/{turn_id}"),
                        metadata=getattr(round_node, "metadata", {}) or {},
                        activation=float(getattr(round_node, "activation", 0.0) or 0.0),
                    ))
        except Exception:
            pass
        try:
            focus_node_id = execution_node_id or message_id
            if not mg.update_focus_to_node(focus_node_id):
                mg.set_active_nodes([session_id, round_id, focus_node_id])
        except Exception:
            try:
                mg.set_active_nodes([session_id, round_id, execution_node_id or message_id])
            except Exception:
                pass
        try:
            from zulong.core.event_bus import event_bus
            from zulong.core.types import EventPriority, EventType, ZulongEvent

            event_bus.publish(ZulongEvent(
                type=EventType.MEMORY_GRAPH_UPDATED,
                priority=EventPriority.LOW,
                source="MemoryMirror",
                payload={
                    "update_type": "delta",
                    "ts": time.time(),
                    "nodes": [],
                    "edges": [],
                    "changes": changes,
                    "changed_node_ids": sorted(changed_node_ids),
                    "changed_edge_ids": sorted(changed_edge_ids),
                    "active_node_ids": list(getattr(mg, "_active_node_ids", []) or []),
                    "stats": {
                        "transport": "delta",
                        "changed_nodes": len(changed_node_ids),
                        "changed_edges": len(changed_edge_ids),
                    },
                },
            ))
        except Exception:
            pass
    except Exception as exc:
        logger.debug("[MemoryMirror] mirror skipped: %s", exc)


def _find_latest_round_for_conversation(mg: Any, conversation_id: str) -> Optional[str]:
    try:
        from zulong.memory.memory_graph import NodeType

        candidates = []
        for node in mg.get_nodes_by_type(NodeType.DIALOGUE):
            if (
                node.metadata.get("sub_type") == "round"
                and node.metadata.get("conversation_id") == conversation_id
            ):
                candidates.append(node)
        if not candidates:
            return None
        candidates.sort(key=lambda n: n.created_at)
        return candidates[-1].node_id
    except Exception:
        return None


def _count_round_children(mg: Any, session_id: str) -> int:
    try:
        return sum(
            1 for node in mg.get_children(session_id)
            if node.metadata.get("sub_type") == "round"
        )
    except Exception:
        return 0


def _mirror_execution_event(
    *,
    mg: Any,
    session_id: str,
    round_id: str,
    conversation_id: str,
    turn_id: str,
    role: str,
    text: str,
    event_type: str,
    source: str,
    payload: Dict[str, Any],
    record_node_change: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    record_edge_change: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> str:
    """Project tool/approval cards into dedicated MemoryGraph execution nodes."""
    interaction = _get_payload_interaction(payload)
    node_type = _event_node_type(event_type, interaction, payload)
    if node_type is None:
        return ""

    try:
        from zulong.memory.memory_graph import EdgeType, GraphNode, Importance
    except Exception:
        return ""

    node_type_value = getattr(node_type, "value", str(node_type))
    pair_id = str(
        interaction.get("pair_id")
        or payload.get("pair_id")
        or payload.get("call_id")
        or payload.get("tool_call_id")
        or payload.get("approval_id")
        or payload.get("approvalId")
        or _payload_source_event_id(payload)
        or event_type
    )
    tool_name = str(interaction.get("tool_name") or payload.get("tool_name") or "")
    source_event_id = _payload_source_event_id(payload)
    task_graph_id = _payload_task_graph_id(payload)
    status = str(interaction.get("status") or payload.get("status") or "")
    phase = str(interaction.get("phase") or payload.get("phase") or event_type)
    node_id = _stable_execution_node_id(round_id, node_type, event_type, interaction, payload)

    if node_type_value == "tool_call":
        title = interaction.get("title") or f"工具调用: {tool_name or pair_id}"
        content = interaction.get("detail") or text
        metadata = {
            "pair_id": pair_id,
            "tool_name": tool_name,
            "arguments_summary": interaction.get("tool_args") or payload.get("tool_args") or payload.get("args"),
            "fc_turn": interaction.get("turn") or payload.get("turn"),
            "task_graph_id": task_graph_id,
            "risk_level": interaction.get("risk_level") or payload.get("risk_level") or "",
            "source_event_id": source_event_id,
        }
    elif node_type_value == "tool_result":
        title = interaction.get("title") or f"工具结果: {tool_name or pair_id}"
        content = interaction.get("result_preview") or payload.get("result_preview") or text
        is_failed = status in {"failed", "error"} or bool(payload.get("is_error"))
        files = _payload_files(payload, interaction)
        metadata = {
            "pair_id": pair_id,
            "tool_name": tool_name,
            "success": not is_failed,
            "latency_ms": payload.get("latency_ms") or interaction.get("latency_ms"),
            "result_preview": content[:1000] if isinstance(content, str) else str(content)[:1000],
            "affected_files": files,
            "source_event_id": source_event_id,
        }
    else:
        approval_id = str(
            payload.get("approval_id")
            or payload.get("approvalId")
            or interaction.get("approval_id")
            or pair_id
        )
        decision = ""
        if "approved" in payload:
            decision = "approved" if payload.get("approved") else "rejected"
        elif status in {"approved", "rejected", "denied"}:
            decision = status
        title = interaction.get("title") or payload.get("action_summary") or "审批事件"
        content = interaction.get("detail") or payload.get("risk_reason") or text
        metadata = {
            "approval_id": approval_id,
            "pair_id": pair_id,
            "tool_name": tool_name,
            "phase": phase,
            "decision": decision,
            "risk_level": interaction.get("risk_level") or payload.get("risk_level") or "",
            "approval_mode": interaction.get("approval_mode") or payload.get("approval_mode") or "",
            "action_summary": payload.get("action_summary") or payload.get("summary") or interaction.get("title") or "",
            "source_event_id": source_event_id,
        }

    common_metadata = {
        "content": content,
        "conversation_id": conversation_id,
        "request_id": turn_id,
        "event_type": event_type,
        "role": role,
        "source": source,
        "interaction": interaction,
        "payload": payload,
        "parent_round": round_id,
        "parent_session": session_id,
        "full_path": node_id,
        "node_role": node_type_value,
        "graph_level": 2,
        "created_from": "interaction_store_mirror",
        "importance": Importance.IMPORTANT.value,
    }
    common_metadata.update({k: v for k, v in metadata.items() if v is not None})

    backend_ref = f"interaction:{conversation_id}/{turn_id}/{event_type}/{source_event_id or pair_id}"
    execution_exists = mg.has_node(node_id)
    mg.add_node(GraphNode(
        node_id=node_id,
        node_type=node_type,
        label=str(title)[:120],
        backend_ref=backend_ref,
        metadata=common_metadata,
    ))
    if record_node_change:
        record_node_change("update_node" if execution_exists else "add_node", _node_delta_payload(
            node_id=node_id,
            node_type=node_type,
            label=str(title)[:120],
            backend_ref=backend_ref,
            metadata=common_metadata,
        ))
    if not _edge_exists(mg, round_id, node_id):
        edge_metadata = {"link_type": "round_execution_event"}
        if mg.add_edge(
            round_id,
            node_id,
            EdgeType.HIERARCHY,
            weight=1.0,
            protected=True,
            metadata=edge_metadata,
        ) and record_edge_change:
            record_edge_change("add_edge", _edge_delta_payload(
                round_id,
                node_id,
                EdgeType.HIERARCHY,
                weight=1.0,
                protected=True,
                metadata=edge_metadata,
            ))

    _attach_execution_task_edges(
        mg,
        session_id,
        round_id,
        node_id,
        node_type_value,
        payload,
        record_edge_change=record_edge_change,
    )
    if node_type_value == "tool_call":
        previous_tool_call = _LAST_TOOL_CALL_BY_ROUND.get(round_id)
        if previous_tool_call and previous_tool_call != node_id and mg.has_node(previous_tool_call):
            if not _edge_exists(mg, previous_tool_call, node_id):
                edge_metadata = {"link_type": "tool_chain_order"}
                if mg.add_edge(
                    previous_tool_call,
                    node_id,
                    EdgeType.TEMPORAL,
                    weight=0.8,
                    protected=True,
                    metadata=edge_metadata,
                ) and record_edge_change:
                    record_edge_change("add_edge", _edge_delta_payload(
                        previous_tool_call,
                        node_id,
                        EdgeType.TEMPORAL,
                        weight=0.8,
                        protected=True,
                        metadata=edge_metadata,
                    ))
        _LAST_TOOL_CALL_BY_ROUND[round_id] = node_id
    elif node_type_value == "tool_result":
        call_node_id = _find_execution_node_by_pair(mg, node_type.__class__.TOOL_CALL, round_id, pair_id)
        if call_node_id and not _edge_exists(mg, call_node_id, node_id):
            edge_metadata = {"link_type": "tool_call_result", "pair_id": pair_id}
            if mg.add_edge(
                call_node_id,
                node_id,
                EdgeType.CAUSAL,
                weight=1.0,
                protected=True,
                metadata=edge_metadata,
            ) and record_edge_change:
                record_edge_change("add_edge", _edge_delta_payload(
                    call_node_id,
                    node_id,
                    EdgeType.CAUSAL,
                    weight=1.0,
                    protected=True,
                    metadata=edge_metadata,
                ))
        _add_file_references(
            mg,
            node_id,
            _payload_files(payload, interaction),
            {"pair_id": pair_id},
            record_node_change=record_node_change,
            record_edge_change=record_edge_change,
        )
    elif node_type_value == "approval":
        if status in {"approved", "rejected", "denied"} or common_metadata.get("decision"):
            request_node_id = _find_matching_approval_request(mg, round_id, pair_id, node_id)
            if request_node_id and not _edge_exists(mg, request_node_id, node_id):
                edge_metadata = {"link_type": "approval_request_decision", "pair_id": pair_id}
                if mg.add_edge(
                    request_node_id,
                    node_id,
                    EdgeType.CAUSAL,
                    weight=1.0,
                    protected=True,
                    metadata=edge_metadata,
                ) and record_edge_change:
                    record_edge_change("add_edge", _edge_delta_payload(
                        request_node_id,
                        node_id,
                        EdgeType.CAUSAL,
                        weight=1.0,
                        protected=True,
                        metadata=edge_metadata,
                    ))
            call_node_id = _find_execution_node_by_pair(mg, node_type.__class__.TOOL_CALL, round_id, pair_id)
            if call_node_id and not _edge_exists(mg, node_id, call_node_id):
                edge_metadata = {"link_type": "approval_decision_tool_call", "pair_id": pair_id}
                if mg.add_edge(
                    node_id,
                    call_node_id,
                    EdgeType.CAUSAL,
                    weight=0.9,
                    protected=True,
                    metadata=edge_metadata,
                ) and record_edge_change:
                    record_edge_change("add_edge", _edge_delta_payload(
                        node_id,
                        call_node_id,
                        EdgeType.CAUSAL,
                        weight=0.9,
                        protected=True,
                        metadata=edge_metadata,
                    ))
    return node_id


def _find_task_node_id(mg: Any, session_id: str, task_graph_id: str) -> str:
    if not task_graph_id:
        return ""
    if hasattr(mg, "get_task_node_id_for_graph"):
        try:
            node_id = mg.get_task_node_id_for_graph(task_graph_id)
            if node_id:
                return node_id
        except Exception:
            pass
    task_candidates = [
        f"task:{task_graph_id}",
        f"{session_id}/task:{task_graph_id}",
    ]
    for candidate in task_candidates:
        try:
            if mg.has_node(candidate):
                return candidate
        except Exception:
            pass
    try:
        from zulong.memory.memory_graph import NodeType
        for node in mg.get_nodes_by_type(NodeType.TASK):
            if _metadata_of(node).get("graph_id") == task_graph_id:
                return _node_id_of(node)
    except Exception:
        pass
    try:
        for nid, node in getattr(mg, "_nodes", {}).items():
            if _node_type_value(node) == "task" and _metadata_of(node).get("graph_id") == task_graph_id:
                return nid
    except Exception:
        pass
    return ""


def _attach_execution_task_edges(
    mg: Any,
    session_id: str,
    round_id: str,
    execution_node_id: str,
    execution_type: str,
    payload: Dict[str, Any],
    record_edge_change: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> None:
    task_graph_id = _payload_task_graph_id(payload)
    task_node_id = _find_task_node_id(mg, session_id, task_graph_id)
    if not task_node_id:
        return
    try:
        from zulong.memory.memory_graph import EdgeType
        edge_type = EdgeType.DEPENDENCY if execution_type == "tool_call" else EdgeType.REFERENCE
        source_id = task_node_id if execution_type == "tool_call" else execution_node_id
        target_id = execution_node_id if execution_type == "tool_call" else task_node_id
        if not _edge_exists(mg, source_id, target_id):
            edge_metadata = {
                "link_type": "task_execution_event",
                "task_graph_id": task_graph_id,
                "execution_type": execution_type,
            }
            if mg.add_edge(
                source_id,
                target_id,
                edge_type,
                weight=0.9,
                protected=True,
                metadata=edge_metadata,
            ) and record_edge_change:
                record_edge_change("add_edge", _edge_delta_payload(
                    source_id,
                    target_id,
                    edge_type,
                    weight=0.9,
                    protected=True,
                    metadata=edge_metadata,
                ))
    except Exception:
        pass


def _find_matching_approval_request(mg: Any, round_id: str, pair_id: str, exclude_node_id: str) -> str:
    try:
        from zulong.memory.memory_graph import NodeType
        for node in mg.get_nodes_by_type(NodeType.APPROVAL):
            node_id = _node_id_of(node)
            if node_id == exclude_node_id:
                continue
            meta = _metadata_of(node)
            if meta.get("parent_round") != round_id:
                continue
            if str(meta.get("pair_id") or "") != str(pair_id):
                continue
            if not meta.get("decision"):
                return node_id
    except Exception:
        pass
    return ""


def _event_has_execution_signal(event: Dict[str, Any]) -> bool:
    event_type = str(event.get("event_type") or "").lower()
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    interaction = _get_payload_interaction(payload)
    return bool(
        interaction
        or "tool" in event_type
        or "approval" in event_type
        or event_type.startswith("pipeline.")
        or event_type.startswith("ide:")
    )


def backfill_recent_interactions_to_memory_graph(limit: int = 50, events_per_conversation: int = 500) -> int:
    """Rebuild recent interaction events into MemoryGraph from InteractionStore.

    This uses raw events instead of display messages so tool calls, tool
    observations, approvals, and IDE bridge events can be projected into the
    dedicated TOOL_CALL / TOOL_RESULT / APPROVAL node types after a restart.
    """
    try:
        from zulong.launcher.interaction_store import get_interaction_store

        store = get_interaction_store()
        count = 0
        for conv in reversed(store.list_conversations(limit=limit)):
            conversation_id = conv.get("conversation_id")
            if not conversation_id:
                continue
            try:
                events = store.get_events(conversation_id, limit=events_per_conversation, include_system=True)
            except Exception:
                events = store.get_messages(conversation_id, limit=events_per_conversation)
            for msg in events:
                payload = dict(msg.get("payload") or {})
                if msg.get("event_id"):
                    payload.setdefault("source_event_id", msg.get("event_id"))
                if msg.get("task_graph_id") and not payload.get("task_graph_id"):
                    payload["task_graph_id"] = msg.get("task_graph_id")
                if not _event_has_execution_signal(msg):
                    # Keep ordinary dialogue backfill small; execution nodes are
                    # the reason this compensation path exists.
                    continue
                mirror_interaction_to_memory_graph(
                    conversation_id=conversation_id,
                    turn_id=msg.get("turn_id") or msg.get("event_id"),
                    role=msg.get("role") or "system",
                    text=msg.get("text") or msg.get("content") or "",
                    event_type=msg.get("event_type") or "message",
                    source=msg.get("source") or "interaction_store",
                    payload=payload,
                )
                count += 1
        return count
    except Exception as exc:
        logger.debug("[MemoryMirror] backfill skipped: %s", exc)
        return 0
