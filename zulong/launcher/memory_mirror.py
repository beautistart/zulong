"""Mirror Web conversation events into MemoryGraph.

MemoryGraph is the authoritative long-term store for conversations, tasks,
tools, approvals, and other interaction events. InteractionStore is only a
small SQLite cache/ledger for fast Web session restoration and audit queries.
This module makes sure every user-facing interaction is represented in the
graph even when the L1-B/L2 route changes.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_LAST_ROUND_BY_CONVERSATION: Dict[str, str] = {}


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
        session_id = f"dialogue:session_{safe_conversation_id}"
        round_id = f"{session_id}/round_{safe_turn_id}"
        message_id = f"{round_id}/{safe_role}_{safe_event_type}"

        if not mg.has_node(session_id):
            mg.add_node(GraphNode(
                node_id=session_id,
                node_type=NodeType.DIALOGUE,
                label=f"Web 会话 {conversation_id[-8:]}",
                backend_ref=f"interaction:{conversation_id}",
                metadata={
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
                },
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

        if not mg.has_node(round_id):
            prev_round_id = _LAST_ROUND_BY_CONVERSATION.get(conversation_id)
            if not prev_round_id:
                prev_round_id = _find_latest_round_for_conversation(mg, conversation_id)
            mg.add_node(GraphNode(
                node_id=round_id,
                node_type=NodeType.DIALOGUE,
                label=text[:80],
                backend_ref=f"interaction:{conversation_id}/{turn_id}",
                metadata={
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
                },
            ))
            if not _edge_exists(mg, session_id, round_id):
                mg.add_edge(session_id, round_id, EdgeType.HIERARCHY, weight=1.0, protected=True)
            if prev_round_id and mg.has_node(prev_round_id) and not _edge_exists(mg, prev_round_id, round_id):
                mg.add_edge(prev_round_id, round_id, EdgeType.TEMPORAL, weight=1.0, protected=True)
            sess_node = mg.get_node(session_id)
            if sess_node:
                sess_node.metadata["round_count"] = _count_round_children(mg, session_id)
                sess_node.metadata["last_round_id"] = round_id
                sess_node.metadata["last_active_at"] = now
                _persist_node_update(mg, sess_node)
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

        _LAST_ROUND_BY_CONVERSATION[conversation_id] = round_id

        mg.add_node(GraphNode(
            node_id=message_id,
            node_type=NodeType.DIALOGUE,
            label=(role or "message")[:20],
            backend_ref=f"interaction:{conversation_id}/{turn_id}/{event_type}/{role}",
            metadata={
                "sub_type": "agent_turn" if role != "user" else "user_turn",
                "conversation_id": conversation_id,
                "request_id": turn_id,
                "event_type": event_type,
                "role": role,
                "source": source,
                "content": text,
                "payload": payload or {},
                "created_at": now,
                "parent_round": round_id,
                "parent_session": session_id,
                "full_path": message_id,
                "is_root_node": False,
                "graph_level": 2,
                "node_role": "message",
            },
        ))
        if not _edge_exists(mg, round_id, message_id):
            mg.add_edge(round_id, message_id, EdgeType.HIERARCHY, weight=1.0, protected=True)
        if role == "assistant":
            node = mg.get_node(round_id)
            if node:
                node.metadata["bot_text"] = text
                node.metadata["status"] = "completed"
                node.metadata["completed_at"] = now
                _persist_node_update(mg, node)
            try:
                mg.index_summary(round_id, f"{node.metadata.get('goal', '') if node else ''} {text}"[:500])
            except Exception:
                pass
        _attach_task_if_present(mg, session_id, round_id, payload or {})
        if event_type in {"approval_required", "checkpoint_created", "ide:file_changed"}:
            try:
                mg.set_importance(message_id, Importance.IMPORTANT)
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
        except Exception:
            pass
        try:
            if not mg.update_focus_to_node(message_id):
                mg.set_active_nodes([session_id, round_id, message_id])
        except Exception:
            try:
                mg.set_active_nodes([session_id, round_id, message_id])
            except Exception:
                pass
        try:
            from zulong.core.event_bus import event_bus
            from zulong.core.types import EventPriority, EventType, ZulongEvent

            event_bus.publish(ZulongEvent(
                type=EventType.MEMORY_GRAPH_UPDATED,
                priority=EventPriority.LOW,
                source="MemoryMirror",
                payload=mg.to_frontend_dict(depth=0),
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


def _attach_task_if_present(mg: Any, session_id: str, round_id: str, payload: Dict[str, Any]) -> None:
    try:
        from zulong.memory.memory_graph import EdgeType, NodeType
    except Exception:
        return

    task_graph_id = (
        payload.get("task_graph_id")
        or payload.get("graph_id")
        or payload.get("task_id")
        or ""
    )
    if not task_graph_id:
        return
    task_candidates = [
        f"task:{task_graph_id}",
        f"{session_id}/task:{task_graph_id}",
    ]
    task_node_id = ""
    for candidate in task_candidates:
        if mg.has_node(candidate):
            task_node_id = candidate
            break
    if not task_node_id:
        try:
            for nid, node in getattr(mg, "_nodes", {}).items():
                if node.node_type == NodeType.TASK and node.metadata.get("graph_id") == task_graph_id:
                    task_node_id = nid
                    break
        except Exception:
            pass
    if not task_node_id:
        return
    if not _edge_exists(mg, round_id, task_node_id):
        mg.add_edge(
            round_id,
            task_node_id,
            EdgeType.REFERENCE,
            weight=0.9,
            protected=True,
            metadata={"link_type": "dialogue_round_task"},
        )
    task_node = mg.get_node(task_node_id)
    if task_node:
        task_node.metadata.setdefault("parent_session", session_id)
        task_node.metadata.setdefault("parent_round", round_id)
        task_node.metadata.setdefault("full_path", f"{session_id}/{task_node.node_id}")
        _persist_node_update(mg, task_node)


def backfill_recent_interactions_to_memory_graph(limit: int = 50) -> int:
    """Rebuild recent Web conversations into MemoryGraph from InteractionStore."""
    try:
        from zulong.launcher.interaction_store import get_interaction_store

        store = get_interaction_store()
        count = 0
        for conv in reversed(store.list_conversations(limit=limit)):
            conversation_id = conv.get("conversation_id")
            if not conversation_id:
                continue
            for msg in store.get_messages(conversation_id, limit=200):
                mirror_interaction_to_memory_graph(
                    conversation_id=conversation_id,
                    turn_id=msg.get("turn_id") or msg.get("event_id"),
                    role=msg.get("role") or "system",
                    text=msg.get("text") or msg.get("content") or "",
                    event_type=msg.get("event_type") or "message",
                    source=msg.get("source") or "interaction_store",
                )
                count += 1
        return count
    except Exception as exc:
        logger.debug("[MemoryMirror] backfill skipped: %s", exc)
        return 0
