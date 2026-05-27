"""Web-first conversation routing and task orchestration."""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from zulong.launcher.interaction_store import InteractionStore, get_interaction_store
from zulong.launcher.memory_mirror import mirror_interaction_to_memory_graph


EXPLICIT_VOICE_HINT = re.compile(
    r"(?:继续|切换到|打开|回到)\s*(?:会话|任务|聊天页)?\s*[:：]?\s*(?P<hint>[\w\-\u4e00-\u9fff ]{2,64})"
)


@dataclass
class RouteDecision:
    conversation_id: str
    turn_id: str
    text: str
    source: str = "web_chat"
    workspace_path: Optional[str] = None
    project_id: Optional[str] = None
    task_graph_id: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "source": self.source,
            "workspace_path": self.workspace_path,
            "project_id": self.project_id,
            "task_graph_id": self.task_graph_id,
        }


class ConversationOrchestrator:
    """Records web-first conversation turns without routing-class side effects."""

    transition_after_seconds = 0.7

    def __init__(self, store: Optional[InteractionStore] = None):
        self.store = store or get_interaction_store()

    def prepare_turn(self, data: Dict[str, Any], *, source: str = "web_chat") -> RouteDecision:
        text = (data.get("text") or data.get("task") or "").strip()
        conversation_id = (
            data.get("conversation_id")
            or data.get("session_id")
            or f"conv_{uuid.uuid4().hex}"
        )
        turn_id = data.get("turn_id") or data.get("request_id") or f"turn_{uuid.uuid4().hex[:12]}"
        workspace_path = data.get("workspace_path") or data.get("cwd")
        project_id = data.get("project_id")
        task_graph_id = data.get("task_graph_id") or data.get("graph_id")

        existing = self.store.get_conversation(conversation_id)
        if existing:
            workspace_path = workspace_path or existing.get("workspace_path")
            project_id = project_id or existing.get("project_id")
            task_graph_id = task_graph_id or existing.get("task_graph_id")

        title = self._make_title(text)
        self.store.upsert_conversation(
            conversation_id,
            title=title,
            source=source,
            workspace_path=workspace_path,
            project_id=project_id,
            task_graph_id=task_graph_id,
            metadata={},
            active=True,
        )
        self.store.append_event(
            conversation_id=conversation_id,
            turn_id=turn_id,
            event_type="user_message",
            role="user",
            source=source,
            text=text,
            payload=data,
            workspace_path=workspace_path,
            project_id=project_id,
            task_graph_id=task_graph_id,
        )
        mirror_interaction_to_memory_graph(
            conversation_id=conversation_id,
            turn_id=turn_id,
            role="user",
            text=text,
            event_type="user_message",
            source=source,
            payload=data,
        )

        return RouteDecision(
            conversation_id=conversation_id,
            turn_id=turn_id,
            text=text,
            source=source,
            workspace_path=workspace_path,
            project_id=project_id,
            task_graph_id=task_graph_id,
        )

    def record_assistant_text(
        self,
        decision: RouteDecision,
        text: str,
        *,
        event_type: str = "assistant_message",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.store.append_event(
            conversation_id=decision.conversation_id,
            turn_id=decision.turn_id,
            event_type=event_type,
            role="assistant",
            source="system",
            text=text,
            payload=payload or {},
            workspace_path=decision.workspace_path,
            project_id=decision.project_id,
            task_graph_id=decision.task_graph_id,
        )
        mirror_interaction_to_memory_graph(
            conversation_id=decision.conversation_id,
            turn_id=decision.turn_id,
            role="assistant",
            text=text,
            event_type=event_type,
            source="system",
            payload=payload or {},
        )

    def record_system_event(
        self,
        decision: RouteDecision,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.store.append_event(
            conversation_id=decision.conversation_id,
            turn_id=decision.turn_id,
            event_type=event_type,
            source="system",
            payload=payload or {},
            workspace_path=decision.workspace_path,
            project_id=decision.project_id,
            task_graph_id=decision.task_graph_id,
        )
        mirror_interaction_to_memory_graph(
            conversation_id=decision.conversation_id,
            turn_id=decision.turn_id,
            role="system",
            text=(payload or {}).get("message") or event_type,
            event_type=event_type,
            source="system",
            payload=payload or {},
        )

    def bind_conversation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        conversation_id = data.get("conversation_id") or data.get("session_id")
        if not conversation_id:
            return {"ok": False, "error": "conversation_id required"}
        conv = self.store.upsert_conversation(
            conversation_id,
            title=data.get("title") or "",
            source=data.get("source") or "web_chat",
            workspace_path=data.get("workspace_path") or data.get("cwd"),
            project_id=data.get("project_id"),
            task_graph_id=data.get("task_graph_id") or data.get("graph_id"),
            metadata=data.get("metadata") or {},
            active=True,
        )
        return {"ok": True, "conversation": conv}

    def route_voice_record(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = (payload.get("text") or "").strip()
        source = payload.get("source") or "voice_page"
        confidence = payload.get("confidence")
        voice_event_id = self.store.save_voice_record(
            text=text,
            source=source,
            confidence=confidence,
            payload=payload,
        )

        matched = None
        reason = "unmatched"
        explicit = EXPLICIT_VOICE_HINT.search(text)
        if explicit:
            hint = explicit.group("hint").strip()
            matched = self.store.find_conversation_by_hint(hint)
            reason = "explicit_hint" if matched else "explicit_hint_not_found"

        if not matched:
            matched = self.store.find_active_conversation(max_age_seconds=1800.0)
            reason = "active_web_conversation" if matched else reason

        if not matched:
            matched = self.store.find_recent_coding_conversation(
                workspace_path=payload.get("workspace_path") or payload.get("cwd"),
                project_id=payload.get("project_id"),
                task_graph_id=payload.get("task_graph_id") or payload.get("graph_id"),
            )
            reason = "recent_coding_context" if matched else reason

        if not matched:
            return {
                "voice_event_id": voice_event_id,
                "linked": False,
                "reason": reason,
            }

        conversation_id = matched["conversation_id"]
        self.store.link_voice_to_conversation(
            voice_event_id=voice_event_id,
            conversation_id=conversation_id,
            payload={"reason": reason},
        )
        mirror_event_id = self.store.append_event(
            conversation_id=conversation_id,
            turn_id=payload.get("turn_id") or f"voice_turn_{uuid.uuid4().hex[:12]}",
            event_type="voice_mirror",
            role="user",
            source=source,
            text=text,
            payload=payload,
            workspace_path=matched.get("workspace_path"),
            project_id=matched.get("project_id"),
            task_graph_id=matched.get("task_graph_id"),
            mirrored_from_voice_event_id=voice_event_id,
        )
        return {
            "voice_event_id": voice_event_id,
            "linked": True,
            "conversation_id": conversation_id,
            "mirror_event_id": mirror_event_id,
            "reason": reason,
            "workspace_path": matched.get("workspace_path"),
            "project_id": matched.get("project_id"),
            "task_graph_id": matched.get("task_graph_id"),
        }

    @staticmethod
    def _make_title(text: str) -> str:
        title = re.sub(r"\s+", " ", text).strip()
        return (title[:24] + "...") if len(title) > 24 else (title or "新会话")


_orchestrator: Optional[ConversationOrchestrator] = None


def get_conversation_orchestrator() -> ConversationOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ConversationOrchestrator()
    return _orchestrator
