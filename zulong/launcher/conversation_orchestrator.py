"""Web-first conversation routing and task orchestration."""

from __future__ import annotations

import os
import re
import uuid
import copy
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from zulong.launcher.interaction_store import InteractionStore, get_interaction_store
from zulong.launcher.memory_mirror import mirror_interaction_to_memory_graph


EXPLICIT_VOICE_HINT = re.compile(
    r"(?:继续|切换到|打开|回到)\s*(?:会话|任务|聊天页)?\s*[:：]?\s*(?P<hint>[\w\-\u4e00-\u9fff ]{2,64})"
)

_SIMPLE_SOCIAL_TEXTS = {
    "你好", "您好", "hi", "hello", "hey",
    "早上好", "下午好", "晚上好",
    "谢谢", "感谢", "多谢", "thanks", "thank you",
    "你好呀", "嗨", "哈喽",
}

_FOLLOWUP_TASK_CUES = (
    "继续", "刚才", "上次", "上一个", "那个", "这个",
    "原有", "原来", "原任务", "任务图", "图谱",
    "当前任务图", "检查", "质量", "覆盖", "节点", "任务进度",
    "任务地址", "焦点", "压力", "完成门",
    "修改", "改一下", "调整", "补充", "增加", "添加",
    "删除", "保留", "修复", "完善", "更新",
    "index.html", ".py", ".ts", ".tsx", ".js", ".jsx",
    ".css", ".json", ".md",
)

_NEW_TASK_CUES = (
    "新建", "创建", "写一个", "做一个", "开发一个",
    "生成一个", "搭建", "实现一个", "从头",
)

_TASK_GRAPH_REFERENCE_CUES = (
    "分析", "借鉴", "参考", "参照", "看看", "看一下", "对比",
    "学习", "复盘", "总结", "只读", "仅分析", "不要修改", "不要改",
    "不要写", "复制", "仿照", "类似", "作为参考",
    "analyze", "reference", "compare", "inspect only", "read only",
)

_TASK_GRAPH_VERSION_CUES = (
    "重新做一版", "另起版本", "新版本", "再做一版", "重做一版",
    "换一版", "fork", "分支", "variant", "new version",
)

_TASK_GRAPH_EDIT_CUES = (
    "继续", "修改", "改一下", "调整", "补充", "增加", "添加",
    "删除", "移除", "修复", "完善", "更新", "继续编辑",
)

CODING_KEYWORDS = {
    "代码", "编程", "程序", "函数", "类", "模块", "接口", "脚本",
    "项目", "文件", "修复", "重构", "实现", "运行", "构建", "测试",
    "bug", "fix", "refactor", "implement", "code", "python",
    "typescript", "javascript", "vscode", "npm", "pytest", "tsc",
}

_CODING_FILE_RE = re.compile(
    r"\.(?:py|ts|tsx|js|jsx|css|html|json|md|yaml|yml|toml|go|rs|java|cpp|c|h)\b",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)
_MIRROR_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="zulong-memory-mirror")


def _compact_dialogue_id(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value or ""))


def _infer_session_node_id(conversation_id: str, explicit_node_id: Any = "") -> str:
    node_id = str(explicit_node_id or "").strip()
    if node_id.startswith("dialogue:session_") and "/" not in node_id:
        return node_id
    conversation_id = str(conversation_id or "").strip()
    if not conversation_id:
        return ""
    if conversation_id.startswith("dialogue:session_") and "/" not in conversation_id:
        return conversation_id
    return f"dialogue:session_{_compact_dialogue_id(conversation_id)}"


def _copy_payload_for_background(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        return copy.deepcopy(payload or {})
    except Exception:
        return dict(payload or {})


def _mirror_interaction_background(**kwargs: Any) -> None:
    """Mirror interactions without blocking the Web → L1-B hot path."""
    kwargs["payload"] = _copy_payload_for_background(kwargs.get("payload"))

    def _run() -> None:
        try:
            mirror_interaction_to_memory_graph(**kwargs)
        except Exception as exc:
            logger.debug("[ConversationOrchestrator] MemoryGraph mirror skipped: %s", exc)

    try:
        _MIRROR_EXECUTOR.submit(_run)
    except Exception as exc:
        logger.debug("[ConversationOrchestrator] MemoryGraph mirror submit failed: %s", exc)


def _is_simple_social_turn(text: str) -> bool:
    normalized = (text or "").lower().strip(" \t\r\n。！？!?.,，~～")
    return len(text or "") <= 30 and normalized in _SIMPLE_SOCIAL_TEXTS


def _should_bind_existing_task(text: str, data: Dict[str, Any]) -> bool:
    """Whether this turn should be treated as attached to the prior task graph."""
    if _is_simple_social_turn(text):
        return False
    lowered = (text or "").lower()
    explicit_graph = bool(data.get("task_graph_id") or data.get("graph_id"))
    if explicit_graph:
        return _task_graph_reference_mode(text, data.get("task_graph_id") or data.get("graph_id")) == "edit"
    has_followup_anchor = any(cue in lowered for cue in _FOLLOWUP_TASK_CUES)
    if not has_followup_anchor:
        return False
    if any(cue in lowered for cue in _NEW_TASK_CUES) and not any(
        cue in lowered for cue in ("继续", "刚才", "原有", "原来", "原任务", "任务图", "图谱")
    ):
        return False
    return True


def _task_graph_reference_mode(text: str, task_graph_id: Any) -> str:
    """Classify an explicit task graph id as edit target or reference hint."""
    if not task_graph_id:
        return "none"
    lowered = (text or "").lower()
    if any(cue in lowered for cue in _TASK_GRAPH_VERSION_CUES):
        return "ambiguous_version"
    if any(cue in lowered for cue in _TASK_GRAPH_REFERENCE_CUES) and not any(
        cue in lowered for cue in _TASK_GRAPH_EDIT_CUES
    ):
        return "reference"
    return "edit"


@dataclass
class RouteDecision:
    conversation_id: str
    turn_id: str
    text: str
    route: str = "general_chat"
    confidence: float = 1.0
    reason: str = ""
    source: str = "web_chat"
    session_node_id: str = ""
    workspace_path: Optional[str] = None
    project_id: Optional[str] = None
    task_graph_id: Optional[str] = None
    referenced_task_graph_id: Optional[str] = None
    task_graph_reference_mode: str = "none"
    last_task_graph_id: Optional[str] = None
    task_graph_binding_policy: str = "reference_only"
    binding_reason: str = ""

    def to_payload(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "session_id": self.conversation_id,
            "session_node_id": self.session_node_id,
            "dialogue_session_id": self.session_node_id,
            "turn_id": self.turn_id,
            "route": self.route,
            "confidence": self.confidence,
            "reason": self.reason,
            "source": self.source,
            "workspace_path": self.workspace_path,
            "project_id": self.project_id,
            "task_graph_id": self.task_graph_id,
            "referenced_task_graph_id": self.referenced_task_graph_id,
            "task_graph_reference_mode": self.task_graph_reference_mode,
            "last_task_graph_id": self.last_task_graph_id,
            "task_graph_binding_policy": self.task_graph_binding_policy,
            "binding_reason": self.binding_reason,
        }


class ConversationOrchestrator:
    """Records web-first conversation turns without routing-class side effects."""

    transition_after_seconds = 0.7

    def __init__(self, store: Optional[InteractionStore] = None):
        self.store = store or get_interaction_store()

    @staticmethod
    def _has_local_workspace_signal(workspace_path: Optional[str]) -> bool:
        return bool(workspace_path)

    @staticmethod
    def _looks_like_coding(text: str) -> bool:
        lowered = (text or "").lower()
        return any(keyword in lowered for keyword in CODING_KEYWORDS) or bool(
            _CODING_FILE_RE.search(text or "")
        )

    @staticmethod
    def _llm_confirm_coding_intent(text: str) -> bool:
        """Confirm ambiguous coding-like text, falling back to regex trust."""
        try:
            from zulong.ide.ide_server import _get_engine

            engine = _get_engine()
            client = getattr(engine, "vllm_client", None) if engine else None
            if client is None:
                return True
            response = client.chat.completions.create(
                model=getattr(engine, "model_name", None) or "default",
                messages=[
                    {
                        "role": "system",
                        "content": "判断用户是否在请求代码、项目、文件、构建或测试相关工作。只回答 YES 或 NO。",
                    },
                    {"role": "user", "content": text or ""},
                ],
                temperature=0,
                max_tokens=2,
            )
            answer = str(response.choices[0].message.content or "").strip().upper()
            return answer.startswith("YES")
        except Exception:
            return True

    def _classify(
        self,
        text: str,
        *,
        workspace_path: Optional[str] = None,
        task_graph_id: Optional[str] = None,
        referenced_nodes: Optional[list] = None,
    ) -> tuple[str, float, str]:
        if task_graph_id and _should_bind_existing_task(text, {"task_graph_id": task_graph_id}):
            return "resume_task", 0.9, "resume_task_graph"
        if referenced_nodes:
            return "coding_task", 0.8, "referenced_nodes"
        if self._looks_like_coding(text):
            if not self._has_local_workspace_signal(workspace_path):
                return "general_chat", 0.75, "no_workspace_signal"
            if self._llm_confirm_coding_intent(text):
                return "coding_task", 0.8, "coding_keyword_confirmed"
            return "general_chat", 0.75, "llm_rejected_coding_intent"
        return "general_chat", 0.75, "default_general_chat"

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
        session_node_id = _infer_session_node_id(
            conversation_id,
            data.get("session_node_id") or data.get("dialogue_session_id"),
        )
        if session_node_id:
            data["session_node_id"] = session_node_id
            data["dialogue_session_id"] = session_node_id
        referenced_task_graph_id = None
        task_graph_reference_mode = "none"
        existing = self.store.get_conversation(conversation_id)
        last_task_graph_id = (existing or {}).get("task_graph_id")
        if task_graph_id:
            try:
                from zulong.tools.task_tools import normalize_task_graph_id

                task_graph_id = normalize_task_graph_id(task_graph_id)
                data["task_graph_id"] = task_graph_id
            except Exception:
                task_graph_id = str(task_graph_id or "").strip()
            task_graph_reference_mode = _task_graph_reference_mode(text, task_graph_id)
            if task_graph_reference_mode in {"reference", "ambiguous_version"}:
                referenced_task_graph_id = task_graph_id
                data["referenced_task_graph_id"] = task_graph_id
                data["task_graph_reference_mode"] = task_graph_reference_mode
                task_graph_id = None
            else:
                data["task_graph_id"] = task_graph_id
                data["graph_id"] = task_graph_id
                data["task_graph_reference_mode"] = task_graph_reference_mode

        bind_existing_task = _should_bind_existing_task(text, data)
        task_graph_binding_policy = "reference_only"
        binding_reason = "no active task binding requested"
        if _is_simple_social_turn(text):
            workspace_path = None
            project_id = None
            task_graph_id = None
            task_graph_binding_policy = "reference_only"
            binding_reason = "simple social turn"
        elif existing and bind_existing_task:
            workspace_path = workspace_path or existing.get("workspace_path")
            project_id = project_id or existing.get("project_id")
            task_graph_id = task_graph_id or existing.get("task_graph_id")
            task_graph_binding_policy = "keep_recent_task_graph"
            binding_reason = "follow-up/check turn keeps recent task graph"
        elif referenced_task_graph_id:
            task_graph_binding_policy = "reference_only"
            binding_reason = "explicit task graph used as reference"
        elif any(cue in (text or "").lower() for cue in _NEW_TASK_CUES):
            task_graph_binding_policy = "clear_for_new_unbound_turn"
            binding_reason = "new task cue without existing binding"

        title = self._make_title(text)
        self.store.upsert_conversation(
            conversation_id,
            title=title,
            source=source,
            workspace_path=workspace_path,
            project_id=project_id,
            task_graph_id=task_graph_id,
            session_node_id=session_node_id or None,
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
        _mirror_interaction_background(
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
            session_node_id=session_node_id,
            workspace_path=workspace_path,
            project_id=project_id,
            task_graph_id=task_graph_id,
            referenced_task_graph_id=referenced_task_graph_id,
            task_graph_reference_mode=task_graph_reference_mode,
            last_task_graph_id=last_task_graph_id,
            task_graph_binding_policy=task_graph_binding_policy,
            binding_reason=binding_reason,
        )

    def record_assistant_text(
        self,
        decision: RouteDecision,
        text: str,
        *,
        event_type: str = "assistant_message",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = dict(payload or {})
        if decision.session_node_id:
            payload.setdefault("session_node_id", decision.session_node_id)
            payload.setdefault("dialogue_session_id", decision.session_node_id)
        self.store.append_event(
            conversation_id=decision.conversation_id,
            turn_id=decision.turn_id,
            event_type=event_type,
            role="assistant",
            source="system",
            text=text,
            payload=payload,
            workspace_path=decision.workspace_path,
            project_id=decision.project_id,
            task_graph_id=decision.task_graph_id,
        )
        _mirror_interaction_background(
            conversation_id=decision.conversation_id,
            turn_id=decision.turn_id,
            role="assistant",
            text=text,
            event_type=event_type,
            source="system",
            payload=payload,
        )

    def record_system_event(
        self,
        decision: RouteDecision,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = dict(payload or {})
        if decision.session_node_id:
            payload.setdefault("session_node_id", decision.session_node_id)
            payload.setdefault("dialogue_session_id", decision.session_node_id)
        self.store.append_event(
            conversation_id=decision.conversation_id,
            turn_id=decision.turn_id,
            event_type=event_type,
            source="system",
            payload=payload,
            workspace_path=decision.workspace_path,
            project_id=decision.project_id,
            task_graph_id=decision.task_graph_id,
        )
        _mirror_interaction_background(
            conversation_id=decision.conversation_id,
            turn_id=decision.turn_id,
            role="system",
            text=payload.get("message") or event_type,
            event_type=event_type,
            source="system",
            payload=payload,
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
