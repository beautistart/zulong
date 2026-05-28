# File: zulong/review/task_execution_extractor.py
"""Task execution trace extraction.

P1 of the task-memory-experience loop: convert the raw InteractionStore event
ledger into a compact TaskExecutionTrace and attach it back to MemoryGraph.
This module does not write experiences. P2 consumes the trace and decides what
is worth long-term experience storage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


TERMINAL_EVENT_TYPES = {
    "pipeline.agent_done",
    "pipeline.agent_error",
    "agent_done",
    "agent_error",
    "FC_DONE",
    "TASK_COMPLETE",
    "TASK_ERROR",
    "completed",
    "error",
    "blocked",
}


def _compact_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value or ""))


def _payload_interaction(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    interaction = payload.get("interaction")
    if isinstance(interaction, dict):
        return interaction
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("interaction"), dict):
        return data["interaction"]
    return {}


def _event_text(event: Dict[str, Any]) -> str:
    payload = event.get("payload") or {}
    text = event.get("text") or event.get("content") or ""
    if text:
        return str(text)
    interaction = _payload_interaction(payload)
    if interaction:
        return str(interaction.get("detail") or interaction.get("title") or "")
    for key in ("message", "summary", "result", "error"):
        if payload.get(key):
            return str(payload[key])
    return ""


def _event_time(event: Dict[str, Any]) -> float:
    try:
        return float(event.get("created_at") or 0.0)
    except Exception:
        return 0.0


def _event_id(event: Dict[str, Any]) -> str:
    return str(event.get("event_id") or event.get("id") or event.get("node_id") or "")


def _event_type(event: Dict[str, Any]) -> str:
    return str(event.get("event_type") or "")


def _collect_files(payload: Dict[str, Any], interaction: Dict[str, Any]) -> List[str]:
    files: List[str] = []
    for key in ("path", "file_path", "workspace_path"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if value:
            files.append(str(value))
    for key in ("files", "affected_files"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, (list, tuple, set)):
            files.extend(str(v) for v in value if v)
        elif isinstance(value, str):
            files.append(value)
    value = interaction.get("affected_files") if isinstance(interaction, dict) else None
    if isinstance(value, (list, tuple, set)):
        files.extend(str(v) for v in value if v)
    elif isinstance(value, str):
        files.append(value)
    for result in payload.get("results") or []:
        if not isinstance(result, dict):
            continue
        for key in ("path", "file_path"):
            if result.get(key):
                files.append(str(result[key]))
    data = payload.get("data") if isinstance(payload, dict) else None
    graph = data.get("graph") if isinstance(data, dict) and isinstance(data.get("graph"), dict) else None
    if graph:
        for node in graph.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            for file_info in node.get("files") or []:
                if isinstance(file_info, dict) and file_info.get("path"):
                    files.append(str(file_info["path"]))
    seen = set()
    deduped = []
    for path in files:
        path = path.strip()
        if path and path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def _is_terminal_event(event_type: str) -> bool:
    if event_type in TERMINAL_EVENT_TYPES:
        return True
    lowered = event_type.lower()
    return lowered.endswith("agent_done") or lowered.endswith("agent_error")


def _is_tool_action(event_type: str, interaction: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    kind = str(interaction.get("kind") or "").lower()
    if kind == "action":
        return True
    event = event_type.lower()
    return event in {"tool_call", "ide_tool_request", "tool_requested"} or event.endswith("tool_requested")


def _is_tool_observation(event_type: str, interaction: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    kind = str(interaction.get("kind") or "").lower()
    if kind == "observation" and (
        interaction.get("tool_name") or payload.get("tool_name") or payload.get("results") or "tool" in event_type.lower()
    ):
        return True
    event = event_type.lower()
    return event in {"ide_tool_result", "ide_tool_exec", "tool_finished"} or event.endswith("tool_finished")


def _is_approval(event_type: str, interaction: Dict[str, Any]) -> bool:
    return str(interaction.get("kind") or "").lower() == "approval" or "approval" in event_type.lower()


@dataclass
class TaskExecutionTraceExtractor:
    """Extract and persist TaskExecutionTrace from InteractionStore events."""

    max_events: int = 1200

    def build_trace(
        self,
        *,
        conversation_id: str,
        turn_id: Optional[str] = None,
        task_graph_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        from zulong.launcher.interaction_store import get_interaction_store

        store = get_interaction_store()
        events = store.get_events(
            conversation_id,
            turn_id=turn_id,
            limit=self.max_events,
            include_system=True,
        )
        if turn_id and events:
            try:
                all_events = store.get_events(
                    conversation_id,
                    limit=self.max_events,
                    include_system=True,
                )
                times = [_event_time(event) for event in events if _event_time(event)]
                if times:
                    start_at = min(times) - 3.0
                    end_at = max(times) + 3.0
                    by_id = {
                        _event_id(event) or f"turn:{idx}": event
                        for idx, event in enumerate(events)
                    }
                    for event in all_events:
                        event_time = _event_time(event)
                        if event_time < start_at or event_time > end_at:
                            continue
                        event_payload = event.get("payload") or {}
                        event_interaction = _payload_interaction(event_payload)
                        event_type = _event_type(event)
                        should_merge = (
                            _is_approval(event_type, event_interaction)
                            or _is_tool_action(event_type, event_interaction, event_payload)
                            or _is_tool_observation(event_type, event_interaction, event_payload)
                            or event_type.startswith("ide:")
                            or event_type in {"diff_ready", "checkpoint_created"}
                        )
                        if not should_merge:
                            continue
                        event_id = _event_id(event) or f"time:{event_time}:{event_type}"
                        by_id.setdefault(event_id, event)
                    events = list(by_id.values())
            except Exception as exc:
                logger.debug("[TaskExecutionTrace] 同轮时间窗事件合并跳过: %s", exc)
        if task_graph_id:
            try:
                graph_events = store.get_events(
                    conversation_id,
                    limit=self.max_events,
                    include_system=True,
                )
                by_id = {_event_id(event): event for event in events if _event_id(event)}
                for event in graph_events:
                    event_payload = event.get("payload") or {}
                    event_data = event_payload.get("data") if isinstance(event_payload.get("data"), dict) else {}
                    event_task_graph_id = (
                        event.get("task_graph_id")
                        or event_payload.get("task_graph_id")
                        or event_payload.get("graph_id")
                        or event_data.get("task_graph_id")
                        or event_data.get("graph_id")
                    )
                    if str(event_task_graph_id or "") != str(task_graph_id):
                        continue
                    event_id = _event_id(event) or f"row:{len(by_id)}"
                    by_id.setdefault(event_id, event)
                if by_id:
                    events = list(by_id.values())
            except Exception as exc:
                logger.debug("[TaskExecutionTrace] task_graph 事件合并跳过: %s", exc)
        if not events:
            return self._empty_trace(conversation_id, turn_id, task_graph_id)
        return self._build_from_events(
            conversation_id=conversation_id,
            turn_id=turn_id,
            task_graph_id=task_graph_id,
            events=events,
        )

    def build_trace_from_events(
        self,
        *,
        conversation_id: str,
        turn_id: Optional[str],
        task_graph_id: Optional[str],
        events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return self._build_from_events(
            conversation_id=conversation_id,
            turn_id=turn_id,
            task_graph_id=task_graph_id,
            events=events,
        )

    def _empty_trace(
        self,
        conversation_id: str,
        turn_id: Optional[str],
        task_graph_id: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "trace_id": self._trace_id(conversation_id, turn_id, task_graph_id),
            "conversation_id": conversation_id,
            "turn_id": turn_id or "",
            "task_graph_id": task_graph_id or "",
            "goal": "",
            "tool_chain": [],
            "approval_trace": [],
            "files": [],
            "result": "",
            "success": False,
            "failure_reason": "no_events",
            "retry_count": 0,
            "verification": [],
            "source_event_ids": [],
            "summary": "未找到可用于生成 TaskExecutionTrace 的事件。",
            "created_at": time.time(),
        }

    def _build_from_events(
        self,
        *,
        conversation_id: str,
        turn_id: Optional[str],
        task_graph_id: Optional[str],
        events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        events = sorted(events, key=_event_time)
        source_event_ids = [_event_id(e) for e in events if _event_id(e)]
        resolved_turn_id = turn_id or self._infer_turn_id(events)
        resolved_task_graph_id = task_graph_id or self._infer_task_graph_id(events)
        goal = self._infer_goal(events)

        tool_map: Dict[str, Dict[str, Any]] = {}
        tool_order: List[str] = []
        approvals: List[Dict[str, Any]] = []
        files: List[str] = []
        terminal_event: Optional[Dict[str, Any]] = None
        verification: List[Dict[str, Any]] = []

        for event in events:
            payload = event.get("payload") or {}
            interaction = _payload_interaction(payload)
            event_type = _event_type(event)
            event_id = _event_id(event)
            text = _event_text(event)
            files.extend(_collect_files(payload, interaction))

            if _is_tool_action(event_type, interaction, payload):
                pair_id = self._pair_id(event_type, interaction, payload, event_id)
                entry = tool_map.setdefault(pair_id, self._new_tool_entry(pair_id, event))
                if pair_id not in tool_order:
                    tool_order.append(pair_id)
                entry["tool_name"] = entry["tool_name"] or self._tool_name(interaction, payload)
                entry["action_event_id"] = event_id
                entry["action_summary"] = text
                entry["status"] = interaction.get("status") or "running"
                entry["started_at"] = event.get("created_at")
                entry["arguments_summary"] = (
                    interaction.get("tool_args")
                    or payload.get("tool_args")
                    or payload.get("args")
                )

            if _is_tool_observation(event_type, interaction, payload):
                result_items = payload.get("results") if isinstance(payload.get("results"), list) else None
                if result_items:
                    for item in result_items:
                        if not isinstance(item, dict):
                            continue
                        pair_id = str(item.get("pair_id") or item.get("call_id") or self._pair_id(event_type, interaction, payload, event_id))
                        entry = tool_map.setdefault(pair_id, self._new_tool_entry(pair_id, event))
                        if pair_id not in tool_order:
                            tool_order.append(pair_id)
                        entry["tool_name"] = entry["tool_name"] or str(item.get("tool_name") or "")
                        self._append_tool_result(entry, event, item)
                else:
                    pair_id = self._pair_id(event_type, interaction, payload, event_id)
                    entry = tool_map.setdefault(pair_id, self._new_tool_entry(pair_id, event))
                    if pair_id not in tool_order:
                        tool_order.append(pair_id)
                    entry["tool_name"] = entry["tool_name"] or self._tool_name(interaction, payload)
                    self._append_tool_result(entry, event, {})

            if _is_approval(event_type, interaction):
                approvals.append(self._approval_record(event, interaction, payload))

            if self._is_verification_event(event_type, interaction, payload, text):
                verification.append({
                    "event_id": event_id,
                    "tool_name": self._tool_name(interaction, payload),
                    "summary": text[:300],
                    "created_at": event.get("created_at"),
                })

            if _is_terminal_event(event_type):
                terminal_event = event

        tool_chain = [tool_map[pair_id] for pair_id in tool_order]
        files = self._dedupe(files)
        retry_count = self._retry_count(tool_chain)
        task_completion = self._task_completion_state(events)
        success = self._success_from_terminal(
            terminal_event,
            tool_chain,
            task_completion=task_completion,
        )
        result = self._terminal_result(terminal_event, events)
        failure_reason = "" if success else self._failure_reason(
            terminal_event,
            tool_chain,
            task_completion=task_completion,
        )

        trace = {
            "trace_id": self._trace_id(conversation_id, resolved_turn_id, resolved_task_graph_id),
            "conversation_id": conversation_id,
            "turn_id": resolved_turn_id or "",
            "task_graph_id": resolved_task_graph_id or "",
            "goal": goal,
            "tool_chain": tool_chain,
            "approval_trace": approvals,
            "files": files,
            "result": result,
            "success": success,
            "failure_reason": failure_reason,
            "retry_count": retry_count,
            "verification": verification,
            "task_completion": task_completion,
            "source_event_ids": source_event_ids,
            "summary": self._summary(goal, tool_chain, approvals, result, success, failure_reason),
            "created_at": time.time(),
        }
        return trace

    def persist_trace_to_memory_graph(self, trace: Dict[str, Any]) -> bool:
        """Attach trace to MemoryGraph round and create/update a summary node."""
        if not trace or not trace.get("conversation_id"):
            return False
        try:
            from zulong.memory.memory_graph import EdgeType, GraphNode, NodeType, get_memory_graph

            mg = get_memory_graph()
            if mg is None:
                return False
            conversation_id = trace.get("conversation_id", "")
            turn_id = trace.get("turn_id", "")
            session_id = f"dialogue:session_{_compact_id(conversation_id)}"
            round_id = f"{session_id}/round_{_compact_id(turn_id)}" if turn_id else self._latest_round_id(mg, conversation_id)
            if not round_id:
                return False

            if mg.has_node(round_id):
                round_node = mg.get_node(round_id)
                if round_node:
                    meta = getattr(round_node, "metadata", {}) or {}
                    meta["task_execution_trace"] = trace
                    meta["task_execution_trace_id"] = trace.get("trace_id", "")
                    meta["task_execution_trace_updated_at"] = time.time()
                    round_node.metadata = meta
                    self._update_node(mg, round_node)

            summary_id = f"{round_id}/episode_task_execution_trace"
            mg.add_node(GraphNode(
                node_id=summary_id,
                node_type=NodeType.EPISODE,
                label="任务执行 Trace",
                backend_ref=f"task_execution_trace:{trace.get('trace_id', '')}",
                metadata={
                    "sub_type": "task_execution_trace",
                    "content": trace.get("summary", ""),
                    "summary": trace.get("summary", ""),
                    "trace": trace,
                    "trace_id": trace.get("trace_id", ""),
                    "conversation_id": conversation_id,
                    "request_id": turn_id,
                    "task_graph_id": trace.get("task_graph_id", ""),
                    "parent_round": round_id,
                    "parent_session": session_id,
                    "full_path": summary_id,
                    "node_role": "task_execution_summary",
                    "graph_level": 2,
                    "created_from": "task_execution_extractor",
                },
            ))
            if mg.has_node(round_id) and not self._edge_exists(mg, round_id, summary_id):
                mg.add_edge(
                    round_id,
                    summary_id,
                    EdgeType.HIERARCHY,
                    weight=1.0,
                    protected=True,
                    metadata={"link_type": "round_task_execution_trace"},
                )
            task_node_id = self._find_task_node_id(mg, session_id, trace.get("task_graph_id", ""))
            if task_node_id and not self._edge_exists(mg, summary_id, task_node_id):
                mg.add_edge(
                    summary_id,
                    task_node_id,
                    EdgeType.REFERENCE,
                    weight=0.9,
                    protected=True,
                    metadata={"link_type": "trace_task_graph"},
                )
            return True
        except Exception as exc:
            logger.debug("[TaskExecutionTrace] 写入 MemoryGraph 失败: %s", exc)
            return False

    def finalize_for_event(
        self,
        *,
        conversation_id: Optional[str],
        turn_id: Optional[str] = None,
        task_graph_id: Optional[str] = None,
        event_type: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Build and persist trace when a terminal event arrives."""
        if not conversation_id or not _is_terminal_event(event_type):
            return None
        trace = self.build_trace(
            conversation_id=conversation_id,
            turn_id=turn_id,
            task_graph_id=task_graph_id,
        )
        self.persist_trace_to_memory_graph(trace)
        try:
            from zulong.review.task_experience_generator import maybe_generate_task_experiences

            saved = maybe_generate_task_experiences(trace)
            if saved:
                trace["experience_ids"] = [item.get("experience_id", "") for item in saved if item.get("experience_id")]
                trace["experience_node_ids"] = [
                    item.get("experience_node_id", "")
                    for item in saved
                    if item.get("experience_node_id")
                ]
                self.persist_trace_to_memory_graph(trace)
        except Exception as exc:
            logger.debug("[TaskExecutionTrace] 经验化跳过: %s", exc)
        return trace

    @staticmethod
    def _new_tool_entry(pair_id: str, event: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pair_id": pair_id,
            "tool_name": "",
            "status": "running",
            "action_event_id": "",
            "action_summary": "",
            "arguments_summary": None,
            "result_event_ids": [],
            "result_preview": "",
            "success": None,
            "started_at": event.get("created_at"),
            "completed_at": None,
        }

    @staticmethod
    def _append_tool_result(entry: Dict[str, Any], event: Dict[str, Any], item: Dict[str, Any]) -> None:
        payload = event.get("payload") or {}
        interaction = _payload_interaction(payload)
        event_id = _event_id(event)
        if event_id and event_id not in entry["result_event_ids"]:
            entry["result_event_ids"].append(event_id)
        preview = (
            item.get("result_preview")
            or item.get("result")
            or interaction.get("result_preview")
            or payload.get("result_preview")
            or payload.get("tool_result")
            or _event_text(event)
            or ""
        )
        is_error = bool(item.get("is_error") or payload.get("is_error"))
        status = interaction.get("status") or payload.get("status") or ""
        entry["status"] = "failed" if is_error or status == "failed" else "succeeded"
        entry["success"] = entry["status"] == "succeeded"
        entry["result_preview"] = str(preview)[:1000]
        entry["completed_at"] = event.get("created_at")

    @staticmethod
    def _approval_record(event: Dict[str, Any], interaction: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        decision = ""
        if "approved" in payload:
            decision = "approved" if payload.get("approved") else "rejected"
        elif interaction.get("status") in {"approved", "rejected", "denied"}:
            decision = str(interaction.get("status"))
        return {
            "event_id": _event_id(event),
            "approval_id": str(payload.get("approval_id") or payload.get("approvalId") or interaction.get("approval_id") or interaction.get("pair_id") or ""),
            "pair_id": str(interaction.get("pair_id") or payload.get("pair_id") or ""),
            "tool_name": str(interaction.get("tool_name") or payload.get("tool_name") or ""),
            "phase": str(interaction.get("phase") or payload.get("phase") or _event_type(event)),
            "status": str(interaction.get("status") or payload.get("status") or ""),
            "decision": decision,
            "risk_level": str(interaction.get("risk_level") or payload.get("risk_level") or ""),
            "approval_mode": str(interaction.get("approval_mode") or payload.get("approval_mode") or ""),
            "action_summary": str(payload.get("action_summary") or payload.get("summary") or interaction.get("title") or _event_text(event)),
            "created_at": event.get("created_at"),
        }

    @staticmethod
    def _pair_id(event_type: str, interaction: Dict[str, Any], payload: Dict[str, Any], event_id: str) -> str:
        return str(
            interaction.get("pair_id")
            or interaction.get("interaction_id")
            or payload.get("pair_id")
            or payload.get("call_id")
            or payload.get("tool_call_id")
            or payload.get("group_id")
            or event_id
            or event_type
        )

    @staticmethod
    def _tool_name(interaction: Dict[str, Any], payload: Dict[str, Any]) -> str:
        return str(interaction.get("tool_name") or payload.get("tool_name") or payload.get("tool") or "")

    @staticmethod
    def _dedupe(values: List[str]) -> List[str]:
        seen = set()
        deduped = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                deduped.append(value)
        return deduped

    @staticmethod
    def _retry_count(tool_chain: List[Dict[str, Any]]) -> int:
        seen = set()
        retries = 0
        for item in tool_chain:
            key = item.get("tool_name") or item.get("pair_id")
            if key in seen:
                retries += 1
            seen.add(key)
        return retries

    @staticmethod
    def _success_from_terminal(
        terminal_event: Optional[Dict[str, Any]],
        tool_chain: List[Dict[str, Any]],
        *,
        task_completion: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if task_completion:
            total = int(task_completion.get("total") or 0)
            completed = int(task_completion.get("completed") or 0)
            pending = int(task_completion.get("pending") or 0)
            blocked = int(task_completion.get("blocked") or 0)
            if total and (completed < total or pending or blocked):
                return False
        if terminal_event:
            event_type = _event_type(terminal_event)
            payload = terminal_event.get("payload") or {}
            interaction = _payload_interaction(payload)
            status = str(interaction.get("status") or payload.get("status") or "").lower()
            if "error" in event_type.lower() or status in {"failed", "blocked", "rejected"}:
                return False
            if event_type in {"pipeline.agent_done", "agent_done", "FC_DONE", "TASK_COMPLETE", "completed"}:
                return True
        if tool_chain:
            return not any(item.get("success") is False for item in tool_chain)
        return False

    @staticmethod
    def _terminal_result(terminal_event: Optional[Dict[str, Any]], events: List[Dict[str, Any]]) -> str:
        if terminal_event:
            return _event_text(terminal_event)[:1000]
        for event in reversed(events):
            if event.get("role") == "assistant":
                return _event_text(event)[:1000]
        return ""

    @staticmethod
    def _failure_reason(
        terminal_event: Optional[Dict[str, Any]],
        tool_chain: List[Dict[str, Any]],
        *,
        task_completion: Optional[Dict[str, Any]] = None,
    ) -> str:
        if task_completion:
            total = int(task_completion.get("total") or 0)
            completed = int(task_completion.get("completed") or 0)
            pending_nodes = task_completion.get("pending_nodes") or []
            blocked_nodes = task_completion.get("blocked_nodes") or []
            if total and completed < total:
                pending_text = ", ".join(
                    f"{item.get('id') or ''}:{item.get('label') or ''}"
                    for item in pending_nodes[:5]
                ).strip(", ")
                blocked_text = ", ".join(
                    f"{item.get('id') or ''}:{item.get('label') or ''}"
                    for item in blocked_nodes[:5]
                ).strip(", ")
                detail = pending_text or blocked_text
                return (
                    f"任务图未完成: {completed}/{total} 个叶子节点完成"
                    + (f"，未完成节点: {detail}" if detail else "")
                )[:500]
        if terminal_event:
            payload = terminal_event.get("payload") or {}
            for key in ("error", "failure_reason", "reason"):
                if payload.get(key):
                    return str(payload[key])[:500]
            text = _event_text(terminal_event)
            if text:
                return text[:500]
        failed_tools = [item for item in tool_chain if item.get("success") is False]
        if failed_tools:
            names = ", ".join(item.get("tool_name") or item.get("pair_id") or "unknown" for item in failed_tools[:5])
            return f"工具失败: {names}"
        return ""

    @staticmethod
    def _is_verification_event(event_type: str, interaction: Dict[str, Any], payload: Dict[str, Any], text: str) -> bool:
        tool_name = str(interaction.get("tool_name") or payload.get("tool_name") or "").lower()
        joined = f"{event_type} {tool_name} {text}".lower()
        return any(k in joined for k in ("test", "pytest", "tsc", "build", "验证", "检查", "checkpoint"))

    @staticmethod
    def _task_completion_state(events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Infer final TaskGraph leaf completion from persisted event payloads."""
        latest_graph: Optional[Dict[str, Any]] = None
        latest_progress: Optional[Dict[str, Any]] = None
        for event in events:
            payload = event.get("payload") or {}
            data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
            graph = None
            if isinstance(data, dict) and isinstance(data.get("graph"), dict):
                graph = data.get("graph")
            elif isinstance(payload.get("graph"), dict):
                graph = payload.get("graph")
            if graph:
                latest_graph = graph
            progress = None
            if isinstance(data, dict) and isinstance(data.get("progress"), dict):
                progress = data.get("progress")
            elif isinstance(payload.get("progress"), dict):
                progress = payload.get("progress")
            if progress:
                latest_progress = progress

        if latest_graph:
            nodes = latest_graph.get("nodes") or []
            parents = set()
            for edge in latest_graph.get("hEdges") or []:
                if isinstance(edge, (list, tuple)) and edge:
                    parents.add(str(edge[0]))
            leaves = []
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                node_id = str(node.get("id") or "")
                if node_id and node_id not in parents:
                    leaves.append(node)
            if not leaves:
                leaves = [node for node in nodes if isinstance(node, dict)]
            completed_nodes = [
                n for n in leaves
                if str(n.get("status") or "").lower() in {"completed", "skipped"}
            ]
            blocked_nodes = [
                n for n in leaves
                if str(n.get("status") or "").lower() in {"blocked", "needs_adjust", "waiting_input"}
            ]
            pending_nodes = [
                n for n in leaves
                if str(n.get("status") or "").lower() not in {"completed", "skipped"}
            ]
            return {
                "task_graph_id": str(latest_graph.get("id") or ""),
                "total": len(leaves),
                "completed": len(completed_nodes),
                "pending": len(pending_nodes),
                "blocked": len(blocked_nodes),
                "percent": int((len(completed_nodes) / len(leaves)) * 100) if leaves else 0,
                "pending_nodes": [
                    {
                        "id": str(n.get("id") or ""),
                        "label": str(n.get("label") or ""),
                        "status": str(n.get("status") or ""),
                    }
                    for n in pending_nodes[:10]
                ],
                "blocked_nodes": [
                    {
                        "id": str(n.get("id") or ""),
                        "label": str(n.get("label") or ""),
                        "status": str(n.get("status") or ""),
                    }
                    for n in blocked_nodes[:10]
                ],
            }

        if latest_progress:
            total = int(latest_progress.get("total") or 0)
            completed = int(latest_progress.get("completed") or 0)
            blocked = int(latest_progress.get("blocked") or 0)
            return {
                "task_graph_id": "",
                "total": total,
                "completed": completed,
                "pending": max(total - completed, 0),
                "blocked": blocked,
                "percent": int(latest_progress.get("percent") or 0),
                "pending_nodes": [],
                "blocked_nodes": [],
            }

        return {
            "task_graph_id": "",
            "total": 0,
            "completed": 0,
            "pending": 0,
            "blocked": 0,
            "percent": 0,
            "pending_nodes": [],
            "blocked_nodes": [],
        }

    @staticmethod
    def _infer_goal(events: List[Dict[str, Any]]) -> str:
        for event in events:
            if event.get("role") == "user":
                text = _event_text(event)
                if text:
                    return text[:500]
        for event in events:
            payload = event.get("payload") or {}
            if payload.get("goal"):
                return str(payload["goal"])[:500]
        return ""

    @staticmethod
    def _infer_turn_id(events: List[Dict[str, Any]]) -> str:
        for event in events:
            if event.get("turn_id"):
                return str(event["turn_id"])
        return ""

    @staticmethod
    def _infer_task_graph_id(events: List[Dict[str, Any]]) -> str:
        for event in events:
            if event.get("task_graph_id"):
                return str(event["task_graph_id"])
            payload = event.get("payload") or {}
            if payload.get("task_graph_id"):
                return str(payload["task_graph_id"])
            data = payload.get("data")
            if isinstance(data, dict):
                graph = data.get("graph")
                if isinstance(graph, dict) and graph.get("id"):
                    return str(graph["id"])
        return ""

    @staticmethod
    def _trace_id(conversation_id: str, turn_id: Optional[str], task_graph_id: Optional[str]) -> str:
        base = f"{conversation_id}:{turn_id or ''}:{task_graph_id or ''}"
        return f"trace:{_compact_id(base)}"

    @staticmethod
    def _summary(
        goal: str,
        tool_chain: List[Dict[str, Any]],
        approvals: List[Dict[str, Any]],
        result: str,
        success: bool,
        failure_reason: str,
    ) -> str:
        tool_names = [item.get("tool_name") or item.get("pair_id") for item in tool_chain]
        tool_part = " -> ".join(tool_names[:8]) if tool_names else "无工具调用"
        approval_part = f"{len(approvals)} 次审批" if approvals else "无审批"
        outcome = "成功" if success else f"失败/未完成: {failure_reason or '未知原因'}"
        return (
            f"目标: {goal or '未记录'}\n"
            f"工具链: {tool_part}\n"
            f"审批: {approval_part}\n"
            f"结果: {outcome}\n"
            f"摘要: {(result or '')[:300]}"
        ).strip()

    @staticmethod
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

    @staticmethod
    def _update_node(graph: Any, node: Any) -> None:
        if hasattr(graph, "update_node"):
            try:
                graph.update_node(node)
            except Exception:
                pass

    @staticmethod
    def _latest_round_id(mg: Any, conversation_id: str) -> str:
        try:
            from zulong.memory.memory_graph import NodeType
            candidates = []
            for node in mg.get_nodes_by_type(NodeType.DIALOGUE):
                meta = getattr(node, "metadata", {}) or {}
                if meta.get("sub_type") == "round" and meta.get("conversation_id") == conversation_id:
                    candidates.append(node)
            candidates.sort(key=lambda n: getattr(n, "created_at", 0.0))
            return getattr(candidates[-1], "node_id", "") if candidates else ""
        except Exception:
            return ""

    @staticmethod
    def _find_task_node_id(mg: Any, session_id: str, task_graph_id: str) -> str:
        if not task_graph_id:
            return ""
        for candidate in (f"task:{task_graph_id}", f"{session_id}/task:{task_graph_id}"):
            try:
                if mg.has_node(candidate):
                    return candidate
            except Exception:
                pass
        try:
            from zulong.memory.memory_graph import NodeType
            for node in mg.get_nodes_by_type(NodeType.TASK):
                meta = getattr(node, "metadata", {}) or {}
                if meta.get("graph_id") == task_graph_id:
                    return getattr(node, "node_id", "")
        except Exception:
            pass
        return ""


_task_execution_trace_extractor: Optional[TaskExecutionTraceExtractor] = None


def get_task_execution_trace_extractor() -> TaskExecutionTraceExtractor:
    global _task_execution_trace_extractor
    if _task_execution_trace_extractor is None:
        _task_execution_trace_extractor = TaskExecutionTraceExtractor()
    return _task_execution_trace_extractor


def maybe_finalize_task_execution_trace(
    *,
    conversation_id: Optional[str],
    turn_id: Optional[str],
    task_graph_id: Optional[str],
    event_type: str,
) -> Optional[Dict[str, Any]]:
    """Convenience hook for Web/IDE terminal events."""
    if not conversation_id or not _is_terminal_event(event_type):
        return None
    try:
        return get_task_execution_trace_extractor().finalize_for_event(
            conversation_id=conversation_id,
            turn_id=turn_id,
            task_graph_id=task_graph_id,
            event_type=event_type,
        )
    except Exception as exc:
        logger.debug("[TaskExecutionTrace] finalize skipped: %s", exc)
        return None
