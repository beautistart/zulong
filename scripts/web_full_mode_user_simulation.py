"""Exercise the Zulong Web FULL-mode coding task lifecycle.

This script behaves like the Web UI:
- sends CHAT_MESSAGE through ws://127.0.0.1:8090/ws
- sends STOP_GENERATION through the same WebSocket
- resumes and then modifies the same task graph
- records observable Web messages and filesystem evidence

It intentionally does not call internal task creation APIs; the running Zulong
FULL service must create the TaskGraph, workspace, and files itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_WS_URL = "ws://127.0.0.1:8090/ws"
AUDIT_PATH = ROOT / "tmp" / "web_full_mode_user_simulation_audit.json"
PROGRESS_PATH = ROOT / "tmp" / "web_full_mode_user_simulation_progress.json"

NORMALIZED_TYPES = {
    "turn:accepted": "TURN_ACCEPTED",
    "task:execution_status": "TASK_EXECUTION_STATUS",
    "task:progress": "THINKING_STEP",
    "reasoning": "THINKING_STEP",
    "graph:task:update": "TASK_GRAPH_UPDATE",
    "attention:update": "ATTENTION_UPDATE",
    "interaction:event": "INTERACTION_EVENT",
    "text:stream": "STREAMING_RESPONSE",
    "text:final": "CHAT_RESPONSE",
    "task:complete": "CHAT_RESPONSE",
    "task:error": "TASK_ERROR",
}


def _now() -> float:
    return time.time()


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text or ""))


def _is_ascii_path(text: str) -> bool:
    return bool(text) and all(ord(ch) < 128 for ch in text)


def _extract_payload(message: Dict[str, Any]) -> Dict[str, Any]:
    payload = message.get("payload")
    return payload if isinstance(payload, dict) else {}


def _message_type(message: Dict[str, Any]) -> str:
    raw_type = str(message.get("type") or "")
    return NORMALIZED_TYPES.get(raw_type, raw_type)


def _message_text(message: Dict[str, Any]) -> str:
    payload = _extract_payload(message)
    parts = [
        message.get("text"),
        message.get("message"),
        message.get("error"),
        payload.get("text"),
        payload.get("message"),
        payload.get("error"),
    ]
    return "\n".join(str(item) for item in parts if item)


def _message_request_id(message: Dict[str, Any]) -> str:
    payload = _extract_payload(message)
    return str(
        message.get("request_id")
        or message.get("turn_id")
        or message.get("msg_id")
        or payload.get("request_id")
        or payload.get("turn_id")
        or ""
    )


def _message_session_id(message: Dict[str, Any]) -> str:
    payload = _extract_payload(message)
    return str(
        message.get("conversation_id")
        or message.get("session_id")
        or payload.get("conversation_id")
        or payload.get("session_id")
        or ""
    )


def _message_state(message: Dict[str, Any]) -> str:
    payload = _extract_payload(message)
    return str(message.get("state") or payload.get("state") or "").lower()


def _message_phase(message: Dict[str, Any]) -> str:
    payload = _extract_payload(message)
    return str(message.get("phase") or payload.get("phase") or "").lower()


def _graph_from_message(message: Dict[str, Any]) -> Dict[str, Any]:
    payload = _extract_payload(message)
    graph = message.get("graph") or payload.get("graph")
    return graph if isinstance(graph, dict) else {}


def _graph_id_from_message(message: Dict[str, Any]) -> str:
    payload = _extract_payload(message)
    graph = _graph_from_message(message)
    value = str(
        message.get("task_graph_id")
        or message.get("graph_id")
        or payload.get("task_graph_id")
        or payload.get("graph_id")
        or graph.get("id")
        or ""
    )
    return value if value.startswith("tg_") else ""


def _workspace_from_message(message: Dict[str, Any]) -> str:
    payload = _extract_payload(message)
    graph = _graph_from_message(message)
    metadata = graph.get("metadata") if isinstance(graph.get("metadata"), dict) else {}
    candidates = [
        message.get("workspace_path"),
        message.get("workspace_dir"),
        payload.get("workspace_path"),
        payload.get("workspace_dir"),
        graph.get("workspace_path"),
        graph.get("workspace_dir"),
        metadata.get("workspace_dir"),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return ""


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class Simulation:
    def __init__(self, ws_url: str, *, test_open_actions: bool, timeout: float) -> None:
        self.ws_url = ws_url
        self.test_open_actions = test_open_actions
        self.timeout = timeout
        self.started_at = _now()
        self.session_id = f"web-full-sim-{uuid.uuid4().hex[:12]}"
        self.session_node_id = f"dialogue:session_{self.session_id}"
        self.repo_workspace = str(ROOT)
        self.messages: List[Dict[str, Any]] = []
        self.own_messages: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.graph_ids: List[str] = []
        self.primary_graph_id = ""
        self.workspace_paths: List[str] = []
        self.turns: Dict[str, str] = {}
        self._tracking_enabled = False
        self.audit: Dict[str, Any] = {
            "ok": False,
            "started_at": self.started_at,
            "session_id": self.session_id,
            "ws_url": self.ws_url,
            "checks": {},
            "turns": self.turns,
            "events": self.events,
            "graph_ids": self.graph_ids,
            "workspace_paths": self.workspace_paths,
            "evidence": {},
            "errors": [],
        }

    def is_relevant_message(self, message: Dict[str, Any]) -> bool:
        request_id = _message_request_id(message)
        session_id = _message_session_id(message)
        if session_id and session_id == self.session_id:
            return True
        if request_id and request_id in self.turns.values():
            return True
        if not self._tracking_enabled:
            return False
        msg_type = _message_type(message)
        if session_id or request_id:
            return False
        if msg_type not in {
            "TASK_EXECUTION_STATUS",
            "TASK_GRAPH_UPDATE",
            "INTERACTION_EVENT",
            "THINKING_STEP",
        }:
            return False
        graph_id = _graph_id_from_message(message)
        workspace = _workspace_from_message(message)
        if graph_id:
            return not self.graph_ids or graph_id in self.graph_ids
        if workspace:
            return not self.workspace_paths or workspace in self.workspace_paths
        return msg_type == "TASK_EXECUTION_STATUS" and not self.graph_ids

    def update_progress(self, phase: str, **extra: Any) -> None:
        payload = {
            "phase": phase,
            "updated_at": _now(),
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "graph_ids": self.graph_ids[-5:],
            "workspace_paths": self.workspace_paths[-5:],
            **extra,
        }
        _write_json(PROGRESS_PATH, payload)

    def remember(self, message: Dict[str, Any]) -> None:
        self.messages.append(message)
        if not self.is_relevant_message(message):
            return
        self.own_messages.append(message)
        msg_type = _message_type(message)
        graph_id = _graph_id_from_message(message)
        workspace = _workspace_from_message(message)
        should_record_graph = msg_type in {
            "TASK_GRAPH_UPDATE",
            "INTERACTION_EVENT",
            "CHAT_RESPONSE",
            "THINKING_STEP",
        }
        if msg_type == "TASK_EXECUTION_STATUS":
            should_record_graph = _message_phase(message) not in {"accepted", "running"}
        if graph_id and should_record_graph and graph_id not in self.graph_ids:
            self.graph_ids.append(graph_id)
            if not self.primary_graph_id:
                self.primary_graph_id = graph_id
        if workspace and workspace not in self.workspace_paths:
            self.workspace_paths.append(workspace)
        if msg_type in {
            "TURN_ACCEPTED",
            "TASK_EXECUTION_STATUS",
            "TASK_GRAPH_UPDATE",
            "INTERACTION_EVENT",
            "CHAT_RESPONSE",
            "STREAMING_RESPONSE",
            "THINKING_STEP",
            "STOP_ACK",
            "ide_action_result",
        }:
            self.events.append({
                "type": msg_type,
                "request_id": _message_request_id(message),
                "session_id": _message_session_id(message),
                "state": _message_state(message),
                "phase": _message_phase(message),
                "graph_id": graph_id,
                "workspace_path": workspace,
                "text": _message_text(message)[:260],
                "ts": message.get("ts") or message.get("timestamp") or _now(),
            })

    async def recv_for(self, ws: Any, seconds: float, *, until: Optional[Any] = None) -> List[Dict[str, Any]]:
        deadline = _now() + seconds
        seen: List[Dict[str, Any]] = []
        while _now() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - _now()))
            except asyncio.TimeoutError:
                break
            try:
                data = json.loads(raw)
            except Exception:
                continue
            if isinstance(data, dict):
                self.remember(data)
                seen.append(data)
                if until and until(data):
                    break
        return seen

    async def send_chat(self, ws: Any, text: str, turn_name: str, **extra: Any) -> str:
        request_id = f"{turn_name}-{uuid.uuid4().hex[:10]}"
        self.turns[turn_name] = request_id
        self._tracking_enabled = True
        payload = {
            "type": "CHAT_MESSAGE",
            "text": text,
            "session_id": self.session_id,
            "conversation_id": self.session_id,
            "request_id": request_id,
            "turn_id": request_id,
            "session_node_id": self.session_node_id,
            "dialogue_session_id": self.session_node_id,
            "source": "web_full_mode_user_simulation",
            "workspace_path": self.repo_workspace,
            "cwd": self.repo_workspace,
            **extra,
        }
        await ws.send(json.dumps(payload, ensure_ascii=False))
        self.update_progress(f"sent_{turn_name}", request_id=request_id)
        return request_id

    async def switch_to_new_session(self, ws: Any) -> None:
        payload = {
            "type": "conversation:switch",
            "session_id": self.session_id,
            "conversation_id": self.session_id,
            "session_node_id": self.session_node_id,
            "dialogue_session_id": self.session_node_id,
            "is_new_session": True,
            "clear_active_graph": True,
            "source": "web_full_mode_user_simulation",
        }
        await ws.send(json.dumps(payload, ensure_ascii=False))
        self.update_progress("switched_new_session")

    async def send_stop(self, ws: Any, request_id: str) -> None:
        await ws.send(json.dumps({
            "type": "STOP_GENERATION",
            "request_id": request_id,
            "session_id": self.session_id,
            "conversation_id": self.session_id,
            "task_graph_id": self.graph_ids[-1] if self.graph_ids else "",
            "source": "web_full_mode_user_simulation",
        }, ensure_ascii=False))
        self.update_progress("sent_stop", request_id=request_id)

    async def wait_for_completion(self, ws: Any, request_id: str, phase: str, seconds: float) -> bool:
        def done(message: Dict[str, Any]) -> bool:
            if not self.is_relevant_message(message):
                return False
            msg_type = _message_type(message)
            if _message_request_id(message) not in {"", request_id}:
                return False
            if msg_type == "TASK_EXECUTION_STATUS":
                return _message_state(message) in {"completed", "succeeded"}
            if msg_type in {"CHAT_RESPONSE", "task_complete"}:
                return True
            return False

        await self.recv_for(ws, seconds, until=done)
        completed = any(done(msg) for msg in self.own_messages)
        self.update_progress(phase, completed=completed, request_id=request_id)
        return completed

    async def wait_for_index_html(self, ws: Any, seconds: float, phase: str) -> bool:
        deadline = _now() + seconds
        while _now() < deadline:
            fs = self.verify_filesystem()
            if fs.get("index_exists"):
                self.update_progress(phase, index_html=fs.get("index_html"), workspace_path=fs.get("workspace_path"))
                return True
            await self.recv_for(ws, min(5.0, max(0.1, deadline - _now())))
        self.update_progress(phase, index_html="")
        return False

    def _candidate_workspaces(self) -> List[Path]:
        candidates: List[Path] = []
        for value in self.workspace_paths:
            if value:
                candidates.append(Path(value))
        graph_id = self.graph_ids[-1] if self.graph_ids else ""
        if graph_id:
            try:
                from zulong.workspace.project_registry import get_project_registry

                project = get_project_registry().get_project_by_graph_id(graph_id)
                if project and project.path:
                    candidates.insert(0, Path(project.path))
                    self.audit["evidence"]["project_registry"] = {
                        "project_id": project.project_id,
                        "name": project.name,
                        "path": project.path,
                        "task_graph_id": project.task_graph_id,
                    }
            except Exception as exc:
                self.audit["evidence"]["project_registry_error"] = str(exc)
        for root in (
            Path(r"C:\Users\HiWin11\Zulong\workspace"),
            ROOT / "agent_workspace",
            ROOT / "workspace",
        ):
            if root.exists():
                try:
                    recent = sorted(
                        [p for p in root.iterdir() if p.is_dir() and p.stat().st_mtime >= self.started_at - 30],
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    candidates.extend(recent[:8])
                except Exception:
                    pass
        unique: List[Path] = []
        seen = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            key = str(resolved).lower()
            if key not in seen:
                seen.add(key)
                unique.append(resolved)
        return unique

    def verify_filesystem(self) -> Dict[str, Any]:
        candidates = self._candidate_workspaces()
        evidence: Dict[str, Any] = {
            "candidate_workspaces": [str(path) for path in candidates],
            "workspace_path": "",
            "index_html": "",
            "index_exists": False,
            "workspace_ascii": False,
            "workspace_has_cjk": True,
            "index_contains_modify_marker": False,
            "index_contains_counter": False,
        }
        for workspace in candidates:
            index_path = workspace / "index.html"
            if index_path.exists():
                text = index_path.read_text(encoding="utf-8", errors="ignore")
                evidence.update({
                    "workspace_path": str(workspace),
                    "index_html": str(index_path),
                    "index_exists": True,
                    "workspace_ascii": _is_ascii_path(str(workspace)),
                    "workspace_has_cjk": _has_cjk(str(workspace)),
                    "index_contains_modify_marker": "已完成二次修改" in text or "祖龙计数" in text,
                    "index_contains_counter": "计数" in text and ("click" in text.lower() or "addEventListener" in text),
                    "index_preview": text[:500],
                })
                break
        self.audit["evidence"]["filesystem"] = evidence
        return evidence

    def check_messages(self) -> Dict[str, Any]:
        by_type = {}
        for message in self.own_messages:
            msg_type = _message_type(message)
            by_type[msg_type] = by_type.get(msg_type, 0) + 1
        status_events = [event for event in self.events if event["type"] == "TASK_EXECUTION_STATUS"]
        visible_start = any(event.get("phase") in {"accepted"} for event in status_events)
        visible_running = any(
            event.get("state") == "running"
            or event.get("phase") in {"running", "model_call", "tool_call", "tool_result", "streaming"}
            for event in status_events
        )
        cancelled = any(
            event["type"] == "STOP_ACK"
            or event.get("state") in {"cancelled", "canceled"}
            or event.get("phase") in {"cancelled", "canceled"}
            for event in self.events
        )
        completed = any(event.get("state") in {"completed", "succeeded"} for event in status_events)
        checks = {
            "turn_accepted_count": by_type.get("TURN_ACCEPTED", 0),
            "task_status_count": by_type.get("TASK_EXECUTION_STATUS", 0),
            "task_graph_update_count": by_type.get("TASK_GRAPH_UPDATE", 0),
            "interaction_event_count": by_type.get("INTERACTION_EVENT", 0),
            "stop_ack_seen": cancelled,
            "visible_start_feedback": visible_start,
            "visible_running_feedback": visible_running,
            "completion_status_seen": completed,
            "graph_id_count": len(self.graph_ids),
            "single_graph_after_resume": len(set(self.graph_ids)) <= 1 if self.graph_ids else False,
            "websocket_channel": True,
        }
        self.audit["evidence"]["message_type_counts"] = by_type
        return checks

    async def run(self) -> Dict[str, Any]:
        try:
            import websockets
        except Exception as exc:
            raise RuntimeError(f"websockets package unavailable: {exc}") from exc

        project_name = "祖龙黑灰计数器验证"
        chinese_target = str(ROOT / "tmp" / "中文父目录")
        initial_text = (
            "请执行一个新的FULL模式Web端编程任务：新建前端项目，"
            f"项目名叫“{project_name}”。目标父目录我故意写成 {chinese_target}，"
            "但实际创建的项目目录必须是英文/ASCII，路径不得包含中文。"
            "请先创建任务图，再创建 index.html。页面必须是黑灰配色，"
            "包含标题“祖龙 FULL 模式验证”，一个按钮初始显示“计数 0”，点击后递增。"
            "每一步开始前都要在前端可见状态里说明正在做什么，中途汇报进度，"
            "完成时回复 index.html 的完整路径。"
        )
        resume_text = (
            "继续刚才被我中断的任务，沿用原任务图和原工作目录，不要新建任务图。"
            "把 index.html 落盘并完成可点击计数功能；完成时回复文件完整路径。"
        )
        modify_text = (
            "继续修改原项目，仍然沿用同一个任务图和工作目录，不要新建任务图。"
            "请只修改 index.html：把按钮文案改成“祖龙计数 0”，"
            "并在页面底部增加“已完成二次修改”的文字。完成后回复文件完整路径。"
        )

        async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as ws:
            await self.recv_for(ws, 1.0)
            await ws.send(json.dumps({"type": "ping"}, ensure_ascii=False))
            await self.recv_for(ws, 2.0)
            await self.switch_to_new_session(ws)
            await self.recv_for(ws, 2.0)

            first_id = await self.send_chat(ws, initial_text, "start")
            await self.recv_for(ws, 14.0, until=lambda msg: bool(_graph_id_from_message(msg)))
            await self.send_stop(ws, first_id)
            await self.recv_for(ws, 8.0)

            graph_id = self.graph_ids[-1] if self.graph_ids else ""
            if self.primary_graph_id:
                graph_id = self.primary_graph_id
            workspace_path = self.workspace_paths[-1] if self.workspace_paths else self.repo_workspace
            resume_id = await self.send_chat(
                ws,
                resume_text,
                "resume",
                task_graph_id=graph_id,
                graph_id=graph_id,
                workspace_path=workspace_path,
                cwd=workspace_path,
            )
            await self.wait_for_completion(ws, resume_id, "resume_wait_complete", self.timeout)
            await self.wait_for_index_html(ws, 45.0, "resume_wait_index_html")

            graph_id = self.primary_graph_id or (self.graph_ids[-1] if self.graph_ids else graph_id)
            workspace_path = self.workspace_paths[-1] if self.workspace_paths else workspace_path
            modify_id = await self.send_chat(
                ws,
                modify_text,
                "modify",
                task_graph_id=graph_id,
                graph_id=graph_id,
                workspace_path=workspace_path,
                cwd=workspace_path,
            )
            await self.wait_for_completion(ws, modify_id, "modify_wait_complete", self.timeout)

            fs = self.verify_filesystem()
            if self.test_open_actions and fs.get("index_html"):
                open_payloads = [
                    {
                        "type": "ide_open_file",
                        "source": "web_message_context_menu",
                        "session_id": self.session_id,
                        "conversation_id": self.session_id,
                        "workspace_path": fs["workspace_path"],
                        "cwd": fs["workspace_path"],
                        "path": fs["index_html"],
                        "task_graph_id": graph_id,
                    },
                    {
                        "type": "ide_open_workspace",
                        "source": "web_message_context_menu",
                        "session_id": self.session_id,
                        "conversation_id": self.session_id,
                        "workspace_path": fs["workspace_path"],
                        "cwd": fs["workspace_path"],
                        "path": fs["workspace_path"],
                    },
                ]
                for payload in open_payloads:
                    await ws.send(json.dumps(payload, ensure_ascii=False))
                await self.recv_for(ws, 8.0)

        checks = self.check_messages()
        fs = self.audit["evidence"].get("filesystem") or self.verify_filesystem()
        checks.update({
            "index_html_exists": bool(fs.get("index_exists")),
            "workspace_path_ascii": bool(fs.get("workspace_ascii")) and not fs.get("workspace_has_cjk"),
            "modify_marker_written": bool(fs.get("index_contains_modify_marker")),
            "counter_logic_written": bool(fs.get("index_contains_counter")),
        })
        if self.test_open_actions:
            ide_results = [event for event in self.events if event["type"] == "ide_action_result"]
            checks["ide_open_actions_over_ws"] = len(ide_results) >= 1
            self.audit["evidence"]["ide_action_results"] = ide_results[-5:]
        required = [
            checks["turn_accepted_count"] >= 3,
            checks["task_status_count"] >= 3,
            checks["stop_ack_seen"],
            checks["visible_start_feedback"],
            checks["visible_running_feedback"],
            checks["completion_status_seen"],
            checks["graph_id_count"] >= 1,
            checks["single_graph_after_resume"],
            checks["index_html_exists"],
            checks["workspace_path_ascii"],
            checks["modify_marker_written"],
            checks["counter_logic_written"],
        ]
        if self.test_open_actions:
            required.append(bool(checks.get("ide_open_actions_over_ws")))
        self.audit["checks"] = checks
        self.audit["ok"] = all(required)
        self.audit["finished_at"] = _now()
        self.audit["duration_seconds"] = round(self.audit["finished_at"] - self.started_at, 1)
        _write_json(AUDIT_PATH, self.audit)
        return self.audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--test-open-actions", action="store_true")
    args = parser.parse_args()

    sim = Simulation(args.ws_url, test_open_actions=args.test_open_actions, timeout=args.timeout)
    try:
        audit = asyncio.run(sim.run())
    except Exception as exc:
        sim.audit["errors"].append(str(exc))
        sim.audit["finished_at"] = _now()
        _write_json(AUDIT_PATH, sim.audit)
        raise
    print(json.dumps({
        "ok": audit.get("ok"),
        "session_id": audit.get("session_id"),
        "checks": audit.get("checks"),
        "filesystem": audit.get("evidence", {}).get("filesystem"),
        "audit_path": str(AUDIT_PATH),
    }, ensure_ascii=False, indent=2))
    if not audit.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
