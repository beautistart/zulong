"""Audit Web-visible interaction events against TSD/OpenHands display rules.

The script connects to the running Web service, sends one real CHAT_MESSAGE
for a scoped programming task, and records the interaction events that the UI
would render. It focuses on display-layer separation:

- user-facing task cards come from model/user-facing progress
- system/background execution details stay out of the main checklist
- task graph and thinking events are observable during a longer task
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_WS_URL = "ws://127.0.0.1:8090/ws"
AUDIT_PATH = ROOT / "tmp" / "web_interaction_display_audit.json"
PROGRESS_PATH = ROOT / "tmp" / "web_interaction_display_audit_progress.json"

NORMALIZED_TYPES = {
    "interaction:event": "INTERACTION_EVENT",
    "task:execution_status": "TASK_EXECUTION_STATUS",
    "task:progress": "THINKING_STEP",
    "reasoning": "THINKING_STEP",
    "graph:task:update": "TASK_GRAPH_UPDATE",
    "turn:accepted": "TURN_ACCEPTED",
    "text:final": "CHAT_RESPONSE",
    "task:complete": "CHAT_RESPONSE",
    "task:error": "TASK_ERROR",
    "attention:update": "ATTENTION_UPDATE",
}

FORBIDDEN_MAIN_TEXT_RE = re.compile(
    r"确认项目上下文|确认当前任务状态|进入执行链路|任务进入执行链路|"
    r"L1-B|L2/FC|推理链路|EventBus|后台桥|workspace trust|心跳|heartbeat|stalled_watch",
    re.IGNORECASE,
)
BACKGROUND_SOURCES = {"internal_control", "system_status"}
BACKGROUND_TOOLS = {
    "recall_memory",
    "read_memory_node",
    "discover_related",
    "ide_get_context",
    "request_tool_supplement",
}


def _now() -> float:
    return time.time()


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _payload(message: Dict[str, Any]) -> Dict[str, Any]:
    payload = message.get("payload")
    return payload if isinstance(payload, dict) else {}


def _data(message: Dict[str, Any]) -> Dict[str, Any]:
    data = message.get("data")
    return data if isinstance(data, dict) else {}


def _normalized_type(message: Dict[str, Any]) -> str:
    raw_type = str(message.get("type") or "")
    return NORMALIZED_TYPES.get(raw_type, raw_type)


def _request_id(message: Dict[str, Any]) -> str:
    payload = _payload(message)
    data = _data(message)
    return str(
        message.get("request_id")
        or message.get("turn_id")
        or message.get("msg_id")
        or payload.get("request_id")
        or payload.get("turn_id")
        or data.get("request_id")
        or ""
    )


def _session_id(message: Dict[str, Any]) -> str:
    payload = _payload(message)
    data = _data(message)
    return str(
        message.get("session_id")
        or message.get("conversation_id")
        or payload.get("session_id")
        or payload.get("conversation_id")
        or data.get("session_id")
        or data.get("conversation_id")
        or ""
    )


def _graph_from_message(message: Dict[str, Any]) -> Dict[str, Any]:
    payload = _payload(message)
    data = _data(message)
    graph = message.get("graph") or payload.get("graph") or data.get("graph")
    return graph if isinstance(graph, dict) else {}


def _graph_id(message: Dict[str, Any]) -> str:
    payload = _payload(message)
    graph = _graph_from_message(message)
    return str(
        message.get("task_graph_id")
        or message.get("graph_id")
        or payload.get("task_graph_id")
        or payload.get("graph_id")
        or graph.get("id")
        or ""
    )


def _workspace(message: Dict[str, Any]) -> str:
    payload = _payload(message)
    graph = _graph_from_message(message)
    metadata = graph.get("metadata") if isinstance(graph.get("metadata"), dict) else {}
    return str(
        message.get("workspace_path")
        or message.get("workspace_dir")
        or payload.get("workspace_path")
        or payload.get("workspace_dir")
        or graph.get("workspace_path")
        or graph.get("workspace_dir")
        or metadata.get("workspace_dir")
        or ""
    )


def _interaction_from(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    payload = _payload(message)
    data = _data(message)
    candidates = [
        message.get("interaction"),
        payload.get("interaction"),
        data.get("interaction"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    return None


def _ux_visibility(interaction: Dict[str, Any], message: Dict[str, Any]) -> str:
    payload = _payload(message)
    return str(
        interaction.get("ux_visibility")
        or message.get("_zulong_ux_visibility")
        or payload.get("_zulong_ux_visibility")
        or ""
    )


def _channel(interaction: Dict[str, Any], message: Dict[str, Any]) -> str:
    payload = _payload(message)
    return str(
        interaction.get("channel")
        or message.get("_zulong_channel")
        or payload.get("_zulong_channel")
        or ""
    )


def _progress_labels(interaction: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    for key in ("progress_items", "plan_steps"):
        value = interaction.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                label = item.get("label") or item.get("title") or item.get("detail")
            else:
                label = item
            if label:
                labels.append(str(label))
    return labels


def _tool_name(interaction: Dict[str, Any]) -> str:
    raw = interaction.get("raw_details")
    raw_tool = raw.get("tool_name") if isinstance(raw, dict) else ""
    return str(interaction.get("tool_name") or raw_tool or "")


def _interaction_text_values(interaction: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    for key in ("title", "detail", "next_step", "thought", "source_channel", "kind"):
        value = interaction.get(key)
        if value:
            yield key, str(value)
    for label in _progress_labels(interaction):
        yield "progress_label", label


def _summarize_interaction(message: Dict[str, Any], interaction: Dict[str, Any]) -> Dict[str, Any]:
    labels = _progress_labels(interaction)
    return {
        "type": _normalized_type(message),
        "request_id": _request_id(message),
        "session_id": _session_id(message),
        "task_graph_id": _graph_id(message),
        "workspace_path": _workspace(message),
        "kind": interaction.get("kind") or "",
        "status": interaction.get("status") or "",
        "source_channel": interaction.get("source_channel") or "",
        "ux_visibility": _ux_visibility(interaction, message),
        "channel": _channel(interaction, message),
        "tool_name": _tool_name(interaction),
        "is_background": bool(interaction.get("is_background")),
        "tool_category": interaction.get("tool_category") or "",
        "title": str(interaction.get("title") or "")[:180],
        "detail": str(interaction.get("detail") or "")[:260],
        "next_step": str(interaction.get("next_step") or "")[:180],
        "progress_labels": labels[:12],
        "progress_label_count": len(labels),
    }


def _event_is_relevant(message: Dict[str, Any], session_id: str, request_id: str) -> bool:
    msg_session = _session_id(message)
    msg_request = _request_id(message)
    if msg_session == session_id or msg_request == request_id:
        return True
    msg_type = _normalized_type(message)
    return msg_type in {"INTERACTION_EVENT", "TASK_GRAPH_UPDATE", "THINKING_STEP", "TASK_EXECUTION_STATUS", "ATTENTION_UPDATE"}


async def _recv_for(ws: Any, seconds: float, *, session_id: str, request_id: str) -> List[Dict[str, Any]]:
    deadline = time.monotonic() + seconds
    messages: List[Dict[str, Any]] = []
    while time.monotonic() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.monotonic()))
        except asyncio.TimeoutError:
            break
        try:
            data = json.loads(raw)
        except Exception:
            continue
        if isinstance(data, dict) and _event_is_relevant(data, session_id, request_id):
            messages.append(data)
    return messages


def analyze_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    type_counts: Dict[str, int] = {}
    interactions: List[Dict[str, Any]] = []
    forbidden_hits: List[Dict[str, Any]] = []
    background_main_hits: List[Dict[str, Any]] = []
    interaction_raw_samples: List[Dict[str, Any]] = []
    graph_raw_samples: List[Dict[str, Any]] = []
    attention_raw_samples: List[Dict[str, Any]] = []
    graph_node_total = 0
    graph_edge_total = 0
    pipeline_graph_updates = 0
    task_graph_updates = 0
    attention_updates = 0
    pressure_values: List[float] = []

    for message in messages:
        msg_type = _normalized_type(message)
        type_counts[msg_type] = type_counts.get(msg_type, 0) + 1
        if msg_type == "INTERACTION_EVENT" and len(interaction_raw_samples) < 8:
            interaction_raw_samples.append(message)
        if msg_type == "TASK_GRAPH_UPDATE":
            task_graph_updates += 1
            if len(graph_raw_samples) < 4:
                graph_raw_samples.append(message)
            graph = _graph_from_message(message)
            nodes = graph.get("nodes")
            edges = graph.get("edges") or graph.get("hEdges")
            if isinstance(nodes, list):
                graph_node_total += len(nodes)
            if isinstance(edges, list):
                graph_edge_total += len(edges)
        if msg_type == "THINKING_STEP":
            payload = _payload(message)
            data = _data(message)
            step_type = str(message.get("step_type") or payload.get("step_type") or "")
            graph = data.get("graph") if isinstance(data.get("graph"), dict) else {}
            if step_type.startswith("pipeline.") and graph:
                pipeline_graph_updates += 1
                if len(graph_raw_samples) < 4:
                    graph_raw_samples.append({
                        "type": msg_type,
                        "step_type": step_type,
                        "request_id": _request_id(message),
                        "session_id": _session_id(message),
                        "task_graph_id": data.get("task_graph_id"),
                        "graph": {
                            "id": graph.get("id"),
                            "nodes": graph.get("nodes", [])[:8],
                            "hEdges": graph.get("hEdges", [])[:8],
                            "activeNodeId": graph.get("activeNodeId"),
                        },
                    })
                nodes = graph.get("nodes")
                edges = graph.get("edges") or graph.get("hEdges")
                if isinstance(nodes, list):
                    graph_node_total += len(nodes)
                if isinstance(edges, list):
                    graph_edge_total += len(edges)
            if step_type == "attention.update":
                attention_updates += 1
                if len(attention_raw_samples) < 4:
                    attention_raw_samples.append({
                        "type": msg_type,
                        "step_type": step_type,
                        "request_id": _request_id(message),
                        "session_id": _session_id(message),
                        "data": data,
                    })
                pressure = data.get("context_pressure")
                if pressure is not None:
                    try:
                        pressure_values.append(float(pressure))
                    except Exception:
                        pass
        if msg_type == "ATTENTION_UPDATE":
            attention_updates += 1
            if len(attention_raw_samples) < 4:
                attention_raw_samples.append(message)
            payload = _payload(message)
            pressure = message.get("context_pressure") or payload.get("context_pressure")
            if pressure is not None:
                try:
                    pressure_values.append(float(pressure))
                except Exception:
                    pass
        interaction = _interaction_from(message)
        if not interaction:
            continue
        summary = _summarize_interaction(message, interaction)
        interactions.append(summary)
        is_main = summary["ux_visibility"] == "main"
        if is_main:
            for field, text in _interaction_text_values(interaction):
                if FORBIDDEN_MAIN_TEXT_RE.search(text):
                    forbidden_hits.append({
                        "field": field,
                        "text": text[:220],
                        "interaction": summary,
                    })
                    break
            tool_name = summary["tool_name"]
            if (
                summary["source_channel"] in BACKGROUND_SOURCES
                or summary["is_background"]
                or summary["tool_category"] == "background"
                or any(tool in tool_name for tool in BACKGROUND_TOOLS)
            ):
                background_main_hits.append(summary)

    main_interactions = [item for item in interactions if item["ux_visibility"] == "main"]
    detail_interactions = [item for item in interactions if item["ux_visibility"] == "details"]
    hidden_interactions = [item for item in interactions if item["ux_visibility"] == "hidden"]
    main_model_progress = [
        item for item in main_interactions
        if item["source_channel"] in {"model_progress", "model_final"} or item["kind"] == "summary"
    ]
    main_checklist = [
        item for item in main_interactions
        if item["progress_label_count"] > 0
    ]
    detail_execution = [
        item for item in detail_interactions
        if item["kind"] in {"action", "observation", "plan", "progress"}
        or item["source_channel"] in BACKGROUND_SOURCES
        or item["tool_name"]
    ]
    quality_like = [
        item for item in interactions
        if item["kind"] == "summary"
        or re.search(r"质量|复核|风险|review|audit|verify|验证", " ".join([
            item["title"],
            item["detail"],
            item["next_step"],
        ]), re.IGNORECASE)
    ]

    checks = {
        "turn_accepted_seen": type_counts.get("TURN_ACCEPTED", 0) >= 1,
        "task_status_seen": type_counts.get("TASK_EXECUTION_STATUS", 0) >= 1,
        "interaction_event_seen": len(interactions) >= 1,
        "main_model_progress_seen": len(main_model_progress) >= 1,
        "main_checklist_seen": len(main_checklist) >= 1,
        "details_execution_seen": len(detail_execution) >= 1,
        "no_forbidden_main_text": not forbidden_hits,
        "no_background_main_cards": not background_main_hits,
        "thinking_steps_seen": type_counts.get("THINKING_STEP", 0) >= 1,
        "task_graph_update_seen": (task_graph_updates + pipeline_graph_updates) >= 1,
        "task_graph_nodes_seen": graph_node_total > 0,
        "attention_update_seen": attention_updates >= 1,
        "context_pressure_seen": bool(pressure_values),
        "quality_review_seen": len(quality_like) >= 1,
    }
    display_required = [
        checks["interaction_event_seen"],
        checks["main_model_progress_seen"],
        checks["details_execution_seen"],
        checks["no_forbidden_main_text"],
        checks["no_background_main_cards"],
    ]
    return {
        "ok": all(display_required),
        "checks": checks,
        "type_counts": type_counts,
        "interaction_counts": {
            "total": len(interactions),
            "main": len(main_interactions),
            "details": len(detail_interactions),
            "hidden": len(hidden_interactions),
            "main_model_progress": len(main_model_progress),
            "main_checklist": len(main_checklist),
            "detail_execution": len(detail_execution),
        },
        "graph_evidence": {
            "node_count_seen": graph_node_total,
            "edge_count_seen": graph_edge_total,
            "task_graph_update_count": task_graph_updates,
            "pipeline_graph_update_count": pipeline_graph_updates,
            "attention_update_count": attention_updates,
            "context_pressure_values": pressure_values[:8],
        },
        "samples": {
            "main": main_interactions[:8],
            "details": detail_interactions[:8],
            "hidden": hidden_interactions[:4],
            "main_checklist": main_checklist[:6],
            "forbidden_hits": forbidden_hits[:6],
            "background_main_hits": background_main_hits[:6],
            "quality_like": quality_like[:6],
            "interaction_raw": interaction_raw_samples,
            "graph_raw": graph_raw_samples,
            "attention_raw": attention_raw_samples,
        },
    }


async def run_audit(ws_url: str, observe_seconds: float) -> Dict[str, Any]:
    try:
        import websockets
    except Exception as exc:
        raise RuntimeError(f"websockets package unavailable: {exc}") from exc

    started_at = _now()
    run_id = uuid.uuid4().hex[:10]
    session_id = f"web-display-audit-{run_id}"
    request_id = f"display-audit-{run_id}"
    target_dir = ROOT / "tmp" / f"zulong-interaction-audit-{run_id}"
    target_dir.mkdir(parents=True, exist_ok=True)

    prompt = (
        f"请在 {target_dir} 中执行一个复杂但范围受控的 Web 编程任务：创建单页任务审计看板 Demo。"
        "功能包括添加任务、切换完成状态、按全部/待办/完成过滤、localStorage 保存、底部统计。"
        "请按 TSD 用户交互规范展示：开始时说明目标和计划，中途每个关键步骤说明正在做什么，"
        "完成后做质量自查并回复 index.html 完整路径。"
        "主任务卡只能展示用户能理解的计划和当前步骤；后台动作、EventBus、L1-B、L2/FC、确认上下文、进入执行链路等不要作为任务步骤展示。"
    )

    messages: List[Dict[str, Any]] = []
    audit: Dict[str, Any] = {
        "ok": False,
        "started_at": started_at,
        "ws_url": ws_url,
        "session_id": session_id,
        "request_id": request_id,
        "target_dir": str(target_dir),
        "observe_seconds": observe_seconds,
    }

    async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps({"type": "ping"}, ensure_ascii=False))
        messages.extend(await _recv_for(ws, 1.5, session_id=session_id, request_id=request_id))
        await ws.send(json.dumps({
            "type": "conversation:switch",
            "session_id": session_id,
            "conversation_id": session_id,
            "session_node_id": f"dialogue:session_{session_id}",
            "dialogue_session_id": f"dialogue:session_{session_id}",
            "is_new_session": True,
            "clear_active_graph": True,
            "source": "web_interaction_display_audit",
        }, ensure_ascii=False))
        messages.extend(await _recv_for(ws, 1.0, session_id=session_id, request_id=request_id))
        await ws.send(json.dumps({
            "type": "CHAT_MESSAGE",
            "text": prompt,
            "session_id": session_id,
            "conversation_id": session_id,
            "request_id": request_id,
            "turn_id": request_id,
            "session_node_id": f"dialogue:session_{session_id}",
            "dialogue_session_id": f"dialogue:session_{session_id}",
            "workspace_path": str(target_dir),
            "cwd": str(target_dir),
            "source": "web_interaction_display_audit",
        }, ensure_ascii=False))

        deadline = _now() + observe_seconds
        while _now() < deadline:
            chunk = await _recv_for(ws, min(5.0, max(0.1, deadline - _now())), session_id=session_id, request_id=request_id)
            messages.extend(chunk)
            analysis = analyze_messages(messages)
            _write_json(PROGRESS_PATH, {
                "session_id": session_id,
                "request_id": request_id,
                "updated_at": _now(),
                "elapsed_seconds": round(_now() - started_at, 1),
                "message_count": len(messages),
                "checks": analysis["checks"],
                "interaction_counts": analysis["interaction_counts"],
            })
            if analysis["ok"] and analysis["checks"].get("task_graph_update_seen"):
                # Keep listening a bit after the core display contract appears.
                if _now() - started_at >= min(observe_seconds, 35.0):
                    break

    analysis = analyze_messages(messages)
    audit.update(analysis)
    audit["finished_at"] = _now()
    audit["duration_seconds"] = round(audit["finished_at"] - started_at, 1)
    audit["message_count"] = len(messages)
    audit["graph_ids"] = sorted({value for value in (_graph_id(msg) for msg in messages) if value})
    audit["workspace_paths"] = sorted({value for value in (_workspace(msg) for msg in messages) if value})
    _write_json(AUDIT_PATH, audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL)
    parser.add_argument("--observe-seconds", type=float, default=90.0)
    parser.add_argument("--strict", action="store_true", help="also require optional attention/quality/checklist evidence")
    args = parser.parse_args()

    audit = asyncio.run(run_audit(args.ws_url, args.observe_seconds))
    print(json.dumps({
        "ok": audit.get("ok"),
        "checks": audit.get("checks"),
        "interaction_counts": audit.get("interaction_counts"),
        "graph_evidence": audit.get("graph_evidence"),
        "target_dir": audit.get("target_dir"),
        "audit_path": str(AUDIT_PATH),
    }, ensure_ascii=False, indent=2))

    if not audit.get("ok"):
        raise SystemExit(1)
        if args.strict:
            checks = audit.get("checks", {})
            strict_required = [
            checks.get("turn_accepted_seen"),
            checks.get("task_status_seen"),
            checks.get("main_checklist_seen"),
            checks.get("thinking_steps_seen"),
            checks.get("task_graph_update_seen"),
            checks.get("task_graph_nodes_seen"),
            checks.get("attention_update_seen"),
            checks.get("context_pressure_seen"),
            checks.get("quality_review_seen"),
        ]
        if not all(strict_required):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
