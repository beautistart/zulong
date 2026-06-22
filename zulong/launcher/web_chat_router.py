"""
Web 聊天路由器 — 通过 EventBus 路由消息到祖龙系统主链路

Web 前端是主系统界面，聊天消息不经过 IDE Server。

消息流:
  Full 模式: Web /ws → EventBus(USER_TEXT) → L1-B → L2 → EventBus(L2_OUTPUT/STREAM) → Web /ws
  IDE  模式: Web /ws → EventBus(USER_TEXT) → L1-B → L2 → EventBus(L2_OUTPUT/STREAM) → Web /ws
"""

import asyncio
import copy
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.routing import APIRouter

from zulong.core.unified_protocol import (
    ProtocolBridge, MessageType, make_unified_message,
)
from zulong.launcher.conversation_orchestrator import (
    RouteDecision,
    get_conversation_orchestrator,
)
from zulong.launcher.interaction_store import get_interaction_store
from zulong.core.message_visibility import (
    CHANNEL_FINAL,
    CHANNEL_LEDGER,
    CHANNEL_STATUS,
    is_public_payload,
    mark_public_payload,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_PRE_LLM_CONTEXT_TIMEOUT = 0.25
_MEMORY_GRAPH_SNAPSHOT_LOCK = threading.Lock()
_MEMORY_GRAPH_SNAPSHOT_TIMEOUT = 6.0
_VISIBLE_MESSAGE_MIRROR_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="zulong-visible-mirror")
_WINDOWS_PATH_IN_TEXT_RE = re.compile(r"(?P<path>[A-Za-z]:[\\/][^\s，。；;,\n\r\"'`]+)")
_POSIX_PATH_IN_TEXT_RE = re.compile(r"(?P<path>/(?:[^\s，。；;,\n\r\"'`]+))")


def _copy_payload_for_background(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return copy.deepcopy(payload or {})
    except Exception:
        return dict(payload or {})


def _mirror_visible_message_background(
    *,
    conversation_id: str,
    turn_id: str,
    role: str,
    text: str,
    event_type: str,
    source: str,
    payload: Dict[str, Any],
) -> None:
    mirror_payload = _copy_payload_for_background(payload)

    def _run() -> None:
        try:
            from zulong.launcher.memory_mirror import mirror_interaction_to_memory_graph

            mirror_interaction_to_memory_graph(
                conversation_id=conversation_id,
                turn_id=turn_id,
                role=role,
                text=text,
                event_type=event_type,
                source=source,
                payload=mirror_payload,
            )
        except Exception as mirror_exc:
            logger.debug(f"[WebChatRouter] MemoryGraph 后台镜像跳过: {event_type}: {mirror_exc}")

    try:
        _VISIBLE_MESSAGE_MIRROR_EXECUTOR.submit(_run)
    except Exception as submit_exc:
        logger.debug(f"[WebChatRouter] MemoryGraph 后台镜像提交失败: {event_type}: {submit_exc}")


def _extract_explicit_workspace_path(text: str) -> str:
    """Return an explicit workspace path mentioned by the user.

    If the mentioned target does not exist yet, use the nearest existing
    parent as the workspace so Web programming tasks can create new folders
    without falling back to the Zulong repo directory.
    """
    raw = str(text or "")
    match = _WINDOWS_PATH_IN_TEXT_RE.search(raw) or _POSIX_PATH_IN_TEXT_RE.search(raw)
    if not match:
        return ""
    candidate = match.group("path").strip().rstrip("\\/。；;，,")
    try:
        path = os.path.abspath(os.path.expanduser(os.path.expandvars(candidate)))
        if os.path.exists(path):
            return path
        parent = os.path.dirname(path)
        while parent and parent != os.path.dirname(parent):
            if os.path.isdir(parent):
                return parent
            parent = os.path.dirname(parent)
    except Exception:
        return ""
    return ""


def _same_path(left: Optional[str], right: Optional[str]) -> bool:
    if not left or not right:
        return False
    try:
        return os.path.normcase(os.path.abspath(str(left))) == os.path.normcase(os.path.abspath(str(right)))
    except Exception:
        return str(left) == str(right)


def _resolve_active_task_workspace(
    preferred: Optional[str] = None,
    task_graph_id: Optional[str] = None,
    *,
    prefer_explicit: bool = False,
) -> str:
    """Resolve the current task workspace without falling back to repo cwd."""
    candidates: List[str] = []

    def add_candidate(value: Optional[str]) -> None:
        if value:
            candidates.append(value)

    # Explicit workspace switching is different from resolving an ordinary
    # tool-call workspace. When the user/model asks to open a concrete
    # workspace, that destination must win over the active TaskGraph's current
    # workspace; otherwise recovery flows can never switch away from a stale
    # graph binding.
    if prefer_explicit:
        add_candidate(preferred)

    if task_graph_id:
        try:
            from zulong.workspace.project_registry import get_project_registry
            project = get_project_registry().get_project_by_graph_id(task_graph_id)
            if project and project.path:
                add_candidate(project.path)
        except Exception:
            pass

    try:
        from zulong.tools.task_tools import get_active_task_graph, get_active_workspace_dir
        active_workspace = get_active_workspace_dir()
        active_graph = get_active_task_graph()
        active_graph_id = getattr(active_graph, "id", None) if active_graph else None
        if active_workspace and (not task_graph_id or active_graph_id == task_graph_id):
            add_candidate(active_workspace)
        if active_graph is not None:
            meta_workspace = getattr(active_graph, "metadata", {}).get("workspace_dir")
            if meta_workspace and (not task_graph_id or active_graph_id == task_graph_id):
                add_candidate(meta_workspace)
    except Exception:
        pass

    if not prefer_explicit:
        # Web payloads can carry stale IDE context. Prefer authoritative task
        # binding first, and only use the requested path as a final fallback.
        add_candidate(preferred)

    for candidate in candidates:
        try:
            path = os.path.abspath(os.path.expanduser(os.path.expandvars(str(candidate))))
            if os.path.exists(path):
                return path
        except Exception:
            continue
    return ""


def _active_task_graph_binding() -> Dict[str, str]:
    """Return active TaskGraph id/workspace for Web message attribution."""
    binding = {"task_graph_id": "", "workspace_path": ""}
    try:
        from zulong.tools.task_tools import get_active_task_graph, get_active_workspace_dir
        tg = get_active_task_graph()
        if tg:
            binding["task_graph_id"] = str(getattr(tg, "id", "") or "")
            meta = getattr(tg, "metadata", {}) or {}
            binding["workspace_path"] = str(meta.get("workspace_dir") or "")
        if not binding["workspace_path"]:
            binding["workspace_path"] = str(get_active_workspace_dir() or "")
    except Exception:
        pass
    return binding


def _extract_bfs_task_graph_id(bfs_ctx: Dict[str, Any], user_text: str = "") -> str:
    """Extract the bound TaskGraph ID from BFS session context.

    TSD 23.11 uses the conversation window as the recovery anchor. Older BFS
    payloads may leave top-level task_graph_id empty while still returning the
    active task as node_id="task:tg_xxx"; promote that stable graph address.
    """
    graph_id = str(bfs_ctx.get("task_graph_id") or "").strip()
    if graph_id:
        return graph_id

    active_tasks = bfs_ctx.get("active_tasks") or []
    if not isinstance(active_tasks, list):
        return ""

    candidates: List[Dict[str, Any]] = []
    for task in active_tasks:
        if not isinstance(task, dict):
            continue
        node_id = str(task.get("node_id") or "").strip()
        if node_id.startswith("task:"):
            task = dict(task)
            task["_graph_id"] = node_id.split("task:", 1)[1].strip()
            candidates.append(task)

    if not candidates:
        return ""

    active_like = [
        task for task in candidates
        if str(task.get("status") or "").lower() in {"active", "running", "in_progress", "pending"}
    ]
    pool = active_like or candidates

    query_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", str(user_text or "").lower()))

    def score(task: Dict[str, Any]) -> float:
        label = str(task.get("label") or "").lower()
        label_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", label))
        lexical = len(query_tokens & label_tokens) / max(1, len(query_tokens | label_tokens))
        substring = 1.0 if label and (label in str(user_text or "").lower() or str(user_text or "").lower() in label) else 0.0
        activation = float(task.get("activation") or 0)
        return substring * 2.0 + lexical + activation

    pool.sort(key=score, reverse=True)
    return str(pool[0].get("_graph_id") or "").strip()

# ── 状态 ──────────────────────────────────────────────

# WebSocket 连接管理
_ws_clients: Set[WebSocket] = set()

# 协议版本追踪 (ws id → protocol_version: "legacy" | "unified")
_ws_protocols: Dict[int, str] = {}

# 客户端类型追踪 (ws id → client_type: "dashboard" | "ide_plugin" | ...)
_ws_client_types: Dict[int, str] = {}

# 每个 WebSocket 连接独立的发送锁。
#
# starlette/uvicorn 底层使用 websockets legacy protocol；同一连接如果被多个
# asyncio task 并发 send_json，会在 drain waiter 上触发 AssertionError。
_ws_send_locks: Dict[int, asyncio.Lock] = {}

# Low-priority visual updates can arrive in bursts while L2 is finishing. Keep
# only one in-flight broadcast per noisy type so they cannot delay chat output.
_NOISY_BROADCAST_TYPES = {"MEMORY_GRAPH_UPDATE", "THINKING_STEP"}
_NOISY_MIN_INTERVALS = {"MEMORY_GRAPH_UPDATE": 5.0, "THINKING_STEP": 0.75}
_noisy_broadcast_pending: Set[str] = set()
_noisy_broadcast_last_sent: Dict[str, float] = {}
_noisy_broadcast_lock = threading.Lock()

# 统一协议桥接器
_protocol_bridge = ProtocolBridge()

# 运行模式（由 LauncherApp 在启动后设置）
_launch_mode: Optional[str] = None  # "full" | "ide"

# asyncio 事件循环引用（用于从 EventBus 分发线程安全地调度协程）
_event_loop: Optional[asyncio.AbstractEventLoop] = None

# 活跃聊天取消事件（IDE 模式使用）
_chat_cancels: Dict[str, asyncio.Event] = {}
_recent_stop_ack_at: Dict[str, float] = {}

# Web 主界面的任务运行状态。TSD §23.3 要求执行中、等待审批、
# 长时间无事件都能被用户明确区分。
_TASK_STATUS_STALE_SECONDS = 90.0
_task_execution_status: Dict[str, Any] = {
    "state": "idle",
    "phase": "idle",
    "message": "当前没有正在执行的任务。",
    "request_id": "",
    "conversation_id": "",
    "session_id": "",
    "workspace_path": "",
    "project_id": "",
    "task_graph_id": "",
    "tool_name": "",
    "awaiting_approval": False,
    "approval": None,
    "last_progress_at": 0.0,
    "last_state_change_at": time.time(),
    "stale_after_seconds": _TASK_STATUS_STALE_SECONDS,
    "progress_items": [],
}


def _get_ws_send_lock(ws: WebSocket) -> asyncio.Lock:
    ws_id = id(ws)
    lock = _ws_send_locks.get(ws_id)
    if lock is None:
        lock = asyncio.Lock()
        _ws_send_locks[ws_id] = lock
    return lock


def _forget_ws_connection(ws: WebSocket) -> None:
    """清理 /ws 连接相关状态，供断连和发送失败路径复用。"""
    ws_id = id(ws)
    _ws_clients.discard(ws)
    _ws_protocols.pop(ws_id, None)
    _ws_client_types.pop(ws_id, None)
    _ws_send_locks.pop(ws_id, None)
    try:
        from zulong.ide.ide_server import _monitor_connections
        _monitor_connections.discard(ws)
    except Exception:
        pass


def _interaction_status_for_state(state: str) -> str:
    state = str(state or "").lower()
    if state in {"completed", "succeeded", "success", "idle"}:
        return "succeeded" if state != "idle" else "pending"
    if state in {"failed", "error"}:
        return "failed"
    if state in {"blocked", "possibly_stalled", "stalled"}:
        return "blocked"
    if state in {"cancelled", "canceled"}:
        return "cancelled"
    if state in {"waiting_approval", "awaiting_approval", "workspace_trust_required"}:
        return "awaiting_approval"
    return "running"


def _build_task_status_interaction(status: Dict[str, Any]) -> Dict[str, Any]:
    state = str(status.get("state") or "running")
    phase = str(status.get("phase") or state)
    message = str(status.get("message") or "任务状态更新")
    conversation_id = str(status.get("conversation_id") or status.get("session_id") or "")
    request_id = str(status.get("request_id") or status.get("turn_id") or "")
    pair_id = "task-status"
    if conversation_id or request_id:
        pair_id += f":{conversation_id or request_id}"
    if phase in {"approval_required", "approval_resolved", "workspace_trust_required"}:
        approval = status.get("approval") if isinstance(status.get("approval"), dict) else {}
        pair_id = str(approval.get("approval_id") or approval.get("approvalId") or pair_id)
    else:
        approval = {}
    raw_progress_items = status.get("progress_items")
    has_explicit_progress_items = isinstance(raw_progress_items, list) and bool(raw_progress_items)
    progress_items = raw_progress_items if has_explicit_progress_items else None
    if not progress_items:
        item_status = "running"
        if state in {"completed", "succeeded"}:
            item_status = "completed"
        elif state in {"blocked", "possibly_stalled", "failed"}:
            item_status = "blocked" if state != "failed" else "failed"
        elif state in {"waiting_approval", "awaiting_approval", "workspace_trust_required"}:
            item_status = "running"
        progress_items = [{
            "id": phase,
            "label": message,
            "status": item_status,
            "source": "heartbeat" if phase in {"heartbeat", "stalled_watch"} else "task_graph",
            "timestamp": status.get("last_progress_at") or time.time(),
        }]
    ux_visibility = "hidden"
    if status.get("awaiting_approval"):
        ux_visibility = "main"
    elif state in {"blocked", "possibly_stalled", "stalled", "failed"}:
        ux_visibility = "main"
    elif has_explicit_progress_items and phase not in {"heartbeat", "stalled_watch"}:
        # TASK_EXECUTION_STATUS is a status bar signal, not the source of the
        # main task checklist. User-facing model/task-plan events render cards.
        ux_visibility = "details"
    interaction = {
        "interaction_id": pair_id,
        "pair_id": pair_id,
        "kind": "approval" if status.get("awaiting_approval") else "progress",
        "status": _interaction_status_for_state(state),
        "title": status.get("title") or _task_status_title(state, phase),
        "detail": _humanize_task_status_text(message),
        "tool_name": approval.get("tool_name") or status.get("tool_name") or "",
        "progress_items": progress_items[:8],
        "next_step": status.get("next_step") or "",
        "source_channel": "system_status",
        "channel": CHANNEL_STATUS,
        "ux_visibility": ux_visibility,
        "is_background": ux_visibility != "main",
        "tool_category": "background" if ux_visibility != "main" else "",
        "timestamp": time.time(),
    }
    for key in (
        "approval_id",
        "approvalId",
        "action_summary",
        "tool_args",
        "risk_level",
        "risk_reason",
        "approval_mode",
        "confirmation_state",
    ):
        value = approval.get(key)
        if value not in (None, ""):
            normalized_key = "approval_id" if key == "approvalId" else key
            interaction[normalized_key] = value
    if approval and not interaction.get("action_summary"):
        interaction["action_summary"] = approval.get("summary") or approval.get("message") or message
    return interaction


def _task_status_title(state: str, phase: str) -> str:
    state = str(state or "").lower()
    phase = str(phase or "").lower()
    if state in {"waiting_approval", "awaiting_approval"}:
        return "任务已暂停，等待审批"
    if state == "workspace_trust_required" or phase == "workspace_trust_required":
        return "任务已暂停，等待 VS Code 信任"
    if state in {"possibly_stalled", "stalled"}:
        return "任务疑似卡住"
    if state in {"completed", "succeeded"}:
        return "任务已完成"
    if state in {"failed", "blocked"}:
        return "任务受阻"
    if state in {"idle", ""}:
        return "当前空闲"
    return "任务执行中"


def _humanize_task_status_text(text: str) -> str:
    """Convert internal event labels into user-facing task status text."""
    raw = str(text or "").strip()
    if not raw:
        return raw

    direct_map = {
        "pipeline.pipeline_start": "任务链路已启动，正在准备上下文。",
        "pipeline.agent_start": "祖龙正在分析任务并准备下一步。",
        "pipeline.agent_done": "本轮推理已完成，正在整理结果。",
        "pipeline.pipeline_done": "任务链路已完成。",
        "agent_tool_call": "正在调用工具处理任务。",
        "agent_tool_result": "工具已返回结果，正在继续判断。",
        "MEMORY_GRAPH_UPDATE": "记忆图谱已更新。",
    }
    if raw in direct_map:
        return direct_map[raw]
    if raw.startswith("pipeline."):
        return "任务链路正在推进。"
    if raw.startswith("agent_"):
        return "祖龙正在推进任务。"
    return raw


def _looks_like_incomplete_final_text(text: str) -> bool:
    """Detect final replies that are actually forced-stop/blockage reports."""
    raw = str(text or "")
    if not raw:
        return False
    stripped = raw.strip()
    positive_markers = (
        "全部完成",
        "任务完成",
        "已完成",
        "已创建",
        "已生成",
        "已写入",
        "创建成功",
        "写入成功",
        "生成成功",
    )
    if any(marker in raw for marker in positive_markers) and "未完成" not in raw:
        return False
    lowered = raw.lower()
    hard_markers = [
        "任务执行中断",
        "强制收敛",
        "无法正常回复",
        "系统当前出问题",
        "安全防护触发",
        "触发循环保护",
    ]
    if any(marker.lower() in lowered for marker in hard_markers):
        return True
    report_shape = ("汇报", "报告", "清单", "列表", "总览", "如下")
    report_context = ("历史", "记忆", "节点", "清理", "删除", "候选", "未完成", "blocked")
    if (
        any(marker in raw for marker in report_shape)
        and sum(1 for marker in report_context if marker.lower() in lowered) >= 2
    ):
        return False
    short_status_markers = (
        "任务受阻",
        "任务疑似卡住",
        "stalled",
        "possibly_stalled",
        "blocked",
        "timed out",
        "interrupted",
    )
    if len(stripped) <= 160 and any(marker.lower() in lowered for marker in short_status_markers):
        return True
    return False


def _active_task_completion_snapshot(task_graph_id: Optional[str] = None) -> Dict[str, Any]:
    """Return authoritative active TaskGraph completion state."""
    snapshot: Dict[str, Any] = {
        "total": 0,
        "completed": 0,
        "uncompleted": 0,
        "blocked": 0,
        "uncompleted_labels": [],
    }
    try:
        from zulong.tools.task_tools import get_active_task_graph
        tg = get_active_task_graph()
        if not tg:
            return snapshot
        if task_graph_id and getattr(tg, "id", None) and getattr(tg, "id", None) != task_graph_id:
            return snapshot
        leaves = [n for n in tg.get_leaf_nodes() if getattr(n, "id", "") != "req"]
        snapshot["total"] = len(leaves)
        snapshot["completed"] = sum(1 for n in leaves if getattr(n, "status", "") in ("completed", "skipped"))
        uncompleted = [n for n in leaves if getattr(n, "status", "") not in ("completed", "skipped")]
        snapshot["uncompleted"] = len(uncompleted)
        snapshot["blocked"] = sum(1 for n in uncompleted if getattr(n, "status", "") == "blocked")
        snapshot["uncompleted_labels"] = [
            f"{getattr(n, 'id', '')}({getattr(n, 'label', '')})"
            for n in uncompleted[:6]
        ]
    except Exception:
        pass
    return snapshot


def _task_status_snapshot() -> Dict[str, Any]:
    snapshot = dict(_task_execution_status)
    now = time.time()
    last_progress = float(snapshot.get("last_progress_at") or 0.0)
    if (
        snapshot.get("state") == "running"
        and not snapshot.get("awaiting_approval")
        and last_progress > 0
        and now - last_progress > _TASK_STATUS_STALE_SECONDS
    ):
        elapsed = int(now - last_progress)
        snapshot.update({
            "state": "possibly_stalled",
            "phase": "stalled_watch",
            "message": f"连接仍正常，但已 {elapsed}s 未观察到任务推进；可能卡在模型或工具调用，建议查看日志或等待恢复。",
        })
    snapshot["now"] = now
    snapshot["elapsed_since_progress"] = round(now - last_progress, 1) if last_progress else None
    if str(snapshot.get("state") or "").lower() == "idle":
        snapshot["interaction"] = None
    else:
        snapshot["interaction"] = _build_task_status_interaction(snapshot)
    return snapshot


def _broadcast_task_execution_status(snapshot: Dict[str, Any]) -> None:
    snapshot = mark_public_payload(dict(snapshot), CHANNEL_STATUS)
    message = {
        "type": "TASK_EXECUTION_STATUS",
        **snapshot,
        "payload": snapshot,
        "_persisted": True,
    }
    mark_public_payload(message, CHANNEL_STATUS)
    _schedule_broadcast(message)


def _is_cancelled_turn(
    request_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> bool:
    request_key = str(request_id or "").strip()
    conversation_key = str(conversation_id or session_id or "").strip()
    current_state = str(_task_execution_status.get("state") or "").lower()
    current_request = str(_task_execution_status.get("request_id") or "").strip()
    current_conversation = str(
        _task_execution_status.get("conversation_id")
        or _task_execution_status.get("session_id")
        or ""
    ).strip()
    if current_state == "cancelled":
        if request_key and current_request and request_key == current_request:
            return True
        if not request_key and conversation_key and conversation_key == current_conversation:
            return True
    try:
        from zulong.core.state_manager import state_manager
        is_cancelled = getattr(state_manager, "is_cancelled_context", None)
        if callable(is_cancelled):
            return bool(is_cancelled(request_key or None, conversation_key or None))
    except Exception:
        pass
    return False


def update_task_execution_status(
    *,
    state: Optional[str] = None,
    phase: Optional[str] = None,
    message: Optional[str] = None,
    request_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    workspace_path: Optional[str] = None,
    project_id: Optional[str] = None,
    task_graph_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    awaiting_approval: Optional[bool] = None,
    approval: Optional[Dict[str, Any]] = None,
    progress_items: Optional[List[Dict[str, Any]]] = None,
    broadcast: bool = True,
) -> Dict[str, Any]:
    """Update and broadcast the Web-visible task execution status."""
    now = time.time()
    previous_state = _task_execution_status.get("state")
    incoming_state = str(state or "").lower()
    if (
        incoming_state in {"running", "completed", "succeeded"}
        and _is_cancelled_turn(request_id, conversation_id, session_id)
    ):
        logger.info(
            "[WebChatRouter] 忽略已取消请求的后续状态: request_id=%s state=%s phase=%s",
            request_id or "-",
            state or "-",
            phase or "-",
        )
        return _task_status_snapshot()
    if state is not None:
        _task_execution_status["state"] = state
    if phase is not None:
        _task_execution_status["phase"] = phase
    if message is not None:
        _task_execution_status["message"] = message
    for key, value in (
        ("request_id", request_id),
        ("conversation_id", conversation_id),
        ("session_id", session_id),
        ("workspace_path", workspace_path),
        ("project_id", project_id),
        ("task_graph_id", task_graph_id),
        ("tool_name", tool_name),
    ):
        if value is not None:
            _task_execution_status[key] = value or ""
    if conversation_id is not None and session_id is None:
        _task_execution_status["session_id"] = conversation_id or ""
    if awaiting_approval is not None:
        _task_execution_status["awaiting_approval"] = bool(awaiting_approval)
    if approval is not None:
        _task_execution_status["approval"] = dict(approval)
    elif awaiting_approval is False:
        _task_execution_status["approval"] = None
    if progress_items is not None:
        _task_execution_status["progress_items"] = list(progress_items)
    progress_states = {
        "accepted",
        "running",
        "model_call",
        "tool_call",
        "tool_result",
        "approval_resolved",
        "streaming",
        "completed",
        "failed",
        "blocked",
    }
    if (
        state in {"running", "completed", "failed", "blocked", "cancelled", "succeeded"}
        or phase in progress_states
    ):
        _task_execution_status["last_progress_at"] = now
    if previous_state != _task_execution_status.get("state") or phase is not None:
        _task_execution_status["last_state_change_at"] = now
    snapshot = _task_status_snapshot()
    if broadcast:
        _broadcast_task_execution_status(snapshot)
    return snapshot


def _extract_text_from_message(message: dict) -> str:
    payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
    for key in ("text", "result", "message", "error"):
        value = message.get(key)
        if value:
            return str(value)
        value = payload.get(key)
        if value:
            return str(value)
    data = message.get("data") if isinstance(message.get("data"), dict) else payload.get("data")
    if isinstance(data, dict):
        interaction = data.get("interaction")
        if isinstance(interaction, dict):
            title = interaction.get("title") or ""
            detail = interaction.get("detail") or ""
            return "\n".join(str(x) for x in (title, detail) if x)
    interaction = message.get("interaction") or payload.get("interaction")
    if isinstance(interaction, dict):
        title = interaction.get("title") or ""
        detail = interaction.get("detail") or ""
        return "\n".join(str(x) for x in (title, detail) if x)
    for key in ("action_summary", "summary", "risk_reason", "tool_name", "phase"):
        value = message.get(key) or payload.get(key)
        if value:
            return str(value)
    return ""


def _resolve_conversation_binding(message: dict) -> Dict[str, Optional[str]]:
    payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
    turn_id = (
        message.get("turn_id")
        or payload.get("turn_id")
        or message.get("request_id")
        or payload.get("request_id")
    )
    explicit_conversation_id = (
        message.get("conversation_id")
        or message.get("session_id")
        or payload.get("conversation_id")
        or payload.get("session_id")
    )

    store = get_interaction_store()
    binding: Dict[str, Optional[str]] = {
        "conversation_id": explicit_conversation_id,
        "turn_id": turn_id,
        "session_node_id": (
            message.get("session_node_id")
            or message.get("dialogue_session_id")
            or payload.get("session_node_id")
            or payload.get("dialogue_session_id")
        ),
        "workspace_path": message.get("workspace_path") or payload.get("workspace_path") or payload.get("cwd"),
        "project_id": message.get("project_id") or payload.get("project_id"),
        "task_graph_id": message.get("task_graph_id") or payload.get("task_graph_id"),
    }
    task_fallback_allowed = True

    if explicit_conversation_id:
        try:
            explicit_conversation = store.get_conversation(explicit_conversation_id)
        except Exception:
            explicit_conversation = None
        if explicit_conversation:
            message_has_task_binding = bool(
                binding.get("workspace_path")
                or binding.get("project_id")
                or binding.get("task_graph_id")
            )
            explicit_has_task_binding = bool(
                explicit_conversation.get("workspace_path")
                or explicit_conversation.get("project_id")
                or explicit_conversation.get("task_graph_id")
            )
            binding["workspace_path"] = binding["workspace_path"] or explicit_conversation.get("workspace_path")
            binding["project_id"] = binding["project_id"] or explicit_conversation.get("project_id")
            binding["task_graph_id"] = binding["task_graph_id"] or explicit_conversation.get("task_graph_id")
            binding["session_node_id"] = binding["session_node_id"] or explicit_conversation.get("session_node_id")
            if not message_has_task_binding and not explicit_has_task_binding:
                task_fallback_allowed = False
        elif not (
            binding.get("workspace_path")
            or binding.get("project_id")
            or binding.get("task_graph_id")
        ):
            task_fallback_allowed = False

    if turn_id:
        try:
            turn_binding = store.find_conversation_for_turn(turn_id)
        except Exception:
            turn_binding = None
        if turn_binding:
            message_has_task_binding = bool(
                binding.get("workspace_path")
                or binding.get("project_id")
                or binding.get("task_graph_id")
            )
            turn_has_task_binding = bool(
                turn_binding.get("workspace_path")
                or turn_binding.get("project_id")
                or turn_binding.get("task_graph_id")
            )
            resolved_conversation_id = turn_binding.get("conversation_id")
            if (
                explicit_conversation_id
                and resolved_conversation_id
                and explicit_conversation_id != resolved_conversation_id
            ):
                logger.debug(
                    "[WebChatRouter] turn 绑定覆盖会话: explicit=%s -> resolved=%s, turn_id=%s",
                    explicit_conversation_id,
                    resolved_conversation_id,
                    turn_id,
                )
            binding["conversation_id"] = turn_binding.get("conversation_id")
            binding["workspace_path"] = binding["workspace_path"] or turn_binding.get("workspace_path")
            binding["project_id"] = binding["project_id"] or turn_binding.get("project_id")
            binding["task_graph_id"] = binding["task_graph_id"] or turn_binding.get("task_graph_id")
            binding["session_node_id"] = binding["session_node_id"] or turn_binding.get("session_node_id")
            if not message_has_task_binding and not turn_has_task_binding:
                task_fallback_allowed = False

    if not binding["conversation_id"]:
        try:
            active = store.find_active_conversation(max_age_seconds=3600)
            binding["conversation_id"] = active.get("conversation_id") if active else None
            if active:
                binding["workspace_path"] = binding["workspace_path"] or active.get("workspace_path")
                binding["project_id"] = binding["project_id"] or active.get("project_id")
                binding["task_graph_id"] = binding["task_graph_id"] or active.get("task_graph_id")
                binding["session_node_id"] = binding["session_node_id"] or active.get("session_node_id")
        except Exception:
            binding["conversation_id"] = None

    if binding.get("conversation_id") and not binding.get("session_node_id"):
        conv_id = str(binding.get("conversation_id") or "")
        binding["session_node_id"] = (
            conv_id if _is_dialogue_session_node_id(conv_id)
            else f"dialogue:session_{_compact_dialogue_id(conv_id)}"
        )

    status_request_id = str(_task_execution_status.get("request_id") or "")
    status_conversation_id = str(
        _task_execution_status.get("conversation_id")
        or _task_execution_status.get("session_id")
        or ""
    )
    if (
        task_fallback_allowed
        and (
        (turn_id and status_request_id and str(turn_id) == status_request_id)
        or (
            binding.get("conversation_id")
            and status_conversation_id
            and binding.get("conversation_id") == status_conversation_id
        )
        )
    ):
        binding["workspace_path"] = (
            binding.get("workspace_path")
            or _task_execution_status.get("workspace_path")
            or None
        )
        binding["project_id"] = (
            binding.get("project_id")
            or _task_execution_status.get("project_id")
            or None
        )
        binding["task_graph_id"] = (
            binding.get("task_graph_id")
            or _task_execution_status.get("task_graph_id")
            or None
        )

    if (
        task_fallback_allowed
        and not explicit_conversation_id
        and (not binding.get("workspace_path") or not binding.get("task_graph_id"))
    ):
        active_binding = _active_task_graph_binding()
        if active_binding:
            binding["workspace_path"] = (
                binding.get("workspace_path")
                or active_binding.get("workspace_path")
                or None
            )
            binding["task_graph_id"] = (
                binding.get("task_graph_id")
                or active_binding.get("task_graph_id")
                or None
            )

    return binding


def _persist_web_visible_message(message: dict) -> None:
    """Persist user-visible Web messages that were generated outside prepare_turn.

    Web localStorage intentionally no longer stores message bodies.  This small
    bridge keeps assistant outputs and important tool/status cards recoverable
    after refresh without introducing another event channel.
    """
    if not is_public_payload(message):
        logger.debug(
            "[WebChatRouter] 跳过内部控制消息持久化: type=%s",
            message.get("type"),
        )
        return
    if message.get("_persisted"):
        return
    original_payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
    binding = _resolve_conversation_binding(message)
    conversation_id = binding.get("conversation_id")
    if not conversation_id:
        return
    event_type = str(message.get("type") or "")
    role = _event_role_for_persistence(event_type, message)
    if not role:
        return
    text = _extract_text_from_message(message)
    payload = dict(original_payload) if isinstance(original_payload, dict) and original_payload else dict(message)
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    if event_type == "THINKING_STEP":
        step_type = payload.get("step_type") or message.get("step_type")
        if step_type:
            payload["step_type"] = step_type
            event_type = str(step_type)
        if not data and isinstance(message.get("data"), dict):
            data = message["data"]
            payload["data"] = data
    if conversation_id and not payload.get("conversation_id"):
        payload["conversation_id"] = conversation_id
    if conversation_id and not payload.get("session_id"):
        payload["session_id"] = conversation_id
    if binding.get("session_node_id") and not payload.get("session_node_id"):
        payload["session_node_id"] = binding["session_node_id"]
    if binding.get("session_node_id") and not payload.get("dialogue_session_id"):
        payload["dialogue_session_id"] = binding["session_node_id"]
    if binding.get("turn_id") and not payload.get("turn_id"):
        payload["turn_id"] = binding["turn_id"]
    if binding.get("turn_id") and not payload.get("request_id"):
        payload["request_id"] = binding["turn_id"]
    interaction = payload.get("interaction") if isinstance(payload, dict) else None
    if not interaction and isinstance(data, dict):
        interaction = data.get("interaction")
        if interaction:
            payload["interaction"] = interaction
    if not text and isinstance(interaction, dict):
        text = interaction.get("detail") or interaction.get("title") or ""
    if not text and event_type == "IDE_APPROVAL_STATUS":
        text = (
            payload.get("action_summary")
            or payload.get("summary")
            or payload.get("risk_reason")
            or payload.get("tool_name")
            or "需要确认操作"
        )
    if not text:
        return
    try:
        event_id = get_interaction_store().append_event(
            conversation_id=conversation_id,
            turn_id=binding.get("turn_id"),
            event_type=event_type or "web_event",
            role=role,
            source="web_runtime" if role == "assistant" else "ide_bridge",
            text=text,
            payload=payload,
            workspace_path=binding.get("workspace_path"),
            project_id=binding.get("project_id"),
            task_graph_id=binding.get("task_graph_id"),
        )
        mirror_payload = dict(payload)
        mirror_payload.setdefault("source_event_id", event_id)
        if binding.get("task_graph_id") and not mirror_payload.get("task_graph_id"):
            mirror_payload["task_graph_id"] = binding["task_graph_id"]
        _mirror_visible_message_background(
            conversation_id=conversation_id,
            turn_id=binding.get("turn_id") or event_id,
            role=role,
            text=text,
            event_type=event_type or "web_event",
            source="web_runtime" if role == "assistant" else "ide_bridge",
            payload=mirror_payload,
        )
        try:
            from zulong.review.task_execution_extractor import maybe_finalize_task_execution_trace

            maybe_finalize_task_execution_trace(
                conversation_id=conversation_id,
                turn_id=binding.get("turn_id"),
                task_graph_id=binding.get("task_graph_id") or payload.get("task_graph_id"),
                event_type=event_type or "web_event",
            )
        except Exception as trace_exc:
            logger.debug(f"[WebChatRouter] TaskExecutionTrace 跳过: {event_type}: {trace_exc}")
    except Exception as exc:
        logger.debug(f"[WebChatRouter] 消息持久化跳过: {event_type}: {exc}")


def _event_role_for_persistence(event_type: str, message: dict) -> Optional[str]:
    payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
    if event_type == "STREAMING_RESPONSE":
        if message.get("complete") is not True and payload.get("complete") is not True:
            return None
        return "assistant"
    if event_type in ("CHAT_RESPONSE", "DISPLAY_TEXT", "TASK_COMPLETE", "FC_DONE"):
        return "assistant"
    if event_type in (
        "IDE_TERMINAL_STATUS",
        "IDE_APPROVAL_STATUS",
        "IDE_DIFF_STATUS",
        "IDE_CHECKPOINT_STATUS",
        "IDE_FILE_CHANGED",
        "IDE_TOOL_REQUEST",
        "IDE_TOOL_RESULT",
        "IDE_TOOL_EXEC",
        "THINKING_STEP",
    ):
        return "tool"
    return None


def _failure_response_text(reason: str) -> str:
    return f"系统当前出问题了，{reason}，因此无法正常回复。"


def _compact_dialogue_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value or ""))


def _memory_node_id(node: Any) -> str:
    if node is None:
        return ""
    if isinstance(node, dict):
        return str(node.get("node_id") or node.get("id") or "")
    return str(getattr(node, "node_id", "") or getattr(node, "id", "") or "")


def _memory_node_metadata(node: Any) -> Dict[str, Any]:
    if node is None:
        return {}
    if isinstance(node, dict):
        meta = node.get("metadata") or {}
    else:
        meta = getattr(node, "metadata", {}) or {}
    return dict(meta) if isinstance(meta, dict) else {}


def _memory_node_type_value(node: Any) -> str:
    if node is None:
        return ""
    if isinstance(node, dict):
        raw = node.get("node_type") or node.get("type") or ""
    else:
        raw = getattr(node, "node_type", "") or getattr(node, "type", "")
    return str(getattr(raw, "value", raw) or "").lower()


def _is_dialogue_session_node_id(node_id: str) -> bool:
    node_id = str(node_id or "")
    return node_id.startswith("dialogue:session_") and "/" not in node_id


def _is_dialogue_session_node(node: Any) -> bool:
    node_id = _memory_node_id(node)
    meta = _memory_node_metadata(node)
    node_type = _memory_node_type_value(node)
    return (
        _is_dialogue_session_node_id(node_id)
        and (not node_type or node_type == "dialogue")
        and meta.get("sub_type") in {"session", None, ""}
    )


def _normalize_dialogue_title(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\s*\[记忆\]\s*$", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _looks_corrupt_dialogue_title(value: Any) -> bool:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return False
    question_count = text.count("?")
    return question_count >= 5 and (question_count / max(len(text), 1)) > 0.25


def _safe_dialogue_title(value: Any, fallback: str = "对话记录") -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text or _looks_corrupt_dialogue_title(text):
        text = re.sub(r"\s+", " ", str(fallback or "")).strip()
    if not text or _looks_corrupt_dialogue_title(text):
        return "对话记录"
    return text


def _session_candidate_text(session: Dict[str, Any]) -> str:
    return _normalize_dialogue_title(
        session.get("title")
        or session.get("preview")
        or session.get("topic_summary")
        or ""
    )


def _graph_session_has_dialogue_round(mg: Any, session_id: str) -> bool:
    if not mg or not session_id:
        return False
    try:
        if not mg.has_node(session_id):
            return False
        for child in (mg.get_children(session_id) or []):
            if child and _memory_node_metadata(child).get("sub_type") == "round":
                return True
    except Exception:
        return False
    return False


def _safe_has_memory_node(mg: Any, node_id: str) -> bool:
    if not mg or not node_id:
        return False
    try:
        return bool(mg.has_node(node_id))
    except Exception:
        return False


def _safe_get_memory_node(mg: Any, node_id: str) -> Any:
    if not mg or not node_id:
        return None
    try:
        return mg.get_node(node_id)
    except Exception:
        return None


def _resolve_dialogue_session_node_id(mg: Any, session_id: str) -> str:
    raw = str(session_id or "").strip()
    if not raw:
        return ""

    candidates: List[str] = []

    def add(value: Optional[str]) -> None:
        value = str(value or "").strip()
        if value and value not in candidates:
            candidates.append(value)

    add(raw)
    try:
        store = get_interaction_store()
        conv = store.get_conversation(raw)
        if conv:
            add(conv.get("session_node_id"))
        by_node = store.find_conversation_by_session_node(raw)
        if by_node:
            add(by_node.get("session_node_id"))
            add(f"dialogue:session_{_compact_dialogue_id(by_node.get('conversation_id') or '')}")
    except Exception:
        pass
    if raw.startswith("dialogue:session_"):
        add(raw)
    else:
        add(f"dialogue:session_{_compact_dialogue_id(raw)}")

    for candidate in candidates:
        if _safe_has_memory_node(mg, candidate):
            node = _safe_get_memory_node(mg, candidate)
            if _is_dialogue_session_node(node) or _is_dialogue_session_node_id(candidate):
                return candidate
    for candidate in candidates:
        if _is_dialogue_session_node_id(candidate):
            return candidate
    return candidates[0] if candidates else raw


def _memory_children_ids(mg: Any, node_id: str) -> List[str]:
    if not mg or not node_id:
        return []
    try:
        return [
            _memory_node_id(child)
            for child in (mg.get_children(node_id) or [])
            if _memory_node_id(child)
        ]
    except Exception:
        return []
    return []


def _collect_hierarchy_node_ids(mg: Any, root_id: str) -> List[str]:
    if not mg or not root_id or not _safe_has_memory_node(mg, root_id):
        return []
    ordered: List[str] = []
    seen: Set[str] = set()
    queue: List[str] = [root_id]
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        ordered.append(current)
        for child_id in _memory_children_ids(mg, current):
            if child_id not in seen:
                queue.append(child_id)
    return ordered


def _save_memory_graph_if_possible(mg: Any) -> None:
    try:
        if hasattr(mg, "save_all"):
            mg.save_all()
    except Exception:
        pass


def _publish_memory_nodes_removed(deleted_node_ids: List[str], *, source: str) -> None:
    if not deleted_node_ids:
        return
    try:
        from zulong.core.event_bus import event_bus
        from zulong.core.types import EventPriority, EventType, ZulongEvent

        event_bus.publish(ZulongEvent(
            type=EventType.MEMORY_GRAPH_UPDATED,
            priority=EventPriority.LOW,
            source=source,
            payload={
                "update_type": "delta",
                "ts": time.time(),
                "nodes": [],
                "edges": [],
                "changes": [
                    {"action": "remove_node", "data": {"id": node_id}}
                    for node_id in deleted_node_ids
                ],
                "changed_node_ids": list(deleted_node_ids),
                "deleted_node_ids": list(deleted_node_ids),
                "stats": {
                    "transport": "delta",
                    "changed_nodes": len(deleted_node_ids),
                    "changed_edges": 0,
                },
            },
        ))
    except Exception as exc:
        logger.debug(f"[WebChatRouter] MemoryGraph 删除事件发布跳过: {exc}")


def _cleanup_dialogue_session_indexes(session_id: str, session_node_id: str = "") -> List[str]:
    removed: List[str] = []
    candidates: List[str] = []

    def add(value: Optional[str]) -> None:
        value = str(value or "").strip()
        if value and value not in candidates:
            candidates.append(value)

    add(session_id)
    add(session_node_id)
    if session_node_id and _is_dialogue_session_node_id(session_node_id):
        add(session_node_id.replace("dialogue:session_", "", 1))
    try:
        store = get_interaction_store()
        for value in list(candidates):
            conv = store.get_conversation(value)
            if conv:
                add(conv.get("conversation_id"))
                add(conv.get("session_node_id"))
            by_node = store.find_conversation_by_session_node(value)
            if by_node:
                add(by_node.get("conversation_id"))
                add(by_node.get("session_node_id"))
        for value in candidates:
            conv = store.get_conversation(value)
            if conv and store.delete_conversation(value):
                removed.append(value)
    except Exception as exc:
        logger.debug(f"[WebChatRouter] 会话索引清理跳过: {exc}")
    return removed


def _cleanup_runtime_session(session_id: str) -> bool:
    cleared = False
    try:
        from zulong.ide.ide_server import get_session_store
        store = get_session_store()
        store.delete(session_id)
    except Exception:
        pass
    try:
        from zulong.ide.ide_server import _sessions
        if session_id in _sessions:
            sess = _sessions[session_id]
            if sess.fc_task and not sess.fc_task.done():
                sess.cancel_event.set()
                sess.fc_task.cancel()
            _sessions.pop(session_id, None)
            cleared = True
    except Exception:
        pass
    return cleared


def _remove_memory_nodes(mg: Any, node_ids: List[str]) -> List[Dict[str, Any]]:
    deleted: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for node_id in reversed(node_ids):
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        node = _safe_get_memory_node(mg, node_id)
        if node is None:
            continue
        node_info = {
            "node_id": node_id,
            "type": _memory_node_type_value(node),
            "label": getattr(node, "label", "") if not isinstance(node, dict) else node.get("label", ""),
        }
        try:
            if mg.remove_node(node_id):
                deleted.append(node_info)
        except Exception as exc:
            logger.debug(f"[WebChatRouter] 删除 MemoryGraph 节点失败 {node_id}: {exc}")
    deleted.reverse()
    return deleted


def _delete_dialogue_session_cascade(session_id: str, *, cascade: bool = True) -> Dict[str, Any]:
    mg = _get_active_memory_graph()
    session_node_id = _resolve_dialogue_session_node_id(mg, session_id)
    nodes_to_remove: List[str] = []
    deleted: List[Dict[str, Any]] = []
    if cascade and mg and _safe_has_memory_node(mg, session_node_id):
        nodes_to_remove = _collect_hierarchy_node_ids(mg, session_node_id)
        deleted = _remove_memory_nodes(mg, nodes_to_remove)
        _save_memory_graph_if_possible(mg)

    deleted_ids = [item["node_id"] for item in deleted]
    removed_sessions = _cleanup_dialogue_session_indexes(session_id, session_node_id)
    session_ids = []
    for value in (session_id, session_node_id, *removed_sessions):
        value = str(value or "").strip()
        if value and value not in session_ids:
            session_ids.append(value)
    ide_session_cleared = _cleanup_runtime_session(session_id)
    if session_node_id and session_node_id != session_id:
        _cleanup_runtime_session(session_node_id)

    _publish_memory_nodes_removed(deleted_ids, source="WebChatRouter/DeleteSession")
    return {
        "session_id": session_id,
        "session_node_id": session_node_id,
        "session_ids": session_ids,
        "deleted": True,
        "mg_nodes_deleted": len(deleted),
        "nodes_deleted": len(deleted),
        "deleted_node_ids": deleted_ids,
        "ide_session_cleared": ide_session_cleared,
    }


def _message_event_matches_node(event: Optional[Dict[str, Any]], node_id: str, meta: Dict[str, Any]) -> bool:
    if not event:
        return False
    payload = meta.get("payload") if isinstance(meta.get("payload"), dict) else {}
    event_id = str(event.get("event_id") or "")
    if event_id and event_id in {
        str(meta.get("source_event_id") or ""),
        str(payload.get("source_event_id") or ""),
        str(payload.get("event_id") or ""),
        str(payload.get("id") or ""),
    }:
        return True
    conversation_id = str(event.get("conversation_id") or "")
    turn_id = str(event.get("turn_id") or "")
    role = str(event.get("role") or "")
    event_type = str(event.get("event_type") or "")
    return bool(
        conversation_id
        and turn_id
        and meta.get("conversation_id") == conversation_id
        and meta.get("request_id") == turn_id
        and (not role or meta.get("role") == role)
        and (not event_type or meta.get("event_type") == event_type)
    )


def _resolve_message_memory_node_ids(
    mg: Any,
    session_id: str,
    message_id: str,
    event: Optional[Dict[str, Any]] = None,
) -> List[str]:
    message_id = str(message_id or "").strip()
    if not mg or not message_id:
        return []
    candidates: List[str] = []

    def add(value: Optional[str]) -> None:
        value = str(value or "").strip()
        if value and value not in candidates and _safe_has_memory_node(mg, value):
            candidates.append(value)

    add(message_id)
    session_node_id = _resolve_dialogue_session_node_id(
        mg,
        session_id or (event or {}).get("conversation_id") or "",
    )
    if event:
        turn_id = str(event.get("turn_id") or "")
        role = str(event.get("role") or "")
        event_type = str(event.get("event_type") or "")
        if session_node_id and turn_id:
            round_id = f"{session_node_id}/round_{_compact_dialogue_id(turn_id)}"
            if role == "user":
                add(round_id)
            add(f"{round_id}/{_compact_dialogue_id(role or 'message')}_{_compact_dialogue_id(event_type or 'message')}")

    try:
        from zulong.memory.memory_graph import NodeType
        node_types = [
            NodeType.DIALOGUE,
            NodeType.TOOL_CALL,
            NodeType.TOOL_RESULT,
            NodeType.APPROVAL,
        ]
    except Exception:
        node_types = ["dialogue", "tool_call", "tool_result", "approval"]

    for node_type in node_types:
        try:
            nodes = mg.get_nodes_by_type(node_type) or []
        except Exception:
            continue
        for node in nodes:
            node_id = _memory_node_id(node)
            if not node_id or node_id in candidates:
                continue
            meta = _memory_node_metadata(node)
            payload = meta.get("payload") if isinstance(meta.get("payload"), dict) else {}
            if (
                node_id == message_id
                or meta.get("full_path") == message_id
                or message_id in {
                    str(meta.get("source_event_id") or ""),
                    str(payload.get("source_event_id") or ""),
                    str(payload.get("event_id") or ""),
                    str(payload.get("id") or ""),
                }
                or _message_event_matches_node(event, node_id, meta)
            ):
                candidates.append(node_id)

    if event and str(event.get("role") or "") == "user":
        round_candidates = [
            node_id for node_id in candidates
            if _memory_node_metadata(_safe_get_memory_node(mg, node_id)).get("sub_type") == "round"
        ]
        if round_candidates:
            return round_candidates
    return candidates


def _delete_dialogue_message_cascade(session_id: str, message_id: str) -> Dict[str, Any]:
    store = get_interaction_store()
    event = None
    try:
        event = store.get_event(message_id)
    except Exception:
        event = None
    mg = _get_active_memory_graph()
    target_ids = _resolve_message_memory_node_ids(mg, session_id, message_id, event)
    nodes_to_remove: List[str] = []
    for target_id in target_ids:
        for node_id in _collect_hierarchy_node_ids(mg, target_id):
            if node_id not in nodes_to_remove:
                nodes_to_remove.append(node_id)

    deleted = _remove_memory_nodes(mg, nodes_to_remove) if mg else []
    if deleted:
        _save_memory_graph_if_possible(mg)
        _publish_memory_nodes_removed(
            [item["node_id"] for item in deleted],
            source="WebChatRouter/DeleteMessage",
        )
    event_deleted = False
    try:
        event_deleted = store.delete_event(message_id)
    except Exception:
        event_deleted = False
    return {
        "status": "ok",
        "session_id": session_id,
        "message_id": message_id,
        "removed": len(deleted),
        "mg_nodes_deleted": len(deleted),
        "deleted_node_ids": [item["node_id"] for item in deleted],
        "event_deleted": event_deleted,
    }


def _collect_dialogue_sessions(limit: int = 200) -> Dict[str, Any]:
    store = get_conversation_orchestrator().store
    store_sessions = []
    mg = None
    try:
        if _launcher_ready_for_memory_backfill():
            mg = _get_active_memory_graph()
    except Exception:
        mg = None

    try:
        for conv in store.list_conversations(limit=limit):
            conv_id = conv.get("conversation_id") or ""
            session_node_id = conv.get("session_node_id")
            inferred_session_node_id = session_node_id
            if mg and not inferred_session_node_id and conv_id:
                try:
                    if hasattr(mg, "get_session_node_id_for_conversation"):
                        inferred_session_node_id = mg.get_session_node_id_for_conversation(conv_id)
                except Exception:
                    inferred_session_node_id = None
                if not inferred_session_node_id:
                    inferred_session_node_id = (
                        conv_id if _is_dialogue_session_node_id(conv_id)
                        else f"dialogue:session_{_compact_dialogue_id(conv_id)}"
                    )
            if not inferred_session_node_id and conv_id:
                inferred_session_node_id = (
                    conv_id if _is_dialogue_session_node_id(conv_id)
                    else f"dialogue:session_{_compact_dialogue_id(conv_id)}"
                )
            # InteractionStore 是 Web 窗口身份账本。MemoryGraph 节点可能异步创建、
            # 延迟加载或被修复流程重建；列表阶段不能因为图节点暂缺删除窗口记录。
            store_sessions.append({
                "id": conv_id,
                "title": _safe_dialogue_title(conv.get("title"), "对话记录"),
                "created_at": conv.get("created_at") or 0,
                "last_active_at": conv.get("last_active_at") or conv.get("created_at") or 0,
                "preview": "",
                "round_count": 0,
                "source": conv.get("source") or "interaction_store",
                "workspace_path": conv.get("workspace_path"),
                "cwd": conv.get("workspace_path"),
                "project_id": conv.get("project_id"),
                "task_graph_id": conv.get("task_graph_id"),
                "dialogue_session_id": inferred_session_node_id,
                "session_node_id": inferred_session_node_id,
            })
    except Exception as store_err:
        logger.debug(f"[WebChatRouter] interaction store session list skipped: {store_err}")

    # 会话窗口列表以 InteractionStore 的窗口账本为准。
    # MemoryGraph session 是记忆地址，不能反向升级成聊天窗口，否则 Web 窗口 ID
    # 和图谱 session ID 会分裂，产生空的 [记忆] 会话。
    sessions = store_sessions

    active = None
    try:
        active = store.find_active_conversation(max_age_seconds=86400)
    except Exception:
        active = None

    return {
        "activeSessionId": (
            active.get("conversation_id")
            if isinstance(active, dict)
            else (sessions[0].get("id") if sessions else None)
        ),
        "sessions": sessions,
    }


def _get_active_memory_graph(*, allow_fallback: bool = False):
    """Return the MemoryGraph instance owned by LauncherApp when available.

    The launcher module manager is the owner of the running memory graph. Web
    routes must not create a fallback graph while the launcher is still
    selecting/launching, because frontend dashboard probes can otherwise load
    the full sharded MemoryGraph before the startup sequence reaches that
    module.
    """
    try:
        from zulong.launcher import app as launcher_app_module

        launcher = getattr(launcher_app_module, "_app_instance", None)
        if launcher is not None:
            mg = getattr(launcher, "manager", None)
            if mg is not None:
                graph = mg.context.get("memory_graph")
                if graph is not None:
                    return graph
            return None
    except Exception:
        pass
    if not allow_fallback:
        return None
    try:
        from zulong.memory.memory_graph import get_memory_graph

        return get_memory_graph()
    except Exception:
        return None

# EventBus 是否已订阅
_eventbus_subscribed = False


# ── 公共接口 ──────────────────────────────────────────

_heartbeat_task: Optional[asyncio.Task] = None

async def _ws_heartbeat_loop():
    """🔧 定期向所有 /ws 客户端发送 THINKING_STEP 心跳
    防止 Web 前端 120 秒看门狗在长 FC 循环中超时。
    """
    while True:
        await asyncio.sleep(30)
        if _ws_clients:
            try:
                status = _task_status_snapshot()
                state = str(status.get("state") or "").lower()
                if (
                    state in {"idle", "cancelled", "canceled", "completed", "succeeded", "failed"}
                    and time.time() - float(status.get("last_state_change_at") or 0) > 10
                ):
                    continue
                _broadcast_task_execution_status(status)
                if (
                    status.get("interaction") is not None
                    and state not in {"idle", "cancelled", "canceled", "completed", "succeeded", "failed"}
                ):
                    await _broadcast({
                        "type": "THINKING_STEP",
                        "step_type": "heartbeat",
                        "request_id": status.get("request_id") or "",
                        "conversation_id": status.get("conversation_id") or "",
                        "session_id": status.get("session_id") or status.get("conversation_id") or "",
                        "workspace_path": status.get("workspace_path") or "",
                        "project_id": status.get("project_id") or "",
                        "task_graph_id": status.get("task_graph_id") or "",
                        "data": {
                            "message": status.get("message") or "连接正常。",
                            "state": status.get("state"),
                            "phase": status.get("phase"),
                            "elapsed_since_progress": status.get("elapsed_since_progress"),
                            "interaction": status.get("interaction"),
                        },
                        "timestamp": time.time(),
                        "_persisted": True,  # 跳过持久化
                    })
            except Exception:
                pass

def _start_ws_heartbeat():
    """启动 WebSocket 心跳任务（幂等）"""
    global _heartbeat_task
    if _heartbeat_task is None or _heartbeat_task.done():
        try:
            loop = asyncio.get_event_loop()
            _heartbeat_task = loop.create_task(_ws_heartbeat_loop())
            logger.info("[WebChatRouter] ✅ 已启动 WebSocket 心跳 (每30秒)")
        except RuntimeError as e:
            logger.warning(f"[WebChatRouter] 无法启动心跳: {e}")

def set_launch_mode(mode: str) -> None:
    """设置运行模式并初始化对应的事件订阅"""
    global _launch_mode
    _launch_mode = mode
    logger.info(f"[WebChatRouter] 运行模式: {mode}")

    if mode == "full":
        _subscribe_eventbus()
    else:
        # IDE 模式同样走 EventBus 主链，因此也要订阅主会话下行事件。
        _subscribe_eventbus_lite()
    
    # 启动 /ws 心跳，防止 Web 前端 120s 看门狗在长 FC 循环中超时
    _start_ws_heartbeat()


def _is_full_mode() -> bool:
    return _launch_mode == "full"


# ── EventBus 订阅（Full 模式） ────────────────────────

_eventbus_lite_subscribed = False


def _subscribe_eventbus_lite() -> None:
    """IDE 模式精简订阅 — 订阅主会话链下行事件与图谱事件

    为了与 TSD 的 USER -> L1-B -> L2 主链保持一致，IDE 模式下的 /ws
    聊天也必须消费 EventBus 下行事件，而不能直接旁路调用 InferenceEngine。
    """
    global _eventbus_lite_subscribed
    if _eventbus_lite_subscribed:
        return
    try:
        from zulong.core.event_bus import event_bus
        from zulong.core.types import EventType

        event_bus.subscribe(EventType.L2_OUTPUT, _on_l2_output, "WebChatRouter")
        event_bus.subscribe(EventType.L2_OUTPUT_STREAM, _on_l2_output_stream, "WebChatRouter")
        event_bus.subscribe(EventType.L2_THINKING_STEP, _on_l2_thinking_step, "WebChatRouter")
        event_bus.subscribe(EventType.MEMORY_GRAPH_UPDATED, _on_memory_graph_updated, "WebChatRouter")
        event_bus.subscribe(EventType.ACTION_SPEAK, _on_action_speak, "WebChatRouter")
        _eventbus_lite_subscribed = True
        logger.info(
            "[WebChatRouter] IDE 模式: 已订阅 L2_OUTPUT/L2_OUTPUT_STREAM/"
            "L2_THINKING_STEP/MEMORY_GRAPH_UPDATED/ACTION_SPEAK"
        )
    except Exception as e:
        logger.error(f"[WebChatRouter] IDE EventBus 精简订阅失败: {e}")

def _subscribe_eventbus() -> None:
    """订阅 EventBus 下行事件 — 将 L2 输出转发到 /ws 客户端"""
    global _eventbus_subscribed
    if _eventbus_subscribed:
        return
    try:
        from zulong.core.event_bus import event_bus
        from zulong.core.types import EventType

        event_bus.subscribe(EventType.L2_OUTPUT, _on_l2_output, "WebChatRouter")
        event_bus.subscribe(EventType.L2_OUTPUT_STREAM, _on_l2_output_stream, "WebChatRouter")
        event_bus.subscribe(EventType.L2_THINKING_STEP, _on_l2_thinking_step, "WebChatRouter")
        event_bus.subscribe(EventType.MEMORY_GRAPH_UPDATED, _on_memory_graph_updated, "WebChatRouter")
        event_bus.subscribe(EventType.ACTION_SPEAK, _on_action_speak, "WebChatRouter")
        event_bus.subscribe(EventType.PROJECT_CREATED, _on_project_created, "WebChatRouter")
        _eventbus_subscribed = True
        logger.info("[WebChatRouter] 已订阅 EventBus 下行事件")
    except Exception as e:
        logger.error(f"[WebChatRouter] EventBus 订阅失败: {e}")


def _schedule_broadcast(message: dict) -> None:
    """从非 asyncio 线程安全地调度广播到所有 /ws 客户端"""
    loop = _event_loop
    msg_type = message.get("type", "?")
    if not is_public_payload(message):
        logger.debug(
            "[WebChatRouter] _schedule_broadcast 跳过内部控制消息: type=%s",
            msg_type,
        )
        return
    noisy_key = str(msg_type) if msg_type in _NOISY_BROADCAST_TYPES else ""
    if noisy_key:
        now = time.monotonic()
        min_interval = _NOISY_MIN_INTERVALS.get(noisy_key, 0.0)
        with _noisy_broadcast_lock:
            if noisy_key in _noisy_broadcast_pending:
                logger.info(
                    "[WebChatRouter] _schedule_broadcast: 合并低优先级过程事件 type=%s",
                    msg_type,
                )
                return
            last_sent = _noisy_broadcast_last_sent.get(noisy_key, 0.0)
            if min_interval > 0 and now - last_sent < min_interval:
                logger.debug(
                    "[WebChatRouter] _schedule_broadcast: 节流低优先级过程事件 type=%s",
                    msg_type,
                )
                return
            _noisy_broadcast_pending.add(noisy_key)
            _noisy_broadcast_last_sent[noisy_key] = now

    if loop and loop.is_running():
        logger.info(f"[WebChatRouter] _schedule_broadcast: type={msg_type}, loop_running=True, ws_clients={len(_ws_clients)}")
        future = asyncio.run_coroutine_threadsafe(_broadcast(message), loop)
        # 捕获异步广播的异常
        def _on_done(f):
            if noisy_key:
                with _noisy_broadcast_lock:
                    _noisy_broadcast_pending.discard(noisy_key)
            try:
                exc = f.exception()
            except asyncio.CancelledError:
                logger.warning(f"[WebChatRouter] _broadcast 已取消: type={msg_type}")
                return
            if exc:
                logger.error(f"[WebChatRouter] _broadcast 异常: {exc}")
            else:
                logger.info(f"[WebChatRouter] _broadcast 完成: type={msg_type}")
        future.add_done_callback(_on_done)
    else:
        if noisy_key:
            with _noisy_broadcast_lock:
                _noisy_broadcast_pending.discard(noisy_key)
        logger.warning(f"[WebChatRouter] _schedule_broadcast 跳过: type={msg_type}, loop={loop}, loop_running={loop.is_running() if loop else 'N/A'}")


async def _broadcast(message: dict) -> None:
    """向所有 /ws 客户端广播消息，根据各自协议版本自动选择格式"""
    msg_type = message.get("type", "?")
    if not is_public_payload(message):
        logger.debug("[WebChatRouter] 跳过内部控制消息广播: type=%s", msg_type)
        return
    _persist_web_visible_message(message)
    if not _ws_clients:
        logger.warning(f"[WebChatRouter] _broadcast: type={msg_type}, 无客户端连接")
        return
    logger.info(f"[WebChatRouter] _broadcast: type={msg_type}, 发送给 {len(_ws_clients)} 个客户端")
    dead: Set[WebSocket] = set()
    clients = list(_ws_clients)
    tasks = [
        asyncio.create_task(_send_broadcast_to_client(ws, message, str(msg_type)))
        for ws in clients
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for ws, result in zip(clients, results):
        if result is True:
            logger.info(f"[WebChatRouter] _broadcast: 成功发送 {msg_type}")
            continue
        if isinstance(result, Exception):
            logger.error(f"[WebChatRouter] _broadcast: 发送失败 {msg_type}: {result}")
        dead.add(ws)
    _ws_clients.difference_update(dead)


def _broadcast_timeout_for_type(msg_type: str) -> float:
    if msg_type in {"CHAT_RESPONSE", "TASK_EXECUTION_STATUS", "CHAT_STREAM"}:
        return 15.0
    if msg_type in {"THINKING_STEP", "MEMORY_GRAPH_UPDATE"}:
        return 0.75
    if msg_type in {"TASK_GRAPH_UPDATE", "CODE_ANCHOR_UPDATE"}:
        return 2.0
    return 3.0


async def _send_broadcast_to_client(ws: WebSocket, message: dict, msg_type: str) -> bool:
    if msg_type in {"MEMORY_GRAPH_UPDATE", "THINKING_STEP"}:
        lock = _ws_send_locks.get(id(ws))
        if lock and lock.locked():
            logger.info(
                "[WebChatRouter] _broadcast: 跳过繁忙连接的过程事件，避免阻塞聊天响应 type=%s",
                msg_type,
            )
            return True

    timeout = _broadcast_timeout_for_type(msg_type)
    try:
        if msg_type == "MEMORY_GRAPH_UPDATE" and (message.get("nodes") or message.get("edges")):
            await asyncio.wait_for(_send_memory_graph_payload(ws, message), timeout=timeout)
            return True
        return await asyncio.wait_for(_send_to_ws(ws, message, persist=False), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "[WebChatRouter] _broadcast: 发送超时，保留客户端连接 type=%s",
            msg_type,
        )
        return True
    except Exception:
        _forget_ws_connection(ws)
        raise


# ── EventBus 回调（从分发线程调用，非 asyncio） ───────

def _on_l2_output(event) -> None:
    payload = event.payload or {}
    if not is_public_payload(payload):
        logger.debug("[WebChatRouter] 丢弃内部 L2_OUTPUT 控制消息")
        return
    text = payload.get("display_text") or payload.get("text", "")
    speech_text = payload.get("speech_text") or ""
    request_id = payload.get("request_id")
    binding = _resolve_conversation_binding({
        "request_id": request_id,
        "turn_id": payload.get("turn_id"),
        "conversation_id": payload.get("conversation_id"),
        "session_id": payload.get("session_id"),
        "workspace_path": payload.get("workspace_path"),
        "project_id": payload.get("project_id"),
        "task_graph_id": payload.get("task_graph_id"),
        "payload": payload,
    })
    conversation_id = binding.get("conversation_id")
    logger.info(f"[WebChatRouter] _on_l2_output 被调用: text_len={len(text)}, request_id={request_id}, ws_clients={len(_ws_clients)}")
    if text:
        if _is_cancelled_turn(request_id, conversation_id, conversation_id):
            logger.info(
                "[WebChatRouter] 丢弃已取消请求的迟到 L2_OUTPUT: request_id=%s text_len=%s",
                request_id or "-",
                len(text),
            )
            return
        bound_task_graph_id = binding.get("task_graph_id")
        completion = (
            _active_task_completion_snapshot(bound_task_graph_id)
            if bound_task_graph_id
            else {"total": 0, "completed": 0, "uncompleted": 0, "blocked": 0, "uncompleted_labels": []}
        )
        incomplete_final = _looks_like_incomplete_final_text(text)
        has_uncompleted_work = bool(completion.get("total") and completion.get("uncompleted"))
        if incomplete_final or has_uncompleted_work:
            state = "blocked"
            phase = "forced_convergence" if incomplete_final else "incomplete_task_graph"
            if has_uncompleted_work:
                message = (
                    f"任务未完成：任务图进度 "
                    f"{completion.get('completed')}/{completion.get('total')}，"
                    f"剩余 {completion.get('uncompleted')} 项。"
                )
            elif completion.get("total"):
                message = (
                    f"任务图已完成 {completion.get('completed')}/{completion.get('total')}，"
                    "但最终回复仍是中断/受阻说明，不能标记为正常完成。"
                )
            else:
                message = "任务返回了中断/受阻说明，不能标记为完成。"
            item_label = message
            item_status = "blocked"
        else:
            state = "completed"
            phase = "completed"
            message = "任务已返回最终回复。"
            item_label = "生成最终回复"
            item_status = "completed"
        chat_message = {
            "type": "CHAT_RESPONSE",
            "text": text,
            "display_text": text,
            "speech_text": speech_text,
            "request_id": request_id,
            "session_id": conversation_id,
            "conversation_id": conversation_id,
            "workspace_path": binding.get("workspace_path"),
            "project_id": binding.get("project_id"),
            "task_graph_id": binding.get("task_graph_id"),
            "_persisted": True,
        }
        mark_public_payload(chat_message, CHANNEL_FINAL)
        _schedule_broadcast(chat_message)
        update_task_execution_status(
            state=state,
            phase=phase,
            message=message,
            request_id=request_id,
            conversation_id=conversation_id,
            session_id=conversation_id,
            workspace_path=binding.get("workspace_path"),
            project_id=binding.get("project_id"),
            task_graph_id=binding.get("task_graph_id"),
            tool_name="",
            awaiting_approval=False,
            progress_items=[{
                "label": item_label,
                "status": item_status,
                "source": "summary",
                "timestamp": time.time(),
            }],
        )
        if conversation_id:
            def _record_assistant_output() -> None:
                try:
                    decision = RouteDecision(
                        conversation_id=conversation_id,
                        turn_id=request_id or "",
                        text="",
                        workspace_path=binding.get("workspace_path"),
                        project_id=binding.get("project_id"),
                        task_graph_id=binding.get("task_graph_id"),
                    )
                    get_conversation_orchestrator().record_assistant_text(
                        decision,
                        text,
                        payload={
                            "raw_markdown": text,
                            "display_text": text,
                            "speech_text": speech_text,
                        },
                    )
                except Exception as record_exc:
                    logger.debug(f"[WebChatRouter] assistant 输出后台持久化跳过: {record_exc}")

            threading.Thread(
                target=_record_assistant_output,
                name="zulong-web-assistant-persist",
                daemon=True,
            ).start()


def _on_l2_output_stream(event) -> None:
    payload = event.payload or {}
    if not is_public_payload(payload):
        logger.debug("[WebChatRouter] 丢弃内部 L2_OUTPUT_STREAM 控制消息")
        return
    text = payload.get("text", "")
    chunk = payload.get("chunk", "")
    request_id = payload.get("request_id")
    binding = _resolve_conversation_binding({
        "request_id": request_id,
        "turn_id": payload.get("turn_id"),
        "conversation_id": payload.get("conversation_id"),
        "session_id": payload.get("session_id"),
        "workspace_path": payload.get("workspace_path"),
        "project_id": payload.get("project_id"),
        "task_graph_id": payload.get("task_graph_id"),
        "payload": payload,
    })
    conversation_id = binding.get("conversation_id")
    if chunk or text:
        if _is_cancelled_turn(request_id, conversation_id, conversation_id):
            logger.info(
                "[WebChatRouter] 丢弃已取消请求的迟到流式输出: request_id=%s",
                request_id or "-",
            )
            return
        update_task_execution_status(
            state="running",
            phase="streaming",
            message="任务正在输出结果。",
            request_id=request_id,
            conversation_id=conversation_id,
            session_id=conversation_id,
            workspace_path=binding.get("workspace_path"),
            project_id=binding.get("project_id"),
            task_graph_id=binding.get("task_graph_id"),
            broadcast=False,
        )
        stream_message = {
            "type": "STREAMING_RESPONSE",
            "text": text,
            "chunk": chunk,
            "request_id": request_id,
            "session_id": conversation_id,
            "conversation_id": conversation_id,
            "workspace_path": binding.get("workspace_path"),
            "project_id": binding.get("project_id"),
            "task_graph_id": binding.get("task_graph_id"),
        }
        mark_public_payload(stream_message, CHANNEL_FINAL)
        _schedule_broadcast(stream_message)


def _on_l2_thinking_step(event) -> None:
    payload = event.payload
    if payload and not is_public_payload(payload):
        logger.debug("[WebChatRouter] 丢弃内部 L2_THINKING_STEP 控制消息")
        return
    if payload:
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        step_type = str(payload.get("step_type") or "")
        if step_type == "graph.task_update" and isinstance(data.get("graph"), dict):
            graph_payload = {
                "type": "TASK_GRAPH_UPDATE",
                "request_id": payload.get("request_id"),
                "turn_id": payload.get("turn_id") or payload.get("request_id"),
                "session_id": payload.get("session_id") or payload.get("conversation_id"),
                "conversation_id": payload.get("conversation_id") or payload.get("session_id"),
                "workspace_path": payload.get("workspace_path"),
                "project_id": payload.get("project_id"),
                "task_graph_id": data.get("task_graph_id"),
                "graph_id": data.get("task_graph_id"),
                "graph": data.get("graph"),
                "payload": {
                    "request_id": payload.get("request_id"),
                    "turn_id": payload.get("turn_id") or payload.get("request_id"),
                    "session_id": payload.get("session_id") or payload.get("conversation_id"),
                    "conversation_id": payload.get("conversation_id") or payload.get("session_id"),
                    "workspace_path": payload.get("workspace_path"),
                    "project_id": payload.get("project_id"),
                    "task_graph_id": data.get("task_graph_id"),
                    "graph_id": data.get("task_graph_id"),
                    "graph": data.get("graph"),
                    "progress": data.get("progress"),
                    "update_type": data.get("update_type") or "pipeline",
                    "node_count": data.get("node_count"),
                    "edge_count": data.get("edge_count"),
                },
            }
            mark_public_payload(graph_payload, CHANNEL_LEDGER)
            _schedule_broadcast(graph_payload)
        elif step_type == "attention.update":
            attention_payload = {
                "type": "ATTENTION_UPDATE",
                "request_id": payload.get("request_id"),
                "turn_id": payload.get("turn_id") or payload.get("request_id"),
                "session_id": payload.get("session_id") or payload.get("conversation_id"),
                "conversation_id": payload.get("conversation_id") or payload.get("session_id"),
                "workspace_path": payload.get("workspace_path"),
                "project_id": payload.get("project_id"),
                "task_graph_id": data.get("task_graph_id"),
                "mode": data.get("mode"),
                "turn": data.get("turn"),
                "focus_node_id": data.get("focus_node_id"),
                "budget_usage": data.get("budget_usage"),
                "context_pressure": data.get("context_pressure"),
                "pressure_tier": data.get("pressure_tier"),
                "payload": {
                    "request_id": payload.get("request_id"),
                    "turn_id": payload.get("turn_id") or payload.get("request_id"),
                    "session_id": payload.get("session_id") or payload.get("conversation_id"),
                    "conversation_id": payload.get("conversation_id") or payload.get("session_id"),
                    "workspace_path": payload.get("workspace_path"),
                    "project_id": payload.get("project_id"),
                    "task_graph_id": data.get("task_graph_id"),
                    "mode": data.get("mode"),
                    "turn": data.get("turn"),
                    "focus_node_id": data.get("focus_node_id"),
                    "budget_usage": data.get("budget_usage"),
                    "context_pressure": data.get("context_pressure"),
                    "pressure_tier": data.get("pressure_tier"),
                    "progress": data.get("progress"),
                },
            }
            mark_public_payload(attention_payload, CHANNEL_STATUS)
            _schedule_broadcast(attention_payload)
        message = data.get("message") or payload.get("message") or payload.get("step_type") or "任务执行中。"
        interaction = data.get("interaction") if isinstance(data, dict) else None
        progress_items = interaction.get("progress_items") if isinstance(interaction, dict) else data.get("progress_items")
        if isinstance(interaction, dict) and is_public_payload({"interaction": interaction}):
            interaction_payload = {
                "type": "INTERACTION_EVENT",
                "request_id": payload.get("request_id"),
                "turn_id": payload.get("turn_id") or payload.get("request_id"),
                "session_id": payload.get("session_id") or payload.get("conversation_id"),
                "conversation_id": payload.get("conversation_id") or payload.get("session_id"),
                "workspace_path": payload.get("workspace_path"),
                "project_id": payload.get("project_id"),
                "task_graph_id": payload.get("task_graph_id") or data.get("task_graph_id"),
                "step_type": payload.get("step_type"),
                "interaction": interaction,
                "payload": {
                    "interaction": interaction,
                    "request_id": payload.get("request_id"),
                    "turn_id": payload.get("turn_id") or payload.get("request_id"),
                    "session_id": payload.get("session_id") or payload.get("conversation_id"),
                    "conversation_id": payload.get("conversation_id") or payload.get("session_id"),
                    "workspace_path": payload.get("workspace_path"),
                    "project_id": payload.get("project_id"),
                    "task_graph_id": payload.get("task_graph_id") or data.get("task_graph_id"),
                    "step_type": payload.get("step_type"),
                },
            }
            mark_public_payload(
                interaction_payload,
                str(interaction.get("channel") or CHANNEL_LEDGER),
                str(interaction.get("ux_visibility") or "main"),
            )
            _schedule_broadcast(interaction_payload)
        update_task_execution_status(
            state="running",
            phase=str(payload.get("step_type") or "thinking"),
            message=str(message),
            request_id=payload.get("request_id"),
            conversation_id=payload.get("conversation_id") or payload.get("session_id"),
            session_id=payload.get("session_id") or payload.get("conversation_id"),
            workspace_path=payload.get("workspace_path"),
            project_id=payload.get("project_id"),
            task_graph_id=payload.get("task_graph_id"),
            progress_items=progress_items if isinstance(progress_items, list) and progress_items else None,
            broadcast=False,
        )
        thinking_message = {"type": "THINKING_STEP", **payload}
        mark_public_payload(thinking_message, CHANNEL_LEDGER)
        _schedule_broadcast(thinking_message)


def _compact_memory_graph_update(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Keep background graph notifications light; full graph is request-driven."""
    payload = dict(payload or {})
    if payload.get("update_type") == "delta" or payload.get("changes"):
        nodes = payload.pop("nodes", None) or []
        edges = payload.pop("edges", None) or []
        payload.pop("execution_view", None)
        stats = dict(payload.get("stats") or {})
        stats.setdefault("transport", "delta")
        stats.setdefault("changed_nodes", len(payload.get("changed_node_ids") or []))
        stats.setdefault("changed_edges", len(payload.get("changed_edge_ids") or []))
        stats.setdefault("returned_nodes", len(nodes))
        stats.setdefault("returned_edges", len(edges))
        payload["stats"] = stats
        payload["nodes"] = []
        payload["edges"] = []
        payload["update_type"] = "delta"
        payload["summary_only"] = False
        payload.setdefault("ts", time.time())
        return payload

    nodes = payload.pop("nodes", None) or []
    edges = payload.pop("edges", None) or []
    payload.pop("execution_view", None)
    stats = dict(payload.get("stats") or {})
    stats.setdefault("total_nodes", len(nodes))
    stats.setdefault("total_edges", len(edges))
    stats["transport"] = "summary"
    payload["stats"] = stats
    payload["nodes"] = []
    payload["edges"] = []
    payload["update_type"] = "summary"
    payload["summary_only"] = True
    payload.setdefault("ts", time.time())
    return payload


def _on_memory_graph_updated(event) -> None:
    payload = event.payload
    if payload:
        _sync_deleted_dialogue_sessions_from_memory_update(payload)
        _schedule_broadcast({
            "type": "MEMORY_GRAPH_UPDATE",
            **_compact_memory_graph_update(payload),
        })


def _deleted_node_ids_from_memory_payload(payload: Dict[str, Any]) -> List[str]:
    deleted: List[str] = []
    for node_id in payload.get("deleted_node_ids") or []:
        value = str(node_id or "").strip()
        if value and value not in deleted:
            deleted.append(value)
    for change in payload.get("changes") or []:
        if not isinstance(change, dict) or change.get("action") != "remove_node":
            continue
        data = change.get("data") if isinstance(change.get("data"), dict) else {}
        value = str(data.get("id") or data.get("node_id") or "").strip()
        if value and value not in deleted:
            deleted.append(value)
    return deleted


def _sync_deleted_dialogue_sessions_from_memory_update(payload: Dict[str, Any]) -> None:
    deleted_ids = _deleted_node_ids_from_memory_payload(payload)
    if not deleted_ids:
        return
    for node_id in deleted_ids:
        if not _is_dialogue_session_node_id(node_id):
            continue
        removed_sessions = _cleanup_dialogue_session_indexes(node_id, node_id)
        session_ids = list(dict.fromkeys([node_id] + removed_sessions))
        _schedule_broadcast({
            "type": "SESSION_DELETED",
            "ts": time.time(),
            "session_id": removed_sessions[0] if removed_sessions else node_id,
            "session_node_id": node_id,
            "session_ids": session_ids,
            "nodes_deleted": 1,
            "deleted_node_ids": deleted_ids,
        })


def _on_action_speak(event) -> None:
    text = event.payload.get("text", "")
    if text:
        logger.info(
            "[WebChatRouter] ACTION_SPEAK 仅后端记录，不广播到前端: "
            "text_len=%s, voice_mode=%s, preview=%r",
            len(text),
            event.payload.get("voice_mode"),
            text[:120],
        )


def _on_project_created(event) -> None:
    """项目创建事件 → 通知 Web 前端"""
    payload = event.payload
    if payload:
        conversation_id = payload.get("conversation_id") or payload.get("session_id")
        workspace_path = payload.get("path", "")
        task_graph_id = payload.get("task_graph_id", "")
        if conversation_id and workspace_path:
            try:
                get_interaction_store().upsert_conversation(
                    conversation_id,
                    source="workspace",
                    workspace_path=workspace_path,
                    project_id=payload.get("project_id"),
                    task_graph_id=task_graph_id,
                    active=True,
                )
            except Exception:
                pass
        _schedule_broadcast({
            "type": "PROJECT_CREATED",
            "project_id": payload.get("project_id", ""),
            "name": payload.get("name", ""),
            "path": payload.get("path", ""),
            "workspace_path": payload.get("path", ""),
            "task_graph_id": payload.get("task_graph_id", ""),
            "status": payload.get("status", ""),
        })


# ── 消息处理 ──────────────────────────────────────────

async def _handle_chat_message(ws: WebSocket, data: dict) -> None:
    """处理 CHAT_MESSAGE: Web 总入口，按需自动激活编程能力"""
    text = data.get("text", "")
    referenced_nodes = data.get("referenced_nodes", [])

    if not text:
        return

    orchestrator = get_conversation_orchestrator()
    decision = orchestrator.prepare_turn(data, source=data.get("source") or "web_chat")
    data["session_id"] = decision.conversation_id
    data["conversation_id"] = decision.conversation_id
    data["request_id"] = decision.turn_id
    data["turn_id"] = decision.turn_id
    binding_policy = getattr(decision, "task_graph_binding_policy", "") or "reference_only"
    clear_active_graph_for_turn = binding_policy == "clear_for_new_unbound_turn"
    if clear_active_graph_for_turn:
        data["clear_active_graph"] = True
        data["active_graph_policy"] = "clear"
        try:
            from zulong.tools.task_tools import set_active_task_graph

            set_active_task_graph(None, None)
            logger.info(
                "[WebChatRouter] 未绑定任务图的会话消息，发送前清空 active graph: session=%s request=%s",
                decision.conversation_id,
                decision.turn_id,
            )
        except Exception as exc:
            logger.debug(f"[WebChatRouter] unbound turn active graph clear skipped: {exc}")
    else:
        data["active_graph_policy"] = binding_policy
        data["task_graph_binding"] = {
            "task_graph_id": decision.task_graph_id or getattr(decision, "last_task_graph_id", None),
            "policy": binding_policy,
            "reason": getattr(decision, "binding_reason", ""),
            "active_source": "session_restore" if decision.task_graph_id else "none",
        }

    await _send_turn_accepted(ws, decision)
    update_task_execution_status(
        state="running",
        phase="accepted",
        message="任务已接收，正在准备执行。",
        request_id=decision.turn_id,
        conversation_id=decision.conversation_id,
        session_id=decision.conversation_id,
        workspace_path=decision.workspace_path,
        project_id=decision.project_id,
        task_graph_id=decision.task_graph_id or "",
        tool_name="",
        awaiting_approval=False,
    )

    logger.info(
        "[WebChatRouter] 消息: text=%s, mode=%s",
        text[:80],
        _launch_mode,
    )

    await _chat_via_eventbus(
        ws,
        text,
        decision.turn_id,
        decision.conversation_id,
        referenced_nodes,
        decision=decision,
        clear_active_graph=clear_active_graph_for_turn,
    )

async def _send_turn_accepted(ws: WebSocket, decision: RouteDecision) -> None:
    payload = decision.to_payload()
    msg = {
        "type": "TURN_ACCEPTED",
        "request_id": decision.turn_id,
        "session_id": decision.conversation_id,
        "conversation_id": decision.conversation_id,
        "payload": payload,
    }
    await _send_to_ws(ws, msg)


async def _chat_via_eventbus(ws, text, request_id, session_id, referenced_nodes, decision=None, clear_active_graph=False):
    """Full 模式: 发布 USER_TEXT 到核心 EventBus → L1-B → L2"""
    try:
        from zulong.core.event_bus import event_bus
        from zulong.core.types import EventType, EventPriority, ZulongEvent

        payload = {"text": text, "confidence": 1.0}
        explicit_workspace = _extract_explicit_workspace_path(text)
        if session_id:
            payload["session_id"] = session_id
        if request_id:
            payload["request_id"] = request_id
        if referenced_nodes:
            payload["referenced_nodes"] = referenced_nodes
        if decision:
            payload.update({
                "conversation_id": decision.conversation_id,
                "session_id": decision.conversation_id,
                "session_node_id": decision.session_node_id,
                "dialogue_session_id": decision.session_node_id,
                "turn_id": decision.turn_id,
                "workspace_path": decision.workspace_path,
                "project_id": decision.project_id,
                "task_graph_id": decision.task_graph_id,
                "referenced_task_graph_id": decision.referenced_task_graph_id,
                "task_graph_reference_mode": decision.task_graph_reference_mode,
                "task_graph_binding": {
                    "task_graph_id": decision.task_graph_id or getattr(decision, "last_task_graph_id", None),
                    "policy": getattr(decision, "task_graph_binding_policy", "reference_only"),
                    "reason": getattr(decision, "binding_reason", ""),
                    "active_source": "session_restore" if decision.task_graph_id else "none",
                },
                "source": decision.source,
            })
        if clear_active_graph:
            payload["clear_active_graph"] = True
            payload["active_graph_policy"] = "clear"
        if payload.get("task_graph_id") or payload.get("graph_id"):
            try:
                from zulong.tools.task_tools import normalize_task_graph_id

                normalized_graph_id = normalize_task_graph_id(
                    payload.get("task_graph_id") or payload.get("graph_id")
                )
                payload["task_graph_id"] = normalized_graph_id
                payload["graph_id"] = normalized_graph_id
                if decision:
                    decision.task_graph_id = normalized_graph_id
            except Exception:
                pass
        if explicit_workspace:
            payload["workspace_path"] = explicit_workspace
            payload["cwd"] = explicit_workspace
            if decision:
                decision.workspace_path = explicit_workspace

        # TSD 23.11: Web 只传递 session/conversation anchor。
        # BFS 自恢复、冷热路径检索和上下文打包属于 L1-B 主链职责，
        # 不在发布 USER_TEXT 前同步等待，避免 Web Router 抢占 L1-B 热链路。

        event = ZulongEvent(
            type=EventType.USER_TEXT,
            source="launcher/web_ui",
            payload=payload,
            priority=EventPriority.NORMAL,
        )
        # 在线程池中发布事件，避免阻塞 asyncio 事件循环
        # event_bus.publish(USER_TEXT) 会同步调用 L1-B → L2 全链路，
        # 如果在事件循环线程中执行会阻塞 WebSocket ping/pong 和消息广播
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, event_bus.publish, event)
        update_task_execution_status(
            state="running",
            phase="running",
            message="任务执行中，祖龙正在推进下一步。",
            request_id=request_id,
            conversation_id=session_id,
            session_id=session_id,
            workspace_path=payload.get("workspace_path"),
            project_id=payload.get("project_id"),
            task_graph_id=payload.get("task_graph_id") or "",
            tool_name="",
            awaiting_approval=False,
        )
        logger.info("[WebChatRouter] USER_TEXT 已发布到 EventBus (via executor)")
    except Exception as e:
        logger.error(f"[WebChatRouter] EventBus 发布失败: {e}", exc_info=True)
        update_task_execution_status(
            state="failed",
            phase="failed",
            message=f"消息无法提交到推理链路：{e}",
            request_id=request_id,
            conversation_id=session_id,
            session_id=session_id,
            progress_items=[{
                "label": "提交推理链路失败",
                "status": "failed",
                "source": "summary",
                "detail": str(e),
                "timestamp": time.time(),
            }],
        )
        msg = {
            "type": "CHAT_RESPONSE",
            "text": _failure_response_text(f"消息无法提交到推理链路：{e}"),
            "request_id": request_id,
        }
        try:
            await _send_to_ws(ws, msg)
        except Exception:
            await _broadcast(msg)


async def _chat_via_engine(ws, text, request_id, session_id, referenced_nodes, decision=None):
    """兼容入口：统一转发到 EventBus 主链，禁止绕过 L1-B 直达 L2。"""
    logger.warning("[WebChatRouter] _chat_via_engine 已弃用，转发到 _chat_via_eventbus")
    await _chat_via_eventbus(
        ws,
        text,
        request_id,
        session_id,
        referenced_nodes,
        decision=decision,
    )


async def _retrieve_memory_context_for_prompt(text: str, top_k: int = 3) -> str:
    """为首轮 prompt 获取轻量记忆上下文。

    调用方会用短超时等待；这里独立成 task，避免阻塞 LLM 首字。
    """
    try:
        mg = _get_active_memory_graph()
        if not mg:
            return ""
        context_results = await mg.retrieve_context(text, top_k=top_k)
        memory_lines = []
        for r in context_results or []:
            label = r.get("label", "")
            content = r.get("content", "")
            graph_memory_id = r.get("graph_memory_id") or r.get("node_id", "")
            shard_id = r.get("shard_id", "")
            if label or content:
                address_hint = f" [graph_memory_id={graph_memory_id}" + (f", shard={shard_id}" if shard_id else "") + "]"
                memory_lines.append(f"- {label}: {content[:200]}{address_hint}")
        return "\n".join(memory_lines)
    except Exception as e:
        logger.debug(f"[WebChatRouter] retrieve memory context failed: {e}")
        return ""


async def _sync_ide_cancel(cancel_evt: asyncio.Event, session_id: str):
    """监听IDE端cancel_event并同步到Web端cancel_evt"""
    while True:
        await asyncio.sleep(1.0)
        try:
            from zulong.ide.ide_server import _sessions
            # 检查是否有IDE端对应session被取消
            for sid, sess in _sessions.items():
                if hasattr(sess, 'cancel_event') and sess.cancel_event.is_set():
                    cancel_evt.set()
                    logger.info(f"[WebChatRouter] IDE端取消已同步到Web端")
                    return
            # 也检查引擎级中断标志
            from zulong.l2.inference_engine import get_inference_engine
            engine = get_inference_engine()
            if engine and engine._interrupt_flag:
                cancel_evt.set()
                logger.info(f"[WebChatRouter] 引擎中断标志已同步到Web端")
                return
        except Exception:
            pass


async def _handle_stop_generation(data: dict) -> None:
    """处理停止生成请求 — 支持单session停止和全局停止"""
    request_id = data.get("request_id")
    session_id = data.get("session_id")
    stop_key = str(request_id or session_id or "__global__")
    now = time.time()
    last_stop_ack = _recent_stop_ack_at.get(stop_key, 0.0)
    is_duplicate_stop = bool(last_stop_ack and now - last_stop_ack < 1.0)
    _recent_stop_ack_at[stop_key] = now
    for key, ts in list(_recent_stop_ack_at.items()):
        if now - ts > 30:
            _recent_stop_ack_at.pop(key, None)

    if request_id and request_id in _chat_cancels:
        _chat_cancels[request_id].set()
        logger.info(f"[WebChatRouter] 已设置 Web chat cancel_event: {request_id}")

    try:
        from zulong.core.state_manager import state_manager
        if hasattr(state_manager, "request_interrupt"):
            state_manager.request_interrupt(request_id=request_id, session_id=session_id)
        else:
            state_manager.set_interrupt_flag(True)
        logger.info("[WebChatRouter] 停止生成: 全局 scoped interrupt 已记录")
    except Exception as e:
        logger.warning(f"[WebChatRouter] 设置全局中断标志失败: {e}")

    try:
        from zulong.l2.task_state_manager import task_state_manager
        task_state_manager.clear_active_task(clear_stack=False)
    except Exception as e:
        logger.debug(f"[WebChatRouter] 清理 TaskStateManager active 跳过: {e}")

    try:
        from zulong.tools.task_tools import (
            get_active_task_graph,
            normalize_task_graph_id,
            set_active_task_graph,
        )

        expected_graph_id = normalize_task_graph_id(
            data.get("task_graph_id")
            or _task_execution_status.get("task_graph_id")
            or ""
        )
        active_tg = get_active_task_graph()
        active_graph_id = normalize_task_graph_id(getattr(active_tg, "id", "")) if active_tg else ""
        if expected_graph_id and active_graph_id and active_graph_id != expected_graph_id:
            set_active_task_graph(None, None)
            logger.info(
                "[WebChatRouter] 停止后清除错焦点任务图: active=%s expected=%s",
                active_graph_id,
                expected_graph_id,
            )
    except Exception as e:
        logger.debug(f"[WebChatRouter] 停止后任务图焦点检查跳过: {e}")

    if not is_duplicate_stop:
        update_task_execution_status(
            state="cancelled",
            phase="cancelled",
            message="用户已停止当前任务。",
            request_id=request_id,
            conversation_id=session_id,
            session_id=session_id,
            awaiting_approval=False,
            progress_items=[{
                "label": "用户已停止任务",
                "status": "completed",
                "source": "user_interject",
                "timestamp": time.time(),
            }],
        )
        await _broadcast({
            "type": "STOP_ACK",
            "request_id": request_id,
            "session_id": session_id,
            "message": "停止指令已确认",
        })
    else:
        logger.info("[WebChatRouter] 忽略重复停止确认: request_id=%s", request_id or "-")

    try:
        import sys
        l2_module = sys.modules.get("zulong.l2.inference_engine")
        engine = getattr(l2_module, "inference_engine", None) if l2_module else None
        if engine and hasattr(engine, "request_interrupt"):
            engine.request_interrupt(request_id=request_id, session_id=session_id)
            logger.info("[WebChatRouter] 停止生成: 已同步到已加载的 L2 引擎")
    except Exception as e:
        logger.debug(f"[WebChatRouter] 同步 L2 引擎中断跳过: {e}")

    # ── 核心：设置 IDE FC Runner 的 cancel_event ──
    stopped_sessions = 0
    try:
        from zulong.ide.ide_server import _sessions, _engine_instance
        if session_id:
            # 停止指定 session
            sess = _sessions.get(session_id)
            if sess and hasattr(sess, 'cancel_event') and sess.cancel_event:
                sess.cancel_event.set()
                stopped_sessions = 1
                logger.info(f"[WebChatRouter] 停止指定 session: {session_id[:12]}")
        else:
            # 无 session_id 时停止所有 session
            for sid, sess in _sessions.items():
                if hasattr(sess, 'cancel_event') and sess.cancel_event:
                    sess.cancel_event.set()
                    stopped_sessions += 1
        if _engine_instance and hasattr(_engine_instance, '_interrupt_flag'):
            _engine_instance._interrupt_flag = True
        if stopped_sessions:
            logger.info(
                f"[WebChatRouter] 停止生成: 已设置 {stopped_sessions} 个 session 的 cancel_event")
    except Exception as e:
        logger.warning(f"[WebChatRouter] 设置 cancel_event 失败: {e}")

    try:
        from zulong.core.event_bus import event_bus
        from zulong.core.types import EventType, EventPriority, ZulongEvent
        event = ZulongEvent(
            type=EventType.SYSTEM_INTERRUPT,
            source="launcher/web_ui",
            payload={
                "reason": "user_stop_generation",
                "request_id": request_id,
                "session_id": session_id,
            },
            priority=EventPriority.HIGH,
        )
        event_bus.publish(event)
    except Exception as e:
        logger.error(f"[WebChatRouter] 发布停止中断失败: {e}")


async def _handle_conversation_switch(data: dict) -> None:
    payload = data.get("payload") if isinstance(data.get("payload"), dict) else data
    conversation_id = payload.get("conversation_id") or payload.get("session_id") or data.get("session_id")
    if not conversation_id:
        return
    try:
        get_conversation_orchestrator().store.set_active_conversation(conversation_id)
    except Exception as e:
        logger.debug(f"[WebChatRouter] conversation switch skipped: {e}")

    raw_graph_id = payload.get("task_graph_id") or payload.get("graph_id") or ""
    workspace_path = payload.get("workspace_path") or payload.get("cwd") or ""
    session_node_id = payload.get("session_node_id") or payload.get("dialogue_session_id") or ""
    clear_requested = bool(payload.get("clear_active_graph") or payload.get("is_new_session"))
    try:
        from zulong.tools.task_tools import (
            _interaction_store_claims_graph,
            infer_task_graph_owner_session_node_id,
            load_task_graph_deterministic,
            normalize_task_graph_id,
            set_active_task_graph,
        )

        graph_id = normalize_task_graph_id(raw_graph_id)
        session_node_id = infer_task_graph_owner_session_node_id(conversation_id, session_node_id)
        if clear_requested and payload.get("is_new_session"):
            set_active_task_graph(None, None)
            logger.info(
                "[WebChatRouter] 新会话切换优先清空 active graph: session=%s",
                conversation_id,
            )
        elif graph_id:
            claim_unowned = _interaction_store_claims_graph(conversation_id, graph_id)
            loaded = load_task_graph_deterministic(
                graph_id,
                workspace_dir=workspace_path or None,
                conversation_id=conversation_id,
                session_node_id=session_node_id,
                claim_unowned=claim_unowned,
            )
            if loaded:
                actual_workspace = workspace_path or ""
                try:
                    from zulong.tools.task_tools import get_active_task_graph, get_active_workspace_dir
                    active_tg = get_active_task_graph()
                    if active_tg is not None and normalize_task_graph_id(getattr(active_tg, "id", "")) == graph_id:
                        actual_workspace = (
                            getattr(active_tg, "metadata", {}).get("workspace_dir")
                            or get_active_workspace_dir()
                            or actual_workspace
                        )
                except Exception:
                    pass
                logger.info(
                    "[WebChatRouter] Conversation switch activated TaskGraph: "
                    "session=%s graph=%s requested_workspace=%s actual_workspace=%s",
                    conversation_id,
                    graph_id,
                    workspace_path or "-",
                    actual_workspace or "-",
                )
            else:
                set_active_task_graph(None, None)
                logger.warning(
                    "[WebChatRouter] 会话切换拒绝/未能激活任务图，已清空 active graph: session=%s graph=%s",
                    conversation_id,
                    graph_id,
                )
        elif clear_requested:
            set_active_task_graph(None, None)
            logger.info(
                "[WebChatRouter] 新会话/空会话切换，已清空 active graph: session=%s",
                conversation_id,
            )
    except Exception as e:
        logger.debug(f"[WebChatRouter] conversation active graph sync skipped: {e}")


async def _handle_chat_visible_message(data: dict) -> None:
    payload = data.get("payload") if isinstance(data.get("payload"), dict) else data
    conversation_id = payload.get("conversation_id") or payload.get("session_id")
    session_node_id = payload.get("session_node_id") or payload.get("dialogue_session_id") or ""
    text = (payload.get("text") or "").strip()
    role = payload.get("role") or "assistant"
    if not conversation_id or not text or role not in ("user", "assistant", "tool"):
        return
    if not session_node_id:
        session_node_id = (
            conversation_id if _is_dialogue_session_node_id(conversation_id)
            else f"dialogue:session_{_compact_dialogue_id(conversation_id)}"
        )
    payload["session_node_id"] = session_node_id
    payload["dialogue_session_id"] = session_node_id
    try:
        get_interaction_store().upsert_conversation(
            conversation_id,
            title=text[:20] if role == "user" else "",
            source=payload.get("source") or "web_ui",
            workspace_path=payload.get("workspace_path") or payload.get("cwd"),
            project_id=payload.get("project_id"),
            task_graph_id=payload.get("task_graph_id"),
            session_node_id=session_node_id,
            active=True,
        )
        get_interaction_store().append_event(
            conversation_id=conversation_id,
            turn_id=payload.get("turn_id") or payload.get("request_id"),
            event_type="visible_message",
            role=role,
            source=payload.get("source") or "web_ui",
            text=text,
            payload=payload,
            workspace_path=payload.get("workspace_path") or payload.get("cwd"),
            project_id=payload.get("project_id"),
            task_graph_id=payload.get("task_graph_id"),
        )
    except Exception as exc:
        logger.debug(f"[WebChatRouter] visible message 持久化失败: {exc}")


async def _handle_voice_bind(ws: WebSocket, data: dict) -> None:
    payload = data.get("payload") if isinstance(data.get("payload"), dict) else data
    try:
        result = get_conversation_orchestrator().route_voice_record(payload)
        await _send_to_ws(ws, {
            "type": "voice:record_created",
            "payload": result,
            "voice_event_id": result.get("voice_event_id"),
            "conversation_id": result.get("conversation_id"),
        })
        if result.get("linked"):
            await _send_to_ws(ws, {
                "type": "voice:record_linked",
                "payload": result,
                "voice_event_id": result.get("voice_event_id"),
                "conversation_id": result.get("conversation_id"),
            })
    except Exception as e:
        logger.error(f"[WebChatRouter] voice bind failed: {e}", exc_info=True)
        await _send_to_ws(ws, {
            "type": "voice:record_created",
            "payload": {"linked": False, "error": str(e)},
        })


async def _handle_voice_list(ws: WebSocket, data: dict) -> None:
    """列出语音记录，支持过滤和分页。"""
    store = get_interaction_store()
    limit = data.get("limit", 50)
    offset = data.get("offset", 0)
    linked_only = data.get("linked_only")
    if linked_only is not None:
        linked_only = bool(linked_only)
    records = store.list_voice_records(limit=limit, offset=offset, linked_only=linked_only)
    await _send_to_ws(ws, {
        "type": "voice:list",
        "payload": {
            "records": records,
            "count": len(records),
            "limit": limit,
            "offset": offset,
        },
    })


async def _handle_voice_delete(ws: WebSocket, data: dict) -> None:
    """删除单条语音记录。"""
    voice_event_id = data.get("voice_event_id")
    if not voice_event_id:
        await _send_to_ws(ws, {
            "type": "voice:delete",
            "payload": {"ok": False, "error": "缺少 voice_event_id"},
        })
        return
    store = get_interaction_store()
    ok = store.delete_voice_record(voice_event_id)
    await _send_to_ws(ws, {
        "type": "voice:delete",
        "payload": {"ok": ok, "voice_event_id": voice_event_id},
    })


async def _handle_ide_action(ws: WebSocket, action: str, data: dict) -> None:
    payload = data.get("payload") if isinstance(data.get("payload"), dict) else dict(data)
    payload.setdefault("conversation_id", data.get("conversation_id") or data.get("session_id"))
    payload.setdefault("turn_id", data.get("turn_id") or data.get("request_id"))
    requested_workspace = payload.get("workspace_path") or payload.get("cwd")
    is_explicit_workspace_switch = action in (
        MessageType.IDE_OPEN_WORKSPACE,
        "ide_open_workspace",
        "ide:open_workspace",
    )
    workspace_path = _resolve_active_task_workspace(
        requested_workspace,
        payload.get("task_graph_id") or payload.get("graph_id"),
        prefer_explicit=is_explicit_workspace_switch,
    )
    if workspace_path:
        payload["workspace_path"] = workspace_path
        payload["cwd"] = workspace_path
        conversation_id = payload.get("conversation_id") or payload.get("session_id")
        try:
            if conversation_id:
                get_interaction_store().upsert_conversation(
                    conversation_id,
                    source=payload.get("source") or "web_ui",
                    workspace_path=workspace_path,
                    task_graph_id=payload.get("task_graph_id") or payload.get("graph_id"),
                    active=True,
                )
        except Exception:
            pass
    try:
        from zulong.ide.ide_server import request_ide_action
        logger.info(
            "[WebChatRouter] IDE action=%s requested_workspace=%s resolved_workspace=%s task_graph_id=%s",
            action,
            requested_workspace,
            workspace_path,
            payload.get("task_graph_id") or payload.get("graph_id"),
        )
        result = await request_ide_action(action, payload)
        await _send_to_ws(ws, make_unified_message(
            MessageType.IDE_ACTION_RESULT,
            {
                "action": action,
                "payload": result,
                "conversation_id": payload.get("conversation_id"),
                "turn_id": payload.get("turn_id"),
            },
            session_id=payload.get("conversation_id") or payload.get("session_id") or "",
            msg_id=payload.get("turn_id") or None,
        ))
    except Exception as e:
        await _send_to_ws(ws, make_unified_message(
            MessageType.IDE_ACTION_RESULT,
            {
                "action": action,
                "payload": {"ok": False, "error": str(e)},
                "conversation_id": payload.get("conversation_id"),
                "turn_id": payload.get("turn_id"),
            },
            session_id=payload.get("conversation_id") or payload.get("session_id") or "",
            msg_id=payload.get("turn_id") or None,
        ))


_EXECUTION_NODE_TYPES = {"tool_call", "tool_result", "approval"}
_MEMORY_GRAPH_EXECUTION_VIEW_LIMIT = 120
_MEMORY_GRAPH_EXECUTION_BACKFILL_DONE = False
_MEMORY_GRAPH_EXECUTION_BACKFILL_STARTED = False
_MEMORY_GRAPH_CHUNK_MAX_BYTES = 520 * 1024


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _memory_node_id(node: Any) -> str:
    if isinstance(node, dict):
        return str(node.get("id") or node.get("node_id") or "")
    return str(getattr(node, "node_id", "") or getattr(node, "id", "") or "")


def _memory_node_type(node: Any) -> str:
    if isinstance(node, dict):
        return _enum_value(node.get("type") or node.get("node_type"))
    return _enum_value(getattr(node, "node_type", "") or getattr(node, "type", ""))


def _memory_node_metadata(node: Any) -> Dict[str, Any]:
    if isinstance(node, dict):
        meta = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        return dict(meta)
    meta = getattr(node, "metadata", None)
    return dict(meta or {}) if isinstance(meta, dict) else {}


def _get_memory_node(mg: Any, node_id: str) -> Optional[Any]:
    try:
        if hasattr(mg, "get_node"):
            return mg.get_node(node_id)
    except Exception:
        return None
    return None


def _serialize_memory_node_for_snapshot(
    mg: Any,
    node: Any,
    *,
    include_children_count: bool = True,
) -> Dict[str, Any]:
    node_id = _memory_node_id(node)
    node_type = _memory_node_type(node)
    metadata = _memory_node_metadata(node)
    content = getattr(node, "content", None)
    if content and "content" not in metadata:
        metadata["content"] = content
    content_summary = getattr(node, "content_summary", None)
    if content_summary and "content_summary" not in metadata:
        metadata["content_summary"] = content_summary
    if node_type in _EXECUTION_NODE_TYPES:
        metadata["is_execution_event"] = True

    label = ""
    activation = 0.0
    if isinstance(node, dict):
        label = str(node.get("label") or node_id)
        try:
            activation = float(node.get("activation") or 0.0)
        except (TypeError, ValueError):
            activation = 0.0
    else:
        label = str(getattr(node, "label", "") or node_id)
        try:
            activation = float(getattr(node, "activation", 0.0) or 0.0)
        except (TypeError, ValueError):
            activation = 0.0

    data = {
        "id": node_id,
        "type": node_type,
        "label": label,
        "activation": round(activation, 3),
        "metadata": metadata,
    }
    for attr in ("backend_ref", "created_at", "last_accessed", "access_count"):
        value = node.get(attr) if isinstance(node, dict) else getattr(node, attr, None)
        if value not in (None, ""):
            data[attr] = value
    try:
        if hasattr(mg, "get_children"):
            if include_children_count and node_type not in _EXECUTION_NODE_TYPES:
                data["children_count"] = len(mg.get_children(node_id))
    except Exception:
        pass
    return data


def _serialize_memory_edge_for_snapshot(src: str, dst: str, edge_type: Any, edge_data: Any = None) -> Dict[str, Any]:
    if isinstance(edge_data, dict):
        edge_type = edge_data.get("edge_type") or edge_data.get("type") or edge_type
        weight = edge_data.get("weight", 1.0)
        protected = edge_data.get("protected", False)
        metadata = edge_data.get("metadata", {})
    else:
        edge_type = getattr(edge_data, "edge_type", edge_type)
        weight = getattr(edge_data, "weight", 1.0)
        protected = getattr(edge_data, "protected", False)
        metadata = getattr(edge_data, "metadata", {})
    try:
        weight = round(float(weight), 3)
    except (TypeError, ValueError):
        weight = 1.0
    return {
        "source": src,
        "target": dst,
        "type": _enum_value(edge_type) or "reference",
        "weight": weight,
        "protected": bool(protected),
        "metadata": metadata if isinstance(metadata, dict) else {},
    }


def _collect_execution_edges_sharded(mg: Any, execution_ids: Set[str]) -> List[Dict[str, Any]]:
    if not hasattr(mg, "list_all_shards") or not hasattr(mg, "get_shard"):
        return []
    edges: List[Dict[str, Any]] = []
    for shard_id in mg.list_all_shards():
        try:
            shard = mg.get_shard(shard_id, load_if_missing=True)
        except TypeError:
            shard = mg.get_shard(shard_id)
        except Exception:
            shard = None
        if not shard or not getattr(shard, "topology", None):
            continue
        graph = getattr(shard.topology, "graph", None)
        if graph is None:
            continue
        for edge in graph.es:
            try:
                src = mg._vertex_node_id(graph.vs[edge.source]) if hasattr(mg, "_vertex_node_id") else graph.vs[edge.source]["name"]
                dst = mg._vertex_node_id(graph.vs[edge.target]) if hasattr(mg, "_vertex_node_id") else graph.vs[edge.target]["name"]
                if src not in execution_ids and dst not in execution_ids:
                    continue
                edge_type = edge["type"] if "type" in edge.attributes() else "association"
                edge_props = shard.get_edge(src, dst) if hasattr(shard, "get_edge") else None
                edges.append(_serialize_memory_edge_for_snapshot(str(src), str(dst), edge_type, edge_props))
            except Exception:
                continue
    return edges


def _collect_sharded_execution_nodes(
    mg: Any,
    *,
    limit: int = _MEMORY_GRAPH_EXECUTION_VIEW_LIMIT,
) -> Optional[Dict[str, Any]]:
    if not hasattr(mg, "list_all_shards") or not hasattr(mg, "get_shard"):
        return None

    candidates: List[Any] = []
    type_counts: Dict[str, int] = {node_type: 0 for node_type in sorted(_EXECUTION_NODE_TYPES)}
    try:
        shard_ids = sorted(list(mg.list_all_shards()), reverse=True)
    except Exception:
        shard_ids = []

    for shard_id in shard_ids:
        try:
            shard = mg.get_shard(shard_id, load_if_missing=True)
        except TypeError:
            shard = mg.get_shard(shard_id)
        except Exception:
            shard = None
        if not shard:
            continue
        properties = getattr(shard, "properties", None)
        if not properties or not hasattr(properties, "get_nodes_by_type"):
            continue
        for node_type in sorted(_EXECUTION_NODE_TYPES):
            try:
                iterator = properties.get_nodes_by_type(node_type)
            except Exception:
                continue
            for node in iterator or []:
                type_counts[node_type] += 1
                candidates.append(node)

    candidates.sort(
        key=lambda node: (
            float(getattr(node, "created_at", 0.0) or 0.0),
            _memory_node_id(node),
        ),
        reverse=True,
    )
    visible_nodes = candidates[:max(1, int(limit or _MEMORY_GRAPH_EXECUTION_VIEW_LIMIT))]
    nodes_by_id = {
        _memory_node_id(node): _serialize_memory_node_for_snapshot(
            mg,
            node,
            include_children_count=False,
        )
        for node in visible_nodes
        if _memory_node_id(node)
    }
    execution_ids = set(nodes_by_id)
    return {
        "nodes_by_id": nodes_by_id,
        "execution_ids": execution_ids,
        "type_counts": type_counts,
        "total_execution_nodes": sum(type_counts.values()),
        "truncated": len(candidates) > len(visible_nodes),
    }


def _collect_execution_edges_for_visible_nodes(
    mg: Any,
    execution_ids: Set[str],
    *,
    max_edges: int = 360,
    max_neighbors_per_node: int = 24,
) -> List[Dict[str, Any]]:
    if not execution_ids:
        return []
    if not hasattr(mg, "list_all_shards") or not hasattr(mg, "get_shard"):
        return []

    collected: List[Dict[str, Any]] = []
    seen: Set[tuple] = set()

    def add_edge(src: str, dst: str, edge_type: Any = None, edge_props: Any = None) -> bool:
        if not src or not dst:
            return False
        edge_payload = _serialize_memory_edge_for_snapshot(src, dst, edge_type, edge_props)
        key = (str(edge_payload.get("source") or ""), str(edge_payload.get("target") or ""), str(edge_payload.get("type") or ""))
        if not key[0] or not key[1] or key in seen:
            return False
        seen.add(key)
        collected.append(edge_payload)
        return len(collected) >= max_edges

    for node_id in sorted(execution_ids):
        if len(collected) >= max_edges:
            break
        try:
            shard, _ = mg._get_indexed_shard(node_id) if hasattr(mg, "_get_indexed_shard") else (None, None)
        except Exception:
            shard = None
        if not shard:
            continue

        for mode in ("out", "in"):
            if len(collected) >= max_edges:
                break
            try:
                neighbors = list(shard.get_topology_neighbors(node_id, mode=mode) or [])
            except Exception:
                neighbors = []
            for neighbor_id in neighbors[:max_neighbors_per_node]:
                if len(collected) >= max_edges:
                    break
                if mode == "out":
                    src_id, dst_id = node_id, str(neighbor_id)
                else:
                    src_id, dst_id = str(neighbor_id), node_id
                try:
                    edge_info = shard.get_topology_edge_info(src_id, dst_id)
                except Exception:
                    edge_info = None
                edge_type = edge_info[0] if edge_info else "reference"
                try:
                    edge_props = shard.get_edge(src_id, dst_id)
                except Exception:
                    edge_props = None
                if add_edge(src_id, dst_id, edge_type, edge_props):
                    break

        for getter in ("_get_cross_edges_from", "_get_cross_edges_to"):
            if len(collected) >= max_edges or not hasattr(mg, getter):
                continue
            try:
                cross_edges = getattr(mg, getter)(node_id) or []
            except Exception:
                cross_edges = []
            for edge in cross_edges[:max_neighbors_per_node]:
                if len(collected) >= max_edges:
                    break
                src_id = str(edge.get("src_id") or edge.get("source") or "")
                dst_id = str(edge.get("dst_id") or edge.get("target") or "")
                edge_type = edge.get("edge_type") or edge.get("type") or "reference"
                if add_edge(src_id, dst_id, edge_type, edge):
                    break

    return collected


def _get_memory_graph_execution_view(
    mg: Any,
    *,
    limit: int = _MEMORY_GRAPH_EXECUTION_VIEW_LIMIT,
) -> Dict[str, Any]:
    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    execution_ids: Set[str] = set()
    type_counts: Dict[str, int] = {node_type: 0 for node_type in sorted(_EXECUTION_NODE_TYPES)}
    truncated = False

    sharded_view = _collect_sharded_execution_nodes(mg, limit=limit)
    if sharded_view is not None:
        nodes_by_id = dict(sharded_view.get("nodes_by_id") or {})
        execution_ids = set(sharded_view.get("execution_ids") or set())
        type_counts.update(sharded_view.get("type_counts") or {})
        truncated = bool(sharded_view.get("truncated"))
    elif hasattr(mg, "get_nodes_by_type"):
        for node_type in sorted(_EXECUTION_NODE_TYPES):
            try:
                nodes = mg.get_nodes_by_type(node_type)
            except Exception:
                nodes = []
            for node in nodes or []:
                node_id = _memory_node_id(node)
                if not node_id:
                    continue
                type_counts[node_type] += 1
                if len(nodes_by_id) >= limit:
                    truncated = True
                    continue
                execution_ids.add(node_id)
                nodes_by_id[node_id] = _serialize_memory_node_for_snapshot(
                    mg,
                    node,
                    include_children_count=False,
                )

    edges = _collect_execution_edges_for_visible_nodes(mg, execution_ids)

    for edge in edges:
        for endpoint in (edge.get("source"), edge.get("target")):
            if endpoint and endpoint not in nodes_by_id:
                node = _get_memory_node(mg, str(endpoint))
                if node:
                    nodes_by_id[str(endpoint)] = _serialize_memory_node_for_snapshot(
                        mg,
                        node,
                        include_children_count=False,
                    )

    nodes = list(nodes_by_id.values())
    nodes.sort(key=lambda item: (item.get("created_at") or 0, item.get("id") or ""))
    visible_execution_ids = {str(node.get("id")) for node in nodes if node.get("id")}

    return {
        "nodes": nodes,
        "edges": edges,
        "execution_node_ids": sorted(node_id for node_id in execution_ids if node_id in visible_execution_ids),
        "stats": {
            "total_execution_nodes": sum(type_counts.values()) or len(execution_ids),
            "total_execution_edges": len(edges),
            "returned_execution_nodes": len(execution_ids),
            "returned_nodes_with_context": len(nodes),
            "returned_execution_edges": len(edges),
            "view_limit": limit,
            "truncated": truncated,
            **type_counts,
        },
    }


def _ensure_execution_events_backfilled(mg: Any) -> None:
    global _MEMORY_GRAPH_EXECUTION_BACKFILL_DONE, _MEMORY_GRAPH_EXECUTION_BACKFILL_STARTED
    if _MEMORY_GRAPH_EXECUTION_BACKFILL_DONE or _MEMORY_GRAPH_EXECUTION_BACKFILL_STARTED:
        return
    if not _launcher_ready_for_memory_backfill():
        return
    _MEMORY_GRAPH_EXECUTION_BACKFILL_STARTED = True

    def _run_backfill() -> None:
        global _MEMORY_GRAPH_EXECUTION_BACKFILL_DONE, _MEMORY_GRAPH_EXECUTION_BACKFILL_STARTED
        try:
            from zulong.launcher.memory_mirror import backfill_recent_interactions_to_memory_graph

            count = backfill_recent_interactions_to_memory_graph(limit=5, events_per_conversation=100)
            if count:
                logger.info("[WebChatRouter] MemoryGraph 执行事件回填完成: %s events", count)
        except Exception as exc:
            logger.debug("[WebChatRouter] MemoryGraph 执行事件回填跳过: %s", exc)
        finally:
            _MEMORY_GRAPH_EXECUTION_BACKFILL_DONE = True
            _MEMORY_GRAPH_EXECUTION_BACKFILL_STARTED = False

    try:
        loop = _event_loop
        if loop and loop.is_running():
            loop.run_in_executor(None, _run_backfill)
        else:
            threading.Thread(
                target=_run_backfill,
                name="zulong-memory-backfill",
                daemon=True,
            ).start()
        logger.info("[WebChatRouter] MemoryGraph 执行事件回填已转入后台")
    except Exception as exc:
        _MEMORY_GRAPH_EXECUTION_BACKFILL_STARTED = False
        _MEMORY_GRAPH_EXECUTION_BACKFILL_DONE = True
        logger.debug("[WebChatRouter] MemoryGraph 执行事件回填调度失败: %s", exc)


def _launcher_ready_for_memory_backfill() -> bool:
    """Backfill is optional; never let it compete with startup modules."""
    try:
        from zulong.launcher import app as launcher_app_module

        launcher = getattr(launcher_app_module, "_app_instance", None)
        if launcher is None:
            return False
        if getattr(launcher, "phase", "") != "running":
            return False
        manager = getattr(launcher, "manager", None)
        if manager is None:
            return False
        status = manager.get_status() if hasattr(manager, "get_status") else {}
        return bool(status.get("launched"))
    except Exception:
        return False


def _merge_execution_view_into_payload(payload: Dict[str, Any], execution_view: Dict[str, Any]) -> Dict[str, Any]:
    if not execution_view:
        return payload
    payload = dict(payload or {})
    nodes = list(payload.get("nodes") or [])
    edges = list(payload.get("edges") or [])

    node_ids = {str(node.get("id")) for node in nodes if isinstance(node, dict) and node.get("id")}
    for node in execution_view.get("nodes") or []:
        node_id = str(node.get("id") or "")
        if node_id and node_id not in node_ids:
            nodes.append(node)
            node_ids.add(node_id)

    edge_keys = {
        (
            str(edge.get("source") if not isinstance(edge.get("source"), dict) else edge.get("source", {}).get("id")),
            str(edge.get("target") if not isinstance(edge.get("target"), dict) else edge.get("target", {}).get("id")),
            str(edge.get("type") or ""),
        )
        for edge in edges
        if isinstance(edge, dict)
    }
    for edge in execution_view.get("edges") or []:
        key = (str(edge.get("source") or ""), str(edge.get("target") or ""), str(edge.get("type") or ""))
        if key[0] and key[1] and key not in edge_keys:
            edges.append(edge)
            edge_keys.add(key)

    stats = dict(payload.get("stats") or {})
    execution_stats = dict(execution_view.get("stats") or {})
    if "returned_nodes" in stats:
        stats.setdefault("active_skeleton_returned_nodes", stats.get("returned_nodes"))
    if "returned_edges" in stats:
        stats.setdefault("active_skeleton_returned_edges", stats.get("returned_edges"))
    stats["returned_nodes"] = len(nodes)
    stats["returned_edges"] = len(edges)
    stats["execution"] = execution_stats
    stats["execution_nodes"] = execution_stats.get("total_execution_nodes", 0)

    payload["nodes"] = nodes
    payload["edges"] = edges
    payload["stats"] = stats
    payload["execution_view"] = execution_view
    payload["execution_node_ids"] = execution_view.get("execution_node_ids") or []
    return payload


async def _push_memory_graph_snapshot(ws: WebSocket) -> None:
    """推送记忆图谱快照到指定 WebSocket"""
    try:
        if not _launcher_ready_for_memory_backfill():
            await _send_memory_graph_payload(ws, _empty_memory_graph_payload())
            return
        payload = await _get_memory_graph_snapshot_payload_async(include_execution=True)
        if payload:
            await _send_memory_graph_payload(ws, payload)
    except Exception as e:
        logger.debug(f"[WebChatRouter] 推送记忆图谱失败: {e}")
        try:
            await _send_to_ws(ws, {
                "type": "MEMORY_GRAPH_UPDATE",
                "update_type": "error",
                "ts": time.time(),
                "nodes": [],
                "edges": [],
                "stats": {"total_nodes": 0, "total_edges": 0},
                "error": str(e),
            })
        except Exception:
            pass


async def _push_memory_graph_summary(ws: WebSocket) -> None:
    try:
        if not _launcher_ready_for_memory_backfill():
            return
        payload = await _get_memory_graph_snapshot_payload_async(summary_only=True)
        await _send_to_ws(ws, {
            "type": "MEMORY_GRAPH_UPDATE",
            **payload,
        }, persist=False)
    except Exception as e:
        logger.debug(f"[WebChatRouter] 推送记忆图谱摘要失败: {e}")


async def _send_memory_graph_payload(ws: WebSocket, payload: Dict[str, Any]) -> None:
    """Send a MemoryGraph snapshot in bounded frames without dropping content."""
    chunks, snapshot_id = _chunk_memory_graph_payload(payload)
    for index, chunk_payload in enumerate(chunks):
        chunk_payload["chunk_index"] = index
        chunk_payload["chunk_total"] = len(chunks)
        chunk_payload["is_last_chunk"] = index == len(chunks) - 1
        chunk_payload["update_type"] = "full" if index == 0 else "full_chunk"
        chunk_payload["snapshot_id"] = snapshot_id
        await _send_to_ws(ws, chunk_payload, persist=False)


def _chunk_memory_graph_payload(payload: Dict[str, Any]) -> tuple[List[Dict[str, Any]], str]:
    """Build bounded MemoryGraph frames without dropping nodes or edges."""
    nodes = list((payload or {}).get("nodes") or [])
    edges = list((payload or {}).get("edges") or [])
    snapshot_id = f"mg-{int(time.time() * 1000)}"

    base_payload = {
        key: value
        for key, value in (payload or {}).items()
        if key not in ("nodes", "edges", "execution_view")
    }
    execution_view = payload.get("execution_view") if isinstance(payload, dict) else None
    execution_nodes_by_id: Dict[str, Dict[str, Any]] = {}
    if isinstance(execution_view, dict):
        for node in execution_view.get("nodes") or []:
            if isinstance(node, dict) and node.get("id"):
                execution_nodes_by_id[str(node["id"])] = node

    base_stats = dict(base_payload.get("stats") or {})
    base_stats.setdefault("total_nodes", len(nodes))
    base_stats.setdefault("total_edges", len(edges))
    base_stats["returned_nodes"] = len(nodes)
    base_stats["returned_edges"] = len(edges)
    base_stats["transport"] = "chunked"
    base_payload["stats"] = base_stats

    chunks: List[Dict[str, Any]] = []
    node_index = 0
    edge_index = 0
    while node_index < len(nodes) or edge_index < len(edges) or not chunks:
        chunk_nodes: List[Dict[str, Any]] = []
        chunk_edges: List[Dict[str, Any]] = []
        chunk_node_ids: Set[str] = set()

        while node_index < len(nodes):
            candidate = nodes[node_index]
            candidate_nodes = chunk_nodes + [candidate]
            candidate_node_ids = {
                str(node.get("id"))
                for node in candidate_nodes
                if isinstance(node, dict) and node.get("id")
            }
            candidate_payload = _build_memory_graph_chunk_payload(
                base_payload,
                snapshot_id,
                0,
                1,
                False,
                candidate_nodes,
                chunk_edges,
                candidate_node_ids,
                execution_view,
                execution_nodes_by_id,
            )
            if (
                chunk_nodes
                and _json_size_bytes(candidate_payload) > _MEMORY_GRAPH_CHUNK_MAX_BYTES
            ):
                break
            chunk_nodes.append(candidate)
            node_index += 1
            chunk_node_ids = candidate_node_ids
            if _json_size_bytes(candidate_payload) > _MEMORY_GRAPH_CHUNK_MAX_BYTES:
                break

        while edge_index < len(edges):
            candidate_edge = edges[edge_index]
            candidate_edges = chunk_edges + [candidate_edge]
            candidate_payload = _build_memory_graph_chunk_payload(
                base_payload,
                snapshot_id,
                0,
                1,
                False,
                chunk_nodes,
                candidate_edges,
                chunk_node_ids,
                execution_view,
                execution_nodes_by_id,
            )
            if (
                chunk_edges
                and _json_size_bytes(candidate_payload) > _MEMORY_GRAPH_CHUNK_MAX_BYTES
            ):
                break
            chunk_edges.append(candidate_edge)
            edge_index += 1
            if _json_size_bytes(candidate_payload) > _MEMORY_GRAPH_CHUNK_MAX_BYTES:
                break

        chunks.append(_build_memory_graph_chunk_payload(
            base_payload,
            snapshot_id,
            0,
            1,
            False,
            chunk_nodes,
            chunk_edges,
            chunk_node_ids,
            execution_view,
            execution_nodes_by_id,
        ))

    transport = "chunked" if len(chunks) > 1 else "single"
    for chunk in chunks:
        chunk.setdefault("stats", {})["transport"] = transport
    return chunks, snapshot_id


def _json_size_bytes(data: Dict[str, Any]) -> int:
    return len(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _build_memory_graph_chunk_payload(
    base_payload: Dict[str, Any],
    snapshot_id: str,
    index: int,
    total_chunks: int,
    is_last_chunk: bool,
    chunk_nodes: List[Dict[str, Any]],
    chunk_edges: List[Dict[str, Any]],
    chunk_node_ids: Set[str],
    execution_view: Any,
    execution_nodes_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    chunk_execution_nodes = [
        node for node in chunk_nodes
        if isinstance(node, dict) and str(node.get("id")) in execution_nodes_by_id
    ]
    chunk_execution_edges = [
        edge for edge in chunk_edges
        if (
            str(edge.get("source") if not isinstance(edge.get("source"), dict) else edge.get("source", {}).get("id")) in chunk_node_ids
            or str(edge.get("target") if not isinstance(edge.get("target"), dict) else edge.get("target", {}).get("id")) in chunk_node_ids
        )
    ]
    chunk_payload = dict(base_payload)
    chunk_payload.update({
        "type": "MEMORY_GRAPH_UPDATE",
        "update_type": "full" if index == 0 else "full_chunk",
        "snapshot_id": snapshot_id,
        "chunk_index": index,
        "chunk_total": total_chunks,
        "is_last_chunk": is_last_chunk,
        "nodes": chunk_nodes,
        "edges": chunk_edges,
    })
    if isinstance(execution_view, dict):
        chunk_payload["execution_view"] = {
            "nodes": chunk_execution_nodes,
            "edges": chunk_execution_edges,
            "execution_node_ids": [
                str(node.get("id"))
                for node in chunk_execution_nodes
                if isinstance(node, dict) and node.get("id")
            ],
            "stats": execution_view.get("stats") or {},
        }
    return chunk_payload


def _get_memory_graph_snapshot_payload(
    summary_only: bool = False,
    include_execution: bool = False,
) -> dict:
    try:
        if not _launcher_ready_for_memory_backfill():
            return _empty_memory_graph_payload(summary_only=summary_only, unavailable_reason="launcher_not_ready")
        mg = _get_active_memory_graph()
        if not mg:
            return _empty_memory_graph_payload(summary_only=summary_only, unavailable_reason="memory_graph_unavailable")
        if summary_only:
            try:
                stats = dict(mg.get_total_stats()) if hasattr(mg, "get_total_stats") else {}
            except Exception:
                stats = {}
            if not stats:
                stats = dict(getattr(mg, "_stats", {}) or {})
            try:
                skeleton = mg.get_active_skeleton() if hasattr(mg, "get_active_skeleton") else {}
            except Exception:
                skeleton = {}
            skeleton_stats = dict(skeleton.get("stats") or {}) if isinstance(skeleton, dict) else {}
            return {
                "update_type": "summary",
                "ts": time.time(),
                "nodes": [],
                "edges": [],
                "stats": {
                    "total_nodes": stats.get("total_nodes", 0),
                    "total_edges": stats.get("total_edges", 0),
                    "active_skeleton_nodes": skeleton_stats.get(
                        "node_count",
                        len(skeleton.get("node_ids") or []) if isinstance(skeleton, dict) else 0,
                    ),
                    "active_skeleton_edges": skeleton_stats.get(
                        "edge_count",
                        len(skeleton.get("edges") or []) if isinstance(skeleton, dict) else 0,
                    ),
                    "transport": "summary",
                },
            }
        acquired = _MEMORY_GRAPH_SNAPSHOT_LOCK.acquire(blocking=False)
        if not acquired:
            return _empty_memory_graph_payload(unavailable_reason="snapshot_busy")
        try:
            if hasattr(mg, "to_frontend_dict"):
                payload = mg.to_frontend_dict(depth=0)
            elif hasattr(mg, "get_snapshot_for_frontend"):
                payload = mg.get_snapshot_for_frontend()
            else:
                payload = {}
        finally:
            _MEMORY_GRAPH_SNAPSHOT_LOCK.release()
        if include_execution:
            _ensure_execution_events_backfilled(mg)
            execution_view = _get_memory_graph_execution_view(mg)
            payload = _merge_execution_view_into_payload(payload or {}, execution_view)
        return {
            "update_type": "full",
            "ts": time.time(),
            **(payload or {}),
        }
    except Exception as exc:
        return {
            "update_type": "error",
            "ts": time.time(),
            "nodes": [],
            "edges": [],
            "stats": {"total_nodes": 0, "total_edges": 0},
            "error": str(exc),
        }


async def _get_memory_graph_snapshot_payload_async(
    *,
    summary_only: bool = False,
    include_execution: bool = False,
) -> dict:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                _get_memory_graph_snapshot_payload,
                summary_only,
                include_execution,
            ),
            timeout=1.0 if summary_only else _MEMORY_GRAPH_SNAPSHOT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return _empty_memory_graph_payload(
            summary_only=True,
            unavailable_reason="snapshot_timeout",
        )


def _empty_memory_graph_payload(
    *,
    summary_only: bool = False,
    unavailable_reason: Optional[str] = None,
) -> dict:
    payload = {
        "update_type": "summary" if summary_only else "full",
        "ts": time.time(),
        "nodes": [],
        "edges": [],
        "stats": {"total_nodes": 0, "total_edges": 0},
    }
    if unavailable_reason:
        payload["unavailable_reason"] = unavailable_reason
    return payload


@router.get("/api/memory-graph/snapshot")
async def get_memory_graph_snapshot(
    summary_only: bool = True,
    include_execution: bool = True,
):
    """Return a fast MemoryGraph frontend snapshot for Web fallback loading."""
    return await _get_memory_graph_snapshot_payload_async(
        summary_only=summary_only,
        include_execution=include_execution,
    )


@router.get("/api/memory-graph/context-seed")
async def get_memory_graph_context_seed(
    conversation_id: Optional[str] = None,
    task_graph_id: Optional[str] = None,
):
    """Return indexed MemoryGraph addresses for Web restore paths."""
    try:
        mg = _get_active_memory_graph()
        if not mg:
            return {"ok": False, "error": "memory graph unavailable"}

        payload: Dict[str, Any] = {"ok": True}
        if conversation_id:
            if hasattr(mg, "get_context_seed_for_conversation"):
                payload["conversation"] = mg.get_context_seed_for_conversation(conversation_id)
            else:
                payload["conversation"] = None
        if task_graph_id:
            if hasattr(mg, "get_task_node_id_for_graph"):
                task_node_id = mg.get_task_node_id_for_graph(task_graph_id)
            else:
                task_node_id = f"task:{task_graph_id}" if hasattr(mg, "has_node") and mg.has_node(f"task:{task_graph_id}") else None
            payload["task_graph"] = {
                "task_graph_id": task_graph_id,
                "task_node_id": task_node_id,
            }
        return payload
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _handle_expand_node(node_id: str, ws: WebSocket) -> None:
    """处理展开节点请求"""
    try:
        mg = _get_active_memory_graph()
        if not mg:
            return
        if hasattr(mg, "get_node_children_for_frontend"):
            result = mg.get_node_children_for_frontend(node_id)
        elif hasattr(mg, "get_neighbors"):
            neighbors = mg.get_neighbors(node_id)
            result = {"node_id": node_id, "neighbors": neighbors}
        else:
            return
        await _send_to_ws(ws, {
            "type": "MEMORY_GRAPH_EXPAND_RESULT",
            "ts": time.time(),
            **result,
        })
    except Exception as e:
        logger.debug(f"[WebChatRouter] 展开节点失败: {e}")


async def _handle_memory_context_seed(ws: WebSocket, data: Dict[str, Any]) -> None:
    payload = await get_memory_graph_context_seed(
        conversation_id=data.get("conversation_id") or data.get("session_id"),
        task_graph_id=data.get("task_graph_id") or data.get("graph_id"),
    )
    await _send_to_ws(ws, {
        "type": "MEMORY_GRAPH_CONTEXT_SEED",
        "ts": time.time(),
        "payload": payload,
    })


# ── 对话会话管理（Web 前端会话栏重建） ─────────────────

async def _handle_list_dialogue_sessions(ws: WebSocket) -> None:
    """查询 MemoryGraph 中所有对话会话节点，返回会话列表"""
    try:
        loop = asyncio.get_running_loop()
        session_store = await loop.run_in_executor(None, _collect_dialogue_sessions)
        sent = await _send_to_ws(ws, {
            "type": "SESSION_LIST",
            "ts": time.time(),
            "sessions": session_store["sessions"],
            "activeSessionId": session_store.get("activeSessionId"),
        }, persist=False)
        if sent:
            logger.info(f"[WebChatRouter] SESSION_LIST: 返回 {len(session_store['sessions'])} 个会话")
    except Exception as e:
        logger.error(f"[WebChatRouter] LIST_DIALOGUE_SESSIONS 失败: {e}", exc_info=True)
        try:
            await _send_to_ws(ws, {
            "type": "SESSION_LIST",
            "ts": time.time(),
            "sessions": [],
            "error": str(e),
            }, persist=False)
        except Exception:
            pass


async def _handle_get_session_messages(ws: WebSocket, session_id: str) -> None:
    """获取指定会话的完整消息列表"""
    try:
        messages = []
        events = []
        try:
            store = get_conversation_orchestrator().store
            store_messages = store.get_messages(session_id)
            if store_messages:
                messages = store_messages
            raw_events = store.get_events(session_id, limit=1500, include_system=True)
            if raw_events:
                events = raw_events
        except Exception as store_err:
            logger.debug(f"[WebChatRouter] interaction store messages skipped: {store_err}")

        await _send_to_ws(ws, {
            "type": "SESSION_MESSAGES",
            "ts": time.time(),
            "session_id": session_id,
            "messages": messages,
            "events": events,
        })
        logger.info(
            f"[WebChatRouter] SESSION_MESSAGES: {session_id} -> "
            f"{len(messages)} messages, {len(events)} raw events")
    except Exception as e:
        logger.error(f"[WebChatRouter] GET_SESSION_MESSAGES 失败: {e}", exc_info=True)
        try:
            await _send_to_ws(ws, {
            "type": "SESSION_MESSAGES",
            "ts": time.time(),
            "session_id": session_id,
            "messages": [],
            "events": [],
            "error": str(e),
            })
        except Exception:
            pass


async def _handle_delete_dialogue_session(ws: WebSocket, session_id: str) -> None:
    """删除对话会话及其所有子节点（BFS 级联删除），同步清理 TaskGraph/AgentSessionStore/IDESession"""
    try:
        result = _delete_dialogue_session_cascade(session_id, cascade=True)
        await _broadcast({
            "type": "SESSION_DELETED",
            "ts": time.time(),
            "session_id": session_id,
            "session_node_id": result.get("session_node_id") or "",
            "session_ids": result.get("session_ids") or [session_id],
            "nodes_deleted": result.get("mg_nodes_deleted", 0),
            "deleted_node_ids": result.get("deleted_node_ids") or [],
        })
        logger.info(
            f"[WebChatRouter] DELETE_DIALOGUE_SESSION: {session_id} → "
            f"MG删除{result.get('mg_nodes_deleted', 0)}个, "
            f"session_node={result.get('session_node_id') or '-'}")
    except Exception as e:
        logger.error(f"[WebChatRouter] DELETE_DIALOGUE_SESSION 失败: {e}", exc_info=True)
        await _send_to_ws(ws, {
            "type": "SESSION_DELETED",
            "ts": time.time(),
            "session_id": session_id,
            "nodes_deleted": 0,
            "error": str(e),
        })


# ── REST API 端点 ────────────────────────────────────

@router.get("/api/chat/sessions")
async def list_sessions_rest():
    return _collect_dialogue_sessions()

@router.delete("/api/chat/sessions/{session_id}")
async def delete_session_rest(session_id: str, cascade: bool = True):
    """REST API 删除会话（级联清理 MemoryGraph/TaskGraph/AgentSessionStore）"""
    result = _delete_dialogue_session_cascade(session_id, cascade=cascade)
    await _broadcast({
        "type": "SESSION_DELETED",
        "ts": time.time(),
        "session_id": session_id,
        "session_node_id": result.get("session_node_id") or "",
        "session_ids": result.get("session_ids") or [session_id],
        "nodes_deleted": result.get("mg_nodes_deleted", 0),
        "deleted_node_ids": result.get("deleted_node_ids") or [],
    })
    return {
        **result,
        "tg_nodes_deleted": 0,
    }


@router.delete("/api/chat/sessions/{session_id}/messages/{message_id:path}")
async def delete_chat_message_rest(session_id: str, message_id: str):
    """删除单条 Web 对话内容，并同步删除对应 MemoryGraph 子节点。"""
    result = _delete_dialogue_message_cascade(session_id, message_id)
    if result.get("removed") or result.get("event_deleted"):
        try:
            session_store = _collect_dialogue_sessions()
            await _broadcast({
                "type": "SESSION_LIST",
                "ts": time.time(),
                "sessions": session_store["sessions"],
                "activeSessionId": session_store.get("activeSessionId"),
            })
        except Exception:
            pass
    return result


async def _send_to_ws(ws: WebSocket, message: dict, *, persist: bool = True) -> bool:
    """向 WebSocket 发送消息，根据协议版本自动选择格式

    统一协议客户端收到 `{msg_id, type: ns:action, session_id, ts, payload}` 格式。
    旧客户端收到传统的 `{type: UPPER_CASE, ...fields}` 格式。
    """
    if not is_public_payload(message):
        logger.debug(
            "[WebChatRouter] 跳过内部控制消息发送: type=%s",
            message.get("type"),
        )
        return True
    ws_id = id(ws)
    protocol = _ws_protocols.get(ws_id, "legacy")
    client_type = _ws_client_types.get(ws_id, "unknown")
    if persist:
        _persist_web_visible_message(message)

    if protocol == "unified" and ":" not in message.get("type", ""):
        # 旧格式 → 统一格式。只要客户端完成 handshake，就按统一协议下发。
        message = _protocol_bridge.to_unified(message, format_version="web")
    elif protocol == "unified" and ":" in message.get("type", "") and "payload" not in message:
        payload = {k: v for k, v in message.items() if k not in ("type", "request_id", "session_id", "conversation_id")}
        message = make_unified_message(
            message["type"],
            payload,
            session_id=message.get("conversation_id") or message.get("session_id", ""),
        )

    try:
        async with _get_ws_send_lock(ws):
            await ws.send_json(message)
        return True
    except WebSocketDisconnect:
        logger.debug("[WebChatRouter] WebSocket 已断开，跳过发送")
        _forget_ws_connection(ws)
        return False
    except Exception as e:
        text = str(e)
        exc_name = type(e).__name__
        if (
            "close message has been sent" in text
            or "ClientDisconnected" in exc_name
            or "AssertionError" in exc_name
        ):
            logger.debug(f"[WebChatRouter] WebSocket 发送不可用，跳过发送: {exc_name}: {text}")
            _forget_ws_connection(ws)
            return False
        raise


async def _send_initial_dashboard_state(ws: WebSocket) -> None:
    """发送 Dashboard 建连后的初始状态。"""
    engine_ready = False
    task_graph_snapshot = None
    memory_graph_stats = None
    code_anchor_stats = None
    launcher_ready = _launcher_ready_for_memory_backfill()
    if launcher_ready:
        try:
            from zulong.ide.ide_server import _get_engine
            engine_ready = _get_engine() is not None
        except Exception:
            pass
        try:
            from zulong.tools.task_tools import get_active_task_graph
            tg = get_active_task_graph()
            if tg:
                task_graph_snapshot = tg.to_frontend_dict()
        except Exception:
            pass
        try:
            mg = _get_active_memory_graph()
            if mg:
                memory_graph_stats = {
                    "total_nodes": mg._stats.get("total_nodes", 0),
                    "total_edges": mg._stats.get("total_edges", 0),
                }
        except Exception:
            pass
        try:
            from zulong.memory.code_anchor import get_code_anchor_store
            store = get_code_anchor_store()
            if store:
                code_anchor_stats = store.get_stats()
        except Exception:
            pass

    active_sessions_info = []
    if launcher_ready:
        try:
            from zulong.ide.ide_server import _sessions as _ide_sessions
            active_sessions_info = [s.to_info_dict() for s in _ide_sessions.values()]
        except Exception:
            pass

    return await _send_to_ws(ws, {
        "type": "WELCOME",
        "ts": time.time(),
        "payload": {
            "engine_ready": engine_ready,
            "launch_mode": _launch_mode,
            "task_graph": task_graph_snapshot,
            "task_execution_status": _task_status_snapshot(),
            "memory_graph_stats": memory_graph_stats,
            "code_anchor_stats": code_anchor_stats,
            "active_sessions": active_sessions_info,
        },
    }, persist=False)


async def handle_unified_root_ws(
    ws: WebSocket,
    *,
    accepted: bool = False,
    initial_msg: Optional[dict] = None,
) -> None:
    """统一根 WebSocket 入口中的 dashboard/monitor 分支。

    该函数复用 /ws 的处理语义，只是连接已经完成 handshake，因此不再要求
    客户端走旧的 /ws 路径。
    """
    global _event_loop
    if not accepted:
        await ws.accept()
    _ws_clients.add(ws)
    _ws_protocols[id(ws)] = "unified"

    if _event_loop is None:
        _event_loop = asyncio.get_running_loop()

    if _launcher_ready_for_memory_backfill():
        try:
            from zulong.ide.ide_server import _monitor_connections
            _monitor_connections.add(ws)
            import zulong.ide.ide_server as _ide_srv
            if _ide_srv._main_event_loop is None:
                _ide_srv._main_event_loop = _event_loop
        except Exception:
            pass

    if initial_msg:
        payload = initial_msg.get("payload", {}) if isinstance(initial_msg, dict) else {}
        _ws_client_types[id(ws)] = payload.get("client_type", "dashboard")

    if not await _send_to_ws(ws, make_unified_message(
        MessageType.TASK_ACK,
        {
            "status": "ok",
            "client_type": _ws_client_types.get(id(ws), "dashboard"),
        },
    ), persist=False):
        _ws_clients.discard(ws)
        _ws_protocols.pop(id(ws), None)
        _ws_client_types.pop(id(ws), None)
        return
    if not await _send_initial_dashboard_state(ws):
        _ws_clients.discard(ws)
        _ws_protocols.pop(id(ws), None)
        _ws_client_types.pop(id(ws), None)
        return
    try:
        await _push_memory_graph_summary(ws)
    except WebSocketDisconnect:
        _ws_clients.discard(ws)
        _ws_protocols.pop(id(ws), None)
        _ws_client_types.pop(id(ws), None)
        return

    try:
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
                fmt = _protocol_bridge.detect_format(data)
                unified = _protocol_bridge.to_unified(data, fmt)
                msg_type = unified.get("type", "")

                if msg_type == MessageType.HANDSHAKE:
                    payload = unified.get("payload", {})
                    _ws_client_types[id(ws)] = payload.get("client_type", "dashboard")
                    await _send_to_ws(ws, make_unified_message(
                        MessageType.TASK_ACK,
                        {"status": "ok", "client_type": _ws_client_types[id(ws)]},
                    ), persist=False)
                    continue

                if msg_type == MessageType.PING:
                    await _send_to_ws(ws, make_unified_message(MessageType.PONG, {}), persist=False)
                    continue

                legacy = _protocol_bridge.from_unified(unified, target_format="web", direction="uplink")
                legacy_type = legacy.get("type", "")
                if legacy_type == "CHAT_MESSAGE":
                    asyncio.create_task(_handle_chat_message(ws, legacy))
                elif legacy_type in ("STOP_GENERATION", "STOP_TASK"):
                    asyncio.create_task(_handle_stop_generation(legacy))
                elif msg_type == "conversation:switch":
                    asyncio.create_task(_handle_conversation_switch(unified.get("payload", {})))
                elif msg_type == "chat:visible_message":
                    asyncio.create_task(_handle_chat_visible_message(unified))
                elif msg_type == MessageType.VOICE_BIND:
                    asyncio.create_task(_handle_voice_bind(ws, unified.get("payload", {})))
                elif msg_type in (
                    MessageType.IDE_OPEN_WORKSPACE,
                    MessageType.IDE_OPEN_FILE,
                    MessageType.IDE_OPEN_TERMINAL,
                    MessageType.IDE_SHOW_DIFF,
                    MessageType.IDE_GET_CONTEXT,
                    MessageType.IDE_APPROVAL_RESULT,
                    "ide:execute_tool",
                ):
                    asyncio.create_task(_handle_ide_action(ws, msg_type, unified))
                elif legacy_type == "REQUEST_MEMORY_GRAPH":
                    asyncio.create_task(_push_memory_graph_snapshot(ws))
                elif legacy_type in ("REQUEST_MEMORY_CONTEXT_SEED", "GET_MEMORY_CONTEXT_SEED"):
                    asyncio.create_task(_handle_memory_context_seed(ws, legacy))
                elif legacy_type == "EXPAND_NODE":
                    node_id = legacy.get("node_id")
                    if node_id:
                        asyncio.create_task(_handle_expand_node(node_id, ws))
                elif legacy_type == "LIST_DIALOGUE_SESSIONS":
                    asyncio.create_task(_handle_list_dialogue_sessions(ws))
                elif legacy_type == "GET_SESSION_MESSAGES":
                    session_id = legacy.get("session_id")
                    if session_id:
                        asyncio.create_task(_handle_get_session_messages(ws, session_id))
                elif legacy_type == "DELETE_DIALOGUE_SESSION":
                    session_id = legacy.get("session_id")
                    if session_id:
                        asyncio.create_task(_handle_delete_dialogue_session(ws, session_id))
                elif legacy_type == "audio_start":
                    asyncio.create_task(_handle_audio_start_web(ws, legacy))
                elif legacy_type == "audio_chunk":
                    asyncio.create_task(_handle_audio_chunk_web(ws, legacy))
                elif legacy_type == "audio_end":
                    asyncio.create_task(_handle_audio_end_web(ws, legacy))
            except json.JSONDecodeError:
                pass
            except Exception as e:
                logger.debug(f"[WebChatRouter] 统一入口消息处理异常: {e}")
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)
        _ws_protocols.pop(id(ws), None)
        _ws_client_types.pop(id(ws), None)
        if _launcher_ready_for_memory_backfill():
            try:
                from zulong.ide.ide_server import _monitor_connections
                _monitor_connections.discard(ws)
            except Exception:
                pass


# ── WebSocket 端点 ────────────────────────────────────

@router.websocket("/ws")
async def ws_chat_endpoint(ws: WebSocket):
    """Web 聊天 WebSocket — 主系统前端通信端点"""
    global _event_loop
    await ws.accept()
    _ws_clients.add(ws)

    # 捕获事件循环引用（首次）
    if _event_loop is None:
        _event_loop = asyncio.get_running_loop()

    # 启动完成后才接入 IDE 监控集，避免选择页/Dashboard 探针提前导入重模块。
    if _launcher_ready_for_memory_backfill():
        try:
            from zulong.ide.ide_server import _monitor_connections
            _monitor_connections.add(ws)
            # 确保 ide_server 的 _main_event_loop 也被设置（Launcher 模式下 startup 不走）
            import zulong.ide.ide_server as _ide_srv
            if _ide_srv._main_event_loop is None:
                _ide_srv._main_event_loop = _event_loop
        except Exception:
            pass

    logger.info(f"[WebChatRouter] /ws 已连接 (total={len(_ws_clients)})")

    if not await _send_initial_dashboard_state(ws):
        _ws_clients.discard(ws)
        _ws_protocols.pop(id(ws), None)
        _ws_client_types.pop(id(ws), None)
        if _launcher_ready_for_memory_backfill():
            try:
                from zulong.ide.ide_server import _monitor_connections
                _monitor_connections.discard(ws)
            except Exception:
                pass
        logger.info("[WebChatRouter] /ws 初始状态发送前客户端已断开")
        return

    # 建连只推轻量摘要；完整图谱由前端在用户打开记忆图谱时按需请求。
    try:
        await _push_memory_graph_summary(ws)
    except WebSocketDisconnect:
        _ws_clients.discard(ws)
        _ws_protocols.pop(id(ws), None)
        _ws_client_types.pop(id(ws), None)
        if _launcher_ready_for_memory_backfill():
            try:
                from zulong.ide.ide_server import _monitor_connections
                _monitor_connections.discard(ws)
            except Exception:
                pass
        logger.info("[WebChatRouter] /ws 初始图谱发送前客户端已断开")
        return

    try:
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
                msg_type = data.get("type", "")

                # ── 统一协议: handshake ──
                if msg_type == "handshake":
                    _ws_protocols[id(ws)] = "unified"
                    client_type = data.get("payload", {}).get("client_type", "unknown")
                    _ws_client_types[id(ws)] = client_type
                    logger.info(f"[WebChatRouter] 统一协议握手: client_type={client_type}")
                    await _send_to_ws(ws, make_unified_message(
                        MessageType.TASK_ACK,
                        {"status": "ok", "client_type": client_type},
                    ), persist=False)
                    continue

                # ── 统一协议: 未显式握手但直接发送 namespace 类型 ──
                if ":" in msg_type and _ws_protocols.get(id(ws)) != "unified":
                    _ws_protocols[id(ws)] = "unified"
                    _ws_client_types.setdefault(id(ws), "dashboard")

                # ── 统一协议: 新格式消息 → 旧格式处理 ──
                if _ws_protocols.get(id(ws)) == "unified" and ":" in msg_type:
                    data = _protocol_bridge.from_unified(data, target_format="web", direction="uplink")
                    msg_type = data.get("type", "")

                logger.info(
                    "[WebChatRouter] /ws 收到消息: type=%s, protocol=%s",
                    msg_type,
                    _ws_protocols.get(id(ws), "legacy"),
                )

                if msg_type == "ping":
                    await _send_to_ws(ws, {"type": "pong", "ts": time.time()}, persist=False)
                elif msg_type == "CHAT_MESSAGE":
                    asyncio.create_task(_handle_chat_message(ws, data))
                elif msg_type == "STOP_GENERATION":
                    asyncio.create_task(_handle_stop_generation(data))
                elif msg_type == "STOP_TASK":
                    asyncio.create_task(_handle_stop_generation(data))
                elif msg_type == "settings_update":
                    asyncio.create_task(_handle_settings_update(ws, data.get("payload", data)))
                elif msg_type == "conversation:switch":
                    asyncio.create_task(_handle_conversation_switch(data.get("payload", data)))
                elif msg_type in ("CHAT_VISIBLE_MESSAGE", "chat:visible_message"):
                    asyncio.create_task(_handle_chat_visible_message(data))
                elif msg_type == MessageType.VOICE_BIND or msg_type == "voice_bind":
                    asyncio.create_task(_handle_voice_bind(ws, data.get("payload", data)))
                elif msg_type == MessageType.VOICE_LIST or msg_type == "voice:list":
                    asyncio.create_task(_handle_voice_list(ws, data.get("payload", data)))
                elif msg_type == MessageType.VOICE_DELETE or msg_type == "voice:delete":
                    asyncio.create_task(_handle_voice_delete(ws, data.get("payload", data)))
                elif msg_type in (
                    MessageType.IDE_OPEN_WORKSPACE,
                    MessageType.IDE_OPEN_FILE,
                    MessageType.IDE_OPEN_TERMINAL,
                    MessageType.IDE_SHOW_DIFF,
                    MessageType.IDE_GET_CONTEXT,
                    MessageType.IDE_APPROVAL_RESULT,
                    "ide_open_workspace",
                    "ide_open_file",
                    "ide_open_terminal",
                    "ide_show_diff",
                    "ide_get_context",
                    "ide_approval_result",
                    "ide:open_workspace",
                    "ide:open_file",
                    "ide:open_terminal",
                    "ide:show_diff",
                    "ide:get_context",
                    "ide:approval_result",
                    "ide:execute_tool",
                    "ide_execute_tool",
                ):
                    asyncio.create_task(_handle_ide_action(ws, msg_type, data))
                elif msg_type == "REQUEST_MEMORY_GRAPH":
                    asyncio.create_task(_push_memory_graph_snapshot(ws))
                elif msg_type in ("REQUEST_MEMORY_CONTEXT_SEED", "GET_MEMORY_CONTEXT_SEED"):
                    asyncio.create_task(_handle_memory_context_seed(ws, data))
                elif msg_type == "EXPAND_NODE":
                    node_id = data.get("node_id")
                    if node_id:
                        asyncio.create_task(_handle_expand_node(node_id, ws))
                elif msg_type == "LIST_DIALOGUE_SESSIONS":
                    asyncio.create_task(_handle_list_dialogue_sessions(ws))
                elif msg_type == "GET_SESSION_MESSAGES":
                    session_id = data.get("session_id")
                    if session_id:
                        asyncio.create_task(_handle_get_session_messages(ws, session_id))
                elif msg_type == "DELETE_DIALOGUE_SESSION":
                    session_id = data.get("session_id")
                    if session_id:
                        asyncio.create_task(_handle_delete_dialogue_session(ws, session_id))
                elif msg_type == "audio_start":
                    asyncio.create_task(_handle_audio_start_web(ws, data))
                elif msg_type == "audio_chunk":
                    asyncio.create_task(_handle_audio_chunk_web(ws, data))
                elif msg_type == "audio_end":
                    asyncio.create_task(_handle_audio_end_web(ws, data))
                else:
                    logger.warning(f"[WebChatRouter] 未处理的 /ws 消息类型: {msg_type}")
            except json.JSONDecodeError:
                logger.warning("[WebChatRouter] /ws 收到无效 JSON")
            except Exception as e:
                logger.exception(f"[WebChatRouter] 消息处理异常: {e}")
                try:
                    await _send_to_ws(ws, {
                        "type": "CHAT_RESPONSE",
                        "text": _failure_response_text(f"WebSocket 消息处理异常：{e}"),
                    })
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"[WebChatRouter] 连接异常: {e}")
    finally:
        _ws_clients.discard(ws)
        _ws_protocols.pop(id(ws), None)
        _ws_client_types.pop(id(ws), None)
        if _launcher_ready_for_memory_backfill():
            try:
                from zulong.ide.ide_server import _monitor_connections
                _monitor_connections.discard(ws)
            except Exception:
                pass
        logger.info(f"[WebChatRouter] /ws 已断开 (total={len(_ws_clients)})")


# ── 音频处理（Web 前端） ────────────────────────────────

async def _handle_audio_start_web(ws: WebSocket, data: dict):
    """处理 Web 前端音频流开始 - 触发麦克风设备录音"""
    try:
        from zulong.l0.devices.microphone_device import MicrophoneDevice
        from zulong.launcher.app import LauncherApp
        
        mic_device = None
        
        # 从 LauncherApp 全局单例获取
        try:
            import zulong.launcher.app as app_module
            if hasattr(app_module, '_app_instance') and app_module._app_instance:
                launcher_app = app_module._app_instance
                mm = launcher_app.manager
                mic_module = mm._modules.get("microphone")
                if mic_module and hasattr(mic_module, '_mic') and mic_module._mic:
                    mic_device = mic_module._mic
                    logger.debug("[WebChatRouter] 获取麦克风设备成功")
        except Exception as e:
            logger.error(f"[WebChatRouter] 获取麦克风设备失败: {e}")
        
        if mic_device and hasattr(mic_device, 'start_manual_recording'):
            mic_device.start_manual_recording()
            logger.info(f"[WebChatRouter] 手动录音已启动")
            await _send_to_ws(ws, {
                "type": "audio_start_ack",
                "ts": time.time(),
                "payload": {"status": "ok"},
            })
        else:
            logger.warning("[WebChatRouter] 麦克风设备不可用")
            await _send_to_ws(ws, {
                "type": "audio_start_ack",
                "ts": time.time(),
                "payload": {"status": "error", "message": "microphone_not_available"},
            })
    except Exception as e:
        logger.error(f"[WebChatRouter] audio_start 失败: {e}")


async def _handle_audio_chunk_web(ws: WebSocket, data: dict):
    """处理 Web 前端音频块 - 已废弃，使用麦克风设备直接采集"""
    pass


async def _handle_audio_end_web(ws: WebSocket, data: dict):
    """处理 Web 前端音频流结束 - 停止麦克风录音并触发 ASR"""
    try:
        from zulong.l0.devices.microphone_device import MicrophoneDevice
        from zulong.launcher.app import LauncherApp
        
        mic_device = None
        
        # 从 LauncherApp 获取
        try:
            import zulong.launcher.app as app_module
            if hasattr(app_module, '_app_instance') and app_module._app_instance:
                launcher_app = app_module._app_instance
                mm = launcher_app.manager
                mic_module = mm._modules.get("microphone")
                if mic_module and hasattr(mic_module, '_mic') and mic_module._mic:
                    mic_device = mic_module._mic
                    logger.debug("[WebChatRouter] 获取麦克风设备成功")
        except Exception as e:
            logger.error(f"[WebChatRouter] 获取麦克风设备失败: {e}")
        
        if mic_device and hasattr(mic_device, 'stop_manual_recording'):
            try:
                audio_data = await mic_device.stop_manual_recording()
                if audio_data is None:
                    audio_data = b""
                logger.info(f"[WebChatRouter] 手动录音结束：{len(audio_data)} bytes")
                transcript = await _transcribe_pcm16_audio(audio_data)
                if transcript.get("status") == "ok" and transcript.get("text"):
                    bind_result = get_conversation_orchestrator().route_voice_record({
                        **transcript,
                        "source": "voice_page",
                        "session_id": data.get("session_id"),
                        "conversation_id": data.get("conversation_id") or data.get("session_id"),
                    })
                    transcript["voice_event_id"] = bind_result.get("voice_event_id")
                    transcript["linked_conversation_id"] = bind_result.get("conversation_id")
                    await _send_to_ws(ws, {
                        "type": "voice:record_created",
                        "payload": bind_result,
                        "voice_event_id": bind_result.get("voice_event_id"),
                        "conversation_id": bind_result.get("conversation_id"),
                    })
                    if bind_result.get("linked"):
                        await _send_to_ws(ws, {
                            "type": "voice:record_linked",
                            "payload": bind_result,
                            "voice_event_id": bind_result.get("voice_event_id"),
                            "conversation_id": bind_result.get("conversation_id"),
                        })
                await _send_to_ws(ws, {
                    "type": "audio_transcript",
                    "ts": time.time(),
                    "payload": transcript,
                })
            except Exception as e:
                logger.error(f"[WebChatRouter] stop_manual_recording 异常: {e}")
                await _send_to_ws(ws, {
                    "type": "audio_transcript",
                    "ts": time.time(),
                    "payload": {"status": "error", "text": "", "message": str(e)},
                })
        else:
            logger.warning("[WebChatRouter] 麦克风设备不可用或方法不存在")
            await _send_to_ws(ws, {
                "type": "audio_transcript",
                "ts": time.time(),
                "payload": {"status": "error", "text": "", "message": "microphone_not_available"},
            })
    except Exception as e:
        logger.error(f"[WebChatRouter] audio_end 失败: {e}", exc_info=True)


async def _transcribe_pcm16_audio(audio_data: bytes) -> dict:
    """转写麦克风手动录音的 PCM16/16k/mono 音频。"""
    if not audio_data:
        return {"status": "error", "text": "", "message": "no_audio_data"}
    try:
        import numpy as np
        from zulong.models.audio_model_container import get_audio_model_container

        container = get_audio_model_container()
        if not getattr(container, "_initialized", False):
            try:
                from zulong.config.config_manager import ConfigManager
                cm = ConfigManager()
                sensevoice_model_path = cm.get(
                    "audio.asr.model_path",
                    "./models/OpenASR/sensevoice-small-onnx",
                )
                asr_device = cm.get("audio.asr.device", "auto")
            except Exception:
                sensevoice_model_path = "./models/OpenASR/sensevoice-small-onnx"
                asr_device = "auto"
            try:
                from zulong.utils.device import resolve_audio_model_devices
                audio_devices = resolve_audio_model_devices(asr_device, prefer_gpu=True)
            except Exception:
                audio_devices = {
                    "sensevoice": "cpu",
                    "whisper": "cpu",
                    "yamnet": "cpu",
                }
            container.initialize(
                enable_yamnet=False,
                enable_sensevoice=True,
                enable_whisper=True,
                sensevoice_device=audio_devices["sensevoice"],
                whisper_device=audio_devices["whisper"],
                sensevoice_model_path=sensevoice_model_path,
            )

        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype("float32") / 32768.0
        loop = asyncio.get_running_loop()
        start = time.perf_counter()
        result = await loop.run_in_executor(
            None,
            lambda: container.transcribe_speech(audio_np, 16000, "zh"),
        )
        elapsed = time.perf_counter() - start
        text = (getattr(result, "text", "") or "").strip()
        logger.info(
            "[WebChatRouter] ASR转写完成: len=%s, %.2fs, engine=%s",
            len(text),
            elapsed,
            getattr(result, "engine", ""),
        )
        return {
            "status": "ok",
            "text": text,
            "is_final": True,
            "engine": getattr(result, "engine", ""),
            "emotion": getattr(result, "emotion", ""),
            "language": getattr(result, "language", ""),
            "confidence": getattr(result, "confidence", 0.0),
            "elapsed": elapsed,
        }
    except Exception as e:
        logger.error(f"[WebChatRouter] ASR转写失败: {e}", exc_info=True)
        return {"status": "error", "text": "", "message": str(e)}


async def _handle_settings_update(ws: WebSocket, payload: dict) -> None:
    """Apply Web runtime settings that affect task execution."""
    try:
        from zulong.config.approval_config import set_runtime_approval_mode

        mode = payload.get("approval_mode")
        response: Dict[str, Any] = {"status": "ok"}
        if mode:
            response["approval_mode"] = set_runtime_approval_mode(str(mode))
            logger.info("[WebChatRouter] 审批模式更新: %s", response["approval_mode"])
            try:
                from zulong.ide.ide_server import broadcast_runtime_settings
                response["ide_bridges_synced"] = await broadcast_runtime_settings({
                    "approval_mode": response["approval_mode"],
                })
            except Exception as sync_exc:
                response["ide_bridges_synced"] = 0
                response["ide_sync_error"] = str(sync_exc)
                logger.debug("[WebChatRouter] IDE 审批模式同步失败: %s", sync_exc)
        await _send_to_ws(ws, {
            "type": "settings_ack",
            "payload": response,
        })
    except Exception as exc:
        logger.warning("[WebChatRouter] settings_update 失败: %s", exc)
        await _send_to_ws(ws, {
            "type": "settings_ack",
            "payload": {"status": "error", "error": str(exc)},
        })


# ── TSD 23.11.3: BFS 自恢复辅助函数 ──────────────────────────

async def _retrieve_bfs_context(conversation_id: str) -> Optional[Dict]:
    """Web 端 BFS 会话上下文检索（薄 wrapper，避免循环导入）"""
    try:
        from zulong.ide.ide_server import _retrieve_session_context
        return await _retrieve_session_context(conversation_id)
    except Exception:
        return None


def _format_bfs_context_for_prompt(bfs_ctx: Dict) -> str:
    """将 BFS 上下文格式化为 LLM system prompt 注入文本"""
    parts = []

    session_label = bfs_ctx.get("session_label", "")
    if session_label:
        parts.append(f"当前会话主题: {session_label}")

    rounds = bfs_ctx.get("recent_rounds", [])
    if rounds:
        parts.append("近期对话:")
        for r in rounds[-3:]:
            user = (r.get("user_text", "") or r.get("label", ""))[:200]
            bot = (r.get("bot_text", ""))[:200]
            parts.append(f"- 用户: {user}")
            if bot:
                parts.append(f"  助手: {bot}")

    tasks = bfs_ctx.get("active_tasks", [])
    if tasks:
        parts.append("进行中的任务:")
        for t in tasks:
            status = t.get("status", "未知")
            label = t.get("label", "")
            desc = t.get("desc", "")[:200]
            parts.append(f"- [{status}] {label}: {desc}")

    return "\n".join(parts)
