"""Runtime folder-access authorization for workspace-outside paths.

The Web runtime may receive an explicit folder path from the user that is
outside the currently active Zulong workspace.  That must not be silently
promoted into the tool safety boundary.  Instead, tools call this module to
ask the Web UI for a one-session folder authorization and then wait for the
user decision.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class FolderAccessDecision:
    approved: bool
    approval_id: str = ""
    status: str = "approved"
    workspace_path: str = ""
    message: str = ""
    timed_out: bool = False
    rejected: bool = False
    auto_approved: bool = False
    approval_mode: str = ""

    def to_payload(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "approval_required": not self.approved,
            "approval_id": self.approval_id,
            "approval_type": "folder_access",
            "status": self.status,
            "workspace_path": self.workspace_path,
            "message": self.message,
            "timed_out": self.timed_out,
            "rejected": self.rejected,
            "auto_approved": self.auto_approved,
            "approval_mode": self.approval_mode,
        }


@dataclass
class _PendingFolderApproval:
    approval_id: str
    workspace_path: str
    scope: str
    tool_name: str
    action_summary: str
    conversation_id: str = ""
    request_id: str = ""
    event: threading.Event = field(default_factory=threading.Event)
    approved: Optional[bool] = None
    resolved_payload: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


_lock = threading.RLock()
_authorized_by_scope: Dict[str, Set[str]] = {}
_pending_by_id: Dict[str, _PendingFolderApproval] = {}


def normalize_workspace_path(value: Any) -> str:
    raw = str(value or "").strip().strip('"').strip("'")
    if not raw:
        return ""
    try:
        return os.path.abspath(os.path.normpath(os.path.expanduser(os.path.expandvars(raw))))
    except Exception:
        return raw


def _path_key(value: Any) -> str:
    return os.path.normcase(normalize_workspace_path(value))


def _scope_key(conversation_id: Any = "", session_id: Any = "") -> str:
    scope = str(conversation_id or session_id or "").strip()
    return scope or "global"


def _is_inside_path(child: Any, parent: Any) -> bool:
    child_key = _path_key(child)
    parent_key = _path_key(parent)
    if not child_key or not parent_key:
        return False
    try:
        return os.path.commonpath([child_key, parent_key]) == parent_key
    except Exception:
        return child_key == parent_key


def _is_already_authorized(workspace_path: str, scope: str, current_workspace: str = "") -> bool:
    workspace_key = _path_key(workspace_path)
    current_key = _path_key(current_workspace)
    if current_key and _is_inside_path(workspace_key, current_key):
        return True
    with _lock:
        for allowed in _authorized_by_scope.get(scope, set()):
            if _is_inside_path(workspace_key, allowed):
                return True
        for allowed in _authorized_by_scope.get("global", set()):
            if _is_inside_path(workspace_key, allowed):
                return True
    return False


def grant_folder_access(
    workspace_path: Any,
    *,
    conversation_id: Any = "",
    session_id: Any = "",
) -> str:
    workspace = normalize_workspace_path(workspace_path)
    if not workspace:
        return ""
    scope = _scope_key(conversation_id, session_id)
    with _lock:
        _authorized_by_scope.setdefault(scope, set()).add(_path_key(workspace))
    logger.info("[FolderAccess] 已授权文件夹: scope=%s path=%s", scope, workspace)
    return workspace


def revoke_folder_access(
    workspace_path: Any,
    *,
    conversation_id: Any = "",
    session_id: Any = "",
) -> None:
    workspace_key = _path_key(workspace_path)
    scope = _scope_key(conversation_id, session_id)
    with _lock:
        _authorized_by_scope.get(scope, set()).discard(workspace_key)


def reset_folder_access_authorizations() -> None:
    with _lock:
        _authorized_by_scope.clear()
        _pending_by_id.clear()


def _approval_id(scope: str, workspace_path: str, tool_name: str, request_id: str = "") -> str:
    digest = hashlib.sha1(
        f"{scope}|{workspace_path}|{tool_name}|{request_id}|{time.time()}".encode("utf-8", "replace")
    ).hexdigest()[:12]
    return f"folder_access:{digest}"


def _notify_folder_access_required(pending: _PendingFolderApproval) -> None:
    payload = {
        "approval_id": pending.approval_id,
        "approval_type": "folder_access",
        "status": "pending",
        "approved": None,
        "tool_name": pending.tool_name or "folder_access",
        "action_summary": pending.action_summary,
        "summary": pending.action_summary,
        "workspace_path": pending.workspace_path,
        "cwd": pending.workspace_path,
        "risk_level": "HIGH",
        "risk_reason": "该文件夹不在当前任务工作区内，允许后祖龙才能读取其中的文件。",
        "approval_mode": "manual",
        "conversation_id": pending.conversation_id,
        "session_id": pending.conversation_id,
        "request_id": pending.request_id,
        "message": pending.action_summary,
        "interaction": {
            "interaction_id": pending.approval_id,
            "pair_id": pending.approval_id,
            "kind": "approval",
            "status": "pending",
            "title": "需要文件夹访问授权",
            "detail": pending.action_summary,
            "tool_name": pending.tool_name or "folder_access",
            "risk_level": "HIGH",
            "approval_id": pending.approval_id,
            "approval_type": "folder_access",
            "ux_visibility": "main",
            "source_channel": "system_status",
        },
    }
    try:
        from zulong.launcher.web_chat_router import update_task_execution_status

        update_task_execution_status(
            state="waiting_approval",
            phase="approval_required",
            message=pending.action_summary,
            request_id=pending.request_id,
            conversation_id=pending.conversation_id,
            session_id=pending.conversation_id,
            workspace_path=pending.workspace_path,
            project_id=pending.workspace_path,
            tool_name=pending.tool_name or "folder_access",
            awaiting_approval=True,
            approval=payload,
            progress_items=[{
                "label": "等待文件夹访问授权",
                "status": "running",
                "source": "approval",
                "detail": pending.workspace_path,
                "timestamp": time.time(),
            }],
        )
    except Exception as exc:
        logger.debug("[FolderAccess] 任务状态通知跳过: %s", exc)

    try:
        from zulong.core.message_visibility import CHANNEL_STATUS, mark_public_payload
        from zulong.launcher.web_chat_router import _schedule_broadcast

        message = mark_public_payload({
            "type": "IDE_APPROVAL_STATUS",
            "payload": payload,
            "session_id": pending.conversation_id,
            "conversation_id": pending.conversation_id,
            "request_id": pending.request_id,
            "ts": time.time(),
        }, CHANNEL_STATUS)
        _schedule_broadcast(message)
    except Exception as exc:
        logger.debug("[FolderAccess] 审批弹窗广播跳过: %s", exc)


def require_folder_access_authorization(
    workspace_path: Any,
    *,
    current_workspace: Any = "",
    tool_name: str = "folder_access",
    action_summary: str = "",
    conversation_id: Any = "",
    session_id: Any = "",
    request_id: Any = "",
    timeout: float = 180.0,
) -> FolderAccessDecision:
    workspace = normalize_workspace_path(workspace_path)
    if not workspace:
        return FolderAccessDecision(
            approved=False,
            status="invalid_workspace",
            message="缺少需要授权的文件夹路径。",
        )
    if not os.path.isdir(workspace):
        return FolderAccessDecision(
            approved=False,
            status="invalid_workspace",
            workspace_path=workspace,
            message=f"需要授权的路径不是文件夹: {workspace}",
        )

    scope = _scope_key(conversation_id, session_id)
    current = normalize_workspace_path(current_workspace)
    if _is_already_authorized(workspace, scope, current):
        return FolderAccessDecision(
            approved=True,
            status="approved",
            workspace_path=workspace,
            message="文件夹已在当前任务授权范围内。",
        )

    runtime_mode = ""
    runtime_auto_approved = False
    try:
        from zulong.config.approval_config import (
            get_runtime_approval_mode,
            should_runtime_auto_approve,
        )

        runtime_mode = str(get_runtime_approval_mode() or "").strip().lower()
        runtime_auto_approved = should_runtime_auto_approve(
            tool_name or "folder_access",
            {
                "path": workspace,
                "workspace_path": workspace,
                "cwd": workspace,
            },
            risk_level="HIGH",
        )
    except Exception as exc:
        logger.debug("[FolderAccess] runtime approval mode unavailable: %s", exc)
    if runtime_auto_approved:
        auto_approval_id = _approval_id(scope, workspace, tool_name, str(request_id or "auto"))
        grant_folder_access(
            workspace,
            conversation_id=conversation_id,
            session_id=session_id,
        )
        _notify_folder_access_resolved(
            approval_id=auto_approval_id,
            approved=True,
            workspace_path=workspace,
            conversation_id=conversation_id or session_id or "",
            request_id=request_id or "",
            tool_name=tool_name or "folder_access",
            action_summary=action_summary
            or f"允许祖龙访问文件夹：{workspace}",
        )
        logger.info(
            "[FolderAccess] full_auto auto-approves folder access: scope=%s path=%s tool=%s",
            scope,
            workspace,
            tool_name,
        )
        return FolderAccessDecision(
            approved=True,
            approval_id=auto_approval_id,
            status="approved",
            workspace_path=workspace,
            message=(
                "完全访问模式已自动允许文件夹访问。"
                if runtime_mode == "full_auto"
                else "白名单模式已自动允许文件夹访问。"
            ),
            auto_approved=True,
            approval_mode=runtime_mode or "auto",
        )

    summary = action_summary or f"允许祖龙访问文件夹：{workspace}"
    approval_id = _approval_id(scope, workspace, tool_name, str(request_id or ""))
    pending = _PendingFolderApproval(
        approval_id=approval_id,
        workspace_path=workspace,
        scope=scope,
        tool_name=tool_name,
        action_summary=summary,
        conversation_id=str(conversation_id or session_id or ""),
        request_id=str(request_id or ""),
    )
    with _lock:
        _pending_by_id[approval_id] = pending

    logger.info(
        "[FolderAccess] 等待文件夹授权: approval_id=%s scope=%s path=%s tool=%s",
        approval_id,
        scope,
        workspace,
        tool_name,
    )
    _notify_folder_access_required(pending)

    approved = pending.event.wait(max(0.0, float(timeout or 0.0)))
    with _lock:
        _pending_by_id.pop(approval_id, None)
    if not approved:
        return FolderAccessDecision(
            approved=False,
            approval_id=approval_id,
            status="timeout",
            workspace_path=workspace,
            message="等待文件夹访问授权超时，已停止本次文件访问。",
            timed_out=True,
        )
    if pending.approved:
        grant_folder_access(workspace, conversation_id=scope)
        return FolderAccessDecision(
            approved=True,
            approval_id=approval_id,
            status="approved",
            workspace_path=workspace,
            message="文件夹访问已授权。",
        )
    return FolderAccessDecision(
        approved=False,
        approval_id=approval_id,
        status="rejected",
        workspace_path=workspace,
        message="用户拒绝了文件夹访问授权。",
        rejected=True,
    )


def handle_folder_access_approval_result(payload: Dict[str, Any]) -> bool:
    payload = dict(payload or {})
    approval_type = str(payload.get("approval_type") or payload.get("type") or "").lower()
    approval_id = str(
        payload.get("approval_id")
        or payload.get("approvalId")
        or payload.get("pair_id")
        or ""
    ).strip()
    if approval_type != "folder_access" and not approval_id.startswith("folder_access:"):
        return False

    approved = bool(payload.get("approved"))
    workspace = normalize_workspace_path(payload.get("workspace_path") or payload.get("cwd") or "")
    pending: Optional[_PendingFolderApproval] = None
    with _lock:
        pending = _pending_by_id.get(approval_id)
        if pending:
            pending.approved = approved
            pending.resolved_payload = payload
            workspace = workspace or pending.workspace_path
            if approved:
                _authorized_by_scope.setdefault(pending.scope, set()).add(_path_key(workspace))
            pending.event.set()
        elif approved and workspace:
            grant_folder_access(
                workspace,
                conversation_id=payload.get("conversation_id") or payload.get("session_id") or "",
            )

    _notify_folder_access_resolved(
        approval_id=approval_id,
        approved=approved,
        workspace_path=workspace,
        conversation_id=(
            payload.get("conversation_id")
            or payload.get("session_id")
            or (pending.conversation_id if pending else "")
        ),
        request_id=payload.get("request_id") or (pending.request_id if pending else ""),
        tool_name=payload.get("tool_name") or (pending.tool_name if pending else "folder_access"),
        action_summary=payload.get("action_summary") or (pending.action_summary if pending else ""),
    )
    logger.info(
        "[FolderAccess] 文件夹授权结果: approval_id=%s approved=%s path=%s",
        approval_id or "-",
        approved,
        workspace or "-",
    )
    return True


def _notify_folder_access_resolved(
    *,
    approval_id: str,
    approved: bool,
    workspace_path: str,
    conversation_id: Any = "",
    request_id: Any = "",
    tool_name: str = "folder_access",
    action_summary: str = "",
) -> None:
    payload = {
        "approval_id": approval_id,
        "approval_type": "folder_access",
        "status": "approved" if approved else "rejected",
        "approved": approved,
        "tool_name": tool_name or "folder_access",
        "action_summary": action_summary or ("文件夹访问已授权" if approved else "文件夹访问已拒绝"),
        "workspace_path": workspace_path,
        "cwd": workspace_path,
        "risk_level": "HIGH",
        "confirmation_state": "approved" if approved else "rejected",
        "conversation_id": str(conversation_id or ""),
        "session_id": str(conversation_id or ""),
        "request_id": str(request_id or ""),
        "message": "文件夹访问已授权，任务继续执行。" if approved else "文件夹访问已拒绝，任务已暂停。",
    }
    try:
        from zulong.launcher.web_chat_router import update_task_execution_status

        update_task_execution_status(
            state="running" if approved else "blocked",
            phase="approval_resolved",
            message=payload["message"],
            request_id=str(request_id or ""),
            conversation_id=str(conversation_id or ""),
            session_id=str(conversation_id or ""),
            workspace_path=workspace_path,
            project_id=workspace_path,
            tool_name=tool_name or "folder_access",
            awaiting_approval=False,
            approval=payload,
            progress_items=[{
                "label": "文件夹访问已授权" if approved else "文件夹访问已拒绝",
                "status": "completed" if approved else "blocked",
                "source": "approval",
                "detail": workspace_path,
                "timestamp": time.time(),
            }],
        )
    except Exception as exc:
        logger.debug("[FolderAccess] 授权结果状态通知跳过: %s", exc)

    try:
        from zulong.core.message_visibility import CHANNEL_STATUS, mark_public_payload
        from zulong.launcher.web_chat_router import _schedule_broadcast

        _schedule_broadcast(mark_public_payload({
            "type": "IDE_APPROVAL_STATUS",
            "payload": payload,
            "session_id": str(conversation_id or ""),
            "conversation_id": str(conversation_id or ""),
            "request_id": str(request_id or ""),
            "ts": time.time(),
        }, CHANNEL_STATUS))
    except Exception as exc:
        logger.debug("[FolderAccess] 授权结果广播跳过: %s", exc)


def should_treat_as_folder_access_approval(payload: Dict[str, Any]) -> bool:
    payload = dict(payload or {})
    approval_type = str(payload.get("approval_type") or payload.get("type") or "").lower()
    approval_id = str(
        payload.get("approval_id")
        or payload.get("approvalId")
        or payload.get("pair_id")
        or ""
    ).strip()
    return approval_type == "folder_access" or approval_id.startswith("folder_access:")
