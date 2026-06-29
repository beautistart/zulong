"""Tools that expose the VS Code backend bridge to L2."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Tuple

from .base import BaseTool, ToolCategory, ToolRequest, ToolResult

logger = logging.getLogger(__name__)


def _safe_log_summary(value: Any, max_len: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _resolve_ide_target_path(file_path: str, workspace_path: str, result: Dict[str, Any]) -> Path:
    """Resolve the real filesystem target used by the IDE bridge."""
    target = Path(file_path)
    if target.is_absolute():
        return target

    base = (
        result.get("workspace_path")
        or result.get("workspace")
        or result.get("cwd")
        or workspace_path
        or ""
    )
    if base:
        return Path(str(base)) / target
    return target


def _extract_applied_path(result: Dict[str, Any]) -> Path | None:
    """Best-effort parse of bridge messages such as '已应用文件变更: C:\\x\\a.py'."""
    haystack = " ".join(
        str(result.get(key) or "")
        for key in ("result", "message", "detail", "output")
    )
    if not haystack:
        return None
    match = re.search(r"([A-Za-z]:\\[^\r\n]+)", haystack)
    if not match:
        return None
    return Path(match.group(1).strip().strip('"'))


def _ide_result_not_applied(result: Dict[str, Any]) -> bool:
    """Return True when the IDE bridge reports an unapplied protected change."""
    haystack = " ".join(
        str(result.get(key) or "")
        for key in ("result", "error", "message", "detail")
    ).lower()
    if not haystack:
        return False
    markers = (
        "用户未应用",
        "用户未允许",
        "用户拒绝",
        "审批拒绝",
        "审批超时",
        "审批未通过",
        "未应用写入",
        "未允许",
    )
    return any(marker.lower() in haystack for marker in markers)


def _nearest_existing_parent(path: Path) -> Path | None:
    current = path
    while current and current != current.parent:
        if current.is_dir():
            return current
        current = current.parent
    return None


def _normalize_workspace_for_target(file_path: str, workspace_path: str) -> str:
    """Preserve the explicit task workspace selected by L2/TaskGraph.

    Previous logic replaced a not-yet-created workspace with its nearest
    existing parent. That made writes appear under a parent directory and broke
    TSD v2.9.6/v2.9.8's "current TaskGraph workspace is the sole task
    directory" invariant. Creation paths may create the explicit workspace; the
    framework must not silently reinterpret it as a parent.
    """
    return workspace_path


def _active_workspace() -> str:
    try:
        from zulong.tools.task_tools import get_active_workspace_dir

        active = get_active_workspace_dir()
    except Exception:
        active = ""
    if active:
        try:
            return str(Path(str(active)).resolve())
        except Exception:
            return str(active)
    return ""


def _normalize_optional_path(value: str) -> str:
    if not value:
        return ""
    try:
        return str(Path(str(value)).resolve())
    except Exception:
        return str(value)


def _workspace_binding_state(workspace_path: str) -> Dict[str, str | bool]:
    """Resolve active/explicit workspace without silently choosing on conflict."""
    explicit = _normalize_optional_path(workspace_path)
    active = _active_workspace()
    conflict = False
    if active and explicit:
        try:
            conflict = Path(active).resolve() != Path(explicit).resolve()
        except Exception:
            conflict = active != explicit
    return {
        "active": active,
        "explicit": explicit,
        "workspace": explicit or active,
        "conflict": conflict,
    }


def _active_or_explicit_workspace(workspace_path: str) -> str:
    state = _workspace_binding_state(workspace_path)
    return str(state.get("workspace") or "")


def _target_workspace_policy_violation(
    file_path: str,
    workspace_path: str,
    *,
    create_directory: bool,
) -> Dict[str, Any] | None:
    state = _workspace_binding_state(workspace_path)
    if state.get("conflict"):
        return {
            "ok": False,
            "status": "workspace_conflict",
            "error": (
                "工具请求的 workspace_path 与当前 TaskGraph 工作区不一致，"
                "已拒绝执行以避免写入错误目录。请先由 L2 修正 TaskGraph 绑定或调整工具参数。"
            ),
            "active_workspace": state.get("active") or "",
            "requested_workspace": state.get("explicit") or "",
            "workspace_path": state.get("workspace") or "",
            "file_path": file_path,
            "applied": False,
            "verified": False,
        }
    workspace = str(state.get("workspace") or "")
    if not workspace:
        return None
    try:
        workspace_resolved = Path(workspace).resolve()
        target = Path(file_path)
        if not target.is_absolute():
            return None
        target_resolved = target.resolve()
        try:
            target_resolved.relative_to(workspace_resolved)
            return None
        except Exception:
            pass
        return {
            "ok": False,
            "status": "path_outside_workspace",
            "error": (
                "目标路径不在当前任务工作区内，已拒绝静默改写或压扁路径。"
                "请由 L2 重新确认任务工作区，或把 file_path 改为工作区内相对路径。"
            ),
            "workspace_path": str(workspace_resolved),
            "resolved_path": str(target_resolved),
            "file_path": str(target_resolved),
            "applied": False,
            "verified": False,
        }
    except Exception:
        return None
    return None


def _coerce_target_into_workspace(
    file_path: str,
    workspace_path: str,
    *,
    create_directory: bool,
) -> Tuple[str, str, bool, str]:
    """Normalize only safe in-workspace targets; never basename-redirect.

    Older behavior silently rewrote an absolute path outside the active task
    workspace to its basename inside the workspace. That violated the TSD
    workspace binding contract and flattened nested project files.  The
    separate policy validator now returns a structured error for such cases.
    """
    workspace = _active_or_explicit_workspace(workspace_path)
    if not workspace:
        return file_path, workspace_path, False, ""
    try:
        workspace_resolved = Path(workspace).resolve()
        target = Path(file_path)
        if not target.is_absolute():
            return file_path, str(workspace_resolved), False, ""
        target_resolved = target.resolve()
        target_resolved.relative_to(workspace_resolved)
        return str(target_resolved), str(workspace_resolved), False, ""
    except Exception:
        return file_path, str(workspace) if workspace else workspace_path, False, ""


def _ide_bridge_available(workspace_path: str) -> bool:
    # Unit tests replace _run_async_request with a fake bridge.  Treat that as
    # an available bridge so verification paths still exercise IDE result
    # handling instead of the no-bridge local fallback.
    original = globals().get("_ORIGINAL_RUN_ASYNC_REQUEST")
    if original is not None and globals().get("_run_async_request") is not original:
        return True
    try:
        from zulong.ide import ide_server

        selector = getattr(ide_server, "_select_ide_bridge", None)
        if callable(selector):
            return bool(selector(workspace_path))
    except Exception:
        pass
    return False


def _local_workspace_write(
    *,
    file_path: str,
    workspace_path: str,
    content: str,
    effective_content: str,
    create_directory: bool,
    write_mode: str,
    reason: str,
    content_bytes: int,
    effective_content_bytes: int,
    content_hash: str,
) -> Dict[str, Any]:
    workspace = _active_or_explicit_workspace(workspace_path)
    if not workspace:
        return {
            "ok": False,
            "status": "workspace_required",
            "error": "workspace_path is required for local Web write fallback",
            "applied": False,
            "verified": False,
        }
    try:
        workspace_resolved = Path(workspace).resolve()
        workspace_resolved.mkdir(parents=True, exist_ok=True)
        target = _resolve_ide_target_path(file_path, str(workspace_resolved), {})
        target = target.resolve()
        try:
            target.relative_to(workspace_resolved)
        except Exception:
            return {
                "ok": False,
                "status": "path_outside_workspace",
                "error": f"target path must stay inside the active task workspace: {target}",
                "workspace_path": str(workspace_resolved),
                "resolved_path": str(target),
                "applied": False,
                "verified": False,
            }
        if create_directory:
            target.mkdir(parents=True, exist_ok=True)
            verified = target.is_dir()
            actual_bytes = 0
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(effective_content, encoding="utf-8")
            verified = target.is_file() and target.read_text(encoding="utf-8") == effective_content
            actual_bytes = target.stat().st_size if target.exists() else -1
        return {
            "ok": bool(verified),
            "status": "local_workspace_write",
            "message": reason or "written by Web task local workspace fallback",
            "workspace_path": str(workspace_resolved),
            "cwd": str(workspace_resolved),
            "resolved_path": str(target),
            "file_path": str(target),
            "write_mode": write_mode,
            "content_bytes": content_bytes,
            "effective_content_bytes": effective_content_bytes,
            "content_sha256_12": content_hash,
            "actual_bytes": actual_bytes,
            "applied": bool(target.exists()),
            "verified": bool(verified),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "local_workspace_write_failed",
            "error": str(exc),
            "workspace_path": workspace,
            "applied": False,
            "verified": False,
        }


def _local_apply_edits(
    *,
    file_path: str,
    edits: list,
    workspace_path: str,
    reason: str = "",
) -> Dict[str, Any]:
    """本地执行 edit 模式行操作（VS Code 桥未连接时的静默兜底）。

    读取文件内容按行分割，逐条应用 insert/delete/replace/append 操作，写回文件。
    行为等价于 VS Code WorkspaceEdit，但不走编辑器 API。
    """
    workspace = _active_or_explicit_workspace(workspace_path)
    if not workspace:
        return {
            "ok": False,
            "status": "workspace_required",
            "error": "workspace_path is required for local edit fallback",
            "applied": False,
            "verified": False,
        }
    try:
        workspace_resolved = Path(workspace).resolve()
        target = _resolve_ide_target_path(file_path, str(workspace_resolved), {})
        target = target.resolve()
        try:
            target.relative_to(workspace_resolved)
        except Exception:
            return {
                "ok": False,
                "status": "path_outside_workspace",
                "error": f"target path must stay inside the active task workspace: {target}",
                "workspace_path": str(workspace_resolved),
                "applied": False,
                "verified": False,
            }
        if not target.exists():
            return {
                "ok": False,
                "status": "file_not_found",
                "error": f"edit 模式要求文件已存在，但文件不存在: {target}",
                "applied": False,
                "verified": False,
            }
        # 读取文件内容
        original = target.read_text(encoding="utf-8")
        lines = original.split("\n")
        summary_parts = []

        for e in edits:
            op = str(e.get("op", "")).lower()
            if op == "insert":
                line_idx = max(0, int(e.get("line", 1)) - 1)
                text = str(e.get("text", ""))
                new_lines = text.split("\n")
                lines[line_idx:line_idx] = new_lines
                summary_parts.append(f"L{line_idx + 1} +{len(new_lines)}行")
            elif op == "delete":
                line_idx = max(0, int(e.get("line", 1)) - 1)
                count = max(1, int(e.get("count", 1)))
                del lines[line_idx:line_idx + count]
                summary_parts.append(f"L{line_idx + 1} -{count}行")
            elif op == "replace":
                sl = max(0, int(e.get("start_line", 1)) - 1)
                el = max(sl, int(e.get("end_line", sl + 1)) - 1)
                text = str(e.get("text", ""))
                new_lines = text.split("\n")
                lines[sl:el + 1] = new_lines
                summary_parts.append(f"L{sl + 1}-{el + 1} →{len(new_lines)}行")
            elif op == "append":
                text = str(e.get("text", ""))
                lines.append("")  # 确保末尾换行
                lines.extend(text.split("\n"))
                summary_parts.append(f"末尾 +{len(text.split(chr(10)))}行")

        next_content = "\n".join(lines)
        summary = ", ".join(summary_parts) or f"{len(edits)} 个编辑操作"
        target.write_text(next_content, encoding="utf-8")
        logger.info("[exec_write_file][edit] local apply_edits: %s (%s)", target, summary)
        return {
            "ok": True,
            "status": "local_edit",
            "result": f"已本地应用 {summary}: {target}",
            "workspace_path": str(workspace_resolved),
            "resolved_path": str(target),
            "applied": True,
            "verified": True,
            "edit_count": len(edits),
            "summary": summary,
            "original": original,
            "next": next_content,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "local_edit_failed",
            "error": str(exc),
            "applied": False,
            "verified": False,
        }


def _broadcast_local_file_change(
    *,
    file_path: str,
    operation: str,
    original: str,
    next: str,
    summary: str = "",
) -> None:
    """本地 edit/写入成功后，向 Web 端推送 file_changed + diff 事件。

    复用 web_chat_router 的 WebSocket 广播通道，让 Web 端 diff 面板能显示本地写入的变更。
    """
    try:
        from zulong.launcher import web_chat_router

        path = str(file_path)
        # 截断过长的 diff 内容（避免 WebSocket 消息过大）
        max_chars = 20000
        orig_trunc = original[:max_chars] if len(original) > max_chars else original
        next_trunc = next[:max_chars] if len(next) > max_chars else next

        # 推送 IDE_FILE_CHANGED（文件变更通知，触发 Web diff 面板更新）
        web_chat_router._schedule_broadcast({
            "type": "IDE_FILE_CHANGED",
            "payload": {
                "workspace_path": "",
                "path": path,
                "operation": operation,
                "original": orig_trunc,
                "next": next_trunc,
                "summary": summary,
            },
        })
    except Exception as exc:
        logger.debug("[exec_write_file] 本地变更广播失败（不影响写入结果）: %s", exc)


class IdeOpenWorkspaceTool(BaseTool):
    """Open or switch a VS Code workspace through the Zulong IDE bridge."""

    def __init__(self):
        super().__init__(name="ide_open_workspace", category=ToolCategory.CODE)
        self.description = (
            "打开或切换当前任务绑定的 VS Code 工作区。"
            "仅当用户明确要求打开/切换 VS Code，或已确认需要用户查看当前任务代码时调用；"
            "普通任务创建与后台执行不得为了预热桥接而自动弹出前台 VS Code。"
            "workspace_path 必须是当前 TaskGraph 绑定的任务目录，缺失时失败，不允许回退到祖龙源码目录。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.time()
        params = request.parameters or {}
        workspace_path = (
            params.get("workspace_path")
            or params.get("folder_path")
            or params.get("path")
            or ""
        )
        if not workspace_path:
            return self._create_result(
                success=False,
                error="workspace_path 不能为空",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        payload = {
            "workspace_path": workspace_path,
            "cwd": workspace_path,
            "reason": params.get("reason", ""),
            "vscode_command": params.get("vscode_command") or params.get("vscode_path") or "",
            "source": "l2_tool",
        }
        try:
            result = _run_async_request("ide:open_workspace", payload)
            return self._create_result(
                success=bool(result.get("ok")),
                data=result,
                error=None if result.get("ok") else result.get("error", "打开 VS Code 工作区失败"),
                execution_time=time.time() - start,
                request_id=request.request_id,
            )
        except Exception as exc:
            return self._create_result(
                success=False,
                error=str(exc),
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "workspace_path": {
                    "type": "string",
                    "description": "当前 TaskGraph 绑定的任务目录或 .code-workspace 文件路径，必须明确提供。",
                },
                "reason": {
                    "type": "string",
                    "description": "为什么需要打开或切换这个 VS Code 工作区，给用户看的简短说明。",
                },
                "vscode_command": {
                    "type": "string",
                    "description": "可选：VS Code 启动命令或完整路径。留空时后端自动从设置和 PATH 查找。",
                },
            },
            "required": ["workspace_path"],
        }


def _run_async_request(action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    from zulong.core.unified_protocol import MessageType
    from zulong.ide import ide_server

    if action == "ide:open_workspace":
        action = MessageType.IDE_OPEN_WORKSPACE
    timeout = float(payload.get("timeout") or payload.get("bridge_timeout") or 180)
    wait_timeout = max(30.0, timeout + 15.0)
    coro = ide_server.request_ide_action(action, payload)
    main_loop = getattr(ide_server, "_main_event_loop", None) or getattr(ide_server, "_main_loop", None)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if main_loop and main_loop.is_running():
        if loop is main_loop:
            coro.close()
            return {
                "ok": False,
                "error": "IDE 桥同步工具不能在主事件循环线程内阻塞等待。",
            }
        future = asyncio.run_coroutine_threadsafe(coro, main_loop)
        return future.result(timeout=wait_timeout)
    if loop and loop.is_running():
        coro.close()
        return {
            "ok": False,
            "error": "未找到 IDE 主事件循环，无法安全等待 VS Code 工具结果。",
        }
    if loop:
        return loop.run_until_complete(coro)
    return asyncio.run(coro)


_ORIGINAL_RUN_ASYNC_REQUEST = _run_async_request
