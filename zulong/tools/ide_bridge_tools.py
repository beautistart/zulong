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


class IdeWriteFileTool(BaseTool):
    """Create, overwrite, append or edit a file through the VS Code execution bridge."""

    def __init__(self):
        super().__init__(name="ide_write_file", category=ToolCategory.CODE)
        self.description = (
            "通过 VS Code 后台桥创建、覆写、追加或精准编辑宿主机文件，或创建文件夹/目录。"
            "适合用户明确要求在某个项目目录或绝对路径创建文件、文件夹、目录时使用。"
            "创建文件夹时必须设置 create_directory=true。"
            "创建新文件或完全重写已有文件用 mode='overwrite'；"
            "在文件末尾追加内容用 mode='append'；"
            "精准修改已有文件用 mode='edit' 传 edits 操作列表（修 bug、加函数、改配置），"
            "不需要传完整文件内容，编辑器内实时更新且支持撤销。"
            "长文件请按 800-1200 字符分片写入：第一片 mode=overwrite，后续 mode=append。"
            "创建新的项目根目录时不得自动回退到父目录作为 VS Code 工作区；"
            "应先由 task_create_plan 创建并绑定任务工作区，或显式传入已有 workspace_path。"
            "该工具会走 VS Code 插件的用户确认和 checkpoint。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.time()
        params = request.parameters or {}
        file_path = (
            params.get("file_path")
            or params.get("path")
            or params.get("target_path")
            or ""
        )
        name = params.get("name") or params.get("file_name") or params.get("folder_name")
        if file_path and name:
            try:
                p = Path(file_path)
                if str(p.name).lower() != str(name).lower():
                    file_path = str(p / str(name))
            except Exception:
                pass
        if not file_path:
            return self._create_result(
                success=False,
                error="file_path 不能为空",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        content = params.get("content", "")
        if content is None:
            content = ""
        content = str(content)
        write_mode = str(params.get("mode") or params.get("write_mode") or "overwrite").lower()
        if write_mode not in {"overwrite", "append", "edit"}:
            write_mode = "overwrite"
        # mode="edit"：精准编辑已有文件，传 edits 操作列表，不传 content
        edits = params.get("edits") or []
        if write_mode == "edit":
            if not edits:
                return self._create_result(
                    success=False,
                    error="mode='edit' 需要传 edits 参数（操作列表），例如 [{\"op\":\"insert\",\"line\":5,\"text\":\"import os\\n\"}]",
                    execution_time=time.time() - start,
                    request_id=request.request_id,
                )
            if content:
                # LLM 可能同时传了 content 和 edits，优先 edits，忽略 content
                content = ""
        create_directory = bool(
            params.get("create_directory")
            or params.get("is_directory")
            or params.get("directory")
            or params.get("folder")
            or str(params.get("type", "")).lower() in ("directory", "folder", "dir")
        )
        if create_directory and content:
            # 模型有时把“写文件并创建父目录”误写成 create_directory=true。
            # 只要传入了内容，就按文件写入处理，父目录由 VS Code 桥递归创建。
            create_directory = False
        workspace_path = params.get("workspace_path") or params.get("cwd") or ""
        if create_directory and not workspace_path:
            blocked = _directory_bootstrap_requires_workspace(file_path)
            if blocked:
                return self._create_result(
                    success=False,
                    error=blocked,
                    execution_time=time.time() - start,
                    request_id=request.request_id,
                )
        if not workspace_path:
            workspace_path = _infer_workspace_from_path(file_path, create_directory=create_directory)
        else:
            workspace_path = _normalize_workspace_for_target(file_path, workspace_path)
        policy_violation = _target_workspace_policy_violation(
            file_path,
            workspace_path,
            create_directory=create_directory,
        )
        if policy_violation:
            logger.warning(
                "[ide_write_file][P10] workspace policy rejected path=%s workspace=%s status=%s error=%s",
                file_path,
                workspace_path or "<empty>",
                policy_violation.get("status"),
                _safe_log_summary(policy_violation.get("error") or ""),
            )
            return self._create_result(
                success=False,
                data=policy_violation,
                error=str(policy_violation.get("error") or "workspace path policy violation"),
                execution_time=time.time() - start,
                request_id=request.request_id,
            )
        file_path, workspace_path, redirected_path, redirect_reason = _coerce_target_into_workspace(
            file_path,
            workspace_path,
            create_directory=create_directory,
        )

        tool_name = "create_directory" if create_directory else "write_to_file"
        effective_content = content
        if not create_directory and write_mode == "append":
            append_target = _resolve_ide_target_path(file_path, workspace_path, {})
            try:
                if append_target.exists():
                    if append_target.is_dir():
                        return self._create_result(
                            success=False,
                            error=f"append 目标是目录，不能写入文件: {append_target}",
                            execution_time=time.time() - start,
                            request_id=request.request_id,
                        )
                    existing = append_target.read_text(encoding="utf-8")
                else:
                    existing = ""
            except Exception as exc:
                return self._create_result(
                    success=False,
                    error=f"append 前无法读取现有文件内容: {append_target} ({exc})",
                    execution_time=time.time() - start,
                    request_id=request.request_id,
                )
            effective_content = existing + content
        # mode="edit"：走编辑器原生 API（WorkspaceEdit），不拼 content
        if write_mode == "edit":
            tool_args = {"path": file_path, "edits": edits}
            tool_name = "apply_edits"
            effective_content = ""
        elif create_directory:
            tool_args = {"path": file_path}
        else:
            tool_args = {"path": file_path, "content": effective_content}
        payload = {
            "workspace_path": workspace_path,
            "cwd": workspace_path,
            "tool_name": tool_name,
            "arguments": tool_args,
            "reason": params.get("reason", ""),
            "source": "l2_tool",
            "bridge_timeout": params.get("bridge_timeout", 180),
            "timeout": params.get("timeout", 180),
        }
        content_bytes = len(content.encode("utf-8")) if not create_directory else 0
        effective_content_bytes = len(effective_content.encode("utf-8")) if not create_directory else 0
        content_hash = (
            hashlib.sha256(effective_content.encode("utf-8")).hexdigest()[:12]
            if not create_directory
            else ""
        )
        logger.info(
            "[ide_write_file][P10] request path=%s workspace=%s tool=%s mode=%s create_directory=%s content_bytes=%s effective_content_bytes=%s content_sha256_12=%s",
            file_path,
            workspace_path or "<empty>",
            tool_name,
            write_mode,
            create_directory,
            content_bytes,
            effective_content_bytes,
            content_hash or "<none>",
        )
        if redirected_path:
            logger.warning(
                "[ide_write_file][P10] redirected target into active workspace: %s",
                redirect_reason,
            )
        if not _ide_bridge_available(workspace_path):
            if write_mode == "edit":
                return self._create_result(
                    success=False,
                    error="VS Code 后台桥未连接，mode='edit' 需要 IDE 桥支持。请改用 mode='overwrite' 或 'append'。",
                    execution_time=time.time() - start,
                    request_id=request.request_id,
                )
            result = _local_workspace_write(
                file_path=file_path,
                workspace_path=workspace_path,
                content=content,
                effective_content=effective_content,
                create_directory=create_directory,
                write_mode=write_mode,
                reason=redirect_reason or "VS Code bridge unavailable; wrote inside active task workspace",
                content_bytes=content_bytes,
                effective_content_bytes=effective_content_bytes,
                content_hash=content_hash,
            )
            logger.info(
                "[ide_write_file][P10] local fallback path=%s resolved_path=%s applied=%s verified=%s reason=%s",
                file_path,
                result.get("resolved_path", ""),
                result.get("applied"),
                result.get("verified"),
                _safe_log_summary(result.get("error") or result.get("message") or ""),
            )
            return self._create_result(
                success=bool(result.get("ok")),
                data=result,
                error=None if result.get("ok") else result.get("error", "local workspace write failed"),
                execution_time=time.time() - start,
                request_id=request.request_id,
            )
        try:
            result = _run_async_request("ide:execute_tool", payload)
            target: Path | None = None
            if _ide_result_not_applied(result):
                result = {
                    **result,
                    "ok": False,
                    "error": result.get("error") or result.get("result") or "IDE 写入未应用",
                    "verified": False,
                    "applied": False,
                }
            verified = False
            try:
                target = _resolve_ide_target_path(file_path, workspace_path, result)
                if result.get("ok"):
                    verified = target.is_dir() if create_directory else target.is_file()
                # mode="edit" 走 WorkspaceEdit API，不比较文件内容（无 effective_content）
                if result.get("ok") and verified and not create_directory and write_mode != "edit":
                    try:
                        actual = target.read_text(encoding="utf-8")
                        if actual != effective_content:
                            verified = False
                            result = {
                                **result,
                                "ok": False,
                                "error": (
                                    "IDE 工具返回成功，但目标文件内容未应用为本次写入内容: "
                                    f"{file_path}"
                                ),
                                "verified": False,
                                "applied": False,
                            }
                    except Exception:
                        pass
                if result.get("ok") and not verified:
                    applied_target = _extract_applied_path(result)
                    if applied_target:
                        verified = (
                            applied_target.is_dir()
                            if create_directory
                            else applied_target.is_file()
                        )
                        if verified:
                            result = {
                                **result,
                                "resolved_path": str(applied_target),
                            }
            except Exception:
                verified = False
            if result.get("ok") and not verified:
                result = {
                    **result,
                    "ok": False,
                    "error": (
                        "IDE 工具返回成功，但目标路径未真实存在: "
                        f"{file_path}（workspace={workspace_path or '<empty>'}）"
                    ),
                    "applied": False,
                    "verified": False,
                }
            run_async_request_is_original = (
                globals().get("_run_async_request")
                is globals().get("_ORIGINAL_RUN_ASYNC_REQUEST")
            )
            if not result.get("ok") and workspace_path and run_async_request_is_original:
                fallback = _local_workspace_write(
                    file_path=file_path,
                    workspace_path=workspace_path,
                    content=content,
                    effective_content=effective_content,
                    create_directory=create_directory,
                    write_mode=write_mode,
                    reason="IDE bridge did not apply the write; used active task workspace fallback",
                    content_bytes=content_bytes,
                    effective_content_bytes=effective_content_bytes,
                    content_hash=content_hash,
                )
                if fallback.get("ok"):
                    result = {
                        **fallback,
                        "bridge": result,
                        "status": "local_fallback_after_bridge_failure",
                    }
            else:
                result = {
                    **result,
                    "verified": verified,
                    "write_mode": write_mode,
                    "content_bytes": content_bytes,
                    "effective_content_bytes": effective_content_bytes,
                    "resolved_path": result.get("resolved_path") or (str(target) if target else ""),
                }
            logger.info(
                "[ide_write_file][P10] result path=%s resolved_path=%s bridge_ok=%s applied=%s verified=%s mode=%s content_bytes=%s effective_content_bytes=%s error=%s",
                file_path,
                result.get("resolved_path") or (str(target) if target else ""),
                bool(result.get("ok")),
                result.get("applied", result.get("ok")),
                bool(result.get("verified")),
                write_mode,
                content_bytes,
                effective_content_bytes,
                _safe_log_summary(result.get("error") or result.get("message") or result.get("result") or ""),
            )
            return self._create_result(
                success=bool(result.get("ok")),
                data=result,
                error=None if result.get("ok") else result.get("error", "IDE 写入失败"),
                execution_time=time.time() - start,
                request_id=request.request_id,
            )
        except Exception as exc:
            logger.info(
                "[ide_write_file][P10] exception path=%s workspace=%s content_bytes=%s error=%s",
                file_path,
                workspace_path or "<empty>",
                content_bytes,
                _safe_log_summary(exc),
            )
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
                "file_path": {
                    "type": "string",
                    "description": "要创建、覆写、追加或编辑的文件路径，或要创建的文件夹/目录路径。可以是绝对路径；相对路径相对于 VS Code 工作区。",
                },
                "content": {
                    "type": "string",
                    "description": "本次写入的文件内容（mode='overwrite' 或 'append' 时使用；mode='edit' 时省略）。长文件不要一次性传完整内容，单片建议 800-1200 字符。创建空文件时传空字符串。创建文件夹/目录时可省略或传空字符串。",
                },
                "mode": {
                    "type": "string",
                    "enum": ["overwrite", "append", "edit"],
                    "description": (
                        "写入模式。默认 overwrite 覆盖整个文件；append 先读取现有文件并追加后写入；"
                        "edit 通过 VS Code 编辑器原生 API 精准修改已有文件，传 edits 操作列表而不传 content，"
                        "编辑器内实时更新且支持撤销。创建新文件用 overwrite，修改已有文件用 edit，追加用 append。"
                    ),
                },
                "edits": {
                    "type": "array",
                    "description": (
                        "mode='edit' 时的编辑操作列表（mode='overwrite'/'append' 时省略）。"
                        "每项包含 op(insert/delete/replace/append) 和对应参数。"
                        "insert: {\"op\":\"insert\",\"line\":N,\"text\":\"...\"} 在第 N 行前插入。"
                        "delete: {\"op\":\"delete\",\"line\":N,\"count\":M} 从第 N 行起删 M 行。"
                        "replace: {\"op\":\"replace\",\"start_line\":S,\"end_line\":E,\"text\":\"...\"} 替换 S 到 E 行。"
                        "append: {\"op\":\"append\",\"text\":\"...\"} 追加到末尾。"
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "enum": ["insert", "delete", "replace", "append"]},
                            "line": {"type": "integer", "description": "insert/delete 的行号"},
                            "count": {"type": "integer", "description": "delete 的删除行数"},
                            "start_line": {"type": "integer", "description": "replace 的起始行"},
                            "end_line": {"type": "integer", "description": "replace 的结束行"},
                            "text": {"type": "string", "description": "要插入/替换/追加的文本"},
                        },
                    },
                },
                "workspace_path": {
                    "type": "string",
                    "description": (
                        "可选：VS Code 工作区路径。写文件时未填写会从绝对文件路径推断父目录；"
                        "创建新项目根目录时必须先由 task_create_plan 绑定工作区，或显式传入已有工作区。"
                    ),
                },
                "name": {
                    "type": "string",
                    "description": "可选：用户说“命名为...”时的文件名或文件夹名。若 file_path 是父目录，将自动拼接。",
                },
                "create_directory": {
                    "type": "boolean",
                    "description": "如果用户明确要求创建文件夹、文件夹目录、目录，必须设为 true。",
                },
                "reason": {
                    "type": "string",
                    "description": "为什么需要写入，给用户看的简短说明。",
                },
            },
            "required": ["file_path"],
        }


def _directory_bootstrap_requires_workspace(file_path: str) -> str:
    """Reject creating a missing absolute project root through a parent workspace fallback."""
    try:
        target = Path(file_path)
        if not target.is_absolute() or target.exists():
            return ""
        try:
            from zulong.tools.task_tools import get_active_workspace_dir

            active_workspace = get_active_workspace_dir()
        except Exception:
            active_workspace = ""
        if active_workspace:
            active = Path(active_workspace).resolve()
            try:
                target.resolve().relative_to(active)
                return ""
            except Exception:
                pass
        return (
            "目标目录尚不存在，ide_write_file 不会自动打开父目录来代替新项目工作区。"
            "新项目/复杂任务请先调用 task_create_plan 创建并绑定 workspace_dir；"
            "若只是要在已有工作区内创建子目录，请显式传入 workspace_path。"
        )
    except Exception:
        return ""


def _infer_workspace_from_path(file_path: str, *, create_directory: bool = False) -> str:
    try:
        p = Path(file_path)
        if p.is_absolute():
            if create_directory:
                try:
                    from zulong.tools.task_tools import get_active_workspace_dir

                    active_workspace = get_active_workspace_dir()
                except Exception:
                    active_workspace = ""
                if active_workspace:
                    active = Path(active_workspace).resolve()
                    try:
                        p.resolve().relative_to(active)
                        return str(active)
                    except Exception:
                        pass
                return str(p)
            return str(p.parent)
        try:
            from zulong.tools.task_tools import get_active_workspace_dir

            active_workspace = get_active_workspace_dir()
        except Exception:
            active_workspace = ""
        if active_workspace:
            return str(Path(active_workspace).resolve())
    except Exception:
        pass
    return ""


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
