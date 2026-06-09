"""Tools that expose the VS Code backend bridge to L2."""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any, Dict

from .base import BaseTool, ToolCategory, ToolRequest, ToolResult


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


def _nearest_existing_parent(path: Path) -> Path | None:
    current = path
    while current and current != current.parent:
        if current.is_dir():
            return current
        current = current.parent
    return None


def _normalize_workspace_for_target(file_path: str, workspace_path: str) -> str:
    """Keep explicit paths inside the nearest existing parent workspace.

    Models sometimes pass the not-yet-created target directory as
    workspace_path. For file creation, the bridge should use the nearest
    existing parent, while still rejecting unrelated implicit fallbacks.
    """
    try:
        target = Path(file_path)
        workspace = Path(workspace_path) if workspace_path else None
        if not target.is_absolute() or not workspace or workspace.exists():
            return workspace_path
        parent = _nearest_existing_parent(workspace)
        if not parent:
            return workspace_path
        try:
            target.resolve().relative_to(parent.resolve())
            return str(parent)
        except Exception:
            return workspace_path
    except Exception:
        return workspace_path


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
    """Create a file or directory through the VS Code execution bridge."""

    def __init__(self):
        super().__init__(name="ide_write_file", category=ToolCategory.CODE)
        self.description = (
            "通过 VS Code 后台桥创建或覆写宿主机文件，或创建文件夹/目录。"
            "适合用户明确要求在某个项目目录或绝对路径创建文件、文件夹、目录时使用。"
            "创建文件夹时必须设置 create_directory=true。"
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

        tool_name = "create_directory" if create_directory else "write_to_file"
        tool_args: Dict[str, Any] = (
            {"path": file_path}
            if create_directory
            else {"path": file_path, "content": content}
        )
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
        try:
            result = _run_async_request("ide:execute_tool", payload)
            verified = False
            try:
                target = _resolve_ide_target_path(file_path, workspace_path, result)
                verified = target.is_dir() if create_directory else target.is_file()
                if not verified:
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
                    "verified": False,
                }
            else:
                result = {
                    **result,
                    "verified": verified,
                    "resolved_path": result.get("resolved_path") or str(target),
                }
            return self._create_result(
                success=bool(result.get("ok")),
                data=result,
                error=None if result.get("ok") else result.get("error", "IDE 写入失败"),
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
                "file_path": {
                    "type": "string",
                    "description": "要创建或覆写的文件路径，或要创建的文件夹/目录路径。可以是绝对路径；相对路径相对于 VS Code 工作区。",
                },
                "content": {
                    "type": "string",
                    "description": "文件内容。创建空文件时传空字符串。创建文件夹/目录时可省略或传空字符串。",
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
