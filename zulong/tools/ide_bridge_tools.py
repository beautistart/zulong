"""Tools that expose the VS Code backend bridge to L2."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict

from .base import BaseTool, ToolCategory, ToolRequest, ToolResult


class IdeOpenWorkspaceTool(BaseTool):
    """Open or switch a VS Code workspace through the Zulong IDE bridge."""

    def __init__(self):
        super().__init__(name="ide_open_workspace", category=ToolCategory.CODE)
        self.description = (
            "打开或切换 VS Code 项目文件夹。"
            "当用户要求打开某个项目目录、切换到新工作区、或任务需要先启动 VS Code 后台桥时调用。"
            "如果 VS Code 未启动，后端会先启动 VS Code；如果后台桥已连接，则通过插件打开文件夹。"
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
                    "description": "要打开的项目目录或 .code-workspace 文件路径，可以是绝对路径。",
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
        if not workspace_path:
            workspace_path = _infer_workspace_from_path(file_path)

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
            "bridge_timeout": params.get("bridge_timeout", 25),
        }
        try:
            result = _run_async_request("ide:execute_tool", payload)
            verified = False
            try:
                target = Path(file_path)
                verified = target.is_dir() if create_directory else target.is_file()
            except Exception:
                verified = False
            if result.get("ok") and not verified:
                result = {
                    **result,
                    "ok": False,
                    "error": f"IDE 工具返回成功，但目标路径未真实存在: {file_path}",
                    "verified": False,
                }
            else:
                result = {**result, "verified": verified}
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
                    "description": "可选：VS Code 工作区路径。未填写时会从绝对文件路径推断父目录。",
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


def _infer_workspace_from_path(file_path: str) -> str:
    try:
        p = Path(file_path)
        if p.is_absolute():
            return str(p.parent)
    except Exception:
        pass
    return ""


def _run_async_request(action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    from zulong.core.unified_protocol import MessageType
    from zulong.ide.ide_server import request_ide_action

    if action == "ide:open_workspace":
        action = MessageType.IDE_OPEN_WORKSPACE
    coro = request_ide_action(action, payload)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    if loop.is_running():
        # ToolEngine normally runs in worker threads. If this ever executes
        # inside a running event loop, use a temporary loop in a helper thread
        # so the sync Tool API does not deadlock.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(coro)).result(timeout=30)
    return loop.run_until_complete(coro)
