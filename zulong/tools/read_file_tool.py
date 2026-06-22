import os
import time
from pathlib import Path
from typing import Any, Dict, List

from .base import BaseTool, ToolCategory, ToolRequest, ToolResult
from .workspace_access import normalize_workspace_path


class ReadFileTool(BaseTool):
    """只读文件工具。

    用于代码分析/配置阅读场景的安全 fallback。
    仅允许读取当前工作区内的文件，不执行任何写入。
    """

    def __init__(self):
        super().__init__(name="read_file", category=ToolCategory.CODE)
        self.description = (
            "只读读取工作区内文件内容。"
            "适用于代码分析、配置检查、日志查看等场景。"
            "当结构化代码工具暂时不可用时，可作为源码读取 fallback。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    @staticmethod
    def _same_path(left: Any, right: Any) -> bool:
        if not left or not right:
            return False
        try:
            return os.path.normcase(normalize_workspace_path(left)) == os.path.normcase(
                normalize_workspace_path(right)
            )
        except Exception:
            return str(left) == str(right)

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "要读取的文件路径，优先使用相对工作区路径。",
                },
                "start_line": {
                    "type": "integer",
                    "description": "起始行号（从 1 开始，可选）",
                },
                "end_line": {
                    "type": "integer",
                    "description": "结束行号（包含，可选）",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "最大返回行数，默认 400。",
                    "default": 400,
                },
                "workspace_path": {
                    "type": "string",
                    "description": "可选。本轮显式指定的工作区目录；提供后安全边界以该目录为准。",
                },
                "workspace_dir": {
                    "type": "string",
                    "description": "可选。workspace_path 的兼容别名。",
                },
                "cwd": {
                    "type": "string",
                    "description": "可选。当前执行目录，作为显式工作区兼容字段。",
                },
            },
            "required": ["file_path"],
        }

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        params = request.parameters or {}
        file_path = (params.get("file_path") or params.get("path") or "").strip()
        start_line = params.get("start_line")
        end_line = params.get("end_line")
        max_lines = int(params.get("max_lines") or 400)

        if not file_path:
            return self._create_result(
                success=False,
                error="file_path 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        try:
            from .task_tools import get_active_workspace_dir

            explicit_workspace = (
                params.get("workspace_path")
                or params.get("workspace_dir")
                or params.get("cwd")
                or ""
            )
            explicit_workspace = str(explicit_workspace or "").strip()
            if explicit_workspace:
                workspace = Path(
                    os.path.expandvars(os.path.expanduser(explicit_workspace))
                ).resolve()
            else:
                workspace = Path(get_active_workspace_dir() or ".").resolve()
            if not workspace.exists() or not workspace.is_dir():
                return self._create_result(
                    success=False,
                    error=f"工作区不存在或不是目录: {workspace}",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )
            candidates: List[Path] = []

            raw = Path(file_path)
            if raw.is_absolute():
                candidates.append(raw.resolve())
            else:
                candidates.append((workspace / raw).resolve())
                if not explicit_workspace:
                    candidates.append((Path(".").resolve() / raw).resolve())

            target = next((p for p in candidates if p.exists() and p.is_file()), None)
            if target is None:
                return self._create_result(
                    success=False,
                    error=f"文件不存在: {file_path}",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            try:
                workspace_norm = os.path.normcase(str(workspace))
                target_norm = os.path.normcase(str(target))
                inside_workspace = (
                    os.path.commonpath([workspace_norm, target_norm])
                    == workspace_norm
                )
            except ValueError:
                inside_workspace = False
            target_requires_external_authorization = False
            authorization_workspace = workspace
            if not inside_workspace:
                # 用户直接给出当前任务工作区外的路径时，不应让模型绕到 exec 命令。
                # 将目标文件父目录作为本轮外部文件夹访问范围；非 full_auto
                # 会等待授权，full_auto 会在 workspace_access 中权威短路放行。
                authorization_workspace = target.parent.resolve()
                target_requires_external_authorization = True
                inside_workspace = True

            active_workspace = get_active_workspace_dir() or os.getcwd()
            needs_folder_authorization = bool(
                explicit_workspace
                and not self._same_path(active_workspace, workspace)
            ) or bool(
                target_requires_external_authorization
                or (
                    raw.is_absolute()
                    and not self._same_path(active_workspace, workspace)
                )
            )
            if needs_folder_authorization:
                from .workspace_access import require_folder_access_authorization

                access = require_folder_access_authorization(
                    str(authorization_workspace),
                    current_workspace=active_workspace,
                    tool_name=self.name,
                    action_summary=(
                        f"允许祖龙访问文件夹：{authorization_workspace}\n"
                        f"待读取文件：{target.name}"
                    ),
                    conversation_id=(
                        params.get("conversation_id")
                        or params.get("session_id")
                        or ""
                    ),
                    session_id=params.get("session_id") or "",
                    request_id=params.get("request_id") or request.request_id,
                    timeout=float(params.get("approval_timeout") or 180.0),
                )
                if not access.approved:
                    return self._create_result(
                        success=False,
                        data=access.to_payload(),
                        error=access.message,
                        status_code=403,
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )

            content = target.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            total_lines = len(lines)

            s_idx = max(1, int(start_line)) if start_line is not None else 1
            e_idx = min(total_lines, int(end_line)) if end_line is not None else total_lines
            if e_idx < s_idx:
                e_idx = s_idx

            selected = lines[s_idx - 1:e_idx]
            truncated = False
            if len(selected) > max_lines:
                selected = selected[:max_lines]
                truncated = True

            numbered = [f"{idx}: {line}" for idx, line in enumerate(selected, start=s_idx)]
            snippet = "\n".join(numbered)
            if truncated:
                snippet += f"\n... (已截断，剩余 {max(0, e_idx - s_idx + 1 - len(selected))} 行)"

            return self._create_result(
                success=True,
                data={
                    "file_path": str(target),
                    "content": snippet,
                    "total_lines": total_lines,
                    "returned_start_line": s_idx,
                    "returned_end_line": s_idx + len(selected) - 1 if selected else s_idx,
                    "truncated": truncated,
                    "size_bytes": target.stat().st_size,
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )
        except Exception as e:
            return self._create_result(
                success=False,
                error=f"读取文件失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )
