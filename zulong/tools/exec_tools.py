# File: zulong/tools/exec_tools.py

# 执行 FC 工具集 — 让模型通过 Function Calling 自主执行文件写入和命令

#

# 2 个工具:

# - exec_write_file: 安全写入文件

# - exec_run_command: 安全执行命令



import logging
import os
import time
import subprocess
import signal
import hashlib
import shutil
import re
from typing import Dict, Any
from pathlib import Path

from .base import BaseTool, ToolCategory, ToolRequest, ToolResult
from zulong.utils.runtime_env import get_runtime_environment


logger = logging.getLogger(__name__)



# 工作区根目录（安全边界）
WORKSPACE_DIR = os.environ.get("ZULONG_WORKSPACE", "./agent_workspace")
# 命令白名单
COMMON_COMMAND_WHITELIST = {
    "python", "python3", "node", "npm", "npx", "pip", "pip3",
    "pytest", "pnpm", "yarn",
    "echo", "mkdir", "cd", "pwd",
    "tree", "git", "cargo", "go", "javac", "java",
    # 扩展: 常用开发工具
    "curl", "wget", "tar", "unzip", "zip",
    "pdflatex", "make", "cmake", "gcc", "g++",
    "ffmpeg", "imagemagick", "convert",
    "docker", "docker-compose", "kubectl",
    "npx", "tsc", "eslint", "prettier",

    "http-server", "live-server",
}
WINDOWS_COMMAND_WHITELIST = {
    "dir", "type", "copy", "move", "del", "where",
    "get-childitem", "select-string", "get-content", "set-content", "new-item",
    # PowerShell 常用别名/命令（LLM 常用 Set-Location 代替 cd 等）
    "set-location", "get-location", "set-variable", "get-variable",
    "remove-item", "test-path", "copy-item", "move-item",
    "write-output", "write-host", "out-string",
    "invoke-expression", "start-process",
}
POSIX_COMMAND_WHITELIST = {
    "cat", "ls", "cp", "mv", "rm", "touch", "chmod", "chown",
    "find", "grep", "head", "tail", "wc", "sort", "diff", "file", "which", "whereis",
}

SUPPORTED_COMMAND_SHELLS = {"auto", "cmd", "powershell", "git_bash"}
SHELL_ALIASES = {
    "": "auto",
    "default": "auto",
    "system": "auto",
    "windows_cmd": "cmd",
    "cmd.exe": "cmd",
    "command_prompt": "cmd",
    "pwsh": "powershell",
    "powershell.exe": "powershell",
    "ps": "powershell",
    "bash": "git_bash",
    "gitbash": "git_bash",
    "git-bash": "git_bash",
    "git_bash": "git_bash",
}


def _command_whitelist_for_current_platform() -> set:
    env = get_runtime_environment()
    if env.os_family == "windows":
        return COMMON_COMMAND_WHITELIST | WINDOWS_COMMAND_WHITELIST
    return COMMON_COMMAND_WHITELIST | POSIX_COMMAND_WHITELIST


def _command_whitelist_for_shell(shell: str) -> set:
    if shell == "git_bash":
        return COMMON_COMMAND_WHITELIST | POSIX_COMMAND_WHITELIST
    return _command_whitelist_for_current_platform()


def _normalize_shell(value: Any) -> str:
    shell = str(value or "auto").strip().lower()
    return SHELL_ALIASES.get(shell, shell)


def _find_git_bash() -> str:
    if os.name == "nt":
        for raw_path in (
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\usr\bin\bash.exe",
        ):
            if Path(raw_path).exists():
                return raw_path
    return shutil.which("bash") or ""


def _build_shell_command(command: str, shell: str) -> tuple:
    if shell == "auto":
        return command, True, shell
    if os.name != "nt":
        if shell == "git_bash":
            return ["/bin/bash", "-lc", command], False, "git_bash"
        return [], False, shell
    if shell == "cmd":
        return ["cmd.exe", "/d", "/s", "/c", command], False, shell
    if shell == "powershell":
        executable = shutil.which("pwsh") or shutil.which("powershell") or "powershell.exe"
        return [executable, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command], False, shell
    if shell == "git_bash":
        bash = _find_git_bash()
        if not bash:
            return [], False, shell
        return [bash, "-lc", command], False, shell
    return [], False, shell


def _is_inline_file_write_command(command: str) -> bool:
    """Detect shell snippets that try to create/write files directly.

    Long source text should be passed to exec_write_file as structured JSON.
    Letting the model embed code in shell one-liners makes quoting/newline
    handling unreliable and often corrupts generated files.
    """
    text = str(command or "").strip()
    if not text:
        return False
    lowered = text.lower()

    direct_write_markers = (
        "set-content",
        "add-content",
        "out-file",
        "new-item",
        "tee-object",
        "writealltext",
        "appendalltext",
        "fs.writefilesync",
        "fs.appendfilesync",
    )
    if any(marker in lowered for marker in direct_write_markers):
        return True

    redirection_pattern = r'(^|[\s;&|])(?:echo|printf|type|cat)\b[\s\S]{0,400}?(?:>>?|2>|1>)'
    heredoc_pattern = r'(^|[\s;&|])cat\s+<<'
    if re.search(redirection_pattern, lowered) or re.search(heredoc_pattern, lowered):
        return True

    inline_interpreter = (
        re.search(r'(^|[\s;&|])python(?:3)?(?:\.exe)?\s+(-c|/c)\b', lowered)
        or re.search(r'(^|[\s;&|])node(?:\.exe)?\s+(-e|--eval)\b', lowered)
    )
    if inline_interpreter and any(
        marker in lowered
        for marker in (
            "open(",
            ".write(",
            "pathlib",
            "write_text",
            "write_bytes",
            "fs.",
            "writefile",
            "appendfile",
        )
    ):
        return True

    return False


def _inline_file_write_block_result(
    *,
    command: str,
    workspace: Path,
    request: ToolRequest,
    start_time: float,
) -> ToolResult:
    return ToolResult(
        success=False,
        data={
            "blocked": True,
            "reason": "inline_file_write_requires_write_tool",
            "recoverable": True,
            "workspace": str(workspace),
            "next_action": (
                "请改用 exec_write_file 写入文件内容：file_path 使用相对路径，"
                "content 直接传真实文本；长文件第一片 mode='overwrite'，后续 mode='append'。"
            ),
            "command_preview": str(command or "")[:240],
        },
        error=(
            "exec_run_command 只用于运行/验证命令，不能用 set-content、python -c、node -e "
            "或重定向来内联写文件。请改用 exec_write_file，避免引号转义、JSON 截断和 \\n 被写成字面量。"
        ),
        status_code=400,
        execution_time=time.time() - start_time,
        request_id=request.request_id,
    )



def _terminate_process_tree(proc: subprocess.Popen, grace_seconds: float = 5.0) -> None:
    """Best-effort cleanup for a command and all children after timeout.

    Windows shell commands often spawn grandchildren such as cargo/rustc. Killing
    only the immediate shell leaves those processes running, which breaks
    interrupt/resume pressure tests. On Windows use taskkill /T; on POSIX use a
    process group created by start_new_session=True.
    """
    if proc.poll() is not None:
        return

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=grace_seconds,
                check=False,
            )
        except Exception as exc:
            logger.warning("[exec_run_command] taskkill 进程树失败 pid=%s: %s", proc.pid, exc)
            try:
                proc.kill()
            except Exception:
                pass
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.05)

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _scan_external_paths(command: str, workspace: Path) -> list:
    """扫描命令字符串中的绝对路径，返回所有在工作区外的路径



    用于防止 exec_run_command 通过绝对路径绕过工作区边界。

    检测模式：

    - Windows 绝对路径: [A-Z]:\\... 或 [A-Z]:/...

    - POSIX 绝对路径: /home/... 或 /tmp/...

    排除 workpace 下的路径。

    """

    import re as _re

    illegal = []



    # 提取所有看起来像绝对路径的片段

    # Windows: 盘符后跟路径分隔符

    # Windows: 盘符后跟 \ 或 /，但排除 :// (URL 协议如 http://x.com 中的 p://x.com)
    win_pattern = _re.compile(r'[A-Za-z]:(?!//)[/\\][^\s\'"`;,&|<>]+')

    # POSIX: / 开头，且不是单个 /

    posix_pattern = _re.compile(r'/(?:home|tmp|usr|opt|var|etc|mnt|media|srv)/[^\s\'"`;,&|<>]*')



    workspace_str = str(workspace).replace('\\', '/').lower()

    workspace_win = str(workspace).replace('/', '\\').lower()



    for match in win_pattern.finditer(command):

        raw_path = match.group(0)

        normalized = raw_path.replace('\\', '/').lower().rstrip('/')

        # 如果路径在工作区内则放行

        if normalized.startswith(workspace_str):

            continue

        illegal.append(raw_path)



    for match in posix_pattern.finditer(command):

        raw_path = match.group(0)

        normalized = raw_path.lower().rstrip('/')

        if normalized.startswith(workspace_str):

            continue

        illegal.append(raw_path)



    return illegal


def _folder_scope_for_external_path(raw_path: str) -> str:
    """Return the existing folder scope that should be authorized for a path."""
    text = str(raw_path or "").strip().strip('"').strip("'")
    if not text:
        return ""
    try:
        candidate = Path(os.path.expanduser(os.path.expandvars(text))).resolve()
        if candidate.exists():
            return str(candidate if candidate.is_dir() else candidate.parent)
        parent = candidate.parent
        while parent and parent != parent.parent:
            if parent.is_dir():
                return str(parent)
            parent = parent.parent
    except Exception:
        return ""
    return ""


def _authorize_external_paths_for_command(
    *,
    command: str,
    illegal_paths: list,
    workspace: Path,
    request: ToolRequest,
    start_time: float,
) -> ToolResult | None:
    """Gate command absolute paths through folder access authorization.

    In manual/popup modes this emits a folder-access approval and waits; in
    full_auto mode require_folder_access_authorization performs the authoritative
    short-circuit and records the authorization audit event.
    """
    scopes = []
    for raw_path in illegal_paths:
        scope = _folder_scope_for_external_path(str(raw_path))
        if not scope:
            return ToolResult(
                success=False,
                error=(
                    f"命令包含无法解析授权范围的工作区外绝对路径: {raw_path}。"
                    f"当前工作区: {workspace}。"
                ),
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )
        if scope not in scopes:
            scopes.append(scope)

    try:
        from .workspace_access import require_folder_access_authorization

        params = request.parameters or {}
        for scope in scopes:
            access = require_folder_access_authorization(
                scope,
                current_workspace=str(workspace),
                tool_name="exec_run_command",
                action_summary=(
                    f"允许祖龙在命令中访问文件夹：{scope}\n"
                    f"待执行命令：{command[:240]}"
                ),
                conversation_id=params.get("conversation_id") or params.get("session_id") or "",
                session_id=params.get("session_id") or "",
                request_id=params.get("request_id") or request.request_id,
                timeout=float(params.get("approval_timeout") or 180.0),
            )
            if not access.approved:
                return ToolResult(
                    success=False,
                    data=access.to_payload(),
                    error=access.message,
                    status_code=403,
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )
    except Exception as exc:
        logger.warning("[exec_run_command] 外部路径授权检查失败: %s", exc)
        return ToolResult(
            success=False,
            error=f"外部路径授权检查失败: {exc}",
            status_code=403,
            execution_time=time.time() - start_time,
            request_id=request.request_id,
        )
    return None




class ExecWriteFileTool(BaseTool):

    """exec_write_file — 安全写入文件"""



    def __init__(self):
        super().__init__(name="exec_write_file", category=ToolCategory.SYSTEM)
        self.description = (
            "创建、覆写、追加或精准编辑工作区中的文件。"
            "用于生成代码、配置文件、文档等。"
            "文件路径会被限制在工作区目录内。"
            "支持一次写入完整文件；写入后会读取校验，失败时返回结构化错误。"
            "mode='overwrite' 创建新文件或完全重写；mode='append' 追加；"
            "mode='edit' 精准修改已有文件（传 edits 操作列表，不传完整文件内容）。"
        )


    def initialize(self) -> bool:

        return True



    def cleanup(self) -> None:

        pass



    def execute(self, request: ToolRequest) -> ToolResult:

        start_time = time.time()
        file_path = request.parameters.get("file_path", "")
        content = request.parameters.get("content", "")
        if content is None:
            content = ""
        content = str(content)
        mode = str(
            request.parameters.get("mode")
            or request.parameters.get("write_mode")
            or "overwrite"
        ).strip().lower()
        if mode not in {"overwrite", "append", "edit"}:
            mode = "overwrite"

        # mode="edit"：精准编辑已有文件，传 edits 操作列表
        edits = request.parameters.get("edits") or []
        if mode == "edit":
            if not edits:
                return self._create_result(
                    success=False,
                    error="mode='edit' 需要传 edits 参数（操作列表），例如 [{\"op\":\"insert\",\"line\":5,\"text\":\"import os\\n\"}]",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

        if not file_path:
            return self._create_result(
                success=False,
                error="file_path 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        try:
            # 路径安全检查：显式 workspace_path 必须与活跃 TaskGraph 工作区一致。
            from .task_tools import get_active_workspace_dir

            active_ws = get_active_workspace_dir()

            requested_workspace = (
                request.parameters.get("workspace_path")
                or request.parameters.get("workspace_dir")
                or request.parameters.get("cwd")
                or ""
            )
            requested_workspace = str(requested_workspace or "").strip()
            active_resolved = Path(active_ws).resolve() if active_ws else None
            requested_resolved = Path(requested_workspace).resolve() if requested_workspace else None
            if active_resolved and requested_resolved and active_resolved != requested_resolved:
                return self._create_result(
                    success=False,
                    data={
                        "status": "workspace_conflict",
                        "error": (
                            "工具请求的 workspace_path 与当前 TaskGraph 工作区不一致，"
                            "已拒绝执行以避免写入错误目录。"
                        ),
                        "active_workspace": str(active_resolved),
                        "requested_workspace": str(requested_resolved),
                        "workspace_path": str(active_resolved),
                        "file_path": file_path,
                        "applied": False,
                        "verified": False,
                    },
                    error=(
                        "工具请求的 workspace_path 与当前 TaskGraph 工作区不一致，"
                        "已拒绝执行以避免写入错误目录。"
                    ),
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            workspace = requested_resolved or active_resolved or Path(WORKSPACE_DIR).resolve()

            workspace.mkdir(parents=True, exist_ok=True)

            raw_target = Path(str(file_path))
            target = raw_target.resolve() if raw_target.is_absolute() else (workspace / raw_target).resolve()



            try:
                target.relative_to(workspace)
            except Exception:

                return self._create_result(

                    success=False,
                    data={
                        "status": "path_outside_workspace",
                        "error": "路径越界：文件必须在工作区目录内。",
                        "workspace_path": str(workspace),
                        "resolved_path": str(target),
                        "file_path": str(target),
                        "applied": False,
                        "verified": False,
                    },

                    error=(

                        f"路径越界：文件必须在工作区目录内。"

                        f"当前工作区: {workspace}"

                    ),

                    execution_time=time.time() - start_time,

                    request_id=request.request_id,

                )



            # 创建父目录
            target.parent.mkdir(parents=True, exist_ok=True)

            # ===== 桥路由：VS Code 桥可用时走编辑器实时通道 =====
            _bridge_used = False
            _bridge_original_content = ""
            _bridge_effective_content = None
            try:
                from .ide_bridge_tools import _ide_bridge_available, _run_async_request
                if _ide_bridge_available(str(workspace)):
                    if mode == "edit":
                        if not target.exists():
                            return self._create_result(
                                success=False,
                                error=f"edit 模式要求文件已存在，但文件不存在: {target}",
                                execution_time=time.time() - start_time,
                                request_id=request.request_id,
                            )
                        _bridge_result = _run_async_request("ide:execute_tool", {
                            "tool_name": "apply_edits",
                            "arguments": {"path": str(target), "edits": edits},
                            "workspace_path": str(workspace),
                            "cwd": str(workspace),
                            "source": "exec_write_file_bridge_edit",
                        })
                    else:
                        # overwrite/append：桥走 write_to_file
                        _pre_existing = ""
                        if mode == "append" and target.exists():
                            try:
                                _pre_existing = target.read_text(encoding="utf-8")
                            except Exception:
                                _pre_existing = ""
                        _bridge_original_content = _pre_existing
                        _bridge_content = _pre_existing + content if mode == "append" else content
                        _bridge_effective_content = _bridge_content
                        _bridge_result = _run_async_request("ide:execute_tool", {
                            "tool_name": "write_to_file",
                            "arguments": {"path": str(target), "content": _bridge_content, "mode": mode},
                            "workspace_path": str(workspace),
                            "cwd": str(workspace),
                            "source": "exec_write_file_bridge_write",
                        })
                    if _bridge_result.get("ok"):
                        _bridge_used = True
                        logger.info("[exec_write_file] 桥写入成功: %s mode=%s", target, mode)
                    else:
                        logger.warning("[exec_write_file] 桥写入失败，回退本地: %s", _bridge_result.get("error", ""))
            except Exception as _bridge_exc:
                logger.debug("[exec_write_file] 桥检测/写入异常，回退本地: %s", _bridge_exc)

            # mode="edit"：精准编辑已有文件（本地兜底，桥已处理则跳过）
            if mode == "edit" and not _bridge_used:
                if not target.exists():
                    return self._create_result(
                        success=False,
                        error=f"edit 模式要求文件已存在，但文件不存在: {target}",
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )
                from .ide_bridge_tools import _local_apply_edits, _broadcast_local_file_change
                edit_result = _local_apply_edits(
                    file_path=str(target),
                    edits=edits,
                    workspace_path=str(workspace),
                    reason="exec_write_file mode=edit (local fallback)",
                )
                if edit_result.get("ok"):
                    _broadcast_local_file_change(
                        file_path=edit_result.get("resolved_path", str(target)),
                        operation="edited",
                        original=edit_result.get("original", ""),
                        next=edit_result.get("next", ""),
                        summary=edit_result.get("summary", ""),
                    )
                return self._create_result(
                    success=bool(edit_result.get("ok")),
                    data=edit_result,
                    error=None if edit_result.get("ok") else edit_result.get("error", "edit failed"),
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )
            elif mode == "edit" and _bridge_used:
                # edit 模式桥写入成功，走统一后处理
                existing = ""
                effective_content = ""
            else:
                # overwrite/append 本地写入（桥未用或失败）
                existing = _bridge_original_content if _bridge_used else ""
                if _bridge_used:
                    effective_content = str(_bridge_effective_content or "")
                else:
                    if mode == "append" and target.exists():
                        try:
                            existing = target.read_text(encoding="utf-8")
                        except Exception as exc:
                            return self._create_result(
                                success=False,
                                data={
                                    "file_path": str(target),
                                    "resolved_path": str(target),
                                    "mode": mode,
                                    "recoverable": True,
                                    "next_action": "请改用 mode='overwrite' 重新写入第一片，后续再 append。",
                                },
                                error=f"append 前无法读取现有文件: {file_path} ({exc})",
                                execution_time=time.time() - start_time,
                                request_id=request.request_id,
                            )
                    effective_content = existing + content if mode == "append" else content
                if not _bridge_used:
                    target.write_text(effective_content, encoding="utf-8")
            expected_bytes = len(effective_content.encode("utf-8"))
            verified = target.is_file()
            actual_bytes = target.stat().st_size if verified else -1
            verified_content = False
            if verified:
                try:
                    verified_content = target.read_text(encoding="utf-8") == effective_content
                except Exception:
                    verified_content = False

            logger.info(
                "[exec_write_file][P10] result path=%s workspace=%s mode=%s applied=%s verified=%s chunk_bytes=%s effective_bytes=%s actual_bytes=%s content_sha256_12=%s",
                target,
                workspace,
                mode,
                True,
                verified_content,
                len(content.encode("utf-8")),
                expected_bytes,
                actual_bytes,
                hashlib.sha256(effective_content.encode("utf-8")).hexdigest()[:12],
            )
            if not verified_content:
                verification_error = (
                    f"文件写入后校验失败，目标文件未真实存在: {file_path}"
                    if not verified
                    else f"文件写入后校验失败，目标文件内容未应用为本次写入内容: {file_path}"
                )
                return self._create_result(
                    success=False,
                    data={
                        "file_path": str(target),
                        "resolved_path": str(target),
                        "bytes_written": expected_bytes,
                        "mode": mode,
                        "applied": target.exists(),
                        "verified": False,
                        "actual_bytes": actual_bytes,
                    },
                    error=verification_error,
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # 推送 diff/file_changed 事件给 Web 端
            try:
                from .ide_bridge_tools import _broadcast_local_file_change
                _broadcast_local_file_change(
                    file_path=str(target),
                    operation="created" if not existing else "updated",
                    original=existing,
                    next=effective_content,
                    summary=f"exec_write_file {mode}",
                )
            except Exception:
                pass

            # 关联到活跃任务图
            try:
                from .task_tools import get_active_task_graph

                tg = get_active_task_graph()

                node_id = request.parameters.get("node_id")

                if tg and node_id:

                    tg.add_file_to_node(

                        node_id, target.name, str(target)

                    )

            except Exception:

                pass



            return self._create_result(

                success=True,

                data={
                    "file_path": str(target),
                    "resolved_path": str(target),
                    "bytes_written": expected_bytes,
                    "chunk_bytes": len(content.encode("utf-8")),
                    "mode": mode,
                    "applied": True,
                    "verified": verified_content,
                    "actual_bytes": actual_bytes,
                    "message": f"文件已写入: {file_path}",
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )



        except Exception as e:
            logger.error(f"[exec_write_file] 写入失败: {e}", exc_info=True)
            logger.info(
                "[exec_write_file][P10] exception path=%s content_bytes=%s error=%s",
                file_path,
                len(content.encode("utf-8")),
                str(e)[:240],
            )
            return self._create_result(
                success=False,
                error=f"文件写入失败: {e}",

                execution_time=time.time() - start_time,

                request_id=request.request_id,

            )



    def _get_parameters_schema(self) -> Dict[str, Any]:

        return {

            "type": "object",

            "properties": {

                "file_path": {

                    "type": "string",

                    "description": "文件路径。必须是相对于任务工作目录的路径（如 index.html, src/main.py），不支持工作目录外的绝对路径。工作目录由 task_create_plan 返回的 workspace_dir 指定。",

                },
                "content": {
                    "type": "string",
                    "description": "要写入的完整文件内容；append 模式会追加到现有内容后并整体校验。",
                },
                "mode": {
                    "type": "string",
                    "enum": ["overwrite", "append", "edit"],
                    "description": "写入模式。overwrite 覆盖整个文件；append 读取现有文件并追加；edit 精准修改已有文件传 edits 操作列表。创建新文件用 overwrite，修改已有文件优先用 edit。",
                },
                "edits": {

                    "type": "array",

                    "description": "mode='edit' 时的编辑操作列表（其他模式省略）。insert:{op,line,text}; delete:{op,line,count}; replace:{op,start_line,end_line,text}; append:{op,text}",

                    "items": {

                        "type": "object",

                        "properties": {

                            "op": {"type": "string", "enum": ["insert", "delete", "replace", "append"]},

                            "line": {"type": "integer"},

                            "count": {"type": "integer"},

                            "start_line": {"type": "integer"},

                            "end_line": {"type": "integer"},

                            "text": {"type": "string"},

                        },

                    },

                },
                "node_id": {
                    "type": "string",
                    "description": "关联的任务节点 ID（可选）",

                },

            },

            "required": ["file_path"],

        }





class ExecRunCommandTool(BaseTool):

    """exec_run_command — 安全执行命令"""



    def __init__(self):

        super().__init__(name="exec_run_command", category=ToolCategory.SYSTEM)

        self.description = (
            "在工作区中执行 shell 命令。"
            "支持 python/node/npm/git 等常用命令。"
            "命令有 30 秒超时限制，输出限制 2000 字符。"
            "⚠️ 命令中不得包含工作区外的绝对路径（如 D:/other_project/file），"
            "所有文件操作必须在当前工作区内进行。"
            "命令必须符合当前操作系统和 Shell。"
        )


    def initialize(self) -> bool:

        return True



    def cleanup(self) -> None:

        pass



    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        command = request.parameters.get("command", "")
        shell = _normalize_shell(
            request.parameters.get("shell")
            or request.parameters.get("shell_type")
            or request.parameters.get("terminal")
            or request.parameters.get("terminal_shell")
        )

        if not command:
            return self._create_result(
                success=False,
                error="command 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        if shell not in SUPPORTED_COMMAND_SHELLS:
            return self._create_result(
                success=False,
                error=f"shell 参数不支持: {shell}。允许: {sorted(SUPPORTED_COMMAND_SHELLS)}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )


        # 安全检查：提取命令主体

        cmd_parts = command.strip().split()

        cmd_base = cmd_parts[0].lower() if cmd_parts else ""



        command_whitelist = _command_whitelist_for_shell(shell)
        if cmd_base not in command_whitelist:
            return self._create_result(
                success=False,
                error=f"命令 '{cmd_base}' 不在当前平台白名单中。允许: {sorted(command_whitelist)}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        runtime_env = get_runtime_environment()
        if shell != "git_bash":
            lowered = command.lower()
            matched_markers = [
                marker for marker in runtime_env.forbidden_command_markers
                if marker.lower() in lowered
            ]
            if matched_markers:
                return self._create_result(
                    success=False,
                    error=(
                        f"检测到与当前环境不兼容的命令片段: {matched_markers}。\n"
                        f"{runtime_env.command_guidance}"
                    ),
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )


        # BP7 修复: 扫描命令中的绝对路径，禁止写入工作区外的文件

        try:

            from .task_tools import get_active_workspace_dir

            active_ws = get_active_workspace_dir()

            workspace = Path(active_ws).resolve() if active_ws else Path(WORKSPACE_DIR).resolve()

        except Exception:

            workspace = Path(WORKSPACE_DIR).resolve()



        if _is_inline_file_write_command(command):
            logger.warning(
                "[exec_run_command] blocked inline file write; use exec_write_file: %s",
                command[:240],
            )
            return _inline_file_write_block_result(
                command=command,
                workspace=workspace,
                request=request,
                start_time=start_time,
            )

        illegal_paths = _scan_external_paths(command, workspace)

        if illegal_paths:
            authorization_result = _authorize_external_paths_for_command(
                command=command,
                illegal_paths=illegal_paths,
                workspace=workspace,
                request=request,
                start_time=start_time,
            )
            if authorization_result is not None:
                return authorization_result



        try:

            # 优先使用活跃任务的专属工作目录（复用上面的 workspace）



            # 执行命令

            popen_command, use_shell, resolved_shell = _build_shell_command(command, shell)
            if not popen_command:
                return self._create_result(
                    success=False,
                    error=f"无法找到或启动指定 shell: {shell}",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )
            proc = subprocess.Popen(
                popen_command,
                shell=use_shell,
                cwd=str(workspace),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=(os.name != "nt"),
            )

            try:
                timeout_seconds = min(float(request.parameters.get("timeout") or request.timeout or 30), 300.0)
                stdout, stderr = proc.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:

                _terminate_process_tree(proc)

                try:
                    stdout, stderr = proc.communicate(timeout=5)
                except Exception:
                    stdout, stderr = "", ""

                return self._create_result(

                    success=False,

                    error=f"命令执行超时（{timeout_seconds:g} 秒）",
                    execution_time=time.time() - start_time,

                    request_id=request.request_id,

                )



            # 截断输出

            max_output = 2000

            stdout = stdout or ""

            stderr = stderr or ""



            if len(stdout) > max_output:

                stdout = stdout[:max_output] + f"\n... (截断，共 {len(stdout)} 字符)"

            if len(stderr) > max_output:

                stderr = stderr[:max_output] + f"\n... (截断，共 {len(stderr)} 字符)"



            success = proc.returncode == 0



            logger.info(f"[exec_run_command] '{command}' → returncode={proc.returncode}")



            return self._create_result(

                success=success,

                data={
                    "command": command,
                    "shell": resolved_shell,
                    "returncode": proc.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                },

                execution_time=time.time() - start_time,

                request_id=request.request_id,

            )



        except Exception as e:

            logger.error(f"[exec_run_command] 执行失败: {e}", exc_info=True)

            return self._create_result(

                success=False,

                error=f"命令执行失败: {e}",

                execution_time=time.time() - start_time,

                request_id=request.request_id,

            )



    def _get_parameters_schema(self) -> Dict[str, Any]:

        return {

            "type": "object",

            "properties": {

                "command": {
                    "type": "string",
                    "description": (
                        "要执行的命令（如 'python main.py'、'npm install'）。"
                        "⚠️ 命令中不得包含工作区外的绝对路径，所有文件操作必须在工作区内。"
                        f"{get_runtime_environment().command_guidance}"
                    ),
                },
                "shell": {
                    "type": "string",
                    "enum": ["auto", "cmd", "powershell", "git_bash"],
                    "description": (
                        "选择命令执行 shell。auto 使用当前系统默认 shell；"
                        "cmd 使用 cmd.exe；powershell 使用 PowerShell/pwsh；"
                        "git_bash 使用 Git for Windows 的 bash.exe。"
                    ),
                },
                "timeout": {
                    "type": "number",
                    "description": "命令超时时间（秒），默认 30，最大 300。",
                },
            },
            "required": ["command"],
        }




