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
import hashlib
import shutil
from typing import Dict, Any
from pathlib import Path

from .base import BaseTool, ToolCategory, ToolRequest, ToolResult
from zulong.utils.runtime_env import get_runtime_environment

logger = logging.getLogger(__name__)

# 工作区根目录（安全边界）
WORKSPACE_DIR = os.environ.get("ZULONG_WORKSPACE", "./agent_workspace")
MAX_WRITE_CHUNK_CHARS = 1800
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
    win_pattern = _re.compile(r'[A-Za-z]:[/\\][^\s\'"`;,&|<>]+')
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


class ExecWriteFileTool(BaseTool):
    """exec_write_file — 安全写入文件"""

    def __init__(self):
        super().__init__(name="exec_write_file", category=ToolCategory.SYSTEM)
        self.description = (
            "创建、覆写或追加写入工作区中的文件。"
            "用于生成代码、配置文件、文档等。"
            "文件路径会被限制在工作区目录内。"
            "长文件请按 800-1200 字符分片写入：第一片 mode=overwrite，后续 mode=append。"
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
        if mode not in {"overwrite", "append"}:
            mode = "overwrite"

        if not file_path:
            return self._create_result(
                success=False,
                error="file_path 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        try:
            # 路径安全检查：优先使用活跃任务的专属工作目录
            from .task_tools import get_active_workspace_dir
            active_ws = get_active_workspace_dir()
            workspace = Path(active_ws).resolve() if active_ws else Path(WORKSPACE_DIR).resolve()
            workspace.mkdir(parents=True, exist_ok=True)
            target = (workspace / file_path).resolve()

            if not str(target).startswith(str(workspace)):
                return self._create_result(
                    success=False,
                    error=(
                        f"路径越界：文件必须在工作区目录内。"
                        f"当前工作区: {workspace}"
                    ),
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # 创建父目录
            target.parent.mkdir(parents=True, exist_ok=True)

            # 写入文件
            existing = ""
            if mode == "append" and target.exists():
                try:
                    existing = target.read_text(encoding="utf-8")
                except Exception as exc:
                    return self._create_result(
                        success=False,
                        data={
                            "file_path": str(target),
                            "mode": mode,
                            "recoverable": True,
                            "next_action": "请改用 mode='overwrite' 重新写入第一片，后续再 append。",
                        },
                        error=f"append 前无法读取现有文件: {file_path} ({exc})",
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )
            effective_content = existing + content if mode == "append" else content
            content_chars = len(content)
            if content_chars > MAX_WRITE_CHUNK_CHARS:
                logger.warning(
                    "[exec_write_file][P10] oversized chunk path=%s mode=%s content_chars=%s",
                    target,
                    mode,
                    content_chars,
                )
                return self._create_result(
                    success=False,
                    data={
                        "file_path": str(target),
                        "mode": mode,
                        "content_chars": content_chars,
                        "max_chunk_chars": MAX_WRITE_CHUNK_CHARS,
                        "recoverable": True,
                        "chunk_policy": "openhands_style_file_chunking",
                        "next_action": (
                            "请把文件内容拆成 800-1200 字符的小块；"
                            "第一块 mode='overwrite'，后续块 mode='append'。"
                        ),
                    },
                    error=f"单次写入内容过长: {content_chars} 字符，需分片写入",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )
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
                return self._create_result(
                    success=False,
                    data={
                        "file_path": str(target),
                        "bytes_written": expected_bytes,
                        "mode": mode,
                        "applied": target.exists(),
                        "verified": False,
                        "actual_bytes": actual_bytes,
                    },
                    error=f"文件写入后校验失败: {file_path}",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

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
                    "description": "要写入的文件内容。单块建议 800-1200 字符，最大 1800 字符。",
                },
                "mode": {
                    "type": "string",
                    "enum": ["overwrite", "append"],
                    "description": "写入模式。默认 overwrite 覆盖整个文件；append 会读取现有文件并追加本次 content。",
                },
                "node_id": {
                    "type": "string",
                    "description": "关联的任务节点 ID（可选）",
                },
            },
            "required": ["file_path", "content"],
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

        illegal_paths = _scan_external_paths(command, workspace)
        if illegal_paths:
            return self._create_result(
                success=False,
                error=(
                    f"命令包含工作区外的绝对路径，已被拦截: {illegal_paths}。"
                    f"当前工作区: {workspace}。"
                    f"请使用相对路径或确保所有文件操作在工作区内进行。"
                ),
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

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
            )

            try:
                timeout_seconds = min(float(request.parameters.get("timeout") or request.timeout or 30), 300.0)
                stdout, stderr = proc.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
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

