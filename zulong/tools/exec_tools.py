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
import platform
from typing import Dict, Any
from pathlib import Path

from .base import BaseTool, ToolCategory, ToolRequest, ToolResult

logger = logging.getLogger(__name__)

# 工作区根目录（安全边界）
WORKSPACE_DIR = os.environ.get("ZULONG_WORKSPACE", "./agent_workspace")
WINDOWS_SHELL_HINT = (
    "当前环境是 Windows/PowerShell。"
    "请改用 PowerShell/Windows 命令，例如：Get-ChildItem、Select-String、Get-Content、rg。"
)

# 命令白名单
COMMAND_WHITELIST = {
    "python", "python3", "node", "npm", "npx", "pip", "pip3",
    "cat", "ls", "dir", "echo", "type", "mkdir", "cd", "pwd", "get-content",
    "tree", "git", "cargo", "go", "javac", "java",
    # 扩展: 常用开发工具
    "curl", "wget", "tar", "unzip", "zip",
    "cp", "mv", "rm", "touch", "chmod", "chown",
    "find", "grep", "head", "tail", "wc", "sort",
    "diff", "file", "which", "whereis",
    "pdflatex", "make", "cmake", "gcc", "g++",
    "ffmpeg", "imagemagick", "convert",
    "docker", "docker-compose", "kubectl",
    "npx", "tsc", "eslint", "prettier",
    "http-server", "live-server",
}


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
            "创建或覆写工作区中的文件。"
            "用于生成代码、配置文件、文档等。"
            "文件路径会被限制在工作区目录内。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        file_path = request.parameters.get("file_path", "")
        content = request.parameters.get("content", "")

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
            target.write_text(content, encoding="utf-8")

            logger.info(f"[exec_write_file] 写入 {target} ({len(content)} bytes)")

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
                    "bytes_written": len(content.encode("utf-8")),
                    "message": f"文件已写入: {file_path}",
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[exec_write_file] 写入失败: {e}", exc_info=True)
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
                    "description": "要写入的文件内容",
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
            "当前默认环境通常为 Windows + PowerShell，请避免 Unix 专属命令。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        command = request.parameters.get("command", "")

        if not command:
            return self._create_result(
                success=False,
                error="command 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        # 安全检查：提取命令主体
        cmd_parts = command.strip().split()
        cmd_base = cmd_parts[0].lower() if cmd_parts else ""

        if cmd_base not in COMMAND_WHITELIST:
            return self._create_result(
                success=False,
                error=f"命令 '{cmd_base}' 不在白名单中。允许: {sorted(COMMAND_WHITELIST)}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        if platform.system().lower().startswith("win"):
            unix_only_markers = (
                "find / -name", "ls -la", "pwd &&", "2>/dev/null",
                "| head", " head -", "grep ", "chmod ", "mkdir -p",
            )
            lowered = command.lower()
            if any(marker in lowered for marker in unix_only_markers):
                return self._create_result(
                    success=False,
                    error=(
                        f"检测到与当前 Windows 环境不兼容的 Unix 风格命令: {command}\n"
                        f"{WINDOWS_SHELL_HINT}"
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
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=str(workspace),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
            )

            try:
                stdout, stderr = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
                return self._create_result(
                    success=False,
                    error="命令执行超时（>30 秒）",
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
                    "description": "要执行的命令（如 'python main.py'、'npm install'）。⚠️ 命令中不得包含工作区外的绝对路径（如 D:/other/file），所有文件操作必须在工作区内。当前环境优先使用 Windows/PowerShell 命令，如 Get-ChildItem、Select-String、Get-Content、rg。",
                },
            },
            "required": ["command"],
        }
