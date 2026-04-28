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
from typing import Dict, Any
from pathlib import Path

from .base import BaseTool, ToolCategory, ToolRequest, ToolResult

logger = logging.getLogger(__name__)

# 工作区根目录（安全边界）
WORKSPACE_DIR = os.environ.get("ZULONG_WORKSPACE", "./workspace")

# 命令白名单
COMMAND_WHITELIST = {
    "python", "python3", "node", "npm", "npx", "pip", "pip3",
    "cat", "ls", "dir", "echo", "type", "mkdir", "cd", "pwd",
    "tree", "git", "cargo", "go", "javac", "java",
}


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
            # 路径安全检查
            workspace = Path(WORKSPACE_DIR).resolve()
            workspace.mkdir(parents=True, exist_ok=True)
            target = (workspace / file_path).resolve()

            if not str(target).startswith(str(workspace)):
                return self._create_result(
                    success=False,
                    error="路径越界：文件必须在工作区目录内",
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
                    "description": "文件相对路径（相对于工作区目录）",
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

        try:
            workspace = Path(WORKSPACE_DIR).resolve()
            workspace.mkdir(parents=True, exist_ok=True)

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
                    "description": "要执行的命令（如 'python main.py'、'npm install'）",
                },
            },
            "required": ["command"],
        }
