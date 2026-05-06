"""
MCPModule — 可选模块，启动祖龙记忆 MCP Server 子进程

MCP Server 使用 stdio 协议，这里以子进程方式启动，
方便运行时动态开关。
"""

import asyncio
import logging
import os
import subprocess
import sys
from typing import Set

from zulong.launcher.module_base import Module, ModuleState

logger = logging.getLogger(__name__)


class MCPModule(Module):
    name = "mcp_server"
    display_name = "MCP 记忆服务"
    dependencies = ["memory_graph"]
    mode_tags: Set[str] = {"optional"}

    def __init__(self):
        super().__init__()
        self._process = None

    async def start(self) -> None:
        self.progress_message = "正在启动 MCP Server..."

        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        mcp_path = os.path.join(project_root, "mcp_server.py")

        if not os.path.exists(mcp_path):
            raise FileNotFoundError(f"MCP Server 入口未找到: {mcp_path}")

        # MCP Server 使用 stdio，以子进程方式启动
        self._process = subprocess.Popen(
            [sys.executable, mcp_path],
            cwd=project_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "PYTHONPATH": project_root},
        )
        logger.info(f"[MCPModule] MCP Server 子进程已启动, PID={self._process.pid}")

        # 等待初始化
        await asyncio.sleep(2.0)
        if self._process.poll() is not None:
            stderr_out = ""
            if self._process.stderr:
                stderr_out = self._process.stderr.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"MCP Server 启动失败 (exit={self._process.returncode}): {stderr_out}")

        self.state = ModuleState.RUNNING
        logger.info("[MCPModule] MCP Server 已就绪")

    async def stop(self) -> None:
        if self._process and self._process.poll() is None:
            logger.info(f"[MCPModule] 正在停止 MCP Server (PID={self._process.pid})...")
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
            logger.info("[MCPModule] MCP Server 已停止")
        self._process = None
        self.state = ModuleState.STOPPED
