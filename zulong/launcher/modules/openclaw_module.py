"""
OpenClawModule — 可选模块，启动 OpenClaw Bridge 子进程

依赖 eventbus_ws（需要先有 EventBus WebSocket Server 监听 5555）。
运行时可通过 API 动态开关。
"""

import asyncio
import logging
import os
import subprocess
import sys
from typing import Set

from zulong.launcher.module_base import Module, ModuleState

logger = logging.getLogger(__name__)


class OpenClawModule(Module):
    name = "openclaw"
    display_name = "OpenClaw 桥接"
    dependencies = ["eventbus_ws"]
    mode_tags: Set[str] = {"optional"}

    def __init__(self):
        super().__init__()
        self._process = None

    async def start(self) -> None:
        self.progress_message = "正在启动 OpenClaw Bridge..."

        # 检查配置是否启用
        try:
            from zulong.config.config_manager import get_config
            if not get_config("tools.openclaw.enabled", True):
                logger.info("[OpenClawModule] 配置中已禁用 OpenClaw，跳过")
                self.state = ModuleState.RUNNING
                return
        except Exception:
            pass

        # 定位 openclaw_bridge 入口
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        bootstrap_path = os.path.join(project_root, "openclaw_bridge", "bootstrap.py")

        if not os.path.exists(bootstrap_path):
            raise FileNotFoundError(f"OpenClaw Bridge 入口未找到: {bootstrap_path}")

        # 以子进程方式启动
        self._process = subprocess.Popen(
            [sys.executable, bootstrap_path],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": project_root},
        )
        logger.info(f"[OpenClawModule] OpenClaw Bridge 子进程已启动, PID={self._process.pid}")

        # 等待 Bridge 启动（检查进程是否还活着）
        await asyncio.sleep(3.0)
        if self._process.poll() is not None:
            # 进程已退出
            output = ""
            if self._process.stdout:
                output = self._process.stdout.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"OpenClaw Bridge 启动失败 (exit={self._process.returncode}): {output}")

        self.state = ModuleState.RUNNING
        logger.info("[OpenClawModule] OpenClaw Bridge 已就绪")

    async def stop(self) -> None:
        if self._process and self._process.poll() is None:
            logger.info(f"[OpenClawModule] 正在停止 OpenClaw Bridge (PID={self._process.pid})...")
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
            logger.info("[OpenClawModule] OpenClaw Bridge 已停止")
        self._process = None
        self.state = ModuleState.STOPPED
