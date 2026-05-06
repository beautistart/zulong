"""
LauncherApp — 祖龙统一启动器

启动 FastAPI 服务：
- 初始阶段: GET / 返回 startup.html (模式选择页)
- 启动后:   GET / 返回 index.html (监控 dashboard)

API 端点:
- POST /api/launch       启动指定模式
- GET  /api/status       全局状态
- POST /api/module/{n}/start|stop  动态开关
- WS   /ws/progress      启动进度推送
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

from zulong.launcher.module_manager import ModuleManager
from zulong.launcher.modules.core_modules import (
    ConfigModule,
    SharedMemoryPoolModule,
    MemoryGraphModule,
    EventBusWSModule,
    InferenceEngineModule,
)
from zulong.launcher.modules.ide_server_module import IDEServerModule
from zulong.launcher.web_chat_router import router as web_chat_router, set_launch_mode

logger = logging.getLogger(__name__)

_LAUNCHER_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_DASHBOARD_STATIC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "openclaw_bridge", "web", "static"
)


class LauncherApp:
    """祖龙统一启动器"""

    def __init__(self):
        # 读取配置
        try:
            from zulong.config.config_manager import init_config, get_config
            init_config()
            self.host = get_config("launcher.host", "127.0.0.1")
            self.port = get_config("launcher.port", 8090)
            self.auto_open_browser = get_config("launcher.auto_open_browser", True)
            self._default_mode = get_config("launcher.default_mode", None)
        except Exception:
            self.host = "127.0.0.1"
            self.port = 8090
            self.auto_open_browser = True
            self._default_mode = None

        self.app = FastAPI(title="Zulong Launcher")
        self.manager = ModuleManager()
        self.phase = "selecting"   # selecting → launching → running
        self._start_time = time.time()
        self._progress_clients: Set[WebSocket] = set()

        self._register_all_modules()
        self._register_routes()

        # 挂载 Web 聊天路由器（/ws 端点 — 主系统前端通信）
        self.app.include_router(web_chat_router)

    # ── 模块注册 ──────────────────────────────────────

    def _register_all_modules(self) -> None:
        """注册所有模块到 ModuleManager"""
        # Core 模块
        self.manager.register(ConfigModule())
        self.manager.register(SharedMemoryPoolModule())
        self.manager.register(MemoryGraphModule())
        self.manager.register(EventBusWSModule())
        self.manager.register(InferenceEngineModule())
        self.manager.register(IDEServerModule())

        # 注入 FastAPI app 到 context（IDEServerModule 需要）
        self.manager.context["fastapi_app"] = self.app

        # Full 模式模块
        try:
            from zulong.launcher.modules.full_modules import get_full_modules
            for mod in get_full_modules():
                self.manager.register(mod)
        except ImportError:
            logger.debug("[LauncherApp] full_modules 未找到，跳过 Full 模式模块注册")

        # 可选模块
        try:
            from zulong.launcher.modules.openclaw_module import OpenClawModule
            self.manager.register(OpenClawModule())
        except ImportError:
            logger.debug("[LauncherApp] openclaw_module 未找到，跳过")

        try:
            from zulong.launcher.modules.mcp_module import MCPModule
            self.manager.register(MCPModule())
        except ImportError:
            logger.debug("[LauncherApp] mcp_module 未找到，跳过")

    # ── 路由注册 ──────────────────────────────────────

    def _register_routes(self) -> None:
        app = self.app

        @app.get("/")
        async def root():
            """根路由：启动前返回选择页，启动后返回 Dashboard"""
            # 禁止缓存，否则浏览器可能缓存 startup.html 导致跳转后仍返回旧页面
            headers = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
            if self.phase == "running" and self.manager.context.get("ide_ready"):
                index_path = os.path.join(_DASHBOARD_STATIC, "index.html")
                if os.path.exists(index_path):
                    return FileResponse(index_path, media_type="text/html", headers=headers)
            # 启动选择页
            startup_path = os.path.join(_LAUNCHER_STATIC, "startup.html")
            if os.path.exists(startup_path):
                return FileResponse(startup_path, media_type="text/html", headers=headers)
            return {"message": "Zulong Launcher", "phase": self.phase}

        @app.post("/api/launch")
        async def launch(body: dict):
            """启动指定模式"""
            if self.phase != "selecting":
                return JSONResponse(
                    {"error": f"当前阶段不允许启动: {self.phase}"},
                    status_code=400,
                )
            mode = body.get("mode", "ide")
            if mode not in ("full", "ide"):
                return JSONResponse({"error": f"无效模式: {mode}"}, status_code=400)

            self.phase = "launching"
            asyncio.create_task(self._do_launch(mode))
            return {"status": "ok", "mode": mode}

        @app.get("/api/status")
        async def status():
            """全局状态"""
            info = self.manager.get_status()
            info["phase"] = self.phase
            info["uptime_seconds"] = round(time.time() - self._start_time, 1)
            return info

        @app.post("/api/module/{name}/start")
        async def module_start(name: str):
            """动态启动可选模块"""
            try:
                await self.manager.start_module(name)
                return {"status": "ok", "module": name}
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=400)

        @app.post("/api/module/{name}/stop")
        async def module_stop(name: str):
            """动态停止可选模块"""
            try:
                await self.manager.stop_module(name)
                return {"status": "ok", "module": name}
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=400)

        @app.websocket("/ws/progress")
        async def ws_progress(ws: WebSocket):
            """启动进度实时推送"""
            await ws.accept()
            self._progress_clients.add(ws)
            try:
                # 发送当前状态
                await ws.send_json({
                    "type": "STATUS",
                    "ts": time.time(),
                    "payload": {
                        "phase": self.phase,
                        "modules": {
                            n: m.to_status()
                            for n, m in self.manager._modules.items()
                        },
                    },
                })
                # 保持连接
                while True:
                    msg = await ws.receive_text()
                    # ping/pong
                    if msg == "ping":
                        await ws.send_json({"type": "pong", "ts": time.time()})
            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                self._progress_clients.discard(ws)

    # ── 启动编排 ──────────────────────────────────────

    async def _do_launch(self, mode: str) -> None:
        """执行模块启动序列"""
        try:
            await self.manager.launch(mode=mode, on_progress=self._push_progress)
            self.phase = "running"
            # 设置 Web 聊天路由器的运行模式（Full 模式下订阅 EventBus 下行事件）
            set_launch_mode(mode)
            logger.info(
                f"[LauncherApp] 启动完成: mode={mode}, "
                f"ide_ready={self.manager.context.get('ide_ready')}, "
                f"progress_clients={len(self._progress_clients)}"
            )
            # 推送启动完成
            await self._broadcast_progress({
                "type": "LAUNCH_COMPLETE",
                "ts": time.time(),
                "payload": {
                    "mode": mode,
                    "redirect_url": "/",
                    "duration_seconds": round(time.time() - self._start_time, 1),
                },
            })
            logger.info(f"[LauncherApp] 启动完成: mode={mode}")
        except Exception as e:
            self.phase = "selecting"  # 回退
            await self._broadcast_progress({
                "type": "LAUNCH_ERROR",
                "ts": time.time(),
                "payload": {"error": str(e)},
            })
            logger.error(f"[LauncherApp] 启动失败: {e}", exc_info=True)

    async def _push_progress(
        self, step: int, total: int, name: str, display: str, message: str
    ) -> None:
        """进度回调 → 推送到所有 WS 客户端"""
        percent = int(step / total * 100) if total > 0 else 0
        payload = {
            "type": "PROGRESS",
            "ts": time.time(),
            "payload": {
                "phase": "launching",
                "current_module": name,
                "current_module_display": display,
                "current_step": step,
                "total_steps": total,
                "percent": percent,
                "message": message,
                "modules": {
                    n: m.to_status()
                    for n, m in self.manager._modules.items()
                },
            },
        }
        await self._broadcast_progress(payload)

    async def _broadcast_progress(self, msg: dict) -> None:
        dead: Set[WebSocket] = set()
        for ws in list(self._progress_clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.add(ws)
        self._progress_clients -= dead
