"""
IDEServerModule — 将 IDE Server 路由动态挂载到 LauncherApp 的 FastAPI app

职责：
1. 从 ModuleManager.context 获取 InferenceEngine 实例
2. 设置 ide_server._engine_instance
3. 将 ide_router 挂载到 Launcher 的 FastAPI app
4. 挂载 /static 静态文件
5. 设置标志位使 Launcher 的根路由切换为 index.html
"""

import logging
import os
from typing import Set

from zulong.launcher.module_base import Module, ModuleState

logger = logging.getLogger(__name__)


class IDEServerModule(Module):
    name = "ide_server"
    display_name = "IDE 服务"
    dependencies = ["inference_engine", "eventbus_ws"]
    mode_tags: Set[str] = {"core"}

    async def start(self) -> None:
        self.progress_message = "正在挂载 IDE 服务路由..."

        # 1. 注入 InferenceEngine 到 ide_server
        import zulong.ide.ide_server as ide_mod
        engine = self._context.get("inference_engine")
        if engine is not None:
            ide_mod._engine_instance = engine
            logger.info("[IDEServerModule] InferenceEngine 已注入到 IDE Server")

        # 2. 将 ide_router 挂载到 Launcher app
        launcher_app = self._context.get("fastapi_app")
        if launcher_app is None:
            raise RuntimeError("fastapi_app 未在 context 中，无法挂载 IDE 路由")

        launcher_app.include_router(ide_mod.ide_router)
        logger.info("[IDEServerModule] ide_router 已挂载 (/ide, /monitor, /health)")

        # 3. 挂载静态文件
        from fastapi.staticfiles import StaticFiles
        static_dir = ide_mod._STATIC_DIR
        if os.path.isdir(static_dir):
            launcher_app.mount("/static", StaticFiles(directory=static_dir), name="static")
            logger.info(f"[IDEServerModule] /static 已挂载: {static_dir}")

        # 4. 预热 Embedding 模型（避免首次对话时加载耗时导致前端超时）
        try:
            from zulong.memory.embedding_manager import get_embedding_manager
            emb_mgr = get_embedding_manager()
            if emb_mgr._model is None:
                logger.info("[IDEServerModule] 预热 Embedding 模型...")
                emb_mgr.encode("预热")
                logger.info("[IDEServerModule] Embedding 模型预热完成")
        except Exception as e:
            logger.warning(f"[IDEServerModule] Embedding 模型预热失败（非致命）: {e}")

        # 5. 标记 IDE 就绪（Launcher 用此切换根路由）
        self._context["ide_ready"] = True

        self.state = ModuleState.RUNNING
        logger.info("[IDEServerModule] IDE 服务已就绪")
