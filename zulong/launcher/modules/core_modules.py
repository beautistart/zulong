"""
Core 模块组 — 两种启动模式都需要的基础模块

包含：ConfigModule, SharedMemoryPoolModule, MemoryGraphModule,
      EventBusWSModule, InferenceEngineModule
"""

import asyncio
import logging
import os
import socket
import threading
from typing import Set

from zulong.launcher.module_base import Module, ModuleState

logger = logging.getLogger(__name__)


# ── 1. ConfigModule ──────────────────────────────────

class ConfigModule(Module):
    name = "config"
    display_name = "配置系统"
    dependencies = []
    mode_tags: Set[str] = {"core"}

    async def start(self) -> None:
        self.progress_message = "正在加载配置..."
        from zulong.config.config_manager import init_config, get_config

        config_mgr = init_config()
        self._context["config_manager"] = config_mgr

        # 设置环境变量（与 bootstrap.py 对齐）
        if "USE_VLLM_FOR_L2" not in os.environ:
            backend = get_config("llm.backend", "ollama")
            os.environ["USE_VLLM_FOR_L2"] = "true" if backend == "vllm" else "false"
        if "USE_VLLM_FOR_L2_BACKUP" not in os.environ:
            backup = get_config("llm.backend", "ollama")
            os.environ["USE_VLLM_FOR_L2_BACKUP"] = "true" if backup == "vllm" else "false"
        os.environ.setdefault("ZULONG_LOG_LEVEL", get_config("system.log_level", "INFO"))
        os.environ.setdefault("ZULONG_DEBUG_MODE", str(get_config("system.debug_mode", False)).lower())
        os.environ.setdefault("ZULONG_DATA_DIR", get_config("system.data_dir", "./data"))
        os.environ.setdefault("ZULONG_MODELS_DIR", get_config("system.models_dir", "./models"))

        # 初始化日志
        try:
            from zulong.utils.monitor import setup_logging
            setup_logging()
        except Exception:
            pass

        self.state = ModuleState.RUNNING
        logger.info("[ConfigModule] 配置系统已初始化")


# ── 2. SharedMemoryPoolModule ────────────────────────

class SharedMemoryPoolModule(Module):
    name = "shared_memory_pool"
    display_name = "共享内存池"
    dependencies = ["config"]
    mode_tags: Set[str] = {"core"}

    async def start(self) -> None:
        self.progress_message = "正在初始化共享内存池..."
        from zulong.infrastructure.shared_memory_pool import SharedMemoryPool

        if SharedMemoryPool._instance is not None:
            logger.info("[SharedMemoryPoolModule] 单例已存在，跳过创建")
            pool = SharedMemoryPool._instance
        else:
            pool = await SharedMemoryPool.get_instance()
            logger.info(f"[SharedMemoryPoolModule] 单例已创建: id={id(pool)}")

        self._context["shared_memory_pool"] = pool
        self.state = ModuleState.RUNNING


# ── 3. MemoryGraphModule ─────────────────────────────

class MemoryGraphModule(Module):
    name = "memory_graph"
    display_name = "记忆图谱"
    dependencies = ["shared_memory_pool"]
    mode_tags: Set[str] = {"core"}

    async def start(self) -> None:
        self.progress_message = "正在初始化记忆图谱..."
        from zulong.memory.memory_graph import MemoryGraph

        if MemoryGraph._instance is not None:
            mg = MemoryGraph._instance
            logger.info("[MemoryGraphModule] 单例已存在，跳过创建")
            # 确保适配器已注册（单例可能由 bootstrap 创建但未注册适配器）
            try:
                from zulong.memory.graph_adapters import register_all_adapters
                if not mg._adapters.get("code_graph"):
                    register_all_adapters(mg)
                    logger.info("[MemoryGraphModule] 适配器补注册完成")
            except Exception as e:
                logger.warning(f"[MemoryGraphModule] 适配器补注册失败: {e}")
        else:
            mg = MemoryGraph(persist_path="./data/memory_graph")
            try:
                from zulong.memory.graph_adapters import register_all_adapters
                register_all_adapters(mg)
            except Exception as e:
                logger.warning(f"[MemoryGraphModule] register_all_adapters 失败: {e}")
            try:
                mg.sync_all()
            except Exception as e:
                logger.warning(f"[MemoryGraphModule] sync_all 失败: {e}")
            logger.info(f"[MemoryGraphModule] MemoryGraph 初始化完成: {mg.stats['total_nodes']} 节点")

        self._context["memory_graph"] = mg
        self.state = ModuleState.RUNNING


# ── 4. EventBusWSModule ──────────────────────────────

class EventBusWSModule(Module):
    name = "eventbus_ws"
    display_name = "事件总线"
    dependencies = ["config"]
    mode_tags: Set[str] = {"core"}

    def __init__(self):
        super().__init__()
        self._thread = None

    async def start(self) -> None:
        self.progress_message = "正在启动事件总线 WebSocket..."
        try:
            from zulong.config.config_manager import get_config
            host = get_config("event_bus.websocket.host", "localhost")
            port = get_config("event_bus.websocket.port", 5555)
        except Exception:
            host, port = "localhost", 5555

        # 检测端口占用
        bind_host = "127.0.0.1" if host == "localhost" else host
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((bind_host, port))
        sock.close()
        if result == 0:
            logger.info(f"[EventBusWSModule] 端口 {port} 已被占用，跳过启动（外部已启动）")
            self.state = ModuleState.RUNNING
            return

        def _run():
            try:
                from zulong.core.websocket_server import start_websocket_server
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(start_websocket_server(host=host, port=port))
            except Exception as e:
                logger.error(f"[EventBusWSModule] WebSocket 服务器异常: {e}", exc_info=True)

        self._thread = threading.Thread(target=_run, daemon=True, name="EventBusWS")
        self._thread.start()
        # 等待 WS 服务器实际就绪
        await asyncio.sleep(1.0)
        self.state = ModuleState.RUNNING
        logger.info(f"[EventBusWSModule] EventBus WebSocket 已启动: ws://{host}:{port}/eventbus")

    async def stop(self) -> None:
        # daemon 线程会随进程退出
        self.state = ModuleState.STOPPED


# ── 5. InferenceEngineModule ─────────────────────────

class InferenceEngineModule(Module):
    name = "inference_engine"
    display_name = "推理引擎"
    dependencies = ["shared_memory_pool", "memory_graph"]
    mode_tags: Set[str] = {"core"}

    async def start(self) -> None:
        self.progress_message = "正在初始化推理引擎（可能需要 30 秒）..."
        from zulong.l2.inference_engine import InferenceEngine

        engine = InferenceEngine()
        self._context["inference_engine"] = engine
        self.state = ModuleState.RUNNING
        logger.info("[InferenceEngineModule] InferenceEngine 初始化完成")
