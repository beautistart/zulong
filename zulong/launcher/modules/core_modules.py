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
        from zulong.memory.memory_graph_factory import create_memory_graph, get_memory_graph_type
        from zulong.config.config_manager import get_config

        if MemoryGraph._instance is not None:
            mg = MemoryGraph._instance
            backend_type = get_memory_graph_type(mg)

            if backend_type == "networkx":
                raise RuntimeError("MemoryGraph 已禁止使用 NetworkX 单 JSON 后端，请检查启动顺序")
            logger.info(f"[MemoryGraphModule] 单例已存在，使用 {backend_type} 分片后端")
        else:
            mg = create_memory_graph(persist_path="./data/memory_graph")
            backend_type = get_memory_graph_type(mg)
            
            # 设置 MemoryGraph 单例引用，防止后续代码创建第二个后端实例（split-brain）
            MemoryGraph._instance = mg
            
            if backend_type == "networkx":
                raise RuntimeError("MemoryGraph 工厂返回了已禁用的 NetworkX 单 JSON 后端")
            stats = mg.get_stats() if hasattr(mg, 'get_stats') else {}
            node_count = stats.get('total_nodes', stats.get('node_count', 0))
            logger.info(f"[MemoryGraphModule] {mg.__class__.__name__} 初始化完成 ({backend_type}): {node_count} 节点")

        self._context["memory_graph"] = mg
        self.state = ModuleState.RUNNING


# ── 4. EventStoreModule ─────────────────────────────

class EventStoreModule(Module):
    """事件持久化模块 - 将所有 EventBus 事件异步写入 SQLite
    
    功能：
    - 启动时初始化 SQLite EventStore
    - 注入 EventBus，所有后续事件自动持久化
    - 30 天自动清理过期事件
    """
    name = "event_store"
    display_name = "事件持久化"
    dependencies = ["eventbus_ws"]
    mode_tags: Set[str] = {"core"}

    async def start(self) -> None:
        self.progress_message = "正在初始化事件持久化..."
        try:
            from zulong.config.config_manager import get_config
            from zulong.core.event_bus import event_bus
            from zulong.events import get_event_store

            event_persistence_enabled = get_config('memory.event_persistence.enabled', False)
            if event_persistence_enabled:
                db_path = get_config('memory.event_persistence.db_path', './data/events.db')
                retention_days = get_config('memory.event_persistence.retention_days', 30)
                batch_size = get_config('memory.event_persistence.batch_size', 100)
                event_store = get_event_store(
                    db_path=db_path,
                    retention_days=retention_days,
                    batch_size=batch_size,
                )
                event_bus.set_event_store(event_store)
                logger.info(f"✅ [EventStoreModule] EventStore 已启用: {db_path} (retention={retention_days}天, batch={batch_size})")
            else:
                logger.info("ℹ️ [EventStoreModule] EventStore 未启用 (event_persistence.enabled=false)")
        except Exception as e:
            logger.warning(f"⚠️ [EventStoreModule] EventStore 初始化失败（非致命）: {e}")
        self.state = ModuleState.RUNNING


# ── 5. EventBusWSModule ──────────────────────────────

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
