from __future__ import annotations

import asyncio
import logging
from typing import Optional

from test_system.definition.loader import TestCaseLoader
from test_system.engine.executor import TestExecutor
from test_system.engine.progress_tracker import ProgressTracker, StagnationDetector
from test_system.engine.snapshot_manager import SnapshotManager
from test_system.integration.backend_proxy import BackendProxy
from test_system.integration.event_bridge import EventBridge, TestMonitorWSService
from test_system.storage.log_store import TestLogStore
from test_system.storage.result_store import TestResultStore
from test_system.api.routes import set_executor

logger = logging.getLogger(__name__)


class TestSystemApp:
    def __init__(
        self,
        cases_dir: str = "",
        db_path: str = "test_system.db",
        backend_uri: str = "ws://127.0.0.1:8090/ide",
        monitor_host: str = "127.0.0.1",
        monitor_port: int = 8091,
        enabled: bool = True,
    ):
        self.cases_dir = cases_dir
        self.db_path = db_path
        self.backend_uri = backend_uri
        self.monitor_host = monitor_host
        self.monitor_port = monitor_port
        self.enabled = enabled

        self._store: Optional[TestResultStore] = None
        self._log_store: Optional[TestLogStore] = None
        self._proxy: Optional[BackendProxy] = None
        self._bridge: Optional[EventBridge] = None
        self._monitor_ws: Optional[TestMonitorWSService] = None
        self._progress: Optional[ProgressTracker] = None
        self._snapshot_mgr: Optional[SnapshotManager] = None
        self._executor: Optional[TestExecutor] = None
        self._loader: Optional[TestCaseLoader] = None
        self._running = False

    async def start(self) -> None:
        if not self.enabled:
            logger.info("测试系统未启用")
            return

        logger.info("测试系统启动中...")

        self._store = TestResultStore(self.db_path)
        await self._store.initialize()

        self._log_store = TestLogStore()

        self._bridge = EventBridge()

        self._proxy = BackendProxy(self.backend_uri)

        self._progress = ProgressTracker(self._bridge, StagnationDetector())

        self._snapshot_mgr = SnapshotManager(self._store, self._proxy)

        self._executor = TestExecutor(
            store=self._store,
            log_store=self._log_store,
            backend_proxy=self._proxy,
            event_bridge=self._bridge,
            progress_tracker=self._progress,
            snapshot_manager=self._snapshot_mgr,
        )

        set_executor(self._executor)

        if self.cases_dir:
            self._loader = TestCaseLoader(self.cases_dir)
            test_cases = self._loader.load_all()
            for tc in test_cases:
                await self._store.save_test_case(tc)
            logger.info("已加载 %d 个测试用例", len(test_cases))

        self._monitor_ws = TestMonitorWSService(self._bridge)
        await self._monitor_ws.start(host=self.monitor_host, port=self.monitor_port)

        self._running = True
        logger.info("测试系统启动完成 (monitor: ws://%s:%d)", self.monitor_host, self.monitor_port)

    async def stop(self) -> None:
        if not self._running:
            return

        logger.info("测试系统关闭中...")

        active = self._executor.get_active_runs() if self._executor else []
        for run in active:
            await self._executor.stop_test(run.execution_id)

        if self._monitor_ws:
            await self._monitor_ws.stop()

        if self._proxy:
            await self._proxy.disconnect()

        if self._store:
            await self._store.close()

        self._running = False
        logger.info("测试系统已关闭")

    @property
    def executor(self) -> Optional[TestExecutor]:
        return self._executor

    @property
    def is_running(self) -> bool:
        return self._running
