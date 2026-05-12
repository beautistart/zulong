from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Optional

from test_system.definition.enums import AttentionRestoreStatus, TestStatus
from test_system.engine.models import AttentionContext, AttentionSwitchEvent
from test_system.integration.backend_proxy import BackendProxy
from test_system.integration.event_bridge import EventBridge, TestEvent
from test_system.storage.log_store import TestLogStore

logger = logging.getLogger(__name__)


class AttentionSimulator:
    def __init__(self, backend_proxy: BackendProxy, event_bridge: EventBridge, log_store: TestLogStore):
        self._proxy = backend_proxy
        self._bridge = event_bridge
        self._log = log_store
        self._active_sessions: dict[str, AttentionContext] = {}
        self._current_session_id: Optional[str] = None
        self._switch_queue: asyncio.Queue = asyncio.Queue()

    async def start_parallel_tasks(self, task_configs: list[dict]) -> list[str]:
        session_ids = []
        for config in task_configs:
            session_id = f"sess_{uuid.uuid4().hex[:8]}"
            try:
                await self._proxy.start_session(config)
                context = AttentionContext(
                    session_id=session_id,
                    attention_window_data=config.get("attention_window_data", {}),
                )
                self._active_sessions[session_id] = context
                session_ids.append(session_id)
                self._log.log(session_id, "INFO", "AttentionSimulator", f"并行任务启动: {session_id}")
            except Exception as e:
                self._log.log(session_id, "ERROR", "AttentionSimulator", f"并行任务启动失败: {e}")
        if session_ids:
            self._current_session_id = session_ids[0]
        return session_ids

    async def switch_attention(self, from_session_id: str, to_session_id: str, execution_id: str = "") -> AttentionSwitchEvent:
        start_ms = int(time.time() * 1000)

        if from_session_id in self._active_sessions:
            self._active_sessions[from_session_id].attention_window_data["_suspended_at"] = datetime.now().isoformat()

        restore_status = AttentionRestoreStatus.RESTORED
        if to_session_id not in self._active_sessions:
            new_session_id = f"sess_{uuid.uuid4().hex[:8]}"
            context = AttentionContext(session_id=new_session_id, attention_window_data={})
            self._active_sessions[to_session_id] = context
            restore_status = AttentionRestoreStatus.COLD_START
            self._log.log(execution_id, "WARN", "AttentionSimulator", f"冷启动: 目标Session {to_session_id} 过期，创建新Session")
        else:
            try:
                target = self._active_sessions[to_session_id]
                if target.attention_window_data:
                    pass
            except Exception as e:
                restore_status = AttentionRestoreStatus.FAILED
                self._active_sessions[to_session_id] = AttentionContext(session_id=to_session_id, attention_window_data={})
                self._log.log(execution_id, "WARN", "AttentionSimulator", f"上下文恢复失败: {e}")

        elapsed_ms = int(time.time() * 1000) - start_ms

        event = AttentionSwitchEvent(
            from_session_id=from_session_id,
            to_session_id=to_session_id,
            restore_status=restore_status,
            elapsed_ms=elapsed_ms,
        )
        self._current_session_id = to_session_id

        await self._bridge.emit(TestEvent.ATTENTION_SWITCH, event.to_dict())
        self._log.log(execution_id, "INFO", "AttentionSimulator", f"注意力切换: {from_session_id} → {to_session_id} ({restore_status.value}, {elapsed_ms}ms)")
        return event

    @property
    def current_session_id(self) -> Optional[str]:
        return self._current_session_id

    @property
    def active_sessions(self) -> dict[str, AttentionContext]:
        return dict(self._active_sessions)
