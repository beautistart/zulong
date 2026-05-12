from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from datetime import datetime
from typing import Optional

from test_system.definition.enums import TestStatus

logger = logging.getLogger(__name__)


class TestLogStore:
    def __init__(self, max_entries: int = 10000):
        self._entries: deque = deque(maxlen=max_entries)
        self._flush_queue: deque = deque(maxlen=max_entries)
        self._lock = asyncio.Lock()

    def log(self, execution_id: str, level: str, source: str, message: str) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "source": source,
            "message": message,
            "execution_id": execution_id,
        }
        self._entries.append(entry)
        self._flush_queue.append(entry)
        if level == "ERROR":
            logger.error("[%s] %s: %s", source, execution_id, message)
        elif level == "WARN":
            logger.warning("[%s] %s: %s", source, execution_id, message)
        elif level == "DEBUG":
            logger.debug("[%s] %s: %s", source, execution_id, message)
        else:
            logger.info("[%s] %s: %s", source, execution_id, message)

    async def query(
        self,
        execution_id: Optional[str] = None,
        level: Optional[str] = None,
        page: int = 1,
        size: int = 200,
    ) -> list[dict]:
        entries = list(self._entries)
        if execution_id:
            entries = [e for e in entries if e["execution_id"] == execution_id]
        if level:
            entries = [e for e in entries if e["level"] == level]
        start = (page - 1) * size
        return entries[start : start + size]

    async def get_recent(self, count: int = 200, since: Optional[str] = None) -> list[dict]:
        entries = list(self._entries)
        if since:
            entries = [e for e in entries if e["timestamp"] > since]
        return entries[-count:]

    def flush_buffer(self) -> list[dict]:
        items = list(self._flush_queue)
        self._flush_queue.clear()
        return items
