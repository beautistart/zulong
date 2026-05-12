from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

from test_system.engine.models import StateSnapshot
from test_system.storage.result_store import TestResultStore

logger = logging.getLogger(__name__)


class SnapshotCorruptionError(Exception):
    pass


class SnapshotManager:
    def __init__(self, store: TestResultStore, backend_proxy=None):
        self._store = store
        self._backend_proxy = backend_proxy

    async def capture(self, execution_id: str, step_id: str, state_data: dict) -> StateSnapshot:
        snapshot_id = f"snap_{uuid.uuid4().hex[:12]}"
        snapshot = StateSnapshot(
            snapshot_id=snapshot_id,
            execution_id=execution_id,
            step_id=step_id,
            state_data=state_data,
            integrity_hash="",
            timestamp=datetime.now().isoformat(),
        )
        snapshot.integrity_hash = snapshot.compute_hash()

        await self._store.save_snapshot(
            snapshot_id=snapshot.snapshot_id,
            execution_id=execution_id,
            step_id=step_id,
            state_data=state_data,
            integrity_hash=snapshot.integrity_hash,
        )
        logger.info("快照已保存: %s (execution=%s, step=%s)", snapshot_id, execution_id, step_id)
        return snapshot

    async def load(self, snapshot_id: str) -> StateSnapshot:
        row = await self._store.get_snapshot(snapshot_id)
        if not row:
            raise FileNotFoundError(f"快照不存在: {snapshot_id}")

        snapshot = StateSnapshot(
            snapshot_id=row["snapshot_id"],
            execution_id=row["execution_id"],
            step_id=row["step_id"],
            state_data=json.loads(row["state_data_json"]) if isinstance(row["state_data_json"], str) else row["state_data_json"],
            integrity_hash=row["integrity_hash"],
            timestamp=row["timestamp"],
        )

        if not snapshot.verify_integrity():
            raise SnapshotCorruptionError(f"快照完整性校验失败: {snapshot_id}")

        return snapshot

    async def cleanup_old(self, days: int = 30) -> int:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        async with self._store._db.execute("DELETE FROM state_snapshots WHERE timestamp < ?", (cutoff,)) as cursor:
            count = cursor.rowcount
        await self._store._db.commit()
        return count


import json
