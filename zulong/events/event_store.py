# File: zulong/events/event_store.py
# 事件持久化存储 (基于 SQLite, 零外部依赖)
#
# 设计原则:
#   - 非侵入式: 通过 EventBus 可选钩子集成，不动核心逻辑
#   - 异步写入: 持久化不阻塞事件分发
#   - 可配置: 通过 config 控制开关、保留天数

import json
import logging
import os
import sqlite3
import threading
import time
from typing import Dict, List, Optional

from zulong.core.types import ZulongEvent, EventType

logger = logging.getLogger(__name__)

# 建表 SQL
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    source TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    priority INTEGER NOT NULL DEFAULT 1,
    task_id TEXT DEFAULT '',
    timestamp REAL NOT NULL,
    sequence INTEGER NOT NULL DEFAULT 0
)
"""

_CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_task_id ON events(task_id);
CREATE INDEX IF NOT EXISTS idx_events_sequence ON events(sequence);
"""


class EventStore:
    """基于 SQLite 的事件持久化存储

    使用示例:
        store = EventStore("./data/events.db")
        store.persist(event)
        results = store.query_by_time(start, end)
    """

    def __init__(self, db_path: str, retention_days: int = 30, batch_size: int = 100):
        self._db_path = db_path
        self._retention_days = retention_days
        self._batch_size = batch_size
        self._lock = threading.Lock()
        self._sequence = 0
        self._batch: List[dict] = []
        self._conn: Optional[sqlite3.Connection] = None  # :memory: 模式共享连接

        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        with self._get_conn() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            for stmt in _CREATE_INDEXES_SQL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
            conn.commit()

        logger.info(f"EventStore 初始化: {db_path} (retention={retention_days}天)")

    # ── 公共接口 ─────────────────────────────────

    def persist(self, event: ZulongEvent, task_id: str = "") -> None:
        """持久化单个事件 (异步批量写入)

        Args:
            event: ZulongEvent 实例
            task_id: 关联任务 ID (从 payload 或调用方传入)
        """
        row = {
            "type": event.type.value if isinstance(event.type, EventType) else str(event.type),
            "source": event.source,
            "payload_json": json.dumps(event.payload, ensure_ascii=False, default=str),
            "priority": event.priority.value if hasattr(event.priority, 'value') else 1,
            "task_id": task_id or event.payload.get("task_id", event.payload.get("session_id", "")),
            "timestamp": time.time(),
            "sequence": self._next_sequence(),
        }

        with self._lock:
            self._batch.append(row)
            if len(self._batch) >= self._batch_size:
                self._flush_locked()

    def query_by_time(
        self,
        start: float,
        end: float,
        event_type: Optional[EventType] = None,
        limit: int = 1000,
    ) -> List[dict]:
        """按时间范围查询事件

        Args:
            start: 起始时间戳
            end: 结束时间戳
            event_type: 可选的事件类型过滤
            limit: 最大返回条数

        Returns:
            [{"type": ..., "source": ..., "payload": ..., ...}, ...]
        """
        self._flush()

        type_filter = ""
        params: tuple = (start, end)
        if event_type is not None:
            type_filter = " AND type = ?"
            params = (start, end, event_type.value if isinstance(event_type, EventType) else str(event_type))

        sql = f"SELECT * FROM events WHERE timestamp >= ? AND timestamp <= ?{type_filter} ORDER BY timestamp DESC LIMIT ?"
        params = params + (limit,)

        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_dict(r) for r in rows]

    def query_by_task(self, task_id: str, limit: int = 500) -> List[dict]:
        """按任务 ID 查询关联事件"""
        self._flush()

        sql = "SELECT * FROM events WHERE task_id = ? ORDER BY timestamp DESC LIMIT ?"
        with self._get_conn() as conn:
            rows = conn.execute(sql, (task_id, limit)).fetchall()

        return [self._row_to_dict(r) for r in rows]

    def get_stats(self) -> Dict:
        """获取存储统计信息"""
        self._flush()

        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            oldest = conn.execute(
                "SELECT MIN(timestamp) FROM events"
            ).fetchone()[0]
            newest = conn.execute(
                "SELECT MAX(timestamp) FROM events"
            ).fetchone()[0]
            type_counts = conn.execute(
                "SELECT type, COUNT(*) as cnt FROM events GROUP BY type ORDER BY cnt DESC LIMIT 10"
            ).fetchall()
            db_size = os.path.getsize(self._db_path) if os.path.exists(self._db_path) else 0

        return {
            "total_events": total,
            "oldest_ts": oldest,
            "newest_ts": newest,
            "db_size_bytes": db_size,
            "type_counts": [{"type": t, "count": c} for t, c in type_counts],
            "retention_days": self._retention_days,
        }

    def cleanup(self, retention_days: Optional[int] = None) -> int:
        """清理过期事件

        Args:
            retention_days: 保留天数 (默认使用初始化时的值)

        Returns:
            删除的事件数
        """
        self._flush()

        days = retention_days or self._retention_days
        cutoff = time.time() - (days * 86400)

        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
            deleted = cursor.rowcount
            conn.commit()

        if deleted > 0:
            logger.info(f"EventStore 清理: 删除 {deleted} 条过期事件 (cutoff={days}天前)")
        return deleted

    def close(self) -> None:
        """关闭存储 (将缓冲写入磁盘)"""
        self._flush()
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        logger.info("EventStore 已关闭")

    # ── 内部方法 ─────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接

        :memory: 模式使用缓存的单一连接（确保表共享），
        文件模式每次新建（支持多线程安全）。
        """
        if self._db_path == ":memory:":
            if self._conn is None:
                self._conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
            return self._conn
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _flush(self) -> None:
        """强制写入缓冲区"""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """写入缓冲区 (需已持有 _lock)"""
        if not self._batch:
            return

        rows = self._batch[:]
        self._batch.clear()

        try:
            with self._get_conn() as conn:
                conn.executemany(
                    """INSERT INTO events (type, source, payload_json, priority, task_id, timestamp, sequence)
                       VALUES (:type, :source, :payload_json, :priority, :task_id, :timestamp, :sequence)""",
                    rows,
                )
                conn.commit()
        except Exception:
            logger.error("EventStore 批量写入失败", exc_info=True)
            # 不丢失数据: 放回缓冲区头部
            with self._lock:
                self._batch = rows + self._batch

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        """将 SQLite Row 转为字典，还原 payload JSON"""
        d = dict(row)
        try:
            d["payload"] = json.loads(d.pop("payload_json", "{}"))
        except (json.JSONDecodeError, KeyError):
            d["payload"] = {}
            d.pop("payload_json", None)
        return d


# ── 全局单例 ───────────────────────────────────

_event_store: Optional[EventStore] = None


def get_event_store(
    db_path: Optional[str] = None,
    retention_days: int = 30,
    batch_size: int = 100,
) -> EventStore:
    """获取全局 EventStore 单例"""
    global _event_store
    if _event_store is None:
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data", "events.db",
            )
        _event_store = EventStore(
            db_path=db_path,
            retention_days=retention_days,
            batch_size=batch_size,
        )
    return _event_store
