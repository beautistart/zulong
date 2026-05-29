# File: zulong/memory/storage_hybrid/local_index.py
# 分片局部轻量索引 - 冷分片分级唤醒的第一层

import logging
import os
import time
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

try:
    import lmdb
except ImportError:
    logger.error("lmdb 未安装，请执行: pip install lmdb")
    raise

try:
    import msgspec
except ImportError:
    logger.error("msgspec 未安装，请执行: pip install msgspec")
    raise


class LocalShardIndex:
    """单分片轻量索引。

    只保存节点头信息，用于在打开完整 properties/topology 前确认节点
    是否属于该冷分片。完整内容仍以 properties.lmdb 为权威。
    """

    def __init__(self, db_path: str, shard_id: str, map_size_mb: int = 64, max_readers: int = 64):
        os.makedirs(db_path, exist_ok=True)
        map_size = max(16, int(map_size_mb or 64)) * 1024**2

        self.db_path = db_path
        self.shard_id = shard_id
        self.env = lmdb.open(
            db_path,
            map_size=map_size,
            max_dbs=8,
            max_readers=max_readers,
        )
        self.node_headers_db = self.env.open_db(b"node_headers")
        self.node_type_db = self.env.open_db(b"node_type_idx")
        self.metadata_db = self.env.open_db(b"metadata")
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder()
        logger.debug(f"LocalShardIndex 初始化: shard={shard_id}, path={db_path}")

    @staticmethod
    def _key(value: str) -> bytes:
        return str(value).encode("utf-8")

    @staticmethod
    def _type_value(node_type: Any) -> str:
        return str(getattr(node_type, "value", node_type) or "")

    def set_node_header(
        self,
        node_id: str,
        node_type: Any = "",
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        sync: bool = False,
    ) -> None:
        if not node_id:
            return
        payload = {
            "node_id": node_id,
            "shard_id": self.shard_id,
            "node_type": self._type_value(node_type),
            "label": label or "",
            "metadata": metadata or {},
            "updated_at": time.time(),
        }
        with self.env.begin(write=True) as txn:
            key = self._key(node_id)
            txn.put(key, self.encoder.encode(payload), db=self.node_headers_db)
            if payload["node_type"]:
                txn.put(self._key(f"{payload['node_type']}:{node_id}"), key, db=self.node_type_db)
        if sync:
            self.env.sync()

    def set_many_node_headers(self, items: Iterable[Dict[str, Any]], sync: bool = False) -> int:
        count = 0
        with self.env.begin(write=True) as txn:
            for item in items:
                node_id = item.get("node_id")
                if not node_id:
                    continue
                node_type = self._type_value(item.get("node_type", ""))
                payload = {
                    "node_id": node_id,
                    "shard_id": self.shard_id,
                    "node_type": node_type,
                    "label": item.get("label") or "",
                    "metadata": item.get("metadata") or {},
                    "updated_at": time.time(),
                }
                key = self._key(node_id)
                txn.put(key, self.encoder.encode(payload), db=self.node_headers_db)
                if node_type:
                    txn.put(self._key(f"{node_type}:{node_id}"), key, db=self.node_type_db)
                count += 1
        if sync:
            self.env.sync()
        return count

    def has_node(self, node_id: str) -> bool:
        with self.env.begin() as txn:
            return txn.get(self._key(node_id), db=self.node_headers_db) is not None

    def get_node_header(self, node_id: str) -> Optional[Dict[str, Any]]:
        with self.env.begin() as txn:
            data = txn.get(self._key(node_id), db=self.node_headers_db)
            if not data:
                return None
            try:
                return self.decoder.decode(data)
            except Exception as exc:
                logger.warning(f"local_index 反序列化失败 node={node_id}: {exc}")
                return None

    def count_nodes(self) -> int:
        with self.env.begin() as txn:
            return txn.stat(self.node_headers_db)["entries"]

    def clear(self) -> None:
        with self.env.begin(write=True) as txn:
            txn.drop(self.node_headers_db, delete=False)
            txn.drop(self.node_type_db, delete=False)
            txn.drop(self.metadata_db, delete=False)

    def mark_rebuilt(self, source: str = "") -> None:
        with self.env.begin(write=True) as txn:
            txn.put(
                b"last_rebuilt",
                self.encoder.encode({"source": source, "timestamp": time.time()}),
                db=self.metadata_db,
            )

    def get_stats(self) -> Dict[str, Any]:
        with self.env.begin() as txn:
            return {
                "db_path": self.db_path,
                "shard_id": self.shard_id,
                "node_headers": txn.stat(self.node_headers_db)["entries"],
                "node_type_entries": txn.stat(self.node_type_db)["entries"],
                "env_info": self.env.info(),
            }

    def sync(self) -> None:
        self.env.sync()

    def close(self) -> None:
        self.env.sync()
        self.env.close()
