# File: zulong/memory/storage_hybrid/global_index.py
# 全局分片索引 - LMDB/mmap 点查层

import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional

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


class GlobalMemoryIndex:
    """MemoryGraph 全局定位索引。

    P1 先落高频点查：node_id -> shard_id，并同步维护
    session/conversation/task_graph 的轻量二级索引。后续可把
    cross_edges.lmdb、summary routing index 也并入该目录。
    """

    def __init__(self, db_path: str, map_size_mb: int = 256, max_readers: int = 128):
        os.makedirs(db_path, exist_ok=True)
        map_size = max(16, map_size_mb) * 1024**2

        self.db_path = db_path
        self.env = lmdb.open(
            db_path,
            map_size=map_size,
            max_dbs=12,
            max_readers=max_readers,
        )

        self.node_to_shard_db = self.env.open_db(b"node_to_shard")
        self.session_to_node_db = self.env.open_db(b"session_to_node")
        self.conversation_to_node_db = self.env.open_db(b"conversation_to_node")
        self.task_graph_to_node_db = self.env.open_db(b"task_graph_to_node")
        self.cross_edges_by_src_db = self.env.open_db(b"cross_edges_by_src")
        self.cross_edges_by_dst_db = self.env.open_db(b"cross_edges_by_dst")
        self.metadata_db = self.env.open_db(b"metadata")

        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder()
        logger.info(f"GlobalMemoryIndex 初始化完成: {db_path} (map_size={map_size_mb}MB)")

    @staticmethod
    def _key(value: str) -> bytes:
        return value.encode("utf-8")

    @staticmethod
    def _is_dialogue_session_root(node_id: str) -> bool:
        return node_id.startswith("dialogue:session_") and "/" not in node_id

    @staticmethod
    def _decode_text(value: Optional[bytes]) -> Optional[str]:
        if not value:
            return None
        try:
            return value.decode("utf-8")
        except Exception:
            return None

    def _put_conversation_target(
        self,
        txn,
        conversation_id: str,
        target: str,
    ) -> None:
        """维护 conversation -> session seed 索引，session 根优先。

        Dialogue round 往往也携带 conversation_id，但恢复入口需要稳定指向
        session 根节点。非根节点只在没有任何映射时作为兜底，避免后写 round
        覆盖已存在的 session seed。
        """
        if not conversation_id or not target:
            return
        key = self._key(str(conversation_id))
        existing = self._decode_text(txn.get(key, db=self.conversation_to_node_db))
        if existing and not self._is_dialogue_session_root(target):
            return
        txn.put(key, self._key(target), db=self.conversation_to_node_db)

    def set_node_location(
        self,
        node_id: str,
        shard_id: str,
        node_type: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        sync: bool = False,
    ) -> None:
        metadata = metadata or {}
        payload = {
            "node_id": node_id,
            "shard_id": shard_id,
            "node_type": node_type,
            "updated_at": time.time(),
        }

        with self.env.begin(write=True) as txn:
            txn.put(
                self._key(node_id),
                self.encoder.encode(payload),
                db=self.node_to_shard_db,
            )

            if self._is_dialogue_session_root(node_id):
                txn.put(self._key(node_id), self._key(node_id), db=self.session_to_node_db)

            session_id = metadata.get("session_id")
            if session_id:
                txn.put(self._key(str(session_id)), self._key(str(session_id)), db=self.session_to_node_db)

            parent_session = metadata.get("parent_session")
            if parent_session:
                txn.put(self._key(str(parent_session)), self._key(str(parent_session)), db=self.session_to_node_db)

            conversation_id = metadata.get("conversation_id") or metadata.get("bound_window_id")
            if conversation_id:
                target = node_id
                if self._is_dialogue_session_root(node_id):
                    target = node_id
                elif session_id:
                    target = str(session_id)
                elif parent_session:
                    target = str(parent_session)
                self._put_conversation_target(txn, str(conversation_id), target)

            task_graph_id = (
                metadata.get("task_graph_id")
                or metadata.get("graph_id")
                or metadata.get("task_id")
            )
            if task_graph_id and node_type == "task":
                txn.put(self._key(str(task_graph_id)), self._key(node_id), db=self.task_graph_to_node_db)

        if sync:
            self.env.sync()

    def set_many_node_locations(self, items: Iterable[Dict[str, Any]], sync: bool = False) -> int:
        count = 0
        with self.env.begin(write=True) as txn:
            for item in items:
                node_id = item.get("node_id")
                shard_id = item.get("shard_id")
                if not node_id or not shard_id:
                    continue
                node_type = item.get("node_type", "")
                metadata = item.get("metadata") or {}
                payload = {
                    "node_id": node_id,
                    "shard_id": shard_id,
                    "node_type": node_type,
                    "updated_at": time.time(),
                }
                txn.put(self._key(node_id), self.encoder.encode(payload), db=self.node_to_shard_db)

                if self._is_dialogue_session_root(node_id):
                    txn.put(self._key(node_id), self._key(node_id), db=self.session_to_node_db)

                session_id = metadata.get("session_id")
                if session_id:
                    txn.put(self._key(str(session_id)), self._key(str(session_id)), db=self.session_to_node_db)

                conversation_id = metadata.get("conversation_id") or metadata.get("bound_window_id")
                if conversation_id:
                    target = node_id if self._is_dialogue_session_root(node_id) else str(session_id or metadata.get("parent_session") or node_id)
                    self._put_conversation_target(txn, str(conversation_id), target)

                task_graph_id = (
                    metadata.get("task_graph_id")
                    or metadata.get("graph_id")
                    or metadata.get("task_id")
                )
                if task_graph_id and node_type == "task":
                    txn.put(self._key(str(task_graph_id)), self._key(node_id), db=self.task_graph_to_node_db)

                count += 1

        if sync:
            self.env.sync()
        return count

    def get_node_shard(self, node_id: str) -> Optional[str]:
        with self.env.begin() as txn:
            data = txn.get(self._key(node_id), db=self.node_to_shard_db)
            if not data:
                return None
            try:
                payload = self.decoder.decode(data)
                return payload.get("shard_id")
            except Exception as exc:
                logger.warning(f"全局索引反序列化失败 node_id={node_id}: {exc}")
                return None

    def delete_node_location(self, node_id: str, sync: bool = False) -> bool:
        if not node_id:
            return False
        removed = False
        with self.env.begin(write=True) as txn:
            node_key = self._key(node_id)
            removed = bool(txn.delete(node_key, db=self.node_to_shard_db))
            for db in (
                self.session_to_node_db,
                self.conversation_to_node_db,
                self.task_graph_to_node_db,
            ):
                self._delete_secondary_refs(txn, db, node_id)
            self._delete_cross_edges_for_node(txn, node_id)
        if sync:
            self.env.sync()
        return removed

    def get_session_node(self, session_id: str) -> Optional[str]:
        return self._get_text(session_id, self.session_to_node_db)

    def get_conversation_node(self, conversation_id: str) -> Optional[str]:
        return self._get_text(conversation_id, self.conversation_to_node_db)

    def get_task_graph_node(self, task_graph_id: str) -> Optional[str]:
        return self._get_text(task_graph_id, self.task_graph_to_node_db)

    def set_cross_edge(self, edge: Dict[str, Any], sync: bool = False) -> None:
        src_id = edge.get("src_id")
        dst_id = edge.get("dst_id")
        if not src_id or not dst_id:
            return

        payload = dict(edge)
        payload.setdefault("updated_at", time.time())

        with self.env.begin(write=True) as txn:
            self._upsert_edge_list(txn, self.cross_edges_by_src_db, src_id, payload, compare_field="dst_id")
            self._upsert_edge_list(txn, self.cross_edges_by_dst_db, dst_id, payload, compare_field="src_id")

        if sync:
            self.env.sync()

    def set_many_cross_edges(self, edges: Iterable[Dict[str, Any]], sync: bool = False) -> int:
        count = 0
        with self.env.begin(write=True) as txn:
            for edge in edges:
                src_id = edge.get("src_id")
                dst_id = edge.get("dst_id")
                if not src_id or not dst_id:
                    continue
                payload = dict(edge)
                payload.setdefault("updated_at", time.time())
                self._upsert_edge_list(txn, self.cross_edges_by_src_db, src_id, payload, compare_field="dst_id")
                self._upsert_edge_list(txn, self.cross_edges_by_dst_db, dst_id, payload, compare_field="src_id")
                count += 1

        if sync:
            self.env.sync()
        return count

    def get_cross_edges_from(self, node_id: str, edge_type: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._get_cross_edges(node_id, self.cross_edges_by_src_db, edge_type=edge_type)

    def get_cross_edges_to(self, node_id: str, edge_type: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._get_cross_edges(node_id, self.cross_edges_by_dst_db, edge_type=edge_type)

    def has_cross_edge(self, src_id: str, dst_id: str) -> bool:
        return any(edge.get("dst_id") == dst_id for edge in self.get_cross_edges_from(src_id))

    def delete_cross_edge(
        self,
        src_id: str,
        dst_id: str,
        edge_type: Optional[str] = None,
        sync: bool = False,
    ) -> bool:
        edge_type_value = getattr(edge_type, "value", edge_type)
        if not src_id or not dst_id:
            return False

        matched = [
            edge for edge in self.get_cross_edges_from(src_id)
            if edge.get("dst_id") == dst_id
            and (edge_type_value is None or edge.get("edge_type") == edge_type_value)
        ]
        if not matched:
            return False

        with self.env.begin(write=True) as txn:
            self._remove_edge_list_item(
                txn,
                self.cross_edges_by_src_db,
                src_id,
                compare_field="dst_id",
                compare_value=dst_id,
                edge_type=edge_type_value,
            )
            self._remove_edge_list_item(
                txn,
                self.cross_edges_by_dst_db,
                dst_id,
                compare_field="src_id",
                compare_value=src_id,
                edge_type=edge_type_value,
            )

        if sync:
            self.env.sync()
        return True

    def count_cross_edges(self, txn=None) -> int:
        count = 0
        own_txn = txn is None
        if own_txn:
            txn = self.env.begin()
        try:
            cursor = txn.cursor(db=self.cross_edges_by_src_db)
            for _, data in cursor:
                try:
                    count += len(self.decoder.decode(data) or [])
                except Exception:
                    continue
        finally:
            if own_txn:
                txn.abort()
        return count

    def iter_cross_edges(self) -> Iterable[Dict[str, Any]]:
        seen = set()
        with self.env.begin() as txn:
            cursor = txn.cursor(db=self.cross_edges_by_src_db)
            for _, data in cursor:
                try:
                    edges = self.decoder.decode(data) or []
                except Exception:
                    continue
                for edge in edges:
                    edge_key = f"{edge.get('src_id')}→{edge.get('dst_id')}"
                    if edge_key in seen:
                        continue
                    seen.add(edge_key)
                    yield edge

    def _get_cross_edges(self, node_id: str, db, edge_type: Optional[str] = None) -> List[Dict[str, Any]]:
        edge_type_value = getattr(edge_type, "value", edge_type)
        with self.env.begin() as txn:
            data = txn.get(self._key(node_id), db=db)
            if not data:
                return []
            try:
                edges = self.decoder.decode(data) or []
            except Exception as exc:
                logger.warning(f"跨分片边反序列化失败 node={node_id}: {exc}")
                return []
        if edge_type_value:
            return [edge for edge in edges if edge.get("edge_type") == edge_type_value]
        return edges

    def _delete_secondary_refs(self, txn, db, node_id: str) -> None:
        node_key_text = str(node_id)
        stale_keys = []
        cursor = txn.cursor(db=db)
        for key, value in cursor:
            key_text = self._decode_text(key)
            value_text = self._decode_text(value)
            if key_text == node_key_text or value_text == node_key_text:
                stale_keys.append(key)
        for key in stale_keys:
            txn.delete(key, db=db)

    def _decode_edge_list(self, data: Optional[bytes]) -> List[Dict[str, Any]]:
        if not data:
            return []
        try:
            return self.decoder.decode(data) or []
        except Exception:
            return []

    def _remove_edge_list_item(
        self,
        txn,
        db,
        node_id: str,
        compare_field: str,
        compare_value: str,
        edge_type: Optional[str],
    ) -> None:
        key = self._key(node_id)
        edges = self._decode_edge_list(txn.get(key, db=db))
        if not edges:
            return
        kept = [
            edge for edge in edges
            if not (
                edge.get(compare_field) == compare_value
                and (edge_type is None or edge.get("edge_type") == edge_type)
            )
        ]
        if kept:
            txn.put(key, self.encoder.encode(kept), db=db)
        else:
            txn.delete(key, db=db)

    def _delete_cross_edges_for_node(self, txn, node_id: str) -> None:
        node_key = self._key(node_id)

        outgoing = self._decode_edge_list(txn.get(node_key, db=self.cross_edges_by_src_db))
        for edge in outgoing:
            dst_id = edge.get("dst_id")
            if dst_id:
                self._remove_edge_list_item(
                    txn,
                    self.cross_edges_by_dst_db,
                    dst_id,
                    compare_field="src_id",
                    compare_value=node_id,
                    edge_type=edge.get("edge_type"),
                )
        txn.delete(node_key, db=self.cross_edges_by_src_db)

        incoming = self._decode_edge_list(txn.get(node_key, db=self.cross_edges_by_dst_db))
        for edge in incoming:
            src_id = edge.get("src_id")
            if src_id:
                self._remove_edge_list_item(
                    txn,
                    self.cross_edges_by_src_db,
                    src_id,
                    compare_field="dst_id",
                    compare_value=node_id,
                    edge_type=edge.get("edge_type"),
                )
        txn.delete(node_key, db=self.cross_edges_by_dst_db)

    def _upsert_edge_list(self, txn, db, node_id: str, edge: Dict[str, Any], compare_field: str) -> None:
        key = self._key(node_id)
        data = txn.get(key, db=db)
        edges = []
        if data:
            try:
                edges = self.decoder.decode(data) or []
            except Exception:
                edges = []

        compare_value = edge.get(compare_field)
        edge_type = edge.get("edge_type")
        replaced = False
        for idx, item in enumerate(edges):
            if item.get(compare_field) == compare_value and item.get("edge_type") == edge_type:
                edges[idx] = edge
                replaced = True
                break
        if not replaced:
            edges.append(edge)

        txn.put(key, self.encoder.encode(edges), db=db)

    def _get_text(self, key: str, db) -> Optional[str]:
        with self.env.begin() as txn:
            data = txn.get(self._key(key), db=db)
            return data.decode("utf-8") if data else None

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
                "node_to_shard": txn.stat(self.node_to_shard_db)["entries"],
                "session_to_node": txn.stat(self.session_to_node_db)["entries"],
                "conversation_to_node": txn.stat(self.conversation_to_node_db)["entries"],
                "task_graph_to_node": txn.stat(self.task_graph_to_node_db)["entries"],
                "cross_edge_src_nodes": txn.stat(self.cross_edges_by_src_db)["entries"],
                "cross_edge_dst_nodes": txn.stat(self.cross_edges_by_dst_db)["entries"],
                "cross_edges": self.count_cross_edges(txn),
                "env_info": self.env.info(),
            }

    def sync(self) -> None:
        self.env.sync()

    def close(self) -> None:
        self.env.sync()
        self.env.close()
