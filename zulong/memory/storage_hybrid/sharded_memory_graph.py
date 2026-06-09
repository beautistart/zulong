# File: zulong/memory/storage_hybrid/sharded_memory_graph.py
# 分片管理器 - 按时间切分的大规模记忆图谱存储
#
# 核心特性:
# - 时间分片策略（按月/周切分）
# - LRU缓存活跃分片（近3个月常驻内存）
# - 跨分片关联发现
# - 单分片50-200MB，总规模可达年级别

import asyncio
import logging
import os
import time
import json
import pickle
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
from collections import OrderedDict, deque
import threading

logger = logging.getLogger(__name__)

from .memory_graph_hybrid import MemoryGraphHybrid, NodeProperties, EdgeProperties
from .global_index import GlobalMemoryIndex
from .local_index import LocalShardIndex


class ShardStrategy:
    """分片策略"""
    MONTHLY = "month"
    WEEKLY = "week"
    DAILY = "day"


class ShardedMemoryGraph:
    """
    分片管理器 - 大规模记忆图谱存储
    
    分片策略:
    - 按时间切分（默认按月）
    - 单分片目标<50万节点
    - 近3个月分片常驻内存
    - 更早分片按需加载（LRU淘汰）
    
    性能指标:
    - 单分片加载: 100-500ms
    - 跨分片查询: 10-20ms（含加载）
    """
    
    def __init__(
        self,
        base_dir: str,
        shard_strategy: str = ShardStrategy.MONTHLY,
        max_active_shards: int = 3,
        map_size_gb: int = 10,
        enable_vector_index: bool = False,
        max_nodes_per_shard: int = 150_000,
        max_shard_property_mb_warning: int = 150,
        max_shard_property_mb_split: int = 200,
        max_shard_topology_mb_warning: int = 64,
        max_shard_topology_delta_mb_compact: float = 32,
        max_active_skeleton_nodes: int = 50_000,
        local_index_map_size_mb: int = 64,
        shard_size_check_interval_nodes: int = 100,
        global_index_map_size_mb: int = 256
    ):
        """
        初始化分片管理器
        
        Args:
            base_dir: 数据基础目录
            shard_strategy: 分片策略（month/week/day）
            max_active_shards: 最大活跃分片数
            map_size_gb: 每个分片的LMDB映射大小（GB）
            enable_vector_index: 是否启用向量索引
            max_nodes_per_shard: 单分片节点数上限（超限触发警告/自动分裂）
            max_shard_property_mb_warning: properties LMDB 使用量预警阈值（MB，<=0禁用）
            max_shard_property_mb_split: properties LMDB 使用量分裂阈值（MB，<=0禁用）
            max_shard_topology_mb_warning: topology.graphml 预警阈值（MB，<=0禁用）
            max_shard_topology_delta_mb_compact: topology_delta.log 后台压实阈值（MB，<=0禁用）
            max_active_skeleton_nodes: active skeleton 节点预算（<=0 表示不限制）
            local_index_map_size_mb: 单分片局部索引 LMDB 映射大小（MB）
            shard_size_check_interval_nodes: 每写入多少节点刷新一次物理大小统计
            global_index_map_size_mb: 全局索引 LMDB 映射大小（MB）
        """
        os.makedirs(base_dir, exist_ok=True)
        
        self.base_dir = base_dir
        self.shard_strategy = shard_strategy
        self.max_active_shards = max_active_shards
        self.map_size_gb = map_size_gb
        self.enable_vector_index = enable_vector_index
        self.max_nodes_per_shard = max_nodes_per_shard
        self.max_shard_property_mb_warning = max_shard_property_mb_warning
        self.max_shard_property_mb_split = max_shard_property_mb_split
        self.max_shard_topology_mb_warning = max_shard_topology_mb_warning
        self.max_shard_topology_delta_mb_compact = max_shard_topology_delta_mb_compact
        self.max_active_skeleton_nodes = max_active_skeleton_nodes
        self.local_index_map_size_mb = local_index_map_size_mb
        self.shard_size_check_interval_nodes = max(1, shard_size_check_interval_nodes)
        self.global_index = GlobalMemoryIndex(
            db_path=os.path.join(base_dir, "global_index", "index.lmdb"),
            map_size_mb=global_index_map_size_mb
        )
        self._rag_manager = None  # RAGManager 引用
        
        self.active_shards: OrderedDict[str, MemoryGraphHybrid] = OrderedDict()
        self.local_indexes: OrderedDict[str, LocalShardIndex] = OrderedDict()
        self.shard_lock = threading.RLock()
        
        self.shard_index = self._load_shard_index()
        self._migrate_json_cross_edges_to_global_index()
        
        self._stats = {
            "shard_load_count": 0,
            "shard_evict_count": 0,
            "cross_shard_query_count": 0,
            "auto_split_count": 0,
            "shard_size_warning_count": 0,
            "global_index_hit_count": 0,
            "global_index_miss_count": 0,
            "global_index_repair_count": 0,
            "cross_edge_migration_count": 0,
            "topology_rebuild_count": 0,
            "topology_compact_count": 0,
            "topology_compact_enqueue_count": 0,
            "topology_compact_failure_count": 0,
            "topology_compact_retry_count": 0,
            "active_skeleton_prune_count": 0,
            "local_index_hit_count": 0,
            "local_index_miss_count": 0,
            "local_index_rebuild_count": 0,
            "active_skeleton_enqueue_count": 0,
            "active_skeleton_background_refresh_count": 0,
        }
        self._last_size_check_node_counts: Dict[str, int] = {}
        self._compacting_shards: Set[str] = set()
        self._topology_compact_queue = deque()
        self._queued_topology_compactions: Set[str] = set()
        self._topology_compact_failures: Dict[str, int] = {}
        self._topology_compact_retry_after: Dict[str, float] = {}
        self._topology_compact_worker: Optional[threading.Thread] = None
        self._last_topology_write_at = 0.0
        self._active_node_ids: Set[str] = set()
        self._last_focus_context: Optional[Dict[str, Any]] = None
        self._last_activated_edges: List[Tuple[str, str]] = []  # (src_id, dst_id) pairs for Hebbian learning
        self._last_retrieved_node_ids: List[str] = []  # top-3 node IDs from last retrieve_context
        self._active_skeleton: Dict[str, Any] = self._load_active_skeleton()
        self._skeleton_refresh_pending_ids: Set[str] = set()
        self._skeleton_refresh_worker: Optional[threading.Thread] = None
        self._last_skeleton_refresh_at = 0.0
        
        logger.info(
            f"ShardedMemoryGraph 初始化完成: "
            f"strategy={shard_strategy}, "
            f"max_active={max_active_shards}, "
            f"max_nodes_per_shard={max_nodes_per_shard}, "
            f"property_warning_mb={max_shard_property_mb_warning}, "
            f"property_split_mb={max_shard_property_mb_split}, "
            f"active_skeleton_budget={max_active_skeleton_nodes}"
        )
        
    def _get_shard_id(self, timestamp: float) -> str:
        """时间戳 → 分片ID"""
        dt = datetime.fromtimestamp(timestamp)
        
        if self.shard_strategy == ShardStrategy.MONTHLY:
            return f"{dt.year}_{dt.month:02d}"
        elif self.shard_strategy == ShardStrategy.WEEKLY:
            week = dt.isocalendar()[1]
            return f"{dt.year}_W{week:02d}"
        elif self.shard_strategy == ShardStrategy.DAILY:
            return f"{dt.year}_{dt.month:02d}_{dt.day:02d}"
        else:
            return f"{dt.year}_{dt.month:02d}"
            
    def _get_shard_path(self, shard_id: str) -> str:
        """分片ID → 文件路径"""
        return os.path.join(self.base_dir, f"shard_{shard_id}")

    def _get_local_index_path(self, shard_id: str) -> str:
        return os.path.join(self._get_shard_path(shard_id), "local_index.lmdb")

    def get_local_index(
        self,
        shard_id: str,
        create: bool = True,
    ) -> Optional[LocalShardIndex]:
        """获取分片局部小索引，不打开完整 properties/topology。"""
        if not shard_id:
            return None
        if shard_id in self.local_indexes:
            self.local_indexes.move_to_end(shard_id)
            return self.local_indexes[shard_id]

        index_path = self._get_local_index_path(shard_id)
        shard_exists = os.path.exists(self._get_shard_path(shard_id))
        index_exists = os.path.exists(index_path)
        if not create and not index_exists:
            return None
        if not shard_exists and not create:
            return None

        try:
            local_index = LocalShardIndex(
                db_path=index_path,
                shard_id=shard_id,
                map_size_mb=self.local_index_map_size_mb,
            )
        except Exception as exc:
            logger.debug(f"local_index 打开失败 shard={shard_id}: {exc}")
            return None

        while len(self.local_indexes) >= max(self.max_active_shards * 2, 4):
            oldest_id, oldest_index = self.local_indexes.popitem(last=False)
            try:
                oldest_index.close()
            except Exception:
                pass
            logger.debug(f"淘汰 local_index: {oldest_id}")

        self.local_indexes[shard_id] = local_index
        return local_index

    def _index_local_node_header(
        self,
        shard_id: str,
        node_id: str,
        node_type: str = "",
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            local_index = self.get_local_index(shard_id, create=True)
            if local_index:
                local_index.set_node_header(
                    node_id=node_id,
                    node_type=node_type,
                    label=label,
                    metadata=metadata or {},
                )
                self._record_local_index_update(shard_id)
        except Exception as exc:
            logger.debug(f"local_index 写入失败 shard={shard_id}, node={node_id}: {exc}")

    def _local_index_has_node(self, shard_id: str, node_id: str) -> Optional[bool]:
        """轻量判断节点是否在某冷分片。None 表示索引不存在/不可用。"""
        local_index = self.get_local_index(shard_id, create=False)
        if not local_index:
            return None
        try:
            found = local_index.has_node(node_id)
            if found:
                self._stats["local_index_hit_count"] += 1
                return True
            if not self._is_local_index_complete(shard_id):
                return None
            self._stats["local_index_miss_count"] += 1
            return False
        except Exception as exc:
            logger.debug(f"local_index 查询失败 shard={shard_id}, node={node_id}: {exc}")
            return None

    def _is_local_index_complete(self, shard_id: str) -> bool:
        info = self.shard_index.get("shards", {}).get(shard_id, {}).get("local_index") or {}
        if not info.get("complete"):
            return False
        indexed = int(info.get("node_count") or 0)
        expected = int(self.shard_index.get("shards", {}).get(shard_id, {}).get("node_count") or 0)
        return expected <= 0 or indexed >= expected

    def _record_local_index_update(self, shard_id: str) -> None:
        shard_info = self.shard_index.setdefault("shards", {}).setdefault(shard_id, {})
        local_info = shard_info.setdefault("local_index", {})
        local_info["updated_at"] = time.time()
        if local_info.get("complete"):
            local_info["node_count"] = max(
                int(local_info.get("node_count") or 0),
                int(shard_info.get("node_count") or 0),
            )

    def _active_skeleton_path(self) -> str:
        return os.path.join(self.base_dir, "active_topology", "skeleton.bin")

    def _load_active_skeleton(self) -> Dict[str, Any]:
        path = self._active_skeleton_path()
        if not os.path.exists(path):
            return {
                "version": 1,
                "updated_at": 0.0,
                "node_ids": [],
                "edges": [],
                "center_ids": [],
                "stats": {"node_count": 0, "edge_count": 0},
            }
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict):
                raise ValueError("active skeleton payload is not dict")
            data.setdefault("version", 1)
            data.setdefault("node_ids", [])
            data.setdefault("edges", [])
            data.setdefault("center_ids", [])
            data.setdefault("stats", {
                "node_count": len(data.get("node_ids", [])),
                "edge_count": len(data.get("edges", [])),
            })
            return data
        except Exception as exc:
            logger.debug(f"active skeleton 加载失败，使用空骨架: {exc}")
            return {
                "version": 1,
                "updated_at": 0.0,
                "node_ids": [],
                "edges": [],
                "center_ids": [],
                "stats": {"node_count": 0, "edge_count": 0},
            }

    def _save_active_skeleton(self) -> None:
        path = self._active_skeleton_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(self._active_skeleton, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
        except Exception as exc:
            logger.debug(f"active skeleton 保存失败: {exc}")

    def _drop_node_from_active_state(self, node_id: str) -> None:
        self._active_node_ids.discard(node_id)
        self._last_retrieved_node_ids = [
            nid for nid in self._last_retrieved_node_ids if nid != node_id
        ]
        self._last_activated_edges = [
            (src_id, dst_id)
            for src_id, dst_id in self._last_activated_edges
            if src_id != node_id and dst_id != node_id
        ]

        skeleton = dict(self._active_skeleton or {})
        if not skeleton:
            return
        node_ids = [nid for nid in skeleton.get("node_ids", []) if nid != node_id]
        center_ids = [nid for nid in skeleton.get("center_ids", []) if nid != node_id]
        edges = [
            edge for edge in skeleton.get("edges", [])
            if edge.get("source") != node_id and edge.get("target") != node_id
        ]
        stats = dict(skeleton.get("stats") or {})
        stats["node_count"] = len(node_ids)
        stats["edge_count"] = len(edges)
        stats["center_count"] = len(center_ids)

        skeleton.update({
            "updated_at": time.time(),
            "node_ids": node_ids,
            "edges": edges,
            "center_ids": center_ids,
            "stats": stats,
        })
        self._active_skeleton = skeleton
        self._save_active_skeleton()

    def _score_active_skeleton_node(
        self,
        node: Optional[NodeProperties],
        node_id: str,
        center_ids: Set[str],
        now: float,
    ) -> float:
        """为 active skeleton 候选节点打分，用于预算淘汰。"""
        score = 0.0
        if node_id in center_ids:
            score += 1_000_000.0
        if not node:
            return score

        score += float(getattr(node, "activation", 0.0) or 0.0) * 100.0
        score += min(float(getattr(node, "access_count", 0) or 0), 100.0)

        importance = str(getattr(node, "importance", "normal") or "normal").lower()
        score += {
            "must_remember": 500.0,
            "identity": 420.0,
            "important": 340.0,
            "fact": 260.0,
            "normal": 80.0,
            "trivial": 5.0,
        }.get(importance, 60.0)

        node_type = str(getattr(node, "node_type", "") or "").lower()
        if node_type in {"task", "dialogue", "tool_call", "tool_result", "approval"}:
            score += 120.0

        last_accessed = float(getattr(node, "last_accessed", 0.0) or 0.0)
        if last_accessed > 0:
            age_seconds = max(0.0, now - last_accessed)
            if age_seconds <= 30 * 60:
                score += 220.0
            elif age_seconds <= 6 * 3600:
                score += 120.0
            elif age_seconds <= 24 * 3600:
                score += 60.0

        metadata = getattr(node, "metadata", None) or {}
        if metadata.get("task_graph_id") or metadata.get("conversation_id"):
            score += 80.0
        return score

    def refresh_active_skeleton(
        self,
        center_ids: Optional[List[str]] = None,
        max_neighbors_per_center: int = 24,
    ) -> Dict[str, Any]:
        """刷新热/活跃骨架快照。

        兼容期只生成 active 节点及一跳邻接，不替换运行态 BFS。
        """
        center_list = []
        seen_centers = set()
        for node_id in list(center_ids or []) + list(self._active_node_ids) + list(self._last_retrieved_node_ids):
            if node_id and node_id not in seen_centers:
                center_list.append(node_id)
                seen_centers.add(node_id)

        mandatory_ids: Set[str] = set(center_list)
        node_ids: Set[str] = set(center_list)
        edges: List[Dict[str, Any]] = []
        seen_edges: Set[str] = set()

        for center_id in center_list:
            for node in self.get_neighbors(center_id, max_depth=1)[:max_neighbors_per_center]:
                node_ids.add(node.node_id)
                edge = self.get_edge(center_id, node.node_id)
                if not edge:
                    edge = self.get_edge(node.node_id, center_id)
                src_id = edge.get("src_id", center_id) if isinstance(edge, dict) else center_id
                dst_id = edge.get("dst_id", node.node_id) if isinstance(edge, dict) else node.node_id
                edge_key = f"{src_id}->{dst_id}:{edge.get('edge_type', edge.get('type', 'association')) if isinstance(edge, dict) else 'association'}"
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edges.append({
                    "source": src_id,
                    "target": dst_id,
                    "type": edge.get("edge_type", edge.get("type", "association")) if isinstance(edge, dict) else "association",
                    "weight": edge.get("weight", 1.0) if isinstance(edge, dict) else 1.0,
                })

        original_node_count = len(node_ids)
        budget = int(self.max_active_skeleton_nodes or 0)
        pruned = False
        retained_ids = set(node_ids)
        effective_budget = max(budget, len(mandatory_ids)) if budget > 0 else 0
        if effective_budget > 0 and len(node_ids) > effective_budget:
            now = time.time()
            scored_nodes = []
            node_cache: Dict[str, Optional[NodeProperties]] = {}
            for node_id in node_ids:
                node = self.get_node(node_id)
                node_cache[node_id] = node
                scored_nodes.append((
                    self._score_active_skeleton_node(node, node_id, mandatory_ids, now),
                    float(getattr(node, "last_accessed", 0.0) or 0.0) if node else 0.0,
                    node_id,
                ))

            scored_nodes.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
            retained_ids = {node_id for _, _, node_id in scored_nodes[:effective_budget]}
            retained_ids.update(mandatory_ids)

            if len(retained_ids) > effective_budget:
                mandatory_sorted = sorted(
                    retained_ids,
                    key=lambda node_id: (
                        0 if node_id in mandatory_ids else 1,
                        -self._score_active_skeleton_node(node_cache.get(node_id), node_id, mandatory_ids, now),
                        node_id,
                    )
                )
                retained_ids = set(mandatory_sorted[:effective_budget])

            edges = [
                edge for edge in edges
                if edge.get("source") in retained_ids and edge.get("target") in retained_ids
            ]
            pruned = len(retained_ids) < original_node_count
            if pruned:
                self._stats["active_skeleton_prune_count"] += 1

        self._active_skeleton = {
            "version": 1,
            "updated_at": time.time(),
            "node_ids": sorted(retained_ids),
            "edges": edges,
            "center_ids": center_list,
            "stats": {
                "node_count": len(retained_ids),
                "edge_count": len(edges),
                "center_count": len(center_list),
                "budget": budget,
                "effective_budget": effective_budget,
                "pruned": pruned,
                "candidate_node_count": original_node_count,
                "pruned_node_count": max(0, original_node_count - len(retained_ids)),
            },
        }
        self._save_active_skeleton()
        self._last_skeleton_refresh_at = time.time()
        return self._active_skeleton

    def get_active_skeleton(self) -> Dict[str, Any]:
        return dict(self._active_skeleton or {})

    def enqueue_active_skeleton_refresh(self, center_ids: Optional[List[str]] = None) -> None:
        """防抖排队刷新 active skeleton，减少热链路同步写快照。"""
        with self.shard_lock:
            for node_id in center_ids or []:
                if node_id:
                    self._skeleton_refresh_pending_ids.add(node_id)
            self._stats["active_skeleton_enqueue_count"] += 1
            worker = self._skeleton_refresh_worker
            if worker and worker.is_alive():
                return
            worker = threading.Thread(
                target=self._active_skeleton_worker_loop,
                name="zulong-active-skeleton-refresh",
                daemon=True,
            )
            self._skeleton_refresh_worker = worker
            worker.start()

    def _active_skeleton_worker_loop(self) -> None:
        time.sleep(0.2)
        try:
            with self.shard_lock:
                pending = list(self._skeleton_refresh_pending_ids)
                self._skeleton_refresh_pending_ids.clear()
                self._skeleton_refresh_worker = None
            if pending or time.time() - self._last_skeleton_refresh_at > 30:
                self._stats["active_skeleton_background_refresh_count"] += 1
                self.refresh_active_skeleton(pending)
        except Exception as exc:
            logger.debug(f"active skeleton 后台刷新失败: {exc}")
            with self.shard_lock:
                self._skeleton_refresh_worker = None

    def _resolve_writable_shard_id(self, shard_id: str) -> str:
        """将时间分片路由到当前可写子分片。

        自动分裂采用 parent -> parent_part_N 的索引形态。若父分片已经
        分裂，后续写入必须落到最新子分片，否则会持续写爆父分片。
        """
        shard_info = self.shard_index.get("shards", {}).get(shard_id)
        if not shard_info:
            return shard_id

        parts = shard_info.get("parts") or []
        if parts:
            return parts[-1]
        return shard_id

    def _get_split_parent_id(self, shard_id: str) -> str:
        """获取分裂父分片 ID，避免生成嵌套的 part_part 分片。"""
        shard_info = self.shard_index.get("shards", {}).get(shard_id, {})
        parent_id = shard_info.get("parent_shard")
        if parent_id:
            return parent_id
        return shard_id.split("_part_", 1)[0]
        
    def _load_shard_index(self) -> Dict[str, Any]:
        """加载分片索引"""
        index_path = os.path.join(self.base_dir, "shard_index.json")
        
        if os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"分片索引加载失败: {e}")
                
        return {
            "shards": {},
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        
    def _save_shard_index(self):
        """保存分片索引"""
        index_path = os.path.join(self.base_dir, "shard_index.json")
        
        self.shard_index["updated_at"] = time.time()
        
        try:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(self.shard_index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"分片索引保存失败: {e}")

    def _migrate_json_cross_edges_to_global_index(self) -> None:
        """把旧 shard_index.json.cross_edges 迁移到 LMDB 全局索引。"""
        cross_edges = list((self.shard_index.get("cross_edges") or {}).values())
        if not cross_edges:
            return
        try:
            migrated = self.global_index.set_many_cross_edges(cross_edges)
            self.shard_index["legacy_cross_edge_count"] = len(cross_edges)
            self.shard_index["cross_edges"] = {}
            self.shard_index["cross_edges_migrated_at"] = time.time()
            self._save_shard_index()
            logger.info(f"旧 cross_edges 已迁移到 global_index: {migrated} 条")
        except Exception as exc:
            logger.error(f"旧 cross_edges 迁移失败，运行态不会继续读取 JSON 跨边: {exc}")
            
    def get_shard(
        self,
        shard_id: str,
        load_if_missing: bool = True
    ) -> Optional[MemoryGraphHybrid]:
        """
        获取分片 - 按需加载，LRU淘汰
        
        Args:
            shard_id: 分片ID
            load_if_missing: 不存在时是否加载
            
        Returns:
            分片对象或None
        """
        with self.shard_lock:
            if shard_id in self.active_shards:
                self.active_shards.move_to_end(shard_id)
                return self.active_shards[shard_id]
                
            if not load_if_missing:
                return None
                
            shard_path = self._get_shard_path(shard_id)
            
            if not os.path.exists(shard_path):
                os.makedirs(shard_path, exist_ok=True)
                
            shard = MemoryGraphHybrid(
                data_dir=shard_path,
                shard_id=shard_id,
                map_size_gb=self.map_size_gb,
                enable_vector_index=self.enable_vector_index
            )
            
            topology_path = os.path.join(shard_path, "topology.graphml")
            topology_bin_path = os.path.join(shard_path, "topology.bin")
            if os.path.exists(topology_bin_path) or os.path.exists(topology_path):
                shard.load(topology_path)
                self._ensure_topology_consistent(shard_id, shard, reason="load")
            elif shard.properties.count_nodes() > 0:
                self._rebuild_shard_topology(shard_id, shard, reason="missing_graphml")
                 
            while len(self.active_shards) >= self.max_active_shards:
                oldest_id, oldest_shard = self.active_shards.popitem(last=False)
                oldest_shard.close()
                self._stats["shard_evict_count"] += 1
                logger.info(f"淘汰分片: {oldest_id}")
                
            self.active_shards[shard_id] = shard
            self._stats["shard_load_count"] += 1
            
            return shard

    def _ensure_topology_consistent(
        self,
        shard_id: str,
        shard: MemoryGraphHybrid,
        reason: str = "load",
    ) -> bool:
        """校验 GraphML 拓扑缓存与 LMDB 属性库数量是否明显不一致。

        LMDB properties 是权威数据；GraphML 只是兼容期拓扑缓存。若
        GraphML 非空但显著少于 LMDB，会导致 Web 快照和 BFS 看不到
        已落盘节点，因此加载分片时必须自动修复。
        """
        try:
            property_nodes = shard.properties.count_nodes()
            property_edges = shard.properties.count_edges()
            topology_nodes = len(shard.topology)
            topology_edges = getattr(shard.topology, "_edge_count", 0)
        except Exception as exc:
            logger.debug(f"分片拓扑一致性统计失败 shard={shard_id}: {exc}")
            return False

        inconsistent = False
        details = []

        node_gap = property_nodes - topology_nodes
        edge_gap = property_edges - topology_edges

        if property_nodes > 0 and topology_nodes == 0:
            inconsistent = True
            details.append(f"topology_nodes=0 property_nodes={property_nodes}")
        elif node_gap >= max(3, int(property_nodes * 0.1)):
            inconsistent = True
            details.append(f"topology_nodes={topology_nodes} property_nodes={property_nodes}")

        if property_edges > 0 and topology_edges == 0:
            inconsistent = True
            details.append(f"topology_edges=0 property_edges={property_edges}")
        elif edge_gap >= max(3, int(property_edges * 0.2)):
            inconsistent = True
            details.append(f"topology_edges={topology_edges} property_edges={property_edges}")

        self._record_topology_health(
            shard_id,
            topology_nodes=topology_nodes,
            property_nodes=property_nodes,
            topology_edges=topology_edges,
            property_edges=property_edges,
            stale=inconsistent,
            reason=reason,
            details=details,
        )

        if not inconsistent:
            return False

        logger.warning(
            f"分片 {shard_id} 拓扑缓存与 LMDB 不一致，准备重建: {', '.join(details)}"
        )
        self._rebuild_shard_topology(shard_id, shard, reason=";".join(details) or reason)
        return True

    def _rebuild_shard_topology(
        self,
        shard_id: str,
        shard: MemoryGraphHybrid,
        reason: str,
    ) -> None:
        """从 LMDB properties 重建单分片 GraphML 拓扑缓存。"""
        before_nodes = len(shard.topology)
        before_edges = getattr(shard.topology, "_edge_count", 0)
        shard.rebuild_topology_from_properties()
        after_nodes = len(shard.topology)
        after_edges = getattr(shard.topology, "_edge_count", 0)
        property_nodes = shard.properties.count_nodes()
        property_edges = shard.properties.count_edges()
        self._stats["topology_rebuild_count"] += 1
        self._record_topology_health(
            shard_id,
            topology_nodes=after_nodes,
            property_nodes=property_nodes,
            topology_edges=after_edges,
            property_edges=property_edges,
            stale=False,
            reason=reason,
            details=[
                f"rebuilt_from={before_nodes}/{before_edges}",
                f"rebuilt_to={after_nodes}/{after_edges}",
            ],
        )
        self._save_shard_index()

    def _record_topology_health(
        self,
        shard_id: str,
        topology_nodes: int,
        property_nodes: int,
        topology_edges: int,
        property_edges: int,
        stale: bool,
        reason: str,
        details: Optional[List[str]] = None,
    ) -> None:
        shard_info = self.shard_index.setdefault("shards", {}).setdefault(shard_id, {})
        shard_info["topology_health"] = {
            "checked_at": time.time(),
            "topology_nodes": topology_nodes,
            "property_nodes": property_nodes,
            "topology_edges": topology_edges,
            "property_edges": property_edges,
            "stale": stale,
            "reason": reason,
            "details": details or [],
        }
            
    def get_current_shard(self) -> MemoryGraphHybrid:
        """获取当前时间对应的分片"""
        current_shard_id = self._get_shard_id(time.time())
        return self.get_shard(current_shard_id)
        
    def add_node(
        self,
        node=None,
        node_id=None,
        node_type=None,
        label=None,
        timestamp: Optional[float] = None,
        touch: bool = True,
        **kwargs
    ) -> bool:
        """
        添加节点到指定时间分片
        
        兼容两种调用方式：
        1. MemoryGraph 风格: add_node(gnode, touch=True)  # GraphNode 对象
        2. Sharded 风格: add_node(node_id=..., node_type=..., label=..., ...)
        
        Args:
            node: GraphNode 对象（兼容旧接口）
            node_id: 节点ID
            node_type: 节点类型
            label: 标签
            timestamp: 时间戳（默认当前时间）
            touch: 是否刷新 last_accessed（仅在 GraphNode 风格下使用）
            **kwargs: 其他参数（content, importance, backend_ref, metadata等）
        """
        # 🔥 兼容 MemoryGraph 风格：第一个参数是 GraphNode 对象
        if node is not None and hasattr(node, 'node_id'):
            gnode = node
            node_id = gnode.node_id
            node_type_str = getattr(gnode.node_type, 'value', gnode.node_type) if hasattr(gnode, 'node_type') else "unknown"
            if isinstance(node_type_str, str):
                node_type = node_type_str
            else:
                node_type = str(node_type_str)
            label = getattr(gnode, 'label', "")
            # 提取 metadata 中的属性
            meta = getattr(gnode, 'metadata', {}) or {}
            timestamp = getattr(gnode, 'created_at', timestamp)
            if 'timestamp' in meta and timestamp is None:
                timestamp = meta.get('timestamp')
            if 'content' in meta:
                kwargs.setdefault('content', meta.get('content'))
            if 'importance' in meta:
                kwargs.setdefault('importance', meta.get('importance'))
            if 'backend_ref' in meta:
                kwargs.setdefault('backend_ref', meta.get('backend_ref'))
            if hasattr(gnode, 'backend_ref') and 'backend_ref' not in kwargs:
                kwargs['backend_ref'] = getattr(gnode, 'backend_ref', "")
            # 剩余属性合并到 metadata
            reserved = {'timestamp', 'content', 'importance', 'backend_ref'}
            remaining = {k: v for k, v in meta.items() if k not in reserved}
            if remaining:
                existing_meta = kwargs.get('metadata', {}) or {}
                existing_meta.update(remaining)
                kwargs['metadata'] = existing_meta
            # content 属性可能在 GraphNode 上直接
            if hasattr(gnode, 'content') and 'content' not in kwargs:
                kwargs['content'] = getattr(gnode, 'content', None)
        
        if node_id is None:
            logger.error("[ShardedMemoryGraph] add_node 缺少 node_id")
            return False
        if node_type is None:
            node_type = "unknown"
        if label is None:
            label = ""
            
        if timestamp is None:
            timestamp = time.time()

        with self.shard_lock:
            shard_id = self._resolve_writable_shard_id(self._get_shard_id(timestamp))
            shard = self.get_shard(shard_id)

            # 提取 properties 并映射到 MemoryGraphHybrid.add_node 接受的参数
            properties = kwargs.pop("properties", {}) or {}
            if "content" not in kwargs and "content" in properties:
                kwargs["content"] = properties.pop("content")
            if "importance" not in kwargs and "importance" in properties:
                kwargs["importance"] = properties.pop("importance")
            if "backend_ref" not in kwargs and "backend_ref" in properties:
                kwargs["backend_ref"] = properties.pop("backend_ref")
            # 剩余的 properties 合并到 metadata
            if properties:
                existing_meta = kwargs.get("metadata", {}) or {}
                existing_meta.update(properties)
                kwargs["metadata"] = existing_meta

            success = shard.add_node(
                node_id=node_id,
                node_type=node_type,
                label=label,
                **kwargs
            )

            if success:
                self._last_topology_write_at = time.time()
                if shard_id not in self.shard_index["shards"]:
                    self.shard_index["shards"][shard_id] = {
                        "created_at": timestamp,
                        "node_count": 0,
                        "edge_count": 0,
                    }
                self.shard_index["shards"][shard_id]["node_count"] += 1

                # 分片大小控制: 检查是否超过节点数或物理大小阈值
                node_count = self.shard_index["shards"][shard_id]["node_count"]
                self._check_shard_size(shard_id, shard, node_count)
                self._check_topology_delta_compaction(shard_id, shard)

                self._index_node_location(
                    node_id,
                    shard_id,
                    node_type,
                    kwargs.get("metadata", {}) or {},
                    label=label,
                )

                self._save_shard_index()

            return success

    def _check_shard_size(
        self,
        shard_id: str,
        shard: Optional[MemoryGraphHybrid],
        node_count: int
    ) -> None:
        """检查分片大小，超阈值时发出警告或触发自动分裂。

        大规模方案下不能只看节点数：某些节点 content/metadata 很大，
        即使没到 15 万节点也可能让 LMDB payload 超过 TSD 推荐的
        50-200MB 单分片目标。因此这里同时记录物理使用量。
        """
        usage = self._refresh_shard_usage_if_needed(shard_id, shard, node_count)

        split_reasons = []
        warning_reasons = []

        if self.max_nodes_per_shard > 0:
            ratio = node_count / self.max_nodes_per_shard
            if ratio >= 1.10:
                split_reasons.append(
                    f"{node_count} 节点 ({ratio:.1%}, limit={self.max_nodes_per_shard})"
                )
            elif ratio >= 0.95:
                remaining = self.max_nodes_per_shard - node_count
                warning_reasons.append(
                    f"{node_count} 节点 ({ratio:.1%}, remaining={remaining})"
                )

        property_mb = usage.get("property_used_mb", 0.0)
        if self.max_shard_property_mb_split > 0 and property_mb >= self.max_shard_property_mb_split:
            split_reasons.append(
                f"properties={property_mb:.1f}MB (limit={self.max_shard_property_mb_split}MB)"
            )
        elif self.max_shard_property_mb_warning > 0 and property_mb >= self.max_shard_property_mb_warning:
            warning_reasons.append(
                f"properties={property_mb:.1f}MB (warning={self.max_shard_property_mb_warning}MB)"
            )

        topology_mb = usage.get("topology_graphml_mb", 0.0)
        if self.max_shard_topology_mb_warning > 0 and topology_mb >= self.max_shard_topology_mb_warning:
            warning_reasons.append(
                f"topology.graphml={topology_mb:.1f}MB "
                f"(warning={self.max_shard_topology_mb_warning}MB)"
            )

        delta_mb = usage.get("topology_delta_mb", 0.0)
        if (
            self.max_shard_topology_delta_mb_compact > 0
            and delta_mb >= self.max_shard_topology_delta_mb_compact
        ):
            warning_reasons.append(
                f"topology_delta={delta_mb:.3f}MB "
                f"(compact={self.max_shard_topology_delta_mb_compact}MB)"
            )
            self._schedule_topology_compaction(shard_id)

        if split_reasons:
            logger.critical(
                f"分片 {shard_id} 严重超限: {', '.join(split_reasons)}，触发自动分裂"
            )
            self._auto_split_shard(shard_id)
        elif warning_reasons:
            self._stats["shard_size_warning_count"] += 1
            logger.warning(
                f"分片 {shard_id} 接近上限: {', '.join(warning_reasons)}"
            )

    def _schedule_topology_compaction(self, shard_id: str) -> None:
        """将 topology_delta.log 压实任务放入去重队列。"""
        if not shard_id:
            return
        if shard_id in self._compacting_shards or shard_id in self._queued_topology_compactions:
            return
        if shard_id not in self.active_shards:
            return

        self._queued_topology_compactions.add(shard_id)
        self._topology_compact_queue.append(shard_id)
        self._stats["topology_compact_enqueue_count"] += 1
        self._ensure_topology_compact_worker()

    def _ensure_topology_compact_worker(self) -> None:
        """确保只有一个 topology compactor worker 在后台消费队列。"""
        worker = self._topology_compact_worker
        if worker and worker.is_alive():
            return
        worker = threading.Thread(
            target=self._topology_compact_worker_loop,
            name="zulong-topology-compact-worker",
            daemon=True,
        )
        self._topology_compact_worker = worker
        worker.start()

    def _topology_compact_worker_loop(self) -> None:
        """串行压实 topology delta，避免高频写入反复起线程。"""
        while True:
            idle_delay = 0.25 - (time.time() - self._last_topology_write_at)
            if idle_delay > 0:
                time.sleep(min(idle_delay, 0.25))
            with self.shard_lock:
                if not self._topology_compact_queue:
                    self._topology_compact_worker = None
                    return
                shard_id = self._topology_compact_queue.popleft()
                self._queued_topology_compactions.discard(shard_id)
                if shard_id in self._compacting_shards:
                    continue
                self._compacting_shards.add(shard_id)

            retry_needed = False
            compacted = False
            try:
                with self.shard_lock:
                    active = self.active_shards.get(shard_id)
                    if active is None:
                        continue
                    active.save()
                    node_count = self.shard_index.get("shards", {}).get(shard_id, {}).get("node_count", 0)
                    usage = self._collect_shard_usage(shard_id, active)
                    shard_info = self.shard_index["shards"].setdefault(shard_id, {})
                    shard_info["usage"] = usage
                    shard_info["last_compacted_at"] = time.time()
                    shard_info.pop("last_compact_error", None)
                    shard_info.pop("last_compact_failed_at", None)
                    self._last_size_check_node_counts[shard_id] = node_count
                    self._topology_compact_failures.pop(shard_id, None)
                    self._topology_compact_retry_after.pop(shard_id, None)
                    self._stats["topology_compact_count"] += 1
                    self._save_shard_index()
                    compacted = True
                    logger.info(f"分片 {shard_id} topology_delta 已后台压实")
            except Exception as exc:
                retry_count = self._topology_compact_failures.get(shard_id, 0) + 1
                self._topology_compact_failures[shard_id] = retry_count
                self._stats["topology_compact_failure_count"] += 1
                with self.shard_lock:
                    shard_info = self.shard_index["shards"].setdefault(shard_id, {})
                    shard_info["last_compact_error"] = str(exc)
                    shard_info["last_compact_failed_at"] = time.time()
                    self._save_shard_index()
                logger.warning(
                    f"分片 {shard_id} topology_delta 后台压实失败 "
                    f"(attempt={retry_count}): {exc}"
                )
                retry_needed = retry_count < 3
                if retry_needed:
                    self._topology_compact_retry_after[shard_id] = time.time() + min(0.5 * retry_count, 2.0)
            finally:
                with self.shard_lock:
                    self._compacting_shards.discard(shard_id)
                    active = self.active_shards.get(shard_id)
                    if active is not None and compacted:
                        self._check_topology_delta_compaction(shard_id, active)
                    elif retry_needed and shard_id not in self._queued_topology_compactions:
                        self._stats["topology_compact_retry_count"] += 1
                        self._queued_topology_compactions.add(shard_id)
                        self._topology_compact_queue.append(shard_id)
            retry_after = self._topology_compact_retry_after.get(shard_id, 0.0)
            if retry_after > 0:
                delay = retry_after - time.time()
                if delay > 0:
                    time.sleep(delay)

    def _check_topology_delta_compaction(
        self,
        shard_id: str,
        shard: Optional[MemoryGraphHybrid],
    ) -> None:
        """独立检查 delta 大小，避免被节点数间隔缓存跳过。"""
        if self.max_shard_topology_delta_mb_compact <= 0 or shard is None:
            return
        delta_path = os.path.join(self._get_shard_path(shard_id), "topology_delta.log")
        delta_bytes = self._safe_getsize(delta_path)
        threshold_bytes = int(self.max_shard_topology_delta_mb_compact * 1024**2)
        if threshold_bytes <= 0:
            threshold_bytes = 1
        if delta_bytes >= threshold_bytes:
            self._schedule_topology_compaction(shard_id)

    def _refresh_shard_usage_if_needed(
        self,
        shard_id: str,
        shard: Optional[MemoryGraphHybrid],
        node_count: int
    ) -> Dict[str, Any]:
        """刷新分片物理使用量统计，避免每个节点都触发文件系统/LMDB查询。"""
        shard_info = self.shard_index["shards"].setdefault(shard_id, {})
        last_count = self._last_size_check_node_counts.get(shard_id)

        if last_count is not None and node_count - last_count < self.shard_size_check_interval_nodes:
            return shard_info.get("usage", {})

        usage = self._collect_shard_usage(shard_id, shard)
        shard_info["usage"] = usage
        shard_info["last_size_check_at"] = time.time()
        self._last_size_check_node_counts[shard_id] = node_count
        return usage

    def _collect_shard_usage(
        self,
        shard_id: str,
        shard: Optional[MemoryGraphHybrid]
    ) -> Dict[str, Any]:
        """收集分片 LMDB 与拓扑缓存的轻量使用量。"""
        shard_path = self._get_shard_path(shard_id)
        topology_path = os.path.join(shard_path, "topology.graphml")
        topology_bin_path = os.path.join(shard_path, "topology.bin")
        topology_delta_path = os.path.join(shard_path, "topology_delta.log")
        property_path = os.path.join(shard_path, "properties")
        data_mdb_path = os.path.join(property_path, "data.mdb")
        local_index_bytes = self._dir_size(self._get_local_index_path(shard_id))

        property_file_bytes = self._safe_getsize(data_mdb_path)
        topology_graphml_bytes = self._safe_getsize(topology_path)
        topology_bin_bytes = self._safe_getsize(topology_bin_path)
        topology_delta_bytes = self._safe_getsize(topology_delta_path)
        property_used_bytes = property_file_bytes
        property_map_size_bytes = self.map_size_gb * 1024**3

        if shard is not None:
            try:
                prop_stats = shard.properties.get_stats()
                env_info = prop_stats.get("env_info", {}) or {}
                map_size = env_info.get("map_size") or property_map_size_bytes
                last_pgno = env_info.get("last_pgno")
                page_size = env_info.get("psize", 4096)
                if last_pgno is not None:
                    property_used_bytes = int((last_pgno + 1) * page_size)
                property_map_size_bytes = int(map_size)
            except Exception as exc:
                logger.debug(f"读取分片 {shard_id} LMDB 使用量失败: {exc}")

        usage = {
            "property_used_bytes": property_used_bytes,
            "property_used_mb": round(property_used_bytes / 1024**2, 3),
            "property_file_bytes": property_file_bytes,
            "property_file_mb": round(property_file_bytes / 1024**2, 3),
            "property_map_size_bytes": property_map_size_bytes,
            "property_map_size_gb": round(property_map_size_bytes / 1024**3, 3),
            "local_index_bytes": local_index_bytes,
            "local_index_mb": round(local_index_bytes / 1024**2, 3),
            "topology_graphml_bytes": topology_graphml_bytes,
            "topology_graphml_mb": round(topology_graphml_bytes / 1024**2, 3),
            "topology_bin_bytes": topology_bin_bytes,
            "topology_bin_mb": round(topology_bin_bytes / 1024**2, 3),
            "topology_delta_bytes": topology_delta_bytes,
            "topology_delta_mb": round(topology_delta_bytes / 1024**2, 3),
        }

        if property_map_size_bytes > 0:
            usage["property_map_used_ratio"] = round(
                property_used_bytes / property_map_size_bytes,
                6
            )

        return usage

    def rebuild_global_index(self, shard_ids: Optional[List[str]] = None) -> int:
        """从分片 properties 重建/补齐 global_index.lmdb。"""
        target_shards = shard_ids or self.list_all_shards()
        total = 0
        for shard_id in target_shards:
            shard = self.get_shard(shard_id, load_if_missing=True)
            if not shard:
                continue
            items = []
            try:
                for node in shard.properties.iter_nodes():
                    items.append({
                        "node_id": node.node_id,
                        "shard_id": shard_id,
                        "node_type": node.node_type,
                        "metadata": node.metadata or {},
                    })
                    if len(items) >= 500:
                        total += self.global_index.set_many_node_locations(items)
                        items = []
                if items:
                    total += self.global_index.set_many_node_locations(items)
            except Exception as exc:
                logger.warning(f"重建全局索引失败 shard={shard_id}: {exc}")

        if total:
            self.global_index.mark_rebuilt(source="sharded_memory_graph")
            self._stats["global_index_repair_count"] += total
            logger.info(f"全局索引已重建/补齐: {total} 个节点")
        return total

    def rebuild_local_index(self, shard_ids: Optional[List[str]] = None) -> int:
        """从分片 properties 重建/补齐 local_index.lmdb。"""
        target_shards = shard_ids or self.list_all_shards()
        total = 0
        for shard_id in target_shards:
            shard = self.get_shard(shard_id, load_if_missing=True)
            local_index = self.get_local_index(shard_id, create=True)
            if not shard or not local_index:
                continue
            items = []
            try:
                local_index.clear()
                for node in shard.properties.iter_nodes():
                    items.append({
                        "node_id": node.node_id,
                        "node_type": node.node_type,
                        "label": node.label,
                        "metadata": node.metadata or {},
                    })
                    if len(items) >= 500:
                        total += local_index.set_many_node_headers(items)
                        items = []
                if items:
                    total += local_index.set_many_node_headers(items)
                local_index.mark_rebuilt(source="sharded_memory_graph")
                shard_info = self.shard_index["shards"].setdefault(shard_id, {})
                shard_info["local_index"] = {
                    "rebuilt_at": time.time(),
                    "node_count": local_index.count_nodes(),
                    "complete": True,
                }
            except Exception as exc:
                logger.warning(f"重建 local_index 失败 shard={shard_id}: {exc}")

        if total:
            self._stats["local_index_rebuild_count"] += total
            self._save_shard_index()
            logger.info(f"local_index 已重建/补齐: {total} 个节点")
        return total

    @staticmethod
    def _safe_getsize(path: str) -> int:
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    @staticmethod
    def _dir_size(path: str) -> int:
        if not os.path.exists(path):
            return 0
        if os.path.isfile(path):
            return ShardedMemoryGraph._safe_getsize(path)
        total = 0
        for root, _, files in os.walk(path):
            for filename in files:
                total += ShardedMemoryGraph._safe_getsize(os.path.join(root, filename))
        return total

    def _auto_split_shard(self, shard_id: str) -> None:
        """自动分裂分片：创建子分片并将后续节点路由到新分片
        
        分裂策略: 在原分片 ID 后追加 _part_N 后缀。
        例如 "2026_05" → "2026_05_part_1"
        """
        parent_shard_id = self._get_split_parent_id(shard_id)
        parent_info = self.shard_index["shards"].setdefault(parent_shard_id, {
            "created_at": time.time(),
            "node_count": 0,
            "edge_count": 0,
        })

        existing_parts = [
            key for key in self.shard_index["shards"]
            if key.startswith(f"{parent_shard_id}_part_") and key != parent_shard_id
        ]
        part_num = len(existing_parts) + 1
        new_shard_id = f"{parent_shard_id}_part_{part_num}"

        if shard_id != parent_shard_id:
            shard_info = self.shard_index["shards"].setdefault(shard_id, {})
            shard_info["sealed_at"] = time.time()
            shard_info["sealed_reason"] = "split_threshold_exceeded"
        
        # 在 shard_index 中注册新子分片
        self.shard_index["shards"][new_shard_id] = {
            "created_at": time.time(),
            "node_count": 0,
            "edge_count": 0,
            "parent_shard": parent_shard_id,
        }
        
        # 更新父分片索引，记录已分裂的子分片
        if "parts" not in parent_info:
            parent_info["parts"] = []
        parent_info["parts"].append(new_shard_id)
        parent_info["active_part"] = new_shard_id
        
        self._stats["auto_split_count"] += 1
        self._save_shard_index()
        
        logger.info(
            f"分片 {shard_id} 已自动分裂 → {new_shard_id} "
            f"(子分片 #{part_num})"
        )
        
    def add_edge(
        self,
        src_id: str,
        dst_id: str,
        edge_type: str,
        timestamp: Optional[float] = None,
        **kwargs
    ) -> bool:
        """
        添加边到指定时间分片
        
        Args:
            src_id: 源节点ID
            dst_id: 目标节点ID
            edge_type: 边类型
            timestamp: 时间戳
            **kwargs: 其他参数
        """
        if timestamp is None:
            timestamp = time.time()
        edge_type = getattr(edge_type, "value", edge_type) or "association"

        with self.shard_lock:
            shard_id = self._resolve_writable_shard_id(self._get_shard_id(timestamp))
            shard = self.get_shard(shard_id)

            if shard and (src_id not in shard.topology or dst_id not in shard.topology):
                src_shard_id = self._find_node_shard_id(src_id)
                dst_shard_id = self._find_node_shard_id(dst_id)
                if src_shard_id and src_shard_id == dst_shard_id:
                    shard_id = src_shard_id
                    shard = self.get_shard(shard_id)
                elif src_shard_id and dst_shard_id:
                    return self._add_cross_shard_edge(
                        src_id=src_id,
                        dst_id=dst_id,
                        edge_type=edge_type,
                        src_shard_id=src_shard_id,
                        dst_shard_id=dst_shard_id,
                        **kwargs
                    )
                elif src_shard_id or dst_shard_id:
                    logger.debug(
                        f"跨分片边暂按时间分片路由: {src_id}@{src_shard_id} → "
                        f"{dst_id}@{dst_shard_id}, write_shard={shard_id}"
                    )

            success = shard.add_edge(
                src_id=src_id,
                dst_id=dst_id,
                edge_type=edge_type,
                **kwargs
            )

            if success:
                self._last_topology_write_at = time.time()
                if shard_id in self.shard_index["shards"]:
                    self.shard_index["shards"][shard_id]["edge_count"] += 1
                    node_count = self.shard_index["shards"][shard_id].get("node_count", 0)
                    self._check_shard_size(shard_id, shard, node_count)
                    self._check_topology_delta_compaction(shard_id, shard)
                    self._save_shard_index()

            return success

    def _index_node_location(
        self,
        node_id: str,
        shard_id: str,
        node_type: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        label: str = "",
    ) -> None:
        try:
            self.global_index.set_node_location(
                node_id=node_id,
                shard_id=shard_id,
                node_type=str(node_type or ""),
                metadata=metadata or {},
            )
        except Exception as exc:
            logger.warning(f"全局索引写入失败 node={node_id}, shard={shard_id}: {exc}")
        self._index_local_node_header(
            shard_id=shard_id,
            node_id=node_id,
            node_type=str(node_type or ""),
            label=label,
            metadata=metadata or {},
        )

    def _add_cross_shard_edge(
        self,
        src_id: str,
        dst_id: str,
        edge_type: str,
        src_shard_id: str,
        dst_shard_id: str,
        **kwargs
    ) -> bool:
        """记录跨分片边到 global_index LMDB。"""
        weight = float(kwargs.get("weight", 1.0))
        protected = bool(kwargs.get("protected", False))
        metadata = kwargs.get("metadata", {}) or {}
        now = time.time()

        edge_payload = {
            "src_id": src_id,
            "dst_id": dst_id,
            "edge_type": edge_type,
            "weight": weight,
            "created_at": now,
            "last_activated": now,
            "activation_count": 1,
            "protected": protected,
            "metadata": metadata,
            "src_shard_id": src_shard_id,
            "dst_shard_id": dst_shard_id,
        }

        try:
            self.global_index.set_cross_edge(edge_payload)
        except Exception as exc:
            logger.error(f"跨分片边写入 global_index 失败: {exc}")
            return False
        logger.debug(
            f"记录跨分片边: {src_id}@{src_shard_id} → "
            f"{dst_id}@{dst_shard_id} ({edge_type})"
        )
        return True
        
    def has_edge(self, src_id: str, dst_id: str) -> bool:
        """检查边是否存在（兼容 MemoryGraph 接口）
        
        Args:
            src_id: 源节点ID
            dst_id: 目标节点ID
            
        Returns:
            bool: 边是否存在
        """
        for shard_id in self.list_all_shards():
            shard = self.get_shard(shard_id, load_if_missing=True)
            if shard and shard.get_topology_edge_info(src_id, dst_id) is not None:
                return True
        if self.global_index.has_cross_edge(src_id, dst_id):
            return True
        return False
        
    def discover_across_shards(
        self,
        seed_ids: List[str],
        seed_shard_id: str,
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> List[Tuple[str, int, str]]:
        """
        跨分片关联发现
        
        Args:
            seed_ids: 种子节点ID列表
            seed_shard_id: 种子所在分片ID
            max_depth: 最大扩散深度
            max_nodes: 最大返回节点数
            
        Returns:
            [(node_id, distance, shard_id), ...]
        """
        visited: Dict[str, Tuple[int, str]] = {}
        queue: List[Tuple[str, int, str]] = [
            (sid, 0, seed_shard_id) for sid in seed_ids
        ]
        
        for sid in seed_ids:
            visited[sid] = (0, seed_shard_id)
            
        while queue and len(visited) < max_nodes:
            current_id, current_dist, current_shard_id = queue.pop(0)
            
            if current_dist >= max_depth:
                continue
                
            shard = self.get_shard(current_shard_id, load_if_missing=True)
            if shard is None:
                continue
                
            neighbors = shard.get_topology_neighbors(current_id, mode="out")
            for edge in self._get_cross_edges_from(current_id):
                dst_id = edge.get("dst_id")
                if not dst_id or dst_id in visited:
                    continue
                dst_shard_id = edge.get("dst_shard_id") or current_shard_id
                visited[dst_id] = (current_dist + 1, dst_shard_id)
                queue.append((dst_id, current_dist + 1, dst_shard_id))
            
            for neighbor_id in neighbors:
                if neighbor_id in visited:
                    continue

                visited[neighbor_id] = (current_dist + 1, current_shard_id)
                queue.append((neighbor_id, current_dist + 1, current_shard_id))
                
        result = [
            (node_id, dist, shard_id)
            for node_id, (dist, shard_id) in visited.items()
        ]
        result.sort(key=lambda x: x[1])
        
        self._stats["cross_shard_query_count"] += 1
        return result[:max_nodes]
        
    def list_all_shards(self) -> List[str]:
        """列出所有分片ID"""
        return list(self.shard_index["shards"].keys())
        
    def get_shard_stats(self, shard_id: str) -> Optional[Dict]:
        """获取分片统计信息"""
        return self.shard_index["shards"].get(shard_id)

    def get_session_node_id_for_conversation(self, conversation_id: str) -> Optional[str]:
        """通过 conversation_id 定位 Dialogue session 根节点。"""
        if not conversation_id:
            return None
        try:
            node_id = self.global_index.get_conversation_node(conversation_id)
            if node_id and self.has_node(node_id):
                return node_id
        except Exception as exc:
            logger.debug(f"conversation 全局索引查询失败 {conversation_id}: {exc}")
        return self._infer_session_node_id(conversation_id)

    def get_task_node_id_for_graph(self, task_graph_id: str) -> Optional[str]:
        """通过 task_graph_id 定位 MemoryGraph 中的任务节点。"""
        if not task_graph_id:
            return None
        try:
            node_id = self.global_index.get_task_graph_node(task_graph_id)
            if node_id and self.has_node(node_id):
                return node_id
        except Exception as exc:
            logger.debug(f"task_graph 全局索引查询失败 {task_graph_id}: {exc}")

        candidates = [f"task:{task_graph_id}"]
        for candidate in candidates:
            if self.has_node(candidate):
                return candidate

        for node in self.get_nodes_by_type("task"):
            metadata = node.metadata or {}
            if (
                metadata.get("task_graph_id") == task_graph_id
                or metadata.get("graph_id") == task_graph_id
            ):
                self._index_node_location(node.node_id, node.storage_shard, node.node_type, metadata)
                return node.node_id
        return None

    def get_context_seed_for_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """返回 Web/IDE 恢复上下文使用的 session seed。"""
        session_node_id = self.get_session_node_id_for_conversation(conversation_id)
        if not session_node_id:
            return None
        node, shard_id = self._get_node_with_shard(session_node_id)
        if not node:
            return None
        return {
            "conversation_id": conversation_id,
            "session_node_id": session_node_id,
            "session_label": node.label or "",
            "shard_id": shard_id,
            "metadata": node.metadata or {},
        }

    @staticmethod
    def _infer_session_node_id(conversation_id: str) -> str:
        safe_id = "".join(
            ch if ch.isalnum() or ch in "-_" else "_"
            for ch in str(conversation_id or "")
        )
        return f"dialogue:session_{safe_id}"

    def _get_indexed_shard(self, node_id: str) -> Tuple[Optional[MemoryGraphHybrid], Optional[str]]:
        shard, shard_id, _ = self._get_indexed_shard_status(node_id)
        return shard, shard_id

    def _get_indexed_shard_status(self, node_id: str) -> Tuple[Optional[MemoryGraphHybrid], Optional[str], bool]:
        """返回 (shard, shard_id, definitive_miss)。"""
        shard_id = None
        try:
            shard_id = self.global_index.get_node_shard(node_id)
        except Exception as exc:
            logger.debug(f"全局索引读取失败 node={node_id}: {exc}")

        if not shard_id:
            self._stats["global_index_miss_count"] += 1
            return None, None, False

        local_hit = self._local_index_has_node(shard_id, node_id)
        if local_hit is False:
            self._stats["global_index_miss_count"] += 1
            return None, shard_id, True

        shard = self.get_shard(shard_id, load_if_missing=True)
        if shard and node_id in shard.topology:
            self._stats["global_index_hit_count"] += 1
            return shard, shard_id, False

        self._stats["global_index_miss_count"] += 1
        return None, shard_id, False
        
    def get_node(self, node_id: str) -> Optional[NodeProperties]:
        """
        获取节点属性（兼容 MemoryGraph 接口）
        
        Returns:
            NodeProperties 或 None
        """
        shard, _, definitive_miss = self._get_indexed_shard_status(node_id)
        if shard:
            return shard.get_node(node_id)
        if definitive_miss:
            return None

        for shard_id in self.list_all_shards():
            shard = self.get_shard(shard_id, load_if_missing=True)
            if shard and node_id in shard.topology:
                node = shard.get_node(node_id)
                if node:
                    self._index_node_location(node.node_id, shard_id, node.node_type, node.metadata or {})
                return node
                
        return None

    def resolve_address(self, address: str) -> Optional[NodeProperties]:
        """
        解析图地址字符串，定位对应的记忆节点（兼容 MemoryGraph 接口）。

        支持格式:
        - "tg:{graph_id}/task:{node_id}" → TaskGraph 地址
        - "task:o1_1"                    → 直接 node_id
        - "dialogue:round_42"            → 对话节点
        - 任意 node_id 字符串            → 直接查找

        性能设计 (TSD v2.4 <50ms):
        - 格式1-2: O(1) 通过 global_index (LMDB mmap)，<1ms
        - 兜底: O(n) 通过 search_nodes 遍历所有分片

        Returns:
            NodeProperties 或 None（未找到）
        """
        if not address:
            return None

        # ── 格式1: tg:{graph_id}/task:{node_id} ──
        if address.startswith("tg:"):
            parts = address.split("/")
            graph_id = parts[0][3:]  # 去掉 "tg:" 前缀
            for part in parts[1:]:
                if part.startswith("task:"):
                    local_node_id = part[5:]
                    # 尝试 task:{node_id} 格式（TaskGraphAdapter 写入的 node_id）
                    node = self.get_node(f"task:{local_node_id}")
                    if node:
                        return node
                    # 尝试 task:{graph_id}/{node_id} 格式（带 session 前缀的完整路径）
                    node = self.get_node(f"task:{graph_id}/{local_node_id}")
                    if node:
                        return node
                    # 尝试原始 part 作为 node_id
                    node = self.get_node(part)
                    if node:
                        return node

        # ── 格式2: 直接 node_id 查找 ──
        node = self.get_node(address)
        if node:
            return node

        # ── 格式3: tg: 地址通过 global_index 的 task_graph_node 映射 ──
        if address.startswith("tg:"):
            parts = address.split("/")
            graph_id = parts[0][3:]
            try:
                indexed_nid = self.global_index.get_task_graph_node(graph_id)
                if indexed_nid:
                    node = self.get_node(indexed_nid)
                    if node:
                        return node
            except Exception as exc:
                logger.debug(
                    f"[resolve_address] global_index 查询失败 "
                    f"tg={graph_id}: {exc}"
                )

        # ── 格式4 (兜底): 通过 metadata graph_address/task_graph_address 遍历 ──
        search_results = self.search_nodes(address, max_results=10)
        for result in search_results:
            meta = result.get("metadata", {}) or {}
            if meta.get("graph_address") == address:
                nid = result.get("node_id")
                if nid:
                    return self.get_node(nid)
            if meta.get("task_graph_address") == address:
                nid = result.get("node_id")
                if nid:
                    return self.get_node(nid)

        return None

    def update_node(self, node: NodeProperties) -> bool:
        """更新节点属性（兼容 MemoryGraph 接口）。"""
        shard, shard_id, definitive_miss = self._get_indexed_shard_status(node.node_id)
        if shard and shard_id:
            updated = bool(shard.update_node(node))
            if updated:
                self._index_node_location(node.node_id, shard_id, node.node_type, node.metadata or {})
            return updated
        if definitive_miss:
            return False

        for shard_id in self.list_all_shards():
            shard = self.get_shard(shard_id, load_if_missing=True)
            if shard and node.node_id in shard.topology:
                updated = bool(shard.update_node(node))
                if updated:
                    self._index_node_location(node.node_id, shard_id, node.node_type, node.metadata or {})
                return updated
        return False

    def _get_node_with_shard(self, node_id: str) -> Tuple[Optional[NodeProperties], Optional[str]]:
        """
        获取节点属性及其所在分片ID（内部使用）
        
        Returns:
            (NodeProperties, shard_id) 或 (None, None)
        """
        shard, shard_id, definitive_miss = self._get_indexed_shard_status(node_id)
        if shard and shard_id:
            node = shard.get_node(node_id)
            if node:
                return node, shard_id
        if definitive_miss:
            return None, None

        for shard_id in self.list_all_shards():
            shard = self.get_shard(shard_id, load_if_missing=True)
            if shard and node_id in shard.topology:
                node = shard.get_node(node_id)
                if node:
                    self._index_node_location(node.node_id, shard_id, node.node_type, node.metadata or {})
                return node, shard_id
                
        return None, None

    def _find_node_shard_id(self, node_id: str) -> Optional[str]:
        """定位节点所在分片，优先走 global_index.lmdb。"""
        shard, shard_id, definitive_miss = self._get_indexed_shard_status(node_id)
        if shard and shard_id:
            return shard_id
        if definitive_miss:
            return None

        for shard_id in self.list_all_shards():
            shard = self.get_shard(shard_id, load_if_missing=True)
            if shard and node_id in shard.topology:
                node = shard.get_node(node_id)
                if node:
                    self._index_node_location(node.node_id, shard_id, node.node_type, node.metadata or {})
                return shard_id
        return None

    def _get_cross_edges_from(self, node_id: str, edge_type: Optional[str] = None) -> List[Dict[str, Any]]:
        edge_type_value = getattr(edge_type, "value", edge_type)
        try:
            return self.global_index.get_cross_edges_from(node_id, edge_type=edge_type_value)
        except Exception as exc:
            logger.debug(f"读取 LMDB 跨分片出边失败 node={node_id}: {exc}")
            return []

    def _get_cross_edges_to(self, node_id: str, edge_type: Optional[str] = None) -> List[Dict[str, Any]]:
        edge_type_value = getattr(edge_type, "value", edge_type)
        try:
            return self.global_index.get_cross_edges_to(node_id, edge_type=edge_type_value)
        except Exception as exc:
            logger.debug(f"读取 LMDB 跨分片入边失败 node={node_id}: {exc}")
            return []
        
    def has_node(self, node_id: str) -> bool:
        """检查节点是否存在（兼容 MemoryGraph 接口）"""
        shard, _ = self._get_indexed_shard(node_id)
        if shard:
            return True

        for shard_id in self.list_all_shards():
            shard = self.get_shard(shard_id, load_if_missing=True)
            if shard and node_id in shard.topology:
                node = shard.get_node(node_id)
                if node:
                    self._index_node_location(node.node_id, shard_id, node.node_type, node.metadata or {})
                return True
        return False

    def get_nodes_by_type(self, node_type: str) -> list:
        """按类型查询节点（兼容 MemoryGraph 接口）
        
        Args:
            node_type: 节点类型字符串
            
        Returns:
            NodeProperties 列表
        """
        node_type_value = getattr(node_type, "value", node_type)
        node_type_value = str(node_type_value)
        results = []
        for shard_id in self.list_all_shards():
            shard = self.get_shard(shard_id, load_if_missing=True)
            if shard:
                for vertex in shard.topology.graph.vs:
                    vt = vertex["type"] if "type" in vertex.attributes() else "unknown"
                    if vt == node_type_value:
                        nid = self._vertex_node_id(vertex)
                        node = shard.get_node(nid)
                        if node:
                            results.append(node)
        return results
        
    async def start_prune_loop(self, interval_seconds: int = 1800):
        """异步修剪循环（分片存储按时间维度的独立管理，不执行全局修剪）
        
        分片存储的驱逐和清理由 LRU 淘汰机制和时间分片自动管理，
        无需传统的修剪循环。此方法仅提供接口兼容。
        """
        logger.info(f"ShardedMemoryGraph 使用 LRU 淘汰 + 时间分片管理，无需修剪循环")

    def get_stats(self) -> Dict[str, Any]:
        """获取总统计信息（兼容 MemoryGraph 接口的别名）"""
        return self.get_total_stats()

    @property
    def stats(self) -> Dict[str, Any]:
        """兼容 NetworkX MemoryGraph.stats 属性。"""
        return self.get_total_stats()

    def get_total_stats(self) -> Dict[str, Any]:
        """获取总统计信息"""
        total_nodes = 0
        total_edges = 0
        
        for shard_info in self.shard_index["shards"].values():
            total_nodes += shard_info.get("node_count", 0)
            total_edges += shard_info.get("edge_count", 0)
        try:
            cross_edge_count = self.global_index.count_cross_edges()
        except Exception:
            cross_edge_count = 0
        total_edges += cross_edge_count
        global_index_stats = {}
        try:
            global_index_stats = self.global_index.get_stats()
        except Exception as exc:
            global_index_stats = {"error": str(exc)}
            
        return {
            "total_shards": len(self.shard_index["shards"]),
            "active_shards": len(self.active_shards),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "cross_edge_count": cross_edge_count,
            "active_skeleton": (self._active_skeleton or {}).get("stats", {}),
            "global_index": global_index_stats,
            "operations": self._stats,
            "shard_strategy": self.shard_strategy,
        }
        
    async def retrieve_context(
        self,
        query_text: str,
        top_k: int = 10,
        hot_window_minutes: int = 30,
        session_id: str = "",
    ) -> List[Dict[str, Any]]:
        """检索上下文 — 检索热路径 + 摘要导航并行，热命中后做 historical BFS。

        这里的“热路径”按 hot_window_minutes 时间窗口扫描近期节点；
        historical BFS 表示从热命中继续跨分片发现关联历史节点，
        不等同于存储生命周期里的 temperature=cold。
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数
            hot_window_minutes: 热窗口（分钟），在此时间窗口内的节点走热路径
            session_id: 会话 ID（用于 session 节点优先级加权）
            
        Returns:
            [{node_id, node_type, label, content, score, source}, ...]
        """
        now = time.time()
        hot_threshold = now - hot_window_minutes * 60
        results: Dict[str, Dict[str, Any]] = {}
        
        try:
            loop = asyncio.get_running_loop()
            hot_task = loop.run_in_executor(
                None,
                self._retrieve_hot_path,
                query_text,
                hot_threshold,
                session_id,
            )
            summary_task = loop.run_in_executor(
                None,
                self._retrieve_summary_navigation,
                query_text,
                top_k,
            )

            hot_results, summary_results = await asyncio.gather(hot_task, summary_task)
            for item in hot_results:
                node_id = item.get("node_id")
                if node_id:
                    results[node_id] = item

            # ── 历史扩展: 从热命中做 BFS 扩散，发现关联但不在首屏热扫描里的节点 ──
            hot_node_ids = list(results.keys())[:20]
            if hot_node_ids:
                historical_results = await loop.run_in_executor(
                    None,
                    self._retrieve_historical_bfs_from_hot,
                    query_text,
                    hot_node_ids,
                    now,
                    set(results.keys()),
                )
                for item in historical_results:
                    node_id = item.get("node_id")
                    if node_id and node_id not in results:
                        results[node_id] = item

            # ── 摘要导航: SQLite L1 摘要命中后带图地址回跳 ──
            for item in summary_results:
                node_id = item.get("node_id") or item.get("graph_memory_id")
                if not node_id:
                    continue
                if node_id in results:
                    self._merge_summary_navigation_result(results[node_id], item)
                else:
                    results[node_id] = item
            
            # ── 排序取 top_k ──
            sorted_results = sorted(results.values(), key=lambda x: x['score'], reverse=True)[:top_k]
            
            # ── 缓存 top-3 节点 ID 供 BFS 种子扩展 ──
            self._last_retrieved_node_ids = [r['node_id'] for r in sorted_results[:3]]
            self.enqueue_active_skeleton_refresh(self._last_retrieved_node_ids)
            
            logger.debug(
                f"[ShardedMemoryGraph] retrieve_context: query='{query_text[:30]}...', "
                f"results={len(sorted_results)}"
            )
            return sorted_results
            
        except Exception as e:
            logger.warning(f"[ShardedMemoryGraph] retrieve_context 异常: {e}")
            return []

    def _retrieve_summary_navigation(
        self,
        query_text: str,
        top_k: int,
        exclude_node_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """检索 L1 摘要导航层，并返回可回跳 MemoryGraph 的图地址。

        优先使用 DualIndexSummaryStore.hybrid_search()，即 SQL 过滤 +
        FAISS 摘要向量并行检索；不可用时降级到 SQLite 摘要导航。
        """
        exclude_node_ids = exclude_node_ids or set()
        if not query_text:
            return []

        try:
            from ..summary_store import get_dual_index_summary_store
            store = get_dual_index_summary_store()
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            if hasattr(store, "hybrid_search") and not (running_loop and running_loop.is_running()):
                summary_hits = asyncio.run(
                    store.hybrid_search(
                        query_text,
                        top_k=max(top_k, 3),
                        include_details=False,
                    )
                )
            else:
                summary_hits = store.search_graph_summaries(query_text, top_k=max(top_k, 3))
        except Exception as exc:
            logger.debug(f"[ShardedMemoryGraph] 摘要导航检索跳过: {exc}")
            return []

        results: List[Dict[str, Any]] = []
        for hit in summary_hits:
            graph_memory_id = (
                getattr(hit, "graph_memory_id", "")
                or (getattr(hit, "source_node_ids", []) or [""])[0]
            )
            if not graph_memory_id or graph_memory_id in exclude_node_ids:
                continue

            node, resolved_shard_id = self._get_node_with_shard(graph_memory_id)
            shard_id = getattr(hit, "shard_id", "") or resolved_shard_id or ""
            full_path = getattr(hit, "full_path", "") or graph_memory_id
            summary_text = getattr(hit, "summary_text", "") or ""
            lexical_score = self._bigram_overlap_score(query_text, summary_text)
            try:
                importance_score = float(getattr(hit, "relevance_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                importance_score = 0.0
            score = max(0.12, lexical_score * 0.9 + importance_score * 0.1)

            if node is not None:
                metadata = dict(getattr(node, "metadata", {}) or {})
                metadata.setdefault("summary_ref_id", getattr(hit, "summary_id", ""))
                metadata.setdefault("source_node_ids", getattr(hit, "source_node_ids", []) or [graph_memory_id])
                address = self._build_memory_address(
                    node,
                    shard_id=shard_id,
                    source="summary_navigation",
                )
                address["summary_ref_id"] = getattr(hit, "summary_id", "") or address.get("summary_ref_id", "")
                address["source_node_ids"] = getattr(hit, "source_node_ids", []) or address.get("source_node_ids", [])
                results.append({
                    "node_id": graph_memory_id,
                    "graph_memory_id": graph_memory_id,
                    "shard_id": shard_id,
                    "full_path": full_path,
                    "node_type": getattr(node, "node_type", "unknown"),
                    "label": getattr(node, "label", "") or full_path,
                    "content": summary_text,
                    "summary": summary_text,
                    "score": score,
                    "source": "summary_navigation",
                    "importance": getattr(node, "importance", "NORMAL"),
                    "metadata": metadata,
                    "memory_address": address,
                    "summary_ref_id": getattr(hit, "summary_id", ""),
                    "recall_hint": "这是摘要导航命中；需要详情时，用 graph_memory_id 调用 read_memory_node 或 discover_related 做增量回忆。",
                })
                continue

            source_node_ids = getattr(hit, "source_node_ids", []) or [graph_memory_id]
            results.append({
                "node_id": graph_memory_id,
                "graph_memory_id": graph_memory_id,
                "shard_id": shard_id,
                "full_path": full_path,
                "node_type": "summary",
                "label": full_path,
                "content": summary_text,
                "summary": summary_text,
                "score": score,
                "source": "summary_navigation",
                "importance": "normal",
                "metadata": {
                    "summary_ref_id": getattr(hit, "summary_id", ""),
                    "source_node_ids": source_node_ids,
                    "summary_navigation_only": True,
                },
                "memory_address": {
                    "graph_memory_id": graph_memory_id,
                    "node_id": graph_memory_id,
                    "shard_id": shard_id,
                    "full_path": full_path,
                    "source_node_ids": source_node_ids,
                    "backend_ref": "",
                    "summary_ref_id": getattr(hit, "summary_id", ""),
                    "source": "summary_navigation",
                },
                "summary_ref_id": getattr(hit, "summary_id", ""),
                "recall_hint": "这是摘要导航命中；需要详情时，用 graph_memory_id 调用 read_memory_node 或 discover_related 做增量回忆。",
            })

        return results

    @staticmethod
    def _merge_summary_navigation_result(existing: Dict[str, Any], summary_item: Dict[str, Any]) -> None:
        """把摘要导航命中合并到已有热/冷结果，避免同节点重复或摘要丢失。"""
        summary_text = summary_item.get("summary") or summary_item.get("content") or ""
        if summary_text:
            existing["summary"] = summary_text

        summary_ref_id = summary_item.get("summary_ref_id") or ""
        if summary_ref_id:
            existing["summary_ref_id"] = summary_ref_id

        existing["graph_memory_id"] = existing.get("graph_memory_id") or summary_item.get("graph_memory_id")
        existing["shard_id"] = existing.get("shard_id") or summary_item.get("shard_id", "")
        existing["full_path"] = existing.get("full_path") or summary_item.get("full_path", "")
        existing["recall_hint"] = summary_item.get("recall_hint") or existing.get("recall_hint", "")

        try:
            existing["score"] = max(float(existing.get("score", 0.0)), float(summary_item.get("score", 0.0)))
        except (TypeError, ValueError):
            pass

        sources = existing.get("sources")
        if not sources:
            sources = [existing.get("source")] if existing.get("source") else []
        if "summary_navigation" not in sources:
            sources.append("summary_navigation")
        existing["sources"] = sources

        metadata = existing.setdefault("metadata", {})
        if isinstance(metadata, dict):
            if summary_ref_id:
                metadata["summary_ref_id"] = summary_ref_id
            source_node_ids = (summary_item.get("memory_address") or {}).get("source_node_ids")
            if source_node_ids:
                metadata.setdefault("source_node_ids", source_node_ids)

        address = existing.setdefault("memory_address", {})
        item_address = summary_item.get("memory_address") or {}
        if isinstance(address, dict):
            for key in ("graph_memory_id", "shard_id", "full_path", "source_node_ids"):
                if item_address.get(key) and not address.get(key):
                    address[key] = item_address[key]
            if summary_ref_id:
                address["summary_ref_id"] = summary_ref_id
            address.setdefault("source", existing.get("source", ""))
            address["summary_navigation_matched"] = True
    
    @staticmethod
    def _bigram_overlap_score(query: str, text: str) -> float:
        """Bigram 重叠度评分 — 适用于中文文本"""
        if not query or not text:
            return 0.0
        q = query.lower()
        t = text.lower()
        q_grams = {q[i:i+2] for i in range(len(q)-1)} if len(q) >= 2 else {q}
        t_grams = {t[i:i+2] for i in range(len(t)-1)} if len(t) >= 2 else {t}
        if not q_grams or not t_grams:
            return 0.0
        intersection = q_grams & t_grams
        return len(intersection) / max(len(q_grams), len(t_grams))
    
    def hebbian_strengthen(self):
        """赫布增强 — 加强共激活边的权重（兼容 MemoryGraph 接口）
        
        Hebbian 公式: new_weight = old_weight + η × (1.0 - old_weight)
        其中 η = 0.1，权重渐近逼近 1.0。
        仅更新 compute_activations 所记录的 _last_activated_edges。
        """
        if not self._last_activated_edges:
            return
        
        eta = 0.1
        strengthened = 0
        skipped = 0
        cross_edge_updated = False
        
        for src_id, dst_id in self._last_activated_edges:
            # 查找边所在的分片
            found = False
            for shard_id in self.list_all_shards():
                shard = self.get_shard(shard_id, load_if_missing=True)
                if not shard:
                    continue
                if src_id not in shard.topology or dst_id not in shard.topology:
                    continue
                edge_info = shard.get_topology_edge_info(src_id, dst_id)
                if edge_info is None:
                    continue
                
                # 读取边属性
                edge = shard.properties.get_edge(src_id, dst_id)
                if edge is None:
                    continue
                if getattr(edge, 'protected', False):
                    skipped += 1
                    found = True
                    break
                
                # 应用赫布公式
                old_weight = getattr(edge, 'weight', 0.5)
                edge.weight = old_weight + eta * (1.0 - old_weight)
                edge.last_activated = time.time()
                edge.activation_count = getattr(edge, 'activation_count', 0) + 1
                
                try:
                    shard.properties.set_edge(edge)
                    strengthened += 1
                except Exception:
                    skipped += 1
                found = True
                break
            
            if not found:
                cross_edges = self._get_cross_edges_from(src_id)
                cross_edge = next((edge for edge in cross_edges if edge.get("dst_id") == dst_id), None)
                if cross_edge:
                    updated_edge = dict(cross_edge)
                    if updated_edge.get("protected", False):
                        skipped += 1
                    else:
                        old_weight = float(updated_edge.get("weight", 0.5))
                        updated_edge["weight"] = old_weight + eta * (1.0 - old_weight)
                        updated_edge["last_activated"] = time.time()
                        updated_edge["activation_count"] = int(updated_edge.get("activation_count", 0)) + 1
                        self.global_index.set_cross_edge(updated_edge)
                        strengthened += 1
                        cross_edge_updated = True
                else:
                    skipped += 1

        if cross_edge_updated:
            self.global_index.sync()
        
        logger.info(
            f"[ShardedMemoryGraph] 赫布增强: {strengthened}/{len(self._last_activated_edges)} 条边权重已更新"
            + (f" ({skipped} skipped)" if skipped else "")
        )
    
    def set_active_nodes(self, node_ids: List[str]):
        """设置当前活跃节点 ID 集合（前端高亮用）。"""
        self._active_node_ids = set(node_ids or [])
        self._last_focus_context = {
            "active_node_ids": list(node_ids or []),
            "focus_path": list(node_ids or []),
            "focus_depth": max(0, len(node_ids or []) - 1),
            "saved_at": time.time(),
        }
        self.enqueue_active_skeleton_refresh(list(node_ids or []))

    def get_active_node_ids(self) -> List[str]:
        """获取当前活跃节点 ID 列表。"""
        return list(self._active_node_ids)

    def update_focus_to_node(self, node_id: str) -> bool:
        """沿 HIERARCHY 入边构造当前焦点路径。"""
        if not self.has_node(node_id):
            return False
        path = [node_id]
        current = node_id
        visited = {node_id}
        for _ in range(20):
            parent_id = self._find_parent_id(current)
            if not parent_id or parent_id in visited:
                break
            visited.add(parent_id)
            path.append(parent_id)
            current = parent_id
        path.reverse()
        self._active_node_ids = set(path)
        self._last_focus_context = {
            "active_node_ids": list(path),
            "focus_path": list(path),
            "focus_depth": len(path) - 1,
            "saved_at": time.time(),
        }
        self.enqueue_active_skeleton_refresh(path)
        return True

    def _find_parent_id(self, node_id: str) -> Optional[str]:
        shard, _ = self._get_indexed_shard(node_id)
        candidate_shards = []
        if shard:
            candidate_shards.append(shard)
        else:
            for shard_id in self.list_all_shards():
                shard = self.get_shard(shard_id, load_if_missing=True)
                if shard:
                    candidate_shards.append(shard)

        for shard in candidate_shards:
            for src_id in shard.get_topology_neighbors(node_id, edge_type="hierarchy", mode="in"):
                return src_id
        for edge in self._get_cross_edges_to(node_id, edge_type="hierarchy"):
            src_id = edge.get("src_id")
            if src_id:
                return src_id
        return None

    def get_last_focus_context(self) -> Optional[Dict]:
        """获取上次保存的焦点上下文（兼容 MemoryGraph 接口）
        
        分片存储模式下保存轻量焦点路径。
        """
        return self._last_focus_context

    def _retrieve_hot_path(
        self,
        query_text: str,
        hot_threshold: float,
        session_id: str = "",
    ) -> List[Dict[str, Any]]:
        """热路径：当前分片热节点扫描 + 关键词打分。"""
        current_shard = self.get_current_shard()
        if not current_shard:
            return []

        hot_nodes = []
        try:
            for node in current_shard.properties.iter_nodes():
                if getattr(node, "last_accessed", 0) >= hot_threshold:
                    hot_nodes.append(node)
                if len(hot_nodes) >= 500:
                    break
        except Exception:
            return []

        results: List[Dict[str, Any]] = []
        for node in hot_nodes:
            text_parts = [
                getattr(node, "label", "") or "",
                getattr(node, "content", "") or "",
            ]
            combined = " ".join(filter(None, text_parts))
            if not combined.strip():
                continue
            score = self._bigram_overlap_score(query_text, combined)
            if score <= 0:
                continue
            if session_id and node.node_id.startswith(f"dialogue:session_{session_id}"):
                score *= 2.0

            importance = getattr(node, "importance", "NORMAL")
            importance_boost = {
                "MUST_REMEMBER": 2.0,
                "IMPORTANT": 1.5,
                "IDENTITY": 1.8,
                "FACT": 1.3,
            }
            score *= importance_boost.get(importance, 1.0)

            metadata = getattr(node, "metadata", {}) or {}
            results.append({
                "node_id": node.node_id,
                "graph_memory_id": node.node_id,
                "shard_id": getattr(node, "storage_shard", "") or self._find_node_shard_id(node.node_id) or "",
                "full_path": metadata.get("full_path") or metadata.get("graph_address") or node.node_id,
                "node_type": getattr(node, "node_type", "unknown"),
                "label": getattr(node, "label", ""),
                "content": getattr(node, "content", ""),
                "summary": getattr(node, "content_summary", "") or metadata.get("content_summary", ""),
                "score": score,
                "source": "hot",
                "importance": importance,
                "metadata": metadata,
                "memory_address": self._build_memory_address(node, source="hot"),
                "recall_hint": "需要详情时，用 graph_memory_id 调用 read_memory_node 或 discover_related。",
            })
        return results

    def _retrieve_historical_bfs_from_hot(
        self,
        query_text: str,
        hot_node_ids: List[str],
        now: float,
        seen_ids: Set[str],
    ) -> List[Dict[str, Any]]:
        """历史扩展：从热命中做跨分片 BFS，发现关联的历史节点。

        这里的 historical 表示检索层面的“热命中之外的关联历史”，
        不等同于存储生命周期里的 temperature=cold。
        """
        if not hot_node_ids:
            return []

        results: List[Dict[str, Any]] = []
        discovered = self.discover_across_shards(
            seed_ids=hot_node_ids[:20],
            seed_shard_id=self._get_shard_id(now),
            max_depth=2,
            max_nodes=200,
        )
        for node_id, distance, shard_id in discovered:
            if node_id in seen_ids:
                continue
            shard = self.get_shard(shard_id, load_if_missing=True)
            if not shard:
                continue
            node = shard.get_node(node_id)
            if node is None:
                continue
            combined = " ".join(filter(None, [
                getattr(node, "label", "") or "",
                getattr(node, "content", "") or "",
            ]))
            score = self._bigram_overlap_score(query_text, combined) * 0.8
            if score <= 0:
                continue
            metadata = getattr(node, "metadata", {}) or {}
            results.append({
                "node_id": node_id,
                "graph_memory_id": node_id,
                "shard_id": shard_id,
                "full_path": metadata.get("full_path") or metadata.get("graph_address") or node_id,
                "node_type": getattr(node, "node_type", "unknown"),
                "label": getattr(node, "label", ""),
                "content": getattr(node, "content", ""),
                "summary": getattr(node, "content_summary", "") or metadata.get("content_summary", ""),
                "score": score,
                "source": "historical_bfs",
                "importance": getattr(node, "importance", "NORMAL"),
                "metadata": metadata,
                "memory_address": self._build_memory_address(node, shard_id=shard_id, source="historical_bfs"),
                "recall_hint": "需要详情时，用 graph_memory_id 调用 read_memory_node 或 discover_related。",
            })
            seen_ids.add(node_id)
        return results
    
    def remove_node(self, node_id: str) -> bool:
        """删除节点及其索引记录（兼容 MemoryGraph 接口）。"""
        with self.shard_lock:
            shard, shard_id, definitive_miss = self._get_indexed_shard_status(node_id)
            if definitive_miss:
                return False

            if not shard or not shard_id:
                shard_id = self._find_node_shard_id(node_id)
                shard = self.get_shard(shard_id, load_if_missing=True) if shard_id else None

            if not shard or not shard_id:
                return False

            removed_edge_count = 0
            try:
                removed_edge_count = int(shard.topology.get_node_degree(node_id, mode="all"))
            except Exception:
                try:
                    removed_edge_count = (
                        len(shard.get_topology_neighbors(node_id, mode="out"))
                        + len(shard.get_topology_neighbors(node_id, mode="in"))
                    )
                except Exception:
                    removed_edge_count = 0

            if not shard.remove_node(node_id):
                return False

            try:
                self.global_index.delete_node_location(node_id)
            except Exception as exc:
                logger.debug(f"全局索引删除跳过 node={node_id}: {exc}")

            try:
                local_index = self.get_local_index(shard_id, create=False)
                if local_index:
                    local_index.delete_node_header(node_id)
                    self._record_local_index_update(shard_id)
            except Exception as exc:
                logger.debug(f"local_index 删除跳过 node={node_id}: {exc}")

            shard_info = self.shard_index.setdefault("shards", {}).setdefault(shard_id, {})
            shard_info["node_count"] = max(0, int(shard_info.get("node_count") or 0) - 1)
            if removed_edge_count:
                shard_info["edge_count"] = max(0, int(shard_info.get("edge_count") or 0) - removed_edge_count)

            self._drop_node_from_active_state(node_id)
            self._last_topology_write_at = time.time()
            self._check_topology_delta_compaction(shard_id, shard)
            self._save_shard_index()
            logger.info(f"[ShardedMemoryGraph] 已删除节点: {node_id} (shard={shard_id})")
            return True

    def get_importance(self, node_id: str):
        """读取节点重要度（兼容 MemoryGraph 接口）。"""
        node = self.get_node(node_id)
        if not node:
            return None
        raw = (
            getattr(node, "importance", None)
            or (getattr(node, "metadata", None) or {}).get("importance")
            or "normal"
        )
        raw_value = str(getattr(raw, "value", raw) or "normal").lower()
        try:
            from zulong.memory.memory_graph import Importance
            return Importance(raw_value)
        except Exception:
            return raw_value
    
    def set_importance(self, node_id: str, importance) -> None:
        """设置节点重要性（兼容 MemoryGraph 接口）"""
        node = self.get_node(node_id)
        if not node:
            return
        importance_value = getattr(importance, "value", importance) or "normal"
        node.importance = str(importance_value)
        metadata = dict(getattr(node, "metadata", {}) or {})
        metadata["importance"] = str(importance_value)
        node.metadata = metadata
        self.update_node(node)
    
    def get_children(self, node_id: str, edge_type=None) -> list:
        """获取子节点列表（兼容 MemoryGraph 接口）
        """
        edge_type_str = getattr(edge_type, "value", edge_type) or "hierarchy"
        children = []
        shard, _ = self._get_indexed_shard(node_id)
        candidate_shards = []
        if shard:
            candidate_shards.append(shard)
        else:
            for shard_id in self.list_all_shards():
                shard = self.get_shard(shard_id, load_if_missing=True)
                if shard:
                    candidate_shards.append(shard)

        for shard in candidate_shards:
            for child_id in shard.get_topology_neighbors(node_id, edge_type=edge_type_str, mode="out"):
                node = self.get_node(child_id)
                if node:
                    children.append(node)
        seen = {child.node_id for child in children}
        for edge in self._get_cross_edges_from(node_id, edge_type=edge_type_str):
            child_id = edge.get("dst_id")
            if not child_id or child_id in seen:
                continue
            node = self.get_node(child_id)
            if node:
                children.append(node)
                seen.add(child_id)
        children.sort(key=lambda n: getattr(n, "created_at", 0.0))
        return children

    def get_parent(self, node_id: str, edge_type=None) -> Optional[NodeProperties]:
        """沿 HIERARCHY 入边查找父节点（兼容 MemoryGraph 接口）。"""
        edge_type_str = getattr(edge_type, "value", edge_type) or "hierarchy"
        parent_id = self._find_parent_id(node_id) if edge_type_str == "hierarchy" else None
        if parent_id:
            return self.get_node(parent_id)

        shard, _ = self._get_indexed_shard(node_id)
        candidate_shards = [shard] if shard else []
        if not candidate_shards:
            for shard_id in self.list_all_shards():
                candidate = self.get_shard(shard_id, load_if_missing=True)
                if candidate:
                    candidate_shards.append(candidate)

        for shard in candidate_shards:
            for src_id in shard.get_topology_neighbors(node_id, edge_type=edge_type_str, mode="in"):
                parent = self.get_node(src_id)
                if parent:
                    return parent
        for edge in self._get_cross_edges_to(node_id, edge_type=edge_type_str):
            parent = self.get_node(edge.get("src_id", ""))
            if parent:
                return parent
        return None

    def get_edge(self, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        """获取边属性（兼容 MemoryGraph 接口）。"""
        for shard_id in self.list_all_shards():
            shard = self.get_shard(shard_id, load_if_missing=True)
            if not shard:
                continue
            edge_info = shard.get_topology_edge_info(source_id, target_id)
            if edge_info is None:
                continue
            edge_props = shard.get_edge(source_id, target_id)
            if edge_props:
                return edge_props.to_dict()
            edge_type, weight = edge_info
            return {"edge_type": edge_type, "weight": weight}

        for edge in self._get_cross_edges_from(source_id):
            if edge.get("dst_id") == target_id:
                return dict(edge)
        return None

    def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[Set[Any]] = None,
        max_depth: int = 1,
    ) -> List[NodeProperties]:
        """获取邻域节点（兼容 MemoryGraph 接口，入边/出边均可达）。"""
        edge_type_values = None
        if edge_types:
            edge_type_values = {getattr(edge_type, "value", edge_type) for edge_type in edge_types}

        visited = {node_id}
        current_layer = [node_id]
        result: List[NodeProperties] = []

        for _ in range(max(0, max_depth)):
            next_layer = []
            for current_id in current_layer:
                shard, _ = self._get_indexed_shard(current_id)
                candidate_shards = [shard] if shard else []
                if not candidate_shards:
                    for shard_id in self.list_all_shards():
                        candidate = self.get_shard(shard_id, load_if_missing=True)
                        if candidate:
                            candidate_shards.append(candidate)

                neighbor_ids: List[str] = []
                for shard in candidate_shards:
                    for mode in ("out", "in"):
                        raw_neighbors = shard.get_topology_neighbors(current_id, mode=mode)
                        for neighbor_id in raw_neighbors:
                            if edge_type_values:
                                if mode == "out":
                                    edge_info = shard.get_topology_edge_info(current_id, neighbor_id)
                                else:
                                    edge_info = shard.get_topology_edge_info(neighbor_id, current_id)
                                if not edge_info or edge_info[0] not in edge_type_values:
                                    continue
                            neighbor_ids.append(neighbor_id)

                cross_edges = self._get_cross_edges_from(current_id) + self._get_cross_edges_to(current_id)
                for edge in cross_edges:
                    edge_type = edge.get("edge_type")
                    if edge_type_values and edge_type not in edge_type_values:
                        continue
                    if edge.get("src_id") == current_id:
                        neighbor_ids.append(edge.get("dst_id", ""))
                    elif edge.get("dst_id") == current_id:
                        neighbor_ids.append(edge.get("src_id", ""))

                for neighbor_id in neighbor_ids:
                    if not neighbor_id or neighbor_id in visited:
                        continue
                    visited.add(neighbor_id)
                    next_layer.append(neighbor_id)
                    node = self.get_node(neighbor_id)
                    if node:
                        result.append(node)

            current_layer = next_layer
            if not current_layer:
                break

        return result

    def get_subgraph_summary(self, node_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """提取以 node_id 为中心的子图摘要（兼容 MemoryGraph 接口）。"""
        center = self.get_node(node_id)
        if not center:
            return {}
        neighbors = self.get_neighbors(node_id, max_depth=max_depth)
        type_counts: Dict[str, int] = {}
        for node in neighbors:
            node_type = getattr(node, "node_type", "unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return {
            "center": center.to_dict(),
            "neighbor_count": len(neighbors),
            "type_distribution": type_counts,
            "neighbors": [node.to_dict() for node in neighbors[:20]],
        }

    def search_nodes(
        self,
        query: str,
        node_types: Optional[List[Any]] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """关键词搜索节点（兼容 MemoryGraph 降级检索接口）。"""
        query_lower = (query or "").lower()
        if not query_lower:
            return []

        type_values = None
        if node_types:
            type_values = {getattr(node_type, "value", node_type) for node_type in node_types}

        results: List[Dict[str, Any]] = []
        for shard_id in self.list_all_shards():
            shard = self.get_shard(shard_id, load_if_missing=True)
            if not shard:
                continue
            for node in shard.properties.iter_nodes():
                if type_values and node.node_type not in type_values:
                    continue
                metadata = node.metadata or {}
                searchable = " ".join([
                    node.node_id or "",
                    node.label or "",
                    node.content or "",
                    node.content_summary or "",
                    json.dumps(metadata, ensure_ascii=False),
                ]).lower()

                score = 0.0
                if query_lower in (node.node_id or "").lower():
                    score = 1.0
                elif query_lower in (node.label or "").lower():
                    score = 0.8
                elif query_lower in searchable:
                    score = 0.5
                else:
                    score = self._bigram_overlap_score(query_lower, searchable)

                if score <= 0:
                    continue
                score += float(getattr(node, "activation", 0.0) or 0.0) * 0.2
                results.append({
                    "node_id": node.node_id,
                    "graph_memory_id": node.node_id,
                    "shard_id": shard_id,
                    "full_path": metadata.get("full_path") or metadata.get("graph_address") or node.node_id,
                    "type": node.node_type,
                    "node_type": node.node_type,
                    "label": node.label,
                    "content": node.content or node.content_summary or metadata.get("content", ""),
                    "summary": node.content_summary or metadata.get("content_summary", ""),
                    "score": round(score, 4),
                    "activation": round(float(getattr(node, "activation", 0.0) or 0.0), 3),
                    "source": "keyword_scan",
                    "metadata": metadata,
                    "memory_address": self._build_memory_address(node, shard_id=shard_id, source="keyword_scan"),
                    "recall_hint": "需要详情时，用 graph_memory_id 调用 read_memory_node 或 discover_related。",
                })

        results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return results[:max_results]

    def to_frontend_dict(self, depth: Optional[int] = None) -> Dict[str, Any]:
        """序列化为前端记忆图谱格式。"""
        if depth == 0:
            skeleton = self.get_active_skeleton()
            skeleton_node_ids: List[str] = []
            if isinstance(skeleton, dict):
                seen_skeleton_ids: Set[str] = set()
                for node_id in list(skeleton.get("center_ids") or []) + list(skeleton.get("node_ids") or []):
                    node_id = str(node_id or "")
                    if not node_id or node_id in seen_skeleton_ids:
                        continue
                    seen_skeleton_ids.add(node_id)
                    skeleton_node_ids.append(node_id)

            if skeleton_node_ids:
                # TSD v2.9.16 / 21.9.5: 首屏显示活跃骨架的一跳邻接，而不是孤立根节点。
                first_view_limit = 200
                returned_node_ids = skeleton_node_ids[:first_view_limit]
                visible_ids = set(returned_node_ids)
                visible_child_counts: Dict[str, int] = {}
                edges: List[Dict[str, Any]] = []
                seen_edges: Set[Tuple[str, str, str]] = set()

                for edge in skeleton.get("edges") or []:
                    if not isinstance(edge, dict):
                        continue
                    src_id = str(edge.get("source") or edge.get("src_id") or "")
                    dst_id = str(edge.get("target") or edge.get("dst_id") or "")
                    edge_type = str(edge.get("type") or edge.get("edge_type") or "association")
                    if not src_id or not dst_id or src_id not in visible_ids or dst_id not in visible_ids:
                        continue
                    if self._should_hide_frontend_edge(src_id, dst_id, edge_type, edge=edge):
                        continue
                    edge_key = (src_id, dst_id, edge_type)
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)
                    metadata = dict(edge.get("metadata") or {})
                    edges.append({
                        "source": src_id,
                        "target": dst_id,
                        "type": edge_type,
                        "weight": edge.get("weight", 1.0),
                        "protected": edge.get("protected", False),
                        "metadata": metadata,
                    })
                    if edge_type == "hierarchy":
                        visible_child_counts[src_id] = visible_child_counts.get(src_id, 0) + 1

                nodes: List[Dict[str, Any]] = []
                for node_id in returned_node_ids:
                    node = self.get_node(node_id)
                    if not node:
                        continue
                    data = self._serialize_node_for_frontend(node, include_children_count=False)
                    data["children_count"] = visible_child_counts.get(node_id, 0)
                    nodes.append(data)

                visible_node_ids = {str(node.get("id")) for node in nodes if node.get("id")}
                edges = [
                    edge for edge in edges
                    if edge.get("source") in visible_node_ids and edge.get("target") in visible_node_ids
                ]

                stats = self.get_total_stats()
                skeleton_stats = dict(skeleton.get("stats") or {}) if isinstance(skeleton, dict) else {}
                stats["returned_nodes"] = len(nodes)
                stats["returned_edges"] = len(edges)
                stats["active_skeleton_nodes"] = skeleton_stats.get(
                    "node_count",
                    len(skeleton.get("node_ids") or []) if isinstance(skeleton, dict) else len(skeleton_node_ids),
                )
                stats["active_skeleton_edges"] = skeleton_stats.get(
                    "edge_count",
                    len(skeleton.get("edges") or []) if isinstance(skeleton, dict) else len(edges),
                )
                stats["view_limit"] = first_view_limit
                stats["limited"] = len(skeleton_node_ids) > first_view_limit
                stats["transport"] = "active_skeleton"

                center_ids = list(skeleton.get("center_ids") or []) if isinstance(skeleton, dict) else list(self._active_node_ids)
                return {
                    "nodes": nodes,
                    "edges": edges,
                    "stats": stats,
                    "active_node_ids": list(self._active_node_ids),
                    "thought_view": {"nodes": nodes, "edges": edges, "center_ids": center_ids},
                    "active_skeleton": skeleton,
                }

            hierarchy_children: Set[str] = set()
            hierarchy_parents: Set[str] = set()
            child_counts: Dict[str, int] = {}

            for shard_id in self.list_all_shards():
                shard = self.get_shard(shard_id, load_if_missing=True)
                if not shard:
                    continue
                for edge in shard.topology.graph.es:
                    edge_type = edge["type"] if "type" in edge.attributes() else "association"
                    if edge_type != "hierarchy":
                        continue
                    src_id = self._vertex_node_id(shard.topology.graph.vs[edge.source])
                    dst_id = self._vertex_node_id(shard.topology.graph.vs[edge.target])
                    hierarchy_parents.add(src_id)
                    hierarchy_children.add(dst_id)
                    child_counts[src_id] = child_counts.get(src_id, 0) + 1

            for edge in self._iter_cross_edges():
                if edge.get("edge_type", "association") != "hierarchy":
                    continue
                src_id = edge.get("src_id")
                dst_id = edge.get("dst_id")
                if not src_id or not dst_id:
                    continue
                hierarchy_parents.add(src_id)
                hierarchy_children.add(dst_id)
                child_counts[src_id] = child_counts.get(src_id, 0) + 1

            root_ids = sorted(hierarchy_parents - hierarchy_children)
            root_limit = 200
            returned_root_ids = root_ids[:root_limit]
            nodes = []
            for node_id in returned_root_ids:
                node = self.get_node(node_id)
                if not node:
                    continue
                data = self._serialize_node_for_frontend(node, include_children_count=False)
                data["children_count"] = child_counts.get(node_id, 0)
                nodes.append(data)

            stats = self.get_total_stats()
            stats["returned_nodes"] = len(nodes)
            stats["returned_edges"] = 0
            stats["root_count"] = len(root_ids)
            stats["root_limit"] = root_limit
            stats["limited"] = len(root_ids) > root_limit
            stats["transport"] = "limited_root"

            return {
                "nodes": nodes,
                "edges": [],
                "stats": stats,
                "active_node_ids": list(self._active_node_ids),
                "thought_view": {"nodes": [], "edges": [], "center_ids": list(self._active_node_ids)},
                "active_skeleton": self.get_active_skeleton(),
            }

        nodes = []
        edges = []
        hierarchy_children: Set[str] = set()
        hierarchy_parents: Set[str] = set()

        for shard_id in self.list_all_shards():
            shard = self.get_shard(shard_id, load_if_missing=True)
            if not shard:
                continue
            for vertex in shard.topology.graph.vs:
                node_id = self._vertex_node_id(vertex)
                node = shard.get_node(node_id)
                if node:
                    nodes.append(self._serialize_node_for_frontend(node, include_children_count=True))
            for edge in shard.topology.graph.es:
                src_id = self._vertex_node_id(shard.topology.graph.vs[edge.source])
                dst_id = self._vertex_node_id(shard.topology.graph.vs[edge.target])
                edge_type = edge["type"] if "type" in edge.attributes() else "association"
                edge_props = shard.get_edge(src_id, dst_id)
                if self._should_hide_frontend_edge(src_id, dst_id, edge_type, edge_props=edge_props):
                    continue
                edges.append(self._serialize_edge_for_frontend(src_id, dst_id, edge_type, edge_props))
                if edge_type == "hierarchy":
                    hierarchy_parents.add(src_id)
                    hierarchy_children.add(dst_id)

        for edge in self._iter_cross_edges():
            src_id = edge.get("src_id")
            dst_id = edge.get("dst_id")
            edge_type = edge.get("edge_type", "association")
            if not src_id or not dst_id:
                continue
            if self._should_hide_frontend_edge(src_id, dst_id, edge_type, edge=edge):
                continue
            edges.append(self._serialize_cross_edge_for_frontend(edge))
            if edge_type == "hierarchy":
                hierarchy_parents.add(src_id)
                hierarchy_children.add(dst_id)

        if depth == 0:
            root_ids = hierarchy_parents - hierarchy_children
            if root_ids:
                nodes = [n for n in nodes if n["id"] in root_ids]
                root_set = set(root_ids)
                edges = [e for e in edges if e["source"] in root_set and e["target"] in root_set]

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": self.get_total_stats(),
            "active_node_ids": list(self._active_node_ids),
            "thought_view": {"nodes": [], "edges": [], "center_ids": list(self._active_node_ids)},
            "active_skeleton": self.get_active_skeleton(),
        }

    def get_node_children_for_frontend(self, node_id: str) -> Dict[str, Any]:
        visible_child_ids = self._frontend_hierarchy_child_ids(node_id)
        children = [
            child for child in self.get_children(node_id)
            if child.node_id in visible_child_ids
        ]
        child_ids = {c.node_id for c in children}
        nodes = [self._serialize_node_for_frontend(c, include_children_count=True) for c in children]
        edges = []
        for shard_id in self.list_all_shards():
            shard = self.get_shard(shard_id, load_if_missing=True)
            if not shard:
                continue
            for edge in shard.topology.graph.es:
                src_id = self._vertex_node_id(shard.topology.graph.vs[edge.source])
                dst_id = self._vertex_node_id(shard.topology.graph.vs[edge.target])
                if (src_id == node_id and dst_id in child_ids) or (src_id in child_ids and dst_id in child_ids):
                    edge_type = edge["type"] if "type" in edge.attributes() else "association"
                    edge_props = shard.get_edge(src_id, dst_id)
                    if self._should_hide_frontend_edge(src_id, dst_id, edge_type, edge_props=edge_props):
                        continue
                    edges.append(self._serialize_edge_for_frontend(src_id, dst_id, edge_type, edge_props))
        for edge in self._iter_cross_edges():
            src_id = edge.get("src_id")
            dst_id = edge.get("dst_id")
            if (src_id == node_id and dst_id in child_ids) or (src_id in child_ids and dst_id in child_ids):
                if self._should_hide_frontend_edge(
                    src_id,
                    dst_id,
                    edge.get("edge_type", "association"),
                    edge=edge,
                ):
                    continue
                edges.append(self._serialize_cross_edge_for_frontend(edge))
        return {"parent_id": node_id, "nodes": nodes, "edges": edges}

    def _vertex_node_id(self, vertex) -> str:
        attrs = vertex.attributes()
        if "node_id" in attrs:
            return vertex["node_id"]
        return vertex["name"]

    def _serialize_node_for_frontend(self, node: NodeProperties, include_children_count: bool = False) -> Dict[str, Any]:
        metadata = dict(node.metadata or {})
        node_type = str(getattr(getattr(node, "node_type", ""), "value", getattr(node, "node_type", "")) or "")
        if node_type == "dialogue":
            metadata["task_graph_address"] = ""
            full_path = str(metadata.get("full_path") or "")
            if full_path.startswith("task:") or "/task:" in full_path:
                metadata["full_path"] = node.node_id
        metadata.setdefault("content", node.content or "")
        metadata.setdefault("importance", node.importance)
        metadata.setdefault("temperature", node.temperature)
        data = {
            "id": node.node_id,
            "type": node.node_type,
            "label": node.label,
            "activation": node.activation,
            "metadata": metadata,
            "backend_ref": node.backend_ref,
            "created_at": node.created_at,
            "last_accessed": node.last_accessed,
            "access_count": node.access_count,
        }
        if include_children_count:
            data["children_count"] = len(self._frontend_hierarchy_child_ids(node.node_id))
        data["graph_memory_id"] = node.node_id
        data["shard_id"] = getattr(node, "storage_shard", "") or metadata.get("shard_id", "")
        data["full_path"] = metadata.get("full_path") or metadata.get("graph_address") or node.node_id
        data["memory_address"] = self._build_memory_address(node)
        return data

    _LEGACY_DIALOGUE_TASK_LINK_TYPES = {
        "dialogue_round_task",
        "dialogue_to_task",
        "sub_dialogue_to_task",
    }

    def _frontend_hierarchy_child_ids(self, node_id: str) -> Set[str]:
        child_ids: Set[str] = set()
        for shard_id in self.list_all_shards():
            shard = self.get_shard(shard_id, load_if_missing=True)
            if not shard:
                continue
            for edge in shard.topology.graph.es:
                src_id = self._vertex_node_id(shard.topology.graph.vs[edge.source])
                if src_id != node_id:
                    continue
                dst_id = self._vertex_node_id(shard.topology.graph.vs[edge.target])
                edge_type = edge["type"] if "type" in edge.attributes() else "association"
                if edge_type != "hierarchy":
                    continue
                edge_props = shard.get_edge(src_id, dst_id)
                if self._should_hide_frontend_edge(src_id, dst_id, edge_type, edge_props=edge_props):
                    continue
                child_ids.add(dst_id)
        for edge in self._iter_cross_edges():
            if edge.get("src_id") != node_id or edge.get("edge_type", "association") != "hierarchy":
                continue
            dst_id = edge.get("dst_id")
            if not dst_id:
                continue
            if self._should_hide_frontend_edge(node_id, dst_id, "hierarchy", edge=edge):
                continue
            child_ids.add(dst_id)
        return child_ids

    def _node_type_value_for_frontend_filter(self, node_id: str) -> str:
        node = self.get_node(node_id)
        if node:
            node_type = getattr(node, "node_type", "") or ""
            return str(getattr(node_type, "value", node_type) or "").lower()
        if "/dialogue:" in str(node_id) or str(node_id).startswith("dialogue:"):
            return "dialogue"
        if "/task:" in str(node_id) or str(node_id).startswith("task:"):
            return "task"
        return ""

    def _should_hide_frontend_edge(
        self,
        src_id: str,
        dst_id: str,
        edge_type: str,
        *,
        edge_props: Optional[EdgeProperties] = None,
        edge: Optional[Dict[str, Any]] = None,
    ) -> bool:
        metadata = dict(edge.get("metadata") or {}) if edge else {}
        if edge_props and getattr(edge_props, "metadata", None):
            metadata.update(getattr(edge_props, "metadata", {}) or {})
        link_type = str(metadata.get("link_type") or "").strip()
        if link_type in self._LEGACY_DIALOGUE_TASK_LINK_TYPES:
            return True
        if str(edge_type).lower() != "hierarchy":
            return False
        return (
            self._node_type_value_for_frontend_filter(src_id) == "dialogue"
            and self._node_type_value_for_frontend_filter(dst_id) == "task"
        )

    def _serialize_edge_for_frontend(self, src_id: str, dst_id: str, edge_type: str, edge_props: Optional[EdgeProperties]) -> Dict[str, Any]:
        return {
            "source": src_id,
            "target": dst_id,
            "type": edge_type,
            "weight": getattr(edge_props, "weight", 1.0) if edge_props else 1.0,
            "protected": getattr(edge_props, "protected", False) if edge_props else False,
            "metadata": getattr(edge_props, "metadata", {}) if edge_props else {},
        }

    def _serialize_cross_edge_for_frontend(self, edge: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(edge.get("metadata") or {})
        metadata.setdefault("src_shard_id", edge.get("src_shard_id"))
        metadata.setdefault("dst_shard_id", edge.get("dst_shard_id"))
        metadata.setdefault("cross_shard", True)
        return {
            "source": edge.get("src_id"),
            "target": edge.get("dst_id"),
            "type": edge.get("edge_type", "association"),
            "weight": edge.get("weight", 1.0),
            "protected": edge.get("protected", False),
            "metadata": metadata,
        }

    def _build_memory_address(
        self,
        node: NodeProperties,
        shard_id: Optional[str] = None,
        source: str = "",
    ) -> Dict[str, Any]:
        """构造 RAG/工具可回跳的图记忆地址。"""
        metadata = dict(getattr(node, "metadata", {}) or {})
        node_id = node.node_id
        node_type = str(getattr(getattr(node, "node_type", ""), "value", getattr(node, "node_type", "")) or "")
        if node_type == "dialogue":
            metadata["task_graph_address"] = ""
            full_path_value = str(metadata.get("full_path") or "")
            if full_path_value.startswith("task:") or "/task:" in full_path_value:
                metadata["full_path"] = node_id
        resolved_shard_id = (
            shard_id
            or getattr(node, "storage_shard", "")
            or metadata.get("shard_id", "")
        )
        if not resolved_shard_id:
            try:
                resolved_shard_id = self.global_index.get_node_shard(node_id) or ""
            except Exception:
                resolved_shard_id = ""
        full_path = (
            metadata.get("full_path")
            or metadata.get("graph_address")
            or metadata.get("task_graph_address")
            or node_id
        )
        source_node_ids = metadata.get("source_node_ids") or [node_id]
        if isinstance(source_node_ids, str):
            source_node_ids = [source_node_ids]
        return {
            "graph_memory_id": metadata.get("graph_memory_id") or node_id,
            "node_id": node_id,
            "shard_id": resolved_shard_id,
            "full_path": full_path,
            "source_node_ids": source_node_ids,
            "backend_ref": getattr(node, "backend_ref", "") or metadata.get("backend_ref", ""),
            "summary_ref_id": metadata.get("summary_ref_id", ""),
            "source": source,
        }

    def _iter_cross_edges(self) -> List[Dict[str, Any]]:
        try:
            return list(self.global_index.iter_cross_edges())
        except Exception as exc:
            logger.debug(f"读取 LMDB 跨分片边列表失败: {exc}")
            return []
    
    def index_summary(self, node_id: str, summary_text: str) -> None:
        """索引节点摘要（兼容 MemoryGraph 接口）
        
        分片存储下先把摘要写回节点属性；向量索引后续由异步语义边流程接管。
        """
        node = self.get_node(node_id)
        if not node:
            return
        node.content_summary = summary_text
        node.metadata = dict(node.metadata or {})
        node.metadata["content_summary"] = summary_text
        node.metadata.setdefault("graph_memory_id", node_id)
        node.metadata.setdefault("full_path", node.metadata.get("graph_address", node_id))
        node.metadata.setdefault("source_node_ids", [node_id])
        _, shard_id = self._get_node_with_shard(node_id)
        if shard_id:
            node.metadata.setdefault("shard_id", shard_id)
        try:
            from ..summary_store import get_dual_index_summary_store
            store = get_dual_index_summary_store()
            summary_ref_id = store.store_graph_summary(
                summary_text=summary_text,
                graph_memory_id=node_id,
                shard_id=shard_id or "",
                full_path=node.metadata.get("full_path", node_id),
                source_node_ids=node.metadata.get("source_node_ids") or [node_id],
                topic=node.metadata.get("summary_topic", ""),
                importance=float(node.metadata.get("summary_importance", 0.5) or 0.5),
            )
            if summary_ref_id:
                node.metadata["summary_ref_id"] = summary_ref_id
        except Exception as exc:
            logger.debug(f"[ShardedMemoryGraph] 摘要导航写入跳过 node={node_id}: {exc}")
        self.update_node(node)
    
    def register_adapter(self, name: str, adapter) -> None:
        """注册适配器（兼容 MemoryGraph 接口）
        
        分片存储暂不需要适配器。
        """
        pass
    
    def compute_activations(
        self,
        seed_node_ids: List[str] = None,
        max_depth: int = 3,
        decay: float = 0.5,
        min_activation: float = 0.01,
        node_type_filter: Optional[Set[str]] = None,
    ) -> Dict[str, float]:
        """BFS 扩散激活 — 从种子节点沿图边传播激活值（兼容 MemoryGraph 接口）
        
        Args:
            seed_node_ids: 种子节点ID列表
            max_depth: 最大扩散深度
            decay: 衰减因子（每层乘以此值）
            min_activation: 最低激活阈值，低于此值的节点不返回
            node_type_filter: 可选的节点类型过滤
            
        Returns:
            {node_id: activation_score} dict
        """
        all_activations: Dict[str, float] = {}
        activated_edges: List[Tuple[str, str]] = []
        
        if not seed_node_ids:
            return {}
        
        try:
            # 1. 按分片分组种子节点
            shard_seeds: Dict[str, List[str]] = {}
            for node_id in seed_node_ids:
                _, shard_id = self._get_node_with_shard(node_id)
                if shard_id:
                    shard_seeds.setdefault(shard_id, []).append(node_id)
            
            if not shard_seeds:
                return {}
            
            # 2. 每个分片执行 bfs_spread_weighted
            for shard_id, seeds in shard_seeds.items():
                shard = self.get_shard(shard_id, load_if_missing=True)
                if not shard:
                    continue
                try:
                    results = shard.bfs_spread_weighted(
                        seeds, max_depth=max_depth, decay_factor=decay
                    )
                    # 合并分数：跨分片取最大值
                    for node_id, score in results:
                        if score < min_activation:
                            continue
                        if node_type_filter and hasattr(shard, 'get_node'):
                            node = shard.get_node(node_id)
                            if node and getattr(node, 'node_type', None) not in node_type_filter:
                                continue
                        all_activations[node_id] = max(all_activations.get(node_id, 0.0), score)
                except Exception as e:
                    logger.warning(f"[ShardedMemoryGraph] 分片 {shard_id} BFS 异常: {e}")
                    continue

            # 跨分片边由 global_index 的 cross_edges_by_src / cross_edges_by_dst
            # 提供一跳增量传播，避免为边界节点扫描全部冷分片。
            cross_updates: Dict[str, float] = {}
            for node_id, score in list(all_activations.items()):
                for edge in self._get_cross_edges_from(node_id):
                    target_id = edge.get("dst_id")
                    if not target_id:
                        continue
                    try:
                        weight = float(edge.get("weight", 1.0))
                    except (TypeError, ValueError):
                        weight = 1.0
                    new_score = score * decay * weight
                    if new_score < min_activation:
                        continue
                    if node_type_filter:
                        target_node, _ = self._get_node_with_shard(target_id)
                        if target_node and getattr(target_node, "node_type", None) not in node_type_filter:
                            continue
                    cross_updates[target_id] = max(cross_updates.get(target_id, 0.0), new_score)

            for node_id, score in cross_updates.items():
                all_activations[node_id] = max(all_activations.get(node_id, 0.0), score)
            
            # 3. 更新节点激活值到 LMDB
            updated_count = 0
            for node_id, score in all_activations.items():
                node, sid = self._get_node_with_shard(node_id)
                if node is None or sid is None:
                    continue
                try:
                    node.activation = score
                    node.last_accessed = time.time()
                    shard = self.get_shard(sid, load_if_missing=False)
                    if shard:
                        shard.update_node(node)
                        updated_count += 1
                except Exception:
                    continue
            
            # 4. 记录激活边（用于赫布学习）
            for shard_id in shard_seeds:
                shard = self.get_shard(shard_id, load_if_missing=True)
                if not shard:
                    continue
                for node_id in all_activations:
                    if node_id not in shard.topology:
                        continue
                    neighbors = shard.get_topology_neighbors(node_id, mode="out")
                    for neighbor_id in neighbors:
                        if neighbor_id in all_activations:
                            activated_edges.append((node_id, neighbor_id))
                    for edge in self._get_cross_edges_from(node_id):
                        neighbor_id = edge.get("dst_id")
                        if neighbor_id in all_activations:
                            activated_edges.append((node_id, neighbor_id))
            
            self._last_activated_edges = activated_edges
            self._active_node_ids = set(all_activations.keys())
            self.enqueue_active_skeleton_refresh(seed_node_ids)
            logger.info(
                f"[ShardedMemoryGraph] BFS扩散: seeds={len(seed_node_ids)}, "
                f"activated={len(all_activations)}, edges={len(activated_edges)}"
            )
            
            return all_activations
            
        except Exception as e:
            logger.warning(f"[ShardedMemoryGraph] compute_activations 异常: {e}")
            return {}
    
    def set_rag_manager(self, rag_manager):
        """注入 RAGManager 供 backend_ref 反查使用（兼容 MemoryGraph 接口）"""
        self._rag_manager = rag_manager

    def resolve_backend_ref(self, node_id: str) -> Optional[Dict[str, Any]]:
        """通过 backend_ref 反查后端完整数据（兼容 MemoryGraph 接口）。"""
        node = self.get_node(node_id)
        if not node:
            return None
        metadata = dict(getattr(node, "metadata", {}) or {})
        ref = getattr(node, "backend_ref", "") or metadata.get("backend_ref", "")
        if not ref:
            return None

        parts = str(ref).split(":", 1)
        if len(parts) != 2:
            return None
        backend_type, backend_id = parts[0], parts[1]

        if backend_type == "task_graph":
            return {"source": "task_graph", "ref": backend_id, "data": metadata}

        rag_mgr = getattr(self, "_rag_manager", None)
        if rag_mgr is None:
            return None

        lib_map = {
            "experience_rag": "experience",
            "knowledge": "knowledge",
            "memory": "memory",
        }
        library = lib_map.get(backend_type, backend_type)
        try:
            if hasattr(rag_mgr, "get_document"):
                doc = rag_mgr.get_document(library, backend_id)
                if doc:
                    return {
                        "source": backend_type,
                        "doc_id": backend_id,
                        "content": getattr(doc, "content", "") or getattr(doc, "text", "") or str(doc),
                        "metadata": getattr(doc, "metadata", {}) or {},
                    }
            if hasattr(rag_mgr, "get"):
                doc = rag_mgr.get(library, backend_id)
                if doc:
                    return {
                        "source": backend_type,
                        "doc_id": backend_id,
                        "content": getattr(doc, "content", "") or getattr(doc, "text", "") or str(doc),
                        "metadata": getattr(doc, "metadata", {}) or {},
                    }
        except Exception as exc:
            logger.debug(f"[ShardedMemoryGraph] resolve_backend_ref 失败 ({ref}): {exc}")
        return None

    def save_all(self):
        """保存所有活跃分片"""
        with self.shard_lock:
            for shard_id, shard in self.active_shards.items():
                shard.save()
                logger.info(f"分片已保存: {shard_id}")
                
        self._save_shard_index()
        self.global_index.sync()
        
    def close_all(self):
        """关闭所有分片"""
        with self.shard_lock:
            for shard_id, shard in self.active_shards.items():
                shard.close()
                logger.info(f"分片已关闭: {shard_id}")
                
            self.active_shards.clear()
            for shard_id, local_index in self.local_indexes.items():
                local_index.close()
                logger.debug(f"local_index 已关闭: {shard_id}")
            self.local_indexes.clear()
        self.global_index.close()
            
    def compact(self):
        """压缩所有分片（减少文件大小）"""
        logger.info("开始压缩所有分片...")
        
        for shard_id in self.list_all_shards():
            shard = self.get_shard(shard_id, load_if_missing=True)
            if shard:
                shard.properties.compact()
                
        logger.info("分片压缩完成")
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()
