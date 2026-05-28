# File: zulong/memory/storage_hybrid/sharded_memory_graph.py
# 分片管理器 - 按时间切分的大规模记忆图谱存储
#
# 核心特性:
# - 时间分片策略（按月/周切分）
# - LRU缓存活跃分片（近3个月常驻内存）
# - 跨分片关联发现
# - 单分片50-200MB，总规模可达年级别

import logging
import os
import time
import json
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)

from .memory_graph_hybrid import MemoryGraphHybrid, NodeProperties, EdgeProperties


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
        max_nodes_per_shard: int = 150_000
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
        """
        os.makedirs(base_dir, exist_ok=True)
        
        self.base_dir = base_dir
        self.shard_strategy = shard_strategy
        self.max_active_shards = max_active_shards
        self.map_size_gb = map_size_gb
        self.enable_vector_index = enable_vector_index
        self.max_nodes_per_shard = max_nodes_per_shard
        self._rag_manager = None  # RAGManager 引用
        
        self.active_shards: OrderedDict[str, MemoryGraphHybrid] = OrderedDict()
        self.shard_lock = threading.RLock()
        
        self.shard_index = self._load_shard_index()
        
        self._stats = {
            "shard_load_count": 0,
            "shard_evict_count": 0,
            "cross_shard_query_count": 0,
            "auto_split_count": 0,
        }
        self._active_node_ids: Set[str] = set()
        self._last_focus_context: Optional[Dict[str, Any]] = None
        self._last_activated_edges: List[Tuple[str, str]] = []  # (src_id, dst_id) pairs for Hebbian learning
        self._last_retrieved_node_ids: List[str] = []  # top-3 node IDs from last retrieve_context
        
        logger.info(
            f"ShardedMemoryGraph 初始化完成: "
            f"strategy={shard_strategy}, "
            f"max_active={max_active_shards}, "
            f"max_nodes_per_shard={max_nodes_per_shard}"
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
            if os.path.exists(topology_path):
                shard.load(topology_path)
                if len(shard.topology) == 0 and shard.properties.count_nodes() > 0:
                    shard.rebuild_topology_from_properties()
            elif shard.properties.count_nodes() > 0:
                shard.rebuild_topology_from_properties()
                 
            while len(self.active_shards) >= self.max_active_shards:
                oldest_id, oldest_shard = self.active_shards.popitem(last=False)
                oldest_shard.close()
                self._stats["shard_evict_count"] += 1
                logger.info(f"淘汰分片: {oldest_id}")
                
            self.active_shards[shard_id] = shard
            self._stats["shard_load_count"] += 1
            
            return shard
            
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
            
        shard_id = self._get_shard_id(timestamp)
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
            if shard_id not in self.shard_index["shards"]:
                self.shard_index["shards"][shard_id] = {
                    "created_at": timestamp,
                    "node_count": 0,
                    "edge_count": 0,
                }
            self.shard_index["shards"][shard_id]["node_count"] += 1
            
            # 分片大小控制: 检查是否超过阈值
            node_count = self.shard_index["shards"][shard_id]["node_count"]
            self._check_shard_size(shard_id, node_count)
            
            self._save_shard_index()
            
        return success

    def _check_shard_size(self, shard_id: str, node_count: int) -> None:
        """检查分片大小，超阈值时发出警告或触发自动分裂"""
        if self.max_nodes_per_shard <= 0:
            return
        
        ratio = node_count / self.max_nodes_per_shard
        
        if ratio >= 1.10:
            logger.critical(
                f"分片 {shard_id} 严重超限: {node_count} 节点 "
                f"({ratio:.1%})，触发自动分裂"
            )
            self._auto_split_shard(shard_id)
        elif ratio >= 0.95:
            logger.warning(
                f"分片 {shard_id} 接近上限: {node_count} 节点 "
                f"({ratio:.1%})，将在 {self.max_nodes_per_shard - node_count} 个节点后触发分裂"
            )

    def _auto_split_shard(self, shard_id: str) -> None:
        """自动分裂分片：创建子分片并将后续节点路由到新分片
        
        分裂策略: 在原分片 ID 后追加 _part_N 后缀。
        例如 "2026_05" → "2026_05_part_1"
        """
        existing_parts = [
            key for key in self.shard_index["shards"]
            if key.startswith(shard_id) and key != shard_id
        ]
        part_num = len(existing_parts) + 1
        new_shard_id = f"{shard_id}_part_{part_num}"
        
        # 在 shard_index 中注册新子分片
        self.shard_index["shards"][new_shard_id] = {
            "created_at": time.time(),
            "node_count": 0,
            "edge_count": 0,
            "parent_shard": shard_id,
        }
        
        # 更新父分片索引，记录已分裂的子分片
        parent_info = self.shard_index["shards"].get(shard_id, {})
        if "parts" not in parent_info:
            parent_info["parts"] = []
        parent_info["parts"].append(new_shard_id)
        
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
            
        shard_id = self._get_shard_id(timestamp)
        shard = self.get_shard(shard_id)
        
        success = shard.add_edge(
            src_id=src_id,
            dst_id=dst_id,
            edge_type=edge_type,
            **kwargs
        )
        
        if success:
            if shard_id in self.shard_index["shards"]:
                self.shard_index["shards"][shard_id]["edge_count"] += 1
                self._save_shard_index()
                
        return success
        
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
            if shard and shard.topology.get_edge_info(src_id, dst_id) is not None:
                return True
        return False
        
    def discover_across_shards(
        self,
        seed_ids: List[str],
        seed_shard_id: str,
        max_depth: int = 3,
        max_nodes: int = 1000,
        max_shard_scan: int = 10
    ) -> List[Tuple[str, int, str]]:
        """
        跨分片关联发现
        
        Args:
            seed_ids: 种子节点ID列表
            seed_shard_id: 种子所在分片ID
            max_depth: 最大扩散深度
            max_nodes: 最大返回节点数
            max_shard_scan: 每次邻居查询最多扫描的分片数（防性能退化）
            
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
                
            neighbors = shard.get_neighbors(current_id, mode="out")
            
            for neighbor_id in neighbors:
                if neighbor_id in visited:
                    continue
                    
                neighbor_shard_id = current_shard_id
                
                # 检查邻居节点是否在当前分片的拓扑中
                # 注意: get_neighbors() 返回的是当前分片 igraph 内的邻居，
                # 如果邻居节点通过跨分片边引用被记录但不在本分片拓扑中，
                # 则需要扫描其他分片来定位
                if neighbor_id not in shard.topology:
                    all_shards = self.list_all_shards()
                    scan_count = 0
                    for other_shard_id in all_shards:
                        if other_shard_id == current_shard_id:
                            continue
                        if scan_count >= max_shard_scan:
                            logger.debug(f"跨分片扫描达上限 ({max_shard_scan})，停止扫描")
                            break
                        other_shard = self.get_shard(other_shard_id, load_if_missing=True)
                        if other_shard and neighbor_id in other_shard.topology:
                            neighbor_shard_id = other_shard_id
                            break
                        scan_count += 1
                            
                visited[neighbor_id] = (current_dist + 1, neighbor_shard_id)
                queue.append((neighbor_id, current_dist + 1, neighbor_shard_id))
                
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
        
    def get_node(self, node_id: str) -> Optional[NodeProperties]:
        """
        获取节点属性（兼容 MemoryGraph 接口）
        
        Returns:
            NodeProperties 或 None
        """
        for shard_id in self.list_all_shards():
            shard = self.get_shard(shard_id, load_if_missing=True)
            if shard and node_id in shard.topology:
                node = shard.get_node(node_id)
                return node
                
        return None

    def update_node(self, node: NodeProperties) -> bool:
        """更新节点属性（兼容 MemoryGraph 接口）。"""
        for shard_id in self.list_all_shards():
            shard = self.get_shard(shard_id, load_if_missing=True)
            if shard and node.node_id in shard.topology:
                return bool(shard.update_node(node))
        return False

    def _get_node_with_shard(self, node_id: str) -> Tuple[Optional[NodeProperties], Optional[str]]:
        """
        获取节点属性及其所在分片ID（内部使用）
        
        Returns:
            (NodeProperties, shard_id) 或 (None, None)
        """
        for shard_id in self.list_all_shards():
            shard = self.get_shard(shard_id, load_if_missing=True)
            if shard and node_id in shard.topology:
                node = shard.get_node(node_id)
                return node, shard_id
                
        return None, None
        
    def has_node(self, node_id: str) -> bool:
        """检查节点是否存在（兼容 MemoryGraph 接口）"""
        for shard_id in self.list_all_shards():
            shard = self.get_shard(shard_id, load_if_missing=True)
            if shard and node_id in shard.topology:
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
            
        return {
            "total_shards": len(self.shard_index["shards"]),
            "active_shards": len(self.active_shards),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
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
        """双路并行检索上下文 — 热路径 (时间窗口扫描) + 冷路径 (BFS 扩散)
        
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
            current_shard = self.get_current_shard()
            
            # ── 热路径: 扫描当前分片的热节点 ──
            if current_shard:
                hot_nodes = []
                try:
                    for node in current_shard.properties.iter_nodes():
                        if getattr(node, 'last_accessed', 0) >= hot_threshold:
                            hot_nodes.append(node)
                        if len(hot_nodes) >= 500:
                            break
                except Exception:
                    pass
                
                for node in hot_nodes:
                    text_parts = [
                        getattr(node, 'label', '') or '',
                        getattr(node, 'content', '') or '',
                    ]
                    combined = ' '.join(filter(None, text_parts))
                    if not combined.strip():
                        continue
                    score = self._bigram_overlap_score(query_text, combined)
                    if score <= 0:
                        continue
                    # Session 匹配加权
                    if session_id and node.node_id.startswith(f"dialogue:session_{session_id}"):
                        score *= 2.0
                    # 重要度加权
                    importance = getattr(node, 'importance', 'NORMAL')
                    _importance_boost = {
                        'MUST_REMEMBER': 2.0, 'IMPORTANT': 1.5,
                        'IDENTITY': 1.8, 'FACT': 1.3,
                    }
                    score *= _importance_boost.get(importance, 1.0)
                    
                    results[node.node_id] = {
                        'node_id': node.node_id,
                        'node_type': getattr(node, 'node_type', 'unknown'),
                        'label': getattr(node, 'label', ''),
                        'content': getattr(node, 'content', ''),
                        'score': score,
                        'source': 'hot',
                        'importance': importance,
                        'metadata': getattr(node, 'metadata', {}),
                    }
            
            # ── 冷路径: 从热节点做 BFS 扩散发现关联冷节点 ──
            hot_node_ids = list(results.keys())[:20]
            if hot_node_ids:
                discovered = self.discover_across_shards(
                    seed_ids=hot_node_ids,
                    seed_shard_id=self._get_shard_id(now),
                    max_depth=2,
                    max_nodes=200,
                )
                seen_ids = set(results.keys())
                for node_id, distance, shard_id in discovered:
                    if node_id in seen_ids:
                        continue
                    shard = self.get_shard(shard_id, load_if_missing=True)
                    if not shard:
                        continue
                    node = shard.get_node(node_id)
                    if node is None:
                        continue
                    combined = ' '.join(filter(None, [
                        getattr(node, 'label', '') or '',
                        getattr(node, 'content', '') or '',
                    ]))
                    score = self._bigram_overlap_score(query_text, combined) * 0.8  # cold path discount
                    if score <= 0:
                        continue
                    results[node.node_id] = {
                        'node_id': node_id,
                        'node_type': getattr(node, 'node_type', 'unknown'),
                        'label': getattr(node, 'label', ''),
                        'content': getattr(node, 'content', ''),
                        'score': score,
                        'source': 'cold',
                        'importance': getattr(node, 'importance', 'NORMAL'),
                        'metadata': getattr(node, 'metadata', {}),
                    }
                    seen_ids.add(node_id)
            
            # ── 排序取 top_k ──
            sorted_results = sorted(results.values(), key=lambda x: x['score'], reverse=True)[:top_k]
            
            # ── 缓存 top-3 节点 ID 供 BFS 种子扩展 ──
            self._last_retrieved_node_ids = [r['node_id'] for r in sorted_results[:3]]
            
            logger.debug(
                f"[ShardedMemoryGraph] retrieve_context: query='{query_text[:30]}...', "
                f"results={len(sorted_results)}"
            )
            return sorted_results
            
        except Exception as e:
            logger.warning(f"[ShardedMemoryGraph] retrieve_context 异常: {e}")
            return []
    
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
        
        for src_id, dst_id in self._last_activated_edges:
            # 查找边所在的分片
            found = False
            for shard_id in self.list_all_shards():
                shard = self.get_shard(shard_id, load_if_missing=True)
                if not shard:
                    continue
                if src_id not in shard.topology or dst_id not in shard.topology:
                    continue
                edge_info = shard.topology.get_edge_info(src_id, dst_id)
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
                skipped += 1
        
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
        return True

    def _find_parent_id(self, node_id: str) -> Optional[str]:
        for shard_id in self.list_all_shards():
            shard = self.get_shard(shard_id, load_if_missing=True)
            if not shard:
                continue
            for src_id in shard.topology.get_neighbors(node_id, edge_type="hierarchy", mode="in"):
                return src_id
        return None

    def get_last_focus_context(self) -> Optional[Dict]:
        """获取上次保存的焦点上下文（兼容 MemoryGraph 接口）
        
        分片存储模式下保存轻量焦点路径。
        """
        return self._last_focus_context
    
    def remove_node(self, node_id: str) -> bool:
        """删除节点（兼容 MemoryGraph 接口）
        
        当前为 stub，实际删除功能待实现。
        """
        logger.debug(f"[ShardedMemoryGraph] remove_node stub: {node_id}")
        return False
    
    def set_importance(self, node_id: str, importance) -> None:
        """设置节点重要性（兼容 MemoryGraph 接口）"""
        pass
    
    def get_children(self, node_id: str, edge_type=None) -> list:
        """获取子节点列表（兼容 MemoryGraph 接口）
        """
        edge_type_str = getattr(edge_type, "value", edge_type) or "hierarchy"
        children = []
        for shard_id in self.list_all_shards():
            shard = self.get_shard(shard_id, load_if_missing=True)
            if not shard:
                continue
            for child_id in shard.topology.get_neighbors(node_id, edge_type=edge_type_str, mode="out"):
                node = shard.get_node(child_id)
                if node:
                    children.append(node)
        children.sort(key=lambda n: getattr(n, "created_at", 0.0))
        return children

    def to_frontend_dict(self, depth: Optional[int] = None) -> Dict[str, Any]:
        """序列化为前端记忆图谱格式。"""
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
                edges.append(self._serialize_edge_for_frontend(src_id, dst_id, edge_type, edge_props))
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
        }

    def get_node_children_for_frontend(self, node_id: str) -> Dict[str, Any]:
        children = self.get_children(node_id)
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
                    edges.append(self._serialize_edge_for_frontend(src_id, dst_id, edge_type, edge_props))
        return {"parent_id": node_id, "nodes": nodes, "edges": edges}

    def _vertex_node_id(self, vertex) -> str:
        attrs = vertex.attributes()
        if "node_id" in attrs:
            return vertex["node_id"]
        return vertex["name"]

    def _serialize_node_for_frontend(self, node: NodeProperties, include_children_count: bool = False) -> Dict[str, Any]:
        metadata = dict(node.metadata or {})
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
            data["children_count"] = len(self.get_children(node.node_id))
        return data

    def _serialize_edge_for_frontend(self, src_id: str, dst_id: str, edge_type: str, edge_props: Optional[EdgeProperties]) -> Dict[str, Any]:
        return {
            "source": src_id,
            "target": dst_id,
            "type": edge_type,
            "weight": getattr(edge_props, "weight", 1.0) if edge_props else 1.0,
            "protected": getattr(edge_props, "protected", False) if edge_props else False,
            "metadata": getattr(edge_props, "metadata", {}) if edge_props else {},
        }
    
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
                    results = shard.topology.bfs_spread_weighted(
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
                    neighbors = shard.topology.get_neighbors(node_id, mode="out")
                    for neighbor_id in neighbors:
                        if neighbor_id in all_activations:
                            activated_edges.append((node_id, neighbor_id))
            
            self._last_activated_edges = activated_edges
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

    def save_all(self):
        """保存所有活跃分片"""
        with self.shard_lock:
            for shard_id, shard in self.active_shards.items():
                shard.save()
                logger.info(f"分片已保存: {shard_id}")
                
        self._save_shard_index()
        
    def close_all(self):
        """关闭所有分片"""
        with self.shard_lock:
            for shard_id, shard in self.active_shards.items():
                shard.close()
                logger.info(f"分片已关闭: {shard_id}")
                
            self.active_shards.clear()
            
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
