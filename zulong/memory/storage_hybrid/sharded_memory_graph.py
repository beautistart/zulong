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
        enable_vector_index: bool = False
    ):
        """
        初始化分片管理器
        
        Args:
            base_dir: 数据基础目录
            shard_strategy: 分片策略（month/week/day）
            max_active_shards: 最大活跃分片数
            map_size_gb: 每个分片的LMDB映射大小（GB）
            enable_vector_index: 是否启用向量索引
        """
        os.makedirs(base_dir, exist_ok=True)
        
        self.base_dir = base_dir
        self.shard_strategy = shard_strategy
        self.max_active_shards = max_active_shards
        self.map_size_gb = map_size_gb
        self.enable_vector_index = enable_vector_index
        
        self.active_shards: OrderedDict[str, MemoryGraphHybrid] = OrderedDict()
        self.shard_lock = threading.RLock()
        
        self.shard_index = self._load_shard_index()
        
        self._stats = {
            "shard_load_count": 0,
            "shard_evict_count": 0,
            "cross_shard_query_count": 0,
        }
        
        logger.info(
            f"ShardedMemoryGraph 初始化完成: "
            f"strategy={shard_strategy}, "
            f"max_active={max_active_shards}"
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
        node_id: str,
        node_type: str,
        label: str,
        timestamp: Optional[float] = None,
        **kwargs
    ) -> bool:
        """
        添加节点到指定时间分片
        
        Args:
            node_id: 节点ID
            node_type: 节点类型
            label: 标签
            timestamp: 时间戳（默认当前时间）
            **kwargs: 其他参数传递给MemoryGraphHybrid.add_node
        """
        if timestamp is None:
            timestamp = time.time()
            
        shard_id = self._get_shard_id(timestamp)
        shard = self.get_shard(shard_id)
        
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
            self._save_shard_index()
            
        return success
        
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
        
    def discover_across_shards(
        self,
        seed_ids: List[str],
        seed_shard_id: str,
        max_depth: int = 3,
        max_nodes: int = 1000
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
                
            neighbors = shard.get_neighbors(current_id, mode="out")
            
            for neighbor_id in neighbors:
                if neighbor_id in visited:
                    continue
                    
                neighbor_shard_id = current_shard_id
                
                if neighbor_id not in shard.topology:
                    for other_shard_id in self.list_all_shards():
                        if other_shard_id == current_shard_id:
                            continue
                        other_shard = self.get_shard(other_shard_id, load_if_missing=False)
                        if other_shard and neighbor_id in other_shard.topology:
                            neighbor_shard_id = other_shard_id
                            break
                            
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
        
    def get_node(self, node_id: str) -> Tuple[Optional[NodeProperties], Optional[str]]:
        """
        获取节点属性（需要指定分片）
        
        Returns:
            (NodeProperties, shard_id) 或 (None, None)
        """
        for shard_id in self.list_all_shards():
            shard = self.get_shard(shard_id, load_if_missing=True)
            if shard and node_id in shard.topology:
                node = shard.get_node(node_id)
                return node, shard_id
                
        return None, None
        
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
        
    def migrate_from_networkx(
        self,
        networkx_graph,
        node_timestamp_field: str = "created_at"
    ):
        """
        从NetworkX图迁移数据
        
        Args:
            networkx_graph: NetworkX DiGraph对象
            node_timestamp_field: 节点属性中的时间戳字段名
        """
        import networkx as nx
        
        logger.info(f"开始从NetworkX迁移，节点数={networkx_graph.number_of_nodes()}")
        
        node_count = 0
        edge_count = 0
        
        for node_id, node_data in networkx_graph.nodes(data=True):
            timestamp = node_data.get(node_timestamp_field, time.time())
            
            node_type = node_data.get("node_type", node_data.get("type", "unknown"))
            label = node_data.get("label", str(node_id))
            
            success = self.add_node(
                node_id=str(node_id),
                node_type=node_type,
                label=label,
                timestamp=timestamp,
                content=node_data.get("content"),
                importance=node_data.get("importance", "normal"),
                backend_ref=node_data.get("backend_ref", ""),
                metadata=node_data
            )
            
            if success:
                node_count += 1
                
        for src_id, dst_id, edge_data in networkx_graph.edges(data=True):
            timestamp = edge_data.get("created_at", time.time())
            
            edge_type = edge_data.get("edge_type", edge_data.get("type", "association"))
            weight = edge_data.get("weight", 1.0)
            
            success = self.add_edge(
                src_id=str(src_id),
                dst_id=str(dst_id),
                edge_type=edge_type,
                timestamp=timestamp,
                weight=weight,
                protected=edge_data.get("protected", False),
                metadata=edge_data
            )
            
            if success:
                edge_count += 1
                
        self.save_all()
        
        logger.info(
            f"NetworkX迁移完成: "
            f"节点={node_count}, 边={edge_count}"
        )
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()
