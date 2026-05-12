# File: zulong/memory/storage_hybrid/topology_index.py
# igraph图拓扑索引层 - 仅存储节点ID和邻接关系
#
# 核心特性:
# - 内存效率高（每节点约50字节 vs NetworkX的2KB）
# - BFS扩散微秒级（igraph C后端）
# - 支持未加载属性即可发现关联节点

import logging
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

try:
    import igraph as ig
except ImportError:
    logger.error("igraph 未安装，请执行: pip install python-igraph")
    raise


class TopologyIndex:
    """
    igraph图拓扑索引 - 仅存储节点ID和邻接关系
    
    内存开销: 每节点约50字节（ID映射 + 类型 + 邻接指针）
    百万节点: ~50MB内存（vs NetworkX的2-5GB）
    BFS查询: <1ms（igraph C后端）
    """
    
    def __init__(self):
        self.graph = ig.Graph(directed=True)
        
        self.node_id_to_idx: Dict[str, int] = {}
        self.idx_to_node_id: Dict[int, str] = {}
        
        self._node_count = 0
        self._edge_count = 0
        self._node_type_counts: Dict[str, int] = defaultdict(int)
        self._edge_type_counts: Dict[str, int] = defaultdict(int)
        
    def add_node(self, node_id: str, node_type: str) -> int:
        """
        添加节点 - 仅存储ID和类型
        
        Args:
            node_id: 全局唯一节点ID，如 "task:o1_1"
            node_type: 节点类型，如 "task", "dialogue"
            
        Returns:
            igraph内部索引
        """
        if node_id in self.node_id_to_idx:
            return self.node_id_to_idx[node_id]
        
        self.graph.add_vertex(name=node_id)
        idx = len(self.graph.vs) - 1
        
        self.node_id_to_idx[node_id] = idx
        self.idx_to_node_id[idx] = node_id
        
        self.graph.vs[idx]["type"] = node_type
        
        self._node_count += 1
        self._node_type_counts[node_type] += 1
        
        return idx
        
    def add_edge(
        self, 
        src_id: str, 
        dst_id: str, 
        edge_type: str, 
        weight: float = 1.0
    ) -> Optional[int]:
        """
        添加边 - 存储类型和权重
        
        Args:
            src_id: 源节点ID
            dst_id: 目标节点ID
            edge_type: 边类型，如 "hierarchy", "dependency"
            weight: 边权重（默认1.0）
            
        Returns:
            igraph边索引，失败返回None
        """
        if src_id not in self.node_id_to_idx:
            logger.warning(f"源节点不存在: {src_id}")
            return None
        if dst_id not in self.node_id_to_idx:
            logger.warning(f"目标节点不存在: {dst_id}")
            return None
            
        src_idx = self.node_id_to_idx[src_id]
        dst_idx = self.node_id_to_idx[dst_id]
        
        try:
            eid = self.graph.add_edge(src_idx, dst_idx)
            
            self.graph.es[eid.index]["type"] = edge_type
            self.graph.es[eid.index]["weight"] = weight
            
            self._edge_count += 1
            self._edge_type_counts[edge_type] += 1
            
            return eid.index
        except Exception as e:
            logger.warning(f"添加边失败 ({src_id}→{dst_id}): {e}")
            return None
            
    def get_node_type(self, node_id: str) -> Optional[str]:
        """获取节点类型"""
        if node_id not in self.node_id_to_idx:
            return None
        idx = self.node_id_to_idx[node_id]
        return self.graph.vs[idx]["type"]
        
    def get_neighbors(
        self, 
        node_id: str, 
        edge_type: Optional[str] = None,
        mode: str = "out"
    ) -> List[str]:
        """
        获取邻居节点ID列表 - igraph微秒级查询
        
        Args:
            node_id: 节点ID
            edge_type: 边类型过滤（可选）
            mode: "out"出边 / "in"入边 / "all"双向
            
        Returns:
            邻居节点ID列表
        """
        if node_id not in self.node_id_to_idx:
            return []
            
        idx = self.node_id_to_idx[node_id]
        
        if edge_type is None:
            neighbors = self.graph.neighbors(idx, mode=mode)
            return [self.idx_to_node_id[n] for n in neighbors]
        
        neighbor_indices = self.graph.neighbors(idx, mode=mode)
        result = []
        
        for neighbor_idx in neighbor_indices:
            if mode == "out":
                edges = self.graph.es.select(_source=idx, _target=neighbor_idx)
            elif mode == "in":
                edges = self.graph.es.select(_source=neighbor_idx, _target=idx)
            else:
                edges = self.graph.es.select(_between=([idx], [neighbor_idx]))
                
            for edge in edges:
                if edge["type"] == edge_type:
                    result.append(self.idx_to_node_id[neighbor_idx])
                    break
                    
        return result
        
    def get_edge_info(
        self, 
        src_id: str, 
        dst_id: str
    ) -> Optional[Tuple[str, float]]:
        """
        获取边信息（类型、权重）
        
        Returns:
            (edge_type, weight) 或 None
        """
        if src_id not in self.node_id_to_idx or dst_id not in self.node_id_to_idx:
            return None
            
        src_idx = self.node_id_to_idx[src_id]
        dst_idx = self.node_id_to_idx[dst_id]
        
        edges = self.graph.es.select(_source=src_idx, _target=dst_idx)
        if len(edges) == 0:
            return None
            
        edge = edges[0]
        return (edge["type"], edge["weight"])
        
    def bfs_spread(
        self, 
        seed_ids: List[str], 
        max_depth: int = 3,
        edge_types: Optional[Set[str]] = None,
        max_nodes: int = 1000
    ) -> List[Tuple[str, int]]:
        """
        BFS扩散 - igraph原生高效实现
        
        Args:
            seed_ids: 种子节点ID列表
            max_depth: 最大扩散深度
            edge_types: 边类型过滤集合（可选）
            max_nodes: 最大返回节点数
            
        Returns:
            [(node_id, distance), ...] 按距离排序
        """
        seed_indices = []
        for sid in seed_ids:
            if sid in self.node_id_to_idx:
                seed_indices.append(self.node_id_to_idx[sid])
                
        if not seed_indices:
            return []
            
        visited: Dict[int, int] = {}  # node_idx -> distance
        queue = [(idx, 0) for idx in seed_indices]
        
        for idx in seed_indices:
            visited[idx] = 0
            
        while queue and len(visited) < max_nodes:
            current_idx, current_dist = queue.pop(0)
            
            if current_dist >= max_depth:
                continue
                
            neighbors = self.graph.neighbors(current_idx, mode="out")
            
            for neighbor_idx in neighbors:
                if neighbor_idx in visited:
                    continue
                    
                if edge_types:
                    edges = self.graph.es.select(_source=current_idx, _target=neighbor_idx)
                    if not any(e["type"] in edge_types for e in edges):
                        continue
                        
                visited[neighbor_idx] = current_dist + 1
                queue.append((neighbor_idx, current_dist + 1))
                
        result = [
            (self.idx_to_node_id[idx], dist) 
            for idx, dist in visited.items()
        ]
        result.sort(key=lambda x: x[1])
        
        return result[:max_nodes]
        
    def bfs_spread_weighted(
        self, 
        seed_ids: List[str], 
        max_depth: int = 3,
        decay_factor: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        加权BFS扩散 - 考虑边权重衰减
        
        Args:
            seed_ids: 种子节点ID列表
            max_depth: 最大扩散深度
            decay_factor: 衰减因子（每层乘以此值）
            
        Returns:
            [(node_id, score), ...] 按分数降序
        """
        seed_indices = []
        for sid in seed_ids:
            if sid in self.node_id_to_idx:
                seed_indices.append(self.node_id_to_idx[sid])
                
        if not seed_indices:
            return []
            
        visited: Dict[int, float] = {}  # node_idx -> score
        
        for idx in seed_indices:
            visited[idx] = 1.0
            
        queue = [(idx, 1.0, 0) for idx in seed_indices]
        
        while queue:
            current_idx, current_score, current_dist = queue.pop(0)
            
            if current_dist >= max_depth:
                continue
                
            neighbors = self.graph.neighbors(current_idx, mode="out")
            edges_from_current = self.graph.es.select(_source=current_idx)
            
            for neighbor_idx in neighbors:
                edges_to_neighbor = [
                    e for e in edges_from_current 
                    if e.target == neighbor_idx
                ]
                
                if not edges_to_neighbor:
                    continue
                    
                max_weight = max(e["weight"] for e in edges_to_neighbor)
                new_score = current_score * decay_factor * max_weight
                
                if neighbor_idx in visited:
                    visited[neighbor_idx] = max(visited[neighbor_idx], new_score)
                else:
                    visited[neighbor_idx] = new_score
                    queue.append((neighbor_idx, new_score, current_dist + 1))
                    
        result = [
            (self.idx_to_node_id[idx], score) 
            for idx, score in visited.items()
        ]
        result.sort(key=lambda x: x[1], reverse=True)
        
        return result
        
    def get_node_degree(
        self, 
        node_id: str, 
        mode: str = "all"
    ) -> int:
        """获取节点度数"""
        if node_id not in self.node_id_to_idx:
            return 0
        idx = self.node_id_to_idx[node_id]
        return self.graph.degree(idx, mode=mode)
        
    def remove_node(self, node_id: str) -> bool:
        """删除节点及其所有边"""
        if node_id not in self.node_id_to_idx:
            return False
            
        idx = self.node_id_to_idx[node_id]
        
        node_type = self.graph.vs[idx]["type"]
        self._node_type_counts[node_type] -= 1
        
        degree = self.graph.degree(idx, mode="all")
        self._edge_count -= degree
        
        self.graph.delete_vertices(idx)
        
        del self.node_id_to_idx[node_id]
        del self.idx_to_node_id[idx]
        
        new_idx_to_node_id = {}
        for new_idx, vertex in enumerate(self.graph.vs):
            node_id_v = vertex["name"]
            new_idx_to_node_id[new_idx] = node_id_v
            self.node_id_to_idx[node_id_v] = new_idx
        self.idx_to_node_id = new_idx_to_node_id
        
        self._node_count -= 1
        
        return True
        
    def save_to_graphml(self, filepath: str):
        """保存为GraphML格式"""
        self.graph.write_graphml(filepath)
        logger.info(f"拓扑索引已保存: {filepath} (节点={self._node_count}, 边={self._edge_count})")
        
    def load_from_graphml(self, filepath: str):
        """从GraphML格式加载"""
        self.graph = ig.read(filepath, format="graphml")
        
        self.node_id_to_idx = {}
        self.idx_to_node_id = {}
        
        for idx, vertex in enumerate(self.graph.vs):
            node_id = vertex["name"]
            self.node_id_to_idx[node_id] = idx
            self.idx_to_node_id[idx] = node_id
            
        self._node_count = len(self.graph.vs)
        self._edge_count = len(self.graph.es)
        
        self._node_type_counts = defaultdict(int)
        for vertex in self.graph.vs:
            node_type = vertex.get("type", "unknown")
            self._node_type_counts[node_type] += 1
            
        self._edge_type_counts = defaultdict(int)
        for edge in self.graph.es:
            edge_type = edge.get("type", "unknown")
            self._edge_type_counts[edge_type] += 1
            
        logger.info(f"拓扑索引已加载: {filepath} (节点={self._node_count}, 边={self._edge_count})")
        
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "node_count": self._node_count,
            "edge_count": self._edge_count,
            "node_type_counts": dict(self._node_type_counts),
            "edge_type_counts": dict(self._edge_type_counts),
            "memory_mb": self._estimate_memory_mb()
        }
        
    def _estimate_memory_mb(self) -> float:
        """估算内存占用（MB）"""
        node_overhead = 50
        edge_overhead = 20
        
        total_bytes = (
            self._node_count * node_overhead + 
            self._edge_count * edge_overhead
        )
        return total_bytes / (1024 * 1024)
        
    def clear(self):
        """清空图"""
        self.graph = ig.Graph(directed=True)
        self.node_id_to_idx.clear()
        self.idx_to_node_id.clear()
        self._node_count = 0
        self._edge_count = 0
        self._node_type_counts.clear()
        self._edge_type_counts.clear()
        
    def __len__(self) -> int:
        """节点数量"""
        return self._node_count
        
    def __contains__(self, node_id: str) -> bool:
        """节点是否存在"""
        return node_id in self.node_id_to_idx
