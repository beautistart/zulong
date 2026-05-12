"""
图谱变更检测器

检测图谱节点的增删改，生成增量推送数据
- 维护图谱状态快照
- 对比检测节点增删改
- 生成增量推送消息
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import hashlib
import json


@dataclass
class NodeSnapshot:
    """节点快照"""
    node_id: str
    label: str
    node_type: str
    status: str
    desc: str
    parent_id: Optional[str]
    timestamp: float
    checksum: str

    @classmethod
    def from_node_data(cls, node_id: str, data: Dict[str, Any]) -> "NodeSnapshot":
        """从节点数据创建快照"""
        relevant_data = {
            "label": data.get("label", ""),
            "type": data.get("type", "task"),
            "status": data.get("status", "pending"),
            "desc": data.get("desc", ""),
            "parent_id": data.get("parent_id"),
        }
        checksum = hashlib.md5(
            json.dumps(relevant_data, sort_keys=True).encode()
        ).hexdigest()

        return cls(
            node_id=node_id,
            label=relevant_data["label"],
            node_type=relevant_data["type"],
            status=relevant_data["status"],
            desc=relevant_data["desc"],
            parent_id=relevant_data["parent_id"],
            timestamp=time.time(),
            checksum=checksum,
        )


@dataclass
class GraphDelta:
    """图谱增量数据"""
    added_nodes: List[Dict[str, Any]] = field(default_factory=list)
    updated_nodes: List[Dict[str, Any]] = field(default_factory=list)
    deleted_nodes: List[str] = field(default_factory=list)
    added_edges: List[Dict[str, Any]] = field(default_factory=list)
    deleted_edges: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def is_empty(self) -> bool:
        """检查是否为空增量"""
        return (
            len(self.added_nodes) == 0
            and len(self.updated_nodes) == 0
            and len(self.deleted_nodes) == 0
            and len(self.added_edges) == 0
            and len(self.deleted_edges) == 0
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "added_nodes": self.added_nodes,
            "updated_nodes": self.updated_nodes,
            "deleted_nodes": self.deleted_nodes,
            "added_edges": self.added_edges,
            "deleted_edges": self.deleted_edges,
            "timestamp": self.timestamp,
        }


class GraphChangeDetector:
    """图谱变更检测器"""

    def __init__(self):
        self.node_snapshots: Dict[str, NodeSnapshot] = {}
        self.edge_set: Set[Tuple[str, str, str]] = set()  # (source, target, type)
        self.last_sync_time: float = 0

    def initialize_snapshot(self, nodes: Dict[str, Dict[str, Any]], edges: List[Dict[str, Any]]):
        """初始化图谱快照"""
        self.node_snapshots.clear()
        self.edge_set.clear()

        for node_id, node_data in nodes.items():
            self.node_snapshots[node_id] = NodeSnapshot.from_node_data(node_id, node_data)

        for edge in edges:
            edge_tuple = (edge.get("source", ""), edge.get("target", ""), edge.get("type", "dependency"))
            self.edge_set.add(edge_tuple)

        self.last_sync_time = time.time()
        print(f"[GraphChangeDetector] Initialized with {len(self.node_snapshots)} nodes, {len(self.edge_set)} edges")

    def detect_changes(
        self,
        current_nodes: Dict[str, Dict[str, Any]],
        current_edges: List[Dict[str, Any]],
    ) -> GraphDelta:
        """检测图谱变更"""
        delta = GraphDelta()

        current_node_ids = set(current_nodes.keys())
        snapshot_node_ids = set(self.node_snapshots.keys())

        added_node_ids = current_node_ids - snapshot_node_ids
        for node_id in added_node_ids:
            delta.added_nodes.append({
                "id": node_id,
                **current_nodes[node_id],
            })

        deleted_node_ids = snapshot_node_ids - current_node_ids
        delta.deleted_nodes = list(deleted_node_ids)

        common_node_ids = current_node_ids & snapshot_node_ids
        for node_id in common_node_ids:
            current_snapshot = NodeSnapshot.from_node_data(node_id, current_nodes[node_id])
            old_snapshot = self.node_snapshots[node_id]

            if current_snapshot.checksum != old_snapshot.checksum:
                delta.updated_nodes.append({
                    "id": node_id,
                    **current_nodes[node_id],
                })

        current_edge_set = set()
        for edge in current_edges:
            edge_tuple = (edge.get("source", ""), edge.get("target", ""), edge.get("type", "dependency"))
            current_edge_set.add(edge_tuple)

        added_edges = current_edge_set - self.edge_set
        for source, target, edge_type in added_edges:
            delta.added_edges.append({
                "source": source,
                "target": target,
                "type": edge_type,
            })

        deleted_edges = self.edge_set - current_edge_set
        delta.deleted_edges = [f"{s}_{t}_{e}" for s, t, e in deleted_edges]

        return delta

    def apply_changes(self, delta: GraphDelta):
        """应用变更到快照"""
        for node_id in delta.deleted_nodes:
            self.node_snapshots.pop(node_id, None)

        for node_data in delta.added_nodes:
            node_id = node_data.get("id", "")
            if node_id:
                self.node_snapshots[node_id] = NodeSnapshot.from_node_data(node_id, node_data)

        for node_data in delta.updated_nodes:
            node_id = node_data.get("id", "")
            if node_id:
                self.node_snapshots[node_id] = NodeSnapshot.from_node_data(node_id, node_data)

        for edge_data in delta.added_edges:
            edge_tuple = (
                edge_data.get("source", ""),
                edge_data.get("target", ""),
                edge_data.get("type", "dependency"),
            )
            self.edge_set.add(edge_tuple)

        for edge_id in delta.deleted_edges:
            parts = edge_id.split("_")
            if len(parts) >= 3:
                edge_tuple = (parts[0], parts[1], parts[2])
                self.edge_set.discard(edge_tuple)

        self.last_sync_time = time.time()

    def create_incremental_message(self, delta: GraphDelta) -> Dict[str, Any]:
        """创建增量推送消息"""
        return {
            "type": "graph_delta",
            "delta": delta.to_dict(),
            "timestamp": time.time(),
        }

    def get_snapshot_stats(self) -> Dict[str, Any]:
        """获取快照统计信息"""
        return {
            "node_count": len(self.node_snapshots),
            "edge_count": len(self.edge_set),
            "last_sync_time": self.last_sync_time,
        }

    def clear(self):
        """清空快照"""
        self.node_snapshots.clear()
        self.edge_set.clear()
        self.last_sync_time = 0
        print("[GraphChangeDetector] Cleared")


class IncrementalPushManager:
    """增量推送管理器"""

    def __init__(self):
        self.detector = GraphChangeDetector()
        self.pending_deltas: List[GraphDelta] = []
        self.push_interval: float = 0.2  # 200ms
        self.last_push_time: float = 0
        self.enabled: bool = True

    def enable(self):
        """启用增量推送"""
        self.enabled = True
        print("[IncrementalPushManager] Enabled")

    def disable(self):
        """禁用增量推送"""
        self.enabled = False
        print("[IncrementalPushManager] Disabled")

    def initialize(self, nodes: Dict[str, Dict[str, Any]], edges: List[Dict[str, Any]]):
        """初始化"""
        self.detector.initialize_snapshot(nodes, edges)
        self.pending_deltas.clear()

    def record_change(
        self,
        action: str,
        data: Dict[str, Any],
        nodes: Dict[str, Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> Optional[GraphDelta]:
        """记录变更并生成增量"""
        if not self.enabled:
            return None

        delta = GraphDelta()

        if action == "node_created":
            node_id = data.get("id", "")
            if node_id:
                delta.added_nodes.append(data)
                self.detector.apply_changes(delta)
                return delta

        elif action == "node_updated":
            node_id = data.get("id", "")
            if node_id:
                delta.updated_nodes.append(data)
                self.detector.apply_changes(delta)
                return delta

        elif action == "node_deleted":
            node_id = data.get("id", "")
            if node_id:
                delta.deleted_nodes.append(node_id)
                self.detector.apply_changes(delta)
                return delta

        elif action == "edge_created":
            delta.added_edges.append(data)
            self.detector.apply_changes(delta)
            return delta

        elif action == "edge_deleted":
            edge_id = data.get("id", "")
            if edge_id:
                delta.deleted_edges.append(edge_id)
                self.detector.apply_changes(delta)
                return delta

        elif action in ("batch_created", "batch_updated"):
            batch_nodes = data.get("nodes", [])
            for node_data in batch_nodes:
                node_id = node_data.get("id", "")
                if node_id and node_id not in self.detector.node_snapshots:
                    delta.added_nodes.append(node_data)
                elif node_id:
                    delta.updated_nodes.append(node_data)
            
            if not delta.is_empty():
                self.detector.apply_changes(delta)
                return delta

        return None

    def should_push(self) -> bool:
        """判断是否应该推送"""
        if not self.enabled:
            return False

        current_time = time.time()
        if current_time - self.last_push_time < self.push_interval:
            return False

        return len(self.pending_deltas) > 0

    def get_pending_delta(self) -> Optional[GraphDelta]:
        """获取待推送的增量"""
        if not self.pending_deltas:
            return None

        merged = GraphDelta()
        for delta in self.pending_deltas:
            merged.added_nodes.extend(delta.added_nodes)
            merged.updated_nodes.extend(delta.updated_nodes)
            merged.deleted_nodes.extend(delta.deleted_nodes)
            merged.added_edges.extend(delta.added_edges)
            merged.deleted_edges.extend(delta.deleted_edges)

        self.pending_deltas.clear()
        self.last_push_time = time.time()

        return merged

    def add_pending_delta(self, delta: GraphDelta):
        """添加待推送增量"""
        if not delta.is_empty():
            self.pending_deltas.append(delta)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "enabled": self.enabled,
            "pending_count": len(self.pending_deltas),
            "push_interval": self.push_interval,
            "last_push_time": self.last_push_time,
            "snapshot": self.detector.get_snapshot_stats(),
        }
