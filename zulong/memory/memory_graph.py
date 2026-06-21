"""Native MemoryGraph facade backed only by sharded storage.

The public module name remains ``zulong.memory.memory_graph`` because that is
the domain concept used by L1/L2, tools, Web, and IDE code.  The old NetworkX +
single JSON implementation has been removed; all runtime access goes through
``storage_hybrid.ShardedMemoryGraph`` created by ``memory_graph_factory``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from zulong.memory.storage_hybrid.memory_graph_hybrid import (
    EdgeType,
    Importance,
    NodeType,
)


class Temperature(Enum):
    """节点温度标签。

    这是存储生命周期/前端展示层的温度标签；检索冷热路径仍由
    ShardedMemoryGraph.retrieve_context(hot_window_minutes=...) 控制。
    """

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass
class GraphNode:
    """Public graph node DTO accepted by the native sharded MemoryGraph."""

    node_id: str
    node_type: NodeType
    label: str
    activation: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    backend_ref: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    content: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "node_id": self.node_id,
            "node_type": _enum_value(self.node_type),
            "label": self.label,
            "activation": self.activation,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "backend_ref": self.backend_ref,
            "metadata": self.metadata,
        }
        if self.content is not None:
            data["content"] = self.content
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphNode":
        raw_type = data.get("node_type", NodeType.KNOWLEDGE.value)
        return cls(
            node_id=str(data["node_id"]),
            node_type=raw_type if isinstance(raw_type, NodeType) else NodeType(str(raw_type)),
            label=str(data.get("label", "")),
            activation=float(data.get("activation", 0.0) or 0.0),
            created_at=float(data.get("created_at", time.time()) or time.time()),
            last_accessed=float(data.get("last_accessed", time.time()) or time.time()),
            access_count=int(data.get("access_count", 0) or 0),
            backend_ref=str(data.get("backend_ref", "") or ""),
            metadata=dict(data.get("metadata", {}) or {}),
            content=data.get("content"),
        )


class MemoryGraph:
    """Compatibility constructor for the native sharded MemoryGraph.

    ``MemoryGraph()`` no longer creates a local in-memory/JSON graph.  It returns
    the process singleton produced by ``get_memory_graph()``, whose concrete
    type is ``ShardedMemoryGraph``.
    """

    _instance = None

    def __new__(cls, persist_path: str = "./data/memory_graph", *args, **kwargs):
        return get_memory_graph(persist_path=persist_path)


def get_memory_graph(persist_path: str = None):
    """Return the single native sharded MemoryGraph instance."""

    if MemoryGraph._instance is not None:
        return MemoryGraph._instance

    from zulong.memory.memory_graph_factory import (
        assert_native_memory_graph,
        create_memory_graph,
    )

    graph = create_memory_graph(persist_path=persist_path or "./data/memory_graph")
    assert_native_memory_graph(graph)
    MemoryGraph._instance = graph
    return graph


def _enum_value(value: Any) -> str:
    return getattr(value, "value", str(value))


__all__ = [
    "MemoryGraph",
    "get_memory_graph",
    "GraphNode",
    "NodeType",
    "EdgeType",
    "Importance",
    "Temperature",
]
