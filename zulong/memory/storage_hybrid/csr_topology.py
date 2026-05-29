# File: zulong/memory/storage_hybrid/csr_topology.py
# mmap 友好的 CSR/CSC 拓扑侧车索引

import mmap
import os
import pickle
import struct
import time
import uuid
from collections import deque
from typing import Dict, List, Optional, Tuple


CSR_TOPOLOGY_MAGIC = b"ZLCS"
CSR_TOPOLOGY_VERSION = 1
CSR_HEADER_STRUCT = struct.Struct("<4sIQ")  # magic, version, manifest_len


class CSRTopologyReader:
    """只读 CSR/CSC mmap 拓扑索引。

    节点 ID 映射保留在内存中；邻接、边类型和权重数组直接从 mmap
    文件按偏移读取。它是 P3 运行态迁移的读侧车，不负责写入。
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._file = open(filepath, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        magic, version, manifest_len = CSR_HEADER_STRUCT.unpack_from(self._mmap, 0)
        if magic != CSR_TOPOLOGY_MAGIC:
            self.close()
            raise ValueError(f"无效 csr topology magic: {magic!r}")
        if version != CSR_TOPOLOGY_VERSION:
            self.close()
            raise ValueError(f"不支持 csr topology version={version}")

        manifest_start = CSR_HEADER_STRUCT.size
        manifest_end = manifest_start + manifest_len
        manifest = pickle.loads(self._mmap[manifest_start:manifest_end])

        self.node_ids: List[str] = manifest["node_ids"]
        self.node_types: List[str] = manifest.get("node_types") or ["unknown"] * len(self.node_ids)
        self.edge_types: List[str] = manifest["edge_types"]
        self.node_id_to_idx: Dict[str, int] = {node_id: idx for idx, node_id in enumerate(self.node_ids)}
        self.node_count = int(manifest["node_count"])
        self.edge_count = int(manifest["edge_count"])
        self.offsets = manifest["offsets"]

    def close(self) -> None:
        try:
            if getattr(self, "_mmap", None):
                self._mmap.close()
        finally:
            if getattr(self, "_file", None):
                self._file.close()

    def get_node_type(self, node_id: str) -> Optional[str]:
        idx = self.node_id_to_idx.get(node_id)
        if idx is None:
            return None
        return self.node_types[idx]

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        mode: str = "out",
    ) -> List[str]:
        if mode == "all":
            seen = set()
            result = []
            for neighbor in self.get_neighbors(node_id, edge_type=edge_type, mode="out"):
                if neighbor not in seen:
                    seen.add(neighbor)
                    result.append(neighbor)
            for neighbor in self.get_neighbors(node_id, edge_type=edge_type, mode="in"):
                if neighbor not in seen:
                    seen.add(neighbor)
                    result.append(neighbor)
            return result

        idx = self.node_id_to_idx.get(node_id)
        if idx is None:
            return []

        edge_type_id = self._edge_type_id(edge_type)
        if edge_type and edge_type_id is None:
            return []

        offsets_name = "out_offsets" if mode == "out" else "in_offsets"
        nodes_name = "out_targets" if mode == "out" else "in_sources"
        types_name = "out_type_ids" if mode == "out" else "in_type_ids"

        start, end = self._offset_range(offsets_name, idx)
        result = []
        for pos in range(start, end):
            if edge_type_id is not None and self._u16(types_name, pos) != edge_type_id:
                continue
            neighbor_idx = self._u32(nodes_name, pos)
            result.append(self.node_ids[neighbor_idx])
        return result

    def get_edge_info(self, src_id: str, dst_id: str) -> Optional[Tuple[str, float]]:
        src_idx = self.node_id_to_idx.get(src_id)
        dst_idx = self.node_id_to_idx.get(dst_id)
        if src_idx is None or dst_idx is None:
            return None

        start, end = self._offset_range("out_offsets", src_idx)
        for pos in range(start, end):
            if self._u32("out_targets", pos) != dst_idx:
                continue
            type_id = self._u16("out_type_ids", pos)
            weight = self._f32("out_weights", pos)
            edge_type = self.edge_types[type_id] if type_id < len(self.edge_types) else "association"
            return edge_type, weight
        return None

    def bfs_spread_weighted(
        self,
        seed_ids: List[str],
        max_depth: int = 3,
        decay_factor: float = 0.5,
    ) -> List[Tuple[str, float]]:
        visited: Dict[int, float] = {}
        queue = deque()

        for seed_id in seed_ids:
            idx = self.node_id_to_idx.get(seed_id)
            if idx is None:
                continue
            visited[idx] = max(visited.get(idx, 0.0), 1.0)
            queue.append((idx, 1.0, 0))

        while queue:
            current_idx, current_score, current_depth = queue.popleft()
            if current_depth >= max_depth:
                continue
            start, end = self._offset_range("out_offsets", current_idx)
            for pos in range(start, end):
                neighbor_idx = self._u32("out_targets", pos)
                weight = self._f32("out_weights", pos)
                new_score = current_score * decay_factor * weight
                if new_score <= visited.get(neighbor_idx, 0.0):
                    continue
                visited[neighbor_idx] = new_score
                queue.append((neighbor_idx, new_score, current_depth + 1))

        result = [(self.node_ids[idx], score) for idx, score in visited.items()]
        result.sort(key=lambda item: item[1], reverse=True)
        return result

    def _edge_type_id(self, edge_type: Optional[str]) -> Optional[int]:
        if edge_type is None:
            return None
        edge_type_value = str(getattr(edge_type, "value", edge_type))
        try:
            return self.edge_types.index(edge_type_value)
        except ValueError:
            return None

    def _offset_range(self, array_name: str, idx: int) -> Tuple[int, int]:
        base = self.offsets[array_name]
        start = struct.unpack_from("<Q", self._mmap, base + idx * 8)[0]
        end = struct.unpack_from("<Q", self._mmap, base + (idx + 1) * 8)[0]
        return int(start), int(end)

    def _u32(self, array_name: str, pos: int) -> int:
        return struct.unpack_from("<I", self._mmap, self.offsets[array_name] + pos * 4)[0]

    def _u16(self, array_name: str, pos: int) -> int:
        return struct.unpack_from("<H", self._mmap, self.offsets[array_name] + pos * 2)[0]

    def _f32(self, array_name: str, pos: int) -> float:
        return struct.unpack_from("<f", self._mmap, self.offsets[array_name] + pos * 4)[0]


def save_topology_to_csr(topology, filepath: str) -> None:
    """把 TopologyIndex 导出为 CSR/CSC mmap 侧车文件。"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    node_count = len(topology.graph.vs)
    edges = []
    edge_type_ids: Dict[str, int] = {}
    edge_types: List[str] = []

    def _edge_type_id(edge_type: str) -> int:
        if edge_type not in edge_type_ids:
            edge_type_ids[edge_type] = len(edge_types)
            edge_types.append(edge_type)
        return edge_type_ids[edge_type]

    node_ids = []
    node_types = []
    for vertex in topology.graph.vs:
        node_ids.append(vertex["name"])
        node_types.append(vertex["type"] if "type" in vertex.attributes() else "unknown")

    for edge in topology.graph.es:
        edge_type = edge["type"] if "type" in edge.attributes() else "association"
        weight = float(edge["weight"]) if "weight" in edge.attributes() else 1.0
        edges.append((edge.source, edge.target, _edge_type_id(edge_type), weight))

    out_edges = sorted(edges, key=lambda item: (item[0], item[1], item[2]))
    in_edges = sorted(edges, key=lambda item: (item[1], item[0], item[2]))

    out_offsets = _build_offsets(out_edges, node_count, source_pos=0)
    in_offsets = _build_offsets(in_edges, node_count, source_pos=1)

    manifest = {
        "version": CSR_TOPOLOGY_VERSION,
        "created_at": time.time(),
        "node_count": node_count,
        "edge_count": len(edges),
        "node_ids": node_ids,
        "node_types": node_types,
        "edge_types": edge_types,
        "offsets": {},
    }
    array_specs = [
        ("out_offsets", "<Q", out_offsets),
        ("out_targets", "<I", [edge[1] for edge in out_edges]),
        ("out_type_ids", "<H", [edge[2] for edge in out_edges]),
        ("out_weights", "<f", [edge[3] for edge in out_edges]),
        ("in_offsets", "<Q", in_offsets),
        ("in_sources", "<I", [edge[0] for edge in in_edges]),
        ("in_type_ids", "<H", [edge[2] for edge in in_edges]),
        ("in_weights", "<f", [edge[3] for edge in in_edges]),
    ]
    arrays = [b"".join(struct.pack(fmt, value) for value in values) for _, fmt, values in array_specs]
    manifest_blob = b""
    for _ in range(5):
        cursor = CSR_HEADER_STRUCT.size + len(manifest_blob)
        offsets = {}
        for (name, _, _), blob in zip(array_specs, arrays):
            offsets[name] = cursor
            cursor += len(blob)
        manifest["offsets"] = offsets
        next_blob = pickle.dumps(manifest, protocol=pickle.HIGHEST_PROTOCOL)
        if len(next_blob) == len(manifest_blob):
            manifest_blob = next_blob
            break
        manifest_blob = next_blob
    else:
        raise RuntimeError("csr topology manifest offset did not stabilize")

    tmp_path = f"{filepath}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(CSR_HEADER_STRUCT.pack(CSR_TOPOLOGY_MAGIC, CSR_TOPOLOGY_VERSION, len(manifest_blob)))
        f.write(manifest_blob)
        for blob in arrays:
            f.write(blob)
    try:
        os.replace(tmp_path, filepath)
    except PermissionError:
        if os.path.exists(filepath):
            os.remove(filepath)
        os.replace(tmp_path, filepath)


def _build_offsets(edges: List[Tuple[int, int, int, float]], node_count: int, source_pos: int) -> List[int]:
    offsets = [0] * (node_count + 1)
    cursor = 0
    for node_idx in range(node_count):
        while cursor < len(edges) and edges[cursor][source_pos] < node_idx:
            cursor += 1
        offsets[node_idx] = cursor
    offsets[node_count] = len(edges)
    return offsets
