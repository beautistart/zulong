# File: zulong/pipeline/task_graph.py
"""
TaskGraph 数据结构

Agent 自循环架构的核心骨架，与前端 addTaskGraph() 格式完全对齐。
贯穿整个 Agent 生命周期，模型通过 FC 工具读写它的节点和边。
"""

import logging
import uuid
import time
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Callable, Any

logger = logging.getLogger(__name__)


@dataclass
class FileRef:
    """节点关联文件引用"""
    name: str          # 文件名（显示用），如 "task_graph.py"
    path: str          # 文件路径，如 "zulong/pipeline/task_graph.py"

    def to_dict(self) -> dict:
        return {"name": self.name, "path": self.path}

    @classmethod
    def from_dict(cls, data: dict) -> "FileRef":
        return cls(name=data["name"], path=data["path"])


@dataclass
class TaskNode:
    """任务图谱节点

    type 由节点在树中的深度自动决定（depth_to_type()）:
      深度0: "requirement"  (req 根节点)
      深度1: "analysis"     (需求分析)
      深度2: "outline"      (大纲)
      深度3: "task"         (任务)
      深度4+: "subtask"     (子任务, 无限深度)
    """
    id: str                          # 唯一标识, 如 "req", "o1", "o1_1", "o1_1_1"
    label: str                       # 显示名称
    type: str                        # 由深度决定，见上方说明
    status: str                      # "pending"|"in_progress"|"completed"|"blocked"|"skipped"|"needs_adjust"|"waiting_input"
    desc: str                        # 描述文本
    result: Optional[str] = None     # 执行结果（Agent 执行后填充）
    files: List["FileRef"] = field(default_factory=list)  # 关联文件列表

    def add_file(self, name: str, path: str):
        """添加关联文件（去重）"""
        for f in self.files:
            if f.path == path:
                return
        self.files.append(FileRef(name=name, path=path))

    def to_dict(self) -> dict:
        """转换为字典（兼容前端格式）"""
        d = {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "status": self.status,
            "desc": self.desc,
        }
        if self.result is not None:
            d["result"] = self.result
        if self.files:
            d["files"] = [f.to_dict() for f in self.files]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "TaskNode":
        files = [FileRef.from_dict(f) for f in data.get("files", [])]
        return cls(
            id=data["id"],
            label=data["label"],
            type=data.get("type", "task"),
            status=data.get("status", "pending"),
            desc=data.get("desc", ""),
            result=data.get("result"),
            files=files,
        )


@dataclass
class DependencyEdge:
    """依赖边（数据传递关系）"""
    s: str           # 源节点 ID
    t: str           # 目标节点 ID
    via: str = ""    # 传递的数据描述
    cross: bool = False  # 是否跨层

    def to_dict(self) -> dict:
        return {"s": self.s, "t": self.t, "via": self.via, "cross": self.cross}

    @classmethod
    def from_dict(cls, data: dict) -> "DependencyEdge":
        return cls(
            s=data["s"],
            t=data["t"],
            via=data.get("via", ""),
            cross=data.get("cross", False),
        )


class TaskGraph:
    """任务图谱

    Agent 自循环架构的核心数据结构：
    - 模型通过 plan_* 工具创建图谱、添加子任务节点和层级边
    - 模型通过 update_node_status 更新节点状态 pending -> in_progress -> completed/blocked
    - 模型通过 submit_final_answer 完成任务
    """

    def __init__(self, title: str, graph_id: str = None):
        self.id: str = graph_id or uuid.uuid4().hex[:12]
        self.title: str = title
        self.created_at: float = time.time()

        self._nodes: Dict[str, TaskNode] = {}    # id -> TaskNode
        self._h_edges: List[Tuple[str, str]] = []  # 层级边 (parent, child)
        self._d_edges: List[DependencyEdge] = []    # 依赖边

        self.parallel_groups: List[List[str]] = []  # Stage2 计算的并行组
        self.metadata: Dict[str, Any] = {}          # 扩展信息

        # 节点变化回调（由 Orchestrator 注入，用于推送前端）
        self.on_change_callback: Optional[Callable] = None

    @property
    def address(self) -> str:
        """任务图谱的全局唯一地址"""
        return f"tg:{self.id}"

    def get_node_address(self, node_id: str) -> str:
        """获取节点的完整层级地址
        
        从根节点沿 h_edges 追溯到当前节点，拼接为路径。
        例如: tg:tg_123/req/o1/o1_1
        """
        # 构建 child -> parent 映射
        child_to_parent = {}
        for s, t in self._h_edges:
            child_to_parent[t] = s

        # 从当前节点向上追溯到根
        path_parts = [node_id]
        current = node_id
        visited = set()
        while current in child_to_parent and current not in visited:
            visited.add(current)
            current = child_to_parent[current]
            path_parts.insert(0, current)

        return f"{self.address}/{'/'.join(path_parts)}"

    # ─── 节点操作 ───────────────────────────────────────────

    def add_node(
        self,
        id: str,
        label: str,
        type: str = "task",
        status: str = "pending",
        desc: str = "",
        result: str = None,
        files: List[FileRef] = None,
    ) -> TaskNode:
        """添加节点"""
        node = TaskNode(
            id=id, label=label, type=type, status=status,
            desc=desc, result=result, files=files or [],
        )
        self._nodes[id] = node
        logger.debug(f"[TaskGraph] 添加节点: {id} ({label}, type={type}, status={status})")
        return node

    def add_file_to_node(self, node_id: str, file_name: str, file_path: str) -> bool:
        """向节点添加关联文件"""
        node = self._nodes.get(node_id)
        if not node:
            return False
        node.add_file(file_name, file_path)
        if self.on_change_callback:
            try:
                self.on_change_callback(
                    "node_file_add",
                    {"node_id": node_id, "file": {"name": file_name, "path": file_path}},
                )
            except Exception as e:
                logger.debug(f"[TaskGraph] 文件回调失败（非致命）: {e}")
        return True

    def update_node_status(
        self, node_id: str, status: str, result: str = None
    ) -> bool:
        """更新节点状态，触发变化回调"""
        node = self._nodes.get(node_id)
        if not node:
            logger.warning(f"[TaskGraph] 节点 {node_id} 不存在")
            return False

        old_status = node.status
        node.status = status
        if result is not None:
            node.result = result

        logger.info(
            f"[TaskGraph] 节点状态更新: {node_id} {old_status} → {status}"
        )

        # 同步到 MemoryGraph（如果可用）
        self._sync_node_to_memory_graph(node_id, status, result)

        # 触发变化回调（推送到前端）
        if self.on_change_callback:
            try:
                self.on_change_callback(
                    "node_update",
                    {"node_id": node_id, "status": status, "result": result},
                )
            except Exception as e:
                logger.debug(f"[TaskGraph] 回调执行失败（非致命）: {e}")

        # 自动级联：当节点完成/跳过时，检查父节点是否应自动标记为完成
        if status in ("completed", "skipped"):
            self._cascade_parent_status(node_id)

        return True

    def _cascade_parent_status(self, node_id: str):
        """子节点完成后自动向上级联父节点状态

        当某节点的所有兄弟节点均为 completed/skipped 时，
        自动将父节点标记为 completed，并递归向上检查。
        """
        if getattr(self, '_cascading', False):
            return
        self._cascading = True
        try:
            parent_id = self.get_parent(node_id)
            if not parent_id:
                return
            children = self.get_children(parent_id)
            if not children:
                return
            all_done = all(
                c.status in ("completed", "skipped") for c in children
            )
            if all_done:
                parent_node = self.get_node(parent_id)
                if parent_node and parent_node.status != "completed":
                    parent_node.status = "completed"
                    logger.info(
                        f"[TaskGraph] 自动级联: {parent_id} → completed "
                        f"(所有子节点已完成)"
                    )
                    self._sync_node_to_memory_graph(parent_id, "completed")
                    if self.on_change_callback:
                        try:
                            self.on_change_callback(
                                "node_update",
                                {
                                    "node_id": parent_id,
                                    "status": "completed",
                                    "auto_cascaded": True,
                                },
                            )
                        except Exception:
                            pass
                    # 递归向上
                    self._cascading = False
                    self._cascade_parent_status(parent_id)
        finally:
            self._cascading = False

    def _sync_node_to_memory_graph(
        self, node_id: str, status: str, result: str = None
    ):
        """将节点状态更新同步到 MemoryGraph 对应的 TASK 节点"""
        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            if mg is None:
                return
            # 查找对应的 MemoryGraph 节点（兼容多种地址格式）
            mem_node_id = f"task:{self.id}/{node_id}"
            mem_node = mg.get_node(mem_node_id)
            if mem_node is None:
                # 尝试不带图 ID 的格式
                mem_node_id = f"task:{node_id}"
                mem_node = mg.get_node(mem_node_id)
            if mem_node is None:
                # 遍历查找：TaskGraphAdapter 可能使用带 session 前缀的完整路径
                for nid, n in mg._nodes.items():
                    if (nid.endswith(f"/{node_id}")
                            and n.metadata.get("graph_id") == self.id):
                        mem_node = n
                        mem_node_id = nid
                        break
            if mem_node is not None:
                mem_node.metadata["task_status"] = status
                if result is not None:
                    mem_node.metadata["task_result"] = result[:500]
                mem_node.last_accessed = time.time()
                # 根据任务状态调整 MemoryGraph 激活值
                _activation_map = {
                    "completed": 0.1,
                    "skipped": 0.1,
                    "in_progress": 0.9,
                    "blocked": 0.5,
                }
                _target_activation = _activation_map.get(status)
                if _target_activation is not None:
                    try:
                        mg.update_node_activation(mem_node_id, _target_activation)
                    except Exception:
                        pass
                logger.debug(
                    f"[TaskGraph] MemoryGraph 同步: {mem_node_id} → {status}"
                )
        except Exception as e:
            # 非致命：MemoryGraph 同步失败不影响 TaskGraph 正常工作
            logger.debug(f"[TaskGraph] MemoryGraph 同步跳过: {e}")

    def get_node(self, node_id: str) -> Optional[TaskNode]:
        """获取节点"""
        return self._nodes.get(node_id)

    def get_nodes_by_type(self, node_type: str) -> List[TaskNode]:
        """按类型过滤节点"""
        return [n for n in self._nodes.values() if n.type == node_type]

    def get_nodes_by_status(self, status: str) -> List[TaskNode]:
        """按状态过滤节点"""
        return [n for n in self._nodes.values() if n.status == status]

    def get_subtask_nodes(self) -> List[TaskNode]:
        """获取所有子任务节点（type="task"）— 兼容旧接口"""
        return self.get_nodes_by_type("task")

    def get_children(self, parent_id: str) -> List[TaskNode]:
        """获取某节点的直接子节点（从 h_edges 查找）"""
        child_ids = [t for s, t in self._h_edges if s == parent_id]
        return [self._nodes[cid] for cid in child_ids if cid in self._nodes]

    def get_leaf_nodes(self) -> List[TaskNode]:
        """获取所有叶子节点（没有子节点的非根节点）

        叶子节点 = 不在任何 h_edge 的 source 侧 且 不是根节点(req)
        这些节点是实际需要执行的工作项。
        """
        parent_ids = {s for s, t in self._h_edges}
        return [
            n for n in self._nodes.values()
            if n.id not in parent_ids and n.id != "req"
        ]

    def get_node_depth(self, node_id: str) -> int:
        """获取节点在树中的深度（从 req 到该节点的距离）"""
        child_to_parent = {}
        for s, t in self._h_edges:
            child_to_parent[t] = s

        depth = 0
        current = node_id
        while current in child_to_parent:
            current = child_to_parent[current]
            depth += 1
            if depth > 50:
                break
        return depth

    @staticmethod
    def depth_to_type(depth: int) -> str:
        """根据深度返回节点类型

        深度0: requirement, 深度1: analysis, 深度2: outline,
        深度3: task, 深度4+: subtask
        """
        mapping = {0: "requirement", 1: "analysis", 2: "outline", 3: "task"}
        return mapping.get(depth, "subtask")

    @property
    def nodes(self) -> List[TaskNode]:
        """所有节点列表"""
        return list(self._nodes.values())

    def remove_node(self, node_id: str) -> List[str]:
        """移除节点及其所有后代，清理相关边

        Args:
            node_id: 要移除的节点 ID

        Returns:
            被移除的所有节点 ID 列表
        """
        if node_id in ("req", "analysis"):
            logger.warning(f"[TaskGraph] 不能移除根节点: {node_id}")
            return []

        if node_id not in self._nodes:
            return []

        # 收集所有要移除的节点
        descendants = self.get_all_descendants(node_id)
        removed = [node_id] + descendants

        removed_set = set(removed)
        for nid in removed:
            self._nodes.pop(nid, None)

        # 清理层级边
        self._h_edges = [
            (s, t) for s, t in self._h_edges
            if s not in removed_set and t not in removed_set
        ]
        # 清理依赖边
        self._d_edges = [
            e for e in self._d_edges
            if e.s not in removed_set and e.t not in removed_set
        ]
        # 清理并行组
        self.parallel_groups = [
            [nid for nid in group if nid not in removed_set]
            for group in self.parallel_groups
        ]
        self.parallel_groups = [g for g in self.parallel_groups if g]

        logger.info(f"[TaskGraph] 移除节点: {node_id} 及 {len(descendants)} 个后代")

        if self.on_change_callback:
            try:
                self.on_change_callback(
                    "node_remove", {"removed_ids": removed},
                )
            except Exception as e:
                logger.debug(f"[TaskGraph] 移除回调失败（非致命）: {e}")

        return removed

    # ─── 边操作 ─────────────────────────────────────────────

    def add_h_edge(self, source: str, target: str):
        """添加层级边（父子关系）"""
        self._h_edges.append((source, target))

    def add_d_edge(
        self, source: str, target: str, via: str = "", cross: bool = False
    ):
        """添加依赖边（数据传递关系）"""
        self._d_edges.append(DependencyEdge(s=source, t=target, via=via, cross=cross))

    def get_dependencies(self, node_id: str) -> List[str]:
        """获取某节点的前置依赖节点 ID 列表"""
        return [e.s for e in self._d_edges if e.t == node_id]

    def get_dependents(self, node_id: str) -> List[str]:
        """获取依赖于某节点的后续节点 ID 列表"""
        return [e.t for e in self._d_edges if e.s == node_id]

    @property
    def h_edges(self) -> List[Tuple[str, str]]:
        return list(self._h_edges)

    @property
    def d_edges(self) -> List[DependencyEdge]:
        return list(self._d_edges)

    # ─── 导出 ───────────────────────────────────────────────

    def to_frontend_dict(self) -> dict:
        """导出为前端 addTaskGraph() 兼容格式

        非叶子节点的 status 从子节点状态动态聚合（不修改原始数据）。
        每个节点附带完整层级地址。
        """
        parent_ids = {s for s, t in self._h_edges}
        aggregated_nodes = []

        for n in self._nodes.values():
            d = n.to_dict()
            # 非叶子、非模板节点 → 动态聚合 status
            if n.id in parent_ids and n.id not in ("req", "analysis"):
                d["status"] = self._aggregate_status(n.id)
            # 附带完整层级地址
            d["address"] = self.get_node_address(n.id)
            aggregated_nodes.append(d)

        return {
            "id": self.id,
            "title": self.title,
            "graphAddress": self.address,
            "createdAt": int(self.created_at * 1000),
            "nodes": aggregated_nodes,
            "hEdges": [list(e) for e in self._h_edges],
            "dEdges": [e.to_dict() for e in self._d_edges],
        }

    def _aggregate_status(self, node_id: str, _nodes_with_children: set = None) -> str:
        """从子节点状态聚合父节点的显示状态（递归）"""
        children = self.get_children(node_id)
        if not children:
            node = self._nodes.get(node_id)
            return node.status if node else "pending"

        if _nodes_with_children is None:
            _nodes_with_children = {s for s, t in self._h_edges}

        statuses = set()
        for child in children:
            if child.id in _nodes_with_children:
                statuses.add(self._aggregate_status(child.id, _nodes_with_children))
            else:
                statuses.add(child.status)

        if statuses == {"completed"}:
            return "completed"
        if "in_progress" in statuses:
            return "in_progress"
        if "blocked" in statuses:
            return "blocked"
        if "needs_adjust" in statuses:
            return "needs_adjust"
        if "completed" in statuses:
            return "in_progress"
        return "pending"

    def to_planning_table(self) -> str:
        """生成自然语言规划表（按大纲分组的递归 Markdown 表）

        注入到子任务 prompt 中，让 LLM 了解整体进展。
        遵循"规划表即结构"范式。
        """
        STATUS_TEXT = {
            "completed": "完成",
            "in_progress": "进行中",
            "pending": "待开始",
            "blocked": "阻塞",
            "skipped": "跳过",
            "needs_adjust": "需调整",
        }

        lines = ["## 当前任务规划\n"]

        # 获取 analysis 的子节点（大纲层）
        outlines = self.get_children("analysis")
        if not outlines:
            # 兜底：直接列出所有叶子节点
            outlines = self.get_children("req")

        for outline in outlines:
            lines.append(f"\n### {outline.label}\n")
            lines.append("| 编号 | 子任务 | 状态 | 结果摘要 |")
            lines.append("|------|--------|------|----------|")
            self._append_leaf_rows(outline.id, lines, STATUS_TEXT)

        return "\n".join(lines)

    def _append_leaf_rows(
        self, parent_id: str, lines: List[str], status_text: dict, indent: int = 0
    ):
        """递归收集某节点下的叶子节点行"""
        children = self.get_children(parent_id)
        for child in children:
            grandchildren = self.get_children(child.id)
            if grandchildren:
                # 非叶子节点：作为分组标题
                prefix = "  " * indent
                lines.append(f"| | {prefix}**{child.label}** | | |")
                self._append_leaf_rows(child.id, lines, status_text, indent + 1)
            else:
                # 叶子节点：输出为表格行
                prefix = "  " * indent
                st = status_text.get(child.status, child.status)
                result_summary = ""
                if child.result:
                    result_summary = child.result[:80].replace("\n", " ")
                    if len(child.result) > 80:
                        result_summary += "..."
                lines.append(
                    f"| {child.id} | {prefix}{child.label} | {st} | {result_summary} |"
                )

    # ─── 序列化（挂起/恢复支持） ────────────────────────────

    def serialize(self) -> dict:
        """序列化为可存储的字典（含节点层级地址）"""
        nodes_data = {}
        for nid, n in self._nodes.items():
            nd = n.to_dict()
            nd["address"] = self.get_node_address(nid)
            nodes_data[nid] = nd
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "address": self.address,
            "nodes": nodes_data,
            "h_edges": [list(e) for e in self._h_edges],
            "d_edges": [e.to_dict() for e in self._d_edges],
            "parallel_groups": self.parallel_groups,
            "metadata": self.metadata,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "TaskGraph":
        """从字典反序列化"""
        # 前置校验：必要字段
        if not isinstance(data, dict):
            raise ValueError(f"TaskGraph.deserialize: 数据类型错误，期望 dict，实际 {type(data)}")
        if "title" not in data or "id" not in data:
            raise ValueError(f"TaskGraph.deserialize: 缺少必要字段 title/id，keys={list(data.keys())}")
        
        graph = cls(title=data["title"], graph_id=data["id"])
        graph.created_at = data.get("created_at", time.time())

        for nid, ndata in data.get("nodes", {}).items():
            try:
                node = TaskNode.from_dict(ndata)
                graph._nodes[nid] = node
            except Exception as e:
                logger.warning(f"[TaskGraph] 反序列化节点 {nid} 失败，跳过: {e}")

        # 后置校验：清理引用不存在节点的层级边
        valid_node_ids = set(graph._nodes.keys())
        for edge in data.get("h_edges", []):
            src, tgt = edge[0], edge[1]
            if src in valid_node_ids and tgt in valid_node_ids:
                graph._h_edges.append(tuple(edge))
            else:
                logger.warning(f"[TaskGraph] 层级边 {src}->{tgt} 引用不存在节点，已跳过")

        for edata in data.get("d_edges", []):
            try:
                dep = DependencyEdge.from_dict(edata)
                if dep.s in valid_node_ids and dep.t in valid_node_ids:
                    graph._d_edges.append(dep)
                else:
                    logger.warning(f"[TaskGraph] 依赖边引用不存在节点，已跳过")
            except Exception as e:
                logger.warning(f"[TaskGraph] 反序列化依赖边失败，跳过: {e}")

        graph.parallel_groups = data.get("parallel_groups", [])
        graph.metadata = data.get("metadata", {})

        return graph

    # ─── 树导航辅助 ──────────────────────────────────────────

    def get_parent(self, node_id: str) -> Optional[str]:
        """获取节点的父节点 ID"""
        for s, t in self._h_edges:
            if t == node_id:
                return s
        return None

    def get_ancestor_chain(self, node_id: str) -> List[TaskNode]:
        """获取从当前节点到根节点的祖先链（不含自身，从父到根）"""
        chain = []
        current = node_id
        visited = set()
        while True:
            parent_id = self.get_parent(current)
            if not parent_id or parent_id in visited:
                break
            visited.add(parent_id)
            parent_node = self._nodes.get(parent_id)
            if parent_node:
                chain.append(parent_node)
            current = parent_id
        return chain

    def get_ancestor_at_depth(self, node_id: str, target_depth: int) -> Optional[str]:
        """获取节点在指定深度的祖先节点 ID"""
        current = node_id
        current_depth = self.get_node_depth(node_id)
        visited = set()
        while current_depth > target_depth:
            parent_id = self.get_parent(current)
            if not parent_id or parent_id in visited:
                return None
            visited.add(parent_id)
            current = parent_id
            current_depth -= 1
        return current if current_depth == target_depth else None

    def get_all_descendants(self, node_id: str) -> List[str]:
        """获取某节点下的所有后代节点 ID（BFS）"""
        result = []
        queue = [node_id]
        visited = {node_id}
        while queue:
            current = queue.pop(0)
            children_ids = [t for s, t in self._h_edges if s == current]
            for cid in children_ids:
                if cid not in visited:
                    visited.add(cid)
                    result.append(cid)
                    queue.append(cid)
        return result

    # ─── 增量拆解支持 ────────────────────────────────────────

    def convert_leaf_to_parent(
        self, node_id: str, children: List[Dict[str, str]]
    ) -> List[str]:
        """将叶子节点转为中间节点，挂载子节点（增量拆解）

        Args:
            node_id: 当前叶子节点 ID
            children: 子节点列表 [{"name": "...", "desc": "..."}, ...]

        Returns:
            新生成的子节点 ID 列表
        """
        depth = self.get_node_depth(node_id) + 1
        new_ids = []

        for i, child in enumerate(children):
            child_id = f"{node_id}_{i + 1}"
            child_type = self.depth_to_type(depth)
            self.add_node(
                id=child_id,
                label=child["name"],
                type=child_type,
                status="pending",
                desc=child.get("desc", child["name"]),
            )
            self.add_h_edge(node_id, child_id)
            new_ids.append(child_id)
            logger.info(
                f"[TaskGraph] 增量拆解: {node_id} -> {child_id} ({child['name']})"
            )

        # 推送变化
        if self.on_change_callback:
            try:
                self.on_change_callback(
                    "incremental_decompose",
                    {"parent_id": node_id, "new_node_ids": new_ids},
                )
            except Exception as e:
                logger.debug(f"[TaskGraph] 增量拆解回调失败（非致命）: {e}")

        return new_ids

    # ─── 分级规划表（局部注意力） ─────────────────────────────

    def to_focused_planning_table(self, focus_node_id: str) -> str:
        """生成聚焦规划表：单链路全局注意

        - 当前节点所在大纲：展开到叶子级别（完整细节）
        - 同级其他大纲：只显示标题 + 进度摘要（一行）
        - 跨大纲的直接依赖：展示详细结果
        """
        STATUS_TEXT = {
            "completed": "完成", "in_progress": "进行中",
            "pending": "待开始", "blocked": "阻塞",
            "skipped": "跳过", "needs_adjust": "需调整",
        }

        # 找到焦点节点所在的大纲（depth=2 祖先）
        focus_outline_id = self.get_ancestor_at_depth(focus_node_id, 2)

        lines = ["## 当前任务规划\n"]

        outlines = self.get_children("analysis")
        if not outlines:
            outlines = self.get_children("req")

        for outline in outlines:
            if outline.id == focus_outline_id:
                # 当前大纲：完整展开
                lines.append(f"\n### {outline.label} [当前焦点]\n")
                lines.append("| 编号 | 子任务 | 状态 | 结果摘要 |")
                lines.append("|------|--------|------|----------|")
                self._append_leaf_rows(outline.id, lines, STATUS_TEXT)
            else:
                # 其他大纲：仅摘要
                descendants = self.get_all_descendants(outline.id)
                total = len([d for d in descendants
                             if d not in {s for s, t in self._h_edges}
                             and d not in {"req", "analysis"}])
                done = len([d for d in descendants
                            if self._nodes.get(d) and
                            self._nodes[d].status in ("completed", "skipped")
                            and d not in {s for s, t in self._h_edges}])
                lines.append(
                    f"- {outline.label}: {done}/{total} 已完成"
                )

        # 附加：跨大纲依赖的详细结果
        dep_ids = self.get_dependencies(focus_node_id)
        cross_deps = []
        for dep_id in dep_ids:
            dep_outline = self.get_ancestor_at_depth(dep_id, 2)
            if dep_outline and dep_outline != focus_outline_id:
                cross_deps.append(dep_id)

        if cross_deps:
            lines.append("\n### 跨模块依赖（详细）\n")
            lines.append("| 编号 | 任务 | 状态 | 结果 |")
            lines.append("|------|------|------|------|")
            for dep_id in cross_deps:
                dep_node = self._nodes.get(dep_id)
                if dep_node:
                    st = STATUS_TEXT.get(dep_node.status, dep_node.status)
                    result_sum = ""
                    if dep_node.result:
                        result_sum = dep_node.result[:120].replace("\n", " ")
                        if len(dep_node.result) > 120:
                            result_sum += "..."
                    lines.append(
                        f"| {dep_id} | {dep_node.label} | {st} | {result_sum} |"
                    )

        # 全局进度概览
        all_leaves = self.get_leaf_nodes()
        total = len(all_leaves)
        done = len([n for n in all_leaves if n.status in ("completed", "skipped")])
        blocked = len([n for n in all_leaves if n.status == "blocked"])
        lines.append(f"\n**全局进度**: {done}/{total} 完成")
        if blocked:
            lines.append(f", {blocked} 阻塞")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"TaskGraph(id={self.id!r}, title={self.title!r}, "
            f"nodes={len(self._nodes)}, h_edges={len(self._h_edges)}, "
            f"d_edges={len(self._d_edges)})"
        )
