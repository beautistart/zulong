# File: zulong/memory/graph_adapters.py
# 记忆图谱后端适配器
#
# 将各个现有记忆模块的数据投射为 MemoryGraph 的 GraphNode 和 GraphEdge。
# 适配器是只读的 -- 只从后端读取数据投射到图中，不修改后端。
# MemoryGraph 是后端的投影，可随时重建。

import logging
import re
import time
from typing import Any, Optional, List, Dict, Tuple
from abc import ABC, abstractmethod

from .memory_graph import (
    MemoryGraph, GraphNode, NodeType, EdgeType, Importance,
)

logger = logging.getLogger(__name__)


# ============================================================
# 适配器基类
# ============================================================

class BaseGraphAdapter(ABC):
    """后端适配器基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """适配器名称"""
        ...

    @abstractmethod
    def sync(self, graph: MemoryGraph, source: Any) -> int:
        """全量同步: 将后端数据投射到图中

        Args:
            graph: MemoryGraph 实例
            source: 后端数据源实例 (可为 None, 适配器自行获取单例)

        Returns:
            int: 新增/更新的节点数
        """
        ...

    def incremental_sync(self, graph: MemoryGraph, event_type: str, data: dict):
        """增量同步: 根据事件更新图 (可选实现)"""
        pass


# ============================================================
# TaskGraph 适配器
# ============================================================

class TaskGraphAdapter(BaseGraphAdapter):
    """将 pipeline.TaskGraph 投射到 MemoryGraph

    - TaskNode -> GraphNode(TASK)
    - FileRef -> GraphNode(FILE) + REFERENCE 边
    - h_edges -> HIERARCHY 边 (protected)
    - d_edges -> DEPENDENCY 边 (protected)
    """

    @property
    def name(self) -> str:
        return "task_graph"

    def sync(self, graph: MemoryGraph, source: Any) -> int:
        """全量同步 TaskGraph

        Args:
            source: TaskGraph 实例 (必须显式传入, 非单例)
        """
        if source is None:
            return 0

        count = 0
        graph_id = getattr(source, 'id', '')

        # 地址继承：查找任务根节点在 MemoryGraph 中的完整路径
        # 根节点可能用完整路径作为 node_id（如 "dialogue:session_xxx/task:tg_xxx"）
        # 也可能只用 "task:{graph_id}"（未分配 session 时）
        task_root_mg_id = f"task:{graph_id}"
        root_node = graph.get_node(task_root_mg_id)
        if root_node is None:
            # 尝试查找完整路径格式（task_create_plan 已创建的情况）
            for nid, node in graph._nodes.items():
                if (node.node_type == NodeType.TASK
                        and node.metadata.get("graph_id") == graph_id):
                    root_node = node
                    task_root_mg_id = nid
                    break

        parent_prefix = ""
        if root_node:
            parent_prefix = root_node.metadata.get("full_path", task_root_mg_id)

        # 计算每个节点在 TaskGraph 中的深度，用于确定 sub_type
        _child_to_parent: Dict[str, str] = {}
        for _p, _c in source._h_edges:
            _child_to_parent[_c] = _p

        def _node_depth(nid: str) -> int:
            """计算节点在 TaskGraph 层级中的深度（根=0）"""
            d = 0
            cur = nid
            visited = set()
            while cur in _child_to_parent and cur not in visited:
                visited.add(cur)
                cur = _child_to_parent[cur]
                d += 1
            return d

        # 投射节点（使用完整路径地址）
        for node_id, task_node in source._nodes.items():
            if parent_prefix:
                full_node_id = f"{parent_prefix}/{node_id}"
            else:
                # 无父级路径时，用 graph_id 限定避免跨图碰撞
                full_node_id = f"task:{graph_id}/{node_id}" if graph_id else f"task:{node_id}"

            # 如果节点已存在，跳过（避免重复创建）
            if graph.has_node(full_node_id):
                # 更新已有节点的 metadata
                existing = graph.get_node(full_node_id)
                if existing:
                    existing.metadata["status"] = task_node.status
                    existing.metadata["desc"] = task_node.desc[:200] if task_node.desc else ""
                    # 注意：sync 不更新 last_accessed，保留原始时间戳
                count += 1
                continue

            gnode = GraphNode(
                node_id=full_node_id,
                node_type=NodeType.TASK,
                label=task_node.label,
                backend_ref=f"task_graph:{graph_id}/{node_id}" if graph_id else f"task_graph:{node_id}",
                metadata={
                    "type": task_node.type,
                    "status": task_node.status,
                    "desc": task_node.desc[:200] if task_node.desc else "",
                    "graph_id": graph_id,
                    "graph_address": f"tg:{graph_id}/task:{node_id}" if graph_id else "",
                    "full_path": full_node_id,
                    "parent_session": parent_prefix.split("/")[0] if "/" in parent_prefix else "",
                    "sub_type": (
                        "task_root" if _node_depth(node_id) == 0
                        else "subtask" if _node_depth(node_id) == 1
                        else "sub_subtask"
                    ),
                },
            )
            graph.add_node(gnode, touch=False)
            count += 1

            # 关联文件 -> FILE 节点 + REFERENCE 边
            for fref in task_node.files:
                file_id = f"file:{fref.path}"
                if not graph.has_node(file_id):
                    fnode = GraphNode(
                        node_id=file_id,
                        node_type=NodeType.FILE,
                        label=fref.name,
                        backend_ref=f"file:{fref.path}",
                        metadata={"path": fref.path},
                    )
                    graph.add_node(fnode, touch=False)
                    count += 1
                graph.add_edge(
                    full_node_id, file_id,
                    EdgeType.REFERENCE, weight=0.8,
                )

        # 投射层级边（使用完整路径地址）
        for parent_id, child_id in source._h_edges:
            if parent_prefix:
                p_full = f"{parent_prefix}/{parent_id}"
                c_full = f"{parent_prefix}/{child_id}"
            elif graph_id:
                p_full = f"task:{graph_id}/{parent_id}"
                c_full = f"task:{graph_id}/{child_id}"
            else:
                p_full = f"task:{parent_id}"
                c_full = f"task:{child_id}"
            graph.add_edge(
                p_full, c_full,
                EdgeType.HIERARCHY, weight=1.0, protected=True,
            )

        # 投射依赖边
        for dep_edge in source._d_edges:
            if parent_prefix:
                s_full = f"{parent_prefix}/{dep_edge.s}"
                t_full = f"{parent_prefix}/{dep_edge.t}"
            elif graph_id:
                s_full = f"task:{graph_id}/{dep_edge.s}"
                t_full = f"task:{graph_id}/{dep_edge.t}"
            else:
                s_full = f"task:{dep_edge.s}"
                t_full = f"task:{dep_edge.t}"
            graph.add_edge(
                s_full, t_full,
                EdgeType.DEPENDENCY, weight=1.0, protected=True,
                metadata={"via": dep_edge.via, "cross": dep_edge.cross},
            )

        return count

    def incremental_sync(self, graph: MemoryGraph, event_type: str, data: dict):
        """TaskGraph 变更时增量同步"""
        graph_id = data.get("_graph_id", "")

        def _resolve_mg_id(node_id: str) -> str:
            """将 TaskGraph node_id 映射到 MemoryGraph node_id（优先查路径地址）"""
            # 1. 先在图中搜索路径地址节点
            for nid, nd in graph._nodes.items():
                if (nd.node_type == NodeType.TASK
                        and nd.metadata.get("graph_id") == graph_id
                        and nid.endswith(f"/{node_id}")):
                    return nid
            # 2. 用 graph_id 限定的 fallback
            if graph_id:
                candidate = f"task:{graph_id}/{node_id}"
                if graph.has_node(candidate):
                    return candidate
            # 3. 最终 fallback
            return f"task:{graph_id}/{node_id}" if graph_id else f"task:{node_id}"

        if event_type == "node_add":
            node_id = data.get("node_id", "")
            mg_id = _resolve_mg_id(node_id)
            gnode = GraphNode(
                node_id=mg_id,
                node_type=NodeType.TASK,
                label=data.get("label", ""),
                backend_ref=f"task_graph:{graph_id}/{node_id}" if graph_id else f"task_graph:{node_id}",
                metadata={
                    "type": data.get("type", "task"),
                    "status": data.get("status", "pending"),
                    "desc": data.get("desc", "")[:200],
                    "graph_id": graph_id,
                    "full_path": mg_id,
                    "graph_address": f"tg:{graph_id}/task:{node_id}" if graph_id else "",
                },
            )
            graph.add_node(gnode)

        elif event_type == "node_update":
            node_id = data.get("node_id", "")
            mg_id = _resolve_mg_id(node_id)
            gnode = graph.get_node(mg_id)
            if gnode:
                gnode.metadata["status"] = data.get("status", gnode.metadata.get("status"))
                if data.get("result"):
                    gnode.metadata["result_preview"] = data["result"][:100]
                gnode.last_accessed = time.time()

        elif event_type == "h_edge_add":
            parent = data.get("parent", "")
            child = data.get("child", "")
            graph.add_edge(
                _resolve_mg_id(parent), _resolve_mg_id(child),
                EdgeType.HIERARCHY, weight=1.0, protected=True,
            )

        elif event_type == "d_edge_add":
            graph.add_edge(
                _resolve_mg_id(data.get('source', '')),
                _resolve_mg_id(data.get('target', '')),
                EdgeType.DEPENDENCY, weight=1.0, protected=True,
                metadata={"via": data.get("via", "")},
            )


# ============================================================
# KnowledgeGraph 适配器
# ============================================================

class KnowledgeGraphAdapter(BaseGraphAdapter):
    """将 KnowledgeGraph 投射到 MemoryGraph

    - Entity(PERSON) -> GraphNode(PERSON)
    - Entity(CONCEPT) -> GraphNode(CONCEPT)
    - Entity(其他) -> GraphNode(KNOWLEDGE)
    - Relation(CAUSED) -> CAUSAL 边
    - Relation(RELATED_TO) -> ASSOCIATION 边
    - Relation(其他) -> REFERENCE 边
    """

    @property
    def name(self) -> str:
        return "knowledge_graph"

    def sync(self, graph: MemoryGraph, source: Any) -> int:
        if source is None:
            try:
                from .knowledge_graph import KnowledgeGraph
                source = KnowledgeGraph()
                if not hasattr(source, '_initialized') or not source._initialized:
                    return 0
            except Exception:
                return 0

        count = 0

        # 投射实体
        for entity_id, entity in source.entities.items():
            # 根据实体类型映射到 NodeType
            if entity.entity_type.value == "person":
                node_type = NodeType.PERSON
            elif entity.entity_type.value == "concept":
                node_type = NodeType.CONCEPT
            else:
                node_type = NodeType.KNOWLEDGE

            gnode = GraphNode(
                node_id=f"kg:{entity_id}",
                node_type=node_type,
                label=entity.name,
                backend_ref=f"knowledge_graph:{entity_id}",
                metadata={
                    "entity_type": entity.entity_type.value,
                    "confidence": entity.confidence,
                    "source": entity.source,
                    "attributes": dict(entity.attributes),
                },
            )
            graph.add_node(gnode, touch=False)
            count += 1

        # 投射关系边
        for u, v, data in source.graph.edges(data=True):
            rel_type = data.get("relation_type", "related_to")

            # 映射关系类型
            if rel_type == "caused":
                edge_type = EdgeType.CAUSAL
            elif rel_type == "related_to":
                edge_type = EdgeType.ASSOCIATION
            else:
                edge_type = EdgeType.REFERENCE

            graph.add_edge(
                f"kg:{u}", f"kg:{v}",
                edge_type,
                weight=data.get("weight", 0.8),
                metadata={"relation_type": rel_type},
            )

        return count


# ============================================================
# Dialogue 适配器
# ============================================================

class DialogueAdapter(BaseGraphAdapter):
    """将 ShortTermMemory 对话轮次投射到 MemoryGraph

    三层结构:
    - Session (对话会话): 一组相关话题的对话轮次容器
    - Round (对话轮次): 一次 USER_SPEECH -> 完整响应
    - Sub-dialogue (子对话): Agent 的一个重要推理步骤

    层级关系:
    - session →[HIERARCHY]→ round →[HIERARCHY]→ sub_dialogue
    - session →[TEMPORAL]→ session (跨会话时间线)
    - round →[TEMPORAL]→ round (同一会话内时间线)
    - sub_dialogue →[REFERENCE]→ task_node (关联任务节点)

    话题边界检测:
    - 恢复类消息(继续/接着) → 归入被恢复任务所在的原 session
    - 与上一轮话题相关 → 归入同一 session
    - 无关话题 → 创建新 session
    """

    # 话题边界检测用的无关词(排除后再做关键词匹配)
    _STOPWORDS = {'的', '了', '吗', '呢', '吧', '啊', '嗯', '好',
                  '请', '帮', '我', '你', '一下', '一个', '这个', '那个'}

    # ---- 重要信息自动检测规则（按优先级排序，首个匹配即返回） ----
    _IMPORTANCE_RULES: List[Tuple[re.Pattern, Importance, str]] = [
        # must_remember: 用户显式要求
        (re.compile(r'帮我记住|记得|别忘了|一定要记住|不要忘记|你要记得', re.IGNORECASE),
         Importance.MUST_REMEMBER, "explicit_remember"),
        # identity: 身份信息
        (re.compile(r'我叫|我的名字|我姓|我是.{0,4}[人]|我今年.{0,4}岁|我的名字叫', re.IGNORECASE),
         Importance.IDENTITY, "identity"),
        # fact: 客观事实
        (re.compile(r'我家在|我住在|我的电话|我的手机|我的生日|我的地址|我的邮箱|号码是', re.IGNORECASE),
         Importance.FACT, "fact"),
        # important: 偏好/承诺/任务指令
        (re.compile(r'我喜欢|我不喜欢|我讨厌|我爱|我习惯|我每次都|我答应|我保证|以后都', re.IGNORECASE),
         Importance.IMPORTANT, "preference"),
        # trivial: 纯语气词/极短回复/日常闲聊
        (re.compile(r'^(嗯|好|好的|哦|ok|OK|行|是的|对|谢谢|感谢|拜拜|再见|没了|没有了|可以|你好|嗨|哈喽|hi|hello)$', re.IGNORECASE),
         Importance.TRIVIAL, "filler"),
        # trivial: 闲聊问候/寒暄（短句无实质内容）
        (re.compile(r'^(你好|您好|嗨|哈喽|hi |hello |早上好|下午好|晚上好|晚安|早安|最近怎么样|你好吗|最近如何|在吗|在干嘛|忙吗|干嘛呢|在不在|有空吗|聊聊天|随便聊聊|没什么事|随便问问)$', re.IGNORECASE),
         Importance.TRIVIAL, "greeting"),
    ]

    @classmethod
    def _detect_importance(cls, text: str) -> Tuple[Importance, List[Dict]]:
        """轻量级重要信息检测

        Args:
            text: 待检测文本

        Returns:
            (importance, detected_entities)
            detected_entities: [{"type": "identity"|"fact"|..., "text": "匹配片段"}]
        """
        if not text or not text.strip():
            return Importance.TRIVIAL, []

        text_stripped = text.strip()

        # 极短内容（< 3 字符且不在规则中）
        if len(text_stripped) < 3:
            return Importance.TRIVIAL, []

        for pattern, importance, entity_type in cls._IMPORTANCE_RULES:
            match = pattern.search(text_stripped)
            if match:
                entities = [{"type": entity_type, "text": match.group(0)}]
                return importance, entities

        return Importance.NORMAL, []

    @property
    def name(self) -> str:
        return "dialogue"

    def sync(self, graph: MemoryGraph, source: Any) -> int:
        if source is None:
            try:
                from .short_term_memory import ShortTermMemory
                source = ShortTermMemory()
                if not hasattr(source, '_initialized') or not source._initialized:
                    return 0
            except Exception:
                return 0

        count = 0
        prev_node_id = None

        # 按轮次序号排序
        sorted_turns = sorted(source._turn_index.keys())
        if not sorted_turns:
            return 0

        # 创建一个历史 session 节点（避免旧对话成为孤立子树）
        history_session_id = "dialogue:session_history"
        if not graph.has_node(history_session_id):
            session_node = GraphNode(
                node_id=history_session_id,
                node_type=NodeType.DIALOGUE,
                label="历史会话（STM导入）",
                metadata={
                    "sub_type": "session",
                    "topic_summary": "从 ShortTermMemory 导入的历史对话",
                    "round_count": len(sorted_turns),
                },
            )
            graph.add_node(session_node, touch=False)
            count += 1

        for turn_id in sorted_turns:
            node_id = f"dialogue:{turn_id}"

            gnode = GraphNode(
                node_id=node_id,
                node_type=NodeType.DIALOGUE,
                label=f"对话轮次 {turn_id}",
                backend_ref=f"stm:turn_{turn_id}",
                metadata={
                    "turn_id": turn_id,
                    "sub_type": "round",
                    "session_id": history_session_id,
                },
            )
            graph.add_node(gnode, touch=False)
            count += 1

            # HIERARCHY: session → round
            graph.add_edge(
                history_session_id, node_id,
                EdgeType.HIERARCHY, weight=1.0, protected=True,
            )

            # 与前一轮建立 TEMPORAL 边
            if prev_node_id:
                graph.add_edge(
                    prev_node_id, node_id,
                    EdgeType.TEMPORAL, weight=1.0, protected=True,
                )
            prev_node_id = node_id

        return count

    # ── 增量同步方法 (运行时 Gatekeeper / Orchestrator 调用) ──

    # ---------- Session 管理 ----------

    def ensure_session(
        self, graph: MemoryGraph, text: str,
        task_graph_id: Optional[str] = None,
    ) -> str:
        """确保当前对话属于正确的 Session 节点

        话题边界检测策略:
        1. 如果有 task_graph_id → 通过 TaskGraph ID 查找关联的 session
        2. 如果与当前活跃 session 的最新 round 话题相关 → 沿用
        3. 否则 → 创建新 session

        Args:
            graph: MemoryGraph 实例
            text: 用户输入文本
            task_graph_id: TaskGraph ID（用于定位关联的 session）

        Returns:
            session_id: 当前应使用的 session 节点 ID
        """
        # 收集所有 session 节点
        all_sessions = self._get_sessions(graph)

        # 1) 通过 TaskGraph ID 查找关联的 session
        if task_graph_id:
            task_node_id = f"task:{task_graph_id}"
            if graph.has_node(task_node_id):
                # 查找引用该任务节点的对话会话
                for sess in all_sessions:
                    if graph.has_edge(sess.node_id, task_node_id):
                        logger.debug(
                            f"[DialogueAdapter] 通过 TaskGraph ID {task_graph_id} "
                            f"找到 session {sess.node_id}"
                        )
                        return sess.node_id

        # 2) 取最新的活跃 session
        current_session = self._get_latest_session(all_sessions)

        if current_session:
            # 检查话题相关性
            if self._is_same_topic(graph, current_session, text):
                logger.debug(
                    f"[DialogueAdapter] 话题延续 → {current_session.node_id}"
                )
                return current_session.node_id

        # 4) 创建新 session
        session_id = self._create_session(
            graph, text, prev_session_id=current_session.node_id if current_session else None,
        )
        logger.info(f"[DialogueAdapter] 新对话会话: {session_id}")
        return session_id

    # ---------- L2 Session 分配（Embedding 相似度） ----------

    def assign_session_by_similarity(
        self, graph: MemoryGraph, round_id: str,
        user_input: str, response: str,
        similarity_threshold: float = 0.55,
    ) -> str:
        """由 L2 调用：基于 Embedding 余弦相似度将 round 归属到正确的 Session

        两级策略:
        1. Embedding 相似度 — 与各 session 最新 round 的向量比较
        2. 创建新 session — 无匹配时新建

        注意：任务恢复场景下的 session 绑定由 TaskResumeTool 在恢复后
        通过 bind_session_to_task() 事后完成，不在此处预判。

        Args:
            graph:  MemoryGraph 实例
            round_id: 当前 round 节点 ID
            user_input: 用户输入文本
            response: 模型回复文本
            similarity_threshold: 余弦相似度阈值（>=此值视为同话题）

        Returns:
            session_id: 最终归属的 session 节点 ID
        """
        all_sessions = self._get_sessions(graph)

        # ── 策略 1: Embedding 余弦相似度 ──
        best_session_id = None
        best_score = -1.0

        try:
            from zulong.memory.embedding_manager import EmbeddingModelManager
            emb_mgr = EmbeddingModelManager()

            # 对当前输入做向量编码
            import numpy as np
            query_vec = emb_mgr.encode_query(user_input)

            for sess in all_sessions:
                # 取该 session 最新 round 的文本
                latest_text = self._latest_round_text(graph, sess.node_id)
                if not latest_text:
                    continue
                doc_vec = emb_mgr.encode_document(latest_text)

                # 余弦相似度
                cos_sim = float(
                    np.dot(query_vec, doc_vec)
                    / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-9)
                )
                if cos_sim > best_score:
                    best_score = cos_sim
                    best_session_id = sess.node_id

            if best_session_id and best_score >= similarity_threshold:
                logger.info(
                    f"[SessionAssign] Embedding 匹配: score={best_score:.3f} "
                    f"→ {best_session_id}"
                )
                self._link_round_to_session(graph, round_id, best_session_id)
                return best_session_id

        except Exception as e:
            logger.warning(f"[SessionAssign] Embedding 匹配异常，降级创建新 session: {e}")

        # ── 策略 2: 创建新 session ──
        prev_session_id = all_sessions[-1].node_id if all_sessions else None
        new_session_id = self._create_session(
            graph, user_input,
            prev_session_id=prev_session_id,
        )
        self._link_round_to_session(graph, round_id, new_session_id)
        logger.info(f"[SessionAssign] 新建 session: {new_session_id}")
        return new_session_id

    def _link_round_to_session(
        self, graph: MemoryGraph, round_id: str, session_id: str,
    ):
        """将 round 节点挂载到 session（HIERARCHY 边 + 元数据更新 + 地址继承）"""
        if not graph.has_node(session_id) or not graph.has_node(round_id):
            return

        # 建立 HIERARCHY 边: session → round
        if not graph.has_edge(session_id, round_id):
            graph.add_edge(
                session_id, round_id,
                EdgeType.HIERARCHY, weight=1.0, protected=True,
            )

        # 更新 round 的 session_id 元数据
        round_node = graph.get_node(round_id)
        if round_node:
            round_node.metadata["session_id"] = session_id
            # 地址继承：更新 round 的完整路径
            # 格式：dialogue:session_xxx/dialogue:round_xxx
            round_node.metadata["full_path"] = f"{session_id}/{round_id}"
            # 同步更新 task_graph_address 为完整路径（如果之前是相对路径）
            old_tg_addr = round_node.metadata.get("task_graph_address", "")
            if old_tg_addr and not old_tg_addr.startswith("dialogue:"):
                round_node.metadata["task_graph_address"] = f"{session_id}/{old_tg_addr}"

            # 地址传播：查找通过 REFERENCE 边关联到本 round 的任务图节点
            # 将它们的前缀也更新为 session 路径
            self._propagate_address_to_tasks(graph, round_id, session_id)

        # 更新 session 的 round_count 和 last_active_at
        sess_node = graph.get_node(session_id)
        if sess_node:
            sess_node.metadata["round_count"] = sess_node.metadata.get("round_count", 0) + 1
            sess_node.metadata["last_active_at"] = time.time()

    def _propagate_address_to_tasks(
        self, graph: MemoryGraph, round_id: str, session_id: str,
    ):
        """将 session 地址传播到与 round 关联的所有任务图节点
        
        当 round 被分配到 session 后，通过 HIERARCHY/REFERENCE 边找到关联的任务节点，
        将其地址前缀从 "task:tg_xxx" 更新为 "dialogue:session_xxx/task:tg_xxx"。
        """
        if not hasattr(graph, '_graph'):
            return

        # 查找所有通过 HIERARCHY 或 REFERENCE 边与本 round 相连的任务节点
        linked_nodes = []
        for neighbor_id in graph._graph.neighbors(round_id):
            neighbor = graph.get_node(neighbor_id)
            if neighbor and neighbor.node_type == NodeType.TASK:
                edge_data = graph._graph[round_id].get(neighbor_id, {})
                if edge_data.get("edge_type") in ("hierarchy", "reference"):
                    linked_nodes.append(neighbor)

        # 更新每个任务节点的地址（node_id 不变，只更新 metadata 中的 full_path）
        for task_node in linked_nodes:
            old_path = task_node.metadata.get("full_path", task_node.node_id)
            # 如果任务节点地址还没携带 session 路径，则更新
            if not old_path.startswith("dialogue:session_"):
                new_path = f"{session_id}/{task_node.node_id}"
                task_node.metadata["full_path"] = new_path
                task_node.metadata["parent_session"] = session_id
                logger.debug(
                    f"[地址传播] {task_node.node_id} → full_path={new_path}"
                )
                # 递归传播到子任务节点
                self._propagate_address_to_task_children(graph, task_node.node_id, session_id)

    def _propagate_address_to_task_children(
        self, graph: MemoryGraph, parent_node_id: str, session_id: str = "",
    ):
        """将地址传播到任务节点的子节点（HIERARCHY 边下游）
        
        格式：parent_full_path/child_id
        """
        if not hasattr(graph, '_graph'):
            return

        # 查找所有子任务节点（HIERARCHY 边）
        if parent_node_id not in graph._graph:
            return

        for child_id in graph._graph.successors(parent_node_id):
            edge_data = graph._graph[parent_node_id].get(child_id, {})
            if edge_data.get("edge_type") == "hierarchy":
                child_node = graph.get_node(child_id)
                if child_node and child_node.node_type == NodeType.TASK:
                    # 子节点地址 = 父节点完整路径/子节点ID
                    parent_node = graph.get_node(parent_node_id)
                    if parent_node:
                        parent_path = parent_node.metadata.get("full_path", parent_node_id)
                        new_child_path = f"{parent_path}/{child_node.node_id}"
                        child_node.metadata["full_path"] = new_child_path
                        if session_id:
                            child_node.metadata["parent_session"] = session_id
                        logger.debug(
                            f"[地址传播] 子节点 {child_node.node_id} → {new_child_path}"
                        )
                        # 递归传播到更深层
                        self._propagate_address_to_task_children(graph, child_node.node_id, session_id)

    def _latest_round_text(self, graph: MemoryGraph, session_id: str) -> Optional[str]:
        """获取 session 下最新 round 的用户输入文本（用于 Embedding 比较）"""
        if not hasattr(graph, '_graph'):
            return None
        successors = list(graph._graph.successors(session_id))
        rounds = []
        for nid in successors:
            node = graph.get_node(nid)
            if node and node.metadata.get("sub_type") == "round":
                rounds.append(node)
        if not rounds:
            return None
        rounds.sort(key=lambda n: n.created_at, reverse=True)
        return rounds[0].metadata.get("goal", "") or rounds[0].metadata.get("user_text", "")

    def _get_sessions(self, graph: MemoryGraph) -> List[GraphNode]:
        """获取所有 session 节点，按创建时间排序"""
        dialogue_nodes = graph.get_nodes_by_type(NodeType.DIALOGUE)
        sessions = [n for n in dialogue_nodes if n.metadata.get("sub_type") == "session"]
        sessions.sort(key=lambda n: n.created_at)
        return sessions

    def _get_latest_session(self, sessions: List[GraphNode]) -> Optional[GraphNode]:
        """获取最新的 session"""
        return sessions[-1] if sessions else None

    def _create_session(
        self, graph: MemoryGraph, text: str,
        prev_session_id: Optional[str] = None,
        bound_task_id: Optional[str] = None,
    ) -> str:
        """创建新的对话会话节点"""
        import hashlib
        ts = int(time.time() * 1000)
        hash_suffix = hashlib.md5(f"{text}{ts}".encode()).hexdigest()[:8]
        session_id = f"dialogue:session_{hash_suffix}"

        node = GraphNode(
            node_id=session_id,
            node_type=NodeType.DIALOGUE,
            label=text[:60],
            metadata={
                "sub_type": "session",
                "topic_summary": text[:200],
                "bound_task_id": bound_task_id or "",
                "round_count": 0,
                "full_path": session_id,  # Session 是地址根节点
            },
        )
        graph.add_node(node)

        # TEMPORAL 边连接上一个 session
        if prev_session_id and graph.has_node(prev_session_id):
            graph.add_edge(
                prev_session_id, session_id,
                EdgeType.TEMPORAL, weight=1.0, protected=True,
            )

        return session_id

    def bind_session_to_task(
        self, graph: MemoryGraph, session_id: str, task_id: str,
    ):
        """将 session 绑定到一个复杂任务 ID（挂起时回溯用）

        除 metadata 记录外，同时创建 REFERENCE 边使图遍历可发现此关系，
        并更新任务节点的地址前缀为 session 路径。
        """
        node = graph.get_node(session_id)
        if node:
            node.metadata["bound_task_id"] = task_id
            # 地址继承：查找并更新任务根节点地址
            task_root_id = f"task:{task_id}"
            task_node = graph.get_node(task_root_id)
            if task_node is None:
                # 尝试查找完整路径格式
                for nid, nd in graph._nodes.items():
                    if (nd.node_type == NodeType.TASK
                            and nd.metadata.get("graph_id") == task_id):
                        task_node = nd
                        task_root_id = nid
                        break

            if task_node:
                # 更新任务节点的地址和路径
                if not task_node.node_id.startswith("dialogue:"):
                    new_path = f"{session_id}/{task_node.node_id}"
                    task_node.metadata["full_path"] = new_path
                    task_node.metadata["parent_session"] = session_id
                    logger.debug(
                        f"[bind_session_to_task] 地址继承: {task_node.node_id} → {new_path}"
                    )

                # 创建 REFERENCE 边（使 BFS/遍历能发现此跨空间绑定）
                if not graph.has_edge(session_id, task_node.node_id):
                    graph.add_edge(
                        session_id, task_node.node_id, EdgeType.REFERENCE,
                        weight=0.9, metadata={"binding_type": "session_task"},
                    )

                # 传播地址到任务子节点
                self._propagate_address_to_task_children(graph, task_node.node_id)

    def _is_same_topic(
        self, graph: MemoryGraph, session: GraphNode, new_text: str,
    ) -> bool:
        """判断新消息是否与当前 session 属于同一话题

        策略:
        1. session 最近一个 round 的 goal 与新文本做关键词交集
        2. 交集 >= 2 个有意义词 → 同一话题
        3. 如果 session 绑定了任务且任务仍挂起 → 同一话题
        """
        import re

        # 如果 session 绑定了任务，检查任务是否仍存在
        bound_task = session.metadata.get("bound_task_id", "")
        if bound_task:
            # 有绑定任务的 session 默认延续（直到用户明确开始新话题）
            return True

        # 取 session 下最新 round 的 goal
        latest_goal = self._get_latest_round_goal(graph, session.node_id)
        if not latest_goal:
            # session 还没有 round（刚创建），使用 topic_summary
            latest_goal = session.metadata.get("topic_summary", "")

        if not latest_goal:
            return False

        # 关键词交集检测
        def extract_keywords(text: str) -> set:
            segments = re.split(r'[，、。！？\s,.\-:;/\\()\[\]{}""\'\']+', text.lower())
            return {seg for seg in segments if len(seg) >= 2 and seg not in self._STOPWORDS}

        old_kw = extract_keywords(latest_goal)
        new_kw = extract_keywords(new_text)
        overlap = old_kw & new_kw

        return len(overlap) >= 2

    def _get_latest_round_goal(
        self, graph: MemoryGraph, session_id: str,
    ) -> Optional[str]:
        """获取 session 下最新 round 的 goal 文本"""
        # 查找 session 的 HIERARCHY 子节点中 sub_type=round 的最新一个
        if not hasattr(graph, '_graph'):
            return None
        successors = list(graph._graph.successors(session_id))
        rounds = []
        for nid in successors:
            node = graph.get_node(nid)
            if node and node.metadata.get("sub_type") == "round":
                rounds.append(node)
        if not rounds:
            return None
        rounds.sort(key=lambda n: n.created_at, reverse=True)
        return rounds[0].metadata.get("goal", "")

    # ---------- Round 管理 ----------

    def add_round(
        self, graph: MemoryGraph, request_id: str, goal: str,
        prev_round_id: Optional[str] = None,
        task_graph_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """创建一个对话轮次节点（一次 USER_SPEECH -> 完整响应）

        Args:
            graph: MemoryGraph 实例
            request_id: 请求 ID
            goal: 用户输入文本
            prev_round_id: 上一轮对话节点 ID（用于建立 TEMPORAL 边）
            task_graph_id: 关联的 TaskGraph ID（用于地址继承）
            session_id: 所属 session 节点 ID（用于建立 HIERARCHY 边）

        Returns:
            对话轮次节点 ID
        """
        round_id = f"dialogue:round_{request_id}"
        node = GraphNode(
            node_id=round_id,
            node_type=NodeType.DIALOGUE,
            label=goal[:80],
            backend_ref=f"stm:request_{request_id}",
            metadata={
                "sub_type": "round",
                "request_id": request_id,
                "goal": goal,
                "task_graph_address": f"tg:{task_graph_id}" if task_graph_id else "",
                "session_id": session_id or "",
                "full_path": round_id,  # 完整地址路径（session 分配后更新）
            },
        )
        graph.add_node(node)

        # 重要信息自动检测
        importance, entities = self._detect_importance(goal)
        if importance != Importance.NORMAL:
            graph.set_importance(round_id, importance)
        if entities:
            node_obj = graph.get_node(round_id)
            if node_obj:
                node_obj.metadata["detected_entities"] = entities

        # TEMPORAL 边连接上一轮（同一 session 内的时间线）
        if prev_round_id and graph.has_node(prev_round_id):
            graph.add_edge(
                prev_round_id, round_id,
                EdgeType.TEMPORAL, weight=1.0, protected=True,
            )

        # HIERARCHY 边: session → round
        if session_id and graph.has_node(session_id):
            graph.add_edge(
                session_id, round_id,
                EdgeType.HIERARCHY, weight=1.0, protected=True,
            )
            # 更新 session 的 round_count
            sess_node = graph.get_node(session_id)
            if sess_node:
                sess_node.metadata["round_count"] = sess_node.metadata.get("round_count", 0) + 1

        logger.debug(f"[DialogueAdapter] 创建对话轮次: {round_id} (session={session_id})")

        # 地址继承：对话 round → 任务图节点（REFERENCE 边）
        # 当存在关联的 TaskGraph 时，建立双向关联边，使 BFS 扩散能跨类型传播
        if task_graph_id:
            task_node_id = f"task:{task_graph_id}"
            if graph.has_node(task_node_id):
                graph.add_edge(
                    round_id, task_node_id,
                    EdgeType.REFERENCE, weight=0.8,
                    metadata={"link_type": "dialogue_to_task", "inherited_at": time.time()},
                )
                logger.debug(
                    f"[DialogueAdapter] 地址继承: {round_id} ──REFERENCE──> {task_node_id}"
                )
            else:
                logger.debug(
                    f"[DialogueAdapter] 任务节点不存在，暂存地址: {task_node_id}"
                )

        # 跨类型关联：对话 → KG 实体（KNOWLEDGE/PERSON）
        self._link_to_knowledge(graph, round_id, goal)

        return round_id

    def _link_to_knowledge(
        self, graph: MemoryGraph, round_id: str, text: str,
    ):
        """将对话 round 与匹配的 KG 实体建立 REFERENCE 边

        对 KG 中已存在的 KNOWLEDGE / PERSON 节点做 label 子串匹配，
        命中则创建 REFERENCE 边，使图谱产生跨类型连接。
        """
        if not text or len(text) < 4:
            return
        text_lower = text.lower()
        linked = 0
        for node_type in (NodeType.KNOWLEDGE, NodeType.PERSON, NodeType.CONCEPT):
            for node in graph.get_nodes_by_type(node_type):
                if not node.label or len(node.label) < 2:
                    continue
                if node.label.lower() in text_lower:
                    graph.add_edge(
                        round_id, node.node_id,
                        EdgeType.REFERENCE, weight=0.6,
                    )
                    linked += 1
                    if linked >= 5:
                        return  # 限制每轮最多 5 条跨类型边

    def add_sub_dialogue(
        self, graph: MemoryGraph, round_id: str, turn: int,
        tool_name: Optional[str] = None, task_node_id: Optional[str] = None,
        content: Optional[str] = None, role: Optional[str] = None,
    ) -> str:
        """创建子对话节点（agent 的一个重要推理步骤）

        Args:
            graph: MemoryGraph 实例
            round_id: 父对话轮次节点 ID
            turn: Agent 循环轮次号
            tool_name: 工具名称（用于标签）
            task_node_id: 关联的 TaskGraph 节点 ID
            content: 该轮的对话内容（assistant 回复或 tool 调用摘要）
            role: 消息角色（assistant / tool / user）

        Returns:
            子对话节点 ID
        """
        # 从 round_id 提取 request_id 部分
        round_suffix = round_id.split("_", 1)[-1] if "_" in round_id else round_id
        sub_id = f"dialogue:turn_{round_suffix}_{turn}"

        label = f"Turn {turn}"
        if tool_name:
            label = tool_name

        node = GraphNode(
            node_id=sub_id,
            node_type=NodeType.DIALOGUE,
            label=label,
            metadata={
                "sub_type": "agent_turn",
                "turn": turn,
                "tool_name": tool_name or "",
                "content": content or "",
                "role": role or "assistant",
            },
        )
        graph.add_node(node)

        # 重要信息自动检测（对子对话内容检测）
        if content:
            importance, entities = self._detect_importance(content)
            if importance != Importance.NORMAL:
                graph.set_importance(sub_id, importance)
            if entities:
                node_obj = graph.get_node(sub_id)
                if node_obj:
                    node_obj.metadata["detected_entities"] = entities

        # HIERARCHY: 子对话 -> 父轮次
        graph.add_edge(
            round_id, sub_id,
            EdgeType.HIERARCHY, weight=1.0, protected=True,
        )

        # REFERENCE: 子对话 → 关联任务节点（地址继承：子级细粒度关联）
        # 子对话是 agent 推理的重要步骤，与具体任务节点关联使检索更精准
        if task_node_id:
            task_graph_id = f"task:{task_node_id}"
            if graph.has_node(task_graph_id):
                graph.add_edge(
                    sub_id, task_graph_id,
                    EdgeType.REFERENCE, weight=0.7,
                    metadata={
                        "link_type": "sub_dialogue_to_task",
                        "linked_at": time.time(),
                    },
                )
                # 同步更新子对话节点的 metadata 记录地址
                node.metadata["task_graph_address"] = task_graph_id

        return sub_id

    def finalize_round(
        self, graph: MemoryGraph, round_id: str,
        total_turns: int, status: str = "completed",
    ):
        """完成对话轮次，更新元数据并索引到 FAISS 摘要侧车

        Args:
            graph: MemoryGraph 实例
            round_id: 对话轮次节点 ID
            total_turns: Agent 总轮次数
            status: 完成状态
        """
        node = graph.get_node(round_id)
        if node:
            node.metadata["total_turns"] = total_turns
            node.metadata["status"] = status
            node.metadata["completed_at"] = time.time()

            # ---- 修复: 将完成的对话轮次索引到 FAISS 摘要侧车 ----
            # 此前 finalize_round 不调用 index_summary()，导致冷记忆
            # FAISS 索引永远为空，_retrieve_cold() 无法检索到历史对话。
            try:
                summary_parts = [node.metadata.get("goal", "")]
                children = graph.get_children(round_id, EdgeType.HIERARCHY)
                for child in children:
                    content = child.metadata.get("content", "")
                    if content:
                        summary_parts.append(content[:100])
                summary_text = " ".join(p for p in summary_parts if p)[:500]
                if len(summary_text) > 10:
                    graph.index_summary(round_id, summary_text)
                    logger.debug(
                        f"[DialogueAdapter] 对话摘要已索引到 FAISS: {round_id} "
                        f"({len(summary_text)} chars)"
                    )
            except Exception as e:
                logger.debug(f"[DialogueAdapter] FAISS 索引跳过: {e}")

            logger.debug(
                f"[DialogueAdapter] 完成对话轮次: {round_id} "
                f"({total_turns} turns, {status})"
            )


# ============================================================
# Episode 适配器
# ============================================================

class EpisodeAdapter(BaseGraphAdapter):
    """将 EpisodicMemory 摘要投射到 MemoryGraph

    - 每个 episode -> GraphNode(EPISODE)
    - 相邻 episode -> TEMPORAL 边
    """

    @property
    def name(self) -> str:
        return "episode"

    def sync(self, graph: MemoryGraph, source: Any) -> int:
        if source is None:
            try:
                from .episodic_memory import EpisodicMemory
                source = EpisodicMemory()
                if not hasattr(source, '_initialized') or not source._initialized:
                    return 0
            except Exception:
                return 0

        count = 0
        prev_node_id = None

        sorted_episodes = sorted(
            source._episode_index.keys(),
            key=lambda k: source._episode_index[k].get("timestamp", 0),
        )

        for ep_id in sorted_episodes:
            ep_data = source._episode_index[ep_id]
            node_id = f"episode:{ep_id}"

            gnode = GraphNode(
                node_id=node_id,
                node_type=NodeType.EPISODE,
                label=ep_data.get("summary", f"Episode {ep_id}")[:60],
                backend_ref=f"episodic:{ep_id}",
                metadata={
                    "episode_id": ep_id,
                    "summary_type": ep_data.get("summary_type", "quick"),
                    "timestamp": ep_data.get("timestamp", 0),
                },
            )
            graph.add_node(gnode, touch=False)
            count += 1

            if prev_node_id:
                graph.add_edge(
                    prev_node_id, node_id,
                    EdgeType.TEMPORAL, weight=0.8,
                )
            prev_node_id = node_id

        return count


# ============================================================
# PersonProfile 适配器
# ============================================================

class PersonProfileAdapter(BaseGraphAdapter):
    """将 PersonProfileManager 投射到 MemoryGraph

    - 每个 PersonProfile -> GraphNode(PERSON)
    - 如果有 knowledge_graph_id -> REFERENCE 边连到 KG 节点
    """

    @property
    def name(self) -> str:
        return "person_profile"

    def sync(self, graph: MemoryGraph, source: Any) -> int:
        if source is None:
            try:
                from .person_profile import PersonProfileManager
                source = PersonProfileManager()
                if not hasattr(source, '_initialized') or not source._initialized:
                    return 0
            except Exception:
                return 0

        count = 0

        profiles = getattr(source, 'profiles', {})
        for profile_id, profile in profiles.items():
            node_id = f"person:{profile_id}"

            gnode = GraphNode(
                node_id=node_id,
                node_type=NodeType.PERSON,
                label=getattr(profile, 'name', None) or profile_id,
                backend_ref=f"person_profile:{profile_id}",
                metadata={
                    "first_seen": getattr(profile, 'first_seen', 0),
                    "last_seen": getattr(profile, 'last_seen', 0),
                },
            )
            graph.add_node(gnode, touch=False)
            count += 1

            # 如果有 KG 关联
            kg_id = getattr(profile, 'knowledge_graph_id', None)
            if kg_id:
                kg_node_id = f"kg:{kg_id}"
                if graph.has_node(kg_node_id):
                    graph.add_edge(
                        node_id, kg_node_id,
                        EdgeType.REFERENCE, weight=0.9,
                    )

        return count


# ============================================================
# Experience 适配器
# ============================================================

class ExperienceAdapter(BaseGraphAdapter):
    """将 ExperienceRAG 文档投射到 MemoryGraph

    - 每个 RAGDocument -> GraphNode(EXPERIENCE)
    - 无自动边 (靠语义边层补充)
    """

    @property
    def name(self) -> str:
        return "experience"

    def sync(self, graph: MemoryGraph, source: Any) -> int:
        if source is None:
            try:
                from .rag_manager import RAGManager
                rag = RAGManager()
                if not hasattr(rag, '_initialized') or not rag._initialized:
                    return 0
                source = rag.experience_rag
                if source is None:
                    return 0
            except Exception:
                return 0

        count = 0

        documents = getattr(source, 'documents', {})
        for doc_id, doc in documents.items():
            node_id = f"exp:{doc_id}"

            gnode = GraphNode(
                node_id=node_id,
                node_type=NodeType.EXPERIENCE,
                label=doc.content[:60] if hasattr(doc, 'content') else doc_id,
                backend_ref=f"experience_rag:{doc_id}",
                metadata={
                    "importance": getattr(doc, 'importance', 'pending'),
                    "domain": getattr(doc, 'domain', 'general'),
                    "memorability": getattr(doc, 'memorability', 'pending'),
                },
            )
            graph.add_node(gnode, touch=False)
            count += 1

        return count


# ============================================================
# 注册所有适配器
# ============================================================

def register_all_adapters(graph: MemoryGraph):
    """注册所有内置适配器到 MemoryGraph"""
    adapters = [
        TaskGraphAdapter(),
        KnowledgeGraphAdapter(),
        DialogueAdapter(),
        EpisodeAdapter(),
        PersonProfileAdapter(),
        ExperienceAdapter(),
    ]
    for adapter in adapters:
        graph.register_adapter(adapter.name, adapter)

    logger.info(f"[GraphAdapters] 已注册 {len(adapters)} 个适配器")
