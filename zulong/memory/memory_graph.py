# File: zulong/memory/memory_graph.py
# 记忆图谱 (Memory Graph) - 统一异构类型图集成层
#
# 将所有记忆子系统（TaskGraph / KnowledgeGraph / ShortTermMemory /
# EpisodicMemory / RAG / PersonProfile）的数据投射为统一的图节点和图边，
# 通过加权 BFS 扩散激活实现跨类型上下文发现与图注意力。
#
# 核心能力:
# - 异构节点/边类型图 (NetworkX DiGraph)
# - BFS 扩散激活算法 (从任意种子节点追溯全局关联)
# - 赫布学习 (共激活增强边权)
# - 突触修剪 (艾宾浩斯衰减 + 弱连接移除)
# - 语义边自动发现 (FAISS 侧车索引)
# - JSON 持久化 (跨会话保留图结构)

import logging
import json
import time
import os
import math
import base64
import asyncio
import threading
import atexit
from collections import deque
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

try:
    import networkx as nx
except ImportError:
    logger.error("networkx 未安装，请执行: pip install networkx")
    raise

try:
    import numpy as np
except ImportError:
    np = None
    logger.warning("numpy 未安装，embedding 功能将不可用")


# ============================================================
# 类型枚举
# ============================================================

class NodeType(Enum):
    """图节点类型"""
    TASK = "task"               # 来自 TaskGraph.TaskNode
    DIALOGUE = "dialogue"       # 来自 ShortTermMemory 对话轮次
    KNOWLEDGE = "knowledge"     # 来自 KnowledgeGraph.Entity
    EXPERIENCE = "experience"   # 来自 ExperienceRAG 文档
    EPISODE = "episode"         # 来自 EpisodicMemory 摘要
    FILE = "file"               # 来自 TaskNode.files
    CONCEPT = "concept"         # 来自 KG 中 entity_type=CONCEPT
    PERSON = "person"           # 来自 PersonProfile / KG 中 PERSON 实体
    DOCUMENT = "document"       # 预留: 未来文档/知识切片摄入
    CODE_SYMBOL = "code_symbol" # 来自 Tree-sitter AST (函数/类/方法)
    MODULE = "module"           # 目录/包/模块 (PROJECT→MODULE→FILE 层次链)


class EdgeType(Enum):
    """图边类型"""
    HIERARCHY = "hierarchy"         # 父子关系 (task h_edges)
    DEPENDENCY = "dependency"       # 数据依赖 (task d_edges)
    REFERENCE = "reference"         # 跨类型引用 (task->file, dialogue->knowledge)
    TEMPORAL = "temporal"           # 时间序列 (dialogue->dialogue, episode->episode)
    SEMANTIC = "semantic"           # 语义相似 (embedding cosine > 阈值)
    CAUSAL = "causal"               # 因果关系 (KG 中 CAUSED 关系)
    ASSOCIATION = "association"     # 赫布学习产生的关联


# 结构性边类型 -- 永不被修剪
_STRUCTURAL_EDGE_TYPES = {EdgeType.HIERARCHY, EdgeType.DEPENDENCY, EdgeType.TEMPORAL}


class Importance(Enum):
    """节点重要度标签（有序，值越大越重要）"""
    TRIVIAL = "trivial"                # 无意义闲聊（"嗯"/"好的"）
    NORMAL = "normal"                  # 普通对话
    IDENTITY = "identity"              # 身份信息（姓名/年龄/称呼）
    FACT = "fact"                      # 客观事实（日期/电话/地址）
    IMPORTANT = "important"            # 承诺/任务指令/偏好
    MUST_REMEMBER = "must_remember"    # 用户显式要求记住


class Temperature(Enum):
    """节点温度标签"""
    HOT = "hot"        # 最近被访问/激活
    WARM = "warm"      # 中等时间未激活
    COLD = "cold"      # 长期未激活


# 重要度排序映射（用于比较大小）
_IMPORTANCE_ORDER = {
    Importance.TRIVIAL: 0,
    Importance.NORMAL: 1,
    Importance.IDENTITY: 2,
    Importance.FACT: 3,
    Importance.IMPORTANT: 4,
    Importance.MUST_REMEMBER: 5,
}

# 重要度 → 衰减半衰期（小时），用于 decay_and_prune()
_IMPORTANCE_HALF_LIFE = {
    Importance.TRIVIAL: 6.0,            # 6h
    Importance.NORMAL: 24.0,            # 24h（当前默认值）
    Importance.IDENTITY: 720.0,         # 30 天
    Importance.FACT: 360.0,             # 15 天
    Importance.IMPORTANT: 168.0,        # 7 天
    Importance.MUST_REMEMBER: float('inf'),  # 永不衰减
}

# 温度阈值（秒）
_TEMPERATURE_THRESHOLDS = {
    "hot_max": 3600,        # 1 小时内 → hot
    "warm_max": 86400,      # 1h - 24h → warm
    # > 24h → cold
}


# ============================================================
# 数据结构
# ============================================================

@dataclass
class GraphNode:
    """记忆图谱节点"""
    node_id: str                    # 全局唯一, 带类型前缀: "task:o1_1", "dialogue:42"
    node_type: NodeType             # 节点类型
    label: str                      # 人类可读标签
    activation: float = 0.0         # 当前激活水平 (0.0-1.0)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    backend_ref: str = ""           # 后端来源指针, 如 "stm:turn_42"
    metadata: Dict[str, Any] = field(default_factory=dict)
    # embedding 不在 dataclass 中存储, 放在 _embeddings dict 中管理

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "label": self.label,
            "activation": self.activation,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "backend_ref": self.backend_ref,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphNode":
        """从字典反序列化"""
        return cls(
            node_id=data["node_id"],
            node_type=NodeType(data["node_type"]),
            label=data["label"],
            activation=data.get("activation", 0.0),
            created_at=data.get("created_at", time.time()),
            last_accessed=data.get("last_accessed", time.time()),
            access_count=data.get("access_count", 0),
            backend_ref=data.get("backend_ref", ""),
            metadata=data.get("metadata", {}),
        )


def _make_edge_data(
    edge_type: EdgeType,
    weight: float = 1.0,
    protected: bool = False,
    metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """构建边属性字典"""
    now = time.time()
    return {
        "edge_type": edge_type.value,
        "weight": weight,
        "created_at": now,
        "last_activated": now,
        "activation_count": 0,
        "protected": protected or (edge_type in _STRUCTURAL_EDGE_TYPES),
        "metadata": metadata or {},
    }


# ============================================================
# FAISS 摘要侧车索引
# ============================================================

class SummarySidecarIndex:
    """MemoryGraph 的 FAISS 摘要侧车索引

    只索引 Session / EPISODE 节点的摘要向量 + node_id 指针，
    不向量化记忆节点的实际内容。FAISS 命中后通过 node_id 回到图中
    BFS 下钻获取详情。

    复用项目已有的 FAISSVectorStore 和 EmbeddingModelManager。
    """

    def __init__(self, dimension: int = 512, persist_path: str = ""):
        self._dimension = dimension
        self._persist_path = persist_path
        self._store = None          # FAISSVectorStore，延迟初始化
        self._emb_manager = None    # EmbeddingModelManager，延迟初始化
        self._node_to_faiss: Dict[str, str] = {}   # node_id → faiss_doc_id
        self._faiss_to_node: Dict[str, str] = {}   # faiss_doc_id → node_id
        self._initialized = False
        # P0-3: 关键词索引 — 存储摘要文本以支持冷路径 BM25 关键词检索
        self._text_index: Dict[str, str] = {}  # node_id → summary_text

    def _ensure_init(self) -> bool:
        """延迟初始化 FAISS 和 Embedding（首次调用时加载）"""
        if self._initialized:
            return self._store is not None
        self._initialized = True
        try:
            from .base_rag_library import FAISSVectorStore
            self._store = FAISSVectorStore(dimension=self._dimension, index_type="Flat")
            return True
        except Exception as e:
            logger.warning(f"[SummarySidecarIndex] FAISS 初始化失败: {e}")
            return False

    def _get_embedding_manager(self):
        """延迟获取 EmbeddingModelManager"""
        if self._emb_manager is None:
            try:
                from .embedding_manager import get_embedding_manager
                self._emb_manager = get_embedding_manager()
            except Exception as e:
                logger.warning(f"[SummarySidecarIndex] Embedding 模型加载失败: {e}")
        return self._emb_manager

    def add_summary(self, node_id: str, summary_text: str) -> bool:
        """为节点添加摘要向量到 FAISS 索引

        Args:
            node_id: 图节点 ID
            summary_text: 摘要文本

        Returns:
            是否添加成功
        """
        if not self._ensure_init() or not summary_text:
            return False
        # 已存在则先移除旧的
        if node_id in self._node_to_faiss:
            self.remove(node_id)
        emb_mgr = self._get_embedding_manager()
        if emb_mgr is None:
            return False
        try:
            vector = emb_mgr.encode_document(summary_text)
            if vector is None:
                return False
            vector = np.array(vector, dtype=np.float32).reshape(1, -1)
            faiss_doc_id = f"summary_{node_id}"
            self._store.add_vectors_with_ids(
                vectors=vector,
                metadata=[{"node_id": node_id, "summary": summary_text[:200]}],
                vector_ids=[faiss_doc_id],
            )
            self._node_to_faiss[node_id] = faiss_doc_id
            self._faiss_to_node[faiss_doc_id] = node_id
            # P0-3: 同步索引摘要文本以支持关键词检索
            self._text_index[node_id] = summary_text
            return True
        except Exception as e:
            logger.warning(f"[SummarySidecarIndex] 添加摘要失败 {node_id}: {e}")
            return False

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        exclude_node_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """向量检索摘要，返回匹配的 node_id 列表

        Args:
            query_text: 查询文本
            top_k: 返回条数
            exclude_node_ids: 需排除的节点 ID 集合（用于互斥过滤）

        Returns:
            [(node_id, score), ...] score 越小越相似（L2 距离）
        """
        if not self._ensure_init():
            return []
        emb_mgr = self._get_embedding_manager()
        if emb_mgr is None:
            return []
        try:
            query_vec = emb_mgr.encode_query(query_text)
            if query_vec is None:
                return []
            query_vec = np.array(query_vec, dtype=np.float32).reshape(1, -1)
            # 多请求一些以弥补过滤损失
            search_k = top_k * 3 if exclude_node_ids else top_k
            indices, distances = self._store.search(query_vec, top_k=search_k)

            results = []
            for idx, dist in zip(indices, distances):
                faiss_doc_id = self._store.reverse_id_map.get(idx)
                if not faiss_doc_id:
                    continue
                node_id = self._faiss_to_node.get(faiss_doc_id)
                if not node_id:
                    continue
                if exclude_node_ids and node_id in exclude_node_ids:
                    continue
                # 将 L2 距离转换为相似度分数 (0~1)
                score = 1.0 / (1.0 + dist)
                results.append((node_id, score))
                if len(results) >= top_k:
                    break
            return results
        except Exception as e:
            logger.warning(f"[SummarySidecarIndex] 搜索失败: {e}")
            return []

    def remove(self, node_id: str) -> bool:
        """移除节点的摘要索引"""
        faiss_doc_id = self._node_to_faiss.pop(node_id, None)
        self._text_index.pop(node_id, None)  # P0-3: 同步清理文本索引
        if faiss_doc_id:
            self._faiss_to_node.pop(faiss_doc_id, None)
            if self._store:
                self._store.delete_vectors([faiss_doc_id])
            return True
        return False

    def has_node(self, node_id: str) -> bool:
        """检查节点是否已被索引"""
        return node_id in self._node_to_faiss

    def save(self, path: str) -> bool:
        """持久化 FAISS 索引和 ID 映射（原子写入：先写临时路径再替换）

        FAISSVectorStore.save(path) 生成 {path}.index + {path}.maps.json 两个文件。
        本方法额外生成 {path}.node_maps.json（node_id ↔ faiss_doc_id 映射）。

        原子策略：先保存到 {path}_tmp 前缀，全部成功后原子替换为正式文件。
        """
        if not self._store:
            return False
        try:
            tmp_prefix = f"{path}_tmp"

            # Step 1: 保存 FAISS 索引到临时前缀（生成 {tmp_prefix}.index + {tmp_prefix}.maps.json）
            self._store.save(tmp_prefix)

            # Step 2: 保存 node_id 映射到临时文件
            maps_path = f"{path}.node_maps.json"
            tmp_maps_path = f"{tmp_prefix}.node_maps.json"
            with open(tmp_maps_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "node_to_faiss": self._node_to_faiss,
                    "faiss_to_node": self._faiss_to_node,
                    "text_index": self._text_index,  # P0-3: 持久化关键词文本索引
                }, f, ensure_ascii=False)

            # Step 3: 原子替换（逐个文件 replace）
            _file_pairs = [
                (f"{tmp_prefix}.index", f"{path}.index"),
                (f"{tmp_prefix}.maps.json", f"{path}.maps.json"),
                (tmp_maps_path, maps_path),
            ]
            for src, dst in _file_pairs:
                if os.path.exists(src):
                    os.replace(src, dst)

            return True
        except Exception as e:
            logger.warning(f"[SummarySidecarIndex] 保存失败: {e}")
            # 清理可能残留的临时文件
            tmp_prefix = f"{path}_tmp"
            for suffix in [".index", ".maps.json", ".node_maps.json"]:
                try:
                    tmp_file = f"{tmp_prefix}{suffix}"
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)
                except Exception:
                    pass
            return False

    def load(self, path: str) -> bool:
        """从磁盘加载 FAISS 索引和 ID 映射"""
        if not self._ensure_init():
            return False
        try:
            if not self._store.load(path):
                return False
            # 验证 reverse_id_map 已正确加载（防止 .maps.json 缺失导致搜索静默失败）
            if self._store.index.ntotal > 0 and not self._store.reverse_id_map:
                logger.warning(
                    f"[SummarySidecarIndex] FAISS 索引有 {self._store.index.ntotal} 条向量，"
                    f"但 reverse_id_map 为空，搜索将无法返回结果。请检查 .maps.json 文件。"
                )
            maps_path = f"{path}.node_maps.json"
            if os.path.exists(maps_path):
                with open(maps_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._node_to_faiss = data.get("node_to_faiss", {})
                    self._faiss_to_node = data.get("faiss_to_node", {})
                    self._text_index = data.get("text_index", {})  # P0-3: 恢复关键词索引
            else:
                # node_maps 缺失时，尝试从 reverse_id_map 重建映射
                if self._store.reverse_id_map:
                    logger.warning(
                        f"[SummarySidecarIndex] node_maps.json 缺失，"
                        f"尝试从 FAISSVectorStore 映射重建..."
                    )
                    for idx, faiss_doc_id in self._store.reverse_id_map.items():
                        # faiss_doc_id 格式为 "summary_{node_id}"
                        if faiss_doc_id.startswith("summary_"):
                            node_id = faiss_doc_id[len("summary_"):]
                            self._node_to_faiss[node_id] = faiss_doc_id
                            self._faiss_to_node[faiss_doc_id] = node_id
                    logger.info(
                        f"[SummarySidecarIndex] 重建了 {len(self._node_to_faiss)} 条映射"
                    )
            logger.info(
                f"[SummarySidecarIndex] 已加载: {len(self._node_to_faiss)} 条摘要索引"
            )
            return True
        except Exception as e:
            logger.warning(f"[SummarySidecarIndex] 加载失败: {e}")
            return False

    @property
    def count(self) -> int:
        """索引中的摘要数量"""
        return len(self._node_to_faiss)

    def keyword_search(
        self,
        query_text: str,
        top_k: int = 5,
        exclude_node_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """P0-3: 关键词检索（bigram + 子串匹配），补齐冷路径的关键词能力

        对 _text_index 中存储的摘要文本执行关键词匹配，
        使用与热路径 _bigram_overlap_score 一致的算法。

        Returns:
            [(node_id, score), ...] score 0.0~1.0
        """
        if not query_text or not self._text_index:
            return []
        query_lower = query_text.lower()
        scored: List[Tuple[str, float]] = []
        for node_id, summary in self._text_index.items():
            if exclude_node_ids and node_id in exclude_node_ids:
                continue
            text_lower = summary.lower()
            # 1. 精确子串匹配
            if query_lower in text_lower:
                score = 0.8
            else:
                # 2. Bigram 重叠匹配
                score = self._bigram_score(query_lower, text_lower)
            if score > 0.05:
                scored.append((node_id, score))
        # 按分数降序，取 top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _bigram_score(query: str, text: str) -> float:
        """bigram 重叠分数（与 MemoryGraph._bigram_overlap_score 一致）"""
        if not query or not text or len(query) < 2:
            return 0.0
        q_bigrams = {query[i:i+2] for i in range(len(query)-1)}
        t_bigrams = {text[i:i+2] for i in range(len(text)-1)}
        if not q_bigrams:
            return 0.0
        overlap = len(q_bigrams & t_bigrams) / len(q_bigrams)
        return overlap * 0.6 if overlap >= 0.2 else 0.0


# ============================================================
# MemoryGraph 核心类
# ============================================================

class MemoryGraph:
    """记忆图谱 -- 统一异构类型图

    单例模式，参考 KnowledgeGraph 的 __new__ + _initialized 模式。
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, persist_path: str = "./data/memory_graph"):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._graph = nx.DiGraph()
        self._nodes: Dict[str, GraphNode] = {}
        self._embeddings: Dict[str, Any] = {}  # node_id -> np.ndarray (512-dim)
        self.persist_path = persist_path

        os.makedirs(persist_path, exist_ok=True)

        # FAISS 摘要侧车索引（延迟初始化，首次使用时才加载 FAISS）
        self._summary_index = SummarySidecarIndex(
            dimension=512, persist_path=persist_path,
        )

        # 适配器注册表
        self._adapters: Dict[str, Any] = {}

        # RAGManager 引用（由外部注入，用于 backend_ref 反查）
        self._rag_manager = None

        # 赫布学习: 共激活计数器
        self._coactivation_counter: Dict[Tuple[str, str], int] = {}

        # 变更追踪 (增量 delta 推送到前端)
        self._pending_changes: List[Dict[str, Any]] = []

        # 异步修剪控制
        self._running = False
        self._prune_task: Optional[asyncio.Task] = None

        # 自动保存（防抖写盘）
        self._dirty = False                # 是否有未保存的变更
        self._auto_save_delay = 2          # 防抖延迟（秒），VS Code 风格：变更后最多 2 秒落盘
        self._auto_save_timer = None       # threading.Timer 实例
        self._last_save_time = 0.0         # 上次保存时间戳
        self._save_lock = threading.Lock() # 保存操作锁（防并发写）
        self._data_lock = threading.RLock()  # 内存数据结构读写锁（_nodes/_graph/_embeddings/_coactivation_counter）
        self._candidates_lock = threading.Lock()  # _pending_llm_candidates 读写锁
        self._pending_llm_candidates: List[str] = []

        # 活跃节点追踪（前端高亮用，不持久化）
        self._active_node_ids: Set[str] = set()

        # 上次焦点上下文（持久化到 meta，用于重启后恢复注意力）
        self._last_focus_context: Optional[Dict] = None

        # 最近一次 retrieve_context 的 top 命中节点 ID（供 BFS 种子扩展）
        self._last_retrieved_node_ids: List[str] = []

        # P0-1: 记忆上下文过期标记（焦点漂移后标记为 True，触发系统 prompt 刷新）
        self._memory_context_stale: bool = False

        # P1-4: 地址反向索引（graph_address / task_graph_address → node_id）
        self._address_index: Dict[str, str] = {}

        # 统计
        self._stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "total_activations": 0,
            "total_prune_cycles": 0,
        }

        # 尝试加载已有数据
        self._load()

        # 启动时健康检查: 清理膨胀的 ASSOCIATION 边
        self._sanitize_association_edges()

        self._initialized = True

        # P2-3: atexit 安全网 — 进程退出时强制刷盘防抖窗口内未保存的变更
        atexit.register(self._atexit_flush)

        # P1-5: 启动异步修剪循环（尝试在现有事件循环中调度）
        self._try_start_prune_loop()

        logger.info(
            f"[MemoryGraph] 初始化完成: "
            f"{self._stats['total_nodes']} 节点, "
            f"{self._stats['total_edges']} 边"
        )

    # ============================================================
    # 节点 CRUD
    # ============================================================

    def _index_node_address(self, node_id: str, metadata: Dict):
        """P1-4: 将节点的 graph_address / task_graph_address 加入反向索引"""
        for key in ("graph_address", "task_graph_address"):
            addr = metadata.get(key)
            if addr:
                self._address_index[addr] = node_id

    def _unindex_node_address(self, node_id: str):
        """P1-4: 从反向索引中移除指定节点的所有地址"""
        stale = [addr for addr, nid in self._address_index.items() if nid == node_id]
        for addr in stale:
            del self._address_index[addr]

    def add_node(self, node: GraphNode, touch: bool = True) -> str:
        """添加或更新节点

        如果 node_id 已存在，更新 label/metadata。
        touch=True（默认）时同时刷新 last_accessed；
        touch=False 时保留原始时间戳（适用于 sync_all 等批量导入场景）。

        Returns:
            str: node_id
        """
        with self._data_lock:
            if node.node_id in self._nodes:
                existing = self._nodes[node.node_id]
                existing.label = node.label
                existing.metadata.update(node.metadata)
                if touch:
                    existing.last_accessed = time.time()
                existing.backend_ref = node.backend_ref or existing.backend_ref
                self._index_node_address(node.node_id, existing.metadata)
                self._graph.nodes[node.node_id].update(node.to_dict())
                self._pending_changes.append({
                    "action": "update_node",
                    "data": {"id": node.node_id, "type": node.node_type.value, "label": node.label},
                })
                self._mark_dirty()
                return node.node_id

            node.metadata.setdefault("temperature", Temperature.HOT.value)
            node.metadata.setdefault("importance", Importance.NORMAL.value)
            node.metadata.setdefault("memory_strength", {
                "base_strength": 1.0,
                "current_strength": 1.0,
                "rehearsal_count": 0,
                "last_rehearsal": node.created_at,
            })

            self._nodes[node.node_id] = node
            self._graph.add_node(node.node_id, **node.to_dict())
            self._stats["total_nodes"] += 1
            self._index_node_address(node.node_id, node.metadata)
            self._pending_changes.append({
                "action": "add_node",
                "data": {"id": node.node_id, "type": node.node_type.value, "label": node.label},
            })
            self._mark_dirty()

        self._auto_embed_node(node)

        return node.node_id

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """获取节点"""
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> bool:
        """移除节点及其所有关联边"""
        with self._data_lock:
            if node_id not in self._nodes:
                return False
            self._unindex_node_address(node_id)
            del self._nodes[node_id]
            self._embeddings.pop(node_id, None)
            self._summary_index.remove(node_id)
            self._graph.remove_node(node_id)
            stale_keys = [k for k in self._coactivation_counter if node_id in k]
            for k in stale_keys:
                del self._coactivation_counter[k]
            self._active_node_ids.discard(node_id)
            self._stats["total_nodes"] = len(self._nodes)
            self._stats["total_edges"] = self._graph.number_of_edges()
            self._pending_changes.append({"action": "remove_node", "data": {"id": node_id}})
            self._mark_dirty()
            return True

    def update_node_activation(self, node_id: str, activation: float):
        """更新节点激活水平"""
        node = self._nodes.get(node_id)
        if node:
            node.activation = activation
            node.last_accessed = time.time()
            node.access_count += 1
            # 更新 memory_strength 的 rehearsal 记录
            ms = node.metadata.get("memory_strength")
            if ms and isinstance(ms, dict):
                ms["rehearsal_count"] = ms.get("rehearsal_count", 0) + 1
                ms["last_rehearsal"] = node.last_accessed
                ms["current_strength"] = min(
                    1.0, ms.get("base_strength", 1.0) + 0.05 * ms["rehearsal_count"]
                )

    def get_nodes_by_type(self, node_type: NodeType) -> List[GraphNode]:
        """按类型查询节点"""
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def has_node(self, node_id: str) -> bool:
        """检查节点是否存在"""
        return node_id in self._nodes

    # ============================================================
    # 多维标签辅助方法
    # ============================================================

    def get_temperature(self, node_id: str) -> Optional[Temperature]:
        """动态计算节点温度（基于 last_accessed 实时算，不依赖存储值）

        Returns:
            Temperature 枚举值，节点不存在返回 None
        """
        node = self._nodes.get(node_id)
        if not node:
            return None
        elapsed = time.time() - node.last_accessed
        if elapsed < _TEMPERATURE_THRESHOLDS["hot_max"]:
            return Temperature.HOT
        elif elapsed < _TEMPERATURE_THRESHOLDS["warm_max"]:
            return Temperature.WARM
        else:
            return Temperature.COLD

    def get_importance(self, node_id: str) -> Optional[Importance]:
        """读取节点重要度标签

        Returns:
            Importance 枚举值，节点不存在返回 None
        """
        node = self._nodes.get(node_id)
        if not node:
            return None
        raw = node.metadata.get("importance", "normal")
        try:
            return Importance(raw)
        except ValueError:
            return Importance.NORMAL

    def set_importance(self, node_id: str, importance: Importance) -> bool:
        """设置节点重要度标签

        Args:
            node_id: 节点 ID
            importance: 目标重要度

        Returns:
            是否设置成功
        """
        node = self._nodes.get(node_id)
        if not node:
            return False
        node.metadata["importance"] = importance.value
        self._mark_dirty()
        return True

    def update_temperature(self, node_id: str) -> Optional[Temperature]:
        """动态计算温度并同步写入 metadata 缓存（前端展示用）

        Returns:
            Temperature 枚举值，节点不存在返回 None
        """
        temp = self.get_temperature(node_id)
        if temp is not None:
            node = self._nodes[node_id]
            node.metadata["temperature"] = temp.value
        return temp

    def is_recent(self, node_id: str, window_seconds: int = 1800) -> bool:
        """判断节点是否在热窗口内（检索路由用）

        Args:
            node_id: 节点 ID
            window_seconds: 热窗口秒数，默认 30 分钟

        Returns:
            True 表示节点在热窗口内
        """
        node = self._nodes.get(node_id)
        if not node:
            return False
        return (time.time() - node.last_accessed) < window_seconds

    # ============================================================
    # 重要度动态提升
    # ============================================================

    def promote_importance(self, node_id: str, target: Importance) -> bool:
        """提升节点重要度（只升不降）

        Args:
            node_id: 节点 ID
            target: 目标重要度

        Returns:
            是否实际提升
        """
        node = self._nodes.get(node_id)
        if not node:
            return False

        current = self.get_importance(node_id) or Importance.NORMAL
        current_order = _IMPORTANCE_ORDER.get(current, 1)
        target_order = _IMPORTANCE_ORDER.get(target, 1)

        # 只允许向上提升
        if target_order <= current_order:
            return False

        # 执行提升
        node.metadata["importance"] = target.value

        # 记录提升历史
        history = node.metadata.setdefault("importance_history", [])
        history.append({
            "from": current.value,
            "to": target.value,
            "timestamp": time.time(),
        })

        # 提升为 must_remember 时，自动将所有关联边设为 protected
        if target == Importance.MUST_REMEMBER:
            for _, neighbor, data in self._graph.out_edges(node_id, data=True):
                data["protected"] = True
            for predecessor, _, data in self._graph.in_edges(node_id, data=True):
                data["protected"] = True

        self._mark_dirty()
        logger.info(
            f"[MemoryGraph] 节点 {node_id} 重要度提升: "
            f"{current.value} → {target.value}"
        )
        return True

    def _auto_promote_importance(self) -> int:
        """仅自动提升 NORMAL→IMPORTANT，不提交LLM审查（修剪循环调用）"""
        auto_promoted = 0
        for node_id, node in self._nodes.items():
            imp = self.get_importance(node_id) or Importance.NORMAL
            count = node.access_count
            if count >= 3 and imp == Importance.NORMAL:
                if self.promote_importance(node_id, Importance.IMPORTANT):
                    auto_promoted += 1
        if auto_promoted > 0:
            logger.info(f"[MemoryGraph] 自动提升了 {auto_promoted} 个节点的重要度")
        return auto_promoted

    def run_importance_review(self) -> Dict[str, Any]:
        """扫描所有节点，自动提升重要度并将候选入列（不提交LLM审查）

        LLM审查提交由 _idle_review_loop 在系统空闲5分钟时统一处理。

        规则:
        - access_count >= 3 且 importance == NORMAL → 自动提升为 IMPORTANT
        - access_count >= 5 且 importance == IMPORTANT → 入列 _pending_llm_candidates 等待空闲审查

        Returns:
            {"auto_promoted": int, "llm_candidates": List[str], "submitted_count": 0}
        """
        auto_promoted = 0
        llm_candidates = []

        for node_id, node in self._nodes.items():
            imp = self.get_importance(node_id) or Importance.NORMAL
            count = node.access_count

            if count >= 3 and imp == Importance.NORMAL:
                if self.promote_importance(node_id, Importance.IMPORTANT):
                    auto_promoted += 1

            elif count >= 5 and imp == Importance.IMPORTANT:
                llm_candidates.append(node_id)

        if auto_promoted > 0:
            logger.info(f"[MemoryGraph] 自动提升了 {auto_promoted} 个节点的重要度")

        if llm_candidates:
            with self._candidates_lock:
                self._pending_llm_candidates = list(set(self._pending_llm_candidates + llm_candidates))
                pending_len = len(self._pending_llm_candidates)
            logger.info(f"[MemoryGraph] {len(llm_candidates)} 个节点入列待审查（积压={pending_len}），等待空闲时提交")

        return {"auto_promoted": auto_promoted, "llm_candidates": llm_candidates, "submitted_count": 0}

    # ============================================================
    # 边 CRUD
    # ============================================================

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        protected: bool = False,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """添加边（源和目标节点必须已存在）

        如果边已存在，更新权重和 last_activated
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return False

        with self._data_lock:
            if self._graph.has_edge(source_id, target_id):
                data = self._graph.edges[source_id, target_id]
                data["weight"] = max(data["weight"], weight)
                data["last_activated"] = time.time()
                data["activation_count"] = data.get("activation_count", 0) + 1
                self._mark_dirty()
                return True

            edge_data = _make_edge_data(edge_type, weight, protected, metadata)
            self._graph.add_edge(source_id, target_id, **edge_data)
            self._stats["total_edges"] = self._graph.number_of_edges()
            self._pending_changes.append({
                "action": "add_edge",
                "data": {"source": source_id, "target": target_id, "type": edge_type.value, "weight": weight},
            })
            self._mark_dirty()
            return True

    def get_edge(self, source_id: str, target_id: str) -> Optional[Dict]:
        """获取边属性"""
        if self._graph.has_edge(source_id, target_id):
            return dict(self._graph.edges[source_id, target_id])
        return None

    def has_edge(self, source_id: str, target_id: str) -> bool:
        """检查边是否存在（任一方向）"""
        return self._graph.has_edge(source_id, target_id) or self._graph.has_edge(target_id, source_id)

    def remove_edge(self, source_id: str, target_id: str) -> bool:
        """移除边"""
        with self._data_lock:
            if self._graph.has_edge(source_id, target_id):
                self._graph.remove_edge(source_id, target_id)
                self._stats["total_edges"] = self._graph.number_of_edges()
                self._pending_changes.append({
                    "action": "remove_edge", "data": {"source": source_id, "target": target_id},
                })
                self._mark_dirty()
                return True
        return False

    # ============================================================
    # 图查询
    # ============================================================

    def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[Set[EdgeType]] = None,
        max_depth: int = 1,
    ) -> List[GraphNode]:
        """获取邻居节点

        Args:
            node_id: 中心节点 ID
            edge_types: 限定边类型 (None = 所有)
            max_depth: 搜索深度

        Returns:
            邻居节点列表
        """
        if node_id not in self._nodes:
            return []

        visited = {node_id}
        current_layer = [node_id]
        result = []

        for _ in range(max_depth):
            next_layer = []
            for nid in current_layer:
                # 出边
                for _, neighbor, data in self._graph.out_edges(nid, data=True):
                    if neighbor in visited:
                        continue
                    if edge_types and EdgeType(data["edge_type"]) not in edge_types:
                        continue
                    visited.add(neighbor)
                    next_layer.append(neighbor)
                    node = self._nodes.get(neighbor)
                    if node:
                        result.append(node)

                # 入边 (视为无向查询)
                for predecessor, _, data in self._graph.in_edges(nid, data=True):
                    if predecessor in visited:
                        continue
                    if edge_types and EdgeType(data["edge_type"]) not in edge_types:
                        continue
                    visited.add(predecessor)
                    next_layer.append(predecessor)
                    node = self._nodes.get(predecessor)
                    if node:
                        result.append(node)

            current_layer = next_layer

        return result

    # ---- 有方向层级遍历 API（思维深度索引基础设施） ----

    def get_parent(
        self, node_id: str, edge_type: EdgeType = EdgeType.HIERARCHY
    ) -> Optional[GraphNode]:
        """沿 HIERARCHY 入边找父节点

        HIERARCHY 边方向: parent → child，因此 in_edges 的源节点即为父。

        Args:
            node_id: 目标节点 ID
            edge_type: 要查找的边类型（默认 HIERARCHY）

        Returns:
            父节点的 GraphNode，若无则返回 None
        """
        if node_id not in self._nodes:
            return None
        for predecessor, _, data in self._graph.in_edges(node_id, data=True):
            if EdgeType(data.get("edge_type", "")) == edge_type:
                node = self._nodes.get(predecessor)
                if node:
                    return node
        return None

    def get_ancestors(
        self, node_id: str, edge_type: EdgeType = EdgeType.HIERARCHY
    ) -> List[GraphNode]:
        """从 node_id 沿 HIERARCHY 入边向上走到根

        Returns:
            有序列表 [root, ..., grandparent, parent]，index 0 为根节点。
            若 node_id 无父节点，返回空列表。
        """
        if node_id not in self._nodes:
            return []
        ancestors: List[GraphNode] = []
        visited: set = {node_id}
        current = node_id
        for _ in range(20):  # 安全上限防环
            parent = self.get_parent(current, edge_type)
            if parent is None or parent.node_id in visited:
                break
            visited.add(parent.node_id)
            ancestors.append(parent)
            current = parent.node_id
        ancestors.reverse()  # 翻转: root 在前
        return ancestors

    def get_children(
        self, node_id: str, edge_type: EdgeType = EdgeType.HIERARCHY
    ) -> List[GraphNode]:
        """沿 HIERARCHY 出边找子节点

        Args:
            node_id: 父节点 ID
            edge_type: 要查找的边类型（默认 HIERARCHY）

        Returns:
            子节点列表，按 created_at 排序
        """
        if node_id not in self._nodes:
            return []
        children: List[GraphNode] = []
        for _, target, data in self._graph.out_edges(node_id, data=True):
            if EdgeType(data.get("edge_type", "")) == edge_type:
                child = self._nodes.get(target)
                if child:
                    children.append(child)
        children.sort(key=lambda n: n.created_at)
        return children

    def get_subgraph_summary(self, node_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """提取以 node_id 为中心的子图摘要"""
        neighbors = self.get_neighbors(node_id, max_depth=max_depth)
        center = self.get_node(node_id)
        if not center:
            return {}

        type_counts: Dict[str, int] = {}
        for n in neighbors:
            type_counts[n.node_type.value] = type_counts.get(n.node_type.value, 0) + 1

        return {
            "center": center.to_dict(),
            "neighbor_count": len(neighbors),
            "type_distribution": type_counts,
            "neighbors": [n.to_dict() for n in neighbors[:20]],  # 最多返回20个
        }

    def search_nodes(
        self,
        query: str,
        node_types: Optional[List['NodeType']] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """按关键词搜索记忆图谱节点

        搜索策略（按优先级）：
        1. 精确匹配 node_id
        2. label 包含查询词
        3. metadata 中 desc/content 包含查询词

        Args:
            query: 搜索关键词
            node_types: 限定节点类型（None = 搜索全部）
            max_results: 最大返回数量

        Returns:
            [{"node_id", "type", "label", "score", "activation", "metadata"}]
        """
        query_lower = query.lower()
        results = []

        for nid, node in self._nodes.items():
            if node_types and node.node_type not in node_types:
                continue

            score = 0.0
            if query_lower in nid.lower():
                score = 1.0
            elif query_lower in node.label.lower():
                score = 0.8
            else:
                meta_str = json.dumps(node.metadata, ensure_ascii=False).lower()
                if query_lower in meta_str:
                    score = 0.5

            if score > 0:
                activation = node.activation
                score += activation * 0.2
                results.append({
                    "node_id": nid,
                    "type": node.node_type.value,
                    "label": node.label,
                    "score": round(score, 3),
                    "activation": round(activation, 3),
                    "metadata": {k: str(v)[:100] for k, v in node.metadata.items()},
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]

    def resolve_address(self, address: str) -> Optional[GraphNode]:
        """解析 graph_address 字符串，返回对应 GraphNode

        支持格式:
        - "tg:{graph_id}/task:{node_id}" → 提取 node_id，查找 "task:{node_id}"
        - "dialogue:round_{id}" → 直接作为 node_id 查找
        - 任意字符串 → 兜底作为 node_id 直接查找

        Returns:
            匹配的 GraphNode，未找到返回 None
        """
        if not address:
            return None

        # 格式 1: tg:{graph_id}/task:{node_id}
        if address.startswith("tg:"):
            parts = address.split("/")
            for part in parts:
                if part.startswith("task:"):
                    # 直接用 task:{node_id} 作为图内 node_id 查找
                    node = self._nodes.get(f"task:{part[5:]}")
                    if node:
                        return node
                    # 也尝试原始 part 作为 node_id
                    node = self._nodes.get(part)
                    if node:
                        return node

        # 格式 2/3: 直接 node_id 查找
        node = self._nodes.get(address)
        if node:
            return node

        # P1-4: 使用地址反向索引 O(1) 查找，替代 O(n) 遍历
        indexed_nid = self._address_index.get(address)
        if indexed_nid:
            node = self._nodes.get(indexed_nid)
            if node:
                return node

        # 兜底: 遍历 metadata 中的 graph_address 字段匹配（索引未命中时降级）
        for nid, n in self._nodes.items():
            if n.metadata.get("graph_address") == address:
                # 补充索引以加速后续查找
                self._address_index[address] = nid
                return n
            if n.metadata.get("task_graph_address") == address:
                self._address_index[address] = nid
                return n

        return None

    # ============================================================
    # BFS 扩散激活 (核心算法)
    # ============================================================

    def compute_activations(
        self,
        seed_node_ids: List[str],
        max_depth: int = 3,
        decay: float = 0.5,
        min_activation: float = 0.01,
        node_type_filter: Optional[set] = None,
    ) -> Dict[str, float]:
        """加权 BFS 扩散激活

        从种子节点出发，沿着边传播激活值，边权越高传播越多。
        这是记忆图谱的核心算法 -- 实现"从任意节点追溯全局关联"。

        Args:
            seed_node_ids: 种子节点 ID 列表
            max_depth: 最大扩散深度 (默认3跳)
            decay: 每跳衰减因子 (默认0.5)
            min_activation: 最小激活阈值，低于此值停止传播
            node_type_filter: 允许传播的节点类型集合 (None=不限制)。
                例如 {NodeType.CODE_SYMBOL, NodeType.FILE} 只在代码节点间传播。

        Returns:
            Dict[node_id, activation_score]
        """
        activations: Dict[str, float] = {}
        queue: deque = deque()
        # 使用 visited 集合防止同一节点在同一深度被重复入队
        in_queue: set = set()

        # 初始化种子节点
        for seed in seed_node_ids:
            if seed in self._nodes:
                activations[seed] = 1.0
                queue.append((seed, 0, 1.0))
                in_queue.add(seed)

        # 记录共激活的边 (用于赫布学习)
        activated_edges: List[Tuple[str, str]] = []

        while queue:
            node_id, depth, act = queue.popleft()

            if depth >= max_depth:
                continue

            # 遍历出边
            if self._graph.has_node(node_id):
                for _, neighbor, data in self._graph.out_edges(node_id, data=True):
                    # 节点类型过滤
                    if node_type_filter is not None:
                        neighbor_node = self._nodes.get(neighbor)
                        if neighbor_node and neighbor_node.node_type not in node_type_filter:
                            continue

                    edge_weight = data.get("weight", 1.0)
                    propagated = act * edge_weight * decay

                    if propagated < min_activation:
                        continue

                    if neighbor not in activations or activations[neighbor] < propagated:
                        activations[neighbor] = max(activations.get(neighbor, 0), propagated)
                        activated_edges.append((node_id, neighbor))
                        # 只有未在队列中的节点才入队，防止重复处理
                        if neighbor not in in_queue:
                            queue.append((neighbor, depth + 1, propagated))
                            in_queue.add(neighbor)

                # 遍历入边 (视为无向传播)
                for predecessor, _, data in self._graph.in_edges(node_id, data=True):
                    # 节点类型过滤
                    if node_type_filter is not None:
                        pred_node = self._nodes.get(predecessor)
                        if pred_node and pred_node.node_type not in node_type_filter:
                            continue

                    edge_weight = data.get("weight", 1.0)
                    propagated = act * edge_weight * decay

                    if propagated < min_activation:
                        continue

                    if predecessor not in activations or activations[predecessor] < propagated:
                        activations[predecessor] = max(activations.get(predecessor, 0), propagated)
                        activated_edges.append((node_id, predecessor))
                        if predecessor not in in_queue:
                            queue.append((predecessor, depth + 1, propagated))
                            in_queue.add(predecessor)

        # 更新节点激活值
        # 只对种子节点增加 access_count（代表主动访问）；
        # BFS 被动传播的节点仅更新 activation 值，不膨胀 access_count
        seed_set = set(seed_node_ids)
        for nid, act_val in activations.items():
            if nid in seed_set:
                self.update_node_activation(nid, act_val)
            else:
                # 仅更新激活水平，不增加 access_count，不更新 last_accessed
                node = self._nodes.get(nid)
                if node:
                    node.activation = act_val

        self._stats["total_activations"] += 1

        # 返回激活映射和边信息 (赫布学习在阶段4使用)
        self._last_activated_edges = activated_edges

        if activations:
            logger.info(
                f"[MemoryGraph] BFS扩散激活: {len(seed_node_ids)} 种子 → "
                f"{len(activations)} 节点激活, {len(activated_edges)} 条边共激活"
            )

        return activations

    def compute_activations_dynamic(
        self,
        seed_node_ids: List[str],
        context_window_size: int = 131072,
        usage_ratio: float = 0.0,
        node_type_filter: Optional[set] = None,
    ) -> Dict[str, float]:
        """上下文感知的加权 BFS 扩散激活

        根据 context_window_size 和 usage_ratio 动态计算 BFS 扩散参数:
        - 大窗口(>64K) + 低占用(<0.5): 扩散更深层(max_depth=5)，检索更广
        - 小窗口(<32K) 或 高占用(>0.8): 收窄扩散(max_depth=2)，减少噪声
        - 中间状态: 按比例线性插值

        Args:
            seed_node_ids: 种子节点 ID 列表
            context_window_size: 模型上下文窗口大小（tokens）
            usage_ratio: 当前上下文使用率 (0.0~1.0)
            node_type_filter: 允许传播的节点类型集合

        Returns:
            Dict[node_id, activation_score]
        """
        remaining_ratio = max(0.0, 1.0 - usage_ratio)

        window_tier = 0.0
        if context_window_size >= 200000:
            window_tier = 2.0
        elif context_window_size >= 128000:
            window_tier = 1.5
        elif context_window_size >= 64000:
            window_tier = 1.0
        else:
            window_tier = 0.5

        capacity_factor = remaining_ratio * (0.5 + 0.5 * min(window_tier / 2.0, 1.0))

        max_depth = max(2, min(6, int(2 + 4 * capacity_factor)))
        decay = 0.3 + 0.4 * (1.0 - capacity_factor)
        min_activation = 0.01 + 0.09 * (1.0 - capacity_factor)

        logger.info(
            f"[MemoryGraph] 动态BFS参数: context_window={context_window_size}, "
            f"usage_ratio={usage_ratio:.2f}, capacity_factor={capacity_factor:.2f} → "
            f"max_depth={max_depth}, decay={decay:.2f}, min_activation={min_activation:.3f}"
        )

        return self.compute_activations(
            seed_node_ids,
            max_depth=max_depth,
            decay=decay,
            min_activation=min_activation,
            node_type_filter=node_type_filter,
        )

    def retrieve_context_dynamic(
        self,
        query_text: str,
        context_window_size: int = 131072,
        usage_ratio: float = 0.0,
        session_id: str = "",
    ) -> "asyncio.coroutine":
        """上下文感知的记忆检索

        根据 context_window_size 和 usage_ratio 动态调整:
        - top_k: 按剩余容量比例计算，而非固定值
        - hot_window_minutes: 高占用时缩窄热窗口，低占用时扩大
        - 结果注入时按token预算裁剪
        """
        remaining_ratio = max(0.0, 1.0 - usage_ratio)

        window_factor = min(context_window_size / 128000, 2.0)

        base_top_k = 10
        top_k = max(3, min(20, int(base_top_k * remaining_ratio * window_factor)))

        hot_window_minutes = max(5, min(60, int(30 * remaining_ratio + 10)))

        logger.info(
            f"[MemoryGraph] 动态检索参数: remaining={remaining_ratio:.2f}, "
            f"window_factor={window_factor:.2f} → "
            f"top_k={top_k}, hot_window={hot_window_minutes}min"
        )

        return self.retrieve_context(
            query_text,
            top_k=top_k,
            hot_window_minutes=hot_window_minutes,
            session_id=session_id,
        )

    # ============================================================
    # 赫布学习
    # ============================================================

    def hebbian_strengthen(self):
        """赫布增强: 对上一次 compute_activations 中共激活的边增加权重

        公式: new_weight = old_weight + eta * (1 - old_weight)
        渐近趋向 1.0，永远不会超过 1.0
        """
        eta = 0.1  # 学习率
        edges = getattr(self, '_last_activated_edges', [])

        for src, tgt in edges:
            if not self._graph.has_edge(src, tgt):
                continue
            data = self._graph.edges[src, tgt]
            if data.get("protected"):
                continue

            old_w = data["weight"]
            new_w = old_w + eta * (1.0 - old_w)
            data["weight"] = new_w
            data["last_activated"] = time.time()
            data["activation_count"] = data.get("activation_count", 0) + 1

        # 共激活计数 (用于自动创建 ASSOCIATION 边)
        self._update_coactivation_counter(edges)

        if edges:
            strengthened = sum(1 for s, t in edges if self._graph.has_edge(s, t))
            logger.info(
                f"[MemoryGraph] 赫布增强: {strengthened}/{len(edges)} 条边权重已更新"
            )

    def _update_coactivation_counter(self, activated_edges: List[Tuple[str, str]]):
        """更新共激活计数，超过阈值时自动创建 ASSOCIATION 边

        安全限制:
        - 只对直接相邻的激活边对计数（而非所有激活节点的全排列）
        - 每个节点的 ASSOCIATION 出度上限 _MAX_ASSOC_PER_NODE (默认 10)
        - coactivation_counter 总容量上限 _MAX_COACTIVATION_PAIRS (默认 5000)
        """
        coactivation_threshold = 3
        _MAX_ASSOC_PER_NODE = 10
        _MAX_COACTIVATION_PAIRS = 5000

        # 只对直接相邻的激活边对计数，不做 N*N 全排列
        for src, tgt in activated_edges:
            if src == tgt:
                continue
            a, b = (src, tgt) if src < tgt else (tgt, src)
            pair = (a, b)

            # 跳过已有边的节点对
            if self.has_edge(a, b):
                continue

            self._coactivation_counter[pair] = self._coactivation_counter.get(pair, 0) + 1

            if self._coactivation_counter[pair] >= coactivation_threshold:
                # 检查两端节点的 ASSOCIATION 出度是否已达上限
                a_assoc_count = sum(
                    1 for _, _, d in self._graph.out_edges(a, data=True)
                    if d.get("edge_type") == EdgeType.ASSOCIATION.value
                ) if self._graph.has_node(a) else 0
                b_assoc_count = sum(
                    1 for _, _, d in self._graph.out_edges(b, data=True)
                    if d.get("edge_type") == EdgeType.ASSOCIATION.value
                ) if self._graph.has_node(b) else 0

                if a_assoc_count < _MAX_ASSOC_PER_NODE and b_assoc_count < _MAX_ASSOC_PER_NODE:
                    self.add_edge(a, b, EdgeType.ASSOCIATION, weight=0.3)
                    logger.info(
                        f"[MemoryGraph] 赫布学习创建 ASSOCIATION 边: {a} -> {b}"
                    )
                del self._coactivation_counter[pair]

        # 防止 counter 无限膨胀: 超过容量时丢弃最旧的条目
        if len(self._coactivation_counter) > _MAX_COACTIVATION_PAIRS:
            excess = len(self._coactivation_counter) - _MAX_COACTIVATION_PAIRS
            keys_to_remove = list(self._coactivation_counter.keys())[:excess]
            for k in keys_to_remove:
                del self._coactivation_counter[k]

    # ============================================================
    # 突触修剪 (衰减 + 清理)
    # ============================================================

    def decay_and_prune(self):
        """衰减非结构性边权，移除弱连接和孤立节点

        衰减公式: decayed = weight * exp(-elapsed_hours * ln(2) / half_life)
        不同 importance 标签的节点使用不同半衰期：
            trivial=6h, normal=24h, important=168h, fact=360h, identity=720h, must_remember=永不衰减
        边的半衰期取两端节点中更高重要度的值。
        """
        now = time.time()
        prune_threshold = 0.05
        review_threshold = 0.15  # 边权 < 0.15 但 > 0.05 → 候选 LLM 审查
        ln2 = math.log(2)
        edges_to_remove = []
        edges_pending_review = []

        for src, tgt, data in self._graph.edges(data=True):
            if data.get("protected"):
                continue
            # 正在等待 LLM 审查的边暂不处理
            if data.get("metadata", {}).get("pending_review"):
                continue

            # 获取两端节点的重要度，取更高的
            imp_src = self.get_importance(src) or Importance.NORMAL
            imp_tgt = self.get_importance(tgt) or Importance.NORMAL
            higher_imp = imp_src if _IMPORTANCE_ORDER.get(imp_src, 1) >= _IMPORTANCE_ORDER.get(imp_tgt, 1) else imp_tgt

            # must_remember 节点的边自动设为 protected，跳过衰减
            half_life = _IMPORTANCE_HALF_LIFE.get(higher_imp, 24.0)
            if half_life == float('inf'):
                data["protected"] = True
                continue

            elapsed_hours = (now - data.get("last_activated", data.get("created_at", now))) / 3600
            if elapsed_hours <= 0:
                continue

            decayed = data["weight"] * math.exp(-elapsed_hours * ln2 / half_life)

            if decayed < prune_threshold:
                # 太弱，直接移除
                edges_to_remove.append((src, tgt))
            elif decayed < review_threshold and higher_imp != Importance.TRIVIAL:
                # 濒危边且非 trivial → 候选 LLM 审查
                edges_pending_review.append((src, tgt))
                data.setdefault("metadata", {})["pending_review"] = True
                data.setdefault("metadata", {})["review_requested_at"] = now
            else:
                data["weight"] = decayed

        for src, tgt in edges_to_remove:
            self._graph.remove_edge(src, tgt)

        if edges_to_remove:
            logger.info(f"[MemoryGraph] 修剪了 {len(edges_to_remove)} 条弱边")
        if edges_pending_review:
            logger.info(f"[MemoryGraph] {len(edges_pending_review)} 条边待 LLM 审查")

        # 更新所有节点的温度缓存
        for node_id in list(self._nodes.keys()):
            self.update_temperature(node_id)

        # 移除孤立节点（度为0），按 importance 分级容忍时间
        nodes_to_remove = []
        for node_id in list(self._nodes.keys()):
            node = self._nodes[node_id]
            # TASK 类型节点永不因孤立而删除
            if node.node_type == NodeType.TASK:
                continue
            if self._graph.degree(node_id) == 0:
                imp = self.get_importance(node_id) or Importance.NORMAL
                imp_order = _IMPORTANCE_ORDER.get(imp, 1)
                # importance >= identity (order >= 2) → 永不因孤立删除（沉睡记忆）
                if imp_order >= _IMPORTANCE_ORDER[Importance.IDENTITY]:
                    continue
                # 按重要度选择容忍时间
                if imp == Importance.TRIVIAL:
                    orphan_age = 21600   # 6h
                elif imp_order >= _IMPORTANCE_ORDER[Importance.IMPORTANT]:
                    orphan_age = 604800  # 7 天
                else:
                    orphan_age = 86400   # 24h (normal 默认)
                if (now - node.last_accessed) > orphan_age:
                    nodes_to_remove.append(node_id)

        for nid in nodes_to_remove:
            self.remove_node(nid)

        if nodes_to_remove:
            logger.info(f"[MemoryGraph] 移除了 {len(nodes_to_remove)} 个孤立节点")

        self._stats["total_prune_cycles"] += 1
        self._stats["total_nodes"] = len(self._nodes)
        self._stats["total_edges"] = self._graph.number_of_edges()
        self._mark_dirty()

    def cleanup_orphan_nodes(self):
        """语义孤立节点清理 + 极短内容节点清理

        "语义孤立"定义: 节点的所有连边均为结构性边（TEMPORAL / HIERARCHY），
        没有任何语义关联（SEMANTIC / ASSOCIATION / REFERENCE / CAUSAL / DEPENDENCY）。
        这些节点是边剪枝后留下的"骨架空壳"。

        处理规则:
        - importance >= identity → 保留（沉睡记忆），不做任何处理
        - trivial/normal + cold → 第一次标记 semantically_isolated；
          第二次（仍满足条件）→ 移除
        - content 为空或极短（<5字符）的 trivial/normal 节点 → 直接移除
        """
        nodes_to_remove: List[str] = []
        now = time.time()

        for node_id in list(self._nodes.keys()):
            node = self._nodes[node_id]

            # TASK 节点永不清理
            if node.node_type == NodeType.TASK:
                continue

            imp = self.get_importance(node_id) or Importance.NORMAL
            imp_order = _IMPORTANCE_ORDER.get(imp, 1)

            # ---- 极短内容节点清理 ----
            content = node.metadata.get("content", "") or ""
            if isinstance(content, str) and len(content.strip()) < 5:
                # trivial/normal 的极短内容直接移除
                if imp_order <= _IMPORTANCE_ORDER[Importance.NORMAL]:
                    nodes_to_remove.append(node_id)
                    continue

            # ---- 语义孤立检测 ----
            # importance >= identity → 沉睡记忆，跳过
            if imp_order >= _IMPORTANCE_ORDER[Importance.IDENTITY]:
                continue

            # 检查是否语义孤立：所有边都是结构性边
            has_semantic_edge = False
            for _, _, edata in self._graph.edges(node_id, data=True):
                etype_val = edata.get("edge_type", "")
                try:
                    etype = EdgeType(etype_val) if isinstance(etype_val, str) else etype_val
                except ValueError:
                    etype = None
                if etype and etype not in _STRUCTURAL_EDGE_TYPES:
                    has_semantic_edge = True
                    break
            # 也检查入边
            if not has_semantic_edge:
                for _, _, edata in self._graph.in_edges(node_id, data=True):
                    etype_val = edata.get("edge_type", "")
                    try:
                        etype = EdgeType(etype_val) if isinstance(etype_val, str) else etype_val
                    except ValueError:
                        etype = None
                    if etype and etype not in _STRUCTURAL_EDGE_TYPES:
                        has_semantic_edge = True
                        break

            if has_semantic_edge:
                # 有语义边，清除可能的旧标记
                node.metadata.pop("semantically_isolated", None)
                continue

            # 到此：该节点语义孤立 + importance < identity
            temp = self.get_temperature(node_id)

            if temp != Temperature.COLD:
                # 非 cold 节点暂不处理，清除旧标记
                node.metadata.pop("semantically_isolated", None)
                continue

            # cold + 语义孤立
            if node.metadata.get("semantically_isolated"):
                # 第二次检测仍语义孤立 → 移除
                nodes_to_remove.append(node_id)
            else:
                # 第一次标记，等待下次审查
                node.metadata["semantically_isolated"] = True
                node.metadata["isolated_marked_at"] = now

        for nid in nodes_to_remove:
            self.remove_node(nid)

        if nodes_to_remove:
            logger.info(
                f"[MemoryGraph] 清理了 {len(nodes_to_remove)} 个语义孤立/极短内容节点"
            )

    async def _idle_review_loop(self):
        """空闲时自动批量审批待审批节点

        当系统空闲5分钟且有待审批节点时，启动节点审查流程。
        审查一旦启动，不会被新任务打断，直到所有待审查节点处理完毕。
        """
        _IDLE_THRESHOLD = 300.0  # 5分钟空闲阈值
        _IDLE_CHECK_INTERVAL = 60.0  # 每60秒检查一次空闲状态
        _IDLE_BATCH_SIZE = 10  # 每批审批10个节点
        _IDLE_BATCH_INTERVAL = 3.0  # 批次间间隔3秒，避免LLM过载

        while self._running:
            await asyncio.sleep(_IDLE_CHECK_INTERVAL)

            try:
                from zulong.core.state_manager import state_manager
                is_idle = state_manager.is_system_idle(_IDLE_THRESHOLD)
            except Exception:
                is_idle = False

            if not is_idle:
                continue

            # 检查 L2-PRIME 是否空闲，避免审查任务与用户请求竞争 LLM 资源
            try:
                from zulong.core.types import L2Status
                if state_manager._l2_status not in (L2Status.IDLE, L2Status.WAITING):
                    continue
            except Exception:
                pass

            with self._candidates_lock:
                pending = list(self._pending_llm_candidates)
            if not pending:
                continue

            logger.info(f"[MemoryGraph] 🔔 系统空闲5分钟，启动节点审查流程，共 {len(pending)} 个待审查节点（每批{_IDLE_BATCH_SIZE}个）")

            try:
                from zulong.memory.llm_memory_reviewer import get_llm_memory_reviewer
                reviewer = get_llm_memory_reviewer()
            except Exception:
                continue

            if not reviewer:
                continue

            total_reviewed = 0
            while self._running:
                with self._candidates_lock:
                    if not self._pending_llm_candidates:
                        break
                # 每批前检查：用户有活动则立即中断审查
                try:
                    is_idle = state_manager.is_system_idle(_IDLE_THRESHOLD)
                except Exception:
                    is_idle = False

                if not is_idle:
                    with self._candidates_lock:
                        remaining = len(self._pending_llm_candidates)
                    logger.info(f"[MemoryGraph] 用户活动，中断审查（已审查 {total_reviewed} 个，剩余 {remaining} 个）")
                    break

                with self._candidates_lock:
                    batch = list(self._pending_llm_candidates[:_IDLE_BATCH_SIZE])
                batch_ids = set(batch)
                candidate_memories = []
                for cid in batch:
                    node = self._nodes.get(cid)
                    if node:
                        imp = self.get_importance(cid)
                        candidate_memories.append({
                            "id": cid,
                            "label": getattr(node, 'label', ''),
                            "content": getattr(node, 'content', ''),
                            "importance": str(imp.value if imp else "NORMAL"),
                            "access_count": node.access_count
                        })

                if not candidate_memories:
                    with self._candidates_lock:
                        self._pending_llm_candidates = [c for c in self._pending_llm_candidates if c not in batch_ids]
                    continue

                try:
                    await reviewer.review_importance_candidates(candidate_memories)
                    reviewed_ids = set(m["id"] for m in candidate_memories)
                    with self._candidates_lock:
                        self._pending_llm_candidates = [c for c in self._pending_llm_candidates if c not in reviewed_ids]
                        remaining = len(self._pending_llm_candidates)
                    total_reviewed += len(reviewed_ids)
                    logger.debug(f"[MemoryGraph] 空闲审批: 提交 {len(reviewed_ids)} 个，累计 {total_reviewed}，剩余 {remaining}")
                except Exception as e:
                    logger.warning(f"[MemoryGraph] 空闲审批提交失败: {e}")
                    break

                await asyncio.sleep(_IDLE_BATCH_INTERVAL)

            if total_reviewed > 0:
                with self._candidates_lock:
                    final_remaining = len(self._pending_llm_candidates)
                logger.info(f"[MemoryGraph] 空闲审批完成: 共审批 {total_reviewed} 个节点，剩余待审批 {final_remaining} 个")
                self.save()

    async def start_prune_loop(self, interval_seconds: int = 1800):
        """启动异步修剪循环 (每30分钟，积压时自动加速)"""
        if self._running:
            return
        self._running = True

        async def _loop():
            cycle_count = 0
            while self._running:
                # 自适应间隔：积压时缩短等待时间
                with self._candidates_lock:
                    pending = len(self._pending_llm_candidates)
                if pending > 500:
                    actual_interval = min(interval_seconds, 300)
                elif pending > 200:
                    actual_interval = min(interval_seconds, 600)
                else:
                    actual_interval = interval_seconds
                await asyncio.sleep(actual_interval)
                try:
                    cycle_count += 1
                    self.decay_and_prune()
                    self.run_importance_review()
                    if cycle_count % 3 == 0:
                        self.cleanup_orphan_nodes()
                    if cycle_count % 2 == 0:
                        try:
                            await self.submit_prune_review()
                        except Exception as review_err:
                            logger.warning(f"[MemoryGraph] submit_prune_review 异常: {review_err}")
                    self.save()
                except Exception as e:
                    logger.error(f"[MemoryGraph] 修剪循环异常: {e}")

        self._prune_task = asyncio.create_task(_loop())
        # 启动空闲时自动审批循环
        self._idle_review_task = asyncio.create_task(self._idle_review_loop())
        logger.info(f"[MemoryGraph] 修剪循环已启动 (间隔 {interval_seconds}s)")

    def _try_start_prune_loop(self, interval_seconds: int = 1800):
        """尝试在现有事件循环中启动修剪循环（__init__中调用）"""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._start_prune_coro(interval_seconds))
            logger.info("[MemoryGraph] 修剪循环已在运行事件循环中调度")
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._start_prune_coro(interval_seconds))
                    logger.info("[MemoryGraph] 修剪循环已在事件循环中调度")
                else:
                    logger.info("[MemoryGraph] 无运行中事件循环，修剪循环将在首次retrieve时启动")
            except Exception:
                logger.info("[MemoryGraph] 无法获取事件循环，修剪循环延迟启动")

    async def _start_prune_coro(self, interval_seconds: int):
        """修剪循环的coroutine入口，等待首次调用后自动运行"""
        await asyncio.sleep(60)
        if not self._running:
            await self.start_prune_loop(interval_seconds)

    def stop_prune_loop(self):
        """停止异步修剪循环"""
        self._running = False
        if self._prune_task and not self._prune_task.done():
            self._prune_task.cancel()
        if hasattr(self, '_idle_review_task') and self._idle_review_task and not self._idle_review_task.done():
            self._idle_review_task.cancel()
        logger.info("[MemoryGraph] 修剪循环已停止")

    # ============================================================
    # LLM 剪枝守卫
    # ============================================================

    async def submit_prune_review(self):
        """提交 pending_review 边给 LLM 审查（异步，不阻塞修剪循环）

        收集所有标记了 pending_review 的边，构建记忆列表提交给 LLMMemoryReviewer。
        """
        pending_edges = []
        now = time.time()

        for src, tgt, data in self._graph.edges(data=True):
            meta = data.get("metadata", {})
            if not meta.get("pending_review"):
                continue
            # 超过 2 小时未返回审查结果 → 按 DISCARD 处理
            requested_at = meta.get("review_requested_at", now)
            elapsed = now - requested_at
            if elapsed > 7200:
                pending_edges.append((src, tgt, "timeout"))
            elif elapsed > 1800 and not meta.get("review_retried", False):
                # 超过30分钟未响应且未重试过 → 标记重试
                meta["review_retried"] = True
                meta["review_requested_at"] = now
                pending_edges.append((src, tgt, "pending"))
            else:
                pending_edges.append((src, tgt, "pending"))

        if not pending_edges:
            return

        # 超时的边直接移除
        timeout_edges = [(s, t) for s, t, status in pending_edges if status == "timeout"]
        for src, tgt in timeout_edges:
            if self._graph.has_edge(src, tgt):
                data = self._graph.edges[src, tgt]
                data.get("metadata", {}).pop("pending_review", None)
                self._graph.remove_edge(src, tgt)
        if timeout_edges:
            logger.info(f"[MemoryGraph] {len(timeout_edges)} 条边审查超时，已移除")

        # 待审查的边提交给 LLMMemoryReviewer
        review_edges = [(s, t) for s, t, status in pending_edges if status == "pending"]
        if not review_edges:
            return

        # 构建记忆列表供 LLM 审查
        memories = []
        for src, tgt in review_edges[:50]:  # 批量限制 50 条（从10提升）
            src_node = self.get_node(src)
            tgt_node = self.get_node(tgt)
            if src_node and tgt_node:
                memories.append({
                    "edge": (src, tgt),
                    "source_label": src_node.label,
                    "target_label": tgt_node.label,
                    "source_content": src_node.metadata.get("content", "")[:200],
                    "target_content": tgt_node.metadata.get("content", "")[:200],
                    "importance": max(
                        _IMPORTANCE_ORDER.get(self.get_importance(src) or Importance.NORMAL, 1),
                        _IMPORTANCE_ORDER.get(self.get_importance(tgt) or Importance.NORMAL, 1),
                    ),
                })

        try:
            from .llm_memory_reviewer import get_llm_memory_reviewer
            reviewer = get_llm_memory_reviewer()
            if reviewer:
                await reviewer.review_before_evict(
                    memories=[m for m in memories],
                    usage_ratio=0.5,
                    target_free=len(review_edges),
                )
                logger.info(f"[MemoryGraph] 已提交 {len(memories)} 条记忆给 LLM 审查")
        except Exception as e:
            logger.warning(f"[MemoryGraph] LLM 审查提交失败: {e}")

    def process_review_result(
        self, edge_key: Tuple[str, str], decision: str
    ):
        """处理 LLM 审查回调结果

        Args:
            edge_key: (source_id, target_id) 边标识
            decision: "KEEP" / "DISCARD" / "COMPRESS" / "PROMOTE" / "MERGE"
        """
        src, tgt = edge_key
        if not self._graph.has_edge(src, tgt):
            return

        data = self._graph.edges[src, tgt]
        # 清除 pending 标记
        data.get("metadata", {}).pop("pending_review", None)
        data.get("metadata", {}).pop("review_requested_at", None)

        decision_upper = decision.upper()

        if decision_upper == "KEEP":
            # 强化边权 + 提升重要度
            data["weight"] = max(data["weight"], 0.3)
            data["last_activated"] = time.time()
            for nid in (src, tgt):
                imp = self.get_importance(nid) or Importance.NORMAL
                if _IMPORTANCE_ORDER.get(imp, 1) < _IMPORTANCE_ORDER[Importance.IMPORTANT]:
                    self.promote_importance(nid, Importance.IMPORTANT)

        elif decision_upper == "DISCARD":
            self._graph.remove_edge(src, tgt)

        elif decision_upper == "COMPRESS":
            # 保留边但压缩内容
            for nid in (src, tgt):
                node = self.get_node(nid)
                if node and node.metadata.get("content"):
                    # 截断为摘要
                    content = node.metadata["content"]
                    if len(content) > 100:
                        node.metadata["content"] = content[:100] + "..."
                        node.metadata["compressed"] = True
            data["weight"] = 0.2  # 重置边权
            data["last_activated"] = time.time()

        elif decision_upper == "PROMOTE":
            for nid in (src, tgt):
                self.promote_importance(nid, Importance.IMPORTANT)
            data["weight"] = max(data["weight"], 0.3)

        elif decision_upper == "MERGE":
            # 将 target 内容合并到 source
            src_node = self.get_node(src)
            tgt_node = self.get_node(tgt)
            if src_node and tgt_node:
                merged = src_node.metadata.get("content", "") + "\n" + tgt_node.metadata.get("content", "")
                src_node.metadata["content"] = merged[:500]

        self._mark_dirty()

    # ============================================================
    # Embedding 管理
    # ============================================================

    def _auto_embed_node(self, node: GraphNode):
        """为新节点自动生成 embedding 并触发语义邻居发现

        提取节点文本内容 → EmbeddingModelManager → set_embedding →
        discover_semantic_neighbors（自动创建 SEMANTIC 边）。
        同步执行，失败静默降级。
        """
        if np is None:
            return

        # 提取有意义的文本内容
        text = (
            node.metadata.get("content")
            or node.metadata.get("goal")
            or node.metadata.get("desc")
            or node.metadata.get("summary")
            or node.label
        )
        if not text or len(str(text).strip()) < 10:
            return  # 内容太短，不值得向量化

        # 已有 embedding 则跳过
        if node.node_id in self._embeddings:
            return

        try:
            emb_mgr = self._summary_index._get_embedding_manager()
            if emb_mgr is None:
                return
            vector = emb_mgr.encode_document(str(text)[:512])
            if vector is not None:
                self.set_embedding(node.node_id, vector)
                # 有足够多节点后再做语义发现（图太小时无意义）
                if len(self._embeddings) >= 3:
                    self.discover_semantic_neighbors(
                        node.node_id, top_k=3, threshold=0.75
                    )
        except Exception as e:
            logger.debug(f"[MemoryGraph] 自动 embedding 失败 {node.node_id}: {e}")

    def set_embedding(self, node_id: str, embedding) -> bool:
        """设置节点的 embedding 向量"""
        if node_id not in self._nodes:
            return False
        if np is not None:
            self._embeddings[node_id] = np.array(embedding, dtype=np.float32)
        return True

    def get_embedding(self, node_id: str):
        """获取节点的 embedding 向量"""
        return self._embeddings.get(node_id)

    # ============================================================
    # 语义边发现 (FAISS 侧车索引)
    # ============================================================

    def discover_semantic_neighbors(
        self,
        node_id: str,
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> List[Tuple[str, float]]:
        """按需发现语义邻居并创建 SEMANTIC 边

        使用内积计算 (embeddings 已归一化 = cosine similarity)

        Args:
            node_id: 查询节点
            top_k: 返回最近邻数量
            threshold: 相似度阈值

        Returns:
            List[(neighbor_id, similarity)]
        """
        if np is None:
            return []

        query_emb = self._embeddings.get(node_id)
        if query_emb is None:
            return []

        # 简单向量搜索 (不依赖 FAISS，小规模图直接 numpy)
        results = []
        for other_id, other_emb in self._embeddings.items():
            if other_id == node_id:
                continue
            # cosine similarity (embeddings 已归一化)
            sim = float(np.dot(query_emb, other_emb))
            if sim >= threshold:
                results.append((other_id, sim))

        # 按相似度排序，取 top_k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        # 创建 SEMANTIC 边
        for neighbor_id, sim in results:
            if not self.has_edge(node_id, neighbor_id):
                self.add_edge(
                    node_id, neighbor_id,
                    EdgeType.SEMANTIC,
                    weight=sim,
                )

        return results

    # ============================================================
    # FAISS 摘要索引操作（供适配器调用）
    # ============================================================

    def index_summary(self, node_id: str, summary_text: str) -> bool:
        """为节点添加摘要到 FAISS 侧车索引

        适配器在 finalize_round() / EpisodeAdapter.sync() 时调用。
        """
        return self._summary_index.add_summary(node_id, summary_text)

    def search_summaries(
        self,
        query_text: str,
        top_k: int = 5,
        exclude_node_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """向量检索摘要索引，返回 (node_id, score) 列表"""
        return self._summary_index.search(query_text, top_k, exclude_node_ids)

    @property
    def summary_index_count(self) -> int:
        """摘要索引中的条目数量"""
        return self._summary_index.count

    # ============================================================
    # 并行检索策略（热数据遍历 + 非热数据向量检索）
    # ============================================================

    async def retrieve_context(
        self,
        query_text: str,
        top_k: int = 10,
        hot_window_minutes: int = 30,
        session_id: str = "",
    ) -> List[Dict[str, Any]]:
        """核心检索方法：在同一张图上通过标签互斥过滤实现两条并行检索路径

        路径 A: 遍历热数据（time_scope=recent），排除温/冷数据
        路径 B: FAISS 向量检索非热数据，排除热数据

        两条路径通过 asyncio.gather() 并行执行，结果合并后按分数排序。

        Args:
            query_text: 用户查询文本
            top_k: 最终返回的 Top-K 结果数
            hot_window_minutes: 热窗口分钟数（默认 30 分钟）
            session_id: 当前会话 ID（用于会话内节点 boost）

        Returns:
            [{"node_id", "node_type", "label", "content", "score", "source"}, ...]
        """
        window_seconds = hot_window_minutes * 60

        # 并行执行两条路径
        loop = asyncio.get_event_loop()
        hot_task = loop.run_in_executor(
            None, self._retrieve_hot, query_text, window_seconds, session_id
        )
        cold_task = loop.run_in_executor(
            None, self._retrieve_cold, query_text, top_k, window_seconds
        )
        hot_results, cold_results = await asyncio.gather(hot_task, cold_task)

        # 合并结果（两路互斥，天然无重复），按 score 降序排序
        combined = hot_results + cold_results
        combined.sort(key=lambda x: x["score"], reverse=True)
        final = combined[:top_k]

        # 缓存 top-3 命中节点 ID，供 BFS 种子扩展使用
        self._last_retrieved_node_ids = [
            r["node_id"] for r in final[:3] if r.get("node_id")
        ]

        # P2-8: 检索结果共激活赫布通路
        # 类比：人同时回忆起多件事时，这些记忆因"同时出现在意识中"而建立弱关联
        if len(self._last_retrieved_node_ids) >= 2:
            _coact_pairs = []
            _rids = self._last_retrieved_node_ids[:5]  # 限制为 top-5 避免过多配对
            for i in range(len(_rids)):
                for j in range(i + 1, len(_rids)):
                    _coact_pairs.append((_rids[i], _rids[j]))
            if _coact_pairs:
                self._update_coactivation_counter(_coact_pairs)

        return final

    @staticmethod
    def _bigram_overlap_score(query: str, text: str) -> float:
        """基于 bigram 重叠的中文关键词匹配

        当完全子串匹配失败时，使用 bigram 交集比作为回退。
        例如 "帮我写代码" 的 bigrams {"帮我","我写","写代","代码"}
        能与 "编写Python代码" 的 bigrams 交集到 {"代码"}。

        Returns:
            0.0 ~ 0.6 的匹配分数，overlap < 30% 返回 0
        """
        if not query or not text or len(query) < 2:
            return 0.0
        q_bigrams = {query[i:i+2] for i in range(len(query)-1)}
        t_bigrams = {text[i:i+2] for i in range(len(text)-1)}
        if not q_bigrams:
            return 0.0
        overlap = len(q_bigrams & t_bigrams) / len(q_bigrams)
        return overlap * 0.6 if overlap >= 0.2 else 0.0

    def _compute_semantic_similarity(self, query_text: str, node: GraphNode, query_embedding=None) -> float:
        """计算查询文本与节点的语义相似度

        优先使用 query_embedding 与节点 embedding 的余弦相似度，
        退化时回退到词汇重叠 + embedding丰富度代理。

        Args:
            query_text: 查询文本
            node: 目标节点
            query_embedding: 查询文本的向量表示（可选，优先使用）

        Returns:
            0.0 ~ 1.0 的语义相似度分数
        """
        if np is None or node.node_id not in self._embeddings:
            return 0.0
        
        try:
            node_embedding = self._embeddings[node.node_id]
            
            if query_embedding is not None:
                query_vec = np.asarray(query_embedding, dtype=np.float32)
                node_vec = np.asarray(node_embedding, dtype=np.float32)
                q_norm = np.linalg.norm(query_vec)
                n_norm = np.linalg.norm(node_vec)
                if q_norm > 1e-8 and n_norm > 1e-8:
                    cosine_sim = float(np.dot(query_vec, node_vec) / (q_norm * n_norm))
                    return max(0.0, (cosine_sim + 1.0) / 2.0)
            
            node_text_parts = [node.label]
            for field in ["content", "desc", "goal"]:
                field_value = node.metadata.get(field, "")
                if field_value and isinstance(field_value, str):
                    node_text_parts.append(field_value)
            node_text = " ".join(node_text_parts)
            
            query_words = set(query_text.lower().split())
            node_words = set(node_text.lower().split())
            
            if not query_words or not node_words:
                word_overlap = 0.0
            else:
                overlap = len(query_words & node_words)
                word_overlap = overlap / len(query_words)
            
            embedding_norm = np.linalg.norm(node_embedding)
            semantic_richness = min(embedding_norm / 10.0, 1.0)
            
            semantic_score = 0.7 * word_overlap + 0.3 * semantic_richness
            
            return semantic_score if word_overlap > 0.1 else 0.0
            
        except Exception as e:
            logger.debug(f"[MemoryGraph] 语义相似度计算失败: {e}")
            return 0.0

    def _retrieve_hot(
        self, query_text: str, window_seconds: int, session_id: str = ""
    ) -> List[Dict[str, Any]]:
        """路径 A: 热数据图遍历（排除温/冷数据）

        筛选热窗口内的节点，关键词匹配 + BFS 扩散。
        当提供 session_id 时，优先返回当前会话节点。
        当检测到元记忆查询（"记住/记得/回忆"）时，主动注入身份/知识节点。
        """
        query_lower = query_text.lower()
        results = []
        now = time.time()

        # --- 0. 当前会话最近对话（最高优先级，确保当轮对话总被检索到） ---
        if session_id:
            for nid, node in self._nodes.items():
                if (node.metadata.get("session_id") == session_id
                        and self.is_recent(nid, window_seconds)):
                    results.append(self._node_to_result(node, 2.0, "session_current"))
            if results:
                logger.debug(f"[MemoryGraph] 当前会话注入 {len(results)} 个节点")
        else:
            # 兜底：无 session_id 时，注入最近 5 分钟内的对话节点
            _recent_window = 300  # 5 分钟
            recent_dialogues = []
            for nid, node in self._nodes.items():
                if (node.node_type == NodeType.DIALOGUE
                        and (now - node.last_accessed) < _recent_window):
                    recent_dialogues.append((nid, node, node.last_accessed))
            recent_dialogues.sort(key=lambda x: x[2], reverse=True)
            for nid, node, _ in recent_dialogues[:6]:
                results.append(self._node_to_result(node, 1.8, "recent_dialogue"))
            if recent_dialogues:
                logger.debug(f"[MemoryGraph] 兜底注入 {min(6, len(recent_dialogues))} 个最近对话节点")

        # --- 元记忆查询检测 ---
        _meta_kw = ("记住", "记得", "回忆", "记忆", "知道关于", "了解关于",
                     "还记得", "记下", "保存的信息", "关于我")
        is_meta_query = any(kw in query_lower for kw in _meta_kw)

        if is_meta_query:
            # 收集候选节点，按 recency 加权（避免旧节点垄断结果）
            meta_candidates = []
            for nid, node in self._nodes.items():
                if any(r["node_id"] == nid for r in results):
                    continue
                imp = self.get_importance(nid)
                if imp in (Importance.IDENTITY, Importance.FACT,
                           Importance.MUST_REMEMBER, Importance.IMPORTANT):
                    age_hours = (now - node.last_accessed) / 3600
                    recency = 1.0 / (1.0 + age_hours * 0.3)
                    meta_candidates.append((nid, node, 1.5 * recency))
            for nid, node in self._nodes.items():
                if any(r["node_id"] == nid for r in results):
                    continue
                if node.node_type in (NodeType.PERSON, NodeType.CONCEPT, NodeType.KNOWLEDGE):
                    if any(c[0] == nid for c in meta_candidates):
                        continue
                    age_hours = (now - node.last_accessed) / 3600
                    recency = 1.0 / (1.0 + age_hours * 0.3)
                    meta_candidates.append((nid, node, 1.2 * recency))
            # 按分数排序，取 top-15 避免大量旧节点
            meta_candidates.sort(key=lambda x: x[2], reverse=True)
            for nid, node, score in meta_candidates[:15]:
                results.append(self._node_to_result(node, score, "meta_recall"))
            if meta_candidates:
                logger.debug(f"[MemoryGraph] 元记忆查询命中 {len(meta_candidates)} 个候选, 取 top-{min(15, len(meta_candidates))}")

        # P2-9: 热路径增加向量语义检索
        # 1. 筛选热节点并做关键词匹配 + 语义相似度计算
        hot_seed_ids = []
        
        # 预计算查询向量（如果可能）
        query_embedding = None
        if np is not None and len(query_text) >= 3:
            try:
                emb_mgr = self._summary_index._get_embedding_manager()
                if emb_mgr is not None:
                    query_embedding = emb_mgr.encode_document(query_text[:512])
            except Exception:
                pass
        
        # 收集热节点的关键词得分和语义得分
        hot_node_scores = {}
        for node_id, node in self._nodes.items():
            if not self.is_recent(node_id, window_seconds):
                continue
            
            # 关键词匹配得分
            keyword_score = 0.0
            if query_lower in node.label.lower():
                keyword_score = 0.8
            else:
                content = node.metadata.get("content", "") or ""
                goal = node.metadata.get("goal", "") or ""
                desc = node.metadata.get("desc", "") or ""
                combined_text = f"{content} {goal} {desc}".lower()
                if query_lower in combined_text:
                    keyword_score = 0.5
                else:
                    keyword_score = self._bigram_overlap_score(query_lower, combined_text)
            
            # 语义相似度得分（仅对有关键词匹配或 embedding 的节点）
            semantic_score = 0.0
            if (keyword_score > 0 or query_embedding is not None) and node_id in self._embeddings:
                semantic_score = self._compute_semantic_similarity(query_text, node, query_embedding=query_embedding)
            
            # 合并得分：60% 关键词 + 40% 语义
            if keyword_score > 0 or semantic_score > 0:
                combined_score = 0.6 * keyword_score + 0.4 * semantic_score
                hot_node_scores[node_id] = {
                    'keyword_score': keyword_score,
                    'semantic_score': semantic_score,
                    'combined_score': combined_score,
                    'node': node
                }
        
        # 按合并得分排序并添加到结果
        sorted_hot_nodes = sorted(
            hot_node_scores.items(), 
            key=lambda x: x[1]['combined_score'], 
            reverse=True
        )
        
        for node_id, scores in sorted_hot_nodes:
            node = scores['node']
            combined_score = scores['combined_score']
            
            if combined_score > 0:
                hot_seed_ids.append(node_id)
                imp = self.get_importance(node_id) or Importance.NORMAL
                imp_boost = {
                    Importance.IDENTITY: 0.15,
                    Importance.FACT: 0.15,
                    Importance.IMPORTANT: 0.2,
                    Importance.MUST_REMEMBER: 0.3,
                }.get(imp, 0.0)
                
                # 基础得分计算
                final_score = combined_score * 1.5 + imp_boost + node.activation * 0.1
                
                # session_id boost: 当前会话节点 +0.3
                if session_id and node.metadata.get("session_id") == session_id:
                    final_score += 0.3
                
                # 焦点子树 boost: 当前注意力路径内的节点 +0.2
                if self._last_focus_context and self._last_focus_context.get("focus_path"):
                    _fp = self._last_focus_context["focus_path"]
                    if node_id in _fp:
                        final_score += 0.2
                
                if any(r["node_id"] == node_id for r in results):
                    continue
                
                # 标注检索来源
                source_type = "hot_hybrid"
                if scores['keyword_score'] > 0 and scores['semantic_score'] > 0:
                    source_type = "hot_hybrid"
                elif scores['semantic_score'] > 0:
                    source_type = "hot_semantic"
                else:
                    source_type = "hot_keyword"
                
                results.append(self._node_to_result(node, final_score, source_type))

        # 2. BFS 扩散: 从匹配的热节点向外 1 跳，仍只取热节点
        # 焦点路径节点注入 BFS 种子（取最后 3 个，即最接近当前焦点的节点）
        if self._last_focus_context and self._last_focus_context.get("focus_path"):
            _focus_seeds = self._last_focus_context["focus_path"][-3:]
            for fid in _focus_seeds:
                if fid not in hot_seed_ids and self.is_recent(fid, window_seconds):
                    hot_seed_ids.append(fid)

        if hot_seed_ids:
            for seed_id in hot_seed_ids[:10]:
                neighbors = self.get_neighbors(seed_id, max_depth=1)
                for neighbor in neighbors:
                    if not self.is_recent(neighbor.node_id, window_seconds):
                        continue
                    if any(r["node_id"] == neighbor.node_id for r in results):
                        continue
                    bfs_score = neighbor.activation * 1.2
                    if session_id and neighbor.metadata.get("session_id") == session_id:
                        bfs_score += 0.3
                    results.append(
                        self._node_to_result(neighbor, bfs_score, "hot_bfs")
                    )

        return results

    def _retrieve_cold(
        self, query_text: str, top_k: int, window_seconds: int
    ) -> List[Dict[str, Any]]:
        """路径 B: 非热数据混合检索（FAISS 向量 + 关键词并行，排除热数据）

        P0-3 修正: 原冷路径仅依赖 FAISS 向量检索，缺失关键词匹配导致
        长期低热度但关键词高度相关的记忆被遗漏。现在同时执行向量检索和
        关键词匹配，以 0.7*向量 + 0.3*关键词 的权重融合。

        通过 FAISS 摘要索引检索，命中后 BFS 下钻获取详情。
        """
        # 收集热节点 ID 用于排除
        hot_node_ids = set()
        for node_id in self._nodes:
            if self.is_recent(node_id, window_seconds):
                hot_node_ids.add(node_id)

        # P0-3: 并行执行向量检索和关键词检索
        faiss_results = self._summary_index.search(
            query_text, top_k=top_k * 2, exclude_node_ids=hot_node_ids
        )
        keyword_results = self._summary_index.keyword_search(
            query_text, top_k=top_k * 2, exclude_node_ids=hot_node_ids
        )

        # P0-3: 融合两路检索结果 (0.7 向量 + 0.3 关键词)
        _VECTOR_WEIGHT = 0.7
        _KEYWORD_WEIGHT = 0.3
        merged_scores: Dict[str, float] = {}
        for node_id, vec_score in faiss_results:
            merged_scores[node_id] = _VECTOR_WEIGHT * vec_score
        for node_id, kw_score in keyword_results:
            merged_scores[node_id] = merged_scores.get(node_id, 0.0) + _KEYWORD_WEIGHT * kw_score

        results = []
        # 按融合分数排序
        for node_id, base_score in sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            node = self._nodes.get(node_id)
            if not node:
                continue
            imp = self.get_importance(node_id) or Importance.NORMAL
            imp_boost = {
                Importance.IDENTITY: 0.1,
                Importance.FACT: 0.1,
                Importance.IMPORTANT: 0.15,
                Importance.MUST_REMEMBER: 0.2,
            }.get(imp, 0.0)
            # 焦点子树 boost (冷数据路径)
            focus_boost = 0.0
            if self._last_focus_context and self._last_focus_context.get("focus_path"):
                _cold_fp = set(self._last_focus_context["focus_path"])
                if node_id in _cold_fp:
                    focus_boost = 0.15
            final_score = base_score + imp_boost + focus_boost
            results.append(self._node_to_result(node, final_score, "cold_hybrid"))

            # BFS 下钻: 获取命中摘要节点的子节点详情
            children = self.get_neighbors(node_id, max_depth=1)
            for child in children[:5]:  # 限制子节点数量
                if child.node_id in hot_node_ids:
                    continue
                if any(r["node_id"] == child.node_id for r in results):
                    continue
                child_score = final_score * 0.6  # 子节点分数衰减
                results.append(self._node_to_result(child, child_score, "cold_drill"))

        return results

    def _node_to_result(
        self, node: GraphNode, score: float, source: str
    ) -> Dict[str, Any]:
        """将 GraphNode 转换为检索结果字典

        对 KNOWLEDGE / EXPERIENCE 类型节点，如果本地 content 为空或极短，
        自动通过 resolve_backend_ref() 获取后端完整内容。
        """
        content = (
            node.metadata.get("content")
            or node.metadata.get("goal")
            or node.metadata.get("desc")
            or node.metadata.get("summary")
            or node.label
        )

        # KNOWLEDGE / EXPERIENCE 节点：尝试 backend_ref 反查补充完整内容
        if node.node_type in (NodeType.KNOWLEDGE, NodeType.EXPERIENCE):
            if not content or len(str(content)) < 20:
                backend_data = self.resolve_backend_ref(node.node_id)
                if backend_data and backend_data.get("content"):
                    content = backend_data["content"]

        return {
            "node_id": node.node_id,
            "node_type": node.node_type.value,
            "label": node.label,
            "content": content,
            "score": round(score, 4),
            "source": source,
            "importance": node.metadata.get("importance", "normal"),
            "metadata": node.metadata,
        }

    # ============================================================
    # Backend Ref 反查
    # ============================================================

    def set_rag_manager(self, rag_manager):
        """注入 RAGManager 供 backend_ref 反查使用"""
        self._rag_manager = rag_manager

    def resolve_backend_ref(self, node_id: str) -> Optional[Dict[str, Any]]:
        """通过 backend_ref 反查后端获取完整数据

        支持前缀路由:
        - knowledge_graph:{entity_id}  → KnowledgeGraph 实体
        - experience_rag:{doc_id}      → ExperienceRAG 文档
        - knowledge_rag:{doc_id}       → KnowledgeRAG 文档（如存在）

        Args:
            node_id: 图节点 ID

        Returns:
            包含后端完整数据的字典，反查失败返回 None
        """
        node = self._nodes.get(node_id)
        if not node or not node.backend_ref:
            return None

        ref = node.backend_ref
        parts = ref.split(":", 1)
        if len(parts) != 2:
            return None

        backend_type, backend_id = parts[0], parts[1]

        try:
            if backend_type == "knowledge_graph":
                return self._resolve_knowledge_graph(backend_id)
            elif backend_type in ("experience_rag", "knowledge", "memory"):
                return self._resolve_rag_doc(backend_type, backend_id)
            elif backend_type == "task_graph":
                # task_graph 节点信息已在图内，直接返回 metadata
                return {"source": "task_graph", "ref": backend_id, "data": node.metadata}
            else:
                logger.debug(f"[MemoryGraph] 未知 backend_ref 类型: {backend_type}")
                return None
        except Exception as e:
            logger.warning(f"[MemoryGraph] resolve_backend_ref 失败 ({ref}): {e}")
            return None

    def _resolve_knowledge_graph(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """从 KnowledgeGraph 获取完整实体信息"""
        try:
            from .knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            if not hasattr(kg, 'entities') or entity_id not in kg.entities:
                return None
            entity = kg.entities[entity_id]
            return {
                "source": "knowledge_graph",
                "entity_id": entity_id,
                "name": entity.name,
                "entity_type": entity.entity_type.value,
                "confidence": entity.confidence,
                "attributes": dict(entity.attributes),
                "content": f"{entity.name}: {dict(entity.attributes)}",
            }
        except Exception:
            return None

    def _resolve_rag_doc(self, lib_name: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """从 RAGManager 获取完整文档"""
        rag_mgr = getattr(self, "_rag_manager", None)
        if not rag_mgr:
            return None
        # 将 backend_ref 前缀映射到 RAGManager 的 library 名称
        lib_map = {
            "experience_rag": "experience",
            "knowledge": "knowledge",
            "memory": "memory",
        }
        actual_lib = lib_map.get(lib_name, lib_name)
        lib = rag_mgr.rag_libraries.get(actual_lib)
        if not lib:
            return None
        try:
            if hasattr(lib, 'index') and isinstance(lib.index, dict):
                doc = lib.index.get(doc_id)
                if doc:
                    content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
                    return {
                        "source": actual_lib,
                        "doc_id": doc_id,
                        "content": content,
                    }
            # 备选: 通过 search 查找
            results = rag_mgr.search(actual_lib, doc_id, top_k=1)
            if results:
                doc = results[0]
                return {
                    "source": actual_lib,
                    "doc_id": getattr(doc, "doc_id", doc_id),
                    "content": getattr(doc, "content", ""),
                }
        except Exception:
            pass
        return None

    # ============================================================
    # 适配器注册
    # ============================================================

    def register_adapter(self, name: str, adapter):
        """注册后端适配器"""
        self._adapters[name] = adapter
        logger.debug(f"[MemoryGraph] 注册适配器: {name}")

    def sync_all(self, sources: Optional[Dict[str, Any]] = None):
        """调用所有已注册适配器进行全量同步

        Args:
            sources: 可选的后端实例映射 {adapter_name: source_instance}
        """
        sources = sources or {}
        total = 0
        for name, adapter in self._adapters.items():
            try:
                source = sources.get(name)
                count = adapter.sync(self, source)
                total += count
                logger.debug(f"[MemoryGraph] 适配器 {name} 同步了 {count} 个节点")
            except Exception as e:
                logger.warning(f"[MemoryGraph] 适配器 {name} 同步失败: {e}")

        self._stats["total_nodes"] = len(self._nodes)
        self._stats["total_edges"] = self._graph.number_of_edges()
        logger.info(f"[MemoryGraph] 全量同步完成: 共 {total} 个节点更新")

    # ============================================================
    # 持久化
    # ============================================================

    def _mark_dirty(self):
        """标记数据已变更，调度防抖保存

        写操作（add_node / add_edge / remove_node / remove_edge）调用此方法。
        采用防抖策略：最后一次变更后延迟 _auto_save_delay 秒再落盘，
        避免频繁写入，同时保证数据不会长时间停留在内存中。
        """
        self._dirty = True
        # 取消上一个未触发的定时器，重新计时
        if self._auto_save_timer is not None:
            self._auto_save_timer.cancel()
        self._auto_save_timer = threading.Timer(
            self._auto_save_delay, self._do_auto_save,
        )
        self._auto_save_timer.daemon = True
        self._auto_save_timer.start()

    def _do_auto_save(self):
        """定时器回调：执行实际的磁盘保存"""
        if not self._dirty:
            return
        try:
            self.save()
        except Exception as e:
            logger.warning(f"[MemoryGraph] 自动保存失败: {e}")

    def _atexit_flush(self):
        """P2-3: 进程退出安全网 — 取消挂起的定时器并立即刷盘"""
        if self._auto_save_timer is not None:
            self._auto_save_timer.cancel()
            self._auto_save_timer = None
        if self._dirty:
            try:
                self.save()
                logger.info("[MemoryGraph] atexit: 已刷盘未保存的变更")
            except Exception as e:
                logger.error(f"[MemoryGraph] atexit: 刷盘失败: {e}")

    def save(self) -> bool:
        """保存记忆图谱到磁盘（原子写入：temp + rename，防止崩溃导致数据损坏）"""
        with self._save_lock:
            try:
                filepath = os.path.join(self.persist_path, "memory_graph.json")
                temp_filepath = filepath + ".tmp"
                backup_filepath = filepath + ".bak"

                data = {
                    "version": self._SCHEMA_VERSION,
                    "nodes": {
                        nid: node.to_dict() for nid, node in self._nodes.items()
                    },
                    "edges": [
                        {"source": u, "target": v, **d}
                        for u, v, d in self._graph.edges(data=True)
                    ],
                    "embeddings": self._serialize_embeddings(),
                    "coactivation_counter": {
                        f"{a}||{b}": count
                        for (a, b), count in self._coactivation_counter.items()
                    },
                    "meta": {
                        "saved_at": time.time(),
                        "node_count": len(self._nodes),
                        "edge_count": self._graph.number_of_edges(),
                        "stats": self._stats,
                        "last_focus_context": self._last_focus_context,
                    },
                }

                # Step 1: 写入临时文件（P1-1: 添加 fsync 确保数据刷入磁盘）
                with open(temp_filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())

                # Step 2: 备份当前文件（安全网）
                if os.path.exists(filepath):
                    try:
                        if os.path.exists(backup_filepath):
                            os.remove(backup_filepath)
                        os.rename(filepath, backup_filepath)
                    except OSError:
                        pass  # 备份失败不阻塞保存

                # Step 3: 原子替换（同卷 rename 在 NTFS/POSIX 上是原子的）
                os.replace(temp_filepath, filepath)

                # Step 4: 保存 FAISS 摘要侧车索引（独立于 JSON）
                try:
                    sidecar_path = os.path.join(self.persist_path, "summary_sidecar")
                    self._summary_index.save(sidecar_path)
                except Exception as e:
                    logger.warning(f"[MemoryGraph] FAISS 侧车索引保存失败: {e}")

                self._dirty = False
                self._last_save_time = time.time()
                logger.info(
                    f"[MemoryGraph] 已保存: {len(self._nodes)} 节点, "
                    f"{self._graph.number_of_edges()} 边"
                )
                return True
            except Exception as e:
                logger.error(f"[MemoryGraph] 保存失败: {e}")
                return False

    _SCHEMA_VERSION = "1.0"

    def _load(self) -> bool:
        """从磁盘加载记忆图谱（支持 .bak 崩溃恢复）"""
        filepath = os.path.join(self.persist_path, "memory_graph.json")
        backup_filepath = filepath + ".bak"

        # 尝试主文件 -> 备份文件
        for candidate, label in [(filepath, "主文件"), (backup_filepath, "备份文件")]:
            if not os.path.exists(candidate):
                continue
            try:
                with open(candidate, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 版本号校验
                file_version = data.get("version", "unknown")
                if file_version != self._SCHEMA_VERSION:
                    logger.warning(
                        f"[MemoryGraph] {label}版本不匹配: 文件={file_version}, "
                        f"当前={self._SCHEMA_VERSION}，将尝试兼容加载"
                    )

                # 恢复节点
                for nid, ndata in data.get("nodes", {}).items():
                    node = GraphNode.from_dict(ndata)
                    self._nodes[nid] = node
                    self._graph.add_node(nid, **node.to_dict())

                # P1-4: 重建地址反向索引
                self._address_index.clear()
                for nid, node in self._nodes.items():
                    self._index_node_address(nid, node.metadata)

                # 恢复边（验证节点存在性，跳过孤儿边）
                _skipped_edges = 0
                for edge in data.get("edges", []):
                    edge_copy = dict(edge)
                    source = edge_copy.pop("source")
                    target = edge_copy.pop("target")
                    if source not in self._nodes or target not in self._nodes:
                        _skipped_edges += 1
                        continue
                    self._graph.add_edge(source, target, **edge_copy)
                if _skipped_edges:
                    logger.warning(
                        f"[MemoryGraph] 跳过 {_skipped_edges} 条孤儿边（引用不存在的节点）"
                    )

                # 恢复 embeddings
                self._deserialize_embeddings(data.get("embeddings", {}))

                # 恢复共激活计数器
                for key, count in data.get("coactivation_counter", {}).items():
                    parts = key.split("||")
                    if len(parts) == 2:
                        self._coactivation_counter[(parts[0], parts[1])] = count

                # 恢复统计
                meta = data.get("meta", {})
                self._stats = meta.get("stats", self._stats)

                # 恢复焦点上下文（用于重启后恢复注意力）
                self._last_focus_context = meta.get("last_focus_context")
                if self._last_focus_context:
                    restored_ids = self._last_focus_context.get("active_node_ids", [])
                    self._active_node_ids = set(restored_ids)

                logger.info(
                    f"[MemoryGraph] 已加载({label}): "
                    f"{len(self._nodes)} 节点, {self._graph.number_of_edges()} 边"
                )

                # 尝试加载 FAISS 摘要侧车索引
                try:
                    sidecar_path = os.path.join(self.persist_path, "summary_sidecar")
                    self._summary_index.load(sidecar_path)
                except Exception as e:
                    logger.warning(f"[MemoryGraph] FAISS 侧车索引加载失败(不影响图谱): {e}")

                return True
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"[MemoryGraph] {label}损坏({e})，尝试备份...")
                continue
            except Exception as e:
                logger.error(f"[MemoryGraph] 加载{label}失败: {e}")
                continue

        logger.info("[MemoryGraph] 无可用数据，从空图谱开始")
        return False

    def _sanitize_association_edges(self):
        """启动时健康检查: 清理膨胀的 ASSOCIATION 边

        当 ASSOCIATION 边数量超过节点数的 MAX_ASSOC_RATIO 倍时,
        视为数据异常(历史 bug 导致的 O(N²) 边爆炸), 执行清理:
        - 每个节点只保留权重最高的 MAX_ASSOC_PER_NODE 条 ASSOCIATION 出边
        - 其余全部删除

        清理后自动触发一次保存。
        """
        _MAX_ASSOC_RATIO = 15  # ASSOCIATION 边数 / 节点数 > 此值触发清理
        _MAX_ASSOC_PER_NODE = 10  # 每节点最多保留的 ASSOCIATION 出边数

        node_count = len(self._nodes)
        if node_count == 0:
            return

        # 统计 ASSOCIATION 边数量
        assoc_edges = [
            (u, v, d) for u, v, d in self._graph.edges(data=True)
            if d.get("edge_type") == EdgeType.ASSOCIATION.value
        ]
        assoc_count = len(assoc_edges)

        if assoc_count <= node_count * _MAX_ASSOC_RATIO:
            return  # 健康, 无需清理

        logger.warning(
            f"[MemoryGraph] ⚠️ ASSOCIATION 边数异常: {assoc_count} 条 "
            f"(节点 {node_count}, 比例 {assoc_count/node_count:.0f}:1), "
            f"执行清理..."
        )

        # 按源节点分组, 每个源只保留权重最高的 _MAX_ASSOC_PER_NODE 条
        from collections import defaultdict
        assoc_by_source = defaultdict(list)
        for u, v, d in assoc_edges:
            assoc_by_source[u].append((u, v, d.get("weight", 0)))

        edges_to_remove = []
        for src, edge_list in assoc_by_source.items():
            if len(edge_list) <= _MAX_ASSOC_PER_NODE:
                continue
            # 按权重降序排序, 保留 top-N, 删除其余
            edge_list.sort(key=lambda x: x[2], reverse=True)
            for u, v, _ in edge_list[_MAX_ASSOC_PER_NODE:]:
                edges_to_remove.append((u, v))

        for u, v in edges_to_remove:
            if self._graph.has_edge(u, v):
                self._graph.remove_edge(u, v)

        removed = len(edges_to_remove)
        remaining = self._graph.number_of_edges()
        self._stats["total_edges"] = remaining

        logger.info(
            f"[MemoryGraph] ✅ ASSOCIATION 边清理完成: "
            f"删除 {removed} 条, 剩余 {remaining} 条边"
        )

        # 同时清理膨胀的 coactivation counter
        if len(self._coactivation_counter) > 5000:
            old_size = len(self._coactivation_counter)
            self._coactivation_counter.clear()
            logger.info(
                f"[MemoryGraph] 清理 coactivation counter: {old_size} → 0"
            )

        # 触发保存
        self._dirty = True
        self.save()

    def _serialize_embeddings(self) -> Dict[str, str]:
        """将 embeddings 序列化为 base64 字符串"""
        if np is None:
            return {}
        result = {}
        for nid, emb in self._embeddings.items():
            result[nid] = base64.b64encode(emb.tobytes()).decode('ascii')
        return result

    def _deserialize_embeddings(self, data: Dict[str, str]):
        """从 base64 字符串反序列化 embeddings"""
        if np is None:
            return
        for nid, b64_str in data.items():
            try:
                raw = base64.b64decode(b64_str)
                self._embeddings[nid] = np.frombuffer(raw, dtype=np.float32).copy()
            except Exception:
                pass

    # ============================================================
    # 活跃节点管理（前端高亮用）
    # ============================================================

    def set_active_nodes(self, node_ids: List[str]):
        """设置当前活跃节点 ID 集合（前端高亮用，不持久化）"""
        self._active_node_ids = set(node_ids)

    def get_active_node_ids(self) -> List[str]:
        """获取当前活跃节点 ID 列表"""
        return list(self._active_node_ids)

    def set_last_focus_context(
        self,
        dialogue_round_id: Optional[str] = None,
        focused_task_node_id: Optional[str] = None,
        active_node_ids: Optional[List[str]] = None,
        focus_path: Optional[List[str]] = None,
        focus_depth: Optional[int] = None,
    ):
        """记录当前焦点上下文（关机前 / 检查点 / 注意力导航时调用）

        Args:
            dialogue_round_id: 当前对话轮次 ID
            focused_task_node_id: 当前聚焦的任务节点 ID
            active_node_ids: 活跃节点 ID 列表
            focus_path: 从根到当前焦点的节点 ID 有序列表（思维深度索引）
            focus_depth: 当前焦点深度（0 为根）
        """
        self._last_focus_context = {
            "dialogue_round_id": dialogue_round_id,
            "focused_task_node_id": focused_task_node_id,
            "active_node_ids": active_node_ids or list(self._active_node_ids),
            "focus_path": focus_path,
            "focus_depth": focus_depth,
            "saved_at": time.time(),
        }
        self._mark_dirty()

    def get_last_focus_context(self) -> Optional[Dict]:
        """获取上次保存的焦点上下文（启动时调用）"""
        return self._last_focus_context

    # ---- 思维深度索引: 焦点导航 ----

    _NODE_TYPE_LABELS = {
        "task": "任务",
        "dialogue": "对话",
        "knowledge": "知识",
        "experience": "经验",
        "episode": "摘要",
        "file": "文件",
        "concept": "概念",
        "person": "人物",
        "document": "文档",
    }

    def get_focus_path_summary(self) -> str:
        """返回格式化的思维导航树，用于注入 L2 prompt

        输出示例::

            【思维导航】
            L1 [会话] 用户讨论部署策略 @dialogue:session_abc
             └─ L2 [对话] 帮我配置nginx @dialogue:session_abc/round_1 ← 当前焦点
            提示: navigate_attention deeper/broader/jump 调整注意力焦点

        Returns:
            格式化字符串（限制 500 字符），若无焦点返回空串
        """
        ctx = self._last_focus_context
        if not ctx or not ctx.get("focus_path"):
            return ""

        focus_path: List[str] = ctx["focus_path"]
        lines = ["【思维导航】"]
        for i, nid in enumerate(focus_path):
            node = self.get_node(nid)
            if not node:
                continue
            depth = i + 1  # L1, L2, L3, ...
            ntype = node.node_type.value
            type_label = self._NODE_TYPE_LABELS.get(ntype, ntype)
            # 根据 sub_type 精细化标签
            sub_type = node.metadata.get("sub_type", "")
            if sub_type == "session":
                type_label = "会话"
            elif sub_type == "round":
                type_label = "对话"
            elif sub_type == "agent_turn":
                type_label = "执行"
            elif sub_type == "task_root":
                type_label = "任务根"
            elif sub_type == "subtask":
                type_label = "子任务"
            elif sub_type == "sub_subtask":
                type_label = "子子任务"

            label = node.label[:40]
            indent = " " * (i * 2)
            prefix = "└─ " if i > 0 else ""
            cursor = " ← 当前焦点" if i == len(focus_path) - 1 else ""
            # 节点地址：使用 node_id 作为短地址（自带完整层级路径）
            short_addr = nid
            # 截断过长的地址（保留首尾）
            if len(short_addr) > 50:
                short_addr = short_addr[:22] + "..." + short_addr[-22:]
            lines.append(f"{indent}{prefix}L{depth} [{type_label}] {label} @{short_addr}{cursor}")

        # 简化导航指引（仅在焦点路径非空时追加）
        if len(focus_path) > 0:
            lines.append("提示: navigate_attention deeper/broader/jump 调整注意力焦点")

        result = "\n".join(lines)
        # 智能截断：优先保留首行和末行（当前焦点+提示）
        if len(result) > 500:
            head = lines[0]  # 【思维导航】
            tail = "\n".join(lines[-2:])  # 当前焦点行 + 提示行
            mid_budget = 500 - len(head) - len(tail) - 10
            mid_text = "\n".join(lines[1:-2])
            if mid_budget > 0 and mid_text:
                mid_text = mid_text[:mid_budget] + "..."
            result = head + "\n" + mid_text + "\n" + tail
        return result

    def update_focus_to_node(self, node_id: str) -> bool:
        """自动构建焦点路径并更新焦点上下文

        通过 get_ancestors() 构建从根到 node_id 的完整路径，
        同步更新焦点上下文和前端活跃节点高亮。

        Args:
            node_id: 目标焦点节点 ID

        Returns:
            True 成功更新，False 节点不存在
        """
        node = self.get_node(node_id)
        if not node:
            return False

        # 构建焦点路径: [root, ..., parent, node_id]
        ancestors = self.get_ancestors(node_id)
        focus_path = [a.node_id for a in ancestors] + [node_id]
        focus_depth = len(focus_path) - 1

        # 确定 dialogue_round_id 和 focused_task_node_id
        dialogue_round_id = None
        focused_task_node_id = None
        for nid_in_path in focus_path:
            n = self.get_node(nid_in_path)
            if n:
                if n.node_type == NodeType.DIALOGUE and n.metadata.get("sub_type") == "round":
                    dialogue_round_id = nid_in_path
                elif n.node_type == NodeType.TASK:
                    focused_task_node_id = nid_in_path

        # 更新焦点上下文
        self.set_last_focus_context(
            dialogue_round_id=dialogue_round_id,
            focused_task_node_id=focused_task_node_id,
            active_node_ids=focus_path,
            focus_path=focus_path,
            focus_depth=focus_depth,
        )

        # 同步前端高亮
        self._active_node_ids = set(focus_path)

        # P0-1: 标记记忆上下文为过期状态，触发下次调用时刷新
        self._memory_context_stale = True

        logger.debug(
            f"[MemoryGraph] 焦点已更新: depth={focus_depth}, "
            f"path={'→'.join(focus_path[-3:])}"  # 只打印最后3级
        )
        logger.info("[MemoryGraph] 记忆上下文已标记为过期，将在下次检索时刷新")
        return True

    # ============================================================
    # P0-1: 记忆上下文刷新机制
    # ============================================================

    def is_memory_context_stale(self) -> bool:
        """检查记忆上下文是否过期（焦点漂移后需要刷新）"""
        return getattr(self, '_memory_context_stale', False)

    def clear_memory_context_stale(self) -> None:
        """清除记忆上下文过期标记（系统 prompt 刷新后调用）"""
        self._memory_context_stale = False
        logger.debug("[MemoryGraph] 记忆上下文过期标记已清除")

    def get_focused_node_label(self) -> str:
        """获取当前焦点节点的标签，用于记忆上下文刷新"""
        ctx = self.get_last_focus_context()
        if not ctx:
            return ""
        
        focused_task_node_id = ctx.get("focused_task_node_id")
        if focused_task_node_id:
            node = self.get_node(focused_task_node_id)
            if node:
                return node.label
        
        # 如果没有任务节点，返回焦点路径中的最后一个节点
        focus_path = ctx.get("focus_path", [])
        if focus_path:
            last_node_id = focus_path[-1]
            node = self.get_node(last_node_id)
            if node:
                return node.label
        
        return ""

    # ============================================================
    # 前端序列化 + 变更追踪
    # ============================================================

    # ------ 分层查询辅助方法 ------

    def _get_root_node_ids(self) -> Set[str]:
        """获取无 HIERARCHY 父节点的顶层节点 ID"""
        has_hierarchy_parent: Set[str] = set()
        has_hierarchy_child: Set[str] = set()
        for src, target, data in self._graph.edges(data=True):
            if data.get("edge_type") == "hierarchy":
                has_hierarchy_parent.add(target)
                has_hierarchy_child.add(src)
        # 层级树根：有子节点但无父节点
        hierarchy_roots = has_hierarchy_child - has_hierarchy_parent
        return hierarchy_roots

    def _count_hierarchy_children(self, node_id: str) -> int:
        """计算节点的 HIERARCHY 子节点数（轻量版，不实例化 GraphNode）"""
        count = 0
        for _, _, data in self._graph.out_edges(node_id, data=True):
            if data.get("edge_type") == "hierarchy":
                count += 1
        return count

    def _serialize_node(self, node: "GraphNode", include_children_count: bool = False) -> Dict[str, Any]:
        """将单个节点序列化为前端 dict"""
        d = {
            "id": node.node_id,
            "type": node.node_type.value,
            "label": node.label,
            "activation": round(node.activation, 3),
            "metadata": node.metadata,
        }
        if include_children_count:
            d["children_count"] = self._count_hierarchy_children(node.node_id)
        return d

    def _serialize_edge(self, src: str, dst: str, data: Dict) -> Dict[str, Any]:
        """将单条边序列化为前端 dict"""
        return {
            "source": src,
            "target": dst,
            "type": data.get("edge_type", "reference"),
            "weight": round(data.get("weight", 0.5), 3),
            "protected": data.get("protected", False),
        }

    # ------ 前端序列化 ------

    # 虚拟分组节点的中文标签
    _GROUP_LABELS: Dict[str, str] = {
        "knowledge": "知识实体",
        "experience": "经验记忆",
        "episode": "情景记忆",
        "concept": "概念",
        "person": "人物",
        "file": "文件",
        "document": "文档",
        "dialogue": "对话",
        "task": "任务",
    }

    def to_frontend_dict(self, depth: Optional[int] = None) -> Dict[str, Any]:
        """序列化为前端渲染格式（不含 embeddings）

        Args:
            depth: 层级深度过滤。
                   None  — 返回全部节点和边（向后兼容）。
                   0     — 返回层级树根 + 虚拟类型分组节点。
                   N > 0 — 从根节点沿 HIERARCHY 向下 N 层。
        """
        if depth is None:
            # ---- 全量模式（向后兼容）----
            nodes = []
            for node in self._nodes.values():
                nodes.append(self._serialize_node(node, include_children_count=True))
            edges = []
            for src, dst, data in self._graph.edges(data=True):
                edges.append(self._serialize_edge(src, dst, data))
            return {
                "nodes": nodes,
                "edges": edges,
                "stats": self.stats,
                "active_node_ids": list(self._active_node_ids),
                "thought_view": self.get_thought_view_data(),
            }

        # ---- 分层模式 ----
        # 1) 扫描所有 HIERARCHY 边，确定层级关系
        has_hierarchy_parent: Set[str] = set()
        has_hierarchy_child: Set[str] = set()
        for src, target, data in self._graph.edges(data=True):
            if data.get("edge_type") == "hierarchy":
                has_hierarchy_parent.add(target)
                has_hierarchy_child.add(src)

        # 2) 层级树根 = 有子节点但无父节点
        hierarchy_roots = has_hierarchy_child - has_hierarchy_parent

        # 3) 孤立节点 = 不参与任何层级关系
        in_hierarchy = has_hierarchy_parent | has_hierarchy_child
        orphan_by_type: Dict[str, List[str]] = {}
        for nid in self._nodes:
            if nid not in in_hierarchy:
                node = self._nodes[nid]
                type_str = node.node_type.value
                orphan_by_type.setdefault(type_str, []).append(nid)

        if depth == 0:
            # 返回层级树根 + 虚拟分类容器
            nodes = []
            for nid in hierarchy_roots:
                node = self._nodes.get(nid)
                if node:
                    nodes.append(self._serialize_node(node, include_children_count=True))

            # 为每种孤立类型创建虚拟分组节点
            for type_str, nids in orphan_by_type.items():
                if nids:
                    label = self._GROUP_LABELS.get(type_str, type_str)
                    nodes.append({
                        "id": f"__group:{type_str}",
                        "type": "group",
                        "label": f"{label} ({len(nids)})",
                        "activation": 0,
                        "metadata": {"is_virtual_group": True, "group_type": type_str},
                        "children_count": len(nids),
                    })

            # 仅收集层级树根之间的边
            edges = []
            for src, dst, data in self._graph.edges(data=True):
                if src in hierarchy_roots and dst in hierarchy_roots:
                    edges.append(self._serialize_edge(src, dst, data))

            return {
                "nodes": nodes,
                "edges": edges,
                "stats": self.stats,
                "active_node_ids": list(self._active_node_ids),
                "thought_view": self.get_thought_view_data(),
            }

        # depth > 0: 从根节点沿 HIERARCHY 向下收集 depth 层
        visible_ids: Set[str] = set(hierarchy_roots)
        current_layer = set(hierarchy_roots)
        for _ in range(depth):
            next_layer: Set[str] = set()
            for nid in current_layer:
                for _, target, data in self._graph.out_edges(nid, data=True):
                    if data.get("edge_type") == "hierarchy" and target not in visible_ids:
                        next_layer.add(target)
                        visible_ids.add(target)
            current_layer = next_layer
            if not current_layer:
                break

        nodes = []
        for nid in visible_ids:
            node = self._nodes.get(nid)
            if node:
                nodes.append(self._serialize_node(node, include_children_count=True))

        edges = []
        for src, dst, data in self._graph.edges(data=True):
            if src in visible_ids and dst in visible_ids:
                edges.append(self._serialize_edge(src, dst, data))

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": self.stats,
            "active_node_ids": list(self._active_node_ids),
            "thought_view": self.get_thought_view_data(),
        }

    def get_node_children_for_frontend(self, node_id: str) -> Dict[str, Any]:
        """返回指定节点的直接子节点及相关边（用于前端展开操作）

        支持两种节点 ID：
        - 虚拟分组节点 ``__group:<type>`` → 返回该类型的所有孤立节点
        - 普通节点 → 返回 HIERARCHY 直接子节点
        """
        # ---- 虚拟分组节点展开 ----
        if node_id.startswith("__group:"):
            type_str = node_id[len("__group:"):]
            # 收集所有参与层级关系的节点 ID
            in_hierarchy: Set[str] = set()
            for src, target, data in self._graph.edges(data=True):
                if data.get("edge_type") == "hierarchy":
                    in_hierarchy.add(src)
                    in_hierarchy.add(target)
            # 该类型的孤立节点
            group_child_ids: Set[str] = set()
            nodes = []
            for nid, node in self._nodes.items():
                if node.node_type.value == type_str and nid not in in_hierarchy:
                    group_child_ids.add(nid)
                    nodes.append(self._serialize_node(node, include_children_count=False))
            # 孤立节点之间的边
            edges = []
            for src, dst, data in self._graph.edges(data=True):
                if src in group_child_ids and dst in group_child_ids:
                    edges.append(self._serialize_edge(src, dst, data))
            return {
                "parent_id": node_id,
                "nodes": nodes,
                "edges": edges,
            }

        # ---- 普通层级节点展开 ----
        children = self.get_children(node_id)
        child_ids = {c.node_id for c in children}

        nodes = [self._serialize_node(c, include_children_count=True) for c in children]

        edges = []
        for src, dst, data in self._graph.edges(data=True):
            # 父→子 或 子↔子 的边
            if (src == node_id and dst in child_ids) or \
               (src in child_ids and dst in child_ids):
                edges.append(self._serialize_edge(src, dst, data))

        return {
            "parent_id": node_id,
            "nodes": nodes,
            "edges": edges,
        }

    def flush_changes(self) -> Dict[str, Any]:
        """返回并清空待处理变更（增量 delta）"""
        changes = list(self._pending_changes)
        self._pending_changes.clear()
        return {
            "type": "delta",
            "changes": changes,
            "active_node_ids": list(self._active_node_ids),
            "thought_view": self.get_thought_view_data(),
        }

    def get_thought_view_data(self) -> Dict[str, Any]:
        """为思维可视化浮动窗口提供数据：活跃节点 + 1跳邻居 + 连接边"""
        active_ids = set(self._active_node_ids)
        if not active_ids:
            return {"nodes": [], "edges": [], "center_ids": []}

        # 收集 1 跳邻居
        visible_ids = set(active_ids)
        for nid in active_ids:
            for neighbor in self.get_neighbors(nid, max_depth=1):
                visible_ids.add(neighbor.node_id)

        # 性能保护：上限 50 个节点
        if len(visible_ids) > 50:
            neighbor_only = visible_ids - active_ids
            sorted_neighbors = sorted(
                neighbor_only,
                key=lambda nid: self._nodes[nid].activation if nid in self._nodes else 0,
                reverse=True,
            )
            visible_ids = active_ids | set(sorted_neighbors[:50 - len(active_ids)])

        # 序列化节点
        nodes = []
        for nid in visible_ids:
            node = self._nodes.get(nid)
            if node:
                d = self._serialize_node(node)
                d["is_center"] = nid in active_ids
                nodes.append(d)

        # 序列化边（两端都在 visible_ids 中的边）
        edges = []
        for src, dst, data in self._graph.edges(data=True):
            if src in visible_ids and dst in visible_ids:
                edges.append(self._serialize_edge(src, dst, data))

        return {
            "nodes": nodes,
            "edges": edges,
            "center_ids": list(active_ids),
        }

    # ============================================================
    # 统计与调试
    # ============================================================

    @property
    def stats(self) -> Dict[str, Any]:
        """图谱统计信息"""
        type_counts: Dict[str, int] = {}
        for node in self._nodes.values():
            t = node.node_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        edge_type_counts: Dict[str, int] = {}
        for _, _, data in self._graph.edges(data=True):
            et = data.get("edge_type", "unknown")
            edge_type_counts[et] = edge_type_counts.get(et, 0) + 1

        return {
            "total_nodes": len(self._nodes),
            "total_edges": self._graph.number_of_edges(),
            "node_types": type_counts,
            "edge_types": edge_type_counts,
            "embeddings_count": len(self._embeddings),
            "coactivation_pairs": len(self._coactivation_counter),
            **self._stats,
        }

    def __repr__(self) -> str:
        return (
            f"MemoryGraph(nodes={len(self._nodes)}, "
            f"edges={self._graph.number_of_edges()})"
        )


# ============================================================
# 便捷访问函数
# ============================================================

def get_memory_graph(persist_path: str = None) -> MemoryGraph:
    """获取 MemoryGraph 单例（P3-23: 优先从配置读取路径）"""
    if persist_path is None:
        try:
            import yaml
            import os
            cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "config", "zulong_config.yaml")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                persist_path = (cfg.get("memory") or {}).get("persist_path") or "./data/memory_graph"
        except Exception:
            persist_path = "./data/memory_graph"
    return MemoryGraph(persist_path=persist_path)
