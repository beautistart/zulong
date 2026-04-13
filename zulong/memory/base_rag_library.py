# File: zulong/memory/base_rag_library.py
# RAG 库基础结构 - 定义通用接口和 FAISS 实现

import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class RAGDocument:
    """RAG 文档数据结构
    
    TSD v1.7 对应规则:
    - 信息分类打标
    - 重要性：必须学习/待定/不需要
    - 记忆性：必须记住/待定/不用记住
    - 领域：导航/操作/视觉/对话/...
    """
    content: str  # 文档内容
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    # 标签维度
    importance: str = "pending"  # 重要性：must_learn, pending, not_needed
    memorability: str = "pending"  # 记忆性：must_remember, pending, forget
    domain: str = "general"  # 领域：navigation, manipulation, vision, dialog, general
    
    # 时间戳
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # 向量（计算后存储）
    embedding: Optional[np.ndarray] = None
    embedding_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "importance": self.importance,
            "memorability": self.memorability,
            "domain": self.domain,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGDocument":
        """从字典加载"""
        doc = cls(
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            importance=data.get("importance", "pending"),
            memorability=data.get("memorability", "pending"),
            domain=data.get("domain", "general"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time())
        )
        return doc


class BaseVectorStore(ABC):
    """向量存储基类 - 通用接口
    
    TSD v1.7 对应规则:
    - 支持向量库切换（FAISS/Chroma/Qdrant）
    - 统一 CRUD 操作接口
    """
    
    @abstractmethod
    def __init__(self, dimension: int, **kwargs):
        """初始化向量存储
        
        Args:
            dimension: 向量维度
        """
        self.dimension = dimension
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, metadata: Optional[List[Dict]] = None) -> List[str]:
        """添加向量
        
        Args:
            vectors: 向量数组 (n, dimension)
            metadata: 元数据列表
            
        Returns:
            List[str]: 添加的向量 ID 列表
        """
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5, 
               filter_func: Optional[callable] = None) -> Tuple[List[int], List[float]]:
        """搜索相似向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回数量
            filter_func: 过滤函数（可选）
            
        Returns:
            Tuple[List[int], List[float]]: (索引列表，距离列表)
        """
        pass
    
    @abstractmethod
    def delete_vectors(self, ids: List[str]) -> bool:
        """删除向量
        
        Args:
            ids: 要删除的向量 ID 列表
            
        Returns:
            bool: 是否成功
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> bool:
        """持久化到磁盘
        
        Args:
            path: 保存路径
            
        Returns:
            bool: 是否成功
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> bool:
        """从磁盘加载
        
        Args:
            path: 加载路径
            
        Returns:
            bool: 是否成功
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        pass


class FAISSVectorStore(BaseVectorStore):
    """FAISS 向量存储实现
    
    特点:
    - 高性能（Facebook 开源）
    - 支持多种索引类型
    - 轻量级（无需额外服务）
    """
    
    def __init__(self, dimension: int, index_type: str = "Flat", **kwargs):
        """初始化 FAISS 向量存储
        
        Args:
            dimension: 向量维度
            index_type: 索引类型（Flat, IVF, HNSW 等）
        """
        super().__init__(dimension, **kwargs)
        
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            logger.error("FAISS not installed. Install with: pip install faiss-cpu")
            raise
        
        self.index_type = index_type
        
        # 创建索引
        if index_type == "Flat":
            # 扁平索引（精确搜索，适合小规模数据）
            self.index = self.faiss.IndexFlatL2(dimension)
        elif index_type.startswith("IVF"):
            # IVF 索引（近似搜索，适合大规模数据）
            nlist = 100  # 聚类中心数
            quantizer = self.faiss.IndexFlatL2(dimension)
            self.index = self.faiss.IndexIVFFlat(quantizer, dimension, nlist)
        else:
            # 默认使用 Flat
            self.index = self.faiss.IndexFlatL2(dimension)
        
        # ID 映射（外部 ID -> 内部索引）
        self.id_map: Dict[str, int] = {}  # id -> index
        self.reverse_id_map: Dict[int, str] = {}  # index -> id
        self.metadata_store: Dict[str, Dict] = {}  # id -> metadata
        
        # 统计信息
        self.total_adds = 0
        self.total_searches = 0
        self.total_deletes = 0
        
        logger.info(f"[FAISSVectorStore] Initialized: dim={dimension}, type={index_type}")
    
    def add_vectors(self, vectors: np.ndarray, metadata: Optional[List[Dict]] = None) -> List[str]:
        """添加向量"""
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)
        
        # 维度校验：若不一致则截断或填充到目标维度
        if vectors.shape[1] != self.dimension:
            logger.warning(
                f"[FAISSVectorStore] 向量维度不匹配：输入 {vectors.shape[1]}，期望 {self.dimension}，自动调整"
            )
            if vectors.shape[1] > self.dimension:
                vectors = vectors[:, :self.dimension]
            else:
                pad = np.zeros((vectors.shape[0], self.dimension - vectors.shape[1]), dtype=np.float32)
                vectors = np.hstack([vectors, pad])
        
        n = vectors.shape[0]
        start_idx = self.index.ntotal
        
        # 添加到索引
        self.index.add(vectors.astype(np.float32))
        
        # 生成 ID 并更新映射
        ids = []
        for i in range(n):
            idx = start_idx + i
            doc_id = f"doc_{idx}_{int(time.time()*1000)}"
            self.id_map[doc_id] = idx
            self.reverse_id_map[idx] = doc_id
            
            # 存储元数据
            if metadata and i < len(metadata):
                self.metadata_store[doc_id] = metadata[i]
            else:
                self.metadata_store[doc_id] = {}
            
            ids.append(doc_id)
        
        self.total_adds += n
        logger.debug(f"[FAISSVectorStore] Added {n} vectors")
        
        return ids
    
    def add_vectors_with_ids(self, vectors: np.ndarray, metadata: Optional[List[Dict]] = None,
                            vector_ids: Optional[List[str]] = None) -> List[str]:
        """添加向量（使用自定义 ID）
        
        Args:
            vectors: 向量数组 (n, dimension)
            metadata: 元数据列表
            vector_ids: 自定义 ID 列表
            
        Returns:
            List[str]: 使用的 ID 列表
        """
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)
        
        # 维度校验：若不一致则截断或填充到目标维度
        if vectors.shape[1] != self.dimension:
            logger.warning(
                f"[FAISSVectorStore] 向量维度不匹配：输入 {vectors.shape[1]}，期望 {self.dimension}，自动调整"
            )
            if vectors.shape[1] > self.dimension:
                vectors = vectors[:, :self.dimension]
            else:
                pad = np.zeros((vectors.shape[0], self.dimension - vectors.shape[1]), dtype=np.float32)
                vectors = np.hstack([vectors, pad])
        
        n = vectors.shape[0]
        start_idx = self.index.ntotal
        
        # 添加到索引
        self.index.add(vectors.astype(np.float32))
        
        # 使用自定义 ID 或生成默认 ID
        ids = []
        for i in range(n):
            idx = start_idx + i
            doc_id = vector_ids[i] if vector_ids and i < len(vector_ids) else f"doc_{idx}_{int(time.time()*1000)}"
            self.id_map[doc_id] = idx
            self.reverse_id_map[idx] = doc_id
            
            # 存储元数据
            if metadata and i < len(metadata):
                self.metadata_store[doc_id] = metadata[i]
            else:
                self.metadata_store[doc_id] = {}
            
            ids.append(doc_id)
        
        self.total_adds += n
        logger.debug(f"[FAISSVectorStore] Added {n} vectors with custom IDs")
        
        return ids
    
    def search(self, query_vector: np.ndarray, top_k: int = 5,
               filter_func: Optional[callable] = None) -> Tuple[List[int], List[float]]:
        """搜索相似向量（自动过滤已删除向量）"""
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # 维度校验：若不一致则截断或填充到目标维度
        if query_vector.shape[1] != self.dimension:
            logger.warning(
                f"[FAISSVectorStore] 查询向量维度不匹配：输入 {query_vector.shape[1]}，期望 {self.dimension}，自动调整"
            )
            if query_vector.shape[1] > self.dimension:
                query_vector = query_vector[:, :self.dimension]
            else:
                pad = np.zeros((query_vector.shape[0], self.dimension - query_vector.shape[1]), dtype=np.float32)
                query_vector = np.hstack([query_vector, pad])
        
        # 请求更多结果以弥补过滤掉已删除向量的损失
        search_k = min(top_k * 2, self.index.ntotal) if self.index.ntotal > 0 else top_k
        
        # FAISS 搜索
        distances, indices = self.index.search(query_vector.astype(np.float32), search_k)
        
        # 转换为列表
        distances = distances[0].tolist()
        indices = indices[0].tolist()
        
        # 过滤已删除向量和无效索引
        filtered_indices = []
        filtered_distances = []
        for idx, dist in zip(indices, distances):
            if idx == -1:  # 无效索引
                continue
            doc_id = self.reverse_id_map.get(idx)
            if not doc_id:  # 已从映射中移除（已删除）
                continue
            meta = self.metadata_store.get(doc_id, {})
            if meta.get("deleted", False):  # 标记为已删除
                continue
            # 应用自定义过滤函数
            if filter_func and not filter_func(meta):
                continue
            filtered_indices.append(idx)
            filtered_distances.append(dist)
            if len(filtered_indices) >= top_k:
                break
        
        self.total_searches += 1
        logger.debug(f"[FAISSVectorStore] Searched top_k={top_k}, results={len(filtered_indices)}")
        
        return filtered_indices, filtered_distances
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """删除向量（标记删除 + 触发条件重建）
        
        FAISS 不支持直接删除单个向量，采用标记删除 + 定期重建策略：
        1. 在元数据中标记为已删除
        2. 从映射中移除
        3. 当删除数量超过阈值时自动触发索引重建
        """
        for doc_id in ids:
            if doc_id in self.metadata_store:
                self.metadata_store[doc_id]["deleted"] = True
            
            if doc_id in self.id_map:
                idx = self.id_map[doc_id]
                del self.reverse_id_map[idx]
                del self.id_map[doc_id]
        
        self.total_deletes += len(ids)
        
        # 当已删除向量占比超过 20% 时自动触发重建
        if self.index.ntotal > 0:
            active_count = len(self.id_map)
            delete_ratio = 1.0 - (active_count / self.index.ntotal)
            if delete_ratio > 0.2:
                logger.info(
                    f"[FAISSVectorStore] 已删除比例 {delete_ratio:.1%} 超过 20%，触发索引重建"
                )
                self.rebuild_index()
        
        return True
    
    def rebuild_index(self) -> bool:
        """重建 FAISS 索引，物理移除已删除的向量
        
        流程：
        1. 收集所有活跃（未删除）的向量及其元数据
        2. 创建新的 FAISS 索引
        3. 批量添加活跃向量
        4. 重建 ID 映射
        
        Returns:
            bool: 重建是否成功
        """
        try:
            old_total = self.index.ntotal
            
            # 1. 收集活跃向量数据
            active_vectors = []
            active_ids = []
            active_metadata = {}
            
            for doc_id, idx in list(self.id_map.items()):
                # 跳过已标记删除的
                meta = self.metadata_store.get(doc_id, {})
                if meta.get("deleted", False):
                    continue
                
                # 从 FAISS 索引中提取向量
                if idx < self.index.ntotal:
                    vector = self.index.reconstruct(idx)
                    active_vectors.append(vector)
                    active_ids.append(doc_id)
                    active_metadata[doc_id] = meta
            
            # 2. 创建新索引
            if self.index_type == "Flat":
                new_index = self.faiss.IndexFlatL2(self.dimension)
            elif self.index_type.startswith("IVF"):
                nlist = 100
                quantizer = self.faiss.IndexFlatL2(self.dimension)
                new_index = self.faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            else:
                new_index = self.faiss.IndexFlatL2(self.dimension)
            
            # 3. 批量添加活跃向量
            new_id_map = {}
            new_reverse_id_map = {}
            
            if active_vectors:
                vectors_array = np.array(active_vectors, dtype=np.float32)
                new_index.add(vectors_array)
                
                for new_idx, doc_id in enumerate(active_ids):
                    new_id_map[doc_id] = new_idx
                    new_reverse_id_map[new_idx] = doc_id
            
            # 4. 替换旧索引和映射
            self.index = new_index
            self.id_map = new_id_map
            self.reverse_id_map = new_reverse_id_map
            self.metadata_store = active_metadata
            
            new_total = self.index.ntotal
            logger.info(
                f"[FAISSVectorStore] 索引重建完成：{old_total} → {new_total} 向量 "
                f"(清理了 {old_total - new_total} 个已删除向量)"
            )
            return True
            
        except Exception as e:
            logger.error(f"[FAISSVectorStore] 索引重建失败：{e}")
            return False
    
    def save(self, path: str) -> bool:
        """持久化到磁盘"""
        try:
            # 保存 FAISS 索引
            index_path = f"{path}.index"
            self.faiss.write_index(self.index, index_path)
            
            # 保存 ID 映射
            map_path = f"{path}.maps.json"
            with open(map_path, "w") as f:
                json.dump({
                    "id_map": self.id_map,
                    "reverse_id_map": {str(k): v for k, v in self.reverse_id_map.items()},
                    "metadata_store": self.metadata_store
                }, f)
            
            logger.info(f"[FAISSVectorStore] Saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"[FAISSVectorStore] Save failed: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """从磁盘加载"""
        try:
            # 加载 FAISS 索引
            index_path = f"{path}.index"
            if os.path.exists(index_path):
                self.index = self.faiss.read_index(index_path)
            else:
                logger.warning(f"[FAISSVectorStore] Index file not found: {index_path}")
                return False
            
            # 加载 ID 映射
            map_path = f"{path}.maps.json"
            if os.path.exists(map_path):
                with open(map_path, "r") as f:
                    data = json.load(f)
                    self.id_map = data["id_map"]
                    self.reverse_id_map = {int(k): v for k, v in data["reverse_id_map"].items()}
                    self.metadata_store = data["metadata_store"]
            else:
                logger.warning(f"[FAISSVectorStore] Map file not found: {map_path}")
            
            logger.info(f"[FAISSVectorStore] Loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"[FAISSVectorStore] Load failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "total_vectors": self.index.ntotal,
            "total_adds": self.total_adds,
            "total_searches": self.total_searches,
            "total_deletes": self.total_deletes,
            "memory_usage_mb": self.index.ntotal * self.dimension * 4 / 1024 / 1024  # 估算
        }


class BaseRAGLibrary(ABC):
    """RAG 库基类
    
    TSD v1.7 对应规则:
    - 2.2.5 基础设施层：RAG 记忆系统
    - 统一接口，支持不同类型 RAG 库
    
    功能:
    - 文档存储与检索
    - 向量化（嵌入）
    - 分类打标
    - 持久化
    """
    
    def __init__(self, name: str, dimension: int = 768, 
                 vector_store_type: str = "faiss", **kwargs):
        """初始化 RAG 库
        
        Args:
            name: 库名称
            dimension: 向量维度
            vector_store_type: 向量存储类型（faiss, chroma, qdrant）
        """
        self.name = name
        self.dimension = dimension
        
        # 创建向量存储
        if vector_store_type == "faiss":
            self.vector_store = FAISSVectorStore(dimension, **kwargs)
        else:
            # 默认使用 FAISS
            self.vector_store = FAISSVectorStore(dimension, **kwargs)
        
        # 文档存储
        self.documents: Dict[str, RAGDocument] = {}
        
        # 统计信息
        self.total_adds = 0
        self.total_searches = 0
        
        logger.info(f"[BaseRAGLibrary] Initialized: name={name}, dim={dimension}")
    
    @abstractmethod
    def add_document(self, document: RAGDocument) -> str:
        """添加文档"""
        pass
    
    @abstractmethod
    def search_documents(self, query: str, top_k: int = 5, 
                        filters: Optional[Dict] = None) -> List[RAGDocument]:
        """搜索文档"""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        pass
    
    def save(self, path: str) -> bool:
        """持久化"""
        try:
            # 保存向量索引
            self.vector_store.save(f"{path}.vector")
            
            # 保存文档元数据
            doc_path = f"{path}.docs.json"
            with open(doc_path, "w") as f:
                json.dump({
                    doc_id: doc.to_dict() 
                    for doc_id, doc in self.documents.items()
                }, f)
            
            logger.info(f"[BaseRAGLibrary] Saved to {path}")
            return True
        except Exception as e:
            logger.error(f"[BaseRAGLibrary] Save failed: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """加载"""
        try:
            # 加载向量索引
            self.vector_store.load(f"{path}.vector")
            
            # 加载文档元数据
            doc_path = f"{path}.docs.json"
            if os.path.exists(doc_path):
                with open(doc_path, "r") as f:
                    data = json.load(f)
                    self.documents = {
                        doc_id: RAGDocument.from_dict(doc_data)
                        for doc_id, doc_data in data.items()
                    }
            
            logger.info(f"[BaseRAGLibrary] Loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"[BaseRAGLibrary] Load failed: {e}")
            return False
