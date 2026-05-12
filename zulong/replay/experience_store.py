"""
AT-16: 经验向量库搭建
建立长期记忆库，存储"场景特征 + 修正逻辑"
关键指标: 支持语义检索，非结构化数据存储
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import logging
import json
import os
import asyncio
from datetime import datetime

from .clock_synchronizer import get_unified_timestamp
from .patch_compiler import SystemPatch, PatchStatus

logger = logging.getLogger(__name__)


class StoreBackend(Enum):
    MEMORY = "memory"
    QDRANT = "qdrant"
    FILE = "file"


@dataclass
class ExperienceEntry:
    """经验条目"""
    entry_id: str
    patch: SystemPatch
    embedding: Optional[List[float]] = None
    keywords: List[str] = field(default_factory=list)
    last_accessed: float = 0.0
    access_count: int = 0
    created_at: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "patch": self.patch.to_dict(),
            "embedding": self.embedding,
            "keywords": self.keywords,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperienceEntry':
        return cls(
            entry_id=data["entry_id"],
            patch=SystemPatch.from_dict(data["patch"]),
            embedding=data.get("embedding"),
            keywords=data.get("keywords", []),
            last_accessed=data.get("last_accessed", 0.0),
            access_count=data.get("access_count", 0),
            created_at=data.get("created_at", 0.0)
        )


class ExperienceStore:
    """
    经验向量库 (AT-16)
    
    长期记忆库，存储"场景特征 + 修正逻辑"
    支持语义检索，非结构化数据存储
    """
    
    def __init__(
        self,
        backend: StoreBackend = StoreBackend.MEMORY,
        storage_path: str = "experience_store",
        embedding_dim: int = 768
    ):
        """
        初始化经验向量库
        
        Args:
            backend: 存储后端
            storage_path: 存储路径
            embedding_dim: 嵌入维度
        """
        self.backend = backend
        self.storage_path = storage_path
        self.embedding_dim = embedding_dim
        
        self._entries: Dict[str, ExperienceEntry] = {}
        self._keyword_index: Dict[str, List[str]] = {}
        self._entry_counter = 0
        
        self._embedding_model = None
        self._qdrant_client = None
        
        os.makedirs(storage_path, exist_ok=True)
        
        self._stats = {
            "entries_stored": 0,
            "entries_retrieved": 0,
            "searches_performed": 0,
            "cache_hits": 0
        }
        
        if backend == StoreBackend.FILE:
            self._load_from_disk()
        
        logger.info(f"[ExperienceStore] 初始化完成，后端: {backend.value}")
    
    async def initialize_embedding_model(self, model_name: str = "text-embedding-ada-002"):
        """
        初始化嵌入模型
        
        Args:
            model_name: 模型名称
        """
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("[ExperienceStore] 嵌入模型加载完成")
        except Exception as e:
            logger.warning(f"[ExperienceStore] 嵌入模型加载失败: {e}，将使用关键词匹配")
            self._embedding_model = None
    
    def _generate_entry_id(self) -> str:
        """生成条目ID"""
        self._entry_counter += 1
        return f"EXP_{get_unified_timestamp():.0f}_{self._entry_counter}"
    
    def _compute_embedding(self, text: str) -> Optional[List[float]]:
        """计算文本嵌入"""
        if self._embedding_model is None:
            return None
        try:
            embedding = self._embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"[ExperienceStore] 嵌入计算失败: {e}")
            return None
    
    def _extract_keywords(self, patch: SystemPatch) -> List[str]:
        """提取关键词"""
        keywords = []
        keywords.extend(patch.scene_features)
        keywords.append(patch.fault_layer)
        keywords.append(patch.adjustment_type)
        
        if patch.delta_t > 0:
            keywords.append("execution_delayed")
        elif patch.delta_t < 0:
            keywords.append("prediction_ahead")
        
        return list(set(keywords))
    
    async def store(
        self,
        patch: SystemPatch,
        additional_keywords: Optional[List[str]] = None
    ) -> str:
        """
        存储经验
        
        Args:
            patch: 系统补丁
            additional_keywords: 附加关键词
        
        Returns:
            str: 条目ID
        """
        entry_id = self._generate_entry_id()
        keywords = self._extract_keywords(patch)
        if additional_keywords:
            keywords.extend(additional_keywords)
        
        text_for_embedding = f"{patch.condition} {patch.adjustment} {' '.join(patch.scene_features)}"
        embedding = self._compute_embedding(text_for_embedding)
        
        entry = ExperienceEntry(
            entry_id=entry_id,
            patch=patch,
            embedding=embedding,
            keywords=keywords,
            last_accessed=get_unified_timestamp(),
            access_count=0,
            created_at=get_unified_timestamp()
        )
        
        self._entries[entry_id] = entry
        
        for keyword in keywords:
            if keyword not in self._keyword_index:
                self._keyword_index[keyword] = []
            self._keyword_index[keyword].append(entry_id)
        
        self._stats["entries_stored"] += 1
        
        if self.backend == StoreBackend.FILE:
            await self._save_to_disk()
        
        logger.info(f"[ExperienceStore] 经验存储: {entry_id}")
        return entry_id
    
    async def search(
        self,
        query: str,
        scene_tags: Optional[List[str]] = None,
        top_k: int = 5,
        min_confidence: float = 0.5
    ) -> List[Tuple[ExperienceEntry, float]]:
        """
        搜索经验
        
        Args:
            query: 查询文本
            scene_tags: 场景标签
            top_k: 返回数量
            min_confidence: 最小置信度
        
        Returns:
            List[Tuple[ExperienceEntry, float]]: (条目, 相似度) 列表
        """
        self._stats["searches_performed"] += 1
        
        candidates = set()
        
        if scene_tags:
            for tag in scene_tags:
                if tag in self._keyword_index:
                    candidates.update(self._keyword_index[tag])
        
        if not candidates:
            candidates = set(self._entries.keys())
        
        query_embedding = self._compute_embedding(query)
        
        results = []
        for entry_id in candidates:
            entry = self._entries.get(entry_id)
            if not entry:
                continue
            if entry.patch.status != PatchStatus.ACTIVE:
                continue
            if entry.patch.confidence < min_confidence:
                continue
            
            if query_embedding and entry.embedding:
                similarity = self._cosine_similarity(query_embedding, entry.embedding)
            else:
                similarity = self._keyword_match_score(query, entry.keywords)
            
            if similarity > 0.3:
                results.append((entry, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:top_k]
        
        for entry, _ in top_results:
            entry.last_accessed = get_unified_timestamp()
            entry.access_count += 1
            self._stats["entries_retrieved"] += 1
        
        return top_results
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        import numpy as np
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))
    
    def _keyword_match_score(self, query: str, keywords: List[str]) -> float:
        """关键词匹配分数"""
        query_lower = query.lower()
        matches = sum(1 for kw in keywords if kw.lower() in query_lower)
        return matches / max(len(keywords), 1)
    
    async def get_applicable_patches(
        self,
        scene_tags: List[str],
        top_k: int = 3
    ) -> List[SystemPatch]:
        """
        获取适用的补丁
        
        Args:
            scene_tags: 场景标签
            top_k: 返回数量
        
        Returns:
            List[SystemPatch]: 补丁列表
        """
        query = " ".join(scene_tags)
        results = await self.search(query, scene_tags=scene_tags, top_k=top_k)
        return [entry.patch for entry, _ in results]
    
    async def record_patch_result(self, patch_id: str, success: bool):
        """记录补丁应用结果"""
        for entry in self._entries.values():
            if entry.patch.patch_id == patch_id:
                entry.patch.record_application(success)
                if self.backend == StoreBackend.FILE:
                    await self._save_to_disk()
                break
    
    async def _save_to_disk(self):
        """保存到磁盘"""
        file_path = os.path.join(self.storage_path, "experience_store.json")
        data = {
            "entries": {eid: e.to_dict() for eid, e in self._entries.items()},
            "keyword_index": self._keyword_index,
            "stats": self._stats
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_from_disk(self):
        """从磁盘加载"""
        file_path = os.path.join(self.storage_path, "experience_store.json")
        if not os.path.exists(file_path):
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._entries = {
                eid: ExperienceEntry.from_dict(e) 
                for eid, e in data.get("entries", {}).items()
            }
            self._keyword_index = data.get("keyword_index", {})
            self._stats = data.get("stats", self._stats)
            
            logger.info(f"[ExperienceStore] 从磁盘加载 {len(self._entries)} 条经验")
        except Exception as e:
            logger.error(f"[ExperienceStore] 加载失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_entries": len(self._entries),
            "active_entries": sum(1 for e in self._entries.values() if e.patch.status == PatchStatus.ACTIVE),
            "stats": self._stats.copy(),
            "keyword_count": len(self._keyword_index)
        }
    
    async def cleanup_expired(self):
        """清理过期条目"""
        current_time = get_unified_timestamp()
        expired_ids = [
            eid for eid, entry in self._entries.items()
            if entry.patch.expires_at and entry.patch.expires_at < current_time
        ]
        
        for eid in expired_ids:
            entry = self._entries.pop(eid)
            for keyword in entry.keywords:
                if keyword in self._keyword_index:
                    self._keyword_index[keyword] = [
                        x for x in self._keyword_index[keyword] if x != eid
                    ]
        
        if expired_ids:
            logger.info(f"[ExperienceStore] 清理 {len(expired_ids)} 条过期经验")
            if self.backend == StoreBackend.FILE:
                await self._save_to_disk()


_global_experience_store: Optional[ExperienceStore] = None


def get_experience_store() -> ExperienceStore:
    """获取全局经验向量库"""
    global _global_experience_store
    if _global_experience_store is None:
        _global_experience_store = ExperienceStore()
    return _global_experience_store
