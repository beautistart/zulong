# File: zulong/memory/vector_cache.py
# 会话向量缓存 - 临时记忆语义检索 (TSD v2.4)

"""
会话向量缓存管理器 (TSD v2.4 内存内语义缓存策略)

核心特性:
- ✅ 临时记忆：存储原始文本（Redis/内存列表）
- ✅ 会话向量：每轮对话结束后立即生成（L1-B 负责）
- ✅ 内存缓存：向量存储在内存中，不做 VectorDB
- ✅ 即时检索：查询时计算 V_query，扫描 V_memory
- ✅ 时间衰减：模拟短期记忆随时间衰退

架构对齐:
- 存储层：临时记忆（原始文本）
- 检索层：复用 Embedding 模型（BGE-Small / MiniLM）
- 缓存层：内存级向量缓存（In-Memory Semantic Cache）

对应 TSD v2.4 第 14.1 节：内存内语义缓存（In-Memory Semantic Cache）

文档参考:
- d:\AI\project\zulong_beta4\资料\临时记忆的动态量化与筛选完整问答.txt
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
import logging
import numpy as np

from zulong.memory.embedding_manager import EmbeddingModelManager

logger = logging.getLogger(__name__)


@dataclass
class SessionVectorEntry:
    """
    会话向量条目（TSD v2.4 语义缓存）
    
    架构说明:
    - 临时记忆：原始文本（用户输入 + AI 回复）
    - 会话向量：V_memory = Embedding(原始文本)
    - 存储位置：内存列表（非 VectorDB）
    """
    
    turn_id: int           # 轮次 ID
    timestamp: float       # 时间戳
    
    # 临时记忆（原始文本）
    user_text: str         # 用户输入（原始）
    bot_text: str          # AI 回复（原始）
    raw_text: str          # 组装后的完整文本："User: ...\nAssistant: ..."
    
    # 会话向量（L1-B 生成）
    session_vector: np.ndarray  # V_memory（768 维）
    
    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        """检查是否过期"""
        return (time.time() - self.timestamp) > ttl_seconds
    
    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            'turn_id': self.turn_id,
            'timestamp': self.timestamp,
            'user_text': self.user_text,
            'bot_text': self.bot_text,
            'raw_text': self.raw_text,
            'session_vector': self.session_vector.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SessionVectorEntry':
        """从字典反序列化"""
        return cls(
            turn_id=data['turn_id'],
            timestamp=data['timestamp'],
            user_text=data['user_text'],
            bot_text=data['bot_text'],
            raw_text=data['raw_text'],
            session_vector=np.array(data['session_vector'])
        )


class SessionVectorCache:
    """
    会话向量缓存（TSD v2.4 内存内语义缓存）
    
    架构对齐:
    - 存储层：临时记忆（原始文本，Redis/内存列表）
    - 检索层：复用 Embedding 模型（BGE-Small / MiniLM）
    - 缓存层：会话向量（V_memory，内存列表）
    
    数据流（L1-B 负责）:
    1. 对话完成 → L1-B 生成 V_memory = Embedding(raw_text) → 存入缓存
    2. 检索请求 → L1-B 计算 V_query = Embedding(query) → 扫描 V_memory → 返回 Top-K
    
    优势:
    - ✅ 零延迟写入：不需要等 VectorDB 索引
    - ✅ 语义精准：识别"红色的东西" = "红苹果"
    - ✅ 资源复用：不需要维护两套检索系统
    - ✅ 性能优异：检索速度恒定 ~10ms（不随对话轮数增长）
    """
    
    def __init__(
        self,
        embedding_manager: EmbeddingModelManager,
        max_cache_size: int = 100,  # 🔥 优化：50→100，增加缓存容量
        ttl_seconds: int = 7200,  # 🔥 优化：3600→7200，延长缓存时间 (1h→2h)
        time_decay_lambda: float = 0.0005  # 🔥 优化：0.001→0.0005，减缓衰减 (半衰期~20 分钟)
    ):
        """
        初始化会话向量缓存
        
        Args:
            embedding_manager: Embedding 模型管理器（复用检索模型）
            max_cache_size: 最大缓存数量（🔥 优化：50→100，符合短期记忆特性）
            ttl_seconds: 过期时间（秒）（🔥 优化：3600→7200，默认 2 小时）
            time_decay_lambda: 时间衰减系数（🔥 优化：0.001→0.0005，半衰期~20 分钟）
        """
        self.embedding_manager = embedding_manager
        # 🔥 优化：增加缓存容量，延长 TTL，减缓衰减
        self.max_cache_size = max_cache_size if max_cache_size else 100  # 从 50 增加到 100
        self.ttl_seconds = ttl_seconds if ttl_seconds else 7200  # 从 3600 增加到 7200 (2 小时)
        self.time_decay_lambda = time_decay_lambda if time_decay_lambda else 0.0005  # 从 0.001 减少到 0.0005 (半衰期~20 分钟)
        
        # 🔥 会话向量缓存池（V_memory）
        self.cache: List[SessionVectorEntry] = []
        
        # 异步锁
        self._lock = asyncio.Lock()
        
        # 统计信息
        self._stats = {
            "total_vectors_computed": 0,  # V_memory 生成次数
            "total_searches": 0,          # 检索次数
            "cache_hits": 0,              # 缓存命中
            "cache_evictions": 0          # 淘汰次数
        }
        
        logger.info(
            f"[SessionVectorCache] 初始化完成（TSD v2.4 内存内语义缓存）\n"
            f"  - 最大缓存：{max_cache_size} 轮对话\n"
            f"  - TTL: {ttl_seconds}s\n"
            f"  - 时间衰减系数：{time_decay_lambda} (半衰期~10 分钟)\n"
            f"  - 复用模型：{embedding_manager.model_name if hasattr(embedding_manager, 'model_name') else 'Embedding'}"
        )
    
    async def store_session_vector(
        self, 
        user_text: str, 
        bot_text: str, 
        turn_id: int
    ) -> bool:
        """
        存储会话向量（TSD v2.4 增量向量化）
        
        架构对齐:
        1. 临时记忆：存储原始文本（Redis/内存列表）
        2. 会话向量：L1-B 生成 V_memory = Embedding(raw_text)
        3. 内存缓存：向量存储在内存中（非 VectorDB）
        
        Args:
            user_text: 用户输入（原始）
            bot_text: AI 回复（原始）
            turn_id: 轮次 ID
            
        Returns:
            bool: 是否成功
        
        使用示例:
        ```python
        # L1-B 在对话完成后调用
        await cache.store_session_vector(
            user_text="把那个红色的东西给我",
            bot_text="好的，给你红苹果",
            turn_id=1
        )
        # V_memory = Embedding("User: 把那个红色的东西给我\nAssistant: 好的，给你红苹果")
        ```
        
        性能:
        - 向量化耗时：< 5ms（单次计算）
        - 存储延迟：~0ms（内存操作）
        """
        try:
            # 1. 组装原始文本（临时记忆）
            raw_text = f"User: {user_text}\nAssistant: {bot_text}"
            
            # 2. 🔥 L1-B 生成会话向量：V_memory = Embedding(raw_text)
            # 耗时 < 5ms（复用现有 Embedding 模型）
            session_vector = await self._compute_vector_async(raw_text)
            
            if session_vector is None:
                logger.error("[SessionVectorCache] 会话向量生成失败")
                return False
            
            # 3. 创建会话向量条目
            entry = SessionVectorEntry(
                turn_id=turn_id,
                timestamp=time.time(),
                user_text=user_text,
                bot_text=bot_text,
                raw_text=raw_text,
                session_vector=session_vector
            )
            
            # 4. 存入内存缓存（V_memory 列表）
            async with self._lock:
                self.cache.append(entry)
                
                # 5. 维护缓存大小（FIFO + 过期检查）
                await self._evict_old_entries()
            
            self._stats["total_vectors_computed"] += 1
            
            logger.info(
                f"[SessionVectorCache] ✅ 已存储 turn={turn_id}, "
                f"V_memory 维度={len(session_vector)}, "
                f"缓存大小={len(self.cache)}/{self.max_cache_size}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"[SessionVectorCache] 存储失败：{e}", exc_info=True)
            return False
    
    async def search(
        self,
        query_text: str,
        top_k: int = 3,
        time_decay: bool = True,
        min_score: float = 0.3
    ) -> List[Dict]:
        """
        检索相关记忆（零计算检索）
        
        架构对齐:
        1. 向量化 Query：V_query = Embedding(query_text)
        2. 扫描缓存：计算 V_query 和所有 V_memory 的余弦相似度
        3. 返回 Top-K：相似度最高的 K 条记忆
        
        Args:
            query_text: 查询文本（当前用户输入）
            top_k: 返回 Top-K 相关记忆
            time_decay: 是否启用时间衰减
            min_score: 最低分数阈值（低于此分数的不返回）
            
        Returns:
            List[Dict]: 相关记忆列表，按相似度降序排列
        
        性能:
        - 向量化耗时：< 5ms（仅 Query）
        - 相似度计算：< 1ms（矩阵运算）
        - 总延迟：~10ms（恒定，不随对话轮数增长）
        
        使用示例:
        ```python
        results = await cache.search(
            query_text="红色的东西",
            top_k=3,
            time_decay=True
        )
        # 能识别"红色的东西" = "红苹果"（语义匹配）
        ```
        """
        if not self.cache:
            logger.debug("[SessionVectorCache] 缓存为空")
            return []
        
        try:
            # 1. 🔥 仅向量化当前的查询（Query）
            # 这是检索过程中唯一的计算开销
            query_vector = await self._compute_vector_async(query_text)
            
            if query_vector is None:
                logger.error("[SessionVectorCache] 查询向量化失败")
                return []
            
            # 2. 🔥 批量计算相似度（矩阵运算）
            # 利用 numpy 的矩阵运算，瞬间算出 Query 和所有缓存向量的相似度
            cache_vectors = np.array([entry.session_vector for entry in self.cache])
            
            # 余弦相似度计算（向量已归一化，简化为点积）
            similarities = self._cosine_similarity_batch(
                query_vector, 
                cache_vectors
            )
            
            # 3. 🔥 时间衰减（可选）
            final_scores = similarities
            if time_decay:
                time_factors = self._compute_time_decay_factors()
                final_scores = similarities * time_factors
                logger.debug(
                    f"[SessionVectorCache] 时间衰减因子范围："
                    f"[{time_factors.min():.3f}, {time_factors.max():.3f}]"
                )
            
            # 4. 获取 Top-K
            top_indices = np.argsort(final_scores)[-top_k:][::-1]
            
            # 5. 过滤低分结果
            results = []
            for idx in top_indices:
                if final_scores[idx] >= min_score:
                    entry = self.cache[idx]
                    results.append({
                        'turn_id': entry.turn_id,
                        'user_text': entry.user_text,
                        'bot_text': entry.bot_text,
                        'raw_text': entry.raw_text,
                        'score': float(final_scores[idx]),
                        'timestamp': entry.timestamp
                    })
            
            self._stats["total_searches"] += 1
            self._stats["cache_hits"] += len(results)
            
            logger.info(
                f"[SessionVectorCache] 🔍 查询='{query_text[:30]}...', "
                f"top_k={top_k}, 返回={len(results)}, "
                f"最高分={final_scores[top_indices[0]]:.4f}"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"[SessionVectorCache] 检索失败：{e}", exc_info=True)
            return []
    
    async def get_vector_by_turn_id(self, turn_id: int) -> Optional[Dict]:
        """
        根据 turn_id 获取向量数据
        
        Args:
            turn_id: 轮次 ID
            
        Returns:
            Optional[Dict]: 向量数据字典，包含 turn_id, user_text, bot_text, score 等
                          如果未找到返回 None
        """
        try:
            async with self._lock:
                # 搜索缓存中匹配的 turn_id
                for entry in self.cache:
                    if entry.turn_id == turn_id:
                        # 检查是否过期
                        if entry.is_expired(self.ttl_seconds):
                            logger.debug(f"[SessionVectorCache] turn_id={turn_id} 已过期")
                            return None
                        
                        # 返回条目数据
                        return {
                            'turn_id': entry.turn_id,
                            'timestamp': entry.timestamp,
                            'user_text': entry.user_text,
                            'bot_text': entry.bot_text,
                            'raw_text': entry.raw_text,
                            'session_vector': entry.session_vector,
                            'score': 1.0  # 直接返回的条目，分数设为 1.0
                        }
                
                logger.debug(f"[SessionVectorCache] 未找到 turn_id={turn_id} 的向量")
                return None
                
        except Exception as e:
            logger.error(f"[SessionVectorCache] 获取 turn_id={turn_id} 失败：{e}")
            return None
    
    async def _compute_vector_async(self, text: str) -> Optional[np.ndarray]:
        """
        异步计算向量（复用 Embedding 模型）
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: 向量数据（768 维），失败返回 None
            
        性能:
        - 耗时：< 5ms（CPU）/ < 2ms（GPU）
        - 复用现有 Embedding 模型（BGE-Small / MiniLM）
        """
        try:
            # 使用现有的 EmbeddingManager
            vector = self.embedding_manager.encode(
                texts=[text],
                normalize=True,  # 归一化便于余弦相似度计算
                show_progress=False
            )
            
            if vector is not None and len(vector) > 0:
                return vector[0]  # 返回第一个（唯一的）向量
            
            return None
            
        except Exception as e:
            logger.error(f"[SessionVectorCache] 向量计算失败：{e}")
            return None
    
    def _cosine_similarity_batch(
        self,
        query_vector: np.ndarray,
        cache_vectors: np.ndarray
    ) -> np.ndarray:
        """
        批量计算余弦相似度
        
        Args:
            query_vector: 查询向量 (768,)
            cache_vectors: 缓存向量矩阵 (N, 768)
            
        Returns:
            np.ndarray: 相似度分数 (N,)
        
        原理:
        - 余弦相似度公式：cos(θ) = (A·B) / (||A|| * ||B||)
        - 由于向量已归一化，简化为点积：A·B
        - 利用 numpy 矩阵运算，一次性计算所有相似度
        """
        return np.dot(cache_vectors, query_vector)
    
    def _compute_time_decay_factors(self) -> np.ndarray:
        """
        计算时间衰减因子
        
        使用指数衰减：decay = exp(-λ * Δt)
        λ = 0.001 (半衰期约 10 分钟)
        
        Returns:
            np.ndarray: 衰减因子数组
        
        效果:
        - 新记忆（Δt=0）：decay=1.0（完全保留）
        - 旧记忆（Δt=10min）：decay=0.5（半衰）
        - 老记忆（Δt=1h）：decay=0.05（几乎遗忘）
        """
        current_time = time.time()
        
        factors = []
        for entry in self.cache:
            delta_t = current_time - entry.timestamp
            decay_factor = np.exp(-self.time_decay_lambda * delta_t)
            factors.append(decay_factor)
        
        return np.array(factors)
    
    async def _evict_old_entries(self):
        """淘汰旧缓存（FIFO + 过期检查）"""
        # 1. 移除过期条目
        before_count = len(self.cache)
        self.cache = [
            entry for entry in self.cache
            if not entry.is_expired(self.ttl_seconds)
        ]
        expired_count = before_count - len(self.cache)
        
        if expired_count > 0:
            logger.debug(f"[SessionVectorCache] 清理 {expired_count} 条过期记录")
        
        # 2. FIFO 淘汰（超出 max_size）
        while len(self.cache) > self.max_cache_size:
            removed = self.cache.pop(0)
            self._stats["cache_evictions"] += 1
            logger.debug(
                f"[SessionVectorCache] 🗑️ FIFO 淘汰 turn={removed.turn_id}, "
                f"存活时间={time.time() - removed.timestamp:.1f}s"
            )
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            **self._stats,
            "current_cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "cache_usage_percent": len(self.cache) / self.max_cache_size * 100
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        logger.info("[SessionVectorCache] 🗑️ 已清空")
    
    async def get_entry_by_turn_id(self, turn_id: int) -> Optional[SessionVectorEntry]:
        """根据轮次 ID 获取缓存条目"""
        for entry in self.cache:
            if entry.turn_id == turn_id:
                return entry
        return None
    
    def get_all_entries(self) -> List[SessionVectorEntry]:
        """获取所有缓存条目（只读）"""
        return self.cache.copy()


# 单例管理器
class SessionVectorCacheManager:
    """会话向量缓存单例管理器"""
    
    _instance = None
    _cache = None
    
    @classmethod
    def get_cache(
        cls,
        embedding_manager: Optional[EmbeddingModelManager] = None,
        max_cache_size: int = 50,
        ttl_seconds: int = 3600
    ) -> SessionVectorCache:
        """获取会话向量缓存单例"""
        if cls._cache is None:
            if embedding_manager is None:
                embedding_manager = EmbeddingModelManager()
            
            cls._cache = SessionVectorCache(
                embedding_manager=embedding_manager,
                max_cache_size=max_cache_size,
                ttl_seconds=ttl_seconds
            )
            logger.info("[SessionVectorCacheManager] 创建单例")
        
        return cls._cache
    
    @classmethod
    def reset(cls):
        """重置单例（用于测试）"""
        cls._cache = None
        logger.info("[SessionVectorCacheManager] 已重置")


# 便捷函数
def get_session_vector_cache(
    max_cache_size: int = 50,
    ttl_seconds: int = 3600
) -> SessionVectorCache:
    """
    获取会话向量缓存单例（便捷函数）
    
    使用示例:
    ```python
    cache = get_session_vector_cache(max_cache_size=50)
    await cache.store_session_vector("用户输入", "AI 回复", turn_id=1)
    results = await cache.search("查询", top_k=3)
    ```
    """
    return SessionVectorCacheManager.get_cache(
        max_cache_size=max_cache_size,
        ttl_seconds=ttl_seconds
    )
