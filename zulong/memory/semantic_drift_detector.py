# File: zulong/memory/semantic_drift_detector.py
"""
语义漂移检测器 - TSD v2.4

功能：
1. 基于 Embedding 计算话题相似度
2. 检测对话主题是否发生显著变化
3. 当余弦相似度<0.4 时触发复盘

对应 TSD v2.4: 语义漂移检测、智能触发机制
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import asyncio

logger = logging.getLogger(__name__)


class SemanticDriftDetector:
    """
    语义漂移检测器
    
    核心原理：
    1. 将每轮对话转换为 Embedding 向量
    2. 计算当前对话与历史对话的余弦相似度
    3. 当相似度低于阈值（默认 0.4）时，判定为话题转换
    
    触发条件：
    - 余弦相似度 < 0.4: 显著漂移，立即触发复盘
    - 余弦相似度 < 0.6: 轻微漂移，标记待观察
    - 余弦相似度 >= 0.6: 话题稳定
    """
    
    def __init__(self, embedding_model: str = "text-embedding-ada-002"):
        """
        初始化语义漂移检测器
        
        Args:
            embedding_model: Embedding 模型名称
        """
        self.embedding_model = embedding_model
        self.embedding_cache: Dict[str, np.ndarray] = {}  # 文本 → Embedding 缓存
        
        # 阈值配置
        self.drift_threshold = 0.4  # 显著漂移阈值
        self.warning_threshold = 0.6  # 警告阈值
        
        # 历史对话 Embedding
        self._conversation_embeddings: List[np.ndarray] = []
        self._conversation_texts: List[str] = []
        
        # 统计信息
        self._stats = {
            "total_comparisons": 0,
            "drift_detected": 0,
            "warnings": 0
        }
        
        logger.info(f"[SemanticDriftDetector] 初始化完成，模型：{embedding_model}")
    
    async def get_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的 Embedding 向量（带缓存）
        
        🔥 TSD v2.4 优化：复用系统现有的 Embedding 模型管理器
        无需重复部署模型，直接使用 RAG 模块的 EmbeddingModelManager
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: Embedding 向量
        """
        # 检查缓存
        if text in self.embedding_cache:
            logger.debug(f"[SemanticDriftDetector] Embedding 缓存命中：'{text[:20]}...'")
            return self.embedding_cache[text]
        
        try:
            # 🔥 TSD v2.4 优化：复用 EmbeddingModelManager（单例模式）
            from zulong.memory.embedding_manager import get_embedding_manager
            
            embedding_manager = get_embedding_manager()
            embedding = embedding_manager.encode(text)
            
            # 缓存结果
            self.embedding_cache[text] = embedding
            
            logger.debug(f"[SemanticDriftDetector] Embedding 计算完成：维度 {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"[SemanticDriftDetector] 获取 Embedding 失败：{e}")
            # 降级：返回随机向量（用于测试）
            logger.warning("[SemanticDriftDetector] 使用随机向量降级")
            return np.random.rand(512)  # BAAI/bge-small-zh-v1.5 输出 512 维
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度
        
        Args:
            vec1: 向量 1
            vec2: 向量 2
            
        Returns:
            float: 余弦相似度（0-1）
        """
        try:
            # flatten：EmbeddingManager 可能返回 (1, dim) 形状的 2D 数组
            v1 = vec1.flatten()
            v2 = vec2.flatten()
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            # 归一化到 0-1（原始范围是 -1 到 1）
            return (similarity + 1) / 2
            
        except Exception as e:
            logger.error(f"[SemanticDriftDetector] 计算相似度失败：{e}")
            return 0.0
    
    async def add_conversation_turn(self, user_input: str, ai_response: str):
        """
        添加一轮对话到历史记录
        
        Args:
            user_input: 用户输入
            ai_response: AI 回复
        """
        try:
            # 合并对话文本
            conversation_text = f"{user_input} {ai_response}"
            
            # 获取 Embedding
            embedding = await self.get_embedding(conversation_text)
            
            # 添加到历史
            self._conversation_embeddings.append(embedding)
            self._conversation_texts.append(conversation_text)
            
            # 限制历史记录长度（最近 20 轮）
            if len(self._conversation_embeddings) > 20:
                self._conversation_embeddings.pop(0)
                self._conversation_texts.pop(0)
                
        except Exception as e:
            logger.error(f"[SemanticDriftDetector] 添加对话失败：{e}")
    
    async def detect_drift(self, current_user_input: str) -> Tuple[bool, float, str]:
        """
        检测是否发生语义漂移
        
        Args:
            current_user_input: 当前用户输入
            
        Returns:
            Tuple[bool, float, str]: (是否漂移，相似度，原因)
        """
        try:
            if len(self._conversation_embeddings) == 0:
                return False, 1.0, "无历史对话"
            
            # 获取当前输入的 Embedding
            current_embedding = await self.get_embedding(current_user_input)
            
            # 计算与历史对话的平均相似度
            similarities = []
            for hist_embedding in self._conversation_embeddings[-5:]:  # 只比较最近 5 轮
                sim = self.cosine_similarity(current_embedding, hist_embedding)
                similarities.append(sim)
            
            avg_similarity = np.mean(similarities)
            min_similarity = np.min(similarities)
            
            # 更新统计
            self._stats["total_comparisons"] += 1
            
            # 判断漂移
            if avg_similarity < self.drift_threshold:
                self._stats["drift_detected"] += 1
                return True, avg_similarity, f"显著漂移（相似度：{avg_similarity:.3f} < {self.drift_threshold}）"
            
            elif avg_similarity < self.warning_threshold:
                self._stats["warnings"] += 1
                return False, avg_similarity, f"轻微漂移（相似度：{avg_similarity:.3f} < {self.warning_threshold}）"
            
            else:
                return False, avg_similarity, f"话题稳定（相似度：{avg_similarity:.3f}）"
                
        except Exception as e:
            logger.error(f"[SemanticDriftDetector] 检测漂移失败：{e}")
            return False, 0.0, f"检测失败：{e}"
    
    async def detect_drift_with_context(self, 
                                       user_input: str, 
                                       ai_response: str) -> Tuple[bool, float, str]:
        """
        检测完整对话的语义漂移
        
        Args:
            user_input: 用户输入
            ai_response: AI 回复
            
        Returns:
            Tuple[bool, float, str]: (是否漂移，相似度，原因)
        """
        # 先添加到历史
        await self.add_conversation_turn(user_input, ai_response)
        
        # 然后检测漂移
        return await self.detect_drift(user_input)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "history_length": len(self._conversation_embeddings),
            "cache_size": len(self.embedding_cache),
            "drift_threshold": self.drift_threshold,
            "warning_threshold": self.warning_threshold
        }
    
    def clear_history(self):
        """清空历史（用于复盘后）"""
        self._conversation_embeddings.clear()
        self._conversation_texts.clear()
        logger.info("[SemanticDriftDetector] 历史已清空")


# 全局单例
_semantic_drift_detector: Optional[SemanticDriftDetector] = None


def get_semantic_drift_detector(embedding_model: str = "text-embedding-ada-002") -> SemanticDriftDetector:
    """获取语义漂移检测器单例"""
    global _semantic_drift_detector
    if _semantic_drift_detector is None:
        _semantic_drift_detector = SemanticDriftDetector(embedding_model)
    return _semantic_drift_detector
