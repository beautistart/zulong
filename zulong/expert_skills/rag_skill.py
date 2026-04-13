# RAG 专家技能模块

"""
功能:
- 提供领域知识检索能力
- 支持 L2 按需调用
- LRU 内存管理
- 向量检索 + 关键词检索混合
- 中文分词优化
- 缓存策略

对应 TSD v2.3 第 14 章
"""

import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import numpy as np
from collections import OrderedDict
import hashlib
import time

logger = logging.getLogger(__name__)


class RAGExpertSkill:
    """RAG 专家技能类
    
    功能:
    - 领域知识检索
    - 向量相似度搜索
    - 关键词匹配
    - 结果重排序
    
    接口规范:
    - query(): 查询领域知识
    - add_knowledge(): 添加新知识
    - get_stats(): 获取统计信息
    """
    
    def __init__(self, 
                 skill_id: str = "rag_general",
                 knowledge_base_path: Optional[str] = None,
                 top_k: int = 5,
                 use_real_embedding: bool = False):  # 新增参数
        """初始化 RAG 专家技能
        
        Args:
            skill_id: 技能 ID
            knowledge_base_path: 知识库路径
            top_k: 默认返回数量
            use_real_embedding: 是否使用真实 Embedding 模型（默认 False）
        """
        self.skill_id = skill_id
        self.knowledge_base_path = knowledge_base_path
        self.top_k = top_k
        self.use_real_embedding = use_real_embedding
        
        # 知识库（内存中的简化实现）
        self.knowledge_base: List[Dict[str, Any]] = []
        self.embeddings: List[np.ndarray] = []
        
        # Embedding 模型（可选）
        self._embedding_model = None
        
        # 缓存系统
        # 1. 向量缓存：避免重复编码
        self._vector_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._vector_cache_max_size = 1000
        
        # 2. 搜索结果缓存：避免重复查询（带 TTL）
        self._search_cache: OrderedDict[str, Dict] = OrderedDict()  # 改为 Dict，包含结果 + 时间戳
        self._search_cache_max_size = 500
        self._search_cache_ttl_seconds = 300  # 5 分钟 TTL
        
        # 统计信息（先初始化，避免模型加载时访问未定义属性）
        self.stats = {
            'total_queries': 0,
            'total_knowledge': 0,
            'avg_response_time': 0.0,
            'vector_cache_hits': 0,
            'vector_cache_misses': 0,
            'search_cache_hits': 0,
            'search_cache_misses': 0,
            'search_cache_expired': 0,  # 新增：过期淘汰统计
            'embedding_model_loaded': False  # 先设为 False，模型加载后会更新
        }
        
        # 混合检索权重
        self._vector_weight = 0.7
        self._keyword_weight = 0.3
        
        # 加载 Embedding 模型（在 stats 初始化之后）
        if use_real_embedding:
            self._load_embedding_model()
        
        logger.info(f"[RAGExpertSkill] 初始化完成：id={skill_id}, use_real_embedding={use_real_embedding}")
    
    def set_hybrid_weights(self, vector_weight: float = 0.7, keyword_weight: float = 0.3):
        """设置混合检索权重
        
        Args:
            vector_weight: 向量权重（默认 0.7）
            keyword_weight: 关键词权重（默认 0.3）
        """
        if vector_weight < 0 or keyword_weight < 0:
            raise ValueError("权重必须非负")
        
        total = vector_weight + keyword_weight
        if total <= 0:
            raise ValueError("权重之和必须大于 0")
        
        # 归一化
        self._vector_weight = vector_weight / total
        self._keyword_weight = keyword_weight / total
        
        logger.info(f"[RAGExpertSkill] 混合检索权重：向量={self._vector_weight:.2f}, 关键词={self._keyword_weight:.2f}")
    
    def _load_embedding_model(self):
        """加载真实 Embedding 模型（BAAI/bge-small-zh-v1.5）"""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info("[RAGExpertSkill] 加载真实 Embedding 模型...")
            self._embedding_model = SentenceTransformer(
                'BAAI/bge-small-zh-v1.5',
                device='cpu',  # 使用 CPU
                model_kwargs={'torch_dtype': 'float32'}  # 使用 float32
            )
            
            logger.info("[RAGExpertSkill] Embedding 模型加载成功")
            self.stats['embedding_model_loaded'] = True
            
        except Exception as e:
            logger.error(f"[RAGExpertSkill] Embedding 模型加载失败：{e}，使用模拟向量")
            self._embedding_model = None
            self.stats['embedding_model_loaded'] = False
    
    async def query(self, 
                    query_text: str,
                    top_k: Optional[int] = None,
                    filter_tags: Optional[List[str]] = None,
                    use_vector_search: bool = True,  # 新增参数
                    **kwargs) -> List[Dict[str, Any]]:
        """查询领域知识
        
        Args:
            query_text: 查询文本
            top_k: 返回数量（可选，覆盖默认值）
            filter_tags: 标签过滤（可选）
            use_vector_search: 是否使用向量检索（默认 True）
            **kwargs: 其他参数
            
        Returns:
            List[Dict]: 知识列表，包含：
                - content: 知识内容
                - score: 相关性分数
                - metadata: 元数据
        """
        import time
        start_time = time.time()
        
        k = top_k or self.top_k
        self.stats['total_queries'] += 1
        
        # 生成缓存键（包含 use_vector_search 参数）
        cache_key = self._generate_search_cache_key(query_text, k, filter_tags, use_vector_search)
        
        # 检查搜索缓存
        cached_result = self._get_search_cache(cache_key)
        if cached_result is not None:
            self.stats['search_cache_hits'] += 1
            logger.debug(f"[RAGExpertSkill] 搜索缓存命中：{query_text[:30]}...")
            return cached_result
        
        self.stats['search_cache_misses'] += 1
        
        logger.debug(f"[RAGExpertSkill] 查询：{query_text[:50]}...")
        
        # 根据参数选择检索方式
        if use_vector_search and self.embeddings:
            # 向量检索 + 关键词混合
            results = self._hybrid_search(query_text, k, filter_tags)
        else:
            # 纯关键词检索
            results = self._keyword_search(query_text, k, filter_tags)
        
        # 更新搜索缓存
        self._set_search_cache(cache_key, results)
        
        # 更新统计
        elapsed = time.time() - start_time
        self._update_response_time(elapsed)
        
        logger.debug(f"[RAGExpertSkill] 返回 {len(results)} 条结果，耗时 {elapsed*1000:.2f}ms")
        
        return results
    
    def _keyword_search(self, 
                        query: str, 
                        top_k: int,
                        filter_tags: Optional[List[str]] = None) -> List[Dict]:
        """关键词搜索（简化实现，支持中文）
        
        Args:
            query: 查询文本
            top_k: 返回数量
            filter_tags: 标签过滤
            
        Returns:
            List[Dict]: 搜索结果
        """
        if not self.knowledge_base:
            return []
        
        # 中文关键词提取（简化：提取 2-4 字短语作为关键词）
        # TODO: 集成中文分词（如 jieba）
        query_keywords = self._extract_chinese_keywords(query)
        
        scored_results = []
        for knowledge in self.knowledge_base:
            # 标签过滤
            if filter_tags:
                knowledge_tags = knowledge.get('tags', [])
                if not any(tag in knowledge_tags for tag in filter_tags):
                    continue
            
            # 计算相关性分数（基于子串匹配）
            content = knowledge.get('content', '').lower()
            
            # 统计有多少关键词在内容中
            match_count = 0
            for keyword in query_keywords:
                if keyword.lower() in content:
                    match_count += 1
            
            # 计算分数（匹配的关键词比例）
            score = match_count / len(query_keywords) if query_keywords else 0.0
            
            if score > 0:
                result = knowledge.copy()
                result['score'] = score
                result['source'] = self.skill_id
                scored_results.append(result)
        
        # 按分数排序
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_results[:top_k]
    
    def _vector_search(self, 
                       query: str, 
                       top_k: int,
                       filter_tags: Optional[List[str]] = None) -> List[Dict]:
        """向量相似度搜索（余弦相似度）
        
        Args:
            query: 查询文本
            top_k: 返回数量
            filter_tags: 标签过滤
            
        Returns:
            List[Dict]: 搜索结果
        """
        if not self.knowledge_base or not self.embeddings:
            logger.warning("[RAGExpertSkill] 向量检索不可用：缺少知识库或嵌入")
            return []
        
        try:
            # 获取查询向量（使用缓存）
            query_vector = self._get_vector_cache(query)
            if query_vector is None:
                # 使用真实模型或模拟向量
                query_vector = self._encode_text(query)
                self._set_vector_cache(query, query_vector)
            
            # 计算余弦相似度
            scores = []
            for i, knowledge in enumerate(self.knowledge_base):
                # 标签过滤
                if filter_tags:
                    knowledge_tags = knowledge.get('tags', [])
                    if not any(tag in knowledge_tags for tag in filter_tags):
                        continue
                
                # 计算余弦相似度
                if i < len(self.embeddings) and self.embeddings[i] is not None:
                    similarity = self._cosine_similarity(query_vector, self.embeddings[i])
                    
                    if similarity > 0:  # 只保留正相似度
                        result = knowledge.copy()
                        result['score'] = float(similarity)  # 转换为 Python float
                        result['source'] = self.skill_id
                        scores.append(result)
            
            # 按相似度排序
            scores.sort(key=lambda x: x['score'], reverse=True)
            
            return scores[:top_k]
            
        except Exception as e:
            logger.error(f"[RAGExpertSkill] 向量检索失败：{e}")
            return []
    
    def _hybrid_search(self, 
                       query: str, 
                       top_k: int,
                       filter_tags: Optional[List[str]] = None) -> List[Dict]:
        """混合检索（向量 + 关键词）
        
        Args:
            query: 查询文本
            top_k: 返回数量
            filter_tags: 标签过滤
            
        Returns:
            List[Dict]: 搜索结果
        """
        # 获取两种检索结果
        vector_results = self._vector_search(query, top_k * 2, filter_tags)  # 多取一些
        keyword_results = self._keyword_search(query, top_k * 2, filter_tags)
        
        # 合并结果（加权平均）
        merged_results = {}
        
        # 添加向量检索结果
        for result in vector_results:
            result_id = result.get('id', result.get('content'))
            merged_results[result_id] = {
                **result,
                'vector_score': result['score'],
                'keyword_score': 0.0
            }
        
        # 合并关键词检索结果
        for result in keyword_results:
            result_id = result.get('id', result.get('content'))
            if result_id in merged_results:
                # 已存在，更新关键词分数
                merged_results[result_id]['keyword_score'] = result['score']
            else:
                # 不存在，添加
                merged_results[result_id] = {
                    **result,
                    'vector_score': 0.0,
                    'keyword_score': result['score']
                }
        
        # 计算加权分数（使用实例权重）
        final_results = []
        for result in merged_results.values():
            result['score'] = (
                result['vector_score'] * self._vector_weight + 
                result['keyword_score'] * self._keyword_weight
            )
            final_results.append(result)
        
        # 按加权分数排序
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        return final_results[:top_k]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度
        
        Args:
            vec1: 向量 1
            vec2: 向量 2
            
        Returns:
            float: 余弦相似度（-1 到 1）
        """
        # 归一化
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 余弦相似度公式
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        
        # 确保在 [-1, 1] 范围内
        return float(np.clip(similarity, -1.0, 1.0))
    
    def _encode_text(self, text: str) -> np.ndarray:
        """编码文本为向量（支持真实模型）
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: 向量表示
        """
        # 优先使用真实模型
        if self._embedding_model is not None:
            try:
                # 使用真实 Embedding 模型
                embedding = self._embedding_model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # 归一化，便于余弦相似度计算
                )
                return embedding.astype(np.float32)
                
            except Exception as e:
                logger.error(f"[RAGExpertSkill] 真实模型编码失败：{e}，降级到模拟向量")
        
        # 降级方案：模拟向量
        return self._mock_encode(text)
    
    def _mock_encode(self, text: str) -> np.ndarray:
        """模拟向量编码（降级方案）
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: 随机向量（512 维）
        """
        # 使用文本哈希生成可重复的随机向量
        hash_value = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
        np.random.seed(hash_value % (2**32))
        
        # 生成 512 维向量
        embedding = np.random.rand(512).astype(np.float32)
        
        # 归一化
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def _extract_chinese_keywords(self, text: str) -> List[str]:
        """提取中文关键词（jieba 分词优化版）
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 关键词列表
        """
        try:
            import jieba
        except ImportError:
            logger.warning("jieba 未安装，使用简化分词")
            return self._simple_keyword_extraction(text)
        
        # 移除常见疑问词
        stop_words = {'什么', '为什么', '如何', '怎样', '怎么', '哪个', '哪些', '是', '的', '了', '在', '和'}
        
        # jieba 分词
        words = jieba.lcut(text)
        
        # 过滤停用词和单字
        keywords = [w for w in words if w not in stop_words and len(w) > 1]
        
        # 如果没有有效分词，使用简化版
        if not keywords:
            keywords = self._simple_keyword_extraction(text)
        
        logger.debug(f"[RAGExpertSkill] 分词结果：{keywords}")
        
        return keywords
    
    def _simple_keyword_extraction(self, text: str) -> List[str]:
        """简化关键词提取（降级方案）
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 关键词列表
        """
        keywords = []
        
        # 移除常见疑问词
        stop_words = ['什么是', '为什么', '如何', '怎样', '怎么', '哪个', '哪些']
        cleaned_text = text
        for sw in stop_words:
            cleaned_text = cleaned_text.replace(sw, '')
        
        # 如果清理后有内容，添加到关键词
        if cleaned_text and len(cleaned_text.strip()) > 0:
            keywords.append(cleaned_text.strip())
        
        # 同时保留原始查询
        keywords.append(text.strip())
        
        # 去重
        return list(set(keywords))
    
    async def add_knowledge(self,
                           content: str,
                           metadata: Optional[Dict[str, Any]] = None,
                           tags: Optional[List[str]] = None,
                           embedding: Optional[np.ndarray] = None) -> str:
        """添加领域知识
        
        Args:
            content: 知识内容
            metadata: 元数据
            tags: 标签列表
            embedding: 向量表示（可选，不提供则自动生成）
            
        Returns:
            str: 知识 ID
        """
        import uuid
        
        knowledge_id = str(uuid.uuid4())
        
        # 如果没有提供 embedding，使用真实模型或模拟向量生成
        if embedding is None:
            embedding = self._encode_text(content)
            logger.debug(f"[RAGExpertSkill] 自动生成知识向量：{knowledge_id[:8]}...")
        
        knowledge = {
            'id': knowledge_id,
            'content': content,
            'metadata': metadata or {},
            'tags': tags or [],
            'embedding': embedding
        }
        
        self.knowledge_base.append(knowledge)
        
        if embedding is not None:
            self.embeddings.append(embedding)
        
        self.stats['total_knowledge'] = len(self.knowledge_base)
        
        logger.info(f"[RAGExpertSkill] 添加知识：{knowledge_id}, 总数：{self.stats['total_knowledge']}")
        
        return knowledge_id
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            'skill_id': self.skill_id,
            'total_knowledge': self.stats['total_knowledge'],
            'total_queries': self.stats['total_queries'],
            'avg_response_time_ms': self.stats['avg_response_time'] * 1000
        }
    
    def _update_response_time(self, elapsed: float):
        """更新平均响应时间（移动平均）
        
        Args:
            elapsed: 本次查询耗时（秒）
        """
        alpha = 0.1  # 平滑系数
        self.stats['avg_response_time'] = (
            alpha * elapsed + 
            (1 - alpha) * self.stats['avg_response_time']
        )
    
    def _generate_search_cache_key(self, 
                                   query_text: str,
                                   top_k: int,
                                   filter_tags: Optional[List[str]],
                                   use_vector_search: bool = True) -> str:
        """生成搜索缓存键（MD5 哈希）
        
        Args:
            query_text: 查询文本
            top_k: 返回数量
            filter_tags: 标签过滤
            use_vector_search: 是否使用向量检索
            
        Returns:
            str: 缓存键
        """
        key_data = f"{query_text}:{top_k}:{sorted(filter_tags or [])}:{use_vector_search}"
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()
    
    def _get_search_cache(self, cache_key: str) -> Optional[List[Dict]]:
        """获取搜索缓存结果（带 TTL 检查）
        
        Args:
            cache_key: 缓存键
            
        Returns:
            Optional[List[Dict]]: 缓存结果，不存在或过期返回 None
        """
        if cache_key not in self._search_cache:
            return None
        
        cache_entry = self._search_cache[cache_key]
        current_time = time.time()
        
        # 检查 TTL
        if current_time - cache_entry['timestamp'] > self._search_cache_ttl_seconds:
            # 缓存已过期，删除并返回 None
            del self._search_cache[cache_key]
            self.stats['search_cache_expired'] += 1
            logger.debug(f"[RAGExpertSkill] 搜索缓存过期：{cache_key[:8]}...")
            return None
        
        # 移动到最近使用
        self._search_cache.move_to_end(cache_key)
        
        return cache_entry['results']
    
    def _set_search_cache(self, cache_key: str, results: List[Dict]):
        """设置搜索缓存（带时间戳）
        
        Args:
            cache_key: 缓存键
            results: 搜索结果
        """
        # 如果缓存已满，淘汰最旧的
        if len(self._search_cache) >= self._search_cache_max_size:
            oldest_key = next(iter(self._search_cache))
            del self._search_cache[oldest_key]
            logger.debug(f"[RAGExpertSkill] 搜索缓存淘汰：{oldest_key[:8]}...")
        
        # 添加到缓存（包含时间戳）
        self._search_cache[cache_key] = {
            'results': results,
            'timestamp': time.time()
        }
        logger.debug(f"[RAGExpertSkill] 搜索缓存添加：{cache_key[:8]}...")
    
    def _get_vector_cache(self, text: str) -> Optional[np.ndarray]:
        """获取向量缓存
        
        Args:
            text: 文本
            
        Returns:
            Optional[np.ndarray]: 缓存的向量
        """
        if text not in self._vector_cache:
            return None
        
        # 移动到最近使用
        self._vector_cache.move_to_end(text)
        
        self.stats['vector_cache_hits'] += 1
        return self._vector_cache[text]
    
    def _set_vector_cache(self, text: str, vector: np.ndarray):
        """设置向量缓存
        
        Args:
            text: 文本
            vector: 向量
        """
        # 如果缓存已满，淘汰最旧的
        if len(self._vector_cache) >= self._vector_cache_max_size:
            oldest_key = next(iter(self._vector_cache))
            del self._vector_cache[oldest_key]
            logger.debug(f"[RAGExpertSkill] 向量缓存淘汰：{oldest_key[:20]}...")
        
        self._vector_cache[text] = vector
        self.stats['vector_cache_misses'] += 1  # 新添加的算作 miss
    
    def set_cache_ttl(self, ttl_seconds: int):
        """设置缓存 TTL 时间
        
        Args:
            ttl_seconds: TTL 时间（秒）
        """
        self._search_cache_ttl_seconds = ttl_seconds
        logger.info(f"[RAGExpertSkill] 缓存 TTL 设置为：{ttl_seconds}秒")
    
    def cleanup_expired_cache(self) -> int:
        """清理所有过期的缓存
        
        Returns:
            int: 清理的缓存数量
        """
        current_time = time.time()
        expired_keys = []
        
        # 找出所有过期的缓存
        for key, entry in self._search_cache.items():
            if current_time - entry['timestamp'] > self._search_cache_ttl_seconds:
                expired_keys.append(key)
        
        # 删除过期缓存
        for key in expired_keys:
            del self._search_cache[key]
            self.stats['search_cache_expired'] += 1
        
        if expired_keys:
            logger.info(f"[RAGExpertSkill] 清理了 {len(expired_keys)} 个过期缓存")
        
        return len(expired_keys)
    
    def clear(self):
        """清空知识库和缓存（用于 LRU 卸载）"""
        self.knowledge_base.clear()
        self.embeddings.clear()
        self._vector_cache.clear()
        self._search_cache.clear()
        logger.info(f"[RAGExpertSkill] 已清空知识库和缓存：{self.skill_id}")


# 工厂函数
def get_rag_expert_skill(
    skill_id: str = "rag_general",
    knowledge_base_path: Optional[str] = None,
    top_k: int = 5
) -> RAGExpertSkill:
    """获取 RAG 专家技能实例
    
    Args:
        skill_id: 技能 ID
        knowledge_base_path: 知识库路径
        top_k: 默认返回数量
        
    Returns:
        RAGExpertSkill: 实例
    """
    return RAGExpertSkill(
        skill_id=skill_id,
        knowledge_base_path=knowledge_base_path,
        top_k=top_k
    )
