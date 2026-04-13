# File: zulong/memory/rag_libraries.py
# 三个 RAG 库实现：经验 RAG、记忆 RAG、知识 RAG

import logging
import time  # 🔥 添加 time 模块导入
from typing import Dict, Any, List, Optional
import numpy as np

from .base_rag_library import BaseRAGLibrary, RAGDocument
from ..models.embedding_model import embedding_model

logger = logging.getLogger(__name__)


class ExperienceRAG(BaseRAGLibrary):
    """经验 RAG 库
    
    TSD v1.7 对应规则:
    - 存储系统提示词归类
    - 任务执行经验（成功/失败案例）
    - 技能使用经验
    
    标签:
    - importance: must_learn（重要经验）
    - domain: navigation/manipulation/vision/dialog
    """
    
    def __init__(self, **kwargs):
        # 移除 kwargs 中的 dimension，避免重复传递
        kwargs.pop('dimension', None)
        # ✅ 修复：BAAI/bge-small-zh-v1.5 实际输出 512 维
        super().__init__(name="experience_rag", dimension=512, **kwargs)
        
        # 经验分类
        self.experience_categories = {
            "task_success": [],  # 任务成功案例
            "task_failure": [],  # 任务失败案例
            "skill_usage": [],   # 技能使用经验
            "system_prompt": []  # 系统提示词归类
        }
        
        logger.info("[ExperienceRAG] Initialized")
    
    def add_document(self, document: RAGDocument) -> str:
        """添加经验文档"""
        # 生成文档 ID
        doc_id = f"exp_{len(self.documents)}_{int(document.created_at)}"
        
        # 存储文档
        self.documents[doc_id] = document
        
        # 添加到向量索引（传入 doc_id 作为向量 ID）
        if document.embedding is not None:
            # 使用 doc_id 作为向量 ID
            self.vector_store.add_vectors_with_ids(
                document.embedding,
                metadata=[document.to_dict()],
                vector_ids=[doc_id]
            )
        
        # 分类到经验类别
        category = document.metadata.get("category", "general")
        if category in self.experience_categories:
            self.experience_categories[category].append(doc_id)
        
        self.total_adds += 1
        logger.debug(f"[ExperienceRAG] Added document: {doc_id}, category={category}")
        
        return doc_id
    
    def search_documents(self, query: str, top_k: int = 5,
                        filters: Optional[Dict] = None) -> List[RAGDocument]:
        """搜索经验文档
        
        Args:
            query: 查询文本（需要向量化）
            top_k: 返回数量
            filters: 过滤器（如 category, importance 等）
            
        Returns:
            List[RAGDocument]: 相关文档列表
        """
        try:
            # 1. 确保 embedding 模型已加载
            if embedding_model.model is None:
                embedding_model.load()
            
            # 2. 将 query 转换为向量
            query_vector = embedding_model.encode_query(query)
            
            # 3. 在向量空间中搜索
            indices, distances = self.vector_store.search(query_vector, top_k=top_k)
            
            # 4. 转换为相似度分数（距离 -> 相似度）
            # FAISS 返回 L2 距离，转换为相似度：similarity = 1 / (1 + distance)
            similarities = [1.0 / (1.0 + dist) for dist in distances]
            
            # 5. 获取文档
            results = []
            for idx, sim in zip(indices, similarities):
                doc_id = self.vector_store.reverse_id_map.get(idx)
                if doc_id and doc_id in self.documents:
                    doc = self.documents[doc_id]
                    doc.similarity = sim  # 添加相似度
                    results.append(doc)
            
            logger.info(f"[ExperienceRAG] Found {len(results)} documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            import traceback
            logger.error(f"[ExperienceRAG] Search error: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def add_experience(self, content: str, category: str, 
                      importance: str = "must_learn",
                      domain: str = "general") -> str:
        """便捷方法：添加经验
        
        Args:
            content: 经验内容
            category: 类别（task_success/task_failure/skill_usage/system_prompt）
            importance: 重要性
            domain: 领域
            
        Returns:
            str: 文档 ID
        """
        # 🔥 v3.0 新增：存储层去重检查 (原子操作)
        if self._is_duplicate(content, threshold=0.95):
            logger.warning(f"⚠️ [去重拦截] 发现高度相似经验，已跳过写入。内容：{content[:50]}...")
            return ""  # 返回空字符串表示未写入
        
        doc = RAGDocument(
            content=content,
            metadata={"category": category},
            importance=importance,
            domain=domain
        )
        return self.add_document(doc)
    
    def _is_duplicate(self, new_text: str, threshold: float = 0.95) -> bool:
        """🔥 v3.0 新增：检查是否为重复经验
        
        策略:
        1. 先查 SimHash (快)
        2. 再查向量相似度 (准)
        
        Args:
            new_text: 新文本
            threshold: 相似度阈值 (默认 0.95)
            
        Returns:
            bool: 是否重复
        """
        try:
            # 策略 1: SimHash 快速过滤 (防止海量数据全表扫描)
            new_fingerprint = self._simhash(new_text)
            
            # 遍历检查已有文档 (可优化为批量查询)
            for doc_id, doc in self.documents.items():
                # 获取已有文档的 SimHash 指纹 (缓存)
                existing_fingerprint = doc.metadata.get("simhash_fingerprint")
                
                if existing_fingerprint is None:
                    # 计算并缓存
                    existing_fingerprint = self._simhash(doc.content)
                    doc.metadata["simhash_fingerprint"] = existing_fingerprint
                
                # SimHash 相似 (海明距离<3)
                if self._hamming_distance(new_fingerprint, existing_fingerprint) < 3:
                    # 进行精确的向量相似度比对
                    similarity = self._vector_similarity(new_text, doc)
                    if similarity > threshold:
                        return True  # 确认为重复
            
            return False
            
        except Exception as e:
            logger.error(f"[ExperienceRAG] 去重检查失败：{e}")
            return False  # 失败时不拦截
    
    def _simhash(self, text: str) -> int:
        """🔥 v3.0 新增：计算 SimHash 指纹
        
        Args:
            text: 文本
            
        Returns:
            int: SimHash 指纹值
        """
        try:
            from simhash import Simhash
            # 使用 jieba 分词
            import jieba
            features = list(jieba.cut(text))
            return Simhash(features).value
        except ImportError:
            # 如果 simhash 未安装，降级到简单哈希
            logger.warning("[ExperienceRAG] simhash 未安装，使用简单哈希")
            return hash(text) & 0xFFFFFFFFFFFFFFFF
    
    def _hamming_distance(self, h1: int, h2: int) -> int:
        """🔥 v3.0 新增：计算海明距离
        
        Args:
            h1: 哈希 1
            h2: 哈希 2
            
        Returns:
            int: 海明距离
        """
        return bin(h1 ^ h2).count('1')
    
    def _vector_similarity(self, text: str, doc: RAGDocument) -> float:
        """🔥 v3.0 新增：计算向量余弦相似度
        
        Args:
            text: 新文本
            doc: 已有文档
            
        Returns:
            float: 余弦相似度 (0-1)
        """
        try:
            # 1. 确保 embedding 模型已加载
            if embedding_model.model is None:
                embedding_model.load()
            
            # 2. 计算新文本的向量
            new_vector = embedding_model.encode_query(text)
            
            # 3. 获取已有文档的向量
            if doc.embedding is not None:
                existing_vector = np.array(doc.embedding)
                
                # 4. 余弦相似度
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity([new_vector], [existing_vector])[0][0]
                
                return float(similarity)
            else:
                # 没有向量，返回低相似度
                return 0.0
                
        except Exception as e:
            logger.error(f"[ExperienceRAG] 向量相似度计算失败：{e}")
            return 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        vector_stats = self.vector_store.get_stats()
        
        return {
            "name": self.name,
            "total_documents": len(self.documents),
            "total_adds": self.total_adds,
            "total_searches": self.total_searches,
            "category_counts": {
                cat: len(ids) 
                for cat, ids in self.experience_categories.items()
            },
            "vector_store": vector_stats
        }


class MemoryRAG(BaseRAGLibrary):
    """🔥 阶段 3：记忆 RAG 库（支持淡忘机制）
    
    TSD v1.7 对应规则:
    - 存储上下文和事件分类
    - 形成长中短期记忆
    - 记忆自进化
    - 🔥 淡忘机制（基于艾宾浩斯曲线 + 访问频率）
    
    标签:
    - memorability: must_remember（必须记住）
    - time_span: short_term/medium_term/long_term
    """
    
    def __init__(self, **kwargs):
        # 移除 kwargs 中的 dimension，避免重复传递
        kwargs.pop('dimension', None)
        # ✅ 修复：BAAI/bge-small-zh-v1.5 实际输出 512 维
        super().__init__(name="memory_rag", dimension=512, **kwargs)
        
        # 记忆时间跨度分类
        self.memory_time_spans = {
            "short_term": [],    # 短期记忆（< 1 小时）
            "medium_term": [],   # 中期记忆（1 小时 - 1 天）🔥 L2 半固定层
            "long_term": []      # 长期记忆（> 1 天）🔥 L3 固定层
        }
        
        # 记忆类型
        self.memory_types = {
            "context": [],       # 上下文记忆
            "event": [],         # 事件记忆
            "conversation": []   # 对话记忆
        }
        
        # 🔥 阶段 3：记忆强度追踪
        self.memory_strengths: Dict[str, MemoryStrength] = {}
        
        # 🔥 阶段 3：淡忘检查配置
        self.forget_check_interval = 6 * 3600  # 6 小时检查一次
        self.last_forget_check = time.time()
        
        logger.info("[MemoryRAG] Initialized with forgetting mechanism")
    
    def add_document(self, document: RAGDocument) -> str:
        """🔥 阶段 3：添加记忆文档（支持强度追踪）"""
        doc_id = f"mem_{len(self.documents)}_{int(document.created_at)}"
        
        # 存储文档
        self.documents[doc_id] = document
        
        # 添加到向量索引（使用自定义 ID）
        if document.embedding is not None:
            self.vector_store.add_vectors_with_ids(
                document.embedding,
                metadata=[document.to_dict()],
                vector_ids=[doc_id]
            )
        
        # 分类到时间跨度
        time_span = document.metadata.get("time_span", "short_term")
        if time_span in self.memory_time_spans:
            self.memory_time_spans[time_span].append(doc_id)
        
        # 分类到记忆类型
        mem_type = document.metadata.get("memory_type", "context")
        if mem_type in self.memory_types:
            self.memory_types[mem_type].append(doc_id)
        
        # 🔥 阶段 3：初始化记忆强度
        level = document.metadata.get("level", "L1")
        importance = document.metadata.get("importance", 0.5)
        
        # 🔥 延迟导入，避免循环依赖
        from .memory_evolution import MemoryStrength
        
        self.memory_strengths[doc_id] = MemoryStrength(
            initial_strength=importance,
            level=level,
            importance_level=document.memorability
        )
        
        self.total_adds += 1
        logger.debug(f"[MemoryRAG] Added document: {doc_id}, "
                    f"time_span={time_span}, type={mem_type}, level={level}")
        
        return doc_id
    
    def search_documents(self, query: str, top_k: int = 5,
                        filters: Optional[Dict] = None) -> List[RAGDocument]:
        """搜索记忆文档"""
        try:
            # 1. 确保 embedding 模型已加载
            if embedding_model.model is None:
                embedding_model.load()
            
            # 2. 将 query 转换为向量
            query_vector = embedding_model.encode_query(query)
            
            # 3. 在向量空间中搜索
            indices, distances = self.vector_store.search(query_vector, top_k=top_k)
            
            # 4. 转换为相似度分数
            similarities = [1.0 / (1.0 + dist) for dist in distances]
            
            # 5. 获取文档
            results = []
            for idx, sim in zip(indices, similarities):
                doc_id = self.vector_store.reverse_id_map.get(idx)
                if doc_id and doc_id in self.documents:
                    doc = self.documents[doc_id]
                    doc.similarity = sim
                    results.append(doc)
            
            logger.info(f"[MemoryRAG] Found {len(results)} documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"[MemoryRAG] Search error: {e}")
            return []
    
    def add_memory(self, content: str, memory_type: str,
                  time_span: str = "short_term",
                  memorability: str = "pending") -> str:
        """便捷方法：添加记忆
        
        Args:
            content: 记忆内容
            memory_type: 类型（context/event/conversation）
            time_span: 时间跨度（short_term/medium_term/long_term）
            memorability: 记忆重要性
            
        Returns:
            str: 文档 ID
        """
        doc = RAGDocument(
            content=content,
            metadata={
                "memory_type": memory_type,
                "time_span": time_span
            },
            memorability=memorability
        )
        return self.add_document(doc)
    
    def consolidate_memories(self) -> int:
        """🔥 阶段 3：记忆巩固：将短期记忆转为长期记忆（支持层级升降）
        
        Returns:
            int: 巩固的记忆数量
        """
        consolidated_count = 0
        
        # 遍历短期记忆
        for doc_id in self.memory_time_spans["short_term"]:
            doc = self.documents.get(doc_id)
            if doc:
                # 如果标记为必须记住，转为长期记忆
                if doc.memorability == "must_remember":
                    doc.metadata["time_span"] = "long_term"
                    self.memory_time_spans["short_term"].remove(doc_id)
                    self.memory_time_spans["long_term"].append(doc_id)
                    consolidated_count += 1
                    logger.info(f"[MemoryRAG] Consolidated memory: {doc_id}")
        
        return consolidated_count
    
    def check_and_update_forgetting(self):
        """🔥 阶段 3：检查并更新淡忘状态（每 6 小时执行）
        
        执行逻辑：
        1. 更新所有记忆强度（基于时间衰减 + 访问频率）
        2. L2 记忆评估：
           - 强度 >= 0.6 + 访问>=3 次 → 升级到 L3
           - 强度 < 0.3 → 降级到 L1 或删除
        3. L1 记忆评估：
           - 强度 < 0.2 + 访问 0 次 → 删除
        """
        current_time = time.time()
        elapsed_hours = (current_time - self.last_forget_check) / 3600
        
        logger.info(
            f"🔄 [MemoryRAG] 执行淡忘检查...\n"
            f"  距离上次检查：{elapsed_hours:.2f} 小时"
        )
        
        # 1. 更新所有记忆强度
        for doc_id, strength in self.memory_strengths.items():
            # 计算经过的时间（从最后访问时间）
            elapsed = (current_time - strength.last_access_time) / 3600
            new_strength = strength.decay(elapsed)
            
            logger.debug(
                f"  📊 {doc_id}: level={strength.level}, "
                f"strength={new_strength:.3f}, accesses={strength.access_count}"
            )
        
        # 2. 🔥 评估 L2 记忆（半固定层）
        l2_to_promote = []
        l2_to_demote = []
        
        for doc_id in self.memory_time_spans["medium_term"]:
            strength = self.memory_strengths.get(doc_id)
            if strength:
                # 升级到 L3
                if strength.should_promote():
                    l2_to_promote.append(doc_id)
                # 降级到 L1
                elif strength.should_demote():
                    l2_to_demote.append(doc_id)
        
        # 执行 L2 升级
        for doc_id in l2_to_promote:
            self._promote_to_L3(doc_id)
        
        # 执行 L2 降级
        for doc_id in l2_to_demote:
            self._demote_to_L1(doc_id)
        
        # 3. 🔥 评估 L1 记忆（短期记忆）
        l1_to_forget = []
        
        for doc_id in self.memory_time_spans["short_term"]:
            strength = self.memory_strengths.get(doc_id)
            if strength and strength.current_strength < 0.2 and strength.access_count == 0:
                l1_to_forget.append(doc_id)
        
        # 执行 L1 淡忘
        for doc_id in l1_to_forget:
            self._forget_memory(doc_id)
        
        # 更新检查时间
        self.last_forget_check = current_time
        
        logger.info(
            f"✅ [MemoryRAG] 淡忘检查完成：\n"
            f"  L2→L3 升级：{len(l2_to_promote)} 条\n"
            f"  L2→L1 降级：{len(l2_to_demote)} 条\n"
            f"  L1 淡忘：{len(l1_to_forget)} 条"
        )
    
    def _promote_to_L3(self, doc_id: str):
        """🔥 升级记忆：L2 → L3（半固定→固定）"""
        doc = self.documents.get(doc_id)
        if doc:
            # 更新时间跨度
            doc.metadata["time_span"] = "long_term"
            doc.metadata["level"] = "L3"
            
            # 更新分类
            self.memory_time_spans["medium_term"].remove(doc_id)
            self.memory_time_spans["long_term"].append(doc_id)
            
            # 更新强度
            strength = self.memory_strengths.get(doc_id)
            if strength:
                strength.level = "L3"
                strength.reinforce(boost=0.3)  # 升级时强化
            
            logger.info(f"⬆️ [MemoryRAG] 记忆升级：{doc_id} L2→L3")
    
    def _demote_to_L1(self, doc_id: str):
        """🔥 降级记忆：L2 → L1（半固定→短期）"""
        doc = self.documents.get(doc_id)
        if doc:
            # 更新时间跨度
            doc.metadata["time_span"] = "short_term"
            doc.metadata["level"] = "L1"
            
            # 更新分类
            self.memory_time_spans["medium_term"].remove(doc_id)
            self.memory_time_spans["short_term"].append(doc_id)
            
            # 更新强度
            strength = self.memory_strengths.get(doc_id)
            if strength:
                strength.level = "L1"
            
            logger.info(f"⬇️ [MemoryRAG] 记忆降级：{doc_id} L2→L1")
    
    def _forget_memory(self, doc_id: str):
        """🔥 淡忘记忆：从 L1 删除"""
        # 从向量索引删除
        # TODO: 实现向量删除逻辑
        
        # 从文档列表删除
        if doc_id in self.documents:
            del self.documents[doc_id]
        
        # 从强度追踪删除
        if doc_id in self.memory_strengths:
            del self.memory_strengths[doc_id]
        
        # 从时间跨度分类删除
        for time_span in self.memory_time_spans.values():
            if doc_id in time_span:
                time_span.remove(doc_id)
        
        logger.info(f"🗑️ [MemoryRAG] 记忆淡忘：{doc_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        vector_stats = self.vector_store.get_stats()
        
        return {
            "name": self.name,
            "total_documents": len(self.documents),
            "total_adds": self.total_adds,
            "total_searches": self.total_searches,
            "time_span_counts": {
                ts: len(ids) 
                for ts, ids in self.memory_time_spans.items()
            },
            "memory_type_counts": {
                mt: len(ids) 
                for mt, ids in self.memory_types.items()
            },
            "vector_store": vector_stats
        }


class KnowledgeRAG(BaseRAGLibrary):
    """知识 RAG 库
    
    TSD v1.7 对应规则:
    - 存储确定事实
    - 整理过的知识
    - 领域知识
    
    标签:
    - domain: navigation/manipulation/vision/general
    - certainty: confirmed/uncertain
    """
    
    def __init__(self, **kwargs):
        # 移除 kwargs 中的 dimension，避免重复传递
        kwargs.pop('dimension', None)
        # ✅ 修复：BAAI/bge-small-zh-v1.5 实际输出 512 维
        super().__init__(name="knowledge_rag", dimension=512, **kwargs)
        
        # 知识领域分类
        self.knowledge_domains = {
            "navigation": [],     # 导航知识
            "manipulation": [],   # 操作知识
            "vision": [],         # 视觉知识
            "general": []         # 通用知识
        }
        
        # 知识确定性
        self.knowledge_certainty = {
            "confirmed": [],      # 已确认事实
            "uncertain": []       # 不确定信息
        }
        
        logger.info("[KnowledgeRAG] Initialized")
    
    def add_document(self, document: RAGDocument) -> str:
        """添加知识文档"""
        doc_id = f"know_{len(self.documents)}_{int(document.created_at)}"
        
        # 存储文档
        self.documents[doc_id] = document
        
        # 添加到向量索引（使用自定义 ID）
        if document.embedding is not None:
            self.vector_store.add_vectors_with_ids(
                document.embedding,
                metadata=[document.to_dict()],
                vector_ids=[doc_id]
            )
        
        # 分类到领域
        domain = document.domain
        if domain in self.knowledge_domains:
            self.knowledge_domains[domain].append(doc_id)
        
        # 分类到确定性
        certainty = document.metadata.get("certainty", "confirmed")
        if certainty in self.knowledge_certainty:
            self.knowledge_certainty[certainty].append(doc_id)
        
        self.total_adds += 1
        logger.debug(f"[KnowledgeRAG] Added document: {doc_id}, "
                    f"domain={domain}, certainty={certainty}")
        
        return doc_id
    
    def search_documents(self, query: str, top_k: int = 5,
                        filters: Optional[Dict] = None) -> List[RAGDocument]:
        """搜索知识文档"""
        try:
            # 1. 确保 embedding 模型已加载
            if embedding_model.model is None:
                embedding_model.load()
            
            # 2. 将 query 转换为向量
            query_vector = embedding_model.encode_query(query)
            
            # 3. 在向量空间中搜索
            indices, distances = self.vector_store.search(query_vector, top_k=top_k)
            
            # 4. 转换为相似度分数
            similarities = [1.0 / (1.0 + dist) for dist in distances]
            
            # 5. 获取文档
            results = []
            for idx, sim in zip(indices, similarities):
                doc_id = self.vector_store.reverse_id_map.get(idx)
                if doc_id and doc_id in self.documents:
                    doc = self.documents[doc_id]
                    doc.similarity = sim
                    results.append(doc)
            
            logger.info(f"[KnowledgeRAG] Found {len(results)} documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"[KnowledgeRAG] Search error: {e}")
            return []
    
    def add_knowledge(self, content: str, domain: str,
                     certainty: str = "confirmed") -> str:
        """便捷方法：添加知识
        
        Args:
            content: 知识内容
            domain: 领域（navigation/manipulation/vision/general）
            certainty: 确定性（confirmed/uncertain）
            
        Returns:
            str: 文档 ID
        """
        doc = RAGDocument(
            content=content,
            metadata={"certainty": certainty},
            domain=domain,
            importance="must_learn"
        )
        return self.add_document(doc)
    
    def verify_knowledge(self, doc_id: str, is_confirmed: bool) -> bool:
        """验证知识
        
        Args:
            doc_id: 文档 ID
            is_confirmed: 是否确认
            
        Returns:
            bool: 是否成功
        """
        if doc_id not in self.documents:
            return False
        
        doc = self.documents[doc_id]
        old_certainty = doc.metadata.get("certainty", "uncertain")
        new_certainty = "confirmed" if is_confirmed else "uncertain"
        
        # 更新确定性
        doc.metadata["certainty"] = new_certainty
        doc.updated_at = __import__("time").time()
        
        # 更新分类
        if old_certainty in self.knowledge_certainty:
            if doc_id in self.knowledge_certainty[old_certainty]:
                self.knowledge_certainty[old_certainty].remove(doc_id)
        
        if new_certainty in self.knowledge_certainty:
            self.knowledge_certainty[new_certainty].append(doc_id)
        
        logger.info(f"[KnowledgeRAG] Verified knowledge: {doc_id}, "
                   f"certainty={new_certainty}")
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        vector_stats = self.vector_store.get_stats()
        
        return {
            "name": self.name,
            "total_documents": len(self.documents),
            "total_adds": self.total_adds,
            "total_searches": self.total_searches,
            "domain_counts": {
                dom: len(ids) 
                for dom, ids in self.knowledge_domains.items()
            },
            "certainty_counts": {
                cert: len(ids) 
                for cert, ids in self.knowledge_certainty.items()
            },
            "vector_store": vector_stats
        }
