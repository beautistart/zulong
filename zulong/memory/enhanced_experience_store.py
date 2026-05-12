# File: zulong/memory/enhanced_experience_store.py
# 增强版经验库：支持真实 Embedding、混合检索、时间衰减、多标签过滤
# TSD v2.2 增强实现

import numpy as np
import logging
import time
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import json
import os
import pickle
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """经验数据结构（增强版）"""
    id: str
    content: str
    experience_type: str  # logic/failure/success/preference
    task_id: Optional[str] = None
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    
    # 新增字段
    keywords: List[str] = field(default_factory=list)  # 关键词（用于 BM25）
    tags: List[str] = field(default_factory=list)  # 多标签（用于多标签过滤）
    importance_score: float = 1.0  # 重要性分数（用于加权）
    access_count: int = 0  # 访问次数（用于热度计算）
    last_accessed: float = field(default_factory=time.time)  # 最后访问时间


class BM25Search:
    """BM25 关键词检索引擎
    
    用于混合检索中的关键词匹配部分
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """初始化 BM25
        
        Args:
            k1: 词频饱和度参数（默认 1.5）
            b: 长度归一化参数（默认 0.75）
        """
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, List[str]] = {}  # doc_id -> 分词列表
        self.doc_lengths: Dict[str, float] = {}  # doc_id -> 文档长度
        self.avg_doc_length: float = 0.0
        self.idf: Dict[str, float] = {}  # 词项 -> IDF 值
        self.total_docs: int = 0
    
    def add_document(self, doc_id: str, text: str):
        """添加文档到索引
        
        Args:
            doc_id: 文档 ID
            text: 文档文本
        """
        # 中文分词（简单按字符分割，实际使用可用 jieba）
        tokens = self._tokenize(text)
        
        self.documents[doc_id] = tokens
        self.doc_lengths[doc_id] = len(tokens)
        self.total_docs = len(self.documents)
        
        # 更新平均文档长度
        self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs
        
        # 更新 IDF
        self._update_idf(tokens)
    
    def _tokenize(self, text: str) -> List[str]:
        """文本分词（支持 jieba）
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 分词列表
        """
        text = text.lower()
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        
        # 尝试使用 jieba 分词
        try:
            import jieba
            tokens = list(jieba.cut(text))
            logger.debug(f"[BM25] jieba 分词结果：{tokens[:10]}")
        except ImportError:
            # 降级方案：按字符分割
            logger.warning("[BM25] jieba 未安装，使用字符级分词")
            tokens = list(text.replace(' ', ''))
        
        return [t for t in tokens if t.strip()]
    
    def _update_idf(self, tokens: List[str]):
        """更新 IDF 值
        
        Args:
            tokens: 词项列表
        """
        doc_freq: Dict[str, int] = {}
        
        for doc_id, doc_tokens in self.documents.items():
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        
        # 计算 IDF: log((N - df + 0.5) / (df + 0.5))
        for token, df in doc_freq.items():
            self.idf[token] = np.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """BM25 搜索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            List[Tuple[str, float]]: (doc_id, score) 列表
        """
        query_tokens = self._tokenize(query)
        
        scores: Dict[str, float] = {}
        
        for doc_id, doc_tokens in self.documents.items():
            score = 0.0
            doc_len = self.doc_lengths[doc_id]
            
            # 计算每个查询词项的得分
            for token in query_tokens:
                if token not in self.idf:
                    continue
                
                # 词频
                tf = doc_tokens.count(token)
                
                # BM25 公式
                idf = self.idf[token]
                numerator = idf * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                
                score += numerator / denominator
            
            if score > 0:
                scores[doc_id] = score
        
        # 排序返回 Top-K
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]


class EnhancedExperienceStore:
    """增强版经验库
    
    支持功能:
    1. ✅ 真实 Embedding 模型集成
    2. ✅ 混合检索（向量 + BM25 关键词）
    3. ✅ 时间衰减因子
    4. ✅ 多标签组合过滤
    """
    
    _instance = None
    
    def __new__(cls, db_path: Optional[str] = None, 
                enable_persistence: bool = True,
                enable_smart_tagging: bool = True,
                hot_update_engine=None):
        """单例模式
        
        Args:
            db_path: 数据库路径
            enable_persistence: 是否启用持久化
            enable_smart_tagging: 是否启用智能打标系统
            hot_update_engine: 热更新引擎实例（用于事件驱动）
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, db_path: Optional[str] = None, 
                 enable_persistence: bool = True,
                 enable_smart_tagging: bool = True,
                 hot_update_engine=None):  # 新增：热更新引擎引用
        """初始化增强版经验库
        
        Args:
            db_path: 数据库路径
            enable_persistence: 是否启用持久化
            enable_smart_tagging: 是否启用智能打标系统
            hot_update_engine: 热更新引擎实例（用于事件驱动）
        """
        if not hasattr(self, '_initialized'):
            self.db_path = db_path or "data/experience_db"
            self._experiences: Dict[str, Experience] = {}
            self.hot_update_engine = hot_update_engine  # 保存引用
            
            # Embedding 模型
            self._embedding_model = None
            
            # BM25 索引
            self.bm25_index = BM25Search()
            
            # 检索配置
            self.hybrid_alpha = 0.7  # 向量检索权重（0-1，越大越依赖向量）
            self.time_decay_factor = 0.1  # 时间衰减因子（每天衰减比例）
            self.max_age_days = 30  # 最大保留天数
            
            # 智能打标配置
            self.enable_smart_tagging = enable_smart_tagging
            
            # 持久化支持
            self.enable_persistence = enable_persistence
            self._sqlite_conn: Optional[sqlite3.Connection] = None
            self._pickle_path: Optional[Path] = None
            
            if enable_persistence:
                self._init_persistence()
                self._load_from_disk()
            
            self._initialized = True
            logger.info(f"[EnhancedExperienceStore] 初始化完成，路径：{self.db_path}, "
                       f"持久化：{enable_persistence}, "
                       f"智能打标：{enable_smart_tagging}, "
                       f"事件驱动：{hot_update_engine is not None}")
    
    def set_embedding_model(self, model):
        """设置 Embedding 模型
        
        Args:
            model: Embedding 模型实例（如 sentence-transformers, BAAI/bge 等）
        """
        self._embedding_model = model
        logger.info(f"[EnhancedExperienceStore] Embedding 模型已设置：{type(model)}")
    
    def configure_hybrid_search(self, 
                                alpha: float = 0.7,
                                time_decay: float = 0.1,
                                max_age_days: int = 30):
        """配置混合检索参数
        
        Args:
            alpha: 向量检索权重（0-1）
            time_decay: 时间衰减因子（每天衰减比例）
            max_age_days: 经验最大保留天数
        """
        self.hybrid_alpha = max(0.0, min(1.0, alpha))
        self.time_decay_factor = time_decay
        self.max_age_days = max_age_days
        
        logger.info(f"[EnhancedExperienceStore] 混合检索配置：alpha={alpha}, "
                   f"decay={time_decay}, max_age={max_age_days}天")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的向量表示（支持真实 Embedding）
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: 向量表示
        """
        if self._embedding_model is None:
            # 降级方案：使用模拟向量（512 维，BGE-small-zh-v1.5 维度）
            logger.warning("Embedding 模型未加载，使用模拟向量")
            return np.random.rand(512).astype(np.float32)
        
        try:
            # 支持多种 Embedding 模型接口
            if hasattr(self._embedding_model, 'encode'):
                # sentence-transformers 风格
                embedding = self._embedding_model.encode([text])[0]
            elif hasattr(self._embedding_model, 'embed_query'):
                # LangChain 风格
                embedding = np.array(self._embedding_model.embed_query(text))
            else:
                logger.warning("未知的 Embedding 模型接口，使用模拟向量")
                return np.random.rand(768).astype(np.float32)
            
            # 归一化（便于余弦相似度计算）
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"[EnhancedExperienceStore] Embedding 失败：{e}")
            return np.random.rand(768).astype(np.float32)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 关键词列表
        """
        # 简单实现：提取名词性词汇
        # TODO: 集成 jieba 词性标注
        keywords = []
        
        # 简单规则：提取 2-4 字连续词
        for i in range(len(text) - 1):
            for j in range(i + 2, min(i + 5, len(text))):
                word = text[i:j]
                if word not in keywords:
                    keywords.append(word)
        
        return keywords[:20]  # 限制关键词数量
    
    def _extract_tags(self, content: str, experience_type: str) -> List[str]:
        """自动提取标签（增强版：智能打标系统）
        
        Args:
            content: 经验内容
            experience_type: 经验类型
            
        Returns:
            List[str]: 标签列表
        """
        tags = set()
        
        # 基于类型的标签
        tags.add(experience_type)
        
        # 智能打标（如果启用）
        if hasattr(self, 'enable_smart_tagging') and self.enable_smart_tagging:
            try:
                from .smart_tagging import get_smart_tagger
                
                tagger = get_smart_tagger(self._embedding_model)
                tagged_domains = tagger.tag(
                    content,
                    use_default=True,
                    default_tag="general",
                    min_confidence=0.3
                )
                
                # 添加置信度 > 0.4 的领域标签
                for domain, confidence in tagged_domains:
                    if confidence > 0.4:
                        tags.add(domain)
                        logger.debug(f"[SmartTagging] {content[:30]}... → "
                                   f"{domain} (置信度：{confidence:.2f})")
                
            except Exception as e:
                logger.warning(f"[SmartTagging] 打标失败，降级到规则匹配：{e}")
                # 降级到规则匹配
        
        # 基于内容的标签（简单关键词提取）
        keywords = self._extract_keywords(content)
        tags.update(keywords[:5])  # 添加前 5 个关键词作为标签
        
        # 领域识别（基于关键词，降级方案）
        if len(tags) == 1:  # 只有 experience_type
            domain_keywords = {
                "network": ["网络", "WiFi", "路由器", "网速", "DNS"],
                "navigation": ["导航", "路径", "避障", "移动", "定位"],
                "manipulation": ["抓取", "操作", "物体", "机械臂", "夹持"],
                "vision": ["视觉", "图像", "识别", "检测", "摄像头"],
                "dialog": ["对话", "聊天", "回复", "回答", "问题"]
            }
            
            for domain, keywords in domain_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in content.lower():
                        tags.add(domain)
                        break
            
            # 默认标签
            if len(tags) == 1:
                tags.add("general")
        
        return list(tags)
    
    def add_experience(self, content: str,
                       experience_type: str = "logic",
                       task_id: Optional[str] = None,
                       success: bool = True,
                       metadata: Optional[Dict] = None,
                       tags: Optional[List[str]] = None,
                       importance_score: float = 1.0) -> str:
        """添加新经验（增强版 + 事件驱动）
        
        Args:
            content: 经验内容
            experience_type: 经验类型
            task_id: 关联的任务 ID
            success: 是否成功
            metadata: 元数据
            tags: 自定义标签（可选，自动提取）
            importance_score: 重要性分数
            
        Returns:
            str: 经验 ID
        """
        import uuid
        import asyncio
        exp_id = str(uuid.uuid4())
        
        # 计算向量（使用真实 Embedding）
        embedding = self._get_embedding(content)
        
        # 提取关键词（用于 BM25）
        keywords = self._extract_keywords(content)
        
        # 提取标签（用于多标签过滤）
        if tags:
            exp_tags = tags
        else:
            exp_tags = self._extract_tags(content, experience_type)
        
        # 创建经验对象
        experience = Experience(
            id=exp_id,
            content=content,
            experience_type=experience_type,
            task_id=task_id,
            success=success,
            metadata=metadata or {},
            embedding=embedding,
            keywords=keywords,
            tags=exp_tags,
            importance_score=importance_score
        )
        
        # 存储
        self._experiences[exp_id] = experience
        
        # 添加到 BM25 索引
        self.bm25_index.add_document(exp_id, content)
        
        logger.info(f"[EnhancedExperienceStore] 添加经验：{exp_id[:8]}, "
                   f"类型={experience_type}, 标签={exp_tags[:3]}")
        
        # 自动保存（如果启用持久化）
        if self.enable_persistence:
            self._save_to_disk()
        
        # 【关键】写入成功后，立即触发补丁生成（事件驱动）
        # 这样只有新数据进来时才计算，没有数据时零开销
        if self.hot_update_engine:
            try:
                # 异步调用，不阻塞主流程
                asyncio.create_task(
                    self.hot_update_engine.on_experience_added(experience)
                )
                logger.debug(f"[EnhancedExperienceStore] 已触发补丁生成：{exp_id[:8]}")
            except Exception as e:
                logger.error(f"[EnhancedExperienceStore] 触发补丁生成失败：{e}")
        
        return exp_id
    
    def search(self, 
               query_vector: np.ndarray,
               query_text: Optional[str] = None,
               filter_types: Optional[List[str]] = None,
               filter_tags: Optional[List[str]] = None,
               tag_logic: str = "OR",
               use_hybrid: bool = True,
               apply_time_decay: bool = True,
               limit: int = 5) -> List[Tuple[float, Experience]]:
        """增强版检索（支持混合检索、时间衰减、多标签过滤）
        
        Args:
            query_vector: 查询向量
            query_text: 查询文本（用于 BM25，可选）
            filter_types: 经验类型过滤列表（如 ["logic", "success"]）
            filter_tags: 标签过滤列表（如 ["network", "hardware"]）
            tag_logic: 标签过滤逻辑（"OR" 或 "AND"）
            use_hybrid: 是否使用混合检索
            apply_time_decay: 是否应用时间衰减
            limit: 返回数量
            
        Returns:
            List[Tuple[float, Experience]]: (综合得分，经验) 列表
        """
        results = []
        current_time = time.time()
        
        for exp in self._experiences.values():
            # 1. 类型过滤（支持多类型）
            if filter_types and exp.experience_type not in filter_types:
                continue
            
            # 2. 标签过滤（支持多标签）
            if filter_tags:
                if tag_logic == "AND":
                    # 必须包含所有标签
                    if not all(tag in exp.tags for tag in filter_tags):
                        continue
                else:  # OR
                    # 至少包含一个标签
                    if not any(tag in exp.tags for tag in filter_tags):
                        continue
            
            # 3. 计算向量相似度
            if exp.embedding is not None:
                vector_score = np.dot(query_vector, exp.embedding)
                # 归一化到 0-1
                vector_score = (vector_score + 1) / 2
            else:
                vector_score = 0.0
            
            # 4. 计算 BM25 关键词得分（如果启用混合检索）
            bm25_score = 0.0
            if use_hybrid and query_text:
                bm25_results = self.bm25_index.search(query_text, top_k=limit * 2)
                for doc_id, score in bm25_results:
                    if doc_id == exp.id:
                        # 归一化 BM25 得分
                        bm25_score = min(1.0, score / 10.0)
                        break
            
            # 5. 混合得分
            if use_hybrid:
                combined_score = (
                    self.hybrid_alpha * vector_score + 
                    (1 - self.hybrid_alpha) * bm25_score
                )
            else:
                combined_score = vector_score
            
            # 6. 应用时间衰减
            if apply_time_decay:
                age_days = (current_time - exp.timestamp) / (24 * 3600)
                
                # 指数衰减
                if age_days > self.max_age_days:
                    time_factor = 0.0  # 超过最大年龄，直接淘汰
                else:
                    time_factor = np.exp(-self.time_decay_factor * age_days)
                
                combined_score *= time_factor
            
            # 7. 应用重要性权重
            combined_score *= exp.importance_score
            
            # 8. 应用热度权重（访问次数）
            heat_factor = np.log(exp.access_count + 1) / np.log(100)  # 归一化
            combined_score *= (1 + 0.1 * heat_factor)
            
            results.append((combined_score, exp))
        
        # 排序
        results.sort(key=lambda x: x[0], reverse=True)
        
        # 更新访问记录
        for score, exp in results[:limit]:
            exp.access_count += 1
            exp.last_accessed = current_time
        
        return results[:limit]
    
    def search_by_text(self, query: str,
                       filter_types: Optional[List[str]] = None,
                       filter_tags: Optional[List[str]] = None,
                       tag_logic: str = "OR",
                       use_hybrid: bool = True,
                       apply_time_decay: bool = True,
                       limit: int = 5) -> List[Experience]:
        """通过文本查询（增强版）
        
        Args:
            query: 查询文本
            filter_types: 经验类型过滤列表
            filter_tags: 标签过滤列表
            tag_logic: 标签过滤逻辑（"OR" 或 "AND"）
            use_hybrid: 是否使用混合检索
            apply_time_decay: 是否应用时间衰减
            limit: 返回数量
            
        Returns:
            List[Experience]: 匹配的经验列表
        """
        # 向量化查询
        query_vector = self._get_embedding(query)
        
        # 执行检索
        results = self.search(
            query_vector=query_vector,
            query_text=query,
            filter_types=filter_types,
            filter_tags=filter_tags,
            tag_logic=tag_logic,
            use_hybrid=use_hybrid,
            apply_time_decay=apply_time_decay,
            limit=limit
        )
        
        # 只返回经验对象
        return [exp for _, exp in results]
    
    def to_prompt_context(self, experiences: List[Experience]) -> str:
        """将经验转换为 Prompt 上下文
        
        Args:
            experiences: 经验列表
            
        Returns:
            str: 格式化的上下文字符串
        """
        if not experiences:
            return ""
        
        context_parts = ["## 相关经验"]
        
        for exp in experiences:
            status = "✅" if exp.success else "❌"
            
            # 基础信息
            line = f"\n{status} [{exp.experience_type}] {exp.content}"
            
            # 添加标签（如果存在）
            if exp.tags:
                tags_str = ", ".join(exp.tags[:5])
                line += f" (标签：{tags_str})"
            
            # 添加时间信息（如果较新）
            age_days = (time.time() - exp.timestamp) / (24 * 3600)
            if age_days < 7:
                line += f" [新，{age_days:.1f}天]"
            
            context_parts.append(line)
        
        return "\n".join(context_parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._experiences:
            return {"total": 0}
        
        # 类型分布
        type_counts: Dict[str, int] = {}
        for exp in self._experiences.values():
            type_counts[exp.experience_type] = type_counts.get(exp.experience_type, 0) + 1
        
        # 标签分布
        tag_counts: Dict[str, int] = {}
        for exp in self._experiences.values():
            for tag in exp.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # 时间分布
        current_time = time.time()
        age_buckets = {"0-1 天": 0, "1-7 天": 0, "7-30 天": 0, "30+ 天": 0}
        for exp in self._experiences.values():
            age_days = (current_time - exp.timestamp) / (24 * 3600)
            if age_days < 1:
                age_buckets["0-1 天"] += 1
            elif age_days < 7:
                age_buckets["1-7 天"] += 1
            elif age_days < 30:
                age_buckets["7-30 天"] += 1
            else:
                age_buckets["30+ 天"] += 1
        
        return {
            "total": len(self._experiences),
            "type_distribution": type_counts,
            "top_tags": sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "age_distribution": age_buckets,
            "avg_access_count": np.mean([exp.access_count for exp in self._experiences.values()])
        }
    
    # ========== 持久化方法 ==========
    
    def _init_persistence(self):
        """初始化持久化存储"""
        try:
            # 创建目录
            db_dir = Path(self.db_path)
            db_dir.mkdir(parents=True, exist_ok=True)
            
            # SQLite 数据库（存储元数据）
            sqlite_path = db_dir / "experiences.db"
            self._sqlite_conn = sqlite3.connect(str(sqlite_path), check_same_thread=False)
            self._create_tables()
            
            # Pickle 文件（存储向量和 BM25 索引）
            self._pickle_path = db_dir / "experiences_data.pkl"
            
            logger.info(f"[EnhancedExperienceStore] 持久化初始化完成：{self.db_path}")
            
        except Exception as e:
            logger.error(f"[EnhancedExperienceStore] 持久化初始化失败：{e}")
            self.enable_persistence = False
    
    def _create_tables(self):
        """创建数据库表"""
        if not self._sqlite_conn:
            return
        
        cursor = self._sqlite_conn.cursor()
        
        # 创建经验表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiences (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                experience_type TEXT NOT NULL,
                task_id TEXT,
                success INTEGER DEFAULT 1,
                metadata TEXT,
                timestamp REAL,
                keywords TEXT,
                tags TEXT,
                importance_score REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_type ON experiences(experience_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON experiences(timestamp)')
        
        self._sqlite_conn.commit()
        logger.debug("[EnhancedExperienceStore] 数据库表已创建")
    
    def _save_to_disk(self):
        """保存到磁盘"""
        if not self.enable_persistence:
            return
        
        try:
            # 1. 保存元数据到 SQLite
            if self._sqlite_conn:
                cursor = self._sqlite_conn.cursor()
                
                # 清空表
                cursor.execute('DELETE FROM experiences')
                
                # 插入所有经验
                for exp in self._experiences.values():
                    cursor.execute('''
                        INSERT INTO experiences 
                        (id, content, experience_type, task_id, success, metadata, 
                         timestamp, keywords, tags, importance_score, access_count, last_accessed)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        exp.id,
                        exp.content,
                        exp.experience_type,
                        exp.task_id,
                        1 if exp.success else 0,
                        json.dumps(exp.metadata),
                        exp.timestamp,
                        json.dumps(exp.keywords),
                        json.dumps(exp.tags),
                        exp.importance_score,
                        exp.access_count,
                        exp.last_accessed
                    ))
                
                self._sqlite_conn.commit()
            
            # 2. 保存向量和 BM25 索引到 Pickle 文件
            if self._pickle_path:
                pickle_data = {
                    'embeddings': {
                        exp_id: (exp.embedding.tobytes() if exp.embedding is not None else None, 
                                exp.embedding.shape if exp.embedding is not None else None)
                        for exp_id, exp in self._experiences.items()
                    },
                    'bm25_index': {
                        'documents': self.bm25_index.documents,
                        'doc_lengths': self.bm25_index.doc_lengths,
                        'avg_doc_length': self.bm25_index.avg_doc_length,
                        'idf': self.bm25_index.idf,
                        'total_docs': self.bm25_index.total_docs
                    }
                }
                
                with open(self._pickle_path, 'wb') as f:
                    pickle.dump(pickle_data, f)
            
            logger.debug(f"[EnhancedExperienceStore] 已保存 {len(self._experiences)} 条经验到磁盘")
            
        except Exception as e:
            logger.error(f"[EnhancedExperienceStore] 保存失败：{e}")
    
    def _load_from_disk(self):
        """从磁盘加载"""
        if not self.enable_persistence:
            return
        
        try:
            # 1. 从 SQLite 加载元数据
            if self._sqlite_conn:
                cursor = self._sqlite_conn.cursor()
                cursor.execute('SELECT * FROM experiences')
                rows = cursor.fetchall()
                
                for row in rows:
                    exp = Experience(
                        id=row[0],
                        content=row[1],
                        experience_type=row[2],
                        task_id=row[3],
                        success=bool(row[4]),
                        metadata=json.loads(row[5]) if row[5] else {},
                        timestamp=row[6],
                        keywords=json.loads(row[7]) if row[7] else [],
                        tags=json.loads(row[8]) if row[8] else [],
                        importance_score=row[9] if row[9] is not None else 1.0,
                        access_count=row[10] if row[10] is not None else 0,
                        last_accessed=row[11] if row[11] is not None else time.time()
                    )
                    self._experiences[exp.id] = exp
                
                logger.info(f"[EnhancedExperienceStore] 从 SQLite 加载了 {len(self._experiences)} 条经验")
            
            # 2. 从 Pickle 文件加载向量和 BM25 索引
            if self._pickle_path and self._pickle_path.exists():
                with open(self._pickle_path, 'rb') as f:
                    pickle_data = pickle.load(f)
                
                # 恢复向量
                for exp_id, (embedding_bytes, shape) in pickle_data['embeddings'].items():
                    if exp_id in self._experiences and embedding_bytes is not None:
                        embedding = np.frombuffer(embedding_bytes, dtype=np.float32).reshape(shape)
                        self._experiences[exp_id].embedding = embedding
                
                # 恢复 BM25 索引
                bm25_data = pickle_data['bm25_index']
                self.bm25_index.documents = bm25_data['documents']
                self.bm25_index.doc_lengths = bm25_data['doc_lengths']
                self.bm25_index.avg_doc_length = bm25_data['avg_doc_length']
                self.bm25_index.idf = bm25_data['idf']
                self.bm25_index.total_docs = bm25_data['total_docs']
                
                logger.info(f"[EnhancedExperienceStore] 从 Pickle 加载了向量和 BM25 索引")
            
        except Exception as e:
            logger.error(f"[EnhancedExperienceStore] 加载失败：{e}")
    
    def save(self):
        """手动保存到磁盘"""
        self._save_to_disk()
        logger.info("[EnhancedExperienceStore] 手动保存完成")
    
    def close(self):
        """关闭并保存"""
        if self.enable_persistence:
            self._save_to_disk()
            if self._sqlite_conn:
                self._sqlite_conn.close()
            logger.info("[EnhancedExperienceStore] 已关闭并保存")
_enhanced_instance: Optional[EnhancedExperienceStore] = None


def get_enhanced_experience_store(db_path: Optional[str] = None, 
                                  enable_persistence: bool = True,
                                  enable_smart_tagging: bool = True) -> EnhancedExperienceStore:
    """获取增强版经验库单例
    
    Args:
        db_path: 数据库路径
        enable_persistence: 是否启用持久化
        enable_smart_tagging: 是否启用智能打标系统
        
    Returns:
        EnhancedExperienceStore: 单例实例
    """
    global _enhanced_instance
    if _enhanced_instance is None:
        _enhanced_instance = EnhancedExperienceStore(
            db_path=db_path,
            enable_persistence=enable_persistence,
            enable_smart_tagging=enable_smart_tagging
        )
    return _enhanced_instance
