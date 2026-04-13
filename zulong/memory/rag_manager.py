# File: zulong/memory/rag_manager.py
# RAG 管理器 - 统一管理三个 RAG 库

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import os

from .base_rag_library import BaseRAGLibrary, RAGDocument
from .rag_libraries import ExperienceRAG, MemoryRAG, KnowledgeRAG
from ..models.embedding_model import embedding_model

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """RAG 配置
    
    TSD v1.7 对应规则:
    - 向量维度配置
    - 持久化路径配置
    - 向量库类型配置
    """
    vector_dimension: int = 512  # ✅ 修复：BAAI/bge-small-zh-v1.5 实际输出 512 维
    vector_store_type: str = "faiss"  # 向量库类型
    base_path: str = "./data/rag"  # 持久化路径
    experience_rag_enabled: bool = True
    memory_rag_enabled: bool = True
    knowledge_rag_enabled: bool = True


class RAGManager:
    """RAG 管理器
    
    TSD v1.7 对应规则:
    - 2.2.5 基础设施层：RAG 记忆系统
    - 统一管理三个 RAG 库
    - 提供统一的 CRUD 接口
    
    功能:
    - 经验/记忆/知识管理
    - 跨库搜索
    - 持久化
    - 统计监控
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """初始化 RAG 管理器
        
        Args:
            config: RAG 配置
        """
        self.config = config or RAGConfig()
        
        # 初始化三个 RAG 库
        self.rag_libraries: Dict[str, BaseRAGLibrary] = {}
        
        if self.config.experience_rag_enabled:
            self.rag_libraries["experience"] = ExperienceRAG(
                dimension=self.config.vector_dimension,
                vector_store_type=self.config.vector_store_type
            )
            logger.info("[RAGManager] Experience RAG initialized")
        
        if self.config.memory_rag_enabled:
            self.rag_libraries["memory"] = MemoryRAG(
                dimension=self.config.vector_dimension,
                vector_store_type=self.config.vector_store_type
            )
            logger.info("[RAGManager] Memory RAG initialized")
        
        if self.config.knowledge_rag_enabled:
            self.rag_libraries["knowledge"] = KnowledgeRAG(
                dimension=self.config.vector_dimension,
                vector_store_type=self.config.vector_store_type
            )
            logger.info("[RAGManager] Knowledge RAG initialized")
        
        # 统计信息
        self.total_adds = 0
        self.total_searches = 0
        
        logger.info(f"[RAGManager] Initialized with {len(self.rag_libraries)} libraries")
        
        # 🔥 预加载 embedding 模型，避免首次使用时延迟
        try:
            logger.info("[RAGManager] Pre-loading embedding model...")
            if embedding_model.model is None:
                embedding_model.load()
            logger.info("[RAGManager] Embedding model pre-loaded successfully")
        except Exception as e:
            logger.warning(f"[RAGManager] Failed to pre-load embedding model: {e}")
    
    def add_document(self, library_name: str, document: RAGDocument) -> str:
        """添加文档到指定库
        
        Args:
            library_name: 库名称（experience/memory/knowledge）
            document: 文档对象
            
        Returns:
            str: 文档 ID
        """
        if library_name not in self.rag_libraries:
            logger.error(f"[RAGManager] Unknown library: {library_name}")
            return ""
        
        library = self.rag_libraries[library_name]
        doc_id = library.add_document(document)
        
        self.total_adds += 1
        logger.info(f"[RAGManager] Added document to {library_name}: {doc_id}")
        
        return doc_id
    
    def search(self, library_name: str, query: str, top_k: int = 5,
              filters: Optional[Dict] = None) -> List[RAGDocument]:
        """从指定库搜索文档
        
        Args:
            library_name: 库名称
            query: 查询文本
            top_k: 返回数量
            filters: 过滤器
            
        Returns:
            List[RAGDocument]: 相关文档列表
        """
        if library_name not in self.rag_libraries:
            logger.error(f"[RAGManager] Unknown library: {library_name}")
            return []
        
        library = self.rag_libraries[library_name]
        results = library.search_documents(query, top_k, filters)
        
        self.total_searches += 1
        logger.debug(f"[RAGManager] Searched {library_name}: found {len(results)} results")
        
        return results
    
    def search_all(self, query: str, top_k: int = 5,
                  filters: Optional[Dict] = None) -> Dict[str, List[RAGDocument]]:
        """跨库搜索
        
        Args:
            query: 查询文本
            top_k: 每个库的返回数量
            filters: 过滤器
            
        Returns:
            Dict[str, List[RAGDocument]]: {库名：结果列表}
        """
        results = {}
        
        for library_name, library in self.rag_libraries.items():
            lib_results = library.search_documents(query, top_k, filters)
            results[library_name] = lib_results
        
        logger.info(f"[RAGManager] Cross-library search: {len(results)} libraries")
        
        return results
    
    def add_experience(self, content: str, category: str,
                      importance: str = "must_learn",
                      domain: str = "general") -> str:
        """便捷方法：添加经验
        
        Args:
            content: 经验内容
            category: 类别
            importance: 重要性
            domain: 领域
            
        Returns:
            str: 文档 ID
        """
        if "experience" not in self.rag_libraries:
            logger.error("[RAGManager] Experience RAG not enabled")
            return ""
        
        library = self.rag_libraries["experience"]
        return library.add_experience(content, category, importance, domain)
    
    def add_memory(self, content: str, memory_type: str,
                  time_span: str = "short_term",
                  memorability: str = "pending") -> str:
        """便捷方法：添加记忆
        
        Args:
            content: 记忆内容
            memory_type: 类型
            time_span: 时间跨度
            memorability: 记忆重要性
            
        Returns:
            str: 文档 ID
        """
        if "memory" not in self.rag_libraries:
            logger.error("[RAGManager] Memory RAG not enabled")
            return ""
        
        library = self.rag_libraries["memory"]
        return library.add_memory(content, memory_type, time_span, memorability)
    
    def add_knowledge(self, content: str, domain: str,
                     certainty: str = "confirmed") -> str:
        """便捷方法：添加知识
        
        Args:
            content: 知识内容
            domain: 领域
            certainty: 确定性
            
        Returns:
            str: 文档 ID
        """
        if "knowledge" not in self.rag_libraries:
            logger.error("[RAGManager] Knowledge RAG not enabled")
            return ""
        
        library = self.rag_libraries["knowledge"]
        return library.add_knowledge(content, domain, certainty)
    
    def consolidate_memories(self) -> int:
        """记忆巩固"""
        if "memory" not in self.rag_libraries:
            return 0
        
        library = self.rag_libraries["memory"]
        return library.consolidate_memories()
    
    def save_all(self, base_path: Optional[str] = None) -> bool:
        """保存所有 RAG 库
        
        Args:
            base_path: 保存路径（可选）
            
        Returns:
            bool: 是否成功
        """
        save_path = base_path or self.config.base_path
        
        # 创建目录
        os.makedirs(save_path, exist_ok=True)
        
        success_count = 0
        for library_name, library in self.rag_libraries.items():
            path = os.path.join(save_path, library_name)
            if library.save(path):
                success_count += 1
                logger.info(f"[RAGManager] Saved {library_name} to {path}")
            else:
                logger.error(f"[RAGManager] Failed to save {library_name}")
        
        logger.info(f"[RAGManager] Saved {success_count}/{len(self.rag_libraries)} libraries")
        return success_count == len(self.rag_libraries)
    
    def load_all(self, base_path: Optional[str] = None) -> bool:
        """加载所有 RAG 库
        
        Args:
            base_path: 加载路径（可选）
            
        Returns:
            bool: 是否成功
        """
        load_path = base_path or self.config.base_path
        
        success_count = 0
        for library_name, library in self.rag_libraries.items():
            path = os.path.join(load_path, library_name)
            if library.load(path):
                success_count += 1
                logger.info(f"[RAGManager] Loaded {library_name} from {path}")
            else:
                logger.warning(f"[RAGManager] No data found for {library_name}")
        
        logger.info(f"[RAGManager] Loaded {success_count}/{len(self.rag_libraries)} libraries")
        return success_count > 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "total_adds": self.total_adds,
            "total_searches": self.total_searches,
            "libraries": {}
        }
        
        for library_name, library in self.rag_libraries.items():
            stats["libraries"][library_name] = library.get_statistics()
        
        return stats
    
    def print_status(self):
        """打印状态信息"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("RAG 管理器状态")
        print("=" * 60)
        print(f"总添加数：{stats['total_adds']}")
        print(f"总搜索数：{stats['total_searches']}")
        print(f"活跃库数：{len(stats['libraries'])}")
        
        for lib_name, lib_stats in stats["libraries"].items():
            print(f"\n  {lib_name.upper()}:")
            print(f"    文档数：{lib_stats['total_documents']}")
            print(f"    添加次数：{lib_stats['total_adds']}")
            print(f"    搜索次数：{lib_stats['total_searches']}")
            
            # 打印特定统计
            if "category_counts" in lib_stats:
                print(f"    分类统计：{lib_stats['category_counts']}")
            if "time_span_counts" in lib_stats:
                print(f"    时间跨度：{lib_stats['time_span_counts']}")
            if "domain_counts" in lib_stats:
                print(f"    领域统计：{lib_stats['domain_counts']}")
        
        print("=" * 60 + "\n")
