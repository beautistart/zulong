# File: zulong/memory/tool_rag.py
# Tool RAG 库 - 工具索引与检索
#
# 将注册的工具摘要向量化存入 FAISS，支持按用户意图语义检索工具。
# 设计目的：当工具数量增长到超上下文窗口时，仅将"热工具"放入 prompt，
# 其余"冷工具"通过 search_tools 元工具按需检索并动态注入。

import logging
import time
from typing import Dict, Any, List, Optional
import numpy as np

from .base_rag_library import BaseRAGLibrary, RAGDocument
from ..models.embedding_model import embedding_model

logger = logging.getLogger(__name__)


class ToolRAG(BaseRAGLibrary):
    """工具 RAG 库
    
    存储工具摘要（名称 + 描述 + 参数概要），支持语义检索。
    与 ExperienceRAG/MemoryRAG/KnowledgeRAG 并列，由 RAGManager 统一管理。
    
    每条文档的 content 格式:
        "工具名: <name>\n描述: <description>\n参数: <param1>, <param2>, ..."
    metadata 包含:
        - tool_name: 工具注册名
        - source: 来源（builtin / skill_pack_id）
        - function_schema: 完整的 OpenAI Function Calling schema（用于动态注入）
    """
    
    def __init__(self, **kwargs):
        kwargs.pop('dimension', None)
        # 与其他 RAG 库一致：BAAI/bge-small-zh-v1.5 输出 512 维
        super().__init__(name="tool_rag", dimension=512, **kwargs)
        
        # tool_name -> doc_id 的快速索引
        self._tool_index: Dict[str, str] = {}
        
        logger.info("[ToolRAG] Initialized")
    
    # ==================== 核心 CRUD ====================
    
    def add_document(self, document: RAGDocument) -> str:
        """添加工具摘要文档"""
        tool_name = document.metadata.get("tool_name", "unknown")
        doc_id = f"tool_{tool_name}_{int(document.created_at)}"
        
        # 如果同名工具已存在，先删除旧版（保证唯一性）
        if tool_name in self._tool_index:
            old_doc_id = self._tool_index[tool_name]
            self._remove_document(old_doc_id)
        
        # 存储文档
        self.documents[doc_id] = document
        self._tool_index[tool_name] = doc_id
        
        # 生成向量并写入 FAISS
        if document.embedding is None:
            document.embedding = self._encode(document.content)
        
        if document.embedding is not None:
            self.vector_store.add_vectors_with_ids(
                document.embedding,
                metadata=[document.to_dict()],
                vector_ids=[doc_id]
            )
        
        self.total_adds += 1
        logger.debug(f"[ToolRAG] Added tool: {tool_name} -> {doc_id}")
        return doc_id
    
    def search_documents(self, query: str, top_k: int = 5,
                         filters: Optional[Dict] = None) -> List[RAGDocument]:
        """根据用户意图语义检索最相关的工具"""
        try:
            query_vector = self._encode(query)
            if query_vector is None:
                return []
            
            indices, distances = self.vector_store.search(query_vector, top_k=top_k)
            similarities = [1.0 / (1.0 + dist) for dist in distances]
            
            results = []
            for idx, sim in zip(indices, similarities):
                doc_id = self.vector_store.reverse_id_map.get(idx)
                if doc_id and doc_id in self.documents:
                    doc = self.documents[doc_id]
                    doc.metadata["similarity"] = sim
                    results.append(doc)
            
            self.total_searches += 1
            logger.info(f"[ToolRAG] Search '{query[:40]}...' -> {len(results)} tools")
            return results
        except Exception as e:
            logger.error(f"[ToolRAG] Search error: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "name": self.name,
            "total_documents": len(self.documents),
            "total_adds": self.total_adds,
            "total_searches": self.total_searches,
            "indexed_tools": list(self._tool_index.keys()),
            "vector_store": self.vector_store.get_stats()
        }
    
    # ==================== 工具专用便捷方法 ====================
    
    def add_tool(self, tool_name: str, description: str,
                 param_names: List[str], source: str = "builtin",
                 function_schema: Optional[Dict] = None) -> str:
        """便捷方法：添加工具摘要
        
        Args:
            tool_name: 工具注册名（如 "task_decompose"）
            description: 工具描述
            param_names: 参数名列表（如 ["goal", "context"]）
            source: 来源（"builtin" 或技能包 ID）
            function_schema: 完整的 OpenAI Function Calling schema（动态注入时用）
        
        Returns:
            str: 文档 ID
        """
        # 构建检索友好的文本（名称 + 描述 + 参数）
        content = (
            f"工具名: {tool_name}\n"
            f"描述: {description}\n"
            f"参数: {', '.join(param_names) if param_names else '无'}"
        )
        
        doc = RAGDocument(
            content=content,
            metadata={
                "tool_name": tool_name,
                "source": source,
                "function_schema": function_schema or {},
                "param_names": param_names,
            },
            importance="must_learn",
            domain="tool",
        )
        return self.add_document(doc)
    
    def remove_tool(self, tool_name: str) -> bool:
        """移除工具摘要
        
        Args:
            tool_name: 工具注册名
        
        Returns:
            bool: 是否成功
        """
        if tool_name not in self._tool_index:
            logger.warning(f"[ToolRAG] Tool not found: {tool_name}")
            return False
        
        doc_id = self._tool_index[tool_name]
        self._remove_document(doc_id)
        del self._tool_index[tool_name]
        logger.info(f"[ToolRAG] Removed tool: {tool_name}")
        return True
    
    def search_tools(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """按意图检索工具，返回结构化结果
        
        Args:
            query: 用户意图描述（如"我想分析代码"、"帮我搜索网页"）
            top_k: 返回数量
        
        Returns:
            List[Dict]: [{tool_name, description, similarity, function_schema}, ...]
        """
        docs = self.search_documents(query, top_k=top_k)
        
        results = []
        for doc in docs:
            results.append({
                "tool_name": doc.metadata.get("tool_name", ""),
                "description": doc.metadata.get("function_schema", {}).get("function", {}).get("description", doc.content),
                "similarity": doc.metadata.get("similarity", 0.0),
                "function_schema": doc.metadata.get("function_schema", {}),
                "source": doc.metadata.get("source", "unknown"),
            })
        
        return results
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """获取指定工具的完整 Function Calling schema
        
        Args:
            tool_name: 工具注册名
        
        Returns:
            完整的 OpenAI Function Calling schema，或 None
        """
        if tool_name not in self._tool_index:
            return None
        
        doc_id = self._tool_index[tool_name]
        doc = self.documents.get(doc_id)
        if doc:
            return doc.metadata.get("function_schema")
        return None
    
    def has_tool(self, tool_name: str) -> bool:
        """检查工具是否已索引"""
        return tool_name in self._tool_index
    
    def list_all_tools(self) -> List[str]:
        """列出所有已索引的工具名"""
        return list(self._tool_index.keys())
    
    # ==================== 内部方法 ====================
    
    def _encode(self, text: str) -> Optional[np.ndarray]:
        """将文本编码为向量"""
        try:
            if embedding_model.model is None:
                embedding_model.load()
            return embedding_model.encode_query(text)
        except Exception as e:
            logger.error(f"[ToolRAG] Encoding error: {e}")
            return None
    
    def _remove_document(self, doc_id: str):
        """内部方法：从存储和索引中移除文档"""
        if doc_id in self.documents:
            del self.documents[doc_id]
        self.vector_store.delete_vectors([doc_id])
