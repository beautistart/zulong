# File: zulong/l2/rag_node.py
# L2 RAG 集成节点 - LangGraph 原生节点

import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
import time
import operator

from ..memory import RAGManager, TaggingEngine
from ..memory.enhanced_experience_store import BM25Search
from ..core.types import ZulongEvent, EventType

logger = logging.getLogger(__name__)


@dataclass
class RAGRetrievalResult:
    """RAG 检索结果
    
    TSD v1.7 对应规则:
    - 结构化检索结果
    - 包含相似度评分
    - 支持多库检索
    """
    query: str
    documents: List[Dict[str, Any]]
    target_rag: str
    total_results: int
    search_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "query": self.query,
            "documents": self.documents,
            "target_rag": self.target_rag,
            "total_results": self.total_results,
            "search_time": self.search_time,
            "metadata": self.metadata
        }


class RAGNodeState(TypedDict):
    """RAG 节点状态（LangGraph 强类型状态）
    
    TSD v1.7 对应规则:
    - 使用 TypedDict 定义状态
    - 支持状态累积
    - 类型安全
    """
    query: str  # 当前查询
    context: Dict[str, Any]  # 上下文
    rag_results: List[Dict[str, Any]]  # RAG 结果（累积）
    retrieved_docs: Annotated[List[Dict], operator.add]  # 检索到的文档（累加）
    target_rag: str  # 目标 RAG 库
    search_metadata: Dict[str, Any]  # 搜索元数据
    messages: List[Dict[str, Any]]  # 消息历史


class RAGIntegrationNode:
    """RAG 集成节点（LangGraph 原生）
    
    TSD v1.7 对应规则:
    - 2.2.5 基础设施层：RAG 记忆系统
    - LangGraph StateGraph 节点
    - 支持中断和恢复
    
    功能:
    - RAG 检索
    - 结果排序
    - 上下文增强
    - 记忆追踪
    """
    
    def __init__(
        self,
        rag_manager: Optional[RAGManager] = None,
        tagging_engine: Optional[TaggingEngine] = None,
        max_results: int = 5,
        min_similarity: float = 0.3,  # 🔥 关键修复：降低阈值（从 0.6 到 0.3），确保相关文档不被过滤
        use_shared_rag: bool = True,  # 新增：是否使用共享的 RAG 管理器
        enable_hybrid_search: bool = True,  # ✅ 第 3 周：启用混合检索
        bm25_weight: float = 0.4,  # BM25 权重（默认 0.4，向量检索权重 0.6）
        top_k_bm25: int = 10  # BM25 初检数量
    ):
        """初始化 RAG 节点
        
        Args:
            rag_manager: RAG 管理器
            tagging_engine: 打标引擎
            max_results: 最大结果数
            min_similarity: 最小相似度阈值（🔥 关键修复：降低到 0.3，避免过滤掉相关记忆）
            use_shared_rag: 是否使用共享的 RAG 管理器（从 InferenceEngine）
            enable_hybrid_search: ✅ 是否启用混合检索（BM25 + 向量）
            bm25_weight: BM25 权重（默认 0.4，向量检索权重 0.6）
            top_k_bm25: BM25 初检数量
        """
        # 如果使用共享 RAG，稍后通过 set_rag_manager 设置
        if not use_shared_rag:
            self.rag_manager = rag_manager or RAGManager()
        else:
            self.rag_manager = None  # 稍后设置
        
        self.tagging_engine = tagging_engine or TaggingEngine()
        
        self.max_results = max_results
        self.min_similarity = min_similarity
        
        # ✅ 第 3 周：混合检索配置
        self.enable_hybrid_search = enable_hybrid_search
        self.bm25_weight = bm25_weight
        self.vector_weight = 1.0 - bm25_weight
        self.top_k_bm25 = top_k_bm25
        
        # 初始化 BM25 搜索引擎（从经验库同步文档）
        self.bm25_search = BM25Search() if enable_hybrid_search else None
        
        # 统计信息
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.total_search_time = 0.0
        
        logger.info(f"[RAGIntegrationNode] Initialized (max_results={max_results}, hybrid_search={enable_hybrid_search})")
    
    def set_rag_manager(self, rag_manager: RAGManager):
        """设置共享的 RAG 管理器（由 InferenceEngine 注入）
        
        Args:
            rag_manager: RAG 管理器实例
        """
        self.rag_manager = rag_manager
        logger.info("[RAGIntegrationNode] Using shared RAG manager")
        
        # ✅ 第 3 周：同步 BM25 索引
        if self.enable_hybrid_search and self.bm25_search:
            self._sync_bm25_index()
    
    def _sync_bm25_index(self):
        """同步 BM25 索引（从经验库）"""
        try:
            if not self.rag_manager:
                return
            
            # 从经验库同步文档到 BM25
            experience_rag = self.rag_manager.rag_libraries.get('experience_rag')
            if experience_rag and hasattr(experience_rag, 'index'):
                logger.info("📚 [RAGNode] 同步 BM25 索引...")
                
                # 遍历经验库的所有文档
                for doc_id, doc in experience_rag.index.items():
                    content = doc.get('content', '') if isinstance(doc, dict) else str(doc)
                    self.bm25_search.add_document(doc_id, content)
                
                logger.info(f"✅ [RAGNode] BM25 索引同步完成：{len(self.bm25_search.documents)} 个文档")
        except Exception as e:
            logger.error(f"[RAGNode] BM25 索引同步失败：{e}", exc_info=True)
    
    def _hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """混合检索（BM25 + 向量检索）
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            List[Dict]: 混合检索结果
        """
        logger.info(f"🔍 [RAGNode] 执行混合检索：query={query[:50]}...")
        
        # 1. BM25 关键词检索
        bm25_results = {}
        if self.bm25_search:
            bm25_scores = self.bm25_search.search(query, top_k=self.top_k_bm25)
            bm25_results = dict(bm25_scores)
            logger.debug(f"[RAGNode] BM25 检索结果：{len(bm25_results)} 个文档")
        
        # 2. 向量检索
        vector_results = {}
        if self.rag_manager:
            search_results = self.rag_manager.search_all(query=query, top_k=top_k)
            for library_name, docs in search_results.items():
                for doc in docs:
                    doc_id = getattr(doc, 'doc_id', '')
                    similarity = getattr(doc, 'similarity', 0.0)
                    if doc_id:
                        vector_results[doc_id] = {
                            'similarity': similarity,
                            'doc': doc,
                            'library': library_name
                        }
            logger.debug(f"[RAGNode] 向量检索结果：{len(vector_results)} 个文档")
        
        # 3. 融合分数（加权求和）
        fused_scores: Dict[str, float] = {}
        all_doc_ids = set(bm25_results.keys()) | set(vector_results.keys())
        
        # 归一化 BM25 分数
        if bm25_results:
            max_bm25 = max(bm25_results.values())
            if max_bm25 > 0:
                bm25_results = {k: v / max_bm25 for k, v in bm25_results.items()}
        
        for doc_id in all_doc_ids:
            bm25_score = bm25_results.get(doc_id, 0.0)
            vector_score = vector_results.get(doc_id, {}).get('similarity', 0.0)
            
            # 加权融合
            fused_score = (self.bm25_weight * bm25_score) + (self.vector_weight * vector_score)
            fused_scores[doc_id] = fused_score
        
        # 4. 排序返回 Top-K
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 5. 构建返回结果
        final_results = []
        for doc_id, score in sorted_results[:top_k]:
            if doc_id in vector_results:
                doc_info = vector_results[doc_id]
                doc = doc_info['doc']
                final_results.append({
                    "rag": doc_info['library'],
                    "content": doc.content[:200] if hasattr(doc, 'content') else str(doc),
                    "similarity": score,  # 融合后的分数
                    "metadata": getattr(doc, 'metadata', {}),
                    "doc_id": doc_id,
                    "bm25_score": bm25_results.get(doc_id, 0.0),
                    "vector_score": vector_results.get(doc_id, {}).get('similarity', 0.0)
                })
        
        logger.info(f"✅ [RAGNode] 混合检索完成：返回{len(final_results)}个文档")
        return final_results
    
    def retrieve(self, state: RAGNodeState) -> RAGNodeState:
        """RAG 检索节点函数（LangGraph 节点）
        
        Args:
            state: 节点状态
            
        Returns:
            RAGNodeState: 更新后的状态
        """
        start_time = time.time()
        
        query = state.get("query", "")
        if not query:
            logger.warning("[RAGIntegrationNode] Empty query")
            return state
        
        logger.info(f"[RAGIntegrationNode] Retrieving for query: {query[:50]}...")
        
        try:
            # 1. 智能打标（确定目标 RAG 库）
            tagging_result = self.tagging_engine.tag_content(query)
            target_rag = tagging_result.target_rag
            
            logger.debug(f"[RAGIntegrationNode] Target RAG: {target_rag}")
            
            # ✅ 第 3 周：使用混合检索
            if self.enable_hybrid_search:
                documents = self._hybrid_search(query=query, top_k=self.max_results)
            else:
                # 原有向量检索逻辑
                if not self.rag_manager:
                    logger.warning("[RAGIntegrationNode] rag_manager 未初始化，跳过检索")
                    return state
                # 2. 跨库搜索
                search_results = self.rag_manager.search_all(
                    query=query,
                    top_k=self.max_results
                )
                
                # 3. 结果处理
                documents = []
                filtered_count = 0
                for library_name, docs in search_results.items():
                    for doc in docs:
                        doc_dict = {
                            "rag": library_name,
                            "content": doc.content[:200] if hasattr(doc, 'content') else str(doc),
                            "similarity": getattr(doc, 'similarity', 0.0),
                            "metadata": getattr(doc, 'metadata', {}),
                            "doc_id": getattr(doc, 'doc_id', '')
                        }
                        
                        # 🔥 调试日志：打印所有检索结果
                        logger.debug(f"[RAGNode] {library_name}: similarity={doc_dict['similarity']:.3f}, content={doc_dict['content'][:50]}...")
                        
                        # 过滤低相似度结果
                        if doc_dict["similarity"] >= self.min_similarity:
                            documents.append(doc_dict)
                        else:
                            filtered_count += 1
                
                # 🔥 调试日志：打印过滤统计
                logger.info(f"[RAGNode] 检索统计：total={len(documents) + filtered_count}, retained={len(documents)}, filtered={filtered_count}")
            
            # 🔥 关键修复 3: 打印 RAG 文档内容
            if documents:
                logger.info(f"📚 [RAG 文档内容] 前 3 条结果:")
                for i, doc in enumerate(documents[:3]):
                    logger.info(f"  [{i+1}] {doc['rag']}: {doc['content'][:100]}... (相似度：{doc['similarity']:.3f})")
                    # ✅ 第 3 周：显示混合检索分数详情
                    if 'bm25_score' in doc and 'vector_score' in doc:
                        logger.info(f"      BM25: {doc['bm25_score']:.3f}, Vector: {doc['vector_score']:.3f}, Fused: {doc['similarity']:.3f}")
            else:
                logger.warning("⚠️ [RAG 文档内容] 未检索到任何文档")
            
            # 4. 按相似度排序
            documents.sort(key=lambda x: x["similarity"], reverse=True)
            
            # 5. 更新状态
            retrieval_result = RAGRetrievalResult(
                query=query,
                documents=documents,
                target_rag=target_rag,
                total_results=len(documents),
                search_time=time.time() - start_time
            )
            
            # 6. 更新统计
            self.total_queries += 1
            self.successful_queries += 1
            self.total_search_time += retrieval_result.search_time
            
            # 7. 返回更新后的状态
            updated_state = state.copy()
            updated_state["rag_results"] = [retrieval_result.to_dict()]
            updated_state["retrieved_docs"] = documents
            updated_state["target_rag"] = target_rag
            updated_state["search_metadata"] = {
                "search_time": retrieval_result.search_time,
                "total_found": len(documents),
                "min_similarity": self.min_similarity,
                "timestamp": time.time()
            }
            
            logger.info(f"[RAGIntegrationNode] Retrieved {len(documents)} documents in {retrieval_result.search_time:.3f}s")
            
            return updated_state
            
        except Exception as e:
            logger.error(f"[RAGIntegrationNode] Retrieval error: {e}")
            
            self.total_queries += 1
            self.failed_queries += 1
            
            # 返回原状态（不修改）
            return state
    
    def enhance_context(self, state: RAGNodeState) -> RAGNodeState:
        """上下文增强节点（使用 RAG 结果）
        
        Args:
            state: 节点状态
            
        Returns:
            RAGNodeState: 增强后的状态
        """
        logger.info("[RAGIntegrationNode] Enhancing context with RAG results")
        
        try:
            # 获取 RAG 结果
            rag_results = state.get("rag_results", [])
            retrieved_docs = state.get("retrieved_docs", [])
            
            if not retrieved_docs:
                logger.debug("[RAGIntegrationNode] No documents to enhance context")
                return state
            
            # 构建增强的上下文
            context_parts = []
            
            # 添加检索到的文档
            for i, doc in enumerate(retrieved_docs[:3], 1):  # 最多使用前 3 个
                context_parts.append(
                    f"[{doc['rag'].upper()}] {doc['content']} "
                    f"(相似度：{doc['similarity']:.2f})"
                )
            
            # 更新上下文
            updated_state = state.copy()
            existing_context = updated_state.get("context", {})
            
            existing_context["rag_enhancement"] = {
                "enabled": True,
                "documents_count": len(retrieved_docs),
                "context_parts": context_parts,
                "timestamp": time.time()
            }
            
            updated_state["context"] = existing_context
            
            logger.info(f"[RAGIntegrationNode] Context enhanced with {len(context_parts)} parts")
            
            return updated_state
            
        except Exception as e:
            logger.error(f"[RAGIntegrationNode] Context enhancement error: {e}")
            return state
    
    def format_response(self, state: RAGNodeState) -> RAGNodeState:
        """格式化响应节点（包含 RAG 引用）
        
        Args:
            state: 节点状态
            
        Returns:
            RAGNodeState: 包含格式化响应的状态
        """
        logger.info("[RAGIntegrationNode] Formatting response with RAG citations")
        
        try:
            retrieved_docs = state.get("retrieved_docs", [])
            query = state.get("query", "")
            
            # 构建响应
            response_parts = []
            
            # 添加 RAG 引用
            if retrieved_docs:
                response_parts.append("📚 参考信息:")
                for i, doc in enumerate(retrieved_docs[:3], 1):
                    response_parts.append(
                        f"{i}. [{doc['rag']}] {doc['content'][:100]}..."
                    )
            
            # 更新消息历史
            updated_state = state.copy()
            messages = updated_state.get("messages", [])
            
            # 添加助手消息
            if response_parts:
                messages.append({
                    "role": "assistant",
                    "content": "\n".join(response_parts),
                    "metadata": {
                        "rag_used": True,
                        "documents_count": len(retrieved_docs),
                        "timestamp": time.time()
                    }
                })
            
            updated_state["messages"] = messages
            
            logger.info(f"[RAGIntegrationNode] Response formatted with {len(response_parts)} parts")
            
            return updated_state
            
        except Exception as e:
            logger.error(f"[RAGIntegrationNode] Response formatting error: {e}")
            return state
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate": self.successful_queries / max(self.total_queries, 1),
            "avg_search_time": self.total_search_time / max(self.successful_queries, 1),
            "max_results": self.max_results,
            "min_similarity": self.min_similarity
        }
    
    def print_status(self):
        """打印状态信息"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("RAG 集成节点状态")
        print("=" * 60)
        print(f"总查询数：{stats['total_queries']}")
        print(f"成功查询：{stats['successful_queries']}")
        print(f"失败查询：{stats['failed_queries']}")
        print(f"成功率：{stats['success_rate']:.2%}")
        print(f"平均搜索时间：{stats['avg_search_time']:.3f}s")
        print(f"最大结果数：{stats['max_results']}")
        print(f"最小相似度：{stats['min_similarity']}")
        print("=" * 60 + "\n")
