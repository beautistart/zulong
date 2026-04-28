# File: zulong/l2/expert_invoker.py
# L2 专家调用器（增强版）- 集成 RAG 和工具系统

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import time

from ..memory import RAGManager, TaggingEngine
# MemoryEvolutionEngine 已重命名为 MemoryConsolidator
from ..memory.memory_evolution import MemoryConsolidator as MemoryEvolutionEngine
from ..tools import ToolEngine, BaseTool, ToolResult
from ..l3 import DualBrainContainer

logger = logging.getLogger(__name__)


@dataclass
class ExpertCallResult:
    """专家调用结果
    
    TSD v1.7 对应规则:
    - 统一返回格式
    - 包含 RAG 检索结果
    - 包含工具执行结果
    - 支持多轮对话上下文
    """
    success: bool
    response: str
    rag_results: Dict[str, Any] = field(default_factory=dict)
    tool_results: List[ToolResult] = field(default_factory=list)
    context_updated: bool = False
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "response": self.response,
            "rag_results": self.rag_results,
            "tool_results": [r.to_dict() for r in self.tool_results],
            "context_updated": self.context_updated,
            "execution_time": self.execution_time,
            "metadata": self.metadata
        }


class ExpertInvoker:
    """L2 专家调用器（增强版）
    
    TSD v1.7 对应规则:
    - 2.3.2 专家模型层：专家调用器
    - 集成 RAG 检索
    - 集成工具调用
    - 支持多轮对话
    - 支持上下文管理
    
    功能:
    - 智能路由（判断是否需要 RAG/工具）
    - RAG 检索增强
    - 工具调用编排
    - 上下文管理
    - 结果聚合
    """
    
    def __init__(
        self,
        rag_manager: Optional[RAGManager] = None,
        tool_engine: Optional[ToolEngine] = None,
        brain_container: Optional[DualBrainContainer] = None,
        auto_init: bool = True
    ):
        """初始化专家调用器
        
        Args:
            rag_manager: RAG 管理器
            tool_engine: 工具引擎
            brain_container: 双脑容器
            auto_init: 是否自动初始化
        """
        self.rag_manager = rag_manager or RAGManager()
        self.tool_engine = tool_engine or ToolEngine()
        self.brain_container = brain_container
        
        # 打标引擎（用于智能路由）
        self.tagging_engine = TaggingEngine()
        
        # 记忆进化引擎
        self.memory_evolution = MemoryEvolutionEngine(self.rag_manager)
        
        # 上下文管理
        self.context: Dict[str, Any] = {
            "conversation_history": [],
            "current_task": None,
            "active_tools": [],
            "rag_queries": []
        }
        
        # 统计信息
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        
        if auto_init:
            self.initialize()
        
        logger.info("[ExpertInvoker] Initialized (enhanced with RAG & Tools)")
    
    def initialize(self) -> bool:
        """初始化
        
        Returns:
            bool: 是否成功
        """
        try:
            # RAG 已经在构造函数中初始化
            # 初始化工具
            self._register_default_tools()
            
            logger.info("[ExpertInvoker] Initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"[ExpertInvoker] Initialization error: {e}")
            return False
    
    def _register_default_tools(self) -> None:
        """注册默认工具"""
        # 从工具引擎获取已注册的工具
        # 如果需要，可以在这里注册额外的专家专用工具
        pass
    
    def invoke(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        use_rag: bool = True,
        use_tools: bool = True,
        max_rag_results: int = 3,
        timeout: float = 60.0
    ) -> ExpertCallResult:
        """调用专家（增强版）
        
        Args:
            query: 用户查询
            context: 上下文信息
            use_rag: 是否使用 RAG
            use_tools: 是否使用工具
            max_rag_results: 最大 RAG 结果数
            timeout: 超时时间
            
        Returns:
            ExpertCallResult: 调用结果
        """
        start_time = time.time()
        
        logger.info(f"[ExpertInvoker] Invoking expert with query: {query[:50]}...")
        
        try:
            # 1. 智能路由决策
            routing_decision = self._route_query(query, context)
            
            # 2. RAG 检索（如果需要）
            rag_results = {}
            if use_rag and routing_decision.get("need_rag", False):
                rag_results = self._retrieve_from_rag(
                    query,
                    max_results=max_rag_results,
                    routing_decision=routing_decision
                )
            
            # 3. 工具调用（如果需要）
            tool_results = []
            if use_tools and routing_decision.get("need_tools", False):
                tool_results = self._invoke_tools(
                    query,
                    context,
                    routing_decision=routing_decision,
                    timeout=timeout
                )
            
            # 4. 生成响应（调用 L2 专家模型）
            response = self._generate_response(
                query,
                context,
                rag_results,
                tool_results,
                routing_decision
            )
            
            # 5. 更新上下文
            self._update_context(query, response, rag_results, tool_results)
            
            # 6. 记忆进化（追踪访问）
            if rag_results:
                for doc_id in rag_results.get("retrieved_ids", []):
                    self.memory_evolution.track_access(doc_id)
            
            # 构建结果
            execution_time = time.time() - start_time
            
            result = ExpertCallResult(
                success=True,
                response=response,
                rag_results=rag_results,
                tool_results=tool_results,
                context_updated=True,
                execution_time=execution_time,
                metadata={
                    "routing_decision": routing_decision,
                    "rag_enabled": use_rag,
                    "tools_enabled": use_tools
                }
            )
            
            # 更新统计
            self.total_calls += 1
            self.successful_calls += 1
            
            logger.info(f"[ExpertInvoker] Completed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"[ExpertInvoker] Invoke error: {e}")
            
            self.total_calls += 1
            self.failed_calls += 1
            
            return ExpertCallResult(
                success=False,
                response=f"Error: {str(e)}",
                execution_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def _route_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """智能路由决策
        
        Args:
            query: 用户查询
            context: 上下文
            
        Returns:
            Dict: 路由决策
        """
        # 使用打标引擎分析查询
        tagging_result = self.tagging_engine.tag_content(query)
        
        # 决策逻辑
        decision = {
            "need_rag": False,
            "need_tools": False,
            "target_rag": tagging_result.target_rag,
            "domain": tagging_result.domain.value if hasattr(tagging_result.domain, 'value') else str(tagging_result.domain),
            "urgency": tagging_result.urgency if isinstance(tagging_result.urgency, str) else str(tagging_result.urgency),
            "confidence": tagging_result.confidence
        }
        
        # 需要 RAG 的情况
        if tagging_result.target_rag in ["experience", "knowledge", "memory"]:
            decision["need_rag"] = True
        
        # 需要工具的情况
        domain_value = tagging_result.domain.value if hasattr(tagging_result.domain, 'value') else str(tagging_result.domain)
        if domain_value in ["manipulation", "navigation"]:
            decision["need_tools"] = True
        
        # 检查是否包含工具调用关键词
        tool_keywords = ["打开", "创建", "删除", "搜索", "运行", "执行"]
        if any(kw in query for kw in tool_keywords):
            decision["need_tools"] = True
        
        logger.debug(f"[ExpertInvoker] Routing decision: {decision}")
        
        return decision
    
    def _retrieve_from_rag(
        self,
        query: str,
        max_results: int = 3,
        routing_decision: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """从 RAG 检索
        
        Args:
            query: 查询
            max_results: 最大结果数
            routing_decision: 路由决策
            
        Returns:
            Dict: 检索结果
        """
        logger.info(f"[ExpertInvoker] Retrieving from RAG (max={max_results})")
        
        try:
            # 根据路由决策选择 RAG 库
            target_rag = routing_decision.get("target_rag", "knowledge") if routing_decision else "knowledge"
            
            results = {
                "query": query,
                "target_rag": target_rag,
                "documents": [],
                "retrieved_ids": []
            }
            
            # 跨库搜索
            search_results = self.rag_manager.search_all(
                query=query,
                top_k=max_results
            )
            
            # search_all 返回 Dict[str, List[RAGDocument]]
            for library_name, docs in search_results.items():
                for doc in docs:
                    # doc 是 RAGDocument 对象
                    results["documents"].append({
                        "rag": library_name,
                        "content": doc.content[:200] if hasattr(doc, 'content') else str(doc),
                        "similarity": getattr(doc, 'similarity', 0.0),
                        "metadata": getattr(doc, 'metadata', {})
                    })
                    results["retrieved_ids"].append(getattr(doc, 'doc_id', ''))
            
            logger.info(f"[ExpertInvoker] Retrieved {len(results['documents'])} documents")
            
            return results
            
        except Exception as e:
            logger.error(f"[ExpertInvoker] RAG retrieval error: {e}")
            return {
                "query": query,
                "error": str(e),
                "documents": [],
                "retrieved_ids": []
            }
    
    def _invoke_tools(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        routing_decision: Optional[Dict] = None,
        timeout: float = 30.0
    ) -> List[ToolResult]:
        """调用工具
        
        Args:
            query: 查询
            context: 上下文
            routing_decision: 路由决策
            timeout: 超时
            
        Returns:
            List[ToolResult]: 工具结果
        """
        logger.info(f"[ExpertInvoker] Invoking tools")
        
        tool_results = []
        
        try:
            # 根据路由决策选择工具
            domain = routing_decision.get("domain", "general") if routing_decision else "general"
            
            # 简单示例：根据领域选择工具
            if domain == "manipulation":
                # 操作相关 - 可能需要文件工具
                result = self.tool_engine.call_tool(
                    tool_name="file_tool",
                    action="list_dir",
                    parameters={"path": "."},
                    timeout=timeout
                )
                tool_results.append(result)
            
            elif domain == "navigation":
                # 导航相关 - 可能需要系统命令
                result = self.tool_engine.call_tool(
                    tool_name="system_command_tool",
                    action="get_system_info",
                    parameters={},
                    timeout=timeout
                )
                tool_results.append(result)
            
            # 检查是否包含 VSCode 相关关键词
            if any(kw in query.lower() for kw in ["vscode", "code", "editor", "文件"]):
                result = self.tool_engine.call_tool(
                    tool_name="vscode_tool",
                    action="get_info",
                    parameters={},
                    timeout=timeout
                )
                tool_results.append(result)
            
            logger.info(f"[ExpertInvoker] Invoked {len(tool_results)} tools")
            
        except Exception as e:
            logger.error(f"[ExpertInvoker] Tool invocation error: {e}")
        
        return tool_results
    
    def _generate_response(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        rag_results: Dict[str, Any],
        tool_results: List[ToolResult],
        routing_decision: Dict[str, Any]
    ) -> str:
        """生成响应
        
        Args:
            query: 查询
            context: 上下文
            rag_results: RAG 结果
            tool_results: 工具结果
            routing_decision: 路由决策
            
        Returns:
            str: 响应
        """
        # 构建增强的提示词
        prompt_parts = []
        
        # 1. RAG 增强信息
        if rag_results.get("documents"):
            prompt_parts.append("📚 检索到的相关信息:")
            for i, doc in enumerate(rag_results["documents"], 1):
                prompt_parts.append(f"{i}. [{doc['rag']}] {doc['content'][:100]}...")
        
        # 2. 工具执行结果
        if tool_results:
            prompt_parts.append("\n🛠️ 工具执行结果:")
            for i, result in enumerate(tool_results, 1):
                if result.success:
                    prompt_parts.append(f"{i}. ✅ {result.data}")
                else:
                    prompt_parts.append(f"{i}. ❌ {result.error}")
        
        # 3. 基础响应（如果没有 L2 模型，使用规则生成）
        if not prompt_parts:
            response = f"收到查询：{query}"
        else:
            response = "\n".join(prompt_parts)
            response += f"\n\n✅ 已处理您的查询：{query[:50]}..."
        
        return response
    
    def _update_context(
        self,
        query: str,
        response: str,
        rag_results: Dict[str, Any],
        tool_results: List[ToolResult]
    ) -> None:
        """更新上下文
        
        Args:
            query: 查询
            response: 响应
            rag_results: RAG 结果
            tool_results: 工具结果
        """
        # 添加到对话历史
        self.context["conversation_history"].append({
            "role": "user",
            "content": query,
            "timestamp": time.time()
        })
        
        self.context["conversation_history"].append({
            "role": "assistant",
            "content": response,
            "timestamp": time.time(),
            "rag_used": bool(rag_results),
            "tools_used": len(tool_results)
        })
        
        # 限制历史长度
        if len(self.context["conversation_history"]) > 100:
            self.context["conversation_history"] = self.context["conversation_history"][-100:]
        
        logger.debug(f"[ExpertInvoker] Context updated (history_size={len(self.context['conversation_history'])})")
    
    def get_context(self) -> Dict[str, Any]:
        """获取当前上下文"""
        return self.context.copy()
    
    def clear_context(self) -> None:
        """清空上下文"""
        self.context = {
            "conversation_history": [],
            "current_task": None,
            "active_tools": [],
            "rag_queries": []
        }
        logger.info("[ExpertInvoker] Context cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / max(self.total_calls, 1),
            "context_size": len(self.context["conversation_history"]),
            "rag_manager": self.rag_manager.get_statistics() if self.rag_manager else {},
            "tool_engine": self.tool_engine.get_statistics() if self.tool_engine else {}
        }
    
    def print_status(self):
        """打印状态信息"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("L2 专家调用器（增强版）状态")
        print("=" * 60)
        print(f"总调用数：{stats['total_calls']}")
        print(f"成功调用：{stats['successful_calls']}")
        print(f"失败调用：{stats['failed_calls']}")
        print(f"成功率：{stats['success_rate']:.2%}")
        print(f"上下文大小：{stats['context_size']} 条")
        
        if stats.get('rag_manager'):
            print(f"\n📚 RAG 系统:")
            print(f"  文档总数：{stats['rag_manager'].get('total_documents', 0)}")
            print(f"  经验库：{stats['rag_manager'].get('experience_count', 0)}")
            print(f"  记忆库：{stats['rag_manager'].get('memory_count', 0)}")
            print(f"  知识库：{stats['rag_manager'].get('knowledge_count', 0)}")
        
        if stats.get('tool_engine'):
            print(f"\n🛠️ 工具系统:")
            print(f"  注册工具：{stats['tool_engine'].get('tools_registered', 0)}")
            print(f"  总调用：{stats['tool_engine'].get('total_calls', 0)}")
            print(f"  成功率：{stats['tool_engine'].get('success_rate', 0):.2%}")
        
        print("=" * 60 + "\n")
