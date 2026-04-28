# File: zulong/tools/search_tools.py
# search_tools 元工具 - LLM 自主检索工具索引
#
# 这是一个"工具的工具"（元工具）：
# 当 LLM 发现当前 prompt 中的工具不够用时，
# 它可以调用 search_tools 去 ToolRAG 中按语义检索更多工具，
# 检索到的工具 schema 会被动态注入到下一轮迭代的 tools 列表中。

import logging
import json
from typing import Dict, Any

from .base import BaseTool, ToolCategory, ToolRequest, ToolResult

logger = logging.getLogger(__name__)


class SearchToolsTool(BaseTool):
    """search_tools 元工具
    
    LLM 通过调用此工具来检索 ToolRAG 中的工具索引。
    返回的工具 schema 会被 inference_engine 动态注入到后续迭代。
    
    这是一个"热工具"（始终在 prompt 中），用于按需发现"冷工具"。
    """
    
    def __init__(self, tool_rag=None):
        super().__init__(name="search_tools", category=ToolCategory.CUSTOM)
        self.description = (
            "在工具索引中搜索可用的工具。当你需要完成某项任务但当前可用工具不够时，"
            "调用此工具描述你的需求，系统会返回匹配的工具列表及其使用方法。"
            "返回的工具将被自动加载到你的可用工具列表中。"
        )
        self._tool_rag = tool_rag
    
    def set_tool_rag(self, tool_rag):
        """设置 ToolRAG 引用（延迟注入，避免循环依赖）"""
        self._tool_rag = tool_rag
    
    def initialize(self) -> bool:
        return True
    
    def cleanup(self) -> None:
        pass
    
    def execute(self, request: ToolRequest) -> ToolResult:
        """执行工具搜索
        
        Args:
            request.parameters:
                - query: 需求描述（如"我需要拆解任务"、"帮我读取文件"）
                - top_k: 返回数量（默认 3）
        """
        import time
        start_time = time.time()
        
        query = request.parameters.get("query", "")
        top_k = request.parameters.get("top_k", 3)
        
        if not query:
            return self._create_result(
                success=False,
                error="缺少搜索查询 (query)",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )
        
        if self._tool_rag is None:
            return self._create_result(
                success=False,
                error="ToolRAG 未初始化，工具索引不可用",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )
        
        # 在 ToolRAG 中检索
        results = self._tool_rag.search_tools(query, top_k=top_k)
        
        if not results:
            return self._create_result(
                success=True,
                data={
                    "message": f"未找到与 '{query}' 匹配的工具",
                    "tools_found": 0,
                    "tools": []
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )
        
        # 构建返回结果
        tool_summaries = []
        discovered_schemas = []
        
        for r in results:
            tool_summaries.append({
                "tool_name": r["tool_name"],
                "description": r.get("description", ""),
                "similarity": round(r.get("similarity", 0.0), 3),
                "source": r.get("source", "unknown"),
            })
            # 收集完整 schema 以便 inference_engine 动态注入
            if r.get("function_schema"):
                discovered_schemas.append(r["function_schema"])
        
        logger.info(
            f"[SearchToolsTool] query='{query[:40]}' -> "
            f"found {len(tool_summaries)} tools: "
            f"{[t['tool_name'] for t in tool_summaries]}"
        )
        
        return self._create_result(
            success=True,
            data={
                "message": f"找到 {len(tool_summaries)} 个相关工具",
                "tools_found": len(tool_summaries),
                "tools": tool_summaries,
                # 这个字段被 inference_engine 读取，用于动态注入
                "_discovered_schemas": discovered_schemas,
            },
            execution_time=time.time() - start_time,
            request_id=request.request_id,
        )
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "描述你需要什么类型的工具（如'拆解复杂任务'、'读取文件'、'深度推理'）"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的最大工具数量，默认 3"
                }
            },
            "required": ["query"]
        }
