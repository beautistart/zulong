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
                - top_k: 返回数量（默认 3，仅 search_tools 自身使用）
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


class RequestToolSupplementTool(BaseTool):
    """request_tool_supplement — L2 常驻工具补充机制。

    当 L1-B 预判注入的工具不够时，L2 调用此工具说明缺少的能力，
    服务端会从工具袋中匹配工具并把 schema 动态注入后续 FC 轮次。
    """

    def __init__(self, registry=None):
        super().__init__(name="request_tool_supplement", category=ToolCategory.CUSTOM)
        self.description = (
            "请求补充当前未注入但任务需要的工具。"
            "当你发现当前工具不足以完成任务时调用，说明缺什么能力、为什么需要、风险等级。"
            "系统会从工具袋匹配工具，并在后续工具循环中补充可用工具。"
        )
        self._registry = registry

    def set_registry(self, registry):
        self._registry = registry

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        import time
        start_time = time.time()
        if self._registry is None:
            return self._create_result(
                success=False,
                error="工具注册表未绑定，无法补充工具",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        params = request.parameters or {}
        try:
            from zulong.tools.tool_bag import supplement_tools
            data = supplement_tools(
                self._registry,
                missing_capability=params.get("missing_capability", ""),
                reason=params.get("reason", ""),
                suggested_tools=params.get("suggested_tools") or [],
                max_results=params.get("max_results"),
                list_all_tools=bool(params.get("list_all_tools", False)),
            )
            logger.info(
                "[RequestToolSupplement] missing=%r suggested=%s -> %s",
                params.get("missing_capability", ""),
                params.get("suggested_tools") or [],
                data.get("supplemented_tools", []),
            )
            return self._create_result(
                success=True,
                data=data,
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )
        except Exception as e:
            logger.warning("[RequestToolSupplement] 补充失败: %s", e)
            return self._create_result(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "missing_capability": {
                    "type": "string",
                    "description": "缺少的能力，例如“读取项目文件”、“运行测试”、“创建任务图”。首轮已常驻记忆检索和 web_search，通常不需要为这些能力请求补充。",
                },
                "reason": {
                    "type": "string",
                    "description": "为什么当前工具包不够，以及补充工具会如何帮助完成任务",
                },
                "suggested_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "希望补充的工具名，可以为空；不确定时只描述 missing_capability",
                    "default": [],
                },
                "risk_level": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "补充能力可能带来的风险等级",
                    "default": "low",
                },
                "user_visible_message": {
                    "type": "string",
                    "description": "给用户看的简短说明，例如“我需要补充读取项目文件的工具来查看代码”。",
                },
                "max_results": {
                    "type": "integer",
                    "description": "可选：最多返回几个匹配工具。不填则返回全部匹配工具。",
                    "minimum": 1,
                    "maximum": 20,
                },
                "list_all_tools": {
                    "type": "boolean",
                    "description": "为 true 时返回工具袋里的全部工具清单和 schema，不做匹配过滤。",
                    "default": False,
                },
            },
            "required": ["missing_capability", "reason"],
        }
