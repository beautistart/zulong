# File: zulong/tools/experience_tool.py
# search_experience FC 工具 - ExperienceRAG 被动检索
#
# 将 ExperienceRAG 从自动注入改为模型按需检索。
# 模型在需要参考历史经验时，通过 Function Calling 调用此工具。

import logging
import time
from typing import Dict, Any, Optional

from .base import BaseTool, ToolCategory, ToolRequest, ToolResult

logger = logging.getLogger(__name__)


class SearchExperienceTool(BaseTool):
    """search_experience 工具

    提供经验库的被动检索能力。模型在判断需要参考历史经验时
    主动调用此工具，而非系统在每轮对话中自动注入。
    """

    def __init__(self, rag_manager=None):
        super().__init__(name="search_experience", category=ToolCategory.CUSTOM)
        self.description = (
            "从经验库中检索与查询相关的历史经验。当你需要参考过去处理类似问题的经验、"
            "已有的解决方案或历史对话中的知识时，调用此工具进行检索。"
        )
        self._rag_manager = rag_manager

    def set_rag_manager(self, rag_manager):
        """延迟注入 RAGManager（避免循环依赖）"""
        self._rag_manager = rag_manager

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        """执行经验检索

        Args:
            request.parameters:
                - query: 检索关键词
                - top_k: 返回数量（默认 3）
        """
        start_time = time.time()

        query = request.parameters.get("query", "")
        top_k = request.parameters.get("top_k", 3)

        if not query:
            return self._create_result(
                success=False,
                error="缺少检索查询 (query)",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        if self._rag_manager is None:
            return self._create_result(
                success=False,
                error="RAGManager 未初始化，经验库不可用",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        try:
            results = self._rag_manager.search("experience", query, top_k=top_k)

            if not results:
                return self._create_result(
                    success=True,
                    data={
                        "message": f"未找到与 '{query}' 相关的经验",
                        "docs_found": 0,
                        "documents": [],
                    },
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            docs = []
            for doc in results[:top_k]:
                content = getattr(doc, "content", "")
                doc_id = getattr(doc, "doc_id", "")
                similarity = getattr(doc, "similarity", 0.0)
                docs.append({
                    "doc_id": doc_id,
                    "content": content[:800] if content else "",
                    "similarity": round(similarity, 3),
                })

            logger.info(
                f"[SearchExperienceTool] query='{query[:40]}' -> "
                f"found {len(docs)} docs"
            )

            return self._create_result(
                success=True,
                data={
                    "message": f"找到 {len(docs)} 条相关经验",
                    "docs_found": len(docs),
                    "documents": docs,
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[SearchExperienceTool] 检索失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"经验检索失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索关键词，描述你需要查找的经验内容"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的最大文档数量，默认 3"
                }
            },
            "required": ["query"]
        }
