# File: zulong/tools/web_search.py
"""
Web 搜索工具 — 直连本地 SearXNG 实例

SearXNG API: GET http://localhost:8101/search?q={query}&format=json&language=zh-CN
"""

import requests
import logging
import time
from typing import Dict, Any, List

from .base import BaseTool, ToolRequest, ToolResult, ToolCategory

logger = logging.getLogger(__name__)

# SearXNG 默认配置
_DEFAULT_SEARXNG_URL = "http://localhost:8101/search"
_DEFAULT_LANGUAGE = "zh-CN"
_DEFAULT_TIMEOUT = 15


class WebSearchTool(BaseTool):
    """Web 搜索工具 — 通过本地 SearXNG 实例执行联网搜索"""

    def __init__(
        self,
        searxng_url: str = _DEFAULT_SEARXNG_URL,
        language: str = _DEFAULT_LANGUAGE,
        timeout: int = _DEFAULT_TIMEOUT,
    ):
        super().__init__(name="web_search", category=ToolCategory.NETWORK)
        self.description = (
            "联网搜索工具。当需要查询实时信息、最新新闻、技术文档、百科知识等"
            "互联网内容时使用。参数：query（搜索关键词），count（结果数量，默认 5）"
        )
        self.searxng_url = searxng_url
        self.language = language
        self.timeout = timeout

    # ── 生命周期 ──

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    # ── 执行 ──

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.time()
        params = request.parameters
        query = params.get("query", "")
        if not query:
            return self._create_result(
                success=False,
                error="缺少搜索关键词 (query)",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        count = min(max(int(params.get("count", 5)), 1), 20)

        try:
            resp = requests.get(
                self.searxng_url,
                params={
                    "q": query,
                    "format": "json",
                    "language": self.language,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.Timeout:
            logger.warning("[WebSearchTool] SearXNG 请求超时: %s", query)
            return self._create_result(
                success=False,
                error=f"搜索超时 ({self.timeout}s)",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )
        except Exception as e:
            logger.error("[WebSearchTool] SearXNG 请求异常: %s", e)
            return self._create_result(
                success=False,
                error=str(e),
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        raw_results: List[Dict[str, Any]] = data.get("results", [])
        results = []
        for item in raw_results[:count]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "engine": item.get("engine", ""),
            })

        logger.info(
            "[WebSearchTool] 搜索 '%s' 返回 %d/%d 条结果",
            query, len(results), len(raw_results),
        )

        # 格式化为 Markdown 文本，方便 LLM 直接引用链接
        formatted = self._format_results_markdown(query, results)

        return self._create_result(
            success=True,
            data=formatted,
            execution_time=time.time() - start,
            request_id=request.request_id,
        )

    def _format_results_markdown(self, query: str, results: List[Dict[str, Any]]) -> str:
        """将搜索结果格式化为 Markdown 文本，便于 LLM 引用来源链接"""
        if not results:
            return f"搜索 \"{query}\" 未找到结果。"

        lines = [f"搜索 \"{query}\" 找到 {len(results)} 条结果：\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            url = r.get("url", "")
            snippet = r.get("snippet", "")
            lines.append(f"{i}. [{title}]({url})")
            if snippet:
                lines.append(f"   {snippet}")
            lines.append("")
        lines.append("请在回复中引用相关来源链接。")
        return "\n".join(lines)

    # ── Schema ──

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                },
                "count": {
                    "type": "integer",
                    "description": "返回结果数量，1-20，默认 5",
                },
            },
            "required": ["query"],
        }
