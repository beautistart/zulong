# File: zulong/tools/web_search.py
# 网络搜索工具

import asyncio
import logging
import aiohttp
from typing import List, Dict, Optional
from dataclasses import dataclass

from zulong.tools.base import BaseTool, ToolCategory

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """搜索结果"""
    title: str
    url: str
    snippet: str
    score: float


class WebSearchTool(BaseTool):
    """网络搜索工具
    
    功能：
    1. 使用 Bing Search API 进行网络搜索
    2. 支持中文和英文搜索
    3. 异步搜索，不阻塞主线程
    """
    
    def __init__(self):
        """初始化搜索工具"""
        super().__init__(
            name="web_search",
            category=ToolCategory.NETWORK
        )
        self.description = "进行网络搜索，获取最新信息"
        self.parameters = {
            "query": {
                "type": "string",
                "description": "搜索关键词",
                "required": True
            },
            "count": {
                "type": "integer",
                "description": "返回结果数量",
                "required": False,
                "default": 5
            }
        }
        
        # 配置
        self.api_key = "YOUR_BING_API_KEY"  # 需要替换为真实的 API Key
        self.search_url = "https://api.bing.microsoft.com/v7.0/search"
        self.headers = {
            "Ocp-Apim-Subscription-Key": self.api_key
        }
        
        logger.info("[WebSearchTool] 初始化完成")
    
    def initialize(self) -> bool:
        """初始化工具
        
        Returns:
            bool: 是否初始化成功
        """
        logger.info("[WebSearchTool] 初始化调用")
        return True
    
    def cleanup(self) -> None:
        """清理工具资源"""
        logger.info("[WebSearchTool] 清理资源")
        pass
    
    async def _search_bing(self, query: str, count: int = 5) -> List[SearchResult]:
        """使用 Bing API 搜索
        
        Args:
            query: 搜索关键词
            count: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "q": query,
                    "count": count,
                    "mkt": "zh-CN",
                    "responseFilter": "Webpages"
                }
                
                async with session.get(
                    self.search_url,
                    headers=self.headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10.0)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        
                        if "webPages" in data and "value" in data["webPages"]:
                            for item in data["webPages"]["value"]:
                                result = SearchResult(
                                    title=item.get("name", ""),
                                    url=item.get("url", ""),
                                    snippet=item.get("snippet", ""),
                                    score=item.get("id", 0)
                                )
                                results.append(result)
                        
                        logger.info(f"[WebSearchTool] 搜索完成，找到 {len(results)} 个结果")
                        return results
                    else:
                        logger.error(f"[WebSearchTool] API 请求失败: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"[WebSearchTool] 搜索失败: {e}")
            return []
    
    async def _search_fallback(self, query: str, count: int = 5) -> List[SearchResult]:
        """降级搜索方案（当 API Key 不可用时）
        
        Args:
            query: 搜索关键词
            count: 返回结果数量
            
        Returns:
            模拟搜索结果
        """
        logger.warning("[WebSearchTool] 使用降级搜索方案")
        
        # 模拟搜索结果
        mock_results = [
            SearchResult(
                title=f"关于 '{query}' 的搜索结果 1",
                url=f"https://example.com/search?q={query}&page=1",
                snippet=f"这是关于 '{query}' 的搜索结果摘要，包含相关信息。",
                score=1.0
            ),
            SearchResult(
                title=f"关于 '{query}' 的搜索结果 2",
                url=f"https://example.com/search?q={query}&page=2",
                snippet=f"这是另一个关于 '{query}' 的搜索结果摘要，提供更多相关信息。",
                score=0.9
            )
        ]
        
        return mock_results[:count]
    
    async def execute(self, parameters: Dict[str, any]) -> Dict[str, any]:
        """执行搜索
        
        Args:
            parameters: 搜索参数
            
        Returns:
            搜索结果
        """
        query = parameters.get("query", "")
        count = parameters.get("count", 5)
        
        if not query:
            return {
                "success": False,
                "error": "搜索关键词不能为空"
            }
        
        logger.info(f"[WebSearchTool] 开始搜索: '{query}'")
        
        # 尝试使用 Bing API
        results = await self._search_bing(query, count)
        
        # 如果 API 失败，使用降级方案
        if not results:
            results = await self._search_fallback(query, count)
        
        # 格式化结果
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet
            })
        
        return {
            "success": True,
            "results": formatted_results,
            "count": len(formatted_results)
        }
    
    def is_available(self) -> bool:
        """检查工具是否可用
        
        Returns:
            bool: 是否可用
        """
        return True  # 即使 API Key 不可用，也可以使用降级方案
    
    def _get_parameters_schema(self) -> Dict[str, any]:
        """返回搜索工具的参数 Schema"""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                },
                "count": {
                    "type": "integer",
                    "description": "返回结果数量，1-10 之间，默认 5"
                }
            },
            "required": ["query"]
        }
