# File: zulong/tools/openclaw_search.py
"""
OpenClaw 搜索工具

使用 OpenClaw 的网络搜索能力，获取更准确的搜索结果
"""

import requests
import logging
from typing import Dict, Any, List, Optional

from .base import BaseTool, ToolRegistry, ToolRequest, ToolResult, ToolCategory

logger = logging.getLogger(__name__)


class OpenClawSearchTool(BaseTool):
    """OpenClaw 搜索工具"""
    
    def __init__(self, openclaw_api_url: str = "http://localhost:3000/api"):
        """初始化 OpenClaw 搜索工具
        
        Args:
            openclaw_api_url: OpenClaw API 地址
        """
        super().__init__(
            name="openclaw_search",
            category=ToolCategory.NETWORK
        )
        self.description = "联网搜索工具。当用户需要查询实时信息、最新数据、新闻、天气、股票、百科知识等互联网信息时，使用此工具搜索。参数：query(搜索关键词)"
        self.openclaw_api_url = openclaw_api_url
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def initialize(self) -> bool:
        """初始化工具
        
        Returns:
            bool: 是否初始化成功
        """
        return True
    
    def execute(self, request: ToolRequest) -> ToolResult:
        """执行搜索操作
        
        Args:
            request: 工具请求
            
        Returns:
            ToolResult: 执行结果
        """
        import time
        start_time = time.time()
        
        try:
            action = request.action
            parameters = request.parameters
            
            if action == "search":
                query = parameters.get("query")
                if not query:
                    return self._create_result(
                        success=False,
                        error="缺少搜索查询",
                        execution_time=time.time() - start_time,
                        request_id=request.request_id
                    )
                
                count = parameters.get("count", 3)
                result = self._search(query, count)
            elif action == "fetch_webpage":
                url = parameters.get("url")
                if not url:
                    return self._create_result(
                        success=False,
                        error="缺少 URL 参数",
                        execution_time=time.time() - start_time,
                        request_id=request.request_id
                    )
                
                result = self._fetch_webpage(url)
            else:
                return self._create_result(
                    success=False,
                    error=f"不支持的操作：{action}",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id
                )
            
            return self._create_result(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
                execution_time=time.time() - start_time,
                request_id=request.request_id
            )
        except Exception as e:
            return self._create_result(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
                request_id=request.request_id
            )
    
    def cleanup(self) -> None:
        """清理工具资源"""
        pass
    
    def _search(self, query: str, count: int = 3) -> Dict[str, Any]:
        """执行网络搜索
        
        Args:
            query: 搜索查询
            count: 结果数量
            
        Returns:
            搜索结果
        """
        try:
            url = f"{self.openclaw_api_url}/search"
            payload = {
                "query": query,
                "count": count
            }
            
            logger.info(f"[OpenClawSearchTool] 执行搜索：{query}, 结果数量：{count}")
            
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"[OpenClawSearchTool] 搜索成功，找到 {len(result.get('results', []))} 个结果")
                return result
            else:
                logger.error(f"[OpenClawSearchTool] 搜索失败，状态码：{response.status_code}")
                return {"success": False, "error": f"HTTP 错误：{response.status_code}"}
        except Exception as e:
            logger.error(f"[OpenClawSearchTool] 搜索异常：{e}")
            return {"success": False, "error": str(e)}
    
    def _fetch_webpage(self, url: str) -> Dict[str, Any]:
        """获取网页内容
        
        Args:
            url: 网页 URL
            
        Returns:
            网页内容
        """
        try:
            api_url = f"{self.openclaw_api_url}/fetch_webpage"
            payload = {
                "url": url
            }
            
            logger.info(f"[OpenClawSearchTool] 获取网页内容：{url}")
            
            response = requests.post(
                api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"[OpenClawSearchTool] 成功获取网页内容，长度：{len(result.get('content', ''))}")
                return result
            else:
                logger.error(f"[OpenClawSearchTool] 获取网页失败，状态码：{response.status_code}")
                return {"success": False, "error": f"HTTP 错误：{response.status_code}"}
        except Exception as e:
            logger.error(f"[OpenClawSearchTool] 获取网页异常：{e}")
            return {"success": False, "error": str(e)}
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """返回 OpenClaw 搜索工具的参数 Schema"""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "动作类型：'search'（搜索）或 'fetch_webpage'（读取网页）",
                    "enum": ["search", "fetch_webpage"]
                },
                "query": {
                    "type": "string",
                    "description": "搜索关键词（action='search' 时需要）"
                },
                "count": {
                    "type": "integer",
                    "description": "搜索结果数量（action='search' 时需要），1-10 之间，默认 3"
                },
                "url": {
                    "type": "string",
                    "description": "网页 URL（action='fetch_webpage' 时需要）"
                }
            },
            "required": ["action"]
        }


# 注册工具
ToolRegistry().register(OpenClawSearchTool())
