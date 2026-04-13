# File: zulong/tools/openclaw_tool.py
"""
OpenClaw 工具适配器

通过 API 调用 OpenClaw 的工具系统，为祖龙提供更多功能
"""

import requests
import json
import logging
from typing import Dict, Any, Optional

from .base import BaseTool, ToolRegistry, ToolRequest, ToolResult, ToolCategory

logger = logging.getLogger(__name__)


class OpenClawToolAdapter(BaseTool):
    """OpenClaw 工具适配器"""
    
    def __init__(self, openclaw_api_url: str = "http://localhost:3000/api"):
        """初始化 OpenClaw 工具适配器
        
        Args:
            openclaw_api_url: OpenClaw API 地址
        """
        super().__init__(
            name="openclaw_tool",
            category=ToolCategory.SYSTEM
        )
        self.description = "调用 OpenClaw 的工具系统"
        self.openclaw_api_url = openclaw_api_url
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def _call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """调用 OpenClaw 工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            工具执行结果
        """
        try:
            url = f"{self.openclaw_api_url}/tools/{tool_name}"
            payload = {
                "arguments": kwargs
            }
            
            logger.info(f"[OpenClawToolAdapter] 调用工具: {tool_name}, 参数: {kwargs}")
            
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"[OpenClawToolAdapter] 工具调用成功: {tool_name}")
                return result
            else:
                logger.error(f"[OpenClawToolAdapter] 工具调用失败: {tool_name}, 状态码: {response.status_code}")
                return {"success": False, "error": f"HTTP 错误: {response.status_code}"}
        except Exception as e:
            logger.error(f"[OpenClawToolAdapter] 工具调用异常: {tool_name}, 错误: {e}")
            return {"success": False, "error": str(e)}
    
    def initialize(self) -> bool:
        """初始化工具
        
        Returns:
            bool: 是否初始化成功
        """
        return True
    
    def execute(self, request: ToolRequest) -> ToolResult:
        """执行工具操作
        
        Args:
            request: 工具请求
            
        Returns:
            ToolResult: 执行结果
        """
        start_time = time.time()
        
        try:
            action = request.action
            parameters = request.parameters
            
            if action == "call_tool":
                tool_name = parameters.get("tool_name")
                if not tool_name:
                    return self._create_result(
                        success=False,
                        error="缺少工具名称",
                        execution_time=time.time() - start_time,
                        request_id=request.request_id
                    )
                
                tool_args = parameters.get("args", {})
                result = self._call_tool(tool_name, **tool_args)
            elif action == "list_tools":
                result = self._list_tools()
            else:
                return self._create_result(
                    success=False,
                    error=f"不支持的操作: {action}",
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
    
    def _list_tools(self) -> Dict[str, Any]:
        """列出可用的 OpenClaw 工具
        
        Returns:
            工具列表
        """
        try:
            url = f"{self.openclaw_api_url}/tools"
            
            response = requests.get(
                url,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("[OpenClawToolAdapter] 工具列表获取成功")
                return result
            else:
                logger.error(f"[OpenClawToolAdapter] 工具列表获取失败, 状态码: {response.status_code}")
                return {"success": False, "error": f"HTTP 错误: {response.status_code}"}
        except Exception as e:
            logger.error(f"[OpenClawToolAdapter] 工具列表获取异常: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """返回 OpenClaw 工具适配器的参数 Schema"""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "动作类型：'call_tool'（调用工具）或 'list_tools'（列出工具）",
                    "enum": ["call_tool", "list_tools"]
                },
                "tool_name": {
                    "type": "string",
                    "description": "要调用的 OpenClaw 工具名称（action='call_tool' 时需要）"
                },
                "args": {
                    "type": "object",
                    "description": "传递给 OpenClaw 工具的参数（action='call_tool' 时需要）"
                }
            },
            "required": ["action"]
        }


# 注册工具
ToolRegistry().register(OpenClawToolAdapter())
