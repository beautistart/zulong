# File: zulong/tools/openclaw_plugin.py
"""
OpenClaw 插件支持

通过 API 调用 OpenClaw 的插件系统，扩展祖龙的功能
"""

import requests
import logging
from typing import Dict, Any, List, Optional

from .base import BaseTool, ToolRegistry, ToolRequest, ToolResult, ToolCategory

logger = logging.getLogger(__name__)


class OpenClawPluginAdapter(BaseTool):
    """OpenClaw 插件适配器"""
    
    def __init__(self, openclaw_api_url: str = "http://localhost:3000/api"):
        """初始化 OpenClaw 插件适配器
        
        Args:
            openclaw_api_url: OpenClaw API 地址
        """
        super().__init__(
            name="openclaw_plugin",
            category=ToolCategory.SYSTEM
        )
        self.description = "调用 OpenClaw 的插件系统"
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
        """执行插件操作
        
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
            
            if action == "call_plugin":
                plugin_name = parameters.get("plugin_name")
                if not plugin_name:
                    return self._create_result(
                        success=False,
                        error="缺少插件名称",
                        execution_time=time.time() - start_time,
                        request_id=request.request_id
                    )
                
                plugin_args = parameters.get("args", {})
                result = self._call_plugin(plugin_name, **plugin_args)
            elif action == "list_plugins":
                result = self._list_plugins()
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
    
    def _call_plugin(self, plugin_name: str, **kwargs) -> Dict[str, Any]:
        """调用 OpenClaw 插件
        
        Args:
            plugin_name: 插件名称
            **kwargs: 插件参数
            
        Returns:
            插件执行结果
        """
        try:
            url = f"{self.openclaw_api_url}/plugins/{plugin_name}"
            payload = {
                "arguments": kwargs
            }
            
            logger.info(f"[OpenClawPluginAdapter] 调用插件: {plugin_name}, 参数: {kwargs}")
            
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"[OpenClawPluginAdapter] 插件调用成功: {plugin_name}")
                return result
            else:
                logger.error(f"[OpenClawPluginAdapter] 插件调用失败: {plugin_name}, 状态码: {response.status_code}")
                return {"success": False, "error": f"HTTP 错误: {response.status_code}"}
        except Exception as e:
            logger.error(f"[OpenClawPluginAdapter] 插件调用异常: {plugin_name}, 错误: {e}")
            return {"success": False, "error": str(e)}
    
    def _list_plugins(self) -> Dict[str, Any]:
        """列出可用的 OpenClaw 插件
        
        Returns:
            插件列表
        """
        try:
            url = f"{self.openclaw_api_url}/plugins"
            
            response = requests.get(
                url,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("[OpenClawPluginAdapter] 插件列表获取成功")
                return result
            else:
                logger.error(f"[OpenClawPluginAdapter] 插件列表获取失败, 状态码: {response.status_code}")
                return {"success": False, "error": f"HTTP 错误: {response.status_code}"}
        except Exception as e:
            logger.error(f"[OpenClawPluginAdapter] 插件列表获取异常: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """返回 OpenClaw 插件适配器的参数 Schema"""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "动作类型：'call_plugin'（调用插件）或 'list_plugins'（列出插件）",
                    "enum": ["call_plugin", "list_plugins"]
                },
                "plugin_name": {
                    "type": "string",
                    "description": "要调用的 OpenClaw 插件名称（action='call_plugin' 时需要）"
                },
                "args": {
                    "type": "object",
                    "description": "传递给 OpenClaw 插件的参数（action='call_plugin' 时需要）"
                }
            },
            "required": ["action"]
        }


# 注册工具
ToolRegistry().register(OpenClawPluginAdapter())
