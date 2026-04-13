# File: zulong/mcp/server.py
# MCP Server 适配器 (Phase 4.2)
# 将 ZULONG 的工具暴露给外部 MCP Client

import asyncio
import json
import logging
import sys
import time
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class MCPServer:
    """
    MCP Server 适配器
    
    功能:
    - 将 ZULONG 的 ToolRegistry 工具暴露为 MCP 工具
    - 支持 stdio 传输 (通过 JSON-RPC)
    - 处理工具发现和调用
    
    使用示例:
    ```python
    from zulong.mcp.server import MCPServer
    from zulong.tools.tool_engine import ToolEngine
    
    # 创建 MCP Server
    server = MCPServer(tool_engine)
    
    # 启动 (阻塞模式)
    server.start_stdio()
    ```
    """
    
    def __init__(self, tool_engine=None):
        """
        初始化 MCP Server
        
        Args:
            tool_engine: ZULONG ToolEngine 实例
        """
        self._tool_engine = tool_engine
        self._request_handlers = {}
        self._running = False
        
        # 注册默认请求处理器
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """注册默认的 JSON-RPC 请求处理器"""
        self._request_handlers = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "ping": self._handle_ping
        }
    
    def set_tool_engine(self, tool_engine):
        """设置 ToolEngine"""
        self._tool_engine = tool_engine
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理 initialize 请求"""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {
                    "listChanged": True
                }
            },
            "serverInfo": {
                "name": "zulong-mcp-server",
                "version": "1.0.0"
            }
        }
    
    async def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理 tools/list 请求"""
        tools = []
        
        if self._tool_engine:
            # 从 ToolRegistry 获取所有工具
            schemas = self._tool_engine.get_all_function_schemas()
            
            for schema in schemas:
                tools.append({
                    "name": schema.name,
                    "description": schema.description or "",
                    "inputSchema": {
                        "type": "object",
                        "properties": schema.parameters.get("properties", {}),
                        "required": schema.parameters.get("required", [])
                    }
                })
        
        return {"tools": tools}
    
    async def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理 tools/call 请求"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if not tool_name:
            return {
                "content": [],
                "isError": True
            }
        
        if not self._tool_engine:
            return {
                "content": [{"type": "text", "text": "ToolEngine 未初始化"}],
                "isError": True
            }
        
        try:
            # 调用工具
            start_time = time.time()
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._tool_engine.call_tool(
                    tool_name=tool_name,
                    **arguments
                )
            )
            execution_time = time.time() - start_time
            
            # 格式化结果
            content = []
            if result and hasattr(result, 'result'):
                content.append({
                    "type": "text",
                    "text": str(result.result)
                })
            elif result:
                content.append({
                    "type": "text",
                    "text": json.dumps(result, ensure_ascii=False)
                })
            
            return {
                "content": content,
                "isError": False
            }
            
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"工具调用失败: {str(e)}"}],
                "isError": True
            }
    
    async def _handle_ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理 ping 请求"""
        return {}
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理 JSON-RPC 请求
        
        Args:
            request: JSON-RPC 请求
            
        Returns:
            Dict: JSON-RPC 响应
        """
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params", {})
        
        handler = self._request_handlers.get(method)
        
        if not handler:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
        
        try:
            result = await handler(params)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    def start_stdio(self):
        """
        启动 stdio 模式的 MCP Server (阻塞)
        
        从标准输入读取 JSON-RPC 请求，写入标准输出
        """
        self._running = True
        logger.info("[MCPServer] 启动 stdio 模式")
        
        async def run():
            loop = asyncio.get_event_loop()
            
            while self._running:
                try:
                    # 读取一行请求
                    line = await loop.run_in_executor(
                        None,
                        sys.stdin.readline
                    )
                    
                    if not line:
                        break
                    
                    # 解析请求
                    request = json.loads(line)
                    
                    # 处理请求
                    response = await self.process_request(request)
                    
                    # 写入响应
                    response_json = json.dumps(response, ensure_ascii=False) + "\n"
                    sys.stdout.write(response_json)
                    sys.stdout.flush()
                    
                except json.JSONDecodeError:
                    logger.error("[MCPServer] 无效的 JSON 请求")
                except Exception as e:
                    logger.error(f"[MCPServer] 处理请求失败: {e}")
        
        try:
            asyncio.run(run())
        except KeyboardInterrupt:
            logger.info("[MCPServer] 收到中断信号，停止服务")
        finally:
            self._running = False
    
    def stop(self):
        """停止 MCP Server"""
        self._running = False
        logger.info("[MCPServer] 已停止")
