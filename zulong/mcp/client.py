# File: zulong/mcp/client.py
# MCP Client 适配器 (Phase 4.1)
# 让 ZULONG 能够作为 MCP Client 调用外部 MCP Server 的工具

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MCPTransportType(str, Enum):
    """MCP 传输类型"""
    STDIO = "stdio"      # 标准输入输出
    SSE = "sse"          # Server-Sent Events
    HTTP = "http"        # HTTP POST


@dataclass
class MCPToolDefinition:
    """MCP 工具定义"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema
        }


@dataclass
class MCPToolResult:
    """MCP 工具执行结果"""
    tool_name: str
    success: bool
    content: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class MCPServerConfig:
    """MCP Server 配置"""
    name: str
    command: str                    # 启动命令 (如: python, node)
    args: List[str]                 # 参数
    env: Dict[str, str] = field(default_factory=dict)  # 环境变量
    transport: MCPTransportType = MCPTransportType.STDIO
    timeout: float = 30.0


class MCPClient:
    """
    MCP Client 适配器
    
    功能:
    - 连接到外部 MCP Server (通过 stdio/sse/http)
    - 发现并注册远程工具
    - 调用远程工具并获取结果
    - 管理多个 MCP Server 连接
    
    使用示例:
    ```python
    client = MCPClient()
    
    # 添加 MCP Server
    await client.add_server("file_server", MCPServerConfig(
        name="file_server",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"]
    ))
    
    # 列出所有可用工具
    tools = await client.list_tools()
    
    # 调用远程工具
    result = await client.call_tool("read_file", {"path": "/path/to/file.txt"})
    ```
    """
    
    def __init__(self):
        """初始化 MCP Client"""
        self._servers: Dict[str, MCPServerConfig] = {}
        self._server_processes: Dict[str, Optional[asyncio.subprocess.Process]] = {}
        self._available_tools: Dict[str, MCPToolDefinition] = {}  # tool_name -> definition
        self._tool_to_server: Dict[str, str] = {}  # tool_name -> server_name
        self._initialized = False
    
    async def initialize(self):
        """初始化 MCP Client"""
        if self._initialized:
            return
        
        self._initialized = True
        logger.info("[MCPClient] 初始化完成")
    
    async def add_server(self, name: str, config: MCPServerConfig):
        """
        添加 MCP Server
        
        Args:
            name: Server 名称
            config: Server 配置
        """
        self._servers[name] = config
        logger.info(f"[MCPClient] 添加 MCP Server: {name}")
    
    async def connect_server(self, name: str):
        """
        连接到 MCP Server
        
        Args:
            name: Server 名称
        """
        if name not in self._servers:
            raise ValueError(f"MCP Server '{name}' 未注册")
        
        config = self._servers[name]
        
        if config.transport == MCPTransportType.STDIO:
            await self._connect_stdio(name, config)
        else:
            raise NotImplementedError(f"传输类型 {config.transport} 暂未实现")
    
    async def _connect_stdio(self, name: str, config: MCPServerConfig):
        """通过 stdio 连接到 MCP Server"""
        try:
            # 启动子进程
            process = await asyncio.create_subprocess_exec(
                config.command,
                *config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**config.env}
            )
            
            self._server_processes[name] = process
            logger.info(f"[MCPClient] 已启动 MCP Server '{name}': PID={process.pid}")
            
            # 发现工具
            await self._discover_tools(name, process)
            
        except Exception as e:
            logger.error(f"[MCPClient] 启动 MCP Server '{name}' 失败: {e}")
            raise
    
    async def _discover_tools(self, server_name: str, process: asyncio.subprocess.Process):
        """
        发现 MCP Server 提供的工具
        
        发送 JSON-RPC 请求获取工具列表
        """
        try:
            # 构建 tools/list 请求
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {}
            }
            
            # 发送请求
            request_json = json.dumps(request) + "\n"
            process.stdin.write(request_json.encode('utf-8'))
            await process.stdin.drain()
            
            # 读取响应 (简化实现，实际应该处理流式响应)
            response_line = await asyncio.wait_for(
                process.stdout.readline(),
                timeout=10.0
            )
            
            if response_line:
                response = json.loads(response_line.decode('utf-8'))
                
                # 解析工具列表
                if "result" in response and "tools" in response["result"]:
                    for tool_def in response["result"]["tools"]:
                        tool = MCPToolDefinition(
                            name=tool_def["name"],
                            description=tool_def.get("description", ""),
                            input_schema=tool_def.get("inputSchema", {}),
                            server_name=server_name
                        )
                        
                        self._available_tools[tool.name] = tool
                        self._tool_to_server[tool.name] = server_name
                        
                        logger.info(f"[MCPClient] 发现工具: {tool.name} (来自 {server_name})")
            
        except asyncio.TimeoutError:
            logger.warning(f"[MCPClient] 发现工具超时")
        except Exception as e:
            logger.error(f"[MCPClient] 发现工具失败: {e}")
    
    async def list_tools(self) -> List[MCPToolDefinition]:
        """
        列出所有可用的 MCP 工具
        
        Returns:
            List[MCPToolDefinition]: 工具列表
        """
        return list(self._available_tools.values())
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """
        调用 MCP 工具
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            
        Returns:
            MCPToolResult: 工具执行结果
        """
        if tool_name not in self._tool_to_server:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                error=f"工具 '{tool_name}' 未找到"
            )
        
        server_name = self._tool_to_server[tool_name]
        process = self._server_processes.get(server_name)
        
        if not process:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                error=f"MCP Server '{server_name}' 未连接"
            )
        
        start_time = time.time()
        
        try:
            # 构建 tools/call 请求
            request_id = int(time.time() * 1000)
            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            # 发送请求
            request_json = json.dumps(request) + "\n"
            process.stdin.write(request_json.encode('utf-8'))
            await process.stdin.drain()
            
            # 读取响应
            response_line = await asyncio.wait_for(
                process.stdout.readline(),
                timeout=30.0
            )
            
            execution_time = time.time() - start_time
            
            if response_line:
                response = json.loads(response_line.decode('utf-8'))
                
                # 解析响应
                if "result" in response:
                    return MCPToolResult(
                        tool_name=tool_name,
                        success=True,
                        content=response["result"].get("content", []),
                        execution_time=execution_time
                    )
                elif "error" in response:
                    return MCPToolResult(
                        tool_name=tool_name,
                        success=False,
                        error=response["error"].get("message", "未知错误"),
                        execution_time=execution_time
                    )
            
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                error="响应格式错误",
                execution_time=execution_time
            )
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                error="工具调用超时",
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def disconnect_server(self, name: str):
        """断开 MCP Server 连接"""
        process = self._server_processes.get(name)
        if process:
            try:
                process.terminate()
                await process.wait()
                logger.info(f"[MCPClient] 已断开 MCP Server '{name}' 的连接")
            except Exception as e:
                logger.error(f"[MCPClient] 断开 MCP Server '{name}' 失败: {e}")
            finally:
                self._server_processes[name] = None
        
        # 清理该 Server 提供的工具
        tools_to_remove = [
            tool_name for tool_name, server_name in self._tool_to_server.items()
            if server_name == name
        ]
        for tool_name in tools_to_remove:
            del self._available_tools[tool_name]
            del self._tool_to_server[tool_name]
        
        if tools_to_remove:
            logger.info(f"[MCPClient] 已移除 {len(tools_to_remove)} 个工具")
    
    async def disconnect_all(self):
        """断开所有 MCP Server 连接"""
        for name in list(self._server_processes.keys()):
            await self.disconnect_server(name)
        
        self._available_tools.clear()
        self._tool_to_server.clear()
        logger.info("[MCPClient] 已断开所有连接")
    
    def get_tool_names(self) -> List[str]:
        """获取所有可用工具名称"""
        return list(self._available_tools.keys())
    
    def has_tool(self, tool_name: str) -> bool:
        """检查工具是否可用"""
        return tool_name in self._available_tools
