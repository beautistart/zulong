# File: zulong/mcp/__init__.py
# MCP 协议支持模块

from zulong.mcp.client import MCPClient
from zulong.mcp.server import MCPServer

__all__ = ["MCPClient", "MCPServer"]
