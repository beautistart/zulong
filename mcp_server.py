"""
祖龙记忆 MCP Server - 轻量级实现

直接将祖龙现有记忆系统通过MCP协议暴露给灵码IDE，无需独立打包。
复用IDE的模型进行推理，祖龙仅提供记忆存储和检索能力。
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server import Server
from mcp.types import Tool, TextContent

# 导入祖龙记忆系统核心组件
from zulong.memory.memory_graph import get_memory_graph, MemoryGraph
from zulong.memory.rag_manager import RAGManager, RAGConfig
from zulong.memory.task_search_index import TaskSearchIndex

logger = logging.getLogger(__name__)

# ============================================================
# 初始化祖龙记忆系统
# ============================================================

def init_zulong_memory():
    """初始化祖龙记忆系统（单例模式）"""
    logger.info("[Zulong MCP] 初始化祖龙记忆系统...")
    
    # 1. 初始化 MemoryGraph
    memory_graph = get_memory_graph()
    logger.info(f"[Zulong MCP] MemoryGraph 已就绪: {memory_graph._stats['total_nodes']} 节点")
    
    # 2. 初始化 RAGManager
    rag_config = RAGConfig(
        vector_dimension=512,
        vector_store_type="faiss",
        base_path="./data/rag",
        experience_rag_enabled=True,
        memory_rag_enabled=True,
        knowledge_rag_enabled=True,
        tool_rag_enabled=True
    )
    rag_manager = RAGManager(rag_config)
    logger.info(f"[Zulong MCP] RAGManager 已就绪: {len(rag_manager.rag_libraries)} 个库")
    
    # 3. 初始化任务搜索索引
    task_index = TaskSearchIndex()
    logger.info(f"[Zulong MCP] TaskSearchIndex 已就绪")
    
    return memory_graph, rag_manager, task_index


# 全局实例
memory_graph: Optional[MemoryGraph] = None
rag_manager: Optional[RAGManager] = None
task_index: Optional[TaskSearchIndex] = None


# ============================================================
# MCP Server 定义
# ============================================================

app = Server("zulong-memory")


@app.list_tools()
async def list_tools() -> List[Tool]:
    """列出所有可用的MCP工具"""
    return [
        Tool(
            name="zulong_memory_search",
            description="【项目级记忆】搜索项目的架构决策、技术选型原因、团队规范、踩坑教训。适用于：修改核心模块前了解历史决策、查询某功能的设计原因、避免重复踩坑。不适用于：个人代码风格偏好、通用编程问题（这些请用灵码内置记忆）。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或自然语言描述（建议包含模块名/技术名，如'auth模块JWT认证'）"
                    },
                    "category": {
                        "type": "string",
                        "description": "可选分类过滤",
                        "enum": ["decision", "lesson", "pattern", "todo", "context"]
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量（默认10）",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="zulong_memory_write",
            description="【项目级记忆】记录重要的项目决策、技术选型原因、团队踩坑教训。适用于：架构决策（如'选择PostgreSQL的原因'）、重要教训（如'修改schema前必须备份'）、可复用模式。不适用于：个人偏好、临时TODO（这些请用灵码内置记忆或TodoList）。",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "记忆内容（建议包含背景+决策+原因，如'auth模块改为JWT，原因是session在微服务下扩展性差'）"
                    },
                    "type": {
                        "type": "string",
                        "description": "记忆类型：decision(架构决策)/lesson(踩坑教训)/pattern(可复用模式)/todo(项目级任务)/context(项目背景)",
                        "enum": ["decision", "lesson", "pattern", "todo", "context"]
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "标签列表（建议2-5个，如['auth', 'jwt', 'microservice']）"
                    }
                },
                "required": ["content", "type"]
            }
        ),
        Tool(
            name="zulong_memory_list",
            description="列出最近的记忆条目",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "返回数量（默认10）",
                        "default": 10
                    },
                    "type": {
                        "type": "string",
                        "description": "按类型过滤（可选）",
                        "enum": ["decision", "lesson", "pattern", "todo", "context"]
                    }
                }
            }
        ),
        Tool(
            name="zulong_context_inject",
            description="一键获取当前项目的关键记忆摘要（用于会话开始时）",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "可选的查询上下文，为空则返回全部关键记忆"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量（默认5）",
                        "default": 5
                    }
                }
            }
        ),
        Tool(
            name="zulong_task_view",
            description="查看当前项目的任务全景（宏观目标、子任务状态、依赖关系）",
            inputSchema={
                "type": "object",
                "properties": {
                    "status_filter": {
                        "type": "string",
                        "description": "按状态过滤（可选）",
                        "enum": ["pending", "in_progress", "completed", "blocked"]
                    }
                }
            }
        ),
        Tool(
            name="zulong_task_create",
            description="创建新的任务节点",
            inputSchema={
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "任务描述"
                    },
                    "parent_id": {
                        "type": "string",
                        "description": "父任务ID（可选）"
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "依赖的任务ID列表（可选）"
                    }
                },
                "required": ["label"]
            }
        ),
        Tool(
            name="zulong_task_update",
            description="更新任务状态",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "任务ID"
                    },
                    "status": {
                        "type": "string",
                        "description": "新状态",
                        "enum": ["pending", "in_progress", "completed", "blocked"]
                    },
                    "note": {
                        "type": "string",
                        "description": "状态更新备注（可选）"
                    }
                },
                "required": ["task_id", "status"]
            }
        ),
        Tool(
            name="zulong_experience_query",
            description="查询相关的历史经验",
            inputSchema={
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "当前工作上下文"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量（默认5）",
                        "default": 5
                    }
                },
                "required": ["context"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """处理工具调用"""
    try:
        if name == "zulong_memory_search":
            return await handle_memory_search(arguments)
        elif name == "zulong_memory_write":
            return await handle_memory_write(arguments)
        elif name == "zulong_memory_list":
            return await handle_memory_list(arguments)
        elif name == "zulong_context_inject":
            return await handle_context_inject(arguments)
        elif name == "zulong_task_view":
            return await handle_task_view(arguments)
        elif name == "zulong_task_create":
            return await handle_task_create(arguments)
        elif name == "zulong_task_update":
            return await handle_task_update(arguments)
        elif name == "zulong_experience_query":
            return await handle_experience_query(arguments)
        else:
            return [TextContent(type="text", text=f"未知工具: {name}")]
    except Exception as e:
        logger.error(f"[Zulong MCP] 工具调用失败: {name}, 错误: {e}", exc_info=True)
        return [TextContent(type="text", text=f"错误: {str(e)}")]


# ============================================================
# 工具处理器实现
# ============================================================

async def handle_memory_search(args: Dict) -> List[TextContent]:
    """处理记忆搜索"""
    query = args.get("query", "")
    category = args.get("category")
    top_k = args.get("top_k", 10)
    
    logger.info(f"[Zulong MCP] 搜索记忆: query='{query[:50]}...', category={category}")
    
    # 使用 MemoryGraph 检索上下文
    results = await memory_graph.retrieve_context(query, top_k=top_k)
    
    # 格式化输出
    if not results:
        return [TextContent(type="text", text="未找到相关记忆")]
    
    output_lines = [f"找到 {len(results)} 条相关记忆:\n"]
    for i, result in enumerate(results, 1):
        node_id = result.get("node_id", "N/A")
        node_type = result.get("node_type", "unknown")
        label = result.get("label", "N/A")
        content = result.get("content", "")
        score = result.get("score", 0.0)
        
        output_lines.append(f"{i}. [{node_type}] {label}")
        output_lines.append(f"   内容: {content[:200]}")
        output_lines.append(f"   相关度: {score:.2f}")
        output_lines.append(f"   ID: {node_id}\n")
    
    return [TextContent(type="text", text="\n".join(output_lines))]


async def handle_memory_write(args: Dict) -> List[TextContent]:
    """处理记忆写入"""
    content = args.get("content", "")
    mem_type = args.get("type", "context")
    tags = args.get("tags", [])
    
    logger.info(f"[Zulong MCP] 写入记忆: type={mem_type}, content_length={len(content)}")
    
    # 创建记忆节点
    from zulong.memory.memory_graph import GraphNode, NodeType
    
    node_type_map = {
        "decision": NodeType.DECISION,
        "lesson": NodeType.LESSON,
        "pattern": NodeType.PATTERN,
        "todo": NodeType.TODO,
        "context": NodeType.CONTEXT
    }
    
    node = GraphNode(
        node_type=node_type_map.get(mem_type, NodeType.CONTEXT),
        label=content[:100],  # 使用前100字符作为标签
        metadata={
            "content": content,
            "tags": tags,
            "source": "lingma_ide"
        }
    )
    
    node_id = memory_graph.add_node(node)
    logger.info(f"[Zulong MCP] 记忆已保存: node_id={node_id}")
    
    return [TextContent(type="text", text=f"✅ 记忆已保存\nID: {node_id}\n类型: {mem_type}")]


async def handle_memory_list(args: Dict) -> List[TextContent]:
    """处理记忆列表"""
    limit = args.get("limit", 10)
    mem_type = args.get("type")
    
    logger.info(f"[Zulong MCP] 列出记忆: limit={limit}, type={mem_type}")
    
    # 获取最近的节点
    nodes = list(memory_graph._nodes.values())
    nodes.sort(key=lambda n: n.last_accessed or 0, reverse=True)
    
    if mem_type:
        node_type_map = {
            "decision": NodeType.DECISION,
            "lesson": NodeType.LESSON,
            "pattern": NodeType.PATTERN,
            "todo": NodeType.TODO,
            "context": NodeType.CONTEXT
        }
        target_type = node_type_map.get(mem_type)
        nodes = [n for n in nodes if n.node_type == target_type]
    
    nodes = nodes[:limit]
    
    if not nodes:
        return [TextContent(type="text", text="暂无记忆")]
    
    output_lines = [f"最近 {len(nodes)} 条记忆:\n"]
    for i, node in enumerate(nodes, 1):
        output_lines.append(f"{i}. [{node.node_type.value}] {node.label}")
        if node.metadata.get("content"):
            output_lines.append(f"   {node.metadata['content'][:150]}")
        output_lines.append("")
    
    return [TextContent(type="text", text="\n".join(output_lines))]


async def handle_context_inject(args: Dict) -> List[TextContent]:
    """处理上下文注入"""
    query = args.get("query", "")
    top_k = args.get("top_k", 5)
    
    logger.info(f"[Zulong MCP] 注入上下文: query='{query[:50] if query else '全部'}'")
    
    if query:
        results = await memory_graph.retrieve_context(query, top_k=top_k)
    else:
        # 返回所有活跃节点
        nodes = list(memory_graph._nodes.values())
        nodes.sort(key=lambda n: n.last_accessed or 0, reverse=True)
        results = [
            {
                "node_id": n.node_id,
                "node_type": n.node_type.value,
                "label": n.label,
                "content": n.metadata.get("content", ""),
                "score": 1.0
            }
            for n in nodes[:top_k]
        ]
    
    if not results:
        return [TextContent(type="text", text="暂无项目记忆")]
    
    output_lines = ["📋 项目关键记忆摘要:\n"]
    for i, result in enumerate(results, 1):
        output_lines.append(f"{i}. [{result['node_type']}] {result['label']}")
        content = result.get('content', '')
        if content:
            output_lines.append(f"   {content[:200]}")
        output_lines.append("")
    
    return [TextContent(type="text", text="\n".join(output_lines))]


async def handle_task_view(args: Dict) -> List[TextContent]:
    """处理任务视图"""
    status_filter = args.get("status_filter")
    
    logger.info(f"[Zulong MCP] 查看任务: status_filter={status_filter}")
    
    # 从 MemoryGraph 中检索 TODO 类型的节点
    todo_nodes = [
        n for n in memory_graph._nodes.values()
        if n.node_type.value == "todo"
    ]
    
    if status_filter:
        todo_nodes = [
            n for n in todo_nodes
            if n.metadata.get("status") == status_filter
        ]
    
    if not todo_nodes:
        return [TextContent(type="text", text="暂无任务")]
    
    output_lines = ["📊 任务全景:\n"]
    for i, node in enumerate(todo_nodes, 1):
        status = node.metadata.get("status", "pending")
        status_icon = {
            "pending": "⏸️",
            "in_progress": "▶️",
            "completed": "✅",
            "blocked": "🚫"
        }.get(status, "❓")
        
        output_lines.append(f"{i}. {status_icon} [{status}] {node.label}")
        if node.metadata.get("note"):
            output_lines.append(f"   备注: {node.metadata['note']}")
        output_lines.append("")
    
    return [TextContent(type="text", text="\n".join(output_lines))]


async def handle_task_create(args: Dict) -> List[TextContent]:
    """处理任务创建"""
    label = args.get("label", "")
    parent_id = args.get("parent_id")
    depends_on = args.get("depends_on", [])
    
    logger.info(f"[Zulong MCP] 创建任务: label='{label[:50]}...'")
    
    from zulong.memory.memory_graph import GraphNode, NodeType
    
    node = GraphNode(
        node_type=NodeType.TODO,
        label=label,
        metadata={
            "status": "pending",
            "parent_id": parent_id,
            "depends_on": depends_on,
            "created_at": __import__('time').time()
        }
    )
    
    node_id = memory_graph.add_node(node)
    logger.info(f"[Zulong MCP] 任务已创建: node_id={node_id}")
    
    return [TextContent(type="text", text=f"✅ 任务已创建\nID: {node_id}\n描述: {label}")]


async def handle_task_update(args: Dict) -> List[TextContent]:
    """处理任务更新"""
    task_id = args.get("task_id", "")
    status = args.get("status", "pending")
    note = args.get("note")
    
    logger.info(f"[Zulong MCP] 更新任务: task_id={task_id}, status={status}")
    
    if task_id not in memory_graph._nodes:
        return [TextContent(type="text", text=f"❌ 任务不存在: {task_id}")]
    
    node = memory_graph._nodes[task_id]
    node.metadata["status"] = status
    if note:
        node.metadata["note"] = note
    node.metadata["updated_at"] = __import__('time').time()
    
    # 触发更新
    memory_graph.add_node(node)
    
    return [TextContent(type="text", text=f"✅ 任务已更新\nID: {task_id}\n状态: {status}")]


async def handle_experience_query(args: Dict) -> List[TextContent]:
    """处理经验查询"""
    context = args.get("context", "")
    top_k = args.get("top_k", 5)
    
    logger.info(f"[Zulong MCP] 查询经验: context='{context[:50]}...'")
    
    # 使用 RAGManager 搜索经验库
    if rag_manager and "experience" in rag_manager.rag_libraries:
        results = rag_manager.search("experience", context, top_k=top_k)
        
        if not results:
            return [TextContent(type="text", text="暂无相关经验")]
        
        output_lines = [f"找到 {len(results)} 条相关经验:\n"]
        for i, doc in enumerate(results, 1):
            output_lines.append(f"{i}. {doc.content[:200]}")
            if doc.metadata:
                output_lines.append(f"   类别: {doc.metadata.get('category', 'N/A')}")
                output_lines.append(f"   重要性: {doc.metadata.get('importance', 'N/A')}")
            output_lines.append("")
        
        return [TextContent(type="text", text="\n".join(output_lines))]
    
    return [TextContent(type="text", text="经验库未启用")]


# ============================================================
# 启动入口
# ============================================================

async def main():
    """启动MCP Server"""
    global memory_graph, rag_manager, task_index
    
    # 初始化祖龙记忆系统
    memory_graph, rag_manager, task_index = init_zulong_memory()
    
    logger.info("[Zulong MCP] 祖龙记忆系统已就绪，启动MCP Server...")
    
    # 运行MCP Server（stdio模式）
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("[Zulong MCP] MCP Server 已停止")
