"""TaskGraph CRUD 工具 Schema 定义与分发入口"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

CRUD_TOOL_NAMES = {
    "graph_create_node",
    "graph_create_edge",
    "graph_batch_create",
    "graph_delete_node",
    "graph_delete_edge",
    "graph_update_node",
    "graph_batch_update",
    "graph_query_nodes",
}


def _generate_node_id(node_type: str) -> str:
    ts = int(time.time() * 1000)
    if node_type == "task":
        return f"task:tg_{ts}"
    elif node_type == "note":
        return f"note:tg_{ts}"
    else:
        return f"{node_type}:tg_{ts}"


def get_crud_tool_schemas() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "graph_create_node",
                "description": "在任务图谱中创建新节点。支持task和note类型，自动生成ID（task:tg_* / note:tg_*）。可指定parent_id建立层级关系。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_type": {"type": "string", "enum": ["task", "note", "requirement", "analysis", "outline", "subtask"], "description": "节点类型"},
                        "label": {"type": "string", "description": "节点名称（必填，≤200字符）"},
                        "desc": {"type": "string", "description": "节点描述（≤2000字符）"},
                        "parent_id": {"type": "string", "description": "父节点ID，指定时自动建立层级边"},
                        "task_domain": {"type": "string", "enum": ["code", "research", "creative", "data", "general"], "description": "任务领域"},
                        "custom_id": {"type": "string", "description": "自定义节点ID（可选，不指定则自动生成）"},
                    },
                    "required": ["node_type", "label"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "graph_create_edge",
                "description": "在任务图谱中创建边。支持层级边(hierarchy)、依赖边(dependency)和引用边(reference)。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "源节点ID（必填）"},
                        "target": {"type": "string", "description": "目标节点ID（必填）"},
                        "edge_type": {"type": "string", "enum": ["hierarchy", "dependency", "reference"], "description": "边类型"},
                        "via": {"type": "string", "description": "依赖边的中介描述"},
                        "cross": {"type": "boolean", "description": "是否跨图谱依赖"},
                    },
                    "required": ["source", "target", "edge_type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "graph_batch_create",
                "description": "批量创建任务图谱节点和边。节点和边按顺序依次创建，部分失败不影响已创建的项。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nodes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "node_type": {"type": "string"},
                                    "label": {"type": "string"},
                                    "desc": {"type": "string"},
                                    "parent_id": {"type": "string"},
                                    "task_domain": {"type": "string"},
                                    "custom_id": {"type": "string"},
                                },
                                "required": ["node_type", "label"],
                            },
                            "description": "待创建节点列表",
                        },
                        "edges": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string"},
                                    "target": {"type": "string"},
                                    "edge_type": {"type": "string"},
                                    "via": {"type": "string"},
                                },
                                "required": ["source", "target", "edge_type"],
                            },
                            "description": "待创建边列表",
                        },
                    },
                    "required": ["nodes"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "graph_delete_node",
                "description": "删除任务图谱节点。支持级联删除（删除子节点）、软删除（标记为deleted而非物理删除）。重要节点需确认。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string", "description": "待删除节点ID（必填）"},
                        "cascade": {"type": "boolean", "default": False, "description": "是否级联删除子节点"},
                        "soft_delete": {"type": "boolean", "default": False, "description": "是否软删除（标记为deleted）"},
                    },
                    "required": ["node_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "graph_delete_edge",
                "description": "删除任务图谱中的边。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "源节点ID（必填）"},
                        "target": {"type": "string", "description": "目标节点ID（必填）"},
                        "edge_type": {"type": "string", "enum": ["hierarchy", "dependency"], "description": "边类型（必填）"},
                    },
                    "required": ["source", "target", "edge_type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "graph_update_node",
                "description": "更新任务图谱节点属性。支持状态流转（需符合合法流转规则）、属性更新。不可修改id和created_at。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string", "description": "节点ID（必填）"},
                        "label": {"type": "string", "description": "新名称"},
                        "desc": {"type": "string", "description": "新描述"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "blocked", "skipped", "needs_adjust", "waiting_input", "deleted"], "description": "新状态"},
                        "result": {"type": "string", "description": "执行结果"},
                        "task_domain": {"type": "string", "description": "任务领域"},
                        "analysis_content": {"type": "string", "description": "分析内容"},
                        "semantic_summary": {"type": "string", "description": "语义摘要"},
                    },
                    "required": ["node_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "graph_batch_update",
                "description": "批量更新任务图谱节点属性。每个更新项需包含node_id和待更新属性。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "updates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "node_id": {"type": "string"},
                                    "label": {"type": "string"},
                                    "desc": {"type": "string"},
                                    "status": {"type": "string"},
                                    "result": {"type": "string"},
                                },
                                "required": ["node_id"],
                            },
                            "description": "待更新节点列表",
                        },
                    },
                    "required": ["updates"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "graph_query_nodes",
                "description": "查询任务图谱节点。支持5种模式：get_node(按ID)、list(按类型列表)、search(关键词搜索)、subgraph(关联子图)、overview(概要统计)。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["get_node", "list", "search", "subgraph", "overview"], "description": "查询模式（必填）"},
                        "node_id": {"type": "string", "description": "节点ID（get_node/subgraph模式使用）"},
                        "node_type": {"type": "string", "description": "节点类型筛选"},
                        "keyword": {"type": "string", "description": "搜索关键词（search模式使用）"},
                        "depth": {"type": "integer", "default": 2, "description": "子图遍历深度（subgraph模式使用）"},
                        "page": {"type": "integer", "default": 1},
                        "page_size": {"type": "integer", "default": 20},
                        "sort_by": {"type": "string", "default": "created_at"},
                        "sort_order": {"type": "string", "enum": ["asc", "desc"], "default": "desc"},
                        "include_deleted": {"type": "boolean", "default": False, "description": "是否包含软删除节点"},
                    },
                    "required": ["mode"],
                },
            },
        },
    ]


def dispatch_crud_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    task_graph: Any,
    session_id: str = "",
    ws_sender: Optional[Callable] = None,
) -> Dict[str, Any]:
    if task_graph is None:
        return {"success": False, "error_code": "TASK_GRAPH_NOT_INITIALIZED", "error": "当前会话无任务图谱"}

    if tool_name not in CRUD_TOOL_NAMES:
        return {"success": False, "error_code": "UNKNOWN_TOOL", "error": f"未知CRUD工具: {tool_name}"}

    from zulong.ide.graph_crud_executor import GraphCRUDExecutor
    executor = GraphCRUDExecutor(task_graph=task_graph, session_id=session_id, ws_sender=ws_sender)

    dispatch_map = {
        "graph_create_node": executor.execute_create_node,
        "graph_create_edge": executor.execute_create_edge,
        "graph_batch_create": executor.execute_batch_create,
        "graph_delete_node": executor.execute_delete_node,
        "graph_delete_edge": executor.execute_delete_edge,
        "graph_update_node": executor.execute_update_node,
        "graph_batch_update": executor.execute_batch_update,
        "graph_query_nodes": executor.execute_query_nodes,
    }

    handler = dispatch_map.get(tool_name)
    if not handler:
        return {"success": False, "error_code": "UNKNOWN_TOOL", "error": f"未实现: {tool_name}"}

    try:
        result = handler(arguments)
        return result.to_dict()
    except Exception as e:
        import logging
        logging.getLogger(__name__).error("CRUD工具执行异常: %s - %s", tool_name, e)
        return {"success": False, "error_code": "INTERNAL_ERROR", "error": str(e)}
