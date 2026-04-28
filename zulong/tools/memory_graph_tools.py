# File: zulong/tools/memory_graph_tools.py
# MemoryGraph FC 工具集 — 让模型通过 Function Calling 自主访问图记忆系统
#
# 4 个工具:
# - recall_memory: 检索记忆（语义搜索 + 热数据遍历）
# - read_memory_node: 读取特定节点详情及邻域
# - save_memory_note: 保存笔记/观察到记忆图
# - discover_related: 从种子节点发现关联节点

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional

from .base import BaseTool, ToolCategory, ToolRequest, ToolResult

logger = logging.getLogger(__name__)


def _run_async(coro):
    """在同步上下文中执行异步协程"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # 已有事件循环运行，使用线程池
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=30)
    else:
        return asyncio.run(coro)


class RecallMemoryTool(BaseTool):
    """recall_memory — 检索记忆

    通过语义查询检索图记忆系统中的相关信息。
    使用 MemoryGraph.retrieve_context() 进行热/冷双路径检索。
    """

    def __init__(self):
        super().__init__(name="recall_memory", category=ToolCategory.CUSTOM)
        self.description = (
            "检索记忆。当你需要回忆之前的对话内容、用户偏好、"
            "历史任务信息或任何存储在记忆中的信息时调用此工具。"
            "提供一个查询描述，系统将返回最相关的记忆片段。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        query = request.parameters.get("query", "")
        top_k = request.parameters.get("top_k", 5)

        if not query:
            return self._create_result(
                success=False,
                error="query 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            if mg is None:
                return self._create_result(
                    success=False,
                    error="MemoryGraph 未初始化",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # retrieve_context 是 async，需要包装
            results = _run_async(
                mg.retrieve_context(query_text=query, top_k=top_k)
            )

            if not results:
                # 降级到关键词搜索
                results = mg.search_nodes(query=query, max_results=top_k)

            # 格式化返回
            memories = []
            for item in results[:top_k]:
                memories.append({
                    "node_id": item.get("node_id", ""),
                    "type": str(item.get("node_type", item.get("type", ""))),
                    "label": item.get("label", ""),
                    "content": item.get("content", item.get("metadata", {}).get("content", "")),
                    "score": round(item.get("score", 0), 3),
                })

            logger.info(f"[recall_memory] 查询 '{query}' 返回 {len(memories)} 条结果")

            return self._create_result(
                success=True,
                data={
                    "query": query,
                    "count": len(memories),
                    "memories": memories,
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[recall_memory] 检索失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"记忆检索失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要检索的内容描述，如'用户之前提到的项目需求'",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量（默认 5）",
                },
            },
            "required": ["query"],
        }


class ReadMemoryNodeTool(BaseTool):
    """read_memory_node — 读取特定记忆节点详情

    根据节点 ID 获取节点完整信息及其邻域概览。
    """

    def __init__(self):
        super().__init__(name="read_memory_node", category=ToolCategory.CUSTOM)
        self.description = (
            "读取特定记忆节点的详细信息。当你从 recall_memory 的结果中"
            "看到感兴趣的节点 ID，想深入了解其内容和关联时调用。"
            "返回节点的完整元数据和邻域摘要。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        node_id = request.parameters.get("node_id", "")

        if not node_id:
            return self._create_result(
                success=False,
                error="node_id 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            if mg is None:
                return self._create_result(
                    success=False,
                    error="MemoryGraph 未初始化",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            node = mg.get_node(node_id)
            if node is None:
                return self._create_result(
                    success=False,
                    error=f"节点 '{node_id}' 不存在",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # 获取邻域摘要
            subgraph = mg.get_subgraph_summary(node_id, max_depth=1)

            # 获取子节点
            children = mg.get_children(node_id)
            children_info = [
                {"node_id": c.node_id, "label": c.label, "type": c.node_type.value}
                for c in children[:10]
            ]

            # 获取父节点
            parent = mg.get_parent(node_id)
            parent_info = None
            if parent:
                parent_info = {
                    "node_id": parent.node_id,
                    "label": parent.label,
                    "type": parent.node_type.value,
                }

            # 解析后端引用
            backend_content = None
            if node.backend_ref:
                ref_data = mg.resolve_backend_ref(node_id)
                if ref_data:
                    backend_content = ref_data.get("content", "")

            result_data = {
                "node_id": node.node_id,
                "type": node.node_type.value,
                "label": node.label,
                "activation": round(node.activation, 3),
                "metadata": node.metadata,
                "parent": parent_info,
                "children_count": len(children),
                "children": children_info,
                "neighbor_count": subgraph.get("neighbor_count", 0),
            }

            if backend_content:
                result_data["content"] = backend_content[:2000]

            logger.info(f"[read_memory_node] 读取节点 {node_id}")

            return self._create_result(
                success=True,
                data=result_data,
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[read_memory_node] 读取失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"节点读取失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "要读取的节点 ID，如 'dlg:session_xxx' 或 'task:node_xxx'",
                },
            },
            "required": ["node_id"],
        }


class SaveMemoryNoteTool(BaseTool):
    """save_memory_note — 保存笔记到记忆图

    创建一个新的知识/观察节点，存入图记忆系统。
    用于模型主动记录重要发现、用户偏好等。
    """

    def __init__(self):
        super().__init__(name="save_memory_note", category=ToolCategory.CUSTOM)
        self.description = (
            "保存笔记到记忆。当你发现需要长期记住的信息时调用，"
            "如用户偏好、重要事实、任务中的关键发现等。"
            "信息会持久化到图记忆系统中，后续可通过 recall_memory 检索。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        content = request.parameters.get("content", "")
        label = request.parameters.get("label", "")
        importance = request.parameters.get("importance", "normal")

        if not content:
            return self._create_result(
                success=False,
                error="content 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        if not label:
            label = content[:50]

        try:
            from zulong.memory.memory_graph import (
                get_memory_graph, GraphNode, NodeType, Importance,
            )
            mg = get_memory_graph()
            if mg is None:
                return self._create_result(
                    success=False,
                    error="MemoryGraph 未初始化",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # 映射重要性级别
            importance_map = {
                "trivial": Importance.TRIVIAL,
                "normal": Importance.NORMAL,
                "identity": Importance.IDENTITY,
                "fact": Importance.FACT,
                "important": Importance.IMPORTANT,
                "must_remember": Importance.MUST_REMEMBER,
            }
            imp_level = importance_map.get(importance.lower(), Importance.NORMAL)

            # 创建节点
            now = time.time()
            node_id = f"note:{int(now * 1000)}"
            node = GraphNode(
                node_id=node_id,
                node_type=NodeType.KNOWLEDGE,
                label=label,
                activation=0.8,
                created_at=now,
                last_accessed=now,
                access_count=1,
                metadata={
                    "content": content,
                    "importance": imp_level.value,
                    "source": "model_note",
                },
            )

            mg.add_node(node)
            mg.set_importance(node_id, imp_level)

            # 索引到 FAISS 以支持语义搜索
            mg.index_summary(node_id, f"{label} {content}")

            # 与当前焦点节点建立关联
            ctx = mg.get_last_focus_context()
            if ctx and ctx.get("focused_task_node_id"):
                from zulong.memory.memory_graph import EdgeType
                mg.add_edge(
                    ctx["focused_task_node_id"],
                    node_id,
                    EdgeType.REFERENCE,
                    weight=0.6,
                )

            logger.info(f"[save_memory_note] 保存节点 {node_id}: {label}")

            return self._create_result(
                success=True,
                data={
                    "node_id": node_id,
                    "label": label,
                    "importance": imp_level.value,
                    "message": "笔记已保存到记忆图",
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[save_memory_note] 保存失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"笔记保存失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "要保存的内容",
                },
                "label": {
                    "type": "string",
                    "description": "简短标签（默认取 content 前 50 字）",
                },
                "importance": {
                    "type": "string",
                    "enum": ["trivial", "normal", "identity", "fact", "important", "must_remember"],
                    "description": "重要性级别（默认 normal）",
                },
            },
            "required": ["content"],
        }


class DiscoverRelatedTool(BaseTool):
    """discover_related — 发现关联节点

    从指定节点出发，探索其图邻域中的关联信息。
    适合发现隐藏的上下文关联和知识连接。
    """

    def __init__(self):
        super().__init__(name="discover_related", category=ToolCategory.CUSTOM)
        self.description = (
            "从一个记忆节点出发，发现与之关联的其他节点。"
            "当你想了解某个话题的更广泛上下文、"
            "或探索与当前任务相关的历史信息时调用。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        node_id = request.parameters.get("node_id", "")
        max_depth = request.parameters.get("max_depth", 2)

        if not node_id:
            return self._create_result(
                success=False,
                error="node_id 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            if mg is None:
                return self._create_result(
                    success=False,
                    error="MemoryGraph 未初始化",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            if not mg.has_node(node_id):
                return self._create_result(
                    success=False,
                    error=f"节点 '{node_id}' 不存在",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # 图邻域搜索
            neighbors = mg.get_neighbors(node_id, max_depth=min(max_depth, 3))

            # 尝试语义邻居发现
            semantic_neighbors = []
            try:
                semantic_neighbors = mg.discover_semantic_neighbors(
                    node_id, top_k=3, threshold=0.75
                )
            except Exception:
                pass

            # 格式化
            related = []
            seen = set()
            for n in neighbors:
                if n.node_id in seen or n.node_id == node_id:
                    continue
                seen.add(n.node_id)
                edge = mg.get_edge(node_id, n.node_id) or mg.get_edge(n.node_id, node_id)
                related.append({
                    "node_id": n.node_id,
                    "type": n.node_type.value,
                    "label": n.label,
                    "activation": round(n.activation, 3),
                    "edge_type": edge.get("edge_type", "unknown") if edge else "indirect",
                })

            # 补充语义邻居
            for sem_id, score in semantic_neighbors:
                if sem_id in seen or sem_id == node_id:
                    continue
                seen.add(sem_id)
                sem_node = mg.get_node(sem_id)
                if sem_node:
                    related.append({
                        "node_id": sem_node.node_id,
                        "type": sem_node.node_type.value,
                        "label": sem_node.label,
                        "activation": round(sem_node.activation, 3),
                        "edge_type": "semantic",
                        "similarity": round(score, 3),
                    })

            logger.info(f"[discover_related] 节点 {node_id} 发现 {len(related)} 个关联")

            return self._create_result(
                success=True,
                data={
                    "seed_node_id": node_id,
                    "count": len(related),
                    "related": related[:20],
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[discover_related] 发现失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"关联发现失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "起始节点 ID",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "图遍历最大深度（默认 2，最大 3）",
                },
            },
            "required": ["node_id"],
        }
