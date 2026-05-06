# File: zulong/tools/code_anchor_tools.py
# CodeAnchor FC 工具集 — 让模型通过 Function Calling 管理代码锚点
#
# 3 个工具:
# - zulong_memory_write_with_code: 保存记忆并关联代码位置
# - zulong_code_query: 查询代码位置相关的记忆/任务/经验
# - zulong_task_link_code: 将任务节点关联到实现代码

import logging
import time
from typing import Dict, Any, List, Optional

from .base import BaseTool, ToolCategory, ToolRequest, ToolResult

logger = logging.getLogger(__name__)


class MemoryWriteWithCodeTool(BaseTool):
    """zulong_memory_write_with_code — 保存记忆并关联代码位置

    原子操作：创建一个记忆节点，同时将其关联到具体的代码位置（文件+符号+行范围）。
    比 save_memory_note + 手动关联更高效，适用于记录代码相关的决策/知识/经验。
    """

    def __init__(self):
        super().__init__(name="zulong_memory_write_with_code", category=ToolCategory.CUSTOM)
        self.description = (
            "保存记忆并关联到具体代码位置。当你发现重要的代码设计决策、"
            "踩坑经验、或需要记住某段代码的上下文信息时调用此工具。"
            "会同时创建记忆节点和代码锚点，实现双向关联。"
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
        code_refs = request.parameters.get("code_refs", [])

        # 参数验证
        if not content:
            return self._create_result(
                success=False,
                error="content 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        if not code_refs or not isinstance(code_refs, list):
            return self._create_result(
                success=False,
                error="code_refs 必须是非空数组",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        # 验证每个 code_ref 至少有 file_path
        for i, ref in enumerate(code_refs):
            if not isinstance(ref, dict) or not ref.get("file_path"):
                return self._create_result(
                    success=False,
                    error=f"code_refs[{i}] 必须包含 file_path 字段",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

        if not label:
            label = content[:50]

        try:
            # 1. 创建 MemoryGraph 节点（复用 save_memory_note 的逻辑）
            from zulong.memory.memory_graph import (
                get_memory_graph, GraphNode, NodeType, Importance, EdgeType,
            )
            mg = get_memory_graph()
            if mg is None:
                return self._create_result(
                    success=False,
                    error="MemoryGraph 未初始化",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # 映射重要性
            importance_map = {
                "trivial": Importance.TRIVIAL,
                "normal": Importance.NORMAL,
                "identity": Importance.IDENTITY,
                "fact": Importance.FACT,
                "important": Importance.IMPORTANT,
                "must_remember": Importance.MUST_REMEMBER,
            }
            imp_level = importance_map.get(importance.lower(), Importance.NORMAL)

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
                    "source": "code_anchor",
                },
            )

            mg.add_node(node)
            mg.set_importance(node_id, imp_level)
            mg.index_summary(node_id, f"{label} {content}")

            # 与当前焦点节点建立关联
            ctx = mg.get_last_focus_context()
            if ctx and ctx.get("focused_task_node_id"):
                mg.add_edge(
                    ctx["focused_task_node_id"],
                    node_id,
                    EdgeType.REFERENCE,
                    weight=0.6,
                )

            # 2. 创建 CodeAnchor 并关联
            from zulong.memory.code_anchor import (
                create_code_anchor, compute_content_hash, build_code_ref_summary,
            )

            owner_ref = f"mg:{node_id}"
            created_anchors = []

            for ref in code_refs:
                file_path = ref["file_path"]
                symbol = ref.get("symbol")
                line_start = ref.get("line_start")
                line_end = ref.get("line_end")
                anchor_type = ref.get("anchor_type", "implementation")
                snippet_preview = ref.get("snippet_preview", "")

                # 计算内容哈希
                content_hash = None
                if snippet_preview:
                    content_hash = compute_content_hash(snippet_preview)

                anchor = create_code_anchor(
                    file_path=file_path,
                    owner_ref=owner_ref,
                    anchor_type=anchor_type,
                    symbol=symbol,
                    line_start=line_start,
                    line_end=line_end,
                    snippet_preview=snippet_preview,
                    content_hash=content_hash,
                )
                created_anchors.append(anchor)

            # 3. 更新节点 metadata
            anchor_ids = [a.id for a in created_anchors]
            code_ref_summary = build_code_ref_summary(created_anchors)

            # 更新 GraphNode metadata
            node_data = mg._graph.nodes.get(node_id)
            if node_data:
                meta = node_data.get("node_obj")
                if meta and hasattr(meta, "metadata"):
                    meta.metadata["code_anchors"] = anchor_ids
                    meta.metadata["code_ref_summary"] = code_ref_summary

            # 4. 自动语义关联
            _semantic_count = 0
            try:
                sem_neighbors = mg.discover_semantic_neighbors(
                    node_id, top_k=3, threshold=0.75,
                )
                _semantic_count = len(sem_neighbors)
            except Exception as _sem_err:
                logger.debug(f"[memory_write_with_code] 语义关联跳过: {_sem_err}")

            logger.info(
                f"[memory_write_with_code] 节点 {node_id} + {len(created_anchors)} 锚点"
            )

            return self._create_result(
                success=True,
                data={
                    "node_id": node_id,
                    "label": label,
                    "importance": imp_level.value,
                    "anchor_count": len(created_anchors),
                    "anchor_ids": anchor_ids,
                    "code_ref_summary": code_ref_summary,
                    "semantic_links": _semantic_count,
                    "message": f"记忆已保存并关联 {len(created_anchors)} 个代码锚点",
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[memory_write_with_code] 失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"保存失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "记忆内容（决策原因、经验教训、设计说明等）",
                },
                "label": {
                    "type": "string",
                    "description": "简短标签（默认取 content 前 50 字）",
                },
                "importance": {
                    "type": "string",
                    "enum": ["trivial", "normal", "fact", "important", "must_remember"],
                    "description": "重要性级别（默认 normal）",
                },
                "code_refs": {
                    "type": "array",
                    "description": "关联的代码位置列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "文件路径（相对项目根）",
                            },
                            "symbol": {
                                "type": "string",
                                "description": "函数/类/变量名（最稳定的标识符）",
                            },
                            "line_start": {
                                "type": "integer",
                                "description": "起始行号",
                            },
                            "line_end": {
                                "type": "integer",
                                "description": "结束行号",
                            },
                            "anchor_type": {
                                "type": "string",
                                "enum": ["implementation", "affected", "created", "deleted"],
                                "description": "锚点类型（默认 implementation）",
                            },
                            "snippet_preview": {
                                "type": "string",
                                "description": "代码预览（前2-3行）",
                            },
                        },
                        "required": ["file_path"],
                    },
                },
            },
            "required": ["content", "code_refs"],
        }


class CodeQueryTool(BaseTool):
    """zulong_code_query — 查询代码位置相关的记忆/任务/经验

    给定文件路径（可选符号/行范围），查找所有关联的记忆节点和任务节点。
    这是唯一返回完整锚点详情的工具（normal recall_memory 只返回一行摘要）。
    """

    def __init__(self):
        super().__init__(name="zulong_code_query", category=ToolCategory.CUSTOM)
        self.description = (
            "查询某段代码相关的记忆、任务和经验。"
            "当你需要了解一段代码的历史背景（为什么这样写？之前改过吗？相关的决策是什么？）"
            "或者准备修改代码前想知道有什么注意事项时调用。"
            "提供文件路径，可选提供函数/类名或行范围来精确定位。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        file_path = request.parameters.get("file_path", "")
        symbol = request.parameters.get("symbol")
        line_range = request.parameters.get("line_range")

        if not file_path:
            return self._create_result(
                success=False,
                error="file_path 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        try:
            from zulong.memory.code_anchor import get_code_anchor_store

            store = get_code_anchor_store()

            # 查询匹配的锚点
            line_start = None
            line_end = None
            if line_range and isinstance(line_range, dict):
                line_start = line_range.get("start")
                line_end = line_range.get("end")

            anchors = store.query_by_file_and_symbol(
                file_path=file_path,
                symbol=symbol,
                line_start=line_start,
                line_end=line_end,
            )

            if not anchors:
                return self._create_result(
                    success=True,
                    data={
                        "file_path": file_path,
                        "symbol": symbol,
                        "memories": [],
                        "tasks": [],
                        "message": "未找到与此代码位置相关的记忆或任务",
                    },
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # 按 owner_ref 分组并解析关联节点
            memories = []
            tasks = []
            seen_owners = set()

            for anchor in anchors:
                if anchor.owner_ref in seen_owners:
                    continue
                seen_owners.add(anchor.owner_ref)

                if anchor.owner_ref.startswith("mg:"):
                    # 解析 MemoryGraph 节点
                    node_id = anchor.owner_ref[3:]
                    node_info = self._resolve_memory_node(node_id)
                    if node_info:
                        node_info["anchor"] = {
                            "type": anchor.anchor_type,
                            "symbol": anchor.symbol,
                            "lines": f"L{anchor.line_start}-{anchor.line_end}" if anchor.line_start else None,
                            "snippet": anchor.snippet_preview[:100] if anchor.snippet_preview else None,
                        }
                        memories.append(node_info)

                elif anchor.owner_ref.startswith("tg:"):
                    # 解析 TaskGraph 节点
                    task_info = self._resolve_task_node(anchor.owner_ref[3:])
                    if task_info:
                        task_info["anchor"] = {
                            "type": anchor.anchor_type,
                            "symbol": anchor.symbol,
                            "lines": f"L{anchor.line_start}-{anchor.line_end}" if anchor.line_start else None,
                        }
                        tasks.append(task_info)

            return self._create_result(
                success=True,
                data={
                    "file_path": file_path,
                    "symbol": symbol,
                    "total_anchors": len(anchors),
                    "memories": memories,
                    "tasks": tasks,
                    "message": f"找到 {len(memories)} 条记忆, {len(tasks)} 个任务关联",
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[code_query] 查询失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"查询失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _resolve_memory_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """解析 MemoryGraph 节点信息"""
        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            if mg is None:
                return None

            node_data = mg._graph.nodes.get(node_id)
            if not node_data:
                return None

            node_obj = node_data.get("node_obj")
            if not node_obj:
                return None

            return {
                "node_id": node_id,
                "type": node_obj.node_type.value if hasattr(node_obj.node_type, "value") else str(node_obj.node_type),
                "label": node_obj.label,
                "content": node_obj.metadata.get("content", ""),
                "importance": node_obj.metadata.get("importance", "normal"),
            }
        except Exception:
            return None

    def _resolve_task_node(self, task_ref: str) -> Optional[Dict[str, Any]]:
        """解析 TaskGraph 节点信息
        
        task_ref 格式: "{graph_id}/{task_node_id}"
        """
        try:
            from .task_tools import get_active_task_graph
            tg = get_active_task_graph()
            if tg is None:
                return None

            # 解析 task_ref
            parts = task_ref.split("/", 1)
            task_node_id = parts[-1] if parts else task_ref

            node = tg.get_node(task_node_id)
            if not node:
                return None

            return {
                "task_id": node.id,
                "label": node.label,
                "status": node.status,
                "type": node.type,
                "desc": node.desc[:200] if node.desc else "",
            }
        except Exception:
            return None

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "要查询的文件路径（相对项目根）",
                },
                "symbol": {
                    "type": "string",
                    "description": "函数/类/变量名（可选，用于精确定位）",
                },
                "line_range": {
                    "type": "object",
                    "description": "行范围（可选）",
                    "properties": {
                        "start": {"type": "integer", "description": "起始行"},
                        "end": {"type": "integer", "description": "结束行"},
                    },
                },
            },
            "required": ["file_path"],
        }


class TaskLinkCodeTool(BaseTool):
    """zulong_task_link_code — 将任务节点关联到实现代码

    当模型完成某个任务的代码实现后，调用此工具记录代码位置。
    实现任务进度的代码级粒度可视化。
    """

    def __init__(self):
        super().__init__(name="zulong_task_link_code", category=ToolCategory.CUSTOM)
        self.description = (
            "将任务关联到实现它的代码位置。"
            "当你完成了一个任务的代码实现，调用此工具记录代码在哪里。"
            "这样后续可以精确追踪每个任务对应了哪些代码修改。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        task_node_id = request.parameters.get("task_node_id", "")
        code_refs = request.parameters.get("code_refs", [])

        if not task_node_id:
            return self._create_result(
                success=False,
                error="task_node_id 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        if not code_refs or not isinstance(code_refs, list):
            return self._create_result(
                success=False,
                error="code_refs 必须是非空数组",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        try:
            from .task_tools import get_active_task_graph
            tg = get_active_task_graph()
            if tg is None:
                return self._create_result(
                    success=False,
                    error="没有活跃的 TaskGraph",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # 查找任务节点（支持模糊匹配）
            node = tg.get_node(task_node_id)
            if node is None:
                # 尝试模糊查找
                node = self._fuzzy_find_node(tg, task_node_id)
                if node is None:
                    return self._create_result(
                        success=False,
                        error=f"未找到任务节点: {task_node_id}",
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )
                task_node_id = node.id

            # 创建 CodeAnchors
            from zulong.memory.code_anchor import (
                create_code_anchor, compute_content_hash, build_code_ref_summary,
            )

            graph_id = tg.id
            owner_ref = f"tg:{graph_id}/{task_node_id}"
            created_anchors = []

            for ref in code_refs:
                if not isinstance(ref, dict) or not ref.get("file_path"):
                    continue

                file_path = ref["file_path"]
                symbol = ref.get("symbol")
                line_start = ref.get("line_start")
                line_end = ref.get("line_end")
                anchor_type = ref.get("anchor_type", "implementation")
                snippet_preview = ref.get("snippet_preview", "")

                content_hash = None
                if snippet_preview:
                    content_hash = compute_content_hash(snippet_preview)

                anchor = create_code_anchor(
                    file_path=file_path,
                    owner_ref=owner_ref,
                    anchor_type=anchor_type,
                    symbol=symbol,
                    line_start=line_start,
                    line_end=line_end,
                    snippet_preview=snippet_preview,
                    content_hash=content_hash,
                )
                created_anchors.append(anchor)

                # 自动添加粗粒度文件关联（如果不存在）
                import os
                file_name = os.path.basename(file_path)
                node.add_file(name=file_name, path=file_path)

            # 存储 anchor_ids 到 TaskGraph.metadata
            if "code_anchors" not in tg.metadata:
                tg.metadata["code_anchors"] = {}
            anchor_ids = [a.id for a in created_anchors]
            existing = tg.metadata["code_anchors"].get(task_node_id, [])
            existing.extend(anchor_ids)
            tg.metadata["code_anchors"][task_node_id] = existing

            # 触发变更通知
            if tg.on_change_callback:
                tg.on_change_callback("node_code_anchor", {
                    "node_id": task_node_id,
                    "anchor_count": len(created_anchors),
                })

            # 同步到 MemoryGraph（如果对应的 TASK 节点存在）
            code_ref_summary = build_code_ref_summary(created_anchors)
            self._sync_to_memory_graph(graph_id, task_node_id, anchor_ids, code_ref_summary)

            logger.info(
                f"[task_link_code] 任务 {task_node_id} 关联 {len(created_anchors)} 个锚点"
            )

            return self._create_result(
                success=True,
                data={
                    "task_node_id": task_node_id,
                    "anchor_count": len(created_anchors),
                    "anchor_ids": anchor_ids,
                    "code_ref_summary": code_ref_summary,
                    "message": f"已关联 {len(created_anchors)} 个代码锚点到任务 '{node.label}'",
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[task_link_code] 失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"关联失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _fuzzy_find_node(self, tg, query: str):
        """模糊查找任务节点（按 ID 或 label 部分匹配）"""
        all_nodes = tg.get_all_nodes() if hasattr(tg, "get_all_nodes") else []
        for n in all_nodes:
            if query in n.id or query.lower() in n.label.lower():
                return n
        return None

    def _sync_to_memory_graph(
        self, graph_id: str, task_node_id: str,
        anchor_ids: List[str], code_ref_summary: str
    ) -> None:
        """将锚点信息同步到 MemoryGraph 中对应的 TASK 节点"""
        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            if mg is None:
                return

            # MemoryGraph 中任务节点 ID 格式: "task:{graph_id}/{node_id}"
            mg_node_id = f"task:{graph_id}/{task_node_id}"
            node_data = mg._graph.nodes.get(mg_node_id)
            if not node_data:
                # 也尝试不带 graph_id 的格式
                mg_node_id = f"task:{task_node_id}"
                node_data = mg._graph.nodes.get(mg_node_id)

            if node_data:
                node_obj = node_data.get("node_obj")
                if node_obj and hasattr(node_obj, "metadata"):
                    existing_anchors = node_obj.metadata.get("code_anchors", [])
                    existing_anchors.extend(anchor_ids)
                    node_obj.metadata["code_anchors"] = existing_anchors
                    node_obj.metadata["code_ref_summary"] = code_ref_summary
        except Exception as e:
            logger.debug(f"[task_link_code] MemoryGraph 同步跳过: {e}")

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_node_id": {
                    "type": "string",
                    "description": "任务节点 ID（如 'o1_1', 'o1_2_1'）",
                },
                "code_refs": {
                    "type": "array",
                    "description": "关联的代码位置列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "文件路径（相对项目根）",
                            },
                            "symbol": {
                                "type": "string",
                                "description": "函数/类/变量名",
                            },
                            "line_start": {
                                "type": "integer",
                                "description": "起始行号",
                            },
                            "line_end": {
                                "type": "integer",
                                "description": "结束行号",
                            },
                            "anchor_type": {
                                "type": "string",
                                "enum": ["implementation", "affected", "created", "deleted"],
                                "description": "关联类型（默认 implementation）",
                            },
                            "snippet_preview": {
                                "type": "string",
                                "description": "代码预览（前2-3行）",
                            },
                        },
                        "required": ["file_path"],
                    },
                },
            },
            "required": ["task_node_id", "code_refs"],
        }
