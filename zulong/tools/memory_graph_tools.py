# File: zulong/tools/memory_graph_tools.py
# MemoryGraph FC 工具集 — 让模型通过 Function Calling 自主访问图记忆系统
#
# 8 个工具:
# - recall_memory: 检索记忆（语义搜索 + 热数据遍历）
# - read_memory_node: 读取特定节点详情及邻域
# - save_memory_note: 保存笔记/观察到记忆图
# - discover_related: 从种子节点发现关联节点
# - activate_memory_network: BFS 扩散激活记忆网络（P1-2）
# - list_memory: 按类型/温度/重要性浏览记忆节点（P1-3）
# - set_importance: 设置节点重要性级别（P2-2）
# - delete_memory_node: 删除记忆节点（P2-8）

import logging
import time
import asyncio
import threading
import re
import json
from typing import Dict, Any, List, Optional

from .base import BaseTool, ToolCategory, ToolRequest, ToolResult

logger = logging.getLogger(__name__)


# ── BFS 触发机制（线程安全）───────────────────────────────────
# 使用 thread-local 存储避免竞态条件
_bfs_local = threading.local()
_bfs_global_lock = threading.Lock()
_bfs_global_triggered = False

def _set_bfs_memory_trigger():
    """设置BFS记忆触发标记，供FC循环检查（线程安全）"""
    global _bfs_global_triggered
    _bfs_local.bfs_triggered = True
    with _bfs_global_lock:
        _bfs_global_triggered = True
    logger.debug("[BFS Trigger] 记忆工具触发BFS扩散标记已设置")

def get_bfs_memory_trigger() -> bool:
    """获取并重置BFS记忆触发标记（线程安全）"""
    global _bfs_global_triggered
    # 优先检查 thread-local 标记
    triggered = getattr(_bfs_local, 'bfs_triggered', False)
    _bfs_local.bfs_triggered = False
    # 同时检查全局标记
    with _bfs_global_lock:
        if _bfs_global_triggered:
            triggered = True
            _bfs_global_triggered = False
    return triggered
# ────────────────────────────────────────────────────


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


def _enum_value(value: Any) -> str:
    """兼容 Enum 和 Hybrid 存储里的字符串类型。"""
    return str(getattr(value, "value", value) or "")


def _node_metadata(node: Any) -> Dict[str, Any]:
    meta = getattr(node, "metadata", None)
    return dict(meta or {})


def _node_type_value(node: Any) -> str:
    return _enum_value(getattr(node, "node_type", ""))


def _safe_round(value: Any, digits: int = 3) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def _safe_get_shard_id(mg: Any, node_id: str, node: Any = None, item: Optional[Dict[str, Any]] = None) -> str:
    if item and item.get("shard_id"):
        return str(item["shard_id"])
    if node is not None:
        storage_shard = getattr(node, "storage_shard", "")
        if storage_shard:
            return str(storage_shard)
    try:
        global_index = getattr(mg, "global_index", None)
        if global_index is not None:
            shard_id = global_index.get_node_shard(node_id)
            if shard_id:
                return str(shard_id)
    except Exception:
        pass
    try:
        finder = getattr(mg, "_find_node_shard_id", None)
        if callable(finder):
            shard_id = finder(node_id)
            if shard_id:
                return str(shard_id)
    except Exception:
        pass
    return ""


def _build_memory_address(
    mg: Any,
    node_id: str,
    node: Any = None,
    item: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构造 LLM 可回跳的图记忆地址，不改变既有工具协议。"""
    item = item or {}
    meta = dict(item.get("metadata") or _node_metadata(node))
    node_type = str(
        item.get("node_type")
        or item.get("type")
        or getattr(getattr(node, "node_type", ""), "value", getattr(node, "node_type", ""))
        or ""
    ).lower()
    if node_type == "dialogue":
        meta["task_graph_address"] = ""
        full_path_value = str(meta.get("full_path") or "")
        if full_path_value.startswith("task:") or "/task:" in full_path_value:
            meta["full_path"] = node_id
    graph_memory_id = (
        item.get("graph_memory_id")
        or meta.get("graph_memory_id")
        or node_id
    )
    shard_id = _safe_get_shard_id(mg, node_id, node=node, item=item)
    full_path = (
        item.get("full_path")
        or meta.get("full_path")
        or meta.get("graph_address")
        or meta.get("task_graph_address")
        or node_id
    )
    source_node_ids = item.get("source_node_ids") or meta.get("source_node_ids") or [node_id]
    if isinstance(source_node_ids, str):
        source_node_ids = [source_node_ids]

    return {
        "graph_memory_id": str(graph_memory_id or node_id),
        "node_id": node_id,
        "shard_id": shard_id,
        "full_path": str(full_path or node_id),
        "source_node_ids": source_node_ids,
        "backend_ref": item.get("backend_ref") or getattr(node, "backend_ref", "") or meta.get("backend_ref", ""),
        "summary_ref_id": item.get("summary_ref_id") or meta.get("summary_ref_id", ""),
        "source": item.get("source") or meta.get("source", ""),
    }


def _attach_address_fields(
    entry: Dict[str, Any],
    mg: Any,
    node_id: str,
    node: Any = None,
    item: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    address = _build_memory_address(mg, node_id, node=node, item=item)
    entry["graph_memory_id"] = address["graph_memory_id"]
    entry["shard_id"] = address["shard_id"]
    entry["full_path"] = address["full_path"]
    entry["memory_address"] = address
    entry["recall_hint"] = (
        "如需展开详情，可用 graph_memory_id 调用 read_memory_node；"
        "如需关联扩散，可用 graph_memory_id 调用 discover_related。"
    )
    return entry


def _safe_get_children(mg: Any, node_id: str, edge_type=None) -> List[Any]:
    try:
        if edge_type is not None:
            return list(mg.get_children(node_id, edge_type=edge_type))
        return list(mg.get_children(node_id))
    except TypeError:
        try:
            return list(mg.get_children(node_id))
        except Exception:
            return []
    except Exception:
        return []


def _safe_get_parent(mg: Any, node_id: str) -> Optional[Any]:
    try:
        parent = mg.get_parent(node_id)
        if parent:
            return parent
    except Exception:
        pass
    try:
        finder = getattr(mg, "_find_parent_id", None)
        if callable(finder):
            parent_id = finder(node_id)
            if parent_id:
                return mg.get_node(parent_id)
    except Exception:
        pass
    return None


def _safe_get_edge(mg: Any, src_id: str, dst_id: str) -> Optional[Dict[str, Any]]:
    try:
        edge = mg.get_edge(src_id, dst_id)
        if edge:
            return dict(edge)
    except Exception:
        pass
    try:
        for edge in getattr(mg, "_get_cross_edges_from")(src_id):
            if edge.get("dst_id") == dst_id:
                return dict(edge)
    except Exception:
        pass
    return None


def _safe_get_neighbors(mg: Any, node_id: str, max_depth: int = 1) -> List[Any]:
    try:
        return list(mg.get_neighbors(node_id, max_depth=max_depth))
    except Exception:
        pass

    nodes: List[Any] = []
    seen = {node_id}
    try:
        node, shard_id = mg._get_node_with_shard(node_id)
        if shard_id and hasattr(mg, "discover_across_shards"):
            for nid, _dist, _sid in mg.discover_across_shards(
                seed_ids=[node_id],
                seed_shard_id=shard_id,
                max_depth=max_depth,
                max_nodes=50,
            ):
                if nid in seen:
                    continue
                seen.add(nid)
                n = mg.get_node(nid)
                if n:
                    nodes.append(n)
            return nodes
    except Exception:
        pass

    for n in _safe_get_children(mg, node_id):
        if getattr(n, "node_id", "") not in seen:
            seen.add(n.node_id)
            nodes.append(n)
    parent = _safe_get_parent(mg, node_id)
    if parent and getattr(parent, "node_id", "") not in seen:
        nodes.append(parent)
    return nodes


def _iter_memory_nodes(mg: Any) -> List[Any]:
    """兼容 NetworkX MemoryGraph 和 ShardedMemoryGraph 的节点遍历。"""
    if hasattr(mg, "_nodes"):
        try:
            return list(mg._nodes.values())
        except Exception:
            pass
    nodes = []
    try:
        for shard_id in mg.list_all_shards():
            shard = mg.get_shard(shard_id, load_if_missing=True)
            if not shard:
                continue
            for node in shard.properties.iter_nodes():
                nodes.append(node)
    except Exception:
        pass
    return nodes


def _safe_get_temperature(mg: Any, node: Any) -> str:
    try:
        temp = mg.get_temperature(node.node_id)
        return _enum_value(temp) or "unknown"
    except Exception:
        return _enum_value(getattr(node, "temperature", "")) or _node_metadata(node).get("temperature", "unknown")


def _safe_get_importance(mg: Any, node: Any) -> str:
    try:
        imp = mg.get_importance(node.node_id)
        return _enum_value(imp) or "normal"
    except Exception:
        return (
            _node_metadata(node).get("importance")
            or _enum_value(getattr(node, "importance", ""))
            or "normal"
        )


def _safe_subgraph_summary(mg: Any, node_id: str, max_depth: int = 1) -> Dict[str, Any]:
    try:
        return mg.get_subgraph_summary(node_id, max_depth=max_depth) or {}
    except Exception:
        neighbors = _safe_get_neighbors(mg, node_id, max_depth=max_depth)
        type_counts: Dict[str, int] = {}
        for n in neighbors:
            ntype = _node_type_value(n)
            type_counts[ntype] = type_counts.get(ntype, 0) + 1
        return {
            "neighbor_count": len(neighbors),
            "type_distribution": type_counts,
        }


def _brief_node(mg: Any, node: Any) -> Dict[str, Any]:
    node_id = getattr(node, "node_id", "")
    data = {
        "node_id": node_id,
        "type": _node_type_value(node),
        "label": getattr(node, "label", ""),
        "activation": _safe_round(getattr(node, "activation", 0.0), 3),
    }
    return _attach_address_fields(data, mg, node_id, node=node)


def _normalize_batch_memory_entries(
    content: str,
    raw_entries: Any = None,
) -> List[Dict[str, str]]:
    """Return structured memory entries while preserving single-note behavior."""
    entries: List[Dict[str, str]] = []
    if isinstance(raw_entries, list):
        for idx, item in enumerate(raw_entries, start=1):
            if isinstance(item, dict):
                item_content = str(item.get("content") or item.get("text") or "").strip()
                item_label = str(item.get("label") or "").strip()
            else:
                item_content = str(item or "").strip()
                item_label = ""
            if item_content:
                entries.append({
                    "label": item_label or item_content[:50],
                    "content": item_content,
                    "source_index": str(idx),
                })
        if entries:
            return entries

    lines = []
    for raw_line in str(content or "").splitlines():
        line = re.sub(r"^\s*(?:[-*+]|\d+[.)]|[（(]?\d+[）)])\s*", "", raw_line).strip()
        if line:
            lines.append(line)
    marker_re = re.compile(r"\bZL-[A-Z0-9-]+-\d{2,}\b", re.IGNORECASE)
    marked_lines = [line for line in lines if marker_re.search(line)]
    if len(marked_lines) >= 2:
        return [
            {
                "label": (marker_re.search(line).group(0) if marker_re.search(line) else line[:50]),
                "content": line,
                "source_index": str(idx),
            }
            for idx, line in enumerate(marked_lines, start=1)
        ]

    marker_matches = list(marker_re.finditer(str(content or "")))
    if len(marker_matches) >= 2:
        text = str(content or "")
        parts = []
        for idx, match in enumerate(marker_matches):
            start = match.start()
            end = marker_matches[idx + 1].start() if idx + 1 < len(marker_matches) else len(text)
            part = text[start:end].strip(" \n\r\t;；,，")
            if part:
                parts.append((match.group(0), part))
        if len(parts) >= 2:
            return [
                {"label": marker, "content": part, "source_index": str(idx)}
                for idx, (marker, part) in enumerate(parts, start=1)
            ]

    return []


def _strong_recall_tokens(query: str) -> List[str]:
    """Extract stable markers such as ZL-MEM-... for exact recall promotion."""
    text = str(query or "")
    tokens: List[str] = []
    for marker in re.findall(r"\bZL-[A-Z0-9-]+(?:-\d{2,})?\b", text, flags=re.IGNORECASE):
        if marker not in tokens:
            tokens.append(marker)
    for token in re.findall(r"\b[A-Za-z0-9][A-Za-z0-9_-]{2,}\b", text):
        if len(token) >= 4 and token not in tokens:
            tokens.append(token)
    return tokens


def _memory_item_search_text(item: Dict[str, Any], node: Any = None) -> str:
    meta = dict(item.get("metadata") or _node_metadata(node))
    parts = [
        item.get("node_id", ""),
        item.get("graph_memory_id", ""),
        item.get("label", ""),
        item.get("content", ""),
        item.get("summary", ""),
        getattr(node, "node_id", "") if node is not None else "",
        getattr(node, "label", "") if node is not None else "",
        getattr(node, "content", "") if node is not None else "",
        getattr(node, "content_summary", "") if node is not None else "",
        json.dumps(meta, ensure_ascii=False, sort_keys=True) if meta else "",
    ]
    return " ".join(str(part or "") for part in parts).lower()


def _exact_recall_rank(query: str, item: Dict[str, Any], node: Any = None) -> float:
    search_text = _memory_item_search_text(item, node=node)
    query_lower = str(query or "").strip().lower()
    tokens = _strong_recall_tokens(query)
    has_match = bool(query_lower and query_lower in search_text)
    if not has_match:
        has_match = any(token.lower() in search_text for token in tokens)
    if not has_match:
        return 0.0

    meta = dict(item.get("metadata") or _node_metadata(node))
    node_type = _enum_value(item.get("node_type") or item.get("type") or getattr(node, "node_type", ""))
    source = str(item.get("source") or meta.get("source") or "").lower()
    node_id = str(item.get("node_id") or getattr(node, "node_id", ""))

    rank = 10.0 + _safe_round(item.get("score", 0.0), 4)
    if node_type == "knowledge":
        rank += 5.0
    if source in {"model_note", "model_note_batch_entry"}:
        rank += 4.0
    if meta.get("importance") == "must_remember":
        rank += 2.0
    if any(token.lower() in str(item.get("label", "")).lower() for token in tokens):
        rank += 2.0
    if node_id.startswith("dialogue:"):
        rank -= 2.0
    return rank


def _merge_exact_recall_results(
    mg: Any,
    query: str,
    results: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Promote exact marker hits without changing the recall_memory contract."""
    try:
        keyword_results = mg.search_nodes(query=query, max_results=max(top_k * 3, 10))
    except Exception:
        keyword_results = []
    if not keyword_results:
        return results

    ranked = []
    for item in keyword_results:
        node_id = item.get("node_id") or item.get("graph_memory_id") or ""
        node = mg.get_node(node_id) if node_id else None
        rank = _exact_recall_rank(query, item, node=node)
        if rank > 0:
            exact_item = dict(item)
            exact_item["score"] = max(float(item.get("score", 0.0) or 0.0), rank)
            exact_item["source"] = "keyword_exact"
            ranked.append((rank, exact_item))
    if not ranked:
        return results

    merged: List[Dict[str, Any]] = []
    seen = set()
    for _rank, item in sorted(ranked, key=lambda pair: (-pair[0], str(pair[1].get("node_id", "")))):
        node_id = item.get("node_id") or item.get("graph_memory_id") or ""
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        merged.append(item)
    for item in results or []:
        node_id = item.get("node_id") or item.get("graph_memory_id") or ""
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        merged.append(item)
    return merged


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
        try:
            top_k = min(max(int(request.parameters.get("top_k", 5) or 5), 1), 20)
        except (TypeError, ValueError):
            top_k = 5
        seed_node_id = (
            request.parameters.get("seed_node_id")
            or request.parameters.get("graph_memory_id")
            or request.parameters.get("node_id")
        )
        try:
            expand_depth = min(max(int(request.parameters.get("expand_depth", 0) or 0), 0), 3)
        except (TypeError, ValueError):
            expand_depth = 0

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
            else:
                results = _merge_exact_recall_results(mg, query, list(results or []), top_k)

            if seed_node_id:
                if not mg.has_node(seed_node_id):
                    return self._create_result(
                        success=False,
                        error=f"种子节点 '{seed_node_id}' 不存在",
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )
                seed_node = mg.get_node(seed_node_id)
                seed_item = {
                    "node_id": seed_node_id,
                    "node_type": _node_type_value(seed_node),
                    "label": getattr(seed_node, "label", ""),
                    "content": (
                        getattr(seed_node, "content", None)
                        or _node_metadata(seed_node).get("content")
                        or _node_metadata(seed_node).get("content_summary")
                        or _node_metadata(seed_node).get("summary")
                        or getattr(seed_node, "label", "")
                    ),
                    "score": 1.0,
                    "source": "seed",
                    "metadata": _node_metadata(seed_node),
                }
                seed_results = [seed_item]
                if expand_depth > 0:
                    for neighbor in _safe_get_neighbors(mg, seed_node_id, max_depth=expand_depth):
                        seed_results.append({
                            "node_id": getattr(neighbor, "node_id", ""),
                            "node_type": _node_type_value(neighbor),
                            "label": getattr(neighbor, "label", ""),
                            "content": (
                                getattr(neighbor, "content", None)
                                or _node_metadata(neighbor).get("content")
                                or _node_metadata(neighbor).get("content_summary")
                                or _node_metadata(neighbor).get("summary")
                                or getattr(neighbor, "label", "")
                            ),
                            "score": 0.8,
                            "source": "seed_bfs",
                            "metadata": _node_metadata(neighbor),
                        })
                seen_seed_ids = set()
                ordered = []
                for item in seed_results + list(results or []):
                    nid = item.get("node_id", "")
                    if not nid or nid in seen_seed_ids:
                        continue
                    seen_seed_ids.add(nid)
                    ordered.append(item)
                results = ordered

            # 格式化返回
            memories = []
            for item in results[:top_k]:
                node_id = item.get("node_id", "")
                node_obj = mg.get_node(node_id) if node_id else None
                entry = {
                    "node_id": node_id,
                    "type": str(item.get("node_type", item.get("type", ""))),
                    "label": item.get("label", ""),
                    "content": item.get("content", item.get("metadata", {}).get("content", "")),
                    "score": round(item.get("score", 0), 3),
                    "source": item.get("source", ""),
                }
                # 附带代码锚点摘要（如有）
                meta = item.get("metadata", {})
                code_ref = meta.get("code_ref_summary", "")
                if code_ref:
                    entry["code_ref"] = code_ref
                _attach_address_fields(entry, mg, node_id, node=node_obj, item=item)
                memories.append(entry)

            logger.info(f"[recall_memory] 查询 '{query}' 返回 {len(memories)} 条结果")

            # 🔥 BFS触发标记：记忆检索工具调用后触发BFS扩散
            _set_bfs_memory_trigger()

            return self._create_result(
                success=True,
                data={
                    "query": query,
                    "count": len(memories),
                    "memories": memories,
                    "addressing": {
                        "graph_memory_id": "每条 memories[*].graph_memory_id 都可作为 read_memory_node/discover_related 的 node_id 使用",
                        "incremental_recall": "需要更多细节时，用 graph_memory_id 作为种子调用 read_memory_node 或 discover_related",
                        "seed_node_id": seed_node_id or "",
                        "expand_depth": expand_depth,
                    },
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
                    "description": "要检索的内容描述，例如“用户之前提到的项目需求”或“上次讨论的偏好”。",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量。",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                },
                "seed_node_id": {
                    "type": "string",
                    "description": "可选：从某个 graph_memory_id/node_id 作为种子做增量回忆",
                },
                "graph_memory_id": {
                    "type": "string",
                    "description": "可选：等同 seed_node_id，来自 recall_memory 返回的图记忆地址",
                },
                "expand_depth": {
                    "type": "integer",
                    "description": "可选：种子增量回忆的 BFS 深度。",
                    "default": 0,
                    "minimum": 0,
                    "maximum": 3,
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
        node_id = request.parameters.get("node_id", "") or request.parameters.get("graph_memory_id", "")

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
            subgraph = _safe_subgraph_summary(mg, node_id, max_depth=1)

            # 获取子节点
            children = _safe_get_children(mg, node_id)
            children_info = [_brief_node(mg, c) for c in children[:10]]

            # 获取父节点
            parent = _safe_get_parent(mg, node_id)
            parent_info = None
            if parent:
                parent_info = _brief_node(mg, parent)

            # 解析后端引用
            backend_content = None
            if getattr(node, "backend_ref", ""):
                resolver = getattr(mg, "resolve_backend_ref", None)
                ref_data = resolver(node_id) if callable(resolver) else None
                if ref_data:
                    backend_content = ref_data.get("content", "")

            result_data = {
                "node_id": node.node_id,
                "type": _node_type_value(node),
                "label": node.label,
                "activation": _safe_round(getattr(node, "activation", 0.0), 3),
                "metadata": _node_metadata(node),
                "parent": parent_info,
                "children_count": len(children),
                "children": children_info,
                "neighbor_count": subgraph.get("neighbor_count", 0),
            }
            _attach_address_fields(result_data, mg, node_id, node=node)

            content = (
                getattr(node, "content", None)
                or _node_metadata(node).get("content")
                or _node_metadata(node).get("content_summary")
                or _node_metadata(node).get("summary")
            )
            if content:
                result_data["content"] = str(content)[:2000]

            if backend_content:
                result_data["content"] = backend_content[:2000]

            # 加载完整代码锚点数据（如有）
            anchor_ids = _node_metadata(node).get("code_anchors", [])
            if anchor_ids:
                try:
                    from zulong.memory.code_anchor import get_code_anchor_store
                    store = get_code_anchor_store()
                    anchors_data = []
                    for aid in anchor_ids:
                        anchor = store.get_anchor(aid)
                        if anchor:
                            anchors_data.append(anchor.to_dict())
                    if anchors_data:
                        result_data["code_anchors"] = anchors_data
                except Exception:
                    pass

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
                    "description": "要读取的节点 ID；可直接使用 recall_memory 返回的 graph_memory_id。",
                },
                "graph_memory_id": {
                    "type": "string",
                    "description": "可选：来自 recall_memory 返回的图记忆地址，等同 node_id。",
                },
            },
            "anyOf": [
                {"required": ["node_id"]},
                {"required": ["graph_memory_id"]},
            ],
            "required": [],
        }


class SaveMemoryNoteTool(BaseTool):
    """save_memory_note — 保存笔记到记忆图

    创建一个新的知识/观察节点，存入图记忆系统。
    用于模型主动记录重要发现、用户偏好等。
    """

    def __init__(self):
        super().__init__(name="save_memory_note", category=ToolCategory.CUSTOM)
        self.description = (
            "保存笔记到记忆。当用户明确要求'记住'、'保存'、'记录'信息时调用。"
            "关键词：记住、保存、记录、存储、备忘、笔记、重要信息。"
            "适用场景：用户偏好、重要事实、关键发现、待办事项等需要长期存储的信息。"
            "注意：如果用户要求'规划'或'制定计划'，请使用task_create_plan工具。"
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
        raw_entries = request.parameters.get("entries")

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
            batch_entries = _normalize_batch_memory_entries(content, raw_entries)
            node_specs = batch_entries or [{
                "label": label,
                "content": content,
                "source_index": "1",
            }]
            batch_root_id = ""
            if batch_entries:
                batch_root_id = f"note_batch:{int(now * 1000)}"
                batch_label = label or f"批量记忆 {len(batch_entries)} 条"
                root_node = GraphNode(
                    node_id=batch_root_id,
                    node_type=NodeType.EPISODE,
                    label=batch_label[:120],
                    activation=0.8,
                    created_at=now,
                    last_accessed=now,
                    access_count=1,
                    metadata={
                        "content": content,
                        "importance": imp_level.value,
                        "source": "model_note_batch",
                        "entry_count": len(batch_entries),
                    },
                )
                mg.add_node(root_node)
                mg.set_importance(batch_root_id, imp_level)
                mg.index_summary(batch_root_id, f"{batch_label} {content}"[:1000])

            ctx = mg.get_last_focus_context()
            focus_node_id = (ctx or {}).get("focused_task_node_id")

            created_nodes = []
            semantic_total = 0
            from zulong.memory.memory_graph import EdgeType
            for index, spec in enumerate(node_specs, start=1):
                item_content = str(spec.get("content") or "").strip()
                if not item_content:
                    continue
                item_label = str(spec.get("label") or label or item_content[:50]).strip()[:120]
                suffix = f":{index:03d}" if batch_entries else ""
                node_id = f"note:{int(now * 1000)}{suffix}"
                node = GraphNode(
                    node_id=node_id,
                    node_type=NodeType.KNOWLEDGE,
                    label=item_label,
                    activation=0.8,
                    created_at=now + index * 0.000001,
                    last_accessed=now,
                    access_count=1,
                    metadata={
                        "content": item_content,
                        "importance": imp_level.value,
                        "source": "model_note_batch_entry" if batch_entries else "model_note",
                        "batch_root_id": batch_root_id,
                        "batch_index": spec.get("source_index") or str(index),
                    },
                )

                mg.add_node(node)
                mg.set_importance(node_id, imp_level)
                mg.index_summary(node_id, f"{item_label} {item_content}")

                if batch_root_id:
                    mg.add_edge(batch_root_id, node_id, EdgeType.HIERARCHY, weight=1.0, protected=True)
                if focus_node_id:
                    mg.add_edge(focus_node_id, node_id, EdgeType.REFERENCE, weight=0.6)

                logger.info(f"[save_memory_note] 保存节点 {node_id}: {item_label}")

                _semantic_count = 0
                try:
                    sem_neighbors = mg.discover_semantic_neighbors(
                        node_id, top_k=3, threshold=0.75,
                    )
                    _semantic_count = len(sem_neighbors)
                except Exception as _sem_err:
                    logger.debug(f"[save_memory_note] 语义关联跳过: {_sem_err}")
                semantic_total += _semantic_count
                node_obj = mg.get_node(node_id)
                created_nodes.append({
                    "node_id": node_id,
                    "graph_memory_id": node_id,
                    "shard_id": _safe_get_shard_id(mg, node_id, node=node_obj),
                    "full_path": node_id,
                    "memory_address": _build_memory_address(mg, node_id, node=node_obj),
                    "label": item_label,
                    "importance": imp_level.value,
                    "semantic_links": _semantic_count,
                })

            if not created_nodes:
                return self._create_result(
                    success=False,
                    error="未解析到可保存的记忆内容",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            primary = created_nodes[0]
            return self._create_result(
                success=True,
                data={
                    "node_id": primary["node_id"],
                    "graph_memory_id": primary["graph_memory_id"],
                    "shard_id": primary["shard_id"],
                    "full_path": primary["full_path"],
                    "memory_address": primary["memory_address"],
                    "label": primary["label"],
                    "importance": imp_level.value,
                    "semantic_links": semantic_total,
                    "created_count": len(created_nodes),
                    "batch_root_id": batch_root_id,
                    "nodes": created_nodes,
                    "message": f"已保存 {len(created_nodes)} 条记忆到记忆图",
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
                "entries": {
                    "type": "array",
                    "description": "可选：批量记忆条目；每项可为字符串或包含 label/content 的对象",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "description": "条目标签"},
                            "content": {"type": "string", "description": "条目内容"},
                        },
                    },
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
        node_id = request.parameters.get("node_id", "") or request.parameters.get("graph_memory_id", "")
        try:
            max_depth = min(max(int(request.parameters.get("max_depth", 2) or 2), 1), 3)
        except (TypeError, ValueError):
            max_depth = 2

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
            neighbors = _safe_get_neighbors(mg, node_id, max_depth=min(max_depth, 3))

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
                neighbor_id = getattr(n, "node_id", "")
                if not neighbor_id or neighbor_id in seen or neighbor_id == node_id:
                    continue
                seen.add(neighbor_id)
                edge = _safe_get_edge(mg, node_id, neighbor_id) or _safe_get_edge(mg, neighbor_id, node_id)
                item = _brief_node(mg, n)
                item["edge_type"] = edge.get("edge_type", edge.get("type", "unknown")) if edge else "indirect"
                related.append(item)

            # 补充语义邻居
            for sem_id, score in semantic_neighbors:
                if sem_id in seen or sem_id == node_id:
                    continue
                seen.add(sem_id)
                sem_node = mg.get_node(sem_id)
                if sem_node:
                    item = _brief_node(mg, sem_node)
                    item["edge_type"] = "semantic"
                    item["similarity"] = round(score, 3)
                    related.append(item)

            logger.info(f"[discover_related] 节点 {node_id} 发现 {len(related)} 个关联")

            return self._create_result(
                success=True,
                data={
                    "seed_node_id": node_id,
                    "seed_graph_memory_id": node_id,
                    "seed_memory_address": _build_memory_address(mg, node_id, node=mg.get_node(node_id)),
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
                    "description": "起始节点 ID；可直接使用 recall_memory 返回的 graph_memory_id。",
                },
                "graph_memory_id": {
                    "type": "string",
                    "description": "可选：来自 recall_memory 返回的图记忆地址，等同 node_id。",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "图遍历最大深度。",
                    "default": 2,
                    "minimum": 1,
                    "maximum": 3,
                },
            },
            "anyOf": [
                {"required": ["node_id"]},
                {"required": ["graph_memory_id"]},
            ],
            "required": [],
        }


# ============================================================
# P1-2: BFS 扩散激活工具
# ============================================================

class ActivateMemoryNetworkTool(BaseTool):
    """activate_memory_network — BFS 扩散激活

    从种子节点出发，通过加权 BFS 沿边传播激活值。
    边权越高传播越多，每跳按 decay 衰减。
    这是记忆图谱的核心能力：从任意节点追溯全局关联。
    """

    def __init__(self):
        super().__init__(name="activate_memory_network", category=ToolCategory.CUSTOM)
        self.description = (
            "从种子节点出发，通过 BFS 扩散激活记忆网络。"
            "当你需要了解某个主题的全局关联、发现隐含联系、"
            "或预热与当前任务相关的记忆网络时调用。"
            "激活值沿边权衰减传播，揭示整个关联图谱。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        seed_node_ids = request.parameters.get("seed_node_ids", [])
        max_depth = request.parameters.get("max_depth", 3)
        min_activation = request.parameters.get("min_activation", 0.01)

        if not seed_node_ids:
            return self._create_result(
                success=False,
                error="seed_node_ids 不能为空",
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

            # 验证种子节点存在
            valid_seeds = [s for s in seed_node_ids if mg.has_node(s)]
            if not valid_seeds:
                return self._create_result(
                    success=False,
                    error=f"所有种子节点均不存在: {seed_node_ids}",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # 调用 BFS 扩散激活
            activations = mg.compute_activations(
                seed_node_ids=valid_seeds,
                max_depth=min(max_depth, 5),
                min_activation=max(min_activation, 0.001),
            )

            # 格式化返回，按激活值降序
            activated = []
            for node_id, score in sorted(activations.items(), key=lambda x: -x[1]):
                node = mg.get_node(node_id)
                if node:
                    item = {
                        "node_id": node_id,
                        "type": _node_type_value(node),
                        "label": node.label,
                        "activation": round(score, 4),
                    }
                    _attach_address_fields(item, mg, node_id, node=node)
                    activated.append(item)

            logger.info(
                f"[activate_memory_network] 种子 {valid_seeds} → "
                f"激活 {len(activated)} 个节点"
            )

            return self._create_result(
                success=True,
                data={
                    "seed_node_ids": valid_seeds,
                    "total_activated": len(activated),
                    "activated_nodes": activated[:30],
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[activate_memory_network] 激活失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"网络激活失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "seed_node_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "种子节点 ID 列表，BFS 将从这些节点开始扩散",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "最大扩散深度（默认 3，最大 5）",
                },
                "min_activation": {
                    "type": "number",
                    "description": "最小激活阈值，低于此值的节点不返回（默认 0.01）",
                },
            },
            "required": ["seed_node_ids"],
        }


# ============================================================
# P1-3: 记忆浏览工具
# ============================================================

class ListMemoryTool(BaseTool):
    """list_memory — 浏览记忆节点列表

    按类型/温度/重要性筛选并列出记忆图中的节点。
    """

    def __init__(self):
        super().__init__(name="list_memory", category=ToolCategory.CUSTOM)
        self.description = (
            "浏览记忆图中的节点列表。"
            "当你需要了解记忆中有哪些信息、按类型/重要性筛选记忆、"
            "或获取记忆图的总体概况时调用。"
            "支持按节点类型、温度、重要性过滤。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        node_type = request.parameters.get("node_type", None)
        importance = request.parameters.get("importance", None)
        temperature = request.parameters.get("temperature", None)
        limit = min(request.parameters.get("limit", 20), 50)

        try:
            from zulong.memory.memory_graph import (
                get_memory_graph, NodeType, Importance, Temperature,
            )
            mg = get_memory_graph()
            if mg is None:
                return self._create_result(
                    success=False,
                    error="MemoryGraph 未初始化",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # 获取所有节点或按类型过滤
            if node_type:
                type_map = {t.value: t for t in NodeType}
                nt = type_map.get(node_type)
                if nt is None:
                    return self._create_result(
                        success=False,
                        error=f"无效的 node_type: {node_type}，"
                              f"可选值: {list(type_map.keys())}",
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )
                nodes = mg.get_nodes_by_type(nt)
            else:
                nodes = _iter_memory_nodes(mg)

            # 按重要性过滤
            if importance:
                imp_map = {i.value: i for i in Importance}
                target_imp = imp_map.get(importance)
                if target_imp:
                    nodes = [
                        n for n in nodes
                        if _safe_get_importance(mg, n) == target_imp.value
                    ]

            # 按温度过滤
            if temperature:
                temp_map = {t.value: t for t in Temperature}
                target_temp = temp_map.get(temperature)
                if target_temp:
                    nodes = [
                        n for n in nodes
                        if _safe_get_temperature(mg, n) == target_temp.value
                    ]

            total_filtered = len(nodes)

            # 按 last_accessed 降序排序
            nodes.sort(key=lambda n: getattr(n, "last_accessed", 0.0), reverse=True)

            # 格式化
            result_nodes = []
            for node in nodes[:limit]:
                temp = _safe_get_temperature(mg, node)
                imp = _safe_get_importance(mg, node)
                item = {
                    "node_id": node.node_id,
                    "type": _node_type_value(node),
                    "label": node.label,
                    "activation": _safe_round(getattr(node, "activation", 0.0), 3),
                    "temperature": temp or "unknown",
                    "importance": imp or "normal",
                }
                _attach_address_fields(item, mg, node.node_id, node=node)
                result_nodes.append(item)

            # 统计
            type_stats = {}
            for t in NodeType:
                try:
                    count = len(mg.get_nodes_by_type(t))
                except Exception:
                    count = sum(1 for n in nodes if _node_type_value(n) == t.value)
                if count > 0:
                    type_stats[t.value] = count

            logger.info(f"[list_memory] 返回 {len(result_nodes)}/{total_filtered} 个节点")

            return self._create_result(
                success=True,
                data={
                    "total_in_graph": len(_iter_memory_nodes(mg)),
                    "filtered_count": total_filtered,
                    "returned_count": len(result_nodes),
                    "type_stats": type_stats,
                    "nodes": result_nodes,
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[list_memory] 查询失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"记忆列表查询失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "node_type": {
                    "type": "string",
                    "enum": [
                        "task", "dialogue", "knowledge", "experience",
                        "episode", "file", "concept", "person", "document",
                    ],
                    "description": "按节点类型过滤（不填则返回全部类型）",
                },
                "importance": {
                    "type": "string",
                    "enum": [
                        "trivial", "normal", "identity",
                        "fact", "important", "must_remember",
                    ],
                    "description": "按重要性过滤",
                },
                "temperature": {
                    "type": "string",
                    "enum": ["hot", "warm", "cold"],
                    "description": "按温度过滤（hot=最近访问, warm=中等, cold=长期未访问）",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回数量上限（默认 20，最大 50）",
                },
            },
            "required": [],
        }


# ============================================================
# P2-9: 记忆边删除工具
# ============================================================

class DeleteMemoryEdgeTool(BaseTool):
    """delete_memory_edge — 删除记忆图中的边

    从图记忆系统中删除指定节点之间的关联边。
    支持按源-目标节点对删除单条边，或按节点ID删除其所有关联边。
    """

    def __init__(self):
        super().__init__(name="delete_memory_edge", category=ToolCategory.CUSTOM)
        self.description = (
            "删除记忆图中的边（关联关系）。当用户要求删除、移除两个记忆节点之间的关联时调用。"
            "关键词：删除关联、移除关系、断开连接。"
            "支持按 source_id + target_id 删除单条边，或按 node_id 删除该节点的所有关联边。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        source_id = request.parameters.get("source_id", "")
        target_id = request.parameters.get("target_id", "")
        node_id = request.parameters.get("node_id", "")

        if not source_id and not node_id:
            return self._create_result(
                success=False,
                error="必须提供 source_id+target_id（删除单条边）或 node_id（删除该节点所有关联边）",
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

            deleted_edges = []

            if source_id and target_id:
                if mg.remove_edge(source_id, target_id):
                    deleted_edges.append({"source": source_id, "target": target_id})
                else:
                    return self._create_result(
                        success=False,
                        error=f"边 ({source_id} → {target_id}) 不存在",
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )
            elif node_id:
                neighbors_in = list(mg._graph.predecessors(node_id)) if mg._graph.has_node(node_id) else []
                neighbors_out = list(mg._graph.successors(node_id)) if mg._graph.has_node(node_id) else []
                if not neighbors_in and not neighbors_out:
                    return self._create_result(
                        success=True,
                        data={"deleted_count": 0, "deleted_edges": [], "message": f"节点 {node_id} 无关联边"},
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )
                for src in neighbors_in:
                    if mg.remove_edge(src, node_id):
                        deleted_edges.append({"source": src, "target": node_id})
                for tgt in neighbors_out:
                    if mg.remove_edge(node_id, tgt):
                        deleted_edges.append({"source": node_id, "target": tgt})

            logger.info(f"[delete_memory_edge] 删除 {len(deleted_edges)} 条边")

            return self._create_result(
                success=True,
                data={
                    "deleted_count": len(deleted_edges),
                    "deleted_edges": deleted_edges,
                    "message": f"已删除 {len(deleted_edges)} 条记忆边",
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[delete_memory_edge] 删除失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"记忆边删除失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "边的源节点 ID（与 target_id 配对使用，删除单条边）",
                },
                "target_id": {
                    "type": "string",
                    "description": "边的目标节点 ID（与 source_id 配对使用，删除单条边）",
                },
                "node_id": {
                    "type": "string",
                    "description": "节点 ID，删除该节点的所有入边和出边",
                },
            },
            "required": [],
        }


# ============================================================
# P2-2: 重要性设置工具
# ============================================================

class SetImportanceTool(BaseTool):
    """set_importance — 设置记忆节点重要性

    调整指定节点的重要性标签，影响其衰减半衰期和检索优先级。
    重要性越高的记忆衰减越慢、检索时权重越高。
    """

    def __init__(self):
        super().__init__(name="set_importance", category=ToolCategory.CUSTOM)
        self.description = (
            "设置记忆节点的重要性级别。当用户明确要求记住某条信息、"
            "标记某条记忆为重要/不重要、或调整信息优先级时调用。"
            "重要性影响记忆的衰减速度和检索排名。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        node_id = request.parameters.get("node_id", "")
        importance = request.parameters.get("importance", "")
        reason = request.parameters.get("reason", "")

        if not node_id or not importance:
            return self._create_result(
                success=False,
                error="node_id 和 importance 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        try:
            from zulong.memory.memory_graph import get_memory_graph, Importance
            mg = get_memory_graph()
            if mg is None:
                return self._create_result(
                    success=False,
                    error="MemoryGraph 未初始化",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # 映射重要性
            imp_map = {i.value: i for i in Importance}
            target_imp = imp_map.get(importance)
            if target_imp is None:
                return self._create_result(
                    success=False,
                    error=f"无效的 importance: {importance}，"
                          f"可选值: {list(imp_map.keys())}",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # 检查节点存在
            node = mg.get_node(node_id)
            if node is None:
                return self._create_result(
                    success=False,
                    error=f"节点 '{node_id}' 不存在",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # 读取旧值
            old_imp = mg.get_importance(node_id)
            old_value = old_imp.value if old_imp else "unknown"

            # 设置新值
            ok = mg.set_importance(node_id, target_imp)

            if not ok:
                return self._create_result(
                    success=False,
                    error="设置重要性失败",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            logger.info(
                f"[set_importance] {node_id}: {old_value} → {target_imp.value}"
                f"{f' (原因: {reason})' if reason else ''}"
            )

            return self._create_result(
                success=True,
                data={
                    "node_id": node_id,
                    "label": node.label,
                    "old_importance": old_value,
                    "new_importance": target_imp.value,
                    "reason": reason,
                    "message": f"重要性已从 {old_value} 调整为 {target_imp.value}",
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[set_importance] 设置失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"重要性设置失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "要设置重要性的节点 ID",
                },
                "importance": {
                    "type": "string",
                    "enum": [
                        "trivial", "normal", "identity",
                        "fact", "important", "must_remember",
                    ],
                    "description": "目标重要性级别",
                },
                "reason": {
                    "type": "string",
                    "description": "调整原因（可选，用于日志记录）",
                },
            },
            "required": ["node_id", "importance"],
        }


# ============================================================
# P2-8: 记忆节点删除工具
# ============================================================

class DeleteMemoryNodeTool(BaseTool):
    """delete_memory_node — 删除记忆节点

    从图记忆系统中删除指定节点及其关联边。
    支持单个删除和批量删除。
    带有安全防护：不允许删除 identity 级别的核心记忆。
    """

    def __init__(self):
        super().__init__(name="delete_memory_node", category=ToolCategory.CUSTOM)
        self.description = (
            "删除记忆节点。当用户明确要求删除、移除、清除某条记忆或相关信息时调用。"
            "关键词：删除、移除、清除、忘记、去掉、不要记住。"
            "支持按节点 ID 删除单个节点，或按关键词批量搜索删除。"
            "注意：identity 级别的核心记忆受保护，不能删除。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        node_ids = request.parameters.get("node_ids", [])
        keyword = request.parameters.get("keyword", "")
        confirm = request.parameters.get("confirm", False)

        if not node_ids and not keyword:
            return self._create_result(
                success=False,
                error="必须提供 node_ids（节点 ID 列表）或 keyword（搜索关键词）之一",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        try:
            from zulong.memory.memory_graph import get_memory_graph, Importance
            mg = get_memory_graph()
            if mg is None:
                return self._create_result(
                    success=False,
                    error="MemoryGraph 未初始化",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # 如果通过关键词搜索，先找到匹配的节点
            if keyword and not node_ids:
                search_results = mg.search_nodes(query=keyword, max_results=20)
                node_ids = [r.get("node_id") for r in search_results if r.get("node_id")]

                if not node_ids:
                    return self._create_result(
                        success=True,
                        data={
                            "keyword": keyword,
                            "deleted_count": 0,
                            "message": f"未找到与 '{keyword}' 相关的记忆节点",
                        },
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )

                # 若未确认，返回待删除列表供确认
                if not confirm:
                    preview = []
                    for nid in node_ids:
                        node = mg.get_node(nid)
                        if node:
                            preview.append({
                                "node_id": nid,
                                "type": _node_type_value(node),
                                "label": node.label,
                            })
                    return self._create_result(
                        success=True,
                        data={
                            "keyword": keyword,
                            "action": "preview",
                            "candidates": preview,
                            "message": (
                                f"找到 {len(preview)} 个与 '{keyword}' 相关的节点。"
                                f"请使用 confirm=true 确认删除，或指定具体 node_ids。"
                            ),
                        },
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )

            # 执行删除
            deleted = []
            protected = []
            not_found = []

            for nid in node_ids:
                node = mg.get_node(nid)
                if node is None:
                    not_found.append(nid)
                    continue

                # 安全防护：identity 级别节点不可删除
                imp = _safe_get_importance(mg, node)
                if imp == Importance.IDENTITY.value:
                    protected.append({
                        "node_id": nid,
                        "label": node.label,
                        "reason": "identity 级别核心记忆受保护",
                    })
                    continue

                # 执行删除（MemoryGraph）
                ok = mg.remove_node(nid)
                if ok:
                    deleted.append({
                        "node_id": nid,
                        "type": _node_type_value(node),
                        "label": node.label,
                    })
                    # 如果是任务节点，同步从 TaskGraph 中移除
                    if nid.startswith("task:tg_"):
                        try:
                            from zulong.tools.task_tools import get_active_task_graph
                            tg = get_active_task_graph()
                            if tg and tg.get_node(nid):
                                tg.remove_node(nid)
                                logger.debug(
                                    f"[delete_memory_node] 同步从 TaskGraph 移除: {nid}"
                                )
                        except Exception as _tg_err:
                            logger.debug(
                                f"[delete_memory_node] TaskGraph 同步移除跳过: {_tg_err}"
                            )

            logger.info(
                f"[delete_memory_node] 删除 {len(deleted)} 个节点，"
                f"保护 {len(protected)} 个，未找到 {len(not_found)} 个"
            )
            if deleted:
                try:
                    from zulong.core.event_bus import event_bus
                    from zulong.core.types import EventPriority, EventType, ZulongEvent

                    deleted_ids = [item["node_id"] for item in deleted if item.get("node_id")]
                    event_bus.publish(ZulongEvent(
                        type=EventType.MEMORY_GRAPH_UPDATED,
                        priority=EventPriority.LOW,
                        source="delete_memory_node",
                        payload={
                            "update_type": "delta",
                            "ts": time.time(),
                            "nodes": [],
                            "edges": [],
                            "changes": [
                                {"action": "remove_node", "data": {"id": node_id}}
                                for node_id in deleted_ids
                            ],
                            "changed_node_ids": deleted_ids,
                            "deleted_node_ids": deleted_ids,
                            "stats": {
                                "transport": "delta",
                                "changed_nodes": len(deleted_ids),
                                "changed_edges": 0,
                            },
                        },
                    ))
                except Exception as _event_err:
                    logger.debug(
                        f"[delete_memory_node] MemoryGraph 删除事件发布跳过: {_event_err}"
                    )

            return self._create_result(
                success=True,
                data={
                    "deleted_count": len(deleted),
                    "deleted": deleted,
                    "protected_count": len(protected),
                    "protected": protected,
                    "not_found": not_found,
                    "message": (
                        f"已删除 {len(deleted)} 个记忆节点"
                        + (f"，{len(protected)} 个核心记忆受保护未删除" if protected else "")
                        + (f"，{len(not_found)} 个节点未找到" if not_found else "")
                    ),
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[delete_memory_node] 删除失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"记忆节点删除失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "node_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要删除的节点 ID 列表，如 ['note:xxx', 'task:tg_xxx']",
                },
                "keyword": {
                    "type": "string",
                    "description": (
                        "按关键词搜索要删除的节点（与 node_ids 二选一）。"
                        "首次调用会返回预览列表，需设置 confirm=true 确认删除。"
                    ),
                },
                "confirm": {
                    "type": "boolean",
                    "description": "确认执行批量删除（keyword 模式下需要确认，默认 false）",
                },
            },
            "required": [],
        }
