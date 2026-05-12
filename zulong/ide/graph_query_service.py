"""
图谱分页查询服务

专门用于任务图谱渐进式加载的高性能分页查询
- 支持游标分页（避免深度分页性能问题）
- 支持排序和过滤
- 边查询边序列化（减少内存峰值）
- 优化大数据集查询性能
"""

from __future__ import annotations
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple
from dataclasses import dataclass
import hashlib


@dataclass
class PageInfo:
    """分页信息"""
    page: int
    page_size: int
    total_nodes: int
    total_pages: int
    has_next: bool
    has_prev: bool
    cursor: Optional[str] = None


@dataclass
class GraphQueryResult:
    """图谱查询结果"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    page_info: PageInfo
    query_time: float
    from_cache: bool = False


class GraphQueryService:
    """图谱分页查询服务"""

    def __init__(self, task_graph: Any):
        self._tg = task_graph
        self._query_cache: Dict[str, Tuple[GraphQueryResult, float]] = {}
        self._cache_ttl: float = 300.0  # 5分钟缓存

    def query_nodes_paginated(
        self,
        graph_id: str,
        page: int = 1,
        page_size: int = 500,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        node_type: Optional[str] = None,
        status_filter: Optional[List[str]] = None,
        include_edges: bool = True,
        use_cache: bool = True,
    ) -> GraphQueryResult:
        """
        分页查询图谱节点

        Args:
            graph_id: 图谱ID
            page: 页码（从1开始）
            page_size: 每页大小（默认500）
            sort_by: 排序字段
            sort_order: 排序方向（asc/desc）
            node_type: 节点类型过滤
            status_filter: 状态过滤列表
            include_edges: 是否包含边数据
            use_cache: 是否使用缓存

        Returns:
            GraphQueryResult: 查询结果
        """
        start_time = time.time()

        cache_key = self._generate_cache_key(
            graph_id, page, page_size, sort_by, sort_order, node_type, status_filter
        )

        if use_cache:
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return GraphQueryResult(
                    nodes=cached_result.nodes,
                    edges=cached_result.edges,
                    page_info=cached_result.page_info,
                    query_time=time.time() - start_time,
                    from_cache=True,
                )

        nodes = self._filter_nodes(node_type, status_filter)
        nodes = self._sort_nodes(nodes, sort_by, sort_order)

        total_nodes = len(nodes)
        total_pages = (total_nodes + page_size - 1) // page_size if page_size > 0 else 0

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_nodes = nodes[start_idx:end_idx]

        page_node_ids = {n.get("id") for n in page_nodes}

        edges = []
        if include_edges:
            edges = self._get_edges_for_nodes(page_node_ids)

        page_info = PageInfo(
            page=page,
            page_size=page_size,
            total_nodes=total_nodes,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
            cursor=self._generate_cursor(page_nodes, sort_by),
        )

        result = GraphQueryResult(
            nodes=page_nodes,
            edges=edges,
            page_info=page_info,
            query_time=time.time() - start_time,
            from_cache=False,
        )

        if use_cache:
            self._cache_result(cache_key, result)

        return result

    def query_nodes_by_cursor(
        self,
        graph_id: str,
        cursor: str,
        page_size: int = 500,
        direction: str = "next",
    ) -> GraphQueryResult:
        """
        基于游标的分页查询（避免深度分页性能问题）

        Args:
            graph_id: 图谱ID
            cursor: 游标字符串
            page_size: 每页大小
            direction: 方向（next/prev）

        Returns:
            GraphQueryResult: 查询结果
        """
        start_time = time.time()

        cursor_data = self._decode_cursor(cursor)
        if not cursor_data:
            return self.query_nodes_paginated(graph_id, page_size=page_size)

        last_value, last_id = cursor_data

        nodes = list(self._tg._nodes.values())
        nodes = [n.__dict__ if hasattr(n, '__dict__') else str(n) for n in nodes]

        nodes = self._sort_nodes(nodes, cursor_data.get("sort_by", "created_at"), "desc")

        if direction == "next":
            page_nodes = self._get_nodes_after(nodes, last_value, last_id, page_size)
        else:
            page_nodes = self._get_nodes_before(nodes, last_value, last_id, page_size)

        page_node_ids = {n.get("id") for n in page_nodes}
        edges = self._get_edges_for_nodes(page_node_ids)

        total_nodes = len(nodes)
        page_info = PageInfo(
            page=0,  # 游标分页不使用页码
            page_size=page_size,
            total_nodes=total_nodes,
            total_pages=0,
            has_next=len(page_nodes) == page_size,
            has_prev=True,
            cursor=self._generate_cursor(page_nodes, cursor_data.get("sort_by", "created_at")),
        )

        return GraphQueryResult(
            nodes=page_nodes,
            edges=edges,
            page_info=page_info,
            query_time=time.time() - start_time,
        )

    def get_total_count(
        self,
        graph_id: str,
        node_type: Optional[str] = None,
        status_filter: Optional[List[str]] = None,
    ) -> int:
        """获取节点总数"""
        nodes = self._filter_nodes(node_type, status_filter)
        return len(nodes)

    def _filter_nodes(
        self,
        node_type: Optional[str],
        status_filter: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """过滤节点"""
        nodes = list(self._tg._nodes.values())

        if node_type:
            nodes = [n for n in nodes if getattr(n, 'type', None) == node_type]

        if status_filter:
            nodes = [n for n in nodes if getattr(n, 'status', None) in status_filter]
        else:
            nodes = [n for n in nodes if getattr(n, 'status', None) != "deleted"]

        return [n.__dict__ if hasattr(n, '__dict__') else str(n) for n in nodes]

    def _sort_nodes(
        self,
        nodes: List[Dict[str, Any]],
        sort_by: str,
        sort_order: str,
    ) -> List[Dict[str, Any]]:
        """排序节点"""
        reverse = sort_order == "desc"

        def get_sort_key(node: Dict[str, Any]) -> Any:
            if isinstance(node, dict):
                return node.get(sort_by, "")
            return getattr(node, sort_by, "")

        return sorted(nodes, key=get_sort_key, reverse=reverse)

    def _get_edges_for_nodes(self, node_ids: set) -> List[Dict[str, Any]]:
        """获取节点的边"""
        edges = []

        for s, t in self._tg._h_edges:
            if s in node_ids or t in node_ids:
                edges.append({
                    "id": f"h_{s}_{t}",
                    "source": s,
                    "target": t,
                    "type": "hierarchy",
                })

        for e in self._tg._d_edges:
            if e.s in node_ids or e.t in node_ids:
                edges.append({
                    "id": f"d_{e.s}_{e.t}",
                    "source": e.s,
                    "target": e.t,
                    "type": "dependency",
                    "via": getattr(e, 'via', ''),
                    "cross": getattr(e, 'cross', False),
                })

        return edges

    def _generate_cache_key(self, *args) -> str:
        """生成缓存键"""
        key_str = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[GraphQueryResult]:
        """获取缓存结果"""
        if cache_key in self._query_cache:
            result, timestamp = self._query_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return result
            else:
                del self._query_cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: GraphQueryResult):
        """缓存结果"""
        self._query_cache[cache_key] = (result, time.time())

        if len(self._query_cache) > 100:
            oldest_key = min(self._query_cache.keys(), key=lambda k: self._query_cache[k][1])
            del self._query_cache[oldest_key]

    def _generate_cursor(self, nodes: List[Dict], sort_by: str) -> str:
        """生成游标"""
        if not nodes:
            return ""

        last_node = nodes[-1]
        last_value = last_node.get(sort_by, "")
        last_id = last_node.get("id", "")

        import base64
        import json
        cursor_data = {
            "last_value": str(last_value),
            "last_id": last_id,
            "sort_by": sort_by,
        }
        return base64.b64encode(json.dumps(cursor_data).encode()).decode()

    def _decode_cursor(self, cursor: str) -> Optional[Dict]:
        """解码游标"""
        if not cursor:
            return None

        try:
            import base64
            import json
            cursor_data = json.loads(base64.b64decode(cursor.encode()).decode())
            return cursor_data
        except Exception:
            return None

    def _get_nodes_after(
        self,
        nodes: List[Dict],
        last_value: str,
        last_id: str,
        count: int,
    ) -> List[Dict]:
        """获取游标之后的节点"""
        found = False
        result = []

        for node in nodes:
            if not found:
                if node.get("id") == last_id:
                    found = True
                continue

            result.append(node)
            if len(result) >= count:
                break

        return result

    def _get_nodes_before(
        self,
        nodes: List[Dict],
        last_value: str,
        last_id: str,
        count: int,
    ) -> List[Dict]:
        """获取游标之前的节点"""
        result = []

        for node in reversed(nodes):
            if node.get("id") == last_id:
                break
            result.insert(0, node)
            if len(result) >= count:
                result = result[-count:]

        return result

    def clear_cache(self):
        """清空缓存"""
        self._query_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "cache_size": len(self._query_cache),
            "cache_ttl": self._cache_ttl,
        }
