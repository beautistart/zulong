"""TaskGraph CRUD 执行引擎 — Create/Delete/Update/Query"""
from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from zulong.ide.graph_crud_audit import (
    AuditLogEntry,
    CASCADE_REQUIRED,
    CRUDResult,
    DELETE_CONFIRM_REJECTED,
    DELETE_CONFIRM_TIMEOUT,
    EDGE_ALREADY_EXISTS,
    EDGE_ENDPOINT_NOT_FOUND,
    GraphCRUDAudit,
    IMMUTABLE_ATTRIBUTE,
    INVALID_STATUS_TRANSITION,
    NODE_ID_CONFLICT,
    NODE_NOT_FOUND,
    ROOT_NODE_DELETE_FORBIDDEN,
    SELF_LOOP_FORBIDDEN,
    SUBGRAPH_DEPTH_TRUNCATED,
    QueryParams,
)

logger = logging.getLogger(__name__)

VALID_STATUS_TRANSITIONS: Dict[str, Set[str]] = {
    "pending": {"in_progress", "blocked", "skipped"},
    "in_progress": {"completed", "blocked", "needs_adjust", "waiting_input"},
    "blocked": {"pending", "in_progress"},
    "needs_adjust": {"in_progress", "pending"},
    "waiting_input": {"in_progress"},
    "completed": set(),
    "skipped": set(),
    "deleted": set(),
}

MUTABLE_NODE_ATTRS = {"label", "desc", "status", "result", "task_domain", "analysis_content", "semantic_summary", "files"}
IMMUTABLE_NODE_ATTRS = {"id", "created_at"}


class GraphCRUDExecutor:
    def __init__(self, task_graph: Any, session_id: str = "", ws_sender: Optional[Callable] = None):
        self._tg = task_graph
        self._session_id = session_id
        self._ws_sender = ws_sender
        self._audit = GraphCRUDAudit()

    def _audit_log(self, tool_name: str, action: str, target_id: str, params: dict, result: dict, duration_ms: int):
        from zulong.ide.graph_crud_audit import AuditLogEntry
        entry = AuditLogEntry(
            timestamp=datetime.now().isoformat(),
            session_id=self._session_id,
            tool_name=tool_name,
            action=action,
            target_id=target_id,
            params=params,
            result=result,
            duration_ms=duration_ms,
        )
        self._audit.log_operation(entry)

    def _push_graph_update(self, action: str, data: Any):
        if not self._ws_sender:
            return
        try:
            msg = {
                "msg_type": "graph_update",
                "payload": {"action": action, "data": data, "session_id": self._session_id, "ts": datetime.now().isoformat()},
            }
            if asyncio.iscoroutinefunction(self._ws_sender):
                import asyncio
                asyncio.create_task(self._ws_sender("graph_update", msg["payload"]))
            else:
                self._ws_sender("graph_update", msg["payload"])
        except Exception as e:
            logger.warning("WebSocket推送失败: %s", e)

    # ── CREATE ──
    def execute_create_node(self, params: Dict) -> CRUDResult:
        start = time.time()
        node_type = params.get("node_type", "task")
        label = params.get("label", "")
        desc = params.get("desc", "")
        parent_id = params.get("parent_id")
        task_domain = params.get("task_domain", "general")
        custom_id = params.get("custom_id")

        from zulong.ide.graph_crud_tools import _generate_node_id
        node_id = custom_id or _generate_node_id(node_type)

        if not label or len(label) > 200:
            return CRUDResult(False, error_code="INVALID_ATTRIBUTE", error="label非空且≤200字符")

        with self._tg._lock:
            if node_id in self._tg._nodes:
                result = CRUDResult(False, error_code=NODE_ID_CONFLICT, error=f"节点ID已存在: {node_id}")
                self._audit_log("graph_create_node", "create_node", node_id, params, result.to_dict(), int((time.time()-start)*1000))
                return result

            self._tg.add_node(id=node_id, label=label, type=node_type, status="pending", desc=desc, task_domain=task_domain)

            if parent_id:
                if parent_id in self._tg._nodes:
                    self._tg.add_h_edge(parent_id, node_id)
                else:
                    logger.warning("parent_id不存在: %s, 跳过建边", parent_id)

        node_data = self._tg.get_node(node_id)
        data = node_data.__dict__ if node_data and hasattr(node_data, '__dict__') else {"id": node_id, "label": label}
        self._push_graph_update("node_created", data)
        result = CRUDResult(True, data=data)
        self._audit_log("graph_create_node", "create_node", node_id, params, result.to_dict(), int((time.time()-start)*1000))
        return result

    def execute_create_edge(self, params: Dict) -> CRUDResult:
        start = time.time()
        source = params.get("source", "")
        target = params.get("target", "")
        edge_type = params.get("edge_type", "hierarchy")
        via = params.get("via", "")
        cross = params.get("cross", False)

        with self._tg._lock:
            if source not in self._tg._nodes or target not in self._tg._nodes:
                result = CRUDResult(False, error_code=EDGE_ENDPOINT_NOT_FOUND, error=f"端点不存在: {source}→{target}")
                self._audit_log("graph_create_edge", "create_edge", f"{source}->{target}", params, result.to_dict(), int((time.time()-start)*1000))
                return result

            if source == target:
                result = CRUDResult(False, error_code=SELF_LOOP_FORBIDDEN, error=f"自环边禁止: {source}")
                self._audit_log("graph_create_edge", "create_edge", source, params, result.to_dict(), int((time.time()-start)*1000))
                return result

            if edge_type == "hierarchy":
                if (source, target) in self._tg._h_edges:
                    result = CRUDResult(False, error_code=EDGE_ALREADY_EXISTS, error=f"层级边已存在: {source}→{target}")
                    self._audit_log("graph_create_edge", "create_edge", source, params, result.to_dict(), int((time.time()-start)*1000))
                    return result
                self._tg.add_h_edge(source, target)
            elif edge_type in ("dependency", "reference"):
                self._tg.add_d_edge(source, target, via=via, cross=cross)

        data = {"source": source, "target": target, "edge_type": edge_type}
        self._push_graph_update("edge_created", data)
        result = CRUDResult(True, data=data)
        self._audit_log("graph_create_edge", "create_edge", f"{source}->{target}", params, result.to_dict(), int((time.time()-start)*1000))
        return result

    def execute_batch_create(self, params: Dict) -> CRUDResult:
        start = time.time()
        nodes_params = params.get("nodes", [])
        edges_params = params.get("edges", [])
        successes, failures = [], []

        for np in nodes_params:
            r = self.execute_create_node(np)
            if r.success:
                successes.append(r.data)
            else:
                failures.append({"params": np, "error": r.error})

        for ep in edges_params:
            r = self.execute_create_edge(ep)
            if r.success:
                successes.append(r.data)
            else:
                failures.append({"params": ep, "error": r.error})

        data = {"successes": successes, "failures": failures}
        self._push_graph_update("batch_created", data)
        result = CRUDResult(True, data=data)
        self._audit_log("graph_batch_create", "batch_create", "batch", params, result.to_dict(), int((time.time()-start)*1000))
        return result

    # ── DELETE ──
    def execute_delete_node(self, params: Dict) -> CRUDResult:
        start = time.time()
        node_id = params.get("node_id", "")
        cascade = params.get("cascade", False)
        soft_delete = params.get("soft_delete", False)

        with self._tg._lock:
            if node_id not in self._tg._nodes:
                result = CRUDResult(False, error_code=NODE_NOT_FOUND, error=f"节点不存在: {node_id}")
                self._audit_log("graph_delete_node", "delete_node", node_id, params, result.to_dict(), int((time.time()-start)*1000))
                return result

            node = self._tg._nodes[node_id]
            depth = self._tg.get_node_depth(node_id) if hasattr(self._tg, 'get_node_depth') else 0

            if node.type == "requirement" and depth == 0:
                result = CRUDResult(False, error_code=ROOT_NODE_DELETE_FORBIDDEN, error=f"根需求节点禁止删除: {node_id}")
                self._audit_log("graph_delete_node", "delete_node", node_id, params, result.to_dict(), int((time.time()-start)*1000))
                return result

            children = [t for s, t in self._tg._h_edges if s == node_id]
            if children and not cascade and not soft_delete:
                result = CRUDResult(False, error_code=CASCADE_REQUIRED, error=f"节点有{len(children)}个子节点，需cascade=true")
                self._audit_log("graph_delete_node", "delete_node", node_id, params, result.to_dict(), int((time.time()-start)*1000))
                return result

        if soft_delete:
            node = self._tg._nodes.get(node_id)
            if node and hasattr(node, 'status'):
                node.status = "deleted"
            data = {"node_id": node_id, "action": "soft_deleted"}
        else:
            removed = self._tg.remove_node(node_id)
            data = {"node_id": node_id, "action": "deleted", "removed_nodes": removed}

        self._push_graph_update("node_deleted", data)
        result = CRUDResult(True, data=data)
        self._audit_log("graph_delete_node", "delete_node", node_id, params, result.to_dict(), int((time.time()-start)*1000))
        return result

    def execute_delete_edge(self, params: Dict) -> CRUDResult:
        start = time.time()
        source = params.get("source", "")
        target = params.get("target", "")
        edge_type = params.get("edge_type", "hierarchy")

        with self._tg._lock:
            if edge_type == "hierarchy":
                self._tg._h_edges = [(s, t) for s, t in self._tg._h_edges if not (s == source and t == target)]
            elif edge_type == "dependency":
                self._tg._d_edges = [e for e in self._tg._d_edges if not (e.s == source and e.t == target)]

        data = {"source": source, "target": target, "edge_type": edge_type}
        self._push_graph_update("edge_deleted", data)
        result = CRUDResult(True, data=data)
        self._audit_log("graph_delete_edge", "delete_edge", f"{source}->{target}", params, result.to_dict(), int((time.time()-start)*1000))
        return result

    # ── UPDATE ──
    def execute_update_node(self, params: Dict) -> CRUDResult:
        start = time.time()
        node_id = params.get("node_id", "")

        with self._tg._lock:
            if node_id not in self._tg._nodes:
                result = CRUDResult(False, error_code=NODE_NOT_FOUND, error=f"节点不存在: {node_id}")
                self._audit_log("graph_update_node", "update_node", node_id, params, result.to_dict(), int((time.time()-start)*1000))
                return result

            node = self._tg._nodes[node_id]

            for key in params:
                if key in IMMUTABLE_NODE_ATTRS:
                    result = CRUDResult(False, error_code=IMMUTABLE_ATTRIBUTE, error=f"不可修改属性: {key}")
                    self._audit_log("graph_update_node", "update_node", node_id, params, result.to_dict(), int((time.time()-start)*1000))
                    return result

            if "label" in params:
                if not params["label"] or len(params["label"]) > 200:
                    return CRUDResult(False, error_code="INVALID_ATTRIBUTE", error="label非空且≤200字符")

            if "status" in params:
                new_status = params["status"]
                old_status = node.status
                if new_status != old_status:
                    allowed = VALID_STATUS_TRANSITIONS.get(old_status, set())
                    if new_status not in allowed and new_status != "deleted":
                        result = CRUDResult(False, error_code=INVALID_STATUS_TRANSITION, error=f"非法状态流转: {old_status}→{new_status}")
                        self._audit_log("graph_update_node", "update_node", node_id, params, result.to_dict(), int((time.time()-start)*1000))
                        return result
                    self._tg.update_node_status(node_id, new_status)

            update_kwargs = {}
            for attr in ("label", "desc", "result", "analysis_content", "semantic_summary"):
                if attr in params:
                    update_kwargs[attr] = params[attr]
            if "task_domain" in params:
                update_kwargs["task_domain"] = params["task_domain"]
            if update_kwargs:
                self._tg.update_node_content(node_id, **update_kwargs)

        node_data = self._tg.get_node(node_id)
        data = node_data.__dict__ if node_data and hasattr(node_data, '__dict__') else {"id": node_id}
        self._push_graph_update("node_updated", data)
        result = CRUDResult(True, data=data)
        self._audit_log("graph_update_node", "update_node", node_id, params, result.to_dict(), int((time.time()-start)*1000))
        return result

    def execute_batch_update(self, params: Dict) -> CRUDResult:
        start = time.time()
        updates = params.get("updates", [])
        successes, failures = [], []
        for u in updates:
            r = self.execute_update_node(u)
            if r.success:
                successes.append(r.data)
            else:
                failures.append({"params": u, "error": r.error})
        data = {"successes": successes, "failures": failures}
        self._push_graph_update("batch_updated", data)
        result = CRUDResult(True, data=data)
        self._audit_log("graph_batch_update", "batch_update", "batch", params, result.to_dict(), int((time.time()-start)*1000))
        return result

    # ── QUERY ──
    def execute_query_nodes(self, params: Dict) -> CRUDResult:
        start = time.time()
        qp = QueryParams(
            mode=params.get("mode", "list"),
            node_id=params.get("node_id"),
            node_type=params.get("node_type"),
            keyword=params.get("keyword"),
            depth=params.get("depth", 2),
            page=params.get("page", 1),
            page_size=params.get("page_size", 20),
            sort_by=params.get("sort_by", "created_at"),
            sort_order=params.get("sort_order", "desc"),
            include_deleted=params.get("include_deleted", False),
        )
        err = qp.validate()
        if err:
            return CRUDResult(False, error_code="INVALID_ATTRIBUTE", error=err)

        with self._tg._lock:
            if qp.mode == "get_node":
                node = self._tg.get_node(qp.node_id or "")
                data = node.__dict__ if node and hasattr(node, '__dict__') else None
                result = CRUDResult(True, data=data)

            elif qp.mode == "list":
                nodes = list(self._tg._nodes.values())
                if qp.node_type:
                    nodes = [n for n in nodes if n.type == qp.node_type]
                if not qp.include_deleted:
                    nodes = [n for n in nodes if n.status != "deleted"]
                total = len(nodes)
                start_idx = (qp.page - 1) * qp.page_size
                page_nodes = nodes[start_idx:start_idx + qp.page_size]
                data = {
                    "nodes": [n.__dict__ if hasattr(n, '__dict__') else str(n) for n in page_nodes],
                    "total": total, "page": qp.page, "page_size": qp.page_size,
                }
                result = CRUDResult(True, data=data)

            elif qp.mode == "search":
                keyword = (qp.keyword or "").lower()
                nodes = list(self._tg._nodes.values())
                matched = [n for n in nodes if keyword in n.label.lower() or keyword in (n.desc or "").lower()]
                if not qp.include_deleted:
                    matched = [n for n in matched if n.status != "deleted"]
                total = len(matched)
                start_idx = (qp.page - 1) * qp.page_size
                page_nodes = matched[start_idx:start_idx + qp.page_size]
                data = {
                    "nodes": [n.__dict__ if hasattr(n, '__dict__') else str(n) for n in page_nodes],
                    "total": total, "page": qp.page, "page_size": qp.page_size,
                }
                result = CRUDResult(True, data=data)

            elif qp.mode == "subgraph":
                center = qp.node_id or ""
                if center not in self._tg._nodes:
                    result = CRUDResult(False, error_code=NODE_NOT_FOUND, error=f"节点不存在: {center}")
                else:
                    max_depth = min(qp.depth, 20)
                    truncated = qp.depth > 20
                    visited = {center}
                    queue = deque([(center, 0)])
                    sub_nodes = set()
                    while queue:
                        nid, d = queue.popleft()
                        sub_nodes.add(nid)
                        if d < max_depth:
                            children = [t for s, t in self._tg._h_edges if s == nid]
                            parents = [s for s, t in self._tg._h_edges if t == nid]
                            for neighbor in children + parents:
                                if neighbor not in visited and neighbor in self._tg._nodes:
                                    visited.add(neighbor)
                                    queue.append((neighbor, d + 1))
                    sub_h = [(s, t) for s, t in self._tg._h_edges if s in sub_nodes and t in sub_nodes]
                    sub_d = [e for e in self._tg._d_edges if e.s in sub_nodes and e.t in sub_nodes]
                    data = {
                        "nodes": [self._tg._nodes[n].__dict__ if hasattr(self._tg._nodes[n], '__dict__') else str(self._tg._nodes[n]) for n in sub_nodes],
                        "h_edges": sub_h,
                        "d_edges": [{"s": e.s, "t": e.t, "via": e.via, "cross": e.cross} for e in sub_d],
                        "truncated": truncated,
                    }
                    if truncated:
                        result = CRUDResult(True, data=data, error_code=SUBGRAPH_DEPTH_TRUNCATED)
                    else:
                        result = CRUDResult(True, data=data)

            elif qp.mode == "overview":
                nodes = list(self._tg._nodes.values())
                by_type = {}
                by_status = {}
                for n in nodes:
                    by_type[n.type] = by_type.get(n.type, 0) + 1
                    by_status[n.status] = by_status.get(n.status, 0) + 1
                data = {
                    "total_nodes": len(nodes),
                    "by_type": by_type,
                    "by_status": by_status,
                    "total_h_edges": len(self._tg._h_edges),
                    "total_d_edges": len(self._tg._d_edges),
                }
                result = CRUDResult(True, data=data)
            else:
                result = CRUDResult(False, error_code="INVALID_ATTRIBUTE", error=f"未知查询模式: {qp.mode}")

        self._audit_log("graph_query_nodes", "query", qp.node_id or "query", params, result.to_dict(), int((time.time()-start)*1000))
        return result


import asyncio
