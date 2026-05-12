"""TaskGraph CRUD 审计日志与数据模型"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── 错误码常量 ──
NODE_ID_CONFLICT = "NODE_ID_CONFLICT"
NODE_NOT_FOUND = "NODE_NOT_FOUND"
EDGE_ENDPOINT_NOT_FOUND = "EDGE_ENDPOINT_NOT_FOUND"
SELF_LOOP_FORBIDDEN = "SELF_LOOP_FORBIDDEN"
EDGE_ALREADY_EXISTS = "EDGE_ALREADY_EXISTS"
ROOT_NODE_DELETE_FORBIDDEN = "ROOT_NODE_DELETE_FORBIDDEN"
CASCADE_REQUIRED = "CASCADE_REQUIRED"
DELETE_CONFIRM_REJECTED = "DELETE_CONFIRM_REJECTED"
DELETE_CONFIRM_TIMEOUT = "DELETE_CONFIRM_TIMEOUT"
INVALID_STATUS_TRANSITION = "INVALID_STATUS_TRANSITION"
INVALID_ATTRIBUTE = "INVALID_ATTRIBUTE"
IMMUTABLE_ATTRIBUTE = "IMMUTABLE_ATTRIBUTE"
TASK_GRAPH_NOT_INITIALIZED = "TASK_GRAPH_NOT_INITIALIZED"
SUBGRAPH_DEPTH_TRUNCATED = "SUBGRAPH_DEPTH_TRUNCATED"
UNKNOWN_TOOL = "UNKNOWN_TOOL"
INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass
class AuditLogEntry:
    timestamp: str
    session_id: str
    tool_name: str
    action: str
    target_id: str
    params: Dict[str, Any]
    result: Dict[str, Any]
    duration_ms: int

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)

    @classmethod
    def from_jsonl(cls, line: str) -> AuditLogEntry:
        d = json.loads(line)
        return cls(**d)


@dataclass
class QueryParams:
    mode: str = "list"
    node_id: Optional[str] = None
    node_type: Optional[str] = None
    keyword: Optional[str] = None
    depth: int = 2
    page: int = 1
    page_size: int = 20
    sort_by: str = "created_at"
    sort_order: str = "desc"
    include_deleted: bool = False

    def validate(self) -> Optional[str]:
        valid_modes = {"get_node", "list", "search", "subgraph", "overview"}
        if self.mode not in valid_modes:
            return f"无效查询模式: {self.mode}, 有效值: {valid_modes}"
        if self.page < 1 or self.page_size < 1:
            return "page和page_size必须≥1"
        if self.depth < 0 or self.depth > 10:
            return "depth必须在0-10之间"
        return None


@dataclass
class CRUDResult:
    success: bool
    data: Any = None
    error_code: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"success": self.success}
        if self.data is not None:
            d["data"] = self.data
        if self.error_code:
            d["error_code"] = self.error_code
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class DeleteConfirmRequest:
    confirm_id: str
    node_id: str
    reason: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    event: Any = None
    result: Optional[bool] = None


class GraphCRUDAudit:
    def __init__(self, log_dir: str = "./logs/graph_crud_audit"):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log_operation(self, entry: AuditLogEntry) -> None:
        filename = f"graph_crud_audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        filepath = self._log_dir / filename
        try:
            with self._lock:
                with open(filepath, "a", encoding="utf-8") as f:
                    f.write(entry.to_jsonl() + "\n")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("审计日志写入失败: %s", e)

    def query_logs(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        action: Optional[str] = None,
        tool_name: Optional[str] = None,
        target_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditLogEntry]:
        results = []
        try:
            for filepath in sorted(self._log_dir.glob("*.jsonl"), reverse=True):
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = AuditLogEntry.from_jsonl(line)
                        except Exception:
                            continue
                        if start_time and entry.timestamp < start_time:
                            continue
                        if end_time and entry.timestamp > end_time:
                            continue
                        if action and entry.action != action:
                            continue
                        if tool_name and entry.tool_name != tool_name:
                            continue
                        if target_id and entry.target_id != target_id:
                            continue
                        results.append(entry)
                        if len(results) >= limit:
                            return results
        except Exception:
            pass
        return results
