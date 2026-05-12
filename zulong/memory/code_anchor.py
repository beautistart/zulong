# File: zulong/memory/code_anchor.py
# 代码锚点 (Code Anchor) — 将记忆/任务节点关联到具体代码位置
#
# 核心能力:
# - CodeAnchor 数据模型 (file_path + symbol + line_range + commit + content_hash)
# - CodeAnchorStore 单例管理 (CRUD + 反向索引 + JSON 持久化)
# - 双向查询: 代码→记忆/任务, 记忆/任务→代码
# - Delta 变更追踪 (_pending_changes, 用于 WebSocket 广播)

import logging
import json
import os
import time
import uuid
import hashlib
import threading
import atexit
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set

logger = logging.getLogger(__name__)


# ============================================================
# 数据模型
# ============================================================

@dataclass
class CodeAnchor:
    """代码锚点 — 将记忆/任务关联到具体代码位置
    
    锚点稳定性优先级: symbol > content_hash > file_path+line_range > commit_sha
    """
    id: str                         # UUID
    file_path: str                  # 相对于项目根的文件路径
    symbol: Optional[str]           # 函数/类/变量名（最稳定的标识符）
    line_start: Optional[int]       # 起始行
    line_end: Optional[int]         # 结束行
    commit_sha: Optional[str]       # 关联时的 commit
    content_hash: Optional[str]     # 代码片段 hash（检测内容变化）
    anchor_type: str                # implementation / affected / created / deleted
    snippet_preview: str            # 代码预览（前2-3行）
    owner_ref: str                  # "mg:{node_id}" 或 "tg:{graph_id}/{task_node_id}"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "symbol": self.symbol,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "commit_sha": self.commit_sha,
            "content_hash": self.content_hash,
            "anchor_type": self.anchor_type,
            "snippet_preview": self.snippet_preview,
            "owner_ref": self.owner_ref,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeAnchor":
        """从字典反序列化"""
        return cls(
            id=data["id"],
            file_path=data["file_path"],
            symbol=data.get("symbol"),
            line_start=data.get("line_start"),
            line_end=data.get("line_end"),
            commit_sha=data.get("commit_sha"),
            content_hash=data.get("content_hash"),
            anchor_type=data.get("anchor_type", "implementation"),
            snippet_preview=data.get("snippet_preview", ""),
            owner_ref=data.get("owner_ref", ""),
            created_at=data.get("created_at", time.time()),
        )

    def to_summary(self) -> str:
        """生成一行摘要字符串 (用于 LLM 上下文中的 code_ref_summary)
        
        格式: "anchor_type: file_path:symbol Lstart-end"
        """
        parts = [self.anchor_type]
        loc = self.file_path
        if self.symbol:
            loc += f":{self.symbol}"
        if self.line_start is not None:
            if self.line_end is not None and self.line_end != self.line_start:
                loc += f" L{self.line_start}-{self.line_end}"
            else:
                loc += f" L{self.line_start}"
        parts.append(loc)
        return ": ".join(parts)


# ============================================================
# 辅助函数
# ============================================================

def compute_content_hash(content: str) -> str:
    """计算代码内容的 SHA-256 哈希前缀 (16字符)"""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def get_current_commit_sha() -> Optional[str]:
    """尝试获取当前 git HEAD commit SHA (失败返回 None)"""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


# ============================================================
# CodeAnchorStore — 单例存储管理器
# ============================================================

class CodeAnchorStore:
    """代码锚点存储管理器
    
    提供锚点的 CRUD、反向索引查询和 JSON 持久化。
    使用 _pending_changes 追踪变更，支持 WebSocket 增量广播。
    """

    def __init__(self, data_dir: str = None):
        # 数据目录
        if data_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(project_root, "data", "memory_graph")
        self._data_dir = data_dir
        self._file_path = os.path.join(data_dir, "code_anchors.json")

        # 核心数据结构
        self._anchors: Dict[str, CodeAnchor] = {}       # anchor_id -> CodeAnchor
        self._file_index: Dict[str, Set[str]] = {}      # file_path -> anchor_ids
        self._owner_index: Dict[str, Set[str]] = {}     # owner_ref -> anchor_ids

        # 线程安全
        self._lock = threading.Lock()

        # 变更追踪 (用于广播)
        self._pending_changes: List[Dict[str, Any]] = []

        # 持久化
        self._dirty = False
        self._auto_save_delay = 3  # 秒
        self._auto_save_timer: Optional[threading.Timer] = None

        # 加载已有数据
        self._load()

        # 进程退出时保存
        atexit.register(self._atexit_flush)

    # ─── 公开 API ───────────────────────────────────────────

    def add_anchor(self, anchor: CodeAnchor) -> str:
        """添加一个锚点，返回 anchor_id"""
        with self._lock:
            self._anchors[anchor.id] = anchor

            # 更新文件索引
            if anchor.file_path not in self._file_index:
                self._file_index[anchor.file_path] = set()
            self._file_index[anchor.file_path].add(anchor.id)

            # 更新 owner 索引
            if anchor.owner_ref:
                if anchor.owner_ref not in self._owner_index:
                    self._owner_index[anchor.owner_ref] = set()
                self._owner_index[anchor.owner_ref].add(anchor.id)

            # 记录变更
            self._pending_changes.append({
                "action": "add_anchor",
                "data": anchor.to_dict(),
            })
            self._mark_dirty()

        return anchor.id

    def remove_anchor(self, anchor_id: str) -> bool:
        """移除一个锚点"""
        with self._lock:
            anchor = self._anchors.pop(anchor_id, None)
            if anchor is None:
                return False

            # 清理文件索引
            file_ids = self._file_index.get(anchor.file_path)
            if file_ids:
                file_ids.discard(anchor_id)
                if not file_ids:
                    del self._file_index[anchor.file_path]

            # 清理 owner 索引
            if anchor.owner_ref:
                owner_ids = self._owner_index.get(anchor.owner_ref)
                if owner_ids:
                    owner_ids.discard(anchor_id)
                    if not owner_ids:
                        del self._owner_index[anchor.owner_ref]

            # 记录变更
            self._pending_changes.append({
                "action": "remove_anchor",
                "data": {"id": anchor_id, "file_path": anchor.file_path, "owner_ref": anchor.owner_ref},
            })
            self._mark_dirty()

        return True

    def get_anchor(self, anchor_id: str) -> Optional[CodeAnchor]:
        """获取单个锚点"""
        return self._anchors.get(anchor_id)

    def get_anchors_by_file(self, file_path: str) -> List[CodeAnchor]:
        """通过文件路径查询所有锚点（反向索引: 代码→记忆/任务）"""
        anchor_ids = self._file_index.get(file_path, set())
        return [self._anchors[aid] for aid in anchor_ids if aid in self._anchors]

    def get_anchors_by_owner(self, owner_ref: str) -> List[CodeAnchor]:
        """通过 owner_ref 查询所有锚点（正向索引: 记忆/任务→代码）"""
        anchor_ids = self._owner_index.get(owner_ref, set())
        return [self._anchors[aid] for aid in anchor_ids if aid in self._anchors]

    def get_all_anchors(self) -> List[CodeAnchor]:
        """获取所有锚点"""
        return list(self._anchors.values())

    def query_by_file_and_symbol(
        self,
        file_path: str,
        symbol: Optional[str] = None,
        line_start: Optional[int] = None,
        line_end: Optional[int] = None,
    ) -> List[CodeAnchor]:
        """组合查询 — 按文件+符号+行范围过滤
        
        稳定性优先: symbol 精确匹配 > 行范围重叠
        """
        candidates = self.get_anchors_by_file(file_path)
        if not candidates:
            return []

        # 按 symbol 过滤
        if symbol:
            symbol_matches = [a for a in candidates if a.symbol == symbol]
            if symbol_matches:
                return symbol_matches
            # symbol 不精确匹配时尝试部分匹配
            partial = [a for a in candidates if a.symbol and symbol in a.symbol]
            if partial:
                return partial

        # 按行范围过滤
        if line_start is not None and line_end is not None:
            range_matches = []
            for a in candidates:
                if a.line_start is not None and a.line_end is not None:
                    # 判断范围重叠
                    if a.line_start <= line_end and a.line_end >= line_start:
                        range_matches.append(a)
            if range_matches:
                return range_matches

        # 无过滤条件，返回该文件所有锚点
        return candidates

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            "total_anchors": len(self._anchors),
            "total_files": len(self._file_index),
            "total_owners": len(self._owner_index),
        }

    def flush_changes(self) -> List[Dict[str, Any]]:
        """消费并返回 pending_changes (用于广播)"""
        with self._lock:
            changes = list(self._pending_changes)
            self._pending_changes.clear()
        return changes

    # ─── 持久化 ───────────────────────────────────────────

    def save(self) -> None:
        """保存到 JSON 文件"""
        with self._lock:
            data = {
                "version": "1.0",
                "saved_at": time.time(),
                "anchors": {aid: a.to_dict() for aid, a in self._anchors.items()},
            }

        # 确保目录存在
        os.makedirs(self._data_dir, exist_ok=True)

        # 原子写入
        tmp_path = self._file_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # 原子替换
            if os.path.exists(self._file_path):
                os.replace(tmp_path, self._file_path)
            else:
                os.rename(tmp_path, self._file_path)
            self._dirty = False
            logger.debug(f"[CodeAnchorStore] 已保存 {len(self._anchors)} 个锚点")
        except Exception as e:
            logger.error(f"[CodeAnchorStore] 保存失败: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _load(self) -> None:
        """从 JSON 文件加载"""
        if not os.path.exists(self._file_path):
            logger.info(f"[CodeAnchorStore] 数据文件不存在，启动为空 store")
            return

        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            anchors_data = data.get("anchors", {})
            for aid, a_dict in anchors_data.items():
                anchor = CodeAnchor.from_dict(a_dict)
                self._anchors[anchor.id] = anchor

                # 重建文件索引
                if anchor.file_path not in self._file_index:
                    self._file_index[anchor.file_path] = set()
                self._file_index[anchor.file_path].add(anchor.id)

                # 重建 owner 索引
                if anchor.owner_ref:
                    if anchor.owner_ref not in self._owner_index:
                        self._owner_index[anchor.owner_ref] = set()
                    self._owner_index[anchor.owner_ref].add(anchor.id)

            logger.info(
                f"[CodeAnchorStore] 已加载 {len(self._anchors)} 个锚点, "
                f"{len(self._file_index)} 个文件, {len(self._owner_index)} 个 owner"
            )
        except Exception as e:
            logger.error(f"[CodeAnchorStore] 加载失败: {e}")

    def _mark_dirty(self) -> None:
        """标记脏数据，启动防抖自动保存"""
        self._dirty = True
        if self._auto_save_timer:
            self._auto_save_timer.cancel()
        self._auto_save_timer = threading.Timer(self._auto_save_delay, self._do_auto_save)
        self._auto_save_timer.daemon = True
        self._auto_save_timer.start()

    def _do_auto_save(self) -> None:
        """防抖定时器触发的自动保存"""
        if self._dirty:
            self.save()

    def _atexit_flush(self) -> None:
        """进程退出时强制保存"""
        if self._auto_save_timer:
            self._auto_save_timer.cancel()
        if self._dirty:
            self.save()


# ============================================================
# 单例工厂
# ============================================================

_store_instance: Optional[CodeAnchorStore] = None
_store_lock = threading.Lock()


def get_code_anchor_store(data_dir: str = None) -> CodeAnchorStore:
    """获取 CodeAnchorStore 单例（懒加载）"""
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                _store_instance = CodeAnchorStore(data_dir=data_dir)
    return _store_instance


# ============================================================
# 便捷函数
# ============================================================

def create_code_anchor(
    file_path: str,
    owner_ref: str,
    anchor_type: str = "implementation",
    symbol: Optional[str] = None,
    line_start: Optional[int] = None,
    line_end: Optional[int] = None,
    snippet_preview: str = "",
    commit_sha: Optional[str] = None,
    content_hash: Optional[str] = None,
) -> CodeAnchor:
    """创建并持久化一个 CodeAnchor
    
    自动生成 UUID、尝试获取 commit_sha（若未提供）。
    """
    anchor_id = uuid.uuid4().hex[:12]

    # 自动获取 commit SHA
    if commit_sha is None:
        commit_sha = get_current_commit_sha()

    anchor = CodeAnchor(
        id=anchor_id,
        file_path=file_path,
        symbol=symbol,
        line_start=line_start,
        line_end=line_end,
        commit_sha=commit_sha,
        content_hash=content_hash,
        anchor_type=anchor_type,
        snippet_preview=snippet_preview,
        owner_ref=owner_ref,
    )

    store = get_code_anchor_store()
    store.add_anchor(anchor)
    return anchor


def build_code_ref_summary(anchors: List[CodeAnchor], max_items: int = 3) -> str:
    """构建 code_ref_summary 一行摘要
    
    格式: "impl: auth.py:login L42-87 | affected: db.py:UserModel"
    """
    if not anchors:
        return ""
    summaries = [a.to_summary() for a in anchors[:max_items]]
    result = " | ".join(summaries)
    if len(anchors) > max_items:
        result += f" (+{len(anchors) - max_items} more)"
    return result
