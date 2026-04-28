"""
已完成任务存档管理器

当复杂任务正常完成后，将完整状态持久化到磁盘，
便于用户后期随时查看历史任务的详情、执行结果和关联文件。

设计模式与 TaskSuspensionManager 完全对称。
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CompletedTaskArchive:
    """已完成任务的完整存档"""
    task_id: str                          # 任务唯一标识（复用 request_id）
    description: str                      # 用户目标描述
    final_answer: str                     # 最终回答
    duration: float                       # 总耗时（秒）
    total_turns: int                      # Agent 推理轮次
    completion_status: str                # "completed" | "exhausted" | "stopped"
    task_graph_snapshot: Dict = field(default_factory=dict)   # TaskGraph.serialize()
    workspace_dir: str = ""               # 工作目录路径
    created_at: float = field(default_factory=time.time)      # 任务开始时间
    completed_at: float = field(default_factory=time.time)    # 任务完成时间
    metadata: Dict = field(default_factory=dict)              # 扩展信息

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'CompletedTaskArchive':
        """从字典反序列化"""
        return cls(
            task_id=data["task_id"],
            description=data.get("description", ""),
            final_answer=data.get("final_answer", ""),
            duration=data.get("duration", 0.0),
            total_turns=data.get("total_turns", 0),
            completion_status=data.get("completion_status", "completed"),
            task_graph_snapshot=data.get("task_graph_snapshot", {}),
            workspace_dir=data.get("workspace_dir", ""),
            created_at=data.get("created_at", 0),
            completed_at=data.get("completed_at", 0),
            metadata=data.get("metadata", {}),
        )


class CompletedTaskArchiveManager:
    """已完成任务存档管理器（单例）"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        cfg = config or {}
        self.enabled = cfg.get("enabled", True)
        self._persistence_path = cfg.get("persistence_path", "./data/completed_tasks")
        self._max_archived_tasks = cfg.get("max_archived_tasks", 200)
        self._max_age_days = cfg.get("max_age_days", 30)

        os.makedirs(self._persistence_path, exist_ok=True)
        logger.info(f"[TaskArchive] 初始化完成，持久化路径: {self._persistence_path}")

    async def archive_task(self, state: CompletedTaskArchive) -> str:
        """归档已完成任务到磁盘，返回 task_id"""
        if not self.enabled:
            logger.warning("[TaskArchive] 归档功能已禁用")
            return ""

        # 检查归档数量上限
        existing = await self.list_tasks(limit=0)
        if len(existing) >= self._max_archived_tasks:
            oldest = min(existing, key=lambda x: x["completed_at"])
            await self.delete_task(oldest["task_id"])
            logger.info(
                f"[TaskArchive] 达到上限({self._max_archived_tasks})，"
                f"已清理最旧归档: {oldest['task_id']}"
            )

        state.completed_at = time.time()

        file_path = os.path.join(self._persistence_path, f"{state.task_id}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(
                f"[TaskArchive] 任务已归档: {state.task_id} "
                f"(status={state.completion_status}, turns={state.total_turns}, "
                f"duration={state.duration:.1f}s)"
            )

            # 增量更新语义索引
            try:
                from zulong.memory.task_search_index import (
                    get_task_search_index, TaskIndexEntry,
                )
                idx = get_task_search_index()
                idx.add_entry(TaskIndexEntry(
                    entry_id=state.task_id,
                    title=state.description,
                    source="completed",
                    file_path=file_path,
                ))
                idx.save()
            except Exception as _idx_err:
                logger.debug(f"[TaskArchive] 语义索引更新失败（非致命）: {_idx_err}")

            return state.task_id
        except Exception as e:
            logger.error(f"[TaskArchive] 归档失败: {e}")
            return ""

    async def get_task(self, task_id: str) -> Optional[CompletedTaskArchive]:
        """按 ID 获取完整归档（不删除文件）"""
        file_path = os.path.join(self._persistence_path, f"{task_id}.json")
        if not os.path.exists(file_path):
            logger.warning(f"[TaskArchive] 归档不存在: {task_id}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return CompletedTaskArchive.from_dict(data)
        except Exception as e:
            logger.error(f"[TaskArchive] 读取归档失败: {e}")
            return None

    async def list_tasks(self, limit: int = 20) -> List[Dict]:
        """列出归档摘要列表，按完成时间倒序

        Args:
            limit: 返回数量上限，0 表示返回全部
        """
        tasks = []
        if not os.path.exists(self._persistence_path):
            return tasks

        for filename in os.listdir(self._persistence_path):
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(self._persistence_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tasks.append({
                    "task_id": data["task_id"],
                    "description": data.get("description", ""),
                    "completion_status": data.get("completion_status", "completed"),
                    "completed_at": data.get("completed_at", 0),
                    "duration": data.get("duration", 0),
                    "total_turns": data.get("total_turns", 0),
                    "workspace_dir": data.get("workspace_dir", ""),
                })
            except Exception as e:
                logger.warning(f"[TaskArchive] 读取归档摘要失败 {filename}: {e}")

        # 按完成时间倒序
        tasks.sort(key=lambda x: x["completed_at"], reverse=True)

        if limit > 0:
            tasks = tasks[:limit]
        return tasks

    async def search_tasks(self, query: str) -> List[Dict]:
        """按描述模糊搜索归档任务

        支持中文匹配：使用字符级子串匹配而非空格分词。
        """
        tasks = await self.list_tasks(limit=0)
        if not tasks:
            return []

        query_lower = query.lower()

        # 去除查询指令关键词，保留实际描述部分
        query_words = ['查看', '历史', '任务', '完成', '做过', '之前', '的', '过']
        query_clean = query_lower
        for w in query_words:
            query_clean = query_clean.replace(w, '')
        query_clean = query_clean.strip()

        if not query_clean:
            # 无具体描述，直接按时间倒序返回
            return tasks[:10]

        scored = []
        for task in tasks:
            desc = task.get("description", "").lower()
            score = 0

            if query_clean in desc:
                score += 10
            elif desc in query_clean:
                score += 8

            for char in query_clean:
                if char in desc and len(char.strip()) > 0:
                    score += 1

            if score > 0:
                scored.append((score, task))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:10]]

    async def delete_task(self, task_id: str) -> bool:
        """删除归档"""
        file_path = os.path.join(self._persistence_path, f"{task_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"[TaskArchive] 归档已删除: {task_id}")
            # 同步移除语义索引
            try:
                from zulong.memory.task_search_index import get_task_search_index
                get_task_search_index().remove_entry(task_id)
            except Exception:
                pass
            return True
        return False

    async def cleanup_expired(self, max_age_days: Optional[float] = None) -> int:
        """清理过期归档，返回清理数量"""
        age_limit = max_age_days or self._max_age_days
        cutoff_time = time.time() - age_limit * 86400
        cleaned = 0

        if not os.path.exists(self._persistence_path):
            return 0

        for filename in os.listdir(self._persistence_path):
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(self._persistence_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("completed_at", 0) < cutoff_time:
                    os.remove(file_path)
                    cleaned += 1
                    logger.info(
                        f"[TaskArchive] 清理过期归档: {data.get('task_id', filename)}"
                    )
            except Exception as e:
                logger.warning(f"[TaskArchive] 清理文件失败 {filename}: {e}")

        if cleaned > 0:
            logger.info(f"[TaskArchive] 共清理 {cleaned} 个过期归档")
        return cleaned
