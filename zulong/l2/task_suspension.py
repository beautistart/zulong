"""
任务挂起/恢复管理器

对于超长程任务（跨天/月），不能在一次推理循环中完成。
当 Circuit Breaker 触发 RED 且任务标记为可挂起时：
1. 序列化当前完整状态（对话历史、CB 状态、已完成迭代数等）
2. 持久化到磁盘
3. 下次用户恢复时，从断点继续执行
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SuspendableTaskState:
    """可挂起任务的完整状态快照"""
    task_id: str
    description: str               # 任务简述（用于后续匹配恢复）
    messages: List[Dict]           # 对话历史
    accumulated_links: str         # 累积链接
    circuit_breaker_state: Dict    # CB 序列化状态
    iteration_count: int           # 已完成的迭代数
    task_graph: Optional[Any] = None  # TaskGraph 实例（运行时）
    task_graph_serialized: Optional[Dict] = None  # TaskGraph 序列化数据（持久化）
    created_at: float = field(default_factory=time.time)
    suspended_at: float = field(default_factory=time.time)
    suspended_reason: str = "complexity"  # "time_limit" | "user_requested" | "complexity"
    metadata: Dict = field(default_factory=dict)  # 扩展信息
    memory_snapshot: Optional[Dict] = None  # MemoryGraph 激活快照（焦点/激活值）

    def to_dict(self) -> Dict:
        """序列化为字典（避免 asdict 的深拷贝触发不可序列化对象）"""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "messages": self.messages,
            "accumulated_links": self.accumulated_links,
            "circuit_breaker_state": self.circuit_breaker_state,
            "iteration_count": self.iteration_count,
            "task_graph_serialized": self.task_graph_serialized,
            "created_at": self.created_at,
            "suspended_at": self.suspended_at,
            "suspended_reason": self.suspended_reason,
            "metadata": self.metadata,
            "memory_snapshot": self.memory_snapshot,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SuspendableTaskState':
        """从字典反序列化"""
        state = cls(
            task_id=data["task_id"],
            description=data["description"],
            messages=data["messages"],
            accumulated_links=data.get("accumulated_links", ""),
            circuit_breaker_state=data.get("circuit_breaker_state", {}),
            iteration_count=data.get("iteration_count", 0),
            task_graph_serialized=data.get("task_graph_serialized"),
            created_at=data.get("created_at", time.time()),
            suspended_at=data.get("suspended_at", time.time()),
            suspended_reason=data.get("suspended_reason", "complexity"),
            metadata=data.get("metadata", {}),
            memory_snapshot=data.get("memory_snapshot"),
        )
        
        # 尝试反序列化 TaskGraph
        if state.task_graph_serialized:
            try:
                from zulong.l2.task_graph import TaskGraph
                state.task_graph = TaskGraph.deserialize(state.task_graph_serialized)
            except Exception as e:
                logger.warning(f"[TaskSuspension] 反序列化 TaskGraph 失败: {e}")
        
        return state


class TaskSuspensionManager:
    """任务挂起/恢复管理器（单例）"""

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
        self._persistence_path = cfg.get("persistence_path", "./data/suspended_tasks")
        self._max_suspended_tasks = cfg.get("max_suspended_tasks", 20)
        self._auto_suspend_on_time_limit = cfg.get("auto_suspend_on_time_limit", True)
        self._max_age_hours = cfg.get("max_age_hours", 72)

        # 确保持久化目录存在
        os.makedirs(self._persistence_path, exist_ok=True)
        logger.info(f"[TaskSuspension] 初始化完成，持久化路径: {self._persistence_path}")

    async def suspend_task(self, state: SuspendableTaskState) -> str:
        """挂起任务，持久化到磁盘，返回 task_id"""
        if not self.enabled:
            logger.warning("[TaskSuspension] 挂起功能已禁用")
            return ""

        # 检查挂起任务数量上限
        existing = await self.list_suspended_tasks()
        if len(existing) >= self._max_suspended_tasks:
            # 清理最旧的任务腾出空间
            oldest = min(existing, key=lambda x: x["suspended_at"])
            await self.cancel_task(oldest["task_id"])
            logger.info(f"[TaskSuspension] 达到上限({self._max_suspended_tasks})，已清理最旧任务: {oldest['task_id']}")

        state.suspended_at = time.time()

        # 清洗 messages：将不可序列化的对象转为字典
        sanitized_messages = []
        for msg in state.messages:
            if isinstance(msg, dict):
                sanitized_messages.append(msg)
            elif hasattr(msg, '__dict__'):
                # ChatCompletionMessage 等 OpenAI 对象
                sanitized_messages.append(self._serialize_message_object(msg))
            else:
                sanitized_messages.append({"role": "unknown", "content": str(msg)})
        state.messages = sanitized_messages

        # 序列化 TaskGraph（如果存在）
        if state.task_graph is not None:
            try:
                state.task_graph_serialized = state.task_graph.serialize()
                logger.info(f"[TaskSuspension] 已序列化 TaskGraph: {state.task_graph.id}")
            except Exception as e:
                logger.warning(f"[TaskSuspension] 序列化 TaskGraph 失败: {e}")

        # 捕获 MemoryGraph 激活快照（焦点上下文 + 活跃节点）
        try:
            from zulong.memory.memory_graph import get_memory_graph
            _mg = get_memory_graph()
            if _mg:
                _focus_ctx = _mg.get_last_focus_context()
                _active_ids = _mg.get_active_node_ids()
                state.memory_snapshot = {
                    "focus_context": _focus_ctx,
                    "active_node_ids": _active_ids,
                }
                logger.info(
                    f"[TaskSuspension] MemoryGraph 快照已保存: "
                    f"focus={'有' if _focus_ctx else '无'}, "
                    f"active_nodes={len(_active_ids)}"
                )
        except Exception as e:
            logger.info(f"[TaskSuspension] MemoryGraph 快照跳过: {e}")

        # 原子写入磁盘：temp file → fsync → os.replace
        # 防止写入过程中崩溃导致 JSON 损坏，任务永久丢失
        file_path = os.path.join(self._persistence_path, f"{state.task_id}.json")
        tmp_path = file_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, file_path)
            logger.info(f"[TaskSuspension] 任务已挂起: {state.task_id} (reason={state.suspended_reason}, iters={state.iteration_count})")
            return state.task_id
        except Exception as e:
            logger.error(f"[TaskSuspension] 挂起失败: {e}")
            # 清理可能残留的临时文件
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            return ""

    async def resume_task(self, task_id: str, consume: bool = True) -> Optional[SuspendableTaskState]:
        """恢复任务，返回状态快照

        Args:
            task_id: 任务 ID
            consume: 是否消费（删除）磁盘文件。
                     True  = 正式恢复，删除文件防止重复恢复（默认）
                     False = 只读加载，不删除文件（用于预览/匹配）
        """
        file_path = os.path.join(self._persistence_path, f"{task_id}.json")
        if not os.path.exists(file_path):
            logger.warning(f"[TaskSuspension] 任务不存在: {task_id}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            state = SuspendableTaskState.from_dict(data)

            # 恢复 MemoryGraph 激活状态
            if consume and state.memory_snapshot:
                try:
                    from zulong.memory.memory_graph import get_memory_graph
                    _mg = get_memory_graph()
                    if _mg:
                        _snap = state.memory_snapshot
                        # 恢复活跃节点
                        _active_ids = _snap.get("active_node_ids", [])
                        if _active_ids:
                            _mg.set_active_nodes(_active_ids)
                        # 恢复焦点上下文
                        _focus_ctx = _snap.get("focus_context")
                        if _focus_ctx and isinstance(_focus_ctx, dict):
                            _mg.set_last_focus_context(
                                dialogue_round_id=_focus_ctx.get("dialogue_round_id", ""),
                                focused_task_node_id=_focus_ctx.get("focused_task_node_id", ""),
                                active_node_ids=_focus_ctx.get("active_node_ids", []),
                                focus_path=_focus_ctx.get("focus_path", []),
                                focus_depth=_focus_ctx.get("focus_depth", 0),
                            )
                        logger.info(
                            f"[TaskSuspension] MemoryGraph 状态已恢复: "
                            f"active_nodes={len(_active_ids)}"
                        )
                except Exception as _mg_err:
                    logger.info(f"[TaskSuspension] MemoryGraph 恢复跳过: {_mg_err}")

            if consume:
                # 恢复后删除磁盘文件（避免重复恢复），TOCTOU 安全
                try:
                    os.remove(file_path)
                except FileNotFoundError:
                    logger.debug(f"[TaskSuspension] 文件已被其他线程删除: {task_id}")
                logger.info(f"[TaskSuspension] 任务已恢复并消费: {task_id}")
            else:
                logger.info(f"[TaskSuspension] 任务已加载（只读）: {task_id}")
            return state
        except Exception as e:
            logger.error(f"[TaskSuspension] 恢复失败: {e}")
            return None

    async def list_suspended_tasks(self) -> List[Dict]:
        """列出所有挂起的任务（摘要信息）"""
        tasks = []
        if not os.path.exists(self._persistence_path):
            return tasks

        for filename in os.listdir(self._persistence_path):
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(self._persistence_path, filename)
            try:
                if os.path.getsize(file_path) == 0:
                    logger.warning(f"[TaskSuspension] 跳过零字节文件: {filename}")
                    continue
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tasks.append({
                    "task_id": data["task_id"],
                    "description": data.get("description", ""),
                    "suspended_at": data.get("suspended_at", 0),
                    "iteration_count": data.get("iteration_count", 0),
                    "suspended_reason": data.get("suspended_reason", "unknown"),
                    "metadata": data.get("metadata", {}),
                })
            except Exception as e:
                logger.warning(f"[TaskSuspension] 读取任务文件失败 {filename}: {e}")
        return tasks

    async def cancel_task(self, task_id: str) -> bool:
        """取消挂起的任务"""
        file_path = os.path.join(self._persistence_path, f"{task_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"[TaskSuspension] 任务已取消: {task_id}")
            return True
        return False

    async def cleanup_expired(self, max_age_hours: Optional[float] = None) -> int:
        """清理过期任务，返回清理数量"""
        age_limit = max_age_hours or self._max_age_hours
        cutoff_time = time.time() - age_limit * 3600
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
                if data.get("suspended_at", 0) < cutoff_time:
                    os.remove(file_path)
                    cleaned += 1
                    logger.info(f"[TaskSuspension] 清理过期任务: {data.get('task_id', filename)}")
            except Exception as e:
                logger.warning(f"[TaskSuspension] 清理文件失败 {filename}: {e}")

        if cleaned > 0:
            logger.info(f"[TaskSuspension] 共清理 {cleaned} 个过期任务")
        return cleaned

    async def find_by_description(self, query: str, return_full_state: bool = False) -> Optional[Dict]:
        """通过描述模糊匹配挂起的任务

        支持中文匹配：使用 bigram 重叠率 + 子串匹配双重评分。
        设置最低匹配阈值，防止不相关任务被错误恢复。

        Args:
            query: 查询字符串（由 L2 模型自主构造）
            return_full_state: 是否返回完整的任务状态（包含 TaskGraph）

        Returns:
            匹配的任务信息或完整状态；匹配质量不足时返回 None
        """
        tasks = await self.list_suspended_tasks()
        if not tasks:
            return None

        # 如果只有一个挂起任务，直接作为最佳匹配
        if len(tasks) == 1:
            best_match = tasks[0]
            best_task_id = tasks[0]["task_id"]
        else:
            query_clean = query.lower().strip()

            best_match = None
            best_score = 0.0
            best_task_id = None

            for task in tasks:
                desc = task.get("description", "").lower()
                score = 0.0

                if query_clean:
                    # 1. 精确子串匹配（最高优先级）
                    if query_clean in desc:
                        score = 10.0
                    elif desc in query_clean:
                        score = 8.0
                    else:
                        # 2. Bigram 重叠率（中文友好的语义匹配）
                        overlap = self._bigram_overlap(query_clean, desc)
                        score = overlap * 10.0  # 将 0~1 映射到 0~10
                else:
                    # 用户只说了"继续"之类的词，无具体描述
                    # 按时间排序，返回最近挂起的
                    score = task.get("suspended_at", 0)

                if score > best_score:
                    best_score = score
                    best_match = task
                    best_task_id = task.get("task_id")

            # 匹配质量门槛：bigram 重叠率 < 30%（score < 3.0）时拒绝匹配
            # 避免"重庆旅游"错误匹配到"北京旅游"
            _MIN_MATCH_SCORE = 3.0
            if query_clean and best_score < _MIN_MATCH_SCORE:
                logger.info(
                    f"[TaskSuspension] 匹配质量不足: query='{query}', "
                    f"best_score={best_score:.1f} < {_MIN_MATCH_SCORE}, "
                    f"best_desc='{best_match.get('description', '') if best_match else ''}'，拒绝匹配"
                )
                return None

            if not best_match:
                # 兜底：匹配不上则返回最近挂起的任务
                best_match = max(tasks, key=lambda t: t.get("suspended_at", 0))
                best_task_id = best_match.get("task_id")

        # 非破坏性加载：只读取完整状态，不删除文件。
        # 调用方确认匹配后应显式调用 resume_task(consume=True) 或 cancel_task() 清理。
        if return_full_state and best_task_id:
            state = await self.resume_task(best_task_id, consume=False)
            return state
        
        return best_match

    @staticmethod
    def _bigram_overlap(a: str, b: str) -> float:
        """计算两个字符串的 bigram 重叠率（Jaccard 系数）

        适用于中文无空格场景。
        Returns:
            0.0 ~ 1.0，值越大越相似
        """
        if not a or not b or len(a) < 2 or len(b) < 2:
            return 0.0
        bigrams_a = {a[i:i+2] for i in range(len(a) - 1)}
        bigrams_b = {b[i:i+2] for i in range(len(b) - 1)}
        intersection = len(bigrams_a & bigrams_b)
        union = len(bigrams_a | bigrams_b)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _serialize_message_object(msg) -> Dict:
        """将 OpenAI ChatCompletionMessage 等对象转为可序列化的字典"""
        result = {"role": getattr(msg, "role", "assistant")}
        if hasattr(msg, "content") and msg.content:
            result["content"] = msg.content
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in msg.tool_calls
            ]
        return result

    @staticmethod
    def generate_task_id() -> str:
        """生成唯一的任务 ID"""
        return f"task_{int(time.time())}_{uuid.uuid4().hex[:8]}"
