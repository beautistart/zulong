# File: zulong/l2/task_snapshot.py
# 任务快照系统 - L2 中断恢复的核心
# 对应 TSD v1.7: L2 任务冻结与恢复机制

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum, auto
import time
import uuid

from .environment_snapshot import EnvironmentSnapshot


class TaskStatus(Enum):
    """任务状态"""
    RUNNING = auto()      # 正在执行
    FROZEN = auto()       # 已冻结（中断保存）
    COMPLETED = auto()    # 已完成
    ABANDONED = auto()    # 已放弃


@dataclass
class ExecutionPointer:
    """执行进度指针
    
    类比：就像你在读一本书时夹的书签，记录读到哪一页、哪一行
    """
    task_type: str = ""                    # 任务类型 (storytelling, coding, navigation...)
    current_step: int = 0                  # 当前步骤索引
    total_steps: int = 0                   # 总步骤数
    step_description: str = ""             # 当前步骤描述
    generated_content: str = ""            # 已生成的内容（如故事已讲到的段落）
    progress_percentage: float = 0.0       # 进度百分比 (0-100)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "task_type": self.task_type,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "step_description": self.step_description,
            "generated_content": self.generated_content,
            "progress_percentage": self.progress_percentage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionPointer":
        """从字典反序列化"""
        return cls(
            task_type=data.get("task_type", ""),
            current_step=data.get("current_step", 0),
            total_steps=data.get("total_steps", 0),
            step_description=data.get("step_description", ""),
            generated_content=data.get("generated_content", ""),
            progress_percentage=data.get("progress_percentage", 0.0)
        )


@dataclass
class IntentFrame:
    """意图栈帧
    
    类比：就像俄罗斯套娃，大意图里可能包含多个子意图
    例如："导航到厨房" -> [找路，避障，开门]
    """
    intent: str = ""                       # 意图名称
    parameters: Dict[str, Any] = field(default_factory=dict)  # 意图参数
    priority: int = 0                      # 优先级
    created_at: float = field(default_factory=time.time)  # 创建时间
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "intent": self.intent,
            "parameters": self.parameters,
            "priority": self.priority,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntentFrame":
        """从字典反序列化"""
        return cls(
            intent=data.get("intent", ""),
            parameters=data.get("parameters", {}),
            priority=data.get("priority", 0),
            created_at=data.get("created_at", time.time())
        )


@dataclass
class KVCacheSnapshot:
    """KV Cache 快照
    
    注意：实际 KV Cache 是 GPU 显存中的张量，这里只保存元数据
    真正的缓存数据由 ModelContainer 管理
    """
    cache_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    num_tokens: int = 0                    # Token 数量
    memory_size_mb: float = 0.0            # 内存占用 (MB)
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_id": self.cache_id,
            "num_tokens": self.num_tokens,
            "memory_size_mb": self.memory_size_mb,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KVCacheSnapshot":
        return cls(
            cache_id=data.get("cache_id", str(uuid.uuid4())),
            num_tokens=data.get("num_tokens", 0),
            memory_size_mb=data.get("memory_size_mb", 0.0),
            created_at=data.get("created_at", time.time())
        )


@dataclass
class TaskSnapshot:
    """任务快照 - L2 在任意时刻的完整状态封装
    
    类比：就像游戏里的"存档点"，保存了当前所有进度
    当需要中断时，可以冻结保存；恢复时，从存档点继续
    
    包含：
    - context_window: 当前的对话历史/思维链 (Token 序列)
    - working_memory: 短期工作变量
    - execution_pointer: 执行进度指针
    - intent_stack: 未完成的子意图栈
    - kv_cache: KV Cache 元数据（实际数据由 ModelContainer 管理）
    - environment_snapshot: 环境快照（用于重评估）
    """
    # 基础信息
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""                      # 任务唯一标识
    task_name: str = ""                    # 任务名称（用于人类识别）
    status: TaskStatus = TaskStatus.RUNNING
    created_at: float = field(default_factory=time.time)
    frozen_at: Optional[float] = None      # 冻结时间
    resumed_at: Optional[float] = None     # 恢复时间
    
    # 核心状态
    context_window: List[Dict[str, Any]] = field(default_factory=list)  # 对话历史
    working_memory: Dict[str, Any] = field(default_factory=dict)        # 工作记忆
    execution_pointer: ExecutionPointer = field(default_factory=ExecutionPointer)  # 执行指针
    intent_stack: List[IntentFrame] = field(default_factory=list)       # 意图栈
    kv_cache: Optional[KVCacheSnapshot] = None                          # KV Cache 快照
    
    # 🎯 TSD v1.7 新增：环境快照（用于重评估）
    environment_snapshot: Optional[EnvironmentSnapshot] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def freeze(self) -> None:
        """冻结任务 - 标记为已保存状态"""
        self.status = TaskStatus.FROZEN
        self.frozen_at = time.time()
    
    def thaw(self) -> None:
        """解冻任务 - 标记为恢复执行"""
        self.status = TaskStatus.RUNNING
        self.resumed_at = time.time()
    
    def update_progress(self, step: int, description: str = "", content: str = "") -> None:
        """更新执行进度"""
        self.execution_pointer.current_step = step
        if description:
            self.execution_pointer.step_description = description
        if content:
            self.execution_pointer.generated_content = content
        if self.execution_pointer.total_steps > 0:
            self.execution_pointer.progress_percentage = (
                step / self.execution_pointer.total_steps * 100
            )
    
    def push_intent(self, intent: str, parameters: Dict[str, Any] = None, priority: int = 0) -> None:
        """压入新意图到栈顶"""
        frame = IntentFrame(
            intent=intent,
            parameters=parameters or {},
            priority=priority
        )
        self.intent_stack.append(frame)
    
    def pop_intent(self) -> Optional[IntentFrame]:
        """弹出栈顶意图"""
        if self.intent_stack:
            return self.intent_stack.pop()
        return None
    
    def peek_intent(self) -> Optional[IntentFrame]:
        """查看栈顶意图（不弹出）"""
        if self.intent_stack:
            return self.intent_stack[-1]
        return None
    
    def add_to_context(self, role: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """添加消息到上下文窗口"""
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.context_window.append(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """完整序列化为字典"""
        return {
            "snapshot_id": self.snapshot_id,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "status": self.status.name,
            "created_at": self.created_at,
            "frozen_at": self.frozen_at,
            "resumed_at": self.resumed_at,
            "context_window": self.context_window,
            "working_memory": self.working_memory,
            "execution_pointer": self.execution_pointer.to_dict(),
            "intent_stack": [frame.to_dict() for frame in self.intent_stack],
            "kv_cache": self.kv_cache.to_dict() if self.kv_cache else None,
            "environment_snapshot": self.environment_snapshot.to_dict() if self.environment_snapshot else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskSnapshot":
        """从字典完整反序列化"""
        snapshot = cls(
            snapshot_id=data.get("snapshot_id", str(uuid.uuid4())),
            task_id=data.get("task_id", ""),
            task_name=data.get("task_name", ""),
            status=TaskStatus[data.get("status", "RUNNING")],
            created_at=data.get("created_at", time.time()),
            frozen_at=data.get("frozen_at"),
            resumed_at=data.get("resumed_at"),
            context_window=data.get("context_window", []),
            working_memory=data.get("working_memory", {}),
            execution_pointer=ExecutionPointer.from_dict(data.get("execution_pointer", {})),
            intent_stack=[IntentFrame.from_dict(f) for f in data.get("intent_stack", [])],
            kv_cache=KVCacheSnapshot.from_dict(data["kv_cache"]) if data.get("kv_cache") else None,
            environment_snapshot=None,  # 简化处理，不反序列化环境快照
            metadata=data.get("metadata", {})
        )
        return snapshot
    
    def get_summary(self) -> str:
        """获取快照摘要（用于日志显示）"""
        current_intent = self.peek_intent()
        return (
            f"TaskSnapshot[{self.task_name}] "
            f"Status={self.status.name}, "
            f"Progress={self.execution_pointer.progress_percentage:.1f}%, "
            f"Intent={current_intent.intent if current_intent else 'None'}, "
            f"Context={len(self.context_window)} msgs"
        )
