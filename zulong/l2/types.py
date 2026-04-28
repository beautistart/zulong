# File: zulong/l2/types.py
# 任务快照类型定义 - 第四阶段细化版
# 对应 TSD v1.7: 任务冻结与恢复机制

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time
import uuid


class TaskStatus(Enum):
    """任务状态"""
    RUNNING = auto()    # 正在运行
    FROZEN = auto()     # 已冻结
    COMPLETED = auto()  # 已完成


@dataclass
class TaskSnapshot:
    """任务快照 - 保存任务的完整状态
    
    对应 TSD v1.7 核心原则：状态完整性
    - 保存完整的上下文、工作变量和执行指针
    """
    # 基础信息
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    status: TaskStatus = TaskStatus.RUNNING
    
    # 核心状态
    context_history: List[Dict] = field(default_factory=list)  # 对话历史 [{'role': 'user', 'content': '...'}]
    working_memory: Dict[str, Any] = field(default_factory=dict)  # 工作变量
    execution_pointer: str = ""  # 断点描述，如 "generating_step_5"
    priority: int = 0  # 优先级
    
    def update(self, **kwargs):
        """增量更新快照"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_updated = time.time()
    
    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "task_id": self.task_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "status": self.status.name,
            "context_history": self.context_history,
            "working_memory": self.working_memory,
            "execution_pointer": self.execution_pointer,
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TaskSnapshot":
        """从字典反序列化"""
        return cls(
            task_id=data.get("task_id", str(uuid.uuid4())),
            created_at=data.get("created_at", time.time()),
            last_updated=data.get("last_updated", time.time()),
            status=TaskStatus[data.get("status", "RUNNING")],
            context_history=data.get("context_history", []),
            working_memory=data.get("working_memory", {}),
            execution_pointer=data.get("execution_pointer", ""),
            priority=data.get("priority", 0)
        )
    
    def get_summary(self) -> str:
        """获取快照摘要"""
        return (
            f"Task[{self.task_id[:8]}] "
            f"Status={self.status.name}, "
            f"Context={len(self.context_history)} msgs, "
            f"Pointer={self.execution_pointer}"
        )
