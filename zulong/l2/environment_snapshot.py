# File: zulong/l2/environment_snapshot.py
# 环境快照管理器 - 保存和对比环境状态
# 对应 TSD v1.7: L2 重评估机制 - 恢复任务前对比环境变化

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time
import threading
import logging
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ObjectState:
    """物体状态"""
    object_id: str
    name: str
    position: Optional[Dict[str, float]] = None  # {x, y, z}
    status: str = "unknown"  # on_ground, held, moved, disappeared
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "name": self.name,
            "position": self.position,
            "status": self.status,
            "timestamp": self.timestamp
        }


@dataclass
class UserState:
    """用户状态"""
    user_id: str = "default"
    position: Optional[Dict[str, float]] = None  # {x, y, z}
    activity: str = "unknown"  # stationary, moving, speaking, silent
    last_interaction: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "position": self.position,
            "activity": self.activity,
            "last_interaction": self.last_interaction
        }


@dataclass
class TaskCondition:
    """任务条件"""
    condition_id: str
    description: str
    is_satisfied: bool = True
    check_method: str = "visual"  # visual, audio, temporal
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "description": self.description,
            "is_satisfied": self.is_satisfied,
            "check_method": self.check_method
        }


@dataclass
class EnvironmentSnapshot:
    """环境快照 - 记录某一时刻的环境状态
    
    TSD v1.7 对应:
    - 5.3 重评估机制：恢复任务前对比环境快照
    - 5.4 执行前确认：检查物体状态、用户位置、任务条件
    """
    snapshot_id: str
    timestamp: float = field(default_factory=time.time)
    task_id: Optional[str] = None  # 关联的任务 ID
    
    # 物体状态列表
    objects: List[ObjectState] = field(default_factory=list)
    
    # 用户状态
    user: Optional[UserState] = None
    
    # 任务条件
    conditions: List[TaskCondition] = field(default_factory=list)
    
    # 视觉场景描述
    scene_description: str = ""
    
    # 音频环境
    audio_context: str = ""
    
    def add_object(self, obj: ObjectState):
        """添加物体状态"""
        self.objects.append(obj)
    
    def set_user(self, user: UserState):
        """设置用户状态"""
        self.user = user
    
    def add_condition(self, condition: TaskCondition):
        """添加任务条件"""
        self.conditions.append(condition)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "objects": [obj.to_dict() for obj in self.objects],
            "user": self.user.to_dict() if self.user else None,
            "conditions": [c.to_dict() for c in self.conditions],
            "scene_description": self.scene_description,
            "audio_context": self.audio_context
        }


@dataclass
class EnvironmentChange:
    """环境变化检测结果"""
    has_changes: bool = False
    changes: List[str] = field(default_factory=list)
    severity: str = "none"  # none, minor, major, critical
    recommendation: str = "CONTINUE"  # CONTINUE, REPLAN, ABORT
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_changes": self.has_changes,
            "changes": self.changes,
            "severity": self.severity,
            "recommendation": self.recommendation
        }


class EnvironmentSnapshotManager:
    """环境快照管理器（单例模式）
    
    核心功能:
    1. 创建环境快照 (create_snapshot)
    2. 对比环境变化 (compare_snapshots)
    3. 检测关键变化 (物体掉落、用户移动、条件消失)
    4. 提供重评估建议 (CONTINUE / REPLAN / ABORT)
    
    类比：就像游戏存档时保存世界状态
    - 记录所有物体的位置
    - 记录 NPC（用户）的位置
    - 记录任务条件是否满足
    - 读档时对比，决定是否需要重新规划
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EnvironmentSnapshotManager, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """初始化环境快照管理器"""
        self._snapshots: Dict[str, EnvironmentSnapshot] = {}
        self._max_snapshots = 10  # 最多保存 10 个环境快照
        logger.info("EnvironmentSnapshotManager initialized")
    
    def create_snapshot(self, task_id: str = None) -> EnvironmentSnapshot:
        """创建当前环境快照
        
        Args:
            task_id: 关联的任务 ID
            
        Returns:
            EnvironmentSnapshot: 环境快照
        """
        import uuid
        
        snapshot = EnvironmentSnapshot(
            snapshot_id=str(uuid.uuid4()),
            timestamp=time.time(),
            task_id=task_id
        )
        
        # 🎯 TODO: 从视觉/听觉系统获取实际状态
        # 这里先创建空快照，后续会填充实际数据
        
        # 示例：添加物体状态
        # snapshot.add_object(ObjectState(
        #     object_id="cup_001",
        #     name="水杯",
        #     position={"x": 0.5, "y": 0.8, "z": 0.2},
        #     status="on_table"
        # ))
        
        # 示例：添加用户状态
        # snapshot.set_user(UserState(
        #     user_id="user_001",
        #     position={"x": 1.0, "y": 2.0, "z": 0.0},
        #     activity="stationary"
        # ))
        
        # 保存快照
        self._snapshots[snapshot.snapshot_id] = snapshot
        
        # 限制快照数量
        if len(self._snapshots) > self._max_snapshots:
            # 删除最旧的快照
            oldest_id = min(self._snapshots.keys(), 
                          key=lambda k: self._snapshots[k].timestamp)
            del self._snapshots[oldest_id]
        
        logger.info(f"Created environment snapshot: {snapshot.snapshot_id}")
        return snapshot
    
    def compare_snapshots(self, old: EnvironmentSnapshot, 
                         new: EnvironmentSnapshot) -> EnvironmentChange:
        """对比两个环境快照，检测变化
        
        Args:
            old: 旧快照
            new: 新快照
            
        Returns:
            EnvironmentChange: 变化检测结果
        """
        changes = []
        severity = "none"
        recommendation = "CONTINUE"
        
        # 1. 检测物体变化
        old_objects = {obj.object_id: obj for obj in old.objects}
        new_objects = {obj.object_id: obj for obj in new.objects}
        
        # 检查消失的物体
        for obj_id, old_obj in old_objects.items():
            if obj_id not in new_objects:
                changes.append(f"物体消失：{old_obj.name}")
                severity = "major"
                recommendation = "REPLAN"
        
        # 检查位置变化的物体
        for obj_id, new_obj in new_objects.items():
            if obj_id in old_objects:
                old_obj = old_objects[obj_id]
                if self._position_changed(old_obj.position, new_obj.position):
                    changes.append(f"物体移动：{new_obj.name}")
                    if severity == "none":
                        severity = "minor"
                        recommendation = "REPLAN"  # 物体移动需要重新规划
            else:
                changes.append(f"新物体出现：{new_obj.name}")
        
        # 2. 检测用户位置变化
        if old.user and new.user:
            if self._position_changed(old.user.position, new.user.position):
                changes.append(f"用户移动")
                if severity in ["none", "minor"]:
                    severity = "minor"
                    recommendation = "REPLAN"  # 用户移动可能需要调整策略
        
        # 3. 检测任务条件变化
        old_conditions = {c.condition_id: c for c in old.conditions}
        new_conditions = {c.condition_id: c for c in new.conditions}
        
        for cond_id, old_cond in old_conditions.items():
            new_cond = new_conditions.get(cond_id)
            if new_cond:
                if old_cond.is_satisfied and not new_cond.is_satisfied:
                    changes.append(f"任务条件不再满足：{old_cond.description}")
                    severity = "critical"
                    recommendation = "ABORT"
        
        # 4. 检测场景变化
        if old.scene_description != new.scene_description:
            if old.scene_description and new.scene_description:
                changes.append(f"场景变化")
                if severity == "none":
                    severity = "minor"
        
        return EnvironmentChange(
            has_changes=len(changes) > 0,
            changes=changes,
            severity=severity,
            recommendation=recommendation
        )
    
    def _position_changed(self, pos1: Optional[Dict], 
                         pos2: Optional[Dict], threshold: float = 0.5) -> bool:
        """检测位置是否发生变化
        
        Args:
            pos1: 位置 1
            pos2: 位置 2
            threshold: 变化阈值（米）
            
        Returns:
            bool: 是否发生变化
        """
        if pos1 is None and pos2 is None:
            return False
        if pos1 is None or pos2 is None:
            return True
        
        # 计算欧几里得距离
        dx = pos1.get("x", 0) - pos2.get("x", 0)
        dy = pos1.get("y", 0) - pos2.get("y", 0)
        dz = pos1.get("z", 0) - pos2.get("z", 0)
        distance = (dx**2 + dy**2 + dz**2) ** 0.5
        
        return distance > threshold
    
    def get_latest_snapshot(self, task_id: str = None) -> Optional[EnvironmentSnapshot]:
        """获取最新的环境快照
        
        Args:
            task_id: 任务 ID（可选）
            
        Returns:
            EnvironmentSnapshot: 最新的快照，如果没有则返回 None
        """
        if not self._snapshots:
            return None
        
        if task_id:
            # 查找指定任务的最新快照
            task_snapshots = [s for s in self._snapshots.values() 
                            if s.task_id == task_id]
            if task_snapshots:
                return max(task_snapshots, key=lambda s: s.timestamp)
        
        # 返回时间最新的快照
        return max(self._snapshots.values(), key=lambda s: s.timestamp)


# 全局环境快照管理器实例
environment_snapshot_manager = EnvironmentSnapshotManager()
