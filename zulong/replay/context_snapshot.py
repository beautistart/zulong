"""
AT-04: L2/L3上下文快照接口
开发接口，允许在任意时刻抓取L2的推理逻辑和L3的参数
关键指标: 能够序列化输出当前的Prompt输入和IK解算参数
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import logging
import json
import time

from .clock_synchronizer import get_unified_timestamp
from .ring_buffer import get_ring_buffer

logger = logging.getLogger(__name__)


class L2ContextType(Enum):
    PROMPT_INPUT = "prompt_input"
    LLM_OUTPUT = "llm_output"
    TASK_PLAN = "task_plan"
    REASONING_CHAIN = "reasoning_chain"
    INTERRUPT_CONTEXT = "interrupt_context"


class L3ContextType(Enum):
    IK_PARAMS = "ik_params"
    TRAJECTORY = "trajectory"
    PID_PARAMS = "pid_params"
    VISION_CONFIG = "vision_config"
    MANIPULATION_CONFIG = "manipulation_config"
    NAV_CONFIG = "nav_config"


@dataclass
class L2Snapshot:
    """L2推理引擎快照"""
    snapshot_id: str
    timestamp: float
    task_id: str
    prompt_input: str
    prompt_tokens: int
    llm_output: Optional[str] = None
    output_tokens: int = 0
    reasoning_chain: List[str] = field(default_factory=list)
    task_plan: Dict[str, Any] = field(default_factory=dict)
    context_window: List[Dict[str, Any]] = field(default_factory=list)
    kv_cache_size: int = 0
    model_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "prompt_input": self.prompt_input,
            "prompt_tokens": self.prompt_tokens,
            "llm_output": self.llm_output,
            "output_tokens": self.output_tokens,
            "reasoning_chain": self.reasoning_chain,
            "task_plan": self.task_plan,
            "context_window": self.context_window,
            "kv_cache_size": self.kv_cache_size,
            "model_name": self.model_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'L2Snapshot':
        return cls(
            snapshot_id=data["snapshot_id"],
            timestamp=data["timestamp"],
            task_id=data["task_id"],
            prompt_input=data["prompt_input"],
            prompt_tokens=data["prompt_tokens"],
            llm_output=data.get("llm_output"),
            output_tokens=data.get("output_tokens", 0),
            reasoning_chain=data.get("reasoning_chain", []),
            task_plan=data.get("task_plan", {}),
            context_window=data.get("context_window", []),
            kv_cache_size=data.get("kv_cache_size", 0),
            model_name=data.get("model_name", "")
        )


@dataclass
class L3Snapshot:
    """L3专家模块快照"""
    snapshot_id: str
    timestamp: float
    expert_type: str
    ik_params: Optional[Dict[str, Any]] = None
    trajectory: Optional[List[Dict[str, Any]]] = None
    pid_params: Optional[Dict[str, Any]] = None
    vision_config: Optional[Dict[str, Any]] = None
    manipulation_config: Optional[Dict[str, Any]] = None
    nav_config: Optional[Dict[str, Any]] = None
    joint_limits: Optional[Dict[str, Any]] = None
    kinematics_params: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "expert_type": self.expert_type,
            "ik_params": self.ik_params,
            "trajectory": self.trajectory,
            "pid_params": self.pid_params,
            "vision_config": self.vision_config,
            "manipulation_config": self.manipulation_config,
            "nav_config": self.nav_config,
            "joint_limits": self.joint_limits,
            "kinematics_params": self.kinematics_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'L3Snapshot':
        return cls(
            snapshot_id=data["snapshot_id"],
            timestamp=data["timestamp"],
            expert_type=data["expert_type"],
            ik_params=data.get("ik_params"),
            trajectory=data.get("trajectory"),
            pid_params=data.get("pid_params"),
            vision_config=data.get("vision_config"),
            manipulation_config=data.get("manipulation_config"),
            nav_config=data.get("nav_config"),
            joint_limits=data.get("joint_limits"),
            kinematics_params=data.get("kinematics_params")
        )


class ContextSnapshotManager:
    """
    上下文快照管理器
    
    提供L2/L3的上下文快照接口，支持序列化输出
    """
    
    def __init__(self, ring_buffer=None):
        self._ring_buffer = ring_buffer or get_ring_buffer()
        self._l2_snapshots: Dict[str, L2Snapshot] = {}
        self._l3_snapshots: Dict[str, L3Snapshot] = {}
        self._snapshot_counter = 0
        
        self._stats = {
            "l2_snapshots_created": 0,
            "l3_snapshots_created": 0,
            "snapshots_serialized": 0
        }
        
        logger.info("[ContextSnapshotManager] 初始化完成")
    
    def generate_snapshot_id(self, layer: str) -> str:
        """生成快照ID"""
        self._snapshot_counter += 1
        return f"{layer}_SNAP_{get_unified_timestamp():.0f}_{self._snapshot_counter}"
    
    def capture_l2_snapshot(
        self,
        task_id: str,
        prompt_input: str,
        prompt_tokens: int,
        llm_output: Optional[str] = None,
        output_tokens: int = 0,
        reasoning_chain: Optional[List[str]] = None,
        task_plan: Optional[Dict[str, Any]] = None,
        context_window: Optional[List[Dict[str, Any]]] = None,
        kv_cache_size: int = 0,
        model_name: str = ""
    ) -> str:
        """
        捕获L2推理快照
        
        Args:
            task_id: 任务ID
            prompt_input: 输入提示词
            prompt_tokens: 输入token数
            llm_output: LLM输出
            output_tokens: 输出token数
            reasoning_chain: 推理链
            task_plan: 任务计划
            context_window: 上下文窗口
            kv_cache_size: KV Cache大小
            model_name: 模型名称
        
        Returns:
            str: 快照ID
        """
        snapshot_id = self.generate_snapshot_id("L2")
        timestamp = get_unified_timestamp()
        
        snapshot = L2Snapshot(
            snapshot_id=snapshot_id,
            timestamp=timestamp,
            task_id=task_id,
            prompt_input=prompt_input,
            prompt_tokens=prompt_tokens,
            llm_output=llm_output,
            output_tokens=output_tokens,
            reasoning_chain=reasoning_chain or [],
            task_plan=task_plan or {},
            context_window=context_window or [],
            kv_cache_size=kv_cache_size,
            model_name=model_name
        )
        
        self._l2_snapshots[snapshot_id] = snapshot
        self._stats["l2_snapshots_created"] += 1
        
        self._ring_buffer.write_event(
            layer="L2",
            event_type="context_snapshot",
            data=snapshot.to_dict(),
            metadata={"snapshot_id": snapshot_id, "task_id": task_id}
        )
        
        logger.debug(f"[ContextSnapshotManager] L2快照创建: {snapshot_id}")
        return snapshot_id
    
    def capture_l3_snapshot(
        self,
        expert_type: str,
        ik_params: Optional[Dict[str, Any]] = None,
        trajectory: Optional[List[Dict[str, Any]]] = None,
        pid_params: Optional[Dict[str, Any]] = None,
        vision_config: Optional[Dict[str, Any]] = None,
        manipulation_config: Optional[Dict[str, Any]] = None,
        nav_config: Optional[Dict[str, Any]] = None,
        joint_limits: Optional[Dict[str, Any]] = None,
        kinematics_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        捕获L3专家模块快照
        
        Args:
            expert_type: 专家类型
            ik_params: IK参数
            trajectory: 轨迹
            pid_params: PID参数
            vision_config: 视觉配置
            manipulation_config: 操作配置
            nav_config: 导航配置
            joint_limits: 关节限制
            kinematics_params: 运动学参数
        
        Returns:
            str: 快照ID
        """
        snapshot_id = self.generate_snapshot_id("L3")
        timestamp = get_unified_timestamp()
        
        snapshot = L3Snapshot(
            snapshot_id=snapshot_id,
            timestamp=timestamp,
            expert_type=expert_type,
            ik_params=ik_params,
            trajectory=trajectory,
            pid_params=pid_params,
            vision_config=vision_config,
            manipulation_config=manipulation_config,
            nav_config=nav_config,
            joint_limits=joint_limits,
            kinematics_params=kinematics_params
        )
        
        self._l3_snapshots[snapshot_id] = snapshot
        self._stats["l3_snapshots_created"] += 1
        
        self._ring_buffer.write_event(
            layer="L3",
            event_type="context_snapshot",
            data=snapshot.to_dict(),
            metadata={"snapshot_id": snapshot_id, "expert_type": expert_type}
        )
        
        logger.debug(f"[ContextSnapshotManager] L3快照创建: {snapshot_id}")
        return snapshot_id
    
    def get_l2_snapshot(self, snapshot_id: str) -> Optional[L2Snapshot]:
        """获取L2快照"""
        return self._l2_snapshots.get(snapshot_id)
    
    def get_l3_snapshot(self, snapshot_id: str) -> Optional[L3Snapshot]:
        """获取L3快照"""
        return self._l3_snapshots.get(snapshot_id)
    
    def get_latest_l2_snapshot(self) -> Optional[L2Snapshot]:
        """获取最新的L2快照"""
        if not self._l2_snapshots:
            return None
        return self._l2_snapshots[max(self._l2_snapshots.keys(), key=lambda k: self._l2_snapshots[k].timestamp)]
    
    def get_latest_l3_snapshot(self, expert_type: Optional[str] = None) -> Optional[L3Snapshot]:
        """获取最新的L3快照"""
        if not self._l3_snapshots:
            return None
        
        filtered = self._l3_snapshots
        if expert_type:
            filtered = {k: v for k, v in self._l3_snapshots.items() if v.expert_type == expert_type}
        
        if not filtered:
            return None
        return filtered[max(filtered.keys(), key=lambda k: filtered[k].timestamp)]
    
    def serialize_snapshot(self, snapshot_id: str, format: str = "json") -> Optional[str]:
        """
        序列化快照
        
        Args:
            snapshot_id: 快照ID
            format: 格式 (json/pickle)
        
        Returns:
            str: 序列化后的字符串
        """
        l2_snapshot = self._l2_snapshots.get(snapshot_id)
        if l2_snapshot:
            self._stats["snapshots_serialized"] += 1
            if format == "json":
                return json.dumps(l2_snapshot.to_dict(), ensure_ascii=False, indent=2)
        
        l3_snapshot = self._l3_snapshots.get(snapshot_id)
        if l3_snapshot:
            self._stats["snapshots_serialized"] += 1
            if format == "json":
                return json.dumps(l3_snapshot.to_dict(), ensure_ascii=False, indent=2)
        
        return None
    
    def serialize_all(self, format: str = "json") -> Dict[str, str]:
        """序列化所有快照"""
        result = {}
        
        for snapshot_id, snapshot in self._l2_snapshots.items():
            if format == "json":
                result[snapshot_id] = json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2)
        
        for snapshot_id, snapshot in self._l3_snapshots.items():
            if format == "json":
                result[snapshot_id] = json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2)
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "l2_snapshots": len(self._l2_snapshots),
            "l3_snapshots": len(self._l3_snapshots),
            "stats": self._stats.copy()
        }


_global_snapshot_manager: Optional[ContextSnapshotManager] = None


def get_snapshot_manager() -> ContextSnapshotManager:
    """获取全局上下文快照管理器"""
    global _global_snapshot_manager
    if _global_snapshot_manager is None:
        _global_snapshot_manager = ContextSnapshotManager()
    return _global_snapshot_manager
