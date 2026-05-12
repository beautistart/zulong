"""
AT-02: L0执行层日志扩展
在L0驱动中增加电流、扭矩、实际位置的采样与输出
关键指标: 日志包含指令时间戳、执行时间戳、关节状态
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import logging
import asyncio

from .clock_synchronizer import get_unified_timestamp
from .ring_buffer import RingBufferSlot, get_ring_buffer

logger = logging.getLogger(__name__)


class L0EventType(Enum):
    COMMAND_ISSUED = "command_issued"
    COMMAND_EXECUTING = "command_executing"
    COMMAND_COMPLETED = "command_completed"
    COMMAND_FAILED = "command_failed"
    MOTOR_STATE = "motor_state"
    JOINT_UPDATE = "joint_update"
    CURRENT_SAMPLE = "current_sample"
    TORQUE_SAMPLE = "torque_sample"


@dataclass
class JointState:
    joint_id: str
    position: float
    velocity: float
    torque: float
    current: float
    temperature: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "joint_id": self.joint_id,
            "position": self.position,
            "velocity": self.velocity,
            "torque": self.torque,
            "current": self.current,
            "temperature": self.temperature,
            "timestamp": self.timestamp
        }


@dataclass
class MotorCommand:
    command_id: str
    command_type: str
    target_position: Optional[float] = None
    target_velocity: Optional[float] = None
    target_torque: Optional[float] = None
    duration_ms: Optional[float] = None
    timestamp_issued: float = 0.0
    timestamp_executed: float = 0.0
    status: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "command_type": self.command_type,
            "target_position": self.target_position,
            "target_velocity": self.target_velocity,
            "target_torque": self.target_torque,
            "duration_ms": self.duration_ms,
            "timestamp_issued": self.timestamp_issued,
            "timestamp_executed": self.timestamp_executed,
            "status": self.status
        }


class L0Logger:
    """
    L0执行层日志记录器
    
    记录电机控制、关节状态、电流扭矩等底层执行数据
    """
    
    def __init__(self, ring_buffer=None):
        self._ring_buffer = ring_buffer or get_ring_buffer()
        self._active_commands: Dict[str, MotorCommand] = {}
        self._joint_states: Dict[str, JointState] = {}
        self._command_counter = 0
        
        self._stats = {
            "commands_issued": 0,
            "commands_completed": 0,
            "commands_failed": 0,
            "state_updates": 0
        }
        
        logger.info("[L0Logger] 初始化完成")
    
    def generate_command_id(self) -> str:
        """生成唯一命令ID"""
        self._command_counter += 1
        return f"L0CMD_{get_unified_timestamp():.0f}_{self._command_counter}"
    
    def log_command_issued(
        self,
        command_type: str,
        target_position: Optional[float] = None,
        target_velocity: Optional[float] = None,
        target_torque: Optional[float] = None,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        记录命令发出
        
        Returns:
            str: 命令ID
        """
        command_id = self.generate_command_id()
        timestamp = get_unified_timestamp()
        
        command = MotorCommand(
            command_id=command_id,
            command_type=command_type,
            target_position=target_position,
            target_velocity=target_velocity,
            target_torque=target_torque,
            duration_ms=duration_ms,
            timestamp_issued=timestamp,
            status="issued"
        )
        
        self._active_commands[command_id] = command
        self._stats["commands_issued"] += 1
        
        self._ring_buffer.write_event(
            layer="L0",
            event_type=L0EventType.COMMAND_ISSUED.value,
            data=command.to_dict(),
            metadata=metadata or {"command_id": command_id}
        )
        
        logger.debug(f"[L0Logger] 命令发出: {command_id}, 类型: {command_type}")
        return command_id
    
    def log_command_executing(self, command_id: str):
        """记录命令开始执行"""
        if command_id in self._active_commands:
            command = self._active_commands[command_id]
            command.timestamp_executed = get_unified_timestamp()
            command.status = "executing"
            
            self._ring_buffer.write_event(
                layer="L0",
                event_type=L0EventType.COMMAND_EXECUTING.value,
                data=command.to_dict(),
                metadata={"command_id": command_id}
            )
    
    def log_command_completed(
        self,
        command_id: str,
        actual_position: Optional[float] = None,
        actual_velocity: Optional[float] = None,
        success: bool = True
    ):
        """记录命令完成"""
        if command_id in self._active_commands:
            command = self._active_commands[command_id]
            command.status = "completed" if success else "failed"
            
            data = command.to_dict()
            data["actual_position"] = actual_position
            data["actual_velocity"] = actual_velocity
            
            event_type = L0EventType.COMMAND_COMPLETED.value if success else L0EventType.COMMAND_FAILED.value
            
            self._ring_buffer.write_event(
                layer="L0",
                event_type=event_type,
                data=data,
                metadata={"command_id": command_id, "success": str(success)}
            )
            
            if success:
                self._stats["commands_completed"] += 1
            else:
                self._stats["commands_failed"] += 1
            
            del self._active_commands[command_id]
            logger.debug(f"[L0Logger] 命令完成: {command_id}, 成功: {success}")
    
    def log_joint_state(
        self,
        joint_id: str,
        position: float,
        velocity: float,
        torque: float,
        current: float,
        temperature: float = 25.0
    ):
        """记录关节状态"""
        timestamp = get_unified_timestamp()
        
        joint_state = JointState(
            joint_id=joint_id,
            position=position,
            velocity=velocity,
            torque=torque,
            current=current,
            temperature=temperature,
            timestamp=timestamp
        )
        
        self._joint_states[joint_id] = joint_state
        self._stats["state_updates"] += 1
        
        self._ring_buffer.write_event(
            layer="L0",
            event_type=L0EventType.JOINT_UPDATE.value,
            data=joint_state.to_dict(),
            metadata={"joint_id": joint_id}
        )
    
    def log_current_sample(
        self,
        joint_id: str,
        current: float,
        voltage: Optional[float] = None
    ):
        """记录电流采样"""
        self._ring_buffer.write_event(
            layer="L0",
            event_type=L0EventType.CURRENT_SAMPLE.value,
            data={
                "joint_id": joint_id,
                "current": current,
                "voltage": voltage,
                "timestamp": get_unified_timestamp()
            },
            metadata={"joint_id": joint_id}
        )
    
    def log_torque_sample(
        self,
        joint_id: str,
        torque: float,
        torque_limit: Optional[float] = None
    ):
        """记录扭矩采样"""
        self._ring_buffer.write_event(
            layer="L0",
            event_type=L0EventType.TORQUE_SAMPLE.value,
            data={
                "joint_id": joint_id,
                "torque": torque,
                "torque_limit": torque_limit,
                "timestamp": get_unified_timestamp()
            },
            metadata={"joint_id": joint_id}
        )
    
    def log_motor_state(
        self,
        motor_id: str,
        state: str,
        speed: Optional[float] = None,
        position: Optional[float] = None
    ):
        """记录电机状态"""
        self._ring_buffer.write_event(
            layer="L0",
            event_type=L0EventType.MOTOR_STATE.value,
            data={
                "motor_id": motor_id,
                "state": state,
                "speed": speed,
                "position": position,
                "timestamp": get_unified_timestamp()
            },
            metadata={"motor_id": motor_id}
        )
    
    def get_active_commands(self) -> List[MotorCommand]:
        """获取活动命令列表"""
        return list(self._active_commands.values())
    
    def get_joint_states(self) -> Dict[str, JointState]:
        """获取所有关节状态"""
        return self._joint_states.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "stats": self._stats.copy(),
            "active_commands": len(self._active_commands),
            "tracked_joints": len(self._joint_states)
        }


_global_l0_logger: Optional[L0Logger] = None


def get_l0_logger() -> L0Logger:
    """获取全局L0日志记录器"""
    global _global_l0_logger
    if _global_l0_logger is None:
        _global_l0_logger = L0Logger()
    return _global_l0_logger
