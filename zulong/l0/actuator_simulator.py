# File: zulong/l0/actuator_simulator.py
# L0 执行器模拟器 - 执行命令
# 集成 AT-02: L0执行层日志扩展

from zulong.core.event_bus import event_bus
from zulong.core.types import EventType, ZulongEvent
from zulong.replay.l0_logger import get_l0_logger
import logging
import time
import random

logging.basicConfig(level=logging.INFO, format='[L0] %(message)s')
logger = logging.getLogger(__name__)


class ActuatorSimulator:
    """执行器模拟器"""
    
    def __init__(self):
        self.l0_logger = get_l0_logger()
        self._joint_states = {
            "joint_1": {"position": 0.0, "velocity": 0.0, "torque": 0.0},
            "joint_2": {"position": 0.0, "velocity": 0.0, "torque": 0.0},
            "joint_3": {"position": 0.0, "velocity": 0.0, "torque": 0.0},
        }
        self.register_callbacks()
    
    def register_callbacks(self):
        """注册命令回调"""
        event_bus.subscribe(EventType.CMD_EMERGENCY_STOP, self.handle_emergency_stop, "L0-Actuator")
        event_bus.subscribe(EventType.CMD_BRAKE, self.handle_brake, "L0-Actuator")
        event_bus.subscribe(EventType.CMD_SLOW_DOWN, self.handle_slow_down, "L0-Actuator")
        event_bus.subscribe(EventType.CMD_BACKUP, self.handle_backup, "L0-Actuator")
        logger.info("L0 ActuatorSimulator initialized and subscribed to command events")
    
    def _log_joint_states(self):
        """记录关节状态 (AT-02)"""
        for joint_id, state in self._joint_states.items():
            self.l0_logger.log_joint_state(
                joint_id=joint_id,
                position=state["position"],
                velocity=state["velocity"],
                torque=state["torque"],
                current=random.uniform(1.0, 3.0),
                temperature=random.uniform(30.0, 45.0)
            )
    
    def handle_emergency_stop(self, event: ZulongEvent):
        """处理紧急停止命令"""
        reason = event.payload.get("reason", "")
        
        cmd_id = self.l0_logger.log_command_issued(
            command_type="emergency_stop",
            target_position=0.0,
            target_velocity=0.0,
            duration_ms=0
        )
        
        self.l0_logger.log_command_executing(cmd_id)
        
        for joint_id in self._joint_states:
            self._joint_states[joint_id]["velocity"] = 0.0
            self._joint_states[joint_id]["torque"] = 0.0
        
        self._log_joint_states()
        
        self.l0_logger.log_command_completed(cmd_id, actual_position=0.0, success=True)
        
        logger.info(f"✅ Executing EMERGENCY_STOP - {reason}")
    
    def handle_brake(self, event: ZulongEvent):
        """处理刹车命令"""
        reason = event.payload.get("reason", "")
        
        cmd_id = self.l0_logger.log_command_issued(
            command_type="brake",
            target_position=self._joint_states["joint_1"]["position"],
            target_velocity=0.0,
            duration_ms=100
        )
        
        self.l0_logger.log_command_executing(cmd_id)
        
        for joint_id in self._joint_states:
            self._joint_states[joint_id]["velocity"] *= 0.5
        
        self._log_joint_states()
        
        self.l0_logger.log_command_completed(cmd_id, success=True)
        
        logger.info(f"✅ Executing BRAKE - {reason}")
    
    def handle_slow_down(self, event: ZulongEvent):
        """处理减速命令"""
        reason = event.payload.get("reason", "")
        
        cmd_id = self.l0_logger.log_command_issued(
            command_type="slow_down",
            target_position=self._joint_states["joint_1"]["position"],
            target_velocity=self._joint_states["joint_1"]["velocity"] * 0.5,
            duration_ms=200
        )
        
        self.l0_logger.log_command_executing(cmd_id)
        
        for joint_id in self._joint_states:
            self._joint_states[joint_id]["velocity"] *= 0.7
        
        self._log_joint_states()
        
        self.l0_logger.log_command_completed(cmd_id, success=True)
        
        logger.info(f"✅ Executing SLOW_DOWN - {reason}")
    
    def handle_backup(self, event: ZulongEvent):
        """处理后退命令"""
        reason = event.payload.get("reason", "")
        
        cmd_id = self.l0_logger.log_command_issued(
            command_type="backup",
            target_position=self._joint_states["joint_1"]["position"] - 0.5,
            target_velocity=-0.3,
            duration_ms=500
        )
        
        self.l0_logger.log_command_executing(cmd_id)
        
        for joint_id in self._joint_states:
            self._joint_states[joint_id]["position"] -= 0.1
            self._joint_states[joint_id]["velocity"] = -0.3
        
        self._log_joint_states()
        
        self.l0_logger.log_command_completed(cmd_id, success=True)
        
        logger.info(f"✅ Executing BACKUP - {reason}")
    
    def execute_motion(self, target_positions: dict, duration_ms: int = 1000) -> str:
        """
        执行运动命令 (通用接口)
        
        Args:
            target_positions: 目标位置字典 {"joint_id": position}
            duration_ms: 执行时间
        
        Returns:
            str: 命令ID
        """
        cmd_id = self.l0_logger.log_command_issued(
            command_type="motion",
            target_position=list(target_positions.values())[0] if target_positions else 0.0,
            target_velocity=0.5,
            duration_ms=duration_ms
        )
        
        self.l0_logger.log_command_executing(cmd_id)
        
        for joint_id, target_pos in target_positions.items():
            if joint_id in self._joint_states:
                self._joint_states[joint_id]["position"] = target_pos
                self._joint_states[joint_id]["velocity"] = 0.5
        
        self._log_joint_states()
        
        self.l0_logger.log_command_completed(cmd_id, success=True)
        
        return cmd_id


actuator_simulator = ActuatorSimulator()
