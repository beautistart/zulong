# File: zulong/plugins/motor/l1a_motor_plugin.py
"""
L1-A 电机控制插件 (重构版)

TSD v1.7 对应:
- 2.2.1 L0 执行与驱动层
- 4.1 L1-A 受控反射引擎

功能:
- 障碍检测反射 (自动刹车)
- 电机速度控制
- 充电状态管理

架构:
- 实现 IL1Module 接口
- 通过 shared_memory 与其他模块通信
- 异常隔离，不影响其他插件
"""

import logging
from typing import Any, Dict, List
import time

from zulong.modules.l1.core.interface import (
    IL1Module, L1PluginBase, ZulongEvent, EventPriority, EventType, create_event
)

logger = logging.getLogger(__name__)


class L1A_MotorPlugin(L1PluginBase):
    """
    L1-A 电机控制插件
    
    职责:
    - 读取超声波/红外传感器数据 (通过 shared_memory)
    - 执行障碍检测反射 (自动刹车)
    - 控制电机速度/方向
    
    输入 (shared_memory):
    - "obstacle.distance": 障碍物距离 (米)
    - "motor.target_speed": 目标速度 (0.0-1.0)
    - "motor.mode": 模式 ("manual", "auto", "charging")
    
    输出 (ZulongEvent):
    - SENSOR_OBSTACLE: 障碍检测事件
    - ACTION_MOTOR: 电机控制指令
    
    TSD v1.7 对应:
    - 4.1 L1-A 受控反射引擎
    - 3.2 智能路由逻辑 (HIGH 优先级)
    """
    
    @property
    def module_id(self) -> str:
        return "L1A/Motor"
    
    @property
    def priority(self) -> EventPriority:
        # HIGH 优先级：障碍检测需要快速响应 (<50ms)
        return EventPriority.HIGH
    
    def initialize(self, shared_memory: Dict) -> bool:
        """
        初始化电机插件
        
        Args:
            shared_memory: 共享内存
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("🔌 [Motor] 正在初始化...")
            
            # 调用基类初始化
            if not super().initialize(shared_memory):
                return False
            
            # 读取配置
            self._obstacle_threshold = self.get_config("obstacle_threshold", 0.3)  # 0.3 米
            self._max_speed = self.get_config("max_speed", 1.0)
            self._brake_decay = self.get_config("brake_decay", 0.5)  # 刹车衰减系数
            
            # 初始化共享内存
            shared_memory["motor.speed"] = 0.0
            shared_memory["motor.mode"] = "auto"
            shared_memory["obstacle.detected"] = False
            shared_memory["obstacle.distance"] = float('inf')
            
            # 🎯 模拟硬件初始化 (实际应连接真实电机驱动)
            self._motor_enabled = True
            self._current_speed = 0.0
            self._last_obstacle_time = 0.0
            
            logger.info(f"✅ [Motor] 初始化完成 (障碍阈值：{self._obstacle_threshold}m)")
            return True
            
        except Exception as e:
            logger.error(f"❌ [Motor] 初始化失败：{e}", exc_info=True)
            return False
    
    def process_cycle(self, shared_memory: Dict) -> List[ZulongEvent]:
        """
        单周期处理 (核心逻辑)
        
        流程:
        1. 读取障碍物距离
        2. 判断是否需要刹车
        3. 计算目标速度
        4. 产生事件
        
        Args:
            shared_memory: 共享内存
        
        Returns:
            List[ZulongEvent]: 事件列表
        """
        events: List[ZulongEvent] = []
        current_time = time.time()
        
        try:
            # ========== 1. 读取传感器数据 ==========
            obstacle_distance = shared_memory.get("obstacle.distance", float('inf'))
            target_speed = shared_memory.get("motor.target_speed", 0.0)
            motor_mode = shared_memory.get("motor.mode", "auto")
            
            # ========== 2. 障碍检测反射 (TSD v1.7 第 4.1 节) ==========
            obstacle_detected = obstacle_distance < self._obstacle_threshold
            
            if obstacle_detected:
                # 记录障碍事件
                self._last_obstacle_time = current_time
                shared_memory["obstacle.detected"] = True
                shared_memory["obstacle.distance"] = obstacle_distance
                
                # 🔥 关键：产生 HIGH 优先级障碍事件 (路由给 L1-A)
                obstacle_event = create_event(
                    event_type=EventType.SENSOR_OBSTACLE,
                    priority=EventPriority.HIGH,
                    source=self.module_id,
                    distance=obstacle_distance,
                    direction="front",
                    requires_brake=True
                )
                events.append(obstacle_event)
                logger.debug(f"🚨 [Motor] 障碍检测：{obstacle_distance:.2f}m")
            
            # ========== 3. 速度控制逻辑 ==========
            if motor_mode == "charging":
                # 充电中：禁止移动
                final_speed = 0.0
            
            elif obstacle_detected:
                # 检测到障碍：自动刹车
                # 距离越近，速度越低
                safety_factor = obstacle_distance / self._obstacle_threshold
                final_speed = target_speed * min(safety_factor, 1.0) * self._brake_decay
                
                logger.debug(f"🛑 [Motor] 自动刹车：{target_speed:.2f} → {final_speed:.2f}")
            
            else:
                # 正常行驶
                final_speed = min(target_speed, self._max_speed)
                shared_memory["obstacle.detected"] = False
            
            # ========== 4. 更新共享内存 ==========
            shared_memory["motor.speed"] = final_speed
            
            # ========== 5. 产生电机控制事件 ==========
            if abs(final_speed - self._current_speed) > 0.01:
                # 速度变化超过阈值，产生 ACTION_MOTOR 事件
                motor_event = create_event(
                    event_type=EventType.ACTION_MOTOR,
                    priority=EventPriority.NORMAL,
                    source=self.module_id,
                    speed=final_speed,
                    direction="forward" if final_speed > 0 else "stop"
                )
                events.append(motor_event)
                self._current_speed = final_speed
            
            # ========== 6. 性能监控 ==========
            if current_time - self._last_obstacle_time > 5.0:
                # 5 秒内无障碍，清除标志
                shared_memory["obstacle.detected"] = False
            
        except Exception as e:
            logger.error(f"❌ [Motor] process_cycle 错误：{e}", exc_info=True)
            # 异常隔离：不抛出异常，返回空事件列表
        
        return events
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            status = "OK"
            details = {
                "motor_enabled": self._motor_enabled,
                "current_speed": self._current_speed,
                "obstacle_threshold": self._obstacle_threshold,
                "last_obstacle_time": self._last_obstacle_time
            }
            
            # 检查电机状态
            if not self._motor_enabled:
                status = "ERROR"
                details["error"] = "Motor disabled"
            
            return {
                "status": status,
                "details": details,
                "last_update": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ [Motor] 健康检查失败：{e}")
            return {
                "status": "ERROR",
                "details": {"error": str(e)},
                "last_update": time.time()
            }
    
    def shutdown(self):
        """关闭电机插件"""
        try:
            logger.info("🛑 [Motor] 正在关闭...")
            
            # 停止电机
            self._motor_enabled = False
            self._current_speed = 0.0
            
            # 调用基类关闭
            super().shutdown()
            
            logger.info("✅ [Motor] 已关闭")
            
        except Exception as e:
            logger.error(f"❌ [Motor] 关闭失败：{e}", exc_info=True)
    
    # ========== 事件回调 (可选实现) ==========
    
    def on_event(self, event: ZulongEvent, shared_memory: Dict):
        """
        处理外部事件
        
        处理的事件:
        - ACTION_MOTOR: 接收 L2 或 L1-B 的电机控制指令
        """
        try:
            if event.type == EventType.ACTION_MOTOR:
                # 接收电机控制指令
                speed = event.payload.get("speed", 0.0)
                shared_memory["motor.target_speed"] = speed
                logger.debug(f"📥 [Motor] 接收速度指令：{speed}")
            
            elif event.type == EventType.SYSTEM_L2_COMMAND:
                # 接收 L2 命令 (如 "去充电")
                command = event.payload.get("command", "")
                if "充电" in command or "charging" in command.lower():
                    shared_memory["motor.mode"] = "charging"
                    logger.info("🔋 [Motor] 切换到充电模式")
        
        except Exception as e:
            logger.error(f"❌ [Motor] 事件处理失败：{e}", exc_info=True)


# ========== 工厂函数 (便于插件管理器动态加载) ==========

def create_plugin(config: Dict = None) -> IL1Module:
    """工厂函数：创建插件实例"""
    return L1A_MotorPlugin(config=config)
