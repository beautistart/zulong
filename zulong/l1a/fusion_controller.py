# File: zulong/l1a/fusion_controller.py
# L1-A 宏微融合执行器 - 三层注意力机制核心组件

"""
祖龙 (ZULONG) 系统 - L1-A 宏微融合执行器

核心功能:
1. 多模态感知融合：整合视觉、听觉、雷达数据
2. 宏观指令解析：将 L2 的 MacroCommand 转换为微观动作
3. 约束检查：验证动作安全性、可行性
4. 实时控制：输出电机 PWM、舵机角度等底层控制信号

TSD v1.8 对应:
- 2.2.0 三层注意力机制 - L1-A 融合执行层
- 3.2.1 宏微融合控制
- 4.3.1 多模态数据流
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from zulong.core.attention_atoms import MacroCommand, AttentionEvent
from zulong.core.types import ZulongEvent, EventType, EventPriority
from zulong.core.event_bus import event_bus

logger = logging.getLogger("FusionController")


@dataclass
class SensorFusionData:
    """
    多模态传感器融合数据
    
    TSD v1.8 对应:
    - 4.3.1 多模态数据流
    """
    # 视觉数据
    vision_target_pos: Optional[List[float]] = None  # [x, y, z] 归一化坐标
    vision_motion_pixels: int = 0
    vision_motion_magnitude: float = 0.0
    vision_last_motion_time: float = 0.0
    
    # 听觉数据 (未来扩展)
    audio_direction: Optional[float] = None  # 声源方向 (角度)
    audio_distance: Optional[float] = None  # 声源距离 (米)
    audio_speech_text: Optional[str] = None  # 语音识别文本
    
    # 雷达数据 (未来扩展)
    radar_obstacles: List[Dict[str, Any]] = None  # 障碍物列表
    radar_distance_front: Optional[float] = None  # 前方距离
    
    def __post_init__(self):
        if self.radar_obstacles is None:
            self.radar_obstacles = []
    
    def is_target_valid(self) -> bool:
        """判断视觉目标是否有效"""
        return (
            self.vision_target_pos is not None and
            len(self.vision_target_pos) == 3 and
            self.vision_motion_pixels > 0
        )
    
    def get_target_position(self) -> Optional[Tuple[float, float, float]]:
        """获取目标 3D 位置"""
        if self.is_target_valid():
            return tuple(self.vision_target_pos)
        return None


@dataclass
class MicroAction:
    """
    微观动作指令
    
    TSD v1.8 对应:
    - 3.2.1 宏微融合控制
    """
    action_id: str
    action_type: str  # "MOVE", "GRASP", "ROTATE", "STOP"
    parameters: Dict[str, Any]  # 动作参数
    priority: int  # 1-10
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class FusionController:
    """
    L1-A 宏微融合执行器
    
    职责:
    - 从共享内存读取多模态感知数据
    - 接收 L2 的宏观指令 (MacroCommand)
    - 融合感知数据生成微观动作
    - 实时输出控制信号到 L0 执行层
    
    TSD v1.8 对应:
    - 2.2.0 三层注意力机制 - L1-A 融合执行层
    - 4.3.1 多模态数据流
    """
    
    def __init__(self):
        """初始化宏微融合执行器"""
        self.event_bus = event_bus
        
        # 多模态融合数据
        self.fusion_data = SensorFusionData()
        
        # 当前宏观指令
        self.current_macro_command: Optional[MacroCommand] = None
        
        # 微观动作队列
        self.micro_action_queue: List[MicroAction] = []
        
        # 安全约束
        self.safety_constraints = {
            'max_speed': 0.5,  # 最大速度 (m/s)
            'max_force': 10.0,  # 最大力度 (N)
            'min_obstacle_distance': 0.3,  # 最小避障距离 (米)
            'emergency_stop': False  # 紧急停止标志
        }
        
        # 运行状态
        self.is_running = False
        self.last_update_time = 0.0
        
        logger.info("🤖 [FusionController] 初始化完成")
        logger.info(f"   - 安全约束：max_speed={self.safety_constraints['max_speed']}m/s")
        logger.info(f"   - 避障距离：{self.safety_constraints['min_obstacle_distance']}m")
    
    def initialize(self):
        """初始化融合控制器"""
        self.is_running = True
        self.last_update_time = time.time()
        logger.info("✅ [FusionController] 已启动")
    
    def update_shared_memory(self, shared_memory: Dict[str, Any]):
        """
        从共享内存更新感知数据
        
        Args:
            shared_memory: 共享内存字典
        """
        # 更新视觉数据
        self.fusion_data.vision_target_pos = shared_memory.get('vision_target_pos')
        self.fusion_data.vision_motion_pixels = shared_memory.get('motion_pixels', 0)
        self.fusion_data.vision_motion_magnitude = shared_memory.get('motion_magnitude', 0.0)
        self.fusion_data.vision_last_motion_time = shared_memory.get('last_motion_time', 0.0)
        
        # 更新听觉数据 (未来扩展)
        # self.fusion_data.audio_direction = shared_memory.get('audio_direction')
        # self.fusion_data.audio_distance = shared_memory.get('audio_distance')
        
        # 更新雷达数据 (未来扩展)
        # self.fusion_data.radar_obstacles = shared_memory.get('radar_obstacles', [])
        
        logger.debug(
            f"📝 [FusionController] 更新共享内存："
            f"vision_pos={self.fusion_data.vision_target_pos}, "
            f"pixels={self.fusion_data.vision_motion_pixels}"
        )
    
    def process_macro_command(self, macro_cmd: MacroCommand):
        """
        处理 L2 宏观指令
        
        TSD v1.8 对应:
        - 3.2.1 宏微融合控制
        
        Args:
            macro_cmd: 宏观指令
        """
        logger.info(
            f"📥 [FusionController] 接收宏观指令："
            f"intent={macro_cmd.intent}, targets={macro_cmd.targets}"
        )
        
        self.current_macro_command = macro_cmd
        
        # 根据意图生成微观动作序列
        if macro_cmd.intent == "GRASP":
            self._generate_grasp_actions(macro_cmd)
        elif macro_cmd.intent == "NAVIGATE":
            self._generate_navigate_actions(macro_cmd)
        elif macro_cmd.intent == "FOLLOW":
            self._generate_follow_actions(macro_cmd)
        elif macro_cmd.intent == "STOP":
            self._generate_stop_actions()
        else:
            logger.warning(f"⚠️ [FusionController] 未知意图：{macro_cmd.intent}")
    
    def _generate_grasp_actions(self, macro_cmd: MacroCommand):
        """
        生成抓取动作序列
        
        Args:
            macro_cmd: 抓取宏观指令
        """
        logger.info("🤖 [FusionController] 生成抓取动作序列")
        
        # 检查目标位置
        target_pos = self.fusion_data.get_target_position()
        if target_pos is None:
            logger.error("❌ [FusionController] 抓取失败：无有效目标位置")
            return
        
        # 生成动作序列
        actions = [
            MicroAction(
                action_id=f"grasp_approach_{int(time.time())}",
                action_type="MOVE",
                parameters={
                    "target": target_pos,
                    "speed": macro_cmd.constraints.get("speed", 0.3),
                    "mode": "approach"
                },
                priority=7
            ),
            MicroAction(
                action_id=f"grasp_close_{int(time.time())}",
                action_type="GRASP",
                parameters={
                    "force": macro_cmd.constraints.get("force", 5.0),
                    "width": 0.08  # 8cm 夹爪宽度
                },
                priority=7
            )
        ]
        
        # 添加到队列
        self.micro_action_queue.extend(actions)
        logger.info(f"✅ [FusionController] 已生成 {len(actions)} 个抓取动作")
    
    def _generate_navigate_actions(self, macro_cmd: MacroCommand):
        """
        生成导航动作序列
        
        Args:
            macro_cmd: 导航宏观指令
        """
        logger.info("🤖 [FusionController] 生成导航动作序列")
        
        # 检查目标位置
        target_pos = self.fusion_data.get_target_position()
        if target_pos is None:
            logger.error("❌ [FusionController] 导航失败：无有效目标位置")
            return
        
        # 检查障碍物
        if self._check_obstacles():
            logger.warning("⚠️ [FusionController] 检测到障碍物，需要避障")
            self._generate_avoidance_actions()
            return
        
        # 生成导航动作
        action = MicroAction(
            action_id=f"navigate_{int(time.time())}",
            action_type="MOVE",
            parameters={
                "target": target_pos,
                "speed": macro_cmd.constraints.get("speed", 0.5),
                "mode": "navigate"
            },
            priority=6
        )
        
        self.micro_action_queue.append(action)
        logger.info(f"✅ [FusionController] 已生成导航动作")
    
    def _generate_follow_actions(self, macro_cmd: MacroCommand):
        """
        生成跟随动作序列
        
        Args:
            macro_cmd: 跟随宏观指令
        """
        logger.info("🤖 [FusionController] 生成跟随动作序列")
        
        # 检查目标位置
        target_pos = self.fusion_data.get_target_position()
        if target_pos is None:
            logger.error("❌ [FusionController] 跟随失败：无有效目标位置")
            return
        
        # 生成跟随动作
        action = MicroAction(
            action_id=f"follow_{int(time.time())}",
            action_type="MOVE",
            parameters={
                "target": target_pos,
                "speed": macro_cmd.constraints.get("speed", 0.3),
                "mode": "follow",
                "distance": 1.0  # 保持 1 米距离
            },
            priority=6
        )
        
        self.micro_action_queue.append(action)
        logger.info(f"✅ [FusionController] 已生成跟随动作")
    
    def _generate_stop_actions(self):
        """生成停止动作"""
        logger.warning("🛑 [FusionController] 紧急停止")
        
        action = MicroAction(
            action_id=f"stop_{int(time.time())}",
            action_type="STOP",
            parameters={},
            priority=10  # 最高优先级
        )
        
        # 清空队列，立即执行停止
        self.micro_action_queue.clear()
        self.micro_action_queue.append(action)
        self.safety_constraints['emergency_stop'] = True
    
    def _generate_avoidance_actions(self):
        """生成避障动作"""
        logger.info("🤖 [FusionController] 生成避障动作")
        
        action = MicroAction(
            action_id=f"avoid_{int(time.time())}",
            action_type="MOVE",
            parameters={
                "mode": "avoid",
                "direction": "left",  # 向左避障
                "distance": 0.5  # 移动 0.5 米
            },
            priority=8
        )
        
        self.micro_action_queue.append(action)
    
    def _check_obstacles(self) -> bool:
        """
        检查是否有障碍物
        
        Returns:
            bool: 是否有障碍物
        """
        # 未来扩展：使用雷达数据
        # if self.fusion_data.radar_distance_front is not None:
        #     return self.fusion_data.radar_distance_front < self.safety_constraints['min_obstacle_distance']
        
        # 临时实现：始终返回 False
        return False
    
    def execute_cycle(self):
        """
        执行周期：处理微观动作队列
        
        TSD v1.8 对应:
        - 4.3.1 实时控制循环
        """
        if not self.is_running:
            return
        
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        # 限制执行频率 (100Hz)
        if dt < 0.01:
            return
        
        self.last_update_time = current_time
        
        # 检查紧急停止
        if self.safety_constraints['emergency_stop']:
            logger.warning("🛑 [FusionController] 紧急停止状态")
            return
        
        # 执行队列中的动作
        if self.micro_action_queue:
            action = self.micro_action_queue[0]
            
            # 检查动作优先级
            if action.priority < 5:
                # 低优先级动作，检查是否有更高优先级事件
                pass
            
            # 执行动作
            self._execute_micro_action(action)
            
            # 移除已执行的动作
            self.micro_action_queue.pop(0)
    
    def _execute_micro_action(self, action: MicroAction):
        """
        执行单个微观动作
        
        Args:
            action: 微观动作
        """
        logger.info(
            f"⚙️ [FusionController] 执行动作："
            f"{action.action_type}, id={action.action_id}"
        )
        
        # 根据动作类型执行
        if action.action_type == "MOVE":
            self._execute_move(action)
        elif action.action_type == "GRASP":
            self._execute_grasp(action)
        elif action.action_type == "ROTATE":
            self._execute_rotate(action)
        elif action.action_type == "STOP":
            self._execute_stop(action)
    
    def _execute_move(self, action: MicroAction):
        """
        执行移动动作
        
        Args:
            action: 移动作
        """
        target = action.parameters.get("target")
        speed = action.parameters.get("speed", 0.3)
        mode = action.parameters.get("mode", "normal")
        
        logger.info(
            f"🚗 [FusionController] 移动：target={target}, speed={speed}m/s, mode={mode}"
        )
        
        # 🎯 关键：发布到底层执行器
        # 未来需要转换为实际的电机控制指令
        event = ZulongEvent(
            type=EventType.ACTION_MOTOR_CONTROL,
            priority=EventPriority.HIGH,
            source="fusion_controller",
            payload={
                "action": "move",
                "target": target,
                "speed": speed,
                "mode": mode
            },
            timestamp=time.time()
        )
        
        self.event_bus.publish(event)
        logger.debug(f"📢 [FusionController] 已发布移动指令到事件总线")
    
    def _execute_grasp(self, action: MicroAction):
        """
        执行抓取动作
        
        Args:
            action: 抓取作
        """
        force = action.parameters.get("force", 5.0)
        width = action.parameters.get("width", 0.08)
        
        logger.info(f"🤏 [FusionController] 抓取：force={force}N, width={width}m")
        
        # 发布到舵机控制器
        event = ZulongEvent(
            type=EventType.ACTION_GRIPPER_CONTROL,
            priority=EventPriority.HIGH,
            source="fusion_controller",
            payload={
                "action": "grasp",
                "force": force,
                "width": width
            },
            timestamp=time.time()
        )
        
        self.event_bus.publish(event)
        logger.debug(f"📢 [FusionController] 已发布抓取指令到事件总线")
    
    def _execute_rotate(self, action: MicroAction):
        """
        执行旋转动作
        
        Args:
            action: 旋转作
        """
        angle = action.parameters.get("angle", 0.0)
        speed = action.parameters.get("speed", 0.5)
        
        logger.info(f"🔄 [FusionController] 旋转：angle={angle}°, speed={speed}°/s")
        
        event = ZulongEvent(
            type=EventType.ACTION_MOTOR_CONTROL,
            priority=EventPriority.NORMAL,
            source="fusion_controller",
            payload={
                "action": "rotate",
                "angle": angle,
                "speed": speed
            },
            timestamp=time.time()
        )
        
        self.event_bus.publish(event)
    
    def _execute_stop(self, action: MicroAction):
        """
        执行停止动作
        
        Args:
            action: 停止作
        """
        logger.warning("🛑 [FusionController] 执行停止")
        
        event = ZulongEvent(
            type=EventType.ACTION_STOP,
            priority=EventPriority.CRITICAL,
            source="fusion_controller",
            payload={"reason": "emergency_stop"},
            timestamp=time.time()
        )
        
        self.event_bus.publish(event)
    
    def reset(self):
        """重置融合控制器"""
        logger.info("🔄 [FusionController] 重置")
        
        self.current_macro_command = None
        self.micro_action_queue.clear()
        self.safety_constraints['emergency_stop'] = False
        self.fusion_data = SensorFusionData()


# 全局单例
_fusion_controller: Optional[FusionController] = None


def get_fusion_controller() -> FusionController:
    """获取融合控制器单例"""
    global _fusion_controller
    if _fusion_controller is None:
        _fusion_controller = FusionController()
    return _fusion_controller
