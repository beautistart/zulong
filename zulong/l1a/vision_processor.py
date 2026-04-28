# File: zulong/l1a/vision_processor.py
# L1-A 视觉处理器 - 主动推送模式 (TSD v1.8 规范)
# 三层注意力机制增强版 (TSD v1.8)

import asyncio
import time
import numpy as np
import threading
from typing import Optional, List, Dict, Any
from datetime import datetime
from collections import deque
import logging

from zulong.l0.motion_detector import OpticalFlowMotionDetector, MotionState, MotionResult
from zulong.core.types import ZulongEvent, EventType, EventPriority
from zulong.core.event_bus import event_bus
from zulong.core.attention_atoms import AttentionEvent, AttentionLayer, SensorFusionData
from zulong.l1a.reflex.vision_node import VisionNode

logger = logging.getLogger("VisionProcessor")


class VisionShortTermMemory:
    """
    视觉短期记忆 (Vision Short-Term Memory, VSTM)
    
    功能 (TSD v1.7):
    - 维护滚动缓冲区 (默认 5 秒，30 FPS = 150 帧)
    - 提供最近 N 帧的访问接口
    - 保存最新帧到共享目录 (供 L2 推理使用)
    """
    
    def __init__(self, duration_sec: float = 5.0, fps: int = 30):
        self.max_frames = int(duration_sec * fps)
        self.buffer = deque(maxlen=self.max_frames)
        self.timestamps = deque(maxlen=self.max_frames)
        self.lock = threading.Lock()
        logger.info(f"🧠 [VSTM] 初始化：保留{duration_sec}秒，最大{self.max_frames}帧")

    def add_frame(self, frame: np.ndarray, timestamp: float):
        """添加帧到缓冲区"""
        with self.lock:
            self.buffer.append(frame.copy())
            self.timestamps.append(timestamp)

    def get_recent_clip(self) -> List[np.ndarray]:
        """获取所有缓冲帧"""
        with self.lock:
            return list(self.buffer)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """获取最新帧"""
        with self.lock:
            if not self.buffer:
                return None
            return self.buffer[-1]

    def get_frames_count(self) -> int:
        """获取缓冲区帧数"""
        with self.lock:
            return len(self.buffer)


class VisionProcessor:
    """
    视觉处理器 (主动推送模式，TSD v1.8 规范)
    
    核心特性:
    - 独立检测器模式：不依赖外部事件触发检测
    - 主动推送：CameraDevice 每帧主动调用 feed_frame
    - 状态机管理：IDLE → MOVING → STOPPED
    - 事件发布：仅在状态翻转时发布事件 (静默注意)
    - 零耦合：与事件总线解耦，仅通过 feed_frame 获取帧
    
    TSD v1.8 三层注意力机制增强:
    - L1_SILENT: 持续运行光流检测，但默认不生成事件
    - 状态翻转时触发 INTERACTION_TRIGGER 事件
    - 紧急运动 (大量像素) 触发 EMERGENCY_ALERT 事件
    - 持续写入共享内存 (vision_target_pos)，即使没有事件
    """
    
    def __init__(self):
        self.event_bus = event_bus
        
        # 🎯 光流法运动检测器 (TSD v1.8: 独立检测器模式)
        self.motion_detector = OpticalFlowMotionDetector(
            frame_size=(640, 480),
            base_threshold=2.5,
            pixel_threshold=8000,
            adaptive_rate=0.15,
            baseline_timeout=5.0,
            motion_stop_timeout=2.0
        )
        
        # 🧠 视觉短期记忆
        self.vstm = VisionShortTermMemory(duration_sec=5.0, fps=30)
        
        # VisionNode (用于保存视频和访问共享内存)
        self.vision_node: Optional[VisionNode] = None
        self.is_initialized = False
        
        # 运行状态
        self.is_running = False
        self.current_motion_state = MotionState.IDLE
        self.last_reported_state = MotionState.IDLE
        
        # 任务线程
        self.process_thread: Optional[threading.Thread] = None
        
        # 基准帧标志
        self.baseline_set = False
        
        # 🎯 三层注意力机制增强
        # 共享内存数据 (持续写入，即使没有事件)
        self.shared_memory: Dict[str, Any] = {
            'vision_target_pos': None,  # 视觉目标位置 [x, y, z]
            'motion_pixels': 0,  # 运动像素数
            'motion_magnitude': 0.0,  # 运动幅值
            'last_motion_time': 0.0,  # 最后运动时间
        }
        
        # 状态机记忆 (用于静默注意)
        self.state_memory = {
            'face_detected': False,  # 未来扩展：人脸检测
            'gesture_active': False,  # 未来扩展：手势识别
            'last_face_pos': None,  # 最后人脸位置
        }
        
        # 触发阈值 (TSD v1.8: 静默注意)
        self.thresholds = {
            'face_conf': 0.8,  # 未来扩展
            'gesture_conf': 0.9,  # 未来扩展
            'emergency_pixels': 15000,  # 紧急运动像素阈值
            'idle_time_sec': 2.0,  # 防抖动
        }
        
        logger.info("👁️ [VisionProcessor] 初始化完成 (三层注意力机制增强版)")
        logger.info(f"   - 静默注意：已启用")
        logger.info(f"   - 共享内存：已启用")
        logger.info(f"   - 紧急阈值：{self.thresholds['emergency_pixels']} 像素")

    async def initialize(self):
        """初始化 VisionProcessor (异步)"""
        if self.is_initialized:
            logger.info("✅ VisionProcessor already initialized")
            return
        
        try:
            logger.info("📥 Initializing VisionNode...")
            self.vision_node = VisionNode()
            logger.info("📥 Calling VisionNode.initialize()...")
            success = await self.vision_node.initialize()
            logger.info(f"📥 VisionNode.initialize() returned: {success}")
            
            if success:
                self.is_initialized = True
                logger.info("✅ VisionNode initialized")
                logger.info(f"📥 is_initialized set to: {self.is_initialized}")
                
                # 🎯 关键修复：启动 VSTM 后台保存线程
                logger.info("📥 Starting VSTM background persist thread...")
                self.vstm.start_background_persist()
                logger.info("✅ VSTM background persist thread started")
                
                # 启动后台处理线程
                logger.info("📥 Calling VisionProcessor.start()...")
                self.start()
                logger.info(f"✅ VisionProcessor.start() called. is_running={self.is_running}")
            else:
                logger.error("❌ VisionNode initialization failed")
                
        except Exception as e:
            logger.error(f"❌ VisionProcessor initialization error: {e}", exc_info=True)
            import traceback
            traceback.print_exc()

    def start(self):
        """启动后台处理线程"""
        if self.is_running:
            logger.warning("⚠️ [VisionProcessor] 已经在运行中")
            return
            
        self.is_running = True
        self.process_thread = threading.Thread(
            target=self._processing_loop, 
            daemon=True,
            name="VisionProcessorThread"
        )
        self.process_thread.start()
        logger.info("👁️ [VisionProcessor] 后台处理线程已启动")

    def stop(self):
        """停止后台处理线程"""
        self.is_running = False
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
            logger.info("🛑 [VisionProcessor] 后台处理线程已停止")

    def feed_frame(self, frame: np.ndarray, timestamp: float):
        """
        外部调用：CameraDevice 每捕获一帧就调用此方法 (TSD v1.8: 主动推送模式)
        
        这是唯一的输入接口，完全解耦事件系统
        
        处理流程 (TSD v1.8 三层注意力机制):
        1. 存入短期记忆
        2. 送入光流检测器
        3. 更新全局状态
        4. 写入共享内存 (持续写入，即使没有事件)
        5. 状态翻转检测 (静默注意核心)
        6. 生成注意力事件 (仅状态变化或紧急事件)
        
        Args:
            frame: BGR 帧
            timestamp: 时间戳
            
        TSD v1.8 对应:
        - 4.1.2 静默注意：持续推理，仅在触发阈值时生成事件
        - 4.1.3 共享内存：持续写入传感器数据
        """
        if not self.is_running or frame is None:
            return
        
        # 🎯 关键调试：记录每次 feed_frame 调用
        logger.debug(f"📥 [VisionProcessor] feed_frame 调用：state={self.current_motion_state.value}")

        # 1. 存入短期记忆
        self.vstm.add_frame(frame, timestamp)

        # 2. 送入光流检测器
        result: MotionResult = self.motion_detector.process_frame(frame)
        
        # 3. 更新全局状态
        self.current_motion_state = result.state
        
        # 🎯 调试：每帧都记录状态
        logger.debug(
            f"📊 [VisionProcessor] 帧处理：state={result.state.value}, "
            f"pixels={result.motion_pixels}, magnitude={result.flow_magnitude:.2f}"
        )

        # 4. 🎯 写入共享内存 (TSD v1.8: 持续写入，即使没有事件)
        self._write_to_shared_memory(result, timestamp)
        
        # 5. 状态翻转检测 (静默注意核心)
        # 只有状态变化，或者运动像素极大时，才发布事件
        if result.state != self.last_reported_state:
            logger.info(
                f"🔄 [VisionProcessor] 状态变化检测：{self.last_reported_state.value} -> {result.state.value}"
            )
            # 生成注意力事件并路由
            attention_event = self._create_attention_event(result, timestamp)
            self._route_attention_event(attention_event)
            
            self.last_reported_state = result.state
            
            # 如果刚进入 MOVING 状态，强制记录一个高优先级事件，触发 L2 关注
            if result.state == MotionState.MOVING:
                logger.info(
                    f"🔥 [VisionProcessor] 显著运动检测："
                    f"{result.motion_pixels} 像素，幅值={result.flow_magnitude:.2f}"
                )
                
                # 🎯 关键修复：触发运动保存标志（保存最新帧和视频片段）
                self.vstm.trigger_motion_save()
                logger.info(f"📸 [事件驱动] 已触发运动保存标志 (状态变化)")
                
            # 如果进入 IDLE 状态，清除共享内存中的位置信息
            elif result.state == MotionState.IDLE:
                self.shared_memory['vision_target_pos'] = None
                logger.debug(f"🧹 [VisionProcessor] 清除共享内存位置信息")
        
        # 🎯 关键修复：每次检测到显著运动都触发保存（即使状态未变化）
        # 这样可以保证持续挥手时也能保存视频
        elif result.state == MotionState.MOVING and result.motion_pixels > 8000:
            # 触发运动保存标志
            self.vstm.trigger_motion_save()
            logger.debug(f"📸 [事件驱动] 已触发运动保存标志 (持续运动：{result.motion_pixels}像素)")
        
        # 6. 紧急事件检测 (TSD v1.8: EMERGENCY_ALERT)
        if result.motion_pixels > self.thresholds['emergency_pixels']:
            logger.warning(
                f"🚨 [VisionProcessor] 紧急运动检测！"
                f"{result.motion_pixels} 像素 > {self.thresholds['emergency_pixels']} 阈值"
            )
            emergency_event = self._create_emergency_event(result, timestamp)
            self._route_attention_event(emergency_event)
        
        # 7. 定期心跳 (可选)：即使静止，每 5 秒也报告一次"一切正常"，防止系统假死
        # 此处省略，依靠状态翻转即可

    def _write_to_shared_memory(self, result: MotionResult, timestamp: float):
        """
        写入共享内存 (TSD v1.8: 持续写入，即使没有事件)
        
        TSD v1.8 对应:
        - 4.1.3 共享内存机制
        - 4.3.1 多模态数据流
        
        Args:
            result: 运动检测结果 (包含 motion_center)
            timestamp: 时间戳
        """
        # 更新共享内存数据
        self.shared_memory['motion_pixels'] = result.motion_pixels
        self.shared_memory['motion_magnitude'] = result.flow_magnitude
        self.shared_memory['last_motion_time'] = timestamp
        
        # 🎯 TSD v1.8: 使用光流检测器计算的真实运动中心坐标
        if result.motion_center is not None:
            # motion_center 是 (x, y) 归一化坐标 (0-1)
            # 扩展为 [x, y, z] 格式，z 暂时假设为 1 米
            self.shared_memory['vision_target_pos'] = [
                result.motion_center[0],  # x: 运动中心 X 坐标 (归一化)
                result.motion_center[1],  # y: 运动中心 Y 坐标 (归一化)
                1.0                        # z: 深度 (假设 1 米，未来使用深度相机或三角测量)
            ]
            logger.debug(
                f"📝 [VisionProcessor] 写入共享内存：center={result.motion_center}, "
                f"pixels={result.motion_pixels}"
            )
        elif result.motion_pixels > 0:
            # 如果有运动像素但没有计算出中心，使用默认中心
            self.shared_memory['vision_target_pos'] = [
                0.5,  # x: 图像中心
                0.5,  # y: 图像中心
                1.0   # z: 深度
            ]
            logger.debug(
                f"📝 [VisionProcessor] 写入共享内存 (默认中心)：pixels={result.motion_pixels}"
            )
        else:
            # 无运动，清除位置信息
            self.shared_memory['vision_target_pos'] = None

    def _create_attention_event(self, result: MotionResult, timestamp: float) -> AttentionEvent:
        """
        创建注意力事件 (TSD v1.8: 状态翻转时触发)
        
        TSD v1.8 对应:
        - 3.2.1 事件数据结构
        - 4.1.2 静默注意触发条件
        
        Args:
            result: 运动检测结果
            timestamp: 时间戳
            
        Returns:
            AttentionEvent: 注意力事件
        """
        # 根据状态确定事件类型和优先级
        if result.state == MotionState.MOVING:
            event_type = "interaction_trigger"
            priority = 5  # 普通交互优先级
            action = "motion_detected"
        elif result.state == MotionState.STOPPED:
            event_type = "interaction_trigger"
            priority = 3  # 低优先级 (运动停止)
            action = "motion_stopped"
        else:  # IDLE
            event_type = "silent_obs"
            priority = 1  # 最低优先级 (静默观察)
            action = "no_motion"
        
        return AttentionEvent(
            id=f"vis_{int(timestamp * 1000)}",
            source="l1a_vision_processor",
            type=self._get_event_type_enum(event_type),
            priority=priority,
            payload={
                "action": action,
                "state": result.state.value,
                "motion_pixels": result.motion_pixels,
                "flow_magnitude": result.flow_magnitude,
                "vision_target_pos": self.shared_memory['vision_target_pos']
            },
            timestamp=timestamp
        )

    def _create_emergency_event(self, result: MotionResult, timestamp: float) -> AttentionEvent:
        """
        创建紧急事件 (TSD v1.8: EMERGENCY_ALERT)
        
        TSD v1.8 对应:
        - 3.2.1 事件数据结构
        - 4.1.2 紧急事件检测
        
        Args:
            result: 运动检测结果
            timestamp: 时间戳
            
        Returns:
            AttentionEvent: 紧急注意力事件
        """
        return AttentionEvent(
            id=f"vis_emergency_{int(timestamp * 1000)}",
            source="l1a_vision_processor",
            type=EventType.EMERGENCY_ALERT,
            priority=10,  # 最高优先级
            payload={
                "action": "emergency_motion",
                "state": result.state.value,
                "motion_pixels": result.motion_pixels,
                "flow_magnitude": result.flow_magnitude,
                "threshold": self.thresholds['emergency_pixels'],
                "vision_target_pos": self.shared_memory['vision_target_pos']
            },
            timestamp=timestamp
        )

    def _get_event_type_enum(self, event_type_str: str) -> EventType:
        """将字符串转换为 EventType 枚举"""
        type_map = {
            "interaction_trigger": EventType.INTERACTION_TRIGGER,
            "silent_obs": EventType.SILENT_OBSERVATION,
            "emergency_alert": EventType.EMERGENCY_ALERT
        }
        return type_map.get(event_type_str, EventType.SILENT_OBSERVATION)

    def _route_attention_event(self, event: AttentionEvent):
        """
        路由注意力事件到 L1-B 注意力控制器
        
        TSD v1.8 对应:
        - 3.2.2 事件路由机制
        - 4.2.1 L1-B 注意力控制器
        
        Args:
            event: 注意力事件
        """
        logger.info(
            f"📢 [VisionProcessor] 路由注意力事件：source={event.source}, "
            f"type={event.type.value}, priority={event.priority}"
        )
        
        # 🎯 关键：将注意力事件转换为 ZulongEvent 以便与现有事件总线兼容
        # 未来可以直接使用 AttentionEvent
        zulong_event = ZulongEvent(
            type=EventType.SENSOR_VISION_STATE,
            priority=(
                EventPriority.CRITICAL 
                if event.priority >= 8 
                else EventPriority.HIGH if event.priority >= 5 else EventPriority.NORMAL
            ),
            source=event.source,
            payload={
                **event.payload,
                "attention_event_id": event.id,
                "attention_event_type": event.type.value,
                "attention_priority": event.priority
            },
            timestamp=event.timestamp
        )
        
        self.event_bus.publish(zulong_event)
        logger.debug(f"📢 [VisionProcessor] 已发布事件到事件总线")

    def _publish_state_event(self, result: MotionResult):
        """发布视觉状态变更事件 (TSD v1.7: SENSOR_VISION_STATE) - 已废弃，使用 _route_attention_event"""
        logger.warning("⚠️ [VisionProcessor] _publish_state_event 已废弃，请使用 _route_attention_event")
        # 保留此方法以兼容旧代码，但不再使用
        pass

    def _processing_loop(self):
        """
        备用轮询循环 (如果 CameraDevice 没有主动调用 feed_frame)
        正常情况下，主要逻辑在 feed_frame 中执行，此循环仅用于监控
        """
        while self.is_running:
            time.sleep(1.0)
            # 可以在这里添加看门狗逻辑，比如检测帧率是否过低

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """获取最新帧"""
        return self.vstm.get_latest_frame()

    def get_recent_frames(self, count: int = 5) -> List[np.ndarray]:
        """获取最近 N 帧"""
        all_frames = self.vstm.get_recent_clip()
        if len(all_frames) <= count:
            return all_frames
        return all_frames[-count:]

    def get_motion_state(self) -> MotionState:
        """获取当前运动状态"""
        return self.current_motion_state

    def is_motion_detected(self) -> bool:
        """是否检测到运动"""
        return self.current_motion_state == MotionState.MOVING


# 全局单例
vision_processor: Optional[VisionProcessor] = None


def init_vision_processor() -> VisionProcessor:
    """初始化全局 VisionProcessor 单例"""
    global vision_processor
    if vision_processor is None:
        vision_processor = VisionProcessor()
    return vision_processor


def get_vision_processor() -> Optional[VisionProcessor]:
    """获取全局 VisionProcessor 单例"""
    return vision_processor
