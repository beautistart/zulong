# File: zulong/l1b/optimized_scheduler.py
"""
L1-B 优化调度器 (整合三层视觉优化)

TSD v1.7 对应:
- 4.2.1 L1-B 注意力控制器
- 4.4 感知预处理
- 5.2 显存约束

优化方案核心:
1. 集成 OptimizedVisionProcessor 事件流
2. 意图分数阈值判断 (挥手/注视/靠近)
3. 鹰眼模式触发条件 (置信度 >0.8)
4. 与 L2 层的协同 (任务冻结/重评估)

架构优势:
- 统一视觉事件路由
- 智能意图判断
- 支持 3 米外手势识别
"""

import logging
from typing import Optional, Dict, Any, List
from collections import deque
import time

from zulong.core.event_bus import event_bus
from zulong.core.types import EventType, EventPriority, ZulongEvent, PowerState, L2Status
from zulong.core.state_manager import state_manager
from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor

logger = logging.getLogger("OptimizedScheduler")


class OptimizedScheduler:
    """
    L1-B 优化调度器 (整合三层视觉优化)
    
    核心逻辑:
    1. 订阅视觉传感器事件
    2. 路由到 OptimizedVisionProcessor
    3. 根据意图分数决策 (忽略/交互/鹰眼)
    4. 与 L2 协同 (任务冻结/重评估)
    
    TSD v1.7 对应:
    - 4.2.1 L1-B 注意力控制器
    - 5.2 显存约束
    """
    
    # 意图阈值配置
    INTENT_THRESHOLDS = {
        'ignore': 0.3,       # 低于此值忽略
        'interact': 0.6,     # 高于此值交互
        'eagle_eye': 0.8,    # 高于此值触发鹰眼
    }
    
    # 冷却时间配置 (秒)
    COOLDOWNS = {
        'attention': 2.0,    # 注意力事件冷却
        'eagle_eye': 5.0,    # 鹰眼模式冷却
    }
    
    def __init__(self):
        """初始化优化调度器"""
        self._processor: Optional[OptimizedVisionProcessor] = None
        
        # 时间戳追踪
        self._last_attention_time = 0.0
        self._last_eagle_eye_time = 0.0
        
        # 上下文缓冲区 (过去 30 秒视觉流)
        self._context_buffer = deque(maxlen=100)
        
        # 注册事件处理器
        self._register_event_handlers()
        
        logger.info("✅ [OptimizedScheduler] 初始化完成")
    
    async def initialize(self):
        """异步初始化"""
        logger.info("📦 [OptimizedScheduler] 正在初始化 OptimizedVisionProcessor...")
        
        self._processor = OptimizedVisionProcessor()
        await self._processor.initialize(load_models=True)
        
        logger.info("✅ [OptimizedScheduler] OptimizedVisionProcessor 加载完成")
    
    def _register_event_handlers(self):
        """注册事件处理器"""
        # 订阅视觉传感器事件
        event_bus.subscribe(
            EventType.SENSOR_VISION,
            self.on_vision_frame,
            "OptimizedScheduler"
        )
        
        # 订阅交互触发事件
        event_bus.subscribe(
            EventType.INTERACTION_TRIGGER,
            self.on_interaction_trigger,
            "OptimizedScheduler"
        )
        
        logger.info("📡 [OptimizedScheduler] 已订阅事件：SENSOR_VISION, INTERACTION_TRIGGER")
    
    async def on_vision_frame(self, event: ZulongEvent):
        """
        处理视觉帧事件
        
        核心逻辑:
        1. 检查冷却时间
        2. 路由到 OptimizedVisionProcessor
        3. 根据意图分数决策
        
        Args:
            event: ZulongEvent (包含 frame, timestamp)
        """
        try:
            timestamp = event.payload.get('timestamp', time.time())
            
            # ========== 1. 检查冷却时间 ==========
            if timestamp - self._last_attention_time < self.COOLDOWNS['attention']:
                logger.debug(f"⏳ [Vision] 注意力冷却中 ({timestamp - self._last_attention_time:.2f}s)")
                return
            
            # ========== 2. 路由到 OptimizedVisionProcessor ==========
            if self._processor is None:
                logger.warning("⚠️ [Vision] OptimizedVisionProcessor 未初始化")
                return
            
            frame = event.payload.get('frame')
            if frame is None:
                logger.warning("⚠️ [Vision] 事件缺少 frame 数据")
                return
            
            # 处理帧 (同步模式，避免异步复杂性)
            result = await self._processor.process_frame_sync(frame, timestamp)
            
            if result is None:
                logger.debug("🔍 [Vision] 未检测到有效目标")
                return
            
            # ========== 3. 解析结果 ==========
            human_detected = result.get('human_detected', False)
            motion_detected = result.get('motion_detected', False)
            intent_score = result.get('intent_score', 0.0)
            intent_type = result.get('intent_type', 'UNKNOWN')
            gesture_result = result.get('gesture_result')
            
            logger.info(
                f"📊 [Vision] 处理结果："
                f"human={human_detected}, motion={motion_detected}, "
                f"intent={intent_type} ({intent_score:.2f})"
            )
            
            # ========== 4. 意图分数决策 ==========
            self._last_attention_time = timestamp
            
            if intent_score < self.INTENT_THRESHOLDS['ignore']:
                logger.debug(f"👻 [Vision] 意图分数过低 ({intent_score:.2f} < 0.3), 忽略")
                return
            
            # 添加到上下文缓冲区
            self._context_buffer.append({
                'timestamp': timestamp,
                'intent_type': intent_type,
                'intent_score': intent_score,
                'gesture': gesture_result,
            })
            
            # 决策分支
            if intent_score >= self.INTENT_THRESHOLDS['eagle_eye']:
                # ========== 4.1 高置信度：触发鹰眼模式 ==========
                await self._handle_high_confidence_intent(result, event)
            elif intent_score >= self.INTENT_THRESHOLDS['interact']:
                # ========== 4.2 中等置信度：触发交互 ==========
                await self._handle_interaction_intent(result, event)
            else:
                # ========== 4.3 低置信度：仅记录 ==========
                logger.info(f"👀 [Vision] 检测到注意意图 ({intent_type}), 但置信度不足 ({intent_score:.2f})")
            
        except Exception as e:
            logger.error(f"❌ [Vision] 处理失败：{e}")
            import traceback
            traceback.print_exc()
    
    async def _handle_high_confidence_intent(
        self, 
        result: Dict[str, Any], 
        event: ZulongEvent
    ):
        """
        处理高置信度意图 (触发鹰眼模式)
        
        核心逻辑:
        1. 检查鹰眼冷却时间
        2. 发布手势识别结果
        3. 路由到 L2 进行复杂交互
        
        Args:
            result: 视觉处理结果
            event: 原始事件
        """
        timestamp = event.payload.get('timestamp', time.time())
        
        # 检查鹰眼冷却
        if timestamp - self._last_eagle_eye_time < self.COOLDOWNS['eagle_eye']:
            logger.debug(f"⏳ [EagleEye] 冷却中 ({timestamp - self._last_eagle_eye_time:.2f}s)")
            return
        
        gesture_result = result.get('gesture_result')
        
        if gesture_result is None:
            logger.warning("⚠️ [EagleEye] 无手势识别结果")
            return
        
        self._last_eagle_eye_time = timestamp
        
        # 发布手势识别事件
        gesture_event = ZulongEvent(
            type=EventType.INTERACTION_TRIGGER,
            priority=EventPriority.HIGH,
            source="OptimizedScheduler",
            payload={
                'modality': 'vision_gesture',
                'gesture': gesture_result.get('gesture'),
                'confidence': gesture_result.get('confidence'),
                'bbox': gesture_result.get('bbox'),
                'context': list(self._context_buffer)[-5:],  # 过去 5 帧上下文
            }
        )
        
        event_bus.publish(gesture_event)
        
        logger.info(
            f"🦅 [EagleEye] 触发成功："
            f"gesture={gesture_result.get('gesture')}, "
            f"confidence={gesture_result.get('confidence'):.2f}"
        )
        
        # 路由到 L2
        await self._route_to_l2(gesture_event)
    
    async def _handle_interaction_intent(
        self, 
        result: Dict[str, Any], 
        event: ZulongEvent
    ):
        """
        处理中等置信度意图 (触发交互)
        
        核心逻辑:
        1. 构建多模态 Prompt
        2. 路由到 L2
        
        Args:
            result: 视觉处理结果
            event: 原始事件
        """
        intent_type = result.get('intent_type')
        intent_score = result.get('intent_score')
        
        logger.info(
            f"🤝 [Interaction] 检测到交互意图："
            f"{intent_type} ({intent_score:.2f})"
        )
        
        # 构建交互事件
        interaction_event = ZulongEvent(
            type=EventType.INTERACTION_TRIGGER,
            priority=EventPriority.MEDIUM,
            source="OptimizedScheduler",
            payload={
                'modality': 'vision_intent',
                'intent_type': intent_type,
                'intent_confidence': intent_score,
                'person_distance': result.get('person_distance', 'unknown'),
                'context': list(self._context_buffer)[-3:],  # 过去 3 帧上下文
            }
        )
        
        event_bus.publish(interaction_event)
        
        # 路由到 L2
        await self._route_to_l2(interaction_event)
    
    async def _route_to_l2(self, event: ZulongEvent):
        """
        路由事件到 L2 层
        
        核心逻辑:
        1. 检查 L2 状态
        2. 若 BUSY 则触发任务冻结
        3. 发布到 L2 事件队列
        
        Args:
            event: 待路由事件
        """
        l2_status = state_manager.get_l2_status()
        
        if l2_status == L2Status.BUSY:
            logger.warning(f"⚠️ [RouteToL2] L2 正忙 ({l2_status.name}), 触发任务冻结")
            # 发布任务冻结事件
            freeze_event = ZulongEvent(
                type=EventType.TASK_FREEZE,
                priority=EventPriority.HIGH,
                source="OptimizedScheduler",
                payload={'reason': 'vision_interrupt'}
            )
            event_bus.publish(freeze_event)
        
        # 发布到 L2
        l2_event = ZulongEvent(
            type=EventType.SYSTEM_L2_COMMAND,
            priority=event.priority,
            source="OptimizedScheduler",
            payload=event.payload
        )
        
        event_bus.publish(l2_event)
        
        logger.info(f"📮 [RouteToL2] 事件已路由到 L2: {event.type.name}")
    
    async def on_interaction_trigger(self, event: ZulongEvent):
        """
        处理交互触发事件
        
        Args:
            event: ZulongEvent
        """
        logger.debug(f"🔔 [OptimizedScheduler] 收到交互触发事件：{event.type.name}")
        # 此处可添加额外的交互后处理逻辑
    
    def get_context_buffer(self, seconds: float = 30.0) -> List[Dict[str, Any]]:
        """
        获取上下文缓冲区
        
        Args:
            seconds: 回溯时间 (秒)
        
        Returns:
            上下文帧列表
        """
        current_time = time.time()
        return [
            ctx for ctx in self._context_buffer
            if current_time - ctx['timestamp'] < seconds
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'processor_initialized': self._processor is not None,
            'context_buffer_size': len(self._context_buffer),
            'last_attention_time': self._last_attention_time,
            'last_eagle_eye_time': self._last_eagle_eye_time,
        }


# 全局单例
_optimized_scheduler: Optional[OptimizedScheduler] = None


async def get_optimized_scheduler() -> OptimizedScheduler:
    """获取或创建优化调度器单例"""
    global _optimized_scheduler
    if _optimized_scheduler is None:
        _optimized_scheduler = OptimizedScheduler()
        await _optimized_scheduler.initialize()
    return _optimized_scheduler


async def init_optimized_scheduler() -> OptimizedScheduler:
    """初始化优化调度器"""
    return await get_optimized_scheduler()
