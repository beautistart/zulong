# File: zulong/l1a/vision_processor_adapter.py
# 视觉处理器适配器 (TSD v2.5 共享池架构)
# 功能：桥接 VisionProcessor 与 SharedMemoryPool

import asyncio
import time
import numpy as np
from typing import Optional, Dict, Any
import logging

from zulong.infrastructure.shared_memory_pool import shared_memory_pool, ZoneType, DataType
from zulong.infrastructure.data_ingestion import data_ingestion
from zulong.l1a.vision_processor import VisionProcessor
from zulong.core.types import ZulongEvent, EventType, EventPriority
from zulong.core.event_bus import event_bus

logger = logging.getLogger(__name__)


class VisionProcessorAdapter:
    """
    视觉处理器适配器 (TSD v2.5)
    
    功能:
    - 监听 DataIngestion 的视频入池事件
    - 从共享池读取原始视频帧
    - 调用 VisionProcessor 进行处理
    - 将处理结果写回共享池 (Feature Zone)
    
    数据流向:
    CameraDevice → DataIngestion → Raw Zone
    → (事件触发) → VisionProcessorAdapter
    → VisionProcessor.process_frame()
    → Feature Zone (motion_detection, vision_target_pos)
    → L1-B 读取
    """
    
    def __init__(self):
        self.vision_processor = VisionProcessor()
        self.event_bus = event_bus
        self.pool = shared_memory_pool
        
        # 订阅视频入池事件
        self._setup_event_handlers()
        
        logger.info("✅ [VisionProcessorAdapter] 初始化完成")
    
    def _setup_event_handlers(self):
        """设置事件监听"""
        # 监听 SENSOR_VISION 事件 (由 DataIngestion 发布)
        self.event_bus.subscribe(
            EventType.SENSOR_VISION,
            self._on_video_ingested,
            subscriber="VisionProcessorAdapter"
        )
        logger.debug("📡 [VisionProcessorAdapter] 已订阅 SENSOR_VISION 事件")
    
    async def _on_video_ingested(self, event: ZulongEvent):
        """
        视频入池事件处理
        
        Args:
            event: SENSOR_VISION 事件
        """
        try:
            trace_id = event.payload.get("trace_id")
            if not trace_id:
                logger.warning(f"⚠️ [VisionProcessorAdapter] 事件缺少 trace_id: {event.payload}")
                return
            
            # 从 Raw Zone 读取原始帧
            envelope = self.pool.read_raw(trace_id)
            if not envelope:
                logger.warning(f"⚠️ [VisionProcessorAdapter] 未找到原始帧：{trace_id[:15]}")
                return
            
            frame = envelope.payload
            timestamp = envelope.timestamp
            
            logger.debug(f"📖 [VisionProcessorAdapter] 读取原始帧：{trace_id[:15]} ({frame.shape})")
            
            # 调用 VisionProcessor 进行处理
            if self.vision_processor.is_initialized:
                # 同步调用 feed_frame
                self.vision_processor.feed_frame(frame, timestamp)
                
                # 获取处理结果 (从 VisionProcessor 的共享内存)
                motion_result = {
                    "vision_target_pos": self.vision_processor.shared_memory.get('vision_target_pos'),
                    "motion_pixels": self.vision_processor.shared_memory.get('motion_pixels'),
                    "motion_magnitude": self.vision_processor.shared_memory.get('motion_magnitude'),
                    "current_state": self.vision_processor.current_motion_state.value
                }
                
                # 将处理结果写回 Feature Zone
                feature_trace_id = f"feature_{trace_id[6:]}"  # trace_xxx → feature_xxx
                self.pool.write_feature(
                    key=feature_trace_id,
                    data=motion_result,
                    data_type=DataType.VIDEO_FRAME,
                    parent_trace_id=trace_id
                )
                
                logger.debug(f"💾 [VisionProcessorAdapter] 处理结果写入 Feature Zone: {feature_trace_id[:15]}")
                
            else:
                logger.warning(f"⚠️ [VisionProcessorAdapter] VisionProcessor 未初始化")
        
        except Exception as e:
            logger.error(f"❌ [VisionProcessorAdapter] 处理视频帧失败：{e}", exc_info=True)
    
    async def initialize(self):
        """初始化 VisionProcessor"""
        await self.vision_processor.initialize()
        logger.info("✅ [VisionProcessorAdapter] VisionProcessor 已初始化")
    
    def shutdown(self):
        """关闭适配器"""
        self.vision_processor.stop()
        logger.info("🛑 [VisionProcessorAdapter] 已关闭")


# 全局单例
vision_processor_adapter = VisionProcessorAdapter()
