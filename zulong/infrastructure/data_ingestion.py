# File: zulong/infrastructure/data_ingestion.py
# 统一数据入口 (TSD v2.5 核心基础设施)
# 对应文档：数据统一共享池化以及增强记忆共享

import asyncio
import time
import uuid
from typing import Optional, Dict, Any, Union, List
import logging
import numpy as np

from zulong.infrastructure.shared_memory_pool import (
    SharedMemoryPool, DataEnvelope, ZoneType, DataType
)
from zulong.core.types import ZulongEvent, EventType, EventPriority
from zulong.core.event_bus import event_bus

logger = logging.getLogger(__name__)


class DataIngestion:
    """
    统一数据入口 (TSD v2.5 核心组件)
    
    功能:
    - 所有感知数据的统一入口
    - 自动封装成 DataEnvelope
    - 写入 Raw Zone
    - 发布事件触发后续处理
    
    数据流向:
    1. 外部数据输入 → 2. 封装成 DataEnvelope → 3. 写入 Raw Zone
    → 4. 发布 "new_data" 事件 → 5. 对应插件订阅处理
    
    使用示例:
    ```python
    ingestion = DataIngestion()
    
    # 视频帧入池
    trace_id = ingestion.ingest_video(frame, timestamp)
    
    # 音频数据入池
    trace_id = ingestion.ingest_audio(audio_data, sample_rate)
    
    # 文本入池
    trace_id = ingestion.ingest_text("你好", source="user")
    ```
    """
    
    _instance = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, pool: Optional[SharedMemoryPool] = None, 
                 event_bus: Optional[Any] = None):
        """初始化数据入口（同步版本，兼容旧代码）
        
        Args:
            pool: 共享数据池实例 (默认延迟创建)
            event_bus: 事件总线实例
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        # 🔥 延迟创建 pool，避免重复实例
        self._pool = pool
        self._pool_initialized = pool is not None
        
        self.event_bus = event_bus
        if event_bus is None:
            # 延迟导入，避免循环导入
            from zulong.core.event_bus import event_bus as global_event_bus
            self.event_bus = global_event_bus
        
        # 计数器 (用于生成有序 trace_id)
        self._counters: Dict[str, int] = {
            "video": 0,
            "audio": 0,
            "text": 0,
            "file": 0
        }
        self._counter_lock = asyncio.Lock()
        
        self._initialized = True
        
        logger.info("✅ [DataIngestion] 初始化完成")
    
    @property
    def pool(self) -> SharedMemoryPool:
        """延迟初始化共享池（关键修复：确保单例）"""
        if not self._pool_initialized:
            from zulong.infrastructure.shared_memory_pool import SharedMemoryPool
            # 🔥 关键修复：使用 __new__ 而不是 __init__，防止清空数据
            self._pool = SharedMemoryPool.__new__(SharedMemoryPool)
            if not hasattr(self._pool, '_initialized'):
                self._pool.__init__()
            self._pool_initialized = True
            logger.info("[DataIngestion] 延迟创建共享池单例")
        return self._pool
    
    @pool.setter
    def pool(self, value: SharedMemoryPool):
        """设置共享池实例"""
        self._pool = value
        self._pool_initialized = True
    
    @classmethod
    async def get_instance(cls):
        """异步单例模式（推荐用法）"""
        if cls._instance is None:
            cls._instance = cls()
            # 🔥 关键：使用异步方式获取共享池单例
            cls._instance.pool = await SharedMemoryPool.get_instance()
            # 获取全局事件总线
            from zulong.core.event_bus import event_bus as global_event_bus
            cls._instance.event_bus = global_event_bus
        return cls._instance
    
    async def _generate_trace_id(self, prefix: str) -> str:
        """生成追踪 ID (有序 + UUID)"""
        async with self._counter_lock:
            self._counters[prefix] = self._counters.get(prefix, 0) + 1
            counter = self._counters[prefix]
        
        # 格式：trace_video_000001_a1b2c3d4
        unique_id = str(uuid.uuid4())[:8]
        return f"trace_{prefix}_{counter:06d}_{unique_id}"
    
    async def ingest_video(self, frame: np.ndarray, timestamp: Optional[float] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        视频帧数据入池
        
        Args:
            frame: 视频帧 (numpy 数组)
            timestamp: 时间戳 (默认当前时间)
            metadata: 附加元数据 (相机 ID、分辨率等)
        
        Returns:
            trace_id: 追踪 ID
        
        使用示例:
        ```python
        # CameraDevice 调用
        trace_id = await ingestion.ingest_video(frame, timestamp=time.time())
        
        # 发布事件后，VisionProcessor 会订阅并处理
        ```
        """
        trace_id = await self._generate_trace_id("video")
        
        envelope = DataEnvelope(
            trace_id=trace_id,
            timestamp=timestamp or time.time(),
            data_type=DataType.VIDEO_FRAME,
            zone=ZoneType.RAW,
            payload=frame,
            metadata=metadata or {}
        )
        
        # 写入 Raw Zone（异步）
        await self.pool.write(envelope)
        
        # 发布事件 (触发 VisionProcessor 处理)
        event = ZulongEvent(
            type=EventType.SENSOR_VISION,
            priority=EventPriority.NORMAL,
            payload={
                "trace_id": trace_id,
                "frame_shape": frame.shape,
                "timestamp": envelope.timestamp
            },
            source="DataIngestion"
        )
        self.event_bus.publish(event)
        
        logger.debug(f"📹 [DataIngestion] 视频帧入池：{trace_id[:15]} ({frame.shape})")
        
        return trace_id
    
    async def ingest_video_clip(self, frames: List[np.ndarray], 
                               timestamps: List[float],
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        视频片段入池 (多帧)
        
        Args:
            frames: 视频帧列表
            timestamps: 时间戳列表
            metadata: 附加元数据
        
        Returns:
            trace_id
        """
        trace_id = await self._generate_trace_id("video_clip")
        
        envelope = DataEnvelope(
            trace_id=trace_id,
            timestamp=time.time(),
            data_type=DataType.VIDEO_CLIP,
            zone=ZoneType.RAW,
            payload={
                "frames": frames,
                "timestamps": timestamps,
                "duration": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
            },
            metadata=metadata or {}
        )
        
        self.pool.write(envelope)
        
        # 发布事件
        event = ZulongEvent(
            type=EventType.SENSOR_VISION,
            priority=EventPriority.NORMAL,
            payload={
                "trace_id": trace_id,
                "frame_count": len(frames),
                "duration": envelope.payload["duration"]
            },
            source="DataIngestion"
        )
        self.event_bus.publish(event)
        
        logger.debug(f"🎬 [DataIngestion] 视频片段入池：{trace_id[:15]} ({len(frames)}帧)")
        
        return trace_id
    
    async def ingest_audio(self, audio_data: np.ndarray, sample_rate: int,
                          timestamp: Optional[float] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        音频数据入池
        
        Args:
            audio_data: 音频数据 (numpy 数组，归一化到 -1.0~1.0)
            sample_rate: 采样率 (e.g. 16000)
            timestamp: 时间戳
            metadata: 附加元数据 (麦克风 ID、增益等)
        
        Returns:
            trace_id
        
        使用示例:
        ```python
        # MicrophoneDevice 调用
        trace_id = await ingestion.ingest_audio(audio_data, sample_rate=16000)
        
        # 发布事件后，AudioPreprocessor 会订阅并处理
        ```
        """
        trace_id = await self._generate_trace_id("audio")
        
        envelope = DataEnvelope(
            trace_id=trace_id,
            timestamp=timestamp or time.time(),
            data_type=DataType.AUDIO_RAW,
            zone=ZoneType.RAW,
            payload=audio_data,
            metadata={
                "sample_rate": sample_rate,
                "duration": len(audio_data) / sample_rate,
                **(metadata or {})
            }
        )
        
        # 写入 Raw Zone（异步）
        await self.pool.write(envelope)
        
        # 发布事件 (触发 AudioPreprocessor 处理)
        event = ZulongEvent(
            type=EventType.SENSOR_AUDIO,
            priority=EventPriority.NORMAL,
            payload={
                "trace_id": trace_id,
                "sample_rate": sample_rate,
                "duration": envelope.metadata["duration"]
            },
            source="DataIngestion"
        )
        self.event_bus.publish(event)
        
        logger.debug(f"🎤 [DataIngestion] 音频入池：{trace_id[:15]} ({sample_rate}Hz, {envelope.metadata['duration']:.2f}s)")
        
        return trace_id
    
    async def ingest_text(self, text: str, source: str = "user",
                         timestamp: Optional[float] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        文本数据入池
        
        Args:
            text: 文本内容
            source: 来源 ("user", "assistant", "system")
            timestamp: 时间戳
            metadata: 附加元数据
        
        Returns:
            trace_id
        
        使用示例:
        ```python
        # 用户输入
        trace_id = await ingestion.ingest_text("你好", source="user")
        
        # AI 回复
        trace_id = await ingestion.ingest_text("你好！有什么可以帮助你的？", source="assistant")
        ```
        """
        trace_id = await self._generate_trace_id("text")
        
        # 根据来源选择数据类型
        data_type_map = {
            "user": DataType.TEXT_USER,
            "assistant": DataType.TEXT_ASSISTANT,
            "system": DataType.TEXT_SYSTEM
        }
        data_type = data_type_map.get(source, DataType.TEXT_SYSTEM)
        
        envelope = DataEnvelope(
            trace_id=trace_id,
            timestamp=timestamp or time.time(),
            data_type=data_type,
            zone=ZoneType.RAW,
            payload=text,
            metadata={
                "source": source,
                "length": len(text),
                **(metadata or {})
            }
        )
        
        # 写入 Raw Zone（异步）
        await self.pool.write(envelope)
        
        # 🔥 修复：不再自动发布事件!
        # 数据入池只是补充，用于复盘
        # 事件路由由 L1-C/D 或用户交互入口直接发布
        
        logger.debug(f"📝 [DataIngestion] 文本入池：{trace_id[:15]} (source={source}, {len(text)}字)")
        
        return trace_id
    
    async def ingest_system_state(self, key: str, state: Any,
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        系统状态入池
        
        Args:
            key: 状态键 (e.g. "power_state", "l2_status")
            state: 状态值
            metadata: 附加元数据
        
        Returns:
            trace_id
        """
        trace_id = await self._generate_trace_id("system")
        
        envelope = DataEnvelope(
            trace_id=trace_id,
            timestamp=time.time(),
            data_type=DataType.SYSTEM_STATE,
            zone=ZoneType.SYSTEM,
            payload=state,
            metadata={
                "key": key,
                **(metadata or {})
            }
        )
        
        # 写入 System Zone（异步）
        await self.pool.write(envelope)
        
        logger.debug(f"⚙️ [DataIngestion] 系统状态入池：{trace_id[:15]} (key={key})")
        
        return trace_id
    
    async def ingest_log(self, level: str, message: str, 
                        module: str = "",
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        日志入池
        
        Args:
            level: 日志级别 ("DEBUG", "INFO", "WARNING", "ERROR")
            message: 日志消息
            module: 模块名
            metadata: 附加元数据
        
        Returns:
            trace_id
        """
        trace_id = await self._generate_trace_id("log")
        
        envelope = DataEnvelope(
            trace_id=trace_id,
            timestamp=time.time(),
            data_type=DataType.SYSTEM_LOG,
            zone=ZoneType.SYSTEM,
            payload=message,
            metadata={
                "level": level,
                "module": module,
                **(metadata or {})
            }
        )
        
        # 写入 System Zone
        self.pool.write(envelope)
        
        logger.debug(f"📋 [DataIngestion] 日志入池：{trace_id[:15]} ({level} - {module})")
        
        return trace_id
    
    async def ingest_memory_node(self, turn_id: int, role: str, text: str,
                                context: Optional[Dict[str, Any]] = None,
                                timestamp: Optional[float] = None) -> str:
        """
        记忆节点入池
        
        Args:
            turn_id: 对话轮数
            role: 角色 ("user", "assistant", "system")
            text: 文本内容
            context: 上下文信息 (视觉、听觉等)
            timestamp: 时间戳
        
        Returns:
            trace_id
        """
        trace_id = await self._generate_trace_id("memory")
        
        envelope = DataEnvelope(
            trace_id=trace_id,
            timestamp=timestamp or time.time(),
            data_type=DataType.MEMORY_NODE,
            zone=ZoneType.MEMORY,
            payload={
                "turn_id": turn_id,
                "role": role,
                "text": text,
                "context": context or {}
            },
            metadata={
                "turn_id": turn_id,
                "role": role
            }
        )
        
        # 写入 Memory Zone
        self.pool.write(envelope)
        
        logger.debug(f"🧠 [DataIngestion] 记忆节点入池：{trace_id[:15]} (turn={turn_id}, role={role})")
        
        return trace_id


# 🔥 全局单例（异步友好）
# 注意：这是同步实例，pool 会在第一次使用时创建
# 新代码应该使用 await DataIngestion.get_instance()
data_ingestion = DataIngestion()
