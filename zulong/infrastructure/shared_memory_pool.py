# File: zulong/infrastructure/shared_memory_pool.py
# 全局共享数据池 (TSD v2.5 核心基础设施)
# 对应文档：数据统一共享池化以及增强记忆共享

import asyncio
import time
import uuid
import threading
import atexit  # 🔥 新增：系统退出钩子
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import json
import gzip
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class ZoneType(str, Enum):
    """数据分区类型 (TSD v2.5 分层存储)"""
    RAW = "raw"           # 原始数据区：二进制流、原始文本、未处理数据
    FEATURE = "feature"   # 特征数据区：结构化 JSON、提取的特征
    SYSTEM = "system"     # 系统数据区：日志、调试信息、状态
    MEMORY = "memory"     # 记忆数据区：对话历史、上下文快照


class DataType(str, Enum):
    """数据类型枚举"""
    VIDEO_FRAME = "video.frame"
    VIDEO_CLIP = "video.clip"
    AUDIO_RAW = "audio.raw"
    AUDIO_FEATURE = "audio.feature"
    TEXT_USER = "text.user"
    TEXT_ASSISTANT = "text.assistant"
    TEXT_SYSTEM = "text.system"
    IMAGE = "image"
    FILE = "file"
    DOCUMENT = "document"
    SYSTEM_LOG = "system.log"
    SYSTEM_STATE = "system.state"
    MEMORY_NODE = "memory.node"
    CONTEXT_PACK = "context.pack"


@dataclass
class DataEnvelope:
    """
    统一数据信封 (TSD v2.5 核心数据结构)
    
    所有进入共享池的数据都必须封装成此格式
    """
    trace_id: str                    # 全局唯一追踪 ID (UUID)
    timestamp: float                 # 纳秒级时间戳
    data_type: DataType              # 数据类型
    zone: ZoneType                   # 存储分区
    payload: Any                     # 原始数据 (支持 numpy 数组、二进制、文本等)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 附加信息
    
    # 关联追踪 (用于多模态数据关联)
    parent_trace_id: Optional[str] = None  # 父级追踪 ID (例如：视频帧来自哪个视频流)
    related_trace_ids: List[str] = field(default_factory=list)  # 关联追踪 ID 列表
    
    # 生命周期管理
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None  # 过期时间 (用于自动清理)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 (用于序列化)"""
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "data_type": self.data_type.value,
            "zone": self.zone.value,
            "payload": self._serialize_payload(),
            "metadata": self.metadata,
            "parent_trace_id": self.parent_trace_id,
            "related_trace_ids": self.related_trace_ids,
            "created_at": self.created_at,
            "expires_at": self.expires_at
        }
    
    def _serialize_payload(self) -> Any:
        """序列化 payload (处理 numpy 数组等特殊类型)"""
        if isinstance(self.payload, np.ndarray):
            return {
                "_type": "numpy.ndarray",
                "shape": self.payload.shape,
                "dtype": str(self.payload.dtype),
                "data": self.payload.tolist()
            }
        elif isinstance(self.payload, (dict, list, str, int, float, bool, type(None))):
            return self.payload
        else:
            # 其他类型尝试转为字符串
            return str(self.payload)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataEnvelope":
        """从字典还原"""
        payload = data.get("payload")
        if isinstance(payload, dict) and payload.get("_type") == "numpy.ndarray":
            payload = np.array(
                payload["data"],
                dtype=payload["dtype"],
                shape=payload["shape"]
            )
        
        return cls(
            trace_id=data["trace_id"],
            timestamp=data["timestamp"],
            data_type=DataType(data["data_type"]),
            zone=ZoneType(data["zone"]),
            payload=payload,
            metadata=data.get("metadata", {}),
            parent_trace_id=data.get("parent_trace_id"),
            related_trace_ids=data.get("related_trace_ids", []),
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at")
        )


class SharedMemoryPool:
    """
    全局共享数据池 (TSD v2.5 核心基础设施) - 纯异步版本
    
    功能:
    - 统一存储所有感知数据、系统状态、记忆
    - 分层存储：Raw Zone, Feature Zone, System Zone, Memory Zone
    - 支持多模态数据关联 (视频 - 音频 - 文本)
    - 自动过期清理
    - 异步非阻塞
    
    数据流向:
    1. 数据采集 → 2. 封装成 DataEnvelope → 3. 写入对应 Zone
    4. 发布事件 → 5. 插件订阅处理 → 6. 写回 Feature Zone
    7. L1-B 读取打包 → 8. 发送到 L2
    """
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        """🔥 关键修复：禁止直接实例化，强制使用 await get_instance()"""
        raise RuntimeError(
            "SharedMemoryPool 必须使用 'await SharedMemoryPool.get_instance()' 初始化！\n"
            "例如：pool = await SharedMemoryPool.get_instance()"
        )
    
    @classmethod
    async def get_instance(cls):
        """🔥 异步单例模式（唯一正确用法）
        
        核心特性:
        - 全局唯一实例，每次启动都加载同一个持久化数据
        - 首次调用时自动加载快照并启动后台任务
        - 后续调用直接返回已加载的实例
        
        使用示例:
        ```python
        # ✅ 正确用法
        pool = await SharedMemoryPool.get_instance()
        
        # ❌ 错误用法 (会抛出异常)
        pool = SharedMemoryPool()
        ```
        """
        logger.debug(f"[SharedMemoryPool] get_instance 被调用，当前_instance: {cls._instance}")
        
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    logger.info("🆕 [SharedMemoryPool] 首次创建全局单例实例...")
                    # 🔥 关键：创建新实例但不调用__init__
                    cls._instance = super().__new__(cls)
                    # 🔥 初始化属性（不清空数据）
                    cls._instance._init_attributes()
                    # 🔥 加载持久化数据（从磁盘快照恢复）
                    logger.info("📥 [SharedMemoryPool] 开始加载持久化快照...")
                    await cls._instance._load_snapshot()
                    logger.info("💾 [SharedMemoryPool] ✅ 已从持久化文件恢复数据")
                    # 🔥 启动后台任务
                    asyncio.create_task(cls._instance._cleanup_loop())
                    asyncio.create_task(cls._instance._snapshot_loop())
                    logger.info("🚀 [SharedMemoryPool] 后台任务已启动")
                else:
                    logger.debug("[SharedMemoryPool] 实例已存在，直接返回")
        else:
            logger.debug("[SharedMemoryPool] 实例已存在，直接返回")
        
        return cls._instance
    
    def __init__(self):
        """🔥 防止直接初始化（关键修复：保护已加载的持久化数据）
        
        注意：此方法现在仅用于向后兼容，实际初始化由 _init_attributes() 完成
        """
        # 🔥 防止重复初始化（关键修复：保护已加载的持久化数据）
        if hasattr(self, '_initialized') and self._initialized:
            logger.debug("[SharedMemoryPool] 已初始化，跳过")
            return
        
        # 🔥 警告：直接调用__init__会清空数据
        logger.warning("⚠️ [SharedMemoryPool] 警告：直接调用__init__会清空数据！请使用 await get_instance()")
        
        # 🔥 注册系统退出钩子，确保异常退出时也能保存
        atexit.register(self._sync_shutdown)
        
        # 四个分区 (内存字典实现)
        self._raw_zone: Dict[str, DataEnvelope] = {}      # 原始数据
        self._feature_zone: Dict[str, DataEnvelope] = {}  # 特征数据
        self._system_zone: Dict[str, DataEnvelope] = {}   # 系统数据
        self._memory_zone: Dict[str, DataEnvelope] = {}   # 记忆数据
        
        # 索引 (加速查询)
        self._trace_index: Dict[str, ZoneType] = {}       # trace_id → zone 映射
        self._type_index: Dict[DataType, List[str]] = {}  # data_type → trace_ids
        self._time_index: List[tuple] = []                # (timestamp, trace_id, zone)
        
        # 锁 (异步非阻塞)
        # 🔥 关键修复：使用 threading.Lock 用于跨线程同步，asyncio.Lock 用于异步操作
        import threading
        self._zone_locks = {
            ZoneType.RAW: threading.Lock(),
            ZoneType.FEATURE: threading.Lock(),
            ZoneType.SYSTEM: threading.Lock(),
            ZoneType.MEMORY: threading.Lock()
        }
        self._index_lock = threading.Lock()
        
        # 🔥 关键修改：异步延迟保存队列和锁（优化频繁 I/O）
        self._save_queue = asyncio.Queue()
        self._save_pending = False
        self._save_lock = asyncio.Lock()
        self._pending_save_task = None
        
        # 配置
        self.max_raw_size = 1000        # Raw Zone 最大条目数 (LRU 清理)
        self.max_feature_size = 5000    # Feature Zone 最大条目数
        self.max_memory_size = 100      # Memory Zone 最大条目数 (对话轮数)
        self.auto_cleanup_interval = 60  # 自动清理间隔 (秒)
        
        # ✅ 第 2 周优化：持久化配置
        self.persistence_enabled = True
        self.persistence_path = Path("./data/shared_memory_pool")
        self.persistence_path.mkdir(parents=True, exist_ok=True)
        self.snapshot_interval = 30  # 🔥 优化：从 300 秒改为 30 秒
        self.max_snapshots = 20  # 🔥 优化：从 10 个改为 20 个
        self.last_snapshot_time = 0.0
        self._write_count = 0  # 🔥 新增：写入计数器
        
        # 后台任务标志
        self._running = True
        self._background_tasks_started = False
        
        # 🔥 关键修复：在初始化完成时标记，而不是在开始
        # 这样 __init__ 只会在第一次执行，后续调用会跳过
        # 但 _load_snapshot() 在 get_instance() 中调用，在__init__之后
        # 所以数据不会被清空
        self._initialized = True
        
        logger.info("✅ [SharedMemoryPool] 初始化完成 (异步版本)")
        logger.info(f"   - Raw Zone: 最大 {self.max_raw_size} 条")
        logger.info(f"   - Feature Zone: 最大 {self.max_feature_size} 条")
        logger.info(f"   - Memory Zone: 最大 {self.max_memory_size} 轮")
        logger.info(f"   - 自动清理：{self.auto_cleanup_interval}秒/次")
        logger.info(f"   - 持久化：✅ 已启用 (路径={self.persistence_path}, 间隔={self.snapshot_interval}秒)")
        logger.info(f"   - 异步延迟保存：✅ 已启用 (减少频繁 I/O)")
        logger.info(f"   - 系统退出钩子：✅ 已注册 (atexit)")
    
    def _init_attributes(self):
        """🔥 初始化实例属性（不清空数据，用于 get_instance()）
        
        此方法在首次创建实例时调用，用于初始化基本属性
        与__init__不同，它不会清空已加载的持久化数据
        """
        # 🔥 防止重复初始化
        if hasattr(self, '_initialized') and self._initialized:
            logger.debug("[SharedMemoryPool] 属性已初始化，跳过")
            return
        
        # 四个分区 (内存字典实现)
        self._raw_zone: Dict[str, DataEnvelope] = {}      # 原始数据
        self._feature_zone: Dict[str, DataEnvelope] = {}  # 特征数据
        self._system_zone: Dict[str, DataEnvelope] = {}   # 系统数据
        self._memory_zone: Dict[str, DataEnvelope] = {}   # 记忆数据
        
        # 索引 (加速查询)
        self._trace_index: Dict[str, ZoneType] = {}       # trace_id → zone 映射
        self._type_index: Dict[DataType, List[str]] = {}  # data_type → trace_ids
        self._time_index: List[tuple] = []                # (timestamp, trace_id, zone)
        
        # 锁 (异步非阻塞)
        # 🔥 关键修复：使用 threading.Lock 用于跨线程同步
        import threading
        self._zone_locks = {
            ZoneType.RAW: threading.Lock(),
            ZoneType.FEATURE: threading.Lock(),
            ZoneType.SYSTEM: threading.Lock(),
            ZoneType.MEMORY: threading.Lock()
        }
        self._index_lock = threading.Lock()
        
        # 🔥 关键修改：异步延迟保存队列和锁（优化频繁 I/O）
        self._save_queue = asyncio.Queue()
        self._save_pending = False
        self._save_lock = asyncio.Lock()
        self._pending_save_task = None
        
        # 配置
        self.max_raw_size = 1000        # Raw Zone 最大条目数 (LRU 清理)
        self.max_feature_size = 5000    # Feature Zone 最大条目数
        self.max_memory_size = 100      # Memory Zone 最大条目数 (对话轮数)
        self.auto_cleanup_interval = 60  # 自动清理间隔 (秒)
        
        # ✅ 第 2 周优化：持久化配置
        self.persistence_enabled = True
        self.persistence_path = Path("./data/shared_memory_pool")
        self.persistence_path.mkdir(parents=True, exist_ok=True)
        self.snapshot_interval = 30  # 🔥 优化：从 300 秒改为 30 秒
        self.max_snapshots = 20  # 🔥 优化：从 10 个改为 20 个
        self.last_snapshot_time = 0.0
        self._write_count = 0  # 🔥 新增：写入计数器
        
        # 后台任务标志
        self._running = True
        self._background_tasks_started = False
        
        # 🔥 标记为已初始化
        self._initialized = True
        
        logger.info("✅ [SharedMemoryPool] 属性初始化完成")
    
    async def start_background_tasks(self):
        """启动后台异步任务（必须在事件循环中调用）"""
        if self._background_tasks_started:
            return
        
        self._background_tasks_started = True
        
        # 创建后台任务
        asyncio.create_task(self._cleanup_loop())
        asyncio.create_task(self._snapshot_loop())
        
        logger.info("🚀 [SharedMemoryPool] 后台任务已启动")
    
    def _get_zone(self, zone: ZoneType) -> Dict[str, DataEnvelope]:
        """获取对应分区字典"""
        zones = {
            ZoneType.RAW: self._raw_zone,
            ZoneType.FEATURE: self._feature_zone,
            ZoneType.SYSTEM: self._system_zone,
            ZoneType.MEMORY: self._memory_zone
        }
        return zones[zone]
    
    async def write(self, envelope: DataEnvelope) -> str:
        """
        写入数据到共享池（异步）
        
        Args:
            envelope: 数据信封
        
        Returns:
            trace_id: 追踪 ID
        
        🔥 优化：每次写入都会触发快照保存，但使用异步延迟保存避免频繁 I/O
        - 写入后立即将保存请求加入队列
        - 等待 0.5 秒合并多个保存请求
        - 实际保存操作在后台异步执行
        """
        zone = envelope.zone
        trace_id = envelope.trace_id
        
        # 获取对应分区的锁（非阻塞）
        # 🔥 关键修复：使用 threading.Lock 的 with 语句
        with self._zone_locks[zone]:
            zone_dict = self._get_zone(zone)
            zone_dict[trace_id] = envelope
            
            # 更新索引
            with self._index_lock:
                self._trace_index[trace_id] = zone
                
                # 类型索引
                if envelope.data_type not in self._type_index:
                    self._type_index[envelope.data_type] = []
                self._type_index[envelope.data_type].append(trace_id)
                
                # 时间索引
                self._time_index.append((envelope.timestamp, trace_id, zone))
        
        logger.debug(f"💾 [SharedMemoryPool] 写入 {zone.value}:{trace_id[:8]} ({envelope.data_type.value})")
        
        # 🔥 关键修改：每次写入都触发快照保存（异步延迟保存）
        # 🔥 用户输入后立即保存，但通过队列合并减少实际 I/O 次数
        await self._queue_snapshot_save()
        
        return trace_id
    
    async def read(self, trace_id: str) -> Optional[DataEnvelope]:
        """
        读取数据（异步）
        
        Args:
            trace_id: 追踪 ID
        
        Returns:
            DataEnvelope 或 None
        """
        # 从索引获取分区
        # 🔥 关键修复：使用 threading.Lock
        with self._index_lock:
            zone = self._trace_index.get(trace_id)
        
        if zone is None:
            logger.warning(f"⚠️ [SharedMemoryPool] 未找到 trace_id: {trace_id[:8]}")
            return None
        
        # 从对应分区读取
        with self._zone_locks[zone]:
            zone_dict = self._get_zone(zone)
            envelope = zone_dict.get(trace_id)
        
        if envelope:
            logger.debug(f"📖 [SharedMemoryPool] 读取 {zone.value}:{trace_id[:8]}")
        
        return envelope
    
    async def read_raw(self, trace_id: str) -> Optional[DataEnvelope]:
        """从 Raw Zone 读取（异步）"""
        return await self._read_from_zone(ZoneType.RAW, trace_id)
    
    async def read_feature(self, trace_id: str) -> Optional[DataEnvelope]:
        """从 Feature Zone 读取（异步）"""
        return await self._read_from_zone(ZoneType.FEATURE, trace_id)
    
    async def read_system(self, key: str) -> Optional[DataEnvelope]:
        """从 System Zone 读取（异步）"""
        return await self._read_from_zone(ZoneType.SYSTEM, key)
    
    async def read_memory(self, trace_id: str) -> Optional[DataEnvelope]:
        """从 Memory Zone 读取（异步，无锁优化）"""
        return await self._read_from_zone(ZoneType.MEMORY, trace_id, use_lock=False)
    
    async def _read_from_zone(self, zone: ZoneType, key: str, use_lock: bool = True) -> Optional[DataEnvelope]:
        """从指定分区读取（异步）
        
        Args:
            zone: 分区类型
            key: 键
            use_lock: 是否使用锁（读取操作可以不用锁）
        """
        if use_lock:
            async with self._zone_locks[zone]:
                zone_dict = self._get_zone(zone)
                return zone_dict.get(key)
        else:
            # 🔥 优化：读取操作不使用锁，提高并发性能
            zone_dict = self._get_zone(zone)
            return zone_dict.get(key)
    
    async def write_feature(self, key: str, data: Any, data_type: DataType = DataType.TEXT_SYSTEM,
                     parent_trace_id: Optional[str] = None) -> str:
        """
        便捷方法：写入到 Feature Zone（异步）
        
        Args:
            key: 存储键
            data: 数据
            data_type: 数据类型
            parent_trace_id: 父级追踪 ID
        
        Returns:
            trace_id
        """
        envelope = DataEnvelope(
            trace_id=key if key.startswith("trace_") else f"trace_{key}",
            timestamp=time.time(),
            data_type=data_type,
            zone=ZoneType.FEATURE,
            payload=data,
            parent_trace_id=parent_trace_id
        )
        return await self.write(envelope)
    
    async def write_text(self, key: str, data: dict, metadata: dict = None) -> str:
        """
        🔥 新增：便捷方法：写入文本数据到 Raw Zone（异步）
        
        Args:
            key: 存储键
            data: 数据字典，包含 'text' 等字段
            metadata: 元数据（可选）
        
        Returns:
            trace_id
        """
        trace_id = key if key.startswith("trace_") else f"trace_{key}"
        
        # 根据数据来源选择合适的数据类型
        source = metadata.get('source', 'unknown') if metadata else 'unknown'
        if 'user' in source.lower():
            data_type = DataType.TEXT_USER
        elif 'assistant' in source.lower() or 'system' in source.lower():
            data_type = DataType.TEXT_SYSTEM
        else:
            data_type = DataType.TEXT_USER  # 默认使用 TEXT_USER
        
        envelope = DataEnvelope(
            trace_id=trace_id,
            timestamp=time.time(),
            data_type=data_type,
            zone=ZoneType.RAW,
            payload=data,
            metadata=metadata or {}
        )
        
        return await self.write(envelope)
    
    async def get_by_type(self, data_type: DataType, limit: int = 10) -> List[DataEnvelope]:
        """
        按类型获取数据 (异步)
        
        Args:
            data_type: 数据类型
            limit: 最大返回数量
        
        Returns:
            DataEnvelope 列表
        """
        # 🔥 关键修复：使用 threading.Lock
        with self._index_lock:
            trace_ids = self._type_index.get(data_type, [])[-limit:]
        
        envelopes = []
        for trace_id in trace_ids:
            envelope = await self.read(trace_id)
            if envelope:
                envelopes.append(envelope)
        
        return envelopes
    
    async def get_recent(self, time_window_sec: Optional[float] = None, 
                   zone: Optional[ZoneType] = None,
                   limit: Optional[int] = None) -> List[DataEnvelope]:
        """
        获取指定分区的数据（支持时间窗口和数量限制）
        
        🔥 优化：
        - 取消默认时间窗口限制（改为 None，表示不限制）
        - 移除 [-100:] 硬编码限制
        - 优先使用分区索引直接访问（更快）
        
        Args:
            time_window_sec: 时间窗口 (秒)，None 表示不限制时间
            zone: 指定分区 (可选)
            limit: 最大返回数量，None 表示不限制
        
        Returns:
            DataEnvelope 列表 (按时间排序)
        """
        current_time = time.time()
        cutoff_time = current_time - (time_window_sec or float('inf'))
        
        recent_envelopes = []
        count = 0
        
        # 🔥 优化 1：如果指定了分区，直接使用分区索引（更快）
        if zone:
            # 获取该分区的所有 trace_id
            with self._zone_locks[zone]:
                zone_data = self._get_zone(zone)
                trace_ids = list(zone_data.keys())
            
            # 按时间排序（需要读取数据后排序）
            # 🔥 优化 2：移除 [-100:] 限制，改用 limit 参数
            for trace_id in trace_ids:
                if limit and count >= limit:
                    break
                
                envelope = await self.read(trace_id)
                if envelope:
                    # 如果指定了时间窗口，过滤过期数据
                    if time_window_sec and envelope.timestamp < cutoff_time:
                        continue
                    
                    recent_envelopes.append(envelope)
                    count += 1
            
            # 按时间排序
            recent_envelopes.sort(key=lambda x: x.timestamp, reverse=True)
            
        else:
            # 🔥 未指定分区时，使用时间索引（向后兼容）
            with self._index_lock:
                # 从时间索引查找（移除 [-100:] 限制）
                for timestamp, trace_id, env_zone in reversed(self._time_index):
                    if limit and count >= limit:
                        break
                    
                    if time_window_sec and timestamp < cutoff_time:
                        break
                    
                    # 如果指定了分区，只返回该分区的数据
                    if zone and env_zone != zone:
                        continue
                    
                    # 释放锁后读取
                    envelope = await self.read(trace_id)
                    if envelope:
                        recent_envelopes.append(envelope)
                        count += 1
        
        # 按时间排序（升序）
        recent_envelopes.sort(key=lambda x: x.timestamp)
        
        return recent_envelopes
    
    async def build_context_pack(self, time_window_sec: float = 30.0,
                          include: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        构建上下文包 (TSD v2.5 核心功能)（异步）
        
        Args:
            time_window_sec: 时间窗口 (秒)
            include: 包含的数据类型列表 (e.g. ["vision", "audio", "system"])
        
        Returns:
            上下文包字典
        """
        context = {
            "timestamp": time.time(),
            "time_window_sec": time_window_sec,
            "vision": [],
            "audio": [],
            "text": [],
            "system": [],
            "memory": []
        }
        
        # 获取最近的数据
        recent_data = await self.get_recent(time_window_sec)
        
        for envelope in recent_data:
            # 根据数据类型分类
            if envelope.data_type in [DataType.VIDEO_FRAME, DataType.VIDEO_CLIP, DataType.IMAGE]:
                if not include or "vision" in include:
                    context["vision"].append(envelope.to_dict())
            
            elif envelope.data_type in [DataType.AUDIO_RAW, DataType.AUDIO_FEATURE]:
                if not include or "audio" in include:
                    context["audio"].append(envelope.to_dict())
            
            elif envelope.data_type in [DataType.TEXT_USER, DataType.TEXT_ASSISTANT, DataType.TEXT_SYSTEM]:
                if not include or "text" in include:
                    context["text"].append(envelope.to_dict())
            
            elif envelope.data_type in [DataType.SYSTEM_LOG, DataType.SYSTEM_STATE]:
                if not include or "system" in include:
                    context["system"].append(envelope.to_dict())
            
            elif envelope.data_type == DataType.MEMORY_NODE:
                if not include or "memory" in include:
                    context["memory"].append(envelope.to_dict())
        
        logger.info(f"📦 [SharedMemoryPool] 构建上下文包：{len(context['vision'])}视觉 + {len(context['audio'])}听觉 + {len(context['text'])}文本")
        
        return context
    
    async def link_traces(self, trace_id1: str, trace_id2: str):
        """
        关联两个追踪 ID (用于多模态数据关联)（异步）
        
        Args:
            trace_id1: 追踪 ID 1
            trace_id2: 追踪 ID 2
        """
        envelope1 = await self.read(trace_id1)
        if envelope1:
            envelope1.related_trace_ids.append(trace_id2)
            await self.write(envelope1)
        
        logger.debug(f"🔗 [SharedMemoryPool] 关联 {trace_id1[:8]} ↔ {trace_id2[:8]}")
    
    async def _cleanup_loop(self):
        """后台清理循环 (删除过期数据)（异步）"""
        while self._running:
            await asyncio.sleep(self.auto_cleanup_interval)
            await self._cleanup_expired()
    
    async def _cleanup_expired(self):
        """清理过期数据 (异步)"""
        current_time = time.time()
        cleaned_count = 0
        
        for zone in ZoneType:
            # 🔥 关键修复：使用 threading.Lock
            with self._zone_locks[zone]:
                zone_dict = self._get_zone(zone)
                to_delete = []
                
                # 找出过期数据
                for trace_id, envelope in zone_dict.items():
                    if envelope.expires_at and envelope.expires_at < current_time:
                        to_delete.append(trace_id)
                
                # 删除过期数据
                for trace_id in to_delete:
                    del zone_dict[trace_id]
                    with self._index_lock:
                        self._trace_index.pop(trace_id, None)
                    cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"🧹 [SharedMemoryPool] 清理 {cleaned_count} 条过期数据")
    
    async def _snapshot_loop(self):
        """后台快照循环（每 5 分钟保存一次）（异步）"""
        while self._running:
            await asyncio.sleep(self.snapshot_interval)
            
            # 🔥 关键修改：定期检查并保存待处理的快照
            if not self._save_queue.empty():
                logger.debug(f"💾 [SharedMemoryPool] 快照检查点 (Raw:{len(self._raw_zone)} Memory:{len(self._memory_zone)})")
                await self._process_save_queue()
    
    async def _queue_snapshot_save(self):
        """🔥 将快照保存请求加入队列（异步延迟保存）
        
        核心原理:
        1. 每次写入都触发保存请求 → 保证数据不丢失
        2. 请求加入队列等待处理 → 异步非阻塞
        3. 合并 0.5 秒内的多个请求 → 减少实际 I/O 次数
        4. 后台异步执行保存 → 不影响主流程
        
        性能优势:
        - 用户输入频繁时：10 次写入 → 1 次实际保存
        - 用户输入稀疏时：1 次写入 → 1 次实际保存
        - 保存过程：压缩 JSON → 原子写入 → 清理旧快照
        """
        if not self.persistence_enabled:
            return
        
        # 将保存请求加入队列
        await self._save_queue.put(time.time())
        
        # 🔥 如果没有正在进行的保存任务，启动一个
        if self._pending_save_task is None or self._pending_save_task.done():
            logger.debug(f"💾 [SharedMemoryPool] 触发快照保存 (队列大小：{self._save_queue.qsize()})")
            self._pending_save_task = asyncio.create_task(self._process_save_queue())
    
    async def _process_save_queue(self):
        """🔥 处理保存队列（合并多次保存请求）
        
        工作流程:
        1. 等待 0.5 秒，让多个写入请求累积
        2. 清空队列，合并所有请求为一次保存
        3. 执行实际保存操作
        4. 释放保存锁，允许下一次保存
        
        性能优化:
        - 0.5 秒内 100 次写入 → 只保存 1 次
        - 每次保存约 10-50ms (压缩 + 写入)
        - 不阻塞主流程，异步后台执行
        """
        async with self._save_lock:
            if self._save_pending:
                return
            self._save_pending = True
        
        try:
            # 🔥 等待 0.5 秒，合并多个保存请求
            # 这是关键优化：减少频繁 I/O，但保证数据不丢失
            await asyncio.sleep(0.5)
            
            # 清空队列（合并所有请求）
            saved_count = 0
            while not self._save_queue.empty():
                try:
                    self._save_queue.get_nowait()
                    saved_count += 1
                except asyncio.QueueEmpty:
                    break
            
            if saved_count > 0:
                logger.info(f"💾 [SharedMemoryPool] 合并保存请求：{saved_count} 次 → 1 次实际保存")
                logger.info(f"  - Raw Zone: {len(self._raw_zone)} 条")
                logger.info(f"  - Memory Zone: {len(self._memory_zone)} 条")
            
            # 执行实际保存
            await self._save_snapshot()
            
        finally:
            async with self._save_lock:
                self._save_pending = False
    
    async def _save_snapshot(self):
        """保存快照到磁盘 (压缩存储)（异步）"""
        if not self.persistence_enabled:
            return
        
        try:
            # 确保目录存在
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            
            snapshot_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 准备快照数据
            snapshot_data = {
                "timestamp": snapshot_time,
                "raw_zone": [env.to_dict() for env in self._raw_zone.values()],
                "feature_zone": [env.to_dict() for env in self._feature_zone.values()],
                "system_zone": [env.to_dict() for env in self._system_zone.values()],
                "memory_zone": [env.to_dict() for env in self._memory_zone.values()],
                "trace_index": {k: v.value for k, v in self._trace_index.items()},
                "type_index": {dt.value: ids for dt, ids in self._type_index.items()},
                "stats": self.get_stats()
            }
            
            # 压缩保存
            snapshot_path = self.persistence_path / f"snapshot_{timestamp}.json.gz"
            temp_path = self.persistence_path / "snapshot_temp.json.gz"
            
            # 如果临时文件已存在，先删除
            if temp_path.exists():
                temp_path.unlink()
            
            # 先写到临时文件
            with gzip.open(temp_path, 'wt', encoding='utf-8') as f:
                json.dump(snapshot_data, f, ensure_ascii=False, indent=2)
            
            # 重命名（原子操作，避免写入中断）
            temp_path.rename(snapshot_path)
            
            # 🔥 新增：日志输出
            logger.info(f"💾 [SharedMemoryPool] ✅ 快照已保存：{snapshot_path.name}")
            logger.info(f"  - 时间戳：{timestamp}")
            logger.info(f"  - 总条目：{len(snapshot_data['raw_zone']) + len(snapshot_data['memory_zone'])} 条")
            
            # 清理旧快照
            await self._cleanup_old_snapshots()
            
            self.last_snapshot_time = snapshot_time
            
            logger.info(f"💾 [SharedMemoryPool] 快照已保存：{snapshot_path.name} " +
                       f"(Raw:{len(snapshot_data['raw_zone'])} " +
                       f"Feature:{len(snapshot_data['feature_zone'])} " +
                       f"System:{len(snapshot_data['system_zone'])} " +
                       f"Memory:{len(snapshot_data['memory_zone'])})")
            
        except Exception as e:
            logger.error(f"[SharedMemoryPool] 保存快照失败：{e}", exc_info=True)
    
    async def _load_snapshot(self):
        """从磁盘加载快照（异步）"""
        if not self.persistence_enabled:
            return
        
        try:
            # 查找最新快照
            snapshots = list(self.persistence_path.glob("snapshot_*.json.gz"))
            if not snapshots:
                logger.info("📂 [SharedMemoryPool] 未找到快照，使用空数据池")
                return
            
            # 按时间排序，取最新
            latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
            
            # 加载快照
            with gzip.open(latest_snapshot, 'rt', encoding='utf-8') as f:
                snapshot_data = json.load(f)
            
            # 恢复数据
            # 🔥 关键修复：使用 threading.Lock
            with self._index_lock:
                # 清空当前数据
                self._raw_zone.clear()
                self._feature_zone.clear()
                self._system_zone.clear()
                self._memory_zone.clear()
                self._trace_index.clear()
                self._type_index.clear()
                
                # 恢复各分区
                for env_dict in snapshot_data.get("raw_zone", []):
                    envelope = DataEnvelope.from_dict(env_dict)
                    self._raw_zone[envelope.trace_id] = envelope
                    self._trace_index[envelope.trace_id] = ZoneType.RAW
                
                for env_dict in snapshot_data.get("feature_zone", []):
                    envelope = DataEnvelope.from_dict(env_dict)
                    self._feature_zone[envelope.trace_id] = envelope
                    self._trace_index[envelope.trace_id] = ZoneType.FEATURE
                
                for env_dict in snapshot_data.get("system_zone", []):
                    envelope = DataEnvelope.from_dict(env_dict)
                    self._system_zone[envelope.trace_id] = envelope
                    self._trace_index[envelope.trace_id] = ZoneType.SYSTEM
                
                for env_dict in snapshot_data.get("memory_zone", []):
                    envelope = DataEnvelope.from_dict(env_dict)
                    self._memory_zone[envelope.trace_id] = envelope
                    self._trace_index[envelope.trace_id] = ZoneType.MEMORY
                
                # 恢复索引
                for trace_id, zone_value in snapshot_data.get("trace_index", {}).items():
                    self._trace_index[trace_id] = ZoneType(zone_value)
                
                for dt_value, ids in snapshot_data.get("type_index", {}).items():
                    data_type = DataType(dt_value)
                    self._type_index[data_type] = ids
                
                total_count = (len(self._raw_zone) + len(self._feature_zone) + 
                              len(self._system_zone) + len(self._memory_zone))
            
            logger.info(f"✅ [SharedMemoryPool] 已恢复快照：{latest_snapshot.name} " +
                       f"(共{total_count}条数据)")
            
        except Exception as e:
            logger.error(f"[SharedMemoryPool] 加载快照失败：{e}", exc_info=True)
    
    async def _cleanup_old_snapshots(self):
        """清理旧快照，保留最近 N 个（异步）"""
        try:
            snapshots = sorted(
                self.persistence_path.glob("snapshot_*.json.gz"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            # 删除超出数量的快照
            for snapshot in snapshots[self.max_snapshots:]:
                snapshot.unlink()
                logger.debug(f"🗑️ [SharedMemoryPool] 删除旧快照：{snapshot.name}")
                
        except Exception as e:
            logger.error(f"[SharedMemoryPool] 清理快照失败：{e}", exc_info=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息（同步方法，无锁安全）"""
        return {
            "raw_zone_size": len(self._raw_zone),
            "feature_zone_size": len(self._feature_zone),
            "system_zone_size": len(self._system_zone),
            "memory_zone_size": len(self._memory_zone),
            "total_traces": len(self._trace_index),
            "data_types": {dt.value: len(ids) for dt, ids in self._type_index.items()},
            "persistence": {
                "enabled": self.persistence_enabled,
                "path": str(self.persistence_path),
                "last_snapshot": self.last_snapshot_time,
                "snapshot_interval": self.snapshot_interval
            }
        }
    
    async def save_snapshot_now(self):
        """手动触发快照保存（异步）"""
        logger.info("📸 [SharedMemoryPool] 手动触发快照保存...")
        await self._save_snapshot()
    
    async def shutdown(self):
        """🔥 关闭共享池（异步）
        
        关闭流程:
        1. 停止后台任务
        2. 立即保存最终快照（不延迟）
        3. 关闭所有资源
        """
        self._running = False
        
        # 🔥 关闭前保存最后快照（立即保存，不延迟）
        if self.persistence_enabled:
            logger.info("💾 [SharedMemoryPool] 关闭前保存最终快照...")
            await self._save_snapshot()
        
        logger.info("🛑 [SharedMemoryPool] 已关闭")
    
    # ========== 任务队列专用方法 (Memory Zone) ==========
    
    async def write_task_queue(self, task_id: str, task_data: Dict[str, Any]) -> str:
        """写入任务到 Memory Zone 任务队列
        
        用于 L2 向 L1-B 传递复杂任务子任务列表。
        
        Args:
            task_id: 任务 ID (父任务 ID)
            task_data: 任务数据 (包含 subtasks, dependencies, parallel_groups 等)
            
        Returns:
            trace_id: 追踪 ID
        """
        trace_id = f"task_queue_{task_id}"
        
        envelope = DataEnvelope(
            trace_id=trace_id,
            timestamp=time.time(),
            data_type=DataType.SYSTEM_STATE,
            zone=ZoneType.MEMORY,
            payload={
                'task_id': task_id,
                'status': 'PENDING',
                'data': task_data,
                'created_at': time.time()
            },
            metadata={'type': 'task_queue', 'parent_task_id': task_id}
        )
        
        await self.write(envelope)
        logger.info(f"📋 [SharedMemoryPool] 任务已写入队列: {task_id}")
        return trace_id
    
    async def read_task_queue(self, task_id: str) -> Optional[Dict[str, Any]]:
        """从 Memory Zone 任务队列读取任务
        
        Args:
            task_id: 任务 ID
            
        Returns:
            任务数据，不存在返回 None
        """
        trace_id = f"task_queue_{task_id}"
        envelope = await self.read_memory(trace_id)
        
        if envelope:
            logger.info(f"📖 [SharedMemoryPool] 任务已从队列读取: {task_id}")
            return envelope.payload
        
        return None
    
    async def update_task_queue_status(self, task_id: str, status: str, results: Optional[Dict[str, Any]] = None) -> bool:
        """更新任务队列中的任务状态
        
        Args:
            task_id: 任务 ID
            status: 新状态 (PENDING, EXECUTING, COMPLETED, FAILED)
            results: 执行结果 (可选)
            
        Returns:
            是否更新成功
        """
        trace_id = f"task_queue_{task_id}"
        envelope = await self.read_memory(trace_id)
        
        if not envelope:
            logger.warning(f"⚠️ [SharedMemoryPool] 任务不存在，无法更新: {task_id}")
            return False
        
        # 更新 payload
        envelope.payload['status'] = status
        envelope.payload['updated_at'] = time.time()
        if results:
            envelope.payload['results'] = results
        
        await self.write(envelope)
        logger.info(f"✏️ [SharedMemoryPool] 任务状态已更新: {task_id} -> {status}")
        return True
    
    async def list_task_queue(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """列出任务队列中的任务
        
        Args:
            status: 过滤状态 (可选)
            limit: 最大返回数量
            
        Returns:
            任务数据列表
        """
        envelopes = await self.get_recent(zone=ZoneType.MEMORY, limit=limit)
        
        result = []
        for env in envelopes:
            if env.metadata.get('type') != 'task_queue':
                continue
            
            if status and env.payload.get('status') != status:
                continue
            
            result.append(env.payload)
        
        return result
    
    async def delete_task_queue_item(self, task_id: str) -> bool:
        """从任务队列删除任务
        
        Args:
            task_id: 任务 ID
            
        Returns:
            是否删除成功
        """
        trace_id = f"task_queue_{task_id}"
        
        # 从 Memory Zone 删除
        with self._zone_locks[ZoneType.MEMORY]:
            if trace_id in self._memory_zone:
                del self._memory_zone[trace_id]
                with self._index_lock:
                    self._trace_index.pop(trace_id, None)
                logger.info(f"🗑️ [SharedMemoryPool] 任务已从队列删除: {task_id}")
                return True
        
        return False
    
    def _sync_shutdown(self):
        """🔥 同步关闭方法 (atexit 回调)
        
        当 Python 解释器退出时自动调用，确保异常退出时也能保存数据
        """
        try:
            logger.info("🛑 [SharedMemoryPool] 系统退出钩子触发，保存最终快照...")
            
            # 尝试获取事件循环
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果事件循环还在运行，安排保存任务
                    future = asyncio.run_coroutine_threadsafe(
                        self._save_snapshot(),
                        loop
                    )
                    # 等待保存完成（最多 5 秒）
                    future.result(timeout=5)
                    logger.info("💾 [SharedMemoryPool] ✅ 最终快照已保存 (异步等待)")
                else:
                    # 如果事件循环已停止，创建新循环
                    loop.run_until_complete(self._save_snapshot())
                    logger.info("💾 [SharedMemoryPool] ✅ 最终快照已保存 (新循环)")
            except RuntimeError:
                # 没有事件循环，创建一个新的
                new_loop = asyncio.new_event_loop()
                new_loop.run_until_complete(self._save_snapshot())
                new_loop.close()
                logger.info("💾 [SharedMemoryPool] ✅ 最终快照已保存 (新事件循环)")
            
        except Exception as e:
            logger.error(f"❌ [SharedMemoryPool] 系统退出时保存快照失败：{e}")


# 🔥 全局异步单例（推荐用法）
async def get_shared_memory_pool() -> SharedMemoryPool:
    """异步获取共享池单例（自动加载持久化数据）"""
    return await SharedMemoryPool.get_instance()
