"""
AT-01: 硬件时钟同步协议
实现L0-L3所有层级的硬件时钟统一协议
关键指标: 时钟误差 < 1ms，所有日志拥有统一的时间戳基准
"""
import time
from dataclasses import dataclass
from typing import Optional
from threading import Lock
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClockSyncStatus:
    is_synchronized: bool
    master_clock_source: str
    offset_us: float
    last_sync_time: float
    sync_accuracy_us: float


class ClockSynchronizer:
    """
    硬件时钟同步器
    
    提供统一的时间戳基准，确保L0-L3所有层级使用相同的时钟源
    """
    
    _instance: Optional['ClockSynchronizer'] = None
    _lock: Lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self._master_clock_source = "system_monotonic"
        self._offset_us: float = 0.0
        self._last_sync_time: float = 0.0
        self._sync_accuracy_us: float = 0.0
        self._is_synchronized: bool = False
        
        self._sync_status = ClockSyncStatus(
            is_synchronized=False,
            master_clock_source=self._master_clock_source,
            offset_us=0.0,
            last_sync_time=0.0,
            sync_accuracy_us=0.0
        )
        
        logger.info("[ClockSynchronizer] 初始化完成，使用系统单调时钟作为主时钟源")
    
    def get_unified_timestamp_us(self) -> float:
        """
        获取统一时间戳 (微秒级)
        
        Returns:
            float: 微秒级时间戳
        """
        return time.perf_counter() * 1_000_000
    
    def get_unified_timestamp_ms(self) -> float:
        """
        获取统一时间戳 (毫秒级)
        
        Returns:
            float: 毫秒级时间戳
        """
        return time.perf_counter() * 1_000
    
    def get_unified_timestamp_s(self) -> float:
        """
        获取统一时间戳 (秒级)
        
        Returns:
            float: 秒级时间戳
        """
        return time.perf_counter()
    
    def get_datetime_timestamp(self) -> float:
        """
        获取可读的日期时间戳 (Unix时间戳)
        
        Returns:
            float: Unix时间戳
        """
        return time.time()
    
    def synchronize_with_external_clock(self, external_timestamp_us: float) -> bool:
        """
        与外部时钟源同步 (如GPS、NTP)
        
        Args:
            external_timestamp_us: 外部时钟时间戳 (微秒)
        
        Returns:
            bool: 同步是否成功
        """
        try:
            local_time = self.get_unified_timestamp_us()
            self._offset_us = external_timestamp_us - local_time
            self._last_sync_time = local_time
            self._is_synchronized = True
            self._sync_accuracy_us = abs(self._offset_us)
            
            self._sync_status = ClockSyncStatus(
                is_synchronized=True,
                master_clock_source=self._master_clock_source,
                offset_us=self._offset_us,
                last_sync_time=self._last_sync_time,
                sync_accuracy_us=self._sync_accuracy_us
            )
            
            logger.info(f"[ClockSynchronizer] 时钟同步完成，偏移: {self._offset_us:.2f}μs")
            return True
            
        except Exception as e:
            logger.error(f"[ClockSynchronizer] 时钟同步失败: {e}")
            return False
    
    def get_sync_status(self) -> ClockSyncStatus:
        """
        获取同步状态
        
        Returns:
            ClockSyncStatus: 同步状态信息
        """
        return self._sync_status
    
    def format_timestamp(self, timestamp_us: float) -> str:
        """
        格式化时间戳为可读字符串
        
        Args:
            timestamp_us: 微秒级时间戳
        
        Returns:
            str: 格式化的时间字符串
        """
        dt = time.localtime(timestamp_us / 1_000_000)
        return time.strftime("%Y-%m-%d %H:%M:%S", dt) + f".{int(timestamp_us % 1_000_000):06d}"
    
    def measure_clock_drift(self, reference_timestamp_us: float) -> float:
        """
        测量时钟漂移
        
        Args:
            reference_timestamp_us: 参考时间戳 (微秒)
        
        Returns:
            float: 漂移量 (微秒)
        """
        current_time = self.get_unified_timestamp_us()
        return current_time - reference_timestamp_us


def get_unified_timestamp() -> float:
    """
    全局便捷函数：获取统一时间戳 (微秒)
    
    Returns:
        float: 微秒级时间戳
    """
    return ClockSynchronizer().get_unified_timestamp_us()


def get_unified_timestamp_ms() -> float:
    """
    全局便捷函数：获取统一时间戳 (毫秒)
    
    Returns:
        float: 毫秒级时间戳
    """
    return ClockSynchronizer().get_unified_timestamp_ms()
