"""
AT-05: 环形缓冲区实现
基于TSD v2.0规范，实现保留最近60s数据的内存缓冲区
关键指标: 缓冲区大小可配置，支持按需"固化"为事件档案
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Deque
from collections import deque
from threading import Lock
import logging
import json

from .clock_synchronizer import get_unified_timestamp, ClockSynchronizer

logger = logging.getLogger(__name__)


@dataclass
class RingBufferSlot:
    """
    环形缓冲区槽位
    
    存储单个时间点的多层级日志数据
    """
    timestamp: float
    layer: str
    event_type: str
    data: Dict[str, Any]
    metadata: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "layer": self.layer,
            "event_type": self.event_type,
            "data": self.data,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RingBufferSlot':
        return cls(
            timestamp=data["timestamp"],
            layer=data["layer"],
            event_type=data["event_type"],
            data=data["data"],
            metadata=data.get("metadata", {})
        )


class MultiLayerRingBuffer:
    """
    多层级环形缓冲区
    
    保留最近N秒的L0-L3所有层级日志数据
    支持按需固化、快照提取、时间范围查询
    """
    
    def __init__(self, duration_seconds: int = 60, max_slots: int = 100000):
        """
        初始化环形缓冲区
        
        Args:
            duration_seconds: 保留时长 (秒)，默认60秒
            max_slots: 最大槽位数，防止内存溢出
        """
        self.duration = duration_seconds
        self.max_slots = max_slots
        self._slots: Deque[RingBufferSlot] = deque(maxlen=max_slots)
        self._lock = Lock()
        self._clock = ClockSynchronizer()
        
        self._stats = {
            "total_writes": 0,
            "total_evictions": 0,
            "total_snapshots": 0,
            "total_freezes": 0
        }
        
        logger.info(f"[RingBuffer] 初始化完成，保留时长: {duration_seconds}s，最大槽位: {max_slots}")
    
    def write(self, slot: RingBufferSlot) -> bool:
        """
        写入新数据，自动清理过期数据
        
        Args:
            slot: 数据槽位
        
        Returns:
            bool: 写入是否成功
        """
        try:
            with self._lock:
                self._slots.append(slot)
                self._stats["total_writes"] += 1
                
                evicted_count = self._evict_expired()
                if evicted_count > 0:
                    self._stats["total_evictions"] += evicted_count
                
                return True
        except Exception as e:
            logger.error(f"[RingBuffer] 写入失败: {e}")
            return False
    
    def write_event(
        self,
        layer: str,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        便捷方法：写入事件数据
        
        Args:
            layer: 层级 (L0/L1-A/L1-B/L2/L3)
            event_type: 事件类型
            data: 事件数据
            metadata: 元数据
        
        Returns:
            bool: 写入是否成功
        """
        slot = RingBufferSlot(
            timestamp=get_unified_timestamp(),
            layer=layer,
            event_type=event_type,
            data=data,
            metadata=metadata or {}
        )
        return self.write(slot)
    
    def _evict_expired(self) -> int:
        """
        清理过期数据
        
        Returns:
            int: 清理的槽位数
        """
        current_time = get_unified_timestamp()
        cutoff_time = current_time - (self.duration * 1_000_000)
        
        evicted_count = 0
        while self._slots and self._slots[0].timestamp < cutoff_time:
            self._slots.popleft()
            evicted_count += 1
        
        return evicted_count
    
    def snapshot(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        layer: Optional[str] = None,
        event_type: Optional[str] = None
    ) -> List[RingBufferSlot]:
        """
        按条件提取快照
        
        Args:
            start_time: 起始时间 (微秒)，默认为最早数据
            end_time: 结束时间 (微秒)，默认为当前时间
            layer: 过滤层级
            event_type: 过滤事件类型
        
        Returns:
            List[RingBufferSlot]: 符合条件的槽位列表
        """
        with self._lock:
            current_time = get_unified_timestamp()
            
            if start_time is None:
                start_time = 0
            if end_time is None:
                end_time = current_time
            
            result = []
            for slot in self._slots:
                if start_time <= slot.timestamp <= end_time:
                    if layer is not None and slot.layer != layer:
                        continue
                    if event_type is not None and slot.event_type != event_type:
                        continue
                    result.append(slot)
            
            self._stats["total_snapshots"] += 1
            return result
    
    def get_recent(self, seconds: float = 30.0) -> List[RingBufferSlot]:
        """
        获取最近N秒的数据
        
        Args:
            seconds: 时间范围 (秒)
        
        Returns:
            List[RingBufferSlot]: 槽位列表
        """
        current_time = get_unified_timestamp()
        start_time = current_time - (seconds * 1_000_000)
        return self.snapshot(start_time=start_time, end_time=current_time)
    
    def freeze(self, event_id: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        固化当前缓冲区为可序列化格式
        
        Args:
            event_id: 事件ID
            output_path: 输出路径 (可选)
        
        Returns:
            Dict: 固化数据
        """
        with self._lock:
            frozen_data = {
                "event_id": event_id,
                "freeze_time": get_unified_timestamp(),
                "duration_seconds": self.duration,
                "slot_count": len(self._slots),
                "slots": [slot.to_dict() for slot in self._slots]
            }
            
            self._stats["total_freezes"] += 1
            
            if output_path:
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(frozen_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"[RingBuffer] 固化完成: {output_path}")
                except Exception as e:
                    logger.error(f"[RingBuffer] 固化写入失败: {e}")
            
            return frozen_data
    
    def search_by_keyword(self, keyword: str) -> List[RingBufferSlot]:
        """
        按关键词搜索
        
        Args:
            keyword: 搜索关键词
        
        Returns:
            List[RingBufferSlot]: 匹配的槽位列表
        """
        with self._lock:
            result = []
            for slot in self._slots:
                slot_str = json.dumps(slot.to_dict(), ensure_ascii=False)
                if keyword.lower() in slot_str.lower():
                    result.append(slot)
            return result
    
    def get_layer_summary(self) -> Dict[str, int]:
        """
        获取各层级数据统计
        
        Returns:
            Dict[str, int]: 各层级槽位数量
        """
        with self._lock:
            summary = {}
            for slot in self._slots:
                if slot.layer not in summary:
                    summary[slot.layer] = 0
                summary[slot.layer] += 1
            return summary
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取缓冲区统计信息
        
        Returns:
            Dict: 统计信息
        """
        with self._lock:
            return {
                "current_slots": len(self._slots),
                "max_slots": self.max_slots,
                "duration_seconds": self.duration,
                "stats": self._stats.copy(),
                "layer_summary": self.get_layer_summary()
            }
    
    def clear(self):
        """清空缓冲区"""
        with self._lock:
            self._slots.clear()
            logger.info("[RingBuffer] 缓冲区已清空")
    
    def __len__(self) -> int:
        return len(self._slots)
    
    def __repr__(self) -> str:
        return f"MultiLayerRingBuffer(slots={len(self._slots)}, duration={self.duration}s)"


_global_ring_buffer: Optional[MultiLayerRingBuffer] = None


def get_ring_buffer(duration_seconds: int = 60) -> MultiLayerRingBuffer:
    """
    获取全局环形缓冲区实例
    
    Args:
        duration_seconds: 保留时长 (秒)
    
    Returns:
        MultiLayerRingBuffer: 全局实例
    """
    global _global_ring_buffer
    if _global_ring_buffer is None:
        _global_ring_buffer = MultiLayerRingBuffer(duration_seconds=duration_seconds)
    return _global_ring_buffer
