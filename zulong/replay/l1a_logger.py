"""
AT-03: L1-A反射层日志扩展
增加传感器原始数据（VAD、视觉流）的低延迟写入接口
关键指标: 感知数据需包含VAD触发时间和物体识别时间
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import logging

from .clock_synchronizer import get_unified_timestamp
from .ring_buffer import get_ring_buffer

logger = logging.getLogger(__name__)


class L1AEventType(Enum):
    VAD_TRIGGER = "vad_trigger"
    VAD_SILENCE = "vad_silence"
    VISION_DETECTION = "vision_detection"
    VISION_TRACKING = "vision_tracking"
    REFLEX_TRIGGERED = "reflex_triggered"
    REFLEX_SUPPRESSED = "reflex_suppressed"
    OBSTACLE_DETECTED = "obstacle_detected"
    TARGET_LOST = "target_lost"
    SENSOR_DATA = "sensor_data"


@dataclass
class VADData:
    is_speech: bool
    confidence: float
    energy: float
    duration_ms: float
    timestamp_trigger: float
    timestamp_end: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_speech": self.is_speech,
            "confidence": self.confidence,
            "energy": self.energy,
            "duration_ms": self.duration_ms,
            "timestamp_trigger": self.timestamp_trigger,
            "timestamp_end": self.timestamp_end
        }


@dataclass
class VisionDetectionData:
    object_id: str
    object_type: str
    confidence: float
    bbox: List[float]
    position_3d: Optional[List[float]] = None
    timestamp_detection: float = 0.0
    tracking_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "position_3d": self.position_3d,
            "timestamp_detection": self.timestamp_detection,
            "tracking_id": self.tracking_id
        }


@dataclass
class ReflexAction:
    reflex_id: str
    reflex_type: str
    trigger_source: str
    action_taken: str
    suppressed: bool = False
    suppress_reason: Optional[str] = None
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reflex_id": self.reflex_id,
            "reflex_type": self.reflex_type,
            "trigger_source": self.trigger_source,
            "action_taken": self.action_taken,
            "suppressed": self.suppressed,
            "suppress_reason": self.suppress_reason,
            "timestamp": self.timestamp
        }


class L1ALogger:
    """
    L1-A反射层日志记录器
    
    记录VAD、视觉检测、反射动作等感知数据
    """
    
    def __init__(self, ring_buffer=None):
        self._ring_buffer = ring_buffer or get_ring_buffer()
        self._vad_counter = 0
        self._vision_counter = 0
        self._reflex_counter = 0
        
        self._stats = {
            "vad_triggers": 0,
            "vision_detections": 0,
            "reflexes_triggered": 0,
            "reflexes_suppressed": 0,
            "obstacles_detected": 0
        }
        
        logger.info("[L1ALogger] 初始化完成")
    
    def log_vad_trigger(
        self,
        confidence: float,
        energy: float,
        duration_ms: float,
        audio_features: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        记录VAD触发
        
        Args:
            confidence: 置信度
            energy: 能量值
            duration_ms: 持续时间 (毫秒)
            audio_features: 音频特征
        
        Returns:
            str: VAD事件ID
        """
        self._vad_counter += 1
        vad_id = f"VAD_{get_unified_timestamp():.0f}_{self._vad_counter}"
        timestamp = get_unified_timestamp()
        
        vad_data = VADData(
            is_speech=True,
            confidence=confidence,
            energy=energy,
            duration_ms=duration_ms,
            timestamp_trigger=timestamp
        )
        
        data = vad_data.to_dict()
        if audio_features:
            data["audio_features"] = audio_features
        
        self._ring_buffer.write_event(
            layer="L1-A",
            event_type=L1AEventType.VAD_TRIGGER.value,
            data=data,
            metadata={"vad_id": vad_id}
        )
        
        self._stats["vad_triggers"] += 1
        logger.debug(f"[L1ALogger] VAD触发: {vad_id}, 置信度: {confidence:.2f}")
        return vad_id
    
    def log_vad_silence(self, duration_ms: float):
        """记录VAD静音"""
        self._ring_buffer.write_event(
            layer="L1-A",
            event_type=L1AEventType.VAD_SILENCE.value,
            data={
                "is_speech": False,
                "duration_ms": duration_ms,
                "timestamp": get_unified_timestamp()
            }
        )
    
    def log_vision_detection(
        self,
        object_type: str,
        confidence: float,
        bbox: List[float],
        position_3d: Optional[List[float]] = None,
        tracking_id: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        记录视觉检测
        
        Args:
            object_type: 物体类型
            confidence: 置信度
            bbox: 边界框 [x1, y1, x2, y2]
            position_3d: 3D位置 [x, y, z]
            tracking_id: 跟踪ID
            additional_info: 附加信息
        
        Returns:
            str: 检测事件ID
        """
        self._vision_counter += 1
        detection_id = f"VIS_{get_unified_timestamp():.0f}_{self._vision_counter}"
        timestamp = get_unified_timestamp()
        
        detection = VisionDetectionData(
            object_id=detection_id,
            object_type=object_type,
            confidence=confidence,
            bbox=bbox,
            position_3d=position_3d,
            timestamp_detection=timestamp,
            tracking_id=tracking_id
        )
        
        data = detection.to_dict()
        if additional_info:
            data["additional_info"] = additional_info
        
        self._ring_buffer.write_event(
            layer="L1-A",
            event_type=L1AEventType.VISION_DETECTION.value,
            data=data,
            metadata={"detection_id": detection_id}
        )
        
        self._stats["vision_detections"] += 1
        logger.debug(f"[L1ALogger] 视觉检测: {detection_id}, 类型: {object_type}")
        return detection_id
    
    def log_vision_tracking(
        self,
        tracking_id: str,
        object_type: str,
        position: List[float],
        velocity: Optional[List[float]] = None
    ):
        """记录视觉跟踪"""
        self._ring_buffer.write_event(
            layer="L1-A",
            event_type=L1AEventType.VISION_TRACKING.value,
            data={
                "tracking_id": tracking_id,
                "object_type": object_type,
                "position": position,
                "velocity": velocity,
                "timestamp": get_unified_timestamp()
            },
            metadata={"tracking_id": tracking_id}
        )
    
    def log_reflex_triggered(
        self,
        reflex_type: str,
        trigger_source: str,
        action_taken: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        记录反射动作触发
        
        Args:
            reflex_type: 反射类型
            trigger_source: 触发源
            action_taken: 执行的动作
            context: 上下文信息
        
        Returns:
            str: 反射ID
        """
        self._reflex_counter += 1
        reflex_id = f"REFLEX_{get_unified_timestamp():.0f}_{self._reflex_counter}"
        timestamp = get_unified_timestamp()
        
        reflex = ReflexAction(
            reflex_id=reflex_id,
            reflex_type=reflex_type,
            trigger_source=trigger_source,
            action_taken=action_taken,
            suppressed=False,
            timestamp=timestamp
        )
        
        data = reflex.to_dict()
        if context:
            data["context"] = context
        
        self._ring_buffer.write_event(
            layer="L1-A",
            event_type=L1AEventType.REFLEX_TRIGGERED.value,
            data=data,
            metadata={"reflex_id": reflex_id}
        )
        
        self._stats["reflexes_triggered"] += 1
        logger.info(f"[L1ALogger] 反射触发: {reflex_id}, 类型: {reflex_type}")
        return reflex_id
    
    def log_reflex_suppressed(
        self,
        reflex_type: str,
        trigger_source: str,
        suppress_reason: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """记录反射动作被抑制"""
        self._reflex_counter += 1
        reflex_id = f"REFLEX_{get_unified_timestamp():.0f}_{self._reflex_counter}"
        
        reflex = ReflexAction(
            reflex_id=reflex_id,
            reflex_type=reflex_type,
            trigger_source=trigger_source,
            action_taken="none",
            suppressed=True,
            suppress_reason=suppress_reason,
            timestamp=get_unified_timestamp()
        )
        
        data = reflex.to_dict()
        if context:
            data["context"] = context
        
        self._ring_buffer.write_event(
            layer="L1-A",
            event_type=L1AEventType.REFLEX_SUPPRESSED.value,
            data=data,
            metadata={"reflex_id": reflex_id, "suppressed": "true"}
        )
        
        self._stats["reflexes_suppressed"] += 1
        logger.info(f"[L1ALogger] 反射抑制: {reflex_id}, 原因: {suppress_reason}")
    
    def log_obstacle_detected(
        self,
        obstacle_type: str,
        distance: float,
        direction: Optional[List[float]] = None,
        urgency: str = "normal"
    ):
        """记录障碍物检测"""
        self._ring_buffer.write_event(
            layer="L1-A",
            event_type=L1AEventType.OBSTACLE_DETECTED.value,
            data={
                "obstacle_type": obstacle_type,
                "distance": distance,
                "direction": direction,
                "urgency": urgency,
                "timestamp": get_unified_timestamp()
            }
        )
        
        self._stats["obstacles_detected"] += 1
        logger.info(f"[L1ALogger] 障碍物检测: {obstacle_type}, 距离: {distance:.2f}m")
    
    def log_target_lost(
        self,
        target_id: str,
        last_known_position: Optional[List[float]] = None,
        reason: Optional[str] = None
    ):
        """记录目标丢失"""
        self._ring_buffer.write_event(
            layer="L1-A",
            event_type=L1AEventType.TARGET_LOST.value,
            data={
                "target_id": target_id,
                "last_known_position": last_known_position,
                "reason": reason,
                "timestamp": get_unified_timestamp()
            },
            metadata={"target_id": target_id, "event": "target_lost"}
        )
        
        logger.info(f"[L1ALogger] 目标丢失: {target_id}, 原因: {reason}")
    
    def log_sensor_data(
        self,
        sensor_type: str,
        sensor_id: str,
        raw_data: Dict[str, Any],
        processed_data: Optional[Dict[str, Any]] = None
    ):
        """记录传感器原始数据"""
        self._ring_buffer.write_event(
            layer="L1-A",
            event_type=L1AEventType.SENSOR_DATA.value,
            data={
                "sensor_type": sensor_type,
                "sensor_id": sensor_id,
                "raw_data": raw_data,
                "processed_data": processed_data,
                "timestamp": get_unified_timestamp()
            },
            metadata={"sensor_id": sensor_id}
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "stats": self._stats.copy()
        }


_global_l1a_logger: Optional[L1ALogger] = None


def get_l1a_logger() -> L1ALogger:
    """获取全局L1-A日志记录器"""
    global _global_l1a_logger
    if _global_l1a_logger is None:
        _global_l1a_logger = L1ALogger()
    return _global_l1a_logger
