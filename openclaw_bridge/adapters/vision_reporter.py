# File: openclaw_bridge\adapters\vision_reporter.py
"""
OpenClaw 视觉状态报告器

功能：
1. 每秒 10 次（或变化时）扫描 OpenClaw 仿真/真实环境中的物体坐标
2. 封装为 ZulongEvent
3. 发送给 L1-B，供 L2 进行"上下文打包"和"状态对比"

对应 TSD v2.2 第 3.1 节（事件模型）、第 4.2 节（L1-B 上下文打包）
关键配置：type="SENSOR_VISION", source="openclaw/camera"
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field

from openclaw_bridge.openclaw_types import ZulongEvent, OpenClawEventType, OpenClawEventPriority
from openclaw_bridge.event_bus_client import EventBusClient

logger = logging.getLogger(__name__)


@dataclass
class VisionObject:
    """视觉物体对象"""
    object_id: str  # 物体唯一标识
    name: str  # 物体名称（如 "苹果", "杯子"）
    position: Dict[str, float]  # 位置 {x, y, z}
    status: str = "normal"  # 状态（normal, moved, missing, dropped）
    confidence: float = 0.95  # 检测置信度
    timestamp: float = field(default_factory=time.time)  # 时间戳
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "object_id": self.object_id,
            "name": self.name,
            "position": self.position,
            "status": self.status,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }
    
    def __eq__(self, other):
        """比较两个物体是否相同（用于检测变化）"""
        if not isinstance(other, VisionObject):
            return False
        return (
            self.object_id == other.object_id and
            self.name == other.name and
            abs(self.position.get("x", 0) - other.position.get("x", 0)) < 0.01 and
            abs(self.position.get("y", 0) - other.position.get("y", 0)) < 0.01 and
            abs(self.position.get("z", 0) - other.position.get("z", 0)) < 0.01
        )


@dataclass
class VisionConfig:
    """视觉配置"""
    report_rate: float = 10.0  # 报告频率（Hz）
    change_threshold: float = 0.05  # 位置变化阈值（米）
    mock_mode: bool = True  # Mock 模式
    enable_change_detection: bool = True  # 启用变化检测


class OpenClawVisionReporter:
    """
    OpenClaw 视觉状态报告器
    
    工作流程：
    1. 定期扫描环境中的物体
    2. 检测物体位置变化（用于 L2 状态对比）
    3. 封装为视觉事件
    4. 通过 EventBus 发送给 L1-B
    """
    
    def __init__(
        self,
        event_bus: EventBusClient,
        config: Optional[VisionConfig] = None
    ):
        """
        初始化视觉报告器
        
        Args:
            event_bus: EventBus 客户端
            config: 视觉配置
        """
        self.event_bus = event_bus
        self.config = config or VisionConfig()
        self._running = False
        self._last_objects: Dict[str, VisionObject] = {}
        self._report_interval = 1.0 / self.config.report_rate
        
        logger.info(
            f"[OpenClawVisionReporter] 初始化完成，报告频率：{self.config.report_rate}Hz"
        )
    
    async def start(self):
        """启动视觉报告"""
        logger.info("[OpenClawVisionReporter] 启动视觉报告...")
        self._running = True
        
        if self.config.mock_mode:
            # Mock 模式：模拟视觉数据
            asyncio.create_task(self._mock_vision_loop())
            logger.info("[OpenClawVisionReporter] ✅ Mock 模式已启动")
        else:
            # 真实模式：连接相机硬件
            asyncio.create_task(self._real_vision_loop())
            logger.info("[OpenClawVisionReporter] ✅ 真实模式已启动")
    
    async def stop(self):
        """停止视觉报告"""
        logger.info("[OpenClawVisionReporter] 停止视觉报告...")
        self._running = False
    
    async def _mock_vision_loop(self):
        """
        Mock 视觉输入循环（开发测试用）
        
        被动模式：等待外部触发，不主动发送模拟数据
        """
        logger.info("[OpenClawVisionReporter] Mock 视觉循环已启动（被动模式）")
        
        # 被动模式：等待外部触发，不主动发送模拟数据
        while self._running:
            await asyncio.sleep(1.0)
    
    def _detect_changes(
        self,
        new_objects: List[VisionObject]
    ) -> List[VisionObject]:
        """
        检测物体变化
        
        Args:
            new_objects: 新的物体列表
        
        Returns:
            List[VisionObject]: 变化的物体列表
        """
        changed_objects = []
        
        for new_obj in new_objects:
            old_obj = self._last_objects.get(new_obj.object_id)
            
            if old_obj is None:
                # 新物体
                new_obj.status = "new"
                changed_objects.append(new_obj)
                logger.info(
                    f"[OpenClawVisionReporter] 🔍 新物体：{new_obj.name}"
                )
            elif old_obj != new_obj:
                # 物体变化
                new_obj.status = "changed"
                changed_objects.append(new_obj)
                logger.info(
                    f"[OpenClawVisionReporter] 🔍 物体变化：{new_obj.name}"
                )
            
            # 更新缓存
            self._last_objects[new_obj.object_id] = new_obj
        
        # 检测消失的物体
        old_ids = set(self._last_objects.keys())
        new_ids = {obj.object_id for obj in new_objects}
        missing_ids = old_ids - new_ids
        
        for missing_id in missing_ids:
            missing_obj = self._last_objects[missing_id]
            missing_obj.status = "missing"
            changed_objects.append(missing_obj)
            logger.info(
                f"[OpenClawVisionReporter] 🔍 物体消失：{missing_obj.name}"
            )
            del self._last_objects[missing_id]
        
        return changed_objects
    
    async def _report_vision(self, objects: List[VisionObject]):
        """
        报告视觉信息
        
        Args:
            objects: 物体列表
        """
        # 转换为字典格式
        objects_data = [obj.to_dict() for obj in objects]
        
        # 创建视觉事件
        event = ZulongEvent(
            type=OpenClawEventType.SENSOR_VISION,
            source="openclaw/camera",
            payload={"objects": objects_data},
            priority=OpenClawEventPriority.NORMAL
        )
        
        # 发布到 EventBus
        self.event_bus.publish(event)
        
        logger.debug(
            f"[OpenClawVisionReporter] 📷 视觉报告已发送，物体数量：{len(objects)}"
        )
    
    async def _real_vision_loop(self):
        """
        真实视觉输入循环
        
        使用 OpenClaw SDK 或 OpenCV 实现真实视觉处理
        TODO: 实现真实硬件对接
        """
        logger.warning(
            "[OpenClawVisionReporter] ⚠️ 真实模式尚未实现，使用 Mock 模式"
        )
        await self._mock_vision_loop()
    
    def get_current_objects(self) -> Dict[str, VisionObject]:
        """
        获取当前物体列表
        
        Returns:
            Dict[str, VisionObject]: 物体字典
        """
        return self._last_objects.copy()


# 便捷创建函数

def create_vision_reporter(
    event_bus: EventBusClient,
    mock_mode: bool = True,
    report_rate: float = 10.0
) -> OpenClawVisionReporter:
    """
    创建视觉报告器
    
    Args:
        event_bus: EventBus 客户端
        mock_mode: 是否使用 Mock 模式
        report_rate: 报告频率（Hz）
    
    Returns:
        OpenClawVisionReporter: 视觉报告器实例
    """
    config = VisionConfig(
        mock_mode=mock_mode,
        report_rate=report_rate
    )
    reporter = OpenClawVisionReporter(event_bus, config)
    return reporter
