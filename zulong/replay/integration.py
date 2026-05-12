"""
事件复盘机制集成模块
将复盘机制与祖龙系统的 EventBus、StateManager 等核心组件集成
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from zulong.core.types import (
    ZulongEvent, EventType, EventPriority,
    TaskStatus, TaskSnapshot
)
from zulong.core.event_bus import EventBus
from zulong.core.state_manager import state_manager

from zulong.replay.clock_synchronizer import get_unified_timestamp
from zulong.replay.ring_buffer import get_ring_buffer
from zulong.replay.l0_logger import get_l0_logger
from zulong.replay.l1a_logger import get_l1a_logger
from zulong.replay.context_snapshot import get_snapshot_manager
from zulong.replay.dossier_serializer import get_serializer
from zulong.replay.attributor import get_attribution_engine, AttributionResult
from zulong.replay.patch_compiler import get_patch_compiler, SystemPatch
from zulong.replay.experience_store import get_experience_store
from zulong.replay.calibration_manager import get_calibration_manager

logger = logging.getLogger(__name__)


@dataclass
class ReplayContext:
    """复盘上下文"""
    task_id: str
    event_id: str
    failure_reason: str
    task_type: str
    predicted_trajectory: Optional[Dict] = None
    actual_trajectory: Optional[Dict] = None
    context_data: Optional[Dict] = None


class ReplayIntegration:
    """
    事件复盘机制集成器
    
    负责:
    1. 监听任务失败事件，触发复盘流程
    2. 协调各复盘组件的工作流
    3. 发布复盘相关事件到 EventBus
    4. 管理经验注入和参数热更新
    """
    
    def __init__(self):
        self.event_bus = EventBus()
        self.ring_buffer = get_ring_buffer()
        self.l0_logger = get_l0_logger()
        self.l1a_logger = get_l1a_logger()
        self.snapshot_manager = get_snapshot_manager()
        self.serializer = get_serializer()
        self.attribution_engine = get_attribution_engine()
        self.patch_compiler = get_patch_compiler()
        self.experience_store = get_experience_store()
        self.calibration_manager = get_calibration_manager()
        
        self._pending_replays: Dict[str, ReplayContext] = {}
        self._stats = {
            "replays_triggered": 0,
            "dossiers_created": 0,
            "attributions_made": 0,
            "patches_generated": 0,
            "patches_applied": 0
        }
        
        self._subscribe_events()
        logger.info("[ReplayIntegration] 初始化完成")
    
    def _subscribe_events(self):
        """订阅相关事件"""
        self.event_bus.subscribe(
            EventType.TASK_COMPLETED,
            self._on_task_completed,
            "ReplayIntegration"
        )
        
        self.event_bus.subscribe(
            EventType.REPLAY_TRIGGERED,
            self._on_replay_triggered,
            "ReplayIntegration"
        )
        
        logger.debug("[ReplayIntegration] 事件订阅完成")
    
    def _on_task_completed(self, event: ZulongEvent):
        """任务完成事件处理"""
        payload = event.payload
        task_id = payload.get("task_id", "")
        success = payload.get("success", True)
        
        if not success:
            self._stats["replays_triggered"] += 1
            
            replay_event = ZulongEvent(
                type=EventType.REPLAY_TRIGGERED,
                source="ReplayIntegration",
                payload={
                    "task_id": task_id,
                    "event_id": f"replay_{get_unified_timestamp():.0f}",
                    "failure_reason": payload.get("error", "unknown"),
                    "task_type": payload.get("task_type", "unknown")
                },
                priority=EventPriority.LOW
            )
            self.event_bus.publish(replay_event)
            logger.info(f"[ReplayIntegration] 任务失败，触发复盘: {task_id}")
    
    def _on_replay_triggered(self, event: ZulongEvent):
        """复盘触发事件处理"""
        asyncio.create_task(self._execute_replay(event.payload))
    
    async def _execute_replay(self, payload: Dict[str, Any]):
        """
        执行复盘流程
        
        六阶段原子任务:
        1. 数据采集 (已完成，通过日志器)
        2. 事件档案创建
        3. 轨迹对齐
        4. 归因分析
        5. System_Patch 生成
        6. 经验注入
        """
        task_id = payload.get("task_id", "")
        event_id = payload.get("event_id", "")
        failure_reason = payload.get("failure_reason", "")
        task_type = payload.get("task_type", "")
        
        logger.info(f"[ReplayIntegration] 开始复盘: {task_id}")
        
        try:
            dossier = await self._create_dossier(task_id, event_id)
            if dossier:
                self._stats["dossiers_created"] += 1
                self._publish_event(EventType.REPLAY_DOSSIER_CREATED, {
                    "dossier_id": dossier.metadata.dossier_id,
                    "task_id": task_id
                })
            
            attribution = await self._run_attribution(dossier, payload)
            if attribution:
                self._stats["attributions_made"] += 1
                self._publish_event(EventType.REPLAY_ATTRIBUTION_DONE, {
                    "task_id": task_id,
                    "fault_layer": attribution.fault_layer.value,
                    "confidence": attribution.confidence
                })
            
            patch = await self._generate_patch(attribution, task_id, event_id)
            if patch:
                self._stats["patches_generated"] += 1
                self._publish_event(EventType.REPLAY_PATCH_GENERATED, {
                    "patch_id": patch.patch_id,
                    "condition": patch.condition,
                    "adjustment": patch.adjustment
                })
            
            if patch:
                applied = await self._apply_patch(patch)
                if applied:
                    self._stats["patches_applied"] += 1
                    self._publish_event(EventType.REPLAY_PATCH_APPLIED, {
                        "patch_id": patch.patch_id
                    })
            
            logger.info(f"[ReplayIntegration] 复盘完成: {task_id}")
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 复盘失败: {e}")
    
    async def _create_dossier(self, task_id: str, event_id: str):
        """创建事件档案"""
        slots = list(self.ring_buffer._slots)
        
        if not slots:
            logger.warning("[ReplayIntegration] 无可用数据创建档案")
            return None
        
        dossier = self.serializer.create_dossier(
            event_id=event_id,
            task_id=task_id,
            slots=slots,
            result="failure"
        )
        
        file_path = self.serializer.save_dossier(dossier, compress=True)
        logger.info(f"[ReplayIntegration] 档案已保存: {file_path}")
        
        return dossier
    
    async def _run_attribution(self, dossier, payload: Dict[str, Any]):
        """运行归因分析"""
        from zulong.replay.attributor import Trajectory, TrajectoryPoint
        
        predicted = Trajectory(source="L2", dimension=3)
        actual = Trajectory(source="L0", dimension=3)
        
        base_time = get_unified_timestamp()
        
        predicted_data = payload.get("predicted_trajectory", {})
        for i, pos in enumerate(predicted_data.get("positions", [])):
            predicted.add_point(TrajectoryPoint(
                timestamp=base_time + i * 10000,
                position=pos if isinstance(pos, list) else [pos, 0.0, 0.0]
            ))
        
        actual_data = payload.get("actual_trajectory", {})
        for i, pos in enumerate(actual_data.get("positions", [])):
            actual.add_point(TrajectoryPoint(
                timestamp=base_time + i * 10000 + 5000,
                position=pos if isinstance(pos, list) else [pos, 0.0, 0.0]
            ))
        
        if len(predicted.points) == 0 or len(actual.points) == 0:
            logger.warning("[ReplayIntegration] 轨迹数据不足，跳过归因")
            return None
        
        _, _, attribution = self.attribution_engine.analyze_and_attribute(predicted, actual)
        return attribution
    
    async def _generate_patch(self, attribution, task_id: str, event_id: str):
        """生成 System_Patch"""
        if not attribution:
            return None
        
        patch = self.patch_compiler.compile(
            attribution=attribution,
            scene_features=[task_id],
            context={"task_type": task_id},
            source_event_id=event_id
        )
        
        self.patch_compiler.validate_patch(patch)
        self.patch_compiler.activate_patch(patch, expires_in_seconds=86400)
        
        entry_id = await self.experience_store.store(patch)
        logger.info(f"[ReplayIntegration] 经验已存储: {entry_id}")
        
        self._publish_event(EventType.EXPERIENCE_STORED, {
            "entry_id": entry_id,
            "patch_id": patch.patch_id
        })
        
        return patch
    
    async def _apply_patch(self, patch: SystemPatch) -> bool:
        """应用补丁"""
        success = await self.calibration_manager.apply_patch(patch)
        
        if success:
            self._publish_event(EventType.CALIBRATION_APPLIED, {
                "patch_id": patch.patch_id,
                "params": self.calibration_manager.get_all_params()
            })
        else:
            self._publish_event(EventType.CALIBRATION_FAILED, {
                "patch_id": patch.patch_id
            })
        
        return success
    
    def _publish_event(self, event_type: EventType, payload: Dict[str, Any]):
        """发布事件"""
        event = ZulongEvent(
            type=event_type,
            source="ReplayIntegration",
            payload=payload,
            priority=EventPriority.LOW
        )
        self.event_bus.publish(event)
    
    def trigger_manual_replay(
        self,
        task_id: str,
        failure_reason: str,
        task_type: str = "manual",
        predicted_trajectory: Optional[Dict] = None,
        actual_trajectory: Optional[Dict] = None,
        context_data: Optional[Dict] = None
    ):
        """
        手动触发复盘
        
        Args:
            task_id: 任务ID
            failure_reason: 失败原因
            task_type: 任务类型
            predicted_trajectory: 预测轨迹
            actual_trajectory: 实际轨迹
            context_data: 上下文数据
        """
        event = ZulongEvent(
            type=EventType.REPLAY_TRIGGERED,
            source="ManualTrigger",
            payload={
                "task_id": task_id,
                "event_id": f"manual_replay_{get_unified_timestamp():.0f}",
                "failure_reason": failure_reason,
                "task_type": task_type,
                "predicted_trajectory": predicted_trajectory or {},
                "actual_trajectory": actual_trajectory or {},
                "context_data": context_data or {}
            },
            priority=EventPriority.NORMAL
        )
        self.event_bus.publish(event)
        logger.info(f"[ReplayIntegration] 手动触发复盘: {task_id}")
    
    async def get_applicable_patches(self, scene_tags: List[str]) -> List[SystemPatch]:
        """获取适用的补丁"""
        return await self.experience_store.get_applicable_patches(scene_tags)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "replay_stats": self._stats.copy(),
            "ring_buffer_stats": self.ring_buffer.get_statistics(),
            "experience_stats": self.experience_store.get_statistics(),
            "calibration_stats": self.calibration_manager.get_statistics()
        }


_global_replay_integration: Optional[ReplayIntegration] = None


def get_replay_integration() -> ReplayIntegration:
    """获取全局复盘集成器"""
    global _global_replay_integration
    if _global_replay_integration is None:
        _global_replay_integration = ReplayIntegration()
    return _global_replay_integration
