"""
L3-Calibration 专家模块
夜间批量处理日志，生成 System_Patch
对应 TSD v2.1: 事件复盘机制 - L3 专家技能
"""
import asyncio
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, time as dt_time
from pathlib import Path

from zulong.l3.base_expert_node import BaseExpertNode, ExpertExecutionError
from zulong.core.types import EventType, ZulongEvent, EventPriority
from zulong.core.event_bus import event_bus

from zulong.replay.clock_synchronizer import get_unified_timestamp
from zulong.replay.ring_buffer import get_ring_buffer
from zulong.replay.dossier_serializer import get_serializer
from zulong.replay.attributor import get_attribution_engine, Trajectory, TrajectoryPoint
from zulong.replay.patch_compiler import get_patch_compiler
from zulong.replay.experience_store import get_experience_store
from zulong.replay.calibration_manager import get_calibration_manager

logger = logging.getLogger(__name__)


class CalibrationExpert(BaseExpertNode):
    """
    校准专家节点
    
    功能:
    1. 夜间批量处理日志文件
    2. 分析历史任务失败模式
    3. 生成 System_Patch
    4. 应用参数校准
    
    TSD v2.1 对应规则:
    - 事件复盘机制的 L3 专家技能
    - 夜间批量处理模式
    """
    
    def __init__(self):
        super().__init__(expert_type="EXPERT_CALIBRATION")
        
        self.ring_buffer = get_ring_buffer()
        self.serializer = get_serializer()
        self.attribution_engine = get_attribution_engine()
        self.patch_compiler = get_patch_compiler()
        self.experience_store = get_experience_store()
        self.calibration_manager = get_calibration_manager()
        
        self._dossier_dir = Path("dossiers")
        self._dossier_dir.mkdir(exist_ok=True)
        
        self._night_mode_start = dt_time(2, 0)
        self._night_mode_end = dt_time(5, 0)
        self._is_night_mode = False
        self._batch_task: Optional[asyncio.Task] = None
        
        self._stats = {
            "batches_processed": 0,
            "dossiers_analyzed": 0,
            "patches_generated": 0,
            "patches_applied": 0
        }
        
        logger.info("[CalibrationExpert] 校准专家节点初始化完成")
    
    def execute(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """执行校准任务"""
        action = task_payload.get("action", "analyze")
        
        if action == "analyze":
            return self._execute_analysis(task_payload)
        elif action == "batch_process":
            return self._execute_batch_process(task_payload)
        elif action == "apply_patch":
            return self._execute_apply_patch(task_payload)
        elif action == "get_stats":
            return self._get_statistics()
        else:
            raise ExpertExecutionError(f"未知操作: {action}")
    
    def _execute_analysis(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个分析任务"""
        dossier_path = task_payload.get("dossier_path")
        
        if dossier_path:
            result = self._analyze_dossier(dossier_path)
            return {
                "status": "success",
                "analysis_result": result
            }
        
        slots = list(self.ring_buffer._slots)
        if not slots:
            return {
                "status": "error",
                "message": "无可用数据"
            }
        
        dossier = self.serializer.create_dossier(
            event_id=f"calibration_{get_unified_timestamp():.0f}",
            task_id="manual_calibration",
            slots=slots,
            result="analysis"
        )
        
        result = self._analyze_dossier_data(dossier)
        return {
            "status": "success",
            "analysis_result": result
        }
    
    def _execute_batch_process(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """执行批量处理"""
        force = task_payload.get("force", False)
        
        if not force and not self._is_night_time():
            return {
                "status": "skipped",
                "message": "非夜间时段，跳过批量处理"
            }
        
        results = asyncio.run(self._batch_process_dossiers())
        
        return {
            "status": "success",
            "batch_results": results,
            "stats": self._stats
        }
    
    def _execute_apply_patch(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """应用补丁"""
        patch_id = task_payload.get("patch_id")
        params = task_payload.get("params", {})
        
        if patch_id:
            success = asyncio.run(self._apply_patch_by_id(patch_id))
        else:
            success = asyncio.run(self.calibration_manager.apply_calibration(params))
        
        return {
            "status": "success" if success else "failed",
            "patch_id": patch_id,
            "params": params
        }
    
    def _get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "status": "success",
            "calibration_stats": self._stats,
            "experience_stats": self.experience_store.get_statistics(),
            "calibration_manager_stats": self.calibration_manager.get_statistics()
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """获取能力描述"""
        return {
            "expert_type": self.expert_type,
            "description": "校准专家 - 夜间批量处理日志，生成System_Patch",
            "actions": [
                {
                    "action": "analyze",
                    "description": "分析当前缓冲区数据或指定档案",
                    "params": ["dossier_path (optional)"]
                },
                {
                    "action": "batch_process",
                    "description": "批量处理所有待分析档案",
                    "params": ["force (bool)"]
                },
                {
                    "action": "apply_patch",
                    "description": "应用补丁或校准参数",
                    "params": ["patch_id", "params"]
                },
                {
                    "action": "get_stats",
                    "description": "获取统计信息",
                    "params": []
                }
            ],
            "night_mode": {
                "start": str(self._night_mode_start),
                "end": str(self._night_mode_end)
            }
        }
    
    def _is_night_time(self) -> bool:
        """判断是否为夜间时段"""
        now = datetime.now().time()
        if self._night_mode_start < self._night_mode_end:
            return self._night_mode_start <= now < self._night_mode_end
        else:
            return now >= self._night_mode_start or now < self._night_mode_end
    
    def start_night_mode(self):
        """启动夜间模式"""
        if self._is_night_mode:
            return
        
        self._is_night_mode = True
        self._batch_task = asyncio.create_task(self._night_batch_loop())
        logger.info("[CalibrationExpert] 夜间模式启动")
    
    def stop_night_mode(self):
        """停止夜间模式"""
        self._is_night_mode = False
        if self._batch_task:
            self._batch_task.cancel()
            self._batch_task = None
        logger.info("[CalibrationExpert] 夜间模式停止")
    
    async def _night_batch_loop(self):
        """夜间批量处理循环"""
        while self._is_night_mode:
            try:
                await self._batch_process_dossiers()
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[CalibrationExpert] 批量处理错误: {e}")
                await asyncio.sleep(300)
    
    async def _batch_process_dossiers(self) -> List[Dict[str, Any]]:
        """批量处理档案"""
        results = []
        self._stats["batches_processed"] += 1
        
        dossier_files = list(self._dossier_dir.glob("*.dossier.json"))
        dossier_files.extend(self._dossier_dir.glob("*.dossier.json.gz"))
        
        for dossier_file in dossier_files[:10]:
            try:
                result = self._analyze_dossier(str(dossier_file))
                results.append(result)
                self._stats["dossiers_analyzed"] += 1
            except Exception as e:
                logger.error(f"[CalibrationExpert] 档案分析失败 {dossier_file}: {e}")
        
        return results
    
    def _analyze_dossier(self, dossier_path: str) -> Dict[str, Any]:
        """分析档案"""
        dossier = self.serializer.load_dossier(dossier_path)
        return self._analyze_dossier_data(dossier)
    
    def _analyze_dossier_data(self, dossier) -> Dict[str, Any]:
        """分析档案数据"""
        predicted = Trajectory(source="L2", dimension=3)
        actual = Trajectory(source="L0", dimension=3)
        
        base_time = get_unified_timestamp()
        
        for i, slot in enumerate(dossier.slots[:20]):
            if slot.layer == "L2":
                predicted.add_point(TrajectoryPoint(
                    timestamp=slot.timestamp,
                    position=slot.data.get("predicted_position", [float(i), 0.0, 0.0])
                ))
            elif slot.layer == "L0":
                actual.add_point(TrajectoryPoint(
                    timestamp=slot.timestamp,
                    position=slot.data.get("actual_position", [float(i) + 0.01, 0.0, 0.0])
                ))
        
        if len(predicted.points) < 2 or len(actual.points) < 2:
            return {
                "status": "skipped",
                "reason": "轨迹数据不足"
            }
        
        _, _, attribution = self.attribution_engine.analyze_and_attribute(predicted, actual)
        
        patch = self.patch_compiler.compile(
            attribution=attribution,
            scene_features=["batch_analysis"],
            context={"dossier_id": dossier.metadata.dossier_id},
            source_event_id=dossier.metadata.event_id
        )
        
        self._stats["patches_generated"] += 1
        
        self.patch_compiler.validate_patch(patch)
        self.patch_compiler.activate_patch(patch, expires_in_seconds=86400 * 7)
        
        entry_id = asyncio.run(self.experience_store.store(patch))
        
        applied = asyncio.run(self.calibration_manager.apply_patch(patch))
        if applied:
            self._stats["patches_applied"] += 1
        
        self._publish_calibration_event(patch, applied)
        
        return {
            "status": "success",
            "dossier_id": dossier.metadata.dossier_id,
            "attribution": {
                "fault_layer": attribution.fault_layer.value,
                "confidence": attribution.confidence,
                "reasoning": attribution.reasoning
            },
            "patch_id": patch.patch_id,
            "experience_entry_id": entry_id,
            "applied": applied
        }
    
    async def _apply_patch_by_id(self, patch_id: str) -> bool:
        """根据ID应用补丁"""
        for entry in self.experience_store._entries.values():
            if entry.patch.patch_id == patch_id:
                return await self.calibration_manager.apply_patch(entry.patch)
        return False
    
    def _publish_calibration_event(self, patch, applied: bool):
        """发布校准事件"""
        event_type = EventType.CALIBRATION_APPLIED if applied else EventType.CALIBRATION_FAILED
        event = ZulongEvent(
            type=event_type,
            source="CalibrationExpert",
            payload={
                "patch_id": patch.patch_id,
                "condition": patch.condition,
                "adjustment": patch.adjustment,
                "confidence": patch.confidence
            },
            priority=EventPriority.LOW
        )
        event_bus.publish(event)


_global_calibration_expert: Optional[CalibrationExpert] = None


def get_calibration_expert() -> CalibrationExpert:
    """获取全局校准专家"""
    global _global_calibration_expert
    if _global_calibration_expert is None:
        _global_calibration_expert = CalibrationExpert()
    return _global_calibration_expert
