"""
AT-14/17: 复盘编译器与System_Patch
将数值差异转化为自然语言/结构化的"修正逻辑文本"
将复盘结果生成标准的System_Patch文本块
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import logging
import json
import uuid
from datetime import datetime

from .clock_synchronizer import get_unified_timestamp
from .attributor import AttributionResult, FaultLayer, AdjustmentType

logger = logging.getLogger(__name__)


class PatchStatus(Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


@dataclass
class SystemPatch:
    """
    系统补丁 (AT-17)
    
    包含修正逻辑的结构化文本块
    """
    patch_id: str
    scene_features: List[str]
    condition: str
    adjustment: str
    confidence: float
    created_at: float
    source_event_id: str
    status: PatchStatus = PatchStatus.DRAFT
    application_count: int = 0
    success_count: int = 0
    success_rate: float = 0.0
    fault_layer: str = ""
    adjustment_type: str = ""
    delta_t: float = 0.0
    trajectory_error: float = 0.0
    expires_at: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "scene_features": self.scene_features,
            "condition": self.condition,
            "adjustment": self.adjustment,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "source_event_id": self.source_event_id,
            "status": self.status.value,
            "application_count": self.application_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "fault_layer": self.fault_layer,
            "adjustment_type": self.adjustment_type,
            "delta_t": self.delta_t,
            "trajectory_error": self.trajectory_error,
            "expires_at": self.expires_at,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemPatch':
        return cls(
            patch_id=data["patch_id"],
            scene_features=data["scene_features"],
            condition=data["condition"],
            adjustment=data["adjustment"],
            confidence=data["confidence"],
            created_at=data["created_at"],
            source_event_id=data["source_event_id"],
            status=PatchStatus(data.get("status", "draft")),
            application_count=data.get("application_count", 0),
            success_count=data.get("success_count", 0),
            success_rate=data.get("success_rate", 0.0),
            fault_layer=data.get("fault_layer", ""),
            adjustment_type=data.get("adjustment_type", ""),
            delta_t=data.get("delta_t", 0.0),
            trajectory_error=data.get("trajectory_error", 0.0),
            expires_at=data.get("expires_at"),
            tags=data.get("tags", [])
        )
    
    def record_application(self, success: bool):
        """记录应用结果"""
        self.application_count += 1
        if success:
            self.success_count += 1
        self.success_rate = self.success_count / self.application_count
    
    def is_applicable(self, scene_tags: List[str]) -> bool:
        """判断是否适用于当前场景"""
        if self.status != PatchStatus.ACTIVE:
            return False
        if self.expires_at and get_unified_timestamp() > self.expires_at:
            return False
        return any(tag in self.scene_features for tag in scene_tags)


class PatchCompiler:
    """
    复盘编译器 (AT-14)
    
    将数值差异转化为自然语言/结构化的"修正逻辑文本"
    输出格式：IF [Condition] THEN [Adjustment]
    """
    
    def __init__(self):
        self._patch_counter = 0
        self._stats = {
            "patches_created": 0,
            "patches_validated": 0,
            "patches_activated": 0
        }
        logger.info("[PatchCompiler] 初始化完成")
    
    def generate_patch_id(self) -> str:
        """生成补丁ID"""
        self._patch_counter += 1
        return f"PATCH_{get_unified_timestamp():.0f}_{self._patch_counter}"
    
    def compile(
        self,
        attribution: AttributionResult,
        scene_features: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        source_event_id: str = ""
    ) -> SystemPatch:
        """
        编译归因结果为System_Patch
        
        Args:
            attribution: 归因结果
            scene_features: 场景特征标签
            context: 上下文信息
            source_event_id: 来源事件ID
        
        Returns:
            SystemPatch: 系统补丁
        """
        patch_id = self.generate_patch_id()
        
        condition = self._build_condition(attribution, context)
        adjustment = self._build_adjustment(attribution)
        
        if scene_features is None:
            scene_features = self._extract_scene_features(attribution, context)
        
        patch = SystemPatch(
            patch_id=patch_id,
            scene_features=scene_features,
            condition=condition,
            adjustment=adjustment,
            confidence=attribution.confidence,
            created_at=get_unified_timestamp(),
            source_event_id=source_event_id,
            status=PatchStatus.DRAFT,
            fault_layer=attribution.fault_layer.value,
            adjustment_type=attribution.adjustment_type.value,
            delta_t=attribution.delta_t,
            trajectory_error=attribution.trajectory_error
        )
        
        self._stats["patches_created"] += 1
        logger.info(f"[PatchCompiler] 补丁创建: {patch_id}, 置信度: {attribution.confidence:.2f}")
        
        return patch
    
    def _build_condition(
        self,
        attribution: AttributionResult,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """构建条件语句"""
        conditions = []
        
        if attribution.delta_t > 0:
            conditions.append(f"Delta_T > {attribution.delta_t/1000:.1f}ms")
        elif attribution.delta_t < 0:
            conditions.append(f"Delta_T < {attribution.delta_t/1000:.1f}ms")
        
        if attribution.trajectory_error > 0:
            conditions.append(f"Trajectory_Error > {attribution.trajectory_error:.4f}")
        
        if context:
            if "task_type" in context:
                conditions.append(f"Task_Type == '{context['task_type']}'")
            if "motion_type" in context:
                conditions.append(f"Motion_Type == '{context['motion_type']}'")
        
        if not conditions:
            return "TRUE"
        
        return " AND ".join(conditions)
    
    def _build_adjustment(self, attribution: AttributionResult) -> str:
        """构建调整语句"""
        adjustments = []
        
        if attribution.adjustment_type == AdjustmentType.INCREASE_SPEED_COMPENSATION:
            compensation_percent = min(20, abs(attribution.delta_t) / 1000)
            adjustments.append(f"L0.speed_compensation += {compensation_percent:.1f}%")
        
        elif attribution.adjustment_type == AdjustmentType.ADD_PREDICTION_DELAY:
            delay_ms = abs(attribution.delta_t) / 1000 * 1.1
            adjustments.append(f"L2.prediction_delay += {delay_ms:.1f}ms")
        
        elif attribution.adjustment_type == AdjustmentType.ADJUST_PID_PARAMS:
            adjustments.append("L3.PID_params = CALIBRATE()")
        
        for action in attribution.recommended_actions[:2]:
            adjustments.append(f"ACTION: {action}")
        
        return "; ".join(adjustments)
    
    def _extract_scene_features(
        self,
        attribution: AttributionResult,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """提取场景特征"""
        features = []
        
        if attribution.fault_layer == FaultLayer.L0_EXECUTOR:
            features.append("high_speed_motion")
            features.append("precision_task")
        elif attribution.fault_layer == FaultLayer.L2_CORTEX:
            features.append("complex_planning")
            features.append("multi_step_task")
        elif attribution.fault_layer == FaultLayer.L3_EXPERT:
            features.append("expert_skill_invocation")
        
        if context:
            if context.get("has_moving_target"):
                features.append("moving_target")
            if context.get("high_precision_required"):
                features.append("high_precision")
            if context.get("time_critical"):
                features.append("time_critical")
        
        return list(set(features))
    
    def compile_to_text(self, patch: SystemPatch) -> str:
        """
        编译为文本格式
        
        Args:
            patch: 系统补丁
        
        Returns:
            str: 文本格式
        """
        lines = [
            f"# System_Patch: {patch.patch_id}",
            f"# Created: {datetime.fromtimestamp(patch.created_at/1_000_000).isoformat()}",
            f"# Confidence: {patch.confidence:.2f}",
            f"# Source: {patch.source_event_id}",
            "",
            f"IF [{patch.condition}]",
            f"THEN [{patch.adjustment}]",
            "",
            f"APPLICABLE_SCENES: {json.dumps(patch.scene_features)}",
            f"FAULT_LAYER: {patch.fault_layer}",
            f"ADJUSTMENT_TYPE: {patch.adjustment_type}",
            "",
            f"# Application Stats: {patch.application_count} times, {patch.success_rate:.1%} success"
        ]
        return "\n".join(lines)
    
    def validate_patch(self, patch: SystemPatch) -> bool:
        """
        验证补丁有效性
        
        Args:
            patch: 系统补丁
        
        Returns:
            bool: 是否有效
        """
        if not patch.condition:
            return False
        if not patch.adjustment:
            return False
        if patch.confidence < 0.3:
            return False
        if not patch.scene_features:
            return False
        
        patch.status = PatchStatus.VALIDATED
        self._stats["patches_validated"] += 1
        return True
    
    def activate_patch(self, patch: SystemPatch, expires_in_seconds: Optional[float] = None):
        """
        激活补丁
        
        Args:
            patch: 系统补丁
            expires_in_seconds: 过期时间 (秒)
        """
        if patch.status != PatchStatus.VALIDATED:
            if not self.validate_patch(patch):
                return False
        
        patch.status = PatchStatus.ACTIVE
        if expires_in_seconds:
            patch.expires_at = get_unified_timestamp() + expires_in_seconds * 1_000_000
        
        self._stats["patches_activated"] += 1
        logger.info(f"[PatchCompiler] 补丁激活: {patch.patch_id}")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._stats.copy()


_global_patch_compiler: Optional[PatchCompiler] = None


def get_patch_compiler() -> PatchCompiler:
    """获取全局补丁编译器"""
    global _global_patch_compiler
    if _global_patch_compiler is None:
        _global_patch_compiler = PatchCompiler()
    return _global_patch_compiler
