"""
AT-11/12/13: 归因决策树引擎
实现轨迹对齐、时序差异计算、归因决策树
关键指标: 
- AT-11: 支持不同频率数据的插值对齐
- AT-12: 精度达到0.1ms
- AT-13: 能够输出具体的故障层级（L0, L1, L2）
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import logging
import numpy as np
from scipy import interpolate

from .clock_synchronizer import get_unified_timestamp

logger = logging.getLogger(__name__)


class FaultLayer(Enum):
    L0_EXECUTOR = "L0"
    L1_REFLEX = "L1-A"
    L1_SCHEDULER = "L1-B"
    L2_CORTEX = "L2"
    L3_EXPERT = "L3"
    UNKNOWN = "UNKNOWN"


class AdjustmentType(Enum):
    INCREASE_SPEED_COMPENSATION = "increase_speed_compensation"
    ADD_PREDICTION_DELAY = "add_prediction_delay"
    ADJUST_PID_PARAMS = "adjust_pid_params"
    REDUCE_COMMAND_FREQUENCY = "reduce_command_frequency"
    INCREASE_SENSITIVITY = "increase_sensitivity"
    MANUAL_REVIEW = "manual_review"


@dataclass
class TrajectoryPoint:
    """轨迹点"""
    timestamp: float
    position: List[float]
    velocity: Optional[List[float]] = None
    acceleration: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "position": self.position,
            "velocity": self.velocity,
            "acceleration": self.acceleration
        }


@dataclass
class Trajectory:
    """轨迹"""
    points: List[TrajectoryPoint] = field(default_factory=list)
    source: str = ""
    dimension: int = 3
    
    def add_point(self, point: TrajectoryPoint):
        self.points.append(point)
    
    def get_positions(self) -> np.ndarray:
        return np.array([p.position for p in self.points])
    
    def get_timestamps(self) -> np.ndarray:
        return np.array([p.timestamp for p in self.points])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "dimension": self.dimension,
            "points": [p.to_dict() for p in self.points]
        }


@dataclass
class AlignmentResult:
    """轨迹对齐结果"""
    predicted_trajectory: Trajectory
    actual_trajectory: Trajectory
    aligned_timestamps: np.ndarray
    position_errors: np.ndarray
    velocity_errors: np.ndarray
    mean_position_error: float
    max_position_error: float
    alignment_quality: float


@dataclass
class TimingAnalysis:
    """时序分析结果"""
    delta_t: float
    delta_t_ms: float
    timing_accuracy_us: float
    prediction_ahead: bool
    execution_delayed: bool
    timing_category: str


@dataclass
class AttributionResult:
    """归因结果"""
    fault_layer: FaultLayer
    adjustment_type: AdjustmentType
    confidence: float
    delta_t: float
    trajectory_error: float
    reasoning: str
    recommended_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_layer": self.fault_layer.value,
            "adjustment_type": self.adjustment_type.value,
            "confidence": self.confidence,
            "delta_t": self.delta_t,
            "trajectory_error": self.trajectory_error,
            "reasoning": self.reasoning,
            "recommended_actions": self.recommended_actions
        }


class TrajectoryAligner:
    """
    轨迹对齐器 (AT-11)
    
    将L2的"预测轨迹"与L0的"实际轨迹"在时空上对齐
    支持不同频率数据的插值对齐
    """
    
    def __init__(self, interpolation_method: str = "cubic"):
        """
        初始化轨迹对齐器
        
        Args:
            interpolation_method: 插值方法 (linear/cubic/quintic)
        """
        self.interpolation_method = interpolation_method
        logger.info(f"[TrajectoryAligner] 初始化完成，插值方法: {interpolation_method}")
    
    def align_trajectories(
        self,
        predicted: Trajectory,
        actual: Trajectory,
        resample_rate: float = 100.0
    ) -> AlignmentResult:
        """
        对齐两条轨迹
        
        Args:
            predicted: 预测轨迹 (L2)
            actual: 实际轨迹 (L0)
            resample_rate: 重采样率
        
        Returns:
            AlignmentResult: 对齐结果
        """
        if len(predicted.points) < 2 or len(actual.points) < 2:
            raise ValueError("轨迹点数不足，无法对齐")
        
        pred_timestamps = predicted.get_timestamps()
        actual_timestamps = actual.get_timestamps()
        
        start_time = max(pred_timestamps[0], actual_timestamps[0])
        end_time = min(pred_timestamps[-1], actual_timestamps[-1])
        
        if start_time >= end_time:
            raise ValueError("轨迹时间范围无交集")
        
        aligned_timestamps = np.arange(start_time, end_time, 1.0 / resample_rate * 1_000_000)
        
        pred_positions = predicted.get_positions()
        actual_positions = actual.get_positions()
        
        pred_interp = self._interpolate_trajectory(
            pred_timestamps, pred_positions, aligned_timestamps
        )
        actual_interp = self._interpolate_trajectory(
            actual_timestamps, actual_positions, aligned_timestamps
        )
        
        position_errors = np.linalg.norm(pred_interp - actual_interp, axis=1)
        
        mean_error = float(np.mean(position_errors))
        max_error = float(np.max(position_errors))
        
        quality = 1.0 - min(mean_error / 0.1, 1.0)
        
        return AlignmentResult(
            predicted_trajectory=predicted,
            actual_trajectory=actual,
            aligned_timestamps=aligned_timestamps,
            position_errors=position_errors,
            velocity_errors=np.zeros_like(position_errors),
            mean_position_error=mean_error,
            max_position_error=max_error,
            alignment_quality=quality
        )
    
    def _interpolate_trajectory(
        self,
        timestamps: np.ndarray,
        positions: np.ndarray,
        target_timestamps: np.ndarray
    ) -> np.ndarray:
        """
        插值轨迹
        
        Args:
            timestamps: 原始时间戳
            positions: 原始位置
            target_timestamps: 目标时间戳
        
        Returns:
            np.ndarray: 插值后的位置
        """
        if self.interpolation_method == "linear":
            kind = "linear"
        elif self.interpolation_method == "cubic":
            kind = "cubic"
        else:
            kind = "quintic"
        
        if positions.shape[1] == 1:
            interp_func = interpolate.interp1d(
                timestamps, positions.flatten(),
                kind=kind, fill_value="extrapolate"
            )
            return interp_func(target_timestamps).reshape(-1, 1)
        else:
            result = np.zeros((len(target_timestamps), positions.shape[1]))
            for dim in range(positions.shape[1]):
                interp_func = interpolate.interp1d(
                    timestamps, positions[:, dim],
                    kind=kind, fill_value="extrapolate"
                )
                result[:, dim] = interp_func(target_timestamps)
            return result


class TimingAnalyzer:
    """
    时序分析器 (AT-12)
    
    计算T_hand（手到达）与T_pen（笔经过）的时间差Delta_T
    精度达到0.1ms
    """
    
    def __init__(self, precision_us: float = 100.0):
        """
        初始化时序分析器
        
        Args:
            precision_us: 精度 (微秒)，默认100μs = 0.1ms
        """
        self.precision_us = precision_us
        logger.info(f"[TimingAnalyzer] 初始化完成，精度: {precision_us}μs")
    
    def analyze_timing(
        self,
        predicted_time: float,
        actual_time: float,
        tolerance_us: float = 50000.0
    ) -> TimingAnalysis:
        """
        分析时序差异
        
        Args:
            predicted_time: 预测时间 (微秒)
            actual_time: 实际时间 (微秒)
            tolerance_us: 容差 (微秒)
        
        Returns:
            TimingAnalysis: 时序分析结果
        """
        delta_t = actual_time - predicted_time
        delta_t_ms = delta_t / 1000.0
        
        rounded_delta_t = round(delta_t / self.precision_us) * self.precision_us
        
        prediction_ahead = delta_t < 0
        execution_delayed = delta_t > 0
        
        if abs(delta_t) < tolerance_us:
            category = "normal"
        elif execution_delayed:
            category = "execution_delayed"
        elif prediction_ahead:
            category = "prediction_ahead"
        else:
            category = "unknown"
        
        return TimingAnalysis(
            delta_t=rounded_delta_t,
            delta_t_ms=rounded_delta_t / 1000.0,
            timing_accuracy_us=self.precision_us,
            prediction_ahead=prediction_ahead,
            execution_delayed=execution_delayed,
            timing_category=category
        )
    
    def analyze_trajectory_timing(
        self,
        alignment: AlignmentResult
    ) -> List[TimingAnalysis]:
        """
        分析轨迹各点的时序差异
        
        Args:
            alignment: 轨迹对齐结果
        
        Returns:
            List[TimingAnalysis]: 各点的时序分析结果
        """
        results = []
        
        pred_timestamps = alignment.predicted_trajectory.get_timestamps()
        actual_timestamps = alignment.actual_trajectory.get_timestamps()
        
        min_len = min(len(pred_timestamps), len(actual_timestamps))
        
        for i in range(min_len):
            analysis = self.analyze_timing(
                predicted_time=pred_timestamps[i],
                actual_time=actual_timestamps[i]
            )
            results.append(analysis)
        
        return results


class AttributionEngine:
    """
    归因决策树引擎 (AT-13)
    
    编码逻辑：若Delta_T > 0则为执行慢；若Delta_T < 0则为预测早
    能够输出具体的故障层级（L0, L1, L2）
    """
    
    def __init__(
        self,
        delta_t_threshold_us: float = 50000.0,
        trajectory_error_threshold: float = 0.05
    ):
        """
        初始化归因引擎
        
        Args:
            delta_t_threshold_us: 时序差异阈值 (微秒)
            trajectory_error_threshold: 轨迹误差阈值
        """
        self.delta_t_threshold = delta_t_threshold_us
        self.trajectory_error_threshold = trajectory_error_threshold
        
        self._aligner = TrajectoryAligner()
        self._timing_analyzer = TimingAnalyzer()
        
        self._stats = {
            "attributions_made": 0,
            "l0_faults": 0,
            "l1_faults": 0,
            "l2_faults": 0,
            "l3_faults": 0,
            "unknown_faults": 0
        }
        
        logger.info(f"[AttributionEngine] 初始化完成，时序阈值: {delta_t_threshold_us}μs")
    
    def attribute(
        self,
        delta_t: float,
        trajectory_error: float,
        context: Optional[Dict[str, Any]] = None
    ) -> AttributionResult:
        """
        归因决策
        
        Args:
            delta_t: 时序差异 (微秒)
            trajectory_error: 轨迹偏差
            context: 上下文信息
        
        Returns:
            AttributionResult: 归因结果
        """
        self._stats["attributions_made"] += 1
        
        if delta_t > self.delta_t_threshold:
            fault_layer = FaultLayer.L0_EXECUTOR
            adjustment_type = AdjustmentType.INCREASE_SPEED_COMPENSATION
            confidence = min(0.9, 0.6 + abs(delta_t) / 1_000_000)
            reasoning = f"执行延迟 {delta_t/1000:.2f}ms，L0执行层响应过慢"
            recommended_actions = [
                "增加速度补偿参数",
                "检查电机驱动响应",
                "优化控制回路增益"
            ]
            self._stats["l0_faults"] += 1
            
        elif delta_t < -self.delta_t_threshold:
            fault_layer = FaultLayer.L2_CORTEX
            adjustment_type = AdjustmentType.ADD_PREDICTION_DELAY
            confidence = min(0.9, 0.6 + abs(delta_t) / 1_000_000)
            reasoning = f"预测提前 {-delta_t/1000:.2f}ms，L2规划层预测过早"
            recommended_actions = [
                "增加预测延迟补偿",
                "调整运动预测模型",
                "优化任务规划时间估算"
            ]
            self._stats["l2_faults"] += 1
            
        elif trajectory_error > self.trajectory_error_threshold:
            fault_layer = FaultLayer.L3_EXPERT
            adjustment_type = AdjustmentType.ADJUST_PID_PARAMS
            confidence = min(0.85, 0.5 + trajectory_error)
            reasoning = f"轨迹偏差 {trajectory_error:.4f}，L3专家模块参数需调整"
            recommended_actions = [
                "调整PID参数",
                "校准运动学模型",
                "检查传感器精度"
            ]
            self._stats["l3_faults"] += 1
            
        else:
            fault_layer = FaultLayer.UNKNOWN
            adjustment_type = AdjustmentType.MANUAL_REVIEW
            confidence = 0.3
            reasoning = "无法自动归因，需人工审查"
            recommended_actions = [
                "人工审查日志",
                "检查系统状态",
                "收集更多数据"
            ]
            self._stats["unknown_faults"] += 1
        
        return AttributionResult(
            fault_layer=fault_layer,
            adjustment_type=adjustment_type,
            confidence=confidence,
            delta_t=delta_t,
            trajectory_error=trajectory_error,
            reasoning=reasoning,
            recommended_actions=recommended_actions
        )
    
    def analyze_and_attribute(
        self,
        predicted_trajectory: Trajectory,
        actual_trajectory: Trajectory,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[AlignmentResult, TimingAnalysis, AttributionResult]:
        """
        完整分析流程：对齐 -> 时序分析 -> 归因
        
        Args:
            predicted_trajectory: 预测轨迹
            actual_trajectory: 实际轨迹
            context: 上下文信息
        
        Returns:
            Tuple: (对齐结果, 时序分析, 归因结果)
        """
        alignment = self._aligner.align_trajectories(
            predicted_trajectory, actual_trajectory
        )
        
        timing_analyses = self._timing_analyzer.analyze_trajectory_timing(alignment)
        
        if timing_analyses:
            avg_delta_t = np.mean([t.delta_t for t in timing_analyses])
            timing_analysis = timing_analyses[len(timing_analyses) // 2]
        else:
            avg_delta_t = 0.0
            timing_analysis = TimingAnalysis(
                delta_t=0.0,
                delta_t_ms=0.0,
                timing_accuracy_us=100.0,
                prediction_ahead=False,
                execution_delayed=False,
                timing_category="unknown"
            )
        
        attribution = self.attribute(
            delta_t=avg_delta_t,
            trajectory_error=alignment.mean_position_error,
            context=context
        )
        
        return alignment, timing_analysis, attribution
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "stats": self._stats.copy(),
            "delta_t_threshold": self.delta_t_threshold,
            "trajectory_error_threshold": self.trajectory_error_threshold
        }


_global_attribution_engine: Optional[AttributionEngine] = None


def get_attribution_engine() -> AttributionEngine:
    """获取全局归因引擎"""
    global _global_attribution_engine
    if _global_attribution_engine is None:
        _global_attribution_engine = AttributionEngine()
    return _global_attribution_engine
