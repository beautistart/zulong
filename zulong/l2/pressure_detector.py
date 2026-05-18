"""
LLM自主注意力模式选择 - 压力检测模块
"""
from typing import List, Optional, TYPE_CHECKING
from datetime import datetime, timedelta
import time
import logging

from .attention_types import (
    PressureMetrics,
    PressureTrend,
    ThresholdCheckResult,
)
from .attention_config import AttentionConfig

if TYPE_CHECKING:
    from .attention_window import AttentionWindowManager

logger = logging.getLogger(__name__)


class PressureDetector:
    """压力检测器
    
    实时监测上下文压力，计算压力指标，判断是否触发LLM自主选择
    """
    
    def __init__(self, awm: "AttentionWindowManager", config: AttentionConfig):
        """初始化压力检测器
        
        Args:
            awm: AttentionWindowManager实例
            config: 配置对象
        """
        self._awm = awm
        self._config = config
        self._pressure_history: List[PressureMetrics] = []
        self._last_detection_time: Optional[datetime] = None
        self._max_history_size = 100
        
    def calculate_pressure(self) -> PressureMetrics:
        """计算当前上下文压力
        
        Returns:
            PressureMetrics压力指标对象
        """
        start_time = time.time()
        
        try:
            total_tokens = sum(env.tokens for env in self._awm.envelopes)
            current_budget = self._awm.budget
            
            if current_budget <= 0:
                pressure_value = 999.0
            else:
                pressure_value = total_tokens / current_budget
            
            velocity = self._calculate_velocity()
            predicted = self._predict_pressure(pressure_value, velocity)
            trend = self._determine_trend(pressure_value)
            
            budget_usage = total_tokens / current_budget if current_budget > 0 else 1.0
            message_count = len(self._awm.envelopes)
            token_density = total_tokens / max(message_count, 1)
            
            metrics = PressureMetrics(
                current_pressure=pressure_value,
                pressure_trend=trend,
                pressure_velocity=velocity,
                predicted_pressure_5s=predicted,
                budget_usage=budget_usage,
                message_count=message_count,
                token_density=token_density,
            )
            
            self._pressure_history.append(metrics)
            if len(self._pressure_history) > self._max_history_size:
                self._pressure_history.pop(0)
            
            self._last_detection_time = datetime.now()
            
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > 5:
                logger.warning(f"[PressureDetector] 压力检测耗时 {elapsed_ms:.2f}ms > 5ms")
            
            return metrics
            
        except Exception as e:
            logger.error(f"[PressureDetector] 压力计算失败: {e}")
            return PressureMetrics(
                current_pressure=999.0,
                pressure_trend=PressureTrend.STABLE,
                pressure_velocity=0.0,
                predicted_pressure_5s=999.0,
            )
    
    def _calculate_velocity(self) -> float:
        """计算压力变化速率
        
        Returns:
            压力变化速率 (单位: 压力值/秒)
        """
        if len(self._pressure_history) < 2:
            return 0.0
        
        recent = self._pressure_history[-2:]
        time_diff = (recent[1].timestamp - recent[0].timestamp).total_seconds()
        
        if time_diff <= 0:
            return 0.0
        
        pressure_diff = recent[1].current_pressure - recent[0].current_pressure
        velocity = pressure_diff / time_diff
        
        return velocity
    
    def _predict_pressure(self, current: float, velocity: float) -> float:
        """预测5秒后的压力值
        
        Args:
            current: 当前压力值
            velocity: 变化速率
            
        Returns:
            预测压力值
        """
        predicted = current + velocity * 5.0
        
        if predicted < 0:
            predicted = 0.0
        if predicted > 999.0:
            predicted = 999.0
        
        return predicted
    
    def _determine_trend(self, current_pressure: float) -> PressureTrend:
        """确定压力趋势
        
        Args:
            current_pressure: 当前压力值
            
        Returns:
            PressureTrend枚举值
        """
        if len(self._pressure_history) < 3:
            return PressureTrend.STABLE
        
        recent_pressures = [m.current_pressure for m in self._pressure_history[-3:]]
        avg_recent = sum(recent_pressures[-2:]) / 2
        avg_older = sum(recent_pressures[:2]) / 2
        
        threshold = 0.05
        
        if avg_recent > avg_older + threshold:
            return PressureTrend.RISING
        elif avg_recent < avg_older - threshold:
            return PressureTrend.FALLING
        else:
            return PressureTrend.STABLE
    
    def check_threshold(self, metrics: PressureMetrics) -> ThresholdCheckResult:
        """检查压力是否超过阈值
        
        Args:
            metrics: 压力指标
            
        Returns:
            ThresholdCheckResult检查结果
        """
        high_threshold = self._config.pressure_threshold_high
        medium_threshold = self._config.pressure_threshold_medium
        
        if metrics.current_pressure >= high_threshold:
            return ThresholdCheckResult(
                should_trigger=True,
                trigger_type="high_pressure",
                message=f"压力值{metrics.current_pressure:.3f}超过高压阈值{high_threshold}",
                pressure_value=metrics.current_pressure,
            )
        
        if metrics.current_pressure >= medium_threshold:
            if metrics.pressure_trend == PressureTrend.RISING:
                return ThresholdCheckResult(
                    should_trigger=True,
                    trigger_type="medium_pressure_rising",
                    message=f"压力值{metrics.current_pressure:.3f}超过中压阈值且趋势上升",
                    pressure_value=metrics.current_pressure,
                )
        
        if metrics.predicted_pressure_5s >= high_threshold:
            return ThresholdCheckResult(
                should_trigger=True,
                trigger_type="predicted_high",
                message=f"预测压力{metrics.predicted_pressure_5s:.3f}将超过高压阈值",
                pressure_value=metrics.current_pressure,
            )
        
        return ThresholdCheckResult(
            should_trigger=False,
            trigger_type="normal",
            message=f"压力值{metrics.current_pressure:.3f}在正常范围",
            pressure_value=metrics.current_pressure,
        )
    
    def get_pressure_trend_description(self, trend: PressureTrend) -> str:
        """获取压力趋势的中文描述
        
        Args:
            trend: 压力趋势枚举
            
        Returns:
            中文描述
        """
        descriptions = {
            PressureTrend.RISING: "上升中",
            PressureTrend.STABLE: "稳定",
            PressureTrend.FALLING: "下降中",
        }
        return descriptions.get(trend, "未知")
    
    def get_recent_metrics(self, count: int = 10) -> List[PressureMetrics]:
        """获取最近的压力指标记录
        
        Args:
            count: 获取数量
            
        Returns:
            压力指标列表
        """
        return self._pressure_history[-count:]
    
    def clear_history(self):
        """清空压力历史记录"""
        self._pressure_history.clear()
        logger.info("[PressureDetector] 压力历史记录已清空")
