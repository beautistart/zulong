"""
LLM自主注意力模式选择 - 控制机制模块

包含冷却时间管理、震荡检测、模式切换控制器
"""
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from .attention_types import (
    SwitchRecord,
    OscillationState,
    OscillationLevel,
    TriggerType,
    CooldownCheckResult,
    ModeSwitchResult,
    DecisionResponse,
)
from .attention_config import AttentionConfig

logger = logging.getLogger(__name__)


class CooldownManager:
    """冷却时间管理器
    
    管理LLM自主选择的冷却时间，防止频繁触发
    """
    
    def __init__(self, config: AttentionConfig):
        """初始化冷却管理器
        
        Args:
            config: 配置对象
        """
        self._config = config
        self._last_switch_time: Optional[datetime] = None
        self._cooldown_factor: float = 1.0
    
    def check_cooldown(self) -> CooldownCheckResult:
        """检查是否处于冷却期
        
        Returns:
            CooldownCheckResult检查结果
        """
        if self._last_switch_time is None:
            return CooldownCheckResult(
                is_allowed=True,
                remaining_time=0.0,
                message="无上次切换记录，允许切换",
            )
        
        elapsed = (datetime.now() - self._last_switch_time).total_seconds()
        cooldown_seconds = self._config.cooldown_base_seconds * self._cooldown_factor
        remaining = cooldown_seconds - elapsed
        
        if remaining <= 0:
            return CooldownCheckResult(
                is_allowed=True,
                remaining_time=0.0,
                message=f"冷却期已过，允许切换",
            )
        else:
            return CooldownCheckResult(
                is_allowed=False,
                remaining_time=remaining,
                message=f"处于冷却期，剩余{remaining:.1f}秒",
            )
    
    def record_switch(self):
        """记录切换时间"""
        self._last_switch_time = datetime.now()
    
    def adjust_cooldown_factor(self, factor: float):
        """调整冷却因子
        
        Args:
            factor: 冷却因子乘数
        """
        self._cooldown_factor *= factor
        self._cooldown_factor = max(1.0, min(5.0, self._cooldown_factor))
        logger.info(f"[CooldownManager] 冷却因子调整为: {self._cooldown_factor:.2f}")
    
    def reset_cooldown_factor(self):
        """重置冷却因子"""
        self._cooldown_factor = 1.0
        logger.info("[CooldownManager] 冷却因子已重置为1.0")
    
    def get_current_cooldown_seconds(self) -> float:
        """获取当前冷却时间(秒)"""
        return self._config.cooldown_base_seconds * self._cooldown_factor


class OscillationDetector:
    """震荡检测器
    
    检测模式切换震荡，调整冷却策略
    """
    
    def __init__(self, config: AttentionConfig):
        """初始化震荡检测器
        
        Args:
            config: 配置对象
        """
        self._config = config
        self._oscillation_count: int = 0
    
    def detect_oscillation(self, history: List[SwitchRecord]) -> OscillationState:
        """检测震荡模式
        
        Args:
            history: 切换历史记录
            
        Returns:
            OscillationState震荡状态
        """
        if len(history) < 3:
            return OscillationState(
                is_oscillating=False,
                oscillation_level=OscillationLevel.NONE,
            )
        
        recent = history[-4:]
        
        aba_pattern = self._check_aba_pattern(recent)
        if aba_pattern:
            return OscillationState(
                is_oscillating=True,
                oscillation_level=OscillationLevel.OBVIOUS,
                oscillation_pattern="ABA",
                adjusted_cooldown_factor=1.5,
            )
        
        abab_pattern = self._check_abab_pattern(recent)
        if abab_pattern:
            return OscillationState(
                is_oscillating=True,
                oscillation_level=OscillationLevel.SEVERE,
                oscillation_pattern="ABAB",
                adjusted_cooldown_factor=2.0,
            )
        
        slight = self._check_frequent_switches(history)
        if slight:
            return OscillationState(
                is_oscillating=True,
                oscillation_level=OscillationLevel.SLIGHT,
                oscillation_pattern="frequent",
                adjusted_cooldown_factor=1.2,
            )
        
        return OscillationState(
            is_oscillating=False,
            oscillation_level=OscillationLevel.NONE,
        )
    
    def _check_aba_pattern(self, records: List[SwitchRecord]) -> bool:
        """检查ABA震荡模式
        
        Args:
            records: 切换记录列表
            
        Returns:
            是否检测到ABA模式
        """
        if len(records) < 3:
            return False
        
        modes = [r.new_mode for r in records[-3:]]
        return modes[0] == modes[2] and modes[0] != modes[1]
    
    def _check_abab_pattern(self, records: List[SwitchRecord]) -> bool:
        """检查ABAB震荡模式
        
        Args:
            records: 切换记录列表
            
        Returns:
            是否检测到ABAB模式
        """
        if len(records) < 4:
            return False
        
        modes = [r.new_mode for r in records[-4:]]
        return (modes[0] == modes[2] and modes[1] == modes[3] and 
                modes[0] != modes[1])
    
    def _check_frequent_switches(self, history: List[SwitchRecord]) -> bool:
        """检查频繁切换
        
        Args:
            history: 完整历史记录
            
        Returns:
            是否频繁切换
        """
        if len(history) < 5:
            return False
        
        recent = history[-5:]
        total_seconds = (recent[-1].timestamp - recent[0].timestamp).total_seconds()
        
        if total_seconds <= 0:
            return False
        
        switch_rate = len(recent) / total_seconds
        
        return switch_rate > 0.2


class ModeSwitchController:
    """模式切换控制器
    
    协调压力检测、LLM决策、冷却管理、震荡检测，执行模式切换
    """
    
    def __init__(
        self,
        config: AttentionConfig,
        cooldown_manager: CooldownManager,
        oscillation_detector: OscillationDetector,
    ):
        """初始化控制器
        
        Args:
            config: 配置对象
            cooldown_manager: 冷却管理器
            oscillation_detector: 震荡检测器
        """
        self._config = config
        self._cooldown_manager = cooldown_manager
        self._oscillation_detector = oscillation_detector
        self._switch_history: List[SwitchRecord] = []
    
    def can_switch(self) -> CooldownCheckResult:
        """检查是否允许切换
        
        Returns:
            CooldownCheckResult检查结果
        """
        return self._cooldown_manager.check_cooldown()
    
    def apply_llm_decision(
        self,
        current_mode: str,
        decision: DecisionResponse,
        pressure_value: float,
    ) -> ModeSwitchResult:
        """应用LLM决策，执行模式切换
        
        Args:
            current_mode: 当前模式
            decision: LLM决策响应
            pressure_value: 当前压力值
            
        Returns:
            ModeSwitchResult切换结果
        """
        new_mode = decision.mode
        
        if new_mode == current_mode:
            return ModeSwitchResult(
                success=True,
                new_mode=new_mode,
                switched=False,
                message="LLM选择保持当前模式",
            )
        
        if decision.is_fallback:
            logger.info(f"[ModeSwitchController] 使用Fallback模式: {new_mode}")
        
        trigger_type = TriggerType.FALLBACK if decision.is_fallback else TriggerType.LLM_AUTONOMOUS
        
        switch_record = SwitchRecord(
            old_mode=current_mode,
            new_mode=new_mode,
            trigger_type=trigger_type,
            pressure_at_switch=pressure_value,
            reason=decision.reason,
            confidence=decision.confidence,
        )
        
        self._switch_history.append(switch_record)
        if len(self._switch_history) > self._config.max_switch_history:
            self._switch_history.pop(0)
        
        self._cooldown_manager.record_switch()
        
        oscillation = self._oscillation_detector.detect_oscillation(self._switch_history)
        if oscillation.is_oscillating:
            self._cooldown_manager.adjust_cooldown_factor(
                oscillation.adjusted_cooldown_factor
            )
            logger.warning(
                f"[ModeSwitchController] 检测到震荡({oscillation.oscillation_pattern})，"
                f"冷却因子调整为{oscillation.adjusted_cooldown_factor:.2f}"
            )
        
        logger.info(
            f"[ModeSwitchController] 模式切换: {current_mode} → {new_mode} "
            f"(触发: {trigger_type.value}, 置信度: {decision.confidence:.2f})"
        )
        
        return ModeSwitchResult(
            success=True,
            new_mode=new_mode,
            switched=True,
            message=f"成功切换到{new_mode}模式",
            switch_record=switch_record,
        )
    
    def record_tool_driven_switch(
        self,
        old_mode: str,
        new_mode: str,
        pressure_value: float,
        reason: str = "",
    ):
        """记录工具驱动的模式切换
        
        Args:
            old_mode: 旧模式
            new_mode: 新模式
            pressure_value: 压力值
            reason: 切换原因
        """
        switch_record = SwitchRecord(
            old_mode=old_mode,
            new_mode=new_mode,
            trigger_type=TriggerType.TOOL_DRIVEN,
            pressure_at_switch=pressure_value,
            reason=reason,
        )
        
        self._switch_history.append(switch_record)
        if len(self._switch_history) > self._config.max_switch_history:
            self._switch_history.pop(0)
        
        logger.debug(f"[ModeSwitchController] 记录工具驱动切换: {old_mode} → {new_mode}")
    
    def get_switch_history(self, count: int = 10) -> List[SwitchRecord]:
        """获取切换历史
        
        Args:
            count: 获取数量
            
        Returns:
            切换记录列表
        """
        return self._switch_history[-count:]
    
    def clear_history(self):
        """清空切换历史"""
        self._switch_history.clear()
        logger.info("[ModeSwitchController] 切换历史已清空")
