"""
LLM自主注意力模式选择 - 配置管理模块
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class AttentionConfig:
    """注意力选择配置类"""
    enabled: bool = True                          # 功能开关
    pressure_threshold_high: float = 0.9          # 高压阈值
    pressure_threshold_medium: float = 0.75       # 中压阈值
    cooldown_base_seconds: float = 30.0           # 基础冷却时间(秒)
    fallback_mode: str = "FOCUS"                  # Fallback模式
    decision_timeout_ms: int = 500                # 决策超时(毫秒)
    oscillation_detection_window: int = 10        # 震荡检测窗口大小
    
    max_switch_history: int = 50                  # 最大切换历史记录数
    min_confidence_threshold: float = 0.3         # 最低置信度阈值
    
    @classmethod
    def load_from_yaml(cls, config_path: str = None) -> "AttentionConfig":
        """从YAML配置文件加载配置
        
        Args:
            config_path: 配置文件路径，默认为config/zulong_config.yaml
            
        Returns:
            AttentionConfig实例
        """
        if config_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            config_path = os.path.join(project_root, "config", "zulong_config.yaml")
        
        default_config = cls()
        
        if not os.path.exists(config_path):
            logger.warning(f"[AttentionConfig] 配置文件不存在: {config_path}，使用默认配置")
            return default_config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                logger.warning(f"[AttentionConfig] 配置文件为空，使用默认配置")
                return default_config
            
            attention_config = config_data.get("attention_selection", {})
            
            if not attention_config:
                logger.info(f"[AttentionConfig] 未找到attention_selection配置段，使用默认配置")
                return default_config
            
            config = cls(
                enabled=attention_config.get("enabled", default_config.enabled),
                pressure_threshold_high=attention_config.get("pressure_threshold_high", default_config.pressure_threshold_high),
                pressure_threshold_medium=attention_config.get("pressure_threshold_medium", default_config.pressure_threshold_medium),
                cooldown_base_seconds=attention_config.get("cooldown_base_seconds", default_config.cooldown_base_seconds),
                fallback_mode=attention_config.get("fallback_mode", default_config.fallback_mode),
                decision_timeout_ms=attention_config.get("decision_timeout_ms", default_config.decision_timeout_ms),
                oscillation_detection_window=attention_config.get("oscillation_detection_window", default_config.oscillation_detection_window),
                max_switch_history=attention_config.get("max_switch_history", default_config.max_switch_history),
                min_confidence_threshold=attention_config.get("min_confidence_threshold", default_config.min_confidence_threshold),
            )
            
            logger.info(f"[AttentionConfig] 配置加载成功: enabled={config.enabled}, "
                       f"pressure_threshold_high={config.pressure_threshold_high}")
            
            return config
            
        except Exception as e:
            logger.error(f"[AttentionConfig] 配置加载失败: {e}，使用默认配置")
            return default_config
    
    def validate(self) -> bool:
        """验证配置参数有效性并自动修正
        
        Returns:
            验证结果布尔值
        """
        is_valid = True
        
        if not (0.5 <= self.pressure_threshold_high <= 1.5):
            old_value = self.pressure_threshold_high
            self.pressure_threshold_high = max(0.5, min(1.5, self.pressure_threshold_high))
            logger.warning(f"[AttentionConfig] pressure_threshold_high超出范围[0.5, 1.5]，"
                          f"从{old_value}修正为{self.pressure_threshold_high}")
            is_valid = False
        
        if not (0.3 <= self.pressure_threshold_medium <= 1.0):
            old_value = self.pressure_threshold_medium
            self.pressure_threshold_medium = max(0.3, min(1.0, self.pressure_threshold_medium))
            logger.warning(f"[AttentionConfig] pressure_threshold_medium超出范围[0.3, 1.0]，"
                          f"从{old_value}修正为{self.pressure_threshold_medium}")
            is_valid = False
        
        if not (10.0 <= self.cooldown_base_seconds <= 300.0):
            old_value = self.cooldown_base_seconds
            self.cooldown_base_seconds = max(10.0, min(300.0, self.cooldown_base_seconds))
            logger.warning(f"[AttentionConfig] cooldown_base_seconds超出范围[10, 300]，"
                          f"从{old_value}修正为{self.cooldown_base_seconds}")
            is_valid = False
        
        if not (100 <= self.decision_timeout_ms <= 2000):
            old_value = self.decision_timeout_ms
            self.decision_timeout_ms = max(100, min(2000, self.decision_timeout_ms))
            logger.warning(f"[AttentionConfig] decision_timeout_ms超出范围[100, 2000]，"
                          f"从{old_value}修正为{self.decision_timeout_ms}")
            is_valid = False
        
        valid_modes = ["GLOBAL", "FOCUS", "SINGLE_CHAIN"]
        if self.fallback_mode not in valid_modes:
            old_value = self.fallback_mode
            self.fallback_mode = "FOCUS"
            logger.warning(f"[AttentionConfig] fallback_mode无效，从{old_value}修正为FOCUS")
            is_valid = False
        
        if is_valid:
            logger.info("[AttentionConfig] 配置验证通过")
        
        return is_valid
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为字典格式"""
        return {
            "enabled": self.enabled,
            "pressure_threshold_high": self.pressure_threshold_high,
            "pressure_threshold_medium": self.pressure_threshold_medium,
            "cooldown_base_seconds": self.cooldown_base_seconds,
            "fallback_mode": self.fallback_mode,
            "decision_timeout_ms": self.decision_timeout_ms,
            "oscillation_detection_window": self.oscillation_detection_window,
            "max_switch_history": self.max_switch_history,
            "min_confidence_threshold": self.min_confidence_threshold,
        }
