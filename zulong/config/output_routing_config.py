# File: zulong/config/output_routing_config.py
"""
输出路由配置 — 定义文本输出 vs 语音输出的决策规则

所有规则集中在此，修改路由行为只需改此文件，无需修改业务代码。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set


class OutputMode(str, Enum):
    """输出模式枚举 (替代硬编码字符串)"""
    TEXT_ONLY = "TEXT_ONLY"       # 仅文本输出到前端
    AUTO_TTS = "AUTO_TTS"         # 语音输入触发的自动语音回复
    FORCED_TTS = "FORCED_TTS"     # 用户明确要求语音回复


class OutputStyle(str, Enum):
    """TTS 语音风格枚举"""
    CONVERSATIONAL = "conversational"  # 自然对话
    EMPHATIC = "emphatic"              # 强调/正式


@dataclass
class OutputRoutingConfig:
    """
    输出路由配置
    
    所有决策规则集中在此，包括:
    - 触发语音输出的输入事件类型
    - 紧急状况自动切换规则
    - TTS 风格映射
    
    注意：FORCED_TTS 检测已移至 VoiceIntentClassifier（语义模型）
    """
    
    # ── 语音输入事件类型 ────────────────────────────────────
    # 以下事件类型来源的输入自动启用 AUTO_TTS
    voice_input_event_types: Set[str] = field(default_factory=lambda: {
        "USER_SPEECH",
        "USER_VOICE",
    })
    
    # ── 紧急状况强制语音 ────────────────────────────────────
    # 紧急状况时强制切换到此模式
    emergency_override_mode: OutputMode = OutputMode.AUTO_TTS
    
    # ── TTS 风格映射 ────────────────────────────────────────
    # 不同输出模式对应的 TTS 语音风格
    tts_style_map: dict = field(default_factory=lambda: {
        OutputMode.AUTO_TTS: OutputStyle.CONVERSATIONAL,
        OutputMode.FORCED_TTS: OutputStyle.EMPHATIC,
    })
    
    # ── 输出模式优先级定义 ──────────────────────────────────
    # 模式升序排列 (索引越大的优先级越高)
    mode_priority: List[OutputMode] = field(default_factory=lambda: [
        OutputMode.TEXT_ONLY,
        OutputMode.AUTO_TTS,
        OutputMode.FORCED_TTS,
    ])
    
    # ── 输出模式对应的 Action 事件 ──────────────────────────
    # True = 发布 ACTION_SPEAK, False = 不发布
    mode_triggers_speech: dict = field(default_factory=lambda: {
        OutputMode.TEXT_ONLY: False,
        OutputMode.AUTO_TTS: True,
        OutputMode.FORCED_TTS: True,
    })
    
    # ── 文本清洗开关 ────────────────────────────────────────
    # 启用后 clean_text_for_tts() 才会生效
    enable_text_cleaning: bool = True
    
    def __post_init__(self):
        """初始化后校验配置完整性"""
        # 确保所有 OutputMode 都被覆盖
        for mode in OutputMode:
            assert mode in self.mode_priority, f"OutputMode.{mode} 未在 mode_priority 中定义"
            assert mode in self.mode_triggers_speech, f"OutputMode.{mode} 未在 mode_triggers_speech 中定义"
    
    def is_voice_input_event(self, event_type: str) -> bool:
        """检测事件类型是否为语音输入"""
        return event_type in self.voice_input_event_types
    
    def should_trigger_speech(self, mode: OutputMode) -> bool:
        """判断某模式是否应该触发语音输出"""
        return self.mode_triggers_speech.get(mode, False)
    
    def get_tts_style(self, mode: OutputMode) -> str:
        """获取某输出模式对应的 TTS 风格"""
        return self.tts_style_map.get(mode, OutputStyle.CONVERSATIONAL).value
    
    def resolve_mode(
        self,
        text: str,
        event_type: str,
        is_emergency: bool = False,
    ) -> OutputMode:
        """
        根据输入解析输出模式（简化版）
        
        优先级: (紧急 → AUTO_TTS) > AUTO_TTS > TEXT_ONLY
        
        注意：FORCED_TTS 检测已移至 VoiceIntentClassifier（语义模型）
        
        Args:
            text: 用户输入文本（保留参数用于向后兼容）
            event_type: 事件类型字符串
            is_emergency: 是否紧急状态
        
        Returns:
            解析后的 OutputMode
        """
        # Condition A: 紧急状态
        if is_emergency:
            return self.emergency_override_mode
        
        # Condition B: 语音输入事件
        if self.is_voice_input_event(event_type):
            return OutputMode.AUTO_TTS
        
        # Default: 仅文本
        return OutputMode.TEXT_ONLY


# ── 全局单例 ──────────────────────────────────────────────

_output_routing_config: OutputRoutingConfig = None


def get_output_routing_config() -> OutputRoutingConfig:
    """获取输出路由配置单例"""
    global _output_routing_config
    if _output_routing_config is None:
        _output_routing_config = OutputRoutingConfig()
    return _output_routing_config
