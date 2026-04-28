# File: zulong/emotion/emotion_response.py
# 情感响应策略 (Phase 9.2)
# 根据用户情感调整 AI 回复策略

import logging
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum

from zulong.emotion.text_emotion import EmotionResult, EmotionType
from zulong.emotion.speech_emotion import SpeechEmotionResult, SpeechEmotionType

logger = logging.getLogger(__name__)


class ResponseTone(str, Enum):
    """回复语气"""
    EMPATHETIC = "empathetic"     # 共情
    ENTHUSIASTIC = "enthusiastic" # 热情
    CALM = "calm"                 # 平静
    SUPPORTIVE = "supportive"     # 支持
    HUMOROUS = "humorous"         # 幽默
    PROFESSIONAL = "professional" # 专业
    GENTLE = "gentle"             # 温柔


@dataclass
class ResponseStrategy:
    """响应策略"""
    tone: ResponseTone                    # 语气
    empathy_level: float                  # 共情程度 0.0-1.0
    use_emojis: bool                      # 是否使用表情
    response_length: str                  # 回复长度: short, medium, long
    special_instructions: Optional[str] = None  # 特殊指令


class EmotionResponseStrategy:
    """
    情感响应策略
    
    功能:
    - 根据用户情感选择合适的回复语气
    - 调整共情程度
    - 生成特殊指令 (如安慰、鼓励等)
    
    使用示例:
    ```python
    strategy = EmotionResponseStrategy()
    
    # 根据文本情感生成策略
    emotion_result = EmotionResult(
        emotion=EmotionType.SADNESS,
        confidence=0.8,
        intensity=0.7,
        details={}
    )
    
    response = strategy.generate(emotion_result)
    print(response.tone)  # empathetic
    print(response.empathy_level)  # 0.8
    ```
    """
    
    def __init__(self):
        """初始化响应策略"""
        # 情感到策略的映射
        self._emotion_strategy_map = {
            EmotionType.JOY: ResponseStrategy(
                tone=ResponseTone.ENTHUSIASTIC,
                empathy_level=0.3,
                use_emojis=True,
                response_length="medium",
                special_instructions="分享喜悦，适当祝贺"
            ),
            EmotionType.SADNESS: ResponseStrategy(
                tone=ResponseTone.EMPATHETIC,
                empathy_level=0.9,
                use_emojis=False,
                response_length="long",
                special_instructions="表达理解，提供安慰和支持建议"
            ),
            EmotionType.ANGER: ResponseStrategy(
                tone=ResponseTone.CALM,
                empathy_level=0.7,
                use_emojis=False,
                response_length="medium",
                special_instructions="保持冷静，避免激化，提供解决方案"
            ),
            EmotionType.FEAR: ResponseStrategy(
                tone=ResponseTone.SUPPORTIVE,
                empathy_level=0.8,
                use_emojis=False,
                response_length="medium",
                special_instructions="提供安全感，给予鼓励和支持"
            ),
            EmotionType.SURPRISE: ResponseStrategy(
                tone=ResponseTone.HUMOROUS,
                empathy_level=0.4,
                use_emojis=True,
                response_length="short",
                special_instructions="适当幽默，增加趣味性"
            ),
            EmotionType.DISGUST: ResponseStrategy(
                tone=ResponseTone.PROFESSIONAL,
                empathy_level=0.5,
                use_emojis=False,
                response_length="medium",
                special_instructions="客观分析，提供替代方案"
            ),
            EmotionType.NEUTRAL: ResponseStrategy(
                tone=ResponseTone.PROFESSIONAL,
                empathy_level=0.2,
                use_emojis=False,
                response_length="medium",
                special_instructions=None
            )
        }
        
        logger.info("[EmotionResponseStrategy] 初始化完成")
    
    def generate(
        self,
        text_emotion: Optional[EmotionResult] = None,
        speech_emotion: Optional[SpeechEmotionResult] = None
    ) -> ResponseStrategy:
        """
        生成响应策略
        
        Args:
            text_emotion: 文本情感分析结果
            speech_emotion: 语音情感分析结果
            
        Returns:
            ResponseStrategy: 响应策略
        """
        # 优先使用文本情感
        if text_emotion and text_emotion.confidence > 0.3:
            strategy = self._generate_from_text_emotion(text_emotion)
        elif speech_emotion and speech_emotion.confidence > 0.3:
            strategy = self._generate_from_speech_emotion(speech_emotion)
        else:
            # 默认策略
            strategy = self._emotion_strategy_map[EmotionType.NEUTRAL]
        
        return strategy
    
    def _generate_from_text_emotion(
        self,
        emotion_result: EmotionResult
    ) -> ResponseStrategy:
        """
        基于文本情感生成策略
        
        Args:
            emotion_result: 文本情感分析结果
            
        Returns:
            ResponseStrategy: 响应策略
        """
        base_strategy = self._emotion_strategy_map.get(
            emotion_result.emotion,
            self._emotion_strategy_map[EmotionType.NEUTRAL]
        )
        
        # 根据强度和置信度调整共情程度
        adjusted_empathy = base_strategy.empathy_level * emotion_result.intensity
        adjusted_empathy *= emotion_result.confidence
        adjusted_empathy = min(max(adjusted_empathy, 0.0), 1.0)
        
        # 创建调整后的策略
        adjusted_strategy = ResponseStrategy(
            tone=base_strategy.tone,
            empathy_level=adjusted_empathy,
            use_emojis=base_strategy.use_emojis and emotion_result.intensity > 0.5,
            response_length=base_strategy.response_length,
            special_instructions=base_strategy.special_instructions
        )
        
        return adjusted_strategy
    
    def _generate_from_speech_emotion(
        self,
        emotion_result: SpeechEmotionResult
    ) -> ResponseStrategy:
        """
        基于语音情感生成策略
        
        Args:
            emotion_result: 语音情感分析结果
            
        Returns:
            ResponseStrategy: 响应策略
        """
        # 映射语音情感到文本情感
        speech_to_text_map = {
            SpeechEmotionType.CALM: EmotionType.NEUTRAL,
            SpeechEmotionType.HAPPY: EmotionType.JOY,
            SpeechEmotionType.EXCITED: EmotionType.JOY,
            SpeechEmotionType.ANGRY: EmotionType.ANGER,
            SpeechEmotionType.SAD: EmotionType.SADNESS,
            SpeechEmotionType.FEARFUL: EmotionType.FEAR,
            SpeechEmotionType.SURPRISED: EmotionType.SURPRISE
        }
        
        text_emotion = speech_to_text_map.get(
            emotion_result.emotion,
            EmotionType.NEUTRAL
        )
        
        base_strategy = self._emotion_strategy_map.get(
            text_emotion,
            self._emotion_strategy_map[EmotionType.NEUTRAL]
        )
        
        # 根据置信度调整
        adjusted_empathy = base_strategy.empathy_level * emotion_result.confidence
        
        return ResponseStrategy(
            tone=base_strategy.tone,
            empathy_level=min(adjusted_empathy, 1.0),
            use_emojis=base_strategy.use_emojis,
            response_length=base_strategy.response_length,
            special_instructions=base_strategy.special_instructions
        )
    
    def generate_prompt_prefix(self, strategy: ResponseStrategy) -> str:
        """
        生成 Prompt 前缀 (用于指导 LLM 回复)
        
        Args:
            strategy: 响应策略
            
        Returns:
            str: Prompt 前缀
        """
        prefix_parts = []
        
        # 语气指令
        tone_instructions = {
            ResponseTone.EMPATHETIC: "请用共情的语气，表达理解",
            ResponseTone.ENTHUSIASTIC: "请用热情的语气，积极回应",
            ResponseTone.CALM: "请用平静的语气，保持理性",
            ResponseTone.SUPPORTIVE: "请用支持的语气，给予鼓励",
            ResponseTone.HUMOROUS: "请用幽默的语气，增加趣味性",
            ResponseTone.PROFESSIONAL: "请用专业的语气，客观分析",
            ResponseTone.GENTLE: "请用温柔的语气，温和回应"
        }
        
        if strategy.tone in tone_instructions:
            prefix_parts.append(tone_instructions[strategy.tone])
        
        # 共情程度
        if strategy.empathy_level > 0.7:
            prefix_parts.append("高度关注用户情感")
        elif strategy.empathy_level > 0.4:
            prefix_parts.append("适当关注用户情感")
        
        # 特殊指令
        if strategy.special_instructions:
            prefix_parts.append(f"特殊要求：{strategy.special_instructions}")
        
        # 表情使用
        if strategy.use_emojis:
            prefix_parts.append("可以适当使用表情符号")
        
        # 回复长度
        length_map = {
            "short": "简短回复 (1-2 句)",
            "medium": "中等长度回复 (3-5 句)",
            "long": "详细回复 (5 句以上)"
        }
        if strategy.response_length in length_map:
            prefix_parts.append(length_map[strategy.response_length])
        
        return " | ".join(prefix_parts)
