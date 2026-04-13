# File: zulong/emotion/__init__.py
# 情感识别模块

from zulong.emotion.text_emotion import TextEmotionAnalyzer
from zulong.emotion.speech_emotion import SpeechEmotionAnalyzer
from zulong.emotion.emotion_response import EmotionResponseStrategy

__all__ = [
    "TextEmotionAnalyzer",
    "SpeechEmotionAnalyzer",
    "EmotionResponseStrategy"
]
