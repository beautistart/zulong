# File: zulong/emotion/speech_emotion.py
# 语音情感分析器 (Phase 9.2)
# 基于声学特征 (音调、语速、音量) 的情感识别

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SpeechEmotionType(str, Enum):
    """语音情感类型"""
    CALM = "calm"           # 平静
    HAPPY = "happy"         # 高兴
    EXCITED = "excited"     # 兴奋
    ANGRY = "angry"         # 愤怒
    SAD = "sad"             # 悲伤
    FEARFUL = "fearful"     # 恐惧
    SURPRISED = "surprised" # 惊讶


@dataclass
class AcousticFeatures:
    """声学特征"""
    pitch_mean: float       # 平均音调 (Hz)
    pitch_std: float        # 音调标准差
    energy_mean: float      # 平均能量 (音量)
    energy_std: float       # 能量标准差
    speech_rate: float      # 语速 (音节/秒)
    duration: float         # 语音时长 (秒)
    pause_count: int        # 停顿次数
    pause_duration: float   # 总停顿时长 (秒)


@dataclass
class SpeechEmotionResult:
    """语音情感分析结果"""
    emotion: SpeechEmotionType
    confidence: float
    acoustic_features: Optional[AcousticFeatures] = None
    details: Dict[str, float] = None


class SpeechEmotionAnalyzer:
    """
    语音情感分析器
    
    功能:
    - 基于声学特征的情感识别
    - 音调分析 (pitch)
    - 音量分析 (energy)
    - 语速分析 (speech rate)
    - 停顿分析 (pause)
    
    使用示例:
    ```python
    analyzer = SpeechEmotionAnalyzer()
    
    # 分析声学特征
    features = AcousticFeatures(
        pitch_mean=200, pitch_std=50,
        energy_mean=0.8, energy_std=0.2,
        speech_rate=6.0, duration=3.0,
        pause_count=1, pause_duration=0.2
    )
    
    result = analyzer.analyze_features(features)
    print(result.emotion)  # happy 或 excited
    ```
    """
    
    def __init__(self):
        """初始化语音情感分析器"""
        # 情感规则 (基于声学特征的范围)
        self._emotion_rules = {
            SpeechEmotionType.CALM: {
                "pitch_range": (100, 200),
                "energy_range": (0.2, 0.5),
                "speech_rate_range": (3.0, 5.0),
                "pause_count_range": (0, 3)
            },
            SpeechEmotionType.HAPPY: {
                "pitch_range": (180, 300),
                "energy_range": (0.5, 0.8),
                "speech_rate_range": (5.0, 7.0),
                "pause_count_range": (0, 2)
            },
            SpeechEmotionType.EXCITED: {
                "pitch_range": (250, 400),
                "energy_range": (0.7, 1.0),
                "speech_rate_range": (6.0, 9.0),
                "pause_count_range": (0, 1)
            },
            SpeechEmotionType.ANGRY: {
                "pitch_range": (200, 350),
                "energy_range": (0.7, 1.0),
                "speech_rate_range": (5.0, 8.0),
                "pause_count_range": (0, 2)
            },
            SpeechEmotionType.SAD: {
                "pitch_range": (80, 150),
                "energy_range": (0.1, 0.4),
                "speech_rate_range": (2.0, 4.0),
                "pause_count_range": (2, 10)
            },
            SpeechEmotionType.FEARFUL: {
                "pitch_range": (250, 450),
                "energy_range": (0.5, 0.9),
                "speech_rate_range": (6.0, 10.0),
                "pause_count_range": (1, 5)
            },
            SpeechEmotionType.SURPRISED: {
                "pitch_range": (220, 380),
                "energy_range": (0.6, 0.9),
                "speech_rate_range": (4.0, 7.0),
                "pause_count_range": (1, 3)
            }
        }
        
        logger.info("[SpeechEmotionAnalyzer] 初始化完成")
    
    def analyze_features(self, features: AcousticFeatures) -> SpeechEmotionResult:
        """
        基于声学特征分析情感
        
        Args:
            features: 声学特征
            
        Returns:
            SpeechEmotionResult: 情感分析结果
        """
        scores = {}
        
        for emotion_type, rules in self._emotion_rules.items():
            score = self._calculate_emotion_score(features, rules)
            scores[emotion_type.value] = score
        
        # 找到主导情感
        dominant_emotion = max(scores, key=scores.get)
        dominant_score = scores[dominant_emotion]
        
        # 计算置信度
        total_score = sum(scores.values())
        confidence = dominant_score / total_score if total_score > 0 else 0.0
        
        return SpeechEmotionResult(
            emotion=SpeechEmotionType(dominant_emotion),
            confidence=min(confidence, 1.0),
            acoustic_features=features,
            details=scores
        )
    
    def _calculate_emotion_score(
        self,
        features: AcousticFeatures,
        rules: Dict[str, Tuple]
    ) -> float:
        """
        计算情感得分
        
        Args:
            features: 声学特征
            rules: 情感规则
            
        Returns:
            float: 情感得分 (0.0-1.0)
        """
        score = 0.0
        feature_count = 0
        
        # 音调匹配
        pitch_range = rules["pitch_range"]
        if pitch_range[0] <= features.pitch_mean <= pitch_range[1]:
            # 在范围内，计算距离中心的接近度
            center = (pitch_range[0] + pitch_range[1]) / 2
            range_width = (pitch_range[1] - pitch_range[0]) / 2
            pitch_score = 1.0 - abs(features.pitch_mean - center) / range_width
            score += max(pitch_score, 0.0)
            feature_count += 1
        
        # 音量匹配
        energy_range = rules["energy_range"]
        if energy_range[0] <= features.energy_mean <= energy_range[1]:
            center = (energy_range[0] + energy_range[1]) / 2
            range_width = (energy_range[1] - energy_range[0]) / 2
            energy_score = 1.0 - abs(features.energy_mean - center) / range_width
            score += max(energy_score, 0.0)
            feature_count += 1
        
        # 语速匹配
        speech_rate_range = rules["speech_rate_range"]
        if speech_rate_range[0] <= features.speech_rate <= speech_rate_range[1]:
            center = (speech_rate_range[0] + speech_rate_range[1]) / 2
            range_width = (speech_rate_range[1] - speech_rate_range[0]) / 2
            rate_score = 1.0 - abs(features.speech_rate - center) / range_width
            score += max(rate_score, 0.0)
            feature_count += 1
        
        # 停顿匹配
        pause_range = rules["pause_count_range"]
        if pause_range[0] <= features.pause_count <= pause_range[1]:
            center = (pause_range[0] + pause_range[1]) / 2
            range_width = (pause_range[1] - pause_range[0]) / 2 if pause_range[1] != pause_range[0] else 1
            pause_score = 1.0 - abs(features.pause_count - center) / range_width
            score += max(pause_score, 0.0)
            feature_count += 1
        
        # 平均得分
        return score / feature_count if feature_count > 0 else 0.0
    
    def analyze_audio_array(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> SpeechEmotionResult:
        """
        分析音频数组 (简化版)
        
        Args:
            audio_data: 音频数据 (numpy array)
            sample_rate: 采样率
            
        Returns:
            SpeechEmotionResult: 情感分析结果
        """
        # 提取简化声学特征
        features = self._extract_acoustic_features(audio_data, sample_rate)
        return self.analyze_features(features)
    
    def _extract_acoustic_features(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> AcousticFeatures:
        """
        从音频数据提取声学特征 (简化实现)
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            AcousticFeatures: 声学特征
        """
        # 计算时长
        duration = len(audio_data) / sample_rate
        
        # 计算音量 (RMS energy)
        energy_mean = np.sqrt(np.mean(audio_data ** 2))
        energy_std = np.std(audio_data)
        
        # 简化：使用过零率估计音调
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_data))))
        pitch_mean = zero_crossings / (2 * duration)  # 估计基频
        
        # 简化：语速估计 (假设每个音节约 0.1-0.2 秒)
        syllable_duration = 0.15
        speech_rate = duration / syllable_duration if duration > 0 else 0
        
        # 简化：停顿检测 (低能量段)
        threshold = energy_mean * 0.3
        pause_mask = np.abs(audio_data) < threshold
        pause_count = 0
        pause_duration = 0.0
        
        # 简单统计连续低能量段
        in_pause = False
        pause_start = 0
        for i, is_pause in enumerate(pause_mask):
            if is_pause and not in_pause:
                in_pause = True
                pause_start = i
            elif not is_pause and in_pause:
                in_pause = False
                pause_seg_duration = (i - pause_start) / sample_rate
                if pause_seg_duration > 0.1:  # 超过 0.1 秒才算停顿
                    pause_count += 1
                    pause_duration += pause_seg_duration
        
        return AcousticFeatures(
            pitch_mean=pitch_mean,
            pitch_std=np.std(audio_data) * 10,  # 简化估计
            energy_mean=energy_mean,
            energy_std=energy_std,
            speech_rate=speech_rate,
            duration=duration,
            pause_count=pause_count,
            pause_duration=pause_duration
        )
