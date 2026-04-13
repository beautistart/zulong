# File: zulong/emotion/text_emotion.py
# 文本情感分析器 (Phase 9.2)
# 基于情感词典和规则的情感分析

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EmotionType(str, Enum):
    """情感类型"""
    JOY = "joy"               # 高兴
    SADNESS = "sadness"       # 悲伤
    ANGER = "anger"           # 愤怒
    FEAR = "fear"             # 恐惧
    SURPRISE = "surprise"     # 惊讶
    DISGUST = "disgust"       # 厌恶
    NEUTRAL = "neutral"       # 中性


@dataclass
class EmotionResult:
    """情感分析结果"""
    emotion: EmotionType
    confidence: float         # 置信度 0.0-1.0
    intensity: float          # 强度 0.0-1.0
    details: Dict[str, float] # 各情感维度得分


class TextEmotionAnalyzer:
    """
    文本情感分析器
    
    功能:
    - 基于情感词典的情感分析
    - 支持程度副词加权
    - 支持否定词反转
    - 支持复合情感检测
    
    使用示例:
    ```python
    analyzer = TextEmotionAnalyzer()
    result = analyzer.analyze("我非常开心！")
    print(result.emotion)  # joy
    print(result.confidence)  # 0.95
    ```
    """
    
    def __init__(self):
        """初始化情感分析器"""
        # 情感词典
        self._emotion_words = {
            EmotionType.JOY: [
                "开心", "快乐", "高兴", "高兴", "喜悦", "兴奋", "愉快", "愉悦",
                "欢喜", "喜欢", "爱", "满足", "幸福", "美好",
                "哈哈", "嘻嘻", "嘿嘿", "呵呵", "笑脸", "庆祝", "恭喜",
                "good", "great", "excellent", "happy", "joy", "wonderful",
                "amazing", "awesome", "fantastic", "love", "like"
            ],
            EmotionType.SADNESS: [
                "伤心", "难过", "悲伤", "悲痛", "痛苦", "哭泣", "哭", "流泪",
                "失落", "失望", "沮丧", "郁闷", "心烦", "忧愁", "忧虑", "担忧",
                "孤独", "孤单", "寂寞", "凄凉", "悲惨", "可怜",
                "sad", "cry", "depressed", "unhappy", "lonely", "miss",
                "呜呜", "唉", "心疼", "遗憾"
            ],
            EmotionType.ANGER: [
                "生气", "愤怒", "恼火", "烦躁", "烦", "怒火", "气愤", "愤恨",
                "不满", "抱怨", "讨厌", "恨", "仇", " rage", "暴怒",
                "angry", "mad", "furious", "hate", "annoyed", "pissed",
                "混蛋", "可恶", "该死", "什么鬼", "见鬼", "气死"
            ],
            EmotionType.FEAR: [
                "害怕", "恐惧", "惊吓", "惊恐", "恐慌", "慌张", "紧张",
                "担心", "担忧", "焦虑", "不安", "畏惧", "胆怯",
                "afraid", "scared", "fear", "terrified", "worried", "nervous",
                "吓死", "妈呀", "天啊", "救命", "危险", "恐怖"
            ],
            EmotionType.SURPRISE: [
                "惊讶", "惊奇", "震惊", "吃惊", "意外", "没想到", "意外",
                "奇怪", "疑惑", "不可思议", "出乎意料",
                "surprise", "shocked", "amazed", "astonished", "unexpected",
                "哇", "啊", "哦", "天哪", "不会吧", "真的吗", "竟然", "居然"
            ],
            EmotionType.DISGUST: [
                "恶心", "讨厌", "厌恶", "反感", "嫌弃", "憎恶", "鄙视",
                "disgust", "dislike", "gross", "yuck", "eww",
                "呕", "呸", "切", "哼", "不屑"
            ]
        }
        
        # 程度副词 (加权系数)
        self._degree_words = {
            "非常": 1.5,
            "很": 1.3,
            "太": 1.4,
            "极其": 1.6,
            "特别": 1.4,
            "十分": 1.4,
            "超级": 1.5,
            "有点": 0.7,
            "有些": 0.7,
            "稍微": 0.6,
            "略": 0.5,
            "微微": 0.6,
            "very": 1.5,
            "really": 1.3,
            "extremely": 1.6,
            "so": 1.3,
            "quite": 1.2,
            "a bit": 0.7,
            "slightly": 0.6
        }
        
        # 否定词
        self._negation_words = [
            "不", "没", "没有", "别", "不要", "不要", "未", "从未",
            "not", "no", "never", "don't", "doesn't", "didn't", "won't", "wouldn't"
        ]
        
        logger.info("[TextEmotionAnalyzer] 初始化完成")
    
    def analyze(self, text: str) -> EmotionResult:
        """
        分析文本情感
        
        Args:
            text: 输入文本
            
        Returns:
            EmotionResult: 情感分析结果
        """
        if not text or not text.strip():
            return EmotionResult(
                emotion=EmotionType.NEUTRAL,
                confidence=0.0,
                intensity=0.0,
                details={}
            )
        
        # 计算各情感维度得分
        scores = {}
        for emotion_type in EmotionType:
            if emotion_type == EmotionType.NEUTRAL:
                continue
            scores[emotion_type.value] = self._calculate_emotion_score(text, emotion_type)
        
        # 找到主导情感
        if not scores or max(scores.values()) == 0:
            return EmotionResult(
                emotion=EmotionType.NEUTRAL,
                confidence=0.0,
                intensity=0.0,
                details=scores
            )
        
        dominant_emotion = max(scores, key=scores.get)
        dominant_score = scores[dominant_emotion]
        
        # 计算置信度和强度
        total_score = sum(scores.values())
        confidence = dominant_score / total_score if total_score > 0 else 0.0
        intensity = min(dominant_score, 1.0)
        
        return EmotionResult(
            emotion=EmotionType(dominant_emotion),
            confidence=min(confidence, 1.0),
            intensity=intensity,
            details=scores
        )
    
    def _calculate_emotion_score(self, text: str, emotion_type: EmotionType) -> float:
        """
        计算特定情感的得分
        
        Args:
            text: 输入文本
            emotion_type: 情感类型
            
        Returns:
            float: 情感得分 (0.0-1.0)
        """
        score = 0.0
        text_lower = text.lower()
        words = self._tokenize(text)
        
        # 查找情感词
        for word in self._emotion_words[emotion_type]:
            word_lower = word.lower()
            if word_lower in text_lower:
                # 基础得分
                word_score = 1.0
                
                # 检查是否有程度副词修饰
                degree_multiplier = self._check_degree_word(text, word)
                word_score *= degree_multiplier
                
                # 检查是否有否定词
                if self._check_negation(text, word):
                    word_score *= -0.5  # 否定词反转并减弱
                
                score += word_score
        
        # 归一化到 0-1 范围
        if score > 0:
            return min(score / 5.0, 1.0)  # 假设最多5个情感词
        return 0.0
    
    def _tokenize(self, text: str) -> List[str]:
        """
        简单分词 (按字符和空格)
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 词列表
        """
        # 简单实现：按空格和标点分割
        words = re.split(r'[\s,;.!?，。；！？、]+', text)
        return [w for w in words if w]
    
    def _check_degree_word(self, text: str, emotion_word: str) -> float:
        """
        检查情感词前是否有程度副词
        
        Args:
            text: 完整文本
            emotion_word: 情感词
            
        Returns:
            float: 程度副词系数 (默认 1.0)
        """
        text_lower = text.lower()
        word_lower = emotion_word.lower()
        
        for degree_word, multiplier in self._degree_words.items():
            degree_lower = degree_word.lower()
            # 检查程度副词是否在情感词前
            pattern = f"{degree_lower}.*{word_lower}"
            if re.search(pattern, text_lower):
                return multiplier
        
        return 1.0
    
    def _check_negation(self, text: str, emotion_word: str) -> bool:
        """
        检查情感词前是否有否定词
        
        Args:
            text: 完整文本
            emotion_word: 情感词
            
        Returns:
            bool: 是否有否定词
        """
        text_lower = text.lower()
        word_lower = emotion_word.lower()
        
        # 找到情感词位置
        word_pos = text_lower.find(word_lower)
        if word_pos == -1:
            return False
        
        # 检查前面的词 (最多6个字符)
        preceding_text = text_lower[max(0, word_pos - 6):word_pos]
        
        for neg_word in self._negation_words:
            neg_lower = neg_word.lower()
            if neg_lower in preceding_text:
                return True
        
        return False
    
    def analyze_batch(self, texts: List[str]) -> List[EmotionResult]:
        """
        批量分析文本情感
        
        Args:
            texts: 文本列表
            
        Returns:
            List[EmotionResult]: 情感分析结果列表
        """
        return [self.analyze(text) for text in texts]
    
    def get_emotion_distribution(self, texts: List[str]) -> Dict[str, float]:
        """
        获取情感分布
        
        Args:
            texts: 文本列表
            
        Returns:
            Dict[str, float]: 情感分布 {emotion: percentage}
        """
        results = self.analyze_batch(texts)
        emotion_counts = {e.value: 0 for e in EmotionType}
        
        for result in results:
            emotion_counts[result.emotion.value] += 1
        
        total = len(texts) if texts else 1
        return {
            emotion: count / total
            for emotion, count in emotion_counts.items()
        }
