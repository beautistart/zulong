# File: zulong/l1b/intent_filter.py
# 意图过滤器 - 分析用户输入的意图和优先级
# 使用 ALBERT-tiny Chinese 模型进行意图分类，关键词匹配作为 fallback
# 覆盖 L2 任务类型、视觉、音频、系统控制等意图

from zulong.core.types import EventPriority
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class IntentFilter:
    """意图过滤器 - ALBERT 模型 + 关键词 fallback"""

    # 意图类别到优先级的映射 (15类)
    INTENT_PRIORITY_MAP = {
        # 高优先级
        "command_stop": EventPriority.CRITICAL,
        "command_start": EventPriority.HIGH,
        "task_execute": EventPriority.HIGH,
        # 中优先级
        "task_code": EventPriority.NORMAL,
        "task_analysis": EventPriority.NORMAL,
        "task_write": EventPriority.NORMAL,
        "task_read": EventPriority.NORMAL,
        "task_search": EventPriority.NORMAL,
        "vision_control": EventPriority.NORMAL,
        "audio_control": EventPriority.NORMAL,
        "command_config": EventPriority.NORMAL,
        # 低优先级
        "vision_query": EventPriority.LOW,
        "audio_query": EventPriority.LOW,
        "chat": EventPriority.LOW,
        "unknown": EventPriority.LOW,
    }

    # 特殊意图的唤醒词标记
    WAKE_INTENTS = {"command_stop", "command_start"}

    def __init__(self, config: Optional[dict] = None):
        """
        初始化意图过滤器

        Args:
            config: 意图分类配置字典 (从 zulong_config.yaml 加载)
        """
        self.config = config or {}
        self._classifier = None
        self._albert_enabled = False
        self._confidence_threshold = 0.6

        # 初始化 ALBERT 分类器 (如果配置允许)
        self._init_albert_classifier()

    def _init_albert_classifier(self):
        """初始化 ALBERT 意图分类器"""
        albert_config = self.config.get("albert", {})
        if not albert_config:
            logger.info("[L1-B] 未配置 ALBERT，使用关键词匹配")
            return

        if not self.config.get("enabled", False):
            logger.info("[L1-B] 意图分类未启用，使用关键词匹配")
            return

        try:
            from zulong.models.albert_intent_classifier import AlbertIntentClassifier

            model_path = albert_config.get("model_path", "./models/albert-tiny-chinese")
            device = albert_config.get("device", "cpu")
            max_length = albert_config.get("max_length", 128)
            self._confidence_threshold = albert_config.get("confidence_threshold", 0.6)

            self._classifier = AlbertIntentClassifier(
                model_path=model_path,
                device=device,
                max_length=max_length,
            )

            if self._classifier.load():
                self._albert_enabled = True
                self._classifier.warmup()
                logger.info("[L1-B] ALBERT 意图分类器已启用 (15类)")
            else:
                logger.warning("[L1-B] ALBERT 模型加载失败，回退到关键词匹配")

        except ImportError as e:
            logger.warning(f"[L1-B] 缺少 transformers 依赖，使用关键词匹配: {e}")
        except Exception as e:
            logger.warning(f"[L1-B] ALBERT 初始化异常，使用关键词匹配: {e}")

    def _albert_classify(self, text: str) -> Optional[dict]:
        """
        使用 ALBERT 模型分类意图

        Args:
            text: 输入文本

        Returns:
            意图结果字典，或 None (如果置信度不足)
        """
        if not self._albert_enabled or not self._classifier:
            return None

        try:
            predicted_label, confidence, all_scores = self._classifier.predict(text)

            logger.debug(
                f"[L1-B] ALBERT 分类结果: {predicted_label} (置信度: {confidence:.3f})"
            )

            # 置信度不足，返回 None 让关键词匹配接手
            if confidence < self._confidence_threshold:
                logger.debug(
                    f"[L1-B] ALBERT 置信度不足 ({confidence:.3f} < {self._confidence_threshold})，使用关键词匹配"
                )
                return None

            # 转换为系统意图格式
            intent = predicted_label.upper()
            priority = self.INTENT_PRIORITY_MAP.get(predicted_label, EventPriority.NORMAL)
            is_wake = predicted_label in self.WAKE_INTENTS

            return {
                "intent": intent,
                "priority": priority,
                "is_wake_word": is_wake,
                "model": "albert",
                "confidence": confidence,
            }

        except Exception as e:
            logger.error(f"[L1-B] ALBERT 推理失败: {e}")
            return None

    def analyze(self, text: str) -> dict:
        """
        分析用户输入的意图

        使用 ALBERT 模型分类，不可用时返回 UNKNOWN

        Args:
            text: 用户输入文本

        Returns:
            dict: 包含 intent, priority, is_wake_word 的字典
        """
        text = text.strip()

        # 1. 优先尝试 ALBERT 模型分类
        albert_result = self._albert_classify(text)
        if albert_result is not None:
            return albert_result

        # 2. ALBERT 不可用或置信度不足，返回 UNKNOWN
        logger.debug("[L1-B] ALBERT 不可用，返回 UNKNOWN")
        return {
            'intent': 'UNKNOWN',
            'priority': EventPriority.LOW,
            'is_wake_word': False
        }
