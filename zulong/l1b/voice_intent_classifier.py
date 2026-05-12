"""
语音意图分类器 - 使用 ALBERT 语义模型检测用户是否要求语音输出

替代原有的硬编码关键词匹配方式，使用语义理解来判断用户意图：
- TEXT_ONLY: 用户仅需要文字回复
- AUTO_TTS: 隐式语音请求（如语音输入事件）
- FORCED_TTS: 用户明确要求语音回复（如"读给我听"、"用语音回答"）

架构：ALBERT 语义编码 + 独立分类头
- 复用 ALBERT-tiny-chinese 基础模型（与任务意图共享）
- 独立的 3 类线性分类头（768 → 3）
- 模型预测为主，规则仅作为安全降级

模型轻量 (~16MB)，适合 CPU 快速推理 (<50ms)。
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VoiceIntentClassifier:
    """语音意图分类器 - 3 类分类（TEXT_ONLY / AUTO_TTS / FORCED_TTS）"""

    # 意图类别定义（3 类）
    LABELS = [
        "TEXT_ONLY",   # 仅文字输出
        "AUTO_TTS",    # 自动语音回复（语音输入触发）
        "FORCED_TTS",  # 用户明确要求语音回复
    ]

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        max_length: int = 128,
        head_weights_path: str = "voice_intent_head.pt",
    ):
        """
        初始化语音意图分类器

        Args:
            model_path: ALBERT 基础模型路径（HuggingFace 格式）
            device: 推理设备 ('cpu' 或 'cuda')
            max_length: 最大输入长度
            head_weights_path: 分类头权重文件路径（相对于 model_path）
        """
        self.model_path = Path(model_path)
        self.device = device
        self.max_length = max_length
        self.head_weights_path = head_weights_path
        
        self._model = None
        self._tokenizer = None
        self._classification_head = None
        self._initialized = False

    def load(self) -> bool:
        """
        加载 ALBERT 基础模型和分类头

        Returns:
            是否加载成功
        """
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            # 验证模型路径
            if not self.model_path.exists():
                logger.error(f"[VoiceIntent] 模型路径不存在: {self.model_path}")
                return False

            config_file = self.model_path / "config.json"
            if not config_file.exists():
                logger.error(f"[VoiceIntent] 缺少 config.json: {config_file}")
                return False

            # 加载 ALBERT 基础模型
            logger.info(f"[VoiceIntent] 从本地加载 ALBERT 基础模型: {self.model_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self._model = AutoModel.from_pretrained(str(self.model_path))

            # 移动到设备并设置为评估模式
            self._model.to(self.device)
            self._model.eval()

            # 加载分类头
            head_path = self.model_path / self.head_weights_path
            if head_path.exists():
                self._classification_head = torch.load(
                    str(head_path), map_location=self.device, weights_only=True
                )
                logger.info(f"[VoiceIntent] 分类头加载成功: {head_path}")
            else:
                logger.warning(
                    f"[VoiceIntent] 分类头不存在: {head_path}。"
                    f"请先运行训练脚本生成分类头权重。"
                )
                return False

            self._initialized = True
            logger.info(
                f"[VoiceIntent] 语音意图分类器加载成功 | "
                f"设备: {self.device} | 意图类别: {len(self.LABELS)}"
            )
            return True

        except ImportError as e:
            logger.error(f"[VoiceIntent] 缺少必要的依赖: {e}")
            logger.info("[VoiceIntent] 请安装: pip install transformers torch")
            return False
        except Exception as e:
            logger.error(f"[VoiceIntent] 模型加载失败: {e}", exc_info=True)
            return False

    def _encode(self, text: str) -> Optional[Dict]:
        """
        使用 ALBERT 编码文本，获取语义特征

        Args:
            text: 输入文本

        Returns:
            {'pooler_output': tensor, 'last_hidden_state': tensor} 或 None
        """
        if not self._initialized or self._model is None:
            return None

        import torch

        with torch.no_grad():
            inputs = self._tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            
            return {
                "pooler_output": outputs.pooler_output,
                "last_hidden_state": outputs.last_hidden_state,
            }

    def _apply_classification_head(self, pooler_output) -> Dict[str, float]:
        """
        应用分类头到 ALBERT 输出

        Args:
            pooler_output: ALBERT 的 pooler_output tensor

        Returns:
            各类别的概率分布
        """
        import torch
        import torch.nn.functional as F

        if self._classification_head is None:
            # 如果分类头未加载，返回均匀分布
            return {label: 1.0 / len(self.LABELS) for label in self.LABELS}

        # 提取分类头参数
        weight = self._classification_head["weight"]  # (3, 768)
        bias = self._classification_head["bias"]      # (3,)

        # 线性变换 + softmax
        logits = torch.nn.functional.linear(pooler_output, weight, bias)
        probs = F.softmax(logits, dim=-1)

        # 转换为字典
        scores = {}
        for i, label in enumerate(self.LABELS):
            scores[label] = probs[0, i].item()

        return scores

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        预测输入文本的语音意图

        Args:
            text: 输入文本

        Returns:
            (predicted_label, confidence, all_scores)
            - predicted_label: 预测的意图类别
            - confidence: 置信度（最高概率）
            - all_scores: 所有类别的概率分布
        """
        if not self._initialized:
            if not self.load():
                raise RuntimeError("[VoiceIntent] 模型未加载")

        # ALBERT 语义编码
        encoding = self._encode(text)
        if encoding is None:
            raise RuntimeError("[VoiceIntent] 文本编码失败")

        # 应用分类头
        all_scores = self._apply_classification_head(encoding["pooler_output"])

        # 获取预测结果
        pred_label = max(all_scores, key=all_scores.get)
        confidence = all_scores[pred_label]

        logger.debug(
            f"[VoiceIntent] 预测结果: {pred_label} (置信度: {confidence:.3f}) | "
            f"文本: '{text[:50]}...'"
        )

        return pred_label, confidence, all_scores

    def is_available(self) -> bool:
        """检查分类器是否可用"""
        return (
            self._initialized
            and self._model is not None
            and self._classification_head is not None
        )

    def warmup(self, sample_text: str = "你好") -> bool:
        """
        预热分类器（首次推理前调用）

        Args:
            sample_text: 用于预热的样本文本

        Returns:
            是否预热成功
        """
        try:
            if not self._initialized:
                return False
            
            self.predict(sample_text)
            logger.info("[VoiceIntent] 语音意图分类器预热成功")
            return True
        except Exception as e:
            logger.error(f"[VoiceIntent] 语音意图分类器预热失败: {e}")
            return False
