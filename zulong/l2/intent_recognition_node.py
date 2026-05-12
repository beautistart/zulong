# File: zulong/l2/intent_recognition_node.py
# 意图识别节点，利用 ALBERT-tiny 快速分类 + L2 模型复杂意图识别

from zulong.models.container import ModelContainer
from zulong.models.config import ModelID
from zulong.l2.intent_schema import IntentResult, SUPPORTED_INTENTS
from zulong.l1b.intent_filter import IntentFilter
from zulong.config.config_manager import get_config
import json
import re
import logging

logger = logging.getLogger(__name__)


class IntentRecognitionNode:
    """意图识别节点 - ALBERT 快速分类 + LLM 兜底"""

    def __init__(self):
        """初始化意图识别节点"""
        self.model_container = ModelContainer()
        self.l2_model = self.model_container.get_model(ModelID.L2_GATEKEEPER)

        # 初始化 ALBERT 意图分类器 (L1-B 层)
        self._intent_filter = None
        self._init_intent_filter()

    def _init_intent_filter(self):
        """初始化 L1-B 意图过滤器 (ALBERT)"""
        try:
            intent_config = get_config("intent_classification", {})
            self._intent_filter = IntentFilter(config=intent_config)
            logger.info("[L2] L1-B IntentFilter 已初始化")
        except Exception as e:
            logger.warning(f"[L2] L1-B IntentFilter 初始化失败: {e}")
            self._intent_filter = None

    def _quick_classify(self, text: str) -> IntentResult | None:
        """
        使用 ALBERT 模型快速分类意图

        Args:
            text: 用户输入文本

        Returns:
            IntentResult 或 None (如果置信度不足或模型不可用)
        """
        if not self._intent_filter:
            return None

        try:
            result = self._intent_filter.analyze(text)
            intent = result.get("intent", "UNKNOWN")
            confidence = result.get("confidence", 0.0)
            model_used = result.get("model", "keyword")

            # 如果置信度足够高，直接返回
            if confidence > 0:
                logger.debug(
                    f"[L2] ALBERT 快速分类: {intent} (置信度: {confidence:.3f}, 模型: {model_used})"
                )
                return IntentResult(
                    intent=intent,
                    confidence=confidence,
                    parameters={"model": model_used},
                    original_text=text,
                )

        except Exception as e:
            logger.warning(f"[L2] ALBERT 快速分类失败: {e}")

        return None

    def recognize_intent(self, text: str) -> IntentResult:
        """
        识别意图 - 优先 ALBERT 快速分类，LLM 兜底

        Args:
            text: 自然语言文本

        Returns:
            IntentResult: 意图识别结果
        """
        # 1. 优先使用 ALBERT 快速分类
        quick_result = self._quick_classify(text)
        if quick_result and quick_result.confidence >= 0.6:
            return quick_result

        # 2. ALBERT 不可用或置信度不足，使用 LLM 进行分类
        logger.debug("[L2] ALBERT 置信度不足，使用 LLM 进行分类")
        return self._llm_classify(text)

    def _llm_classify(self, text: str) -> IntentResult:
        """
        使用 LLM 进行意图分类 (兜底方案)

        Args:
            text: 用户输入文本

        Returns:
            IntentResult: 意图识别结果
        """
        try:
            # 构建 prompt，要求模型输出严格的 JSON 格式
            prompt = f'''
你是一个机器人意图分类器。
分析用户输入并只输出一个有效的 JSON 对象，包含以下键："intent", "confidence", "parameters"。
不要输出任何解释、Markdown、思考过程或额外内容。
不要输出任何 <think>、</think> 或类似的标签。
不要输出任何 "请继续" 或类似的提示。
只输出一个完整的 JSON 对象，不要重复输出。
支持的意图：MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, STOP, QUERY_STATUS, UNKNOWN。

用户输入："{text}"
JSON 输出：
'''

            # 生成回复
            response = self.l2_model.generate(prompt, max_tokens=200)

            # 解析 JSON 输出
            try:
                # 打印原始输出以便调试
                logger.debug(f"[L2] [Model] Raw response: {response}")

                # 清理响应文本
                cleaned_response = response.strip()

                # 移除 Markdown 标记
                cleaned_response = re.sub(r'```json|```', '', cleaned_response)

                # 移除思考内容
                cleaned_response = re.sub(r'<think>[\s\S]*?</think>', '', cleaned_response)

                # 分割可能的多个 JSON 对象
                json_candidates = []
                brace_count = 0
                start_idx = 0

                for i, char in enumerate(cleaned_response):
                    if char == '{':
                        if brace_count == 0:
                            start_idx = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and i > start_idx:
                            json_candidates.append(cleaned_response[start_idx:i + 1])

                # 尝试解析每个候选 JSON
                for candidate in json_candidates:
                    try:
                        # 清理 JSON 字符串
                        json_str = candidate.strip().replace('\n', '').replace('\r', '')
                        result = json.loads(json_str)

                        # 处理参数类型
                        parameters = result.get('parameters', {})
                        if not isinstance(parameters, dict):
                            # 将非字典类型转换为字典
                            parameters = {"value": parameters}

                        # 验证结果
                        if result.get('intent') in SUPPORTED_INTENTS:
                            return IntentResult(
                                intent=result['intent'],
                                confidence=result.get('confidence', 0.0),
                                parameters=parameters,
                                original_text=text
                            )
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(f"[IntentRecognitionNode] JSON 解析失败: {e}")

            # 如果解析失败，返回 UNKNOWN 意图
            return IntentResult(
                intent="UNKNOWN",
                confidence=0.0,
                parameters={},
                original_text=text
            )
        except Exception as e:
            logger.error(f"[IntentRecognitionNode] 意图识别失败: {e}")
            # 返回 UNKNOWN 意图
            return IntentResult(
                intent="UNKNOWN",
                confidence=0.0,
                parameters={},
                original_text=text
            )
