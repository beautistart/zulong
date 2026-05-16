import time
import uuid
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class TimeoutPhase(Enum):
    CORE_TIMEOUT = "CORE_TIMEOUT"
    BACKUP_TIMEOUT = "BACKUP_TIMEOUT"
    BACKUP_UNAVAILABLE = "BACKUP_UNAVAILABLE"
    CORE_BACKUP_SAME_MODEL = "CORE_BACKUP_SAME_MODEL"


class InputIntent(Enum):
    GREETING = "GREETING"
    FAREWELL = "FAREWELL"
    QUESTION = "QUESTION"
    DELETE = "DELETE"
    GENERIC = "GENERIC"


@dataclass
class DegradationContext:
    timeout_phase: TimeoutPhase
    elapsed_seconds: float
    model_id: str
    user_input: str
    request_id: Optional[str] = None


DELETE_MARKERS = ["删除", "移除", "清除", "忘记", "去掉", "不要记住", "删掉", "抹除"]
GREETING_MARKERS = ["你好", "您好", "hello", "hi", "嗨", "早上好", "下午好", "晚上好", "早安", "晚安"]
FAREWELL_MARKERS = ["谢谢", "再见", "拜拜", "bye", "感谢"]
QUESTION_MARKERS = ["？", "?", "吗", "什么", "怎么", "为什么", "哪里", "哪个", "如何", "能不能", "可以"]

DEGRADATION_TEMPLATES: Dict[TimeoutPhase, Dict[InputIntent, str]] = {
    TimeoutPhase.CORE_TIMEOUT: {
        InputIntent.DELETE: "抱歉，当前处理能力受限，无法立即执行删除操作。请稍后重试。",
        InputIntent.GREETING: "你好！我目前响应较慢，请稍后再试。",
        InputIntent.FAREWELL: "不客气，有需要随时找我！",
        InputIntent.QUESTION: "抱歉，模型响应超时，我暂时无法回答这个问题。请稍后再试。",
        InputIntent.GENERIC: "抱歉，模型响应超时，请稍后再试。",
    },
    TimeoutPhase.BACKUP_TIMEOUT: {
        InputIntent.DELETE: "抱歉，当前所有模型响应缓慢，无法执行删除操作。请稍后重试。",
        InputIntent.GREETING: "你好！系统当前负载较高，请稍后再试。",
        InputIntent.FAREWELL: "不客气，有需要随时找我！",
        InputIntent.QUESTION: "抱歉，所有模型均响应超时，暂时无法回答。请稍后再试。",
        InputIntent.GENERIC: "抱歉，我当前响应较慢，请稍后再试。",
    },
    TimeoutPhase.BACKUP_UNAVAILABLE: {
        InputIntent.DELETE: "抱歉，备用模型不可用，无法执行删除操作。请稍后重试。",
        InputIntent.GREETING: "你好！备用模型暂不可用，请稍后再试。",
        InputIntent.FAREWELL: "不客气！",
        InputIntent.QUESTION: "抱歉，备用模型不可用，暂时无法回答。请稍后再试。",
        InputIntent.GENERIC: "抱歉，备用模型不可用，请稍后再试。",
    },
    TimeoutPhase.CORE_BACKUP_SAME_MODEL: {
        InputIntent.DELETE: "抱歉，模型响应超时，无法执行删除操作。请稍后重试。",
        InputIntent.GREETING: "你好！模型响应较慢，请稍后再试。",
        InputIntent.FAREWELL: "不客气！",
        InputIntent.QUESTION: "抱歉，模型响应超时，暂时无法回答。请稍后再试。",
        InputIntent.GENERIC: "抱歉，模型响应超时，请稍后再试。",
    },
}

BACKUP_SUFFIX = "（当前使用备用模型，回复质量可能降低）"


class SmartDegradationHandler:
    def classify_intent(self, user_input: str) -> InputIntent:
        text = user_input.strip().lower() if user_input else ""
        for marker in DELETE_MARKERS:
            if marker in text:
                return InputIntent.DELETE
        for marker in GREETING_MARKERS:
            if marker in text:
                return InputIntent.GREETING
        for marker in FAREWELL_MARKERS:
            if marker in text:
                return InputIntent.FAREWELL
        for marker in QUESTION_MARKERS:
            if marker in text:
                return InputIntent.QUESTION
        return InputIntent.GENERIC

    def generate_response(self, context: DegradationContext) -> str:
        try:
            degradation_id = uuid.uuid4().hex[:12]
        except Exception:
            degradation_id = str(int(time.time()))
        intent = self.classify_intent(context.user_input)
        templates = DEGRADATION_TEMPLATES.get(context.timeout_phase, DEGRADATION_TEMPLATES[TimeoutPhase.CORE_TIMEOUT])
        base_msg = templates.get(intent, templates[InputIntent.GENERIC])
        if context.elapsed_seconds > 0 and intent not in (InputIntent.GREETING, InputIntent.FAREWELL):
            base_msg = base_msg.rstrip("。") + f"（已等待{int(context.elapsed_seconds)}秒）"
        self._last_degradation_id = degradation_id
        self._last_intent = intent
        return base_msg

    def generate_diagnostic_log(self, context: DegradationContext) -> Dict[str, Any]:
        degradation_id = getattr(self, '_last_degradation_id', 'N/A')
        intent = getattr(self, '_last_intent', InputIntent.GENERIC)
        log_data = {
            "degradation_id": degradation_id,
            "timeout_phase": context.timeout_phase.value,
            "elapsed_s": round(context.elapsed_seconds, 2),
            "model_id": context.model_id,
            "input_type": intent.value,
            "fallback_template_used": True,
        }
        logger.info(f"[SmartDegradation] {log_data}")
        return log_data

    def append_backup_hint(self, response_text: str) -> str:
        if not response_text:
            return response_text
        if BACKUP_SUFFIX in response_text:
            return response_text
        return response_text.rstrip() + "\n" + BACKUP_SUFFIX
