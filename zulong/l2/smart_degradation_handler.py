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
    CORE_RATE_LIMIT = "CORE_RATE_LIMIT"
    CORE_QUOTA_EXHAUSTED = "CORE_QUOTA_EXHAUSTED"
    CORE_API_ERROR = "CORE_API_ERROR"
    CIRCUIT_BREAKER_TRIPPED = "CIRCUIT_BREAKER_TRIPPED"  # 非超时: CircuitBreaker 触发 RED 提前终止
    ORCHESTRATOR_NO_OUTPUT = "ORCHESTRATOR_NO_OUTPUT"  # 非超时: 编排器/FC 完成但未生成有效回复


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
    error_reason: Optional[str] = None


DELETE_MARKERS = ["删除", "移除", "清除", "忘记", "去掉", "不要记住", "删掉", "抹除"]
GREETING_MARKERS = ["你好", "您好", "hello", "hi", "嗨", "早上好", "下午好", "晚上好", "早安", "晚安"]
FAREWELL_MARKERS = ["谢谢", "再见", "拜拜", "bye", "感谢"]
QUESTION_MARKERS = ["？", "?", "吗", "什么", "怎么", "为什么", "哪里", "哪个", "如何", "能不能", "可以"]

DEGRADATION_REASONS: Dict[TimeoutPhase, str] = {
    TimeoutPhase.CORE_TIMEOUT: "主模型响应超时",
    TimeoutPhase.BACKUP_TIMEOUT: "主模型和备用模型都响应超时",
    TimeoutPhase.BACKUP_UNAVAILABLE: "主模型不可用，备用模型也未配置或不可用",
    TimeoutPhase.CORE_BACKUP_SAME_MODEL: "主模型不可用，备用模型与主模型相同，无法继续降级",
    TimeoutPhase.CORE_RATE_LIMIT: "主模型触发频率限制（429/rate limit）",
    TimeoutPhase.CORE_QUOTA_EXHAUSTED: "主模型额度或余额不足（402/Insufficient Balance）",
    TimeoutPhase.CORE_API_ERROR: "主模型 API 调用失败",
    TimeoutPhase.CIRCUIT_BREAKER_TRIPPED: "安全防护触发：检测到重复调用或无效循环，已提前终止",
    TimeoutPhase.ORCHESTRATOR_NO_OUTPUT: "推理流程已完成，但模型未生成有效回复（可能因 API 返回空或编排器提前终止）",
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
        reason = context.error_reason or DEGRADATION_REASONS.get(
            context.timeout_phase,
            DEGRADATION_REASONS[TimeoutPhase.CORE_TIMEOUT],
        )
        if context.timeout_phase == TimeoutPhase.CORE_BACKUP_SAME_MODEL and context.error_reason:
            reason = f"{context.error_reason}，备用模型与主模型相同，无法继续降级"
        base_msg = f"系统当前出问题了，{reason}，因此无法正常回复。"
        if context.elapsed_seconds > 0:
            base_msg = base_msg.rstrip("。") + f"（已等待{int(context.elapsed_seconds)}秒）。"
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
        if context.error_reason:
            log_data["error_reason"] = context.error_reason
        logger.info(f"[SmartDegradation] {log_data}")
        return log_data

    def append_backup_hint(self, response_text: str) -> str:
        if not response_text:
            return response_text
        if BACKUP_SUFFIX in response_text:
            return response_text
        return response_text.rstrip() + "\n" + BACKUP_SUFFIX
