import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TimeoutEventLogger:
    def log_timeout(
        self,
        timeout_phase: str,
        elapsed_seconds: float,
        timeout_threshold: float,
        model_id: str,
        request_id: Optional[str] = None,
        consecutive_timeouts: int = 0,
    ):
        log_data = {
            "timeout_phase": timeout_phase,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "timeout_threshold": timeout_threshold,
            "model_id": model_id,
            "request_id": request_id or "N/A",
        }
        msg = f"[TimeoutEvent] {log_data}"
        if consecutive_timeouts >= 3:
            logger.error(msg)
        else:
            logger.warning(msg)

    def log_degradation_decision(
        self,
        decision: str,
        reason: str,
        model_id: str,
        request_id: Optional[str] = None,
    ):
        log_data = {
            "degradation_decision": decision,
            "reason": reason,
            "model_id": model_id,
            "request_id": request_id or "N/A",
        }
        logger.info(f"[DegradationDecision] {log_data}")
