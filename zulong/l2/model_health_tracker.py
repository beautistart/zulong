import time
import threading
import logging
from enum import Enum
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class ModelHealthStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


class ModelHealthRecord:
    __slots__ = ('model_id', 'status', 'consecutive_timeouts',
                 'consecutive_successes', 'last_success_time',
                 'last_timeout_time', 'unavailable_since')

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.status = ModelHealthStatus.HEALTHY
        self.consecutive_timeouts = 0
        self.consecutive_successes = 0
        self.last_success_time: Optional[float] = None
        self.last_timeout_time: Optional[float] = None
        self.unavailable_since: Optional[float] = None


class ModelHealthTracker:
    DEGRADED_THRESHOLD = 3
    UNAVAILABLE_THRESHOLD = 5
    UNAVAILABLE_EXPIRE_SECONDS = 300

    def __init__(self, enabled: bool = True):
        self._records: Dict[str, ModelHealthRecord] = {}
        self._enabled = enabled
        self._lock = threading.Lock()

    def _get_or_create(self, model_id: str) -> ModelHealthRecord:
        if model_id not in self._records:
            self._records[model_id] = ModelHealthRecord(model_id)
        return self._records[model_id]

    def check_health(self, model_id: str) -> ModelHealthStatus:
        if not self._enabled:
            return ModelHealthStatus.HEALTHY
        with self._lock:
            record = self._get_or_create(model_id)
            if record.status == ModelHealthStatus.UNAVAILABLE and record.unavailable_since:
                elapsed = time.time() - record.unavailable_since
                if elapsed > self.UNAVAILABLE_EXPIRE_SECONDS:
                    record.status = ModelHealthStatus.DEGRADED
                    record.consecutive_timeouts = self.UNAVAILABLE_THRESHOLD - 1
                    record.unavailable_since = None
                    logger.info(f"[ModelHealth] {model_id} UNAVAILABLE过期({elapsed:.0f}s>{self.UNAVAILABLE_EXPIRE_SECONDS}s)，重置为DEGRADED")
            return record.status

    def record_success(self, model_id: str):
        if not self._enabled:
            return
        with self._lock:
            record = self._get_or_create(model_id)
            record.consecutive_timeouts = 0
            record.consecutive_successes += 1
            record.last_success_time = time.time()
            old_status = record.status
            record.status = ModelHealthStatus.HEALTHY
            record.unavailable_since = None
            if old_status != ModelHealthStatus.HEALTHY:
                logger.info(f"[ModelHealth] {model_id} 恢复HEALTHY（连续成功{record.consecutive_successes}次）")

    def record_timeout(self, model_id: str):
        if not self._enabled:
            return
        with self._lock:
            record = self._get_or_create(model_id)
            record.consecutive_timeouts += 1
            record.consecutive_successes = 0
            record.last_timeout_time = time.time()
            old_status = record.status
            if record.consecutive_timeouts >= self.UNAVAILABLE_THRESHOLD:
                record.status = ModelHealthStatus.UNAVAILABLE
                if record.unavailable_since is None:
                    record.unavailable_since = time.time()
            elif record.consecutive_timeouts >= self.DEGRADED_THRESHOLD:
                record.status = ModelHealthStatus.DEGRADED
            if old_status != record.status:
                logger.warning(f"[ModelHealth] {model_id} 状态变更: {old_status.value}→{record.status.value}（连续超时{record.consecutive_timeouts}次）")

    def init_health(self, model_id: str, initial_status: ModelHealthStatus):
        if not self._enabled:
            return
        with self._lock:
            record = self._get_or_create(model_id)
            record.status = initial_status
            if initial_status == ModelHealthStatus.UNAVAILABLE:
                record.unavailable_since = time.time()
            elif initial_status == ModelHealthStatus.HEALTHY:
                record.last_success_time = time.time()
                record.consecutive_timeouts = 0
            logger.info(f"[ModelHealth] {model_id} 初始健康状态设为 {initial_status.value}")

    def should_skip(self, model_id: str) -> bool:
        return self.check_health(model_id) == ModelHealthStatus.UNAVAILABLE

    def get_consecutive_timeouts(self, model_id: str) -> int:
        if not self._enabled:
            return 0
        with self._lock:
            record = self._get_or_create(model_id)
            return record.consecutive_timeouts
