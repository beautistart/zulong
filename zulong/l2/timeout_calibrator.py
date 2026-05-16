import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

CORE_TIMEOUT_MIN = 300
BACKUP_TIMEOUT_MIN = 60
FC_LOOP_TIMEOUT_MIN = 90

CORE_TIMEOUT_DEFAULT = 300
BACKUP_TIMEOUT_DEFAULT = 120
FC_LOOP_TIMEOUT_DEFAULT = 600


class TimeoutCalibrator:
    def __init__(self, timeout_config: Dict[str, Any]):
        self._calibration_logs: List[str] = []
        self._core_timeout = CORE_TIMEOUT_DEFAULT
        self._backup_timeout = BACKUP_TIMEOUT_DEFAULT
        self._fc_loop_timeout = FC_LOOP_TIMEOUT_DEFAULT
        self._calibrate(timeout_config)

    def _calibrate(self, config: Dict[str, Any]):
        core_raw = config.get('core', CORE_TIMEOUT_DEFAULT)
        backup_raw = config.get('backup', BACKUP_TIMEOUT_DEFAULT)
        fc_loop_raw = config.get('fc_loop', FC_LOOP_TIMEOUT_DEFAULT)

        for name, raw_val, default_val in [
            ('core', core_raw, CORE_TIMEOUT_DEFAULT),
            ('backup', backup_raw, BACKUP_TIMEOUT_DEFAULT),
            ('fc_loop', fc_loop_raw, FC_LOOP_TIMEOUT_DEFAULT),
        ]:
            if not isinstance(raw_val, (int, float)):
                log_msg = f"超时配置 {name}={raw_val} 非数值，回退默认值 {default_val}s"
                logger.error(f"[TimeoutCalibrator] {log_msg}")
                self._calibration_logs.append(log_msg)
                if name == 'core':
                    core_raw = default_val
                elif name == 'backup':
                    backup_raw = default_val
                else:
                    fc_loop_raw = default_val

        if core_raw < CORE_TIMEOUT_MIN:
            log_msg = f"CORE超时配置 {core_raw}s 低于下限 {CORE_TIMEOUT_MIN}s，校正为 {CORE_TIMEOUT_MIN}s"
            logger.warning(f"[TimeoutCalibrator] {log_msg}")
            self._calibration_logs.append(log_msg)
            self._core_timeout = CORE_TIMEOUT_MIN
        else:
            self._core_timeout = int(core_raw)

        if backup_raw < BACKUP_TIMEOUT_MIN:
            log_msg = f"BACKUP超时配置 {backup_raw}s 低于下限 {BACKUP_TIMEOUT_MIN}s，校正为 {BACKUP_TIMEOUT_MIN}s"
            logger.warning(f"[TimeoutCalibrator] {log_msg}")
            self._calibration_logs.append(log_msg)
            self._backup_timeout = BACKUP_TIMEOUT_MIN
        else:
            self._backup_timeout = int(backup_raw)

        if fc_loop_raw < FC_LOOP_TIMEOUT_MIN:
            log_msg = f"FC_LOOP超时配置 {fc_loop_raw}s 低于下限 {FC_LOOP_TIMEOUT_MIN}s，校正为 {FC_LOOP_TIMEOUT_MIN}s"
            logger.warning(f"[TimeoutCalibrator] {log_msg}")
            self._calibration_logs.append(log_msg)
            self._fc_loop_timeout = FC_LOOP_TIMEOUT_MIN
        else:
            self._fc_loop_timeout = int(fc_loop_raw)

        if self._core_timeout < self._backup_timeout:
            log_msg = f"CORE超时({self._core_timeout}s)不应低于BACKUP超时({self._backup_timeout}s)，CORE校正为{CORE_TIMEOUT_MIN}s"
            logger.error(f"[TimeoutCalibrator] {log_msg}")
            self._calibration_logs.append(log_msg)
            self._core_timeout = CORE_TIMEOUT_MIN

        logger.info(
            f"[TimeoutCalibrator] 校准完成: core={self._core_timeout}s, "
            f"backup={self._backup_timeout}s, fc_loop={self._fc_loop_timeout}s"
        )

    @property
    def core_timeout(self) -> int:
        return self._core_timeout

    @property
    def backup_timeout(self) -> int:
        return self._backup_timeout

    @property
    def fc_loop_timeout(self) -> int:
        return self._fc_loop_timeout

    @property
    def calibration_logs(self) -> List[str]:
        return list(self._calibration_logs)
