"""
结构化异常处理框架

提供统一的异常处理机制，确保所有异常都被正确记录和上报，
避免静默吞没异常（except pass）。
"""

import logging
import traceback
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(Enum):
    """错误码枚举"""

    WS_CONNECTION_FAILED = "WS-0001"
    WS_SEND_FAILED = "WS-0002"
    WS_RECEIVE_FAILED = "WS-0003"
    WS_TIMEOUT = "WS-0004"
    WS_INVALID_MESSAGE = "WS-0005"

    FC_LOOP_ERROR = "FC-0001"
    FC_MODEL_CALL_FAILED = "FC-0002"
    FC_TOOL_EXEC_FAILED = "FC-0003"
    FC_STATE_ERROR = "FC-0004"
    FC_TIMEOUT = "FC-0005"

    TOOL_NOT_FOUND = "TOOL-0001"
    TOOL_EXEC_ERROR = "TOOL-0002"
    TOOL_VALIDATION_ERROR = "TOOL-0003"
    TOOL_TIMEOUT = "TOOL-0004"

    CFG_LOAD_FAILED = "CFG-0001"
    CFG_INVALID = "CFG-0002"
    CFG_SAVE_FAILED = "CFG-0003"

    UNKNOWN_ERROR = "SYS-0001"


@dataclass
class StructuredError:
    """结构化错误信息"""

    error_code: ErrorCode
    error_type: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于日志记录和序列化）"""
        return {
            "error_code": self.error_code.value,
            "error_type": self.error_type,
            "message": self.message,
            "details": self.details,
            "stack_trace": self.stack_trace,
            "timestamp": self.timestamp,
            "context": self.context,
        }

    def to_user_message(self) -> str:
        """转换为用户可读的错误提示（不暴露内部路径和堆栈）"""
        code = self.error_code.value
        category = code.split("-")[0]

        if category == "WS":
            return f"[{code}] WebSocket连接异常，请检查后端服务是否运行"
        elif category == "FC":
            return f"[{code}] 任务执行异常，请稍后重试或查看日志"
        elif category == "TOOL":
            return f"[{code}] 工具执行失败：{self.message}"
        elif category == "CFG":
            return f"[{code}] 配置异常，请检查配置文件"
        else:
            return f"[{code}] 系统异常，请联系管理员"


class ErrorHandler:
    """异常处理器"""

    _logger = logging.getLogger(__name__)

    @staticmethod
    def handle_exception(
        exception: Exception,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        context: Optional[Dict[str, Any]] = None,
        log_level: int = logging.ERROR,
        raise_exception: bool = False,
    ) -> StructuredError:
        """
        处理异常并记录结构化日志

        Args:
            exception: 原始异常对象
            error_code: 错误码
            context: 上下文信息（如函数名、参数等）
            log_level: 日志级别
            raise_exception: 是否重新抛出异常

        Returns:
            StructuredError: 结构化错误对象
        """
        stack_trace = traceback.format_exc()

        structured_error = StructuredError(
            error_code=error_code,
            error_type=type(exception).__name__,
            message=str(exception),
            stack_trace=stack_trace,
            context=context or {},
        )

        log_message = f"[{error_code.value}] {type(exception).__name__}: {exception}"
        if context:
            log_message += f" | Context: {context}"

        ErrorHandler._logger.log(log_level, log_message)
        if log_level >= logging.ERROR:
            ErrorHandler._logger.debug(f"Stack trace:\n{stack_trace}")

        if raise_exception:
            raise exception

        return structured_error

    @staticmethod
    def log_warning(
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录警告日志"""
        log_message = f"[{error_code.value}] {message}"
        if context:
            log_message += f" | Context: {context}"
        ErrorHandler._logger.warning(log_message)

    @staticmethod
    def log_info(
        message: str,
        error_code: Optional[ErrorCode] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录信息日志"""
        if error_code:
            log_message = f"[{error_code.value}] {message}"
        else:
            log_message = message
        if context:
            log_message += f" | Context: {context}"
        ErrorHandler._logger.info(log_message)
