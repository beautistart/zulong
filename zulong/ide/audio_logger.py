"""
音频模块独立日志配置

将音频相关日志输出到单独文件,避免污染主日志
"""
import logging
import logging.handlers
import os
import time
from typing import Optional

_audio_logger: Optional[logging.Logger] = None

def get_audio_logger() -> logging.Logger:
    """
    获取音频模块专用日志器
    
    日志文件: logs/audio_{timestamp}.log
    """
    global _audio_logger
    
    if _audio_logger is not None:
        return _audio_logger
    
    _audio_logger = logging.getLogger("zulong.audio")
    _audio_logger.setLevel(logging.DEBUG)
    _audio_logger.propagate = False
    
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d')
    log_file = os.path.join(log_dir, f"audio_{timestamp}.log")
    
    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in _audio_logger.handlers):
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        _audio_logger.addHandler(file_handler)
    
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) 
               for h in _audio_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        _audio_logger.addHandler(console_handler)
    
    _audio_logger.info(f"音频日志初始化完成: {os.path.abspath(log_file)}")
    
    return _audio_logger


logger = get_audio_logger()
