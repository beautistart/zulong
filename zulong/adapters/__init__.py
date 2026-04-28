# File: zulong/adapters/__init__.py
"""
ZULONG 适配器层
提供硬件抽象、模型加载等适配功能
"""

from .memory_backend import HardwareAwareKVPool, BlockTableManager
from .model_loader import auto_select_model, init_l2_engines

__all__ = [
    'HardwareAwareKVPool',
    'BlockTableManager',
    'auto_select_model',
    'init_l2_engines',
]
