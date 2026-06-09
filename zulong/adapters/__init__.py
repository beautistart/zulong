# File: zulong/adapters/__init__.py
"""
ZULONG 适配器层
提供硬件抽象、模型加载、LLM 后端解析、Ollama 管理等适配功能
"""

from .memory_backend import HardwareAwareKVPool, BlockTableManager
from .model_loader import auto_select_model, init_l2_engines
from .backend_resolver import (
    BackendResolution,
    resolve_llm_backend,
    OPENAI_COMPATIBLE_BACKENDS,
    PROXY_BACKENDS,
    list_ollama_models,
    check_ollama_health,
    pull_ollama_model,
    get_ollama_model_info,
    detect_available_ollama_models,
    default_base_url,
    default_model_id,
    default_api_key,
)

__all__ = [
    'HardwareAwareKVPool',
    'BlockTableManager',
    'auto_select_model',
    'init_l2_engines',
    'BackendResolution',
    'resolve_llm_backend',
    'OPENAI_COMPATIBLE_BACKENDS',
    'PROXY_BACKENDS',
    'list_ollama_models',
    'check_ollama_health',
    'pull_ollama_model',
    'get_ollama_model_info',
    'detect_available_ollama_models',
    'default_base_url',
    'default_model_id',
    'default_api_key',
]
