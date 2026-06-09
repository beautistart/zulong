"""Platform-aware LLM backend resolver.

This module does not replace ConfigManager. It normalizes the active LLM
backend configuration and attaches platform diagnostics so callers can keep
using the existing OpenAI-compatible client path.

支持的后端类型：
- 本地推理：ollama, vllm, sglang, llamacpp, lmstudio
- 云端 API：openai, siliconflow
- 中转站/代理：openrouter, oneapi, custom (通用 OpenAI 兼容端点)

Ollama 增强功能：
- 模型列表自动发现 (list_ollama_models)
- 健康检查 (check_ollama_health)
- 模型拉取管理 (pull_ollama_model)
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from zulong.utils.runtime_env import get_runtime_context


OPENAI_COMPATIBLE_BACKENDS = {
    "ollama",
    "vllm",
    "sglang",
    "llamacpp",
    "llamaccp",
    "lmstudio",
    "openai",
    "siliconflow",
    "openrouter",
    "oneapi",
    "custom",
    "openai_compatible",
}

# 中转站/代理类后端（支持多模型动态切换）
PROXY_BACKENDS = {
    "openrouter",
    "oneapi",
    "siliconflow",
}

PLATFORM_DEFAULT_BACKEND = {
    "windows": "ollama",
    "linux": "ollama",
    "macos": "ollama",
}

MACOS_NON_DEFAULT_BACKENDS = {"vllm", "sglang"}


@dataclass(frozen=True)
class BackendResolution:
    backend: str
    config: Dict[str, Any]
    os_family: str
    compatible: bool
    recommended: bool
    warnings: list[str] = field(default_factory=list)

    def to_config(self) -> Dict[str, Any]:
        data = dict(self.config)
        data["backend"] = self.backend
        data["platform"] = self.os_family
        data["openai_compatible"] = self.compatible
        data["recommended_on_platform"] = self.recommended
        data["warnings"] = list(self.warnings)
        return data


def normalize_backend_name(backend: str | None) -> str:
    name = (backend or "auto").strip().lower()
    if name == "llama.cpp":
        return "llamacpp"
    return name


def resolve_llm_backend(config: Mapping[str, Any], backend: str | None = None) -> BackendResolution:
    """Resolve the active LLM backend with platform diagnostics."""
    runtime = get_runtime_context()
    os_family = runtime.get("os_family", "unknown")
    llm_config = dict(config.get("llm", {}) if isinstance(config, Mapping) else {})

    requested = normalize_backend_name(backend or llm_config.get("backend") or "auto")
    if requested == "auto":
        requested = PLATFORM_DEFAULT_BACKEND.get(os_family, "ollama")

    config_key = requested
    backend_config = dict(llm_config.get(config_key, {}) or {})
    if requested == "llamacpp" and not backend_config and llm_config.get("llamaccp"):
        config_key = "llamaccp"
        backend_config = dict(llm_config.get(config_key, {}) or {})
    warnings: list[str] = []
    compatible = requested in OPENAI_COMPATIBLE_BACKENDS
    recommended = True

    if not compatible:
        warnings.append(f"LLM backend '{requested}' is not marked as OpenAI-compatible.")

    if os_family == "macos" and requested in MACOS_NON_DEFAULT_BACKENDS:
        recommended = False
        warnings.append("macOS does not use vLLM/SGLang as the default acceptance path; prefer Ollama/llama.cpp/MLX.")

    if requested == "llamaccp" or config_key == "llamaccp":
        warnings.append("Backend name 'llamaccp' is kept for existing config compatibility; prefer 'llamacpp' in new config.")

    # 中转站/代理后端：自动从后端配置中查找实际使用的模型列表
    if requested in PROXY_BACKENDS and not backend_config:
        backend_config = _build_proxy_defaults(requested)

    if not backend_config:
        fallback = PLATFORM_DEFAULT_BACKEND.get(os_family, "ollama")
        backend_config = dict(llm_config.get(fallback, {}) or {})
        warnings.append(f"Backend '{requested}' has no config; using '{fallback}' config fields.")

    backend_config.setdefault("base_url", default_base_url(requested))
    backend_config.setdefault("api_key", default_api_key(requested))
    backend_config.setdefault("model_id", default_model_id(requested))

    return BackendResolution(
        backend=requested,
        config=backend_config,
        os_family=os_family,
        compatible=compatible,
        recommended=recommended,
        warnings=warnings,
    )


def default_base_url(backend: str) -> str:
    if backend == "ollama":
        return "http://localhost:11434/v1"
    if backend == "lmstudio":
        return "http://localhost:1234/v1"
    if backend == "vllm":
        return "http://localhost:8000/v1"
    if backend == "sglang":
        return "http://localhost:30000/v1"
    if backend in {"llamacpp", "llamaccp"}:
        return "http://localhost:8080/v1"
    if backend == "openai":
        return "https://api.openai.com/v1"
    if backend == "siliconflow":
        return "https://api.siliconflow.cn/v1"
    if backend == "openrouter":
        return "https://openrouter.ai/api/v1"
    if backend == "oneapi":
        return "http://localhost:3000/v1"
    if backend in {"custom", "openai_compatible"}:
        return "http://localhost:11434/v1"
    return "http://localhost:11434/v1"


def default_api_key(backend: str) -> str:
    if backend == "openai":
        return ""
    if backend == "openrouter":
        return ""  # 需要用户自行配置
    if backend == "oneapi":
        return ""  # 需要用户自行配置
    if backend in {"custom", "openai_compatible"}:
        return ""
    return "EMPTY"


def default_model_id(backend: str) -> str:
    if backend == "openai":
        return "gpt-4o-mini"
    if backend == "siliconflow":
        return "deepseek-ai/DeepSeek-V4-Flash"
    if backend == "openrouter":
        return "openai/gpt-4o-mini"
    if backend == "oneapi":
        return "gpt-4o-mini"
    if backend in {"vllm", "sglang"}:
        return "./models/Qwen/Qwen3___5-0.8B-AWQ"
    if backend in {"custom", "openai_compatible"}:
        return "qwen3.5:4b"
    return "qwen3.5:4b"


def _build_proxy_defaults(backend: str) -> Dict[str, Any]:
    """为中转站/代理后端构建默认配置。"""
    return {
        "backend": backend,
        "base_url": default_base_url(backend),
        "api_key": default_api_key(backend),
        "model_id": default_model_id(backend),
    }


def _ollama_api_url(base_url: str) -> str:
    """从 OpenAI 兼容的 /v1 端点推导出 Ollama 原生 API 基础 URL。"""
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        return url[:-3]
    return url


def list_ollama_models(base_url: str = "http://localhost:11434", timeout: int = 5) -> List[Dict[str, Any]]:
    """从 Ollama 实例获取已安装的模型列表。

    Args:
        base_url: Ollama 原生 API 地址（非 /v1 端点），例如 http://localhost:11434
        timeout: 请求超时秒数

    Returns:
        模型列表，每个模型包含 name, size, modified_at 等字段。
        请求失败时返回空列表。
    """
    api_url = _ollama_api_url(base_url).rstrip("/") + "/api/tags"
    try:
        req = urllib.request.Request(api_url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("models", [])
    except Exception:
        return []


def check_ollama_health(base_url: str = "http://localhost:11434", timeout: int = 3) -> bool:
    """检查 Ollama 服务是否正常运行。

    Args:
        base_url: Ollama 原生 API 地址
        timeout: 请求超时秒数

    Returns:
        True 表示服务健康，False 表示不可达
    """
    api_url = _ollama_api_url(base_url).rstrip("/")
    try:
        req = urllib.request.Request(api_url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def pull_ollama_model(model_name: str, base_url: str = "http://localhost:11434",
                      timeout: int = 600) -> bool:
    """从 Ollama 拉取（下载）模型。

    Args:
        model_name: 模型名称，如 "qwen3.5:4b"
        base_url: Ollama 原生 API 地址
        timeout: 拉取超时秒数（大模型可能需要较长时间）

    Returns:
        True 表示拉取成功，False 表示失败
    """
    api_url = _ollama_api_url(base_url).rstrip("/") + "/api/pull"
    try:
        payload = json.dumps({"name": model_name, "stream": False}).encode("utf-8")
        req = urllib.request.Request(api_url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_ollama_model_info(model_name: str, base_url: str = "http://localhost:11434",
                          timeout: int = 5) -> Optional[Dict[str, Any]]:
    """获取 Ollama 上指定模型的详细信息。

    Args:
        model_name: 模型名称
        base_url: Ollama 原生 API 地址
        timeout: 请求超时秒数

    Returns:
        模型信息字典，包含 modelfile、parameters 等字段。
        失败时返回 None。
    """
    api_url = _ollama_api_url(base_url).rstrip("/") + "/api/show"
    try:
        payload = json.dumps({"name": model_name}).encode("utf-8")
        req = urllib.request.Request(api_url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def detect_available_ollama_models(base_url: str = "http://localhost:11434") -> Dict[str, Any]:
    """检测 Ollama 服务状态和可用模型。

    组合健康检查和模型列表，返回完整的状态报告。

    Args:
        base_url: Ollama 原生 API 地址

    Returns:
        包含 healthy (bool), model_count (int), models (list) 的字典
    """
    healthy = check_ollama_health(base_url)
    result: Dict[str, Any] = {
        "healthy": healthy,
        "base_url": base_url,
        "model_count": 0,
        "models": [],
    }
    if healthy:
        models = list_ollama_models(base_url)
        result["model_count"] = len(models)
        result["models"] = [
            {
                "name": m.get("name", ""),
                "size": m.get("size", 0),
                "modified_at": m.get("modified_at", ""),
                "digest": m.get("digest", ""),
            }
            for m in models
        ]
    return result
