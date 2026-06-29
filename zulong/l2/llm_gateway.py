"""LLM Gateway — 统一 LLM 调用入口，基于 litellm。

支持多种 API 格式（由 registry 的 api_format 字段决定）：
  chat_completions  → OpenAI Chat Completions (/v1/chat/completions)
                      litellm 前缀: openai/，配合 api_base 指向中转站
  anthropic_messages → Anthropic Messages (/v1/messages)
                      litellm 前缀: anthropic/
  openai_responses  → OpenAI Responses API (/v1/responses)
                      litellm 前缀: openai/（litellm 内部路由）

所有 LLM 调用应通过 llm_completion()，不再直接用 openai SDK。
"""
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import litellm
    litellm.drop_params = True  # 自动丢弃 provider 不支持的参数
    litellm.set_verbose = False
    _LITELLM_AVAILABLE = True
except ImportError:
    _LITELLM_AVAILABLE = False
    logger.warning("[LLMGateway] litellm 未安装，回退到 openai SDK")


def _load_retry_config() -> Dict[str, float]:
    """Load LLM retry config from runtime config with safe defaults."""
    default = {"max_attempts": 3, "delay": 1.0, "backoff_factor": 2.0}
    try:
        from zulong.config.config_manager import ConfigManager

        cfg = ConfigManager().get_dict("l2_inference.retry") or {}
    except Exception:
        cfg = {}
    result = dict(default)
    for key in ("max_attempts", "delay", "backoff_factor"):
        value = cfg.get(key, default[key])
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = default[key]
        result[key] = parsed
    result["max_attempts"] = max(1, int(result["max_attempts"]))
    result["delay"] = max(0.0, float(result["delay"]))
    result["backoff_factor"] = max(1.0, float(result["backoff_factor"]))
    return result


def _http_status_from_error(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    return None


def _is_retryable_llm_error(exc: Exception) -> bool:
    """Return whether an LLM API error is worth retrying.

    Retry transient transport/provider instability only. Do not retry malformed
    requests, authentication problems, quota/balance failures, or invalid URLs.
    """
    status = _http_status_from_error(exc)
    text = str(exc or "")
    lower = text.lower()

    if status in {400, 401, 402, 403, 404, 422}:
        return False
    if any(
        marker in lower
        for marker in (
            "insufficient balance",
            "quota",
            "invalid api key",
            "unauthorized",
            "forbidden",
            "authentication",
            "bad request",
            "invalid request",
            "unsupported",
            "model not found",
            "no connection adapters",
            "invalid url",
            "missing schema",
            "unknown url type",
        )
    ):
        return False

    if status == 429 or (status is not None and 500 <= status <= 599):
        return True
    return any(
        marker in lower
        for marker in (
            "timeout",
            "timed out",
            "readtimeout",
            "connecttimeout",
            "connection error",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "internalservererror",
            "internal server error",
            "rate limit",
            "too many requests",
        )
    )


def _completion_with_retry(call_kwargs: Dict[str, Any], *, retry_config: Dict[str, float]):
    max_attempts = max(1, int(retry_config.get("max_attempts", 1)))
    delay = max(0.0, float(retry_config.get("delay", 0.0)))
    backoff = max(1.0, float(retry_config.get("backoff_factor", 1.0)))
    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return litellm.completion(**call_kwargs)
        except Exception as exc:
            last_error = exc
            retryable = _is_retryable_llm_error(exc)
            if attempt >= max_attempts or not retryable:
                if attempt > 1:
                    logger.warning(
                        "[LLMGateway] LLM 调用重试结束: attempt=%s/%s retryable=%s error=%s",
                        attempt,
                        max_attempts,
                        retryable,
                        exc,
                    )
                raise
            sleep_s = delay * (backoff ** (attempt - 1))
            logger.warning(
                "[LLMGateway] LLM 调用失败，将重试: attempt=%s/%s delay=%.2fs error=%s",
                attempt,
                max_attempts,
                sleep_s,
                exc,
            )
            if sleep_s > 0:
                time.sleep(sleep_s)

    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM retry loop exited unexpectedly")


def _build_litellm_model(
    model_id: str,
    api_format: str = "chat_completions",
) -> str:
    """根据 api_format 构建 litellm 的 model 字符串。

    litellm 通过 model 前缀决定调用哪种 API 格式：
      openai/xxx       → Chat Completions（也用于中转站，配合 api_base）
      anthropic/xxx    → Anthropic Messages
      ollama/xxx       → Ollama 原生

    对于中转站/兼容 API，用 openai/ 前缀 + api_base 指向中转站地址。
    """
    if not model_id:
        return model_id
    if "/" in model_id:
        return model_id  # 已有前缀

    fmt = (api_format or "chat_completions").lower().strip()
    prefix_map = {
        "chat_completions": "openai",        # OpenAI Chat Completions 格式（含中转站）
        "anthropic_messages": "anthropic",    # Anthropic Messages 格式
        "openai_responses": "openai",         # OpenAI Responses 格式
        "ollama": "ollama",                   # Ollama 原生格式
    }
    prefix = prefix_map.get(fmt, "openai")
    return f"{prefix}/{model_id}"


def llm_completion(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    api_format: str = "chat_completions",
    stream: bool = False,
    tools: Optional[List[Dict]] = None,
    tool_choice: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    top_p: float = 0.85,
    timeout: float = 300,
    **extra_kwargs,
):
    """通过 litellm 统一调用 LLM，自动处理不同 API 格式。

    Args:
        model: 模型名（裸名，如 "gpt-5.5"、"claude-sonnet-4-..."）
        messages: 消息列表
        api_base: API 地址（中转站/自定义端点）
        api_key: API 密钥
        api_format: API 格式（chat_completions / anthropic_messages / openai_responses / ollama）
        stream: 是否流式
        tools: FC 工具定义
        tool_choice: 工具选择策略
        max_tokens: 最大输出 token
        temperature: 温度
        top_p: top_p
        timeout: 超时秒数
        **extra_kwargs: 额外参数

    Returns:
        litellm 的 Completion 响应（与 OpenAI SDK 格式兼容）
    """
    # 根据 api_format 构建 litellm model 字符串
    # 如果调用方没传 api_format，从全局变量读取（container.py 的 LLM_API_FORMAT）
    if not api_format or api_format == "chat_completions":
        try:
            import zulong.models.container as _mc
            api_format = getattr(_mc, "LLM_API_FORMAT", api_format or "chat_completions")
        except Exception:
            pass

    full_model = _build_litellm_model(model, api_format)

    if not _LITELLM_AVAILABLE:
        return _fallback_openai_completion(
            model=model, messages=messages, api_base=api_base, api_key=api_key,
            stream=stream, tools=tools, tool_choice=tool_choice,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            timeout=timeout, **extra_kwargs,
        )

    call_kwargs: Dict[str, Any] = {
        "model": full_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
        "timeout": timeout,
    }
    if api_base:
        call_kwargs["api_base"] = api_base
    if api_key:
        call_kwargs["api_key"] = api_key
    if tools:
        call_kwargs["tools"] = tools
    if tool_choice:
        call_kwargs["tool_choice"] = tool_choice
    retry_config = extra_kwargs.pop("retry_config", None) or _load_retry_config()
    for key in ("retry_max_attempts", "retry_delay", "retry_backoff_factor"):
        extra_kwargs.pop(key, None)
    call_kwargs.update(extra_kwargs)

    logger.debug("[LLMGateway] litellm model=%s api_base=%s format=%s stream=%s",
                 full_model, api_base, api_format, stream)
    return _completion_with_retry(call_kwargs, retry_config=retry_config)


def _fallback_openai_completion(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
):
    """litellm 未安装时的回退：直接用 openai SDK。

    注意：litellm 路径会丢弃非标准参数（drop_params），但本回退直接调用
    OpenAI SDK，必须主动剔除 SDK 不识别的参数（如 backend / api_format 等
    来自上层调用的透传字段），否则会报 unexpected keyword argument。
    """
    from openai import OpenAI
    client = OpenAI(base_url=api_base or "http://localhost:11434/v1", api_key=api_key or "EMPTY")
    # 剔除本网关专用、但 OpenAI SDK 不识别的参数
    for _drop in ("api_base", "api_key", "backend", "api_format"):
        kwargs.pop(_drop, None)
    return client.chat.completions.create(model=model, messages=messages, **kwargs)


def llm_completion_simple(
    prompt: str,
    *,
    model: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    api_format: str = "chat_completions",
    max_tokens: int = 256,
    timeout: float = 60,
    system_prompt: Optional[str] = None,
) -> str:
    """简化调用：单轮 prompt → 文本回复。"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = llm_completion(
        model=model,
        messages=messages,
        api_base=api_base,
        api_key=api_key,
        api_format=api_format,
        max_tokens=max_tokens,
        temperature=0.3,
        stream=False,
        timeout=timeout,
    )
    return response.choices[0].message.content or ""
