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
    call_kwargs.update(extra_kwargs)

    logger.debug("[LLMGateway] litellm model=%s api_base=%s format=%s stream=%s",
                 full_model, api_base, api_format, stream)
    return litellm.completion(**call_kwargs)


def _fallback_openai_completion(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
):
    """litellm 未安装时的回退：直接用 openai SDK。"""
    from openai import OpenAI
    client = OpenAI(base_url=api_base or "http://localhost:11434/v1", api_key=api_key or "EMPTY")
    kwargs.pop("api_base", None)
    kwargs.pop("api_key", None)
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
