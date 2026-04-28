# File: zulong/utils/model_preloader.py
# 通用模型预加载器 — 支持 Ollama / LM Studio / vLLM 等 OpenAI 兼容后端
#
# 在系统启动时发送一个轻量 warmup 请求，触发模型加载/编译，
# 避免用户第一次对话时经历漫长的冷启动。

import json
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class ModelPreloader:
    """通用模型预加载器

    支持的检测方式:
    1. Ollama: 先尝试 /api/ps 探测，然后用 OpenAI 兼容 API 发送 warmup
    2. LM Studio / vLLM / 其他 OpenAI 兼容服务: 直接用 OpenAI SDK 发送 warmup
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        backend: str = "auto",
        api_key: str = "EMPTY",
        timeout: int = 300,
        warmup_prompt: str = "你好",
        num_ctx: int = 0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.backend = backend  # "ollama" | "lmstudio" | "vllm" | "openai" | "auto"
        self.api_key = api_key
        self.timeout = timeout
        self.warmup_prompt = warmup_prompt
        self.num_ctx = num_ctx  # Ollama num_ctx，0 = 不指定（使用模型默认值）
        self._thread: Optional[threading.Thread] = None
        self._preload_done = False
        self._preload_error: Optional[str] = None

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """在后台线程中启动预加载（不阻塞调用方）"""
        self._thread = threading.Thread(
            target=self._preload,
            name="ModelPreloader",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"🔥 [ModelPreloader] 预热已启动 (backend={self.backend}, "
            f"model={self.model_id}, url={self.base_url})"
        )

    def is_ready(self) -> bool:
        """预加载是否已完成"""
        return self._preload_done

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _detect_backend(self) -> str:
        """自动检测后端类型"""
        import urllib.request

        # 尝试 Ollama 特有端点
        if "11434" in self.base_url:
            try:
                url = self.base_url.replace("/v1", "") + "/api/ps"
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode())
                    if "models" in data:
                        return "ollama"
            except Exception:
                pass

        # 尝试 LM Studio 特有端点
        if "1234" in self.base_url:
            return "lmstudio"

        # 默认回退为 OpenAI 兼容
        return "openai_compatible"

    def _preload(self) -> None:
        """执行预热（在后台线程中运行）"""
        start = time.time()

        # 自动检测后端
        if self.backend == "auto":
            self.backend = self._detect_backend()
            logger.info(f"🔍 [ModelPreloader] 自动检测到后端: {self.backend}")

        if self.backend == "ollama":
            self._preload_ollama()
        else:
            self._preload_openai_compatible()

        elapsed = time.time() - start
        if self._preload_done:
            logger.info(
                f"✅ [ModelPreloader] 预热完成，耗时 {elapsed:.1f}s"
            )
        else:
            logger.warning(
                f"⚠️ [ModelPreloader] 预热超时 ({self.timeout}s)，"
                f"错误: {self._preload_error}"
            )

    def _preload_ollama(self) -> None:
        """Ollama 预热：先用 /api/generate 触发模型加载，再用 OpenAI API 验证"""
        import urllib.request

        ollama_base = self.base_url.replace("/v1", "")

        # 步骤 1: 用原生 /api/generate 触发加载（不等待完成，发完就返回）
        try:
            generate_body = {
                "model": self.model_id,
                "prompt": "hi",
                "stream": False,
                "keep_alive": "60m",  # 让模型在内存中保持 60 分钟
            }
            if self.num_ctx > 0:
                generate_body["options"] = {"num_ctx": self.num_ctx}
            payload = json.dumps(generate_body).encode()
            req = urllib.request.Request(
                f"{ollama_base}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            # 只等 30 秒触发加载，不等完成
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
            logger.info(f"🔥 [ModelPreloader] Ollama 模型加载已触发: {self.model_id}")
        except Exception as e:
            logger.debug(f"[ModelPreloader] /api/generate 预热: {e}")

        # 步骤 2: 用 OpenAI 兼容 API 验证
        self._preload_openai_compatible()

    def _preload_openai_compatible(self) -> None:
        """OpenAI 兼容 API 预热（LM Studio / vLLM / Ollama 通用）"""
        try:
            from openai import OpenAI

            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            logger.info(f"🔥 [ModelPreloader] 发送 warmup 请求...")
            kwargs = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": self.warmup_prompt}],
                "max_tokens": 16,
                "temperature": 0.1,
                "stream": False,
            }
            if self.num_ctx > 0 and self.backend == "ollama":
                kwargs["extra_body"] = {"options": {"num_ctx": self.num_ctx}}
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            logger.info(
                f"✅ [ModelPreloader] 模型就绪，warmup 回复: {text[:80]}"
            )
            self._preload_done = True
        except Exception as e:
            self._preload_error = str(e)
            logger.debug(f"[ModelPreloader] warmup 失败: {e}")
            # 不设为 True，让调用方可以重试或降级


# ------------------------------------------------------------------
# 便捷函数：从配置创建并启动
# ------------------------------------------------------------------

_active_preloader: Optional[ModelPreloader] = None


def preload_model_from_config(config_manager) -> Optional[ModelPreloader]:
    """从祖龙配置创建并启动模型预加载

    Args:
        config_manager: ConfigManager 实例

    Returns:
        ModelPreloader 实例（后台已启动）或 None（配置不可用时）
    """
    global _active_preloader

    backend = config_manager.get("llm.backend", "ollama")
    backend_cfg = config_manager.get(f"llm.{backend}", {})
    if not backend_cfg:
        logger.warning("⚠️ [ModelPreloader] 未找到 LLM 后端配置，跳过预热")
        return None

    base_url = backend_cfg.get("base_url", "http://localhost:11434/v1")
    model_id = backend_cfg.get("model_id", "")
    api_key = backend_cfg.get("api_key", "EMPTY")
    timeout = config_manager.get("llm.preload_timeout", 300)
    num_ctx = int(backend_cfg.get("num_ctx", 0))

    if not model_id:
        logger.warning("⚠️ [ModelPreloader] model_id 为空，跳过预热")
        return None

    preloader = ModelPreloader(
        base_url=base_url,
        model_id=model_id,
        backend=backend,
        api_key=api_key,
        timeout=timeout,
        num_ctx=num_ctx,
    )
    preloader.start()
    _active_preloader = preloader
    return preloader


def get_preloader() -> Optional[ModelPreloader]:
    """获取当前活跃的预加载器"""
    return _active_preloader
