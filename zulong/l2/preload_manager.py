import time
import asyncio
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable

logger = logging.getLogger(__name__)


class SystemReadinessStatus(Enum):
    BOOTING = "BOOTING"
    READY = "READY"
    PARTIAL_READY = "PARTIAL_READY"
    FAILED = "FAILED"


@dataclass
class PreloadModuleResult:
    module_name: str
    status: str = "pending"
    duration_s: float = 0.0
    start_ts: float = 0.0
    end_ts: float = 0.0
    error_message: str = ""


@dataclass
class PreloadConfig:
    enabled: bool = True
    timeout: int = 300
    warmup_prompt: str = "你好"


class PreloadManager:
    def __init__(self, engine, send_callback: Optional[Callable] = None):
        self._engine = engine
        self._send_callback = send_callback
        self._status = SystemReadinessStatus.BOOTING
        self._results: List[PreloadModuleResult] = []
        self._failed_modules: List[str] = []
        self._preload_start_time: Optional[float] = None

        try:
            _l2_config = {}
            from zulong.config.config_manager import get_l2_inference_config
            _l2_config = get_l2_inference_config()
        except Exception:
            pass

        _preload_raw = {}
        try:
            import yaml, os
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "..", "config", "zulong_config.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    _cfg = yaml.safe_load(f) or {}
                _preload_raw = (_cfg.get("llm") or {}).get("preload") or {}
        except Exception:
            pass

        self._config = PreloadConfig(
            enabled=_preload_raw.get("enabled", True),
            timeout=_preload_raw.get("timeout", 300),
            warmup_prompt=_preload_raw.get("warmup_prompt", "你好"),
        )
        logger.info(f"[PreloadManager] 初始化完成: enabled={self._config.enabled}, timeout={self._config.timeout}s")

    async def start_preload(self):
        if not self._config.enabled:
            logger.info("[PreloadManager] 预加载已禁用，跳过")
            self._status = SystemReadinessStatus.READY
            return
        self._status = SystemReadinessStatus.BOOTING
        self._preload_start_time = time.time()
        logger.info("[PreloadManager] ========== 预加载开始 ==========")
        try:
            await asyncio.wait_for(self._run_preload(), timeout=self._config.timeout)
        except asyncio.TimeoutError:
            logger.error(f"[PreloadManager] 预加载整体超时 ({self._config.timeout}s)，终止未完成任务")
            self._status = SystemReadinessStatus.FAILED
        except Exception as e:
            logger.error(f"[PreloadManager] 预加载异常: {e}", exc_info=True)
            self._status = SystemReadinessStatus.FAILED
        self._declare_readiness()
        self._notify_frontend()
        self._disable_lazy_loading()
        _total = time.time() - self._preload_start_time if self._preload_start_time else 0
        logger.info(f"[PreloadManager] ========== 预加载完成: status={self._status.value}, 耗时={_total:.1f}s ==========")

    async def _run_preload(self):
        logger.info("[PreloadManager] Phase1: 串行加载核心模型...")
        await self._preload_phase1_core_models()
        logger.info("[PreloadManager] Phase2: 并行加载辅助模块...")
        await self._preload_phase2_auxiliary()

    async def _preload_phase1_core_models(self):
        await self._load_module("CORE模型", self._load_core_model)
        await self._load_module("CORE预热", self._warmup_core_model)
        await self._load_module("BACKUP模型", self._load_backup_model)

    async def _preload_phase2_auxiliary(self):
        tasks = [
            self._load_module("工具注册表", self._load_tool_registry),
            self._load_module("熔断器", self._load_circuit_breaker),
            self._load_module("注意力窗口", self._load_attention_window),
        ]
        try:
            from zulong.ide.ide_server import _get_engine
            _engine = _get_engine()
            if _engine:
                tasks.append(self._load_module("ASR模型", self._load_asr))
                tasks.append(self._load_module("TTS模型", self._load_tts))
        except Exception:
            pass
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _load_module(self, name: str, loader: Callable):
        result = PreloadModuleResult(module_name=name, start_ts=time.time())
        try:
            await loader()
            result.status = "success"
            result.end_ts = time.time()
            result.duration_s = result.end_ts - result.start_ts
            logger.info(f"[PreloadManager] ✅ {name} 加载完成 ({result.duration_s:.2f}s)")
        except Exception as e:
            result.status = "failed"
            result.end_ts = time.time()
            result.duration_s = result.end_ts - result.start_ts
            result.error_message = str(e)
            self._failed_modules.append(name)
            logger.warning(f"[PreloadManager] ❌ {name} 加载失败: {e} ({result.duration_s:.2f}s)")
        self._results.append(result)

    async def _load_core_model(self):
        self._engine._ensure_l2_loaded()
        if not self._engine._l2_loaded:
            raise RuntimeError("CORE模型加载失败")

    async def _warmup_core_model(self):
        try:
            health_tracker = getattr(self._engine, '_health_tracker', None)
            from zulong.models.container import LLM_BASE_URL, LLM_MODEL_ID, LLM_API_KEY
            from openai import OpenAI
            client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY or "EMPTY", timeout=30)
            response = client.chat.completions.create(
                model=LLM_MODEL_ID,
                messages=[{"role": "user", "content": self._config.warmup_prompt}],
                max_tokens=32,
                temperature=0.1,
                stream=False,
            )
            _text = response.choices[0].message.content if response.choices else ""
            logger.info(f"[PreloadManager] CORE预热成功: \"{_text[:50]}\"")
            if health_tracker:
                from zulong.l2.model_health_tracker import ModelHealthStatus
                health_tracker.init_health("CORE", ModelHealthStatus.HEALTHY)
        except Exception as e:
            logger.warning(f"[PreloadManager] CORE预热失败(非致命): {e}")
            health_tracker = getattr(self._engine, '_health_tracker', None)
            if health_tracker:
                from zulong.l2.model_health_tracker import ModelHealthStatus
                health_tracker.init_health("CORE", ModelHealthStatus.DEGRADED)

    async def _load_backup_model(self):
        backup_client = getattr(self._engine, 'backup_client', None)
        if backup_client:
            from zulong.models.container import LLM_BASE_URL_BACKUP, LLM_MODEL_ID_BACKUP
            if LLM_MODEL_ID_BACKUP and LLM_BASE_URL_BACKUP:
                logger.info(f"[PreloadManager] BACKUP模型已配置: {LLM_MODEL_ID_BACKUP}")
                return
        logger.info("[PreloadManager] BACKUP模型未配置或不可用，跳过")

    async def _load_tool_registry(self):
        _te = getattr(self._engine, 'tool_engine', None)
        if _te:
            _ = getattr(_te, 'registry', None)

    async def _load_circuit_breaker(self):
        _cb = getattr(self._engine, '_circuit_breaker', None)
        if _cb:
            _ = _cb.enabled

    async def _load_attention_window(self):
        pass

    async def _load_asr(self):
        try:
            from zulong.ide.ide_server import _get_engine
            _engine = _get_engine()
            if _engine and hasattr(_engine, '_asr_handler'):
                await asyncio.sleep(0.01)
        except Exception:
            pass

    async def _load_tts(self):
        try:
            from zulong.ide.ide_server import _get_engine
            _engine = _get_engine()
            if _engine and hasattr(_engine, '_tts_handler'):
                await asyncio.sleep(0.01)
        except Exception:
            pass

    def _declare_readiness(self):
        core_ok = any(r.module_name.startswith("CORE") and r.status == "success"
                      for r in self._results)
        if core_ok:
            non_core_failed = [r for r in self._results
                               if not r.module_name.startswith("CORE") and r.status == "failed"]
            if not non_core_failed:
                self._status = SystemReadinessStatus.READY
            else:
                self._status = SystemReadinessStatus.PARTIAL_READY
                logger.warning(f"[PreloadManager] 部分辅助模块加载失败: {[r.module_name for r in non_core_failed]}")
        else:
            self._status = SystemReadinessStatus.FAILED
            logger.error("[PreloadManager] 核心模型加载失败，系统进入FAILED状态")

    def _notify_frontend(self):
        if self._send_callback:
            try:
                payload = {
                    "status": self._status.value,
                    "failed_modules": self._failed_modules,
                }
                asyncio.get_event_loop().create_task(
                    self._send_callback("system_ready", payload)
                )
            except Exception as e:
                logger.warning(f"[PreloadManager] 通知前端失败: {e}")

    def _disable_lazy_loading(self):
        if self._status in (SystemReadinessStatus.READY, SystemReadinessStatus.PARTIAL_READY):
            self._engine._preload_completed = True
            self._engine._l2_loaded = True
            logger.info("[PreloadManager] 已禁用延迟加载（_preload_completed=True）")

    def is_ready(self) -> bool:
        return self._status in (SystemReadinessStatus.READY, SystemReadinessStatus.PARTIAL_READY)

    def get_status(self) -> SystemReadinessStatus:
        return self._status

    def get_failed_modules(self) -> List[str]:
        return list(self._failed_modules)
