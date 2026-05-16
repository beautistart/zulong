# File: zulong/l2/inference_engine.py
# L2 推理引擎 - 修复版（集成 RAG 记忆和优化生成参数）

import time
import threading
import asyncio
import contextvars
import os
import re
import numpy as np
from typing import Optional, Callable, List, Any, Dict

from zulong.core.event_bus import event_bus
from zulong.core.state_manager import state_manager
from zulong.core.types import EventType, EventPriority, ZulongEvent, L2Status
from zulong.models.container import ModelContainer
from zulong.models.config import ModelID
from zulong.l2.rag_node import RAGIntegrationNode
# ShortTermMemory 已废弃（记忆架构改造：MemoryGraph 统一替代）
from zulong.memory.experience_generator import ExperienceGenerator
from zulong.utils.performance import PerformanceTimer
from zulong.tools.tool_engine import ToolEngine
from zulong.l2.info_gap_detector import InformationGapDetector, InfoGapType
from zulong.l2.attention_window import (
    AttentionWindowManager, AttentionMode, MAX_TOOL_RESULT_CHARS,
)
from zulong.l2.circuit_breaker import ToolCallCircuitBreaker, CircuitBreakerState
from zulong.l2.rule_guardian import RuleGuardian
from zulong.l2.unified_fc_runner import run_fc_loop
from zulong.l2.timeout_calibrator import TimeoutCalibrator
from zulong.l2.model_health_tracker import ModelHealthTracker, ModelHealthStatus
from zulong.l2.timeout_event_logger import TimeoutEventLogger
from zulong.l2.smart_degradation_handler import (
    SmartDegradationHandler, DegradationContext, TimeoutPhase,
)

# 导入配置管理器
from zulong.config.config_manager import get_l2_inference_config

import logging
logger = logging.getLogger(__name__)

# 🔥 LLM 后端支持（vLLM / Ollama / LM Studio / OpenAI 云端）
try:
    from openai import OpenAI
    VLLM_AVAILABLE = True
    
    from zulong.models.container import (
        LLM_BACKEND, LLM_BASE_URL, LLM_MODEL_ID, LLM_API_KEY,
        LLM_MODEL_ID_BACKUP, LLM_BASE_URL_BACKUP, LLM_API_KEY_BACKUP,
        VLLM_BASE_URL, LLM_NUM_CTX
    )
    logger.info(f"✅ [LLM] OpenAI SDK 已安装，后端: {LLM_BACKEND}")
    logger.info(f"🔧 [LLM] CORE 模型: {LLM_MODEL_ID} @ {LLM_BASE_URL}")
    logger.info(f"🔧 [LLM] BACKUP 模型: {LLM_MODEL_ID_BACKUP} @ {LLM_BASE_URL_BACKUP}")
except ImportError:
    VLLM_AVAILABLE = False
    LLM_BACKEND = None
    LLM_BASE_URL = None
    LLM_MODEL_ID = None
    LLM_API_KEY = "EMPTY"
    LLM_MODEL_ID_BACKUP = None
    LLM_BASE_URL_BACKUP = None
    LLM_API_KEY_BACKUP = "EMPTY"
    VLLM_BASE_URL = None
    LLM_NUM_CTX = 0
    logger.warning("⚠️ [LLM] OpenAI SDK 未安装，远程推理已禁用")

# 线程安全的 request_id 追踪（替代 self._current_request_id 避免竞态）
_current_request_id_var = contextvars.ContextVar('current_request_id', default=None)


class InferenceEngine:
    """L2 推理引擎（修复版）
    
    修复内容:
    1. ✅ 集成 RAG 记忆系统
    2. ✅ 优化生成参数（temperature, top_p, max_tokens）
    3. ✅ 维护对话历史记忆
    4. ✅ 支持上下文感知
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化推理引擎"""
        if not hasattr(self, '_initialized'):
            self._running = False
            self._current_task_id: Optional[str] = None
            self._interrupt_flag = False
            self._lock = threading.Lock()
            self._processing_done = threading.Event()
            self._processing_done.set()  # 初始为"完成"状态
            self._current_inference_task_id = None  # 当前推理对应的快照task_id
            self._initialized = True
            
            # 信息缺口检测器
            self._info_gap_detector = InformationGapDetector(confidence_threshold=0.6)
            
            # 注意力窗口管理器（FC 循环中按需初始化，此处仅声明）
            self._attn_window: Optional[AttentionWindowManager] = None
            # 远程模型默认 context window（优先使用配置的 num_ctx）
            self._context_window_size = LLM_NUM_CTX if LLM_NUM_CTX > 0 else 32768
            
            # 加载 L2 模型
            self.model_container = ModelContainer()
            self.l2_model = self.model_container.get_model(ModelID.L2_CORE)
            self._l2_loaded = self.l2_model is not None
            
            # 初始化 RAG 节点（记忆系统）
            self.rag_node = RAGIntegrationNode()
            
            # 加载 RAG 数据（从磁盘）
            self.rag_manager = self._load_rag_data()
            
            # 注入 RAGManager 到 RAG 节点
            if self.rag_manager:
                self.rag_node.set_rag_manager(self.rag_manager)
                # 注入 RAGManager 到 MemoryGraph（供 backend_ref 反查）
                try:
                    from zulong.memory.memory_graph import get_memory_graph
                    _mg = get_memory_graph()
                    if _mg:
                        _mg.set_rag_manager(self.rag_manager)
                except Exception:
                    pass
            
            # 近期轮次缓存（仅用于经验提取/纠错检测，NOT 用于 LLM 上下文注入）
            # LLM 上下文注入完全由 MemoryGraph.retrieve_context() 提供
            self._recent_turns_cache: List[Dict] = []
            self._recent_turns_max = 20  # 保留最近 10 轮 (20条消息)
            
            # 活跃任务图 ID 跟踪（用于地址继承）
            self._active_task_graph_id: Optional[str] = None
            
            # 经验生成器
            self.experience_generator = ExperienceGenerator()
            self.experience_generator.set_rag_manager(self.rag_manager)
            logger.info("[InferenceEngine] Experience generator initialized")
            
            # 🔥 修复 3.1: KV Cache 缓存池 (实现跨请求的 KV 状态复用)
            self.kv_cache_pool: Dict[str, Any] = {}  # session_id -> past_key_values
            self.kv_cache_ttl = 1800  # 30 分钟 TTL
            self.kv_cache_last_used: Dict[str, float] = {}  # session_id -> last_used_time
            logger.info("[InferenceEngine] KV Cache 缓存池已初始化")
            
            # 订阅系统 L2 命令事件 (唯一入口 - 所有事件都经过 L1-B 路由)
            event_bus.subscribe(
                EventType.SYSTEM_L2_COMMAND,
                self._on_l2_command,
                "InferenceEngine"
            )
            
            # 订阅系统中断事件
            event_bus.subscribe(
                EventType.SYSTEM_INTERRUPT,
                self._on_interrupt,
                "InferenceEngine"
            )
            
            # 视觉上下文缓存
            self._pending_visual_context: Optional[str] = None
            self._waiting_for_vision = False
            
            # 工具引擎
            self.tool_engine = ToolEngine()
            
            # 🔥 加载超时配置（通过 TimeoutCalibrator 校准）
            _l2_config = get_l2_inference_config()
            _timeout_config = _l2_config.get('timeout', {})
            self._calibrator = TimeoutCalibrator(_timeout_config)
            self._core_timeout = self._calibrator.core_timeout
            self._backup_timeout = self._calibrator.backup_timeout
            self._fc_loop_timeout = self._calibrator.fc_loop_timeout
            
            # 预加载完成标志（预加载完成后禁用延迟加载）
            self._preload_completed = False
            
            # 🔥 模型健康状态跟踪器
            self._health_tracker = ModelHealthTracker()
            
            # 🔥 超时事件结构化日志器
            self._event_logger = TimeoutEventLogger()
            
            # 🔥 智能降级处理器
            self._degradation_handler = SmartDegradationHandler()
            
            # 超时上下文（供降级处理器使用）
            self._last_timeout_phase: Optional[TimeoutPhase] = None
            self._last_timeout_elapsed: float = 0.0
            
            # FC 循环请求间隔（防止 API 被打满）
            self._fc_request_interval = float(_l2_config.get('request_interval', 1.0))
            
            # 🔥 新增：加载步数限制配置（替代时间超时机制）
            _step_config = _l2_config.get('step_limits', {})
            self._max_fc_turns = _step_config.get('max_fc_turns', 100)  # FC 循环最大步数
            self._soft_limit = _step_config.get('soft_limit', 50)  # 软限制
            self._hard_limit = _step_config.get('hard_limit', 100)  # 硬限制
            self._warning_interval = _step_config.get('warning_interval', 10)  # 警告间隔
            
            self._remote_tool_timeout = _timeout_config.get('remote_tool', 600)  # IDE模式远程工具等待超时
            
            # Circuit Breaker: 自适应迭代控制器
            _cb_config = _l2_config.get('circuit_breaker', {})
            self._circuit_breaker = ToolCallCircuitBreaker(_cb_config)
            logger.info(f"🔌 [L2] Circuit Breaker: enabled={self._circuit_breaker.enabled}, hard_cap={self._circuit_breaker.safety_hard_cap}")
            
            # RuleGuardian: 持久化实例（避免每次 eval_response 重建导致 retry_count 重置）
            self._rule_guardian = RuleGuardian(enabled=True)
            
            logger.info(f"⏱️ [L2] 超时配置: core={self._core_timeout}s, backup={self._backup_timeout}s, fc_loop={self._fc_loop_timeout}s, request_interval={self._fc_request_interval}s")
            logger.info(f"🔢 [L2] 步数配置: max={self._max_fc_turns}, soft={self._soft_limit}, hard={self._hard_limit}")
            
            # 注入 RAGManager 到 search_experience 工具（ExperienceRAG 被动检索）
            _exp_tool = self.tool_engine.registry.get("search_experience")
            if _exp_tool and self.rag_manager:
                _exp_tool.set_rag_manager(self.rag_manager)
            
            # 🔥 LLM 后端客户端（统一使用 OpenAI 兼容 API）
            if VLLM_AVAILABLE:
                try:
                    self.vllm_client = OpenAI(
                        base_url=LLM_BASE_URL,
                        api_key=LLM_API_KEY,
                    )
                    logger.info(f"✅ [LLM] CORE 客户端已初始化：{LLM_BACKEND} @ {LLM_BASE_URL}")
                except Exception as e:
                    logger.warning(f"⚠️ [LLM] CORE 客户端初始化失败：{e}，将使用本地模型")
                    self.vllm_client = None
                
                # 🔥 备用模型客户端（L2 BACKUP）
                # 当 CORE 与 BACKUP 使用相同端点时，复用同一客户端对象
                try:
                    if LLM_BASE_URL_BACKUP == LLM_BASE_URL and LLM_API_KEY_BACKUP == LLM_API_KEY:
                        self.backup_client = self.vllm_client
                        logger.info(f"✅ [LLM] BACKUP 复用 CORE 客户端（同一端点），模型: {LLM_MODEL_ID_BACKUP}")
                    else:
                        self.backup_client = OpenAI(
                            base_url=LLM_BASE_URL_BACKUP,
                            api_key=LLM_API_KEY_BACKUP,
                        )
                        logger.info(f"✅ [LLM] BACKUP 客户端已初始化：{LLM_BASE_URL_BACKUP}，模型: {LLM_MODEL_ID_BACKUP}")
                except Exception as e:
                    logger.warning(f"⚠️ [LLM] BACKUP 客户端初始化失败：{e}")
                    self.backup_client = None
            else:
                self.vllm_client = None
                self.backup_client = None
            
            # 🔧 显式调试日志：确认代码已加载
            logger.info("=" * 80)
            logger.info("🔧 InferenceEngine 初始化完成（FC 自主循环模式）")
            logger.info("🔧 已初始化工具引擎（FC 协议：tools= + tool_choice=auto）")
            logger.info(f"🔧 已注册工具：{list(self.tool_engine.registry.tools.keys())}")
            logger.info("=" * 80)

    
    def _ensure_l2_loaded(self):
        """确保 L2 模型已加载（延迟加载 / 预加载完成后直接返回）"""
        if self._preload_completed:
            return self.l2_model is not None
        if not self._l2_loaded:
            logger.info("🔄 延迟加载 L2 模型...")
            try:
                self.l2_model = self.model_container.get_model(ModelID.L2_CORE)
                self._l2_loaded = True
                logger.info("✅ L2 模型加载完成")
            except Exception as e:
                logger.warning(f"⚠️ L2 模型加载失败: {e}，使用降级模式")
                self.l2_model = None
                self._l2_loaded = False
        return self.l2_model is not None
    
    def hot_switch_llm(self, backend: str = None, model_id: str = None,
                       base_url: str = None, api_key: str = None) -> tuple:
        """运行时热切换 LLM 客户端
        
        Args:
            backend: 新后端名称（如 "ollama", "siliconflow"）。None 表示保持当前后端。
            model_id: 覆盖模型 ID。None 表示使用后端默认 model_id。
            base_url: 覆盖 API 地址。None 表示使用后端默认 base_url。
            api_key: 覆盖 API 密钥。None 表示使用后端默认 api_key。
        
        Returns:
            (success: bool, message: str)
        """
        from zulong.config.config_manager import get_config_manager
        cm = get_config_manager()
        
        # 确定目标后端
        target_backend = backend or cm.get('llm.backend', 'ollama')
        target_config = cm.get_dict(f'llm.{target_backend}', {})
        if not target_config:
            return False, f"后端 '{target_backend}' 未配置"
        
        target_base_url = base_url or target_config.get('base_url', '')
        target_model_id = model_id or target_config.get('model_id', '')
        target_api_key = api_key or target_config.get('api_key', 'EMPTY')
        target_num_ctx = int(target_config.get('num_ctx', 0))
        
        # 尝试创建新客户端
        try:
            from openai import OpenAI as _OpenAI
            new_client = _OpenAI(base_url=target_base_url, api_key=target_api_key)
        except Exception as e:
            return False, f"客户端创建失败: {e}"
        
        # 原子替换
        self.vllm_client = new_client
        
        # 更新全局变量（供其他模块引用）
        import zulong.models.container as _mc
        _mc.LLM_BACKEND = target_backend
        _mc.LLM_BASE_URL = target_base_url
        _mc.LLM_MODEL_ID = target_model_id
        _mc.LLM_API_KEY = target_api_key
        _mc.VLLM_BASE_URL = target_base_url
        _mc.LLM_NUM_CTX = target_num_ctx
        
        # 更新 context window size
        if target_num_ctx > 0:
            self._context_window_size = target_num_ctx
        
        # 更新配置文件（持久化）
        cm.config['llm']['backend'] = target_backend
        if model_id:
            cm.config['llm'][target_backend]['model_id'] = model_id
        if base_url:
            cm.config['llm'][target_backend]['base_url'] = base_url
        if api_key:
            cm.config['llm'][target_backend]['api_key'] = api_key
        
        # 同步更新环境覆盖配置，防止重启时被 _apply_environment_overrides 覆盖
        env = cm.environment
        env_cfg = cm.config.get('environments', {}).get(env, {})
        if env_cfg and 'llm' in env_cfg:
            env_cfg['llm']['backend'] = target_backend
        
        cm.save()
        
        logger.info(f"[LLM] 热切换完成: {target_backend} / {target_model_id} @ {target_base_url}")
        return True, f"已切换到 {target_backend} / {target_model_id}"
    
    def _get_llm_extra_kwargs(self) -> dict:
        """根据 LLM 后端类型返回额外参数
        
        vLLM 支持 extra_body（如 repetition_penalty），
        Ollama 支持 extra_body.options（如 num_ctx 控制上下文长度 / GPU 显存占用）。
        Qwen3 系列默认开启思维链，通过 enable_thinking=False 禁用。
        
        Returns:
            dict: 可直接解包到 chat.completions.create() 的额外参数
        """
        # 动态读取模块级变量（热切换后值会更新）
        import zulong.models.container as _mc
        backend = _mc.LLM_BACKEND
        num_ctx = _mc.LLM_NUM_CTX
        if backend == "vllm":
            return {"extra_body": {"repetition_penalty": 1.2, "enable_thinking": False}}
        if backend == "ollama" and num_ctx > 0:
            return {"extra_body": {"options": {"num_ctx": num_ctx}}}
        return {"extra_body": {"enable_thinking": False}}
    
    def _collect_tool_definitions(self) -> List[Dict[str, Any]]:
        """收集所有已启用工具的 OpenAI FC 格式定义
        
        遍历 ToolRegistry 中所有已注册且启用的工具，
        调用 get_function_schema() 生成 OpenAI Function Calling 格式。
        模型通过 tools= 参数接收这些定义，自主决定是否调用。
        
        Returns:
            List[Dict]: OpenAI tools 参数格式的工具定义列表
        """
        tool_definitions = []
        for name, tool in self.tool_engine.registry.tools.items():
            if not tool.enabled:
                continue
            try:
                schema = tool.get_function_schema()
                tool_definitions.append(schema)
            except Exception as e:
                logger.warning(f"[FC] 工具 {name} 的 schema 获取失败: {e}")
        logger.info(f"[FC] 收集到 {len(tool_definitions)} 个工具定义")
        return tool_definitions
    
    def _switch_graph_for_referenced_nodes(self):
        """根据引用节点地址切换活跃任务图

        当用户引用了某个节点（如 @[xxx#tg:tg_1777012532/req]），
        解析其中的 graph_id，如果与当前活跃图不同，则切换到对应任务图。
        必须在 _classify_intent() 之前调用，否则 _handle_resume 会盲目复用旧图。
        """
        _ref_nodes = getattr(self, '_referenced_nodes', [])
        if not _ref_nodes:
            return

        # 1. 从引用节点地址中提取 graph_id（格式: "tg:tg_XXXXX/node_id"）
        referenced_graph_ids = set()
        for ref in _ref_nodes:
            addr = ref.get('address', '') if isinstance(ref, dict) else str(ref)
            m = re.search(r'(tg_\d+)', addr)
            if m:
                referenced_graph_ids.add(m.group(1))

        if not referenced_graph_ids:
            return

        # 2. 检查是否与当前活跃图匹配
        from zulong.tools.task_tools import get_active_task_graph, set_active_task_graph
        current_tg = get_active_task_graph()
        current_graph_id = current_tg.id if current_tg else None

        # 取第一个引用的 graph_id（通常用户只引用同一个任务图的节点）
        target_graph_id = next(iter(referenced_graph_ids))

        if current_graph_id == target_graph_id:
            logger.info(f"[GraphSwitch] 引用的图 {target_graph_id} 与当前活跃图一致，无需切换")
            return

        logger.info(f"[GraphSwitch] 需要切换: 当前={current_graph_id} → 目标={target_graph_id}")

        # 3. 尝试从挂起任务中恢复目标任务图
        try:
            from zulong.l2.task_suspension import TaskSuspensionManager

            def _run_async_safe(coro):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None and loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        return pool.submit(asyncio.run, coro).result(timeout=10)
                else:
                    return asyncio.run(coro)

            mgr = TaskSuspensionManager()

            # 先挂起当前活跃图（如果有），防止丢失
            if current_tg is not None:
                from zulong.l2.task_suspension import SuspendableTaskState
                old_state = SuspendableTaskState(
                    task_id=TaskSuspensionManager.generate_task_id() if hasattr(TaskSuspensionManager, 'generate_task_id') else f"auto_{int(time.time())}",
                    description=current_tg.title or "未命名任务",
                    messages=[],
                    accumulated_links="",
                    circuit_breaker_state=self._circuit_breaker.serialize(),
                    iteration_count=0,
                    task_graph=current_tg,
                    suspended_reason="graph_switch",
                    metadata={"graph_id": current_graph_id or ""},
                )
                _run_async_safe(mgr.suspend_task(old_state))
                logger.info(f"[GraphSwitch] 已自动挂起当前图 {current_graph_id}")

            # 列出所有挂起任务，按 metadata.graph_id 查找目标
            suspended = _run_async_safe(mgr.list_suspended_tasks())
            target_task_id = None
            for task in suspended:
                meta = task.get("metadata", {})
                if meta.get("graph_id") == target_graph_id:
                    target_task_id = task["task_id"]
                    break

            if target_task_id:
                state = _run_async_safe(mgr.resume_task(target_task_id))
                if state and state.task_graph:
                    set_active_task_graph(state.task_graph, target_graph_id)
                    logger.info(f"[GraphSwitch] 已从挂起任务恢复图 {target_graph_id} (task_id={target_task_id})")
                    return
                else:
                    logger.warning(f"[GraphSwitch] 挂起任务 {target_task_id} 无有效 TaskGraph")

            # 4. 挂起任务中未找到 → 尝试从磁盘备份恢复
            from zulong.tools.task_tools import load_graph_from_backup
            backup_tg = load_graph_from_backup(target_graph_id)
            if backup_tg:
                set_active_task_graph(backup_tg, target_graph_id)
                logger.info(f"[GraphSwitch] 已从磁盘备份恢复图 {target_graph_id}")
                return

            # 5. 所有来源都找不到 → 标记图丢失，供后续流程处理
            set_active_task_graph(None, None)
            self._referenced_graph_lost = target_graph_id
            logger.warning(f"[GraphSwitch] 任务图 {target_graph_id} 数据已丢失（挂起任务和磁盘备份中均未找到）")

        except Exception as e:
            logger.error(f"[GraphSwitch] 切换失败: {e}", exc_info=True)

    def _classify_intent(self, user_input: str):
        """两阶段 FC 意图分类 — Round 1
        
        使用极简提示词 + start_session 工具 + tool_choice=required
        强制模型输出意图分类，然后执行对应的骨架操作。
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            Tuple[IntentType, dict]: (意图类型, scaffold 操作返回的数据)
        """
        import json
        import concurrent.futures
        from zulong.l2.intent_prompt_builder import (
            IntentType, build_round1_system_prompt, get_round1_tools
        )
        from zulong.tools.session_tool import StartSessionTool
        from zulong.tools.base import ToolRequest
        
        # 如果没有远程模型客户端，跳过 Round 1，默认 CHAT
        if not self.vllm_client:
            logger.info("[Intent] vllm_client 不可用，跳过 Round 1，默认 CHAT")
            return IntentType.CHAT, {}
        
        try:
            # ── MemoryGraph: 检索相关历史经验注入意图分类 ──
            _history_hint = ""
            try:
                _mg_intent = self._get_memory_graph_safe()
                logger.info(f"[Intent] MemoryGraph 历史检索前置检查: mg={'有' if _mg_intent else '无'}")
                if _mg_intent:
                    import asyncio
                    _loop = asyncio.new_event_loop()
                    try:
                        _ctx_results = _loop.run_until_complete(
                            _mg_intent.retrieve_context(str(user_input), top_k=3)
                        )
                    finally:
                        _loop.close()
                    if _ctx_results:
                        _parts = []
                        for _cr in _ctx_results[:3]:
                            _lbl = _cr.get("label", "")
                            _cnt = (_cr.get("content", "") or "")[:150]
                            if _lbl:
                                _parts.append(f"- {_lbl}: {_cnt}")
                        if _parts:
                            _history_hint = "\n\n相关历史经验:\n" + "\n".join(_parts)
            except Exception as _ctx_err:
                logger.info(f"[Intent] MemoryGraph 检索跳过: {_ctx_err}")

            # 构建 Round 1 消息
            _user_content = str(user_input) + _history_hint
            round1_messages = [
                {"role": "system", "content": build_round1_system_prompt()},
                {"role": "user", "content": _user_content},
            ]
            
            round1_tools = get_round1_tools()
            
            # 调用 LLM API（强制调用 start_session）
            api_kwargs = {
                "model": LLM_MODEL_ID,
                "messages": round1_messages,
                "tools": round1_tools,
                "tool_choice": {"type": "function", "function": {"name": "start_session"}},
                "max_tokens": 256,
                "temperature": 0.1,
                "stream": False,
                **self._get_llm_extra_kwargs(),
            }
            
            def _call(kwargs=api_kwargs):
                return self.vllm_client.chat.completions.create(**kwargs)
            
            api_response = None
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_call)
            try:
                api_response = future.result(timeout=15)
            except Exception as core_err:
                logger.warning(f"[Intent] Round 1 CORE 模型失败: {core_err}")
                # 尝试备用模型进行意图分类
                if self.backup_client and LLM_MODEL_ID_BACKUP:
                    logger.info(f"[Intent] 尝试使用备用模型 {LLM_MODEL_ID_BACKUP} 进行意图分类")
                    backup_kwargs = {
                        "model": LLM_MODEL_ID_BACKUP,
                        "messages": round1_messages,
                        "tools": round1_tools,
                        "tool_choice": {"type": "function", "function": {"name": "start_session"}},
                        "max_tokens": 256,
                        "temperature": 0.1,
                        "stream": False,
                        **self._get_llm_extra_kwargs(),
                    }
                    try:
                        api_response = self.backup_client.chat.completions.create(**backup_kwargs)
                        logger.info(f"[Intent] 备用模型意图分类成功")
                    except Exception as backup_err:
                        logger.warning(f"[Intent] 备用模型也失败: {backup_err}，默认 CHAT")
                        return IntentType.CHAT, {}
                else:
                    logger.warning("[Intent] 无可用备用模型，默认 CHAT")
                    return IntentType.CHAT, {}
            finally:
                executor.shutdown(wait=False)
            
            # 解析 tool_call
            msg = api_response.choices[0].message
            if not msg.tool_calls:
                logger.warning("[Intent] Round 1 未返回 tool_call，尝试启发式分类")
                # 启发式回退：当 4B 模型未遵循 tool_choice=required 时
                _heuristic = self._heuristic_intent_classify(user_input)
                logger.info(f"[Intent] 启发式分类结果: {_heuristic}")
                # 启发式分类也需要执行骨架操作（创建/挂起任务图等）
                _heuristic_intent_str = _heuristic.value  # "chat" / "complex" / "resume"
                if _heuristic_intent_str != "chat":
                    from zulong.tools.session_tool import StartSessionTool
                    from zulong.tools.base import ToolRequest as _TR
                    _h_session = StartSessionTool()
                    _h_request = _TR(
                        tool_name="start_session",
                        action="execute",
                        parameters={
                            "intent": _heuristic_intent_str,
                            "reason": "启发式分类回退",
                            "task_description": user_input[:100],
                            "user_input": user_input,
                        },
                    )
                    _h_result = _h_session.execute(_h_request)
                    _h_scaffold = _h_result.data if _h_result.success and _h_result.data else {}
                    logger.info(f"[Intent] 启发式路径已执行骨架操作: {_h_scaffold.get('message', '')}")
                    # 检测 RESUME 降级
                    if _h_scaffold.get("fallback"):
                        logger.info(f"[Intent] 启发式 RESUME 降级为 CHAT: {_h_scaffold.get('message', '')}")
                        return IntentType.CHAT, _h_scaffold
                    return _heuristic, _h_scaffold
                return _heuristic, {}
            
            tc = msg.tool_calls[0]
            args = json.loads(tc.function.arguments)
            intent_str = args.get("intent", "chat")
            reason = args.get("reason", "")
            task_description = args.get("task_description", "")
            
            logger.info(f"[Intent] Round 1 分类结果: intent={intent_str}, reason={reason}, task_desc={task_description}")
            
            # 映射为 IntentType
            intent_map = {"chat": IntentType.CHAT, "complex": IntentType.COMPLEX, "resume": IntentType.RESUME}
            intent_type = intent_map.get(intent_str, IntentType.CHAT)
            
            # 执行 StartSessionTool 骨架操作
            session_tool = StartSessionTool()
            tool_request = ToolRequest(
                tool_name="start_session",
                action="execute",
                parameters={
                    "intent": intent_str,
                    "reason": reason,
                    "task_description": task_description,
                    "user_input": user_input,  # Rule B: 传递原始用户输入
                },
            )
            result = session_tool.execute(tool_request)
            scaffold_data = result.data if result.success and result.data else {}
            
            # 检测 RESUME 降级：如果未找到挂起任务，降级为 CHAT
            if scaffold_data.get("fallback"):
                logger.info(f"[Intent] RESUME 降级为 CHAT: {scaffold_data.get('message', '')}")
                intent_type = IntentType.CHAT
            
            # 如果 StartSessionTool 返回的 intent 与模型分类不同（如 RESUME 降级）
            actual_intent = scaffold_data.get("intent", intent_str)
            if actual_intent != intent_str:
                intent_type = intent_map.get(actual_intent, intent_type)
            
            return intent_type, scaffold_data
            
        except Exception as e:
            logger.warning(f"[Intent] Round 1 分类失败: {e}，尝试启发式分类", exc_info=True)
            _heuristic = self._heuristic_intent_classify(user_input)
            return _heuristic, {}
    
    @staticmethod
    def _heuristic_intent_classify(user_input: str):
        """启发式意图分类 — LLM 回退方案
        
        当 LLM Round 1 未返回有效 tool_call 时，
        使用关键词匹配进行基础分类。
        """
        from zulong.l2.intent_prompt_builder import IntentType
        
        text = user_input.strip()
        
        # RESUME 信号
        resume_signals = ["继续", "接着做", "恢复", "上次那个", "接着", "继续做", "回到"]
        for s in resume_signals:
            if s in text:
                return IntentType.RESUME
        
        # CHAT 优先信号 — 记忆/记录类请求（save_memory_note 即可处理）
        memory_signals = ["记住", "记下", "记录", "保存", "备忘", "帮我记", "存一下"]
        for s in memory_signals:
            if s in text:
                return IntentType.CHAT
        
        # COMPLEX 信号：多部分请求（含连接词 + 动作动词）
        complex_verbs = ["分析", "对比", "列出", "设计", "开发", "编写", "写一篇",
                         "做一个", "帮我做", "帮我写", "帮我设计", "创建", "搭建",
                         "实现", "规划", "制定", "总结", "归纳"]
        multi_signals = ["同时", "另外", "以及", "还要", "此外", "再加上"]
        
        has_verb = any(v in text for v in complex_verbs)
        has_multi = any(m in text for m in multi_signals)
        
        # 有任务动词 + 多部分信号 → COMPLEX
        if has_verb and has_multi:
            return IntentType.COMPLEX
        
        # 有强任务动词（帮我做/写/设计/开发/创建/搭建/实现）→ COMPLEX
        strong_verbs = ["帮我做", "帮我写", "帮我设计", "帮我开发", "帮我创建",
                        "帮我搭建", "帮我实现", "帮我规划", "帮我制定"]
        if any(v in text for v in strong_verbs):
            return IntentType.COMPLEX
        
        # 输入较长且包含任务动词（动词须在前 40 字符内，表示指令而非内容描述）
        if len(text) > 80 and has_verb:
            first_verb_pos = min(
                (text.find(v) for v in complex_verbs if v in text), default=999
            )
            if first_verb_pos < 40:
                return IntentType.COMPLEX
        
        return IntentType.CHAT
    
    def _collect_tool_definitions_for_intent(self, intent_type) -> List[Dict[str, Any]]:
        """根据意图类型收集过滤后的工具定义
        
        CHAT: 仅对话相关工具（8个）
        COMPLEX: 全部工具（不过滤）
        RESUME: 仅任务恢复相关工具（6个，物理排除 task_create_plan 和 task_add_node）
        
        Args:
            intent_type: IntentType 枚举值
            
        Returns:
            List[Dict]: 过滤后的 OpenAI FC 工具定义列表
        """
        from zulong.l2.intent_prompt_builder import get_round2_tool_names
        
        allowed_names = get_round2_tool_names(intent_type)
        
        # None 表示不过滤（COMPLEX 场景使用全部工具）
        if allowed_names is None:
            return self._collect_tool_definitions()
        
        tool_definitions = []
        for name, tool in self.tool_engine.registry.tools.items():
            if not tool.enabled:
                continue
            if name not in allowed_names:
                continue
            try:
                schema = tool.get_function_schema()
                tool_definitions.append(schema)
            except Exception as e:
                logger.warning(f"[FC] 工具 {name} 的 schema 获取失败: {e}")
        
        logger.info(f"[FC] 为 {intent_type.value} 场景收集到 {len(tool_definitions)} 个工具定义 (过滤自 {len(allowed_names)} 个允许)")
        return tool_definitions
    
    def _execute_tool_call(self, tool_call) -> str:
        """执行单个 FC 工具调用并返回结果文本
        
        Args:
            tool_call: OpenAI API 返回的 tool_call 对象，
                       包含 id, function.name, function.arguments
        
        Returns:
            str: 工具执行结果的文本表示
        """
        import json
        
        func_name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments)
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"[FC] 工具 {func_name} 参数解析失败: {e}")
            return json.dumps({"error": f"参数解析失败: {e}"}, ensure_ascii=False)
        
        logger.info(f"[FC] 执行工具: {func_name}, 参数: {arguments}")
        
        try:
            # 从 arguments 中提取 action 参数，如果未提供则默认为 "execute"
            action = arguments.pop("action", "execute")
            
            # ── 守卫 D: 嵌套参数解包 ──
            # 4B 模型可能将实际参数包裹在 "args" 键内，如:
            #   {"action": "call_tool", "args": {"label": "...", ...}, "tool_name": "xxx"}
            # 需要将 args 内容提取为顶层参数，否则工具读不到真正的字段
            if "args" in arguments and isinstance(arguments["args"], dict):
                inner = arguments["args"]
                arguments.pop("tool_name", None)  # 清除冗余 tool_name
                arguments = inner
                logger.debug(
                    f"[FC] 嵌套参数解包: {func_name} → {arguments}"
                )
            
            result = self.tool_engine.call_tool(
                tool_name=func_name,
                action=action,
                parameters=arguments,
                timeout=30.0,
            )
            
            if result.success:
                data = result.data
                
                # 地址继承：当 task_create_plan 成功时，记录活跃任务图 ID
                if func_name == "task_create_plan" and isinstance(data, dict):
                    graph_id = data.get("graph_id")
                    if graph_id:
                        self._active_task_graph_id = graph_id
                        logger.info(
                            f"[地址继承] task_create_plan 成功，"
                            f"记录活跃任务图 ID: {graph_id}"
                        )
                
                if isinstance(data, str):
                    return data
                return json.dumps(data, ensure_ascii=False, default=str)
            else:
                return json.dumps(
                    {"error": result.error or "工具执行失败"},
                    ensure_ascii=False,
                )
        except Exception as e:
            logger.error(f"[FC] 工具 {func_name} 执行异常: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)
    
    def _publish_task_graph_event(
        self,
        pipeline_type: str,
        fc_turn: int = 0,
        tool_name: str = "",
        tool_result: str = "",
    ) -> None:
        """发布任务图谱更新事件到前端
        
        将当前活跃 TaskGraph 转换为前端 addTaskGraph() 兼容格式，
        通过 L2_THINKING_STEP 消息推送。
        
        Args:
            pipeline_type: pipeline 子类型，如 "agent_tool_call", "agent_done", "pipeline_start"
            fc_turn: FC 轮次
            tool_name: 当前调用的工具名
            tool_result: 工具执行结果摘要
        """
        try:
            from zulong.tools.task_tools import get_active_task_graph
            tg = get_active_task_graph()
            if not tg:
                logger.info(f"[图谱推送] 无活跃任务图，跳过 {pipeline_type}")
                return

            logger.info(f"[图谱推送] 开始处理 {pipeline_type}, 图谱ID={tg.id}, 节点数={len(tg._nodes)}")

            # 转换为前端兼容格式
            try:
                graph_data = self._task_graph_to_frontend(tg)
            except Exception as serialize_err:
                logger.warning(f"[图谱推送] 图谱序列化失败(节点数={len(tg._nodes)}): {serialize_err}")
                graph_data = {
                    "id": tg.id,
                    "title": getattr(tg, 'title', ''),
                    "nodes": [],
                    "hEdges": [],
                    "dEdges": [],
                    "serialize_error": str(serialize_err),
                }

            event_data = {
                "graph": graph_data,
                "turn": fc_turn,
                "tool": tool_name,
                "tool_count": len(tg._nodes),
            }

            # 添加额外上下文
            if pipeline_type == "agent_done":
                event_data["duration"] = 0  # 由调用方填充
            if tool_result:
                event_data["tool_result"] = tool_result[:500]

            step_type = f"pipeline.{pipeline_type}"
            self._send_thinking_step(step_type, event_data)

            logger.info(
                f"[图谱推送] {pipeline_type} → {step_type}, "
                f"节点数: {len(tg._nodes)}, 边数: {len(tg._h_edges) + len(tg._d_edges)}"
            )
        except Exception as e:
            import traceback
            logger.warning(f"[图谱推送] {pipeline_type} 失败: {e}\n{traceback.format_exc()}")
    
    def _task_graph_to_frontend(self, tg) -> Dict:
        """将 TaskGraph 转换为前端 addTaskGraph() 兼容格式
        
        直接复用 TaskGraph.to_frontend_dict()，确保 hEdges/dEdges 完整传递，
        前端 dagre 布局依赖 hEdges 建立层级关系。
        """
        # 使用 TaskGraph 自身的序列化方法，保证 nodes/hEdges/dEdges 完整
        frontend_data = tg.to_frontend_dict()
        
        # 补充 activeNodeId（前端用于高亮当前节点）
        active_node_id = "req"
        for nid, task_node in tg._nodes.items():
            if task_node.status == "in_progress":
                active_node_id = nid
                break
        frontend_data["activeNodeId"] = active_node_id
        
        return frontend_data
    
    def _send_thinking_step(self, step_type: str, data: Dict) -> None:
        """发送 L2_THINKING_STEP 事件到 EventBus

        该事件最终会被 WebSocket 桥转发给前端。
        前端 handleThinkingStep() 期望 payload 包含 request_id, step_type, data。
        """
        try:
            from zulong.core.types import EventType, EventPriority, ZulongEvent

            # 获取当前请求 ID
            try:
                request_id = _current_request_id_var.get() or f"req_{int(time.time() * 1000)}"
            except Exception:
                request_id = f"req_{int(time.time() * 1000)}"

            payload = {
                "request_id": request_id,
                "step_type": step_type,
                "data": data,
                "timestamp": time.time(),
                "iteration": data.get("turn", 0),
            }

            event = ZulongEvent(
                type=EventType.L2_THINKING_STEP,
                priority=EventPriority.NORMAL,
                source="InferenceEngine",
                payload=payload,
            )
            logger.info(f"[图谱推送] 即将发布 {step_type}, request_id={request_id}")
            event_bus.publish(event)
            logger.info(f"[图谱推送] 已发布 {step_type}, request_id={request_id}")
        except Exception as e:
            import traceback
            logger.error(f"[图谱推送] 发布事件失败: {e}\n{traceback.format_exc()}")
    
    async def _generate_with_vllm(self, messages: List[Dict[str, str]]) -> str:
        """使用 vLLM OpenAI API 生成响应（不支持工具调用）
        
        Args:
            messages: 对话历史消息列表
            
        Returns:
            生成的响应文本
        """
        try:
            logger.info(f"🚀 [vLLM] 开始调用 OpenAI API...")
            logger.info(f"🚀 [vLLM] 消息数：{len(messages)}")
            
            vllm_model_id = LLM_MODEL_ID
            
            # 🔥 新增：超时保护（30 秒）
            import concurrent.futures
            
            def call_vllm():
                return self.vllm_client.chat.completions.create(
                    model=vllm_model_id,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.3,
                    top_p=0.85,
                    stream=False,
                    **self._get_llm_extra_kwargs()
                )
            
            # 🔥 关键修复：不使用 with 语句，避免 __exit__ 调用 shutdown(wait=True) 阻塞超时返回
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(call_vllm)
            try:
                response = future.result(timeout=self._core_timeout)  # 从配置读取超时
            except concurrent.futures.TimeoutError:
                logger.error(f"🚨 [vLLM] CORE 模型超时 (>{self._core_timeout} 秒)，尝试备用模型...")
                # 🔥 降级路径：CORE 超时 → BACKUP 模型 → 静态降级
                user_input = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        user_input = msg.get("content", "")
                        break
                return await self._generate_with_backup(messages, user_input)
            finally:
                executor.shutdown(wait=False)
            
            # 提取响应
            response_text = response.choices[0].message.content
            logger.info(f"✅ [vLLM] 生成完成，长度：{len(response_text)}")
            logger.info(f"✅ [vLLM] 响应前 100 字符：{response_text[:100]}...")
            
            return response_text
            
        except Exception as e:
            logger.error(f"❌ [vLLM] CORE 生成失败：{e}，尝试备用模型...")
            # 降级路径：CORE 异常 → BACKUP 模型
            user_input = ""
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_input = msg.get("content", "")
                    break
            return await self._generate_with_backup(messages, user_input)
    
    def _get_memory_graph_safe(self):
        """安全获取 MemoryGraph 实例，失败返回 None"""
        try:
            from zulong.memory.memory_graph import get_memory_graph
            return get_memory_graph()
        except Exception:
            return None

    def _get_fallback_response(self, user_input: str) -> str:
        """根据用户输入类型提供智能降级回复
        
        使用 SmartDegradationHandler 生成自然语言降级消息，
        同时通过 TimeoutEventLogger 输出结构化诊断日志。
        
        Args:
            user_input: 用户的原始输入文本
            
        Returns:
            降级回复文本
        """
        timeout_phase = self._last_timeout_phase or TimeoutPhase.CORE_TIMEOUT
        elapsed = self._last_timeout_elapsed
        model_id = "CORE" if timeout_phase == TimeoutPhase.CORE_TIMEOUT else "BACKUP"
        request_id = _current_request_id_var.get()
        
        context = DegradationContext(
            timeout_phase=timeout_phase,
            elapsed_seconds=elapsed,
            model_id=model_id,
            user_input=user_input or "",
            request_id=request_id,
        )
        
        response = self._degradation_handler.generate_response(context)
        self._degradation_handler.generate_diagnostic_log(context)
        
        return response

    async def _generate_with_backup(self, messages: List[Dict[str, str]], user_input: str = "") -> str:
        """使用备用模型（L2 BACKUP）生成响应
        
        当主模型（L2 CORE）超时或不可用时，使用本地备用模型生成回复。
        备用模型通常更小更快（如 qwen3.5:4b 本地），牺牲质量换取可用性。
        
        Args:
            messages: 对话历史消息列表
            user_input: 用户原始输入（用于日志和最终降级）
            
        Returns:
            生成的响应文本
        """
        if not self.backup_client or not LLM_MODEL_ID_BACKUP:
            logger.warning("⚠️ [BACKUP] 备用客户端不可用，使用静态降级回复")
            self._last_timeout_phase = TimeoutPhase.BACKUP_UNAVAILABLE
            self._event_logger.log_degradation_decision(
                "fallback_static", "backup_client_unavailable", "BACKUP",
                _current_request_id_var.get())
            return self._get_fallback_response(user_input)
        
        # 跳过备用模型调用：如果 CORE 和 BACKUP 是同一个模型，直接降级
        if LLM_MODEL_ID_BACKUP == LLM_MODEL_ID and LLM_BASE_URL_BACKUP == LLM_BASE_URL:
            logger.info("⚠️ [BACKUP] CORE 与 BACKUP 是同一模型，跳过备用调用")
            self._last_timeout_phase = TimeoutPhase.CORE_BACKUP_SAME_MODEL
            self._event_logger.log_degradation_decision(
                "fallback_static", "core_backup_same_model", "BACKUP",
                _current_request_id_var.get())
            return self._get_fallback_response(user_input)
        
        # 检查 BACKUP 健康状态
        if self._health_tracker.should_skip("BACKUP"):
            logger.info("⚠️ [BACKUP] 跳过BACKUP请求：健康状态UNAVAILABLE")
            self._last_timeout_phase = TimeoutPhase.BACKUP_UNAVAILABLE
            self._event_logger.log_degradation_decision(
                "skip_backup", "health_unavailable", "BACKUP",
                _current_request_id_var.get())
            return self._get_fallback_response(user_input)
        
        try:
            logger.info(f"🔄 [BACKUP] 切换到备用模型：{LLM_MODEL_ID_BACKUP} @ {LLM_BASE_URL_BACKUP}")
            
            import concurrent.futures
            
            def call_backup():
                return self.backup_client.chat.completions.create(
                    model=LLM_MODEL_ID_BACKUP,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.3,
                    top_p=0.85,
                    stream=False,
                    **self._get_llm_extra_kwargs()
                )
            
            _backup_start = time.time()
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(call_backup)
            try:
                response = future.result(timeout=self._backup_timeout)
            except concurrent.futures.TimeoutError:
                _elapsed = time.time() - _backup_start
                self._last_timeout_phase = TimeoutPhase.BACKUP_TIMEOUT
                self._last_timeout_elapsed = _elapsed
                self._health_tracker.record_timeout("BACKUP")
                self._event_logger.log_timeout(
                    "BACKUP_TIMEOUT", _elapsed, self._backup_timeout, "BACKUP",
                    _current_request_id_var.get(),
                    self._health_tracker.get_consecutive_timeouts("BACKUP"))
                self._event_logger.log_degradation_decision(
                    "fallback_static", "backup_timeout", "BACKUP",
                    _current_request_id_var.get())
                logger.error(f"🚨 [BACKUP] 备用模型也超时 (>{self._backup_timeout}秒)")
                return self._get_fallback_response(user_input)
            finally:
                executor.shutdown(wait=False)
            
            response_text = response.choices[0].message.content
            self._health_tracker.record_success("BACKUP")
            logger.info(f"✅ [BACKUP] 备用模型生成完成，长度：{len(response_text)}")
            response_text = self._degradation_handler.append_backup_hint(response_text)
            return response_text
            
        except Exception as e:
            self._health_tracker.record_timeout("BACKUP")
            self._last_timeout_phase = TimeoutPhase.BACKUP_TIMEOUT
            self._event_logger.log_degradation_decision(
                "fallback_static", f"backup_exception: {e}", "BACKUP",
                _current_request_id_var.get())
            logger.error(f"❌ [BACKUP] 备用模型生成失败：{e}")
            return self._get_fallback_response(user_input)

    # _detect_needs_tools 已移除：不再使用关键词预路由
    # 工具选择完全由 LLM 通过 Function Calling (tool_choice="auto") 自主判断
    # 参见: docs/Function_Calling架构改进执行方案.md 任务2

    def _on_l2_command(self, event: ZulongEvent):
        """处理 L2 命令事件 (唯一入口 - 所有事件都经过 L1-B 路由)
        
        Args:
            event: L2 命令事件
        """
        # 🔥 检查复盘分析命令
        command = event.payload.get("command", "")
        
        if command == "analyze_for_review":
            logger.info(f"🧠 [复盘] 收到复盘分析命令")
            prompt = event.payload.get("prompt", "")
            session_id = event.payload.get("session_id", "")
            deep_analysis = event.payload.get("deep_analysis", False)
            expect_json = event.payload.get("expect_json_response", False)
            
            logger.info(f"🧠 [复盘] session_id={session_id}, deep={deep_analysis}, expect_json={expect_json}")
            
            # 🔥 调用复盘分析方法（在线程中执行）
            threading.Thread(
                target=self._analyze_for_review,
                args=(prompt, session_id, deep_analysis, expect_json),
                daemon=True
            ).start()
            return
        
        # 🔥 恢复非 Orchestrator 机制：处理普通用户命令
        self._is_resume_task = False  # 每次新命令重置恢复标记
        text = event.payload.get("text", "")
        is_wakeup_command = event.payload.get("wakeup_command", False)
        visual_attention = event.payload.get("visual_attention", False)
        audio_attention = event.payload.get("audio_attention", False)
        sensor_type = event.payload.get("sensor_type", "")
        emergency = event.payload.get("emergency", False)
        vision_ready = event.payload.get("vision_ready", False)
        suspended_task = event.payload.get("suspended_task", None)
        
        # 缓存 GK 传来的节点信息，供 _update_memory() 复用
        self._current_dialogue_round_id = event.payload.get("dialogue_round_id", None)
        
        if is_wakeup_command:
            logger.info(f"🧠 收到唤醒命令：'{text}'")
            time.sleep(1.5)
            state_manager.set_l2_status(L2Status.IDLE)
        
        if emergency:
            logger.warning(f"🚨 [L2] 收到紧急事件: {sensor_type}")
        
        if visual_attention:
            intent_type = event.payload.get("intent_type", "unknown")
            intent_confidence = event.payload.get("intent_confidence", 0)
            person_distance = event.payload.get("person_distance", float('inf'))
            keyframe_b64 = event.payload.get("keyframe_b64")
            crop_b64 = event.payload.get("crop_b64")
            
            if intent_confidence is None:
                intent_confidence = 0.0
            if person_distance is None:
                person_distance = float('inf')
            
            logger.info(
                f"🧠 收到视觉注意力命令：intent={intent_type}, "
                f"confidence={intent_confidence:.2f}, distance={person_distance:.2f}m"
            )
            logger.info(
                f"📸 关键帧状态：keyframe={keyframe_b64 is not None}, crop={crop_b64 is not None}"
            )
            
            visual_context = self._build_visual_context_from_l1c(
                intent_type, intent_confidence, person_distance, keyframe_b64, crop_b64
            )
            
            self._pending_visual_context = visual_context
        
        if audio_attention:
            logger.info(f"🎙️ [L2] 收到音频注意力事件")
        
        if vision_ready:
            video_path = event.payload.get("video_path", "")
            logger.info(f"📹 [L2] 视觉数据就绪: {video_path}")
        
        if sensor_type:
            logger.info(f"📡 [L2] 收到传感器事件: {sensor_type}")
        
        logger.info(f"🧠 收到 L2 命令：'{text}'")
        
        voice_mode = event.payload.get("voice_mode", "TEXT_ONLY")
        if emergency:
            voice_mode = "AUTO_TTS"
        logger.info(f"🎙️ [L2] Voice Mode: {voice_mode}")
        
        # 🎯 提取 L1-B 预分类意图（如果存在则跳过 Round 1）
        pre_classified_intent = event.payload.get("pre_classified_intent")
        if pre_classified_intent:
            logger.info(f"🎯 [L2] 使用 L1-B 预分类意图: {pre_classified_intent} (跳过 Round 1)")
        else:
            logger.debug("[L2] 无预分类意图，将执行 Round 1 分类")
        
        # 🎯 提取 L1-B 预检索的记忆上下文（如果存在则跳过重复检索）
        pre_retrieved_memory = event.payload.get("pre_retrieved_memory")
        if pre_retrieved_memory:
            logger.info(f"🧠 [L2] 使用 L1-B 预检索记忆上下文 ({len(pre_retrieved_memory)} 字符)")
        else:
            logger.debug("[L2] 无预检索记忆，将执行 MemoryGraph 检索")
        
        # 提取用户引用的节点（前端右键引用 → WebSocket → EventBus）
        self._referenced_nodes = event.payload.get("referenced_nodes", [])
        if self._referenced_nodes:
            logger.info(f"[L2] 用户引用了 {len(self._referenced_nodes)} 个节点: {self._referenced_nodes}")
        
        # Phase A1b: 处理恢复事件（Phase C 发布的）
        resume_task_id = event.payload.get("_resume_task_id")
        
        # 等待上一轮推理结束（被中断后最多等10秒）
        if not self._processing_done.wait(timeout=10):
            logger.warning("[L2] 等待上一轮推理超时，强制开始新任务")
            # 强制清理上一轮的生成状态，避免并发锁残留
            try:
                from zulong.l2.interrupt_handler import interrupt_handler as _ih_cleanup
                _ih_cleanup.stop_generation()
            except Exception:
                pass
            try:
                from zulong.core.state_manager import state_manager as _sm_cleanup
                _sm_cleanup.clear_task()
            except Exception:
                pass
        
        if resume_task_id:
            threading.Thread(
                target=self._process_resume_task,
                args=(resume_task_id,),
                daemon=True,
                name="L2-FC-Resume"
            ).start()
        else:
            threading.Thread(
                target=self._process_with_memory,
                args=(text, event.priority, voice_mode, pre_classified_intent, pre_retrieved_memory),
                daemon=True,
                name="L2-FC-Worker"
            ).start()

    def _analyze_for_review(self, prompt: str, session_id: str, deep_analysis: bool, expect_json: bool):
        """执行复盘分析（同步方法，在线程中调用）
        
        Args:
            prompt: 分析提示
            session_id: 会话 ID
            deep_analysis: 是否深度分析
            expect_json: 是否期望 JSON 响应
        """
        logger.info(f"🧠 [复盘] 开始分析，prompt_length={len(prompt)}")
        
        try:
            # 🔥 v3.0 新增：设置状态为 REVIEW_ANALYZING
            state_manager.set_l2_status(L2Status.REVIEW_ANALYZING, task_id=session_id)
            logger.info(f"🧠 [复盘] L2 状态已设置为 REVIEW_ANALYZING")
            
            # 1. 确保 L2 模型已加载
            if not self._ensure_l2_loaded():
                logger.error("🧠 [复盘] L2 模型加载失败")
                self._publish_review_response(session_id, "错误：L2 模型加载失败", deep_analysis)
                state_manager.set_l2_status(L2Status.IDLE)
                return
            
            # 2. 构建对话历史（复盘分析不需要历史）
            messages = [
                {"role": "system", "content": "你是一个专业的复盘分析助手。你的任务是根据提供的对话内容，进行深度分析并总结经验教训。"},
                {"role": "user", "content": prompt}
            ]
            
            # 3. 应用对话模板
            if hasattr(self.l2_model, 'tokenizer') and hasattr(self.l2_model.tokenizer, 'apply_chat_template'):
                prompt_text = self.l2_model.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # 降级：直接使用 prompt
                prompt_text = prompt
            
            # 4. 生成响应
            logger.info(f"🧠 [复盘] 开始生成响应...")
            
            # 使用 generate 方法（同步）
            inputs = self.l2_model.tokenizer(prompt_text, return_tensors="pt")
            input_length = inputs.input_ids.shape[1]
            
            # 移动到正确的设备
            device = self.l2_model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 生成参数
            max_new_tokens = 2048 if deep_analysis else 1024
            temperature = 0.3  # 🔥 降低温度，提高回复质量
            top_p = 0.85       # 🔥 调整 top_p
            
            logger.info(f"🧠 [复盘] 生成参数：max_new_tokens={max_new_tokens}, temperature={temperature}")
            
            # 生成
            import torch
            with torch.no_grad():
                outputs = self.l2_model.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.l2_model.tokenizer.eos_token_id
                )
            
            # 提取生成的文本
            generated_ids = outputs[0][input_length:]
            response = self.l2_model.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            logger.info(f"🧠 [复盘] 生成完成，response_length={len(response)}")
            
            # 5. 发布 L2_OUTPUT 事件
            self._publish_review_response(session_id, response, deep_analysis)
            
            # 🔥 v3.0 新增：分析完成后回到 IDLE 状态
            state_manager.clear_task()
            logger.info(f"🧠 [复盘] 分析完成，L2 状态已设置为 IDLE")
            
        except Exception as e:
            logger.error(f"🧠 [复盘] 分析失败：{e}", exc_info=True)
            self._publish_review_response(session_id, f"错误：{str(e)}", deep_analysis)
            # 🔥 v3.0 新增：失败时也回到 IDLE 状态
            state_manager.clear_task()
    
    def _publish_review_response(self, session_id: str, response: str, deep_analysis: bool):
        """发布复盘分析响应到事件总线
        
        Args:
            session_id: 会话 ID
            response: 分析响应
            deep_analysis: 是否深度分析
        """
        from zulong.core.event_bus import event_bus
        from zulong.core.types import EventType, EventPriority, ZulongEvent
        
        # 构建响应 payload
        payload = {
            'text': response,
            'session_id': session_id,
            'analysis_type': 'review',
            'deep_analysis': deep_analysis
        }
        
        # 如果期望 JSON，尝试解析
        if response.strip().startswith('{') and response.strip().endswith('}'):
            try:
                import json
                parsed = json.loads(response)
                payload['json_data'] = parsed
                logger.info(f"🧠 [复盘] 响应是有效的 JSON 格式")
            except:
                logger.warning(f"🧠 [复盘] 响应看起来像 JSON 但解析失败")
        
        # 发布事件
        output_event = ZulongEvent(
            type=EventType.L2_OUTPUT,
            source="InferenceEngine",
            payload=payload,
            priority=EventPriority.NORMAL
        )
        
        event_bus.publish(output_event)
        logger.info(f"🧠 [复盘] 已发布 L2_OUTPUT 事件，session_id={session_id}")
    
    def _process_with_memory(self, user_input: str, priority: EventPriority, voice_mode: str = "TEXT_ONLY", 
                            pre_classified_intent: str = None, pre_retrieved_memory: str = None):
        """带记忆的推理流程（非 Orchestrator 模式）
        
        Args:
            user_input: 用户输入
            priority: 优先级
            voice_mode: 语音模式 ("TEXT_ONLY", "AUTO_TTS", "FORCED_TTS")
            pre_classified_intent: L1-B 预分类的意图 ("chat"/"complex"/"resume")，如果提供则跳过 Round 1
            pre_retrieved_memory: L1-B 预检索的记忆上下文，如果提供则跳过重复检索
        """
        import time
        
        # Phase A1c: 标记推理开始，阻止并发
        self._processing_done.clear()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 [L2 推理] 收到用户输入：'{user_input}'")
        logger.info(f"🔍 [L2 推理] 时间：{time.strftime('%H:%M:%S')}")
        logger.info(f"{'='*80}\n")
        import sys as _sys  # debug trace
        logger.debug(f"[TRACE] _process_with_memory START: '{user_input[:50]}'")
        _sys.stdout.flush()
        
        # 语音模式已由 L1-B VoiceIntentClassifier 确定，此处仅记录日志
        from zulong.config.output_routing_config import get_output_routing_config
        _routing_config = get_output_routing_config()
        
        logger.info(f"🔊 [语音调试] voice_mode={voice_mode}, 用户输入='{user_input[:50]}...'")
        
        logger.debug(f"[TRACE] voice check done, voice_mode={voice_mode}")
        l2_status = state_manager.get_l2_status()
        logger.debug(f"[TRACE] L2 status={l2_status}")
        if l2_status == L2Status.UNLOADED:
            logger.warning("L2 is UNLOADED, cannot process")
            return
        
        visual_keywords = [
            "周围", "环境", "看到", "看见", "视觉", "图像", "图片",
            "哪里", "外观", "样子", "颜色", "形状",
            "刚才", "刚刚", "画面", "视频",
            "屏幕", "摄像头", "镜头", "眼前", "显示", "呈现",
            "场景", "物体", "物品", "动作", "手势",
        ]
        need_vision = any(kw in user_input for kw in visual_keywords)
        
        logger.info(f"🔧 [DEBUG] 关键词检测结果：need_vision={need_vision}, 用户输入：'{user_input}'")
        
        visual_context = None
        
        if need_vision:
            logger.info(f"👁️ 检测到视觉感知意图：'{user_input}'")
            
            if self._pending_visual_context:
                logger.info("👁️ 使用 L1-C 传递的视觉上下文")
                visual_context = self._pending_visual_context
                self._pending_visual_context = None
            else:
                logger.info("👁️ 无 L1-C 视觉上下文，使用默认提示")
                visual_context = "（当前没有实时视觉数据，请根据上下文回答）"
        
        state_manager.set_l2_status(L2Status.BUSY)
        logger.debug("[TRACE] set BUSY, entering try block")
        
        try:
            logger.debug("[TRACE] calling _retrieve_from_rag...")
            rag_context = self._retrieve_from_rag(user_input)
            logger.debug(f"[TRACE] RAG done, has_context={bool(rag_context)}")
            
            if rag_context:
                logger.info(f"📚 [RAG 调试] 检索到的上下文:\n{rag_context[:500]}...")
            else:
                logger.warning("⚠️ [RAG 调试] 未检索到任何 RAG 上下文")
            
            # ── 引用节点任务图切换（必须在意图分类之前） ──
            self._referenced_graph_lost = None  # 重置标记
            self._switch_graph_for_referenced_nodes()
            
            # ── 两阶段 FC 意图分类（Round 1: 分类 + 骨架操作） ──
            logger.debug("[TRACE] checking for pre-classified intent...")
            
            if pre_classified_intent:
                # L1-B 已经做了意图分类，直接使用
                logger.info(f"🎯 [Intent] 使用 L1-B 预分类结果: {pre_classified_intent} (跳过 Round 1)")
                from zulong.l2.intent_prompt_builder import IntentType
                intent_type = IntentType(pre_classified_intent)
                scaffold_data = {"intent": pre_classified_intent, "from_l1b": True}
            else:
                # 没有预分类，执行 Round 1 分类（兼容旧流程）
                logger.debug("[TRACE] calling _classify_intent...")
                intent_type, scaffold_data = self._classify_intent(user_input)
                logger.debug(f"[TRACE] intent done: {intent_type.value}")
            
            logger.info(f"🎯 [Intent] 最终分类结果: {intent_type.value}, scaffold: {scaffold_data}")
            
            # ── 活跃任务图意图升级 ──────────────────────────────
            # 当用户回答模型的追问时（如"5月1日出发，15天"），Round1 会分类
            # 为 CHAT（无 RESUME/COMPLEX 关键词）。但此时内存中仍有活跃任务图
            # 且有未完成节点 → 应升级为 COMPLEX，确保：
            #   1. 任务工具（task_mark_status 等）被注入
            #   2. 任务管理 prompt 被构建
            #   3. 模型能感知并继续执行任务图
            if intent_type.value == "chat":
                # 社交/问候类短消息不触发活跃任务图升级
                # "你好"、"谢谢" 等不应导致模型去执行任务图
                _stripped_input = user_input.strip().rstrip("。！？~～.!?")
                _is_trivial = (
                    len(_stripped_input) <= 5
                    and not any(c.isdigit() for c in _stripped_input)
                )
                if not _is_trivial:
                    try:
                        from zulong.tools.task_tools import get_active_task_graph as _get_tg_upgrade
                        _upgrade_tg = _get_tg_upgrade()
                        if _upgrade_tg is not None:
                            _upgrade_leaves = _upgrade_tg.get_leaf_nodes()
                            _upgrade_uncompleted = [
                                n for n in _upgrade_leaves
                                if n.status not in ("completed", "skipped")
                            ]
                            if _upgrade_uncompleted:
                                from zulong.l2.intent_prompt_builder import IntentType as _IT
                                intent_type = _IT.COMPLEX
                                scaffold_data = {
                                    "intent": "complex",
                                    "already_exists": True,
                                    "graph_id": getattr(_upgrade_tg, 'id', ''),
                                    "title": _upgrade_tg.title,
                                    "upgraded_from": "chat",
                                    "message": (
                                        f"检测到活跃任务图「{_upgrade_tg.title}」"
                                        f"有 {len(_upgrade_uncompleted)} 个未完成节点，"
                                        f"自动升级为 COMPLEX 意图。"
                                    ),
                                }
                                logger.info(
                                    f"[Intent] CHAT → COMPLEX 升级: "
                                    f"活跃任务图 {getattr(_upgrade_tg, 'id', '?')} "
                                    f"有 {len(_upgrade_uncompleted)}/{len(_upgrade_leaves)} "
                                    f"个未完成节点"
                                )
                    except Exception as _upgrade_err:
                        logger.debug(f"[Intent] 活跃任务图升级检查失败: {_upgrade_err}")
                else:
                    logger.debug(
                        f"[Intent] 跳过 CHAT→COMPLEX 升级: "
                        f"输入 '{user_input}' 为社交短消息"
                    )
            # ── END 活跃任务图意图升级 ────────────────────────────
            
            # ── COMPLEX → CHAT 降级（已完成任务的后续提问） ──────────
            # 当活跃任务图的所有叶子节点已完成，且用户输入较短时，
            # 将 COMPLEX 降级为 CHAT，避免为后续提问创建新任务图。
            # 🔥 [Fix-7B] 仅对短问句降级（如"怎么运行"），不对新任务降级
            # 新任务的 scaffold_data 中不会有 already_exists（已被 Fix-7A 处理）
            if intent_type.value == "complex":
                try:
                    from zulong.tools.task_tools import get_active_task_graph as _get_tg_downgrade
                    _downgrade_tg = _get_tg_downgrade()
                    if _downgrade_tg is not None:
                        _dg_leaves = _downgrade_tg.get_leaf_nodes()
                        _dg_uncompleted = [
                            n for n in _dg_leaves
                            if n.status not in ("completed", "skipped")
                        ]
                        if not _dg_uncompleted and _dg_leaves:
                            _stripped = user_input.strip()
                            # 🔥 [Fix-7B] 降低阈值到15字符，并排除含强任务动词的输入
                            _has_task_verb = any(
                                kw in _stripped
                                for kw in ("帮我", "写一个", "做一个", "设计", "开发",
                                           "创建", "搭建", "实现", "生成", "构建")
                            )
                            if len(_stripped) <= 15 and not _has_task_verb:
                                from zulong.l2.intent_prompt_builder import IntentType as _IT_dg
                                intent_type = _IT_dg.CHAT
                                scaffold_data = {
                                    "intent": "chat",
                                    "downgraded_from": "complex",
                                    "completed_task_title": _downgrade_tg.title,
                                    "message": (
                                        f"检测到已完成任务「{_downgrade_tg.title}」"
                                        f"的后续提问，降级为 CHAT"
                                    ),
                                }
                                logger.info(
                                    f"[Intent] COMPLEX → CHAT 降级: "
                                    f"已完成任务后续提问 '{_stripped}'"
                                )
                except Exception as _dg_err:
                    logger.debug(f"[Intent] COMPLEX→CHAT 降级检查失败: {_dg_err}")
            # ── END COMPLEX → CHAT 降级 ────────────────────────────

            # ── 🔥 [Fix-8] 新任务清除旧图谱 ──────────────────────────
            # 当判定为 COMPLEX（新任务）且旧图谱已全部完成时，清除旧图谱，
            # 让模型调用 task_create_plan 创建全新的任务图谱。
            if intent_type.value == "complex":
                try:
                    from zulong.tools.task_tools import (
                        get_active_task_graph as _get_tg_clear,
                        set_active_task_graph,
                    )
                    _clear_tg = _get_tg_clear()
                    if _clear_tg is not None:
                        _clear_leaves = _clear_tg.get_leaf_nodes()
                        _clear_all_done = (
                            _clear_leaves
                            and all(
                                n.status in ("completed", "skipped")
                                for n in _clear_leaves
                            )
                        )
                        # 检查用户输入是否包含新任务动词
                        _stripped_clear = user_input.strip()
                        _has_new_task_verb = any(
                            kw in _stripped_clear
                            for kw in ("帮我", "写一个", "做一个", "设计", "开发",
                                       "创建", "搭建", "实现", "生成", "构建",
                                       "请帮", "请做")
                        )
                        if _clear_all_done and _has_new_task_verb:
                            old_graph_id = getattr(_clear_tg, 'id', 'unknown')
                            old_title = _clear_tg.title
                            set_active_task_graph(None, None)
                            logger.info(
                                f"[Fix-8] 清除已完成旧图谱: {old_graph_id}「{old_title}」"
                                f"({len(_clear_leaves)} 节点全部完成)，"
                                f"准备创建新任务图谱"
                            )
                            # 在 scaffold 中记录清除信息，供提示词使用
                            scaffold_data["old_graph_cleared"] = True
                            scaffold_data["cleared_graph_id"] = old_graph_id
                            scaffold_data["cleared_graph_title"] = old_title
                except Exception as _clear_err:
                    logger.debug(f"[Fix-8] 清除旧图谱失败: {_clear_err}")
            # ── END 新任务清除旧图谱 ───────────────────────────────

            # 图丢失处理：用户引用了一个找不到的任务图，RESUME 降级为 CHAT
            # 此时应强制走 COMPLEX 路径，让模型重新创建任务计划
            _lost_graph_id = getattr(self, '_referenced_graph_lost', None)
            if _lost_graph_id and intent_type.value == "chat" and scaffold_data.get("fallback"):
                from zulong.l2.intent_prompt_builder import IntentType
                intent_type = IntentType.COMPLEX
                scaffold_data = {
                    "intent": "complex",
                    "graph_lost": True,
                    "lost_graph_id": _lost_graph_id,
                    "message": f"用户引用的任务图 {_lost_graph_id} 数据已丢失，需要重新创建任务计划。",
                }
                logger.info(f"[GraphLost] 检测到图丢失，CHAT → COMPLEX 覆盖 (graph_id={_lost_graph_id})")
            
            # 如果 Round 1 判定为 RESUME 且成功恢复了任务图，标记恢复状态
            if intent_type.value == "resume" and scaffold_data.get("has_task_graph"):
                self._is_resume_task = True
                # RESUME 流程需要逐节点处理，提升 CB 时间和步数预算
                if hasattr(self, '_circuit_breaker'):
                    self._circuit_breaker.escalate_for_resume()

            # COMPLEX 流程也提前升级 CB（不依赖 task_create_plan 工具调用）
            if intent_type.value == "complex" and scaffold_data.get("graph_id"):
                if hasattr(self, '_circuit_breaker'):
                    self._circuit_breaker.escalate_for_planning()
            
            # ── Round 2: 构建场景化提示词 ──
            from zulong.l2.intent_prompt_builder import build_round2_system_prompt
            
            messages = build_round2_system_prompt(
                intent_type, user_input, rag_context, visual_context,
                scaffold_data, rag_manager=self.rag_manager,
                voice_mode=voice_mode,
                pre_retrieved_memory=pre_retrieved_memory,
            )
            
            # 注入用户引用的节点（前端右键引用的节点地址，供模型定位）
            _ref_nodes = getattr(self, '_referenced_nodes', [])
            if _ref_nodes:
                _ref_lines = ["【用户引用的节点】"]
                for _ref in _ref_nodes:
                    if isinstance(_ref, dict):
                        _ref_lines.append(f"- {_ref.get('label', '?')} @{_ref.get('address', '?')}")
                    else:
                        # 兼容旧格式（纯 ID 字符串）
                        _ref_lines.append(f"- @{_ref}")
                _ref_lines.append("请在回答中关注上述引用节点的内容。")
                # 插入到 system 消息后、user 消息前
                _ref_msg = {"role": "system", "content": "\n".join(_ref_lines)}
                if len(messages) >= 2:
                    messages.insert(-1, _ref_msg)
                else:
                    messages.append(_ref_msg)
                logger.info(f"[L2] 已注入 {len(_ref_nodes)} 个引用节点到 messages")
            
            # 注入运行时上下文到 ToolEngine（供工具访问当前会话信息）
            self.tool_engine.set_context(
                user_input=user_input,
                voice_mode=voice_mode,
                has_visual=visual_context is not None,
            )
            
            # ── Round 2: 收集场景过滤后的工具定义 ──
            tool_definitions = self._collect_tool_definitions_for_intent(intent_type)
            
            logger.info(f"🧠 开始推理：'{user_input[:50]}...' " if len(user_input) > 50 else f"🧠 开始推理：'{user_input}'")
            
            generate_start = time.time()
            logger.info(f"🚀 [L2 推理] 开始调用模型生成... (时间：{time.strftime('%H:%M:%S')})")
            
            # 根据 l2_model 类型选择推理路径
            if isinstance(self.l2_model, dict) and self.vllm_client is not None:
                # ====== 远程模型：FC 自主循环 ======
                # 模型完全自主决定：直接回复 or 调用工具 or 多轮工具链
                logger.info(f"🚀 [FC] 远程 API 模式，工具数: {len(tool_definitions)}")
                vllm_model_id = LLM_MODEL_ID
                
                # 初始化 FC 循环变量
                response = None
                fc_turn = 0
                with self._lock:
                    self._interrupt_flag = False  # 重置中断标志
                
                # Phase B1: 注册推理到 InterruptHandler + SnapshotManager
                import uuid as _uuid
                _snapshot_task_id = f"infer_{_uuid.uuid4().hex[:8]}"
                self._current_inference_task_id = _snapshot_task_id
                try:
                    from zulong.l2.snapshot_manager import snapshot_manager
                    from zulong.l2.interrupt_handler import interrupt_handler
                    
                    # 将当前 TaskGraph 序列化存入 metadata
                    _tg_dict = None
                    try:
                        from zulong.tools.task_tools import get_active_task_graph
                        _tg = get_active_task_graph()
                        if _tg:
                            _tg_dict = _tg.serialize()
                    except Exception:
                        pass
                    
                    snapshot_manager.create_snapshot(
                        task_id=_snapshot_task_id,
                        task_name=user_input[:80],
                        context_window=messages,
                        working_memory={},
                        metadata={"task_graph": _tg_dict, "user_input": user_input},
                    )
                    if not interrupt_handler.start_generation(_snapshot_task_id):
                        logger.warning(f"[L2] 检测到残留生成锁，强制清理后重试...")
                        interrupt_handler.stop_generation()
                        if not interrupt_handler.start_generation(_snapshot_task_id):
                            logger.warning(f"[L2] 重试仍失败，跳过本次推理: {_snapshot_task_id}")
                            self._processing_done.set()
                            return
                    logger.info(f"[L2] 快照已创建: {_snapshot_task_id}")
                except Exception as _snap_err:
                    logger.debug(f"[L2] 快照创建失败（非致命）: {_snap_err}")
                
                # 初始化注意力窗口管理器
                _task_graph = None
                try:
                    from zulong.tools.task_tools import get_active_task_graph
                    _task_graph = get_active_task_graph()
                except Exception:
                    pass
                
                # RESUME 场景：第一轮强制调用 task_view_overview
                _force_first_tool = (
                    intent_type.value == "resume"
                    and _task_graph is not None
                )
                
                # Circuit Breaker: 重置状态（每次推理会话独立）
                self._circuit_breaker.reset()
                
                self._attn_window = AttentionWindowManager(
                    context_window_size=self._context_window_size,
                    task_graph=_task_graph,
                    memory_graph=self._get_memory_graph_safe(),
                )
                # 注册初始 messages（system + user）为 pinned
                for _init_msg in messages:
                    self._attn_window.register_message(
                        _init_msg, turn=0, pinned=True,
                    )
                
                # 🔥 发布 pipeline_start 事件（携带初始任务图谱）
                self._publish_task_graph_event("pipeline_start", 0, "", "")
                
                # ★ LangGraph FC Loop（替换原 while 循环）
                # 保存 messages 引用供 Rule C 自动挂起使用
                self._last_fc_messages = messages
                
                # 编排器路由：COMPLEX/RESUME + 编排器开关 → 走编排器
                # ✅ 选择执行引擎：LangGraph Orchestrator / 传统 Orchestrator / FC Loop
                _use_orchestrator = False
                _use_langgraph = False
                try:
                    from zulong.config.config_manager import get_l2_inference_config as _get_l2_cfg
                    _orch_cfg = _get_l2_cfg().get("orchestrator", {})
                    _use_orchestrator = (
                        _orch_cfg.get("enabled", False)
                        and intent_type.value in ("complex", "resume")
                    )
                    # ✅ v2.0 新增：检查是否启用 LangGraph
                    _use_langgraph = (
                        _use_orchestrator
                        and _orch_cfg.get("use_langgraph", False)
                    )
                except Exception:
                    pass
                
                if _use_orchestrator:
                    if _use_langgraph:
                        # ✅ 使用 LangGraph StateGraph 编排器（v2.0）
                        logger.info("[L2] ✅ 使用 LangGraph Orchestrator")
                        try:
                            from zulong.l2.orchestrator_graph import OrchestratorWithLangGraph
                            
                            # 创建或复用编排器实例
                            if not hasattr(self, '_orchestrator_langgraph'):
                                self._orchestrator_langgraph = OrchestratorWithLangGraph(self)
                            
                            orchestrator = self._orchestrator_langgraph
                            
                            # 异步运行编排器
                            import asyncio
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                response, fc_turn, thread_id = loop.run_until_complete(
                                    orchestrator.run(
                                        user_input=user_input,
                                        messages=messages,
                                        tool_definitions=tool_definitions,
                                        vllm_model_id=vllm_model_id,
                                        is_resume=(intent_type.value == "resume"),
                                    )
                                )
                            finally:
                                loop.close()
                            
                            logger.info(f"[L2] LangGraph Orchestrator 完成: thread_id={thread_id}")
                            
                        except Exception as e:
                            logger.error(f"[L2] LangGraph Orchestrator 失败: {e}", exc_info=True)
                            logger.warning("[L2] 降级为传统 Orchestrator")
                            # 降级为传统模式
                            from zulong.l2.orchestrator_graph import run_orchestrator
                            response, fc_turn = run_orchestrator(
                                engine=self,
                                messages=messages,
                                tool_definitions=tool_definitions,
                                vllm_model_id=vllm_model_id,
                                user_input=user_input,
                                is_resume=(intent_type.value == "resume"),
                            )
                    else:
                        # 使用传统状态机编排器
                        logger.info("[L2] 使用传统 Orchestrator")
                        from zulong.l2.orchestrator_graph import run_orchestrator
                        response, fc_turn = run_orchestrator(
                            engine=self,
                            messages=messages,
                            tool_definitions=tool_definitions,
                            vllm_model_id=vllm_model_id,
                            user_input=user_input,
                            is_resume=(intent_type.value == "resume"),
                        )
                else:
                    # 使用原生 FC 循环
                    response, fc_turn = run_fc_loop(
                        engine=self,
                        messages=messages,
                        tool_definitions=tool_definitions,
                        vllm_model_id=vllm_model_id,
                        force_first_tool=_force_first_tool,
                        user_input=user_input,
                        is_resume=(intent_type.value == "resume"),
                        intent_max_tokens=4096 if intent_type.value in ("complex", "resume") else 1024,
                    )
                self._fc_turn_count = fc_turn  # 供 Rule C 自动挂起记录
                
                # Phase B2: 中断检测 + 通过 InterruptHandler 冻结
                with self._lock:
                    was_interrupted = self._interrupt_flag
                    self._interrupt_flag = False  # 重置，避免影响后续新任务
                
                if was_interrupted:
                    logger.info("[L2] FC 循环因中断退出，执行 TSD 冻结流程")
                    try:
                        from zulong.l2.snapshot_manager import snapshot_manager as _sm_b2
                        from zulong.l2.interrupt_handler import interrupt_handler as _ih_b2
                        
                        # 更新快照（保存最新的 messages 和 TaskGraph）
                        _snap_tid = self._current_inference_task_id
                        if _snap_tid:
                            _tg_dict_upd = None
                            try:
                                from zulong.tools.task_tools import get_active_task_graph as _get_tg_b2
                                _tg_upd = _get_tg_b2()
                                if _tg_upd:
                                    _tg_dict_upd = _tg_upd.serialize()
                            except Exception:
                                pass
                            
                            _sm_b2.update_snapshot(
                                task_id=_snap_tid,
                                context_window=messages,
                                working_memory={"task_graph": _tg_dict_upd, "fc_turn": fc_turn},
                                execution_pointer={
                                    "current_step": fc_turn,
                                    "step_description": f"FC loop interrupted at turn {fc_turn}",
                                },
                            )
                        
                        # 通过 InterruptHandler 冻结（调用 snapshot_manager.freeze）
                        frozen_snapshot = _ih_b2.handle_interrupt("user_new_message")
                        if frozen_snapshot:
                            logger.info(f"[L2] 任务已冻结: {frozen_snapshot.task_name}")
                    except Exception as _freeze_err:
                        logger.error(f"[L2] TSD 冻结失败: {_freeze_err}", exc_info=True)
                    
                    # 安全网：磁盘持久化 TaskGraph
                    try:
                        self._auto_suspend_if_needed()
                    except Exception:
                        pass
                    
                    # 中断后跳过后续响应处理，直接进入 finally
                    return
                
                if not response:  # None 或空字符串都需要降级处理
                    # 🔥 优化：根据达到限制的原因选择不同处理方式
                    if fc_turn >= self._hard_limit:
                        logger.warning(f"[FC] 达到硬限制 {self._hard_limit} 步，使用降级回复")
                    else:
                        logger.warning(f"[FC] FC 循环异常终止 (已执行 {fc_turn} 步)")

                    # 任务全完成但回复为空 → 从 TaskGraph 合成摘要（降级路径保底）
                    if not response:
                        try:
                            from zulong.tools.task_tools import get_active_task_graph as _get_tg_synth
                            _synth_tg = _get_tg_synth()
                            if _synth_tg:
                                _synth_leaves = _synth_tg.get_leaf_nodes()
                                _synth_done = [n for n in _synth_leaves if n.status == "completed"]
                                if _synth_done and len(_synth_done) == len(_synth_leaves):
                                    parts = [f"## {_synth_tg.title}\n"]
                                    for _sn in _synth_done:
                                        _sr = getattr(_sn, 'result', '') or ''
                                        parts.append(f"### {_sn.label}\n{_sr or '（已完成）'}\n")
                                    response = "\n".join(parts)
                                    logger.info(
                                        f"[FC][Fallback] 从 {len(_synth_done)} 个已完成节点合成摘要回复"
                                    )
                        except Exception:
                            pass

                    # RESUME 场景专用降级：利用任务图信息生成有意义的提示
                    if getattr(self, '_is_resume_task', False):
                        try:
                            from zulong.tools.task_tools import get_active_task_graph as _get_tg_fb
                            _fb_tg = _get_tg_fb()
                            if _fb_tg:
                                _fb_title = _fb_tg.title or "未命名任务"
                                _fb_leaves = _fb_tg.get_leaf_nodes()
                                _fb_uncompleted = [n for n in _fb_leaves if n.status not in ("completed", "skipped")]
                                _fb_completed = [n for n in _fb_leaves if n.status == "completed"]
                                response = (
                                    f"已恢复任务「{_fb_title}」。"
                                    f"当前进度：{len(_fb_completed)}/{len(_fb_leaves)} 个子任务已完成。"
                                )
                                if _fb_uncompleted:
                                    _fb_next = _fb_uncompleted[0]
                                    response += f"\n下一步需要执行：{_fb_next.label}（{_fb_next.desc or ''}）。"
                                response += "\n由于模型响应超时，请稍后再说「继续」来推进任务。"
                                logger.info(f"[FC][RESUME] 使用任务图信息生成降级回复")
                        except Exception:
                            pass

                    if not response:
                        response = self._get_fallback_response(user_input)
                
                logger.info(f"[FC] 循环完成，共 {fc_turn} 轮")
                
                # Phase B3: 正常完成，清理 InterruptHandler 生成状态
                try:
                    from zulong.l2.interrupt_handler import interrupt_handler as _ih_b3
                    _ih_b3.stop_generation()
                except Exception:
                    pass
                
                # ── MemoryGraph: Hebbian 学习（巩固本轮共激活的边权）──
                try:
                    from zulong.memory.memory_graph import get_memory_graph as _get_mg_hebb
                    _mg_hebb = _get_mg_hebb()
                    logger.info(f"[FC] Hebbian 前置检查: mg={'有' if _mg_hebb else '无'}")
                    if _mg_hebb:
                        _mg_hebb.hebbian_strengthen()
                        logger.info("[FC] Hebbian 学习完成")
                except Exception as _hebb_err:
                    logger.info(f"[FC] Hebbian 学习跳过: {_hebb_err}")
                
                # 🔥 发布 agent_done 事件（携带最终任务图谱）
                self._publish_task_graph_event("agent_done", fc_turn, "", "")
                
                # ── 归档后补丁：回填 final_answer / duration / total_turns ──
                # _auto_archive_completed 在 task_mark_status 级联中触发，此时
                # FC 循环尚未结束，final_answer 等字段无法获取。在此补丁回填。
                try:
                    from zulong.tools.task_tools import get_active_task_graph as _get_tg_patch
                    from zulong.tools.task_tools import _active_graph_id as _patch_gid
                    _patch_tg = _get_tg_patch()
                    if _patch_tg and _patch_gid:
                        _patch_root = _patch_tg.get_node("req")
                        if _patch_root and _patch_root.status == "completed":
                            from zulong.l2.task_archive import CompletedTaskArchiveManager
                            _patch_mgr = CompletedTaskArchiveManager()
                            _patch_mgr.patch_archive(
                                _patch_gid,
                                final_answer=(response or "")[:500],
                                total_turns=fc_turn,
                                duration=round(time.time() - generate_start, 1),
                            )
                except Exception as _patch_err:
                    logger.warning(f"[L2] 归档补丁失败（非致命）: {_patch_err}")
            else:
                # ====== 本地模型：单轮推理（不支持 FC）======
                prompt = self.l2_model.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                prompt_length = len(prompt)
                logger.info(f"📊 Prompt 长度：{prompt_length} tokens")
                if prompt_length > 4000:
                    logger.warning(f"⚠️ Prompt 过长 ({prompt_length})，可能导致生成问题")
                
                session_id = f"session_{hash(user_input) % 10000}"
                past_key_values = self._get_kv_cache(session_id)
                
                generate_kwargs = {
                    "max_tokens": 1024,
                }
                
                if past_key_values is not None:
                    generate_kwargs["past_key_values"] = past_key_values
                    logger.info(f"🚀 [KV Cache] 使用缓存加速生成")
                
                response = self.l2_model.generate(
                    prompt,
                    **generate_kwargs
                )
                
                if hasattr(self.l2_model, 'get_past_key_values'):
                    try:
                        new_kv = self.l2_model.get_past_key_values()
                        if new_kv is not None:
                            self._save_kv_cache(session_id, new_kv)
                    except Exception as e:
                        logger.warning(f"⚠️ [KV Cache] 保存失败：{e}")
            
            generate_elapsed = time.time() - generate_start
            logger.info(f"✅ [L2 推理] 模型生成完成！耗时：{generate_elapsed:.2f}秒")
            
            if "</think>" in response:
                parts = response.split("</think>")
                if len(parts) > 1:
                    response = parts[-1].strip()
                    logger.info(f"📝 已提取</think>后的回复 (共{len(parts)}个</think>标签)")
            
            raw_response = response
            logger.info(f"🔍 [DEBUG] 模型原始输出 (Raw Output): {raw_response[:500]}...")
            logger.info(f"🔍 [DEBUG] 原始回复长度：{len(raw_response)} 字符")
            
            from zulong.utils.text_cleaner import clean_text_for_tts
            cleaned_response = clean_text_for_tts(response)
            
            if cleaned_response != response:
                logger.info(f"✨ TTS 文本清洗完成：{len(response)} → {len(cleaned_response)} 字符")
                logger.info(f"✨ 清洗后文本：'{cleaned_response[:200]}'")
            else:
                logger.debug("✅ 文本已足够纯净，无需清洗")
            
            response_length = len(cleaned_response)
            logger.info(f"💬 生成回复长度：{response_length} 字符")
            logger.info(f"💬 回复前 200 字符：'{cleaned_response[:200]}'")
            if response_length > 200:
                logger.info(f"💬 回复后 200 字符：'{cleaned_response[-200:]}'")
            
            if response_length >= 1024 * 4:
                logger.warning("⚠️ 回复可能达到了 max_tokens 限制")
            
            total_elapsed = time.time() - generate_start
            logger.info(f"\n{'='*80}")
            logger.info(f"✅ [L2 推理] 完整流程完成！总耗时：{total_elapsed:.2f}秒")
            logger.info(f"✅ [L2 推理] 回复内容：'{cleaned_response[:100]}...'")
            logger.info(f"{'='*80}\n")
            
            self._update_memory(user_input, response, fc_turn=fc_turn)
            
            output_event = ZulongEvent(
                type=EventType.L2_OUTPUT,
                priority=EventPriority.NORMAL,
                source="InferenceEngine",
                payload={
                    "text": response,
                    "input_text": user_input,
                    "has_rag_context": rag_context is not None,
                    "history_length": len(self._recent_turns_cache),
                    "visual_context": None,
                    "timestamp": time.time()
                }
            )
            event_bus.publish(output_event)
            
            # 使用配置决定是否触发语音输出
            from zulong.config.output_routing_config import OutputMode
            try:
                current_mode = OutputMode(voice_mode)
            except ValueError:
                current_mode = OutputMode.TEXT_ONLY
            
            logger.info(f"🔊 [语音调试] current_mode={current_mode}, should_trigger={_routing_config.should_trigger_speech(current_mode)}")
            
            if _routing_config.should_trigger_speech(current_mode):
                logger.info(f"🔊 生成语音 (Mode: {voice_mode})...")
                
                speak_event = ZulongEvent(
                    type=EventType.ACTION_SPEAK,
                    priority=EventPriority.NORMAL,
                    source="InferenceEngine",
                    payload={
                        "text": cleaned_response,
                        "style": _routing_config.get_tts_style(current_mode),
                        "voice_mode": voice_mode,
                        "timestamp": time.time()
                    }
                )
                event_bus.publish(speak_event)
                logger.info(f"✅ ACTION_SPEAK 事件已发布 (Mode: {voice_mode}, 文本已清洗, 文本长度={len(cleaned_response)})")
            else:
                logger.warning(f"🔇 [语音调试] Text-only mode. Skipping TTS. current_mode={current_mode}")
            
        except Exception as e:
            logger.error(f"推理失败：{e}", exc_info=True)
        finally:
            # 确保 InterruptHandler 生成状态始终被清理（防止锁残留）
            try:
                from zulong.l2.interrupt_handler import interrupt_handler as _ih_final
                _ih_final.stop_generation()
            except Exception:
                pass
            # Rule C: 自动挂起未完成的任务图（安全网）
            self._auto_suspend_if_needed()
            # Phase B4: 清理当前推理的快照引用
            self._current_inference_task_id = None
            self.tool_engine.clear_context()
            state_manager.set_l2_status(L2Status.IDLE)
            # Phase A1d: 标记推理完成，允许下一轮推理启动
            self._processing_done.set()
            # Phase C3: 检查是否有被中断的冻结任务需要自动恢复
            self._try_resume_frozen_task()
    
    def _auto_suspend_if_needed(self):
        """Rule C: FC 循环结束后，自动挂起未完成的任务图
        
        安全网逻辑：如果还有活跃任务图且有未完成的叶子节点，
        自动执行挂起操作，防止任务状态丢失。
        """
        try:
            from zulong.tools.task_tools import get_active_task_graph, set_active_task_graph
            tg = get_active_task_graph()
            if tg is None:
                return
            
            # 检查是否全部完成
            leaf_nodes = tg.get_leaf_nodes()
            if not leaf_nodes:
                return
            uncompleted = [n for n in leaf_nodes if n.status != "completed"]
            if not uncompleted:
                return  # 全部完成，无需挂起
            
            # 有未完成的任务 → 自动挂起
            from zulong.l2.task_suspension import TaskSuspensionManager, SuspendableTaskState
            root = tg.get_node("req")
            description = root.label if root else tg.title
            graph_id = getattr(tg, 'id', '')
            
            state = SuspendableTaskState(
                task_id=TaskSuspensionManager.generate_task_id(),
                description=description,
                messages=getattr(self, '_last_fc_messages', None) or [],
                accumulated_links="",
                circuit_breaker_state=self._circuit_breaker.serialize(),
                iteration_count=getattr(self, '_fc_turn_count', 0),
                task_graph=tg,
                suspended_reason="auto_safety_net",
                metadata={"graph_id": graph_id, "auto_suspended": True},
            )
            
            mgr = TaskSuspensionManager()
            # 同步调用挂起
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    task_id = pool.submit(asyncio.run, mgr.suspend_task(state)).result(timeout=10)
            else:
                task_id = asyncio.run(mgr.suspend_task(state))
            
            if task_id:
                # 不清空活跃任务图引用：
                # 1. 数据已安全持久化到磁盘（suspended_tasks + graph_backups 双保险）
                # 2. 如果下一条消息是同一任务的后续操作，可以直接访问活跃图
                # 3. 如果是全新任务（COMPLEX），StartSessionTool 的防重复机制会处理
                # 4. 如果是简单聊天（CHAT），不会触及任务图
                logger.info(
                    f"[Rule C] 已持久化未完成任务图: '{description}' "
                    f"({len(uncompleted)}/{len(leaf_nodes)} 未完成), task_id={task_id} "
                    f"（活跃引用保留，供后续消息使用）"
                )
        except Exception as e:
            logger.warning(f"[Rule C] 自动挂起失败（非致命）: {e}")
    
    def _try_resume_frozen_task(self):
        """Phase C1: 新任务完成后，自动恢复被中断的冻结任务
        
        TSD 8B.1.2: resume_generation → 环境重评估 → CONTINUE/REPLAN/ABORT
        """
        try:
            from zulong.l2.snapshot_manager import snapshot_manager
            from zulong.l2.interrupt_handler import interrupt_handler
            
            # 查找冻结的任务
            all_snapshots = snapshot_manager.list_snapshots()
            frozen_tasks = [s for s in all_snapshots if s["status"] == "FROZEN"]
            
            if not frozen_tasks:
                return
            
            # 取最近冻结的（LIFO顺序）
            target = frozen_tasks[-1]
            task_id = target["task_id"]
            task_name = target["task_name"]
            
            logger.info(f"[L2] 发现冻结任务 '{task_name}'({task_id})，尝试自动恢复")
            
            # 通过 InterruptHandler 恢复（含环境重评估）
            snapshot = interrupt_handler.resume_generation(task_id)
            
            if snapshot is None:
                # ABORT: 环境变化过大，任务已被中止
                logger.info(f"[L2] 任务 '{task_name}' 因环境变化被中止")
                return
            
            # CONTINUE 或 REPLAN: 发布恢复事件到 EventBus
            logger.info(f"[L2] 任务 '{task_name}' 环境重评估通过，发布恢复事件")
            
            resume_context = {
                "task_id": task_id,
                "task_name": snapshot.task_name,
                "messages": snapshot.context_window,
                "working_memory": snapshot.working_memory,
                "execution_pointer": snapshot.execution_pointer.to_dict() if snapshot.execution_pointer else {},
            }
            
            event_bus.publish(ZulongEvent(
                type=EventType.SYSTEM_L2_COMMAND,
                priority=EventPriority.LOW,  # 低优先级，让新用户消息优先
                source="L2/AutoResume",
                payload={
                    "text": f"[系统] 自动恢复被中断的任务: {snapshot.task_name}",
                    "_resume_task_id": task_id,
                    "_resume_context": resume_context,
                }
            ))
            logger.info(f"[L2] 恢复事件已发布: {task_id}")
            
        except Exception as e:
            logger.debug(f"[L2] 自动恢复检查失败（非致命）: {e}")
    
    def _process_resume_task(self, task_id: str):
        """Phase C2: 处理恢复事件，从冻结快照重启 FC 循环
        
        恢复流程：
        1. 从 SnapshotManager 获取已 thaw 的快照
        2. 恢复 TaskGraph
        3. 重建 messages 上下文
        4. 运行 FC 循环（is_resume=True）
        """
        self._processing_done.clear()
        import time
        
        logger.info(f"[L2] 开始恢复冻结任务: {task_id}")
        
        try:
            state_manager.set_l2_status(L2Status.BUSY, task_id=task_id)
            
            from zulong.l2.snapshot_manager import snapshot_manager
            
            # 1. 获取已 thaw 的快照（resume_generation 已调用 thaw）
            snapshot = snapshot_manager.get_active_snapshot()
            if not snapshot:
                logger.warning(f"[L2] 恢复失败: 找不到活跃快照 {task_id}")
                return
            
            # 2. 恢复 TaskGraph
            task_graph_dict = snapshot.working_memory.get("task_graph")
            if task_graph_dict:
                try:
                    from zulong.l2.task_graph import TaskGraph
                    from zulong.tools.task_tools import set_active_task_graph
                    tg = TaskGraph.deserialize(task_graph_dict)
                    set_active_task_graph(tg)
                    logger.info(f"[L2] TaskGraph 已恢复: {tg.title}")
                except Exception as tg_err:
                    logger.warning(f"[L2] TaskGraph 恢复失败: {tg_err}")
            
            # 3. 恢复 messages 上下文
            messages = snapshot.context_window or []
            if not messages:
                logger.warning("[L2] 恢复失败: 快照中无对话历史")
                return
            
            original_input = snapshot.metadata.get("user_input", snapshot.task_name)
            logger.info(f"[L2] 恢复上下文: {len(messages)} 条消息, 原始输入: '{original_input[:50]}'")
            
            # 4. 运行 FC 循环
            from zulong.l2.unified_fc_runner import run_fc_loop
            
            with self._lock:
                self._interrupt_flag = False
            
            # 注册到 InterruptHandler
            self._current_inference_task_id = task_id
            try:
                from zulong.l2.interrupt_handler import interrupt_handler
                if not interrupt_handler.start_generation(task_id):
                    logger.warning(f"[L2][RESUME] 检测到残留生成锁，强制清理后重试...")
                    interrupt_handler.stop_generation()
                    if not interrupt_handler.start_generation(task_id):
                        logger.warning(f"[L2][RESUME] 重试仍失败，跳过恢复: {task_id}")
                        self._processing_done.set()
                        return
            except Exception:
                pass
            
            # 获取工具定义
            tool_definitions = self._collect_tool_definitions()
            from zulong.models.container import LLM_MODEL_ID
            vllm_model_id = LLM_MODEL_ID
            
            self._last_fc_messages = messages
            response, fc_turn = run_fc_loop(
                engine=self,
                messages=messages,
                tool_definitions=tool_definitions,
                vllm_model_id=vllm_model_id,
                force_first_tool=None,
                user_input=original_input,
                is_resume=True,
            )
            self._fc_turn_count = fc_turn
            
            # 中断检查
            with self._lock:
                was_interrupted = self._interrupt_flag
                self._interrupt_flag = False
            
            if was_interrupted:
                logger.info("[L2] 恢复任务再次被中断")
                try:
                    from zulong.l2.snapshot_manager import snapshot_manager as _sm_c2
                    from zulong.l2.interrupt_handler import interrupt_handler as _ih_c2
                    _sm_c2.update_snapshot(
                        task_id=task_id,
                        context_window=messages,
                        working_memory={"task_graph": task_graph_dict, "fc_turn": fc_turn},
                    )
                    _ih_c2.handle_interrupt("re_interrupted")
                except Exception:
                    pass
                return
            
            # 正常完成：发送响应
            if response:
                output_event = ZulongEvent(
                    type=EventType.L2_OUTPUT,
                    priority=EventPriority.NORMAL,
                    source="InferenceEngine/Resume",
                    payload={
                        "text": response,
                        "input_text": original_input,
                        "resumed_task": True,
                        "timestamp": time.time()
                    }
                )
                event_bus.publish(output_event)
                logger.info(f"[L2] 恢复任务响应已发布")
            
            # 清理快照（任务已完成）
            try:
                from zulong.l2.snapshot_manager import snapshot_manager as _sm_c2_done
                _sm_c2_done._remove_snapshot(task_id)
            except Exception:
                pass
            
            # 清理 InterruptHandler 状态
            try:
                from zulong.l2.interrupt_handler import interrupt_handler as _ih_c2_done
                _ih_c2_done.stop_generation()
            except Exception:
                pass
        
        except Exception as e:
            logger.error(f"[L2] 恢复任务失败: {e}", exc_info=True)
        finally:
            # 确保 InterruptHandler 生成状态始终被清理（防止锁残留）
            try:
                from zulong.l2.interrupt_handler import interrupt_handler as _ih_resume_final
                _ih_resume_final.stop_generation()
            except Exception:
                pass
            self._current_inference_task_id = None
            self._auto_suspend_if_needed()
            self.tool_engine.clear_context()
            state_manager.set_l2_status(L2Status.IDLE)
            self._processing_done.set()
            # 递归检查：是否还有更多冻结任务
            self._try_resume_frozen_task()
    
    def _build_messages_with_history(self, user_input: str, rag_context: Optional[str], visual_context: Optional[str]) -> list:
        """构建包含历史、RAG 和视觉上下文的 messages（同步版本）
        
        Args:
            user_input: 用户输入
            rag_context: RAG 检索到的上下文
            visual_context: 视觉上下文
            
        Returns:
            构建好的 messages 列表
        """
        from datetime import datetime
        
        now = datetime.now()
        hour = now.hour
        current_time_str = now.strftime("%Y-%m-%d %H:%M")
        
        if 5 <= hour < 11:
            time_period = "早晨"
        elif 11 <= hour < 14:
            time_period = "中午"
        elif 14 <= hour < 18:
            time_period = "下午"
        elif 18 <= hour < 22:
            time_period = "晚上"
        else:
            time_period = "深夜"
        
        system_parts = [
            "**重要身份认知**：",
            "- 你的名字叫  \"  祖龙 (ZULONG)  \"",
            f"\n当前时间：{current_time_str} ({time_period})。",
            "\n【人称代词】",
            "- \"我\" 指的是你自己（祖龙）",
            "- \"你\" 指的是用户",
            "- 当用户说\"我家\"、\"我叫\"时，指的是用户",
            "\n【交流风格】",
            "用自然、友好的口语和用户对话，就像朋友聊天一样。",
            "\n【任务管理规则】",
            "当用户要求你完成复杂的多步骤任务时（如：开发项目、编写代码、设计方案、写报告、做游戏等），",
            "必须按以下步骤操作：",
            "",
            "步骤1. 【创建总节点】调用 task_create_plan，传入：",
            "  - title: 简短标题（如\"Python爬虫项目方案\"）",
            "  - user_requirement: 必须原样复制用户的完整原始需求文本，不得概括或缩写",
            "",
            "步骤2. 【创建子节点大纲】用 task_add_node 逐个添加子任务节点，parent_id='req'（挂到总节点下）",
            "  - 每个子节点代表一个独立的工作模块/步骤",
            "  - 先搭建完整大纲再执行，不要边做边加",
            "",
            "步骤3. 【建立依赖关系】用 task_add_dependency 声明节点间的先后顺序",
            "  - 当任务B需要任务A的产出时，必须建立依赖: task_add_dependency(from_id=A, to_id=B)",
            "  - 依赖关系决定执行顺序，也会在前端图谱中显示为虚线连线",
            "",
            "步骤4. 用 task_view_overview 查看任务概览确认结构",
            "步骤5. 【逐节点执行】按依赖顺序逐个执行子任务：",
            "  - 开始执行前：task_mark_status(node_id='xxx', status='in_progress')",
            "  - 生成该节点的具体内容（不要一次性生成所有节点的内容）",
            "  - 完成后：task_mark_status(node_id='xxx', status='completed', result='该节点的完整结果')",
            "  - 然后继续下一个节点，直到所有节点都标记为 completed",
            "  ⚠️ 严禁跳过 task_mark_status 直接输出完整回复！每个节点必须单独标记。",
            "步骤6. 所有子任务完成后，再生成最终的汇总回复给用户",
            "",
            "【任务图谱模板】下面是一个标准的任务图谱结构示例：",
            "用户说：\"帮我设计一个电商系统，包含用户管理、商品管理、订单管理三个模块\"",
            "",
            "你应该这样调用工具：",
            "① task_create_plan(title=\"电商系统设计方案\", user_requirement=\"帮我设计一个电商系统，包含用户管理、商品管理、订单管理三个模块\")",
            "   → 创建总节点 req，存储用户完整原始需求",
            "② task_add_node(parent_id=\"req\", label=\"用户管理模块\", desc=\"设计用户注册、登录、权限管理功能\")",
            "   → 创建子节点 o1，挂在 req 下",
            "③ task_add_node(parent_id=\"req\", label=\"商品管理模块\", desc=\"设计商品上架、分类、搜索功能\")",
            "   → 创建子节点 o2，挂在 req 下",
            "④ task_add_node(parent_id=\"req\", label=\"订单管理模块\", desc=\"设计下单、支付、物流跟踪功能\")",
            "   → 创建子节点 o3，挂在 req 下",
            "⑤ task_add_dependency(from_id=\"o1\", to_id=\"o3\", via=\"用户身份信息\")",
            "   → 订单模块依赖用户模块（需要用户信息才能下单）",
            "⑥ task_add_dependency(from_id=\"o2\", to_id=\"o3\", via=\"商品数据\")",
            "   → 订单模块依赖商品模块（需要商品信息才能下单）",
            "⑦ task_view_overview() → 确认结构",
            "",
            "最终生成的树形结构（含依赖连线）：",
            "  req (电商系统设计方案) ← 总节点，保存用户原始需求",
            "   ├── o1 (用户管理模块) ──依赖──→ o3",
            "   ├── o2 (商品管理模块) ──依赖──→ o3",
            "   └── o3 (订单管理模块) ← 依赖 o1、o2 完成后才能开始",
            "",
            "如果子任务本身还很复杂，可以继续往下拆分子节点：",
            "  task_add_node(parent_id=\"o1\", label=\"注册功能\", desc=\"手机号+邮箱注册\")",
            "   → 创建子节点 o1_1，挂在 o1 下",
            "",
            "重要规则：",
            "- 第1步的 user_requirement 必须保存用户原话，这是后续恢复任务时还原上下文的关键依据",
            "- 所有子节点必须通过 parent_id 正确挂到父节点下，形成树形结构",
            "- 有先后顺序的任务之间必须用 task_add_dependency 建立依赖，确保执行顺序正确",
            "- 如果任务较简单（单步即可完成），可直接回答无需创建任务图",
            "",
            "【任务挂起与恢复规则】",
            "当用户在任务进行中说\"暂停\"、\"先不做了\"、\"先聊别的\"、\"换个话题\"等意思时：",
            "  → 必须调用 task_suspend(reason=\"user_requested\") 将当前任务图持久化到磁盘",
            "  → 不要只是口头说\"已暂停\"而不调用工具，否则任务状态会丢失",
            "  → 挂起后清除当前活跃任务图，可以正常回答其他问题",
            "",
            "当用户说\"继续\"、\"接着做\"、\"上次那个任务\"、\"恢复之前的任务\"等意思时：",
            "  → 第一步：调用 task_list_suspended(query=\"相关描述\") 查找并恢复挂起的任务",
            "  → 第二步：如果返回 resumed=true，说明之前的任务图（含所有节点和状态）已自动恢复到内存",
            "  → 第三步：调用 task_view_overview() 查看恢复后的任务图当前进度",
            "  → 第四步：阅读概览，找到状态为 pending 或 not_started 的第一个节点，从那里继续执行",
            "  → 第五步：用 task_mark_status 更新该节点为 in_progress，然后执行该节点的具体内容",
            "",
            "  恢复后的绝对禁止事项：",
            "  ✗ 禁止调用 task_create_plan — 这会创建全新图谱，丢弃已恢复的进度",
            "  ✗ 禁止调用 task_add_node — 节点已经在恢复的图谱中了，不需要重新添加",
            "  ✗ 禁止重命名或重建已有节点 — 直接使用恢复后图谱中现有的节点",
            "  ✓ 只使用 task_mark_status 更新现有节点状态，然后继续执行",
        ]
        
        # ========== 推理层级感知 ==========
        _phase = "自由对话"
        _layer_info = "L2 推理引擎（顶层）"
        try:
            _te = getattr(self, 'tool_engine', None)
            _tm = getattr(_te, 'task_manager', None) if _te else None
            _tg = getattr(_tm, 'task_graph', None) if _tm else None
            if _tg and getattr(_tg, 'root_id', None):
                _phase = "任务恢复" if getattr(self, '_is_resume_task', False) else "任务执行"
                _current = getattr(_tg, 'current_node_id', None) or _tg.root_id
                _depth = 0
                _node = _current
                while _node and _node != _tg.root_id:
                    _parent = getattr(_tg.nodes.get(_node), 'parent_id', None) if hasattr(_tg, 'nodes') else None
                    if _parent:
                        _depth += 1
                        _node = _parent
                    else:
                        break
                _layer_info = f"L2 推理引擎 → 任务图第 {_depth} 层"
        except Exception:
            pass
        system_parts.append(
            f"\n【推理层级】当前阶段：{_phase} | 层级：{_layer_info}\n"
            "你可以根据当前层级调整回答深度：自由对话时简洁友好；任务执行时严谨有条理；任务恢复时先确认上下文再继续。\n"
        )
        
        if visual_context:
            is_simple_greeting = any(kw in visual_context for kw in ["挥手", "注视", "走近"])
            
            if is_simple_greeting:
                system_parts.append("""
【回应风格】
用户正在和你互动！请用简短、活泼、口语化的方式回应，就像朋友打招呼一样。
- 挥手：可以说"你好呀~"、"嗨~"、"怎么啦？"、"有什么事吗？"
- 注视：可以说"看什么呢~"、"需要帮忙吗？"、"想聊点什么？"
- 靠近：可以说"来啦~"、"找我有什么事吗？"
打招呼回复控制在 40 字以内，像真人对话一样自然。

【视觉观察】
""" + visual_context + "\n")
            else:
                system_parts.append("""
【回答建议】
1. 直接基于视觉观察回答用户问题
2. 避免复述系统规则或约束条件
3. 如果视觉信息不足，诚实告知用户
4. 使用自然口语，50-150 字
5. 避免使用数字列表或项目符号

【视觉观察】
""" + visual_context + "\n")
        else:
            system_parts.append("""
【回答建议】
1. 友好、专业地回答用户问题
2. 如果信息不足，诚实告知用户
3. 使用自然流畅的口语，50-150 字
""")
        
        if rag_context:
            system_parts.append(f"\n【参考知识】\n{rag_context}\n")
        
        # ========== 思维导航注入（思维深度索引） ==========
        try:
            from zulong.memory.memory_graph import get_memory_graph as _get_mg_nav
            _mg_nav = _get_mg_nav()
            if _mg_nav:
                focus_summary = _mg_nav.get_focus_path_summary()
                if focus_summary:
                    system_parts.append(f"\n{focus_summary}\n")
                    logger.debug(f"[思维导航] 已注入焦点路径 ({len(focus_summary)} chars)")
        except Exception as e:
            logger.debug(f"[思维导航] 注入跳过: {e}")
        
        # ========== 记忆注入：通过 MemoryGraph 统一检索 ==========
        try:
            from zulong.memory.memory_graph import get_memory_graph
            _mg = get_memory_graph()
            if _mg:
                if not getattr(_mg, '_rag_manager', None) and self.rag_manager:
                    _mg.set_rag_manager(self.rag_manager)
                
                def _run_async_bridge(coro):
                    """在同步上下文中执行异步协程"""
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None
                    if loop is not None and loop.is_running():
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            return pool.submit(asyncio.run, coro).result(timeout=30)
                    else:
                        return asyncio.run(coro)
                
                mg_results = _run_async_bridge(
                    _mg.retrieve_context(
                        user_input, top_k=8,
                        session_id=getattr(self, '_current_session_id', ""),
                    )
                )
                if mg_results:
                    memory_sections = []
                    for r in mg_results:
                        ntype = r.get("node_type", "")
                        content = r.get("content", "")
                        label = r.get("label", "")
                        if not content:
                            continue
                        if ntype == "experience":
                            continue  # EXPERIENCE 由 search_experience FC 工具按需获取
                        elif ntype == "dialogue":
                            memory_sections.append(f"【历史对话】{content[:200]}")
                        elif ntype == "task":
                            status = r.get("metadata", {}).get("status", "")
                            memory_sections.append(
                                f"【相关任务】{label}" + (f"（状态：{status}）" if status else "")
                            )
                        elif ntype == "knowledge":
                            memory_sections.append(f"【知识参考】{content[:300]}")
                        elif ntype == "episode":
                            memory_sections.append(f"【历史摘要】{content[:200]}")
                        elif ntype in ("person", "concept"):
                            memory_sections.append(f"【知识参考】{label}: {content[:200]}")
                        else:
                            memory_sections.append(f"【参考】{content[:200]}")
                    
                    if memory_sections:
                        system_parts.append(
                            "\n【记忆上下文】\n" + "\n".join(memory_sections) + "\n"
                        )
                        logger.info(
                            f"[MemoryGraph] 注入 {len(memory_sections)} 条记忆到上下文"
                        )
        except Exception as e:
            logger.warning(f"[MemoryGraph] 记忆检索失败，降级跳过: {e}")
        
        # ========== 注意力状态感知 ==========
        # 统计已注入的记忆条目数量
        _mem_count = sum(1 for p in system_parts if "【记忆上下文】" in p or "【历史对话】" in p)
        _has_memory = _mem_count > 0
        _attn_lines = ["\n【注意力状态】"]
        if _has_memory:
            _attn_lines.append(f"已注入 {_mem_count} 段记忆/上下文到当前对话。")
            _attn_lines.append("如果这些信息不足以回答用户问题，请主动调用 recall_memory 工具检索更多相关记忆。")
        else:
            _attn_lines.append("当前对话未注入任何记忆上下文。")
            _attn_lines.append("如果用户的问题涉及历史信息或个人偏好，请主动调用 recall_memory 工具进行检索。")
        _attn_lines.append("如果需要用户补充信息才能继续，请直接用自然语言向用户提问。")
        _attn_lines.append("如果用户明确要求删除、移除、清除记忆，请调用 delete_memory_node 工具执行删除操作。\n")
        system_parts.append("\n".join(_attn_lines))
        
        system_parts.append("\n请开始回答用户的问题：")
        
        system_prompt = "".join(system_parts)
        
        logger.info("=" * 80)
        logger.info("📋 [SYSTEM PROMPT] 完整内容:")
        logger.info("=" * 80)
        logger.info(system_prompt)
        logger.info("=" * 80)
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": str(user_input)})
        
        return messages
    
    def _update_memory(self, user_input: str, response: str, fc_turn: int = 1):
        """更新对话记忆（同步版本，统一通过 MemoryGraph 写入）
        
        Args:
            user_input: 用户输入
            response: AI 回复
            fc_turn: FC 循环轮次
        """
        self._recent_turns_cache.append({"role": "user", "content": user_input})
        self._recent_turns_cache.append({"role": "assistant", "content": response})
        if len(self._recent_turns_cache) > self._recent_turns_max:
            self._recent_turns_cache = self._recent_turns_cache[-self._recent_turns_max:]
        
        try:
            from zulong.memory.memory_graph import get_memory_graph
            from zulong.memory.graph_adapters import DialogueAdapter
            
            mg = get_memory_graph()
            if mg:
                adapter = DialogueAdapter()
                
                # 优先复用 Gatekeeper 已创建的 round 节点（避免重复创建）
                gk_round_id = getattr(self, '_current_dialogue_round_id', None)
                
                if gk_round_id and mg.get_node(gk_round_id):
                    round_id = gk_round_id
                else:
                    # 降级: GK round 不可用，自行创建（session_id=None，后续由 L2 分配）
                    prev_round_id = getattr(self, '_last_round_id', None)
                    request_id = str(int(time.time() * 1000))
                    round_id = adapter.add_round(
                        mg, request_id, user_input,
                        prev_round_id=prev_round_id,
                        session_id=None,
                    )
                
                # L2 负责 Session 分配（Embedding 相似度匹配）
                session_id = adapter.assign_session_by_similarity(
                    mg, round_id, user_input, response,
                )
                # 缓存 session_id，供 retrieve_context 检索时做会话优先匹配
                self._current_session_id = session_id or ""
                
                # 添加 bot 回复为 sub_dialogue
                adapter.add_sub_dialogue(
                    mg, round_id, turn=fc_turn,
                    content=response, role="assistant",
                )
                
                # 完成对话轮次（索引到 FAISS 摘要侧车，供冷记忆检索）
                try:
                    adapter.finalize_round(
                        mg, round_id,
                        total_turns=fc_turn,
                        status="completed",
                    )
                except Exception as e:
                    logger.warning(f"[MemoryGraph] finalize_round 失败: {e}")
                
                self._last_round_id = round_id
                
                # 更新 round 节点内容
                round_node = mg.get_node(round_id)
                if round_node:
                    round_node.metadata["content"] = (
                        f"用户：{user_input}\n回答：{response[:200]}"
                    )
                    round_node.metadata["user_text"] = user_input
                    round_node.metadata["bot_text"] = response
                
                # 消耗一次性引用
                self._current_dialogue_round_id = None
                
                logger.info(f"[MemoryGraph] 对话已写入: {round_id} (session={session_id})")
        except Exception as e:
            logger.warning(f"[MemoryGraph] 记忆写入失败: {e}")
        
        try:
            if len(self._recent_turns_cache) % 20 == 0:
                logger.info("[经验生成] 批量处理对话历史...")
                stats = self.experience_generator.process_dialogue_batch(self._recent_turns_cache)
                logger.info(f"[经验生成] 处理统计：{stats}")
        except Exception as e:
            logger.warning(f"经验生成失败：{e}")
        
        logger.info(f"[记忆更新] 完成，当前缓存长度：{len(self._recent_turns_cache)}")
    
    def _retrieve_from_rag(self, query: str) -> Optional[str]:
        """从 RAG 检索相关记忆（ExperienceRAG 已被动化，不再自动注入）
        
        ExperienceRAG 改为由模型通过 search_experience 工具主动检索。
        此方法仅保留 knowledge / memory 库的自动检索路径。
        
        Args:
            query: 查询
            
        Returns:
            检索到的上下文文本（不含经验库结果）
        """
        rag_start = time.time()
        logger.info(f"[RAG] 开始检索记忆（跳过 ExperienceRAG）... (查询：'{query[:30]}...')")
        
        try:
            if not self.rag_manager:
                return None
            
            context_parts = []
            
            # 仅检索 knowledge 和 memory 库，跳过 experience 库
            for lib_name in ("knowledge", "memory"):
                try:
                    results = self.rag_manager.search(lib_name, query, top_k=3)
                    for doc in results:
                        content = getattr(doc, "content", "")
                        if content:
                            context_parts.append(content)
                except Exception:
                    pass
            
            rag_elapsed = time.time() - rag_start
            logger.info(f"[RAG] 检索完成！耗时：{rag_elapsed:.3f}秒，结果数：{len(context_parts)}")
            
            if context_parts:
                context = "\n\n".join(context_parts[:3])
                logger.info(f"[RAG] 检索到 {len(context_parts)} 条相关记忆（不含经验库）")
                return context
            
            return None
            
        except Exception as e:
            logger.warning(f"RAG 检索失败：{e}")
            return None
    
    def _load_visual_context(self) -> Optional[List[np.ndarray]]:
        """
        从共享视觉记忆池加载视觉数据 (主动轮询)
        
        策略:
        1. 检查 ./data/shared_vision/recent_clip.mp4 是否存在且新鲜 (<10 秒)
        2. 如果存在，提取关键帧 (3-5 帧)
        3. 如果不存在或过期，降级到 latest_frame.jpg
        4. 如果都失败，返回 None
        
        Returns:
            List[np.ndarray]: 关键帧列表，失败返回 None
        
        TSD v1.7 对应:
        - 4.2 L1-B: 视听信息流回溯
        - 5.2 显存约束：关键帧提取 (3-5 帧)
        """
        try:
            from zulong.l1a.vision_short_term_memory import VisionShortTermMemory
            
            # 创建临时管理器 (只读)
            memory = VisionShortTermMemory(
                duration=5,
                fps=30,
                cache_dir="./data/shared_vision"
            )
            
            # 检查新鲜度
            if memory.is_fresh(max_age=10.0):
                logger.info("👁️ 共享视觉记忆新鲜，提取关键帧...")
                
                # 从视频文件中提取关键帧
                import cv2
                
                if not os.path.exists(str(memory.video_path)):
                    logger.warning(f"👁️ 视频文件不存在：{memory.video_path}")
                    return None
                
                # 打开视频文件
                cap = cv2.VideoCapture(str(memory.video_path))
                
                if not cap.isOpened():
                    logger.error(f"👁️ 无法打开视频文件：{memory.video_path}")
                    return None
                
                # 获取视频信息
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                logger.info(f"👁️ 视频信息：{frame_count}帧，{fps}FPS, {duration:.2f}秒")
                
                # 提取关键帧 (使用 VisionShortTermMemory 的方法)
                # 先读取所有帧到临时缓冲区
                temp_frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    temp_frames.append(frame)
                
                cap.release()
                
                if not temp_frames:
                    logger.warning("👁️ 视频文件为空")
                    return None
                
                # 临时添加到缓冲区以使用 extract_keyframes
                for frame in temp_frames:
                    memory.add_frame(frame, time.time())
                
                # 提取关键帧 (3-5 帧)
                keyframes = memory.extract_keyframes(num_keyframes=5)
                
                logger.info(f"👁️ 提取关键帧：{len(keyframes)}帧")
                
                return keyframes
            
            else:
                # 降级到单帧
                logger.info("👁️ 视频记忆过期，降级到单帧...")
                
                if os.path.exists(str(memory.frame_path)):
                    import cv2
                    frame = cv2.imread(str(memory.frame_path))
                    
                    if frame is not None:
                        logger.info(f"👁️ 加载最新单帧：{memory.frame_path}")
                        return [frame]
                    else:
                        logger.warning(f"👁️ 无法读取单帧文件：{memory.frame_path}")
                else:
                    logger.warning(f"👁️ 单帧文件不存在：{memory.frame_path}")
                
                return None
        
        except Exception as e:
            logger.error(f"❌ 加载视觉上下文失败：{e}", exc_info=True)
            return None
    
    def _analyze_keyframes(self, keyframes: List[np.ndarray], query: str) -> str:
        """
        分析关键帧序列 (动态路由架构：已废弃)
        
        动态路由架构变更：
        - 此方法已废弃，保留仅为兼容性
        - L1-C 已完成视觉感知，不再需要 L2 调用 InternVL
        
        Args:
            keyframes: 关键帧列表
            query: 用户查询
        
        Returns:
            str: 视觉分析结果文本
        """
        logger.warning("⚠️ [_analyze_keyframes] 此方法已废弃，动态路由架构下不应被调用")
        return "（视觉分析已由 L1-C 完成）"
    
    def _request_vision_perception(self, query: str) -> Optional[str]:
        """
        请求视觉感知 (动态路由架构：使用 L1-C 传递的视觉上下文)
        
        动态路由架构变更：
        - L1-C 已完成意图识别 (WAVING/GAZING/APPROACHING)
        - L1-C 已生成关键帧截图 (Base64)
        - L2 不再调用 InternVL，直接使用 L1-C 的结果
        
        Args:
            query: 用户查询
            
        Returns:
            视觉感知结果文本
        
        TSD v1.8 对应:
        - 动态路由架构：L1-C 完成视觉感知，L2 仅做推理
        """
        try:
            logger.info("👁️ [动态路由] 使用 L1-C 视觉上下文...")
            
            # 检查是否有 L1-C 传递的视觉上下文
            if self._pending_visual_context:
                logger.info(f"👁️ [动态路由] 使用缓存的 L1-C 视觉上下文")
                result = self._pending_visual_context
                self._pending_visual_context = None
                return result
            
            # 如果没有 L1-C 上下文，返回默认提示
            logger.warning("👁️ [动态路由] 无 L1-C 视觉上下文，返回默认提示")
            return "（当前没有实时视觉数据，请根据上下文回答）"
        
        except Exception as e:
            logger.error(f"❌ 视觉感知失败：{e}", exc_info=True)
            return "视觉传感器数据未就绪"
    
    def _build_visual_context_from_l1c(
        self, 
        intent_type: str, 
        intent_confidence: float, 
        person_distance: float,
        keyframe_b64: Optional[str],
        crop_b64: Optional[str]
    ) -> str:
        """
        从 L1-C 传递的数据构建视觉上下文 (动态路由架构)
        
        动态路由架构变更：
        - L1-C 已完成意图识别 (WAVING/GAZING/APPROACHING)
        - L1-C 已生成关键帧截图 (Base64)
        - L2 直接使用 L1-C 的结果构建视觉上下文描述
        
        Args:
            intent_type: 意图类型 (WAVING/GAZING/APPROACHING)
            intent_confidence: 意图置信度
            person_distance: 人物距离 (米)
            keyframe_b64: 关键帧截图 (Base64)
            crop_b64: 裁剪人物区域 (Base64)
            
        Returns:
            str: 视觉上下文描述
        """
        # 根据意图类型构建自然语言描述
        intent_descriptions = {
            "WAVING": "用户在向你挥手",
            "GAZING": "用户正在注视着你",
            "APPROACHING": "用户走近了你"
        }
        
        base_desc = intent_descriptions.get(intent_type, f"检测到用户的交互意图：{intent_type}")
        
        # 构建简洁自然的视觉上下文
        visual_context = f"{base_desc}。请简短自然地回应，像朋友一样打招呼。"
        
        logger.info(f"👁️ [动态路由] 构建视觉上下文：{visual_context}")
        
        return visual_context
    
    def _complete_visual_sentence(self, text: str) -> str:
        """补全视觉描述句子 (避免截断)
        
        修复说明:
        1. ✅ 检测未完成的句子 (以逗号、"了"、"着"结尾)
        2. ✅ 截断到最后一个完整句号
        3. ✅ 如果没有句号，添加"..."标记
        
        Args:
            text: 视觉描述文本
            
        Returns:
            补全后的文本
        """
        if not text:
            return ""
        
        text = text.strip()
        
        # 检测是否以未完成标记结尾
        incomplete_endings = ['...', '了', '着', '，', ',', '在', '是']
        
        for ending in incomplete_endings:
            if text.endswith(ending):
                # 找到最后一个句号
                last_period = text.rfind('。')
                if last_period != -1 and last_period > 10:
                    # 截断到最后一个完整句号
                    text = text[:last_period + 1]
                    logger.debug(f"视觉信息已截断到完整句子：{text[:50]}...")
                else:
                    # 没有完整句号，添加标记
                    text = text + " (场景描述)"
                    logger.debug(f"视觉信息不完整，已添加标记")
                break
        
        return text
    
    async def _calculate_similarity_async(self, query: str, text: str) -> float:
        """计算查询与文本的语义相似度（异步版本）
        
        Args:
            query: 查询文本
            text: 待比较文本
            
        Returns:
            float: 相似度分数 (0-1)
        """
        try:
            # 1. 获取 embedding 模型
            from zulong.memory.embedding_manager import embedding_model
            
            # 2. 确保模型已加载
            if embedding_model.model is None:
                success = embedding_model.load_model()
                if not success:
                    logger.warning("[相似度计算] 模型加载失败，返回默认相似度 0.5")
                    return 0.5
            
            # 3. 计算向量
            query_vector = embedding_model.encode_query(query)
            text_vector = embedding_model.encode(text)[0]
            
            # 4. 余弦相似度
            import numpy as np
            similarity = np.dot(query_vector, text_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(text_vector) + 1e-8
            )
            
            # 归一化到 0-1
            similarity = (similarity + 1) / 2  # 从 [-1,1] 转到 [0,1]
            
            logger.debug(f"[相似度计算] query={query[:20]}..., similarity={similarity:.3f}")
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"[相似度计算] 失败：{e}，返回默认值 0.5")
            return 0.5
    
    async def _build_messages_with_history_async(self, user_input: str, rag_context: Optional[str], visual_context: Optional[str], search_context: Optional[str] = None) -> list:
        """构建包含历史、RAG 和视觉上下文的 messages
        
        修复版本:
        1. ✅ 防止模型复述规则 (移除强硬约束)
        2. ✅ 强制视觉融合 (没有视觉信息才能闲聊)
        3. ✅ 优化 Prompt 结构 (避免注意力迷失)
        4. ✅ 修复"你/我"人称混淆问题 (TSD v1.7 第 5.3 节)
        5. ✅ **关键修复：智能记忆检索与注入** (避免上下文污染)
        """
        from datetime import datetime
        
        # 🎯 获取当前时间
        now = datetime.now()
        hour = now.hour
        current_time_str = now.strftime("%Y-%m-%d %H:%M")
        
        # 🎯 根据时间段确定问候语提示
        if 5 <= hour < 11:
            time_period = "早晨"
        elif 11 <= hour < 14:
            time_period = "中午"
        elif 14 <= hour < 18:
            time_period = "下午"
        elif 18 <= hour < 22:
            time_period = "晚上"
        else:
            time_period = "深夜"
        
        # 🔥 关键修复 2: 强化身份认知和人称代词定义
        system_parts = [
            "**重要身份认知**：",
            "- 你的名字叫  \"  祖龙 (ZULONG)  \"",
            f"\n当前时间：{current_time_str} ({time_period})。",
            "\n【人称代词】",
            "- \"我\" 指的是你自己（祖龙）",
            "- \"你\" 指的是用户",
            "- 当用户说\"我家\"、\"我叫\"时，指的是用户",
            "\n【交流风格】",
            "用自然、友好的口语和用户对话，就像朋友聊天一样。",
            "\n【任务管理规则】",
            "当用户要求你完成复杂的多步骤任务时（如：开发项目、编写代码、设计方案、写报告、做游戏等），",
            "必须按以下步骤操作：",
            "",
            "步骤1. 【创建总节点】调用 task_create_plan，传入：",
            "  - title: 简短标题（如\"Python爬虫项目方案\"）",
            "  - user_requirement: 必须原样复制用户的完整原始需求文本，不得概括或缩写",
            "",
            "步骤2. 【创建子节点大纲】用 task_add_node 逐个添加子任务节点，parent_id='req'（挂到总节点下）",
            "  - 每个子节点代表一个独立的工作模块/步骤",
            "  - 先搭建完整大纲再执行，不要边做边加",
            "",
            "步骤3. 【建立依赖关系】用 task_add_dependency 声明节点间的先后顺序",
            "  - 当任务B需要任务A的产出时，必须建立依赖: task_add_dependency(from_id=A, to_id=B)",
            "  - 依赖关系决定执行顺序，也会在前端图谱中显示为虚线连线",
            "",
            "步骤4. 用 task_view_overview 查看任务概览确认结构",
            "步骤5. 【逐节点执行】按依赖顺序逐个执行子任务：",
            "  - 开始执行前：task_mark_status(node_id='xxx', status='in_progress')",
            "  - 生成该节点的具体内容（不要一次性生成所有节点的内容）",
            "  - 完成后：task_mark_status(node_id='xxx', status='completed', result='该节点的完整结果')",
            "  - 然后继续下一个节点，直到所有节点都标记为 completed",
            "  ⚠️ 严禁跳过 task_mark_status 直接输出完整回复！每个节点必须单独标记。",
            "步骤6. 所有子任务完成后，再生成最终的汇总回复给用户",
            "",
            "【任务图谱模板】下面是一个标准的任务图谱结构示例：",
            "用户说：\"帮我设计一个电商系统，包含用户管理、商品管理、订单管理三个模块\"",
            "",
            "你应该这样调用工具：",
            "① task_create_plan(title=\"电商系统设计方案\", user_requirement=\"帮我设计一个电商系统，包含用户管理、商品管理、订单管理三个模块\")",
            "   → 创建总节点 req，存储用户完整原始需求",
            "② task_add_node(parent_id=\"req\", label=\"用户管理模块\", desc=\"设计用户注册、登录、权限管理功能\")",
            "   → 创建子节点 o1，挂在 req 下",
            "③ task_add_node(parent_id=\"req\", label=\"商品管理模块\", desc=\"设计商品上架、分类、搜索功能\")",
            "   → 创建子节点 o2，挂在 req 下",
            "④ task_add_node(parent_id=\"req\", label=\"订单管理模块\", desc=\"设计下单、支付、物流跟踪功能\")",
            "   → 创建子节点 o3，挂在 req 下",
            "⑤ task_add_dependency(from_id=\"o1\", to_id=\"o3\", via=\"用户身份信息\")",
            "   → 订单模块依赖用户模块（需要用户信息才能下单）",
            "⑥ task_add_dependency(from_id=\"o2\", to_id=\"o3\", via=\"商品数据\")",
            "   → 订单模块依赖商品模块（需要商品信息才能下单）",
            "⑦ task_view_overview() → 确认结构",
            "",
            "最终生成的树形结构（含依赖连线）：",
            "  req (电商系统设计方案) ← 总节点，保存用户原始需求",
            "   ├── o1 (用户管理模块) ──依赖──→ o3",
            "   ├── o2 (商品管理模块) ──依赖──→ o3",
            "   └── o3 (订单管理模块) ← 依赖 o1、o2 完成后才能开始",
            "",
            "如果子任务本身还很复杂，可以继续往下拆分子节点：",
            "  task_add_node(parent_id=\"o1\", label=\"注册功能\", desc=\"手机号+邮箱注册\")",
            "   → 创建子节点 o1_1，挂在 o1 下",
            "",
            "重要规则：",
            "- 第1步的 user_requirement 必须保存用户原话，这是后续恢复任务时还原上下文的关键依据",
            "- 所有子节点必须通过 parent_id 正确挂到父节点下，形成树形结构",
            "- 有先后顺序的任务之间必须用 task_add_dependency 建立依赖，确保执行顺序正确",
            "- 如果任务较简单（单步即可完成），可直接回答无需创建任务图",
            "",
            "【任务挂起与恢复规则】",
            "当用户在任务进行中说\"暂停\"、\"先不做了\"、\"先聊别的\"、\"换个话题\"等意思时：",
            "  → 必须调用 task_suspend(reason=\"user_requested\") 将当前任务图持久化到磁盘",
            "  → 不要只是口头说\"已暂停\"而不调用工具，否则任务状态会丢失",
            "  → 挂起后清除当前活跃任务图，可以正常回答其他问题",
            "",
            "当用户说\"继续\"、\"接着做\"、\"上次那个任务\"、\"恢复之前的任务\"等意思时：",
            "  → 第一步：调用 task_list_suspended(query=\"相关描述\") 查找并恢复挂起的任务",
            "  → 第二步：如果返回 resumed=true，说明之前的任务图（含所有节点和状态）已自动恢复到内存",
            "  → 第三步：调用 task_view_overview() 查看恢复后的任务图当前进度",
            "  → 第四步：阅读概览，找到状态为 pending 或 not_started 的第一个节点，从那里继续执行",
            "  → 第五步：用 task_mark_status 更新该节点为 in_progress，然后执行该节点的具体内容",
            "",
            "  恢复后的绝对禁止事项：",
            "  ✗ 禁止调用 task_create_plan — 这会创建全新图谱，丢弃已恢复的进度",
            "  ✗ 禁止调用 task_add_node — 节点已经在恢复的图谱中了，不需要重新添加",
            "  ✗ 禁止重命名或重建已有节点 — 直接使用恢复后图谱中现有的节点",
            "  ✓ 只使用 task_mark_status 更新现有节点状态，然后继续执行",
        ]
        
        # ========== 推理层级感知 ==========
        _phase = "自由对话"
        _layer_info = "L2 推理引擎（顶层）"
        try:
            _te = getattr(self, 'tool_engine', None)
            _tm = getattr(_te, 'task_manager', None) if _te else None
            _tg = getattr(_tm, 'task_graph', None) if _tm else None
            if _tg and getattr(_tg, 'root_id', None):
                _phase = "任务恢复" if getattr(self, '_is_resume_task', False) else "任务执行"
                _current = getattr(_tg, 'current_node_id', None) or _tg.root_id
                _depth = 0
                _node = _current
                while _node and _node != _tg.root_id:
                    _parent = getattr(_tg.nodes.get(_node), 'parent_id', None) if hasattr(_tg, 'nodes') else None
                    if _parent:
                        _depth += 1
                        _node = _parent
                    else:
                        break
                _layer_info = f"L2 推理引擎 → 任务图第 {_depth} 层"
        except Exception:
            pass
        system_parts.append(
            f"\n【推理层级】当前阶段：{_phase} | 层级：{_layer_info}\n"
            "你可以根据当前层级调整回答深度：自由对话时简洁友好；任务执行时严谨有条理；任务恢复时先确认上下文再继续。\n"
        )
        
        # 🔥 关键修复 1: 有视觉信息时，强制要求基于视觉回答
        if visual_context:
            # 检测是否是挥手/注视/靠近等简单交互
            is_simple_greeting = any(kw in visual_context for kw in ["挥手", "注视", "走近"])
            
            if is_simple_greeting:
                system_parts.append("""
【回应风格】
用户正在和你互动！请用简短、活泼、口语化的方式回应，就像朋友打招呼一样。
- 挥手：可以说"你好呀~"、"嗨~"、"怎么啦？"、"有什么事吗？"
- 注视：可以说"看什么呢~"、"需要帮忙吗？"、"想聊点什么？"
- 靠近：可以说"来啦~"、"找我有什么事吗？"
打招呼回复控制在 40 字以内，像真人对话一样自然。

【视觉观察】
""" + visual_context + "\n")
            else:
                system_parts.append("""
【回答建议】
1. 直接基于视觉观察回答用户问题
2. 避免复述系统规则或约束条件
3. 如果视觉信息不足，诚实告知用户
4. 使用自然口语，50-150 字
5. 避免使用数字列表或项目符号

【视觉观察】
""" + visual_context + "\n")
        else:
            # 没有视觉信息，允许闲聊
            system_parts.append("""
【回答建议】
1. 友好、专业地回答用户问题
2. 如果信息不足，诚实告知用户
3. 使用自然流畅的口语，50-150 字
""")
        
        # 🔥 关键修复 2: RAG 信息作为参考，不是约束
        if rag_context:
            system_parts.append(f"\n【参考知识】\n{rag_context}\n")
        
        # 🔥 添加搜索上下文
        if search_context:
            system_parts.append(f"\n{search_context}\n")
        
        # ========== 思维导航注入（思维深度索引） ==========
        # 在记忆上下文之前注入焦点路径摘要，让模型感知自己当前
        # "站在"图的哪个位置（从根到当前焦点的 L1→L2→L3 树形路径）
        try:
            from zulong.memory.memory_graph import get_memory_graph as _get_mg_nav
            _mg_nav = _get_mg_nav()
            if _mg_nav:
                focus_summary = _mg_nav.get_focus_path_summary()
                if focus_summary:
                    system_parts.append(f"\n{focus_summary}\n")
                    logger.debug(f"[思维导航] 已注入焦点路径 ({len(focus_summary)} chars)")
        except Exception as e:
            logger.debug(f"[思维导航] 注入跳过: {e}")
        
        # ========== 记忆注入：通过 MemoryGraph 统一检索 ==========
        # 替代旧的 conversation_history / short_term_memory / episodic_memory 注入
        # 使用 retrieve_context() 的分区并行检索（热数据遍历 + 非热数据 FAISS）
        try:
            from zulong.memory.memory_graph import get_memory_graph
            _mg = get_memory_graph()
            if _mg:
                # 延迟注入 RAGManager（确保此时已完成初始化）
                if not getattr(_mg, '_rag_manager', None) and self.rag_manager:
                    _mg.set_rag_manager(self.rag_manager)
                mg_results = await _mg.retrieve_context(
                    user_input, top_k=8,
                    session_id=getattr(self, '_current_session_id', ""),
                )
                if mg_results:
                    memory_sections = []
                    for r in mg_results:
                        ntype = r.get("node_type", "")
                        content = r.get("content", "")
                        label = r.get("label", "")
                        if not content:
                            continue
                        # 按节点类型格式化
                        if ntype == "experience":
                            # EXPERIENCE 不自动注入（由 search_experience FC 工具按需获取）
                            continue
                        elif ntype == "dialogue":
                            memory_sections.append(f"【历史对话】{content[:200]}")
                        elif ntype == "task":
                            status = r.get("metadata", {}).get("status", "")
                            memory_sections.append(
                                f"【相关任务】{label}" + (f"（状态：{status}）" if status else "")
                            )
                        elif ntype == "knowledge":
                            memory_sections.append(f"【知识参考】{content[:300]}")
                        elif ntype == "episode":
                            memory_sections.append(f"【历史摘要】{content[:200]}")
                        elif ntype in ("person", "concept"):
                            memory_sections.append(f"【知识参考】{label}: {content[:200]}")
                        else:
                            memory_sections.append(f"【参考】{content[:200]}")

                    if memory_sections:
                        system_parts.append(
                            "\n【记忆上下文】\n" + "\n".join(memory_sections) + "\n"
                        )
                        logger.info(
                            f"[MemoryGraph] 注入 {len(memory_sections)} 条记忆到上下文"
                        )
        except Exception as e:
            logger.warning(f"[MemoryGraph] 记忆检索失败，降级跳过: {e}")
        
        # ========== 注意力状态感知 ==========
        _mem_count = sum(1 for p in system_parts if "【记忆上下文】" in p or "【历史对话】" in p)
        _has_memory = _mem_count > 0
        _attn_lines = ["\n【注意力状态】"]
        if _has_memory:
            _attn_lines.append(f"已注入 {_mem_count} 段记忆/上下文到当前对话。")
            _attn_lines.append("如果这些信息不足以回答用户问题，请主动调用 recall_memory 工具检索更多相关记忆。")
        else:
            _attn_lines.append("当前对话未注入任何记忆上下文。")
            _attn_lines.append("如果用户的问题涉及历史信息或个人偏好，请主动调用 recall_memory 工具进行检索。")
        _attn_lines.append("如果需要用户补充信息才能继续，请直接用自然语言向用户提问。")
        _attn_lines.append("如果用户明确要求删除、移除、清除记忆，请调用 delete_memory_node 工具执行删除操作。\n")
        system_parts.append("\n".join(_attn_lines))
        
        # 最后一句话引导
        system_parts.append("\n请开始回答用户的问题：")
        
        system_prompt = "".join(system_parts)
        
        # 🔥 [调试] 打印完整的 System Prompt
        logger.info("=" * 80)
        logger.info("📋 [SYSTEM PROMPT] 完整内容:")
        logger.info("=" * 80)
        logger.info(system_prompt)
        logger.info("=" * 80)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # 多轮上下文注入已完全由 MemoryGraph.retrieve_context() 热窗口覆盖
        # （上方 【记忆上下文】 section 已包含 DIALOGUE 类型的最近对话节点）
        # 不再使用 conversation_history 作为注入源
        
        logger.info(f"[DEBUG] 最终消息结构：system + 1 条用户输入（历史由 MemoryGraph 热窗口提供）")
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": str(user_input)})  # 强制转换为字符串
        
        return messages
    
    async def _update_memory_async(self, user_input: str, response: str, fc_turn: int = 1):
        """更新对话记忆（统一通过 MemoryGraph 写入）
        
        Args:
            user_input: 用户输入
            response: AI 回复
            fc_turn: FC 循环轮次
        """
        # 1. 近期轮次缓存（仅用于经验提取 / 纠错检测，NOT 用于 LLM 上下文注入）
        self._recent_turns_cache.append({"role": "user", "content": user_input})
        self._recent_turns_cache.append({"role": "assistant", "content": response})
        if len(self._recent_turns_cache) > self._recent_turns_max:
            self._recent_turns_cache = self._recent_turns_cache[-self._recent_turns_max:]
        
        # 2. 统一写入 MemoryGraph（替代旧的 STM / EpisodicMemory 双写）
        try:
            from zulong.memory.memory_graph import get_memory_graph
            from zulong.memory.graph_adapters import DialogueAdapter
            
            mg = get_memory_graph()
            if mg:
                adapter = DialogueAdapter()
                
                # 优先复用 Gatekeeper 已创建的 round 节点（避免重复创建）
                gk_round_id = getattr(self, '_current_dialogue_round_id', None)
                
                if gk_round_id and mg.get_node(gk_round_id):
                    round_id = gk_round_id
                else:
                    # 降级: GK round 不可用，自行创建（session_id=None，后续由 L2 分配）
                    prev_round_id = getattr(self, '_last_round_id', None)
                    request_id = _current_request_id_var.get() or str(int(time.time() * 1000))
                    round_id = adapter.add_round(
                        mg, request_id, user_input,
                        prev_round_id=prev_round_id,
                        task_graph_id=self._active_task_graph_id,
                        session_id=None,
                    )
                
                # L2 负责 Session 分配（Embedding 相似度匹配）
                session_id = adapter.assign_session_by_similarity(
                    mg, round_id, user_input, response,
                )
                # 缓存 session_id，供 retrieve_context 检索时做会话优先匹配
                self._current_session_id = session_id or ""
                
                # 将 bot 回复写入 sub_dialogue
                adapter.add_sub_dialogue(
                    mg, round_id, turn=fc_turn,
                    content=response, role="assistant",
                )
                
                # 记录本轮 round_id 供下轮使用
                self._last_round_id = round_id
                
                # 更新 round 节点的 content 以便检索
                round_node = mg.get_node(round_id)
                if round_node:
                    round_node.metadata["content"] = (
                        f"用户：{user_input}\n回答：{response[:200]}"
                    )
                    round_node.metadata["user_text"] = user_input
                    round_node.metadata["bot_text"] = response
                
                # 消耗一次性引用
                self._current_dialogue_round_id = None
                
                logger.info(f"[MemoryGraph] 对话已写入: {round_id} (session={session_id})")
        except Exception as e:
            logger.warning(f"[MemoryGraph] 记忆写入失败: {e}")
        
        # 3. 经验自动生成（保留，写入 ExperienceRAG）
        try:
            if len(self._recent_turns_cache) % 20 == 0:  # 每 10 轮对话
                logger.info("[经验生成] 批量处理对话历史...")
                stats = self.experience_generator.process_dialogue_batch(self._recent_turns_cache)
                logger.info(f"[经验生成] 处理统计：{stats}")
        except Exception as e:
            logger.warning(f"经验生成失败：{e}")
        
        logger.info(f"[记忆更新] 完成，当前缓存长度：{len(self._recent_turns_cache)}")
    
    # 🔥 v3.0 新增：自动经验提取方法 (解决断层 1)
    async def _auto_extract_experience(self):
        """自动提取经验 (解决断层 1)
        
        调用时机:
        - 检测到纠错后自动触发
        - 任务完成后 (可选)
        - ~~每 5 轮对话~~ (v3.0 不启用，避免算力浪费)
        """
        logger.info("[InferenceEngine] 开始自动提取经验...")
        
        try:
            # 1. 从近期轮次缓存中提取
            candidates = self.experience_generator.extract_from_dialogue(
                self._recent_turns_cache
            )
            
            logger.info(f"[InferenceEngine] 提取了 {len(candidates)} 个经验候选")
            
            # 2. 过滤低置信度
            high_confidence = [
                c for c in candidates 
                if c.confidence >= 0.6
            ]
            
            # 3. 🔴 关键：调用存储层的 add_experience (包含去重逻辑)
            success_count = 0
            for candidate in high_confidence:
                # ✅ 这里调用的 add_experience 必须包含 _is_duplicate 检查
                # 如果复盘刚好也在提取同一条，谁先抢到锁谁写，后写的会被拒绝
                success = self.experience_generator.add_experience_to_rag(candidate)
                if success:
                    success_count += 1
                    logger.debug(f"[InferenceEngine] 已保存经验：{candidate.content[:50]}...")
                else:
                    logger.debug(f"[InferenceEngine] 跳过重复经验：{candidate.content[:50]}...")
            
            logger.info(f"[InferenceEngine] 成功保存 {success_count} 条经验")
            
        except Exception as e:
            logger.error(f"[InferenceEngine] 自动提取经验失败：{e}", exc_info=True)
    
    def _is_recent_correction(self) -> bool:
        """检测最近是否有纠错
        
        Returns:
            bool: 是否有纠错
        """
        # 检查最近 3 轮对话是否包含纠错关键词
        correction_keywords = ["错误", "失败", "不对", "有问题", "Exception", "不对", "错了"]
        
        recent_turns = self._recent_turns_cache[-6:]  # 最近 3 轮 (user+assistant)
        
        for turn in recent_turns:
            content = turn.get("content", "")
            if any(kw in content for kw in correction_keywords):
                return True
        
        return False
    
    def _on_interrupt(self, event: ZulongEvent):
        """处理中断事件
        
        Args:
            event: 中断事件
        """
        logger.info("🚨 收到中断信号")
        with self._lock:
            self._interrupt_flag = True
    
    def _get_kv_cache(self, session_id: str) -> Optional[Any]:
        """获取会话的 KV Cache
        
        Args:
            session_id: 会话 ID
            
        Returns:
            Optional[Any]: KV Cache 数据，不存在或过期返回 None
        """
        # 检查是否存在
        if session_id not in self.kv_cache_pool:
            return None
        
        # 检查是否过期
        last_used = self.kv_cache_last_used.get(session_id, 0)
        if time.time() - last_used > self.kv_cache_ttl:
            logger.info(f"🗑️ [KV Cache] 会话 {session_id} 缓存已过期，清除")
            del self.kv_cache_pool[session_id]
            del self.kv_cache_last_used[session_id]
            return None
        
        logger.info(f"♻️ [KV Cache] 复用会话 {session_id} 的缓存")
        return self.kv_cache_pool[session_id]
    
    def _save_kv_cache(self, session_id: str, past_key_values: Any):
        """保存会话的 KV Cache
        
        Args:
            session_id: 会话 ID
            past_key_values: KV Cache 数据
        """
        self.kv_cache_pool[session_id] = past_key_values
        self.kv_cache_last_used[session_id] = time.time()
        logger.debug(f"💾 [KV Cache] 保存会话 {session_id} 的缓存")
    
    def _generate_short_response(self, user_input: str) -> str:
        """生成简短回复 (用于语音指令触发)
        
        Args:
            user_input: 用户输入
            
        Returns:
            str: 简短的回复
        """
        try:
            # 构建简短 Prompt
            messages = [
                {"role": "system", "content": "你是祖龙 (ZULONG) 机器人助手。请用简洁、口语化的方式回答，适合语音播报。控制在 50 字以内。"},
                {"role": "user", "content": user_input}
            ]
            
            prompt = self.l2_model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 生成回复 (限制最大长度)
            # 🔥 修复：使用正确的模型调用方式
            # self.l2_model 是 RealModelLoader 对象，使用属性访问
            
            # 构建输入
            inputs = self.l2_model.tokenizer(prompt, return_tensors="pt")
            input_length = inputs.input_ids.shape[1]
            
            # 移动到正确的设备
            device = self.l2_model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 生成参数
            max_new_tokens = 128  # 限制长度，支持短回复
            temperature = 0.7
            top_p = 0.9
            
            # 生成
            import torch
            with torch.no_grad():
                outputs = self.l2_model.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.l2_model.tokenizer.eos_token_id
                )
            
            # 提取生成的文本
            generated_ids = outputs[0][input_length:]
            response = self.l2_model.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 提取 </think> 之后的内容
            if "</think>" in response:
                parts = response.split("</think>")
                if len(parts) > 1:
                    response = parts[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"生成简短回复失败：{e}")
            return "抱歉，我暂时无法回答。"
        
        video_path = event.payload.get("video_path")
        duration = event.payload.get("duration")
        frame_count = event.payload.get("frame_count")
        
        logger.info(f"   - 视频路径：{video_path}")
        logger.info(f"   - 时长：{duration}秒")
        logger.info(f"   - 帧数：{frame_count}帧")
        
        # 🎯 关键修改：直接同步调用分析方法，避免异步事件循环问题
        # 在后台线程中执行视频分析 (不阻塞事件总线)
        import threading
        thread = threading.Thread(
            target=self._analyze_video_content,
            args=(video_path,),
            daemon=True
        )
        thread.start()
        logger.debug(f"👁️ 启动视频分析线程")
    
    def _analyze_video_content(self, video_path: str) -> Optional[str]:
        """分析视频内容（使用多模态模型）
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            视频内容描述文本
        """
        try:
            logger.info(f"👁️ 开始分析视频内容：{video_path}")
            
            # 检查视频文件是否存在
            if not os.path.exists(video_path):
                logger.error(f"视频文件不存在：{video_path}")
                return None
            
            # 使用 OpenCV 提取关键帧 (提取 10 帧)
            import cv2
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error("无法打开视频文件")
                return None
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            logger.info(f"   - FPS: {fps}, 总帧数：{frame_count}, 时长：{duration:.2f}秒")
            
            # 提取关键帧 (均匀采样 10 帧)
            num_keyframes = min(10, frame_count)
            step = max(1, frame_count // num_keyframes)
            keyframes = []
            
            for i in range(0, frame_count, step):
                ret, frame = cap.read()
                if ret:
                    # 转换 BGR 到 RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    keyframes.append(frame_rgb)
            
            cap.release()
            
            logger.info(f"   - 提取关键帧：{len(keyframes)}帧")
            
            if not keyframes:
                logger.warning("未能提取到关键帧")
                return None
            
            # 使用多模态模型分析关键帧序列
            # 构建提示词
            prompt = """你是一个视觉分析专家。请分析这组从视频中提取的关键帧：

1. 描述视频中的场景和环境
2. 识别视频中的人物、物体及其动作
3. 如果有物体移动或状态变化，请详细描述
4. 总结视频的核心内容

请清晰、简洁地描述，不超过 200 字。
"""
            
            # 使用第一帧和最后一帧进行对比分析 (如果有)
            if len(keyframes) >= 2:
                # 使用多模态模型分析
                # 注意：这里需要使用支持多图像的模型
                # 暂时简化为分析第一帧
                first_frame = keyframes[0]
                
                # 调用视觉模型 (使用 L1-A 的 VisionNode 或 L2 的多模态模型)
                # 这里简化处理，实际需要调用 ModelContainer 中的多模态模型
                visual_description = f"视频包含 {len(keyframes)} 个关键帧，展示了动态场景变化。"
                
                # 🔥 关键修复：视觉信息补全 (避免截断)
                visual_description = self._complete_visual_sentence(visual_description)
                
                # 缓存分析结果
                self._pending_visual_context = visual_description
                logger.info(f"✅ 视频分析完成：{visual_description[:100]}...")
                
                return visual_description
            
            return None
            
        except Exception as e:
            logger.error(f"视频分析失败：{e}", exc_info=True)
            return None
    
    def generate(self, context: list, stop_condition: Optional[Callable] = None):
        """生成过程（使用真实模型）
        
        Args:
            context: 上下文
            stop_condition: 停止条件
            
        Yields:
            str: 生成的 token
        """
        logger.info("Starting generation with real model")
        
        prompt_text = " ".join([str(item) for item in context if isinstance(item, str)])
        prompt = f"请继续：{prompt_text}"
        
        try:
            response = self.l2_model.generate(prompt, max_tokens=100)
            
            for char in response:
                with self._lock:
                    _interrupted = self._interrupt_flag
                if _interrupted:
                    logger.info("Generation interrupted")
                    return
                if stop_condition and stop_condition():
                    logger.info("Stop condition met")
                    return
                yield char
            
            logger.info("Generation completed")
            
        except Exception as e:
            logger.error(f"生成失败：{e}")
    
    def load_model(self):
        """加载模型"""
        logger.info("Loading L2 model...")
        try:
            self.l2_model = self.model_container.get_model(ModelID.L2_CORE)
            state_manager.set_l2_status(L2Status.IDLE)
            logger.info("L2 model loaded successfully")
        except Exception as e:
            logger.error(f"模型加载失败：{e}")
            raise
    
    def unload_model(self):
        """卸载模型"""
        logger.info("Unloading L2 model...")
        try:
            self.model_container.unload_model(ModelID.L2_CORE)
            state_manager.set_l2_status(L2Status.UNLOADED)
            logger.info("L2 model unloaded successfully")
        except Exception as e:
            logger.error(f"模型卸载失败：{e}")
            raise
    
    def is_generating(self) -> bool:
        """
        检查 L2 是否正在生成回复 (动态路由架构核心接口)
        
        Returns:
            bool: True 表示 L2 正在忙碌 (BUSY/WAITING)，False 表示空闲 (IDLE)
        """
        effective_status = state_manager.get_effective_status()
        is_busy = effective_status == "ACTIVE_TASK"
        
        logger.debug(f"🔍 [InferenceEngine] is_generating() = {is_busy} (effective_status={effective_status})")
        return is_busy
    
    def _load_rag_data(self):
        """加载 RAG 数据（从磁盘）
        
        Returns:
            RAGManager: 加载的 RAG 管理器，失败则返回空RAGManager
        """
        try:
            import os
            from zulong.memory.rag_manager import RAGManager, RAGConfig
            
            rag_path = "./data/rag"
            
            # 🔥 关键修复：如果目录不存在，创建它并返回空的RAGManager
            if not os.path.exists(rag_path):
                logger.warning(f"📁 RAG 数据目录不存在，正在创建：{rag_path}")
                os.makedirs(rag_path, exist_ok=True)
            
            # 创建 RAGManager 配置
            config = RAGConfig(
                vector_dimension=512,
                vector_store_type="faiss",
                experience_rag_enabled=True,
                memory_rag_enabled=True,
                knowledge_rag_enabled=True
            )
            rag_manager = RAGManager(config)
            
            # 尝试加载已有数据（如果有）
            try:
                rag_manager.load_all(rag_path)
                logger.info(f"✅ RAG 数据已加载：{rag_path}")
            except Exception as load_err:
                logger.warning(f"⚠️ RAG 数据加载失败（可能是首次运行）：{load_err}")
            
            return rag_manager
        except Exception as e:
            logger.error(f"❌ 创建 RAG 管理器失败：{e}", exc_info=True)
            return None


# 全局推理引擎实例
inference_engine = InferenceEngine()

# 兼容旧代码的别名
mock_inference_engine = inference_engine
