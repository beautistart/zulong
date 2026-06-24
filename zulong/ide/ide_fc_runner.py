"""
IDE 模式 FC 循环运行器

继承自 FCRunner (zulong.l2.fc_runner)，复用通用安全网逻辑。
使用 Python while 循环替代 LangGraph StateGraph，支持跨 HTTP 请求的暂停/恢复。

IDE 特有功能：
1. 流式推送 (display_text 逐句推送到前端)
2. 工具调用分流：内部工具直接执行，远程工具暂停返回
3. 状态完全序列化到 IDEFCState
4. 注意力窗口/RuleGuardian/CircuitBreaker per-runner 实例
5. 参数验证/别名映射
6. 429 限流重试 + 流式处理
"""

import asyncio
import concurrent.futures
import atexit
import hashlib
import json as _json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from zulong.l2.fc_runner import FCRunner
from zulong.l2.attention_pressure_view import build_threshold_pressure_view
from zulong.l2.attention_context_retrieval import (
    build_attention_context_bundle,
    render_attention_context_message,
)
from zulong.ide.ide_session import IDEFCState, AgentSession
from zulong.ide.ide_tool_registry import (
    ANNOUNCE_STEP_TOOL_NAME,
    IDEToolRegistry,
    IDE_REMOTE_TOOLS,
)
from zulong.ide.ide_format_translator import IDEFormatTranslator
from zulong.ide.common.error_handler import ErrorHandler, ErrorCode
from zulong.l2.attention_window import MAX_TOOL_RESULT_CHARS
from zulong.l2.circuit_breaker import CircuitBreakerState
from zulong.core.unified_protocol import MessageType
from zulong.core.message_visibility import (
    CHANNEL_CONTROL,
    CHANNEL_FINAL,
    CHANNEL_LEDGER,
    CHANNEL_STATUS,
    UX_DETAILS,
    UX_HIDDEN,
    UX_MAIN,
    internal_control_message,
    mark_hidden_payload,
    mark_public_payload,
)
from zulong.l2.tool_budget import (
    detect_tool_call_budget,
    engine_tool_budget_exhausted,
    get_engine_tool_budget,
    get_engine_tool_calls_used,
    record_engine_tool_calls_used,
    sync_engine_tool_budget,
)
from zulong.l2.tool_capabilities import tool_capabilities

if TYPE_CHECKING:
    from zulong.l2.inference_engine import InferenceEngine

_SHARED_INTENT_FILTER = None
_SHARED_INTENT_FILTER_LOCK = threading.Lock()
_EMBEDDING_PREWARM_STARTED = False
_EMBEDDING_PREWARM_LOCK = threading.Lock()

_FRIENDLY_TOOL_NAMES = {
    "ide_write_file": "写入文件",
    "exec_write_file": "写入文件",
    "write_to_file": "写入文件",
    "ide_read_file": "读取文件",
    "exec_read_file": "读取文件",
    "read_file": "读取文件",
    "ide_replace_text": "修改文件",
    "replace_in_file": "修改文件",
    "list_files": "查看目录",
    "listFilesTopLevel": "查看目录",
    "listFilesRecursive": "查看目录",
    "search_files": "搜索文件",
    "searchFiles": "搜索文件",
    "search_code_symbols": "检索代码上下文",
    "listCodeDefinitionNames": "检索代码结构",
    "zulong_code_query": "检索代码上下文",
    "get_symbol_context": "读取符号上下文",
    "get_impact_analysis": "分析影响范围",
    "index_code_file": "建立代码索引",
    "exec_run_command": "执行命令",
    "execute_command": "执行命令",
    "task_create_plan": "制定任务计划",
    "start_task_plan": "制定任务计划",
    "task_view_overview": "查看任务进度",
    "task_add_node": "添加任务步骤",
    "task_mark_status": "更新任务状态",
    "submit_final_answer": "整理最终回复",
    "attempt_completion": "整理最终回复",
    "recall_memory": "检索记忆",
    "read_memory_node": "读取记忆",
    "save_memory_note": "保存偏好",
    "discover_related": "发现相关记忆",
    "browser_action": "操作浏览器",
    "web_search": "联网检索",
    "webSearch": "联网检索",
    "request_tool_supplement": "补充工具能力",
    "ide_open_workspace": "打开 VS Code 工作区",
    "ide_open_file": "打开文件",
    "ide_open_terminal": "打开终端",
    "ide_show_diff": "查看差异",
    "ide_get_context": "获取 IDE 上下文",
}

_BACKGROUND_TOOLS = frozenset({
    ANNOUNCE_STEP_TOOL_NAME,
    "recall_memory",
    "read_memory_node",
    "discover_related",
    "ide_get_context",
    "request_tool_supplement",
})

_SUMMARY_ONLY_TOOLS = frozenset({
    "save_memory_note",
})

_READ_TOOLS = frozenset({
    "ide_read_file",
    "read_file",
    "list_files",
    "listFilesTopLevel",
    "listFilesRecursive",
    "search_files",
    "searchFiles",
    "search_code_symbols",
    "listCodeDefinitionNames",
    "zulong_code_query",
    "get_symbol_context",
    "get_impact_analysis",
    "index_code_file",
})

_WRITE_TOOLS = frozenset({
    "ide_write_file",
    "exec_write_file",
    "write_to_file",
    "ide_replace_text",
    "replace_in_file",
})

_COMMAND_TOOLS = frozenset({
    "exec_run_command",
    "execute_command",
    "ide_open_terminal",
})

_TASK_GRAPH_TOOLS = frozenset({
    "task_create_plan",
    "start_task_plan",
    "task_view_overview",
    "task_add_node",
    "task_mark_status",
})

_NETWORK_TOOLS = frozenset({
    "browser_action",
    "web_search",
    "webSearch",
})


def _get_shared_intent_filter():
    """进程级复用 L1-B IntentFilter，避免每个 IDE 会话重复初始化模型。"""
    global _SHARED_INTENT_FILTER
    if _SHARED_INTENT_FILTER is not None:
        return _SHARED_INTENT_FILTER

    with _SHARED_INTENT_FILTER_LOCK:
        if _SHARED_INTENT_FILTER is not None:
            return _SHARED_INTENT_FILTER
        try:
            from zulong.l1b.intent_filter import IntentFilter
            try:
                from zulong.config.config_manager import get_config
                intent_config = get_config("intent_classification", {})
            except Exception:
                intent_config = {}
            _SHARED_INTENT_FILTER = IntentFilter(config=intent_config)
            return _SHARED_INTENT_FILTER
        except Exception as e:
            logger.warning(f"[IDEFCRunner] IntentFilter 初始化失败: {e}")
            return None


def _ensure_embedding_prewarm_async() -> None:
    """后台预热 Embedding，避免首次 FC 会话同步等待模型加载。"""
    global _EMBEDDING_PREWARM_STARTED
    if _EMBEDDING_PREWARM_STARTED:
        return
    with _EMBEDDING_PREWARM_LOCK:
        if _EMBEDDING_PREWARM_STARTED:
            return
        _EMBEDDING_PREWARM_STARTED = True

    def _prewarm() -> None:
        try:
            from zulong.memory.embedding_manager import get_embedding_manager
            emb_mgr = get_embedding_manager()
            if getattr(emb_mgr, "_model", None) is None:
                logger.info("[IDEFCRunner] 后台预热 Embedding 模型...")
                emb_mgr.encode("预热")
                logger.info("[IDEFCRunner] Embedding 模型预热完成")
        except Exception as e:
            logger.warning(f"[IDEFCRunner] Embedding 模型预热失败: {e}")

    threading.Thread(
        target=_prewarm,
        name="zulong-embedding-prewarm",
        daemon=True,
    ).start()

# Web 监控事件广播（延迟导入避免循环依赖：ide_fc_runner ↔ ide_server）
# 注意：ide_server 在模块顶层导入 ide_fc_runner（用于 FC Runner 实例化），
# 若此处也在顶层导入 ide_server，将形成循环引用。因此 _broadcast_sync 内部
# 使用函数级延迟导入，仅在首次调用时解析 ide_server 模块。
def _broadcast_sync(event_type: str, payload: dict) -> None:
    """在同步上下文中安排广播（fire-and-forget），线程安全"""
    try:
        from zulong.ide.ide_server import broadcast_monitor_event, _main_event_loop
        loop = _main_event_loop if _main_event_loop is not None else asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(
                broadcast_monitor_event(event_type, payload), loop
            )
    except (RuntimeError, AttributeError, ImportError) as e:
        logger.debug(f"[IDEFCRunner] 广播失败(预期异常): {e}")
    except Exception as e:
        logger.warning(f"[IDEFCRunner] 广播失败(未预期异常): {type(e).__name__}: {e}")

logger = logging.getLogger(__name__)


def _safe_truncate(text: str, max_len: int = 200) -> str:
    """安全截断字符串，避免多字节字符截断导致乱码"""
    if len(text) <= max_len:
        return text
    # 尝试在字符边界截断
    try:
        truncated = text[:max_len]
        # 验证截断后是否为有效UTF-8
        truncated.encode('utf-8').decode('utf-8')
        return truncated + "..."
    except UnicodeDecodeError:
        # 回退到安全截断
        return text.encode('utf-8')[:max_len].decode('utf-8', errors='ignore') + "..."


_SENSITIVE_ARG_KEYS = frozenset({
    "api_key", "apikey", "authorization", "body", "content", "cookie", "data",
    "diff", "file_content", "key", "password", "replacement", "replacements",
    "secret", "text", "token",
})


def _safe_error_summary(text: Any, max_len: int = 240) -> str:
    """Return a compact log-safe summary without leaking common secrets."""
    import re as _re

    summary = " ".join(str(text or "").split())
    if not summary:
        return ""
    summary = _re.sub(
        r"(?i)(api[_-]?key|authorization|token|password|secret)\s*[:=]\s*['\"]?[^,'\"\s}]+",
        r"\1=<redacted>",
        summary,
    )
    return _safe_truncate(summary, max_len)


def _summarize_usage_for_log(usage: Any) -> Dict[str, Any]:
    if not usage:
        return {}
    if isinstance(usage, dict):
        raw = usage
    elif hasattr(usage, "model_dump"):
        try:
            raw = usage.model_dump()
        except Exception:
            raw = {}
    elif hasattr(usage, "to_dict"):
        try:
            raw = usage.to_dict()
        except Exception:
            raw = {}
    else:
        raw = {
            key: getattr(usage, key, None)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if getattr(usage, key, None) is not None
        }
    return {
        key: raw.get(key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if raw.get(key) is not None
    }


def _summarize_tool_call_for_log(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize a tool call for Phase 10 logs without recording full args."""
    function = tool_call.get("function", {}) or {}
    name = str(function.get("name") or "")
    raw_args = function.get("arguments", "")
    raw_args_text = raw_args if isinstance(raw_args, str) else _json.dumps(raw_args, ensure_ascii=False, default=str)
    summary: Dict[str, Any] = {
        "name": name,
        "arguments_length": len(raw_args_text or ""),
    }
    try:
        args = _json.loads(raw_args_text or "{}") if isinstance(raw_args_text, str) else raw_args_text
        if not isinstance(args, dict):
            summary["arguments_type"] = type(args).__name__
            return summary
    except Exception as exc:
        summary["arguments_parse_error"] = type(exc).__name__
        summary["arguments_preview"] = _safe_error_summary(raw_args_text, 120)
        return summary

    keys = sorted(str(key) for key in args.keys())
    summary["argument_keys"] = keys
    for path_key in ("path", "file_path", "target_path", "workspace_path", "cwd"):
        value = args.get(path_key)
        if value:
            summary[path_key] = _safe_truncate(str(value), 220)
    content_lengths: Dict[str, int] = {}
    redacted_keys: List[str] = []
    for key, value in args.items():
        key_text = str(key)
        if key_text.lower() in _SENSITIVE_ARG_KEYS:
            redacted_keys.append(key_text)
            content_lengths[key_text] = len(str(value or ""))
    if content_lengths:
        summary["redacted_argument_lengths"] = content_lengths
    if redacted_keys:
        summary["redacted_keys"] = sorted(redacted_keys)
    return summary


def _brief_for_feedback(text: str, max_len: int = 80) -> str:
    brief = " ".join(str(text or "").split())
    if len(brief) <= max_len:
        return brief
    return brief[: max_len - 3].rstrip() + "..."


def _plan_steps_for_feedback(
    user_text: str,
    tool_prediction: Optional[Dict[str, Any]],
    policy: str,
) -> List[str]:
    prediction = tool_prediction or {}
    context = prediction.get("context_bundle") or {}
    tools = (
        prediction.get("predicted_tools")
        or prediction.get("suggested_tools")
        or prediction.get("tool_bag")
        or []
    )
    tools = [str(t) for t in tools if t]
    steps: List[str] = []
    if context.get("needs_memory") or any("memory" in t for t in tools):
        steps.append("先取相关记忆和经验")
    if context.get("needs_project_context") or any(t in {"read_file", "search_code_symbols", "index_project", "analyze_module"} for t in tools):
        steps.append("读取或检索项目代码，确认真实实现")
    if policy in {"reuse", "inspect", "inspect_or_create", "continue"} or any(t.startswith("task_") for t in tools):
        steps.append("同步任务图状态")
    if any(t in {"exec_write_file", "ide_write_file", "replace_in_file", "exec_run_command", "execute_command"} for t in tools):
        steps.append("涉及写入或命令时先处理风险与审批")
        if tools:
            steps.append("按当前任务需要执行具体处理步骤")
    if not steps:
        steps.append("判断是否需要工具，不需要时直接组织回答")
    steps.append("结束前整理完成项、验证项和风险")
    return steps[:5]


class ThreadPoolManager:
    """线程池生命周期管理器（单例模式）"""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ide_fc_model"
        )
        self._futures: List[concurrent.futures.Future] = []
        self._shutdown = False

    @classmethod
    def get_instance(cls) -> "ThreadPoolManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    atexit.register(cls._instance.graceful_shutdown)
        return cls._instance

    def submit(self, fn, *args, **kwargs) -> concurrent.futures.Future:
        if self._shutdown:
            raise RuntimeError("ThreadPoolManager has been shutdown")
        future = self._executor.submit(fn, *args, **kwargs)
        self._futures.append(future)
        self._cleanup_completed_futures()
        return future

    def _cleanup_completed_futures(self):
        self._futures = [f for f in self._futures if not f.done()]

    def graceful_shutdown(self, timeout: float = 10.0) -> None:
        if self._shutdown:
            return

        logger.info("[ThreadPoolManager] 开始优雅关闭线程池...")
        self._shutdown = True

        try:
            self._executor.shutdown(wait=True, cancel_futures=False)
            logger.info("[ThreadPoolManager] 线程池已正常关闭")
        except Exception as e:
            logger.warning(f"[ThreadPoolManager] 等待关闭超时，强制终止: {e}")
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

        pending_count = sum(1 for f in self._futures if not f.done())
        if pending_count > 0:
            logger.warning(f"[ThreadPoolManager] 仍有 {pending_count} 个任务未完成")
        else:
            logger.info("[ThreadPoolManager] 所有任务已完成")

    def __del__(self):
        if not self._shutdown:
            self.graceful_shutdown()


@dataclass
class IDEFCResult:
    """FC 循环执行结果"""
    phase: str
    text_response: Optional[str] = None
    pending_call_ids: Optional[List[str]] = None
    reason: Optional[str] = None


@dataclass
class ExecutionEvent:
    """Internal execution event normalized before fan-out.

    It is not a third transport.  send_callback and broadcast_monitor_event are
    the two outward adapters that consume this single in-runner state source.
    """

    phase: str
    message: str
    turn: int = 0
    event_type: str = "TASK_PROGRESS"
    payload: Dict[str, Any] = field(default_factory=dict)
    interaction: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def callback_payload(self, max_turns: int) -> Dict[str, Any]:
        data = {
            "protocol_version": "2.0",
            "phase": self.phase,
            "message": self.message,
            "current_turn": self.turn,
            "max_turns": max_turns,
            "timestamp": self.created_at,
        }
        data.update(self.payload)
        if self.interaction:
            data["interaction"] = self.interaction
        return data


_FILLER_PATTERNS = [
    "我正在思考", "让我继续", "我来继续", "让我想想", "接下来我",
    "我正在处理", "正在分析", "正在执行", "稍等", "我需要",
    "但我需要", "不过我需要", "还需要进一步", "需要更多信息",
]

_PROGRESS_VERB_PATTERNS = (
    "现在", "正在", "接下来", "开始", "创建", "生成", "编写", "实现",
    "完成", "添加", "修改", "构建", "开发", "设计", "部署", "运行",
    "执行", "处理", "分析", "准备", "优化", "更新", "调整", "修复",
    "将", "会", "要", "需要", "首先", "然后", "最后", "接着",
)

# UncompletedGuard 未完成节点拦截阈值（从0.5降至0.3，覆盖更多未完成场景）
_UNCOMPLETED_THRESHOLD = 0.3

# 工具结果缓冲区最大条目数（防止无限增长）
_TOOL_RESULTS_BUFFER_MAX = 100

# Backfill JSON 密度阈值：response 中 JSON 特征字符占比超过此值则跳过
_JSON_DENSITY_THRESHOLD = 0.12


def _is_filler_content(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) < 6:
        return True
    if len(stripped) < 80:
        if stripped.rstrip().endswith(("？", "?")):
            return False
        if any(v in stripped for v in _PROGRESS_VERB_PATTERNS):
            return False
        return True
    pattern_count = sum(1 for p in _FILLER_PATTERNS if p in stripped)
    if len(stripped) > 300:
        return pattern_count >= 3
    return pattern_count >= 2


def _looks_like_audit_or_cleanup_report(text: str) -> bool:
    stripped = str(text or "").strip()
    if len(stripped) < 40:
        return False
    report_shape = ("汇报", "报告", "清单", "列表", "总览", "如下")
    cleanup_context = (
        "记忆",
        "节点",
        "挂起任务",
        "历史任务",
        "未完成任务",
        "经验",
        "清理",
        "删除",
        "候选",
        "blocked",
    )
    return (
        any(marker in stripped for marker in report_shape)
        and sum(1 for marker in cleanup_context if marker.lower() in stripped.lower()) >= 2
    )


def _looks_like_incomplete_result(text: str) -> bool:
    """Detect blockage/status prose that must not be backfilled as completed."""
    raw = str(text or "")
    if not raw:
        return False
    lowered = raw.lower()
    hard_system_markers = (
        "任务执行中断",
        "强制收敛",
        "无法正常回复",
        "系统当前出问题",
        "安全防护触发",
        "触发循环保护",
    )
    if any(marker.lower() in lowered for marker in hard_system_markers):
        return True
    if _looks_like_audit_or_cleanup_report(raw):
        return False
    hard_blockers = (
        "阻塞",
        "被阻塞",
        "无法运行测试",
        "无法验证",
        "无法完成",
        "不能完成",
        "审批拒绝",
        "审批超时",
        "用户未应用",
        "用户拒绝",
        "未应用写入",
        "未真实存在",
        "workspace_trust",
        "approval_blocked",
        "blocked",
        "timed out",
        "timeout",
        "interrupted",
    )
    if any(marker.lower() in lowered for marker in hard_blockers):
        return True

    success_markers = (
        "全部通过",
        "测试通过",
        "全部测试通过",
        "ran ",
        "\nok",
        " ok",
        "无失败",
        "0 failed",
        '"failed_count": 0',
        "'failed_count': 0",
    )
    has_success = any(marker.lower() in lowered for marker in success_markers)
    soft_failures = (
        "工具执行失败",
        "命令执行失败",
        "测试失败",
        "语法错误",
        "returncode=1",
        "returncode\": 1",
        "syntaxerror",
        "traceback",
        "failed (",
        "failed=",
        "error:",
        " errors=",
    )
    if not has_success and any(marker.lower() in lowered for marker in soft_failures):
        return True

    positive_markers = (
        "已创建",
        "已生成",
        "已写入",
        "已完成",
        "已实现",
        "创建成功",
        "写入成功",
        "生成成功",
    )
    if any(marker in raw for marker in positive_markers):
        return False
    markers = (
        "任务执行中断",
        "强制收敛",
        "未产出",
        "进行中但未产出",
        "尚未完成",
        "还未完成",
        "未完成节点",
        "未生成",
        "未创建",
        "没有产出",
        "无法完成",
        "不能完成",
        "触发循环保护",
        "系统当前出问题",
        "stalled",
    )
    return any(marker.lower() in lowered for marker in markers)


def _has_content_match(response: str, node_label: str) -> bool:
    import re as _re
    if not response or not node_label:
        return False
    if node_label in response:
        return True
    cjk_runs = _re.findall("[一-鿿]{2,}", node_label)
    # 短标签（CJK < 4 字符）的 bigram 太少，只接受精确子串匹配
    total_cjk = sum(len(r) for r in cjk_runs)
    if total_cjk < 4:
        return False
    matched = set()
    for run in cjk_runs:
        for i in range(len(run) - 1):
            bg = run[i:i + 2]
            if bg in response:
                matched.add(bg)
    # 要求匹配数量 ≥ max(3, 总 bigram 数的 40%)
    total_bigrams = max(1, sum(len(r) - 1 for r in cjk_runs))
    threshold = max(3, int(total_bigrams * 0.4))
    return len(matched) >= threshold


def _extract_node_content(response: str, node_label: str, max_len: int = 500) -> str:
    import re as _re
    if not response or not node_label:
        return response[:max_len] if response else ""
    idx = response.find(node_label)
    if idx >= 0:
        start = idx
        end = min(len(response), start + max_len)
        ns = response.find("\n\n", start + len(node_label))
        if 0 < ns - start <= max_len:
            end = ns
        return response[start:end].strip()
    for kw in _re.findall("[一-鿿]{2,}", node_label):
        idx = response.find(kw)
        if idx >= 0:
            return response[max(0, idx - 20):min(len(response), idx + max_len)].strip()
    return response[:max_len]


class _FunctionProxy:
    __slots__ = ("name", "arguments")
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _ToolCallProxy:
    __slots__ = ("id", "type", "function")
    def __init__(self, data: Dict):
        self.id = data["id"]
        self.type = data.get("type", "function")
        fd = data["function"]
        self.function = _FunctionProxy(fd["name"], fd["arguments"])


class IDEFCRunner(FCRunner):
    """IDE 模式 FC 循环运行器

    继承 FCRunner，复用通用安全网逻辑 (_detect_duplicate_tool_loop, _detect_progress_stall)。
    添加 IDE 特有：流式推送、工具分流、暂停/恢复、参数验证。
    per-runner 实例隔离注意力窗口/RuleGuardian/CircuitBreaker，
    避免并发 Session 状态冲突。
    """

    _CB_RESTRICTED_RECOVERY_CAPABILITIES = {
        "attention_switch",
        "note_anchor",
        "memory_persist",
        "tag_anchor",
    }
    _CB_RESTRICTED_EXCLUDED_CAPABILITIES = {
        "file_write",
        "verification",
    }

    def __init__(self, engine: "InferenceEngine", session: AgentSession,
                 tool_registry: IDEToolRegistry):
        super().__init__(engine)
        self.session = session
        self.tool_registry = tool_registry
        self.translator = IDEFormatTranslator()
        self._max_fc_turns = getattr(engine, "_max_fc_turns", 100)
        self._soft_limit = getattr(engine, "_soft_limit", 50)
        self._hard_limit = getattr(engine, "_hard_limit", 100)
        self._step_limits_enabled = bool(getattr(engine, "_step_limits_enabled", True))
        self._warning_interval = getattr(engine, "_warning_interval", 10)
        self._fc_loop_timeout = getattr(engine, "_fc_loop_timeout", 600)
        self._fc_request_interval = getattr(engine, "_fc_request_interval", 1.0)
        self._remote_tool_timeout = getattr(engine, "_remote_tool_timeout", 600)
        self._approval_timeout = getattr(engine, "_approval_timeout", 60)
        # P2: 进度报告 + 弹性预算
        self._progress_report_interval = getattr(engine, "_progress_report_interval", 5)
        self._auto_continue = getattr(engine, "_auto_continue", True)
        self._max_reports_before_force_stop = getattr(engine, "_max_reports_before_force_stop", 5)
        self._attn_window = None
        self._rule_guardian = None
        self._circuit_breaker = None
        self._drift_detector = None
        # DialogueAdapter 对话轮次记录
        self._dialogue_adapter = None
        self._current_round_id: Optional[str] = None
        self._current_session_id: Optional[str] = None
        self._model_executor = ThreadPoolManager.get_instance()
        # BFS 调度控制
        self._last_bfs_seeds_hash: str = ""
        self._last_bfs_turn: int = 0
        self._last_pressure_tier: str = "green"  # 压力分级跟踪（green/yellow/red）
        self._bfs_min_interval: int = 3  # 最小间隔轮次
        # WS层IDESession引用（由 ide_server._run_fc_loop 注入）
        self.ide_session = None
        # L1-B 细粒度分类器：只作为工具预判的辅助信号
        self._intent_filter = None
        self._execution_events: List[ExecutionEvent] = []
        self._interaction_seq = 0
        self._tool_interaction_pairs: Dict[str, str] = {}
        self._checklist: List[Dict[str, Any]] = []
        self._checklist_by_id: Dict[str, Dict[str, Any]] = {}
        self._current_visible_step_pair_id: str = ""
        self._init_intent_filter()

    def _reset_checklist(self) -> None:
        self._checklist = []
        self._checklist_by_id = {}
        self._current_visible_step_pair_id = ""

    def _notify_session_linked(self, task_graph_id: str) -> None:
        """通知监控客户端: 会话已关联到任务图谱，同时更新 InteractionStore。

        TSD §23.11: 会话窗口绑定根会话节点，根节点下挂载任务图谱。
        task_graph_id 持久化到 InteractionStore，确保页面刷新后可恢复。
        """
        try:
            ide_sess = self.ide_session
            if ide_sess:
                ide_sess.task_graph_id = task_graph_id
            from zulong.ide.ide_server import _broadcast_sync
            _broadcast_sync("IDE_SESSION_LINKED", {
                "session_id": self.session.session_id,
                "conversation_id": getattr(ide_sess, "conversation_id", None) if ide_sess else None,
                "turn_id": getattr(ide_sess, "web_turn_id", None) if ide_sess else None,
                "workspace_path": getattr(ide_sess, "cwd", None) if ide_sess else None,
                "task_graph_id": task_graph_id,
                "project_id": ide_sess.project_id if ide_sess else None,
            })
            # TSD §23.11.6: 将 task_graph_id 写入 InteractionStore
            # 确保 conversation 表中持久化 task_graph_id，供页面刷新后恢复
            if ide_sess:
                _conv_id = getattr(ide_sess, "conversation_id", None)
                if _conv_id and task_graph_id:
                    try:
                        from zulong.launcher.interaction_store import get_interaction_store
                        _store = get_interaction_store()
                        _store.upsert_conversation(
                            _conv_id,
                            task_graph_id=task_graph_id,
                            workspace_path=getattr(ide_sess, "cwd", None),
                            project_id=getattr(ide_sess, "project_id", None),
                        )
                        logger.debug(
                            f"[IDEFCRunner] InteractionStore 已更新: "
                            f"conv={_conv_id}, graph={task_graph_id}"
                        )
                    except Exception as _ie:
                        logger.debug(
                            f"[IDEFCRunner] InteractionStore 更新跳过: {_ie}"
                        )
        except Exception as e:
            logger.debug(f"[IDEFCRunner] _notify_session_linked 跳过: {e}")

    def run_or_resume(self, new_messages: Optional[List[Dict]] = None,
                      tool_results: Optional[List[Dict]] = None) -> IDEFCResult:
        state = self.session.fc_state
        if tool_results and state and state.phase == "waiting_remote":
            logger.info(f"[IDEFCRunner] 恢复 FC, turn={state.fc_turn}, results={len(tool_results)}")
            self._restore_runner_state()
            self._inject_tool_results(state, tool_results)
            self._maybe_run_bfs(state.fc_turn, "tool_complete")
            state.phase = "running"
        elif new_messages:
            logger.info(f"[IDEFCRunner] 新 FC, messages={len(new_messages)}")
            state = self._init_state(new_messages)
            self.session.fc_state = state
        else:
            return IDEFCResult(phase="done", text_response="")
        return self._run_loop(state)

    async def run_loop_async(
        self,
        messages: List[Dict],
        send_callback,
        tool_result_queue: "asyncio.Queue",
        cancel_event: "asyncio.Event",
    ) -> IDEFCResult:
        """WebSocket 模式异步 FC 循环

        与 run_or_resume + _run_loop 的同步 HTTP 模式不同：
        - 远程工具不暂停返回，而是通过 send_callback 推送 tool_request 后
          等待 tool_result_queue 中的结果，自动注入后继续循环
        - 模型调用和内部工具执行在线程池中运行，避免阻塞事件循环
        - 通过 cancel_event 支持随时取消

        Args:
            messages: 初始消息列表 (system + user)
            send_callback: async callable(msg_type: str, payload: dict) 推送消息到插件
            tool_result_queue: asyncio.Queue 接收插件工具执行结果
            cancel_event: asyncio.Event 取消信号
        """
        # 保存 send_callback 供进度推送使用
        self._ide_send_callback = send_callback
        # 初始化状态（同步，在线程池中运行）
        loop = asyncio.get_event_loop()
        state = await loop.run_in_executor(None, self._init_state, messages)
        self.session.fc_state = state
        self._reset_checklist()
        from zulong.ide.ide_server import broadcast_monitor_event

        # 设置 FC 循环运行状态为 True（禁止节点审查提交）
        try:
            from zulong.core.state_manager import state_manager
            state_manager.set_fc_loop_running(True)
        except Exception:
            pass

        _task_brief = _brief_for_feedback(state.user_input_text or "")
        _policy = getattr(state, "task_graph_policy", "none")
        _tool_pred = getattr(state, "tool_prediction", None)
        _plan_steps = _plan_steps_for_feedback(state.user_input_text or "", _tool_pred, _policy)
        _tool_budget = getattr(state, "tool_call_budget", None)

        await self._emit_execution_event(
            send_callback,
            "started",
            f"任务已进入后台处理队列: {_task_brief[:50]}",
            event_type="FC_START",
            payload={
                "max_turns": self._max_fc_turns,
                "task_graph_policy": _policy,
                "tool_call_budget": _tool_budget,
                "user_input": (state.user_input_text or "")[:500],
                "next_step": "等待 L2 输出可见步骤说明。",
                "interaction": {
                    "kind": "progress",
                    "status": "running",
                    "title": "任务已接收",
                    "detail": "系统已收到任务，等待 L2 生成下一步说明。",
                    "source_channel": "system_status",
                    "channel": CHANNEL_STATUS,
                    "ux_visibility": UX_HIDDEN,
                    "next_step": "等待 L2 输出可见步骤说明。",
                },
            },
            send_status=False,
        )

        # ── TSD v2.7: 发射 task:plan 和 tool:prediction 消息 ──
        try:
            if _tool_pred:
                _pred_ctx = _tool_pred.get("context_bundle", {}) or {}
                _pred_tools = (
                    _tool_pred.get("predicted_tools")
                    or _tool_pred.get("suggested_tools")
                    or []
                )
                await send_callback(MessageType.TOOL_PREDICTION, {
                    "prediction": _tool_pred,
                    "predicted_tools": _pred_tools,
                    "suggested_tools": _pred_tools,
                    "tool_bag": _tool_pred.get("tool_bag", []),
                    "confidence": _tool_pred.get("confidence", 0),
                    "reason": _tool_pred.get("reason", ""),
                    "source": _pred_ctx.get("tool_prediction_source"),
                    "embedding_top_tools": _pred_ctx.get("embedding_top_tools", []),
                    "timestamp": time.time(),
                })
            await send_callback(MessageType.TASK_PLAN, {
                "task": (state.user_input_text or "")[:500],
                "task_graph_policy": getattr(state, "task_graph_policy", "none"),
                "max_turns": self._max_fc_turns,
                "tool_prediction": _tool_pred,
                "tool_call_budget": _tool_budget,
                "interaction": {
                    "kind": "progress",
                    "status": "running",
                    "title": "任务元数据已准备",
                    "detail": "系统状态已更新，用户可见步骤等待 L2 生成。",
                    "source_channel": "system_status",
                    "channel": CHANNEL_STATUS,
                    "ux_visibility": UX_HIDDEN,
                    "next_step": "等待 L2 输出可见步骤说明。",
                },
                "timestamp": time.time(),
            })
        except Exception as _e:
            logger.debug(f"[IDEFCRunner] 发射 task:plan/tool:prediction 失败: {_e}")

        while True:
            # 检查取消
            if cancel_event.is_set():
                return await loop.run_in_executor(
                    None, self._finalize, state, "cancelled")

            # 检查轮次限制
            tr = self._check(state)
            if tr:
                return await loop.run_in_executor(
                    None, self._finalize, state, tr)

            try:
                # 每轮推送含图谱进度的统一状态
                _progress_snapshot = self._get_progress_snapshot()
                await self._emit_execution_event(
                    send_callback,
                    "calling_model",
                    f"正在调用模型进行推理... (Turn {state.fc_turn}/{self._max_fc_turns})",
                    turn=state.fc_turn,
                    event_type="CALLING_MODEL",
                    payload={
                        "progress": _progress_snapshot,
                        "model": getattr(state, "vllm_model_id", ""),
                    },
                    send_progress=False,
                    send_status=True,
                )
                
                # 🔥 修复：持续发送reasoning心跳，保持"思考中"状态稳定
                reasoning_msg = f"正在调用模型进行推理... (Turn {state.fc_turn}/{self._max_fc_turns})"
                await send_callback("display_reasoning", {
                    "reasoning": reasoning_msg
                })
                self._emit_execution_event_sync(
                    "waiting_model",
                    reasoning_msg,
                    turn=state.fc_turn,
                    event_type="TASK_HEARTBEAT",
                    payload={"reasoning": reasoning_msg},
                )

                # FC 请求间隔
                if state.fc_turn > 1 and self._fc_request_interval > 0:
                    await asyncio.sleep(self._fc_request_interval)

                # 在线程池中调用模型（可取消：每 2 秒检查 cancel_event）
                model_future = loop.run_in_executor(
                    None, self._call_model, state)
                tc_data, resp_content = None, None
                while True:
                    if cancel_event.is_set():
                        model_future.cancel()
                        return await loop.run_in_executor(
                            None, self._finalize, state, "cancelled")
                    try:
                        tc_data, resp_content = await asyncio.wait_for(
                            asyncio.shield(model_future), timeout=2.0)
                        break
                    except asyncio.TimeoutError:
                        # 模型调用等待中，推送心跳进度防止前端超时
                        await self._emit_execution_event(
                            send_callback,
                            "waiting_model",
                            f"模型推理中... (Turn {state.fc_turn}/{self._max_fc_turns})",
                            turn=state.fc_turn,
                            event_type="TASK_HEARTBEAT",
                            payload={"progress": self._get_progress_snapshot()},
                            send_progress=False,
                            send_status=True,
                            monitor=True,
                        )
                        try:
                            await send_callback("display_reasoning", {
                                "reasoning": f"模型推理中... (Turn {state.fc_turn}/{self._max_fc_turns})"
                            })
                        except Exception:
                            pass
                        continue  # 继续等待，下次循环检查 cancel_event

                if tc_data is None and resp_content is None:
                    if state.api_timeout_count >= 2:
                        return await loop.run_in_executor(
                            None, self._finalize, state, "api_error")
                    await asyncio.sleep(2)  # API 错误后短暂退避
                    continue

                state.loop_error_count = 0

                # 刷新 Gatekeeper 空闲计时器，防止模型调用间隔被误判空闲
                try:
                    from zulong.l1b.scheduler_gatekeeper import gatekeeper
                    if gatekeeper:
                        gatekeeper.touch_idle_timer()
                except Exception:
                    pass

                if tc_data:
                    tc_data = self._apply_tool_call_budget_ide(
                        state,
                        tc_data,
                        send_callback=send_callback,
                    )
                    if not tc_data:
                        self._log_fc_decision_path(
                            state,
                            path="tool_budget_exhausted",
                            tool_calls=[],
                            response_content=resp_content or "",
                            root_cause="context_pressure",
                            tool_budget=state.tool_call_budget,
                            tool_calls_used=state.tool_calls_used,
                        )
                        await self._emit_execution_event(
                            send_callback,
                            "blocked",
                            "已达到本轮工具上限，转为整理已有结果。",
                            turn=state.fc_turn,
                            event_type="TOOL_BUDGET_LIMIT",
                            payload=self._tool_budget_event_payload(state),
                            send_progress=True,
                            send_status=True,
                        )
                        continue
                    tool_names = [tc["function"]["name"] for tc in tc_data]
                    self._log_fc_decision_path(
                        state,
                        path="tool_calls_dispatch",
                        tool_calls=tc_data,
                        response_content=resp_content or "",
                        tool_names=tool_names,
                        tool_call_count=len(tc_data),
                    )
                    await self._emit_execution_event(
                        send_callback,
                        "executing",
                        f"执行工具: {', '.join(tool_names)}",
                        turn=state.fc_turn,
                        event_type="TOOL_CALL",
                        payload={
                            "tools": tool_names,
                            "count": len(tc_data),
                            "interaction": {
                                "kind": "progress",
                                "status": "running",
                                "title": "工具调度已准备",
                                "detail": "工具调度已准备，等待模型步骤说明完成后执行。",
                                "source_channel": "system_status",
                                "channel": CHANNEL_STATUS,
                                "ux_visibility": UX_HIDDEN,
                            },
                        },
                        send_progress=False,
                        send_status=False,
                        monitor=False,
                    )
                    
                    # CB 模式工具调用计数：防止死循环
                    if state.cb_force_no_tools:
                        state.cb_tool_streak += 1
                    else:
                        state.cb_tool_streak = 0
                    # 有工具调用 → 分流处理
                    should_continue = await self._exec_tools_async(
                        state, tc_data, resp_content or "",
                        send_callback, tool_result_queue, cancel_event, loop)
                    self._update_cb_recovery_progress(
                        state,
                        tool_names,
                        state.tool_definitions,
                    )
                    if should_continue == "cancelled":
                        return await loop.run_in_executor(
                            None, self._finalize, state, "cancelled")
                    await self._emit_execution_event(
                        send_callback,
                        "turn_complete",
                        "本轮工具执行完成，继续判断下一步。",
                        turn=state.fc_turn,
                        event_type="TURN_COMPLETE",
                        payload={"has_tool_calls": True, "tool_names": tool_names},
                        send_progress=False,
                    )
                    # 继续循环
                    continue

                # 纯文本回复 → 先评估，再决定是否推送给前端
                # 避免安全网返回 "continue" 时重复显示 filler 内容
                final_text = state.last_response_content or resp_content
                self._log_fc_decision_path(
                    state,
                    path="no_tool_call_enter_eval",
                    tool_calls=[],
                    response_content=resp_content or "",
                    response_length=len(resp_content or ""),
                    cb_force=state.cb_force_no_tools,
                    null_response_count=state.null_response_count,
                )
                verdict = await loop.run_in_executor(
                    None, self._eval_response, state, resp_content or "")
                self._log_fc_decision_path(
                    state,
                    path=f"no_tool_call_verdict_{verdict}",
                    tool_calls=[],
                    response_content=resp_content or "",
                    verdict=verdict,
                )

                # 🔥 修复：流式推理已在 _call_model 中实时推送，此处不再重复推送
                # 只发送完成标记即可
                if verdict == "done":
                    # 发送包含task_status的完成标记（无text字段，避免重复）
                    complete_msg = {
                        "text": "", 
                        "turn": state.fc_turn,
                        "streaming": False,
                        "complete": True,
                        "task_status": "completed"
                    }
                    logger.info(f"[IDEFCRunner] 发送display_text完成标记（流式已完成，仅发送完成信号）")
                    await send_callback("display_text", complete_msg)
                    await self._emit_execution_event(
                        send_callback,
                        "text_streamed",
                        "模型文本回复已流式输出完成。",
                        turn=state.fc_turn,
                        event_type="DISPLAY_TEXT",
                        payload={"complete": True, "text_length": len(final_text or "")},
                        send_progress=False,
                        monitor=False,
                    )
                    
                elif verdict == "continue":
                    # continue时不推送，避免中间状态重复显示
                    logger.info(f"[FC] Turn {state.fc_turn} verdict=continue, skip display_text push (len={len(final_text or '')})")
                    # 保存内容供下一轮使用
                    if final_text:
                        state.last_response_content = final_text
                
                if verdict == "done" and final_text:
                    await self._emit_execution_event(
                        send_callback,
                        "model_response",
                        "模型已生成最终文本回复。",
                        turn=state.fc_turn,
                        event_type="MODEL_RESPONSE",
                        payload={
                            "text": (final_text or "")[:5000],
                            "text_preview": (final_text or "")[:200],
                            "text_length": len(final_text or ""),
                        },
                        send_progress=False,
                        monitor=True,
                    )
                    await self._emit_execution_event(
                        send_callback,
                        "display_text",
                        "向 Web 端推送最终文本。",
                        turn=state.fc_turn,
                        event_type="DISPLAY_TEXT",
                        payload={"text": (final_text or "")[:10000]},
                        send_progress=False,
                        monitor=True,
                    )

                if verdict == "done":
                    state.phase = "done"
                    _mem_changes = self._get_memory_changes_snapshot()
                    _summary_payload = self._build_task_summary_payload(
                        state,
                        reason="done",
                        final_text=final_text or "",
                        memory_changes=_mem_changes,
                    )
                    _summary_status = str(_summary_payload.get("status") or "succeeded")

                    await self._emit_execution_event(
                        send_callback,
                        "completed",
                        f"任务完成 (共{state.fc_turn}轮推理)",
                        turn=state.fc_turn,
                        event_type="TASK_COMPLETE",
                        payload={
                            "summary": _summary_payload,
                            **_summary_payload,
                            "interaction": {
                                "kind": "summary",
                                "status": _summary_status,
                                "title": "任务完成",
                                "detail": f"本轮共进行了 {state.fc_turn} 轮推理。",
                                "progress": 100,
                                "completed_items": _summary_payload["completed_items"],
                                "verified_items": _summary_payload["verified_items"],
                                "pending_items": _summary_payload["pending_items"],
                                "risks_summary": _summary_payload["risks_summary"],
                                "memory_changes": _mem_changes,
                                "next_step": _summary_payload["next_step"],
                                "progress_items": [
                                    self._progress_item(item, "completed", source="summary")
                                    for item in _summary_payload["completed_items"]
                                ] + [
                                    self._progress_item(item, "pending", source="summary")
                                    for item in _summary_payload["pending_items"]
                                ],
                            },
                        },
                    )
                    
                    # ── TSD v2.7: 发射 task:summary 和 graph:memory:diff ──
                    try:
                        await send_callback(MessageType.TASK_SUMMARY, {
                            **_summary_payload,
                            "interaction": {
                                "kind": "summary",
                                "status": _summary_status,
                                "title": "任务完成",
                                "detail": f"本轮共进行了 {state.fc_turn} 轮推理。",
                                "progress": 100,
                                "completed_items": _summary_payload["completed_items"],
                                "verified_items": _summary_payload["verified_items"],
                                "pending_items": _summary_payload["pending_items"],
                                "risks_summary": _summary_payload["risks_summary"],
                                "memory_changes": _mem_changes,
                                "next_step": _summary_payload["next_step"],
                                "progress_items": [
                                    self._progress_item(item, "completed", source="summary")
                                    for item in _summary_payload["completed_items"]
                                ] + [
                                    self._progress_item(item, "pending", source="summary")
                                    for item in _summary_payload["pending_items"]
                                ],
                            },
                            "timestamp": time.time(),
                        })
                        _graph_diff_payload = mark_hidden_payload({
                            "memory_changes": _mem_changes,
                            "timestamp": time.time(),
                        })
                        await send_callback(MessageType.GRAPH_MEMORY_DIFF, _graph_diff_payload)
                    except Exception as _e:
                        logger.debug(f"[IDEFCRunner] 发射 task:summary/graph:memory:diff 失败: {_e}")
                    
                    # 清除 FC 循环运行状态（允许节点审查提交）
                    try:
                        from zulong.core.state_manager import state_manager
                        state_manager.set_fc_loop_running(False)
                    except Exception:
                        pass
                    
                    # 立即通知前端任务完成（在后处理之前，防止 WS 断开导致丢失）
                    _done_text = state.last_response_content or ""
                    
                    # 🎯 优化3: 完成消息也使用清洗后的文本
                    if _done_text:
                        from zulong.utils.text_cleaner import clean_text_for_tts
                        _done_text = clean_text_for_tts(_done_text)
                    
                    # 🔥 修复：不在FC循环内发送task_complete，由ide_server统一发送
                    # 避免重复发送导致前端状态混乱
                    # await send_callback("task_complete", {"result": _done_text})
                    
                    await self._emit_execution_event(
                        send_callback,
                        "completed",
                        "FC 循环进入完成收束，等待统一质量门写回。",
                        turn=state.fc_turn,
                        event_type="FC_DONE",
                        payload={
                            "total_turns": state.fc_turn,
                            "reason": "done",
                            "summary": {
                                "completed": ["FC 循环已收束"],
                                "verified": [],
                                "remaining": [],
                                "risk": "最终完成状态以统一完成质量门为准。",
                                "next_step": "等待完成质量门写回或转为 partial/blocked。",
                            },
                            "interaction": {
                                "kind": "summary",
                                "status": "running",
                                "title": "质量门收束中",
                                "detail": "祖龙已生成候选结果，正在交由统一完成质量门确认。",
                                "progress": 95,
                                "progress_items": [
                                    self._progress_item("候选结果已生成，等待质量门确认", "running", source="summary")
                                ],
                            },
                        },
                        send_progress=False,
                        monitor=True,
                    )
                    # 🔥 修复：提取 submit_final_answer 内容，确保后处理和 ide_server 能读到完整答案
                    _final_answer = self._extract_final_answer(state)
                    if _final_answer:
                        state.final_answer = _final_answer
                        logger.info(
                            f"[IDEFCRunner] 提取 submit_final_answer: "
                            f"len={len(_final_answer)}")
                    else:
                        state.final_answer = None

                    # 仅记录 FC 统计。最终答案写入/归档必须由统一 _finalize()
                    # 在完成质量门通过后执行，避免 submit_final_answer 绕过证据门。
                    try:
                        from zulong.tools.task_tools import get_active_task_graph
                        _tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
                        if _tg and hasattr(_tg, "metadata"):
                            _tg.metadata["total_turns"] = state.fc_turn
                            _tg.metadata["duration"] = time.time() - getattr(_tg, "created_at", time.time())
                    except Exception:
                        pass

                    # 所有最终写回/归档统一走 _finalize，避免局部 done 路径绕过质量门。
                    return await loop.run_in_executor(
                        None, self._finalize, state, "done")
                elif verdict == "cb_force":
                    state.cb_force_no_tools = True
                # "continue" → 继续循环

            except asyncio.CancelledError:
                # 清除 FC 循环运行状态
                try:
                    from zulong.core.state_manager import state_manager
                    state_manager.set_fc_loop_running(False)
                except Exception:
                    pass
                return await loop.run_in_executor(
                    None, self._finalize, state, "cancelled")
            except Exception as loop_err:
                logger.error(
                    f"[IDEFCRunner] async 循环异常 turn={state.fc_turn}: "
                    f"{loop_err}", exc_info=True)
                state.loop_error_count += 1
                if state.loop_error_count >= 3:
                    # 清除 FC 循环运行状态
                    try:
                        from zulong.core.state_manager import state_manager
                        state_manager.set_fc_loop_running(False)
                    except Exception:
                        pass
                    return await loop.run_in_executor(
                        None, self._finalize, state, "loop_error")
                continue

    async def _exec_tools_async(
        self,
        state: IDEFCState,
        tool_calls_data: List[Dict],
        response_content: str,
        send_callback,
        tool_result_queue: "asyncio.Queue",
        cancel_event: "asyncio.Event",
        loop,
    ) -> Optional[str]:
        """异步版工具执行 + 分流

        内部工具在线程池中执行；远程工具通过 WebSocket 推送后等待结果。
        返回 None 表示正常继续循环，"cancelled" 表示被取消。
        """
        from zulong.ide.ide_server import broadcast_monitor_event
        fc = state.fc_turn
        msgs = state.messages

        # ── TSD v2.9.3: L2 先说再做 ──
        model_step_note = " ".join(str(response_content or "").split())
        announce_calls, real_calls = self._split_announce_step_calls(tool_calls_data)
        turn_pair_id = f"fc:{self.session.session_id}:{fc}:step"
        has_real_tools = bool(real_calls)

        # Mark announce_step as background in tool registry classification
        if announce_calls:
            for ac in announce_calls:
                self._tool_interaction_pairs[ac.get("id", "")] = ac.get("id", "")
            self._append_announce_step_messages(state, announce_calls, response_content, fc, None)

        # Step-note enforcement for real tools
        if has_real_tools:
            have_content_step = bool(model_step_note)
            announce_step = self._first_announce_step(announce_calls)
            have_announce_step = bool(announce_step.get("message"))

            step_retry_key = f"__step_announce_retry_{fc}"
            retry_count = getattr(state, "step_announce_retry_count", 0)

            if have_content_step or have_announce_step:
                # Extract step message: prefer assistant.content, fallback to announce_step
                step_message = model_step_note if have_content_step else announce_step.get("message", "")
                expected = announce_step.get("expected_actions") or []
                await self._emit_model_step_progress(
                    send_callback,
                    state=state,
                    message=step_message,
                    pair_id=turn_pair_id,
                    expected_actions=expected,
                    real_calls=real_calls,
                    source="assistant.content" if have_content_step else "announce_step",
                )
                # Reset retry counter on success
                state.step_announce_retry_count = 0
            else:
                # No step note from L2 → block and request re-generation
                if retry_count < 1:
                    state.step_announce_retry_count = retry_count + 1
                    control_msg = internal_control_message(
                        "请先用一句普通可见中文说明本步将做什么，再重新发起工具调用；不要输出推理过程。"
                        "如果当前模型/Provider 不保留 tool_calls 同轮 assistant.content，请先调用 announce_step(message=...) 再调用实际工具。"
                    )
                    msgs.append(control_msg)
                    if self._attn_window:
                        self._attn_window.register_message(control_msg, turn=fc)
                    logger.warning(
                        "[IDEFCRunner] L2 缺少步骤说明 (retry=%d)，注入 internal_control 要求补说明",
                        retry_count,
                    )
                    return None  # Skip this turn, let next FC iteration re-call model
                else:
                    # Already retried: keep the real tools blocked instead of
                    # creating a system-authored task card or executing silently.
                    logger.warning(
                        "[IDEFCRunner] L2 步骤说明重试后仍缺失，拦截真实工具调用并转为收敛"
                    )
                    control_msg = internal_control_message(
                        "仍未收到可见步骤说明，本轮真实工具调用已被拦截。"
                        "请不要继续调用工具，先向用户说明当前受阻原因；如需继续执行，下一轮必须先输出普通可见步骤说明或调用 announce_step(message=...)。"
                    )
                    msgs.append(control_msg)
                    state.cb_force_no_tools = True
                    if self._attn_window:
                        self._attn_window.register_message(control_msg, turn=fc)
                    return None

        # Use only real calls (non-announce_step) for execution
        tool_calls_data = real_calls
        if not tool_calls_data:
            return None

        internal, remote = [], []
        for td in tool_calls_data:
            cat = self.tool_registry.classify(td["function"]["name"])
            (remote if cat == "remote" else internal).append(td)

        grp = self._attn_window.new_tool_group() if self._attn_window else None

        # ── 内部工具（线程池执行） ──
        if internal:
            a_msg = {
                "role": "assistant", "content": response_content or "",
                "tool_calls": internal}
            msgs.append(a_msg)
            if self._attn_window:
                self._attn_window.register_message(a_msg, turn=fc, group_id=grp)

            for td in internal:
                if cancel_event.is_set():
                    return await loop.run_in_executor(None, self._finalize, state, "cancelled")
                tn = td["function"]["name"]
                friendly_tn = self._friendly_tool_name(tn)
                friendly_action = self._friendly_action_summary(
                    tn,
                    self._safe_parse_tool_arguments(td["function"].get("arguments", "{}")),
                )
                call_id = td.get("id") or self._next_interaction_id("internal_tool")
                event_pair_id = self._current_visible_step_pair_id or turn_pair_id or call_id
                self._tool_interaction_pairs[call_id] = event_pair_id
                await self._emit_execution_event(
                    send_callback,
                    "tool_requested",
                    f"正在{friendly_tn}",
                    turn=fc,
                    event_type="TOOL_CALL",
                    payload={
                        "tool_name": tn,
                        "action_summary": friendly_action,
                        "tool_scope": "internal",
                        "call_id": call_id,
                        "pair_id": event_pair_id,
                        "interaction": {
                            "pair_id": event_pair_id,
                            "kind": "action",
                            "status": "running",
                            "title": f"正在{friendly_action}",
                            "detail": "祖龙正在处理这一步，完成后会把结果接回当前任务。",
                            "tool_name": tn,
                        },
                    },
                    send_progress=False,
                    send_status=True,
                )
                await loop.run_in_executor(
                    None, self._exec_internal, state, td, fc, grp)
                await self._emit_execution_event(
                    send_callback,
                    "tool_finished",
                    f"{friendly_tn}已完成",
                    turn=fc,
                    event_type="IDE_TOOL_EXEC",
                    payload={
                        "tool_name": tn,
                        "action_summary": friendly_action,
                        "tool_scope": "internal",
                        "call_id": call_id,
                        "pair_id": event_pair_id,
                        "interaction": {
                            "pair_id": event_pair_id,
                            "kind": "observation",
                            "status": "succeeded",
                            "title": f"{friendly_action}已完成",
                            "detail": "结果已写回本轮上下文，祖龙会据此继续推进。",
                            "tool_name": tn,
                        },
                    },
                    send_progress=False,
                    send_status=True,
                )

            # CircuitBreaker 评估
            try:
                if self._circuit_breaker:
                    _aw_ratio = (
                        getattr(
                            self._attn_window,
                            "trigger_context_pressure_ratio",
                            getattr(self._attn_window, "context_pressure_ratio", self._attn_window.usage_ratio),
                        )
                        if self._attn_window else -1.0
                    )
                    cb_s, cb_r = self._circuit_breaker.evaluate(fc, msgs, attn_usage_ratio=_aw_ratio)
                    if cb_s == CircuitBreakerState.RED:
                        logger.warning(f"[IDEFCRunner][CB] RED: {cb_r}")
                        state.cb_force_no_tools = True
                        state.cb_trigger_reason = cb_r  # 保存CB原因供_finalize使用
                        cm = self._build_cb_red_control_message(state, cb_r)
                        msgs.append(cm)
                        if self._attn_window:
                            self._attn_window.register_message(cm, turn=fc)
                            try:
                                self._attn_window.on_navigate_attention(direction="single_chain")
                            except Exception:
                                pass
                        remote = []  # 取消远程工具
                    elif cb_s == CircuitBreakerState.YELLOW:
                        logger.warning(f"[IDEFCRunner][CB] YELLOW: {cb_r}")
                        # 如果是 task_add_node 模式重复，检查计划深度
                        cb_msg = f"[Circuit Breaker 警告] {cb_r}\n请尽快总结当前信息并回复用户。"
                        if "task_add_node" in cb_r:
                            try:
                                from zulong.tools.task_tools import get_active_task_graph as _gtg
                                _t = _gtg()
                                if _t:
                                    max_depth = max((_t.get_node_depth(n.id) for n in _t._nodes.values()), default=0)
                                    if max_depth <= 1:
                                        cb_msg = (
                                            "[结构校验] 当前任务计划过浅：所有节点都在第1层。\n"
                                            "【强制要求】请立即为每个阶段添加 2-3 个具体子步骤节点，使用 "
                                            "task_add_node(parent_id='阶段节点ID', label='子步骤名')。\n"
                                            "示例：task_add_node(parent_id='o1', label='分析项目README文档')\n"
                                            "完成子步骤添加后再开始执行。"
                                        )
                            except Exception:
                                pass
                        ch = internal_control_message(cb_msg)
                        msgs.append(ch)
                        if self._attn_window:
                            self._attn_window.register_message(ch, turn=fc)
            except Exception as cb_err:
                logger.warning(
                    f"[IDEFCRunner] CircuitBreaker evaluate 异常: {cb_err}")

            # 上下文压力感知（在 CB 评估之后）
            self._apply_pressure_guidance(state, fc)

            # ── TSD v2.7: 发射 graph:memory:diff（内部工具执行后）──
            await self._emit_memory_diff_async(send_callback, state.fc_turn)

            # ── TSD v2.7: 发射 attention:update ──
            if self._attn_window:
                try:
                    ratio = float(
                        getattr(
                            self._attn_window,
                            "trigger_context_pressure_ratio",
                            getattr(
                                self._attn_window,
                                "context_pressure_ratio",
                                getattr(self._attn_window, "usage_ratio", 0.0),
                            ),
                        )
                        or 0.0
                    )
                    yellow_ratio = 0.90
                    red_ratio = 1.0
                    try:
                        attn_cfg = getattr(self._attn_window, "_llm_config", None)
                        if attn_cfg:
                            yellow_ratio = float(getattr(attn_cfg, "pressure_threshold_medium", yellow_ratio))
                            red_ratio = float(getattr(attn_cfg, "pressure_threshold_high", red_ratio))
                        elif self._circuit_breaker:
                            cb_cfg = getattr(self._circuit_breaker, "_config", {}) or {}
                            yellow_ratio = float(cb_cfg.get("context_yellow_ratio", yellow_ratio))
                            red_ratio = float(cb_cfg.get("context_red_ratio", red_ratio))
                    except Exception:
                        pass
                    if ratio > red_ratio:
                        pressure_tier = "red"
                    elif ratio > yellow_ratio:
                        pressure_tier = "yellow"
                    else:
                        pressure_tier = "green"
                    pressure_view = build_threshold_pressure_view(
                        ratio,
                        yellow_ratio,
                        red_ratio,
                        tier=pressure_tier,
                    ).to_dict()
                    _attn_state = {
                        "mode": self._attn_window.mode.value if self._attn_window.mode else "global",
                        "turn": fc,
                        "focus_node_id": self._attn_window._current_node_id,
                        "budget_usage": round(pressure_view.get("context_pressure_percent", ratio * 100), 1),
                        "context_pressure": round(pressure_view.get("context_pressure_ratio", ratio), 3),
                        "threshold_budget_pressure": round(pressure_view.get("context_pressure_ratio", ratio), 3),
                        "threshold_budget_percent": pressure_view.get("context_pressure_percent"),
                        "active_threshold_ratio": pressure_view.get("active_threshold_ratio"),
                        "budget_reference": pressure_view.get("budget_reference"),
                        "raw_context_pressure": round(ratio, 3),
                        "trigger_context_pressure": round(ratio, 3),
                        "trigger_budget_usage": round(pressure_view.get("context_pressure_percent", ratio * 100), 1),
                        "visible_context_pressure": round(
                            float(getattr(self._attn_window, "active_context_pressure_ratio", ratio) or ratio),
                            3,
                        ),
                        "backing_context_pressure": round(
                            float(getattr(self._attn_window, "context_pressure_ratio", ratio) or ratio),
                            3,
                        ),
                        "pressure_tier": pressure_tier,
                        "pressure_threshold_medium": yellow_ratio,
                        "pressure_threshold_high": red_ratio,
                    }
                    await send_callback(MessageType.ATTENTION_UPDATE, _attn_state)
                except Exception as _e:
                    logger.debug(f"[IDEFCRunner] 发射 attention:update 失败: {_e}")

        # ── 远程工具（WebSocket 推送 + 等待） ──
        if remote:
            valid_remote, rejected = self._validate_and_clean_remote_calls(
                remote)
            if rejected:
                self._log_fc_decision_path(
                    state,
                    path="invalid_tool_args_remote_rejected",
                    tool_calls=remote,
                    root_cause="invalid_tool_args",
                    rejected_count=len(rejected),
                    rejected_tools=[r[0]["function"]["name"] for r in rejected],
                )
            all_calls = valid_remote + [r[0] for r in rejected]
            ra = {
                "role": "assistant",
                "content": "" if internal else (response_content or ""),
                "tool_calls": all_calls,
            }
            msgs.append(ra)
            if self._attn_window:
                self._attn_window.register_message(
                    ra, turn=fc, group_id=grp)

            # 注入被拒绝调用的错误结果
            for rej_tc, err_msg in rejected:
                err_result = {
                    "role": "tool",
                    "tool_call_id": rej_tc["id"],
                    "content": f"[参数验证失败] {err_msg}",
                }
                msgs.append(err_result)
                if self._attn_window:
                    self._attn_window.register_message(
                        err_result, turn=fc,
                        tool_name=rej_tc["function"]["name"])

            if valid_remote:
                # ── TSD v2.7: 高风险工具预检与 approval:required 等待 ──
                approved_remote = []
                approval_rejected_results = []
                for tc in valid_remote:
                    tool_name = tc["function"]["name"]
                    risk_level = self._get_tool_risk_level(tool_name, tc.get("function", {}).get("arguments", "{}"))
                    if risk_level in ("HIGH", "CRITICAL"):
                        try:
                            approved, decision = await self._wait_for_remote_tool_approval(
                                tc,
                                risk_level,
                                send_callback,
                                tool_result_queue,
                                turn=fc,
                            )
                            if approved:
                                approved_remote.append(tc)
                            else:
                                reason = decision.get("reason") or decision.get("action_summary") or "用户拒绝或审批超时"
                                approval_rejected_results.append({
                                    "role": "tool",
                                    "tool_call_id": tc.get("id", ""),
                                    "content": f"[审批未通过] {reason}",
                                })
                        except Exception as _e:
                            logger.warning(f"[IDEFCRunner] 高风险审批等待失败，拒绝执行 {tool_name}: {_e}")
                            approval_rejected_results.append({
                                "role": "tool",
                                "tool_call_id": tc.get("id", ""),
                                "content": f"[审批异常] {_e}",
                            })
                    else:
                        approved_remote.append(tc)

                for approval_result in approval_rejected_results:
                    msgs.append(approval_result)
                    if self._attn_window:
                        self._attn_window.register_message(
                            approval_result,
                            turn=fc,
                            tool_name=approval_result.get("tool_call_id", ""),
                        )

                valid_remote = approved_remote
                if not valid_remote:
                    state.pending_remote_calls = []
                    state.pending_call_ids = list(state.pending_call_turns.keys())
                    self._maybe_run_bfs(fc, "approval_rejected")
                    return None

                # 设置 pending 状态（累积而非覆盖，支持跨轮次）
                state.pending_remote_calls = valid_remote
                new_call_ids = [tc["id"] for tc in valid_remote]

                # 累积到pending_call_turns（而非覆盖pending_call_ids）
                for call_id in new_call_ids:
                    state.pending_call_turns[call_id] = fc
                state.pending_call_ids = list(state.pending_call_turns.keys())

                # 通过 WebSocket 推送 tool_request
                tool_names = [
                    tc["function"]["name"] for tc in valid_remote]
                group_id = f"tool_group:{self.session.session_id}:{fc}:{int(time.time() * 1000)}"
                task_pair_id = self._current_visible_step_pair_id or turn_pair_id or group_id
                for tc in valid_remote:
                    self._tool_interaction_pairs[tc.get("id", "")] = task_pair_id
                logger.info(
                    f"[IDEFCRunner] async 远程工具推送: {tool_names}, call_ids={new_call_ids}")
                llm_tool_reason = str(response_content or "").strip()
                if len(llm_tool_reason) > 500:
                    llm_tool_reason = llm_tool_reason[:497].rstrip() + "..."

                await self._emit_execution_event(
                    send_callback,
                    "tool_requested",
                    f"正在{self._friendly_tool_group(tool_names)}",
                    turn=fc,
                    event_type="IDE_TOOL_REQUEST",
                    payload={
                        "tools": [
                            {"name": tc["function"]["name"],
                             "action_label": self._friendly_tool_name(tc["function"]["name"]),
                             "action_summary": self._friendly_action_summary(
                                 tc["function"]["name"],
                                 self._safe_parse_tool_arguments(tc["function"].get("arguments", "{}")),
                             ),
                             "arguments_preview": tc["function"].get("arguments", "")[:300],
                             "call_id": tc.get("id", ""),
                             "pair_id": tc.get("id", "")}
                            for tc in valid_remote
                        ],
                        "call_ids": state.pending_call_ids,
                        "tool_names": tool_names,
                        "group_id": group_id,
                        "parallel": len(valid_remote) > 1,
                        "completed_count": 0,
                        "total_count": len(valid_remote),
                        "pair_id": task_pair_id,
                        "interaction": {
                            "pair_id": task_pair_id,
                            "kind": "action",
                            "status": "running",
                            "title": f"正在{self._friendly_tool_group(tool_names)}",
                            "detail": "这一步会交给 IDE 后台执行，完成后祖龙会继续判断下一步。",
                            "thought": llm_tool_reason,
                            "tool_name": ",".join(tool_names),
                            "progress": 0,
                            "next_step": "等待这一步完成。",
                            "progress_items": [
                                self._progress_item(
                                    f"正在{self._friendly_tool_name(name)}",
                                    "running",
                                    detail="等待执行结果返回。",
                                    source="tool",
                                    pair_id=call_id,
                                )
                                for name, call_id in zip(tool_names, new_call_ids)
                            ],
                        },
                    },
                    send_progress=True,
                    send_status=True,
                )

                await send_callback("tool_request", {
                    "tool_calls": valid_remote,
                    "call_ids": state.pending_call_ids,
                    "tool_names": tool_names,
                    "group_id": group_id,
                })

                # 等待所有远程工具结果
                results = []
                for i in range(len(valid_remote)):
                    if cancel_event.is_set():
                        return "cancelled"
                    try:
                        deadline = time.time() + self._remote_tool_timeout
                        while True:
                            result = await asyncio.wait_for(
                                tool_result_queue.get(),
                                timeout=max(0.1, deadline - time.time()),
                            )
                            if isinstance(result, dict) and result.get("type") == "approval_result":
                                logger.debug(
                                    "[IDEFCRunner] 忽略迟到审批结果: approval_id=%s call_id=%s",
                                    result.get("approval_id"),
                                    result.get("call_id"),
                                )
                                if time.time() >= deadline:
                                    raise asyncio.TimeoutError()
                                continue
                            results.append(result)
                            break
                    except asyncio.TimeoutError:
                        tc = valid_remote[i]
                        call_id = tc.get("id", "")
                        timeout_tool = tc["function"]["name"]
                        timeout_label = self._friendly_tool_name(timeout_tool)
                        await self._emit_execution_event(
                            send_callback,
                            "blocked",
                            f"{timeout_label}超时，需要确认",
                            turn=fc,
                            event_type="TASK_BLOCKED",
                            payload={
                            "tool_name": timeout_tool,
                            "call_id": call_id,
                                "pair_id": task_pair_id,
                            "timeout_seconds": self._remote_tool_timeout,
                            "interaction": {
                                    "pair_id": task_pair_id,
                                    "kind": "progress",
                                    "status": "blocked",
                                    "title": f"{timeout_label}超时，需要确认",
                                    "detail": f"等待这一步返回超过 {self._remote_tool_timeout}s。",
                                    "tool_name": timeout_tool,
                                    "next_step": "可以重试、取消，或根据当前结果调整任务。",
                                },
                            },
                            send_progress=True,
                            send_status=True,
                        )
                        # ── TSD v2.7: 发射 approval:required ──
                        try:
                            await send_callback(MessageType.APPROVAL_REQUIRED, {
                                "call_id": call_id,
                                "tool_name": timeout_tool,
                                "approval_mode": "manual",
                                "reason": f"{timeout_label}超时，需要确认",
                                "timeout_seconds": self._remote_tool_timeout,
                                "interaction": {
                                    "pair_id": call_id,
                                    "kind": "progress",
                                    "status": "blocked",
                                    "title": f"{timeout_label}超时，需要确认",
                                    "detail": f"等待这一步返回超过 {self._remote_tool_timeout}s。",
                                    "tool_name": timeout_tool,
                                    "next_step": "可以重试、取消，或根据当前结果调整任务。",
                                },
                                "timestamp": time.time(),
                            })
                        except Exception as _e:
                            logger.debug(f"[IDEFCRunner] 发射 approval:required 失败: {_e}")
                        results.append({
                            "call_id": call_id,
                            "tool_name": tc["function"]["name"],
                            "result": f"[工具执行超时 ({self._remote_tool_timeout}s)]",
                            "is_error": True,
                        })

                # 转换为 _inject_tool_results 期望的格式（保留is_error标记）
                formatted_results = []
                call_id_to_remote = {tc["id"]: tc for tc in valid_remote}
                for r in results:
                    call_id = r.get("call_id", "")
                    is_error = r.get("is_error", False)
                    content = r.get("result", "")
                    if is_error and content and not content.startswith("[错误]"):
                        content = f"[错误] {content}"
                    formatted_results.append({
                        "tool_call_id": call_id,
                        "content": content,
                    })

                self._inject_tool_results(state, formatted_results)
                self._maybe_run_bfs(fc, "tool_complete")

                failed_count = sum(1 for r in results if r.get("is_error", False))
                succeeded_count = len(results) - failed_count
                result_title = (
                    f"{failed_count} 项步骤需要复核"
                    if failed_count
                    else f"{self._friendly_tool_group(tool_names)}已完成"
                )
                result_detail = (
                    f"{succeeded_count} 项成功，{failed_count} 项需要复核。"
                    if failed_count
                    else "这一步已经返回结果，祖龙会基于结果继续推进。"
                )
                await self._emit_execution_event(
                    send_callback,
                    "tool_finished",
                    result_title,
                    turn=fc,
                    event_type="IDE_TOOL_RESULT",
                    payload={
                        "results": [
                            {"tool_name": r.get("tool_name", ""),
                             "call_id": r.get("call_id", ""),
                             "pair_id": r.get("call_id", ""),
                             "result_preview": (r.get("result", "") or "")[:500],
                             "is_error": r.get("is_error", False)}
                            for r in results
                        ],
                        "group_id": group_id,
                        "pair_id": task_pair_id,
                        "completed_count": len(results),
                        "total_count": len(valid_remote),
                        "failed_count": failed_count,
                        "succeeded_count": succeeded_count,
                        "interaction": {
                            "pair_id": task_pair_id,
                            "kind": "observation",
                            "status": "failed" if failed_count else "succeeded",
                            "title": result_title,
                            "detail": result_detail,
                            "progress": 100,
                            "next_step": "",
                            "progress_items": [
                                self._progress_item(
                                    self._friendly_tool_name(r.get("tool_name", "工具")),
                                    "failed" if r.get("is_error", False) else "completed",
                                    detail=self._friendly_result_detail(
                                        r.get("result", "") or "",
                                        "failed" if r.get("is_error", False) else "completed",
                                    ),
                                    source="tool",
                                    pair_id=r.get("call_id", ""),
                                )
                                for r in results
                            ],
                        },
                    },
                    send_progress=False,
                    send_status=True,
                )
                # ── TSD v2.7: 发射 graph:memory:diff（远程工具执行后）──
                await self._emit_memory_diff_async(send_callback, state.fc_turn)
                return None  # 继续循环

        self._maybe_run_bfs(fc, "tool_complete")
        # ── TSD v2.7: 发射 graph:memory:diff（内部工具执行后）──
        await self._emit_memory_diff_async(send_callback, state.fc_turn)
        return None  # 继续循环

    def _restore_runner_state(self) -> None:
        if self.session.attention_window_data:
            try:
                from zulong.l2.attention_window import AttentionWindowManager
                from zulong.tools.task_tools import get_active_task_graph
                from zulong.memory.memory_graph import get_memory_graph
                _restore_tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
                _restore_mg = get_memory_graph()
                self._attn_window = AttentionWindowManager.from_serialized(
                    self.session.attention_window_data,
                    task_graph=_restore_tg,
                    memory_graph=_restore_mg,
                )
            except Exception as e:
                logger.warning(f"[IDEFCRunner] 注意力窗口恢复失败: {e}")
        self._create_rule_guardian()
        # 恢复 RuleGuardian 状态
        if self._rule_guardian and self.session.rule_guardian_data:
            try:
                self._rule_guardian.deserialize(self.session.rule_guardian_data)
            except Exception as e:
                logger.warning(f"[IDEFCRunner] RuleGuardian 状态恢复失败: {e}")
        self._create_circuit_breaker()
        # 恢复 CircuitBreaker 状态
        if self._circuit_breaker and self.session.circuit_breaker_data:
            try:
                self._circuit_breaker.deserialize(self.session.circuit_breaker_data)
            except Exception as e:
                logger.warning(f"[IDEFCRunner] CircuitBreaker 状态恢复失败: {e}")
        self._create_drift_detector()
        # 恢复对话轮次跟踪状态
        self._current_round_id = self.session.dialogue_round_id
        self._current_session_id = self.session.dialogue_session_id
        self._init_dialogue_adapter()

    def _create_rule_guardian(self) -> None:
        try:
            if hasattr(self.engine, "_rule_guardian") and self.engine._rule_guardian:
                self._rule_guardian = type(self.engine._rule_guardian)()
            else:
                from zulong.l2.rule_guardian import RuleGuardian
                self._rule_guardian = RuleGuardian()
        except Exception as e:
            logger.warning(f"[IDEFCRunner] RuleGuardian 创建失败: {e}")

    def _create_circuit_breaker(self) -> None:
        try:
            if hasattr(self.engine, "_circuit_breaker") and self.engine._circuit_breaker:
                cb = self.engine._circuit_breaker
                cb_cfg = dict(getattr(cb, "_config", {}))
                # 确保 CB 的 context_window_size 与引擎一致
                cb_cfg["context_window_size"] = getattr(
                    self.engine, "_context_window_size", 32768)
                self._circuit_breaker = type(cb)(cb_cfg)
            else:
                from zulong.l2.circuit_breaker import ToolCallCircuitBreaker
                cb_cfg = {}
                try:
                    from zulong.config.config_manager import ConfigManager
                    cb_cfg = dict(
                        ConfigManager().get("l2_inference.circuit_breaker", {}) or {}
                    )
                except Exception:
                    cb_cfg = {}
                cb_cfg["context_window_size"] = getattr(
                    self.engine, "_context_window_size", 32768)
                self._circuit_breaker = ToolCallCircuitBreaker(cb_cfg or {
                    "context_window_size": getattr(
                        self.engine, "_context_window_size", 32768),
                })
        except Exception as e:
            logger.warning(f"[IDEFCRunner] CircuitBreaker 创建失败: {e}")

    def _create_drift_detector(self) -> None:
        """创建语义漂移检测器（轻量，仅在需要时才计算 Embedding）"""
        try:
            from zulong.memory.semantic_drift_detector import get_semantic_drift_detector
            self._drift_detector = get_semantic_drift_detector()
            logger.info("[IDEFCRunner] SemanticDriftDetector 已创建")
            _ensure_embedding_prewarm_async()
        except Exception as e:
            logger.warning(f"[IDEFCRunner] SemanticDriftDetector 创建失败: {e}")

    def _init_intent_filter(self) -> None:
        """初始化L1-B意图分类器（与Web端统一使用ALBERT模型）"""
        self._intent_filter = _get_shared_intent_filter()
        if self._intent_filter:
            if self._intent_filter._albert_enabled:
                logger.info("[IDEFCRunner] L1-B IntentFilter 已启用（ALBERT 15类）")
            else:
                logger.info("[IDEFCRunner] L1-B IntentFilter 已启用（关键词匹配）")

    def _send_message_safe(self, send_callback, msg_type: str, data: dict) -> bool:
        """线程安全的消息发送（修复no running event loop问题）
        
        三层容错机制：
        1. 优先使用全局主事件循环（_main_event_loop）
        2. 回退到当前线程的事件循环
        3. 异常时记录日志并返回False
        
        Args:
            send_callback: 异步发送回调函数
            msg_type: 消息类型（display_text/task_complete等）
            data: 消息数据
            
        Returns:
            发送是否成功
        """
        try:
            # 第一层：优先使用全局主事件循环（解决线程池工作线程问题）
            from zulong.ide.ide_server import _main_event_loop
            
            if _main_event_loop is not None and _main_event_loop.is_running():
                # 使用线程安全调度
                future = asyncio.run_coroutine_threadsafe(
                    send_callback(msg_type, data),
                    _main_event_loop
                )
                # 等待发送完成（带超时，避免阻塞）
                future.result(timeout=2.0)
                logger.debug(f"✅ [FC] {msg_type} 已发送（主循环）")
                return True
            
            # 第二层：回退到当前线程的事件循环（向后兼容）
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(send_callback(msg_type, data))
                logger.debug(f"✅ [FC] {msg_type} 已发送（当前循环）")
                return True
            except RuntimeError:
                pass
            
            # 第三层：降级处理
            logger.warning(f"⚠️ [FC] {msg_type} 发送跳过：无可用事件循环")
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ [FC] {msg_type} 发送失败: {e}")
            return False

    async def _emit_execution_event(
        self,
        send_callback,
        phase: str,
        message: str,
        *,
        turn: int = 0,
        event_type: str = "TASK_PROGRESS",
        payload: Optional[Dict[str, Any]] = None,
        send_progress: bool = True,
        send_status: bool = False,
        monitor: bool = True,
    ) -> None:
        """Normalize runner state once, then fan out through existing channels."""
        payload = payload or {}
        interaction = self._build_interaction_payload(phase, message, turn, event_type, payload)
        event = ExecutionEvent(
            phase=phase,
            message=message,
            turn=turn,
            event_type=event_type,
            payload=payload,
            interaction=interaction,
        )
        self._execution_events.append(event)
        self._persist_execution_event(event)
        cb_payload = event.callback_payload(self._max_fc_turns)
        public_interaction = not interaction or (
            interaction.get("ux_visibility") != UX_HIDDEN
            and interaction.get("channel") != CHANNEL_CONTROL
        )

        if send_progress and public_interaction:
            try:
                await send_callback("task_progress", cb_payload)
            except Exception:
                pass
        if send_status and public_interaction:
            try:
                await send_callback("status_update", {
                    "protocol_version": "2.0",
                    "turn": turn,
                    "phase": phase,
                    "message": message,
                    **payload,
                    "interaction": interaction,
                })
            except Exception:
                pass
        # ── TSD v2.7: 统一交互事件 emission (Web Dashboard 独立消费) ──
        if interaction and public_interaction:
            try:
                await send_callback(MessageType.INTERACTION_EVENT, {
                    "protocol_version": "2.0",
                    "interaction": interaction,
                    "turn": turn,
                    "event_type": event_type,
                    "phase": phase,
                    "timestamp": time.time(),
                })
            except Exception:
                pass
        if monitor and public_interaction:
            try:
                from zulong.ide.ide_server import broadcast_monitor_event
                await broadcast_monitor_event(event_type, {
                    "session_id": self.session.session_id,
                    "conversation_id": getattr(self.ide_session, "conversation_id", None),
                    "turn_id": getattr(self.ide_session, "web_turn_id", None),
                    "workspace_path": getattr(self.ide_session, "cwd", None),
                    "turn": turn,
                    "phase": phase,
                    "message": message,
                    **payload,
                    "interaction": interaction,
                })
            except Exception:
                pass

    def _emit_execution_event_sync(
        self,
        phase: str,
        message: str,
        *,
        turn: int = 0,
        event_type: str = "TASK_PROGRESS",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = payload or {}
        interaction = self._build_interaction_payload(phase, message, turn, event_type, payload)
        event = ExecutionEvent(
            phase=phase,
            message=message,
            turn=turn,
            event_type=event_type,
            payload=payload,
            interaction=interaction,
        )
        self._execution_events.append(event)
        self._persist_execution_event(event)
        public_interaction = not interaction or (
            interaction.get("ux_visibility") != UX_HIDDEN
            and interaction.get("channel") != CHANNEL_CONTROL
        )
        if public_interaction:
            _broadcast_sync(event_type, {
                "session_id": self.session.session_id,
                "conversation_id": getattr(self.ide_session, "conversation_id", None),
                "turn_id": getattr(self.ide_session, "web_turn_id", None),
                "workspace_path": getattr(self.ide_session, "cwd", None),
                "turn": turn,
                "phase": phase,
                "message": message,
                **payload,
                "interaction": interaction,
            })

    async def _emit_memory_diff_async(self, send_callback, turn: int) -> None:
        """发射 graph:memory:diff 事件，获取 MemoryGraph 自上次快照以来的变化。
        
        TSD v1.7 §23.7: 在工具执行后、注意力变更时发射记忆差异。
        """
        try:
            from zulong.memory.memory_graph import get_memory_graph
            _mg = get_memory_graph()
            _mem_changes = {"created": 0, "strengthened": 0, "pruned": 0}
            if _mg:
                if hasattr(_mg, '_last_activated_edges'):
                    _mem_changes["strengthened"] = len(_mg._last_activated_edges)
                    _created = 0
                    for _sid in _mg.list_all_shards():
                        _shard = _mg.get_shard(_sid)
                        if _shard:
                            try:
                                _created += len(_shard.topology)
                            except Exception:
                                pass
                    _mem_changes["created"] = _created
                elif hasattr(_mg, 'stats'):
                    _mem_changes["created"] = _mg.stats.get("total_nodes", 0)
            await send_callback(MessageType.GRAPH_MEMORY_DIFF, {
                "memory_changes": _mem_changes,
                "turn": turn,
                "timestamp": time.time(),
            })
        except Exception as _e:
            logger.debug(f"[IDEFCRunner] 发射 graph:memory:diff 失败: {_e}")

    # ── TSD v2.7 §23.4.2: 工具风险等级分类 ──
    _HIGH_RISK_TOOLS = frozenset({
        "write_to_file", "replace_in_file", "execute_command",
        "delete_files", "create_rule", "delete_files_by_pattern",
        "preview_url", "use_skill", "web_fetch", "web_search",
        "ask_followup_question", "attempt_completion",
    })
    _CRITICAL_RISK_TOOLS = frozenset({
        "execute_command",  # 命令执行可被 CRITICAL
        "delete_files",     # 不可逆删除
    })
    _INTERACTION_KINDS = frozenset({
        "plan", "action", "observation", "approval",
        "progress", "summary", "user_interject",
    })

    @classmethod
    def _get_tool_risk_level(cls, tool_name: str, tool_args: str = "{}") -> str:
        """返回工具的风险等级: LOW / MEDIUM / HIGH / CRITICAL"""
        import json as _json
        if tool_name in cls._CRITICAL_RISK_TOOLS:
            # 进一步判断: execute_command 若含 rm -rf / sudo 等升级为 CRITICAL
            if tool_name == "execute_command":
                try:
                    args = _json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                    command = args.get("command", "").lower()
                    dangerous = ("rm -rf", "sudo ", "chmod 777", "mkfs.", "dd if=",
                                ":(){ :|:& };:", "> /dev/sda", "format ")
                    if any(d in command for d in dangerous):
                        return "CRITICAL"
                except Exception:
                    pass
            return "HIGH"
        if tool_name in cls._HIGH_RISK_TOOLS:
            return "HIGH"
        return "LOW"

    async def _wait_for_remote_tool_approval(
        self,
        tool_call: Dict[str, Any],
        risk_level: str,
        send_callback,
        tool_result_queue: "asyncio.Queue",
        *,
        turn: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Wait for Web/IDE approval before dispatching a high-risk remote tool."""
        call_id = str(tool_call.get("id") or "")
        function = tool_call.get("function", {}) or {}
        tool_name = str(function.get("name") or "")
        raw_args = function.get("arguments", "{}")
        try:
            tool_args = _json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
        except Exception:
            tool_args = {}
        action_summary = self._friendly_action_summary(tool_name, tool_args)
        risk_text = f"{risk_level} 风险" if risk_level else "风险"

        runtime_mode = "manual"
        try:
            from zulong.config.approval_config import get_runtime_approval_mode, should_runtime_auto_approve

            runtime_mode = get_runtime_approval_mode()
            if should_runtime_auto_approve(tool_name, tool_args, risk_level=risk_level):
                await send_callback(MessageType.APPROVAL_REQUIRED, {
                    "approval_id": f"approval:{call_id}",
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "approval_mode": runtime_mode or "full_auto",
                    "risk_level": risk_level,
                    "reason": f"已按审批模式自动允许: {action_summary}",
                    "tool_args": tool_args,
                    "action_summary": action_summary,
                    "interaction": {
                        "approval_id": f"approval:{call_id}",
                        "pair_id": call_id,
                        "kind": "approval",
                        "status": "approved",
                        "title": "已允许继续执行",
                        "detail": f"{action_summary} 已按当前审批模式自动允许。",
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "risk_level": risk_level,
                        "risk_reason": f"{risk_text}: {action_summary}",
                        "approval_mode": runtime_mode or "full_auto",
                        "confirmation_state": "approved",
                        "turn": turn,
                    },
                    "timestamp": time.time(),
                })
                return True, {"approved": True, "reason": "auto_approved", "approval_id": f"approval:{call_id}"}
        except Exception as exc:
            logger.debug(f"[IDEFCRunner] 自动审批判断失败，转人工审批: {exc}")

        approval_mode = "popup" if risk_level == "CRITICAL" or runtime_mode == "popup" else "manual"
        approval_id = f"approval:{call_id}"
        await send_callback(MessageType.APPROVAL_REQUIRED, {
            "approval_id": approval_id,
            "call_id": call_id,
            "tool_name": tool_name,
            "approval_mode": approval_mode,
            "risk_level": risk_level,
            "reason": f"{risk_text}: {action_summary}",
            "tool_args": tool_args,
            "action_summary": action_summary,
            "interaction": {
                "approval_id": approval_id,
                "pair_id": call_id,
                "kind": "approval",
                "status": "awaiting_approval",
                "title": "需要确认后继续",
                "detail": f"{action_summary} 需要你确认后才会继续执行。",
                "tool_name": tool_name,
                "tool_args": tool_args,
                "risk_level": risk_level,
                "risk_reason": f"{risk_text}: {action_summary}",
                "approval_mode": approval_mode,
                "confirmation_state": "pending",
                "turn": turn,
            },
            "timestamp": time.time(),
        })

        skipped: List[Dict[str, Any]] = []
        deadline = time.time() + self._approval_timeout
        while time.time() < deadline:
            try:
                item = await asyncio.wait_for(
                    tool_result_queue.get(),
                    timeout=max(0.1, min(1.0, deadline - time.time())),
                )
            except asyncio.TimeoutError:
                continue
            if self._matches_approval_result(item, approval_id, call_id):
                for other in skipped:
                    await tool_result_queue.put(other)
                approved = bool(item.get("approved")) or str(item.get("action") or "").lower() in {
                    "approve",
                    "approved",
                    "confirm",
                    "confirmed",
                    "yes",
                    "true",
                }
                item["approved"] = approved
                item.setdefault("approval_id", approval_id)
                item.setdefault("call_id", call_id)
                return approved, item
            skipped.append(item)

        for other in skipped:
            await tool_result_queue.put(other)
        return False, {
            "approved": False,
            "approval_id": approval_id,
            "call_id": call_id,
            "reason": f"审批等待超时 ({self._approval_timeout}s)",
        }

    @staticmethod
    def _matches_approval_result(item: Dict[str, Any], approval_id: str, call_id: str) -> bool:
        if not isinstance(item, dict):
            return False
        if item.get("type") != "approval_result" and "approved" not in item and "action" not in item:
            return False
        candidates = {
            str(item.get("approval_id") or ""),
            str(item.get("approvalId") or ""),
            str(item.get("interaction_id") or ""),
            str(item.get("pair_id") or ""),
            str(item.get("call_id") or ""),
        }
        return approval_id in candidates or call_id in candidates

    def _next_interaction_id(self, prefix: str = "interaction") -> str:
        self._interaction_seq += 1
        return f"{prefix}:{self.session.session_id}:{int(time.time() * 1000)}:{self._interaction_seq}"

    def _normalize_interaction_kind(self, kind: Any, phase: str) -> str:
        """Return the public TSD §23.3.3 kind for legacy/internal phases."""
        kind_text = str(kind or "").strip()
        if kind_text == "user_adjustment":
            return "user_interject"
        if kind_text in self._INTERACTION_KINDS:
            return kind_text
        if phase == "started":
            return "plan"
        if phase in {"completed"}:
            return "summary"
        if phase in {"cancelled", "interrupted"}:
            return "user_interject"
        if phase in {"tool_requested", "executing"}:
            return "action"
        if phase in {"tool_finished", "diff_ready", "checkpoint_created"}:
            return "observation"
        if phase == "approval_required":
            return "approval"
        return "progress"

    @staticmethod
    def _progress_percent(progress: Any) -> Optional[int]:
        if progress is None or progress == "":
            return None
        if isinstance(progress, dict):
            for key in ("percent", "progress"):
                value = progress.get(key)
                if isinstance(value, (int, float)):
                    return int(max(0, min(100, round(value))))
            completed = progress.get("completed_count", progress.get("completed"))
            total = progress.get("total_nodes", progress.get("total"))
            if isinstance(completed, (int, float)) and isinstance(total, (int, float)) and total:
                return int(max(0, min(100, round((completed / total) * 100))))
            return None
        if isinstance(progress, (int, float)):
            return int(max(0, min(100, round(progress))))
        try:
            return int(max(0, min(100, round(float(progress)))))
        except Exception:
            return None

    @staticmethod
    def _progress_steps(progress: Any) -> Tuple[Optional[int], Optional[int]]:
        if not isinstance(progress, dict):
            return None, None
        current = progress.get("completed_count", progress.get("completed"))
        total = progress.get("total_nodes", progress.get("total"))
        return (
            int(current) if isinstance(current, (int, float)) else None,
            int(total) if isinstance(total, (int, float)) else None,
        )

    @staticmethod
    def _copy_summary_fields(interaction: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Copy TSD summary fields from payload/legacy summary into interaction."""
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        field_pairs = (
            ("completed_items", "completed"),
            ("verified_items", "verified"),
            ("pending_items", "remaining"),
            ("risks_summary", "risk"),
            ("next_step", "next_step"),
            ("memory_changes", "memory_changes"),
            ("completion_evidence", "completion_evidence"),
            ("memory_reference_edges", "memory_reference_edges"),
            ("task_graph_binding", "task_graph_binding"),
        )
        for tsd_key, legacy_key in field_pairs:
            if interaction.get(tsd_key) is not None:
                continue
            if payload.get(tsd_key) is not None:
                interaction[tsd_key] = payload.get(tsd_key)
            elif isinstance(summary, dict) and summary.get(tsd_key) is not None:
                interaction[tsd_key] = summary.get(tsd_key)
            elif isinstance(summary, dict) and summary.get(legacy_key) is not None:
                interaction[tsd_key] = summary.get(legacy_key)
        return interaction

    @staticmethod
    def _progress_item(
        label: str,
        status: str,
        *,
        detail: str = "",
        source: str = "",
        pair_id: str = "",
    ) -> Dict[str, Any]:
        label = " ".join(str(label or "").split())
        if not label:
            label = "继续推进当前事项"
        if status not in {"pending", "running", "completed", "blocked", "failed"}:
            status = "pending"
        item = {
            "id": pair_id or label[:48],
            "label": label[:160],
            "status": status,
        }
        if detail:
            item["detail"] = " ".join(str(detail).split())[:220]
        if source:
            item["source"] = source
        if pair_id:
            item["pair_id"] = pair_id
        item["timestamp"] = time.time()
        return item

    @staticmethod
    def _split_tool_names(value: Any) -> List[str]:
        if isinstance(value, dict):
            name = value.get("name") or value.get("tool_name")
            return [str(name).strip()] if name else []
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, list):
            names: List[str] = []
            for item in value:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("tool_name")
                else:
                    name = item
                if name:
                    names.extend(IDEFCRunner._split_tool_names(str(name)))
            return [name for name in names if name]
        return [str(value).strip()] if value else []

    @classmethod
    def _collect_interaction_tool_names(
        cls,
        interaction: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> List[str]:
        names: List[str] = []
        for value in (
            interaction.get("tool_name"),
            payload.get("tool_name"),
            payload.get("tool_names"),
        ):
            for name in cls._split_tool_names(value):
                if name and name not in names:
                    names.append(name)
        tools = payload.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                for name in cls._split_tool_names(tool):
                    if name and name not in names:
                        names.append(name)
        results = payload.get("results")
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    for name in cls._split_tool_names(item.get("tool_name")):
                        if name and name not in names:
                            names.append(name)
        return names

    @staticmethod
    def _is_background_tool(tool_name: Any) -> bool:
        return str(tool_name or "").strip() in _BACKGROUND_TOOLS

    @classmethod
    def _classify_tool_category(
        cls,
        tool_names: List[str],
        interaction: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> str:
        kind = str(interaction.get("kind") or "")
        if kind == "approval" or payload.get("approval_id"):
            return "approval"
        names = {str(name or "").strip() for name in tool_names if name}
        if names and all(name in _BACKGROUND_TOOLS for name in names):
            return "background"
        if names and all(name in _SUMMARY_ONLY_TOOLS for name in names):
            return "summary"
        if names & _COMMAND_TOOLS:
            return "command"
        if names & _WRITE_TOOLS:
            return "write"
        if names & _TASK_GRAPH_TOOLS:
            return "task_graph"
        if names & _NETWORK_TOOLS:
            return "network"
        if names & _READ_TOOLS:
            return "read"
        return "other"

    @staticmethod
    def _classify_source_channel(interaction: Dict[str, Any], payload: Dict[str, Any]) -> str:
        kind = str(interaction.get("kind") or "")
        explicit = str(interaction.get("source_channel") or payload.get("source_channel") or "").strip()
        if explicit:
            return explicit
        if kind == "summary":
            return "model_final"
        if interaction.get("thought") and kind in {"action", "progress", "plan"}:
            return "model_progress"
        if kind == "progress" and payload.get("model_step_note"):
            return "model_progress"
        return "system_status"

    @staticmethod
    def _channel_for_source(source_channel: str) -> str:
        if source_channel == "model_final":
            return CHANNEL_FINAL
        if source_channel == "model_progress":
            return CHANNEL_LEDGER
        if source_channel == "internal_control":
            return CHANNEL_CONTROL
        return CHANNEL_STATUS

    @staticmethod
    def _classify_ux_visibility(
        interaction: Dict[str, Any],
        tool_category: str,
        channel: str,
    ) -> str:
        explicit = str(interaction.get("ux_visibility") or "").strip().lower()
        if explicit in {UX_MAIN, UX_DETAILS, UX_HIDDEN}:
            return explicit
        kind = str(interaction.get("kind") or "")
        status = str(interaction.get("status") or "")
        event_type = str(interaction.get("event_type") or "")
        if channel == CHANNEL_CONTROL:
            return UX_HIDDEN
        if event_type == "GRAPH_MEMORY_DIFF":
            return UX_HIDDEN
        if tool_category in {"background", "summary"}:
            return UX_HIDDEN
        if kind in {"summary", "plan", "approval", "user_interject"}:
            return UX_MAIN
        if status in {"failed", "blocked", "awaiting_approval", "rejected"}:
            return UX_MAIN
        if kind in {"action", "observation"}:
            if tool_category in {"write", "command", "approval"}:
                return UX_MAIN
            return UX_DETAILS
        if kind == "progress":
            return UX_MAIN
        return UX_DETAILS

    @classmethod
    def _build_raw_details(
        cls,
        interaction: Dict[str, Any],
        payload: Dict[str, Any],
        tool_names: List[str],
        event_type: str,
    ) -> Dict[str, Any]:
        raw_details: Dict[str, Any] = {"event_type": event_type}
        if tool_names:
            raw_details["tool_name"] = ",".join(tool_names[:8])
        raw_args = interaction.get("tool_args") or payload.get("tool_args") or payload.get("args")
        if isinstance(raw_args, dict):
            raw_details["tool_args"] = _summarize_tool_args(raw_args)
        result_preview = payload.get("result_preview")
        if not result_preview and isinstance(payload.get("results"), list):
            previews: List[str] = []
            for item in payload["results"][:4]:
                if isinstance(item, dict):
                    preview = item.get("result_preview") or item.get("result") or ""
                    if preview:
                        previews.append(str(preview)[:220])
            if previews:
                result_preview = "\n".join(previews)
        if result_preview:
            raw_details["result_preview"] = str(result_preview)[:800]
        return raw_details

    def _apply_interaction_visibility(
        self,
        interaction: Dict[str, Any],
        payload: Dict[str, Any],
        phase: str,
        event_type: str,
    ) -> Dict[str, Any]:
        tool_names = self._collect_interaction_tool_names(interaction, payload)
        tool_category = self._classify_tool_category(tool_names, interaction, payload)
        source_channel = self._classify_source_channel(interaction, payload)
        channel = str(interaction.get("channel") or payload.get("channel") or self._channel_for_source(source_channel))
        ux_visibility = self._classify_ux_visibility(interaction, tool_category, channel)
        interaction["channel"] = channel
        interaction["source_channel"] = source_channel
        interaction["ux_visibility"] = ux_visibility
        interaction["tool_category"] = tool_category
        interaction["is_background"] = bool(tool_category == "background" or ux_visibility == UX_HIDDEN)
        if "raw_details" not in interaction:
            raw_details = self._build_raw_details(interaction, payload, tool_names, event_type)
            if raw_details:
                interaction["raw_details"] = raw_details
        if isinstance(payload.get("task_graph_binding"), dict):
            interaction.setdefault("task_graph_binding", payload["task_graph_binding"])
        if isinstance(payload.get("completion_evidence"), dict):
            interaction.setdefault("completion_evidence", payload["completion_evidence"])
        if isinstance(payload.get("memory_reference_edges"), list):
            interaction.setdefault("memory_reference_edges", payload["memory_reference_edges"])
        mark_public_payload(interaction, channel, ux_visibility=ux_visibility)
        return interaction

    @staticmethod
    def _friendly_tool_name(name: Any) -> str:
        key = str(name or "").strip()
        if not key:
            return "处理任务步骤"
        mapped = _FRIENDLY_TOOL_NAMES.get(key) or _FRIENDLY_TOOL_NAMES.get(key.lower())
        if mapped:
            return mapped
        if "_" in key:
            return "处理任务步骤"
        return key

    @classmethod
    def _friendly_tool_group(cls, names: Any) -> str:
        if isinstance(names, str):
            raw_names = [part.strip() for part in names.split(",") if part.strip()]
        elif isinstance(names, list):
            raw_names = []
            for item in names:
                if isinstance(item, dict):
                    raw_names.append(str(item.get("name") or item.get("tool_name") or "").strip())
                else:
                    raw_names.append(str(item or "").strip())
            raw_names = [name for name in raw_names if name]
        else:
            raw_names = [str(names or "").strip()] if names else []

        labels: List[str] = []
        for name in raw_names:
            label = cls._friendly_tool_name(name)
            if label and label not in labels:
                labels.append(label)
        if not labels:
            return "处理任务步骤"
        if len(labels) > 2:
            return f"并行处理 {len(labels)} 项步骤"
        return "、".join(labels)

    @staticmethod
    def _safe_parse_tool_arguments(raw_args: Any) -> Dict[str, Any]:
        if isinstance(raw_args, dict):
            return dict(raw_args)
        if not isinstance(raw_args, str):
            return {}
        try:
            parsed = _json.loads(raw_args or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _is_announce_step_tool(tool_name: Any) -> bool:
        return str(tool_name or "").strip() == ANNOUNCE_STEP_TOOL_NAME

    @classmethod
    def _is_announce_step_call(cls, tool_call: Dict[str, Any]) -> bool:
        fn = tool_call.get("function") if isinstance(tool_call, dict) else {}
        return cls._is_announce_step_tool((fn or {}).get("name"))

    @classmethod
    def _split_announce_step_calls(
        cls,
        tool_calls_data: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        announce_calls: List[Dict[str, Any]] = []
        real_calls: List[Dict[str, Any]] = []
        for tool_call in tool_calls_data or []:
            if cls._is_announce_step_call(tool_call):
                announce_calls.append(tool_call)
            else:
                real_calls.append(tool_call)
        return announce_calls, real_calls

    @classmethod
    def _announce_step_from_call(cls, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        fn = tool_call.get("function") if isinstance(tool_call, dict) else {}
        args = cls._safe_parse_tool_arguments((fn or {}).get("arguments", "{}"))
        message = " ".join(str(args.get("message") or "").split())
        if len(message) > 160:
            message = message[:157].rstrip() + "..."
        expected_actions: List[str] = []
        raw_actions = args.get("expected_actions")
        if isinstance(raw_actions, list):
            for value in raw_actions[:3]:
                text = " ".join(str(value or "").split())
                if text:
                    expected_actions.append(text[:80])
        return {
            "message": message,
            "expected_actions": expected_actions,
            "call_id": str(tool_call.get("id") or ""),
        }

    @classmethod
    def _first_announce_step(cls, announce_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        for tool_call in announce_calls or []:
            step = cls._announce_step_from_call(tool_call)
            if step.get("message"):
                return step
        return {"message": "", "expected_actions": [], "call_id": ""}

    @classmethod
    def _step_progress_items(
        cls,
        *,
        message: str,
        expected_actions: Optional[List[str]] = None,
        real_calls: Optional[List[Dict[str, Any]]] = None,
        pair_id: str = "",
    ) -> List[Dict[str, Any]]:
        labels: List[str] = []
        for action in expected_actions or []:
            action_text = " ".join(str(action or "").split())
            if action_text and action_text not in labels:
                labels.append(action_text)
        if not labels:
            for tool_call in (real_calls or [])[:4]:
                fn = tool_call.get("function") or {}
                name = fn.get("name") or ""
                args = cls._safe_parse_tool_arguments(fn.get("arguments", "{}"))
                label = cls._friendly_action_summary(name, args)
                if label and label not in labels:
                    labels.append(label)
        if not labels and message:
            labels.append(message)
        while len(labels) < 2:
            labels.append("根据结果判断下一步" if len(labels) == 1 else "汇总结果和风险")
        items: List[Dict[str, Any]] = []
        for idx, label in enumerate(labels[:6]):
            items.append(cls._progress_item(
                label,
                "running" if idx == 0 else "pending",
                source="model_progress",
                pair_id=f"{pair_id}:step:{idx}" if pair_id else "",
            ))
        return items

    async def _emit_model_step_progress(
        self,
        send_callback,
        *,
        state: IDEFCState,
        message: str,
        pair_id: str,
        expected_actions: Optional[List[str]] = None,
        real_calls: Optional[List[Dict[str, Any]]] = None,
        source: str = "assistant.content",
    ) -> None:
        note = " ".join(str(message or "").split())
        if not note:
            return
        if len(note) > 500:
            note = note[:497].rstrip() + "..."
        self._current_visible_step_pair_id = pair_id
        await self._emit_execution_event(
            send_callback,
            "model_progress",
            note,
            turn=state.fc_turn,
            event_type="MODEL_PROGRESS",
            payload={
                "model_step_note": note,
                "source_text_hash": hashlib.sha256(note.encode("utf-8")).hexdigest()[:16],
                "source_turn": state.fc_turn,
                "source_tool_group_id": pair_id,
                "source": source,
                "interaction": {
                    "pair_id": pair_id,
                    "kind": "progress",
                    "status": "running",
                    "title": "当前步骤",
                    "detail": note,
                    "source_channel": "model_progress",
                    "channel": CHANNEL_LEDGER,
                    "ux_visibility": UX_MAIN,
                    "progress_items": self._step_progress_items(
                        message=note,
                        expected_actions=expected_actions,
                        real_calls=real_calls,
                        pair_id=pair_id,
                    ),
                    "next_step": "等待这一步返回结果。",
                },
            },
            send_progress=True,
            send_status=True,
        )

    def _append_announce_step_messages(
        self,
        state: IDEFCState,
        announce_calls: List[Dict[str, Any]],
        response_content: str,
        fc: int,
        group_id: Any,
    ) -> None:
        if not announce_calls:
            return
        assistant_msg = {
            "role": "assistant",
            "content": response_content or "",
            "tool_calls": announce_calls,
        }
        state.messages.append(assistant_msg)
        if self._attn_window:
            self._attn_window.register_message(assistant_msg, turn=fc, group_id=group_id)
        for tool_call in announce_calls:
            call_id = str(tool_call.get("id") or "")
            step = self._announce_step_from_call(tool_call)
            state.messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": _json.dumps({
                    "ok": True,
                    "message": step.get("message") or "",
                    "expected_actions": step.get("expected_actions") or [],
                    "side_effect": "none",
                }, ensure_ascii=False),
            })
            if self._attn_window:
                self._attn_window.register_message(
                    state.messages[-1],
                    turn=fc,
                    tool_name=ANNOUNCE_STEP_TOOL_NAME,
                    group_id=group_id,
                )

    @classmethod
    def _friendly_action_summary(cls, tool_name: Any, tool_args: Any = None) -> str:
        label = cls._friendly_tool_name(tool_name)
        args = tool_args if isinstance(tool_args, dict) else {}
        path = str(args.get("path") or args.get("file_path") or args.get("workspace_path") or "").strip()
        command = str(args.get("command") or "").strip()
        if path and label in {"写入文件", "读取文件", "修改文件", "查看目录", "搜索文件", "打开文件", "查看差异"}:
            return f"{label}: {path}"
        if command and label == "执行命令":
            command = " ".join(command.split())
            if len(command) > 120:
                command = command[:117].rstrip() + "..."
            return f"{label}: {command}"
        return label if label != "处理任务步骤" else "执行任务步骤"

    @classmethod
    def _friendly_progress_label(cls, label: Any, status: str = "pending") -> str:
        text = " ".join(str(label or "").split())
        if not text:
            return "继续推进当前事项"
        for raw, friendly in _FRIENDLY_TOOL_NAMES.items():
            text = text.replace(raw, friendly)
        if text.startswith("执行 "):
            text = "正在" + cls._friendly_tool_group(text[3:])
        elif text.startswith("使用 "):
            text = "正在" + cls._friendly_tool_group(text[3:])
        if "返回异常" in text:
            text = text.replace("返回异常，需要复核", "需要复核").replace("返回异常", "需要复核")
        if status == "completed" and text in _FRIENDLY_TOOL_NAMES.values():
            text = f"{text}已完成"
        elif status == "running" and text in _FRIENDLY_TOOL_NAMES.values():
            text = f"正在{text}"
        if "_" in text:
            return "任务步骤需要复核" if status in {"failed", "blocked"} else "处理任务步骤"
        return text

    @staticmethod
    def _friendly_result_detail(result: Any, status: str = "completed") -> str:
        text = " ".join(str(result or "").split())
        if not text:
            return ""
        if status in {"failed", "blocked"}:
            return "返回信息需要复核，原始内容已保留在执行细节中。"
        noisy_tokens = ("call_id", "tool_name", "result_preview", "ide_session_id", "返回字段")
        if len(text) > 160 or any(token in text for token in noisy_tokens):
            return "已收到返回结果。"
        return text[:160]

    def _record_model_raw_output(
        self,
        state: IDEFCState,
        fc: int,
        *,
        raw_content: str,
        final_content: str,
        tool_calls: Optional[List[Dict]],
        finish_reason: str = "",
        usage: Any = None,
        source: str = "stream",
    ) -> Dict[str, Any]:
        """Phase 10 evidence log for what the model actually returned."""
        tool_calls = tool_calls or []
        summary = {
            "turn": fc,
            "source": source,
            "assistant_content_length_raw": len(raw_content or ""),
            "assistant_content_length_final": len(final_content or ""),
            "tool_call_count": len(tool_calls),
            "tool_calls": [_summarize_tool_call_for_log(tc) for tc in tool_calls],
            "finish_reason": finish_reason or "unknown",
            "usage": _summarize_usage_for_log(usage),
            "usage_available": bool(_summarize_usage_for_log(usage)),
        }
        try:
            state.last_model_raw_summary = summary
        except Exception:
            pass
        logger.info(
            "[IDEFCRunner][P10ModelOutput] %s",
            _json.dumps(summary, ensure_ascii=False, sort_keys=True),
        )
        return summary

    def _phase10_root_cause(
        self,
        state: IDEFCState,
        *,
        path: str,
        tool_calls: Optional[List[Dict]] = None,
        response_content: str = "",
        rejected_count: int = 0,
    ) -> str:
        """Classify the current blocking evidence using the Phase 10 enum."""
        path_text = str(path or "")
        if rejected_count or "invalid_tool_args" in path_text or "参数验证失败" in str(response_content or ""):
            return "invalid_tool_args"

        reviewer_health = getattr(state, "quality_reviewer_health", {}) or {}
        reviewer_status = str(reviewer_health.get("status") or "")
        if reviewer_status in {"failed", "non_json"} and (
            "reviewer" in path_text or "quality" in path_text
        ):
            return "backup_reviewer_fail"

        if (
            "quality" in path_text
            or "completion_gate" in path_text
            or getattr(state, "quality_iteration_count", 0) > 0
        ):
            return "quality_gate_loop"

        recent = list(getattr(state, "tool_results_buffer", []) or [])[-8:]
        for item in reversed(recent):
            tool = str(item.get("tool_name") or "")
            result = str(item.get("result") or "")
            success = bool(item.get("success", True))
            if tool in {"write_to_file", "replace_in_file", "ide_write_file", "exec_write_file"} and not success:
                if any(marker in result for marker in (
                    "未真实存在", "未应用", "未允许", "内容未应用", "内容不一致", "verified",
                )):
                    return "bridge_not_applied"

        cb_reason = str(getattr(state, "cb_trigger_reason", "") or "")
        attn_ratio = -1.0
        try:
            attn_ratio = float(
                getattr(
                    self._attn_window,
                    "trigger_context_pressure_ratio",
                    getattr(
                        self._attn_window,
                        "context_pressure_ratio",
                        getattr(self._attn_window, "usage_ratio", -1.0),
                    ),
                )
            )
        except Exception:
            attn_ratio = -1.0
        if (
            "context_pressure" in cb_reason
            or "context_pressure" in path_text
            or getattr(state, "pressure_force_attention", False)
            or attn_ratio >= 0.60
        ):
            return "context_pressure"

        if not tool_calls:
            return "no_tool_call"
        return "none"

    def _log_fc_decision_path(
        self,
        state: IDEFCState,
        *,
        path: str,
        tool_calls: Optional[List[Dict]] = None,
        response_content: str = "",
        root_cause: str = "",
        **details: Any,
    ) -> str:
        """Phase 10 decision-path log: why this turn continued, blocked, or finished."""
        cause = root_cause or self._phase10_root_cause(
            state,
            path=path,
            tool_calls=tool_calls,
            response_content=response_content,
            rejected_count=int(details.get("rejected_count") or 0),
        )
        safe_details = {
            key: value
            for key, value in details.items()
            if value is not None
        }
        record = {
            "turn": getattr(state, "fc_turn", 0),
            "path": path,
            "root_cause": cause,
            "details": safe_details,
        }
        try:
            state.last_fc_decision_path = path
            state.last_fc_root_cause = cause
            history = list(getattr(state, "fc_root_cause_history", []) or [])
            history.append(record)
            state.fc_root_cause_history = history[-80:]
        except Exception:
            pass
        logger.info(
            "[IDEFCRunner][P10Decision] %s",
            _json.dumps(record, ensure_ascii=False, default=str, sort_keys=True),
        )
        return cause

    @staticmethod
    def _normalize_progress_items(raw_items: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_items, list):
            return []
        normalized: List[Dict[str, Any]] = []
        seen = set()
        for raw in raw_items:
            if isinstance(raw, str):
                item = IDEFCRunner._progress_item(raw, "pending")
            elif isinstance(raw, dict):
                item = IDEFCRunner._progress_item(
                    raw.get("label") or raw.get("title") or raw.get("text") or "",
                    raw.get("status") or "pending",
                    detail=raw.get("detail") or "",
                    source=raw.get("source") or "",
                    pair_id=raw.get("pair_id") or raw.get("id") or "",
                )
                if raw.get("id"):
                    item["id"] = str(raw.get("id"))
            else:
                continue
            key = (item.get("id"), item.get("label"))
            if item["label"] and key not in seen:
                normalized.append(item)
                seen.add(key)
            if len(normalized) >= 8:
                break
        return normalized

    @staticmethod
    def _tool_result_failed_text(result: str) -> bool:
        raw = str(result or "").strip()
        if not raw:
            return False
        try:
            parsed = _json.loads(raw)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            nested = " ".join(
                str(parsed.get(key) or "")
                for key in ("result", "error", "message", "reason")
            ).lower()
            if any(marker.lower() in nested for marker in (
                "用户未应用",
                "用户未允许",
                "用户拒绝",
                "审批拒绝",
                "审批超时",
                "审批未通过",
                "未应用写入",
                "未允许",
            )):
                return True
            if parsed.get("is_error") is True or parsed.get("failed") is True:
                return True
            if parsed.get("is_error") is False or parsed.get("failed") is False:
                return False
            if parsed.get("ok") is True or parsed.get("success") is True:
                return False
            if parsed.get("ok") is False or parsed.get("success") is False:
                return True
            error_value = parsed.get("error")
            if error_value not in (None, "", False):
                return True
        head = raw[:180].lower()
        return (
            '"error": true' in head
            or '"is_error": true' in head
            or '"success": false' in head
            or '"ok": false' in head
            or head.startswith("error")
            or "失败" in head
            or "异常" in head
        )

    @classmethod
    def _tool_result_success(cls, item: Dict[str, Any]) -> bool:
        """Return success using structured tool result fields before text heuristics.

        The completion gate must not treat a command with ``returncode != 0`` as
        successful only because the command output also contains words such as
        "passed" or "成功".  Prefer objective result fields emitted by tools,
        then fall back to the legacy text detector.
        """
        if not isinstance(item, dict):
            return False
        result = item.get("result", "")
        parsed = cls._structured_tool_result(result)
        if parsed:
            for key in ("returncode", "exit_code", "status_code"):
                if key in parsed:
                    try:
                        return int(parsed.get(key)) == 0
                    except Exception:
                        pass
            for key in ("success", "ok"):
                if key in parsed:
                    return bool(parsed.get(key))
            for key in ("is_error", "failed", "error"):
                if key in parsed:
                    value = parsed.get(key)
                    if isinstance(value, bool):
                        return not value
                    if value not in (None, "", 0, "0", False):
                        return False
            status_text = str(parsed.get("status") or "").strip().lower()
            if status_text:
                if status_text in {"success", "succeeded", "ok", "passed", "pass", "completed"}:
                    return True
                if status_text in {"failed", "failure", "error", "blocked", "timeout", "timed_out"}:
                    return False
        if "success" in item:
            return bool(item.get("success"))
        return not cls._tool_result_failed_text(result)

    @staticmethod
    def _structured_tool_result(result: str) -> Dict[str, Any]:
        raw = str(result or "").strip()
        if not raw:
            return {}
        try:
            parsed = _json.loads(raw)
        except Exception:
            return {}
        if not isinstance(parsed, dict):
            return {}
        if isinstance(parsed.get("data"), dict):
            merged = dict(parsed["data"])
            for key in ("success", "error", "message"):
                if key in parsed and key not in merged:
                    merged[key] = parsed[key]
            return merged
        return parsed

    @classmethod
    def _memory_reference_edges_from_result(cls, result: str) -> List[Dict[str, Any]]:
        parsed = cls._structured_tool_result(result)
        raw_edges = parsed.get("memory_reference_edges")
        if not isinstance(raw_edges, list):
            return []
        edges: List[Dict[str, Any]] = []
        seen = set()
        for raw in raw_edges:
            if not isinstance(raw, dict):
                continue
            source = str(raw.get("source") or raw.get("src") or "").strip()
            target = str(raw.get("target") or raw.get("dst") or "").strip()
            relation = str(raw.get("relation") or raw.get("type") or "reference").strip()
            if not source or not target:
                continue
            key = (source, target, relation)
            if key in seen:
                continue
            seen.add(key)
            edges.append({
                "source": source,
                "target": target,
                "type": str(raw.get("type") or "reference"),
                "relation": relation,
                "task_graph_id": str(raw.get("task_graph_id") or ""),
                "target_role": str(raw.get("target_role") or ""),
                "created": bool(raw.get("created", False)),
            })
        return edges

    def _attention_state(self) -> Tuple[str, str]:
        """Return the current attention mode/focus without mutating the window."""
        if not self._attn_window:
            return "", ""
        mode = getattr(self._attn_window, "mode", "")
        mode_value = getattr(mode, "value", str(mode or ""))
        node_id = str(getattr(self._attn_window, "_current_node_id", "") or "")
        return mode_value, node_id

    def _task_graph_status_signature(self, tg) -> str:
        """Compact status fingerprint used only for navigator map de-dup."""
        try:
            nodes = [
                n for n in getattr(tg, "nodes", [])
                if not str(getattr(n, "id", "")).startswith("crg_")
            ]
            counts: Dict[str, int] = {}
            for node in nodes:
                status = str(getattr(node, "status", "") or "pending")
                counts[status] = counts.get(status, 0) + 1
            return ",".join(f"{k}:{counts[k]}" for k in sorted(counts))
        except Exception:
            return ""

    def _task_node_id_from_memory_id(self, raw_id: str, tg) -> str:
        """Convert MemoryGraph task ids back to TaskGraph local node ids."""
        if not raw_id or not tg:
            return ""
        raw = str(raw_id)
        try:
            if tg.get_node(raw):
                return raw
        except Exception:
            pass
        graph_id = str(getattr(tg, "id", "") or "")
        candidates: List[str] = []
        if raw.startswith("task:"):
            local = raw[5:]
            if graph_id and local.startswith(f"{graph_id}/"):
                local = local[len(graph_id) + 1:]
            candidates.append(local)
        if raw.startswith("tg:"):
            parts = raw.split("/")
            for part in reversed(parts):
                if part.startswith("task:"):
                    candidates.append(part[5:])
                    break
            if parts:
                candidates.append(parts[-1])
        if "/" in raw:
            candidates.append(raw.rsplit("/", 1)[-1])
        for candidate in candidates:
            try:
                if candidate and tg.get_node(candidate):
                    return candidate
            except Exception:
                continue
        return ""

    def _task_node_address_from_memory_id(self, raw_id: str, tg) -> str:
        node_id = self._task_node_id_from_memory_id(raw_id, tg)
        if node_id and tg:
            try:
                return tg.get_node_address(node_id)
            except Exception:
                pass
        return str(raw_id or "unknown")

    def _build_attention_navigation_map(
        self,
        state: IDEFCState,
        fc: int,
        reason: str,
        current_node_id: str = "",
    ) -> Optional[Dict]:
        """Build one navigator map after a real attention switch.

        The caller is responsible for checking that mode/focus actually changed.
        This method only de-duplicates the latest map key and never pins the map.
        """
        try:
            tg = self._get_current_task_graph()
            if not tg or not hasattr(tg, "render_navigator_map"):
                return None
            mode, focus_node_id = self._attention_state()
            focus_node_id = current_node_id or focus_node_id
            if focus_node_id and not tg.get_node(focus_node_id):
                focus_node_id = ""
            if not focus_node_id:
                in_progress = [
                    n for n in tg.get_nodes_by_status("in_progress")
                    if not str(getattr(n, "id", "")).startswith("crg_")
                ]
                focus_node_id = in_progress[0].id if in_progress else "req"
            graph_id = str(getattr(tg, "id", "") or "")
            signature = self._task_graph_status_signature(tg)
            key = f"{graph_id}|{mode}|{focus_node_id}|{reason}|{signature}"
            if key == getattr(state, "last_navigation_map_key", ""):
                return None
            state.last_navigation_map_key = key
            uncovered_node_ids: List[str] = []
            try:
                coverage = self._compute_node_coverage(state, tg)
                uncovered_node_ids = list(coverage.get("uncovered_in_progress") or [])
            except Exception:
                uncovered_node_ids = []
            map_text = tg.render_navigator_map(
                focus_node_id,
                uncovered_node_ids=uncovered_node_ids,
            )
            return {
                "role": "system",
                "content": (
                    f"[导航地图] reason={reason}; mode={mode}; turn={fc}\n"
                    f"{map_text}\n"
                    "请基于当前位置继续推进；没有新的注意力切换时不要请求重复地图。"
                ),
            }
        except Exception as exc:
            logger.debug(f"[IDEFCRunner] 导航地图构建跳过: {exc}")
            return None

    def _build_observation_nudge(
        self,
        state: IDEFCState,
        tool_name: str,
        result_text: str,
        fc: int,
    ) -> Optional[Dict]:
        """Create a short independent nudge for failed or empty observations."""
        skip_tools = {
            "task_view_overview", "recall_memory", "read_memory_node",
            "discover_related", "search_memory", "search_experience",
            "search_tools", "navigate_attention", "adjust_attention_mode",
        }
        if tool_name in skip_tools:
            return None
        if getattr(state, "observation_nudge_turn", 0) != fc:
            state.observation_nudge_turn = fc
            state.observation_nudge_count = 0
        if getattr(state, "observation_nudge_count", 0) >= 2:
            return None

        raw = str(result_text or "").strip()
        failed = False
        try:
            failed = bool(self.engine._tool_result_failed(raw, raw[:200]))
        except Exception:
            failed = self._tool_result_failed_text(raw)
        empty = not raw or raw in {"{}", "[]", "null", "None"} or len(raw) < 8
        if not failed and not empty:
            return None

        state.observation_nudge_count += 1
        if failed:
            content = (
                f"[观察分析] {self._friendly_tool_name(tool_name)} 执行失败。"
                "请先分析失败原因，再选择重试、替代工具或收敛说明风险。"
            )
        else:
            content = (
                f"[观察分析] {self._friendly_tool_name(tool_name)} 返回空结果。"
                "请判断这是否足以推进；若不足，请换用更合适的工具。"
            )
        return internal_control_message(content)

    def _derive_progress_items(
        self,
        interaction: Dict[str, Any],
        payload: Dict[str, Any],
        phase: str,
        message: str,
    ) -> List[Dict[str, Any]]:
        def snapshot() -> List[Dict[str, Any]]:
            return [dict(item) for item in self._checklist[:8]]

        def upsert(item: Dict[str, Any]) -> None:
            item_id = str(item.get("id") or item.get("pair_id") or item.get("label") or "")
            if not item_id:
                return
            item["id"] = item_id
            if item_id in self._checklist_by_id:
                self._checklist_by_id[item_id].update(item)
                return
            self._checklist_by_id[item_id] = item
            self._checklist.append(item)

        existing = self._normalize_progress_items(interaction.get("progress_items"))
        kind = interaction.get("kind")
        pair_id = str(interaction.get("pair_id") or payload.get("pair_id") or payload.get("call_id") or "")
        source_channel = str(interaction.get("source_channel") or payload.get("source_channel") or "")
        if source_channel == "model_progress" and pair_id:
            self._current_visible_step_pair_id = pair_id
        tool_names = self._collect_interaction_tool_names(interaction, payload)
        tool_category = self._classify_tool_category(tool_names, interaction, payload)

        if tool_category in {"background", "summary"}:
            return snapshot()

        if existing:
            for item in existing:
                if kind in {"action", "observation"} and tool_category in {"read", "task_graph", "network"} and interaction.get("status") not in {"failed", "blocked"}:
                    continue
                upsert(item)
            return snapshot() or existing

        if kind == "plan":
            steps = interaction.get("plan_steps") or payload.get("plan_steps") or []
            items: List[Dict[str, Any]] = []
            if isinstance(steps, list):
                self._checklist = []
                self._checklist_by_id = {}
                for idx, step in enumerate(steps[:8]):
                    item = self._progress_item(
                        str(step),
                        "running" if idx == 0 else "pending",
                        source="plan",
                        pair_id=f"{pair_id}:plan:{idx}" if pair_id else "",
                    )
                    items.append(item)
                    upsert(item)
            return snapshot() or items

        if kind == "action":
            if tool_category in {"read", "task_graph", "network"}:
                return snapshot()
            label = self._friendly_progress_label(interaction.get("title") or message, "running")
            detail = self._friendly_result_detail(interaction.get("detail") or "", "running")
            item = self._progress_item(label, "running", detail=detail, source="tool", pair_id=pair_id)
            upsert(item)
            self._current_visible_step_pair_id = pair_id
            return snapshot() or [item]

        if kind == "observation":
            status = "failed" if interaction.get("status") == "failed" else "completed"
            if tool_category in {"read", "task_graph", "network"} and status == "completed":
                return snapshot()
            label = self._friendly_progress_label(interaction.get("title") or message, status)
            detail = self._friendly_result_detail(interaction.get("result_preview") or interaction.get("detail") or "", status)
            item = self._progress_item(label, status, detail=detail, source="tool", pair_id=pair_id)
            upsert(item)
            return snapshot() or [item]

        if kind == "summary":
            self._copy_summary_fields(interaction, payload)
            items: List[Dict[str, Any]] = []
            for value in interaction.get("completed_items") or []:
                items.append(self._progress_item(self._friendly_progress_label(value, "completed"), "completed", source="summary"))
            for value in interaction.get("verified_items") or []:
                items.append(self._progress_item(self._friendly_progress_label(value, "completed"), "completed", source="summary"))
            for value in interaction.get("pending_items") or []:
                items.append(self._progress_item(self._friendly_progress_label(value, "pending"), "pending", source="summary"))
            if not items:
                items.append(self._progress_item(interaction.get("title") or message, "completed", source="summary", pair_id=pair_id))
            return items[:8]

        if interaction.get("status") in {"blocked", "failed"}:
            item = self._progress_item(interaction.get("title") or message, interaction.get("status"), detail=interaction.get("detail") or "", source="heartbeat", pair_id=pair_id)
            upsert(item)
            return snapshot() or [item]

        return snapshot()

    def _apply_tool_call_budget_ide(
        self,
        state: IDEFCState,
        tool_calls_data: List[Dict],
        send_callback=None,
    ) -> List[Dict]:
        """Enforce explicit user tool-call budgets before IDE tool dispatch."""
        if not tool_calls_data:
            return []
        announce_calls, real_calls = self._split_announce_step_calls(tool_calls_data)
        if not real_calls:
            return announce_calls
        budget = state.tool_call_budget
        if budget is None:
            budget = get_engine_tool_budget(self.engine)
            state.tool_call_budget = budget
        if budget is None:
            return tool_calls_data
        used = max(int(state.tool_calls_used or 0), get_engine_tool_calls_used(self.engine))
        remaining = max(0, int(budget) - used)
        if remaining <= 0:
            self._inject_tool_budget_stop_ide(state, int(budget), used, send_callback=send_callback)
            return announce_calls
        allowed_real = real_calls[:remaining]
        allowed_ids = {id(item) for item in allowed_real}
        allowed = [
            item for item in tool_calls_data
            if self._is_announce_step_call(item) or id(item) in allowed_ids
        ]
        skipped = len(real_calls) - len(allowed_real)
        record_engine_tool_calls_used(self.engine, len(allowed_real))
        state.tool_calls_used = used + len(allowed_real)
        if skipped > 0 or engine_tool_budget_exhausted(self.engine):
            self._inject_tool_budget_stop_ide(
                state,
                int(budget),
                state.tool_calls_used,
                skipped=skipped,
                send_callback=send_callback,
            )
        return allowed

    def _inject_tool_budget_stop_ide(
        self,
        state: IDEFCState,
        budget: int,
        used: int,
        *,
        skipped: int = 0,
        send_callback=None,
    ) -> None:
        note = (
            f"[工具预算硬控] 用户要求本轮最多调用 {budget} 个工具；"
            f"当前已允许执行 {used} 个。"
        )
        if skipped > 0:
            note += f" 已拦截 {skipped} 个超额工具调用。"
        note += " 请基于已有工具结果和上下文直接总结，不允许继续调用工具。"
        state.cb_force_no_tools = True
        state.messages.append({"role": "user", "content": note})
        if self._attn_window:
            try:
                self._attn_window.register_message(state.messages[-1], turn=state.fc_turn, pinned=True)
            except Exception:
                pass
        self._emit_execution_event_sync(
            "blocked",
            "已达到本轮工具上限，转为整理已有结果。",
            turn=state.fc_turn,
            event_type="TOOL_BUDGET_LIMIT",
            payload={
                "tool_call_budget": budget,
                "tool_calls_used": used,
                "skipped_tool_calls": skipped,
                "interaction": {
                    "kind": "progress",
                    "status": "blocked",
                    "title": "已达到工具上限",
                    "detail": f"本轮工具上限 {budget} 个，已用 {used} 个；后续会基于已有信息总结。",
                    "next_step": "整理已有工具结果和上下文。",
                    "progress_items": [
                        self._progress_item(
                            f"工具预算 {used}/{budget}",
                            "blocked",
                            source="tool",
                        )
                    ],
                },
            },
        )

    def _tool_budget_event_payload(self, state: IDEFCState) -> Dict[str, Any]:
        budget = int(state.tool_call_budget or 0)
        used = int(state.tool_calls_used or 0)
        return {
            "tool_call_budget": budget,
            "tool_calls_used": used,
            "reason": "tool_budget_limit",
            "interaction": {
                "kind": "progress",
                "status": "blocked",
                "title": "已达到工具上限",
                "detail": f"本轮工具上限 {budget} 个，已用 {used} 个；后续会基于已有信息总结。",
                "next_step": "整理已有工具结果和上下文。",
                "progress_items": [
                    self._progress_item(
                        f"工具预算 {used}/{budget}",
                        "blocked",
                        source="tool",
                    )
                ],
            },
        }

    def _build_interaction_payload(
        self,
        phase: str,
        message: str,
        turn: int,
        event_type: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """OpenHands-style visible interaction semantics layered on old events."""
        existing = payload.get("interaction")
        if isinstance(existing, dict):
            merged = dict(existing)
            merged.setdefault("interaction_id", self._next_interaction_id(phase))
            merged.setdefault("pair_id", merged.get("interaction_id"))
            merged.setdefault("protocol_version", "2.0")
            merged["kind"] = self._normalize_interaction_kind(merged.get("kind"), phase)
            merged.setdefault("title", message)
            merged.setdefault("detail", message)
            merged.setdefault("turn", turn)
            merged.setdefault("event_type", event_type)
            if merged.get("progress") is not None:
                pct = self._progress_percent(merged.get("progress"))
                if pct is not None:
                    merged["progress"] = pct
            current_step, total_steps = self._progress_steps(payload.get("progress"))
            if current_step is not None:
                merged.setdefault("current_step", current_step)
            if total_steps is not None:
                merged.setdefault("total_steps", total_steps)
            if merged["kind"] == "summary":
                self._copy_summary_fields(merged, payload)
            merged["progress_items"] = self._derive_progress_items(merged, payload, phase, message)
            return self._apply_interaction_visibility(merged, payload, phase, event_type)

        kind = "progress"
        status = "running"
        title = message
        detail = message
        pair_id = payload.get("pair_id") or payload.get("call_id") or payload.get("group_id")
        tool_name = payload.get("tool_name")
        risk_level = payload.get("risk_level", "")
        confirmation_state = payload.get("confirmation_state", "")

        if phase in {"tool_requested", "executing"}:
            kind = "action"
            status = "running"
            title = self._tool_action_title(payload, message)
        elif phase == "tool_finished":
            kind = "observation"
            status = "failed" if payload.get("is_error") else "succeeded"
            title = self._tool_observation_title(payload, message)
        elif phase in {"approval_required", "diff_ready", "checkpoint_created"}:
            kind = "approval" if phase == "approval_required" else "observation"
            status = "awaiting_approval" if phase == "approval_required" else payload.get("status", "succeeded")
            title = payload.get("action_summary") or payload.get("summary") or message
            risk_level = risk_level or payload.get("risk", "")
            confirmation_state = confirmation_state or ("awaiting_confirmation" if phase == "approval_required" else "")
        elif phase in {"waiting_model", "calling_model"}:
            kind = "progress"
            status = "running"
            title = "正在推理"
        elif phase in {"blocked"}:
            kind = "progress"
            status = "blocked"
            title = "执行受阻"
        elif phase in {"completed"}:
            kind = "summary"
            status = "succeeded"
            title = "任务完成"
        elif phase in {"cancelled", "interrupted"}:
            kind = "user_interject"
            status = "cancelled"
            title = "任务已被打断"
        elif phase in {"error"}:
            kind = "progress"
            status = "failed"
            title = "执行出错"
        elif phase == "started":
            kind = "plan"
            status = "running"
            title = "已接收任务"

        if not pair_id:
            pair_id = self._next_interaction_id(phase)

        # 审批模式确定 (TSD 23.4.2)
        approval_mode = payload.get("approval_mode", "")
        if phase == "approval_required" and not approval_mode:
            if risk_level == "CRITICAL":
                approval_mode = "popup"
            elif risk_level == "HIGH":
                approval_mode = "manual"
            elif risk_level == "MEDIUM":
                approval_mode = "manual"
            else:
                approval_mode = "manual"

        # L2 thought (为什么选择这个工具 — TSD 23.2.4)
        thought = payload.get("thought", "")
        progress_value = self._progress_percent(payload.get("progress"))
        current_step, total_steps = self._progress_steps(payload.get("progress"))

        interaction = {
            "interaction_id": self._next_interaction_id(phase),
            "pair_id": str(pair_id),
            "kind": kind,
            "status": status,
            "protocol_version": "2.0",
            "title": title,
            "detail": detail,
            "thought": thought,
            "tool_name": tool_name or "",
            "tool_args": payload.get("tool_args", payload.get("args", None)),
            "risk_level": risk_level,
            "risk_reason": payload.get("risk_reason", ""),
            "approval_mode": approval_mode,
            "confirmation_state": confirmation_state,
            "progress": progress_value,
            "current_step": current_step,
            "total_steps": total_steps,
            "next_step": payload.get("next_step", ""),
            "turn": turn,
            "event_type": event_type,
            "phase": phase,
            "memory_changes": payload.get("memory_changes") if isinstance(payload.get("memory_changes"), dict) else None,
        }
        if interaction["kind"] == "summary":
            self._copy_summary_fields(interaction, payload)
        interaction["progress_items"] = self._derive_progress_items(interaction, payload, phase, message)
        return self._apply_interaction_visibility(interaction, payload, phase, event_type)

    def _tool_action_title(self, payload: Dict[str, Any], fallback: str) -> str:
        tools = payload.get("tools")
        if isinstance(tools, list) and tools:
            names = []
            for item in tools:
                if isinstance(item, dict):
                    names.append(item.get("name") or item.get("tool_name") or "")
                else:
                    names.append(str(item))
            names = [n for n in names if n]
            if names:
                return "正在" + self._friendly_tool_group(names[:4])
        if payload.get("tool_name"):
            return "正在" + self._friendly_tool_name(payload["tool_name"])
        return fallback

    def _tool_observation_title(self, payload: Dict[str, Any], fallback: str) -> str:
        results = payload.get("results")
        if isinstance(results, list) and results:
            failed = sum(1 for item in results if isinstance(item, dict) and item.get("is_error"))
            if failed:
                return f"{failed} 项步骤需要复核"
            names = [item.get("tool_name", "") for item in results if isinstance(item, dict)]
            return self._friendly_tool_group(names) + "已完成"
        if payload.get("tool_name"):
            return self._friendly_tool_name(payload["tool_name"]) + "已完成"
        return fallback

    def _get_memory_changes_snapshot(self) -> Dict[str, int]:
        """Return a compact MemoryGraph diff/count snapshot for task summary."""
        changes = {"created": 0, "strengthened": 0, "pruned": 0}
        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            if not mg:
                return changes
            if hasattr(mg, "_last_activated_edges"):
                changes["strengthened"] = len(getattr(mg, "_last_activated_edges", []) or [])
                created = 0
                if hasattr(mg, "list_all_shards") and hasattr(mg, "get_shard"):
                    for shard_id in mg.list_all_shards():
                        shard = mg.get_shard(shard_id)
                        if shard:
                            try:
                                created += len(shard.topology)
                            except Exception:
                                pass
                changes["created"] = created
            elif hasattr(mg, "stats"):
                stats = getattr(mg, "stats", {}) or {}
                changes["created"] = int(stats.get("total_nodes", 0) or 0)
        except Exception as exc:
            logger.debug(f"[IDEFCRunner] memory_changes 统计跳过: {exc}")
        return changes

    @staticmethod
    def _append_unique(items: List[str], text: str, limit: int = 6) -> None:
        text = str(text or "").strip()
        if not text or text in items or len(items) >= limit:
            return
        items.append(text)

    def _build_task_summary_payload(
        self,
        state: IDEFCState,
        *,
        reason: str = "done",
        final_text: str = "",
        memory_changes: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """Build the TSD §23.3 summary from the execution ledger."""
        completed_items: List[str] = []
        verified_items: List[str] = []
        pending_items: List[str] = []
        tool_names: List[str] = []
        failed_tools: List[str] = []

        for event in self._execution_events:
            interaction = event.interaction or {}
            if interaction.get("ux_visibility") == UX_HIDDEN:
                continue
            kind = interaction.get("kind")
            status = interaction.get("status")
            tool_name = (
                interaction.get("tool_name")
                or event.payload.get("tool_name")
                or ""
            )
            tool_category = self._classify_tool_category(
                self._split_tool_names(tool_name),
                interaction,
                event.payload,
            )
            if tool_category in {"background", "summary"}:
                continue
            if kind == "action" and tool_name:
                for name in str(tool_name).split(","):
                    if name.strip() and name.strip() not in tool_names:
                        tool_names.append(name.strip())
            if kind == "observation":
                if status == "failed":
                    if tool_name:
                        failed_tools.append(self._friendly_tool_group(tool_name))
                    self._append_unique(
                        pending_items,
                        self._friendly_progress_label(interaction.get("title") or event.message, "pending"),
                    )
                else:
                    title = interaction.get("title") or event.message
                    self._append_unique(completed_items, self._friendly_progress_label(title, "completed"))
            if kind == "progress" and status in {"blocked", "failed"}:
                self._append_unique(
                    pending_items,
                    self._friendly_progress_label(interaction.get("title") or event.message, "pending"),
                )

        if tool_names:
            self._append_unique(completed_items, "已完成执行步骤: " + self._friendly_tool_group(tool_names[:6]))
        if not completed_items:
            self._append_unique(completed_items, "已完成本轮推理流程")

        latest_tool_results: Dict[str, Dict[str, Any]] = {}
        ordered_tools: List[str] = []
        for item in getattr(state, "tool_results_buffer", []) or []:
            tool = str(item.get("tool_name", "") or "").strip()
            if not tool:
                continue
            tool_category = self._classify_tool_category([tool], {}, {})
            if tool_category in {"background", "summary"}:
                continue
            if tool not in latest_tool_results:
                ordered_tools.append(tool)
            latest_tool_results[tool] = item

        for tool in ordered_tools:
            item = latest_tool_results.get(tool) or {}
            result = item.get("result", "") or ""
            tool_failed = (
                not bool(item.get("success", not self._tool_result_failed_text(result)))
                or self._tool_result_failed_text(result)
            )
            if tool_failed:
                friendly_tool = self._friendly_tool_name(tool)
                failed_tools.append(friendly_tool)
                self._append_unique(pending_items, f"{friendly_tool}需要复核")
            else:
                self._append_unique(verified_items, f"{self._friendly_tool_name(tool)}已返回结果")

        progress = self._get_progress_snapshot()
        if progress:
            total = progress.get("total_nodes", 0) or 0
            done = progress.get("completed_count", 0) or 0
            pending = progress.get("pending_count", 0) or 0
            running = progress.get("in_progress_count", 0) or 0
            if total:
                self._append_unique(completed_items, f"任务清单进度 {done}/{total}")
            if pending or running:
                self._append_unique(pending_items, f"仍有 {pending + running} 个步骤需要继续处理")

        if reason != "done":
            self._append_unique(pending_items, f"FC 以 {reason} 状态结束")

        risks = []
        if failed_tools:
            risks.append("存在步骤失败或异常: " + "、".join(sorted(set(failed_tools))[:6]))
        if pending_items:
            risks.append("存在未完成或需复核事项")
        quality_reasons = list(getattr(state, "quality_last_reasons", []) or [])
        if quality_reasons:
            risks.append("需要关注: " + "；".join(str(r) for r in quality_reasons[:3]))
        risks_summary = "；".join(risks)

        memory_changes = memory_changes or self._get_memory_changes_snapshot()
        completion_evidence = getattr(state, "completion_last_evidence", {}) or {}
        completion_evidence_payload = (
            completion_evidence.get("completion_evidence")
            if isinstance(completion_evidence, dict)
            else {}
        ) or {}
        evidence_constraints = (
            completion_evidence.get("constraints", {})
            if isinstance(completion_evidence, dict)
            else {}
        ) or {}
        evidence_violations = [
            str(item)
            for item in evidence_constraints.get("violated_constraints", []) or []
            if str(item).strip()
        ]
        failed_commands_uncovered = [
            str(item)
            for item in completion_evidence_payload.get("failed_commands_uncovered", []) or []
            if str(item).strip()
        ]
        if evidence_violations:
            self._append_unique(
                pending_items,
                "完成证据需复核: " + "；".join(evidence_violations[:2]),
            )
        if failed_commands_uncovered:
            self._append_unique(
                pending_items,
                "存在失败命令未覆盖: " + "、".join(failed_commands_uncovered[:3]),
            )
        memory_reference_edges: List[Dict[str, Any]] = []
        seen_memory_edges = set()
        for item in getattr(state, "tool_results_buffer", []) or []:
            for edge in item.get("memory_reference_edges") or []:
                if not isinstance(edge, dict):
                    continue
                source = str(edge.get("source") or "")
                target = str(edge.get("target") or "")
                relation = str(edge.get("relation") or edge.get("type") or "reference")
                key = (source, target, relation)
                if not source or not target or key in seen_memory_edges:
                    continue
                seen_memory_edges.add(key)
                memory_reference_edges.append(edge)
        next_step = "等待用户继续补充或提出新调整。"
        if pending_items:
            next_step = "建议先处理未完成项，再继续后续任务。"
        quality_level = self._quality_level_from_state(state)
        risks = []
        if failed_tools:
            risks.append("存在步骤失败或异常: " + "、".join(sorted(set(failed_tools))[:6]))
        if pending_items:
            risks.append("存在未完成或需复核事项")
        if quality_reasons:
            risks.append("需要关注: " + "；".join(str(r) for r in quality_reasons[:3]))
        risks_summary = "；".join(risks)
        if failed_tools:
            summary_status = "failed"
        elif pending_items or reason != "done" or quality_level == "blocked" or evidence_violations:
            summary_status = "blocked"
        else:
            summary_status = "succeeded"

        return {
            "status": summary_status,
            "completed_items": completed_items,
            "verified_items": verified_items,
            "pending_items": pending_items,
            "risks_summary": risks_summary,
            "quality_score": round(float(getattr(state, "quality_last_score", 1.0) or 0.0), 3),
            "quality_level": quality_level,
            "quality_iterations": int(getattr(state, "quality_iteration_count", 0) or 0),
            "quality_dimensions": (
                getattr(state, "completion_last_quality", {}) or {}
            ).get("dimensions") if isinstance(getattr(state, "completion_last_quality", {}), dict) else None,
            "memory_changes": memory_changes,
            "memory_reference_edges": memory_reference_edges,
            "completion_evidence": completion_evidence_payload,
            "next_step": next_step,
            # Legacy fields retained for older Web renderers.
            "completed": completed_items,
            "verified": verified_items,
            "remaining": pending_items,
            "risk": risks_summary,
        }

    def _persist_execution_event(self, event: ExecutionEvent) -> None:
        """Best-effort event ledger + graph memory sink."""
        try:
            from zulong.launcher.interaction_store import get_interaction_store
            event_id = get_interaction_store().append_event(
                conversation_id=getattr(self.ide_session, "conversation_id", None),
                turn_id=getattr(self.ide_session, "web_turn_id", None),
                event_type=event.phase,
                role="system",
                source="ide_fc_runner",
                text=event.message,
                payload={
                    "event_type": event.event_type,
                    "phase": event.phase,
                    "turn": event.turn,
                    "session_id": self.session.session_id,
                    "source_event_id": "",
                    **event.payload,
                    "interaction": event.interaction,
                },
                workspace_path=getattr(self.ide_session, "cwd", None),
                project_id=getattr(self.ide_session, "project_id", None),
                task_graph_id=getattr(self.ide_session, "task_graph_id", None),
            )
            try:
                from zulong.launcher.memory_mirror import mirror_interaction_to_memory_graph
                from zulong.review.task_execution_extractor import maybe_finalize_task_execution_trace

                payload = {
                    "event_type": event.event_type,
                    "phase": event.phase,
                    "turn": event.turn,
                    "session_id": self.session.session_id,
                    "source_event_id": event_id,
                    **event.payload,
                    "interaction": event.interaction,
                }
                mirror_interaction_to_memory_graph(
                    conversation_id=getattr(self.ide_session, "conversation_id", None),
                    turn_id=getattr(self.ide_session, "web_turn_id", None),
                    role="system",
                    text=event.message,
                    event_type=event.phase,
                    source="ide_fc_runner",
                    payload=payload,
                )
                maybe_finalize_task_execution_trace(
                    conversation_id=getattr(self.ide_session, "conversation_id", None),
                    turn_id=getattr(self.ide_session, "web_turn_id", None),
                    task_graph_id=getattr(self.ide_session, "task_graph_id", None),
                    event_type=event.phase,
                )
            except Exception:
                pass
        except Exception as exc:
            logger.debug(f"[IDEFCRunner] InteractionStore 事件记录跳过: {exc}")

        if event.phase in {
            "started",
            "tool_requested",
            "tool_finished",
            "diff_ready",
            "approval_required",
            "checkpoint_created",
            "blocked",
            "completed",
            "cancelled",
            "error",
        }:
            self._persist_event_to_memory(event)

    def _persist_event_to_memory(self, event: ExecutionEvent) -> None:
        try:
            from zulong.memory.memory_graph import (
                get_memory_graph, GraphNode, NodeType, Importance, EdgeType,
            )
            mg = get_memory_graph()
            if not mg:
                return
            now = time.time()
            node_id = f"exec:{self.session.session_id}:{event.created_at:.6f}:{event.phase}"
            node = GraphNode(
                node_id=node_id,
                node_type=NodeType.EPISODE,
                label=f"{event.phase}: {event.message[:80]}",
                activation=0.6,
                created_at=now,
                last_accessed=now,
                access_count=1,
                backend_ref=f"ide_session:{self.session.session_id}",
                metadata={
                    "event_phase": event.phase,
                    "event_type": event.event_type,
                    "message": event.message,
                    "turn": event.turn,
                    "conversation_id": getattr(self.ide_session, "conversation_id", None),
                    "turn_id": getattr(self.ide_session, "web_turn_id", None),
                    "workspace_path": getattr(self.ide_session, "cwd", None),
                    "payload": event.payload,
                    "interaction": event.interaction,
                    "source": "ide_fc_runner",
                },
            )
            mg.add_node(node)
            mg.set_importance(
                node_id,
                Importance.IMPORTANT if event.phase in {"completed", "blocked", "checkpoint_created"} else Importance.NORMAL,
            )
            try:
                mg.index_summary(node_id, f"{event.phase} {event.message} {_json.dumps(event.payload, ensure_ascii=False)[:500]}")
            except Exception:
                pass
            if self._current_round_id:
                try:
                    mg.add_edge(self._current_round_id, node_id, EdgeType.TEMPORAL, weight=0.4)
                except Exception:
                    pass
        except Exception as exc:
            logger.debug(f"[IDEFCRunner] MemoryGraph 事件记录跳过: {exc}")

    def _init_dialogue_adapter(self) -> None:
        """创建 DialogueAdapter 实例（复用或新建）"""
        try:
            from zulong.memory.graph_adapters import DialogueAdapter
            self._dialogue_adapter = DialogueAdapter()
        except Exception as e:
            logger.warning(f"[IDEFCRunner] DialogueAdapter 创建失败: {e}")

    def _init_dialogue_tracking(self, state: IDEFCState) -> None:
        """新 FC 会话开始时初始化对话轮次跟踪

        在 MemoryGraph 中创建 session 和 round 节点，
        使后续工具执行产生的子对话能正确挂载到图谱中。
        """
        try:
            self._init_dialogue_adapter()
            if not self._dialogue_adapter:
                return
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            if not mg:
                return

            user_input = state.user_input_text or ""
            task_graph_id = self.session.active_task_graph_id

            # 确定或创建 session 节点
            self._current_session_id = self._dialogue_adapter.ensure_session(
                mg, user_input, task_graph_id=task_graph_id)

            # 创建本次对话轮次节点
            request_id = f"ide_{self.session.session_id[:12]}_{int(time.time())}"
            self._current_round_id = self._dialogue_adapter.add_round(
                mg, request_id=request_id, goal=user_input,
                task_graph_id=task_graph_id,
                session_id=self._current_session_id)

            # 绑定 session → task（使 BFS 遍历可从会话节点发现关联任务）
            if task_graph_id and self._current_session_id:
                self._dialogue_adapter.bind_session_to_task(
                    mg, self._current_session_id, task_graph_id)

            logger.info(
                f"[IDEFCRunner] 对话跟踪初始化: session={self._current_session_id}, "
                f"round={self._current_round_id}")
        except Exception as e:
            logger.warning(f"[IDEFCRunner] 对话跟踪初始化失败: {e}")

    def _record_sub_dialogue(self, state: IDEFCState,
                             tool_name: str, result: str) -> None:
        """记录一次工具执行为子对话节点"""
        if not self._dialogue_adapter or not self._current_round_id:
            return
        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            if not mg:
                return
            self._dialogue_adapter.add_sub_dialogue(
                mg, round_id=self._current_round_id,
                turn=state.fc_turn, tool_name=tool_name,
                content=result[:200] if result else "",
                role="tool")
        except Exception as e:
            logger.debug(f"[IDEFCRunner] 子对话记录失败: {e}")

    def _finalize_dialogue_round(self, state: IDEFCState,
                                 status: str = "completed") -> None:
        """完成当前对话轮次，更新元数据并索引到 FAISS"""
        if not self._dialogue_adapter or not self._current_round_id:
            return
        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            if not mg:
                return
            self._dialogue_adapter.finalize_round(
                mg, round_id=self._current_round_id,
                total_turns=state.fc_turn, status=status)
            logger.info(
                f"[IDEFCRunner] 对话轮次完成: {self._current_round_id} "
                f"({state.fc_turn} turns, {status})")
        except Exception as e:
            logger.warning(f"[IDEFCRunner] 对话轮次完成记录失败: {e}")

    def _save_runner_state(self) -> None:
        if self._attn_window:
            try:
                self.session.attention_window_data = self._attn_window.serialize()
            except Exception as e:
                logger.warning(f"[IDEFCRunner] 注意力窗口序列化失败: {e}")
        if self._rule_guardian:
            try:
                self.session.rule_guardian_data = self._rule_guardian.serialize()
            except Exception as e:
                logger.warning(f"[IDEFCRunner] RuleGuardian 序列化失败: {e}")
        if self._circuit_breaker:
            try:
                self.session.circuit_breaker_data = self._circuit_breaker.serialize()
            except Exception as e:
                logger.warning(f"[IDEFCRunner] CircuitBreaker 序列化失败: {e}")
        # 对话轮次状态持久化
        self.session.dialogue_round_id = self._current_round_id
        self.session.dialogue_session_id = self._current_session_id
        # P1-8: 持久化session到磁盘
        try:
            if hasattr(self.session, 'session_id'):
                from zulong.ide.ide_server import get_session_store
                store = get_session_store()
                if store:
                    store.save_session(self.session)
        except Exception as e:
            logger.debug(f"[IDEFCRunner] session磁盘持久化跳过: {e}")

    def _init_state(self, messages: List[Dict]) -> IDEFCState:
        # 清除残留的中断标志（防止恢复任务时立即中断）
        if hasattr(self, 'engine') and self.engine:
            if getattr(self.engine, '_interrupt_flag', False):
                logger.info("[IDEFCRunner] 清除残留的中断标志")
                self.engine._interrupt_flag = False
        
        from zulong.models.container import LLM_MODEL_ID
        user_input = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                c = msg.get("content", "")
                if isinstance(c, str):
                    user_input = c
                elif isinstance(c, list):
                    # IDE 多模态格式: [{"type": "text", "text": "..."}, ...]
                    user_input = " ".join(
                        item.get("text", "")
                        for item in c
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
                break

        # ── Layer 1: IDE 任务图策略，仅用于工具边界和任务图复用 ──
        _force_gid = getattr(self, 'force_graph_id', '') or ''
        task_graph_policy = "inspect_or_create"
        has_active_tg = False
        if _force_gid:
            # 确定性恢复: graph_id 已由 ide_server 加载，跳过启发式
            from zulong.tools.task_tools import get_active_task_graph
            _tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
            if _tg and getattr(_tg, 'id', '') == _force_gid:
                task_graph_policy = "reuse"
                has_active_tg = True
                self.session.active_task_graph_id = _force_gid
                self._notify_session_linked(_force_gid)
                logger.info(
                    f"[IDEFCRunner] 确定性恢复模式: graph_id={_force_gid}")
            else:
                # 活跃图加载失败(不应发生), 降级到启发式
                logger.warning(
                    f"[IDEFCRunner] 确定性恢复降级: 活跃图不匹配 {_force_gid}")
                task_graph_policy, has_active_tg = self._detect_task_graph_policy(user_input)
        else:
            task_graph_policy, has_active_tg = self._detect_task_graph_policy(user_input)

        # 非确定性路径下，继续已有任务图时关联活跃图到 session
        if not _force_gid and task_graph_policy in {"reuse", "inspect", "continue"} and has_active_tg:
            from zulong.tools.task_tools import get_active_task_graph
            _tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
            if _tg and hasattr(_tg, 'id') and not self.session.active_task_graph_id:
                self.session.active_task_graph_id = getattr(_tg, 'id', None)
                self._notify_session_linked(self.session.active_task_graph_id)
                logger.info(
                    f"[IDEFCRunner] 继续任务图策略：关联活跃图 "
                    f"{self.session.active_task_graph_id} 到新 session")

        # ── Layer 2: 根据任务图策略获取工具定义 ───────────
        tool_defs = self.tool_registry.get_combined_tool_definitions_for_policy(task_graph_policy)

        # L1-B/策略预判只决定候选工具边界，不能强制 L2 首轮调用某个工具。
        force_first = False

        state = IDEFCState(
            messages=list(messages), fc_turn=0, tool_definitions=tool_defs,
            user_input_text=user_input, vllm_model_id=LLM_MODEL_ID or "",
            phase="running", response_max_tokens=8192,
            is_resume=(task_graph_policy in {"reuse", "inspect", "continue"}),
            task_graph_policy=task_graph_policy,
            force_first_tool=force_first,
        )
        budget = sync_engine_tool_budget(self.engine, user_input)
        state.tool_call_budget = budget
        state.tool_calls_used = get_engine_tool_calls_used(self.engine)
        if budget == 0:
            state.force_first_tool = False
            logger.info("[IDEFCRunner] 用户显式限制本轮不调用工具")
        # ── TSD v2.7: L1BToolPredictor 工具预判接入 ──
        try:
            from zulong.l1b.tool_predictor import L1BToolPredictor
            _predictor = L1BToolPredictor()
            _registry = getattr(getattr(self.engine, "tool_engine", None), "registry", None)
            _conv_history = [
                m for m in messages
                if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            ]
            state.tool_prediction = _predictor.predict_tools(
                user_input, _conv_history, registry=_registry
            )
            _pred_ctx = state.tool_prediction.get("context_bundle", {}) or {}
            _pred_tools = (
                state.tool_prediction.get("predicted_tools")
                or state.tool_prediction.get("suggested_tools")
                or []
            )
            logger.debug(
                f"[IDEFCRunner] L1BToolPredictor: "
                f"turn_shape={_pred_ctx.get('turn_shape')}, "
                f"source={_pred_ctx.get('tool_prediction_source')}, "
                f"policy={state.tool_prediction.get('task_graph_policy')}, "
                f"suggested={_pred_tools[:5]}"
            )
        except Exception as _e:
            logger.debug(f"[IDEFCRunner] L1BToolPredictor 失败: {_e}")
            state.tool_prediction = None
        from zulong.l2.attention_window import AttentionWindowManager
        from zulong.tools.task_tools import get_active_task_graph
        from zulong.memory.memory_graph import get_memory_graph
        _init_tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
        _init_mg = get_memory_graph()
        self._attn_window = AttentionWindowManager(
            context_window_size=getattr(self.engine, "_context_window_size", 32768),
            task_graph=_init_tg,
            memory_graph=_init_mg,
        )
        for msg in messages:
            self._attn_window.register_message(msg, turn=0, pinned=msg.get("role") == "system")
        self._create_rule_guardian()
        self._create_circuit_breaker()
        self._create_drift_detector()
        logger.debug("[IDEFCRunner] 跳过自动建图；任务图创建必须来自 L2 的真实 tool_call")
        self._init_dialogue_tracking(state)
        logger.info(
            f"[IDEFCRunner] task_graph_policy={task_graph_policy}, "
            f"force_first_tool={force_first}, tools={len(tool_defs)}, "
            f"has_active_tg={has_active_tg}"
        )
        return state

    def _detect_task_graph_policy(self, user_input: str) -> tuple:
        """根据引用、活跃任务图和 L1-B 细粒度信号预判任务图策略。"""
        has_active_tg = False
        try:
            from zulong.tools.task_tools import get_active_task_graph
            tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
            if tg is not None:
                has_active_tg = True
        except Exception:
            pass

        ref_graph_id = self._try_activate_from_reference(user_input)
        if ref_graph_id:
            return "reuse", True

        # 任务恢复由 L1-B/L2/LLM 语义判断并调用 task_list_suspended /
        # task_resume_by_address 等恢复工具；这里不再通过关键词自动加载最近备份。

        if self._intent_filter:
            try:
                intent_result = self._intent_filter.analyze(user_input)
                predicted = (intent_result.get("intent", "UNKNOWN") or "").lower()
                if predicted in ("task_code", "task_analysis", "task_write", "task_execute", "task_search", "task_read"):
                    return "inspect_or_create", has_active_tg
            except Exception as e:
                logger.debug(f"[IDEFCRunner] IntentFilter 检查失败: {e}")

        return "inspect_or_create", has_active_tg

    def _try_activate_from_reference(self, user_input: str) -> Optional[str]:
        """尝试从用户输入中的 @[label#address] 引用激活历史 TaskGraph

        流程:
        1. 正则匹配 @[...#tg:xxx/task:yyy] 格式
        2. 通过 MemoryGraph.resolve_address() 定位节点
        3. 从节点 metadata 提取 graph_id
        4. 调用 rebuild_task_graph_from_memory() 重建
        5. 设置为活跃图

        Returns:
            成功时返回 graph_id，失败返回 None
        """
        import re as _re
        if not user_input:
            return None

        # 匹配 @[任意标签#地址] 格式，地址部分以 tg: 开头
        pattern = r'@\[([^#\]]+)#(tg:[^\]]+)\]'
        match = _re.search(pattern, user_input)
        if not match:
            return None

        label = match.group(1)
        address = match.group(2)
        logger.debug(f"[IDEFCRunner] 检测到节点引用: label={label}, address={address}")

        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            node = mg.resolve_address(address)
            if node is None:
                logger.warning(
                    f"[IDEFCRunner] 无法解析地址 {address}，MemoryGraph 中未找到")
                return None

            # 提取 graph_id: 从地址 "tg:{graph_id}/task:{node_id}" 解析
            graph_id = None
            try:
                from zulong.tools.task_tools import normalize_task_graph_id

                graph_id = normalize_task_graph_id(address)
            except Exception:
                if address.startswith("tg:"):
                    parts = address.split("/")
                    graph_id = parts[0][3:]  # 去掉 "tg:" 前缀

            if not graph_id:
                graph_id = node.metadata.get("graph_id")

            if not graph_id:
                logger.warning(
                    f"[IDEFCRunner] 无法从节点引用中提取 graph_id: {address}")
                return None

            # 检查是否已经是活跃图
            from zulong.tools.task_tools import (
                get_active_task_graph, set_active_task_graph
            )
            current_tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
            if current_tg and getattr(current_tg, 'id', '') == graph_id:
                logger.info(
                    f"[IDEFCRunner] 引用的图谱已是活跃图: {graph_id}")
                return graph_id

            # 从 MemoryGraph 重建 TaskGraph
            from zulong.memory.graph_adapters import rebuild_task_graph_from_memory
            rebuilt_tg = rebuild_task_graph_from_memory(mg, graph_id)
            if rebuilt_tg is None:
                logger.warning(
                    f"[IDEFCRunner] 重建 TaskGraph 失败: {graph_id}")
                return None

            # 设置为活跃图
            set_active_task_graph(rebuilt_tg, graph_id, workspace_dir=getattr(self, 'cwd', None))
            logger.info(
                f"[IDEFCRunner] 通过节点引用激活 TaskGraph: "
                f"graph_id={graph_id}, nodes={len(rebuilt_tg._nodes)}")
            return graph_id

        except Exception as e:
            logger.warning(f"[IDEFCRunner] 节点引用激活失败: {e}")
            return None

    def _infer_active_node_id(self) -> Optional[str]:
        """推断当前活跃的 TaskGraph 节点 ID（供远程工具结果关联）"""
        try:
            from zulong.tools.task_tools import get_active_task_graph
            tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
            if tg:
                ip = tg.get_nodes_by_status("in_progress")
                if ip:
                    return ip[0].id
        except Exception:
            pass
        return None

    def _inject_tool_results(self, state: IDEFCState, tool_results: List[Dict]) -> None:
        """注入远程工具执行结果（提升为一等公民，与 _exec_internal 对等处理）"""
        # ── 安全验证 ──
        # 数量上限：不超过 pending_call_ids 长度的 2 倍
        max_results = max(len(state.pending_call_ids) * 2, 1)
        if len(tool_results) > max_results:
            logger.warning(
                f"[IDEFCRunner] 工具结果数量异常: {len(tool_results)} > "
                f"pending*2={max_results}, 截断")
            tool_results = tool_results[:max_results]

        # call_id 白名单
        valid_ids = set(state.pending_call_ids)

        # 构建 call_id → 原始工具信息映射
        call_id_to_func = {}
        for rc in state.pending_remote_calls:
            call_id_to_func[rc["id"]] = rc.get("function", {})

        active_node_id = self._infer_active_node_id()

        for tr in tool_results:
            call_id = tr["tool_call_id"]

            # 验证 call_id 属于 pending 白名单
            if call_id not in valid_ids:
                logger.warning(
                    f"[IDEFCRunner] 拒绝未知 call_id: {call_id}, "
                    f"valid={list(valid_ids)[:5]}")
                continue
            
            # 从白名单移除，防止重复注入
            valid_ids.discard(call_id)
            # 同时从pending_call_turns移除
            if hasattr(state, 'pending_call_turns'):
                state.pending_call_turns.pop(call_id, None)

            func_info = call_id_to_func.get(call_id, {})
            tool_name = func_info.get("name", "ide_remote")
            content = tr["content"]

            # 截断保护（与 _exec_internal 一致）
            if len(content) > MAX_TOOL_RESULT_CHARS:
                orig_len = len(content)
                content = content[:MAX_TOOL_RESULT_CHARS] + \
                    f"\n...(已截断，原始长度 {orig_len} 字符)"

            tm = {"role": "tool", "tool_call_id": call_id, "content": content}
            state.messages.append(tm)

            if self._attn_window:
                self._attn_window.register_message(
                    tm, turn=state.fc_turn,
                    tool_name=tool_name, node_id=active_node_id)
                # 触发注意力模式切换
                try:
                    args_dict = _json.loads(func_info.get("arguments", "{}"))
                except Exception:
                    args_dict = {}
                self._attn_window.observe_tool_call(tool_name, args_dict)

            # CircuitBreaker 记录（与 _exec_internal 一致）
            if self._circuit_breaker:
                self._circuit_breaker.record_call(tool_name, {}, content[:500])

            # 工具结果缓冲（供 InfoGap / Backfill 使用），限制上限
            if len(state.tool_results_buffer) >= _TOOL_RESULTS_BUFFER_MAX:
                state.tool_results_buffer.pop(0)
            try:
                args_dict = _json.loads(func_info.get("arguments", "{}"))
            except Exception:
                args_dict = {}
            memory_reference_edges = self._memory_reference_edges_from_result(content)
            state.tool_results_buffer.append(
                {
                    "tool_name": tool_name,
                    "result": content[:500],
                    "arguments": args_dict,
                    "success": not self._tool_result_failed_text(content),
                    "turn": state.fc_turn,
                    "memory_reference_edges": memory_reference_edges,
                })
            nudge_msg = self._build_observation_nudge(
                state, tool_name, content, state.fc_turn)
            if nudge_msg:
                state.messages.append(nudge_msg)
                if self._attn_window:
                    self._attn_window.register_message(
                        nudge_msg, turn=state.fc_turn, node_id=active_node_id)

        # 仅在所有pending结果都已处理后清空
        # pending_call_turns 通过 discard 机制自动管理，无需手动清空
        state.pending_remote_calls = []
        state.pending_call_ids = list(state.pending_call_turns.keys()) if hasattr(state, 'pending_call_turns') else []
        # 推送 TaskGraph 更新到 web 仪表盘（远程工具结果返回后）
        try:
            self.engine._publish_task_graph_event(
                "agent_tool_call", state.fc_turn,
                "ide_tool_results", f"远程工具结果注入: {len(tool_results)} 条")
        except Exception:
            pass

        # ── 混合自动锚定：write_to_file / replace_in_file 后置钩子 ──
        # 检测写文件工具并自动触发 CRG 索引 + TASK→CODE_SYMBOL 锚定
        self._auto_anchor_on_write(tool_results, call_id_to_func, active_node_id)

    # ── 混合自动锚定核心方法 ──────────────────────────────────

    _WRITE_TOOLS = {"write_to_file", "replace_in_file", "create_file", "insert_code_block"}

    def _auto_anchor_on_write(
        self,
        tool_results: List[Dict],
        call_id_to_func: Dict[str, Dict],
        active_node_id: Optional[str],
    ) -> None:
        """write_to_file / replace_in_file 后自动触发 CRG 索引 + TASK→CODE_SYMBOL 锚定

        策略:
        - 仅对代码文件（支持的扩展名）触发
        - 通过 MD5 去重避免重复索引
        - 自动为当前活跃 TASK 节点建立 TASK→CODE_SYMBOL REFERENCE 边
        - 自动创建 CodeAnchor 记录（owner_ref 指向任务节点）
        - 自动建立 TaskGraph d_edge（任务节点→代码符号节点）
        """
        try:
            # 收集本批次中的写文件路径
            written_files = set()
            for tr in tool_results:
                call_id = tr.get("tool_call_id", "")
                func_info = call_id_to_func.get(call_id, {})
                tool_name = func_info.get("name", "")
                if tool_name not in self._WRITE_TOOLS:
                    continue

                # 从参数中提取 file_path
                args_str = func_info.get("arguments", "{}")
                try:
                    args = _json.loads(args_str)
                except Exception:
                    args = {}
                fp = args.get("path") or args.get("file_path") or ""
                if fp:
                    written_files.add(fp.replace("\\", "/"))

            if not written_files:
                return

            # 判断是否为可索引的代码文件
            import os
            try:
                from zulong.code.graph_builder import ext_to_lang
            except ImportError:
                return

            code_files = []
            for fp in written_files:
                ext = os.path.splitext(fp)[1]
                if ext_to_lang(ext):
                    code_files.append(fp)

            if not code_files:
                return

            # 延迟导入所需模块
            from zulong.memory.memory_graph import get_memory_graph, NodeType, EdgeType

            mg = get_memory_graph()
            if mg is None:
                return

            adapter = getattr(mg, "_adapters", {}).get("code_graph")
            if adapter is None:
                try:
                    from zulong.memory.graph_adapters import register_all_adapters
                    register_all_adapters(mg)
                    adapter = getattr(mg, "_adapters", {}).get("code_graph")
                except Exception:
                    pass
            if adapter is None:
                return

            # 获取或创建 IndexCodeFileTool 实例（复用哈希缓存）
            if not hasattr(self, "_index_tool_instance"):
                from zulong.tools.code_tools import IndexCodeFileTool
                self._index_tool_instance = IndexCodeFileTool()
                self._index_tool_instance.initialize()

            index_tool = self._index_tool_instance

            # 推断 MemoryGraph 中的 TASK 节点 ID
            task_mg_id = None
            if active_node_id:
                # active_node_id 来自 TaskGraph（短 ID），转为 MG 中的完整 ID
                from zulong.tools.task_tools import get_active_task_graph
                tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
                if tg:
                    candidate = f"task:{tg.id}/{active_node_id}"
                    if mg.has_node(candidate):
                        task_mg_id = candidate
                    else:
                        # 退化：搜索含 active_node_id 后缀的 TASK 节点
                        for nd in mg.get_nodes_by_type(NodeType.TASK):
                            nid = getattr(nd, "node_id", "")
                            if nid.endswith(active_node_id):
                                task_mg_id = nid
                                break

            for fp in code_files:
                self._index_and_anchor_file(fp, mg, adapter, index_tool, task_mg_id)

        except Exception as e:
            logger.debug(f"[IDEFCRunner] _auto_anchor_on_write 异常（不影响主流程）: {e}")

    def _index_and_anchor_file(
        self, file_path: str, mg, adapter, index_tool, task_mg_id: Optional[str]
    ) -> None:
        """对单个文件执行 CRG 索引 + 自动锚定边"""
        import os
        import hashlib
        from pathlib import Path
        from zulong.memory.memory_graph import NodeType, EdgeType

        # 读取文件内容
        source_content = ""
        try:
            candidates = [Path(file_path), Path(".") / file_path]
            # 也尝试工作区路径
            if hasattr(self, 'cwd') and self.cwd:
                candidates.insert(0, Path(self.cwd) / file_path)
            for p in candidates:
                if p.exists():
                    source_content = p.read_text(encoding="utf-8", errors="replace")
                    break
        except Exception:
            pass

        if not source_content:
            return

        # MD5 去重：内容未变则跳过
        content_hash = hashlib.md5(
            source_content.encode("utf-8", errors="replace")).hexdigest()
        if index_tool._indexed_hashes.get(file_path) == content_hash:
            return

        # 执行 Tree-sitter 解析 + 增量同步
        try:
            from zulong.code.ast_parser import ASTParser
            from zulong.code.graph_builder import CodeGraphBuilder, CodeEdge, ext_to_lang

            ext = os.path.splitext(file_path)[1]
            lang = ext_to_lang(ext)
            if not lang:
                return

            parser = ASTParser(lang)
            if not parser.available:
                return

            source_bytes = source_content.encode("utf-8", errors="replace")
            result = parser.parse_source(source_bytes, file_path)
            if result.parse_error:
                return

            for sym in result.symbols:
                sym.file_path = file_path

            # 构建边（含跨文件）
            edges = CodeGraphBuilder._build_edges_for_file(result)

            local_sym_names = {s.name for s in result.symbols}
            local_sym_names.update(s.qualified_name for s in result.symbols)
            local_node_ids = {s.node_id for s in result.symbols}

            global_sym_index = {}
            for node in mg.get_nodes_by_type(NodeType.CODE_SYMBOL):
                nid = getattr(node, "node_id", "")
                label = getattr(node, "label", "")
                if not nid or not label:
                    continue
                global_sym_index[label] = nid
                short = label.rsplit(".", 1)[-1]
                if short not in global_sym_index:
                    global_sym_index[short] = nid

            file_node_id = f"file:{file_path}"
            for imp in result.imports:
                if imp.is_from and imp.names:
                    for name in imp.names:
                        target_id = global_sym_index.get(name)
                        if target_id and target_id not in local_node_ids:
                            edges.append(CodeEdge(
                                source_id=file_node_id,
                                target_id=target_id,
                                edge_type="imports",
                                metadata={"line": imp.line, "module": imp.module},
                            ))

            for call in result.calls:
                if call.callee in local_sym_names:
                    continue
                target_id = global_sym_index.get(call.callee)
                if target_id:
                    caller_id = None
                    for s in result.symbols:
                        if s.qualified_name == call.caller:
                            caller_id = s.node_id
                            break
                    if caller_id:
                        edges.append(CodeEdge(
                            source_id=caller_id,
                            target_id=target_id,
                            edge_type="calls",
                            metadata={"line": call.line, "cross_file": True},
                        ))

            # 增量同步到 MemoryGraph
            adapter.incremental_sync(mg, "file_updated", {
                "file_path": file_path,
                "symbols": result.symbols,
                "edges": edges,
                "content_hash": content_hash,
                "project_root": getattr(self, 'cwd', '') or '',
            })

            # 记录哈希
            index_tool._indexed_hashes[file_path] = content_hash

            # 广播 CRG 索引事件到 WEB 面板（双通道）
            crg_update_payload = {
                "file_path": file_path,
                "symbol_count": len(result.symbols),
                "edge_count": len(edges),
                "content_hash": content_hash,
            }
            _broadcast_sync("CRG_INDEX_UPDATE", crg_update_payload)
            try:
                from zulong.launcher.web_chat_router import _schedule_broadcast
                _schedule_broadcast({
                    "type": "CRG_INDEX_UPDATE",
                    "payload": crg_update_payload,
                })
            except Exception:
                pass

            # ── 自动锚定：TASK → CODE_SYMBOL 边 + CodeAnchor 记录 + TaskGraph d_edge ──
            if task_mg_id and mg.has_node(task_mg_id):
                anchored = 0
                anchor_ids = []
                try:
                    from zulong.memory.code_anchor import CodeAnchor, get_code_anchor_store, compute_content_hash, get_current_commit_sha
                    anchor_store = get_code_anchor_store()
                    commit_sha = get_current_commit_sha()
                except Exception:
                    anchor_store = None
                    commit_sha = None

                for sym in result.symbols:
                    code_node_id = sym.node_id
                    if mg.has_node(code_node_id):
                        mg.add_edge(
                            task_mg_id, code_node_id,
                            edge_type=EdgeType.REFERENCE,
                            weight=0.6,
                            metadata={
                                "relation": "auto_anchored",
                                "anchor_type": "implementation",
                                "source": "write_hook",
                            },
                        )
                        anchored += 1

                        # 创建 CodeAnchor 记录
                        if anchor_store:
                            try:
                                snippet = source_content.split("\n")
                                start = max(0, sym.start_line - 1)
                                end = min(len(snippet), sym.end_line)
                                preview = "\n".join(snippet[start:start + 3])
                                sym_content = "\n".join(snippet[start:end])
                                anchor = CodeAnchor(
                                    id=compute_content_hash(f"{file_path}:{sym.node_id}:{time.time()}")[:12],
                                    file_path=file_path,
                                    symbol=sym.name,
                                    line_start=sym.start_line,
                                    line_end=sym.end_line,
                                    commit_sha=commit_sha,
                                    content_hash=compute_content_hash(sym_content),
                                    anchor_type="implementation",
                                    snippet_preview=preview[:200],
                                    owner_ref=task_mg_id.replace("task:", "tg:", 1) if task_mg_id.startswith("task:") else task_mg_id,
                                )
                                anchor_store.add_anchor(anchor)
                                anchor_ids.append(anchor.id)
                            except Exception:
                                pass

                if anchored:
                    logger.info(
                        f"[IDEFCRunner] 自动锚定: {file_path} → "
                        f"TASK({task_mg_id}) ↔ {anchored} CODE_SYMBOL 节点, "
                        f"{len(anchor_ids)} CodeAnchor 记录"
                    )

                # 关联 TaskGraph 符号节点（d_edge）
                try:
                    from zulong.tools.task_tools import get_active_task_graph
                    tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
                    if tg and active_node_id:
                        from zulong.code.graph_builder import ext_to_lang as _ext_to_lang
                        proj_name = getattr(tg, '_project_name', '') or ''
                        if not proj_name:
                            import os
                            proj_name = os.path.basename(getattr(self, 'cwd', '') or '.')
                        for sym in result.symbols:
                            sym_tg_id = f"crg_{proj_name}/sym:{sym.node_id}"
                            if tg.get_node(sym_tg_id) and tg.get_node(active_node_id):
                                tg.add_d_edge(active_node_id, sym_tg_id, via=f"implements {sym.name}", cross=True)
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"[IDEFCRunner] _index_and_anchor_file({file_path}) 异常: {e}")

    def _run_loop(self, state: IDEFCState) -> IDEFCResult:
        # 设置 FC 循环运行状态为 True（禁止节点审查提交）
        try:
            from zulong.core.state_manager import state_manager
            state_manager.set_fc_loop_running(True)
        except Exception:
            pass
        
        while True:
            tr = self._check(state)
            if tr:
                # 清除 FC 循环运行状态
                try:
                    from zulong.core.state_manager import state_manager
                    state_manager.set_fc_loop_running(False)
                except Exception:
                    pass
                return self._finalize(state, tr)
            try:
                # FC 请求间隔：防止 API 被打满（跳过第一轮）
                if state.fc_turn > 1 and self._fc_request_interval > 0:
                    time.sleep(self._fc_request_interval)
                self._publish_fc_progress(state, "calling_model", f"turn={state.fc_turn}")
                tc_data, resp_content = self._call_model(state)
                if tc_data is None and resp_content is None:
                    if state.api_timeout_count >= 2:
                        self._publish_fc_progress(state, "api_error", "连续API错误终止")
                        # 清除 FC 循环运行状态
                        try:
                            from zulong.core.state_manager import state_manager
                            state_manager.set_fc_loop_running(False)
                        except Exception:
                            pass
                        return self._finalize(state, "api_error")
                    time.sleep(2)  # API 错误后短暂退避
                    continue
                # 成功获得模型响应，重置连续错误计数
                state.loop_error_count = 0
                if tc_data:
                    tc_data = self._apply_tool_call_budget_ide(state, tc_data)
                    if not tc_data:
                        self._log_fc_decision_path(
                            state,
                            path="tool_budget_exhausted",
                            tool_calls=[],
                            response_content=resp_content or "",
                            root_cause="context_pressure",
                            tool_budget=state.tool_call_budget,
                            tool_calls_used=state.tool_calls_used,
                        )
                        continue
                    # 模型调用了工具 → 重置独白计数器
                    state.consecutive_text_only_count = 0
                    self._log_fc_decision_path(
                        state,
                        path="tool_calls_dispatch",
                        tool_calls=tc_data,
                        response_content=resp_content or "",
                        tool_names=[tc["function"]["name"] for tc in tc_data],
                        tool_call_count=len(tc_data),
                    )
                    self._publish_fc_progress(state, "exec_tools", f"{len(tc_data)} tool calls")
                    remote = self._exec_tools(state, tc_data, resp_content)
                    self._update_cb_recovery_progress(
                        state,
                        [tc["function"]["name"] for tc in tc_data],
                        state.tool_definitions,
                    )
                    self._update_pressure_recovery_progress(
                        state,
                        [tc["function"]["name"] for tc in tc_data],
                        state.tool_definitions,
                    )
                    if remote:
                        self._publish_fc_progress(state, "pause_for_remote", f"{len(remote)} remote tools")
                        # 注意：暂停不清除状态，恢复后会继续运行
                        return self._pause_for_remote(state, remote)
                    continue
                # 模型纯文本回复（无工具调用）→ 递增独白计数器
                state.consecutive_text_only_count += 1
                self._log_fc_decision_path(
                    state,
                    path="no_tool_call_enter_eval",
                    tool_calls=[],
                    response_content=resp_content or "",
                    response_length=len(resp_content or ""),
                    cb_force=state.cb_force_no_tools,
                    null_response_count=state.null_response_count,
                )
                verdict = self._eval_response(state, resp_content or "")
                self._log_fc_decision_path(
                    state,
                    path=f"no_tool_call_verdict_{verdict}",
                    tool_calls=[],
                    response_content=resp_content or "",
                    verdict=verdict,
                )
                if verdict == "done":
                    state.phase = "done"
                    
                    # 清除 FC 循环运行状态（允许节点审查提交）
                    try:
                        from zulong.core.state_manager import state_manager
                        state_manager.set_fc_loop_running(False)
                    except Exception:
                        pass
                    
                    # 🔥 修复：提取 submit_final_answer 内容
                    _final_answer = self._extract_final_answer(state)
                    if _final_answer:
                        state.final_answer = _final_answer

                    # 仅记录 FC 统计。最终答案写入/归档必须由统一 _finalize()
                    # 在完成质量门通过后执行。
                    try:
                        from zulong.tools.task_tools import get_active_task_graph
                        _tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
                        if _tg and hasattr(_tg, "metadata"):
                            _tg.metadata["total_turns"] = state.fc_turn
                            _tg.metadata["duration"] = time.time() - getattr(_tg, "created_at", time.time())
                    except Exception:
                        pass

                    # 所有最终写回/归档统一走 _finalize，避免局部 done 路径绕过质量门。
                    return self._finalize(state, "done")
                elif verdict == "cb_force":
                    state.cb_force_no_tools = True
                # verdict == "continue" → 继续循环
            except Exception as loop_err:
                logger.error(
                    f"[IDEFCRunner] 循环体异常 turn={state.fc_turn}: {loop_err}",
                    exc_info=True)
                state.loop_error_count += 1
                if state.loop_error_count >= 3:
                    logger.error("[IDEFCRunner] 连续 3 次循环异常，终止 FC")
                    # 清除 FC 循环运行状态
                    try:
                        from zulong.core.state_manager import state_manager
                        state_manager.set_fc_loop_running(False)
                    except Exception:
                        pass
                    return self._finalize(state, "loop_error")
                continue

    def _check(self, state: IDEFCState) -> str:
        """迭代守卫：软限制注入进度提示，硬限制触发弹性续期或终止

        返回值:
          ""           — 继续
          "interrupted" — 外部中断
          "checkpoint"  — 安全阀触发，强制终止
        """
        state.fc_turn += 1
        fc = state.fc_turn
        if getattr(self.engine, "_interrupt_flag", False):
            logger.info("[IDEFCRunner] 外部中断")
            return "interrupted"
        if fc % self._warning_interval == 0:
            logger.info(f"[IDEFCRunner] 进度: {fc}/{self._hard_limit}")
        
        # 重复工具调用死循环检测（从 FCRunner 继承并适配 IDEFCState）
        self._detect_duplicate_tool_loop_ide(state)
        
        # 周期性进度广播（每 _progress_report_interval 轮，独立于 hard_limit）
        # 仅推送到 Web 仪表盘，不注入消息、不影响 FC 循环控制流
        if (fc > 1
                and self._progress_report_interval > 0
                and fc % self._progress_report_interval == 0
                and (not self._step_limits_enabled or fc < self._hard_limit)):
            self._broadcast_periodic_progress(state)
        if self._step_limits_enabled and fc > self._soft_limit and fc % self._warning_interval == 1:
            # 软限制：注入进度提醒到消息列表，引导 LLM 收敛
            report = self._build_progress_hint(state)
            logger.warning(f"[IDEFCRunner] 超软限制 ({self._soft_limit}), 注入进度提示")
            hint_msg = {"role": "system", "content": report}
            state.messages.append(hint_msg)
            if self._attn_window:
                # 独立 group_id：避免被 None 组膨胀后整体淘汰
                gid = self._attn_window.new_tool_group()
                self._attn_window.register_message(hint_msg, turn=fc, group_id=gid)
        if self._step_limits_enabled and fc >= self._hard_limit:
            # 生成结构化进度报告
            progress = self._generate_progress_report(state)
            state.progress_reports.append(progress)
            state.last_report_turn = fc
            # 安全阀: 连续报告无进展 → 强制终止
            if self._is_progress_stalled(state):
                logger.warning(
                    f"[IDEFCRunner] 安全阀: 连续 {self._max_reports_before_force_stop} "
                    f"次报告无进展，强制终止"
                )
                self._save_runner_state()
                return "checkpoint"
            # 自动续期（无次数上限，只要有进展就持续续期）
            if not self._auto_continue:
                logger.warning(
                    f"[IDEFCRunner] 到达硬限制 ({self._hard_limit}), "
                    f"auto_continue=off，终止"
                )
                self._save_runner_state()
                return "checkpoint"
            # 弹性预算续期
            state.auto_continue_count += 1
            old_limit = self._hard_limit
            self._hard_limit += self._progress_report_interval
            logger.info(
                f"[IDEFCRunner] 弹性续期 #{state.auto_continue_count}: "
                f"硬限制 {old_limit} → {self._hard_limit}"
            )
            # 注入进度报告到消息列表，让 LLM 知道当前状态
            renewal_msg = {
                "role": "system",
                "content": (
                    f"[进度报告 #{state.auto_continue_count}] "
                    f"已执行 {fc} 步，预算已自动续期至 {self._hard_limit} 步。"
                    f"已完成 {progress.get('completed_count', 0)} 个节点，"
                    f"进行中 {progress.get('in_progress_count', 0)} 个，"
                    f"待处理 {progress.get('pending_count', 0)} 个。"
                    f"请继续推进任务。"
                ),
            }
            state.messages.append(renewal_msg)
            if self._attn_window:
                # 取消上一次续期消息的 pinned（只保留最新一条 pinned）
                self._unpin_old_renewals()
                # 独立 group_id + pinned：续期指令是 FC 继续运转的核心信号
                gid = self._attn_window.new_tool_group()
                self._attn_window.register_message(
                    renewal_msg, turn=fc, group_id=gid, pinned=True)
        return ""

    def _detect_duplicate_tool_loop_ide(self, state: IDEFCState) -> None:
        """重复工具调用死循环检测（IDEFCState 适配版）

        检查最近几轮中是否连续调用相同的工具且参数相同。
        如果检测到死循环，注入 CB 强制收敛信号。
        """
        fc = state.fc_turn
        if len(state.messages) < 6 or fc <= 5:
            return

        last_tool_calls = []
        for msg in reversed(state.messages[-6:]):
            tool_calls = msg.get("tool_calls", [])
            if tool_calls and len(tool_calls) > 0:
                tc = tool_calls[0]["function"]
                last_tool_calls.append({
                    "name": tc["name"],
                    "args": tc.get("arguments", ""),
                })

        if len(last_tool_calls) >= self._DUPLICATE_TOOL_CHECK_TURNS:
            tool_names = [tc["name"] for tc in last_tool_calls]
            tool_args = [tc["args"] for tc in last_tool_calls]

            if len(set(tool_names)) == 1 and len(set(tool_args)) == 1:
                logger.warning(
                    f"[IDEFCRunner] 检测到死循环: "
                    f"连续{len(last_tool_calls)}轮调用 {tool_names[0]} 且参数相同"
                )
                state.cb_force_no_tools = True
                state.cb_recovery_stage = "restricted_recovery"
                state.cb_recovery_note_saved = False
                state.cb_recovery_attention_switched = False
                cm = {
                    "role": "user",
                    "content": (
                        f"[系统警告] 检测到重复工具调用循环（{tool_names[0]}），"
                        f"请先把当前证据、未完成项、失败原因写入便签并切换注意力；"
                        f"如果仍无法继续，只能输出 partial/blocked summary，不得伪完成。"
                    ),
                }
                state.messages.append(cm)
                if self._attn_window:
                    self._attn_window.register_message(cm, turn=fc)

    def _build_progress_hint(self, state: IDEFCState) -> str:
        """构建进度提示，注入到 LLM 上下文（中性通报，不催促结束）"""
        fc = state.fc_turn
        hint = f"[系统进度通报] 当前已执行 {fc} 步。"
        # 附加任务图进度（如果有）
        from zulong.tools.task_tools import get_active_task_graph
        tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
        if tg:
            all_nodes = [n for n in tg.nodes if n.id != "req"]
            done = sum(1 for n in all_nodes if n.status in ("completed", "skipped"))
            wip = sum(1 for n in all_nodes if n.status == "in_progress")
            todo = sum(1 for n in all_nodes if n.status in ("pending", ""))
            hint += f" 任务进度: {done} 已完成, {wip} 进行中, {todo} 待处理。"
        return hint

    def _unpin_old_renewals(self) -> None:
        """取消旧续期消息的 pinned 状态，避免 pinned 累积膨胀

        只保留最新一条续期消息为 pinned，旧的降级为普通消息参与权重淘汰。
        """
        if not self._attn_window:
            return
        _RENEWAL_PREFIX = "[进度报告 #"
        for env in self._attn_window.envelopes:
            if (env.is_pinned
                    and env.msg.get("role") == "system"
                    and isinstance(env.msg.get("content"), str)
                    and env.msg["content"].startswith(_RENEWAL_PREFIX)):
                env.is_pinned = False

    def _get_progress_snapshot(self) -> dict:
        """获取任务图谱进度快照（轻量，用于每轮 status_update）"""
        try:
            from zulong.tools.task_tools import get_active_task_graph
            tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
            if not tg:
                return {}
            all_nodes = [n for n in tg.nodes if n.id != "req"]
            return {
                "total_nodes": len(all_nodes),
                "completed_count": sum(1 for n in all_nodes if n.status in ("completed", "skipped")),
                "in_progress_count": sum(1 for n in all_nodes if n.status == "in_progress"),
                "pending_count": sum(1 for n in all_nodes if n.status in ("pending", "")),
            }
        except Exception:
            return {}

    def _broadcast_periodic_progress(self, state: IDEFCState) -> None:
        """周期性进度广播（不触发续期，仅通知 Web 仪表盘当前状态）"""
        from zulong.tools.task_tools import get_active_task_graph
        # 刷新 Gatekeeper 空闲计时器，防止 FC 循环执行中被误判空闲挂起
        try:
            from zulong.l1b.scheduler_gatekeeper import gatekeeper
            if gatekeeper:
                gatekeeper.touch_idle_timer()
        except Exception:
            pass
        tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
        report = {"turn": state.fc_turn, "type": "periodic"}
        if tg:
            all_nodes = [n for n in tg.nodes if n.id != "req"]
            report["total_nodes"] = len(all_nodes)
            report["completed_count"] = sum(
                1 for n in all_nodes if n.status in ("completed", "skipped"))
            report["in_progress_count"] = sum(
                1 for n in all_nodes if n.status == "in_progress")
            report["pending_count"] = sum(
                1 for n in all_nodes if n.status in ("pending", ""))
        _broadcast_sync("PROGRESS_REPORT", {
            "session_id": self.session.session_id,
            "turn": state.fc_turn,
            "report": report,
            "type": "periodic",
        })
        # 同时推送到 IDE 端（send_callback）
        try:
            import asyncio
            cb = getattr(self, '_ide_send_callback', None)
            if cb:
                loop = asyncio.get_event_loop()
                asyncio.run_coroutine_threadsafe(
                    cb("status_update", {
                        "turn": state.fc_turn,
                        "phase": "running",
                        "progress": report,
                    }), loop)
        except Exception:
            pass

    def _generate_progress_report(self, state: IDEFCState) -> Dict:
        """生成结构化进度报告，用于弹性续期决策和 Web 推送"""
        from zulong.tools.task_tools import get_active_task_graph
        tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
        report = {
            "turn": state.fc_turn,
            "elapsed_turns": state.fc_turn,
            "completed_count": 0,
            "in_progress_count": 0,
            "pending_count": 0,
            "total_nodes": 0,
            "completed": [],
            "in_progress": [],
            "pending": [],
        }
        if tg:
            # 统计全部节点（排除 req 根节点），不限于叶节点
            all_nodes = [n for n in tg.nodes if n.id != "req"]
            report["total_nodes"] = len(all_nodes)
            for node in all_nodes:
                entry = {"id": node.id, "label": node.label}
                if node.status in ("completed", "skipped"):
                    report["completed_count"] += 1
                    if node.result:
                        entry["result_preview"] = node.result[:100]
                    report["completed"].append(entry)
                elif node.status == "in_progress":
                    report["in_progress_count"] += 1
                    report["in_progress"].append(entry)
                else:
                    report["pending_count"] += 1
                    report["pending"].append(entry)
        logger.info(
            f"[IDEFCRunner] 进度报告: turn={state.fc_turn}, "
            f"total={report['total_nodes']}, "
            f"done={report['completed_count']}, "
            f"wip={report['in_progress_count']}, "
            f"todo={report['pending_count']}"
        )
        # 同步推送到 Web（确保每次报告用户都能看到）
        _broadcast_sync("PROGRESS_REPORT", {
            "session_id": self.session.session_id,
            "turn": state.fc_turn,
            "report": report,
            "auto_continue_count": state.auto_continue_count,
        })
        return report

    def _is_progress_stalled(self, state: IDEFCState) -> bool:
        """检查连续进度报告是否停滞

        停滞条件：最近 N 次报告中 completed_count 和 total_nodes 都没有增长
        （即既没完成节点、也没创建新节点 → 真正的死循环）
        
        🔥 P1修复：增强检测逻辑
        - completed_count增加 → 有进展
        - in_progress_count变化 → 有变化
        - 消息长度增加 → 有新信息
        """
        reports = state.progress_reports
        n = self._max_reports_before_force_stop
        if len(reports) < n:
            return False
        recent = reports[-n:]
        
        # 检查1：completed_count是否增加
        completed_counts = [r.get("completed_count", 0) for r in recent]
        if completed_counts[-1] > completed_counts[0]:
            return False  # 有进展
        
        # 检查2：total_nodes是否增加（新节点创建）
        total_counts = [r.get("total_nodes", 0) for r in recent]
        if total_counts[-1] > total_counts[0]:
            return False  # 有新节点
        
        # 检查3：in_progress_count是否变化（节点状态流转）
        wip_counts = [r.get("in_progress_count", 0) for r in recent]
        if len(set(wip_counts)) > 1:
            return False  # 有状态变化
        
        # 检查4：消息长度是否增加（有新信息注入）
        if hasattr(state, 'last_report_msg_count'):
            if len(state.messages) > state.last_report_msg_count:
                state.last_report_msg_count = len(state.messages)
                return False  # 有新消息
        else:
            state.last_report_msg_count = len(state.messages)
        
        # 所有指标都无变化 → 停滞
        return True

    def _build_dynamic_attention_context_message(
        self,
        state: IDEFCState,
        base_messages: List[Dict],
        *,
        fc: int,
    ) -> Tuple[List[Dict], Optional[Dict]]:
        """Build and inject the per-call active context defined by TSD v2.9.24.

        This is not a backing_pool and is not pinned into the conversation.  It
        reconstructs the current necessary context from TaskGraph / MemoryGraph /
        BFS for this one model call, then lets AttentionWindow remain only the
        low-level safety fallback.
        """
        if not self._attn_window:
            return base_messages, None
        try:
            task_graph = self._get_current_task_graph()
            try:
                from zulong.memory.memory_graph import get_memory_graph
                memory_graph = get_memory_graph()
            except Exception:
                memory_graph = getattr(self._attn_window, "memory_graph", None)

            ratio = float(
                getattr(
                    self._attn_window,
                    "trigger_context_pressure_ratio",
                    getattr(self._attn_window, "context_pressure_ratio", 0.0),
                )
                or 0.0
            )
            mode_value = getattr(getattr(self._attn_window, "mode", None), "value", "global")
            pressure_stage = str(getattr(state, "pressure_stage", "") or "")
            pressure_context = getattr(state, "pressure_attention_context", {}) or {}
            include_navigation = pressure_stage in {"yellow_guidance", "restricted_recovery"}
            if not include_navigation and str(mode_value).lower() != "global":
                include_navigation = True
            trigger_reason = pressure_stage or "model_call"
            if pressure_context.get("tier"):
                trigger_reason = f"pressure_{pressure_context.get('tier')}_{pressure_stage or 'active'}"
            query_text = str(getattr(state, "user_input_text", "") or "")
            if not query_text:
                for msg in reversed(list(getattr(state, "messages", []) or [])):
                    if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                        query_text = str(msg.get("content") or "")
                        break

            uncovered_node_ids: List[str] = []
            try:
                if task_graph:
                    coverage = self._compute_node_coverage(state, task_graph)
                    uncovered_node_ids = list(coverage.get("uncovered_in_progress") or [])
            except Exception:
                uncovered_node_ids = []

            bundle = build_attention_context_bundle(
                mode=mode_value,
                query_text=query_text,
                attention_window=self._attn_window,
                task_graph=task_graph,
                memory_graph=memory_graph,
                pressure_percent=ratio * 100.0,
                trigger_reason=trigger_reason,
                include_navigation_map=include_navigation,
                navigation_reason=trigger_reason if include_navigation else "",
                uncovered_node_ids=uncovered_node_ids,
            )
            rendered_msg = render_attention_context_message(bundle)
            rendered_text = rendered_msg.get("content", "")
            if not rendered_text:
                return base_messages, None

            telemetry = dict(bundle.telemetry or {})
            telemetry.update({
                "pressure_percent": round(ratio * 100.0, 1),
                "pressure_stage": pressure_stage,
                "threshold_budget_tokens": getattr(self._attn_window, "threshold_budget_tokens", None),
                "active_context_rendered_chars": len(rendered_text),
            })
            plan_dict = bundle.plan.to_dict()
            state.attention_context_plan = plan_dict
            state.attention_context_telemetry = telemetry
            state.last_attention_context_rendered = rendered_text[:4000]
            state.last_attention_context_key = "%s|%s|%s|%s" % (
                plan_dict.get("mode"),
                plan_dict.get("focus_node_id"),
                trigger_reason,
                fc,
            )
            try:
                self._attn_window._last_active_context_extra_tokens = int(
                    telemetry.get("active_context_token_estimate") or 0
                )
            except Exception:
                pass
            logger.info(
                "[IDEFCRunner][AttentionContext] turn=%s mode=%s focus=%s memory=%s bfs=%s tokens=%s nav=%s",
                fc,
                plan_dict.get("mode"),
                plan_dict.get("focus_node_address") or plan_dict.get("focus_node_id"),
                telemetry.get("retrieved_memory_count"),
                telemetry.get("bfs_activated_count"),
                telemetry.get("active_context_token_estimate"),
                telemetry.get("navigation_map_injected"),
            )
            return [rendered_msg] + list(base_messages or []), telemetry
        except Exception as exc:
            logger.debug("[IDEFCRunner] ????????????: %s", exc)
            return base_messages, None

    def _call_model(self, state: IDEFCState) -> Tuple[Optional[List[Dict]], Optional[str]]:
        """LLM API 调用。返回 (tool_calls, content)。都为 None 表示超时。"""
        call_start = time.perf_counter()
        fc = state.fc_turn
        msgs = self._attn_window.apply_window() if self._attn_window else state.messages
        msgs, _attention_context_telemetry = self._build_dynamic_attention_context_message(state, msgs, fc=fc)
        extra_kw = self.engine._get_llm_extra_kwargs()
        # Qwen3 系列默认开启思维链（<think>），FC 模式下禁用以避免空 content
        eb = extra_kw.get("extra_body", {})
        eb["enable_thinking"] = False
        extra_kw["extra_body"] = eb
        kw: Dict[str, Any] = {
            "model": state.vllm_model_id, "messages": msgs,
            "max_tokens": state.response_max_tokens, "temperature": 0.3,
            "top_p": 0.85, "stream": True, **extra_kw,  # 启用流式模式
        }
        if state.tool_call_budget == 0 or (
            state.tool_call_budget is not None
            and int(state.tool_calls_used or 0) >= int(state.tool_call_budget)
        ):
            logger.info(
                "[IDEFCRunner] 工具预算已用尽: used=%s budget=%s，移除工具定义",
                state.tool_calls_used,
                state.tool_call_budget,
            )
        elif state.cb_force_no_tools:
            # CB RED 只有两轮口径：YELLOW 纠偏；RED 立即进入受限恢复。
            # RED 不再先保留普通收敛工具，立即进入受限恢复。
            state.cb_recovery_stage = "restricted_recovery"
            cb_retained = self._get_cb_recovery_tools(state.tool_definitions)
            if cb_retained:
                kw["tools"] = cb_retained
                next_tool = self._next_cb_recovery_tool(state, cb_retained)
                if next_tool:
                    kw["tool_choice"] = {
                        "type": "function",
                        "function": {"name": next_tool},
                    }
                else:
                    kw["tool_choice"] = "required"
                logger.warning(
                    f"[IDEFCRunner][CB] RED 进入受限恢复，"
                    f"仅保留便签/标签/记忆落盘+注意力切换能力工具 {len(cb_retained)} 个"
                )
            else:
                logger.warning("[IDEFCRunner][CB] RED 未找到受限恢复工具，回退纯文本")
        elif state.pressure_force_attention:
            # 压力 RED: 只保留注意力切换 + 便签/标签/记忆落盘能力，
            # 由 LLM 自主选择 GLOBAL/FOCUS/SINGLE_CHAIN 并按需保存现场。
            recovery_tools = self._get_cb_recovery_tools(state.tool_definitions)
            if recovery_tools:
                kw["tools"] = recovery_tools
                next_tool = self._next_pressure_recovery_tool(state, recovery_tools)
                kw["tool_choice"] = (
                    {"type": "function", "function": {"name": next_tool}}
                    if next_tool else "required"
                )
                logger.info(
                    "[IDEFCRunner][Pressure] 工具列表约束为注意力/便签/标签/记忆落盘能力工具 (%s个)，selection_context=%s",
                    len(recovery_tools),
                    getattr(state, "pressure_attention_context", {}) or {},
                )
            else:
                logger.warning("[IDEFCRunner][Pressure] 受限恢复工具不在 tool_definitions 中，回退正常模式")
                state.pressure_force_attention = False
                if state.tool_definitions:
                    kw["tools"] = state.tool_definitions
                    kw["tool_choice"] = "auto"
        elif state.tool_definitions:
            kw["tools"] = state.tool_definitions
            if state.force_first_tool:
                logger.debug("[IDEFCRunner] 忽略旧 force_first_tool 状态，保持 L2 tool_choice=auto")
                state.force_first_tool = False
            kw["tool_choice"] = "auto"
        future = self._model_executor.submit(
            lambda: self.engine.vllm_client.chat.completions.create(**kw))
        try:
            # 流式响应处理
            stream_response = future.result(timeout=self._fc_loop_timeout)
            logger.info(
                f"[IDEFCRunner] LLM流连接建立: turn={state.fc_turn}, "
                f"{(time.perf_counter() - call_start) * 1000:.1f}ms"
            )
            
            # 累积 token 并实时推送清洗后的文本
            full_content = ""
            sentence_buffer = ""
            sent_count = 0
            tool_calls_chunks = []  # 累积工具调用片段
            
            stream_start_time = time.time()
            stream_start_perf = time.perf_counter()
            last_heartbeat = stream_start_time
            heartbeat_interval = 10.0  # 每10秒输出一次心跳日志
            first_token_logged = False
            last_flush_perf = stream_start_perf
            first_phase_flush_chars = 24
            first_phase_flush_seconds = 1.5
            first_phase_flush_interval = 0.12
            finish_reason = ""
            usage_data: Any = None
            
            for chunk in stream_response:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    usage_data = chunk_usage
                if chunk.choices:
                    choice = chunk.choices[0]
                    choice_finish = getattr(choice, "finish_reason", None)
                    if choice_finish:
                        finish_reason = str(choice_finish)
                    # 累积文本内容
                    if choice.delta.content:
                        token = choice.delta.content
                        now_perf = time.perf_counter()
                        if not first_token_logged:
                            first_token_logged = True
                            logger.info(
                                f"[IDEFCRunner] LLM首token: turn={state.fc_turn}, "
                                f"{(now_perf - call_start) * 1000:.1f}ms"
                            )
                        full_content += token
                        sentence_buffer += token
                        
                        in_first_phase = (
                            sent_count == 0
                            and now_perf - stream_start_perf <= first_phase_flush_seconds
                        )
                        should_flush_fast = (
                            in_first_phase
                            and len(sentence_buffer) >= first_phase_flush_chars
                            and now_perf - last_flush_perf >= first_phase_flush_interval
                        )
                        # 检测句子边界（句号、问号、感叹号、换行）
                        if should_flush_fast or any(token.endswith(p) for p in ['。', '！', '？', '\n']):
                            # 清洗并推送当前句子
                            from zulong.utils.text_cleaner import clean_text_for_tts
                            cleaned = clean_text_for_tts(sentence_buffer)
                            if cleaned:
                                # 异步推送（使用安全发送方法）
                                cb = getattr(self, '_ide_send_callback', None)
                                if cb:
                                    self._send_message_safe(cb, "display_text", {
                                        "text": cleaned,
                                        "turn": state.fc_turn,
                                        "streaming": True,
                                        "sentence_index": sent_count
                                    })
                                sent_count += 1
                                sentence_buffer = ""  # 清空缓冲区
                                last_flush_perf = now_perf
                    
                    # 累积工具调用片段
                    if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                        tool_calls_chunks.extend(choice.delta.tool_calls)
                
                # 心跳日志：防止长时间无输出时看起来像卡死
                now = time.time()
                if now - last_heartbeat >= heartbeat_interval:
                    elapsed = int(now - stream_start_time)
                    logger.info(f"💓 [FC] 等待模型流式响应中... 已等待 {elapsed}s, 已接收 {len(full_content)} 字符")
                    last_heartbeat = now
            
            # 处理剩余的文本
            if sentence_buffer.strip():
                from zulong.utils.text_cleaner import clean_text_for_tts
                cleaned = clean_text_for_tts(sentence_buffer)
                if cleaned:
                    cb = getattr(self, '_ide_send_callback', None)
                    if cb:
                        self._send_message_safe(cb, "display_text", {
                            "text": cleaned,
                            "turn": state.fc_turn,
                            "streaming": True,
                            "sentence_index": sent_count
                        })
                    sent_count += 1
            
            # 推送完成标记
            cb = getattr(self, '_ide_send_callback', None)
            if cb:
                self._send_message_safe(cb, "display_text", {
                    "text": "",
                    "turn": state.fc_turn,
                    "streaming": False,
                    "complete": True
                })
            
            if sent_count > 0:
                logger.info(f"🌊 [FC] 流式推送完成：共 {sent_count} 个句子，总长度 {len(full_content)}")
            
            # 组装流式工具调用片段
            tc = None
            if tool_calls_chunks:
                # 按 index 分组并累积
                tc_map = {}  # {index: {id, type, function: {name, arguments}}}
                for tc_chunk in tool_calls_chunks:
                    idx = getattr(tc_chunk, 'index', 0)
                    if idx not in tc_map:
                        tc_map[idx] = {
                            'id': '',
                            'type': 'function',
                            'function': {'name': '', 'arguments': ''}
                        }
                    if hasattr(tc_chunk, 'id') and tc_chunk.id:
                        tc_map[idx]['id'] = tc_chunk.id
                    if hasattr(tc_chunk, 'function'):
                        if hasattr(tc_chunk.function, 'name') and tc_chunk.function.name:
                            tc_map[idx]['function']['name'] = tc_chunk.function.name
                        if hasattr(tc_chunk.function, 'arguments') and tc_chunk.function.arguments:
                            tc_map[idx]['function']['arguments'] += tc_chunk.function.arguments
                
                tc = list(tc_map.values())
                if tc:
                    logger.info(f"[IDEFCRunner] Turn {fc}: {len(tc)} 工具调用 (流式)")
                    for t in tc:
                        logger.info(
                            "[IDEFCRunner]   FC tool summary: %s",
                            _json.dumps(
                                _summarize_tool_call_for_log(t),
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                        )
            
            # 返回完整内容供后续处理
            # 注意：task_complete在run_loop_async中统一发送，避免重复
            rc = full_content
            
        except concurrent.futures.TimeoutError:
            future.cancel()
            state.api_timeout_count += 1
            logger.warning(f"[IDEFCRunner] Turn {fc} 超时, count={state.api_timeout_count}")
            return None, None
        except Exception as err:
            logger.error(f"[IDEFCRunner] Turn {fc} API 失败: {err}")
            # 检测 429 Rate Limit 错误：等待更长时间后重试，而非立即放弃
            err_str = str(err)
            is_rate_limit = "429" in err_str or "rate" in err_str.lower() or "TPM" in err_str
            if is_rate_limit:
                # 429 专用处理：等待 20 秒让 TPM 窗口恢复，然后重试一次
                wait_secs = 20
                logger.warning(
                    f"[IDEFCRunner] Turn {fc} 触发 429 限流，等待 {wait_secs}s 后重试...")
                time.sleep(wait_secs)
                try:
                    # 重试也使用流式
                    retry_stream = self.engine.vllm_client.chat.completions.create(**kw)
                    full_content = ""
                    sentence_buffer = ""
                    sent_count = 0
                    retry_tool_calls = []  # 累积工具调用片段
                    retry_finish_reason = ""
                    retry_usage_data: Any = None
                    
                    for chunk in retry_stream:
                        chunk_usage = getattr(chunk, "usage", None)
                        if chunk_usage:
                            retry_usage_data = chunk_usage
                        if chunk.choices:
                            choice = chunk.choices[0]
                            choice_finish = getattr(choice, "finish_reason", None)
                            if choice_finish:
                                retry_finish_reason = str(choice_finish)
                            if choice.delta.content:
                                token = choice.delta.content
                                full_content += token
                                sentence_buffer += token
                                
                                if any(token.endswith(p) for p in ['。', '！', '？', '\n']):
                                    from zulong.utils.text_cleaner import clean_text_for_tts
                                    cleaned = clean_text_for_tts(sentence_buffer)
                                    if cleaned:
                                        cb = getattr(self, '_ide_send_callback', None)
                                        if cb:
                                            self._send_message_safe(cb, "display_text", {
                                                "text": cleaned,
                                                "turn": state.fc_turn,
                                                "streaming": True,
                                                "sentence_index": sent_count
                                            })
                                        sent_count += 1
                                        sentence_buffer = ""
                            
                            # 累积工具调用片段
                            if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                                retry_tool_calls.extend(choice.delta.tool_calls)
                    
                    if sentence_buffer.strip():
                        from zulong.utils.text_cleaner import clean_text_for_tts
                        cleaned = clean_text_for_tts(sentence_buffer)
                        if cleaned:
                            cb = getattr(self, '_ide_send_callback', None)
                            if cb:
                                self._send_message_safe(cb, "display_text", {
                                    "text": cleaned,
                                    "turn": state.fc_turn,
                                    "streaming": True,
                                    "sentence_index": sent_count
                                })
                            sent_count += 1
                    
                    state.api_timeout_count = 0
                    rc = full_content
                    
                    # 组装流式工具调用片段
                    tc = None
                    if retry_tool_calls:
                        tc_map = {}
                        for tc_chunk in retry_tool_calls:
                            idx = getattr(tc_chunk, 'index', 0)
                            if idx not in tc_map:
                                tc_map[idx] = {
                                    'id': '',
                                    'type': 'function',
                                    'function': {'name': '', 'arguments': ''}
                                }
                            if hasattr(tc_chunk, 'id') and tc_chunk.id:
                                tc_map[idx]['id'] = tc_chunk.id
                            if hasattr(tc_chunk, 'function'):
                                if hasattr(tc_chunk.function, 'name') and tc_chunk.function.name:
                                    tc_map[idx]['function']['name'] = tc_chunk.function.name
                                if hasattr(tc_chunk.function, 'arguments') and tc_chunk.function.arguments:
                                    tc_map[idx]['function']['arguments'] += tc_chunk.function.arguments
                        
                        tc = list(tc_map.values())
                        if tc:
                            logger.info(f"[IDEFCRunner] Turn {fc}: {len(tc)} 工具调用 (429重试成功)")
                            for t in tc:
                                logger.info(
                                    "[IDEFCRunner]   FC tool summary: %s",
                                    _json.dumps(
                                        _summarize_tool_call_for_log(t),
                                        ensure_ascii=False,
                                        sort_keys=True,
                                    ),
                                )
                        else:
                            logger.info(f"[IDEFCRunner] Turn {fc}: 文本回复 len={len(rc)} (429重试成功)")
                    else:
                        logger.info(f"[IDEFCRunner] Turn {fc}: 文本回复 len={len(rc)} (429重试成功)")
                    self._record_model_raw_output(
                        state,
                        fc,
                        raw_content=rc,
                        final_content=rc,
                        tool_calls=tc,
                        finish_reason=retry_finish_reason,
                        usage=retry_usage_data,
                        source="stream_retry_429",
                    )
                    return tc, rc
                except Exception as retry_err:
                    logger.warning(f"[IDEFCRunner] Turn {fc} 429重试仍失败: {retry_err}")
                    # 继续走原有的备用模型逻辑
            # 追踪连续 API 错误（含 429 rate limit）
            state.api_timeout_count += 1
            if state.api_timeout_count >= 3:
                logger.error(
                    f"[IDEFCRunner] 连续 {state.api_timeout_count} 次 API 错误，触发退出")
                return None, None
            try:
                from zulong.models.container import LLM_MODEL_ID_BACKUP
                if self.engine.backup_client and LLM_MODEL_ID_BACKUP:
                    br = self.engine.backup_client.chat.completions.create(
                        model=LLM_MODEL_ID_BACKUP, messages=state.messages,
                        max_tokens=state.response_max_tokens, temperature=0.3,
                        stream=False, **self.engine._get_llm_extra_kwargs())
                    c = br.choices[0].message.content or ""
                    state.last_response_content = c
                    self._record_model_raw_output(
                        state,
                        fc,
                        raw_content=c,
                        final_content=c,
                        tool_calls=None,
                        finish_reason=getattr(br.choices[0], "finish_reason", "") or "",
                        usage=getattr(br, "usage", None),
                        source="backup_after_primary_error",
                    )
                    return None, c
            except Exception as be:
                logger.warning(f"[IDEFCRunner] 备用也失败: {be}")
            # 主+备均失败：注入 API 错误提示消息，让循环上层处理
            state.api_timeout_count += 1
            logger.error(
                f"[IDEFCRunner] 主+备均失败，连续 {state.api_timeout_count} 次错误，触发退出")
            return None, None
        # 流式响应已在上面处理，rc 和 tc 已经设置
        # 这里只需要处理 XML 工具调用回退
        
        # 检查是否包含 XML 格式的工具调用
        raw_rc_before_xml = rc
        xml_tc = self.translator.parse_xml_tool_calls(rc)
        if xml_tc:
            logger.info(f"[IDEFCRunner] Turn {fc}: {len(xml_tc)} 工具调用 (XML 回退解析)")
            for xt in xml_tc:
                logger.info(
                    "[IDEFCRunner]   XML tool summary: %s",
                    _json.dumps(
                        _summarize_tool_call_for_log(xt),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )
            tc = xml_tc
            # 从内容中移除 XML 工具标签，保留前置文本
            rc = self._strip_xml_tool_tags(rc)
        else:
            # 即使未解析出工具调用，文本中仍可能含有 XML 残留片段
            # （LLM 输出了不完整/非标准的 XML 工具调用）
            rc = self._strip_xml_tool_tags(rc)
            if rc:
                logger.info(f"[IDEFCRunner] Turn {fc}: 文本回复 len={len(rc)}")
                # 调试：记录文本前 300 字符
                snippet = rc[:300].replace('\n', '\\n')
                logger.info(f"[IDEFCRunner] Turn {fc} 文本预览: {snippet}")
        self._record_model_raw_output(
            state,
            fc,
            raw_content=raw_rc_before_xml,
            final_content=rc,
            tool_calls=tc,
            finish_reason=finish_reason,
            usage=usage_data,
            source="stream",
        )
        
        return tc, rc

    @staticmethod
    @staticmethod
    def _extract_final_answer(state: IDEFCState) -> Optional[str]:
        """从消息历史中提取 submit_final_answer 的 answer 内容。

        如果 FC 循环中调用了 submit_final_answer，从最近的 assistant
        tool_call 消息中提取 answer 参数作为最终回答内容。
        返回 None 表示未调用 submit_final_answer。
        """
        for msg in reversed(state.messages):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function", {}).get("name", "")
                if fn == "submit_final_answer":
                    import json as _json
                    try:
                        args_str = tc.get("function", {}).get("arguments", "{}")
                        args = _json.loads(args_str) if isinstance(args_str, str) else args_str
                        answer = args.get("answer", "")
                        if answer:
                            return answer
                    except Exception:
                        pass
        return None

    @staticmethod
    def _get_cb_retained_tools(tool_definitions: List[Dict]) -> List[Dict]:
        """CB RED 时保留的工具子集

        保留记忆恢复和最终提交类工具，使模型在被限制时仍可：
        1. 通过 recall_memory 恢复被淘汰的上下文
        2. 通过 submit_final_answer 生成最终回复
        3. 通过 task_mark_status 标记当前进度
        """
        _CB_RETAINED_NAMES = {
            "recall_memory", "read_memory_node",
            "submit_final_answer",
            "task_mark_status", "task_view_overview",
        }
        retained = []
        for td in tool_definitions:
            fn = td.get("function", {}).get("name", "")
            if fn in _CB_RETAINED_NAMES:
                retained.append(td)
        return retained

    @staticmethod
    def _tool_capabilities(tool_definition: Dict[str, Any]) -> set[str]:
        """Infer coarse tool capabilities without binding policies to one name."""
        return set(tool_capabilities(tool_definition))

    @classmethod
    def _tool_capability_map(cls, tool_definitions: Optional[List[Dict[str, Any]]]) -> Dict[str, set[str]]:
        capability_by_name: Dict[str, set[str]] = {}
        for td in tool_definitions or []:
            fn = str(td.get("function", {}).get("name", "") or "").strip()
            if fn:
                capability_by_name[fn] = cls._tool_capabilities(td)
        return capability_by_name

    @classmethod
    def _tool_result_has_capability(
        cls,
        item: Dict[str, Any],
        capability: str,
        capability_by_name: Optional[Dict[str, set[str]]] = None,
    ) -> bool:
        name = str(item.get("tool_name", "") or "").strip()
        if not name:
            return False
        caps = (capability_by_name or {}).get(name, set())
        if capability in caps:
            return True
        arguments = item.get("arguments", {}) or {}
        if isinstance(arguments, dict):
            arg_names = {str(k).lower() for k in arguments.keys()}
            if (
                capability == "file_write"
                and bool({"path", "file_path", "target_path"} & arg_names)
                and bool({"content", "diff", "replacement"} & arg_names)
            ):
                return True
            if capability == "verification" and bool({"command", "regex", "query", "url"} & arg_names):
                return True
            if capability == "attention_switch" and bool({"mode", "direction", "target_node_id"} & arg_names):
                result_text = cls._tool_result_text(item).lower()
                return "attention" in result_text or "注意力" in result_text
            if capability == "note_anchor" and bool({"content", "label", "entries"} & arg_names):
                result_text = cls._tool_result_text(item).lower()
                return any(token in result_text for token in ("note", "memory", "便签", "笔记", "记忆"))
        return False

    @classmethod
    def _tool_has_capability(cls, tool_definition: Dict[str, Any], capability: str) -> bool:
        return capability in cls._tool_capabilities(tool_definition)

    @classmethod
    def _get_cb_recovery_tools(cls, tool_definitions: List[Dict]) -> List[Dict]:
        """CB RED 受限恢复工具。

        TSD 对齐口径：
        - 不再直接硬中断任务；
        - 仅保留“便签/标签/记忆落盘能力”和“注意力切换能力”；
        - 按工具能力类别筛选；具体工具名仅作为当前工具描述不足时的兼容映射。
        """
        retained = []
        for td in tool_definitions:
            caps = cls._tool_capabilities(td)
            if caps & cls._CB_RESTRICTED_RECOVERY_CAPABILITIES:
                if caps & cls._CB_RESTRICTED_EXCLUDED_CAPABILITIES:
                    continue
                retained.append(td)
        return retained

    @classmethod
    def _first_tool_with_capability(cls, tool_definitions: List[Dict], capability: str) -> str:
        for td in tool_definitions or []:
            if capability in cls._tool_capabilities(td):
                name = str(td.get("function", {}).get("name", "") or "").strip()
                if name:
                    return name
        return ""

    @classmethod
    def _first_recovery_landing_tool(cls, tool_definitions: List[Dict]) -> str:
        for capability in ("note_anchor", "memory_persist", "tag_anchor"):
            name = cls._first_tool_with_capability(tool_definitions, capability)
            if name:
                return name
        return ""

    @classmethod
    def _next_cb_recovery_tool(cls, state: IDEFCState, tool_definitions: List[Dict]) -> str:
        if not getattr(state, "cb_recovery_note_saved", False):
            return cls._first_recovery_landing_tool(tool_definitions)
        if not getattr(state, "cb_recovery_attention_switched", False):
            return cls._first_tool_with_capability(tool_definitions, "attention_switch")
        return ""

    @classmethod
    def _next_pressure_recovery_tool(cls, state: IDEFCState, tool_definitions: List[Dict]) -> str:
        if (
            getattr(state, "pressure_recovery_requires_note", True)
            and not getattr(state, "pressure_recovery_note_saved", False)
        ):
            return cls._first_recovery_landing_tool(tool_definitions)
        if (
            getattr(state, "pressure_recovery_requires_attention", True)
            and not getattr(state, "pressure_recovery_attention_switched", False)
        ):
            return cls._first_tool_with_capability(tool_definitions, "attention_switch")
        return ""

    def _build_cb_red_control_message(self, state: IDEFCState, reason: str) -> Dict:
        """Build TSD-aligned CircuitBreaker RED control guidance.

        RED 触发后立即进入
        “便签/标签/记忆锚定 + 注意力重选”的受限恢复回路。
        """
        state.cb_recovery_stage = "restricted_recovery"
        state.cb_recovery_note_saved = False
        state.cb_recovery_attention_switched = False
        focus_node = ""
        try:
            focus_node = str(getattr(self._attn_window, "_current_node_id", "") or "")
        except Exception:
            focus_node = ""
        content = (
            f"[Circuit Breaker RED 受限恢复] {reason}\n"
            "不要直接最终收束，也不要继续普通执行/搜索/写文件/命令/验证工具。\n"
            "请按顺序完成两步：\n"
            "1) 调用便签/标签/记忆落盘能力，把当前可用证据、未完成项、失败原因、下一步建议保存下来；"
            f"内容必须关联当前焦点节点 {focus_node or '当前任务节点'}。\n"
            "2) 调用注意力切换能力，基于便签和当前任务状态选择 GLOBAL / FOCUS / SINGLE_CHAIN 之一，"
            "并说明需要注入哪些上下文、暂排哪些上下文。\n"
            "本轮普通执行工具已被收走，只保留便签/标签/记忆落盘和注意力切换能力。"
        )
        return internal_control_message(content)

    @classmethod
    def _update_cb_recovery_progress(
        cls,
        state: IDEFCState,
        tool_names: List[str],
        tool_definitions: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Track RED restricted recovery progress and release the hard tool gate once complete."""
        if getattr(state, "cb_recovery_stage", "") not in {"restricted_recovery", "note_attention"}:
            return
        names = {str(name or "").strip() for name in tool_names if name}
        capability_by_name = cls._tool_capability_map(tool_definitions)
        if any(
            capability_by_name.get(name, set()) & {"note_anchor", "memory_persist", "tag_anchor"}
            for name in names
        ):
            state.cb_recovery_note_saved = True
        if any(
            "attention_switch" in capability_by_name.get(name, set())
            for name in names
        ):
            state.cb_recovery_attention_switched = True
        if state.cb_recovery_note_saved and state.cb_recovery_attention_switched:
            state.cb_force_no_tools = False
            state.cb_tool_streak = 0
            state.cb_recovery_stage = ""

    @classmethod
    def _update_pressure_recovery_progress(
        cls,
        state: IDEFCState,
        tool_names: List[str],
        tool_definitions: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Track context-pressure restricted recovery progress."""
        if getattr(state, "pressure_stage", "") != "restricted_recovery":
            return
        names = {str(name or "").strip() for name in tool_names if name}
        if not names:
            return
        capability_by_name = cls._tool_capability_map(tool_definitions)
        if any(
            capability_by_name.get(name, set()) & {"note_anchor", "memory_persist", "tag_anchor"}
            for name in names
        ):
            state.pressure_recovery_note_saved = True
        if any(
            "attention_switch" in capability_by_name.get(name, set())
            for name in names
        ):
            state.pressure_recovery_attention_switched = True

        note_done = (
            getattr(state, "pressure_recovery_note_saved", False)
            or not getattr(state, "pressure_recovery_requires_note", True)
        )
        attention_done = (
            getattr(state, "pressure_recovery_attention_switched", False)
            or not getattr(state, "pressure_recovery_requires_attention", True)
        )
        if note_done and attention_done:
            ctx = dict(getattr(state, "pressure_attention_context", {}) or {})
            ctx.update({
                "completed": True,
                "resolved_at_turn": getattr(state, "fc_turn", 0),
            })
            state.pressure_attention_context = ctx
            state.pressure_stage = ""
            state.pressure_force_attention = False

    @staticmethod
    def _strip_xml_tool_tags(text: str) -> str:
        """从文本中移除 XML 工具调用标签，保留前置文本"""
        import re as _re
        # 移除 <thinking> 块
        text = _re.sub(r"<thinking>.*?</thinking>", "", text, flags=_re.DOTALL)
        # 移除有闭合标签的远程工具 XML
        for tool_name in IDE_REMOTE_TOOLS:
            text = _re.sub(
                rf"<{_re.escape(tool_name)}>.*?</{_re.escape(tool_name)}>",
                "", text, flags=_re.DOTALL)
        # 移除无闭合标签的远程工具 XML (<tool_name>... 到下一个工具标签或文本末尾)
        _tool_open_re = "|".join(_re.escape(t) for t in IDE_REMOTE_TOOLS)
        for tool_name in IDE_REMOTE_TOOLS:
            text = _re.sub(
                rf"<{_re.escape(tool_name)}>.*?(?=<(?:{_tool_open_re})>|\Z)",
                "", text, flags=_re.DOTALL)
        # 同时移除内部工具的 XML（LLM 可能也用 XML 调用内部工具）
        _internal_xml_tools = {
            "task_create_plan", "task_add_node", "task_mark_status",
            "task_view_overview", "recall_memory", "save_memory_note",
            "navigate_attention", "search_experience", "search_tools",
            "index_project", "index_code_file", "search_code_symbols",
            "get_symbol_context", "get_impact_analysis", "analyze_module",
            "zulong_code_query", "zulong_task_link_code",
        }
        for tool_name in _internal_xml_tools:
            text = _re.sub(
                rf"<{_re.escape(tool_name)}>.*?</{_re.escape(tool_name)}>",
                "", text, flags=_re.DOTALL)
        # 清理通用 XML 包装标签（LLM 可能用 <tool_call>/<function> 等包裹工具调用）
        _generic_wrappers = [
            "tool_call", "function_call", "tool_use",
            "function", "invoke", "tool",
        ]
        for tag in _generic_wrappers:
            text = _re.sub(
                rf"<{tag}(?:\s[^>]*)?>.*?</{tag}>",
                "", text, flags=_re.DOTALL)
        # 清理残留的孤立闭合标签（如 </parameter> </function> </tool_call>）
        text = _re.sub(
            r"</(?:parameter|function|tool_call|function_call|tool_use|"
            r"invoke|tool|name|arguments|thinking)>",
            "", text)
        # 清理残留的孤立开放标签
        text = _re.sub(
            r"<(?:parameter|function|tool_call|function_call|tool_use|"
            r"invoke|tool|name|arguments|thinking)(?:\s[^>]*)?>",
            "", text)
        # 清理 <parameter=name>value 残留
        text = _re.sub(r"<parameter=\w+>", "", text)
        return text.strip()

    def _exec_tools(self, state: IDEFCState, tool_calls_data: List[Dict],
                    response_content: str = "") -> Optional[List[Dict]]:
        """执行工具 + 分流。混合批次拆分为两个 assistant 消息避免 orphaned tool_call IDs。"""
        fc = state.fc_turn
        msgs = state.messages
        model_step_note = " ".join(str(response_content or "").split())
        announce_calls, real_calls = self._split_announce_step_calls(tool_calls_data)
        if announce_calls:
            self._append_announce_step_messages(state, announce_calls, response_content, fc, None)
        if real_calls and not model_step_note and not self._first_announce_step(announce_calls).get("message"):
            retry_count = getattr(state, "step_announce_retry_count", 0)
            if retry_count < 1:
                state.step_announce_retry_count = retry_count + 1
                control_msg = internal_control_message(
                    "请先用一句普通可见中文说明本步将做什么，再重新发起工具调用；不要输出推理过程。"
                    "如果当前模型/Provider 不保留 tool_calls 同轮 assistant.content，请先调用 announce_step(message=...) 再调用实际工具。"
                )
                msgs.append(control_msg)
                if self._attn_window:
                    self._attn_window.register_message(control_msg, turn=fc)
            else:
                state.cb_force_no_tools = True
                control_msg = internal_control_message(
                    "仍未收到可见步骤说明，本轮真实工具调用已被拦截。请不要继续调用工具，先说明当前受阻原因。"
                )
                msgs.append(control_msg)
                if self._attn_window:
                    self._attn_window.register_message(control_msg, turn=fc)
            return None
        if real_calls:
            state.step_announce_retry_count = 0
        tool_calls_data = real_calls
        if not tool_calls_data:
            return None
        internal, remote = [], []
        for td in tool_calls_data:
            (remote if self.tool_registry.classify(td["function"]["name"]) == "remote" else internal).append(td)
        grp = self._attn_window.new_tool_group() if self._attn_window else None

        if internal:
            a_msg = {"role": "assistant", "content": response_content or "", "tool_calls": internal}
            msgs.append(a_msg)
            if self._attn_window:
                self._attn_window.register_message(a_msg, turn=fc, group_id=grp)
            for td in internal:
                self._exec_internal(state, td, fc, grp)
            try:
                if self._circuit_breaker:
                    _aw_ratio = (
                        getattr(
                            self._attn_window,
                            "trigger_context_pressure_ratio",
                            getattr(self._attn_window, "context_pressure_ratio", self._attn_window.usage_ratio),
                        )
                        if self._attn_window else -1.0
                    )
                    cb_s, cb_r = self._circuit_breaker.evaluate(fc, msgs, attn_usage_ratio=_aw_ratio)
                    if cb_s == CircuitBreakerState.RED:
                        logger.warning(f"[IDEFCRunner][CB] RED: {cb_r}")
                        state.cb_force_no_tools = True
                        state.cb_trigger_reason = cb_r  # 保存CB原因供_finalize使用
                        cm = self._build_cb_red_control_message(state, cb_r)
                        msgs.append(cm)
                        if self._attn_window:
                            self._attn_window.register_message(cm, turn=fc)
                            try:
                                self._attn_window.on_navigate_attention(direction="single_chain")
                            except Exception:
                                pass
                        remote = []
                    elif cb_s == CircuitBreakerState.YELLOW:
                        logger.warning(f"[IDEFCRunner][CB] YELLOW: {cb_r}")
                        ch = internal_control_message(
                            f"[Circuit Breaker 警告] {cb_r}\n请尽快总结当前信息并回复用户，避免继续调用更多工具。"
                        )
                        msgs.append(ch)
                        if self._attn_window:
                            self._attn_window.register_message(ch, turn=fc)
                            try:
                                self._attn_window.on_navigate_attention(direction="focus")
                            except Exception:
                                pass
            except Exception as cb_err:
                logger.warning(f"[IDEFCRunner] CircuitBreaker evaluate 异常: {cb_err}")

            # 上下文压力感知（在 CB 评估之后）
            self._apply_pressure_guidance(state, fc)

        if remote:
            valid_remote, rejected = self._validate_and_clean_remote_calls(remote)
            if rejected:
                self._log_fc_decision_path(
                    state,
                    path="invalid_tool_args_remote_rejected",
                    tool_calls=remote,
                    root_cause="invalid_tool_args",
                    rejected_count=len(rejected),
                    rejected_tools=[r[0]["function"]["name"] for r in rejected],
                )
            all_calls = valid_remote + [r[0] for r in rejected]
            ra = {"role": "assistant", "content": "" if internal else (response_content or ""),
                  "tool_calls": all_calls}
            msgs.append(ra)
            if self._attn_window:
                self._attn_window.register_message(ra, turn=fc, group_id=grp)
            # 为被拒绝的调用注入错误结果，让 LLM 重试
            for rej_tc, err_msg in rejected:
                err_result = {"role": "tool", "tool_call_id": rej_tc["id"],
                              "content": f"[参数验证失败] {err_msg}"}
                msgs.append(err_result)
                if self._attn_window:
                    self._attn_window.register_message(
                        err_result, turn=fc, tool_name=rej_tc["function"]["name"])
            if valid_remote:
                self._maybe_run_bfs(fc, "tool_complete")
                return valid_remote

        self._maybe_run_bfs(fc, "tool_complete")
        return None

    def _exec_internal(self, state: IDEFCState, td: Dict, fc: int, grp: Optional[int]) -> None:
        tn = td["function"]["name"]
        ta = {}
        parse_error = ""
        try:
            ta = _json.loads(td["function"]["arguments"] or "{}")
        except Exception as exc:
            parse_error = str(exc)
        if parse_error:
            rt = _json.dumps({
                "error": f"工具参数解析失败: {parse_error}",
                "recoverable": True,
                "reason": "tool_arguments_json_invalid",
                "tool_name": tn,
            }, ensure_ascii=False)
            logger.error("[IDEFCRunner] 内部工具 %s 参数解析失败: %s", tn, parse_error)
            self._log_fc_decision_path(
                state,
                path="invalid_tool_args_internal_parse_error",
                tool_calls=[td],
                response_content=rt,
                root_cause="invalid_tool_args",
                tool_name=tn,
                error=_safe_error_summary(parse_error),
            )
            tm = {"role": "tool", "tool_call_id": td["id"], "content": rt}
            state.messages.append(tm)
            if self._attn_window:
                self._attn_window.register_message(
                    tm, turn=fc, tool_name=tn, group_id=grp)
            if len(state.tool_results_buffer) >= _TOOL_RESULTS_BUFFER_MAX:
                state.tool_results_buffer.pop(0)
            state.tool_results_buffer.append({
                "tool_name": tn,
                "result": rt,
                "arguments": {"_parse_error": parse_error},
                "success": False,
                "turn": fc,
            })
            if self._circuit_breaker:
                self._circuit_breaker.record_call(
                    tn, {"_parse_error": parse_error}, rt)
            return
        attn_before = self._attention_state()
        if self._attn_window:
            self._attn_window.observe_tool_call(tn, ta)
            if tn == "navigate_attention":
                self._attn_window.on_navigate_attention(
                    direction=ta.get("direction", ""), target_node_id=ta.get("target_node_id"))
        try:
            # TaskGraph CRUD 工具拦截
            from zulong.ide.graph_crud_tools import CRUD_TOOL_NAMES, dispatch_crud_tool
            if tn in CRUD_TOOL_NAMES:
                task_graph = self._get_current_task_graph()
                ws_sender = getattr(self, '_ws_send_callback', None)
                crud_result = dispatch_crud_tool(
                    tool_name=tn, arguments=ta, task_graph=task_graph,
                    session_id=self.session.session_id, ws_sender=ws_sender,
                )
                rt = _json.dumps(crud_result, ensure_ascii=False, default=str)
            else:
                rt = self.engine._execute_tool_call(_ToolCallProxy(td))
        except Exception as tool_err:
            logger.error(f"[IDEFCRunner] 内部工具 {tn} 执行异常: {tool_err}")
            rt = f"[工具执行异常] {tool_err}"
        # task_mark_status 完成后自动导航注意力窗口
        if self._attn_window and tn == "task_mark_status":
            new_status = ta.get("new_status") or ta.get("status") or ""
            mark_node = ta.get("node_id") or ""
            if new_status and mark_node:
                self._attn_window.auto_navigate_on_status_change(mark_node, new_status)
        # 🔥 任务完成验证：代码生成节点标记完成时，检查是否真正写入了文件
        if tn == "task_mark_status" and (ta.get("new_status") or ta.get("status")) == "completed":
            mark_node_id = ta.get("node_id")
            if mark_node_id:
                try:
                    task_graph = self._get_current_task_graph()
                    if task_graph:
                        node = task_graph.get_node(mark_node_id)
                        if node:
                            label_lower = (node.label or "").lower()
                            desc_lower = (node.desc or "").lower()
                            # 判断是否为代码生成类节点
                            _code_gen_kw = ("写", "编写", "创建", "生成", "代码", "code", "write",
                                            "create", "文件", "file", "html", "css", "js", "实现",
                                            "开发", "构建", "build", "页面", "page", "组件", "component")
                            is_code_gen = any(kw in label_lower or kw in desc_lower
                                              for kw in _code_gen_kw)
                            if is_code_gen:
                                capability_by_name = self._tool_capability_map(
                                    getattr(state, "tool_definitions", []) or []
                                )
                                recent_results = list(state.tool_results_buffer[-10:] or [])
                                has_write = any(
                                    self._tool_result_has_capability(r, "file_write", capability_by_name)
                                    and r.get("success", not self._tool_result_failed_text(r.get("result", "")))
                                    for r in recent_results
                                )
                                if not has_write:
                                    warn = (
                                        f"\n\n⚠️ [任务完成验证] 该节点({node.label})属于代码生成任务，"
                                        f"但最近未检测到成功的文件写入能力工具结果。"
                                        f"请确认文件已真实落盘；如尚未落盘，请调用当前可用的文件写入工具。"
                                    )
                                    rt = rt + warn
                except Exception as val_err:
                    logger.debug(f"[IDEFCRunner] 任务完成验证异常: {val_err}")
        attn_after = self._attention_state()
        if self._circuit_breaker:
            self._circuit_breaker.record_call(tn, ta, rt)
            if tn in ("task_create_plan", "start_task_plan", "task_add_node"):
                self._circuit_breaker.escalate_for_planning()
        if len(rt) > MAX_TOOL_RESULT_CHARS:
            ol = len(rt)
            rt = rt[:MAX_TOOL_RESULT_CHARS] + f"\n...(已截断，原始长度 {ol} 字符)"
        tm = {"role": "tool", "tool_call_id": td["id"], "content": rt}
        state.messages.append(tm)
        if self._attn_window:
            self._attn_window.register_message(
                tm, turn=fc, tool_name=tn,
                node_id=ta.get("node_id") or ta.get("target_node_id"), group_id=grp)
        if len(state.tool_results_buffer) >= _TOOL_RESULTS_BUFFER_MAX:
            state.tool_results_buffer.pop(0)
        tool_success = not self._tool_result_failed_text(rt)
        memory_reference_edges = self._memory_reference_edges_from_result(rt)
        state.tool_results_buffer.append({
            "tool_name": tn,
            "result": rt,
            "arguments": ta,
            "success": tool_success,
            "turn": fc,
            "memory_reference_edges": memory_reference_edges,
        })
        nudge_msg = self._build_observation_nudge(state, tn, rt, fc)
        if nudge_msg:
            state.messages.append(nudge_msg)
            if self._attn_window:
                self._attn_window.register_message(
                    nudge_msg, turn=fc,
                    node_id=ta.get("node_id") or ta.get("target_node_id"),
                    group_id=grp)
        if attn_before != attn_after and tn in (
            "navigate_attention", "adjust_attention_mode", "task_mark_status"
        ):
            current_node_id = attn_after[1]
            if tn == "task_mark_status" and (ta.get("new_status") or ta.get("status")) in ("completed", "skipped"):
                try:
                    task_graph = self._get_current_task_graph()
                    parent_id = task_graph.get_parent(ta.get("node_id") or "") if task_graph else ""
                    if parent_id and (not current_node_id or current_node_id == ta.get("node_id")):
                        current_node_id = parent_id
                except Exception:
                    pass
            reason = "attention_changed" if tn in ("navigate_attention", "adjust_attention_mode") else "status_auto_navigate"
            nav_msg = self._build_attention_navigation_map(
                state, fc, reason=reason, current_node_id=current_node_id)
            if nav_msg:
                state.messages.append(nav_msg)
                if self._attn_window:
                    self._attn_window.register_message(
                        nav_msg, turn=fc,
                        node_id=current_node_id,
                        group_id=grp)
        self.engine._publish_task_graph_event("agent_tool_call", fc, tn, rt)
        # 记录子对话到 MemoryGraph
        self._record_sub_dialogue(state, tool_name=tn, result=rt)
        friendly_tn = self._friendly_tool_name(tn)
        friendly_action = self._friendly_action_summary(tn, ta)
        self._emit_execution_event_sync(
            "tool_finished",
            f"{friendly_action}已完成",
            turn=fc,
            event_type="IDE_TOOL_EXEC",
            payload={
                "tool_name": tn,
                "action_summary": friendly_action,
                "arguments_preview": _json.dumps(ta, ensure_ascii=False)[:300],
                "result_preview": (rt or "")[:500],
                "interaction": {
                    "kind": "observation",
                    "status": "succeeded",
                    "title": f"{friendly_action}已完成",
                    "detail": "结果已写回本轮上下文，祖龙会据此继续推进。",
                    "tool_name": tn,
                    "result_preview": (rt or "")[:500],
                },
            },
        )

    # 参数默认值：LLM 省略某些参数时自动填充
    # 典型场景：list_files() 不传 path; execute_command() 不传 requires_approval
    _PARAM_DEFAULTS: Dict[str, Dict[str, str]] = {
        "list_files": {"path": "."},
        "search_files": {"path": "."},
        "list_code_definition_names": {"path": "."},
        "execute_command": {"requires_approval": "false"},
    }

    # LLM 常见参数名别名 → schema 标准名映射
    # LLM 有时使用 file_path / filepath 等替代 path，导致参数验证失败
    _PARAM_ALIASES: Dict[str, Dict[str, str]] = {
        "read_file": {
            "file_path": "path", "filepath": "path", "file": "path",
            "filename": "path", "file_name": "path",
        },
        "write_to_file": {
            "file_path": "path", "filepath": "path", "file": "path",
            "filename": "path", "file_name": "path",
            "file_content": "content", "text": "content", "data": "content",
            "write_mode": "mode",
        },
        "replace_in_file": {
            "file_path": "path", "filepath": "path",
            "changes": "diff", "replacement": "diff", "replacements": "diff",
        },
        "execute_command": {
            "cmd": "command", "shell_command": "command",
            "shell_type": "shell", "terminal_shell": "shell", "terminal": "shell",
        },
        "search_files": {
            "directory": "path", "dir": "path", "folder": "path",
            "pattern": "regex", "search_pattern": "regex", "query": "regex",
        },
        "list_files": {
            "directory": "path", "dir": "path", "folder": "path",
        },
        "list_code_definition_names": {
            "file_path": "path", "filepath": "path",
            "directory": "path", "dir": "path",
        },
    }

    def _validate_and_clean_remote_calls(
        self, remote_calls: List[Dict]
    ) -> Tuple[List[Dict], List[Tuple[Dict, str]]]:
        """验证远程工具必需参数 & 清理非 schema 参数（如 task_progress）。

        修复: 增加参数名别名自动映射、非 dict 类型防御、更清晰的错误提示。

        Returns:
            (valid_calls, rejected): rejected 是 [(tool_call_dict, error_msg), ...]
        """
        from zulong.ide.ide_tool_registry import _IDE_TOOL_SCHEMAS
        schema_map: Dict[str, Dict] = {}
        for s in _IDE_TOOL_SCHEMAS:
            f = s.get("function", {})
            p = f.get("parameters", {})
            schema_map[f.get("name", "")] = {
                "required": p.get("required", []),
                "properties": set(p.get("properties", {}).keys()),
            }

        valid: List[Dict] = []
        rejected: List[Tuple[Dict, str]] = []
        for tc in remote_calls:
            fn = tc["function"]["name"]
            args_str = tc["function"]["arguments"]
            try:
                args = _json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
            except Exception:
                args = {}

            # 类型防御：确保 args 是 dict（LLM 可能生成 "null"、纯字符串等非对象 JSON）
            if not isinstance(args, dict):
                logger.warning(
                    f"[IDEFCRunner] 工具 {fn} 参数不是 dict: {type(args).__name__}={str(args)[:100]}")
                args = {}

            info = schema_map.get(fn)
            if not info:
                tc["function"]["arguments"] = _json.dumps(args, ensure_ascii=False)
                valid.append(tc)
                continue

            # ── 参数名别名自动映射（file_path→path 等） ──
            aliases = self._PARAM_ALIASES.get(fn, {})
            remapped = []
            for k in list(args.keys()):
                canonical = aliases.get(k)
                if canonical and canonical not in args:
                    args[canonical] = args.pop(k)
                    remapped.append(f"{k}→{canonical}")
            if remapped:
                logger.info(
                    f"[IDEFCRunner] 参数名自动映射 {fn}: {', '.join(remapped)}")

            # ── 参数默认值填充（LLM 省略有合理默认值的必需参数时自动补全） ──
            defaults = self._PARAM_DEFAULTS.get(fn, {})
            defaulted = []
            for param, default_val in defaults.items():
                if param not in args or not args.get(param):
                    args[param] = default_val
                    defaulted.append(f"{param}='{default_val}'")
            if defaulted:
                logger.info(
                    f"[IDEFCRunner] 参数默认值填充 {fn}: {', '.join(defaulted)}")

            if fn == "write_to_file":
                mode = str(args.get("mode") or "overwrite").strip().lower()
                if mode not in {"overwrite", "append"}:
                    mode = "overwrite"
                args["mode"] = mode
                if mode == "append":
                    path_value = str(args.get("path") or "")
                    content_value = str(args.get("content") or "")
                    try:
                        target = Path(path_value)
                        if not target.is_absolute():
                            try:
                                from zulong.tools.task_tools import get_active_workspace_dir
                                workspace = get_active_workspace_dir()
                            except Exception:
                                workspace = ""
                            if workspace:
                                target = Path(workspace) / target
                        if target.is_absolute() and target.exists() and target.is_file():
                            args["content"] = target.read_text(encoding="utf-8") + content_value
                            args["mode"] = "overwrite"
                            logger.info(
                                "[IDEFCRunner] write_to_file append 已预展开为覆盖完整内容: %s",
                                target,
                            )
                    except Exception as exc:
                        logger.warning(
                            "[IDEFCRunner] write_to_file append 预展开失败，交给客户端处理: %s",
                            exc,
                        )

            # ── 路径矫正：list_code_definition_names 的 path 必须是目录 ──
            # IDE 扩展将 path 用作子进程 cwd，传入文件路径会报错
            # "The cwd option must be a path to a directory"
            if fn == "list_code_definition_names" and args.get("path"):
                import posixpath as _ppath
                p = args["path"]
                # 检测文件路径特征：包含常见文件扩展名
                if "." in _ppath.basename(p.replace("\\", "/")):
                    parent = _ppath.dirname(p.replace("\\", "/")) or "."
                    logger.info(
                        f"[IDEFCRunner] 路径矫正 {fn}: '{p}' → '{parent}' (文件→目录)")
                    args["path"] = parent

            # 清理非 schema 参数（如 task_progress）
            non_schema = [k for k in list(args.keys()) if k not in info["properties"]]
            for k in non_schema:
                logger.info(f"[IDEFCRunner] 清理非 schema 参数: {fn}.{k}={str(args[k])[:80]}")
                del args[k]

            # 检查必需参数
            missing = [p for p in info["required"] if p not in args or not args.get(p)]
            if missing:
                err = (
                    f"工具 {fn} 缺少必需参数: {missing}。"
                    f"该工具的正确参数为: {sorted(info['properties'])}，"
                    f"其中必需: {info['required']}。请使用正确的参数名重新调用。"
                )
                logger.warning(f"[IDEFCRunner] {err}")
                tc["function"]["arguments"] = _json.dumps(args, ensure_ascii=False)
                rejected.append((tc, err))
            else:
                tc["function"]["arguments"] = _json.dumps(args, ensure_ascii=False)
                valid.append(tc)

        return valid, rejected

    def _eval_response(self, state: IDEFCState, response_content: str) -> str:
        fc = state.fc_turn
        msgs = state.messages
        resp = response_content
        is_resume = state.is_resume

        logger.info(
            f"[IDEFCRunner][EvalChain] turn={fc} resp_len={len(resp.strip()) if resp else 0} "
            f"cb_force={state.cb_force_no_tools} null_count={state.null_response_count}"
        )

        if state.cb_force_no_tools:
            if getattr(state, "cb_recovery_stage", "") in {"restricted_recovery", "note_attention"} and not (
                getattr(state, "cb_recovery_note_saved", False)
                and getattr(state, "cb_recovery_attention_switched", False)
            ):
                recovery_hint = internal_control_message(
                    "[Circuit Breaker RED 受限恢复未完成]\n"
                    "请不要直接输出最终总结。必须先完成："
                    "1) 写入并锚定便签/标签/记忆；2) 执行一次注意力切换。"
                )
                msgs.append(recovery_hint)
                if self._attn_window:
                    self._attn_window.register_message(recovery_hint, turn=fc)
                return "continue"

        if getattr(state, "pressure_stage", "") == "restricted_recovery":
            note_done = (
                getattr(state, "pressure_recovery_note_saved", False)
                or not getattr(state, "pressure_recovery_requires_note", True)
            )
            attention_done = (
                getattr(state, "pressure_recovery_attention_switched", False)
                or not getattr(state, "pressure_recovery_requires_attention", True)
            )
            if not (note_done and attention_done):
                recovery_hint = internal_control_message(
                    "[上下文压力 RED 受限恢复未完成]\n"
                    "请不要直接输出最终总结。必须先完成："
                    "1) 写入并锚定便签/标签/记忆；2) 执行一次注意力切换。"
                )
                msgs.append(recovery_hint)
                if self._attn_window:
                    self._attn_window.register_message(recovery_hint, turn=fc)
                return "continue"

        if state.cb_force_no_tools:
            if not resp or len(resp.strip()) < 10:
                resp = self._get_cb_fallback(state)
            self._run_backfill(state, resp, is_cb_path=True)
            state.last_response_content = resp
            state.cb_force_no_tools = False
            state.cb_tool_streak = 0  # 重置 CB 工具连续计数
            # 修复 4: CB 路径检查未完成节点 — 防止大量任务未完成时无条件终止
            # 注意：排除 CRG 自动注入节点（crg_ 前缀），它们不代表用户任务进度
            try:
                from zulong.tools.task_tools import get_active_task_graph as _gtg_cb
                tg_cb = _gtg_cb()
                if tg_cb and state.null_response_count < 6:
                    leaves_cb = tg_cb.get_leaf_nodes()
                    # 只计算 LLM 规划的节点（排除 CRG 自动注入节点）
                    user_leaves = [n for n in leaves_cb if not n.id.startswith("crg_")]
                    unc_cb = [n for n in user_leaves
                              if n.status not in ("completed", "skipped")]
                    if len(unc_cb) > 1 and user_leaves and len(unc_cb) > len(user_leaves) * 0.5:
                        logger.info(
                            f"[IDEFCRunner][CB] 仍有 {len(unc_cb)}/{len(user_leaves)} "
                            f"用户任务未完成，恢复工具调用继续执行"
                        )
                        logger.info(f"[IDEFCRunner][EvalChain] turn={fc} path=cb_continue")
                        return "continue"
            except Exception:
                pass
            gate_verdict, resp = self._completion_quality_gate(state, resp)
            if gate_verdict:
                logger.info(f"[IDEFCRunner][EvalChain] turn={fc} path=cb_quality_{gate_verdict}")
                return gate_verdict
            state.last_response_content = resp
            logger.info(f"[IDEFCRunner][CB] 强制回复, len={len(resp)}")
            logger.info(f"[IDEFCRunner][EvalChain] turn={fc} path=cb_force_done")
            return "done"

        # 安全网 0: 语义漂移检测
        # 检查模型回复是否偏离了原始任务目标
        if self._drift_detector and resp and len(resp) > 50:
            try:
                import asyncio as _aio
                try:
                    loop = _aio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    # 已有事件循环（如 FastAPI 上下文），使用线程池
                    import concurrent.futures as _cf
                    with _cf.ThreadPoolExecutor(max_workers=1) as _pool:
                        drift_result = _pool.submit(
                            lambda: _aio.run(
                                self._drift_detector.detect_drift(resp)
                            )
                        ).result(timeout=5)
                else:
                    drift_result = _aio.run(
                        self._drift_detector.detect_drift(resp))
                is_drifted, similarity, reason = drift_result
                logger.info(
                    f"[IDEFCRunner][DriftDetector] turn={fc} "
                    f"drift={is_drifted}, sim={similarity:.3f}, {reason}")
                if is_drifted:
                    # 显著漂移 → 注入纠偏提示让模型重新聚焦
                    # 注意：必须用 "user" role，SiliconFlow 等 API 要求 system 消息在最前面
                    drift_hint = {
                        "role": "user",
                        "content": (
                            f"[语义漂移检测] {reason}\n"
                            f"原始任务: {state.user_input_text[:200]}\n"
                            f"你的回复偏离了任务目标，请重新聚焦原始任务，"
                            f"调用 task_view_overview 查看当前进度后继续执行。"
                        ),
                    }
                    msgs.append({"role": "assistant", "content": resp})
                    msgs.append(drift_hint)
                    if self._attn_window:
                        self._attn_window.register_message(
                            {"role": "assistant", "content": resp}, turn=fc)
                        self._attn_window.register_message(drift_hint, turn=fc)
                    state.null_response_count += 1
                    return "cb_force" if state.null_response_count >= 4 else "continue"
                # 非漂移：异步记录对话历史供后续检测使用
                try:
                    if loop and loop.is_running():
                        with _cf.ThreadPoolExecutor(max_workers=1) as _pool2:
                            _pool2.submit(
                                lambda: _aio.run(
                                    self._drift_detector.add_conversation_turn(
                                        state.user_input_text or "", resp)
                                )
                            ).result(timeout=5)
                    else:
                        _aio.run(
                            self._drift_detector.add_conversation_turn(
                                state.user_input_text or "", resp))
                except Exception:
                    pass
            except Exception as drift_err:
                logger.warning(f"[IDEFCRunner][DriftDetector] 检测异常: {drift_err}")

        # 安全网 0.5: 独白检测 (OpenHands-style monologue detection)
        # 检测模型连续纯文本回复而不调用工具的退化模式
        # 当连续3次以上纯文本 + 有未完成任务时，注入强纠正提示
        if (state.consecutive_text_only_count >= 3
                and resp and len(resp.strip()) >= 6):
            try:
                from zulong.tools.task_tools import get_active_task_graph as _gtg_mono
                tg_mono = _gtg_mono()
                if tg_mono:
                    leaves_mono = tg_mono.get_leaf_nodes()
                    user_leaves_mono = [n for n in leaves_mono if not n.id.startswith("crg_")]
                    uncompleted_mono = [n for n in user_leaves_mono
                                        if n.status not in ("completed", "skipped")]
                    if not user_leaves_mono:
                        req_node = tg_mono.get_node("req")
                        if req_node and req_node.status not in ("completed", "skipped"):
                            uncompleted_mono = [req_node]
                    if uncompleted_mono:
                        current_mono = next(
                            (n for n in uncompleted_mono if n.status == "in_progress"),
                            uncompleted_mono[0],
                        )
                        # 强纠正提示按工具能力表达，不绑定单个工具名。
                        label_lower = (current_mono.label or "").lower()
                        tool_hint = ""
                        if any(kw in label_lower for kw in ("写", "编写", "创建", "生成", "代码", "code", "write", "create", "文件", "file", "html", "css", "js", "实现", "开发")):
                            tool_hint = (
                                "请立即调用当前可用的文件写入能力工具完成真实落盘。"
                            )
                        elif any(kw in label_lower for kw in ("分析", "分析", "检查", "review", "查看", "阅读", "读", "read")):
                            tool_hint = "请调用当前可用的读取/检索/分析能力工具获取证据。"
                        elif any(kw in label_lower for kw in ("运行", "执行", "测试", "运行", "test", "run", "命令", "command")):
                            tool_hint = "请调用当前可用的命令执行/验证能力工具运行检查。"
                        else:
                            tool_hint = (
                                "请调用与当前节点匹配的执行能力工具推进任务。不要只口头叙述！"
                            )
                        nudge_mono = {
                            "role": "user",
                            "content": (
                                f"[独白检测] 你已经连续 {state.consecutive_text_only_count} 次"
                                f"只输出文本而没有调用任何工具！\n"
                                f"当前应执行: {current_mono.id}({current_mono.label})。\n"
                                f"还有 {len(uncompleted_mono)} 个任务未完成。\n"
                                f"{tool_hint}\n"
                                f"⚠️ 仅在文本中说「开始编写」而不调用工具，等于什么都没做。"
                            ),
                        }
                        msgs.append({"role": "assistant", "content": resp})
                        msgs.append(nudge_mono)
                        if self._attn_window:
                            self._attn_window.register_message(
                                {"role": "assistant", "content": resp}, turn=fc)
                            self._attn_window.register_message(nudge_mono, turn=fc)
                        state.null_response_count += 1
                        mono_count = state.consecutive_text_only_count
                        state.consecutive_text_only_count = 0  # 重置，避免重复nudge
                        logger.warning(
                            f"[IDEFCRunner][Monologue] turn={fc} "
                            f"连续{mono_count}次纯文本，"
                            f"未完成={len(uncompleted_mono)}, 当前={current_mono.id}")
                        return "cb_force" if state.null_response_count >= 4 else "continue"
            except Exception as e:
                logger.warning(f"[IDEFCRunner][Monologue] {e}")

        # 安全网 1: RuleGuardian
        blocked = False
        if self._rule_guardian:
            try:
                from zulong.tools.task_tools import get_active_task_graph as _gtg
                blk, br = self._rule_guardian.check_premature_completion(resp, _gtg())
                if blk:
                    cor = {"role": "user", "content":
                           f"[规则守护] {br}\n请调用 task_view_overview 查看任务图，然后继续执行未完成的任务。"}
                    msgs.append({"role": "assistant", "content": resp})
                    msgs.append(cor)
                    if self._attn_window:
                        self._attn_window.register_message({"role": "assistant", "content": resp}, turn=fc)
                        self._attn_window.register_message(cor, turn=fc)
                    blocked = True
            except Exception as e:
                logger.warning(f"[IDEFCRunner][RuleGuardian] {e}")
        if blocked:
            state.null_response_count += 1
            return "cb_force" if state.null_response_count >= 4 else "continue"

        # 安全网 2: InfoGap
        try:
            from zulong.l2.info_gap_detector import InfoGapType
            sc = self._build_subtask_context()
            gt, gd, gc = self.engine._info_gap_detector.detect(
                llm_output=resp, tool_results=state.tool_results_buffer or None,
                subtask_context=sc)
            if gt == InfoGapType.NEED_SUBTASK_RESULT and gc >= 0.6 and state.gap_continue_count < 5:
                gh = {"role": "user", "content":
                      f"[信息缺口] 缺少前置结果: {gd}\n请先用 task_view_overview 查看任务图。"}
                msgs.append({"role": "assistant", "content": resp})
                msgs.append(gh)
                if self._attn_window:
                    self._attn_window.register_message({"role": "assistant", "content": resp}, turn=fc)
                    self._attn_window.register_message(gh, turn=fc)
                state.gap_continue_count += 1
                state.null_response_count += 1
                return "cb_force" if state.null_response_count >= 4 else "continue"
        except Exception as e:
            logger.warning(f"[IDEFCRunner][InfoGap] {e}")

        # 安全网 3: 继续任务图 AutoMark
        if (is_resume and len(resp) > 100 and state.resume_automark_count < 5
                and not resp.rstrip().endswith(("?", "\uff1f"))
                and not _is_filler_content(resp)
                and not _looks_like_incomplete_result(resp)):
            try:
                from zulong.tools.task_tools import get_active_task_graph as _gtg_am, _save_active_backup
                tg = _gtg_am()
                if tg:
                    leaves = tg.get_leaf_nodes()
                    unc = [n for n in leaves if n.status != "completed"]
                    if unc:
                        tgt = next((n for n in unc if n.status == "in_progress"), unc[0])
                        result_text = resp[:500]
                        if _looks_like_incomplete_result(result_text):
                            return "done"
                        if hasattr(tgt, "metadata"):
                            tgt.metadata["auto_progress_candidate"] = result_text
                            tgt.metadata["auto_progress_candidate_at_turn"] = state.fc_turn
                            tgt.metadata["auto_progress_candidate_source"] = "resume_text_review"
                        try: _save_active_backup()
                        except Exception as e:
                            ErrorHandler.handle_exception(e, ErrorCode.CFG_SAVE_FAILED, context={"operation": "save_active_backup"}, log_level=logging.WARNING)
                        rem = [n for n in tg.get_leaf_nodes() if n.status not in ("completed", "skipped")]
                        if rem:
                            nn = rem[0]
                            if nn.status in ("pending", "needs_adjust", "waiting_input"):
                                tg.update_node_status(nn.id, "in_progress")
                            try: _save_active_backup()
                            except Exception as e:
                                ErrorHandler.handle_exception(e, ErrorCode.CFG_SAVE_FAILED, context={"operation": "save_active_backup"}, log_level=logging.WARNING)
                            cont = {"role": "user", "content":
                                    f"[自动进度候选] 检测到 {tgt.id}({tgt.label}) 可能已有文本产出，但不会自动标记完成。"
                                    f"请基于真实工具/验证证据显式调用 task_mark_status；当前继续 {nn.id}({nn.label})。"}
                            msgs.append({"role": "assistant", "content": resp})
                            msgs.append(cont)
                            if self._attn_window:
                                self._attn_window.register_message(
                                    {"role": "assistant", "content": resp}, turn=fc)
                                self._attn_window.register_message(cont, turn=fc)
                            state.resume_automark_count += 1
                            state.null_response_count += 1
                            return "cb_force" if state.null_response_count >= 4 else "continue"
            except Exception as e:
                logger.warning(f"[IDEFCRunner][AutoMark] {e}")

        # 安全网 4: Backfill
        if (not is_resume and len(resp) > 100
                and not resp.rstrip().endswith(("?", "\uff1f")) and not _is_filler_content(resp)):
            self._run_backfill(state, resp, is_cb_path=False)

        # 安全网 5: 空回复拦截
        if not resp or len(resp.strip()) < 10:
            try:
                from zulong.tools.task_tools import get_active_task_graph as _gtg_e
                _tg = _gtg_e()
                if _tg:
                    lv = _tg.get_leaf_nodes()
                    uc = [n for n in lv if n.status not in ("completed", "skipped")]
                    # 回退: get_leaf_nodes 排除了 req 根节点，当只有 req 且未完成时也应拦截
                    if not uc and not lv:
                        req_node = _tg.get_node("req")
                        if req_node and req_node.status not in ("completed", "skipped"):
                            uc = [req_node]
                    if uc:
                        nx = next((n for n in uc if n.status == "in_progress"), None)
                        if not nx:
                            nx = uc[0]
                            _tg.update_node_status(nx.id, "in_progress")
                        nudge = {"role": "user", "content":
                                 f"[空回复拦截] 任务图有 {len(uc)} 个未完成节点。"
                                 f"请立即调用工具执行任务: {nx.label}。不要输出空内容。"}
                        msgs.append(nudge)
                        if self._attn_window:
                            self._attn_window.register_message(nudge, turn=fc)
                        state.null_response_count += 1
                        return "cb_force" if state.null_response_count >= 4 else "continue"
            except Exception as e:
                logger.warning(f"[IDEFCRunner][EmptyGuard] {e}")

        if not resp or len(resp.strip()) < 10:
            resp = self._synthesize_from_task_graph() or resp

        # 安全网 6: 未完成任务拦截（不依赖关键词，纯状态判断）
        # 模型返回了有效文本，但 TaskGraph 仍有大量未完成节点
        # 跳过 filler 内容：模型持续输出无实质内容时，再次注入提示只会导致循环
        # 注意：排除 CRG 自动注入节点（crg_ 前缀），只看 LLM 规划的用户任务节点
        if (
            resp
            and len(resp.strip()) >= 10
            and not _is_filler_content(resp)
            and not _looks_like_incomplete_result(resp)
        ):
            try:
                from zulong.tools.task_tools import get_active_task_graph as _gtg_uc
                tg_uc = _gtg_uc()
                if tg_uc:
                    leaves_uc = tg_uc.get_leaf_nodes()
                    # 排除 CRG 自动注入节点
                    user_leaves_uc = [n for n in leaves_uc if not n.id.startswith("crg_")]
                    total_uc = len(user_leaves_uc)
                    uncompleted_uc = [n for n in user_leaves_uc
                                      if n.status not in ("completed", "skipped")]
                    # 回退: 无叶子节点时检查 req 根节点（模型尚未创建子节点的情况）
                    if not user_leaves_uc:
                        req_node = tg_uc.get_node("req")
                        if req_node and req_node.status not in ("completed", "skipped"):
                            uncompleted_uc = [req_node]
                            total_uc = 1
                    # 超过阈值比例用户任务节点未完成 → 继续执行
                    if len(uncompleted_uc) > 0 and total_uc > 0 and len(uncompleted_uc) >= total_uc * _UNCOMPLETED_THRESHOLD:
                        current_uc = next(
                            (n for n in uncompleted_uc if n.status == "in_progress"),
                            uncompleted_uc[0],
                        )
                        if current_uc.status != "in_progress":
                            tg_uc.update_node_status(current_uc.id, "in_progress")
                            try:
                                from zulong.tools.task_tools import _save_active_backup
                                _save_active_backup()
                            except Exception:
                                pass
                        nudge_uc = {
                            "role": "user",
                            "content": (
                                f"[任务未完成] 仍有 {len(uncompleted_uc)}/{total_uc} 个子任务未完成。"
                                f"当前应执行: {current_uc.id}({current_uc.label})。"
                                f"请调用工具执行任务，不要只输出文本叙述。"
                                f"\n⚠️ 口头说「开始编写」「我将创建」而不调用工具 = 什么都没做！"
                            ),
                        }
                        msgs.append({"role": "assistant", "content": resp})
                        msgs.append(nudge_uc)
                        if self._attn_window:
                            self._attn_window.register_message(
                                {"role": "assistant", "content": resp}, turn=fc)
                            self._attn_window.register_message(nudge_uc, turn=fc)
                        state.null_response_count += 1
                        return "cb_force" if state.null_response_count >= 4 else "continue"
            except Exception as e:
                logger.warning(f"[IDEFCRunner][UncompletedGuard] {e}")

        # 安全网 6.5: 响应提前中断检测
        # LLM返回短文本进度汇报但function_call未生成时，不应判定为done
        # 条件：短文本(<80字符) + 非filler(含进度动词) + 有未完成节点(>=30%)
        if (resp and 6 <= len(resp.strip()) < 80
                and not _is_filler_content(resp)
                and state.null_response_count < 3):
            try:
                from zulong.tools.task_tools import get_active_task_graph as _gtg_ri
                tg_ri = _gtg_ri()
                if tg_ri:
                    leaves_ri = tg_ri.get_leaf_nodes()
                    user_leaves_ri = [n for n in leaves_ri if not n.id.startswith("crg_")]
                    total_ri = len(user_leaves_ri)
                    uncompleted_ri = [n for n in user_leaves_ri
                                      if n.status not in ("completed", "skipped")]
                    if (total_ri > 0
                            and len(uncompleted_ri) > 0
                            and len(uncompleted_ri) >= total_ri * 0.3):
                        current_ri = next(
                            (n for n in uncompleted_ri if n.status == "in_progress"),
                            uncompleted_ri[0],
                        )
                        # 根据当前节点类型提供能力类别引导，不绑定单个工具名。
                        label_ri = (current_ri.label or "").lower()
                        tool_ri_hint = ""
                        if any(kw in label_ri for kw in ("写", "编写", "创建", "生成", "代码", "code", "write", "create", "文件", "file", "html", "css", "js", "实现", "开发", "页面")):
                            tool_ri_hint = "\n请调用当前可用的文件写入能力工具完成真实落盘。"
                        elif any(kw in label_ri for kw in ("运行", "执行", "测试", "test", "run", "命令")):
                            tool_ri_hint = "\n请调用当前可用的命令执行/验证能力工具运行检查。"
                        elif any(kw in label_ri for kw in ("查看", "分析", "阅读", "检查", "read", "review")):
                            tool_ri_hint = "\n请调用当前可用的读取/检索/分析能力工具获取证据。"
                        nudge_ri = {
                            "role": "user",
                            "content": (
                                f"[响应提前中断] 你的回复仅{len(resp.strip())}字符，"
                                f"疑似function_call未生成。"
                                f"仍有 {len(uncompleted_ri)}/{total_ri} 个子任务未完成。"
                                f"当前应执行: {current_ri.id}({current_ri.label})。"
                                f"{tool_ri_hint}"
                                f"\n⚠️ 只说「开始编写」而不调用工具 = 什么都没做！"
                            ),
                        }
                        msgs.append({"role": "assistant", "content": resp})
                        msgs.append(nudge_ri)
                        if self._attn_window:
                            self._attn_window.register_message(
                                {"role": "assistant", "content": resp}, turn=fc)
                            self._attn_window.register_message(nudge_ri, turn=fc)
                        state.null_response_count += 1
                        logger.info(
                            f"[IDEFCRunner][ResponseIntegrity] turn={fc} "
                            f"resp_len={len(resp.strip())} "
                            f"uncompleted={len(uncompleted_ri)}/{total_ri} "
                            f"疑似响应提前中断(function_call未生成)"
                        )
                        return "cb_force" if state.null_response_count >= 4 else "continue"
            except Exception as e:
                logger.warning(f"[IDEFCRunner][ResponseIntegrity] {e}")

        # 安全网 7: 首轮无效回复拦截
        # 当工具增强策略首轮模型未调用工具，直接返回短回复/问候语时，
        # 注入纠正提示让模型重新聚焦任务并使用工具
        if (_is_filler_content(resp) and fc <= 1
                and getattr(state, "task_graph_policy", "") in {"inspect_or_create", "create", "extend"}
                and state.null_response_count < 3):
            _GREETING_PATTERNS = (
                "你好", "您好", "有什么我可以帮", "有什么可以帮",
                "我可以帮你", "需要帮助", "hello", "hi", "how can i",
                "what can i", "请问", "请说",
            )
            stripped_lower = (resp or "").strip().lower()
            is_greeting = any(p in stripped_lower for p in _GREETING_PATTERNS)
            is_too_short = len((resp or "").strip()) < 50

            if is_greeting or is_too_short:
                nudge_first = {
                    "role": "user",
                    "content": (
                        f"[首轮回复无效] 你返回了一个无意义的问候/短回复，但用户的任务是:\n"
                        f"「{(state.user_input_text or '')[:300]}」\n"
                        f"请不要打招呼或反问，直接分析任务需求并调用工具开始执行。"
                    ),
                }
                msgs.append({"role": "assistant", "content": resp})
                msgs.append(nudge_first)
                if self._attn_window:
                    self._attn_window.register_message(
                        {"role": "assistant", "content": resp}, turn=fc)
                    self._attn_window.register_message(nudge_first, turn=fc)
                state.null_response_count += 1
                logger.info(
                    f"[IDEFCRunner][FirstTurnGuard] 首轮无效回复被拦截: "
                    f"'{resp[:50]}', greeting={is_greeting}")
                return "continue"

        gate_verdict, resp = self._completion_quality_gate(state, resp)
        if gate_verdict:
            logger.info(f"[IDEFCRunner][EvalChain] turn={fc} path=quality_{gate_verdict}")
            return gate_verdict

        state.last_response_content = resp
        logger.info(f"[IDEFCRunner][EvalChain] turn={fc} path=default_done")
        return "done"

    def _pause_for_remote(self, state: IDEFCState, remote_calls: List[Dict]) -> IDEFCResult:
        state.pending_remote_calls = remote_calls
        state.pending_call_ids = [tc["id"] for tc in remote_calls]
        state.phase = "waiting_remote"
        self._save_runner_state()
        self.session.fc_state = state
        tool_names = [tc["function"]["name"] for tc in remote_calls]
        logger.info(f"[IDEFCRunner] FC 暂停, {len(remote_calls)} 远程工具: {tool_names}")
        # 推送 TaskGraph 更新到 web 仪表盘（远程工具分发时）
        try:
            self.engine._publish_task_graph_event(
                "agent_tool_call", state.fc_turn,
                ",".join(tool_names), f"远程工具调用: {tool_names}")
        except Exception:
            pass
        return IDEFCResult(phase="waiting_remote",
                             pending_call_ids=state.pending_call_ids)

    def _finalize(self, state: IDEFCState, reason: str) -> IDEFCResult:
        state.phase = "done"
        # CB相关reason: blocked是进度状态，不是错误（对齐TSD §23.3.6）
        _cb_reasons = {"cb_pattern_loop", "cb_context_pressure", "cb_no_progress", "cb_consecutive_yellow"}
        _is_cb = reason in _cb_reasons or state.cb_trigger_reason in _cb_reasons
        # 如果CB触发但reason为"done"，用存储的CB原因替换
        if not _is_cb and state.cb_trigger_reason:
            _is_cb = True
            reason = state.cb_trigger_reason
        # 如果非正常结束，或“正常结束”时任务图仍有未完成叶子节点，
        # 都应暴露为受阻，避免强制收敛被误标为完成。
        if reason != "done":
            self._mark_unfinished_nodes_blocked(reason)
        elif self._has_unfinished_task_nodes():
            reason = "incomplete_task_graph"
            self._mark_unfinished_nodes_blocked(reason)
        # 完成对话轮次记录（在保存记忆之前，确保 round 已 finalize）
        finalize_status = "completed" if reason == "done" else reason
        self._finalize_dialogue_round(state, status=finalize_status)
        self._auto_save_session_memory(state)
        self._save_runner_state()
        self.session.fc_state = state
        resp = state.last_response_content
        if not resp:
            for m in reversed(state.messages):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    c = m.get("content", "")
                    if c and len(c) > 10:
                        resp = c
                        break
        if not resp:
            resp = self.engine._get_fallback_response(state.user_input_text)
        try:
            from zulong.tools.task_tools import (
                get_active_task_graph,
                _write_final_answer_to_task_graph,
                _auto_archive_completed,
            )
            _tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
            if _tg and hasattr(_tg, "metadata"):
                _tg.metadata["total_turns"] = state.fc_turn
                _tg.metadata["duration"] = time.time() - getattr(_tg, "created_at", time.time())
                if reason == "done" and resp:
                    _write_final_answer_to_task_graph(
                        _tg,
                        resp,
                        source="fc_finalize_quality_pass",
                    )
                    _auto_archive_completed(_tg)
                elif resp:
                    root = _tg.get_node("req") if hasattr(_tg, "get_node") else None
                    if root is not None:
                        root.metadata["non_completion_summary"] = str(resp)
                        root.metadata["non_completion_reason"] = reason
                        root.metadata["non_completion_updated_at"] = time.time()
        except Exception as exc:
            logger.debug("[IDEFCRunner] finalize TaskGraph 写回跳过: %s", exc)
        logger.info(f"[IDEFCRunner] 终止: {reason}, turns={state.fc_turn}, len={len(resp or '')}")
        # 清理共享线程池
        try:
            self._model_executor.shutdown(wait=False)
        except Exception:
            pass
        # 推送终止事件到 web 仪表盘
        try:
            self.engine._publish_task_graph_event(
                "agent_done", state.fc_turn, "finalize", f"FC终止: {reason}")
        except Exception:
            pass
        # Web 监控: FC 终止（同步上下文 fire-and-forget）
        phase = "completed" if reason == "done" else ("cancelled" if reason in {"cancelled", "interrupted"} else "blocked")
        # CB阻断时发送progress(blocked)交互（对齐TSD §23.3.6: blocked→progress归一化）
        if _is_cb:
            self._emit_execution_event_sync(
                "progress",
                f"任务受阻: {reason}",
                turn=state.fc_turn,
                event_type="FC_BLOCKED",
                payload={"total_turns": state.fc_turn, "reason": reason},
            )
        interaction = self._build_interaction_payload(
            phase,
            f"FC终止: {reason}",
            state.fc_turn,
            "FC_DONE",
            {
                "total_turns": state.fc_turn,
                "reason": reason,
                "summary": self._build_task_summary_payload(
                    state,
                    reason=reason,
                    final_text=resp or "",
                    memory_changes=self._get_memory_changes_snapshot(),
                ),
            },
        )
        _broadcast_sync("FC_DONE", {
            "protocol_version": "2.0",
            "session_id": self.session.session_id,
            "total_turns": state.fc_turn,
            "reason": reason,
            "phase": phase,
            "message": f"FC终止: {reason}",
            "interaction": interaction,
        })
        # CB原因返回blocked阶段，让ide_server正常发送task_complete（对齐TSD §23.3.6）
        result_phase = "blocked" if _is_cb else "done"
        return IDEFCResult(phase=result_phase, text_response=resp, reason=reason)

    def _get_current_task_graph(self):
        """获取当前会话的活跃TaskGraph实例"""
        try:
            from zulong.tools.task_tools import get_active_task_graph
            return get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
        except Exception:
            return None

    def _has_unfinished_task_nodes(self) -> bool:
        try:
            tg = self._get_current_task_graph()
            if not tg:
                return False
            leaves = [n for n in tg.get_leaf_nodes() if getattr(n, "id", "") != "req"]
            return bool(leaves) and any(
                getattr(n, "status", "") not in ("completed", "skipped")
                for n in leaves
            )
        except Exception:
            return False

    def _mark_unfinished_nodes_blocked(self, reason: str) -> None:
        """FC 循环异常终止时，将所有 in_progress/pending 的叶节点标记为 blocked"""
        try:
            from zulong.tools.task_tools import get_active_task_graph, _save_active_backup
            tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
            if not tg:
                return
            for node in tg.get_leaf_nodes():
                if node.id != "req" and node.status not in ("completed", "skipped", "blocked"):
                    node.status = "blocked"
                    node.result = f"FC循环异常终止: {reason}"
                    logger.info(f"[IDEFCRunner] 标记 blocked: {node.id} ({node.label})")
            try:
                _save_active_backup()
            except Exception:
                pass
        except Exception:
            pass

    def _get_cb_fallback(self, state: IDEFCState) -> str:
        try:
            from zulong.l2.fc_nodes import _synthesize_cleanup_report_from_tool_results
            synthesized = _synthesize_cleanup_report_from_tool_results(
                state.tool_results_buffer,
                state.user_input_text,
            )
            if synthesized:
                return synthesized
        except Exception as exc:
            logger.debug("[IDEFCRunner] 清理汇报兜底生成跳过: %s", exc)
        if state.tool_results_buffer:
            useful = [r["result"][:300] for r in state.tool_results_buffer
                      if r.get("result") and len(r.get("result", "")) > 20
                      and "error" not in r.get("result", "").lower()[:50]
                      and not r.get("result", "").lstrip().startswith(("{", "["))]
            if useful:
                return (
                    "已停止继续调用工具。我根据已经拿到的工具结果整理如下：\n"
                    + "\n".join(useful[:3])
                )
        try:
            from zulong.tools.task_tools import get_active_task_graph as _gtg_fb
            _tg = _gtg_fb()
            if _tg:
                lv = _tg.get_leaf_nodes()
                cp = [n for n in lv if n.status == "completed"]
                uc = [n for n in lv if n.status not in ("completed", "skipped")]
                fb = (
                    "已停止继续调用工具。我根据当前任务图整理进度：\n"
                    f"当前任务「{_tg.title}」进度：{len(cp)}/{len(lv)} 完成。"
                )
                if uc:
                    fb += f"\n下一步：{uc[0].label}。"
                return fb
        except Exception:
            pass
        return self.engine._get_fallback_response(state.user_input_text)

    def _detect_completion_intent(
        self,
        state: IDEFCState,
        resp: str,
        tool_call: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Detect whether the current turn is trying to complete an FC task."""
        tg = self._get_current_task_graph()
        if not tg:
            return {
                "should_check": False,
                "intent": "chat_only",
                "reason": "no_active_task_graph",
                "active_graph_id": "",
                "high_risk": False,
            }

        graph_id = str(getattr(tg, "id", "") or "")
        if tool_call:
            fn = str(tool_call.get("function", {}).get("name", "") or "")
            if fn in {"submit_final_answer", "attempt_completion"}:
                return {
                    "should_check": True,
                    "intent": "completion_tool",
                    "reason": fn,
                    "active_graph_id": graph_id,
                    "high_risk": True,
                }

        if state.cb_force_no_tools or state.cb_trigger_reason or state.tool_call_budget is not None:
            return {
                "should_check": True,
                "intent": "forced_converge",
                "reason": state.cb_trigger_reason or "forced_no_tools_or_budget",
                "active_graph_id": graph_id,
                "high_risk": True,
            }

        resp_text = str(resp or "").strip()
        if not resp_text:
            return {
                "should_check": True,
                "intent": "final_text",
                "reason": "empty_final_text",
                "active_graph_id": graph_id,
                "high_risk": True,
            }

        return {
            "should_check": True,
            "intent": "final_text",
            "reason": "text_without_tool_calls",
            "active_graph_id": graph_id,
            "high_risk": self._task_or_response_high_risk(state, resp_text),
        }

    def _task_or_response_high_risk(self, state: IDEFCState, resp: str) -> bool:
        text = f"{state.user_input_text or ''}\n{resp or ''}".lower()
        return any(term in text for term in (
            "删除", "支付", "财务", "密钥", "secret", "token", "api key",
            "deploy", "生产", "数据库", "迁移", "权限", "安全", "rm -rf",
        ))

    def _compute_node_coverage(self, state: IDEFCState, tg) -> Dict[str, Any]:
        """Compute structured node visibility from AttentionWindow envelopes."""
        covered = set()
        try:
            if self._attn_window:
                cached_visible = {
                    str(node_id)
                    for node_id in getattr(self._attn_window, "_last_visible_node_ids", set()) or set()
                    if str(node_id)
                }
                if cached_visible:
                    covered.update(cached_visible)
                else:
                    current = str(getattr(self._attn_window, "_current_node_id", "") or "")
                    if current:
                        covered.add(current)
                        try:
                            covered.update(
                                str(getattr(node, "id", "") or "")
                                for node in tg.get_ancestor_chain(current)
                            )
                        except Exception:
                            pass
                    for env in getattr(self._attn_window, "envelopes", []) or []:
                        node_id = str(getattr(env, "node_id", "") or "")
                        if node_id:
                            covered.add(node_id)
        except Exception:
            covered = set()

        all_node_ids = {
            n.id for n in getattr(tg, "nodes", [])
            if not str(getattr(n, "id", "")).startswith("crg_")
        }
        uncovered = sorted(all_node_ids - covered) if covered else []
        uncovered_in_progress = []
        if covered:
            for node in getattr(tg, "nodes", []):
                node_id = str(getattr(node, "id", "") or "")
                if (node_id in all_node_ids
                        and getattr(node, "status", "") == "in_progress"
                        and node_id not in covered):
                    uncovered_in_progress.append(node_id)
        ratio = (len(all_node_ids) - len(uncovered)) / max(len(all_node_ids), 1)
        return {
            "covered": sorted(covered & all_node_ids),
            "uncovered": uncovered,
            "coverage_ratio": ratio,
            "uncovered_in_progress": uncovered_in_progress,
        }

    def _pre_completion_verify(self, state: IDEFCState, resp: str) -> Optional[str]:
        """Hard gate before final completion. Returns a concrete nudge or None."""
        if getattr(state, "quality_forced_risk_summary", False):
            return None
        tg = self._get_current_task_graph()
        if not tg:
            return None
        reasons: List[str] = []

        try:
            leaves = [
                n for n in tg.get_leaf_nodes()
                if not str(getattr(n, "id", "")).startswith("crg_")
            ]
            in_progress = [
                n for n in leaves
                if getattr(n, "status", "") == "in_progress"
            ]
            for node in in_progress[:3]:
                reasons.append(f"{tg.get_node_address(node.id)} 仍为 in_progress")
        except Exception:
            pass

        recent = list(getattr(state, "tool_results_buffer", []) or [])[-5:]
        resp_lower = str(resp or "").lower()
        mentions_risk = any(term in resp_lower for term in (
            "失败", "错误", "异常", "未验证", "风险", "受阻",
            "failed", "error", "exception", "unverified", "risk", "blocked",
        ))
        failed_recent = [
            item for item in recent
            if not self._tool_result_success(item)
        ]
        if failed_recent and not mentions_risk:
            names = "、".join(
                self._friendly_tool_name(item.get("tool_name", "工具"))
                for item in failed_recent[:3]
            )
            reasons.append(f"最近工具结果包含失败但最终回复未说明: {names}")

        coverage = self._compute_node_coverage(state, tg)
        if coverage.get("uncovered_in_progress"):
            addrs = [
                tg.get_node_address(nid)
                for nid in coverage["uncovered_in_progress"][:3]
            ]
            reasons.append("当前窗口不可见的进行中节点: " + "、".join(addrs))

        last_mark = next(
            (
                item for item in reversed(getattr(state, "tool_results_buffer", []) or [])
                if item.get("tool_name") == "task_mark_status"
                and (item.get("arguments", {}).get("new_status")
                     or item.get("arguments", {}).get("status")) == "completed"
            ),
            None,
        )
        if last_mark:
            node_id = last_mark.get("arguments", {}).get("node_id") or ""
            node = tg.get_node(node_id) if node_id else None
            if not last_mark.get("success", True) or not node or node.status != "completed":
                addr = tg.get_node_address(node_id) if node_id else "unknown"
                reasons.append(f"最近完成标记未被 TaskGraph 真实确认: {addr}")

        evidence = self._collect_completion_evidence(state, resp)
        for violated in (evidence.get("constraints", {}) or {}).get("violated_constraints", [])[:5]:
            if violated not in reasons:
                reasons.append(violated)

        if not reasons:
            return None
        listed = "\n".join(f"{idx}. {reason}" for idx, reason in enumerate(reasons[:5], 1))
        return (
            "[完成前验证]\n"
            "不能直接结束，必须先处理以下阻塞项:\n"
            f"{listed}\n"
            "请继续 FC 循环，只处理上述问题；如无法继续，请输出 partial/blocked summary 并列明风险。"
        )

    def _collect_completion_evidence(
        self,
        state: IDEFCState,
        resp: str,
        intent: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tg = self._get_current_task_graph()
        task_coverage: Dict[str, Any] = {
            "leaf_status_counts": {},
            "unfinished_leaf_addresses": [],
            "completed_leaf_addresses": [],
        }
        coverage = {}
        if tg:
            leaves = [
                n for n in tg.get_leaf_nodes()
                if not str(getattr(n, "id", "")).startswith("crg_")
            ]
            counts: Dict[str, int] = {}
            unfinished = []
            completed = []
            for node in leaves:
                status = str(getattr(node, "status", "") or "pending")
                counts[status] = counts.get(status, 0) + 1
                addr = tg.get_node_address(node.id)
                if status in ("completed", "skipped"):
                    completed.append(addr)
                else:
                    unfinished.append(addr)
            task_coverage = {
                "leaf_status_counts": counts,
                "unfinished_leaf_addresses": unfinished,
                "completed_leaf_addresses": completed,
                "total_leaves": len(leaves),
            }
            coverage = self._compute_node_coverage(state, tg)

        all_results = list(getattr(state, "tool_results_buffer", []) or [])
        recent = all_results[-10:]
        failed_tools = [
            item for item in all_results
            if not self._tool_result_success(item)
        ]
        empty_tools = [
            item for item in all_results
            if not str(item.get("result", "") or "").strip()
            or str(item.get("result", "") or "").strip() in {"{}", "[]", "null", "None"}
        ]
        capability_by_name = self._tool_capability_map(getattr(state, "tool_definitions", []) or [])
        write_ops = [
            item for item in all_results
            if self._tool_result_has_capability(item, "file_write", capability_by_name)
        ]
        verify_ops = [
            item for item in all_results
            if self._tool_result_has_capability(item, "verification", capability_by_name)
        ]
        tests_run = []
        for item in verify_ops:
            args_text = _json.dumps(item.get("arguments", {}), ensure_ascii=False).lower()
            if any(term in args_text for term in ("test", "pytest", "npm run", "tsc", "lint", "build", "检查", "测试")):
                tests_run.append(item)

        hard_constraints = self._extract_lightweight_constraints(state.user_input_text)
        target_constraints = self._extract_target_constraints(state.user_input_text)
        if tg:
            meta = getattr(tg, "metadata", {}) or {}
            target_constraints["selected_workspace_dir"] = str(
                meta.get("workspace_dir") or meta.get("workspace_path") or ""
            )
            target_constraints["path_roles"] = meta.get("path_roles") or []
            target_constraints["path_candidates"] = meta.get("path_candidates") or []
        written_paths = self._collect_written_paths(write_ops)
        command_evidence = self._collect_command_evidence(verify_ops)
        violated_constraints = self._target_constraint_violations(
            target_constraints,
            written_paths,
            command_evidence,
        )
        latest_commands: Dict[str, Dict[str, Any]] = {}
        for cmd in command_evidence:
            key = " ".join(str(cmd.get("command") or "").lower().split()) or str(cmd.get("cwd") or "")
            latest_commands[key] = cmd
        failed_commands_uncovered = [
            cmd.get("command", "")
            for cmd in latest_commands.values()
            if cmd.get("status") == "failed"
        ][:5]
        taskspec_coverage = self._collect_taskspec_coverage(
            tg,
            state.user_input_text,
            all_results,
            getattr(state, "tool_definitions", []) or [],
        )
        if taskspec_coverage.get("missing_required_evidence"):
            for missing in taskspec_coverage.get("missing_required_evidence", [])[:5]:
                if missing not in violated_constraints:
                    violated_constraints.append(missing)
        mentioned_constraints = [
            c for c in hard_constraints
            if c.lower() in str(resp or "").lower()
        ]

        return {
            "intent": intent or {},
            "task_coverage": task_coverage,
            "verification": {
                "write_ops": [i.get("tool_name") for i in write_ops],
                "verify_ops": [i.get("tool_name") for i in verify_ops],
                "tests_run": [i.get("tool_name") for i in tests_run],
                "has_write_ops": bool(write_ops),
                "has_verify_ops": bool(verify_ops),
                "has_tests_run": bool(tests_run),
                "written_paths": written_paths,
                "commands": command_evidence,
            },
            "tool_handling": {
                "recent_tool_results": [i.get("tool_name") for i in recent],
                "failed_tool_results": [i.get("tool_name") for i in failed_tools],
                "empty_tool_results": [i.get("tool_name") for i in empty_tools],
            },
            "constraints": {
                "hard_constraints": hard_constraints,
                "mentioned_constraints": mentioned_constraints,
                "violated_constraints": violated_constraints,
                "target_paths": target_constraints.get("target_paths", []),
                "selected_workspace_dir": target_constraints.get("selected_workspace_dir", ""),
            },
            "taskspec_coverage": taskspec_coverage,
            "completion_evidence": {
                "target_paths": target_constraints.get("target_paths", []),
                "selected_workspace_dir": target_constraints.get("selected_workspace_dir", ""),
                "written_paths": written_paths,
                "commands": command_evidence,
                "failed_commands_uncovered": failed_commands_uncovered,
                "taskspec_coverage": taskspec_coverage,
            },
            "risks": {
                "response_mentions_risk": any(term in str(resp or "").lower() for term in (
                    "风险", "未验证", "失败", "受阻", "risk", "unverified", "failed", "blocked",
                )),
                "high_risk": bool((intent or {}).get("high_risk")),
            },
            "attention": coverage,
            "memory_closure": {
                "memory_changes": self._get_memory_changes_snapshot(),
                "task_nodes_synced": None,
            },
        }

    @staticmethod
    def _extract_target_constraints(user_text: str) -> Dict[str, Any]:
        import re as _re

        text = str(user_text or "")
        win_paths = _re.findall(r"[A-Za-z]:[\\/][^\s，。；;,\n\r\"'`]+", text)
        posix_paths = _re.findall(r"/(?:[^\s，。；;,\n\r\"'`]+)", text)
        target_paths: List[str] = []
        for path in [*win_paths, *posix_paths]:
            clean = path.strip().rstrip("。；;,，")
            if clean and clean not in target_paths:
                target_paths.append(clean)
        return {
            "target_paths": target_paths[:6],
            "selected_workspace_dir": "",
        }

    @staticmethod
    def _normalize_path_text(value: Any) -> str:
        return str(value or "").replace("\\", "/").rstrip("/").lower()

    @staticmethod
    def _path_is_absolute_text(value: Any) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        try:
            return Path(text).is_absolute()
        except Exception:
            return bool(re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("/"))

    @classmethod
    def _path_inside_text(cls, path_value: Any, workspace_value: Any) -> bool:
        path_text = str(path_value or "").strip()
        workspace_text = str(workspace_value or "").strip()
        if not path_text or not workspace_text:
            return False
        try:
            Path(path_text).resolve().relative_to(Path(workspace_text).resolve())
            return True
        except Exception:
            path_norm = cls._normalize_path_text(path_text)
            workspace_norm = cls._normalize_path_text(workspace_text)
            return bool(path_norm and workspace_norm and path_norm.startswith(workspace_norm + "/"))

    @classmethod
    def _collect_written_paths(cls, write_ops: List[Dict[str, Any]]) -> List[str]:
        paths: List[str] = []
        for item in write_ops:
            args = item.get("arguments", {}) if isinstance(item, dict) else {}
            if not isinstance(args, dict):
                continue
            for key in ("path", "file_path", "target_path", "workspace_path"):
                value = str(args.get(key) or "").strip()
                if value and value not in paths:
                    paths.append(value)
            result = item.get("result")
            parsed: Any = None
            if isinstance(result, dict):
                parsed = result
            elif isinstance(result, str) and result.strip().startswith(("{", "[")):
                try:
                    parsed = _json.loads(result)
                except Exception:
                    parsed = None
            if isinstance(parsed, dict):
                data = parsed.get("data") if isinstance(parsed.get("data"), dict) else parsed
                for key in ("resolved_path", "file_path", "workspace_path", "cwd"):
                    value = str(data.get(key) or "").strip()
                    if value and value not in paths:
                        paths.append(value)
        return paths[:20]

    @staticmethod
    def _tool_result_text(item: Dict[str, Any]) -> str:
        try:
            result = item.get("result", "")
            if isinstance(result, (dict, list)):
                return _json.dumps(result, ensure_ascii=False)
            return str(result or "")
        except Exception:
            return ""

    @classmethod
    def _collect_taskspec_coverage(
        cls,
        tg: Any,
        user_text: str,
        tool_results: List[Dict[str, Any]],
        tool_definitions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Generic TaskSpec coverage gate built on existing TaskGraph/evidence.

        This intentionally avoids project-specific terms.  It asks only whether
        the original task shape represented in TaskGraph has corresponding
        execution evidence: completed leaf nodes have non-trivial results,
        write-oriented nodes have write evidence, validation-oriented nodes have
        command/read evidence, and failed commands are not left uncovered.
        """
        coverage: Dict[str, Any] = {
            "source": "task_graph_and_tool_evidence",
            "checked": False,
            "node_count": 0,
            "missing_required_evidence": [],
            "covered_nodes": [],
            "uncovered_nodes": [],
        }
        if tg is None:
            return coverage
        leaves = [
            n for n in getattr(tg, "get_leaf_nodes", lambda: [])()
            if not str(getattr(n, "id", "")).startswith("crg_")
        ]
        coverage["checked"] = True
        coverage["node_count"] = len(leaves)
        if not leaves:
            return coverage

        capability_by_name = cls._tool_capability_map(tool_definitions)
        write_evidence = [
            item for item in tool_results
            if cls._tool_result_has_capability(item, "file_write", capability_by_name)
            and cls._tool_result_success(item)
        ]
        verify_evidence = [
            item for item in tool_results
            if cls._tool_result_has_capability(item, "verification", capability_by_name)
            and cls._tool_result_success(item)
        ]
        failed_evidence = [
            item for item in tool_results
            if not cls._tool_result_success(item)
        ]
        all_text = "\n".join(
            [str(user_text or "")]
            + [
                str(item.get("tool_name", ""))
                + " "
                + _json.dumps(item.get("arguments", {}), ensure_ascii=False, default=str)
                + " "
                + cls._tool_result_text(item)[:1200]
                for item in tool_results
            ]
        ).lower()

        missing: List[str] = []
        for node in leaves:
            node_id = str(getattr(node, "id", "") or "")
            label = str(getattr(node, "label", "") or "")
            desc = str(getattr(node, "desc", "") or "")
            result = str(getattr(node, "result", "") or "").strip()
            status = str(getattr(node, "status", "") or "")
            node_text = f"{label} {desc}".lower()
            node_report = {
                "node_id": node_id,
                "label": label,
                "status": status,
                "has_result": bool(result),
            }
            if status not in {"completed", "skipped"}:
                missing.append(f"TaskSpec 节点未完成: {node_id} {label}")
                coverage["uncovered_nodes"].append(node_report)
                continue
            if status == "completed" and len(result) < 20:
                missing.append(f"TaskSpec 节点完成结果证据不足: {node_id} {label}")
                coverage["uncovered_nodes"].append(node_report)
                continue

            # Generic stage signals, not tied to a project or fixed task number.
            needs_write = bool(
                re.search(r"\b(write|create|generate|implement|edit|modify|file|code)\b", node_text)
                or any(token in node_text for token in ("写", "创建", "生成", "实现", "修改", "文件", "代码"))
            )
            needs_verify = bool(
                re.search(r"\b(test|verify|validate|check|build|lint|run)\b", node_text)
                or any(token in node_text for token in ("测试", "验证", "检查", "构建", "运行", "端到端", "集成"))
            )
            node_tokens = [
                token.lower()
                for token in re.findall(r"[A-Za-z0-9_\-.]{3,}", f"{label} {desc} {result}")
            ][:12]
            text_mentions_node = any(token in all_text for token in node_tokens)
            if needs_write and not (write_evidence or text_mentions_node):
                missing.append(f"TaskSpec 写入类节点缺少工具落盘证据: {node_id} {label}")
                coverage["uncovered_nodes"].append(node_report)
                continue
            if needs_verify and not (verify_evidence or text_mentions_node):
                missing.append(f"TaskSpec 验证类节点缺少验证证据: {node_id} {label}")
                coverage["uncovered_nodes"].append(node_report)
                continue
            coverage["covered_nodes"].append(node_report)

        if failed_evidence:
            missing.append(
                "TaskSpec 存在未覆盖失败工具结果: "
                + "、".join(str(item.get("tool_name") or "工具") for item in failed_evidence[:3])
            )
        coverage["missing_required_evidence"] = missing[:12]
        coverage["write_evidence_count"] = len(write_evidence)
        coverage["verify_evidence_count"] = len(verify_evidence)
        coverage["failed_evidence_count"] = len(failed_evidence)
        return coverage

    def _collect_command_evidence(self, verify_ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        commands: List[Dict[str, Any]] = []
        default_cwd = str(getattr(self, "cwd", "") or "")
        for item in verify_ops:
            tool_name = str(item.get("tool_name", ""))
            if tool_name not in {"exec_run_command", "execute_command", "browser_action"}:
                continue
            args = item.get("arguments", {}) if isinstance(item, dict) else {}
            if not isinstance(args, dict):
                args = {}
            command = str(args.get("command") or args.get("cmd") or "").strip()
            cwd = str(args.get("cwd") or args.get("workspace_path") or default_cwd).strip()
            result_text = str(item.get("result", "") or "")
            success = self._tool_result_success(item)
            exit_code = 0 if success else 1
            try:
                parsed = _json.loads(result_text)
                if isinstance(parsed, dict):
                    data = parsed.get("data") if isinstance(parsed.get("data"), dict) else parsed
                    if isinstance(data.get("returncode"), int):
                        exit_code = data["returncode"]
                        success = exit_code == 0
                    elif isinstance(data.get("exit_code"), int):
                        exit_code = data["exit_code"]
                        success = exit_code == 0
            except Exception:
                pass
            commands.append({
                "cwd": cwd,
                "command": command or tool_name,
                "exit_code": exit_code,
                "status": "succeeded" if success else "failed",
                "covers_root": bool(default_cwd and self._normalize_path_text(cwd) == self._normalize_path_text(default_cwd)),
                "covers_target": False,
            })
        return commands[:20]

    @classmethod
    def _target_constraint_violations(
        cls,
        target_constraints: Dict[str, Any],
        written_paths: List[str],
        command_evidence: List[Dict[str, Any]],
    ) -> List[str]:
        violations: List[str] = []
        selected_workspace = str(target_constraints.get("selected_workspace_dir") or "").strip()
        if selected_workspace:
            outside_writes = [
                path for path in written_paths
                if cls._path_is_absolute_text(path)
                and not cls._path_inside_text(path, selected_workspace)
            ]
            if outside_writes:
                violations.append(
                    "写入路径与当前 TaskGraph 工作区不一致: "
                    + "、".join(str(path)[:120] for path in outside_writes[:3])
                )
            outside_commands = [
                cmd for cmd in command_evidence
                if str(cmd.get("cwd") or "").strip()
                and cls._path_is_absolute_text(cmd.get("cwd"))
                and not cls._path_inside_text(cmd.get("cwd"), selected_workspace)
            ]
            if outside_commands:
                violations.append(
                    "命令执行目录与当前 TaskGraph 工作区不一致: "
                    + "、".join(str(cmd.get("cwd") or "")[:120] for cmd in outside_commands[:3])
                )
        latest_by_command: Dict[str, Dict[str, Any]] = {}
        for cmd in command_evidence:
            key = " ".join(str(cmd.get("command") or "").lower().split()) or str(cmd.get("cwd") or "")
            latest_by_command[key] = cmd
        failed_commands = [cmd for cmd in latest_by_command.values() if cmd.get("status") == "failed"]
        if failed_commands:
            violations.append("存在失败命令未被后续成功命令覆盖: " + "、".join(
                str(cmd.get("command") or "命令")[:80] for cmd in failed_commands[:3]
            ))
        return violations[:8]

    @staticmethod
    def _extract_lightweight_constraints(user_text: str) -> List[str]:
        text = str(user_text or "")
        constraints = []
        for token in ("不要", "必须", "只能", "不超过", "最多", "路径", "TSD", "中文", "英文"):
            if token in text:
                constraints.append(token)
        return constraints[:8]

    def _score_completion_quality(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        task = evidence.get("task_coverage", {}) or {}
        verification = evidence.get("verification", {}) or {}
        handling = evidence.get("tool_handling", {}) or {}
        constraints = evidence.get("constraints", {}) or {}
        risks = evidence.get("risks", {}) or {}
        attention = evidence.get("attention", {}) or {}
        taskspec = evidence.get("taskspec_coverage", {}) or {}

        reasons: List[str] = []
        risk_reasons: List[str] = []

        total = int(task.get("total_leaves", 0) or 0)
        unfinished = list(task.get("unfinished_leaf_addresses", []) or [])
        if total == 0:
            task_score = 0.75
            risk_reasons.append("未发现可核验的 TaskGraph 叶子节点")
        elif not unfinished:
            task_score = 1.0
        else:
            done = len(task.get("completed_leaf_addresses", []) or [])
            task_score = max(0.0, min(0.8, done / max(total, 1)))
            reasons.append(f"仍有 {len(unfinished)} 个任务叶子未完成")

        if attention.get("uncovered_in_progress"):
            task_score = min(task_score, 0.55)
            reasons.append("存在当前窗口不可见的进行中节点")

        if verification.get("has_write_ops"):
            if verification.get("has_tests_run"):
                verify_score = 1.0
            elif verification.get("has_verify_ops"):
                verify_score = 0.75
                risk_reasons.append("已有核验工具结果，但未发现明确测试/构建证据")
            else:
                verify_score = 0.35
                reasons.append("检测到写入/修改操作，但缺少后续验证证据")
        else:
            verify_score = 0.85

        failed = list(handling.get("failed_tool_results", []) or [])
        empty = list(handling.get("empty_tool_results", []) or [])
        if failed:
            handling_score = 0.35 if not risks.get("response_mentions_risk") else 0.65
            reasons.append("存在失败工具结果未完全消化")
        elif empty:
            handling_score = 0.75
            risk_reasons.append("存在空工具结果，需要确认是否影响结论")
        else:
            handling_score = 1.0

        violated = list(constraints.get("violated_constraints", []) or [])
        constraint_score = 0.4 if violated else 1.0
        for item in violated[:3]:
            reasons.append(f"用户硬约束可能未满足: {item}")

        taskspec_missing = list(taskspec.get("missing_required_evidence", []) or [])
        if taskspec_missing:
            taskspec_score = 0.35
            for item in taskspec_missing[:3]:
                reasons.append(f"原始任务要求覆盖不足: {item}")
        else:
            taskspec_score = 1.0 if taskspec.get("checked") else 0.85

        failed_commands_uncovered = list(
            (evidence.get("completion_evidence", {}) or {}).get("failed_commands_uncovered", []) or []
        )
        if failed_commands_uncovered:
            verify_score = min(verify_score, 0.2)
            handling_score = min(handling_score, 0.25)
            for command in failed_commands_uncovered[:3]:
                reasons.append(f"存在失败验证命令未被后续成功覆盖: {command}")

        has_known_risk = bool(reasons or risk_reasons or failed or unfinished)
        if has_known_risk and not risks.get("response_mentions_risk"):
            risk_score = 0.45
            if not reasons:
                risk_reasons.append("存在未完全核验事项，但最终回复未主动说明")
        else:
            risk_score = 1.0

        dimensions = {
            "task_coverage": task_score,
            "taskspec_coverage": taskspec_score,
            "verification_evidence": verify_score,
            "tool_result_handling": handling_score,
            "constraint_alignment": constraint_score,
            "risk_transparency": risk_score,
        }
        score = (
            task_score * 0.24
            + taskspec_score * 0.18
            + verify_score * 0.22
            + handling_score * 0.17
            + constraint_score * 0.11
            + risk_score * 0.08
        )
        if score >= 0.80:
            level = "pass"
        elif score >= 0.60:
            level = "warn"
        else:
            level = "iterate"
        if failed_commands_uncovered:
            level = "blocked"
            score = min(score, 0.49)
        return {
            "score": round(score, 3),
            "level": level,
            "dimensions": dimensions,
            "blocking_reasons": reasons[:8],
            "risk_reasons": risk_reasons[:8],
        }

    def _build_quality_review_nudge(
        self,
        evidence: Dict[str, Any],
        quality: Dict[str, Any],
    ) -> str:
        reasons = list(quality.get("blocking_reasons") or [])
        if not reasons:
            reasons = list(quality.get("risk_reasons") or [])
        if not reasons:
            reasons = ["缺少足够结构化证据确认任务已完成"]
        listed = "\n".join(f"{idx}. {reason}" for idx, reason in enumerate(reasons[:5], 1))
        return (
            "[质量复核]\n"
            f"当前完成质量分: {quality.get('score', 0):.2f}，未达到 0.80。\n"
            "必须先处理:\n"
            f"{listed}\n"
            "请继续 FC 循环，只处理上述阻塞项；不要重新规划无关任务。"
        )

    def _maybe_quality_iterate(
        self,
        state: IDEFCState,
        evidence: Dict[str, Any],
        quality: Dict[str, Any],
    ) -> Optional[str]:
        state.quality_last_score = float(quality.get("score", 0.0) or 0.0)
        state.quality_last_reasons = list(
            quality.get("blocking_reasons") or quality.get("risk_reasons") or []
        )
        if getattr(state, "quality_forced_risk_summary", False):
            return None
        if quality.get("level") in {"pass", "warn"}:
            return None

        review_key = _json.dumps({
            "unfinished": evidence.get("task_coverage", {}).get("unfinished_leaf_addresses", [])[:5],
            "failed": evidence.get("tool_handling", {}).get("failed_tool_results", [])[:5],
            "empty": evidence.get("tool_handling", {}).get("empty_tool_results", [])[:5],
            "score": quality.get("score"),
        }, ensure_ascii=False, sort_keys=True)

        if state.quality_iteration_count >= 2 or (
            review_key == getattr(state, "quality_last_review_key", "")
            and state.quality_iteration_count >= 1
        ):
            state.quality_forced_risk_summary = True
            return (
                "[质量复核] 已达到迭代上限或复核问题未变化。"
                "请输出 partial/blocked summary，明确列出完成项、未完成项、未验证项和风险。"
            )

        state.quality_iteration_count += 1
        state.quality_last_review_key = review_key
        return self._build_quality_review_nudge(evidence, quality)

    def _quality_level_from_state(self, state: IDEFCState) -> str:
        if getattr(state, "quality_forced_risk_summary", False):
            return "blocked"
        score = float(getattr(state, "quality_last_score", 1.0) or 0.0)
        if score >= 0.80:
            return "pass"
        if score >= 0.60:
            return "warn"
        return "iterate"

    def _quality_risk_appendix(self, quality: Dict[str, Any]) -> str:
        reasons = list(quality.get("risk_reasons") or quality.get("blocking_reasons") or [])
        if not reasons:
            return ""
        listed = "；".join(str(r) for r in reasons[:3])
        return f"\n\n[质量复核] 当前质量等级为 warn，需注意: {listed}"

    @staticmethod
    def _parse_reviewer_json(text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        if not raw:
            return {}
        try:
            data = _json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = _json.loads(raw[start:end + 1])
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

    def _maybe_run_completion_reviewer(
        self,
        state: IDEFCState,
        evidence: Dict[str, Any],
        quality: Dict[str, Any],
        resp: str,
    ) -> Optional[str]:
        """Use L2-BACKUP as an optional structured reviewer for high-risk finishes."""
        score = float(quality.get("score", 1.0) or 0.0)
        if score >= 0.80:
            state.quality_reviewer_health = {
                "status": "skipped_high_local_score",
                "score": score,
            }
            return None
        if getattr(state, "quality_reviewer_count", 0) >= 1:
            state.quality_reviewer_health = {
                "status": "skipped_limit_reached",
                "score": score,
                "count": getattr(state, "quality_reviewer_count", 0),
            }
            return None
        risks = evidence.get("risks", {}) or {}
        intent = evidence.get("intent", {}) or {}
        task = evidence.get("task_coverage", {}) or {}
        handling = evidence.get("tool_handling", {}) or {}
        is_complex = int(task.get("total_leaves", 0) or 0) >= 6
        is_high_risk = bool(
            risks.get("high_risk")
            or intent.get("high_risk")
            or handling.get("failed_tool_results")
            or is_complex
        )
        if not is_high_risk:
            state.quality_reviewer_health = {
                "status": "skipped_not_high_risk",
                "score": score,
                "total_leaves": int(task.get("total_leaves", 0) or 0),
            }
            return None
        engine = getattr(self, "engine", None)
        backup_client = getattr(engine, "backup_client", None)
        if not backup_client:
            state.quality_reviewer_health = {
                "status": "not_configured",
                "reason": "backup_client_missing",
                "score": score,
            }
            logger.info(
                "[IDEFCRunner][QualityReviewer] L2-BACKUP 外审未配置: backup_client_missing"
            )
            return None
        LLM_MODEL_ID_BACKUP = (
            getattr(engine, "backup_model_id", None)
            or getattr(engine, "l2_backup_model_id", None)
            or getattr(backup_client, "model", None)
            or getattr(backup_client, "model_name", None)
        )
        if not LLM_MODEL_ID_BACKUP:
            try:
                import sys

                container_mod = sys.modules.get("zulong.models.container")
                LLM_MODEL_ID_BACKUP = getattr(container_mod, "LLM_MODEL_ID_BACKUP", "") if container_mod else ""
            except Exception:
                LLM_MODEL_ID_BACKUP = ""
        if not LLM_MODEL_ID_BACKUP and backup_client:
            LLM_MODEL_ID_BACKUP = "backup-model"
        if not LLM_MODEL_ID_BACKUP:
            state.quality_reviewer_health = {
                "status": "not_configured",
                "reason": "backup_model_id_missing",
                "score": score,
            }
            logger.info(
                "[IDEFCRunner][QualityReviewer] L2-BACKUP 外审未配置: backup_model_id_missing"
            )
            return None

        state.quality_reviewer_count += 1
        payload = {
            "local_quality": quality,
            "evidence": evidence,
            "final_response_preview": str(resp or "")[:3000],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "你是祖龙 FC 完成质量外审器。只依据用户任务、结构化证据和最终回复预览评审；"
                    "不得调用工具，不得替代最终答案。只输出 JSON 对象，字段为 "
                    "score_delta, missed_constraints, missing_evidence, suggested_next_actions。"
                ),
            },
            {
                "role": "user",
                "content": _json.dumps(payload, ensure_ascii=False, default=str),
            },
        ]
        try:
            review = backup_client.chat.completions.create(
                model=LLM_MODEL_ID_BACKUP,
                messages=messages,
                max_tokens=800,
                temperature=0.0,
                stream=False,
            )
            text = review.choices[0].message.content or ""
            data = self._parse_reviewer_json(text)
            if not data:
                state.quality_reviewer_health = {
                    "status": "non_json",
                    "raw_length": len(text),
                    "raw_preview": _safe_error_summary(text, 240),
                    "score": score,
                }
                logger.info(
                    "[IDEFCRunner][QualityReviewer] L2-BACKUP 返回非 JSON，降级本地评分: len=%s preview=%s",
                    len(text),
                    _safe_error_summary(text, 160),
                )
                state.quality_last_reviewer = {"raw": text[:1000]}
                return None
            state.quality_last_reviewer = data
        except Exception as exc:
            state.quality_reviewer_health = {
                "status": "failed",
                "error": _safe_error_summary(exc),
                "score": score,
            }
            logger.info(
                "[IDEFCRunner][QualityReviewer] L2-BACKUP 外审失败，降级本地评分: %s",
                exc,
            )
            return None

        reasons: List[str] = []
        for key in ("missed_constraints", "missing_evidence", "suggested_next_actions"):
            values = data.get(key) or []
            if isinstance(values, str):
                values = [values]
            for value in values[:3]:
                if value:
                    reasons.append(str(value))
        try:
            score_delta = float(data.get("score_delta", 0.0) or 0.0)
        except Exception:
            score_delta = 0.0
        if not reasons and score_delta >= 0:
            state.quality_reviewer_health = {
                "status": "pass",
                "score": score,
                "score_delta": score_delta,
            }
            logger.info(
                "[IDEFCRunner][QualityReviewer] L2-BACKUP 外审未要求继续: score_delta=%.3f",
                score_delta,
            )
            return None
        if not reasons:
            reasons.append("外审认为当前完成质量仍低于本地评分，需要补充验证或风险说明")

        reviewer_reasons = [f"L2-BACKUP: {reason}" for reason in reasons[:5]]
        merged = list(getattr(state, "quality_last_reasons", []) or [])
        for reason in reviewer_reasons:
            if reason not in merged:
                merged.append(reason)
        state.quality_last_reasons = merged[:8]

        listed = "\n".join(f"{idx}. {reason}" for idx, reason in enumerate(reasons[:5], 1))
        state.quality_reviewer_health = {
            "status": "iterate",
            "score": score,
            "score_delta": score_delta,
            "reason_count": len(reasons),
        }
        logger.info(
            "[IDEFCRunner][QualityReviewer] L2-BACKUP 要求继续迭代: score_delta=%.3f reasons=%s",
            score_delta,
            len(reasons),
        )
        return (
            "[质量外审]\n"
            "L2-BACKUP 结构化复核认为当前完成仍需处理:\n"
            f"{listed}\n"
            "请继续 FC 循环，只补齐上述证据或风险说明；不要让外审内容直接替代最终答案。"
        )

    def _completion_quality_gate(
        self,
        state: IDEFCState,
        resp: str,
    ) -> Tuple[Optional[str], str]:
        """Run hard completion gate and local quality scoring before final done."""
        intent = self._detect_completion_intent(state, resp)
        if not intent.get("should_check"):
            return None, resp

        verify_hint = self._pre_completion_verify(state, resp)
        if verify_hint and state.null_response_count < 3:
            assistant_msg = {"role": "assistant", "content": resp}
            user_msg = internal_control_message(verify_hint)
            state.messages.append(assistant_msg)
            state.messages.append(user_msg)
            if self._attn_window:
                self._attn_window.register_message(assistant_msg, turn=state.fc_turn)
                self._attn_window.register_message(user_msg, turn=state.fc_turn)
            state.null_response_count += 1
            logger.info(
                "[IDEFCRunner][CompletionGate] hard gate blocked: %s",
                verify_hint[:180],
            )
            self._log_fc_decision_path(
                state,
                path="completion_gate_hard_blocked",
                tool_calls=[],
                response_content=resp,
                root_cause="quality_gate_loop",
                verify_hint=_safe_error_summary(verify_hint),
            )
            return ("cb_force" if state.null_response_count >= 4 else "continue"), resp
        if verify_hint:
            state.quality_forced_risk_summary = True

        evidence = self._collect_completion_evidence(state, resp, intent)
        quality = self._score_completion_quality(evidence)
        try:
            state.completion_last_evidence = evidence
            state.completion_last_quality = quality
        except Exception:
            pass
        state.quality_last_score = float(quality.get("score", 0.0) or 0.0)
        state.quality_last_reasons = list(
            quality.get("blocking_reasons") or quality.get("risk_reasons") or []
        )
        if getattr(state, "quality_forced_risk_summary", False) and quality.get("level") != "pass":
            quality["level"] = "blocked"
            forced_note = (
                "[质量复核] 已达到迭代上限，请输出 partial/blocked summary，"
                "明确列出完成项、未完成项、未验证项和风险。"
            )
            if forced_note not in resp:
                resp = str(resp or "").rstrip() + "\n\n" + forced_note

        reviewer_hint = self._maybe_run_completion_reviewer(state, evidence, quality, resp)
        if reviewer_hint:
            assistant_msg = {"role": "assistant", "content": resp}
            user_msg = internal_control_message(reviewer_hint)
            state.messages.append(assistant_msg)
            state.messages.append(user_msg)
            if self._attn_window:
                self._attn_window.register_message(assistant_msg, turn=state.fc_turn)
                self._attn_window.register_message(user_msg, turn=state.fc_turn)
            state.null_response_count += 1
            logger.info(
                "[IDEFCRunner][CompletionGate] reviewer iterate score=%.3f",
                state.quality_last_score,
            )
            self._log_fc_decision_path(
                state,
                path="completion_gate_reviewer_iterate",
                tool_calls=[],
                response_content=resp,
                score=state.quality_last_score,
            )
            return ("cb_force" if state.null_response_count >= 4 else "continue"), resp

        iterate_hint = self._maybe_quality_iterate(state, evidence, quality)
        if iterate_hint:
            assistant_msg = {"role": "assistant", "content": resp}
            user_msg = internal_control_message(iterate_hint)
            state.messages.append(assistant_msg)
            state.messages.append(user_msg)
            if self._attn_window:
                self._attn_window.register_message(assistant_msg, turn=state.fc_turn)
                self._attn_window.register_message(user_msg, turn=state.fc_turn)
            state.null_response_count += 1
            logger.info(
                "[IDEFCRunner][CompletionGate] quality iterate score=%.3f",
                state.quality_last_score,
            )
            self._log_fc_decision_path(
                state,
                path="completion_gate_quality_iterate",
                tool_calls=[],
                response_content=resp,
                root_cause="quality_gate_loop",
                score=state.quality_last_score,
                quality_level=quality.get("level"),
            )
            return ("cb_force" if state.null_response_count >= 4 else "continue"), resp

        if quality.get("level") == "blocked":
            state.cb_trigger_reason = state.cb_trigger_reason or "quality_blocked"
            logger.info(
                "[IDEFCRunner][CompletionGate] blocked summary score=%.3f",
                state.quality_last_score,
            )
            self._log_fc_decision_path(
                state,
                path="completion_gate_blocked_summary",
                tool_calls=[],
                response_content=resp,
                root_cause="quality_blocked",
                score=state.quality_last_score,
                quality_level=quality.get("level"),
            )
            return None, resp

        if quality.get("level") == "warn":
            appendix = self._quality_risk_appendix(quality)
            if appendix and appendix not in resp:
                resp = str(resp or "").rstrip() + appendix
        logger.info(
            "[IDEFCRunner][CompletionGate] pass level=%s score=%.3f",
            quality.get("level"),
            state.quality_last_score,
        )
        self._log_fc_decision_path(
            state,
            path=f"completion_gate_{quality.get('level') or 'pass'}",
            tool_calls=[],
            response_content=resp,
            score=state.quality_last_score,
            quality_level=quality.get("level"),
        )
        return None, resp

    def _run_backfill(self, state: IDEFCState, response: str, is_cb_path: bool) -> None:
        try:
            from zulong.tools.task_tools import get_active_task_graph as _gtg_bf, _save_active_backup
            tg = _gtg_bf()
            if not tg:
                return
            jc = sum(1 for c in response if c in '{}[]":,')
            if (jc / max(len(response), 1)) > _JSON_DENSITY_THRESHOLD:
                # 密度超标时用 json.loads 验证是否真的是 JSON
                try:
                    _json.loads(response.strip())
                    return  # 确认为 JSON 结构，跳过 backfill
                except (ValueError, TypeError):
                    pass  # 非 JSON（如 Markdown 表格），继续 backfill
            leaves = tg.get_leaf_nodes()
            unc = [n for n in leaves if n.status not in ("completed", "skipped")]
            if not unc or _looks_like_incomplete_result(response):
                return
            candidates = []
            for nd in unc:
                if _has_content_match(response, nd.label):
                    node_content = _extract_node_content(response, nd.label, 500)
                    if _looks_like_incomplete_result(node_content):
                        logger.info(
                            "[IDEFCRunner][Backfill] 跳过未完成片段: "
                            f"{nd.id}({nd.label})"
                        )
                        continue
                    if hasattr(nd, "metadata"):
                        nd.metadata["backfill_candidate_result"] = node_content
                        nd.metadata["backfill_candidate_at_turn"] = state.fc_turn
                        nd.metadata["backfill_candidate_cb_path"] = bool(is_cb_path)
                    candidates.append(nd.id)
            if candidates:
                try: _save_active_backup()
                except Exception as e:
                    ErrorHandler.handle_exception(e, ErrorCode.CFG_SAVE_FAILED, context={"operation": "save_active_backup"}, log_level=logging.WARNING)
                logger.info(
                    "[IDEFCRunner][Backfill] %s记录候选结果: %s/%s，不自动改 completed",
                    "CB " if is_cb_path else "",
                    len(candidates),
                    len(unc),
                )
                self.engine._publish_task_graph_event(
                    "agent_tool_call", state.fc_turn, "task_backfill",
                    _json.dumps({
                        "candidate_backfill": len(candidates),
                        "total_leaf": len(leaves),
                        "auto_completed": 0,
                    }, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"[IDEFCRunner][Backfill] {e}")

    def _apply_pressure_guidance(self, state: IDEFCState, fc: int) -> None:
        """Apply TSD-aligned context-pressure attention guidance.

        Internal tier detection uses raw context-window usage.  Guidance shown
        to the LLM uses a threshold-normalized pressure view: the active
        threshold budget is 100%.
        """
        if not self._attn_window or state.cb_force_no_tools:
            return  # CB RED ?????????
        if getattr(state, "cb_recovery_stage", "") in {"restricted_recovery", "note_attention"}:
            return
        if getattr(state, "pressure_stage", "") == "restricted_recovery":
            return

        ratio = float(
            getattr(
                self._attn_window,
                "trigger_context_pressure_ratio",
                getattr(
                    self._attn_window,
                    "context_pressure_ratio",
                    getattr(self._attn_window, "usage_ratio", 0.0),
                ),
            )
            or 0.0
        )
        yellow_ratio = 0.90
        red_ratio = 1.0
        threshold_budget_ratio = 0.5
        try:
            attn_cfg = getattr(self._attn_window, "_llm_config", None)
            if attn_cfg:
                yellow_ratio = float(getattr(attn_cfg, "pressure_threshold_medium", yellow_ratio))
                red_ratio = float(getattr(attn_cfg, "pressure_threshold_high", red_ratio))
                threshold_budget_ratio = float(getattr(attn_cfg, "threshold_budget_ratio", threshold_budget_ratio))
            elif self._circuit_breaker:
                cb_cfg = getattr(self._circuit_breaker, "_config", {}) or {}
                yellow_ratio = float(cb_cfg.get("context_yellow_ratio", yellow_ratio))
                red_ratio = float(cb_cfg.get("context_red_ratio", red_ratio))
                threshold_budget_ratio = float(cb_cfg.get("threshold_budget_ratio", threshold_budget_ratio))
        except Exception:
            pass

        if ratio > red_ratio:
            tier = "red"
        elif ratio > yellow_ratio:
            tier = "yellow"
        else:
            tier = "green"
        pressure_view = build_threshold_pressure_view(
            ratio,
            yellow_ratio,
            red_ratio,
            tier=tier,
        )
        self._last_pressure_tier = tier

        if tier == "green":
            if getattr(state, "pressure_stage", "") != "restricted_recovery":
                state.pressure_stage = ""
                state.pressure_force_attention = False
            return

        msgs = state.messages
        task_graph = self._get_current_task_graph()

        # 两段式：即使 high==medium，第一次达阈值也只注入注意力引导。
        if getattr(state, "pressure_stage", "") not in {"yellow_guidance", "restricted_recovery"}:
            acts = self._maybe_run_bfs(fc, trigger="pressure_threshold_guidance")
            parts = [
                (
                    "[上下文压力 - 注意力引导] "
                    f"当前上下文压力已达 {pressure_view.threshold_relative_percent:.0f}%。"
                ),
                "请由 LLM 自主判断是否需要切换 GLOBAL / FOCUS / SINGLE_CHAIN。",
                "动态注意力不是压缩上下文，而是重新选择当前必要上下文，暂排/降权无关上下文。",
                "普通读写/命令/检索工具结果只作为证据，不直接决定模式。",
            ]
            if acts:
                seeds_set = set(self._compute_bfs_seeds())
                candidates = [(nid, score) for nid, score in acts.items() if score > 0.6 and nid not in seeds_set]
                if candidates:
                    top_nid, top_score = max(candidates, key=lambda x: x[1])
                    top_task_node_id = self._task_node_id_from_memory_id(top_nid, task_graph)
                    top_display = (
                        task_graph.get_node_address(top_task_node_id)
                        if task_graph and top_task_node_id
                        else top_nid
                    )
                    parts.append(f"可参考高激活节点：{top_display}（激活={top_score:.2f}），但最终由 LLM 自主选择。")
            hint = {"role": "system", "content": "\n".join(parts)}
            msgs.append(hint)
            self._attn_window.register_message(hint, turn=fc)
            state.pressure_stage = "yellow_guidance"
            state.pressure_attention_context = {
                "tier": tier,
                "ratio": round(float(ratio), 4),
                "yellow_ratio": round(float(yellow_ratio), 4),
                "red_ratio": round(float(red_ratio), 4),
                "context_pressure_ratio": round(pressure_view.context_pressure_ratio, 4),
                "context_pressure_percent": round(pressure_view.context_pressure_percent, 1),
                "threshold_budget_ratio": round(float(threshold_budget_ratio), 4),
                "threshold_budget_percent": round(pressure_view.threshold_relative_percent, 1),
                "active_threshold_ratio": round(pressure_view.active_threshold_ratio, 4),
                "budget_reference": "threshold_budget_is_100_percent",
                "fc_turn": fc,
                "decision_owner": "llm",
                "allowed_modes": ["GLOBAL", "FOCUS", "SINGLE_CHAIN"],
                "first_threshold_response": "guidance_only",
            }
            logger.info(
                "[IDEFCRunner][Pressure] YELLOW %.0f%% of threshold budget: 第一次阈值响应，仅注入动态注意力引导",
                pressure_view.threshold_relative_percent,
            )
            return

        if tier != "red":
            return

        recovery_tools = self._get_cb_recovery_tools(state.tool_definitions)
        requires_note = bool(self._first_recovery_landing_tool(recovery_tools))
        requires_attention = bool(self._first_tool_with_capability(recovery_tools, "attention_switch"))
        state.pressure_stage = "restricted_recovery" if recovery_tools else "guidance_only"
        state.pressure_force_attention = bool(recovery_tools)
        state.pressure_recovery_note_saved = False
        state.pressure_recovery_attention_switched = False
        state.pressure_recovery_requires_note = requires_note
        state.pressure_recovery_requires_attention = requires_attention
        state.pressure_recovery_start_result_count = len(state.tool_results_buffer or [])

        # BFS only provides candidate focus evidence; LLM still owns the choice.
        acts = self._maybe_run_bfs(fc, trigger="pressure_restricted_recovery")
        recommended_focus = ""
        recommended_display = ""
        recommended_score = 0.0

        parts = [
            (
                "[动态注意力 RED 受限恢复] "
                f"当前上下文压力已达 {pressure_view.threshold_relative_percent:.0f}%。"
            ),
            "请先保存当前证据/未完成项/失败原因/下一步建议，再完成一次注意力重选。",
            "1) 调用便签/标签/记忆落盘能力，并关联当前焦点节点。",
            "2) 调用注意力切换能力，由 LLM 选择 GLOBAL / FOCUS / SINGLE_CHAIN，并说明注入/暂排上下文。",
        ]
        if acts:
            seeds_set = set(self._compute_bfs_seeds())
            candidates = [(nid, score) for nid, score in acts.items() if score > 0.4 and nid not in seeds_set]
            if candidates:
                top_nid, top_score = max(candidates, key=lambda x: x[1])
                top_task_node_id = self._task_node_id_from_memory_id(top_nid, task_graph)
                recommended_focus = top_task_node_id or top_nid
                recommended_score = float(top_score)
                top_display = (
                    task_graph.get_node_address(top_task_node_id)
                    if task_graph and top_task_node_id
                    else top_nid
                )
                recommended_display = str(top_display)
                parts.append(f"可参考高激活节点：{top_display}（激活={top_score:.2f}），但最终由 LLM 自主选择。")

        state.pressure_attention_context = {
            "tier": "red",
            "ratio": round(float(ratio), 4),
            "yellow_ratio": round(float(yellow_ratio), 4),
            "red_ratio": round(float(red_ratio), 4),
            "context_pressure_ratio": round(pressure_view.context_pressure_ratio, 4),
            "context_pressure_percent": round(pressure_view.context_pressure_percent, 1),
            "threshold_budget_ratio": round(float(threshold_budget_ratio), 4),
            "threshold_budget_percent": round(pressure_view.threshold_relative_percent, 1),
            "active_threshold_ratio": round(pressure_view.active_threshold_ratio, 4),
            "budget_reference": "threshold_budget_is_100_percent",
            "fc_turn": fc,
            "recommended_focus": recommended_focus,
            "recommended_display": recommended_display,
            "recommended_score": round(recommended_score, 4),
            "decision_owner": "llm",
            "allowed_modes": ["GLOBAL", "FOCUS", "SINGLE_CHAIN"],
            "second_threshold_response": "restricted_recovery" if recovery_tools else "guidance_only",
            "requires_note": requires_note,
            "requires_attention": requires_attention,
        }
        hint = {"role": "system", "content": "\n".join(parts)}
        msgs.append(hint)
        self._attn_window.register_message(hint, turn=fc)
        logger.info(
            "[IDEFCRunner][Pressure] RED %.0f%% of threshold budget: 第二次阈值响应，进入受限恢复 context=%s tools=%s",
            pressure_view.threshold_relative_percent,
            state.pressure_attention_context,
            [td.get("function", {}).get("name", "") for td in recovery_tools],
        )

    @classmethod
    def _get_attention_only_tools(cls, tool_definitions: List[Dict]) -> List[Dict]:
        """兼容旧调用：压力 RED 时仅取注意力切换能力工具。"""
        return [
            td for td in tool_definitions
            if cls._tool_has_capability(td, "attention_switch")
        ]

    def _run_bfs_activation(self, fc_turn: int) -> None:
        try:
            from zulong.memory.memory_graph import get_memory_graph
            from zulong.tools.task_tools import get_active_task_graph
            mg = get_memory_graph()
            tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
            if not (mg and tg):
                return
            try:
                from zulong.memory.graph_adapters import TaskGraphAdapter
                s = TaskGraphAdapter().sync(mg, tg)
                if s:
                    logger.info(f"[IDEFCRunner] TG->MG sync: {s}")
            except Exception:
                pass
            ip = tg.get_nodes_by_status("in_progress")
            if not ip:
                return
            seeds = [f"task:{tg.id}/{n.id}" for n in ip]
            rs = getattr(mg, "_last_retrieved_node_ids", [])
            if rs:
                seeds.extend(rs)
            # CRG 增强: 最近被懒索引触碰的 CODE_SYMBOL 节点也加入 seed
            # 使 BFS 激活能发现代码节点与任务节点之间的语义关联
            try:
                from zulong.memory.memory_graph import NodeType as _NT
                recent_code = [
                    getattr(n, "node_id", "")
                    for n in mg.get_nodes_by_type(_NT.CODE_SYMBOL)
                    if getattr(n, "last_accessed", 0)
                    and (time.time() - getattr(n, "last_accessed", 0)) < 120  # 2 分钟内触碰的
                ]
                if recent_code:
                    seeds.extend(recent_code[:10])  # 限制数量避免 BFS 爆炸
            except Exception:
                pass
            valid, seen = [], set()
            for s in seeds:
                if s not in seen and mg.has_node(s):
                    valid.append(s)
                    seen.add(s)
            if valid:
                _cws = getattr(self.engine, "_context_window_size", 131072) if self.engine else 131072
                _ur = self._attn_window.usage_ratio if self._attn_window else 0.0
                if hasattr(mg, 'compute_activations_dynamic'):
                    acts = mg.compute_activations_dynamic(
                        valid,
                        context_window_size=_cws,
                        usage_ratio=_ur,
                    )
                else:
                    _min_act = 0.05 if len(valid) > 5 else 0.01
                    acts = mg.compute_activations(valid, max_depth=3, decay=0.5,
                                                  min_activation=_min_act)
                if acts:
                    # 记录 BFS 激活得分分布（方便诊断注意力切换行为）
                    top_acts = sorted(acts.items(), key=lambda x: -x[1])[:5]
                    logger.info(
                        f"[IDEFCRunner][BFS] turn={fc_turn} seeds={len(valid)}, "
                        f"activated={len(acts)}, top={top_acts}")
                    fc = mg.get_last_focus_context()
                    cf = fc.get("focused_task_node_id", "") if fc else ""
                    top = max(acts, key=acts.get)
                    top_task_node_id = self._task_node_id_from_memory_id(top, tg)
                    if top != cf and acts[top] > 0.6 and top not in valid:
                        logger.info(
                            f"[IDEFCRunner][BFS] 焦点切换: {cf} → {top} "
                            f"(score={acts[top]:.3f})")
                        mg.update_focus_to_node(top)
                        if self._attn_window and top_task_node_id:
                            self._attn_window.on_navigate_attention(
                                direction="jump", target_node_id=top_task_node_id)
                else:
                    logger.info(
                        f"[IDEFCRunner][BFS] turn={fc_turn} seeds={len(valid)}, "
                        f"无激活结果")
        except Exception as e:
            logger.info(f"[IDEFCRunner] BFS skip: {e}")

    def _compute_bfs_seeds(self) -> List[str]:
        """收集 BFS 种子（纯计算，无副作用）"""
        try:
            from zulong.memory.memory_graph import get_memory_graph
            from zulong.tools.task_tools import get_active_task_graph
            mg = get_memory_graph()
            tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
            if not (mg and tg):
                return []
            ip = tg.get_nodes_by_status("in_progress")
            if not ip:
                return []
            seeds = [f"task:{tg.id}/{n.id}" for n in ip]
            rs = getattr(mg, "_last_retrieved_node_ids", [])
            if rs:
                seeds.extend(rs)
            # CRG 增强: 最近被懒索引触碰的 CODE_SYMBOL 节点
            try:
                from zulong.memory.memory_graph import NodeType as _NT
                recent_code = [
                    getattr(n, "node_id", "")
                    for n in mg.get_nodes_by_type(_NT.CODE_SYMBOL)
                    if getattr(n, "last_accessed", 0)
                    and (time.time() - getattr(n, "last_accessed", 0)) < 120
                ]
                if recent_code:
                    seeds.extend(recent_code[:10])
            except Exception:
                pass
            # 去重 + 验证存在性
            valid, seen = [], set()
            for s in seeds:
                if s not in seen and mg.has_node(s):
                    valid.append(s)
                    seen.add(s)
            return valid
        except Exception:
            return []

    def _maybe_run_bfs(self, fc_turn: int, trigger: str = "tool_complete") -> Optional[Dict[str, float]]:
        """条件执行 BFS，返回激活结果或 None

        trigger: "tool_complete" | "pressure_crossing"
        """
        if fc_turn <= 1:
            return None

        seeds = self._compute_bfs_seeds()
        if not seeds:
            return None

        # 变更检测
        import hashlib
        seeds_hash = hashlib.md5("|".join(sorted(seeds)).encode()).hexdigest()[:8]

        if trigger != "pressure_crossing":
            # 非压力触发：检查种子变更 + 最小间隔
            if seeds_hash == self._last_bfs_seeds_hash:
                return None
            if fc_turn - self._last_bfs_turn < self._bfs_min_interval:
                return None

        # TG→MG 同步（仅在 BFS 实际执行时）
        try:
            from zulong.memory.memory_graph import get_memory_graph
            from zulong.tools.task_tools import get_active_task_graph
            from zulong.memory.graph_adapters import TaskGraphAdapter
            mg = get_memory_graph()
            tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
            if mg and tg:
                s = TaskGraphAdapter().sync(mg, tg)
                if s:
                    logger.info(f"[IDEFCRunner] TG->MG sync: {s}")
        except Exception:
            pass

        # 执行 BFS
        from zulong.memory.memory_graph import get_memory_graph
        mg = get_memory_graph()
        if not mg:
            return None
        _min_act = 0.05 if len(seeds) > 5 else 0.01
        _cws = getattr(self.engine, "_context_window_size", 131072) if self.engine else 131072
        _ur = self._attn_window.usage_ratio if self._attn_window else 0.0
        if hasattr(mg, 'compute_activations_dynamic'):
            acts = mg.compute_activations_dynamic(
                seeds,
                context_window_size=_cws,
                usage_ratio=_ur,
            )
        else:
            acts = mg.compute_activations(seeds, max_depth=3, decay=0.5,
                                          min_activation=_min_act)

        self._last_bfs_seeds_hash = seeds_hash
        self._last_bfs_turn = fc_turn

        # 日志
        if acts:
            top_acts = sorted(acts.items(), key=lambda x: -x[1])[:5]
            logger.info(
                f"[IDEFCRunner][BFS] turn={fc_turn} seeds={len(seeds)}, "
                f"activated={len(acts)}, top={top_acts}")

        return acts

    def _build_subtask_context(self) -> Optional[Dict]:
        try:
            from zulong.tools.task_tools import get_active_task_graph
            atg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
            if not atg:
                return None
            ip = atg.get_nodes_by_status("in_progress")
            if not ip:
                return None
            cn = ip[0]
            deps = atg.get_dependencies(cn.id)
            avail = {}
            for did in deps:
                dn = atg.get_node(did)
                if dn and dn.status == "completed":
                    avail[did] = dn.result or ""
            return {"current_subtask": cn.id, "dependencies": deps, "available_results": avail}
        except Exception:
            return None

    def _synthesize_from_task_graph(self) -> Optional[str]:
        try:
            from zulong.tools.task_tools import get_active_task_graph as _gtg_s
            tg = _gtg_s()
            if not tg:
                return None
            lv = tg.get_leaf_nodes()
            cp = [n for n in lv if n.status == "completed"]
            if cp and len(cp) == len(lv):
                parts = [f"## {tg.title}\n"]
                for n in cp:
                    r = getattr(n, "result", "") or ""
                    parts.append(f"### {n.label}\n{r or '（已完成）'}\n")
                return "\n".join(parts)
        except Exception:
            pass
        return None

    # ── IDE 会话自动持久化 ──────────────────────────────────

    def _auto_create_task_plan(self, state: IDEFCState) -> None:
        """IDE 会话开始时自动创建任务计划

        智能判断是否需要创建新任务图：
        - 如果本 session 已有关联的 TG → 复用
        - 如果任务图策略是 reuse/inspect/continue → 复用已完成的旧图（用户想修订/扩展）
        - 只有 L1-B/L2 明确给出 create/extend 策略时才自动创建骨架
        - 如果全局 TG 仍有未完成节点 → 不覆盖（其他会话可能在用）
        """
        try:
            if getattr(state, "task_graph_policy", "none") == "none":
                logger.debug("[IDEFCRunner] 任务图策略为 none，跳过自动任务图")
                return
            if getattr(state, "task_graph_policy", "none") not in {"create", "extend"}:
                logger.debug(
                    "[IDEFCRunner] 任务图策略 %s 交给 LLM 工具选择，跳过自动创建",
                    getattr(state, "task_graph_policy", "none"),
                )
                return

            from zulong.tools.task_tools import get_active_task_graph, set_active_task_graph
            existing_tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))

            # ── 阶段1: 关联已有TaskGraph（不受 user_input 长度限制） ──
            if existing_tg:
                # 本 session 已关联了这个 TG → 直接复用
                if (self.session.active_task_graph_id
                        and hasattr(existing_tg, 'id')
                        and getattr(existing_tg, 'id', '') == self.session.active_task_graph_id):
                    return

                # 会话未关联 → 自动关联到全局活跃TaskGraph
                if hasattr(existing_tg, 'id') and not self.session.active_task_graph_id:
                    self.session.active_task_graph_id = getattr(existing_tg, 'id', '')
                    self._notify_session_linked(self.session.active_task_graph_id)
                    logger.info(
                        f"[IDEFCRunner] 会话自动关联到已有任务图 "
                        f"(graph_id={self.session.active_task_graph_id})"
                    )

                # 检查是否已全部完成
                leaves = existing_tg.get_leaf_nodes()
                uncompleted = [n for n in leaves
                               if n.status not in ("completed", "skipped")]
                if uncompleted:
                    logger.info(
                        f"[IDEFCRunner] 已有活跃任务图（{len(uncompleted)} 未完成），复用")
                    return

                # 继续已有任务图策略 → 复用旧图
                if state.is_resume:
                    logger.info(
                        f"[IDEFCRunner] 继续任务图策略：复用旧图"
                        f"（graph_id={getattr(existing_tg, 'id', '?')}）")
                    return

                # 所有节点已完成 + 非复用策略 → 允许创建新 TG
                logger.info("[IDEFCRunner] 旧任务图已全部完成，创建新任务图")

            # ── 阶段2: 创建新TaskGraph（受 user_input 长度限制） ──
            user_input = state.user_input_text
            if not user_input or len(user_input.strip()) < 5:
                logger.debug("[IDEFCRunner] user_input 过短，跳过创建新任务图")
                return

            import re as _re
            import time as _time
            from zulong.l2.task_graph import TaskGraph

            # 从 <task>...</task> 标签中提取纯任务文本（Cline IDE 会包裹用户输入）
            _task_tag_match = _re.search(
                r"<task>\s*(.*?)\s*</task>", user_input, _re.DOTALL
            )
            if _task_tag_match:
                _clean_input = _task_tag_match.group(1).strip()
            else:
                # 无 <task> 标签时，截断已知噪声段落
                _clean_input = _re.split(
                    r"\n#\s*task_progress|<task_progress>|\n====", user_input
                )[0].strip()

            if not _clean_input or len(_clean_input) < 3:
                _clean_input = user_input.strip()

            title = _clean_input[:80].strip()
            graph_id = f"tg_{int(_time.time())}"
            tg = TaskGraph(title=title, graph_id=graph_id)
            tg.add_node(
                id="req", label=title, type="requirement",
                status="in_progress", desc=title,
            )

            set_active_task_graph(tg, graph_id, workspace_dir=getattr(self, 'cwd', None))

            # 关联到当前 session，避免后续请求重复创建
            self.session.active_task_graph_id = graph_id
            self._notify_session_linked(graph_id)

            # 同步到 MemoryGraph
            try:
                from zulong.memory.memory_graph import get_memory_graph, GraphNode, NodeType
                mg = get_memory_graph()
                if mg:
                    task_node = GraphNode(
                        node_id=f"task:{graph_id}",
                        node_type=NodeType.TASK,
                        label=title,
                        activation=1.0,
                        created_at=_time.time(),
                        last_accessed=_time.time(),
                        access_count=1,
                        metadata={
                            "graph_id": graph_id, "status": "active",
                            "source": "ide_auto",
                        },
                    )
                    mg.add_node(task_node)
                    mg.index_summary(f"task:{graph_id}", title)
            except Exception as me:
                logger.debug(f"[IDEFCRunner] 任务节点同步到记忆图失败: {me}")

            logger.info(
                f"[IDEFCRunner] 自动创建任务计划: {title} (graph_id={graph_id})"
            )

            # 自动创建任务计划后激活规划模式，放宽 CB 模式检测
            # （模型接下来会大量调用 task_add_node 构建子任务节点）
            if self._circuit_breaker:
                self._circuit_breaker.escalate_for_planning()

            # 推送初始图到 web 仪表盘
            try:
                self.engine._publish_task_graph_event(
                    "pipeline_start", 0, "task_auto_create",
                    f"创建任务图: {title}")
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"[IDEFCRunner] 自动创建任务计划失败（不影响FC循环）: {e}")

    def _auto_complete_task(self, state: IDEFCState) -> None:
        """FC 正常完成时不再自动伪完成任务节点。

        任务节点完成必须来自显式 task_mark_status 和完成质量门证据。
        这里仅记录未完成统计，避免旧逻辑把 in_progress 批量改成
        completed、把 pending 批量改成 skipped，从而绕过 TaskSpec Coverage Gate。
        """
        try:
            from zulong.tools.task_tools import get_active_task_graph, _save_active_backup
            tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
            if not tg:
                return
            leaves = tg.get_leaf_nodes()
            in_progress_count = 0
            pending_count = 0
            for leaf in leaves:
                if leaf.id.startswith("crg_"):
                    continue
                if leaf.status == "in_progress":
                    in_progress_count += 1
                elif leaf.status == "pending":
                    pending_count += 1
            if hasattr(tg, "metadata"):
                tg.metadata["last_fc_unfinished_snapshot"] = {
                    "in_progress": in_progress_count,
                    "pending": pending_count,
                    "fc_turn": state.fc_turn,
                    "updated_at": time.time(),
                }
                try:
                    _save_active_backup()
                except Exception:
                    pass
                logger.info(
                    "[IDEFCRunner] 完成阶段未自动改写节点状态: "
                    "in_progress=%s pending=%s",
                    in_progress_count,
                    pending_count,
                )
        except Exception as e:
            logger.debug(f"[IDEFCRunner] 未完成统计记录失败: {e}")

    def _publish_fc_progress(self, state: IDEFCState, stage: str, detail: str = ""):
        """发布 FC 循环进度事件到 EventBus → IDEWebBridge → 仪表盘"""
        try:
            from zulong.core.event_bus import event_bus
            from zulong.core.types import EventType, ZulongEvent
            payload = {
                "component": "FCRunner",
                "fc_turn": state.fc_turn,
                "phase": state.phase,
                "stage": stage,
                "detail": detail,
                "task_graph_policy": getattr(state, "task_graph_policy", "none"),
                "cb_force": state.cb_force_no_tools,
                "max_turns": self._max_fc_turns,
                "timestamp": time.time(),
            }
            event = ZulongEvent(type=EventType("SYSTEM_STATUS"), payload=payload)
            event_bus.publish(event)
        except Exception:
            pass

    def _auto_save_session_memory(self, state: IDEFCState) -> None:
        """FC 会话结束后自动保存记忆节点（包含任务摘要和工具使用记录）"""
        try:
            user_input = state.user_input_text
            # 🔥 修复：优先使用 submit_final_answer 的完整内容，确保记忆持久化包含完整答案
            response = getattr(state, "final_answer", None) or state.last_response_content
            if not user_input or len(user_input.strip()) < 10:
                return

            # 收集本次会话使用的远程工具
            remote_tools_used = set()
            for msg in state.messages:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    for tc in (msg.get("tool_calls") or []):
                        fn = tc.get("function", {}).get("name", "")
                        if fn in IDE_REMOTE_TOOLS:
                            remote_tools_used.add(fn)

            import time as _time
            from zulong.memory.memory_graph import (
                get_memory_graph, GraphNode, NodeType, Importance,
            )
            mg = get_memory_graph()
            if not mg:
                return

            # 构建摘要
            tools_str = ", ".join(sorted(remote_tools_used)) if remote_tools_used else "无"
            summary = (
                f"IDE 任务: {user_input[:200]}\n"
                f"使用工具: {tools_str}\n"
                f"结果摘要: {(response or '')[:300]}"
            )

            node_id = f"note:ide_{int(_time.time() * 1000)}"
            node = GraphNode(
                node_id=node_id,
                node_type=NodeType.KNOWLEDGE,
                label=f"IDE任务: {user_input[:50]}",
                activation=0.8,
                created_at=_time.time(),
                last_accessed=_time.time(),
                access_count=1,
                metadata={
                    "content": summary,
                    "importance": Importance.NORMAL.value,
                    "source": "ide_auto_session",
                    "tools_used": list(remote_tools_used),
                },
            )
            mg.add_node(node)
            mg.set_importance(node_id, Importance.NORMAL)
            mg.index_summary(node_id, summary)

            # 关联到任务节点
            try:
                from zulong.tools.task_tools import get_active_task_graph
                from zulong.memory.memory_graph import EdgeType
                tg = get_active_task_graph(workspace_dir=getattr(self, 'cwd', None))
                if tg:
                    task_mg_id = f"task:{tg.graph_id}" if hasattr(tg, "graph_id") else None
                    if task_mg_id and mg.has_node(task_mg_id):
                        mg.add_edge(task_mg_id, node_id, EdgeType.REFERENCE, weight=0.7)
            except Exception:
                pass

            # [P1 修复] 调用 ExperienceGenerator 从对话中提取经验
            try:
                from zulong.memory.experience_generator import ExperienceGenerator
                from zulong.memory.rag_manager import RAGManager
                rag = RAGManager()
                if hasattr(rag, '_initialized') and rag._initialized:
                    eg = ExperienceGenerator(rag_manager=rag)
                    dialogue_history = [
                        m for m in state.messages
                        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
                        and m.get("content")
                    ]
                    if len(dialogue_history) >= 2:
                        stats = eg.process_dialogue_batch(dialogue_history)
                        if stats.get("added", 0) > 0:
                            logger.info(
                                f"[IDEFCRunner] 经验提取: "
                                f"extracted={stats['extracted']}, added={stats['added']}")
            except Exception as exp_err:
                logger.debug(f"[IDEFCRunner] 经验提取跳过: {exp_err}")

            logger.info(f"[IDEFCRunner] 自动保存会话记忆: {node_id}")
        except Exception as e:
            logger.warning(f"[IDEFCRunner] 自动保存会话记忆失败: {e}")
