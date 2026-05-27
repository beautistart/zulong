"""统一 WebSocket 通信协议

采用 IDE 端的统一帧格式，统一 Web(/ws) 和 IDE(/ide) 两套协议的消息类型。

统一帧格式:
    {
        "msg_id": "uuid4hex12",
        "type": "namespace:action",
        "session_id": "session_xxx",
        "ts": 1716300000.0,
        "payload": { ... }
    }

统一消息类型 (namespace:action):
    连接层:
        handshake (上行) - 客户端角色声明
        ping / pong (双向) - 心跳

    任务流:
        task:start (上行)     ← CHAT_MESSAGE / session_start
        task:resume (上行)    ← (无) / session_resume
        task:cancel (上行)    ← STOP_GENERATION / user_cancel
        task:ack (下行)       ← WELCOME / session_ack
        task:complete (下行)  ← CHAT_RESPONSE / task_complete
        task:error (下行)     ← (无) / task_error
        task:progress (下行)  ← THINKING_STEP / task_progress/status_update

    文本流:
        text:stream (下行)    ← STREAMING_RESPONSE / display_text
        text:final (下行)     ← CHAT_RESPONSE / task_complete (result字段)
        reasoning (下行)      ← (无) / display_reasoning

    工具:
        tool:request (下行)   ← (无) / tool_request
        tool:result (上行)    ← (无) / tool_result

    图谱:
        graph:memory:update (下行) ← MEMORY_GRAPH_UPDATE
        graph:memory:expand (下行) ← MEMORY_GRAPH_EXPAND_RESULT
        graph:task:update (下行)   ← TASK_GRAPH_UPDATE

    会话管理:
        session:list (下行)       ← SESSION_LIST
        session:messages (下行)   ← SESSION_MESSAGES
        session:deleted (下行)    ← SESSION_DELETED

    音频:
        audio:start / audio:end / audio:chunk (双向)
        audio:transcript (下行)   ← audio_transcript

用法:
    from zulong.core.unified_protocol import UnifiedMessage, ProtocolBridge
"""

import json as _json
import logging
import time
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
# 统一消息类型常量
# ═══════════════════════════════════════════════════

class MessageType:
    """统一消息类型常量 (namespace:action)"""

    # 连接层
    HANDSHAKE = "handshake"
    PING = "ping"
    PONG = "pong"

    # 任务流
    TASK_START = "task:start"
    TASK_RESUME = "task:resume"
    TASK_CANCEL = "task:cancel"
    TURN_ACCEPTED = "turn:accepted"
    TASK_ACK = "task:ack"
    TASK_COMPLETE = "task:complete"
    TASK_ERROR = "task:error"
    TASK_PROGRESS = "task:progress"

    # 文本流
    TEXT_STREAM = "text:stream"
    TEXT_FINAL = "text:final"
    REASONING = "reasoning"

    # 工具
    TOOL_REQUEST = "tool:request"
    TOOL_RESULT = "tool:result"

    # 图谱
    GRAPH_MEMORY_UPDATE = "graph:memory:update"
    GRAPH_MEMORY_EXPAND = "graph:memory:expand"
    GRAPH_TASK_UPDATE = "graph:task:update"

    # 会话管理
    SESSION_LIST = "session:list"
    SESSION_MESSAGES = "session:messages"
    SESSION_DELETED = "session:deleted"

    # 音频
    AUDIO_START = "audio:start"
    AUDIO_END = "audio:end"
    AUDIO_CHUNK = "audio:chunk"
    AUDIO_TRANSCRIPT = "audio:transcript"

    # IDE 后台桥
    IDE_OPEN_WORKSPACE = "ide:open_workspace"
    IDE_OPEN_FILE = "ide:open_file"
    IDE_OPEN_TERMINAL = "ide:open_terminal"
    IDE_SHOW_DIFF = "ide:show_diff"
    IDE_GET_CONTEXT = "ide:get_context"
    IDE_FILE_CHANGED = "ide:file_changed"
    IDE_TERMINAL_STATUS = "ide:terminal_status"
    IDE_APPROVAL_STATUS = "ide:approval_status"
    IDE_APPROVAL_RESULT = "ide:approval_result"
    IDE_DIFF_STATUS = "ide:diff_status"
    IDE_CHECKPOINT_STATUS = "ide:checkpoint_status"
    IDE_CONTEXT = "ide:context"

    # 语音记录
    VOICE_BIND = "voice:bind"
    VOICE_RECORD_CREATED = "voice:record_created"
    VOICE_RECORD_LINKED = "voice:record_linked"
    VOICE_LIST = "voice:list"
    VOICE_DELETE = "voice:delete"

    # ===== v2.7 新增: 前端交互体系 (TSD 第23章) =====
    # 工具预判
    TOOL_PREDICTION = "tool:prediction"          # 下行: L1-B 工具预判结果
    # 任务生命周期
    TASK_PLAN = "task:plan"                       # 下行: 任务规划说明 (启动汇报)
    TASK_SUMMARY = "task:summary"                 # 下行: 结束总结 (含 memory_changes)
    # 审批
    APPROVAL_REQUIRED = "approval:required"       # 下行: 审批请求 (含 approval_mode)
    APPROVAL_RESULT = "approval:result"           # 上行: 用户审批决策
    # 注意力
    ATTENTION_UPDATE = "attention:update"         # 下行: 注意力状态变更
    # 图谱记忆增量
    GRAPH_MEMORY_DIFF = "graph:memory:diff"       # 下行: 任务记忆变化 (增量)
    # 交互卡片流
    INTERACTION_EVENT = "interaction:event"       # 下行: 交互事件 (可替代内嵌 interaction_payload)


# ═══════════════════════════════════════════════════
# 旧格式 → 新格式 映射表
# ═══════════════════════════════════════════════════

# Web(/ws) 旧类型 → 统一类型
_WEB_TO_UNIFIED: Dict[str, str] = {
    "CHAT_MESSAGE": MessageType.TASK_START,
    "CHAT_VISIBLE_MESSAGE": "chat:visible_message",
    "STOP_GENERATION": MessageType.TASK_CANCEL,
    "STOP_TASK": MessageType.TASK_CANCEL,
    "TURN_ACCEPTED": MessageType.TURN_ACCEPTED,
    "REQUEST_MEMORY_GRAPH": MessageType.GRAPH_MEMORY_UPDATE,  # 上行请求
    "EXPAND_NODE": MessageType.GRAPH_MEMORY_EXPAND,
    "LIST_DIALOGUE_SESSIONS": MessageType.SESSION_LIST,
    "GET_SESSION_MESSAGES": MessageType.SESSION_MESSAGES,
    "DELETE_DIALOGUE_SESSION": MessageType.SESSION_DELETED,
    "WELCOME": MessageType.TASK_ACK,
    "CHAT_RESPONSE": MessageType.TEXT_FINAL,
    "STREAMING_RESPONSE": MessageType.TEXT_STREAM,
    "THINKING_STEP": MessageType.TASK_PROGRESS,
    "MEMORY_GRAPH_UPDATE": MessageType.GRAPH_MEMORY_UPDATE,
    "MEMORY_GRAPH_EXPAND_RESULT": MessageType.GRAPH_MEMORY_EXPAND,
    "SESSION_LIST": MessageType.SESSION_LIST,
    "SESSION_MESSAGES": MessageType.SESSION_MESSAGES,
    "SESSION_DELETED": MessageType.SESSION_DELETED,
    "TASK_GRAPH_UPDATE": MessageType.GRAPH_TASK_UPDATE,
    "IDE_CONTEXT": MessageType.IDE_CONTEXT,
    "IDE_FILE_CHANGED": MessageType.IDE_FILE_CHANGED,
    "IDE_TERMINAL_STATUS": MessageType.IDE_TERMINAL_STATUS,
    "IDE_APPROVAL_STATUS": MessageType.IDE_APPROVAL_STATUS,
    "IDE_DIFF_STATUS": MessageType.IDE_DIFF_STATUS,
    "IDE_CHECKPOINT_STATUS": MessageType.IDE_CHECKPOINT_STATUS,
    "ide_action_result": MessageType.IDE_APPROVAL_RESULT,
    "ping": MessageType.PING,
    "pong": MessageType.PONG,
    "audio_start": MessageType.AUDIO_START,
    "audio_chunk": MessageType.AUDIO_CHUNK,
    "audio_end": MessageType.AUDIO_END,
    "voice_bind": MessageType.VOICE_BIND,
    "voice:list": MessageType.VOICE_LIST,
    "voice:delete": MessageType.VOICE_DELETE,
    "ide_approval_result": MessageType.IDE_APPROVAL_RESULT,
    "ide:approval_result": MessageType.IDE_APPROVAL_RESULT,
}

# IDE(/ide) 旧类型 → 统一类型
_IDE_TO_UNIFIED: Dict[str, str] = {
    "session_start": MessageType.TASK_START,
    "session_resume": MessageType.TASK_RESUME,
    "user_cancel": MessageType.TASK_CANCEL,
    "tool_result": MessageType.TOOL_RESULT,
    "session_ack": MessageType.TASK_ACK,
    "task_complete": MessageType.TASK_COMPLETE,
    "task_error": MessageType.TASK_ERROR,
    "task_progress": MessageType.TASK_PROGRESS,
    "status_update": MessageType.TASK_PROGRESS,
    "display_text": MessageType.TEXT_STREAM,
    "display_reasoning": MessageType.REASONING,
    "tool_request": MessageType.TOOL_REQUEST,
    "ping": MessageType.PING,
    "pong": MessageType.PONG,
    "audio_start": MessageType.AUDIO_START,
    "audio_chunk": MessageType.AUDIO_CHUNK,
    "audio_end": MessageType.AUDIO_END,
    "audio_start_ack": MessageType.AUDIO_TRANSCRIPT,
    "audio_transcript": MessageType.AUDIO_TRANSCRIPT,
    "ide_approval_result": MessageType.IDE_APPROVAL_RESULT,
    "ide:approval_result": MessageType.IDE_APPROVAL_RESULT,
}

# 统一类型 → Web 旧类型 (用于向后兼容发送)
_UNIFIED_TO_WEB: Dict[str, str] = {v: k for k, v in _WEB_TO_UNIFIED.items()}
_UNIFIED_TO_IDE: Dict[str, str] = {v: k for k, v in _IDE_TO_UNIFIED.items()}

# Web 客户端上行请求优先映射。部分统一类型在旧 Web 协议里同时有
# “请求名”和“下行结果名”，不能简单用反向 dict，否则 graph:memory:update
# 会被误转成 MEMORY_GRAPH_UPDATE 而不是 REQUEST_MEMORY_GRAPH。
_UNIFIED_TO_WEB_UPLINK: Dict[str, str] = {
    MessageType.TASK_START: "CHAT_MESSAGE",
    MessageType.TASK_CANCEL: "STOP_GENERATION",
    MessageType.TURN_ACCEPTED: "TURN_ACCEPTED",
    MessageType.GRAPH_MEMORY_UPDATE: "REQUEST_MEMORY_GRAPH",
    MessageType.GRAPH_MEMORY_EXPAND: "EXPAND_NODE",
    MessageType.SESSION_LIST: "LIST_DIALOGUE_SESSIONS",
    MessageType.SESSION_MESSAGES: "GET_SESSION_MESSAGES",
    MessageType.SESSION_DELETED: "DELETE_DIALOGUE_SESSION",
    MessageType.PING: "ping",
    MessageType.PONG: "pong",
    MessageType.AUDIO_START: "audio_start",
    MessageType.AUDIO_CHUNK: "audio_chunk",
    MessageType.AUDIO_END: "audio_end",
    MessageType.VOICE_BIND: "voice_bind",
    MessageType.VOICE_LIST: "voice:list",
    MessageType.VOICE_DELETE: "voice:delete",
    "chat:visible_message": "CHAT_VISIBLE_MESSAGE",
    MessageType.IDE_APPROVAL_RESULT: "ide_approval_result",
}

_UNIFIED_TO_WEB_DOWNLINK: Dict[str, str] = {
    MessageType.GRAPH_MEMORY_UPDATE: "MEMORY_GRAPH_UPDATE",
    MessageType.GRAPH_MEMORY_EXPAND: "MEMORY_GRAPH_EXPAND_RESULT",
    MessageType.GRAPH_TASK_UPDATE: "TASK_GRAPH_UPDATE",
    MessageType.SESSION_LIST: "SESSION_LIST",
    MessageType.SESSION_MESSAGES: "SESSION_MESSAGES",
    MessageType.SESSION_DELETED: "SESSION_DELETED",
    MessageType.IDE_CONTEXT: "IDE_CONTEXT",
    MessageType.IDE_FILE_CHANGED: "IDE_FILE_CHANGED",
    MessageType.IDE_TERMINAL_STATUS: "IDE_TERMINAL_STATUS",
    MessageType.IDE_APPROVAL_STATUS: "IDE_APPROVAL_STATUS",
    MessageType.IDE_DIFF_STATUS: "IDE_DIFF_STATUS",
    MessageType.IDE_CHECKPOINT_STATUS: "IDE_CHECKPOINT_STATUS",
}


# ═══════════════════════════════════════════════════
# ProtocolBridge
# ═══════════════════════════════════════════════════

class ProtocolBridge:
    """协议桥接器: 新旧格式互转 + 端点路由

    负责:
    1. 检测消息格式 (旧 Web / 旧 IDE / 新统一)
    2. 将旧格式转换为统一格式
    3. 将统一格式转换为旧格式 (向后兼容)
    4. 路由到正确的处理器
    """

    def __init__(self):
        self._sessions: Dict[str, str] = {}  # session_id → client_type

    def detect_format(self, raw_msg: dict) -> str:
        """检测消息格式版本

        Returns:
            "unified" - 新统一格式 (有 namespace:action type)
            "web"     - 旧 Web 格式 (大写下划线 type)
            "ide"     - 旧 IDE 格式 (已有 unified frame, 只是 type 还没用 namespace:action)
            "unknown" - 无法识别
        """
        if not isinstance(raw_msg, dict):
            return "unknown"

        msg_type = raw_msg.get("type", "")

        # 新统一格式: namespace:action，或 handshake/ping/pong 这类连接层类型。
        if ":" in msg_type or msg_type in (
            MessageType.HANDSHAKE,
            MessageType.PING,
            MessageType.PONG,
        ):
            return "unified"

        # IDE 旧格式: 有 msg_id + session_id + ts + payload 结构
        if all(k in raw_msg for k in ("msg_id", "session_id", "ts", "payload")):
            return "ide"

        # Web 旧格式: 大写下划线 style
        if msg_type and (msg_type.isupper() or msg_type in ("ping", "pong")):
            return "web"

        return "unknown"

    def to_unified(self, raw_msg: dict, format_version: str) -> dict:
        """将任意格式转换为统一帧格式

        Args:
            raw_msg: 原始消息
            format_version: detect_format 的结果

        Returns:
            统一帧格式的消息
        """
        if format_version == "unified":
            return self._normalize_payload(raw_msg)

        if format_version == "ide":
            # IDE 已有 frame，只需转换 type
            old_type = raw_msg.get("type", "")
            new_type = _IDE_TO_UNIFIED.get(old_type, old_type)
            return self._normalize_payload({
                **raw_msg,
                "type": new_type,
            })

        if format_version == "web":
            old_type = raw_msg.get("type", "")
            new_type = _WEB_TO_UNIFIED.get(old_type, old_type)
            return self._normalize_payload({
                "msg_id": raw_msg.get("request_id") or uuid.uuid4().hex[:12],
                "type": new_type,
                "session_id": raw_msg.get("session_id", ""),
                "ts": time.time(),
                "payload": {k: v for k, v in raw_msg.items()
                           if k not in ("type", "request_id", "session_id")},
            })

        return raw_msg

    def from_unified(
        self,
        unified_msg: dict,
        target_format: str = "web",
        *,
        direction: str = "downlink",
    ) -> dict:
        """将统一帧格式转换为目标格式

        Args:
            unified_msg: 统一格式消息
            target_format: "web" 或 "ide"

        Returns:
            目标格式的消息
        """
        msg_type = unified_msg.get("type", "")
        payload = unified_msg.get("payload", {})

        if target_format == "ide":
            old_type = _UNIFIED_TO_IDE.get(msg_type, msg_type)
            return {
                "msg_id": unified_msg.get("msg_id", ""),
                "type": old_type,
                "session_id": unified_msg.get("session_id", ""),
                "ts": unified_msg.get("ts", time.time()),
                "payload": payload,
            }

        # target_format == "web"
        if direction == "uplink":
            old_type = _UNIFIED_TO_WEB_UPLINK.get(
                msg_type,
                _UNIFIED_TO_WEB.get(msg_type, msg_type),
            )
        else:
            old_type = _UNIFIED_TO_WEB_DOWNLINK.get(
                msg_type,
                _UNIFIED_TO_WEB.get(msg_type, msg_type),
            )
        result_payload = dict(payload)
        if msg_type == MessageType.TASK_START and "text" not in result_payload:
            result_payload["text"] = result_payload.get("task", "")
        elif msg_type == MessageType.TASK_CANCEL:
            result_payload.setdefault("reason", result_payload.get("message", "cancelled"))

        result = {"type": old_type, **result_payload}
        if unified_msg.get("session_id"):
            result["session_id"] = unified_msg["session_id"]
        if unified_msg.get("msg_id"):
            result["request_id"] = unified_msg["msg_id"]
        result["ts"] = unified_msg.get("ts", time.time())
        return result

    def unified_to_legacy_type(self, unified_type: str, target_format: str = "ide") -> str:
        """返回目标端的旧类型名，不改变 payload。

        服务器内部仍复用旧 handler 时使用这个方法，避免每个入口都维护一份
        namespace:action → legacy 的映射表。
        """
        if target_format == "web":
            return _UNIFIED_TO_WEB.get(unified_type, unified_type)
        return _UNIFIED_TO_IDE.get(unified_type, unified_type)

    def _normalize_payload(self, unified_msg: dict) -> dict:
        """统一不同客户端的 payload 字段命名。

        Web 旧协议使用 text，IDE 旧协议使用 task；统一协议允许两者之一，
        在桥接层补齐另一侧需要的字段。
        """
        if not isinstance(unified_msg, dict):
            return unified_msg
        msg_type = unified_msg.get("type", "")
        payload = unified_msg.get("payload")
        if not isinstance(payload, dict):
            return unified_msg

        payload = dict(payload)
        if msg_type == MessageType.TASK_START:
            if "task" not in payload and "text" in payload:
                payload["task"] = payload["text"]
            if "text" not in payload and "task" in payload:
                payload["text"] = payload["task"]
        elif msg_type == MessageType.TASK_RESUME:
            if "task" not in payload and "text" in payload:
                payload["task"] = payload["text"]
        elif msg_type == MessageType.TOOL_RESULT:
            if "call_id" not in payload and "tool_call_id" in payload:
                payload["call_id"] = payload["tool_call_id"]
            if "result" not in payload and "content" in payload:
                payload["result"] = payload["content"]
        elif msg_type == MessageType.IDE_CONTEXT:
            if "workspace_path" in payload and "cwd" not in payload:
                payload["cwd"] = payload["workspace_path"]
        elif msg_type == MessageType.TASK_CANCEL:
            payload.setdefault("reason", payload.get("message", "cancelled"))

        return {**unified_msg, "payload": payload}


# ═══════════════════════════════════════════════════
# 便捷构建函数
# ═══════════════════════════════════════════════════

def make_unified_message(
    msg_type: str,
    payload: dict = None,
    session_id: str = "",
    msg_id: str = None,
) -> dict:
    """构建统一帧格式消息"""
    return {
        "msg_id": msg_id or uuid.uuid4().hex[:12],
        "type": msg_type,
        "session_id": session_id,
        "ts": time.time(),
        "payload": payload or {},
    }


def parse_raw_message(raw_text: str) -> Optional[dict]:
    """安全解析 JSON 消息"""
    try:
        return _json.loads(raw_text)
    except (_json.JSONDecodeError, TypeError):
        logger.warning(f"[Protocol] 无法解析消息: {raw_text[:100]}")
        return None
