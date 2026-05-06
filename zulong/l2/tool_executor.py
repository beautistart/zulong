"""工具执行抽象层

将工具执行从 FC 循环中解耦，支持多种执行模式：
- LocalToolExecutor: 本地执行（编排器/推理引擎）
- 未来: RemoteToolExecutor / HybridToolExecutor（IDE 远程执行）
"""

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class _ToolCallProxy:
    """轻量代理，模拟 OpenAI ChatCompletionMessageToolCall 接口。

    将 dict 格式的 tool_call 数据转换为属性访问，
    供 engine._execute_tool_call() 使用。
    """

    def __init__(self, data: Dict[str, Any]):
        self.id = data["id"]
        self.type = data.get("type", "function")
        self.function = type("F", (), {
            "name": data["function"]["name"],
            "arguments": data["function"]["arguments"],
        })()


class ToolExecutor:
    """工具执行基类"""

    def execute(self, tool_call_data: Dict[str, Any], engine) -> str:
        """执行单个工具调用，返回结果文本。

        Args:
            tool_call_data: OpenAI 格式 tool_call dict
                {"id": ..., "type": "function",
                 "function": {"name": ..., "arguments": ...}}
            engine: InferenceEngine 实例

        Returns:
            工具执行结果文本
        """
        raise NotImplementedError

    def parse_args(self, tool_call_data: Dict[str, Any]) -> Dict[str, Any]:
        """解析工具参数 JSON 字符串为 dict"""
        try:
            return json.loads(
                tool_call_data["function"]["arguments"] or "{}"
            )
        except Exception:
            return {}


class LocalToolExecutor(ToolExecutor):
    """本地工具执行器

    通过 engine._execute_tool_call() 直接在进程内执行。
    用于编排器 FC 循环和推理引擎批处理模式。
    """

    def execute(self, tool_call_data: Dict[str, Any], engine) -> str:
        proxy = _ToolCallProxy(tool_call_data)
        return engine._execute_tool_call(proxy)
