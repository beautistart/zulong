# File: zulong/tools/session_tools.py
"""
会话交互 FC 工具集

提供 FC 循环中 LLM 与用户交互的通道，适用于所有入口（Web / IDE / EventBus）。
"""

import logging
import time
from typing import Dict, Any

from .base import BaseTool, ToolCategory, ToolRequest, ToolResult

logger = logging.getLogger(__name__)


class AskUserTool(BaseTool):
    """ask_user — LLM 主动向用户提问

    语义：FC 循环遇到歧义或需要用户确认时调用此工具。调用后：
    1. 将 question 返回给调用方（FC 循环 / 编排器）
    2. FC 循环负责将问题推送到前端并暂停等待用户回复
    3. 用户回复后 FC 循环将答案注入下一轮迭代

    注意：这是内部工具，适用于所有入口。IDE 入口的 ask_followup_question
    是远程工具（走 WebSocket），而 ask_user 是 FC 原生工具。
    """

    def __init__(self):
        super().__init__(name="ask_user", category=ToolCategory.CUSTOM)
        self.description = (
            "向用户提出问题并等待回答。当任务存在歧义、需要用户确认方案、"
            "或缺少必要信息时使用。参数：question（问题内容）"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        question = request.parameters.get("question", "")
        if not question:
            return self._create_result(
                success=False,
                error="缺少 question 参数",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        logger.info("[AskUserTool] LLM 向用户提问: %s", question[:100])

        # 返回 success=True + 特殊标记，由 FC 循环检查并切换到等待用户状态
        return self._create_result(
            success=True,
            data={
                "action": "ask_user",
                "question": question,
                "awaiting_response": True,
            },
            execution_time=time.time() - start_time,
            request_id=request.request_id,
        )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "要向用户提出的问题",
                },
            },
            "required": ["question"],
        }
