"""
信息缺口检测器 (Information Gap Detector)

在任务执行过程中，检测模型是否意识到信息缺失，并区分：
1. NEED_SUBTASK_RESULT: 需要其他子任务的执行结果（系统自动等待）
2. NEED_USER_INPUT: 需要用户补充信息（向用户发起自然语言提问）
3. SUFFICIENT: 信息充足，可以继续执行

检测方式：
- 分析 LLM 输出文本中的信息缺口信号词
- 分析工具调用结果中的空结果/错误模式
- 综合判断缺口类型（子任务依赖 vs 用户补充）
"""

import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class InfoGapType(Enum):
    SUFFICIENT = "sufficient"                   # 信息充足
    NEED_SUBTASK_RESULT = "need_subtask_result"  # 需要其他子任务结果
    NEED_USER_INPUT = "need_user_input"          # 需要用户补充信息


class InfoGapSignal:
    """单个信息缺口信号"""

    def __init__(self, gap_type: InfoGapType, description: str, confidence: float):
        self.gap_type = gap_type
        self.description = description
        self.confidence = confidence  # 0.0 ~ 1.0


class InformationGapDetector:
    """信息缺口检测器

    分析 LLM 的输出（文本 + 工具调用），判断当前执行是否遇到信息缺口，
    并区分缺口来源：需要子任务结果，还是需要用户输入。
    """

    # ---- 需要用户输入的信号词 ----
    USER_INPUT_PATTERNS = [
        # 直接请求用户提供信息
        (r"请(?:您|你)(?:提供|告诉|说明|确认|指定|选择|描述)", 0.9),
        (r"(?:需要|缺少)(?:您|你|用户)(?:的|提供的)?(?:信息|输入|确认|选择|说明|描述|详情)", 0.85),
        (r"(?:您|你)(?:希望|想要|期望|偏好|倾向)", 0.7),
        (r"(?:能否|可否|是否可以)(?:告诉|提供|确认|说明)", 0.8),
        # 模型表示不确定需要用户明确
        (r"(?:不确定|不清楚|无法确定).*(?:请|需要).*(?:确认|说明|指定)", 0.85),
        (r"这取决于(?:您|你|用户)的(?:需求|选择|偏好)", 0.75),
        # 多选一场景
        (r"(?:您|你)(?:想|要)(?:选择|使用)(?:哪|什么)", 0.8),
        (r"以下.*(?:方案|选项|选择).*(?:请|需要).*(?:选择|确认|决定)", 0.75),
    ]

    # ---- 需要子任务结果的信号词 ----
    SUBTASK_DEPENDENCY_PATTERNS = [
        # 等待前置任务
        (r"(?:需要|等待|依赖).*(?:前置|前一个|上一步|前序).*(?:任务|步骤|结果)", 0.9),
        (r"(?:在|必须).*(?:完成|获得|得到).*(?:之后|结果后).*(?:才能|方可)", 0.85),
        # 引用其他子任务
        (r"(?:子任务|步骤|阶段)\s*\d+.*(?:结果|输出|数据)", 0.8),
        (r"(?:需要|缺少).*(?:分析|调研|搜索|计算).*(?:结果|数据|报告)", 0.7),
        # 数据依赖
        (r"(?:还没有|尚未获得|缺少).*(?:数据|结果|信息)", 0.65),
        (r"(?:基于|根据).*(?:前面|之前|上一步).*(?:结果|分析)", 0.75),
    ]

    # ---- 工具失败模式（暗示信息缺口）----
    TOOL_FAILURE_PATTERNS = [
        (r"(?:搜索|查询|检索).*(?:未找到|无结果|失败|超时)", 0.6),
        (r"(?:参数|信息|数据).*(?:缺失|不完整|无效)", 0.7),
        (r"(?:无法|不能).*(?:执行|完成|处理)", 0.5),
    ]

    def __init__(self, confidence_threshold: float = 0.6):
        self._confidence_threshold = confidence_threshold

    def detect(
        self,
        llm_output: str,
        tool_results: Optional[List[Dict]] = None,
        subtask_context: Optional[Dict] = None
    ) -> Tuple[InfoGapType, str, float]:
        """检测信息缺口

        Args:
            llm_output: LLM 的文本输出
            tool_results: 最近的工具调用结果列表
            subtask_context: 子任务上下文 {"current_subtask", "dependencies", "available_results"}

        Returns:
            Tuple[InfoGapType, str, float]:
                - 缺口类型
                - 自然语言描述（可直接展示给用户）
                - 置信度 0.0~1.0
        """
        signals: List[InfoGapSignal] = []

        # 1. 检测用户输入需求
        user_signals = self._detect_user_input_need(llm_output)
        signals.extend(user_signals)

        # 2. 检测子任务依赖
        subtask_signals = self._detect_subtask_dependency(llm_output, subtask_context)
        signals.extend(subtask_signals)

        # 3. 检测工具失败模式
        if tool_results:
            tool_signals = self._detect_tool_failure_gaps(tool_results)
            signals.extend(tool_signals)

        # 没有信号 → 信息充足
        if not signals:
            return InfoGapType.SUFFICIENT, "", 0.0

        # 按置信度排序，取最高的
        signals.sort(key=lambda s: s.confidence, reverse=True)
        best = signals[0]

        if best.confidence < self._confidence_threshold:
            return InfoGapType.SUFFICIENT, "", best.confidence

        return best.gap_type, best.description, best.confidence

    def format_user_question(self, gap_description: str, llm_output: str) -> str:
        """将信息缺口格式化为面向用户的自然语言提问

        确保：
        - 使用自然语言，不返回结构化数据
        - 明确说明需要什么信息
        - 语气友好、具体

        Args:
            gap_description: 检测到的缺口描述
            llm_output: LLM 的原始输出（可能已包含对用户的提问）

        Returns:
            str: 自然语言格式的提问文本
        """
        # 如果 LLM 输出本身已经包含了对用户的提问，直接使用
        if self._contains_direct_question(llm_output):
            return llm_output

        # 否则，基于缺口描述生成提问
        return f"在继续执行任务的过程中，我发现还需要一些额外的信息：\n\n{gap_description}\n\n请您提供以上信息，我将据此继续完成任务。"

    # ==================== 内部检测方法 ====================

    def _detect_user_input_need(self, text: str) -> List[InfoGapSignal]:
        """检测文本中是否有向用户请求信息的信号"""
        signals = []
        for pattern, base_confidence in self.USER_INPUT_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                # 提取匹配上下文作为描述
                for match in re.finditer(pattern, text):
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()
                    signals.append(InfoGapSignal(
                        gap_type=InfoGapType.NEED_USER_INPUT,
                        description=context,
                        confidence=base_confidence,
                    ))
        return signals

    def _detect_subtask_dependency(
        self, text: str, subtask_context: Optional[Dict] = None
    ) -> List[InfoGapSignal]:
        """检测文本中是否有对子任务结果的依赖"""
        signals = []

        # 文本模式匹配
        for pattern, base_confidence in self.SUBTASK_DEPENDENCY_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                for match in re.finditer(pattern, text):
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()
                    signals.append(InfoGapSignal(
                        gap_type=InfoGapType.NEED_SUBTASK_RESULT,
                        description=context,
                        confidence=base_confidence,
                    ))

        # 结构化上下文检查：当前子任务声明了依赖但依赖结果尚未就绪
        # 注意：仅结构化条件不足以判定信息缺口，需要文本信号配合
        # 否则任务图刚创建（依赖全 pending）时会持续误报
        if subtask_context:
            deps = subtask_context.get("dependencies", [])
            available = set(subtask_context.get("available_results", {}).keys())
            missing_deps = [d for d in deps if d not in available]
            if missing_deps:
                # 结构 + 文本双重确认：只有当文本模式也检测到信号时才给高置信度
                has_text_signal = len(signals) > 0
                signals.append(InfoGapSignal(
                    gap_type=InfoGapType.NEED_SUBTASK_RESULT,
                    description=f"等待子任务完成: {', '.join(missing_deps)}",
                    confidence=0.85 if has_text_signal else 0.4,
                ))

        return signals

    def _detect_tool_failure_gaps(self, tool_results: List[Dict]) -> List[InfoGapSignal]:
        """检测工具调用结果中的失败模式"""
        signals = []
        for result in tool_results[-3:]:  # 只看最近 3 次
            content = result.get("content", "")
            if not content:
                continue
            for pattern, base_confidence in self.TOOL_FAILURE_PATTERNS:
                if re.search(pattern, content):
                    signals.append(InfoGapSignal(
                        gap_type=InfoGapType.NEED_USER_INPUT,
                        description=f"工具执行遇到问题: {content[:100]}",
                        confidence=base_confidence,
                    ))
        return signals

    @staticmethod
    def _contains_direct_question(text: str) -> bool:
        """判断文本是否已包含对用户的直接提问"""
        question_markers = ["?", "？", "请问", "请您", "请你", "能否", "可否", "是否"]
        return any(marker in text for marker in question_markers)
