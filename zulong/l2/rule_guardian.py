# File: zulong/l2/rule_guardian.py
"""
RuleGuardian — 规则守护者

在 FC 循环中拦截违反系统规则的模型行为。
设计原则：安全网级别，不限制模型正常决策，只在出错时拦截。
"""

import re
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

# 过早完成声明的正则模式
_COMPLETION_PATTERNS = [
    re.compile(r"所有.{0,6}(任务|子任务|工作|步骤).{0,6}(已|都).{0,4}(完成|完毕|做完|结束)"),
    re.compile(r"任务.{0,4}(全部|已经|均已).{0,4}(完成|完毕|做完|结束)"),
    re.compile(r"(已经|已).{0,4}(全部|全都).{0,4}(完成|完毕|做完)"),
    re.compile(r"(整个|全部).{0,4}(项目|任务|工作).{0,6}(完成|完毕|做完)"),
]


class RuleGuardian:
    """规则守护者 — FC 循环级别的规则拦截"""

    def __init__(self, enabled: bool = True, max_retries: int = 2):
        self.enabled = enabled
        self.max_retries = max_retries
        self._retry_count = 0
        self._bypass_redirect_count = 0  # 模型绕过节点执行的次数

    def check_premature_completion(
        self, response_text: str, task_graph
    ) -> Tuple[bool, str]:
        """检查模型回复是否包含过早的完成声明

        Args:
            response_text: 模型的文本回复
            task_graph: 当前活跃的 TaskGraph（可能为 None）

        Returns:
            (should_block, reason): 是否应该阻止, 原因说明
        """
        if not self.enabled or not task_graph:
            return False, ""

        # 1. 检测完成声明关键词
        has_completion_claim = False
        for pattern in _COMPLETION_PATTERNS:
            if pattern.search(response_text):
                has_completion_claim = True
                break

        if not has_completion_claim:
            # ── 新增：检测"绕过节点执行"行为 ──
            # 模型生成了充实的内容但未调用 task_mark_status，
            # 此时任务图有全部未完成的叶节点 → 说明模型在绕过节点执行。
            # 第一次给一次重定向机会，让模型尝试正确使用工具。
            block, reason = self._check_node_bypass(response_text, task_graph)
            if block:
                return block, reason
            return False, ""

        # 2. 检查任务图实际状态
        leaf_nodes = task_graph.get_leaf_nodes()
        if not leaf_nodes:
            return False, ""

        # 排除 CRG 自动注入节点（crg_ 前缀），只看用户任务
        user_leaves = [n for n in leaf_nodes if not n.id.startswith("crg_")]
        uncompleted = [n for n in user_leaves if n.status != "completed"]
        if not uncompleted:
            return False, ""  # 确实全部完成了，放行

        # 3. 有未完成任务 + 有完成声明 → 拦截
        if self._retry_count >= self.max_retries:
            logger.warning(
                f"[RuleGuardian] 过早完成拦截已达重试上限 ({self.max_retries})，"
                f"标记未完成节点为 needs_adjust 后放行"
            )
            self._retry_count = 0
            # 修复：放行前标记未完成节点，避免"假完成"状态
            try:
                for n in uncompleted:
                    task_graph.update_node_status(
                        n.id, "needs_adjust",
                        result="模型多次声明完成但实际未完成",
                    )
                logger.info(
                    f"[RuleGuardian] 已将 {len(uncompleted)} 个节点标记为 needs_adjust"
                )
            except Exception as e:
                logger.warning(f"[RuleGuardian] 标记 needs_adjust 失败: {e}")
            return False, ""

        self._retry_count += 1
        uncompleted_list = ", ".join(
            f"{n.id}({n.label})" for n in uncompleted[:5]
        )
        reason = (
            f"检测到过早完成声明，但仍有 {len(uncompleted)} 个任务未完成: "
            f"{uncompleted_list}。请先执行这些未完成的任务。"
        )
        logger.info(f"[RuleGuardian] 拦截过早完成声明 (retry={self._retry_count}): {reason}")
        return True, reason

    def _check_node_bypass(
        self, response_text: str, task_graph
    ) -> Tuple[bool, str]:
        """检测模型是否绕过任务图节点直接生成内容。

        触发条件（全部满足）：
        - 回复长度 > 200 字符（充实的内容）
        - 回复不是提问（不含多个问号或追问模式）
        - 有活跃任务图且存在叶节点
        - 所有叶节点均未完成（说明模型从未调用 task_mark_status）
        - 第一次绕过时拦截，给模型一次重定向机会

        Returns:
            (should_block, reason)
        """
        if len(response_text) <= 200:
            return False, ""

        # 排除提问型回复：模型向用户提问获取更多信息是合理行为
        if self._is_question_response(response_text):
            return False, ""

        try:
            leaf_nodes = task_graph.get_leaf_nodes()
            if not leaf_nodes:
                return False, ""

            # 排除根节点和 CRG 自动注入节点
            leaf_no_root = [n for n in leaf_nodes
                           if n.id != "req" and not n.id.startswith("crg_")]
            if not leaf_no_root:
                return False, ""

            uncompleted = [
                n for n in leaf_no_root
                if n.status not in ("completed", "skipped")
            ]
            # 只有当 100% 节点未完成时才认定为"绕过"
            if len(uncompleted) < len(leaf_no_root):
                return False, ""

            # 第一次绕过：拦截并重定向
            if self._bypass_redirect_count == 0:
                self._bypass_redirect_count += 1
                first_node = uncompleted[0]
                node_list = ", ".join(
                    f"{n.id}({n.label})" for n in uncompleted[:4]
                )
                reason = (
                    f"检测到你生成了完整回复但未执行任务图中的节点。"
                    f"当前有 {len(uncompleted)} 个待执行节点: {node_list}。"
                    f"请先用 task_mark_status(node_id='{first_node.id}', "
                    f"status='in_progress') 开始执行第一个节点，"
                    f"完成后用 task_mark_status(node_id='{first_node.id}', "
                    f"status='completed', result='节点内容') 提交结果，"
                    f"然后依次执行后续节点。"
                )
                logger.info(
                    f"[RuleGuardian] 拦截节点绕过行为 "
                    f"(bypass={self._bypass_redirect_count}): "
                    f"{len(uncompleted)} 个节点全部未完成"
                )
                return True, reason

            # 后续绕过：不再拦截，交给 Backfill 安全网处理
            return False, ""

        except Exception as e:
            logger.debug(f"[RuleGuardian] 节点绕过检查异常: {e}")
            return False, ""

    @staticmethod
    def _is_question_response(text: str) -> bool:
        """检测回复是否为向用户提问/追问的内容。

        提问是合理行为（如：收集出发日期、预算、偏好等信息），
        不应被 _check_node_bypass 拦截。

        检测方式：
        1. 末尾是问号
        2. 包含多个问号（多问追问）
        3. 包含典型追问模式
        """
        stripped = text.strip()

        # 末尾是问号
        if stripped.endswith(("?", "\uff1f")):
            return True

        # 包含 2 个及以上问号 → 多问追问
        q_count = stripped.count("?") + stripped.count("\uff1f")
        if q_count >= 2:
            return True

        # 典型追问模式
        _QUESTION_PATTERNS = [
            "请问", "请告诉", "能否告诉", "你希望", "你想要",
            "您计划", "您希望", "您想", "您偏好", "您喜欢",
            "需要了解", "需要确认", "想了解", "想确认",
            "是否需要", "是否希望", "是否可以",
            "出发日期", "出发时间", "预算范围", "几天", "多少天",
        ]
        pattern_count = sum(1 for p in _QUESTION_PATTERNS if p in stripped)
        if pattern_count >= 2:
            return True

        return False

    def reset(self):
        """重置重试计数器（每次新的 FC 循环开始时调用）"""
        self._retry_count = 0
        self._bypass_redirect_count = 0

    def serialize(self) -> Dict[str, Any]:
        """导出内部状态（供 FC 暂停时保存）"""
        return {
            "retry_count": self._retry_count,
            "bypass_redirect_count": self._bypass_redirect_count,
            "enabled": self.enabled,
            "max_retries": self.max_retries,
        }

    def deserialize(self, state: Dict[str, Any]) -> None:
        """恢复内部状态（供 FC 恢复时加载）"""
        self._retry_count = state.get("retry_count", 0)
        self._bypass_redirect_count = state.get("bypass_redirect_count", 0)
        logger.info(
            f"[RuleGuardian] 状态已恢复: retry_count={self._retry_count}, "
            f"bypass_redirect_count={self._bypass_redirect_count}"
        )
