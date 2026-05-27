"""
L1-B 工具预判器 (L1BToolPredictor)

基于 TSD v2.7 第 23.2.3 节设计。
对用户 prompt 进行快速工具预判，缩小 L2 的工具选择范围。

核心职责:
1. 分析用户 prompt + 对话历史，预判可能需要的工具
2. 返回预判工具集合 + 完整工具袋(全部工具说明)
3. 返回 turn_shape、task_graph_policy 等上下文信号，供 L2 自主推理
4. 不替代 ALBERT 其余 12 类（特别是语音交互识别保留）

与 zulong/tools/tool_bag.py 的关系:
- tool_bag.py 是核心实现（build_tool_bag, predict_tools_for_turn 等）
- 本模块提供 TSD 定义的标准接口，作为 tool_bag 的薄包装
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ============================================================
# 工具袋定义 — 扁平全量工具清单（按 TSD 23.2.2）
# ============================================================

TOOL_BAG_FULL = [
    # ===== 文件操作 =====
    {
        "name": "read_file",
        "desc": "读取文件内容，支持指定行范围",
        "risk": "LOW",
    },
    {
        "name": "write_to_file",
        "desc": "创建或覆盖文件",
        "risk": "HIGH",
    },
    {
        "name": "replace_in_file",
        "desc": "精确替换文件中的指定内容",
        "risk": "HIGH",
    },
    {
        "name": "delete_file",
        "desc": "删除文件",
        "risk": "CRITICAL",
    },
    {
        "name": "list_files",
        "desc": "列出目录内容",
        "risk": "LOW",
    },
    {
        "name": "search_files",
        "desc": "按正则/glob 搜索文件内容",
        "risk": "LOW",
    },
    # ===== 命令执行 =====
    {
        "name": "execute_command",
        "desc": "在终端执行命令",
        "risk": "CRITICAL",
    },
    # ===== 网络与外部 =====
    {
        "name": "web_fetch",
        "desc": "获取网页内容",
        "risk": "MEDIUM",
    },
    {
        "name": "web_search",
        "desc": "搜索互联网信息",
        "risk": "LOW",
    },
    {
        "name": "use_skill",
        "desc": "调用注册的技能",
        "risk": "MEDIUM",
    },
    # ===== 记忆与图谱 =====
    {
        "name": "recall_memory",
        "desc": "从 MemoryGraph 检索相关记忆",
        "risk": "LOW",
    },
    {
        "name": "read_memory_node",
        "desc": "读取指定记忆节点详情",
        "risk": "LOW",
    },
    {
        "name": "save_memory_note",
        "desc": "保存记忆笔记到 MemoryGraph",
        "risk": "LOW",
    },
    # ===== 知识图谱 =====
    {
        "name": "search_knowledge",
        "desc": "搜索知识图谱中的实体和关系",
        "risk": "LOW",
    },
    {
        "name": "discover_related",
        "desc": "发现关联知识",
        "risk": "LOW",
    },
    # ===== IDE 桥接 =====
    {
        "name": "open_file",
        "desc": "在 VS Code 中打开文件",
        "risk": "LOW",
    },
    {
        "name": "show_diff",
        "desc": "在 VS Code 中显示文件差异",
        "risk": "LOW",
    },
    # ===== VS Code 命令与诊断 (TSD v2.7 工具袋扩充) =====
    {
        "name": "vscode_run_command",
        "desc": "执行 VS Code 内置/扩展命令（格式化、重构、Git 操作等）",
        "risk": "HIGH",
    },
    {
        "name": "get_diagnostics",
        "desc": "获取工作区所有文件 linter/编译器诊断（Error/Warning/Info/Hint）",
        "risk": "LOW",
    },
    # ===== VS Code 用户交互 =====
    {
        "name": "ask_user_input",
        "desc": "弹出 VS Code 输入框向用户提问",
        "risk": "LOW",
    },
    {
        "name": "ask_user_select_file",
        "desc": "弹出系统文件选择对话框，让用户选择文件/文件夹",
        "risk": "LOW",
    },
    # ===== VS Code 扩展与设置 =====
    {
        "name": "vscode_manage_extension",
        "desc": "安装/卸载/启用/禁用 VS Code 扩展",
        "risk": "HIGH",
    },
    {
        "name": "open_settings",
        "desc": "打开 VS Code 设置面板",
        "risk": "LOW",
    },
    {
        "name": "open_problems",
        "desc": "打开 VS Code 问题面板（展示诊断结果）",
        "risk": "LOW",
    },
]


class L1BToolPredictor:
    """L1-B 工具预判器

    使用关键词+规则快速预判用户可能需要哪些工具，
    返回预判工具集合 + 完整工具袋让 L2 自主选择。

    ALBERT 其余 12 类（含语音交互识别）不受影响。
    """

    # 关键词 → 工具映射
    KEYWORDS_MAP: Dict[str, List[str]] = {
        "写代码|修改|改|创建|新建": [
            "read_file", "search_files", "write_to_file", "replace_in_file",
        ],
        "查|搜|找|在哪|搜索": [
            "search_files", "search_knowledge", "web_search", "read_file",
        ],
        "运行|执行|跑|测试": [
            "execute_command", "read_file",
        ],
        "回忆|之前|上次|记得": [
            "recall_memory", "read_memory_node", "discover_related",
        ],
        "知识|关系|谁|什么|哪里": [
            "search_knowledge", "discover_related",
        ],
        # ===== VS Code 完整控制预判 (TSD v2.7 工具袋扩充) =====
        "格式化|格式化代码|整理导入|format|lint|prettier|重命名|rename": [
            "vscode_run_command", "read_file",
        ],
        "git|提交|commit|推送|push|分支|branch|merge|合并|暂存": [
            "vscode_run_command", "execute_command",
        ],
        "扩展|插件|extension|plugin|安装扩展|卸载扩展": [
            "vscode_manage_extension",
        ],
        "设置|setting|配置|preference": [
            "open_settings", "read_file",
        ],
        "错误|报错|error|warning|诊断|diagnostic|lint错误": [
            "get_diagnostics", "open_problems", "read_file",
        ],
        "选择文件|打开文件|browse|选择目录": [
            "ask_user_select_file", "list_files",
        ],
        "输入|填入|回答": [
            "ask_user_input",
        ],
    }

    # 任务图/工具增强指示词
    TOOL_AUGMENTED_INDICATORS = [
        "重构", "写", "创建", "修改", "分析", "实现", "部署", "调试",
        "修复", "开发", "设计", "优化", "配置", "集成", "迁移", "测试",
    ]
    SIMPLE_SOCIAL_PHRASES = {
        "你好", "您好", "嗨", "hi", "hello", "早上好", "下午好",
        "晚上好", "谢谢", "多谢", "辛苦了",
    }

    def predict_tools(
        self,
        prompt: str,
        conversation_history: Optional[list] = None,
    ) -> dict:
        """
        分析用户 prompt + 对话历史，预判可能需要的工具。

        Args:
            prompt: 用户输入文本
            conversation_history: 可选对话历史

        Returns:
            {
                "suggested_tools": ["read_file", "search_files", ...],
                "tool_bag": 全工具清单(所有工具+说明),
                "confidence": 0.85,
                "reason": "任务涉及代码修改，预计需要读取和搜索文件",
                "context_bundle": {"turn_shape": "tool_augmented"},
                "task_graph_policy": "inspect_or_create",
            }
        """
        # 1. 关键词 + 规则快速预判
        suggested = set()

        for pattern, tools in self.KEYWORDS_MAP.items():
            if re.search(pattern, prompt):
                suggested.update(tools)
                logger.debug(
                    f"[L1BToolPredictor] 关键词匹配 '{pattern}' → 工具: {tools}"
                )

        turn_shape = self._detect_turn_shape(prompt)
        task_graph_policy = self._predict_task_graph_policy(prompt, suggested)

        # 轻量寒暄不主动注入工具；L2 如需工具可请求补充。
        if turn_shape != "simple_social":
            suggested.update(["read_file", "search_files"])

        return {
            "suggested_tools": list(suggested),
            "tool_bag": TOOL_BAG_FULL,
            "confidence": self._calc_confidence(suggested, prompt),
            "reason": self._explain_prediction(suggested, turn_shape, task_graph_policy),
            "context_bundle": {"turn_shape": turn_shape},
            "task_graph_policy": task_graph_policy,
        }

    def _detect_turn_shape(self, prompt: str) -> str:
        """识别轮次形态。该信号只用于工具预判，不作为 L2 意图分流。"""
        text = (prompt or "").strip().lower()
        if text in self.SIMPLE_SOCIAL_PHRASES and len(text) <= 18:
            return "simple_social"
        if any(w in prompt for w in self.TOOL_AUGMENTED_INDICATORS):
            return "tool_augmented"
        return "direct_reply"

    def _predict_task_graph_policy(self, prompt: str, suggested: set) -> str:
        """根据工具需求判断是否建议 L2 检查/创建任务图。"""
        text = prompt or ""
        if any(w in text for w in self.TOOL_AUGMENTED_INDICATORS):
            return "inspect_or_create"
        if any(name.startswith("task_") for name in suggested):
            return "inspect"
        return "none"

    def _calc_confidence(self, suggested_tools: set, prompt: str) -> float:
        """计算预判置信度"""
        if not suggested_tools or len(suggested_tools) <= 2:
            return 0.5
        if len(suggested_tools) <= 4:
            return 0.7
        return min(0.95, 0.7 + len(suggested_tools) * 0.05)

    def _explain_prediction(
        self,
        suggested_tools: set,
        turn_shape: str,
        task_graph_policy: str,
    ) -> str:
        """生成预判理由"""
        if not suggested_tools:
            return f"轮次形态={turn_shape}，未检测到明确工具需求"
        tool_names = ", ".join(sorted(suggested_tools)[:5])
        if len(suggested_tools) > 5:
            tool_names += f" 等{len(suggested_tools)}个工具"
        return f"轮次形态={turn_shape}，任务图策略={task_graph_policy}，预计需要: {tool_names}"


# ============================================================
# 便捷函数：从现有 tool_bag 构建预测结果
# ============================================================

def predict_from_tool_bag(
    prompt: str,
    registry=None,
    intent_result: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    使用 L1BToolPredictor 进行快速预判，如果 tool_bag 可用则整合其详细结果。

    这是一个桥接函数：优先使用 L1BToolPredictor 的简化流程，
    当 registry 可用时补充详细信息。
    """
    predictor = L1BToolPredictor()
    result = predictor.predict_tools(prompt)

    # 如果 registry 可用，用 tool_bag 的详细预测增强结果
    if registry is not None:
        try:
            from zulong.tools.tool_bag import predict_tools_for_turn

            detailed = predict_tools_for_turn(
                prompt,
                registry=registry,
                intent_result=intent_result,
            ).to_dict()

            # 合并 tool_bag 的详细结果
            result["detailed_prediction"] = detailed
            result["risk_notes"] = detailed.get("risk_notes", [])
            result["task_graph_policy"] = detailed.get("task_graph_policy", "none")

            logger.info(
                "[L1BToolPredictor] 整合 tool_bag 详细预测: %d 工具, policy=%s",
                len(detailed.get("predicted_tools", [])),
                detailed.get("task_graph_policy"),
            )
        except ImportError:
            logger.debug("[L1BToolPredictor] tool_bag 不可用，使用简化预测")
        except Exception as e:
            logger.warning(f"[L1BToolPredictor] tool_bag 整合失败: {e}")

    return result
