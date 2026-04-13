# File: zulong/skill_packs/module_router.py
"""
ModuleRouter - 双层路由模块

第一层：快速预判（关键词规则，<5ms）
- 短句/问候 -> 闲聊，不需要技能包
- 包含动作指令词 -> 疑似任务，进入第二层

第二层：L2 Function Calling 自主决定
- 模型根据可用工具列表，自主决定是否调用技能包工具
- 需要任务拆解 -> 调用 task_decompose
- 需要深度推理 -> 调用 deep_reasoning
- 不需要 -> 正常回复
"""

import logging
import re
import time

logger = logging.getLogger(__name__)

# 闲聊关键词（命中则直接判定为闲聊）
CHATTER_KEYWORDS = [
    r'^[你你好嗨hellohi]*[啊呀哦]*$',
    r'^(你好|您好|早|早上好|晚上好|午安|嗨|hello|hi)',
    r'(谢谢|感谢|感谢|bye|拜拜|再见)',
    r'(哈哈|呵呵|嗯嗯|好的|知道了|明白|ok)',
]

# 任务关键词（命中则进入第二层）
TASK_KEYWORDS = [
    '搜索', '查找', '查询', '分析', '评估', '比较', '对比',
    '设计', '创建', '写', '报告', '总结', '计算',
    '发送', '邮件', '通知', '执行', '运行',
    '规划', '计划', '安排', '优化', '改进',
]


def quick_class(text: str) -> str:
    """第一层：快速预判
    
    Returns:
        "chatter" -> 闲聊，不需要技能包
        "task" -> 疑似任务，进入第二层
    """
    if not text or len(text.strip()) < 2:
        return "chatter"
    
    # 检查闲聊
    for pattern in CHATTER_KEYWORDS:
        if re.match(pattern, text, re.IGNORECASE):
            return "chatter"
    
    # 检查任务
    for kw in TASK_KEYWORDS:
        if kw in text:
            return "task"
    
    # 默认进入第二层（保守策略：不确定的都交给L2判断）
    return "task"


def classify_with_timing(text: str) -> dict:
    """带性能指标的分类（用于调试）
    
    Returns:
        {
            "classification": "chatter" | "task",
            "timing_ms": float,
            "reason": str,
        }
    """
    start = time.time()
    result = quick_class(text)
    elapsed = (time.time() - start) * 1000
    
    reason = "关键词匹配" if result == "chatter" else "需要L2判断"
    
    return {
        "classification": result,
        "timing_ms": elapsed,
        "reason": reason,
    }
