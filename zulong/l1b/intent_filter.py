# File: zulong/l1b/intent_filter.py
# 意图过滤器 - 分析用户输入的意图和优先级

from zulong.core.types import EventPriority
import re


class IntentFilter:
    """意图过滤器"""
    
    def __init__(self):
        """初始化意图过滤器"""
        # 关键词匹配规则
        self.keyword_rules = {
            # 紧急指令
            r'救命|help|help me|emergency': {
                'intent': 'EMERGENCY',
                'priority': EventPriority.CRITICAL,
                'is_wake_word': True
            },
            # 高优先级指令
            r'停下|stop|停止': {
                'intent': 'STOP',
                'priority': EventPriority.HIGH,
                'is_wake_word': False
            },
            # 唤醒词
            r'你好|hello|hi|祖龙|zulong': {
                'intent': 'WAKE',
                'priority': EventPriority.NORMAL,
                'is_wake_word': True
            },
            # 特殊命令
            r'安静|sleep|silent': {
                'intent': 'CMD_SILENT',
                'priority': EventPriority.NORMAL,
                'is_wake_word': False
            },
            r'醒来|wake up|唤醒': {
                'intent': 'CMD_WAKE',
                'priority': EventPriority.NORMAL,
                'is_wake_word': True
            },
            # 复盘指令 (精确匹配，避免误触)
            r'^启动复盘$': {
                'intent': 'REVIEW',
                'priority': EventPriority.HIGH,
                'is_wake_word': False
            }
        }
    
    def analyze(self, text: str) -> dict:
        """分析用户输入的意图
        
        Args:
            text: 用户输入文本
            
        Returns:
            dict: 包含 intent, priority, is_wake_word 的字典
        """
        text = text.lower().strip()
        
        # 遍历关键词规则
        for pattern, rule in self.keyword_rules.items():
            if re.search(pattern, text):
                return rule
        
        # 默认情况
        return {
            'intent': 'NORMAL',
            'priority': EventPriority.NORMAL,
            'is_wake_word': False
        }
