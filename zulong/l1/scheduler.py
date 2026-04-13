# File: zulong/l1/scheduler.py
# 调度器逻辑节点，实现基于核心状态的中断决策与去抖动机制

from typing import Dict, Any
import time
from zulong.l1.config import INTERRUPT_COOLDOWN_SEC, INTERRUPT_PRIORITY_THRESHOLD


class SchedulerLogicNode:
    """调度器逻辑节点"""
    
    def __init__(self):
        # 记录最后一次中断时间，用于去抖动
        self.last_interrupt_time = 0.0
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行调度器逻辑
        核心逻辑：
        - Core BUSY 时：监听新输入 -> 检查冷却时间 (2.0s) -> 若通过则触发中断 (interrupt_request=True)
        - Core IDLE 时：保持静默，不触发中断，让数据流向 L2 进行常规处理
        - 去抖动：记录 last_interrupt_time，防止高频误触
        """
        try:
            # 获取当前时间
            current_time = time.time()
            
            # 获取核心状态和新输入标志
            core_state = state.get('core_state', 'IDLE')
            has_new_input = state.get('has_new_input', False)
            
            # 默认为不触发中断
            interrupt_request = False
            
            # 当核心繁忙且有新输入时，检查冷却时间
            if core_state == 'BUSY' and has_new_input:
                # 检查冷却时间是否已过
                if current_time - self.last_interrupt_time >= INTERRUPT_COOLDOWN_SEC:
                    # 触发中断
                    interrupt_request = True
                    # 更新最后一次中断时间
                    self.last_interrupt_time = current_time
            
            # 返回更新后的状态
            return {
                **state,
                'interrupt_request': interrupt_request
            }
        
        except Exception as e:
            # 异常捕获，确保单个节点崩溃不会导致整个系统停止
            print(f"SchedulerLogicNode error: {e}")
            # 发生异常时，返回默认状态，不触发中断
            return {
                'interrupt_request': False
            }
