# File: zulong/state.py
# 定义祖龙系统的全局状态 Schema

from typing import TypedDict, List, Dict, Optional
from langchain_core.messages import BaseMessage


class ZulongState(TypedDict):
    """祖龙系统的全局状态 Schema"""
    # 对话历史
    messages: List[BaseMessage]
    # 传感器数据
    sensors: Dict
    # 意图标签
    intent: Optional[Dict]
    # 电机指令
    motor_command: Optional[Dict]
    # 核心状态
    core_status: str  # 枚举: IDLE, BUSY, PAUSED, BUSY_REFLEX
    # 中断标志
    interrupt_flag: bool
    # 中断请求
    interrupt_request: bool
    # 最后中断时间
    last_interrupt_time: float
    # 新的中断上下文
    new_context_for_interrupt: Optional[str]
    # 当前加载的专家列表
    active_experts: List[str]


def custom_reflex_reducer(old_state: ZulongState, new_state: ZulongState) -> ZulongState:
    """
    自定义归约函数，处理状态更新逻辑
    关键逻辑：当新输入的 motor_command 来源为 'REFLEX' 时，必须无条件覆盖旧的指令
    """
    # 创建合并后的状态
    merged_state = old_state.copy()
    
    # 合并新状态中的字段
    for key, value in new_state.items():
        # 特殊处理 motor_command 字段
        if key == 'motor_command' and value is not None:
            # 检查是否为反射指令
            if value.get('source') == 'REFLEX':
                # 反射指令优先级最高，直接覆盖
                merged_state['motor_command'] = value
            else:
                # 普通指令，只有当旧状态中没有指令或旧指令不是反射指令时才更新
                old_motor_command = old_state.get('motor_command')
                if old_motor_command is None or old_motor_command.get('source') != 'REFLEX':
                    merged_state['motor_command'] = value
        else:
            # 其他字段直接覆盖
            merged_state[key] = value
    
    return merged_state
