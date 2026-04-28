# File: zulong/l1/nodes.py
# 实现 L1 感知层的节点

from zulong.state import ZulongState
from zulong.l1.config import INTERRUPT_COOLDOWN_SEC
import time


def visual_analyze_node(state: ZulongState) -> dict:
    """视觉快速分析节点
    
    模拟 L1 模型推理，根据摄像头帧 ID 返回预定义的场景标签列表
    
    Args:
        state: 系统状态
    
    Returns:
        包含更新后传感器数据的字典
    """
    # 从状态中获取传感器数据
    sensors = state.get('sensors', {})
    camera_frame_ref = sensors.get('camera_frame_ref', '')
    
    # 模拟 L1 模型推理（<50ms）
    start_time = time.time()
    
    # 根据帧 ID 映射场景标签
    environment_tags = []
    if 'fall' in camera_frame_ref.lower():
        environment_tags = ['ground_tilted', 'human_nearby']
    elif 'hallway' in camera_frame_ref.lower():
        environment_tags = ['clear_path', 'wall_left']
    elif 'person' in camera_frame_ref.lower():
        environment_tags = ['human_nearby', 'obstacle_front']
    elif 'room' in camera_frame_ref.lower():
        environment_tags = ['indoor', 'furniture_around']
    else:
        # 默认场景标签
        environment_tags = ['normal', 'no_obstacle']
    
    # 模拟推理时间（<50ms）
    elapsed = time.time() - start_time
    if elapsed < 0.05:
        time.sleep(0.05 - elapsed)
    
    # 更新传感器数据，添加环境标签
    updated_sensors = sensors.copy()
    updated_sensors['environment_tags'] = environment_tags
    
    print(f"[L1] [Visual Analyze] Frame: {camera_frame_ref}, Tags: {environment_tags}")
    
    return {'sensors': updated_sensors}


def reflex_logic_node(state: ZulongState) -> dict:
    """硬实时反射逻辑节点
    
    检测触发条件，生成初步动作，并进行环境安全校验
    
    Args:
        state: 系统状态
    
    Returns:
        包含电机指令的字典
    """
    # 从状态中获取传感器数据
    sensors = state.get('sensors', {})
    imu_status = sensors.get('imu_status', 'STABLE')
    environment_tags = sensors.get('environment_tags', [])
    
    print(f"[L1] [Sensor] IMU Status: {imu_status}")
    print(f"[L1] [Vision] Tags: {environment_tags}")
    
    # 检测触发条件
    if imu_status not in ['FALL', 'TILT']:
        # 无触发条件，返回空指令
        print("[L1] [Reflex] No trigger condition met. Skipping.")
        return {'motor_command': None}
    
    # 生成初步动作
    initial_action = ""
    if imu_status == 'FALL':
        # 摔倒默认策略是“立即停止”
        initial_action = "STOP"
    elif imu_status == 'TILT':
        initial_action = "BALANCE_RECOVER"
    
    print(f"[L1] [Reflex] Triggered! Initial Plan: {initial_action}.")
    
    # 环境安全校验
    safety_verified = True
    final_action = initial_action
    
    # 安全规则库
    if "BACKUP" in final_action:
        # 检查是否有后方障碍物
        obstacle_detected = False
        detected_tag = ""
        for tag in ['car_behind', 'human_behind']:
            if tag in environment_tags:
                obstacle_detected = True
                detected_tag = tag
                break
        
        if obstacle_detected:
            print(f"[L1] [Reflex] Safety Check: FAILED! Obstacle '{detected_tag}' detected.")
            print("[L1] [Reflex] OVERRIDING Action to: STOP_IMMEDIATELY.")
            final_action = "STOP_IMMEDIATELY"
            safety_verified = True  # 修正后安全
    else:
        # 对于其他动作，检查环境是否安全
        print("[L1] [Reflex] Safety Check: Passed (No obstacles).")
    
    # 构建电机指令
    motor_command = {
        'action': final_action,
        'source': 'REFLEX',
        'priority': 'HIGH',
        'timestamp': time.time(),
        'safety_verified': safety_verified
    }
    
    print(f"[L1] [Reflex] Command Issued: {motor_command}")
    
    return {'motor_command': motor_command}


def scheduler_logic_node(state: ZulongState) -> dict:
    """外部实时调度器节点
    
    基于核心状态的中断决策与去抖动机制
    
    Args:
        state: 系统状态
    
    Returns:
        包含 interrupt_request, last_interrupt_time, new_context_for_interrupt 的字典片段
    """
    # 获取核心状态
    core_status = state.get('core_status', 'IDLE')
    
    # 获取传感器数据中的文本输入缓冲区
    sensors = state.get('sensors', {})
    text_input_buffer = sensors.get('text_input_buffer', '')
    
    # 获取最后中断时间
    last_interrupt_time = state.get('last_interrupt_time', 0.0)
    
    # 获取当前时间
    current_time = time.time()
    
    # 初始化返回值
    result = {
        'interrupt_request': False,
        'last_interrupt_time': last_interrupt_time,
        'new_context_for_interrupt': None
    }
    
    # 检查是否有新文本输入
    has_new_input = bool(text_input_buffer)
    
    # Case A: Core 处于 BUSY 状态 (正在规划/执行长任务)
    if core_status == 'BUSY':
        if has_new_input:
            # 计算时间差
            delta = current_time - last_interrupt_time
            
            if last_interrupt_time == 0.0:
                # 第一次输入，触发中断
                print("[L1] [Scheduler] 检测到核心忙碌且有新输入，冷却结束。")
                result['interrupt_request'] = True
                result['new_context_for_interrupt'] = text_input_buffer
                result['last_interrupt_time'] = current_time
                # 清空 buffer
                updated_sensors = sensors.copy()
                updated_sensors['text_input_buffer'] = ''
                result['sensors'] = updated_sensors
                print("[L1] [Scheduler] ✅ 触发事务性中断！")
            elif delta >= INTERRUPT_COOLDOWN_SEC:
                # 冷却结束，触发中断
                print("[L1] [Scheduler] 检测到核心忙碌且有新输入，冷却结束。")
                result['interrupt_request'] = True
                result['new_context_for_interrupt'] = text_input_buffer
                result['last_interrupt_time'] = current_time
                # 清空 buffer
                updated_sensors = sensors.copy()
                updated_sensors['text_input_buffer'] = ''
                result['sensors'] = updated_sensors
                print("[L1] [Scheduler] ✅ 触发事务性中断！")
            else:
                # 冷却中，忽略输入
                remaining_time = INTERRUPT_COOLDOWN_SEC - delta
                print(f"[L1] [Scheduler] 冷却中... (剩余 {remaining_time:.1f}s)，忽略输入。")
                # 清空 buffer 以防重复处理
                updated_sensors = sensors.copy()
                updated_sensors['text_input_buffer'] = ''
                result['sensors'] = updated_sensors

    # Case B: Core 处于 IDLE 状态
    elif core_status == 'IDLE':
        if has_new_input:
            print("[L1] [Scheduler] Core 空闲，进入正常流程 (非中断)。")
        # 确保 interrupt_request = False
        result['interrupt_request'] = False

    # Case C: Core 处于 BUSY_REFLEX (正在处理反射)
    elif core_status == 'BUSY_REFLEX':
        print("[L1] [Scheduler] 反射进行中，屏蔽普通中断。")
        # 保持 interrupt_request = False
        result['interrupt_request'] = False
    
    return result

