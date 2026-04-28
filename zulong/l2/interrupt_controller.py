# File: zulong/l2/interrupt_controller.py
# 中断控制器 - 第五阶段总装
# 对应 TSD v1.7: 任务冻结与恢复

import uuid
import time

from zulong.core.event_bus import event_bus
from zulong.core.types import EventType, EventPriority, ZulongEvent
from zulong.l2.task_state_manager import task_state_manager
from zulong.core.state_manager import state_manager
from zulong.core.types import L2Status

import logging
logger = logging.getLogger(__name__)


class InterruptController:
    """中断控制器"""
    
    def __init__(self):
        """初始化中断控制器"""
        self._inference_engine = None
        self._register_event_handlers()
        logger.info("L2 InterruptController initialized and subscribed to system events")
    
    def set_inference_engine(self, engine):
        """设置推理引擎
        
        Args:
            engine: 推理引擎实例
        """
        self._inference_engine = engine
        logger.info("Inference engine injected")
    
    def _register_event_handlers(self):
        """注册事件处理器 - 只接收 L1-B 路由的事件"""
        event_bus.subscribe(EventType.SYSTEM_INTERRUPT, self.on_interrupt, "L2")
        event_bus.subscribe(EventType.SYSTEM_REFLEX, self.on_reflex, "L2")
        logger.info("L2 InterruptController: 只接收 L1-B 路由的事件")
    
    def on_interrupt(self, event: ZulongEvent):
        """处理系统中断
        
        Args:
            event: 中断事件
        """
        logger.info("System interrupt received")
        
        # 冻结当前任务
        task_state_manager.freeze_current()
        
        # 创建新任务
        task_id = f"task_{uuid.uuid4()}"
        task_state_manager.create_task(task_id, [
            {"role": "user", "content": "处理紧急情况"}
        ])
        
        # 处理中断（传递任务 ID）
        self._process_interrupt(event, task_id)
    
    def on_reflex(self, event: ZulongEvent):
        """处理反射事件 - 仅用于任务中断，不发布电机指令
        
        Args:
            event: 反射事件
        """
        command = event.payload.get("command", "")
        event_type = event.payload.get("event_type", "")
        
        # 反射优先级高，需要中断 L2 当前任务
        logger.info(f"Reflex event detected: {command} (from {event_type})")
        
        # 冻结当前任务
        task_state_manager.freeze_current()
        
        # 创建新任务处理反射后续
        task_id = f"task_{uuid.uuid4()}"
        task_state_manager.create_task(task_id, [
            {"role": "user", "content": f"处理反射事件：{command}"}
        ])
        
        # 处理反射（传递任务 ID）
        self._process_reflex(event, task_id)
    
    def _process_reflex(self, event: ZulongEvent, task_id: str):
        """处理反射事件 - 快速处理，不阻塞
        
        Args:
            event: 反射事件
            task_id: 任务 ID
        """
        # 反射事件只需要快速确认，不需要复杂处理
        # 因为 L1-A 已经直接发布了电机指令到 L0
        state_manager.set_l2_status(L2Status.BUSY, task_id)
        logger.info(f"Reflex acknowledged: {event.payload.get('command', '')}")
        time.sleep(0.1)  # 快速处理
        state_manager.set_l2_status(L2Status.IDLE)
        
        # 检查任务栈，恢复被中断的任务
        task_stack = task_state_manager.get_task_stack()
        if task_stack:
            # 恢复栈顶任务
            resumed_task_id = task_stack[-1]
            logger.info(f"Popped task {resumed_task_id} from stack")
            logger.info(f"Auto-resuming task {resumed_task_id} from stack")
            task_state_manager.resume_task(resumed_task_id)
    
    def _process_interrupt(self, event: ZulongEvent, task_id: str):
        """处理中断
        
        Args:
            event: 中断事件
            task_id: 任务 ID
        """
        # 模拟处理中断
        state_manager.set_l2_status(L2Status.BUSY, task_id)
        logger.info("Processing interrupt...")
        time.sleep(1)
        state_manager.set_l2_status(L2Status.IDLE)
        
        # 检查任务栈
        task_stack = task_state_manager.get_task_stack()
        if task_stack:
            # 恢复栈顶任务
            resumed_task_id = task_stack[-1]
            logger.info(f"Popped task {resumed_task_id} from stack")
            logger.info(f"Auto-resuming task {resumed_task_id} from stack")
            task_state_manager.resume_task(resumed_task_id)
    
    def _process_emergency(self, event: ZulongEvent, task_id: str):
        """处理紧急情况
        
        Args:
            event: 紧急事件
            task_id: 任务 ID
        """
        # 模拟处理紧急情况
        state_manager.set_l2_status(L2Status.BUSY, task_id)
        logger.info("Processing emergency...")
        time.sleep(2)
        state_manager.set_l2_status(L2Status.IDLE)
        
        # 检查任务栈
        task_stack = task_state_manager.get_task_stack()
        if task_stack:
            # 恢复栈顶任务
            resumed_task_id = task_stack[-1]
            logger.info(f"Popped task {resumed_task_id} from stack")
            logger.info(f"Auto-resuming task {resumed_task_id} from stack")
            task_state_manager.resume_task(resumed_task_id)
            
            # 继续处理被恢复的任务
            # 这里简化处理，假设恢复的是故事任务
            logger.info("Continuing resumed task...")
            self._process_command("继续讲故事", resumed_task_id)
    
    def _process_command(self, text: str, task_id: str, is_interrupt: bool = False, previous_task: str = None):
        """处理命令
        
        Args:
            text: 命令文本
            task_id: 任务 ID
            is_interrupt: 是否为中断任务
            previous_task: 之前的任务 ID（如果是中断任务）
        """
        # 模拟处理命令
        state_manager.set_l2_status(L2Status.BUSY, task_id)
        logger.info(f"Processing command: {text}")
        
        # 模拟生成过程
        response = ""
        if "故事" in text:
            # 模拟长时间生成（分段任务）
            total_steps = 5
            for i in range(total_steps):
                logger.info(f"Generating step {i+1}/{total_steps}...")
                time.sleep(0.5)
                
            # 生成故事响应
            response = "这是一个火星探险的故事...（故事内容）"
        else:
            time.sleep(1)
            # 生成通用响应
            response = f"我听到了您说：'{text}'。这是一个很好的输入，但我目前的 Mock 模式还只能回答特定关键词（如：你好、故事、状态、睡觉、救命）。"
        
        # 发布 L2 输出事件
        from zulong.core.event_bus import event_bus
        from zulong.core.types import EventType, ZulongEvent, EventPriority
        event_bus.publish(ZulongEvent(
            type=EventType.L2_OUTPUT,
            priority=EventPriority.NORMAL,
            source="InterruptController",
            payload={"text": response}
        ))
        
        # 任务完成，设置为 IDLE
        # 如果有活跃任务 ID，状态管理器会自动转换为 WAITING
        state_manager.set_l2_status(L2Status.IDLE)
        
        # 如果是中断任务，处理完后恢复之前的任务
        if is_interrupt and previous_task:
            logger.info(f"Interrupt task completed. Checking if previous task {previous_task} should be resumed...")
            # 给用户一个选择：继续之前的故事还是保持新话题
            # 这里我们简化处理，询问用户
            logger.info("[ZULONG] 之前的任务已暂停。输入 '继续' 恢复之前的任务，或其他指令开始新任务。")
        # 普通任务完成后，不再自动恢复任务栈中的任务
        # 而是等待用户明确输入 "继续" 指令


# 全局中断控制器实例
interrupt_controller = InterruptController()
