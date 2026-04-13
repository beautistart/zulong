# File: zulong/l1b/attention_controller.py
# L1-B 注意力控制器 - 三层注意力机制核心

"""
祖龙 (ZULONG) 系统 - L1-B 注意力控制器

核心功能:
1. 中断处理：检测紧急事件，强制中断 L2 当前任务
2. 任务冻结：保存 L2 上下文快照 (KV Cache + 对话历史)
3. Prompt 重组：打包 [紧急事件] + [旧任务摘要] + [恢复指令]
4. 事件队列：管理低优先级事件的排队处理
5. 空闲恢复：L2 空闲时恢复暂停的任务

TSD v1.8 对应:
- 2.4 任务冻结与重组算法
- 3.3 上下文快照管理
- 4.2.1 L1-B 注意力控制器
"""

import queue
import time
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from zulong.core.types import ZulongEvent, EventType, EventPriority
from zulong.core.event_bus import event_bus
from zulong.core.attention_atoms import (
    AttentionEvent, AttentionLayer, EventType as AttentionEventType,
    ContextSnapshot, MacroCommand
)
from zulong.l2.l2_snapshot_interface import get_l2_snapshot_interface

logger = logging.getLogger("AttentionController")


@dataclass
class QueuedEvent:
    """排队事件"""
    priority: int
    event: AttentionEvent
    timestamp: float
    
    def __lt__(self, other):
        """用于优先级队列排序 (优先级高的在前)"""
        return self.priority > other.priority  # 注意：优先级数字越大优先级越高


class AttentionController:
    """
    L1-B 注意力控制器
    
    职责:
    - 接收来自 L1 层 (Vision, Audio, Radar) 的注意力事件
    - 判断事件优先级，决定是否中断 L2
    - 管理任务冻结、快照保存、Prompt 重组
    - 在 L2 空闲时恢复任务或处理排队事件
    
    TSD v1.8 对应:
    - 4.2.1 L1-B 注意力控制器
    - 2.4.2 任务冻结机制
    """
    
    def __init__(self):
        """初始化注意力控制器"""
        self.event_bus = event_bus
        
        # 状态管理
        self.status = "IDLE"  # IDLE, BUSY, SUSPENDED
        self.active_snapshot: Optional[ContextSnapshot] = None
        self.current_task_id: Optional[str] = None
        
        # 事件队列 (优先级队列)
        self.event_queue: queue.PriorityQueue = queue.PriorityQueue()
        
        # 配置
        self.interrupt_threshold = 8  # 中断优先级阈值 (>=8 强制中断)
        self.high_priority_threshold = 5  # 高优先级阈值 (>=5 排队优先处理)
        
        # 统计信息
        self.stats = {
            "total_events": 0,
            "interrupts_triggered": 0,
            "tasks_frozen": 0,
            "tasks_resumed": 0,
            "events_queued": 0
        }
        
        logger.info("🧠 [AttentionController] 初始化完成")
        logger.info(f"   - 中断阈值：{self.interrupt_threshold}")
        logger.info(f"   - 高优阈值：{self.high_priority_threshold}")
    
    def tick(self, events: List[AttentionEvent]):
        """
        每一帧调用，处理所有 incoming events
        
        TSD v1.8 对应:
        - 4.2.1 事件处理循环
        - 2.4.1 中断检测逻辑
        
        Args:
            events: 注意力事件列表
        """
        for evt in events:
            self.stats["total_events"] += 1
            
            # 1. 紧急事件检测 (强制中断)
            if evt.type == AttentionEventType.EMERGENCY_ALERT or evt.is_interrupt_level():
                logger.warning(f"⚡ [AttentionController] 紧急事件触发中断：{evt.source}")
                self._handle_interrupt(evt)
                
            # 2. L2 忙碌时的策略
            elif self.status == "BUSY":
                if evt.priority >= self.interrupt_threshold:
                    # 高优打断
                    logger.info(f"⚡ [AttentionController] 高优事件打断：priority={evt.priority}")
                    self._handle_interrupt(evt)
                else:
                    # 低优排队 (静默注意产生的普通交互)
                    logger.debug(f"⏳ [AttentionController] 事件排队：priority={evt.priority}")
                    self._queue_event(evt)
                    
            # 3. L2 空闲时直接路由
            elif self.status == "IDLE":
                if evt.priority >= self.high_priority_threshold:
                    # 高优事件直接路由到 L2
                    logger.info(f"➡️ [AttentionController] 高优事件直通 L2：priority={evt.priority}")
                    self._route_to_l2_direct(evt)
                else:
                    # 低优事件也排队 (批量处理)
                    logger.debug(f"⏳ [AttentionController] 事件排队 (空闲时)：priority={evt.priority}")
                    self._queue_event(evt)
            else:
                # SUSPENDED 状态：所有事件都排队
                logger.debug(f"⏳ [AttentionController] 事件排队 (SUSPENDED 状态)：priority={evt.priority}")
                self._queue_event(evt)
    
    def _handle_interrupt(self, evt: AttentionEvent):
        """
        核心流程：冻结旧任务 -> 打包新上下文 -> 注入 L2
        
        TSD v1.8 对应:
        - 2.4.2 任务冻结机制
        - 3.3.2 Prompt 重组算法
        
        Args:
            evt: 紧急注意力事件
        """
        logger.warning(f"⚡ [AttentionController] ⚡ 中断触发：{evt.payload}")
        
        # 1. 冻结 (Freeze)
        if self.status == "BUSY" and self.current_task_id:
            # 请求 L2 生成摘要并保存状态
            self.active_snapshot = self._create_l2_snapshot()
            logger.info(
                f"🧊 [AttentionController] 任务冻结："
                f"{self.active_snapshot.task_id} -> {self.active_snapshot.summary}"
            )
            self.stats["tasks_frozen"] += 1
        
        self.status = "SUSPENDED"
        
        # 2. 重组 (Recompose)
        # 构造 Prompt：[紧急事件] + [旧任务摘要] + [指令]
        emergency_context = self._format_emergency_context(evt)
        
        old_task_context = ""
        if self.active_snapshot:
            old_task_context = (
                f"📝 [暂停的任务]: '{self.active_snapshot.summary}'\n"
                f"系统提示：请先处理紧急事件。处理完毕后，我会询问是否恢复该任务。\n"
            )
        
        recomposed_prompt = (
            f"{emergency_context}\n"
            f"{old_task_context}\n"
            f"请给出简短的应对指令或宏观控制命令 (JSON 格式)。"
        )
        
        logger.info(f"📦 [AttentionController] 重组 Prompt:\n{recomposed_prompt[:200]}...")
        
        # 3. 注入 (Inject)
        # 强制 L2 清空当前生成流，立即响应新 Prompt
        self._force_l2_respond(recomposed_prompt, priority="IMMEDIATE")
        
        self.stats["interrupts_triggered"] += 1
    
    def on_l2_idle(self):
        """
        当 L2 报告空闲时，检查是否有挂起任务或排队事件
        
        TSD v1.8 对应:
        - 2.4.3 任务恢复机制
        - 4.2.1 空闲事件处理
        """
        logger.debug(f"🔄 [AttentionController] L2 空闲检查：status={self.status}")
        
        # 1. 恢复暂停的任务
        if self.status == "SUSPENDED" and self.active_snapshot:
            logger.info(f"▶️ [AttentionController] 恢复暂停的任务：{self.active_snapshot.task_id}")
            
            resume_prompt = (
                f"✅ 紧急事件已处理。\n"
                f"现在恢复之前的任务：'{self.active_snapshot.summary}'。\n"
                f"请继续之前的工作。"
            )
            
            self._load_l2_snapshot(self.active_snapshot)
            self._force_l2_respond(resume_prompt, priority="NORMAL")
            
            self.active_snapshot = None
            self.status = "IDLE"
            self.stats["tasks_resumed"] += 1
        
        # 2. 处理排队事件
        queued_count = 0
        while not self.event_queue.empty() and queued_count < 3:  # 每次最多处理 3 个
            try:
                queued_event: QueuedEvent = self.event_queue.get_nowait()
                logger.info(
                    f"➡️ [AttentionController] 处理排队事件："
                    f"priority={queued_event.priority}, source={queued_event.event.source}"
                )
                self._route_to_l2_direct(queued_event.event)
                queued_count += 1
            except queue.Empty:
                break
    
    def _queue_event(self, evt: AttentionEvent):
        """
        将事件加入排队队列
        
        Args:
            evt: 注意力事件
        """
        queued = QueuedEvent(
            priority=evt.priority,
            event=evt,
            timestamp=time.time()
        )
        self.event_queue.put(queued)
        self.stats["events_queued"] += 1
        logger.debug(f"📥 [AttentionController] 事件入队：total={self.event_queue.qsize()}")
    
    def _create_l2_snapshot(self) -> Optional[ContextSnapshot]:
        """
        创建 L2 上下文快照
        
        TSD v1.8 对应:
        - 3.3.1 上下文快照数据结构
        
        Returns:
            ContextSnapshot: L2 上下文快照，如果失败则返回 None
        """
        logger.info("📸 [AttentionController] 创建 L2 快照...")
        
        # 🎯 使用 L2 快照接口创建真实快照
        l2_interface = get_l2_snapshot_interface()
        
        # 先冻结当前任务
        frozen_snapshot = l2_interface.freeze_current_task()
        
        if frozen_snapshot:
            logger.info(f"📸 [AttentionController] 快照创建完成：task_id={frozen_snapshot.task_id}")
            logger.info(f"   - 摘要：{frozen_snapshot.summary}")
            logger.info(f"   - 上下文长度：{len(frozen_snapshot.full_history)}")
            return frozen_snapshot
        else:
            # 如果没有活跃任务，创建一个空快照
            logger.warning("⚠️ [AttentionController] 无活跃任务，创建空快照")
            snapshot = ContextSnapshot(
                task_id=self.current_task_id or f"task_{int(time.time())}",
                summary="未知任务",
                full_history=[],
                kv_cache_ptr=None,
                generation_state={},
                pause_reason="emergency_interrupt"
            )
            return snapshot
    
    def _load_l2_snapshot(self, snapshot: ContextSnapshot):
        """
        加载 L2 上下文快照
        
        TSD v1.8 对应:
        - 3.3.1 上下文快照恢复
        
        Args:
            snapshot: L2 上下文快照
        """
        logger.info(f"📥 [AttentionController] 加载 L2 快照：task_id={snapshot.task_id}")
        
        # 🎯 使用 L2 快照接口恢复真实快照
        l2_interface = get_l2_snapshot_interface()
        
        # 恢复快照（KV Cache token 数量从快照中获取）
        kv_cache_tokens = snapshot.kv_cache.num_tokens if snapshot.kv_cache else 0
        
        success = l2_interface.restore_context_snapshot(snapshot, kv_cache_tokens)
        
        if success:
            self.current_task_id = snapshot.task_id
            logger.info(f"📥 [AttentionController] 快照加载完成：task_id={self.current_task_id}")
        else:
            logger.error(f"❌ [AttentionController] 快照加载失败：task_id={snapshot.task_id}")
    
    def _format_emergency_context(self, evt: AttentionEvent) -> str:
        """
        格式化紧急事件上下文
        
        Args:
            evt: 紧急注意力事件
            
        Returns:
            str: 格式化的紧急事件上下文
        """
        source_map = {
            "l1a_vision_processor": "视觉",
            "l1d_audio_processor": "听觉",
            "l1f_radar_processor": "雷达"
        }
        
        source_name = source_map.get(evt.source, evt.source)
        action = evt.payload.get("action", "未知事件")
        details = evt.payload.get("state", "")
        
        return f"⚠️ 紧急事件：{source_name} 检测到 {action} ({details})"
    
    def _route_to_l2_direct(self, evt: AttentionEvent):
        """
        直接路由事件到 L2 (直通模式)
        
        Args:
            evt: 注意力事件
        """
        prompt = f"[{evt.source}] {evt.payload}"
        logger.debug(f"➡️ [AttentionController] 直通 L2: {prompt[:100]}")
        
        # 🎯 简化实现：实际需要调用 L2 的事件处理接口
        # self.l2.process_event(prompt)
        
        # 临时使用事件总线
        zulong_event = ZulongEvent(
            type=EventType.SYSTEM_L2_COMMAND,
            priority=EventPriority.HIGH if evt.priority >= 5 else EventPriority.NORMAL,
            source=evt.source,
            payload=evt.payload
        )
        self.event_bus.publish(zulong_event)
    
    def _force_l2_respond(self, prompt: str, priority: str = "NORMAL"):
        """
        强制 L2 立即响应
        
        Args:
            prompt: Prompt 文本
            priority: 优先级 ("IMMEDIATE" 或 "NORMAL")
        """
        logger.info(f"⚡ [AttentionController] 强制 L2 响应：priority={priority}")
        
        # 🎯 简化实现：实际需要调用 L2 的 force_respond 方法
        # self.l2.force_respond(prompt, priority=priority)
        
        # 临时使用事件总线
        zulong_event = ZulongEvent(
            type=EventType.SYSTEM_INTERRUPT if priority == "IMMEDIATE" else EventType.SYSTEM_L2_COMMAND,
            priority=EventPriority.CRITICAL if priority == "IMMEDIATE" else EventPriority.HIGH,
            source="attention_controller",
            payload={"prompt": prompt, "priority": priority}
        )
        self.event_bus.publish(zulong_event)
    
    def set_current_task(self, task_id: str):
        """
        设置当前任务 ID
        
        Args:
            task_id: 任务 ID
        """
        self.current_task_id = task_id
        self.status = "BUSY"
        logger.debug(f"📝 [AttentionController] 设置当前任务：{task_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            **self.stats,
            "status": self.status,
            "queued_events": self.event_queue.qsize(),
            "active_snapshot": self.active_snapshot.task_id if self.active_snapshot else None
        }


# 全局单例
_attention_controller: Optional[AttentionController] = None


def get_attention_controller() -> AttentionController:
    """获取注意力控制器单例"""
    global _attention_controller
    if _attention_controller is None:
        _attention_controller = AttentionController()
    return _attention_controller
