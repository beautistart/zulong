# File: zulong/l2/re_eval_node.py
# 重评估节点 - 中断后决定继续执行还是重新规划
# 对应 TSD v1.7: 恢复时先运行 re_eval_node，对比原计划与新环境

from .snapshot_manager import snapshot_manager
from .task_snapshot import TaskSnapshot, TaskStatus, IntentFrame
from zulong.core.event_bus import event_bus
from zulong.core.types import EventType, EventPriority, ZulongEvent
from typing import Dict, Any, Optional, List
from enum import Enum, auto
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='[L2-ReEval] %(message)s')
logger = logging.getLogger(__name__)


class ReEvalDecision(Enum):
    """重评估决策结果"""
    RESUME = auto()       # 继续执行原计划
    REPLAN = auto()       # 重新规划
    ABORT = auto()        # 放弃当前任务
    MERGE = auto()        # 合并新旧意图


class ReEvalNode:
    """重评估节点
    
    类比：就像你正在做饭时被打断去接电话，回来后需要决定：
    - 锅里的菜还能继续炒吗？（继续执行）
    - 还是已经糊了需要重新开始？（重新规划）
    - 或者干脆不做饭了叫外卖？（放弃任务）
    
    核心职责：
    1. 对比原计划（冻结时的意图栈）与新环境（当前感知）
    2. 评估继续执行的可行性
    3. 决定：继续、重规划、放弃或合并
    """
    
    def __init__(self):
        """初始化重评估节点"""
        self._eval_history: List[Dict] = []  # 评估历史记录
        logger.info("L2 ReEvalNode initialized")
    
    def evaluate(
        self,
        task_id: str,
        new_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """执行重评估
        
        Args:
            task_id: 要评估的任务 ID
            new_context: 新环境上下文（如传感器数据、用户新指令等）
            
        Returns:
            dict: 评估结果，包含 decision、reason、updated_snapshot 等
        """
        logger.info(f"Starting re-evaluation for task: {task_id}")
        
        # 1. 获取任务快照
        snapshot = snapshot_manager.get_snapshot(task_id)
        if not snapshot:
            logger.error(f"Task snapshot not found: {task_id}")
            return {
                "decision": ReEvalDecision.ABORT,
                "reason": "Task snapshot not found",
                "task_id": task_id
            }
        
        # 2. 分析原计划的意图栈
        original_intents = self._analyze_intents(snapshot)
        logger.info(f"Original intents: {[i['intent'] for i in original_intents]}")
        
        # 3. 分析新环境
        new_environment = self._analyze_environment(new_context or {})
        logger.info(f"New environment: {new_environment}")
        
        # 4. 对比与决策
        decision, reason = self._make_decision(snapshot, original_intents, new_environment)
        
        logger.info(f"Re-evaluation decision: {decision.name}, reason: {reason}")
        
        # 5. 根据决策执行相应操作
        result = self._execute_decision(task_id, snapshot, decision, new_environment)
        
        # 6. 记录评估历史
        eval_record = {
            "timestamp": time.time(),
            "task_id": task_id,
            "decision": decision.name,
            "reason": reason,
            "original_intents": original_intents,
            "new_environment": new_environment
        }
        self._eval_history.append(eval_record)
        
        return result
    
    def _analyze_intents(self, snapshot: TaskSnapshot) -> List[Dict]:
        """分析意图栈
        
        Args:
            snapshot: 任务快照
            
        Returns:
            list: 意图列表
        """
        intents = []
        for frame in snapshot.intent_stack:
            intents.append({
                "intent": frame.intent,
                "priority": frame.priority,
                "parameters": frame.parameters,
                "age_seconds": time.time() - frame.created_at
            })
        return intents
    
    def _analyze_environment(self, new_context: Dict[str, Any]) -> Dict[str, Any]:
        """分析新环境
        
        Args:
            new_context: 新环境上下文
            
        Returns:
            dict: 环境分析结果
        """
        environment = {
            "has_new_user_input": bool(new_context.get("user_input")),
            "has_emergency": new_context.get("emergency_level", 0) > 0,
            "emergency_level": new_context.get("emergency_level", 0),
            "new_user_input": new_context.get("user_input", ""),
            "sensor_data": new_context.get("sensor_data", {}),
            "timestamp": time.time()
        }
        return environment
    
    def _make_decision(
        self,
        snapshot: TaskSnapshot,
        original_intents: List[Dict],
        new_environment: Dict[str, Any]
    ) -> tuple:
        """做出重评估决策
        
        决策逻辑：
        1. 如果有紧急事件 -> ABORT（放弃当前任务，优先处理紧急事件）
        2. 如果有新的高优先级用户指令 -> REPLAN（重新规划）
        3. 如果原任务已完成大部分 (>80%) -> RESUME（继续执行）
        4. 如果原任务刚开始 (<20%) -> REPLAN（重新规划）
        5. 如果新旧意图相关 -> MERGE（合并意图）
        6. 其他情况 -> RESUME（默认继续）
        
        Args:
            snapshot: 任务快照
            original_intents: 原始意图列表
            new_environment: 新环境分析
            
        Returns:
            tuple: (决策, 原因)
        """
        # 规则 1: 紧急事件优先
        if new_environment["has_emergency"]:
            return ReEvalDecision.ABORT, f"Emergency detected (level {new_environment['emergency_level']})"
        
        # 规则 2: 新的高优先级用户指令
        if new_environment["has_new_user_input"]:
            user_input = new_environment["new_user_input"].lower()
            
            # 检查是否是紧急指令
            emergency_keywords = ["救命", "help", "紧急", "emergency", "危险", "danger"]
            if any(kw in user_input for kw in emergency_keywords):
                return ReEvalDecision.ABORT, "Emergency user command received"
            
            # 检查是否是停止指令
            stop_keywords = ["停止", "stop", "取消", "cancel", "别做了"]
            if any(kw in user_input for kw in stop_keywords):
                return ReEvalDecision.ABORT, "User requested to stop"
            
            # 检查是否是切换任务指令
            switch_keywords = ["切换", "switch", "换", "改为"]
            if any(kw in user_input for kw in switch_keywords):
                return ReEvalDecision.REPLAN, "User requested task switch"
        
        # 规则 3 & 4: 根据进度决定
        progress = snapshot.execution_pointer.progress_percentage
        if progress > 80:
            return ReEvalDecision.RESUME, f"Task {progress:.1f}% complete, almost done"
        elif progress < 20:
            return ReEvalDecision.REPLAN, f"Task only {progress:.1f}% complete, better to replan"
        
        # 规则 5: 检查意图相关性
        if new_environment["has_new_user_input"]:
            # 简单检查：如果新输入包含原意图的关键词，则合并
            user_input = new_environment["new_user_input"].lower()
            for intent_info in original_intents:
                intent_name = intent_info["intent"].lower()
                if intent_name in user_input or any(word in user_input for word in intent_name.split("_")):
                    return ReEvalDecision.MERGE, "New input related to original intent"
        
        # 规则 6: 默认继续
        return ReEvalDecision.RESUME, "No significant change, resume original plan"
    
    def _execute_decision(
        self,
        task_id: str,
        snapshot: TaskSnapshot,
        decision: ReEvalDecision,
        new_environment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行决策
        
        Args:
            task_id: 任务 ID
            snapshot: 任务快照
            decision: 决策结果
            new_environment: 新环境
            
        Returns:
            dict: 执行结果
        """
        result = {
            "decision": decision,
            "decision_name": decision.name,
            "task_id": task_id,
            "timestamp": time.time()
        }
        
        if decision == ReEvalDecision.RESUME:
            # 继续执行：恢复任务，保持原计划
            result["action"] = "resume_original_plan"
            result["message"] = "Resuming from where we left off"
            result["updated_snapshot"] = snapshot
            
            # 添加恢复标记到上下文
            snapshot.add_to_context(
                role="system",
                content=f"[Resumed after interruption at {snapshot.execution_pointer.progress_percentage:.1f}%]",
                metadata={"resumed": True, "progress": snapshot.execution_pointer.progress_percentage}
            )
        
        elif decision == ReEvalDecision.REPLAN:
            # 重新规划：保留上下文，但重置执行指针
            result["action"] = "replan"
            result["message"] = "Replanning based on new context"
            
            # 保存原进度到元数据
            snapshot.metadata["original_progress"] = snapshot.execution_pointer.progress_percentage
            snapshot.metadata["replanned_at"] = time.time()
            
            # 重置执行指针（但保留已生成的内容作为参考）
            old_content = snapshot.execution_pointer.generated_content
            snapshot.execution_pointer.current_step = 0
            snapshot.execution_pointer.progress_percentage = 0
            snapshot.execution_pointer.step_description = "Replanning..."
            
            # 添加重新规划标记
            snapshot.add_to_context(
                role="system",
                content=f"[Replanned. Previous progress: {snapshot.metadata['original_progress']:.1f}%. Previous content preserved as reference.]",
                metadata={"replanned": True, "previous_content": old_content}
            )
            
            result["updated_snapshot"] = snapshot
        
        elif decision == ReEvalDecision.ABORT:
            # 放弃任务：标记为放弃状态
            result["action"] = "abort"
            result["message"] = "Aborting current task due to higher priority event"
            
            snapshot.status = TaskStatus.ABANDONED
            snapshot.metadata["aborted_at"] = time.time()
            snapshot.metadata["abort_reason"] = new_environment
            
            result["updated_snapshot"] = snapshot
        
        elif decision == ReEvalDecision.MERGE:
            # 合并意图：将新意图压入栈
            result["action"] = "merge_intents"
            result["message"] = "Merging new intent with original plan"
            
            # 创建新意图帧
            new_intent = IntentFrame(
                intent=f"MERGED_{new_environment.get('new_user_input', 'unknown')[:20]}",
                parameters={"original_intents": [i["intent"] for i in self._analyze_intents(snapshot)]},
                priority=1
            )
            snapshot.push_intent(
                intent=new_intent.intent,
                parameters=new_intent.parameters,
                priority=new_intent.priority
            )
            
            # 添加合并标记
            snapshot.add_to_context(
                role="user",
                content=new_environment.get("new_user_input", ""),
                metadata={"merged": True}
            )
            
            result["updated_snapshot"] = snapshot
        
        return result
    
    def get_eval_history(self) -> List[Dict]:
        """获取评估历史"""
        return self._eval_history.copy()
    
    def clear_history(self):
        """清除评估历史"""
        self._eval_history.clear()


# 全局重评估节点实例
re_eval_node = ReEvalNode()
