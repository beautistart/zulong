# File: zulong/core/attention_atoms.py
# 定义祖龙系统的注意力原子类

"""
祖龙 (ZULONG) 系统 - 注意力机制原子类

核心改进点:
1. L0/L1 分层过滤：
   - L0 (硬件/驱动层): 完全静默，仅做信号采集，不产生任何事件
   - L1 (感知插件层): "静默注意"模式，持续运行微型模型，但默认不生成事件
   - 只有当检测到"交互意图"时，才实例化 AttentionEvent 并向上路由

2. 任务冻结与重组算法：
   - ContextSnapshot: 提取当前对话摘要 + 保存 KV Cache 指针
   - PromptRecomposer: 打包格式 [紧急事件上下文] + [旧任务摘要] + [恢复指令]

3. 宏微融合执行器：
   - 统一接收多模态数据流（视觉坐标、雷达距离、声源定位）
   - 融合 L2 宏观指令，输出电机 PWM

TSD v1.8 对应:
- 2.3 三层注意力机制
- 3.2 事件驱动架构增强
- 4.1 感知预处理 - 静默注意
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import time


class AttentionLayer(Enum):
    """
    注意力层级枚举
    
    TSD v1.8 对应:
    - 2.3.1 L0: 纯数据采集，无注意力，无事件
    - 2.3.2 L1: 静默注意：持续推理，仅在触发阈值时生成事件
    - 2.3.3 L2: 交互注意：已生成事件，等待路由
    - 2.3.4 L3: 认知层：L2 正在处理
    """
    L0_SENSOR = 0           # 纯数据采集，无注意力，无事件
    L1_SILENT = 1           # 静默注意：持续推理，仅在触发阈值时生成事件
    L2_INTERACTIVE = 2      # 交互注意：已生成事件，等待路由
    L3_COGNITIVE = 3        # 认知层：L2 正在处理


class EventType(Enum):
    """
    事件类型枚举
    
    TSD v1.8 对应:
    - SILENT_OBSERVATION: 仅记录，不上报
    - INTERACTION_TRIGGER: 触发路由
    - EMERGENCY_ALERT: 强制中断
    """
    SILENT_OBSERVATION = "silent_obs"      # 仅记录，不上报
    INTERACTION_TRIGGER = "interaction_trigger"  # 触发路由
    EMERGENCY_ALERT = "emergency_alert"    # 强制中断


@dataclass
class AttentionEvent:
    """
    注意力事件原子
    
    只有当 L1 插件从 'SILENT' 状态跃迁到 'TRIGGER' 状态时才生成此对象
    
    TSD v1.8 对应:
    - 3.2.1 事件数据结构
    - 4.1.2 静默注意触发条件
    """
    id: str
    source: str           # e.g., 'l1c_vision', 'l1d_voice', 'l1f_radar'
    type: EventType
    priority: int         # 1-10, 10 最高 (CRITICAL)
    payload: Dict[str, Any]  # 关键数据 (坐标，文本，置信度)
    timestamp: float = field(default_factory=time.time)
    
    def is_interrupt_level(self) -> bool:
        """判断是否为中断级别 (优先级 >= 8)"""
        return self.priority >= 8
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式 (用于日志和调试)"""
        return {
            "id": self.id,
            "source": self.source,
            "type": self.type.value,
            "priority": self.priority,
            "payload": self.payload,
            "timestamp": self.timestamp
        }


@dataclass
class ContextSnapshot:
    """
    任务冻结快照
    
    当紧急事件触发中断时，保存当前 L2 的上下文状态
    
    TSD v1.8 对应:
    - 2.4.2 任务冻结机制
    - 3.3.1 上下文快照数据结构
    
    属性:
        task_id: 任务 ID
        summary: 旧任务的一句话摘要 (由 L2 实时维护或 L1-B 提取)
        full_history: 完整对话历史 (可选，节省空间则只存摘要)
        kv_cache_ptr: 显存中 KV Cache 的指针或序列化块
        generation_state: 当前生成的 token 进度
        pause_reason: 暂停原因
    """
    task_id: str
    summary: str          # 旧任务的一句话摘要
    full_history: List[Dict] = field(default_factory=list)  # 完整对话历史
    kv_cache_ptr: Any = None  # 显存中 KV Cache 的指针或序列化块
    generation_state: Dict = field(default_factory=dict)  # 当前生成的 token 进度
    pause_reason: str = ""  # 暂停原因
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "task_id": self.task_id,
            "summary": self.summary,
            "full_history": self.full_history,
            "kv_cache_ptr": self.kv_cache_ptr,
            "generation_state": self.generation_state,
            "pause_reason": self.pause_reason
        }


@dataclass
class MacroCommand:
    """
    L2 -> L1-A 宏观指令
    
    L2 认知层产生的高级意图指令，由 L1-A 融合执行器解析为微观动作
    
    TSD v1.8 对应:
    - 2.5.1 宏微融合架构
    - 4.3.2 L2 输出格式
    
    属性:
        cmd_id: 指令 ID
        intent: 意图 e.g., "GRASP", "NAVIGATE", "FOLLOW"
        targets: 目标 {"object": "apple", "color": "red", "zone": "kitchen"}
        constraints: 约束 {"speed": "slow", "force": "soft"}
        source_snapshot_id: 关联的任务 ID
    """
    cmd_id: str
    intent: str           # e.g., "GRASP", "NAVIGATE", "FOLLOW"
    targets: Dict[str, Any]  # {"object": "apple", "color": "red", "zone": "kitchen"}
    constraints: Dict[str, Any]  # {"speed": "slow", "force": "soft"}
    source_snapshot_id: Optional[str] = None  # 关联的任务 ID
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "cmd_id": self.cmd_id,
            "intent": self.intent,
            "targets": self.targets,
            "constraints": self.constraints,
            "source_snapshot_id": self.source_snapshot_id
        }


@dataclass
class SensorFusionData:
    """
    多模态传感器融合数据
    
    由 L1 层各传感器插件持续写入共享内存，即使没有事件
    
    TSD v1.8 对应:
    - 4.1.3 共享内存机制
    - 4.3.1 多模态数据流
    
    属性:
        vision_target_pos: 视觉目标位置 [x, y, z]
        radar_obstacles: 雷达障碍物列表
        audio_source_pos: 声源位置 [x, y, z]
        timestamp: 时间戳
    """
    vision_target_pos: Optional[List[float]] = None  # 视觉目标位置 [x, y, z]
    radar_obstacles: Optional[List[Dict]] = None  # 雷达障碍物列表
    audio_source_pos: Optional[List[float]] = None  # 声源位置 [x, y, z]
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "vision_target_pos": self.vision_target_pos,
            "radar_obstacles": self.radar_obstacles,
            "audio_source_pos": self.audio_source_pos,
            "timestamp": self.timestamp
        }
