# File: zulong/core/types.py
# 核心类型定义 - 第五阶段总装
# 对应 TSD v1.7: 增强版全局状态与事件总线

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional


# 电源状态
class PowerState(Enum):
    ACTIVE = "ACTIVE"  # 活跃模式
    SILENT = "SILENT"  # 安静模式


# L2 状态
class L2Status(Enum):
    IDLE = "IDLE"  # 空闲
    BUSY = "BUSY"  # 忙碌
    WAITING = "WAITING"  # 等待间隙（任务未结束，但在等待反馈）
    UNLOADED = "UNLOADED"  # 卸载
    REVIEW_WAITING = "REVIEW_WAITING"      # 🔥 v3.0 新增：等待用户输入
    REVIEW_ANALYZING = "REVIEW_ANALYZING"  # 🔥 v3.0 新增：分析中


# 事件类型
class EventType(Enum):
    # 用户事件
    USER_SPEECH = "USER_SPEECH"  # 用户语音（统一命名）
    USER_VOICE = "USER_VOICE"  # 兼容别名，等同于 USER_SPEECH
    USER_TEXT = "USER_TEXT"  # 用户文本输入（Web/键盘）
    USER_COMMAND = "USER_COMMAND"  # 用户命令
    
    # 传感器事件
    SENSOR_OBSTACLE = "SENSOR_OBSTACLE"  # 障碍传感器
    SENSOR_MOTION = "SENSOR_MOTION"  # 运动传感器
    SENSOR_SOUND = "SENSOR_SOUND"  # 声音传感器
    SENSOR_AUDIO = "SENSOR_AUDIO"  # 音频传感器 (L1-A 音频分析结果)
    SENSOR_FALL = "SENSOR_FALL"  # 摔倒传感器
    SENSOR_VISION = "SENSOR_VISION"  # 视觉传感器 (L1-A 视觉分析结果)
    SENSOR_VISION_STATE = "SENSOR_VISION_STATE"  # 视觉状态变更 (L1-A 运动检测结果)
    SENSOR_VIDEO_MOTION = "SENSOR_VIDEO_MOTION"  # 视频运动检测 (摄像头画面变动)
    SENSOR_VIDEO_FRAME = "SENSOR_VIDEO_FRAME"    # 视频帧事件 (每帧都发布，由 L1-A 检测运动)
    SENSOR_VISION_REQUEST = "SENSOR_VISION_REQUEST"  # 视觉捕获请求 (触发录制短视频)
    VISION_DATA_READY = "VISION_DATA_READY"  # 视觉数据就绪 (短视频文件已保存)
    
    # 🎯 动态路由架构新增事件类型
    DIRECT_WAKEUP = "DIRECT_WAKEUP"  # L2 空闲时直连唤醒 (跳过 L1-B 中断流程)
    INTERACTION_TRIGGER = "INTERACTION_TRIGGER"  # L2 忙碌时触发中断 (走 L1-B 标准流程)
    
    # 系统事件
    SYSTEM_REFLEX = "SYSTEM_REFLEX"  # 反射命令
    SYSTEM_L2_COMMAND = "SYSTEM_L2_COMMAND"  # L2 命令
    SYSTEM_L2_READY = "SYSTEM_L2_READY"  # L2 就绪
    SYSTEM_INTERRUPT = "SYSTEM_INTERRUPT"  # 系统中断
    INTERRUPT = "INTERRUPT"  # 中断（简化）
    INTERRUPT_ACK = "INTERRUPT_ACK"  # 中断确认
    SYSTEM_L3_CALL = "SYSTEM_L3_CALL"  # L3 专家调用
    SYSTEM_L3_RESPONSE = "SYSTEM_L3_RESPONSE"  # L3 专家响应
    
    # L2 输出事件
    L2_OUTPUT = "L2_OUTPUT"  # L2 输出（回复文本）
    L2_OUTPUT_STREAM = "L2_OUTPUT_STREAM"  # L2 流式输出（实时文本片段）
    
    # 🎯 语音合成事件 (TSD v1.7 规范)
    ACTION_SPEAK = "ACTION_SPEAK"  # L2 到 L0 的语音合成指令
    
    # 扬声器事件
    SPEAKER_PLAYING = "SPEAKER_PLAYING"  # 扬声器播放中
    SPEAKER_STOPPED = "SPEAKER_STOPPED"  # 扬声器停止
    
    # 专家调用事件
    EXPERT_CALL = "EXPERT_CALL"  # 专家调用请求
    EXPERT_RESULT = "EXPERT_RESULT"  # 专家调用结果
    EXPERT_ERROR = "EXPERT_ERROR"  # 专家调用错误
    
    # 任务事件
    TASK_CREATED = "TASK_CREATED"  # 任务创建
    TASK_FROZEN = "TASK_FROZEN"  # 任务冻结
    TASK_RESUMED = "TASK_RESUMED"  # 任务恢复
    TASK_COMPLETED = "TASK_COMPLETED"  # 任务完成
    TASK_EXECUTE = "TASK_EXECUTE"  # 任务执行指令
    
    # 动作事件
    ACTION_RESULT = "ACTION_RESULT"  # 动作执行结果
    SYSTEM_STATUS = "SYSTEM_STATUS"  # 系统状态更新
    
    # 电机控制命令（L0 执行器）
    CMD_EMERGENCY_STOP = "CMD_EMERGENCY_STOP"  # 紧急停止
    CMD_BRAKE = "CMD_BRAKE"  # 刹车
    CMD_SLOW_DOWN = "CMD_SLOW_DOWN"  # 减速
    CMD_BACKUP = "CMD_BACKUP"  # 后退
    
    # 🎯 事件复盘机制事件类型 (TSD v2.1)
    REPLAY_TRIGGERED = "REPLAY_TRIGGERED"  # 复盘触发 (任务失败时)
    REPLAY_DOSSIER_CREATED = "REPLAY_DOSSIER_CREATED"  # 事件档案创建完成
    REPLAY_ATTRIBUTION_DONE = "REPLAY_ATTRIBUTION_DONE"  # 归因分析完成
    REPLAY_PATCH_GENERATED = "REPLAY_PATCH_GENERATED"  # System_Patch 生成完成
    REPLAY_PATCH_APPLIED = "REPLAY_PATCH_APPLIED"  # System_Patch 应用完成
    
    # 🎯 参数校准事件类型 (TSD v2.1)
    SYSTEM_CALIBRATION = "SYSTEM_CALIBRATION"  # 系统校准事件
    CALIBRATION_APPLIED = "CALIBRATION_APPLIED"  # 校准参数已应用
    CALIBRATION_FAILED = "CALIBRATION_FAILED"  # 校准参数应用失败
    
    # 🎯 经验库事件类型 (TSD v2.1)
    EXPERIENCE_STORED = "EXPERIENCE_STORED"  # 经验已存储
    EXPERIENCE_RETRIEVED = "EXPERIENCE_RETRIEVED"  # 经验已检索


# 事件优先级
class EventPriority(Enum):
    LOW = 0  # 低优先级
    NORMAL = 1  # 正常优先级
    HIGH = 2  # 高优先级
    CRITICAL = 3  # 临界优先级


# 祖龙事件
@dataclass
class ZulongEvent:
    """祖龙系统事件"""
    type: EventType  # 事件类型
    source: str  # 事件源
    payload: Dict[str, Any]  # 事件载荷
    priority: EventPriority = EventPriority.NORMAL  # 事件优先级（默认 NORMAL）
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保类型正确
        if not isinstance(self.type, EventType):
            raise ValueError("type must be an EventType")
        if not isinstance(self.priority, EventPriority):
            raise ValueError("priority must be an EventPriority")


# 任务状态
class TaskStatus(Enum):
    RUNNING = "RUNNING"  # 运行中
    FROZEN = "FROZEN"  # 已冻结
    COMPLETED = "COMPLETED"  # 已完成


# 任务快照
@dataclass
class TaskSnapshot:
    """任务快照"""
    task_id: str  # 任务 ID
    context_history: list[Dict]  # 上下文历史
    working_memory: Dict[str, Any]  # 工作内存
    execution_pointer: str  # 执行指针
    created_at: float  # 创建时间
    last_updated: float  # 最后更新时间
