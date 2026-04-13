# File: zulong/modules/l1/core/interface.py
"""
L1 层模块化架构 - 标准接口定义

TSD v1.7 对应:
- 2.2.2 L1 层拆分 (L1-A 感知反射，L1-B 调度管理)
- 3.1 事件定义 (ZulongEvent Schema)
- 4.1 L1-A 受控反射引擎

架构原则:
- 模块化软插接：所有 L1 功能模块实现统一接口
- 事件驱动：通过 ZulongEvent 通信
- 优先级调度：CRITICAL > HIGH > NORMAL > LOW
- 异常隔离：单个插件崩溃不影响其他插件
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
import time


# ========== 事件优先级定义 (TSD v1.7 第 3.1 节) ==========

class EventPriority(Enum):
    """事件优先级 (TSD v1.7 第 3.1 节)"""
    CRITICAL = 4  # 紧急事件 (摔倒、火灾、救命) - 穿透任何状态
    HIGH = 3      # 高优先级 (障碍检测、运动触发)
    NORMAL = 2    # 普通事件 (用户语音、文本)
    LOW = 1       # 低优先级 (日志、后台任务)


# ========== 事件类型定义 (TSD v1.7 第 3.1 节) ==========

class EventType(Enum):
    """事件类型 (TSD v1.7 第 3.1 节)"""
    # 传感器事件 (路由给 L1-A)
    SENSOR_OBSTACLE = "SENSOR_OBSTACLE"
    SENSOR_MOTION = "SENSOR_MOTION"
    SENSOR_FALL = "SENSOR_FALL"
    SENSOR_VISION = "SENSOR_VISION"
    SENSOR_GAS = "SENSOR_GAS"
    
    # 用户交互事件 (路由给 L1-B)
    USER_SPEECH = "USER_SPEECH"
    USER_TEXT = "USER_TEXT"
    USER_WAKEWORD = "USER_WAKEWORD"
    
    # 执行器事件 (路由给 L0)
    ACTION_MOTOR = "ACTION_MOTOR"
    ACTION_SPEAK = "ACTION_SPEAK"
    ACTION_LED = "ACTION_LED"
    
    # 系统事件
    SYSTEM_L2_COMMAND = "SYSTEM_L2_COMMAND"
    SYSTEM_STATE_CHANGE = "SYSTEM_STATE_CHANGE"
    
    # 视觉专用事件
    VISION_DATA_READY = "VISION_DATA_READY"
    SENSOR_VISION_REQUEST = "SENSOR_VISION_REQUEST"
    
    # 动态路由事件 (TSD v1.8)
    DIRECT_WAKEUP = "DIRECT_WAKEUP"           # L2 空闲时直连唤醒
    INTERACTION_TRIGGER = "INTERACTION_TRIGGER"  # L2 忙碌时触发中断


# ========== ZulongEvent 数据类 (TSD v1.7 第 3.1 节) ==========

@dataclass
class ZulongEvent:
    """
    Zulong 事件对象 (TSD v1.7 第 3.1 节)
    
    所有 L1 模块间通信必须通过此对象，确保格式统一。
    
    字段:
        id: 事件唯一标识 (UUID)
        type: 事件类型 (SENSOR_*, USER_*, ACTION_*)
        priority: 优先级 (CRITICAL, HIGH, NORMAL, LOW)
        source: 来源模块 (如 "L1A/Motor", "L1C/Vision")
        payload: 业务数据字典
        timestamp: Unix 时间戳
    """
    type: EventType
    priority: EventPriority
    source: str
    payload: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """转换为字典 (用于序列化)"""
        return {
            "id": self.id,
            "type": self.type.value,
            "priority": self.priority.value,
            "source": self.source,
            "payload": self.payload,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ZulongEvent":
        """从字典创建 (用于反序列化)"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=EventType(data["type"]),
            priority=EventPriority(data["priority"]),
            source=data["source"],
            payload=data["payload"],
            timestamp=data.get("timestamp", time.time())
        )


# ========== L1 模块抽象基类 (标准接口) ==========

class IL1Module(ABC):
    """
    L1 层模块抽象基类 (Abstract Base Class)
    
    所有 L1 插件必须实现此接口，确保:
    - 统一的生命周期管理 (initialize → process_cycle → shutdown)
    - 统一的健康检查机制
    - 统一的优先级调度
    - 异常隔离
    
    TSD v1.7 对应:
    - 2.2.2 L1 层拆分
    - 4.1 L1-A 受控反射引擎
    - 4.2 L1-B 调度与电源管理
    """
    
    # ========== 必须实现的抽象方法 ==========
    
    @property
    @abstractmethod
    def module_id(self) -> str:
        """
        模块唯一标识符
        
        格式：L1X_ModuleName (如 "L1A/Motor", "L1C/Vision")
        用于日志、调试、插件管理
        """
        pass
    
    @property
    @abstractmethod
    def priority(self) -> EventPriority:
        """
        模块默认优先级
        
        用于插件管理器排序执行:
        - CRITICAL: 气体检测、摔倒检测
        - HIGH: 障碍检测、运动触发
        - NORMAL: 视觉分析、语音识别
        - LOW: 日志记录
        """
        pass
    
    @abstractmethod
    def initialize(self, shared_memory: Dict) -> bool:
        """
        模块初始化 (插件加载时调用)
        
        Args:
            shared_memory: 共享内存字典，用于模块间数据交换
                          例如：shared_memory["motor_speed"] = 0.5
                          shared_memory["obstacle_distance"] = 1.2
        
        Returns:
            bool: 初始化是否成功 (True=成功，False=失败)
        
        功能:
        - 初始化硬件 (如电机、传感器)
        - 注册共享内存键
        - 加载配置文件
        """
        pass
    
    @abstractmethod
    def process_cycle(self, shared_memory: Dict) -> List[ZulongEvent]:
        """
        单周期处理 (插件管理器每轮循环调用)
        
        Args:
            shared_memory: 共享内存字典 (包含其他模块的输出)
        
        Returns:
            List[ZulongEvent]: 本周期产生的事件列表
        
        功能:
        - 读取传感器数据
        - 执行逻辑判断
        - 产生事件 (如障碍检测、运动触发)
        - 更新共享内存
        
        要求:
        - 快速执行 (<10ms)
        - 无阻塞 I/O
        - 异常隔离 (try-except 包裹)
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查 (定期调用，监控模块状态)
        
        Returns:
            Dict: 健康状态字典
            {
                "status": "OK" | "WARNING" | "ERROR",
                "details": {...},  # 详细信息
                "last_update": timestamp
            }
        
        功能:
        - 检查硬件连接
        - 检查数据有效性
        - 报告错误状态
        """
        pass
    
    @abstractmethod
    def shutdown(self):
        """
        模块关闭 (插件卸载时调用)
        
        功能:
        - 释放硬件资源
        - 保存配置
        - 清理共享内存
        """
        pass
    
    # ========== 可选实现的辅助方法 ==========
    
    def on_event(self, event: ZulongEvent, shared_memory: Dict):
        """
        事件回调 (可选实现)
        
        当其他模块产生与本模块相关的事件时调用
        
        Args:
            event: 事件对象
            shared_memory: 共享内存
        
        示例:
        - Motor 模块接收 "ACTION_MOTOR" 事件
        - Vision 模块接收 "SENSOR_VISION_REQUEST" 事件
        """
        pass
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        获取配置 (可选实现)
        
        Args:
            key: 配置键
            default: 默认值
        
        Returns:
            配置值
        
        用法:
        config = self.get_config("motor.max_speed", 1.0)
        """
        # 从 shared_memory 或外部配置读取
        # 这里提供默认实现，子类可覆盖
        return default


# ========== 插件基类 (提供通用实现) ==========

class L1PluginBase(IL1Module):
    """
    L1 插件基类 (提供通用实现，简化插件开发)
    
    继承此类可避免重复实现通用逻辑，只需关注业务逻辑。
    
    使用示例:
    ```python
    class L1A_MotorPlugin(L1PluginBase):
        @property
        def module_id(self): return "L1A/Motor"
        
        @property
        def priority(self): return EventPriority.HIGH
        
        def process_cycle(self, shared_memory):
            # 业务逻辑
            return events
    ```
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化插件基类
        
        Args:
            config: 配置字典 (可选)
        """
        self._config = config or {}
        self._initialized = False
        self._last_health_check = time.time()
    
    @property
    def module_id(self) -> str:
        """默认实现：使用类名"""
        return self.__class__.__name__
    
    @property
    def priority(self) -> EventPriority:
        """默认实现：NORMAL"""
        return EventPriority.NORMAL
    
    def initialize(self, shared_memory: Dict) -> bool:
        """默认实现：标记为已初始化"""
        self._initialized = True
        self._last_health_check = time.time()
        return True
    
    def health_check(self) -> Dict[str, Any]:
        """默认实现：返回 OK 状态"""
        return {
            "status": "OK",
            "details": {
                "initialized": self._initialized,
                "config": self._config
            },
            "last_update": time.time()
        }
    
    def shutdown(self):
        """默认实现：标记为已关闭"""
        self._initialized = False
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """从配置字典读取"""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


# ========== 辅助工具函数 ==========

def create_event(
    event_type: EventType,
    priority: EventPriority,
    source: str,
    **payload_kwargs
) -> ZulongEvent:
    """
    快速创建事件的辅助函数
    
    用法:
    event = create_event(
        EventType.SENSOR_OBSTACLE,
        EventPriority.HIGH,
        source="L1A/Motor",
        distance=0.5,
        direction="front"
    )
    """
    return ZulongEvent(
        type=event_type,
        priority=priority,
        source=source,
        payload=payload_kwargs
    )


def validate_event(event: ZulongEvent) -> bool:
    """
    验证事件格式是否合法
    
    检查:
    - UUID 格式正确
    - 类型合法
    - 优先级合法
    - 时间戳有效
    
    Returns:
        bool: 是否合法
    """
    try:
        # UUID 验证
        uuid.UUID(event.id)
        
        # 枚举验证
        assert isinstance(event.type, EventType)
        assert isinstance(event.priority, EventPriority)
        
        # 时间戳验证
        assert isinstance(event.timestamp, (int, float))
        assert event.timestamp > 0
        
        return True
    except Exception:
        return False
