# File: zulong/plugins/gas/l1e_gas_plugin.py
"""
L1-E 气体检测插件 (示例 - 高优先级)

TSD v1.7 对应:
- 2.2.2 L1-A 感知与受控反射层
- 4.1 L1-A 受控反射引擎

功能:
- 模拟 MQ-2 烟雾传感器
- 气体浓度检测
- 火灾报警

架构:
- 实现 IL1Module 接口
- CRITICAL 优先级 (安全相关)
- 异常隔离
"""

import logging
from typing import Any, Dict, List
import time
import random

from zulong.modules.l1.core.interface import (
    IL1Module, L1PluginBase, ZulongEvent, EventPriority, EventType, create_event
)

logger = logging.getLogger(__name__)


class L1E_GasPlugin(L1PluginBase):
    """
    L1-E 气体检测插件
    
    职责:
    - 读取 MQ-2 传感器数据
    - 检测烟雾/可燃气体
    - 触发火灾报警
    
    输入 (shared_memory):
    - "gas.simulate": 是否模拟模式
    - "gas.threshold": 报警阈值
    
    输出 (ZulongEvent):
    - SENSOR_GAS: 气体检测事件 (CRITICAL)
    
    TSD v1.7 对应:
    - 4.1 L1-A 受控反射引擎
    - 3.2 智能路由逻辑 (紧急穿透)
    """
    
    @property
    def module_id(self) -> str:
        return "L1E/Gas"
    
    @property
    def priority(self) -> EventPriority:
        # CRITICAL 优先级：安全相关，必须立即响应
        return EventPriority.CRITICAL
    
    def initialize(self, shared_memory: Dict) -> bool:
        """初始化"""
        try:
            logger.info("🔌 [Gas] 正在初始化...")
            
            if not super().initialize(shared_memory):
                return False
            
            # 读取配置
            self._threshold = self.get_config("threshold", 500)  # ppm
            self._simulate = self.get_config("simulate", True)
            
            # 初始化共享内存
            shared_memory["gas.concentration"] = 0
            shared_memory["gas.alarm"] = False
            shared_memory["gas.last_reading"] = 0.0
            
            self._last_alarm_time = 0.0
            self._alarm_cooldown = 60.0  # 报警冷却 60 秒
            
            logger.info(f"✅ [Gas] 初始化完成 (阈值：{self._threshold}ppm, 模拟：{self._simulate})")
            return True
            
        except Exception as e:
            logger.error(f"❌ [Gas] 初始化失败：{e}", exc_info=True)
            return False
    
    def process_cycle(self, shared_memory: Dict) -> List[ZulongEvent]:
        """单周期处理"""
        events: List[ZulongEvent] = []
        current_time = time.time()
        
        try:
            # ========== 1. 读取传感器数据 ==========
            if self._simulate:
                # 模拟模式：随机生成 (1% 概率检测到高浓度)
                if random.random() < 0.01:
                    concentration = random.randint(600, 1000)  # 高浓度
                else:
                    concentration = random.randint(0, 100)  # 正常
            else:
                # 真实传感器 (模拟读取)
                concentration = shared_memory.get("gas.concentration", 0)
            
            # 更新共享内存
            shared_memory["gas.concentration"] = concentration
            shared_memory["gas.last_reading"] = current_time
            
            # ========== 2. 报警检测 ==========
            if concentration > self._threshold:
                # 检查冷却时间
                if current_time - self._last_alarm_time > self._alarm_cooldown:
                    logger.warning(f"🚨 [Gas] 检测到危险气体：{concentration}ppm!")
                    
                    shared_memory["gas.alarm"] = True
                    self._last_alarm_time = current_time
                    
                    # 🔥 产生 CRITICAL 优先级事件 (穿透任何状态)
                    gas_event = create_event(
                        event_type=EventType.SENSOR_GAS,
                        priority=EventPriority.CRITICAL,
                        source=self.module_id,
                        concentration=concentration,
                        threshold=self._threshold,
                        requires_evacuation=True,
                        alarm_sound="fire_alarm"
                    )
                    events.append(gas_event)
                    
                    # 🔥 同时产生 ACTION 事件 (播放警报)
                    action_event = create_event(
                        event_type=EventType.ACTION_SPEAK,
                        priority=EventPriority.CRITICAL,
                        source=self.module_id,
                        text="检测到烟雾！请立即疏散！",
                        style="emergency",
                        voice_mode="AUTO_TTS"
                    )
                    events.append(action_event)
            
            else:
                shared_memory["gas.alarm"] = False
            
        except Exception as e:
            logger.error(f"❌ [Gas] process_cycle 错误：{e}", exc_info=True)
        
        return events
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "OK",
            "details": {
                "threshold": self._threshold,
                "simulate": self._simulate,
                "last_alarm": self._last_alarm_time
            },
            "last_update": time.time()
        }


def create_plugin(config: Dict = None) -> IL1Module:
    """工厂函数"""
    return L1E_GasPlugin(config=config)
