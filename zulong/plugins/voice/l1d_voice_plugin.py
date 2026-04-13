# File: zulong/plugins/voice/l1d_voice_plugin.py
"""
L1-D 语音唤醒插件 (示例)

TSD v1.7 对应:
- 2.2.2 L1-B 调度与意图守门层
- 4.2 L1-B 调度与电源管理

功能:
- 模拟语音唤醒词检测
- 简单命令识别
- 安静模式管理

架构:
- 实现 IL1Module 接口
- CRITICAL 优先级 (唤醒词需要立即响应)
"""

import logging
from typing import Any, Dict, List
import time

from zulong.modules.l1.core.interface import (
    IL1Module, L1PluginBase, ZulongEvent, EventPriority, EventType, create_event
)

logger = logging.getLogger(__name__)


class L1D_VoicePlugin(L1PluginBase):
    """
    L1-D 语音唤醒插件
    
    职责:
    - 监听唤醒词 ("你好", "救命")
    - 识别简单命令 ("安静", "睡觉")
    - 管理安静模式
    
    输入 (shared_memory):
    - "audio.text": 语音识别文本 (模拟)
    - "audio.confidence": 置信度
    
    输出 (ZulongEvent):
    - USER_WAKEWORD: 唤醒词事件
    - USER_SPEECH: 语音命令事件
    
    TSD v1.7 对应:
    - 4.2 L1-B 调度与电源管理
    - 3.2 智能路由逻辑 (安静模式)
    """
    
    @property
    def module_id(self) -> str:
        return "L1D/Voice"
    
    @property
    def priority(self) -> EventPriority:
        # CRITICAL 优先级：唤醒词需要立即响应
        return EventPriority.CRITICAL
    
    def initialize(self, shared_memory: Dict) -> bool:
        """初始化"""
        try:
            logger.info("🔌 [Voice] 正在初始化...")
            
            if not super().initialize(shared_memory):
                return False
            
            # 读取配置
            self._wake_words = self.get_config("wake_words", ["你好", "救命", "小紫"])
            self._silent_commands = self.get_config("silent_commands", ["安静", "睡觉", "去休息"])
            
            # 初始化共享内存
            shared_memory["voice.wakeup_detected"] = False
            shared_memory["voice.silent_mode"] = False
            shared_memory["voice.last_wakeword_time"] = 0.0
            
            self._last_recognition_time = 0.0
            
            logger.info(f"✅ [Voice] 初始化完成 (唤醒词：{self._wake_words})")
            return True
            
        except Exception as e:
            logger.error(f"❌ [Voice] 初始化失败：{e}", exc_info=True)
            return False
    
    def process_cycle(self, shared_memory: Dict) -> List[ZulongEvent]:
        """单周期处理"""
        events: List[ZulongEvent] = []
        current_time = time.time()
        
        try:
            # ========== 1. 读取语音识别文本 (模拟) ==========
            # 🎯 实际部署时应连接麦克风阵列和 VAD
            text = shared_memory.get("audio.text", "")
            confidence = shared_memory.get("audio.confidence", 0.0)
            
            if not text or confidence < 0.6:
                # 无有效语音或置信度太低
                return events
            
            # ========== 2. 唤醒词检测 ==========
            for wake_word in self._wake_words:
                if wake_word in text:
                    logger.info(f"🎙️ [Voice] 检测到唤醒词：{wake_word}")
                    
                    shared_memory["voice.wakeup_detected"] = True
                    shared_memory["voice.last_wakeword_time"] = current_time
                    
                    # 🔥 产生 USER_WAKEWORD 事件 (路由给 L1-B)
                    wakeword_event = create_event(
                        event_type=EventType.USER_WAKEWORD,
                        priority=EventPriority.CRITICAL,
                        source=self.module_id,
                        wake_word=wake_word,
                        full_text=text,
                        confidence=confidence
                    )
                    events.append(wakeword_event)
                    
                    # 如果是"救命"，立即触发紧急中断
                    if wake_word == "救命":
                        logger.warning("🚨 [Voice] 紧急呼救！")
                        # 事件会自动被 EventBus 路由给 L1-A 和 L1-B
                    
                    break
            
            # ========== 3. 安静模式命令 ==========
            for cmd in self._silent_commands:
                if cmd in text:
                    logger.info(f" [Voice] 检测到安静命令：{cmd}")
                    
                    shared_memory["voice.silent_mode"] = True
                    
                    # 🔥 产生 USER_SPEECH 事件 (路由给 L1-B)
                    speech_event = create_event(
                        event_type=EventType.USER_SPEECH,
                        priority=EventPriority.HIGH,
                        source=self.module_id,
                        text=cmd,
                        intent="SILENT_MODE",
                        confidence=confidence
                    )
                    events.append(speech_event)
                    break
            
            # ========== 4. 普通语音命令 ==========
            if not shared_memory.get("voice.wakeup_detected", False):
                # 非唤醒状态，忽略普通语音 (安静模式)
                if current_time - shared_memory.get("voice.last_wakeword_time", 0) > 30.0:
                    # 超过 30 秒未唤醒，进入安静模式
                    shared_memory["voice.silent_mode"] = True
            
        except Exception as e:
            logger.error(f"❌ [Voice] process_cycle 错误：{e}", exc_info=True)
        
        return events
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "OK",
            "details": {
                "silent_mode": self.get_config("silent_mode", False),
                "wake_words": self._wake_words
            },
            "last_update": time.time()
        }


def create_plugin(config: Dict = None) -> IL1Module:
    """工厂函数"""
    return L1D_VoicePlugin(config=config)
