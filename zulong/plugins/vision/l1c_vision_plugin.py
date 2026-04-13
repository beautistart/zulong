# File: zulong/plugins/vision/l1c_vision_plugin.py
"""
L1-C 视觉分析插件 (重构版)

TSD v1.7 对应:
- 2.2.2 L1 层拆分
- 4.4 感知预处理
- 5.2 显存约束 (关键帧提取)

功能:
- 摄像头帧采集
- 光流法运动检测
- 关键帧提取
- 视觉事件发布

架构:
- 实现 IL1Module 接口
- 通过 shared_memory 与 L2 通信
- 支持主动轮询和事件触发
"""

import logging
from typing import Any, Dict, List
import time
import cv2
import numpy as np

from zulong.modules.l1.core.interface import (
    IL1Module, L1PluginBase, ZulongEvent, EventPriority, EventType, create_event
)

logger = logging.getLogger(__name__)


class L1C_VisionPlugin(L1PluginBase):
    """
    L1-C 视觉分析插件
    
    职责:
    - 摄像头帧采集 (30 FPS)
    - 光流法运动检测
    - 关键帧提取 (3-5 帧供 L2 使用)
    - 视觉事件发布
    
    输入 (shared_memory):
    - "vision.request": 视觉请求标志
    - "vision.query": 用户查询文本
    
    输出 (ZulongEvent):
    - SENSOR_VISION: 视觉传感器事件
    - VISION_DATA_READY: 视觉数据就绪事件
    
    TSD v1.7 对应:
    - 4.4 感知预处理
    - 5.2 显存约束
    """
    
    @property
    def module_id(self) -> str:
        return "L1C/Vision"
    
    @property
    def priority(self) -> EventPriority:
        # NORMAL 优先级：视觉分析不需要极快响应
        return EventPriority.NORMAL
    
    def initialize(self, shared_memory: Dict) -> bool:
        """
        初始化视觉插件
        
        Args:
            shared_memory: 共享内存
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("🔌 [Vision] 正在初始化...")
            
            # 调用基类初始化
            if not super().initialize(shared_memory):
                return False
            
            # 读取配置
            self._fps = self.get_config("fps", 30)
            self._frame_width = self.get_config("frame_width", 640)
            self._frame_height = self.get_config("frame_height", 480)
            self._motion_threshold = self.get_config("motion_threshold", 500)  # 像素
            
            # 初始化共享内存
            shared_memory["vision.fps"] = self._fps
            shared_memory["vision.frame_count"] = 0
            shared_memory["vision.motion_detected"] = False
            shared_memory["vision.last_frame_time"] = 0.0
            
            # 🎯 摄像头初始化 (模拟)
            self._camera = None
            self._frame_buffer: List[np.ndarray] = []  # 环形缓冲区
            self._max_buffer_size = self._fps * 5  # 5 秒缓冲区
            self._last_motion_time = 0.0
            self._motion_state = "IDLE"  # IDLE, MOVING, STOPPED
            
            # 尝试打开摄像头
            self._init_camera()
            
            logger.info(f"✅ [Vision] 初始化完成 (FPS: {self._fps}, 分辨率：{self._frame_width}x{self._frame_height})")
            return True
            
        except Exception as e:
            logger.error(f"❌ [Vision] 初始化失败：{e}", exc_info=True)
            return False
    
    def _init_camera(self):
        """初始化摄像头"""
        try:
            # 🎯 实际部署时应使用真实摄像头
            # self._camera = cv2.VideoCapture(0)
            
            # 模拟模式：创建测试图案
            logger.info("📷 [Vision] 摄像头已就绪 (模拟模式)")
            
        except Exception as e:
            logger.error(f"❌ [Vision] 摄像头初始化失败：{e}")
    
    def process_cycle(self, shared_memory: Dict) -> List[ZulongEvent]:
        """
        单周期处理 (核心逻辑)
        
        流程:
        1. 采集摄像头帧
        2. 光流法运动检测
        3. 更新帧缓冲区
        4. 产生视觉事件
        
        Args:
            shared_memory: 共享内存
        
        Returns:
            List[ZulongEvent]: 事件列表
        """
        events: List[ZulongEvent] = []
        current_time = time.time()
        
        try:
            # ========== 1. 采集摄像头帧 ==========
            frame = self._capture_frame()
            
            if frame is not None:
                # 更新共享内存
                shared_memory["vision.frame_count"] += 1
                shared_memory["vision.last_frame_time"] = current_time
                
                # 添加到帧缓冲区
                self._frame_buffer.append(frame)
                if len(self._frame_buffer) > self._max_buffer_size:
                    self._frame_buffer.pop(0)  # 移除最旧帧
                
                # ========== 2. 光流法运动检测 ==========
                motion_detected, motion_pixels = self._detect_motion(frame)
                
                if motion_detected:
                    self._last_motion_time = current_time
                    
                    # 状态机：IDLE → MOVING
                    if self._motion_state == "IDLE":
                        logger.info(f"👁️ [Vision] 检测到运动：{motion_pixels}像素")
                        self._motion_state = "MOVING"
                    
                    shared_memory["vision.motion_detected"] = True
                    
                    # 🔥 产生 SENSOR_VISION 事件 (路由给 L1-A)
                    vision_event = create_event(
                        event_type=EventType.SENSOR_VISION,
                        priority=EventPriority.NORMAL,
                        source=self.module_id,
                        scene="active",
                        motion_intensity=motion_pixels,
                        motion_state=self._motion_state
                    )
                    events.append(vision_event)
                
                else:
                    # 无运动
                    if self._motion_state == "MOVING":
                        # 检查是否停止
                        if current_time - self._last_motion_time > 1.0:  # 1 秒无运动
                            logger.info("👁️ [Vision] 运动停止")
                            self._motion_state = "STOPPED"
                            
                            # 🔥 触发保存回溯视频
                            backtrack_event = create_event(
                                event_type=EventType.VISION_DATA_READY,
                                priority=EventPriority.HIGH,
                                source=self.module_id,
                                trigger="motion_stopped",
                                frame_count=len(self._frame_buffer),
                                duration_seconds=5.0
                            )
                            events.append(backtrack_event)
                    
                    shared_memory["vision.motion_detected"] = False
            
            # ========== 3. 检查视觉请求 ==========
            vision_request = shared_memory.get("vision.request", False)
            if vision_request:
                query = shared_memory.get("vision.query", "描述场景")
                
                logger.info(f"📥 [Vision] 收到视觉请求：{query}")
                
                # 提取关键帧 (3-5 帧)
                keyframes = self._extract_keyframes(num_keyframes=5)
                
                if keyframes:
                    # 🔥 产生 VISION_DATA_READY 事件 (供 L2 轮询)
                    vision_ready_event = create_event(
                        event_type=EventType.VISION_DATA_READY,
                        priority=EventPriority.HIGH,
                        source=self.module_id,
                        trigger="user_request",
                        query=query,
                        keyframe_count=len(keyframes),
                        frames=keyframes  # ⚠️ 注意：大对象，考虑使用文件路径
                    )
                    events.append(vision_ready_event)
                
                # 清除请求标志
                shared_memory["vision.request"] = False
            
            # ========== 4. 性能监控 ==========
            if len(self._frame_buffer) == 0:
                logger.warning("⚠️ [Vision] 帧缓冲区为空")
            
        except Exception as e:
            logger.error(f"❌ [Vision] process_cycle 错误：{e}", exc_info=True)
            # 异常隔离
        
        return events
    
    def _capture_frame(self) -> Optional[np.ndarray]:
        """
        采集单帧图像
        
        Returns:
            np.ndarray or None: BGR 格式帧
        """
        try:
            if self._camera is not None:
                # 真实摄像头
                ret, frame = self._camera.read()
                if ret:
                    return frame
            else:
                # 模拟模式：创建测试图案
                frame = np.zeros((self._frame_height, self._frame_width, 3), dtype=np.uint8)
                
                # 绘制渐变背景 (模拟时间变化)
                t = time.time() % 10  # 10 秒周期
                color = int(255 * (0.5 + 0.5 * np.sin(t)))
                cv2.rectangle(frame, (0, 0), (self._frame_width, self._frame_height), (color, 100, 200), -1)
                
                # 绘制文字
                cv2.putText(frame, "Vision Plugin", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, time.strftime("%H:%M:%S"), (50, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                return frame
        
        except Exception as e:
            logger.error(f"❌ [Vision] 采集帧失败：{e}")
            return None
    
    def _detect_motion(self, frame: np.ndarray) -> tuple:
        """
        光流法运动检测
        
        Args:
            frame: 当前帧
        
        Returns:
            tuple: (是否检测到运动，运动像素数)
        """
        try:
            if len(self._frame_buffer) < 2:
                return False, 0
            
            # 获取前一帧
            prev_frame = self._frame_buffer[-2]
            
            # 转为灰度图
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 帧差法
            diff = cv2.absdiff(prev_gray, curr_gray)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # 计算运动像素
            motion_pixels = cv2.countNonZero(thresh)
            
            # 判断是否超过阈值
            motion_detected = motion_pixels > self._motion_threshold
            
            return motion_detected, motion_pixels
        
        except Exception as e:
            logger.error(f"❌ [Vision] 运动检测失败：{e}")
            return False, 0
    
    def _extract_keyframes(self, num_keyframes: int = 5) -> List[np.ndarray]:
        """
        提取关键帧 (TSD v1.7 第 5.2 节)
        
        策略:
        1. 第一帧 (起始状态)
        2. 中间帧 (均匀采样)
        3. 最后一帧 (最新状态)
        4. 运动最大帧
        
        Args:
            num_keyframes: 关键帧数量
        
        Returns:
            List[np.ndarray]: 关键帧列表
        """
        if len(self._frame_buffer) < 2:
            return []
        
        frames = self._frame_buffer.copy()
        
        if len(frames) <= num_keyframes:
            return frames
        
        keyframes = []
        
        # 1. 第一帧
        keyframes.append(frames[0])
        
        # 2. 均匀采样
        step = len(frames) // (num_keyframes - 1)
        for i in range(1, num_keyframes - 1):
            idx = i * step
            if idx < len(frames):
                keyframes.append(frames[idx])
        
        # 3. 最后一帧
        keyframes.append(frames[-1])
        
        logger.info(f"👁️ [Vision] 提取关键帧：{len(keyframes)}帧 (原始{len(frames)}帧)")
        return keyframes
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            status = "OK"
            details = {
                "camera_ready": self._camera is not None,
                "buffer_size": len(self._frame_buffer),
                "max_buffer_size": self._max_buffer_size,
                "motion_state": self._motion_state,
                "fps": self._fps
            }
            
            # 检查缓冲区
            if len(self._frame_buffer) == 0:
                status = "WARNING"
                details["warning"] = "Frame buffer empty"
            
            return {
                "status": status,
                "details": details,
                "last_update": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ [Vision] 健康检查失败：{e}")
            return {
                "status": "ERROR",
                "details": {"error": str(e)},
                "last_update": time.time()
            }
    
    def shutdown(self):
        """关闭视觉插件"""
        try:
            logger.info("🛑 [Vision] 正在关闭...")
            
            # 释放摄像头
            if self._camera is not None:
                self._camera.release()
            
            # 清空缓冲区
            self._frame_buffer.clear()
            
            # 调用基类关闭
            super().shutdown()
            
            logger.info("✅ [Vision] 已关闭")
            
        except Exception as e:
            logger.error(f"❌ [Vision] 关闭失败：{e}", exc_info=True)
    
    def on_event(self, event: ZulongEvent, shared_memory: Dict):
        """事件回调"""
        try:
            if event.type == EventType.SENSOR_VISION_REQUEST:
                # 接收视觉请求
                query = event.payload.get("query", "描述场景")
                shared_memory["vision.request"] = True
                shared_memory["vision.query"] = query
                logger.debug(f"📥 [Vision] 收到请求：{query}")
        
        except Exception as e:
            logger.error(f"❌ [Vision] 事件处理失败：{e}", exc_info=True)


# ========== 工厂函数 ==========

def create_plugin(config: Dict = None) -> IL1Module:
    """工厂函数：创建插件实例"""
    return L1C_VisionPlugin(config=config)
