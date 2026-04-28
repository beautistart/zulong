#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
祖龙 (ZULONG) 系统 - 摄像头驱动与视频采集模块

文件：zulong/l0/devices/camera_device.py

功能:
- 摄像头设备管理（初始化、启动、停止）
- 视频流采集（30fps, 640x480）
- 人脸检测
- 发布 SENSOR_VIDEO_FRAME 事件

TSD v1.7 对应:
- 4.4 感知预处理 - 视觉感知
- 2.2.2 L1-A - SENSOR_* 事件处理
"""

import asyncio
import logging
import threading
from typing import Optional, Callable, List, Dict, Any
import cv2
import numpy as np
from datetime import datetime

from zulong.core.types import ZulongEvent, EventType, EventPriority
from zulong.core.event_bus import event_bus

logger = logging.getLogger(__name__)


class CameraDevice:
    """
    摄像头设备驱动类
    
    功能:
    - 封装 OpenCV，调用 Windows 摄像头驱动
    - 实现画面变动检测（光流法）
    - 发布视频事件到 EventBus
    
    硬件要求:
    - Windows 系统已安装摄像头驱动
    - 摄像头设备可用
    
    使用示例:
    ```python
    camera = CameraDevice()
    await camera.start()
    # 自动采集视频并发布事件
    await camera.stop()
    ```
    """
    
    # 视频参数配置
    FRAME_WIDTH = 640        # 帧宽度
    FRAME_HEIGHT = 480       # 帧高度
    FPS = 30                 # 帧率
    FRAME_BUFFER_SIZE = 30   # 保留最近 30 帧
    
    # 画面变动检测阈值
    MOTION_THRESHOLD = 10000   # 运动像素数量阈值
    LIGHT_THRESHOLD = 0.1      # 光线变化阈值
    
    def __init__(self, device_index: int = 1):
        """
        初始化摄像头设备
        
        Args:
            device_index: 摄像头设备索引，1 表示 USB Video 摄像头 (默认)
        """
        self.device_index = device_index
        self.camera: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.is_recording = False
        
        # 🎯 线程安全：帧缓冲区锁
        self.frame_lock = threading.Lock()
        
        # 帧缓冲区
        self.frame_buffer: List[np.ndarray] = []
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None
        
        # 状态跟踪
        self.last_motion_time = 0.0
        self.motion_count = 0
        self.user_detected = False
        
        # 🎯 关键修改：自动曝光和亮度补偿参数
        self.auto_exposure = False           # 禁用软件自动曝光 (Windows 驱动不支持)
        self.brightness_offset = 0           # 不再使用固定亮度补偿 (让硬件自动曝光)
        self.target_brightness = 80          # 目标平均亮度
        
        # 回调函数
        self._on_motion_detected: Optional[Callable] = None
        self._on_user_detected: Optional[Callable] = None
        
        # EventBus（单例）
        from zulong.core.event_bus import event_bus
        self.event_bus = event_bus
        
        logger.info("📷 摄像头设备初始化完成")
        logger.info(f"   - 光流法检测器：已启用")
        logger.info(f"   - 线程锁：已启用 (frame_lock)")
    
    def list_cameras(self) -> list:
        """
        列出所有可用的摄像头设备
        
        Returns:
            list: 设备信息列表
        """
        cameras = []
        
        # 尝试打开 0-9 号摄像头
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Windows DirectShow
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        cameras.append({
                            'index': i,
                            'name': f'Camera {i}',
                            'width': width,
                            'height': height,
                            'fps': fps
                        })
                cap.release()
            except Exception as e:
                logger.warning(f"检测摄像头 {i} 失败：{e}")
        
        return cameras
    
    async def initialize(self) -> bool:
        """
        初始化 OpenCV 和摄像头设备
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 打开摄像头（尝试多种后端）
            backends = [
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "Media Foundation"),
                (cv2.CAP_ANY, "Auto"),
            ]
            
            for backend, name in backends:
                self.camera = cv2.VideoCapture(self.device_index, backend)
                if self.camera.isOpened():
                    logger.info(f"✅ 摄像头已打开 (后端: {name}, 索引: {self.device_index})")
                    break
                else:
                    logger.warning(f"⚠️ 后端 {name} 无法打开摄像头 {self.device_index}")
            
            if not self.camera or not self.camera.isOpened():
                logger.error(f"❌ 无法打开摄像头 {self.device_index}")
                return False
            
            # 🎯 关键修改：摄像头预热序列 (等待自动曝光调整)
            logger.info("📷 摄像头传感器预热中 (等待自动曝光调整)...")
            import time
            
            warmup_frames = 0
            max_warmup = 60  # 最多尝试 60 帧
            brightness = 0.0
            
            while warmup_frames < max_warmup:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    warmup_frames += 1
                    time.sleep(0.03)
                    continue
                
                # 计算亮度 (灰度图的平均值)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                
                logger.debug(f"   - 帧 {warmup_frames}: 亮度 = {brightness:.2f}")
                
                # 如果亮度合理 (>20),提前退出
                if brightness > 20:
                    logger.info(f"✅ 摄像头已就绪，预热 {warmup_frames} 帧 (亮度={brightness:.2f})")
                    break
                
                warmup_frames += 1
                time.sleep(0.03)  # 小延迟让硬件调整
            
            if brightness < 20:
                logger.warning(f"⚠️ 预热完成但亮度仍然过低 ({brightness:.2f}),将使用软件亮度补偿")
            
            # 设置参数
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FPS, self.FPS)
            
            # 🎯 关键修改：启用自动曝光和亮度补偿
            if self.auto_exposure:
                # Windows 摄像头自动曝光控制
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 启用自动曝光模式
                self.camera.set(cv2.CAP_PROP_EXPOSURE, -6)      # 曝光值（负值减少曝光，正值增加）
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 128)   # 亮度（0-255）
                self.camera.set(cv2.CAP_PROP_CONTRAST, 128)     # 对比度（0-255）
                self.camera.set(cv2.CAP_PROP_SATURATION, 128)   # 饱和度（0-255）
                logger.info("📷 已启用自动曝光控制")
            
            # 获取实际参数
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            # 检查实际曝光设置
            if self.auto_exposure:
                actual_exposure = self.camera.get(cv2.CAP_PROP_EXPOSURE)
                actual_brightness = self.camera.get(cv2.CAP_PROP_BRIGHTNESS)
                logger.info(f"📷 曝光设置：exposure={actual_exposure}, brightness={actual_brightness}")
            
            logger.info(f"📷 摄像头设备：{self.device_index}")
            logger.info(f"   - 分辨率：{actual_width}x{actual_height}")
            logger.info(f"   - 帧率：{actual_fps} fps")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 摄像头初始化失败：{e}")
            return False
    
    async def start(self) -> bool:
        """
        启动视频流采集
        
        Returns:
            bool: 启动是否成功
        """
        if not self.camera:
            if not await self.initialize():
                return False
        
        self.is_running = True
        self.is_recording = True
        
        logger.info("📷 摄像头已启动，开始采集视频...")
        
        # 启动采集任务
        asyncio.create_task(self._capture_loop())
        
        return True
    
    async def stop(self):
        """停止视频流采集"""
        self.is_running = False
        self.is_recording = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        logger.info("📷 摄像头已停止")
    
    async def _capture_loop(self):
        """视频采集循环 (TSD v1.8: 主动推送帧到 VisionProcessor)"""
        frame_count = 0
        
        # 🎯 关键修复 1: 启动后立即设置基准帧 (等待摄像头预热)
        await asyncio.sleep(0.5)
        ret, warmup_frame = self.camera.read()
        if ret and warmup_frame is not None:
            # 直接调用检测器设置基准
            try:
                from zulong.l0.motion_detector import get_detector
                detector = get_detector()
                detector.set_baseline(warmup_frame)
                logger.info("📷 [Camera] 预热完成，基准帧已强制设置")
                
                # 同时推送给 VisionProcessor (如果已初始化)
                from zulong.l1c.optimized_vision_processor import get_vision_processor
                vp = get_vision_processor()
                if vp and vp.is_running:
                    vp.feed_frame(warmup_frame, datetime.now().timestamp())
            except Exception as e:
                logger.debug(f"设置基准帧失败：{e}")
        
        while self.is_running:
            try:
                # 读取帧
                ret, frame = self.camera.read()
                
                if not ret or frame is None:
                    logger.warning("⚠️ 无法读取视频帧")
                    await asyncio.sleep(0.1)
                    continue
                
                # 软件亮度补偿 (仅在确实需要时应用)
                original_brightness = np.mean(frame)
                if original_brightness < 50:
                    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
                    new_brightness = np.mean(frame)
                    logger.debug(f"📷 亮度补偿：{original_brightness:.2f} -> {new_brightness:.2f}")
                
                frame_count += 1
                current_timestamp = datetime.now().timestamp()
                
                # 🎯 关键修复 2: 主动推送帧到 VisionProcessor (TSD v1.8: 主动推送模式)
                try:
                    from zulong.l1c.optimized_vision_processor import get_vision_processor
                    vp = get_vision_processor()
                    logger.debug(f"📷 [Camera] VP 检查：vp={vp}, is_running={vp.is_running if vp else 'N/A'}")
                    if vp and vp.is_running:
                        logger.debug(f"📷 [Camera] 推送帧到 VP: is_running={vp.is_running}")
                        vp.feed_frame(frame, current_timestamp)
                    else:
                        logger.warning(f"⚠️ [Camera] VP 未运行：vp={vp}, is_running={vp.is_running if vp else 'N/A'}")
                        # 如果 VP 未运行，仅缓存 (兼容旧模式)
                        with self.frame_lock:
                            self.frame_buffer.append(frame)
                            if len(self.frame_buffer) > self.FRAME_BUFFER_SIZE:
                                self.frame_buffer.pop(0)
                except Exception as e:
                    logger.debug(f"推送帧失败：{e}")
                
                # 使用锁保护帧缓冲区 (线程安全，兼容旧模式)
                with self.frame_lock:
                    # 已删除旧版 VisionProcessor，不再需要更新帧缓存
                    pass
                
                # 发布视频帧事件 (轻量级，不包含帧数据)
                event = ZulongEvent(
                    type=EventType.SENSOR_VIDEO_FRAME,
                    priority=EventPriority.LOW,  # 降低优先级，因为主要逻辑在 VP 内部
                    source=f"camera_{self.device_index}",
                    payload={
                        "frame_index": frame_count,
                        "timestamp": current_timestamp,
                        "note": "Frame pushed to VisionProcessor directly"
                    }
                )
                self.event_bus.publish(event)
                
                # 控制帧率 (TSD v1.8: 固定 30FPS 以保证流畅性)
                await asyncio.sleep(1.0 / self.FPS)
                
            except Exception as e:
                logger.error(f"❌ 视频采集错误：{e}")
                await asyncio.sleep(0.1)
    
    async def capture_frame(self) -> Optional[np.ndarray]:
        """
        捕获单帧图像
        
        Returns:
            np.ndarray: 图像数据，失败返回 None
        """
        if not self.camera:
            return None
        
        ret, frame = self.camera.read()
        
        if ret:
            return frame
        else:
            return None
    
    def get_frame_buffer(self) -> List[np.ndarray]:
        """
        获取帧缓冲区
        
        Returns:
            List[np.ndarray]: 最近 30 帧的列表
        """
        return self.frame_buffer.copy()
    
    def set_motion_callback(self, callback: Callable):
        """
        设置运动检测回调
        
        Args:
            callback: 回调函数
        """
        self._on_motion_detected = callback
    
    def set_user_detect_callback(self, callback: Callable):
        """
        设置用户检测回调
        
        Args:
            callback: 回调函数
        """
        self._on_user_detected = callback
