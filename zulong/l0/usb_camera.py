# File: zulong/l0/usb_camera.py
"""
USB 摄像头捕获模块

功能:
- 支持多摄像头自动检测
- 硬件参数配置 (分辨率/帧率/曝光等)
- 低延迟帧捕获 (异步线程)
- 帧同步时间戳

TSD v1.7 对应:
- 4.4 感知预处理
- 7.2 集成测试场景
"""

import cv2
import time
import threading
import numpy as np
from typing import Optional, List, Dict, Any, Callable
from collections import deque
from datetime import datetime
import logging

logger = logging.getLogger("USBCamera")


class USBCamera:
    """
    USB 摄像头捕获类
    
    核心特性:
    1. 异步捕获线程 (避免阻塞主线程)
    2. 帧缓冲区 (防止丢帧)
    3. 硬件参数优化
    4. 自动重连机制
    """
    
    def __init__(
        self, 
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        buffer_size: int = 5
    ):
        """
        初始化摄像头
        
        Args:
            device_id: 摄像头设备 ID (默认 0)
            width: 帧宽度
            height: 帧高度
            fps: 目标帧率
            buffer_size: 帧缓冲区大小
        """
        logger.info(f"📷 [USBCamera] 初始化摄像头 (device={device_id}, {width}x{height}@{fps}fps)")
        
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        
        # OpenCV 捕获器
        self.cap: Optional[cv2.VideoCapture] = None
        
        # 帧缓冲区
        self.frame_buffer = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        
        # 运行状态
        self.is_running = False
        self.is_connected = False
        self.capture_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'dropped_frames': 0,
            'fps_actual': 0.0,
            'last_frame_time': 0.0,
        }
        
        # 回调函数
        self.on_frame_callback: Optional[Callable[[np.ndarray, float], None]] = None
        
        logger.info(f"📷 [USBCamera] 实例创建完成")
    
    def detect_cameras(self) -> List[int]:
        """
        检测可用的 USB 摄像头
        
        Returns:
            摄像头设备 ID 列表
        """
        logger.info("🔍 [USBCamera] 检测可用摄像头...")
        
        available_cameras = []
        
        # 尝试前 10 个设备 ID
        for device_id in range(10):
            cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)  # Windows 使用 DirectShow
            
            if cap.isOpened():
                logger.info(f"✅ [USBCamera] 检测到摄像头：device_id={device_id}")
                available_cameras.append(device_id)
                cap.release()
            else:
                # 尝试失败，跳过
                pass
        
        if not available_cameras:
            logger.warning("⚠️ [USBCamera] 未检测到任何摄像头")
        else:
            logger.info(f"✅ [USBCamera] 共检测到 {len(available_cameras)} 个摄像头：{available_cameras}")
        
        return available_cameras
    
    def connect(self) -> bool:
        """
        连接摄像头
        
        Returns:
            连接是否成功
        """
        logger.info(f"🔌 [USBCamera] 尝试连接摄像头 (device={self.device_id})...")
        
        try:
            # 创建捕获器 (使用 DirectShow 后端)
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                logger.error(f"❌ [USBCamera] 无法打开摄像头 {self.device_id}")
                return False
            
            # 配置硬件参数
            self._configure_hardware()
            
            self.is_connected = True
            logger.info(f"✅ [USBCamera] 摄像头连接成功")
            logger.info(f"   - 分辨率：{self.width}x{self.height}")
            logger.info(f"   - 帧率：{self.fps} fps")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ [USBCamera] 连接失败：{e}")
            return False
    
    def _configure_hardware(self):
        """配置摄像头硬件参数"""
        logger.info("⚙️ [USBCamera] 配置硬件参数...")
        
        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # 设置帧率
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # 获取实际参数
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"   - 实际分辨率：{actual_width}x{actual_height}")
        logger.info(f"   - 实际帧率：{actual_fps} fps")
        
        # 优化参数 (根据摄像头驱动调整)
        # 自动曝光
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1=手动，3=自动
        
        # 曝光值 (手动模式)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # 值越小越亮
        
        # 增益
        self.cap.set(cv2.CAP_PROP_GAIN, 0)
        
        # 白平衡 (自动)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        
        logger.info("✅ [USBCamera] 硬件配置完成")
    
    def start(self):
        """启动捕获线程"""
        if self.is_running:
            logger.warning("⚠️ [USBCamera] 已经在运行中")
            return
        
        if not self.is_connected:
            logger.error("❌ [USBCamera] 未连接，无法启动")
            return
        
        logger.info("🚀 [USBCamera] 启动捕获线程...")
        
        self.is_running = True
        self.stop_event.clear()
        
        # 创建并启动捕获线程
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            name="USBCamera_Capture",
            daemon=True
        )
        self.capture_thread.start()
        
        logger.info("✅ [USBCamera] 捕获线程已启动")
    
    def stop(self):
        """停止捕获线程"""
        if not self.is_running:
            return
        
        logger.info("🛑 [USBCamera] 停止捕获线程...")
        
        self.is_running = False
        self.stop_event.set()
        
        # 等待线程结束
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        logger.info("✅ [USBCamera] 捕获线程已停止")
    
    def disconnect(self):
        """断开摄像头连接"""
        self.stop()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.is_connected = False
        logger.info("🔌 [USBCamera] 已断开连接")
    
    def _capture_loop(self):
        """
        捕获线程主循环
        
        核心逻辑:
        1. 连续读取帧
        2. 添加时间戳
        3. 写入缓冲区
        4. 触发回调
        """
        logger.debug("🔄 [USBCamera] 捕获循环开始")
        
        frame_count = 0
        start_time = time.time()
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # 读取帧
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    logger.warning("⚠️ [USBCamera] 读取帧失败")
                    self.stats['dropped_frames'] += 1
                    time.sleep(0.01)  # 短暂等待
                    continue
                
                # 添加时间戳
                timestamp = time.time()
                
                # 写入缓冲区
                self.frame_buffer.append(frame.copy())
                self.timestamps.append(timestamp)
                
                # 更新统计
                frame_count += 1
                self.stats['total_frames'] += 1
                self.stats['last_frame_time'] = timestamp
                
                # 计算实际 FPS (每秒更新一次)
                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    self.stats['fps_actual'] = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()
                
                # 触发回调
                if self.on_frame_callback:
                    self.on_frame_callback(frame, timestamp)
                
            except Exception as e:
                logger.error(f"❌ [USBCamera] 捕获循环错误：{e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        
        logger.debug("🏁 [USBCamera] 捕获循环结束")
    
    def get_latest_frame(self) -> Optional[tuple]:
        """
        获取最新帧
        
        Returns:
            (frame, timestamp) 或 (None, 0.0) 如果缓冲区为空
        """
        if not self.frame_buffer:
            return None, 0.0
        
        # 从缓冲区尾部获取最新帧
        frame = self.frame_buffer[-1]
        timestamp = self.timestamps[-1]
        
        return frame, timestamp
    
    def get_frame_count(self) -> int:
        """获取已捕获帧数"""
        return self.stats['total_frames']
    
    def get_dropped_frame_count(self) -> int:
        """获取丢帧数"""
        return self.stats['dropped_frames']
    
    def get_actual_fps(self) -> float:
        """获取实际帧率"""
        return self.stats['fps_actual']
    
    def set_on_frame_callback(self, callback: Callable[[np.ndarray, float], None]):
        """
        设置帧回调函数
        
        Args:
            callback: 回调函数 (frame, timestamp) -> None
        """
        self.on_frame_callback = callback
        logger.info(f"✅ [USBCamera] 帧回调已设置")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'is_running': self.is_running,
            'is_connected': self.is_connected,
            'total_frames': self.stats['total_frames'],
            'dropped_frames': self.stats['dropped_frames'],
            'fps_actual': self.stats['fps_actual'],
            'buffer_size': len(self.frame_buffer),
        }


# 全局摄像头实例
_camera_instance: Optional[USBCamera] = None


def get_camera(
    device_id: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30
) -> USBCamera:
    """
    获取或创建摄像头单例
    
    Args:
        device_id: 摄像头设备 ID
        width: 帧宽度
        height: 帧高度
        fps: 帧率
    
    Returns:
        USBCamera 实例
    """
    global _camera_instance
    
    if _camera_instance is None:
        _camera_instance = USBCamera(device_id, width, height, fps)
        _camera_instance.connect()
    
    return _camera_instance


def detect_available_cameras() -> List[int]:
    """
    检测可用摄像头
    
    Returns:
        摄像头设备 ID 列表
    """
    temp_camera = USBCamera()
    return temp_camera.detect_cameras()
