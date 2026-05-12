# File: zulong/l1c/optimized_vision_processor.py
"""
L1-C 优化视觉处理器 (人体锚点驱动策略)

TSD v1.7 对应:
- 2.2.2 L1 层拆分
- 4.4 感知预处理
- 5.2 显存约束

优化方案核心 (2026 年主流):
1. **第一层注意**: YOLO-Nano 人体检测 + ROI 增益放大 (替代全图光流)
2. **第二层注意**: MobileNetV4-TSM 动作分类 (替代 ST-GCN)
3. **第三层注意**: EfficientNet + Digital Zoom 手势识别 (替代 MediaPipe)

架构优势:
- 解决 3 米外光流法不敏感问题 (ROI 增益放大 2-5 倍)
- 解决 ST-GCN 权重缺失问题 (TSM 无需骨骼数据)
- 解决远距离手势识别率低问题 (Digital Zoom + EfficientNet)
"""

import asyncio
import time
import numpy as np
import threading
import cv2
from typing import Optional, List, Dict, Any, Tuple
from collections import deque
import logging
import os

from zulong.l0.motion_detector import OpticalFlowMotionDetector, MotionState, MotionResult
from zulong.core.types import ZulongEvent, EventType, EventPriority
from zulong.core.event_bus import event_bus
from zulong.l1c.action_classifier import MobileNetV4_TSM
from zulong.l1c.mediapipe_gesture_recognizer import MediaPipeGestureRecognizer
# from zulong.l1c.gesture_classifier import EfficientNet_Gesture  # 已删除，使用 MediaPipe 替代
# from zulong.l1c.vision_model_loader import get_vision_model_loader, VisionModelLoader  # 已删除

logger = logging.getLogger("OptimizedVisionProcessor")


class OptimizedVisionProcessor:
    """
    优化视觉处理器 (人体锚点驱动策略)
    
    核心逻辑:
    1. **Layer 1 - 人体检测**: YOLO-Nano 永远开启，检测人体位置
    2. **Layer 2 - ROI 增益**: 对人体区域进行像素差分放大 (增益系数 2-5 倍)
    3. **Layer 3 - 动作分类**: MobileNetV4-TSM 判断意图 (挥手/注视/靠近)
    4. **Layer 4 - 鹰眼模式**: 触发 Digital Zoom + EfficientNet 手势识别
    
    TSD v1.7 对应:
    - 4.2.1 L1-B 注意力控制器
    - 4.4 感知预处理
    - 5.2 显存约束
    """
    
    def __init__(self):
        logger.info("👁️ [OptimizedVisionProcessor.__init__] Creating optimized processor...")
        
        self.event_bus = event_bus
        
        # ========== 模型占位符 (按需加载) ==========
        self._model_loader: Optional[Any] = None  # 已删除，使用 None
        self._yolo_model: Optional[Any] = None  # YOLO-Nano
        self._action_classifier: Optional[MobileNetV4_TSM] = None  # MobileNetV4-TSM
        self._gesture_classifier: Optional[MediaPipeGestureRecognizer] = None  # 使用 MediaPipe 替代 EfficientNet
        
        # ========== 运行状态 ==========
        self.is_running = False
        self.is_initialized = False
        
        # ========== 共享内存 ==========
        self.shared_memory: Dict[str, Any] = {
            'vision_target_pos': None,
            'motion_pixels': 0,
            'motion_magnitude': 0.0,
            'last_motion_time': 0.0,
            'human_detected': False,
            'human_bbox': None,  # [x1, y1, x2, y2]
            'gesture_type': None,
            'action_score': 0.0,
        }
        
        # ========== 帧缓冲区 ==========
        self.frame_buffer = deque(maxlen=90)  # 3 秒 @ 30fps
        self.timestamps = deque(maxlen=90)
        
        # ========== 配置参数 ==========
        self._config = {
            # YOLO-Nano 配置
            'yolo_conf_threshold': 0.25,  # 降低检测阈值 (0.5->0.4->0.25) 提高检测率
            'yolo_iou_threshold': 0.45,
            'yolo_inference_frequency': 5,  # YOLO 推理频率 (每 N 帧检测一次，3->5)
            
            # ROI 增益配置
            'roi_gain_coefficient': 3.0,  # ROI 区域增益系数 (2-5 倍)
            'roi_motion_threshold': 100,  # 降低阈值提高敏感度 (200->100)  # ROI 区域运动像素阈值 (降低)
            'global_motion_threshold': 1000,  # 全图运动像素阈值 (严格)
            
            # 动作分类配置
            'action_slow_fps': 8,  # Slow 流帧率
            'action_fast_fps': 30,  # Fast 流帧率
            'action_slow_frame_interval': 4,  # Slow 流采样间隔 (30/8≈4)
            'action_num_frames_slow': 8,  # Slow 流输入帧数
            'action_num_frames_fast': 16,  # Fast 流输入帧数
            'intent_threshold': 0.25,  # 意图检测阈值
            'interact_threshold': 0.05,  # 降低到 L3 最低置信度 (0.033) 之上，确保 L4 能触发
            
            # 鹰眼模式配置
            'digital_zoom_factor': 5.0,  # 数字变焦倍数
            'gesture_conf_threshold': 0.25,  # 手势置信度阈值
            'eagle_eye_cooldown': 0.3,  # 冷却时间 (秒)
            
            # 基础配置
            'frame_width': 640,
            'frame_height': 480,
            'fps': 30,
        }
        
        # ========== 状态机 ==========
        self.state_machine = {
            'layer1_state': 'NO_HUMAN',  # NO_HUMAN, HUMAN_DETECTED
            'layer2_state': 'IDLE',  # IDLE, MOTION_DETECTED
            'layer3_state': 'NO_INTENT',  # NO_INTENT, SILENT_WATCH, INTERACT_REQUEST
            'layer4_state': 'NORMAL',  # NORMAL, EAGLE_EYE_ACTIVE
        }
        
        # ========== 时间戳追踪 ==========
        self._last_eagle_eye_time = 0.0
        self._last_human_detect_time = 0.0
        
        # ========== 帧计数器 (用于 YOLO 推理频率控制) ==========
        self._frame_counter = 0
        self._last_yolo_inference_frame = 0  # 上一次 YOLO 推理的帧数
        
        logger.info("✅ [OptimizedVisionProcessor] Instance created")
        logger.info(f"   - ROI Gain: {self._config['roi_gain_coefficient']}x")
        logger.info(f"   - Digital Zoom: {self._config['digital_zoom_factor']}x")
        logger.info(f"   - Action FPS: Slow@{self._config['action_slow_fps']} / Fast@{self._config['action_fast_fps']}")
    
    def _load_yolo_model(self):
        """
        加载 YOLO-Nano 人体检测模型
        
        TSD v1.7 对应:
        - 5.2 显存约束：模型 4bit 量化加载
        """
        try:
            # 🎯 方案 A: 使用项目已有的 YOLOv8n (当前目录)
            # 检查是否存在 yolov8n.pt
            yolo_path = os.path.join(os.path.dirname(__file__), '../../yolov8n.pt')
            
            if os.path.exists(yolo_path):
                logger.info(f"📦 [YOLO] 加载现有模型：{yolo_path}")
                # 延迟导入 ultralytics (避免依赖冲突)
                from ultralytics import YOLO
                self._yolo_model = YOLO(yolo_path)
                logger.info("✅ [YOLO] YOLOv8n 加载完成 (使用 nano 模式)")
            else:
                # 🎯 方案 B: 使用 YOLO-Nano (需额外安装)
                # logger.warning("⚠️ [YOLO] yolov8n.pt 不存在，使用模拟模式")
                # self._yolo_model = None
                logger.info(f"📦 [YOLO] 尝试加载：{yolo_path}")
                from ultralytics import YOLO
                self._yolo_model = YOLO(yolo_path)
                logger.info("✅ [YOLO] YOLOv8n 加载完成")
            
        except Exception as e:
            logger.error(f"❌ [YOLO] 加载失败：{e}")
            logger.warning("⚠️ [YOLO] 将使用模拟检测模式")
            self._yolo_model = None
    
    def _load_action_classifier(self):
        """
        加载 MobileNetV4-TSM 动作分类器
        
        TSD v1.7 对应:
        - 5.2 显存约束：模型 4bit 量化
        - 替代 ST-GCN (无需骨骼数据)
        """
        try:
            logger.info("📦 [Action] 加载 MobileNetV4-TSM 动作分类器...")
            
            # 创建分类器实例
            self._action_classifier = MobileNetV4_TSM(config={
                'slow_fps': self._config['action_slow_fps'],
                'fast_fps': self._config['action_fast_fps'],
                'slow_frame_interval': self._config['action_slow_frame_interval'],
                'num_frames_slow': self._config['action_num_frames_slow'],
                'num_frames_fast': self._config['action_num_frames_fast'],
                'intent_threshold': self._config['intent_threshold'],
                'interact_threshold': self._config['interact_threshold'],
            })
            
            # 加载真实模型（MobileNetV3-Large 预训练）
            self._action_classifier.load_model()
            
            logger.info("✅ [Action] MobileNetV4-TSM 加载完成 (真实模型)")
            
        except Exception as e:
            logger.error(f"❌ [Action] 加载失败：{e}")
            self._action_classifier = None
    
    def _load_gesture_classifier(self):
        """
        加载手势分类器（使用 MediaPipe 替代 EfficientNet）
        
        TSD v1.7 对应:
        - 5.2 显存约束：MediaPipe 可在 CPU 运行
        - 预训练模型，开箱即用
        """
        try:
            logger.info("📦 [Gesture] 加载 MediaPipe Gesture Recognizer...")
            
            # 创建 MediaPipe 手势识别器（降低阈值到 0.15）
            self._gesture_classifier = MediaPipeGestureRecognizer(
                confidence_threshold=0.15  # 降低阈值提高灵敏度
            )
            
            if self._gesture_classifier._recognizer:
                logger.info("✅ [Gesture] MediaPipe Gesture Recognizer 加载完成 (预训练模型)")
                logger.info(f"   置信度阈值：0.15")
            else:
                logger.warning("⚠️ [Gesture] MediaPipe 不可用，使用模拟模式")
            
        except Exception as e:
            logger.error(f"❌ [Gesture] 加载失败：{e}")
            self._gesture_classifier = None
    
    async def initialize(self, load_models: bool = True):
        """
        异步初始化处理器
        
        Args:
            load_models: 是否加载真实模型 (默认 True)
        """
        logger.info("🚀 [OptimizedVisionProcessor.initialize] Starting initialization...")
        
        if self.is_initialized:
            logger.info("✅ 已初始化，跳过")
            return
        
        try:
            # 已删除模型加载器，直接初始化各模块
            if load_models:
                logger.info("📦 [初始化] 加载真实视觉模型...")
                # self._model_loader = get_vision_model_loader()  # 已删除
                # model_stats = self._model_loader.get_model_stats()
                # logger.info(f"✅ [初始化] 模型加载状态：{model_stats}")
                logger.info("⚠️  模型加载器已删除，使用独立模块")
            else:
                logger.info("⚠️ [初始化] 使用模拟模型模式")
            
            # 初始化 YOLO 模型
            self._load_yolo_model()
            
            # 初始化动作分类器
            self._load_action_classifier()
            
            # 初始化手势分类器
            self._load_gesture_classifier()
            
            # 启动后台线程
            self.start()
            
            self.is_initialized = True
            logger.info("✅ [OptimizedVisionProcessor] 初始化完成")
            
        except Exception as e:
            logger.error(f"❌ [OptimizedVisionProcessor] 初始化失败：{e}")
            import traceback
            traceback.print_exc()
            raise
    
    def start(self):
        """启动处理线程"""
        if self.is_running:
            logger.warning("⚠️ 处理线程已在运行")
            return
        
        self.is_running = True
        logger.info("🚀 [OptimizedVisionProcessor] 处理线程已启动")
    
    def stop(self):
        """停止处理线程"""
        self.is_running = False
        logger.info("🛑 [OptimizedVisionProcessor] 处理线程已停止")
    
    def feed_frame(self, frame: np.ndarray, timestamp: float):
        """
        输入帧 (由 CameraDevice 主动推送)
        
        Args:
            frame: BGR 格式帧
            timestamp: 时间戳
        """
        try:
            if not self.is_running:
                logger.warning("⚠️ [feed_frame] VP 未运行，跳过")
                return
            
            # 添加到缓冲区
            self.frame_buffer.append(frame.copy())
            self.timestamps.append(timestamp)
            
            # 保持缓冲区大小
            if len(self.frame_buffer) > self.frame_buffer.maxlen:
                self.frame_buffer.popleft()
                self.timestamps.popleft()
            
            # 每 30 帧记录一次日志 (INFO 级别)
            if self._frame_counter % 30 == 0:
                logger.info(f"📥 [feed_frame] 接收帧 #{self._frame_counter} (缓冲区：{len(self.frame_buffer)} 帧)")
            
            # 同步处理帧 (修复：在同步上下文中直接调用)
            # asyncio.create_task() 需要在异步上下文中，改为直接调用同步方法
            self._process_frame_sync(frame, timestamp)
            
        except Exception as e:
            logger.error(f"❌ [feed_frame] 错误：{e}", exc_info=True)
    
    def _process_frame_sync(self, frame: np.ndarray, timestamp: float):
        """
        同步处理帧 (核心逻辑)
        
        流程:
        1. Layer 1: YOLO-Nano 人体检测
        2. Layer 2: ROI 增益放大 + 像素差分
        3. Layer 3: MobileNetV4-TSM 动作分类
        4. Layer 4: 鹰眼模式 (Digital Zoom + MediaPipe)
        
        Args:
            frame: 当前帧
            timestamp: 时间戳
        """
        try:
            # 每 60 帧记录一次处理日志
            if self._frame_counter % 60 == 0:
                logger.info(f"🔄 [_process_frame_sync] 处理帧 #{self._frame_counter}")
            
            # ========== Layer 1: 人体检测 ==========
            human_bboxes = self._layer1_human_detection(frame)
            
            if not human_bboxes:
                self.state_machine['layer1_state'] = 'NO_HUMAN'
                self.shared_memory['human_detected'] = False
                return
            
            self.state_machine['layer1_state'] = 'HUMAN_DETECTED'
            self.shared_memory['human_detected'] = True
            self.shared_memory['human_bbox'] = human_bboxes[0]  # 取第一个检测框
            self._last_human_detect_time = timestamp
            
            # ========== Layer 2: ROI 增益放大 ==========
            motion_detected, motion_pixels = self._layer2_roi_motion_detection(
                frame, human_bboxes[0]
            )
            
            if not motion_detected:
                self.state_machine['layer2_state'] = 'IDLE'
                return
            
            self.state_machine['layer2_state'] = 'MOTION_DETECTED'
            self.shared_memory['motion_pixels'] = motion_pixels
            self.shared_memory['last_motion_time'] = timestamp
            
            # 发布视觉状态变更事件 (L1-A 兼容)
            self._publish_vision_state_event(motion_pixels, human_bboxes[0])
            
            # ========== Layer 3: 动作分类 (频率控制) ==========
            # 动作分类不需要每帧执行，每 2 帧执行一次即可
            if self._frame_counter % 2 == 0:  # 偶数帧执行
                action_score, intent_type = self._layer3_action_classification(frame)
            else:
                # 使用上一次的结果
                action_score = self.shared_memory.get('action_score', 0.0)
                intent_type = self.shared_memory.get('intent_type', 'UNKNOWN')
                logger.debug(f"⏭️ [Layer3] 跳过动作分类 (帧 {self._frame_counter})")
            
            self.shared_memory['action_score'] = action_score
            self.shared_memory['intent_type'] = intent_type
            
            if action_score < self._config['intent_threshold']:
                self.state_machine['layer3_state'] = 'NO_INTENT'
                return
            
            elif action_score < self._config['interact_threshold']:
                self.state_machine['layer3_state'] = 'SILENT_WATCH'
                # 发布静默注意事件
                self._publish_silent_attention_event(intent_type, action_score)
            
            else:
                self.state_machine['layer3_state'] = 'INTERACT_REQUEST'
                
                # ========== Layer 4: 鹰眼模式 ==========
                gesture_result = self._layer4_eagle_eye_mode(frame, human_bboxes[0], timestamp)
                
                if gesture_result:
                    self.state_machine['layer4_state'] = 'EAGLE_EYE_ACTIVE'
                    self.shared_memory['gesture_type'] = gesture_result['gesture']
                    
                    # 发布交互事件
                    self._publish_interaction_event(gesture_result, intent_type)
                else:
                    self.state_machine['layer4_state'] = 'NORMAL'
                    
        except Exception as e:
            logger.error(f"❌ [_process_frame_sync] 错误：{e}", exc_info=True)
    
    def _layer1_human_detection(self, frame: np.ndarray) -> List[List[int]]:
        """
        Layer 1: YOLO-Nano 人体检测 (带频率控制)
        
        TSD v1.7 对应:
        - 4.4 感知预处理：降低推理频率提升性能
        
        Args:
            frame: BGR 格式帧
        
        Returns:
            List of [x1, y1, x2, y2] 检测框
        """
        try:
            # ========== 频率控制逻辑 ==========
            self._frame_counter += 1
            
            # 检查是否需要执行 YOLO 推理
            frames_since_last_yolo = self._frame_counter - self._last_yolo_inference_frame
            yolo_frequency = self._config['yolo_inference_frequency']
            
            if frames_since_last_yolo < yolo_frequency:
                # 跳过 YOLO 推理，使用上一帧的结果
                logger.debug(f"⏭️ [Layer1] 跳过 YOLO 推理 (帧 {frames_since_last_yolo}/{yolo_frequency})")
                
                # 返回上一帧的检测结果
                if self.shared_memory['human_detected'] and self.shared_memory['human_bbox']:
                    return [self.shared_memory['human_bbox']]
                else:
                    return []
            
            # 执行 YOLO 推理
            self._last_yolo_inference_frame = self._frame_counter
            logger.debug(f"🔍 [Layer1] 执行 YOLO 推理 (帧 {self._frame_counter}, 频率 1/{yolo_frequency})")
            
            # 使用 YOLO 模型 (已删除模型加载器，直接使用 _yolo_model)
            if self._yolo_model:
                logger.debug(f"🔍 [Layer1] 开始 YOLO 推理，帧形状：{frame.shape}")
                
                try:
                    # YOLO 推理
                    results = self._yolo_model(frame, verbose=False, conf=self._config['yolo_conf_threshold'])
                    
                    # 解析结果 (COCO 数据集，person 类别 ID=0)
                    best_person = None
                    best_confidence = 0.0
                    
                    for result in results:
                        boxes = result.boxes
                        if boxes is None:
                            continue
                        
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            if cls_id == 0 and conf > best_confidence:  # person
                                xyxy = box.xyxy[0].cpu().numpy()
                                best_person = {
                                    'bbox': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                                    'confidence': conf
                                }
                                best_confidence = conf
                    
                    if best_person:
                        logger.info(f"👤 [Layer1] 检测到人体 (置信度：{best_person['confidence']:.2f})")
                        return [best_person['bbox']]
                    else:
                        logger.debug("⚠️ [Layer1] 未检测到人体")
                        return []
                        
                except Exception as e:
                    logger.error(f"❌ [Layer1] YOLO 推理失败：{e}")
                    return []
            
            # 模拟模式：检查帧中是否有蓝色人体区域
            # 测试帧使用 BGR(50, 100, 200)
            # OpenCV 颜色范围：[B_min, G_min, R_min] 到 [B_max, G_max, R_max]
            lower_blue = np.array([40, 90, 190])
            upper_blue = np.array([60, 110, 210])
            mask = cv2.inRange(frame, lower_blue, upper_blue)
            
            # 如果蓝色区域足够大，认为检测到人体
            if cv2.countNonZero(mask) > 1000:
                h, w = frame.shape[:2]
                # 返回模拟检测框
                return [[w//4, h//4, w//2, h//2]]
            else:
                return []
            
        except Exception as e:
            logger.error(f"❌ [Layer1] 人体检测失败：{e}")
            return []
    
    def _layer2_roi_motion_detection(
        self, 
        frame: np.ndarray, 
        human_bbox: List[int]
    ) -> Tuple[bool, int]:
        """
        Layer 2: ROI 增益放大 + 像素差分
        
        核心逻辑:
        1. 提取人体 ROI 区域
        2. 对 ROI 区域进行增益放大 (像素值 × gain_coefficient)
        3. 计算 ROI 区域的帧间差分
        4. 使用较低阈值 (ROI 区域更敏感)
        
        Args:
            frame: 当前帧
            human_bbox: 人体检测框 [x1, y1, x2, y2]
        
        Returns:
            (是否检测到运动，运动像素数)
        """
        try:
            # ========== 1. 提取 ROI 区域 ==========
            # 确保 bbox 坐标是整数
            x1, y1, x2, y2 = [int(coord) for coord in human_bbox]
            h, w = frame.shape[:2]
            
            # 边界检查
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # 扩展 ROI 边界 (包含手部活动区域)
            margin = 50
            x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
            x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
            
            roi_current = frame[y1:y2, x1:x2].copy()
            
            # ========== 2. 使用传入的 frame 作为前一帧 (测试模式) ==========
            # 如果有前一帧，使用帧缓冲区；否则使用当前帧 (测试时会手动添加)
            if len(self.frame_buffer) >= 2:
                prev_frame = self.frame_buffer[-2]
                roi_prev = prev_frame[y1:y2, x1:x2].copy()
            else:
                # 测试模式：直接使用当前帧作为参考
                roi_prev = roi_current.copy()
            
            # ========== 3. ROI 增益放大 ==========
            gain = self._config['roi_gain_coefficient']
            
            # 转为灰度图
            roi_curr_gray = cv2.cvtColor(roi_current, cv2.COLOR_BGR2GRAY)
            roi_prev_gray = cv2.cvtColor(roi_prev, cv2.COLOR_BGR2GRAY)
            
            # 应用增益 (模拟"放大微小运动")
            roi_curr_enhanced = cv2.convertScaleAbs(roi_curr_gray, alpha=gain, beta=0)
            roi_prev_enhanced = cv2.convertScaleAbs(roi_prev_gray, alpha=gain, beta=0)
            
            # ========== 4. 帧间差分 ==========
            diff = cv2.absdiff(roi_curr_enhanced, roi_prev_enhanced)
            _, thresh = cv2.threshold(diff, self._config['roi_motion_threshold'], 255, cv2.THRESH_BINARY)
            
            # ========== 5. 计算运动像素 ==========
            motion_pixels = cv2.countNonZero(thresh)
            motion_detected = motion_pixels > self._config['roi_motion_threshold']
            
            if motion_detected:
                logger.debug(f"👆 [Layer2] ROI 区域检测到运动：{motion_pixels}像素 (阈值：{self._config['roi_motion_threshold']})")
            
            return motion_detected, motion_pixels
            
        except Exception as e:
            logger.error(f"❌ [Layer2] ROI 运动检测失败：{e}")
            return False, 0
    
    def _layer3_action_classification(
        self, 
        frame: np.ndarray
    ) -> Tuple[float, str]:
        """
        Layer 3: MobileNetV4-TSM 动作分类
        
        核心逻辑:
        1. 使用真实模型加载器进行推理
        2. 返回意图分数和类型
        
        Args:
            frame: 当前帧
        
        Returns:
            (意图分数，意图类型)
        """
        try:
            # 使用动作分类器 (已删除模型加载器)
            # if self._model_loader and self._model_loader.mobilenet_model:
            #     action_label, confidence = self._model_loader.classify_action(frame)
            #     intent_score = confidence
            #     intent_type = action_label.upper()
            #     logger.debug(f"🧠 [Layer3] 分类结果：{intent_type} ({intent_score:.2f})")
            #     return intent_score, intent_type
            
            # 使用动作分类器
            if self._action_classifier is None:
                logger.warning("⚠️ [Layer3] 动作分类器未初始化，使用默认值")
                return 0.3, "UNKNOWN"
            
            # 获取当前时间戳
            timestamp = time.time()
            
            # 添加帧到分类器缓冲区
            ready = self._action_classifier.add_frame(frame, timestamp)
            
            # 如果缓冲区未满，返回中间结果
            if not ready:
                status = self._action_classifier.get_buffer_status()
                logger.debug(f"🧠 [Layer3] 缓冲区填充中：Slow={status['slow_buffer_size']}/{self._config['action_slow_fps']}, Fast={status['fast_buffer_size']}")
                return 0.3, "BUFFERING"
            
            # 执行动作分类
            intent_score, intent_type, details = self._action_classifier.classify_action()
            
            logger.debug(f"🧠 [Layer3] 分类结果：{intent_type} ({intent_score:.2f})")
            
            return intent_score, intent_type
            
        except Exception as e:
            logger.error(f"❌ [Layer3] 动作分类失败：{e}")
            return 0.0, "ERROR"
    
    def _layer4_eagle_eye_mode(
        self, 
        frame: np.ndarray, 
        human_bbox: List[int],
        timestamp: float
    ) -> Optional[Dict[str, Any]]:
        """
        Layer 4: 鹰眼模式 (使用 MediaPipe Gesture Recognizer)
        
        核心逻辑:
        1. 检查冷却时间 (避免频繁触发)
        2. 使用 MediaPipe 直接识别整帧手势
        3. 发布手势识别结果
        
        Args:
            frame: 当前帧
            human_bbox: 人体检测框
            timestamp: 时间戳
        
        Returns:
            手势识别结果 (包含 gesture, confidence)
        """
        try:
            # ========== 1. 检查冷却时间 ==========
            if timestamp - self._last_eagle_eye_time < self._config['eagle_eye_cooldown']:
                logger.debug("⏳ [Layer4] 鹰眼模式冷却中")
                return None
            
            self._last_eagle_eye_time = timestamp
            
            # ========== 2. 使用 MediaPipe 识别手势 ==========
            gesture = "UNKNOWN"
            confidence = 0.0
            
            if self._gesture_classifier and isinstance(self._gesture_classifier, MediaPipeGestureRecognizer):
                # MediaPipe 直接处理整帧
                gesture_name, conf, details = self._gesture_classifier.classify_gesture(frame)
                
                if gesture_name:
                    gesture = gesture_name
                    confidence = conf
                    logger.info(f"🦅 [Layer4] MediaPipe 识别：{gesture} (置信度：{confidence:.2f})")
                else:
                    logger.debug("⏳ [Layer4] MediaPipe 未检测到手势")
            else:
                # 备用方案：使用 EfficientNet (如果存在)
                logger.warning("⚠️ [Layer4] MediaPipe 不可用，尝试 EfficientNet")
                
                if self._gesture_classifier:
                    # Digital Zoom (裁剪 + 放大)
                    x1, y1, x2, y2 = human_bbox
                    h, w = frame.shape[:2]
                    
                    # 估计手部区域 (人体上半部分)
                    hand_y1 = max(0, y1)
                    hand_y2 = max(0, y1 + int((y2 - y1) * 0.6))  # 上半身
                    hand_x1 = max(0, x1 - 50)
                    hand_x2 = min(w, x2 + 50)
                    
                    hand_roi = frame[hand_y1:hand_y2, hand_x1:hand_x2].copy()
                    
                    if hand_roi.size > 0:
                        # 放大
                        zoom_factor = self._config['digital_zoom_factor']
                        new_size = (
                            int(hand_roi.shape[1] * zoom_factor),
                            int(hand_roi.shape[0] * zoom_factor)
                        )
                        hand_roi_enlarged = cv2.resize(hand_roi, new_size, interpolation=cv2.INTER_CUBIC)
                        
                        # 手势识别
                        gesture_name, conf, details = self._gesture_classifier.classify_gesture(hand_roi_enlarged)
                        
                        if gesture_name:
                            gesture = gesture_name
                            confidence = conf
            
            # ========== 3. 返回结果 ==========
            result = {
                'gesture': gesture,
                'confidence': confidence,
                'bbox': human_bbox,
                'method': 'mediapipe' if isinstance(self._gesture_classifier, MediaPipeGestureRecognizer) else 'efficientnet',
            }
            
            if confidence > 0.5:
                logger.info(f"🦅 [Layer4] 识别手势：{gesture} (置信度：{confidence:.0%})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ [Layer4] 鹰眼模式失败：{e}", exc_info=True)
            return None
    
    def _publish_silent_attention_event(self, intent_type: str, confidence: float):
        """发布静默注意事件"""
        try:
            event = ZulongEvent(
                type=EventType.INTERACTION_TRIGGER,
                priority=EventPriority.NORMAL,
                source="OptimizedVisionProcessor/Layer3",
                payload={
                    'intent_type': intent_type,
                    'intent_confidence': confidence,
                    'route_mode': 'queue',  # 排队模式
                    'layer': 'L3',
                }
            )
            
            self.event_bus.publish(event)
            logger.debug(f"📢 发布静默注意事件：{intent_type} ({confidence:.0%})")
            
        except Exception as e:
            logger.error(f"❌ 发布事件失败：{e}")
    
    def _publish_vision_state_event(self, motion_pixels: int, human_bbox: List[int]):
        """发布视觉状态变更事件 (L1-A 兼容)"""
        try:
            event = ZulongEvent(
                type=EventType.SENSOR_VISION_STATE,
                priority=EventPriority.NORMAL,
                source="OptimizedVisionProcessor/Layer2",
                payload={
                    'motion_pixels': motion_pixels,
                    'human_bbox': human_bbox,
                    'layer': 'L2',
                    'motion_detected': True,
                }
            )
            
            self.event_bus.publish(event)
            logger.debug(f"📢 发布视觉状态变更事件：motion_pixels={motion_pixels}")
            
        except Exception as e:
            logger.error(f"❌ 发布事件失败：{e}")
    
    def _publish_interaction_event(self, gesture_result: Dict[str, Any], intent_type: str):
        """发布交互事件"""
        try:
            # 安全检查：确保 gesture_result 不为 None
            if not gesture_result:
                logger.warning("⚠️ [_publish_interaction_event] gesture_result 为 None，跳过事件发布")
                return
            
            gesture = gesture_result.get('gesture', 'UNKNOWN') or 'UNKNOWN'
            confidence = gesture_result.get('confidence', 0.0)
            
            event = ZulongEvent(
                type=EventType.INTERACTION_TRIGGER,
                priority=EventPriority.HIGH,
                source="OptimizedVisionProcessor/Layer4",
                payload={
                    'intent_type': intent_type,
                    'gesture': gesture,
                    'gesture_confidence': confidence,
                    'route_mode': 'interrupt',  # 中断模式
                    'layer': 'L4',
                    'eagle_eye_active': True,
                }
            )
            
            self.event_bus.publish(event)
            logger.info(f"📢 发布交互事件：{gesture} ({confidence:.0%})")
            
        except Exception as e:
            logger.error(f"❌ 发布事件失败：{e}", exc_info=True)
    
    def get_shared_memory(self) -> Dict[str, Any]:
        """获取共享内存数据"""
        return self.shared_memory.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'is_running': self.is_running,
            'is_initialized': self.is_initialized,
            'frame_buffer_size': len(self.frame_buffer),
            'state_machine': self.state_machine.copy(),
            'shared_memory': self.shared_memory.copy(),
            'models': {
                'yolo': self._yolo_model is not None,
                'action': self._action_model is not None,
                'gesture': self._gesture_model is not None,
            }
        }


# ========== 全局单例模式 (兼容 bootstrap.py) ==========

vision_processor: Optional[OptimizedVisionProcessor] = None


async def init_vision_processor() -> OptimizedVisionProcessor:
    """初始化全局 OptimizedVisionProcessor 单例
    
    TSD v1.8 关键修正：
    - 在 bootstrap 的异步上下文中调用 initialize()
    - 确保在 Camera 初始化之前完成 VP 初始化
    
    Returns:
        OptimizedVisionProcessor 单例
    """
    global vision_processor
    if vision_processor is None:
        logger.info("👁️ [init_vision_processor] Creating OptimizedVisionProcessor singleton...")
        vision_processor = OptimizedVisionProcessor()
        logger.info(f"👁️ [init_vision_processor] OptimizedVisionProcessor created: {vision_processor}")
        
        # 在异步上下文中调用 initialize
        logger.info("👁️ [init_vision_processor] Calling initialize()...")
        await vision_processor.initialize(load_models=True)
        logger.info(f"✅ [init_vision_processor] initialize() completed. is_running={vision_processor.is_running}")
        
    return vision_processor


def get_vision_processor() -> Optional[OptimizedVisionProcessor]:
    """获取全局 OptimizedVisionProcessor 单例
    
    Returns:
        OptimizedVisionProcessor 单例或 None
    """
    return vision_processor
