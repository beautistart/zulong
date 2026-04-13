# File: zulong/plugins/vision/l1c_vision_silent_plugin.py
"""
L1-C 视觉静默注意插件 (TSD v1.8 三层注意力机制 + MediaPipe 集成)

TSD v1.8 对应:
- 2.2.6 L1 层模块化插件架构
- 4.1 L1-A 受控反射引擎 (插件化实现)
- 三层注意力机制原子任务
- 4.2.1 L1-B 注意力控制器：视听信息流回溯

核心改进:
1. **静默注意模式**: 持续运行轻量级检测，但默认不生成事件
2. **状态比较**: 仅当状态翻转 (如人脸出现/消失) 时生成 ZulongEvent
3. **环形缓冲区**: 维护最近 2 秒的有效帧，支持时间轴采样
4. **MediaPipe 集成**: 手势检测、视线估计、靠近趋势分析
5. **YOLO 集成**: 人物检测与距离估算
6. **4 帧精选**: 提取 T-2s, T-1.3s, T-0.7s, T-0s 四个关键帧
7. **动态路由**: 根据 L2 状态选择 DIRECT_WAKEUP 或 INTERACTION_TRIGGER

输入 (shared_memory):
- "camera_frame": 摄像头帧 (numpy array)

输出 (ZulongEvent):
- DIRECT_WAKEUP: L2 空闲时直连唤醒
- INTERACTION_TRIGGER: L2 忙碌时触发中断
- EMERGENCY_ALERT: 紧急警报 (摔倒检测)

共享内存输出:
- "vision.target_pos": 目标 3D 坐标
- "vision.face_detected": 人脸检测标志
- "vision.face_bbox": 人脸边界框
"""

import logging
from typing import Any, Dict, List, Optional
import time
import numpy as np
import threading
from collections import deque
import os

from zulong.modules.l1.core.interface import L1PluginBase, ZulongEvent, EventPriority, EventType, create_event
from zulong.core.state_manager import state_manager
from zulong.core.event_bus import event_bus
from zulong.l1c.frame_buffer import FrameBuffer, SampledFrame

logger = logging.getLogger(__name__)

# YOLO 模型 (延迟加载)
_yolo_model = None
_yolo_initialized = False
_yolo_lock = threading.Lock()

# MediaPipe 模型 (延迟加载)
_mp_hands = None
_mp_face_mesh = None
_mp_solutions = None
_mp_initialized = False
_mp_lock = threading.Lock()

# 模型路径
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')


def _lazy_load_yolo():
    """延迟加载 YOLO 模型"""
    global _yolo_model, _yolo_initialized
    
    if _yolo_initialized:
        return _yolo_model is not None
    
    with _yolo_lock:
        if _yolo_initialized:
            return _yolo_model is not None
        
        try:
            from ultralytics import YOLO
            
            # 查找模型文件
            yolo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'yolov10n.pt')
            yolo_path = os.path.normpath(yolo_path)
            
            if not os.path.exists(yolo_path):
                logger.warning(f"⚠️ YOLO 模型文件不存在: {yolo_path}")
                _yolo_initialized = True
                return False
            
            _yolo_model = YOLO(yolo_path)
            _yolo_initialized = True
            logger.info("✅ YOLOv10-Nano 模型加载完成")
            return True
            
        except ImportError:
            logger.warning("⚠️ ultralytics 未安装，人物检测将使用 Mock 模式")
            _yolo_initialized = True
            return False
        except Exception as e:
            logger.error(f"❌ YOLO 加载失败：{e}")
            _yolo_initialized = True
            return False


def _lazy_load_mediapipe():
    """延迟加载 MediaPipe 模型 (使用新版 tasks API)"""
    global _mp_hands, _mp_face_mesh, _mp_solutions, _mp_initialized
    
    if _mp_initialized:
        return _mp_solutions is not False
    
    with _mp_lock:
        if _mp_initialized:
            return _mp_solutions is not False
        
        try:
            import mediapipe as mp
            _mp_solutions = mp.solutions
            
            # 使用新版 tasks API
            hand_model_path = os.path.join(MODELS_DIR, 'hand_landmarker.task')
            face_model_path = os.path.join(MODELS_DIR, 'face_landmarker.task')
            
            # 检查模型文件
            if os.path.exists(hand_model_path):
                base_options_hands = mp.tasks.BaseOptions(model_asset_path=hand_model_path)
                options_hands = mp.tasks.vision.HandLandmarkerOptions(
                    base_options=base_options_hands,
                    num_hands=2
                )
                _mp_hands = mp.tasks.vision.HandLandmarker.create_from_options(options_hands)
                logger.info("✅ MediaPipe Hands 模型加载完成 (tasks API)")
            else:
                logger.warning(f"⚠️ Hand model 不存在: {hand_model_path}，使用旧版 API")
                _mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            
            if os.path.exists(face_model_path):
                base_options_face = mp.tasks.BaseOptions(model_asset_path=face_model_path)
                options_face = mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=base_options_face
                )
                _mp_face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(options_face)
                logger.info("✅ MediaPipe Face 模型加载完成 (tasks API)")
            else:
                logger.warning(f"⚠️ Face model 不存在: {face_model_path}，使用旧版 API")
                _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True
                )
            
            _mp_initialized = True
            logger.info("✅ MediaPipe 模型加载完成")
            return True
            
        except ImportError:
            logger.warning("⚠️ MediaPipe 未安装，手势/视线检测将使用 Mock 模式")
            _mp_solutions = False
            _mp_initialized = True
            return False
        except Exception as e:
            logger.error(f"❌ MediaPipe 加载失败：{e}")
            _mp_solutions = False
            _mp_initialized = True
            return False


class L1C_VisionSilentPlugin(L1PluginBase):
    """
    L1-C 视觉静默注意插件 (增强版)
    
    核心逻辑:
    1. 每帧运行轻量级检测 (MediaPipe Hands/Face)
    2. 内部维护状态记忆
    3. 仅当状态翻转时生成 ZulongEvent
    4. 环形缓冲区存储最近 2 秒的有效帧
    5. 动态路由：根据 L2 状态选择事件类型
    
    状态翻转场景:
    - 场景 A: 检测到交互意图 (挥手/注视/靠近) → 生成 INTERACTION_TRIGGER 或 DIRECT_WAKEUP
    - 场景 B: 紧急异常 (摔倒检测) → 生成 EMERGENCY_ALERT
    - 场景 C: 从有到无 (人脸离开) → 静默更新，不生成事件
    
    TSD v1.8 对应:
    - 第 4.1 节 L1-A 受控反射引擎
    - 三层注意力机制原子任务
    - 第 4.2 节 L1-B 调度与守门
    """
    
    @property
    def module_id(self) -> str:
        return "L1C/Vision_Silent"
    
    @property
    def priority(self) -> EventPriority:
        return EventPriority.HIGH
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化插件"""
        super().__init__(config)
        
        # 帧缓冲区 (核心组件)
        self._frame_buffer: Optional[FrameBuffer] = None
        
        # 状态记忆
        self._state_memory = {
            "face_detected": False,
            "last_face_pos": None,
            "gesture_active": False,
            "fall_detected": False,
            "last_intent_type": None
        }
        
        # 动作分析历史 (用于计算振荡和速度)
        self._wrist_history: deque = deque(maxlen=15)  # 最近 0.5 秒的手腕位置
        self._bbox_history: deque = deque(maxlen=15)   # 最近 0.5 秒的 BBox 大小
        
        # 防抖与冷却
        self._last_trigger_time = 0.0
        self._trigger_cooldown = 2.0  # 秒
        
        # MediaPipe 模型 (延迟加载)
        self._mediapipe_loaded = False
        self._mp_hands = None
        self._mp_face_mesh = None
        
        # 运动检测器 (简化版，用于 L1 过滤)
        self._motion_detector = None
        
        # 线程锁
        self._lock = threading.Lock()
    
    def initialize(self, shared_memory: Dict) -> bool:
        """
        初始化静默视觉插件
        
        Args:
            shared_memory: 共享内存
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("🔌 [VisionSilent] 正在初始化静默注意插件...")
            
            # 调用基类初始化
            if not super().initialize(shared_memory):
                return False
            
            # 读取配置
            self._thresholds = self.get_config("thresholds", {
                "face_conf": 0.8,
                "gesture_conf": 0.9,
                "idle_time_sec": 2.0,
                "fall_conf": 0.95,
                "wave_oscillation": 0.15,
                "approach_growth_rate": 0.5
            })
            
            # 初始化帧缓冲区 (60 帧 ≈ 2 秒 @ 30fps)
            self._frame_buffer = FrameBuffer(
                max_frames=60,
                jpeg_quality=75,
                default_crop_size=(448, 448),
                padding_ratio=0.1
            )
            
            # 延迟加载 MediaPipe
            self._mediapipe_loaded = _lazy_load_mediapipe()
            if self._mediapipe_loaded:
                global _mp_hands, _mp_face_mesh
                self._mp_hands = _mp_hands
                self._mp_face_mesh = _mp_face_mesh
            
            # 初始化共享内存
            shared_memory["vision.face_detected"] = False
            shared_memory["vision.face_bbox"] = None
            shared_memory["vision.target_pos"] = None
            shared_memory["vision.silent_mode"] = True
            
            logger.info(f"✅ [VisionSilent] 初始化完成")
            logger.info(f"   - MediaPipe: {'已加载' if self._mediapipe_loaded else 'Mock 模式'}")
            logger.info(f"   - 帧缓冲区: 60 帧 (≈2秒)")
            logger.info(f"   - 冷却时间: {self._trigger_cooldown}s")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ [VisionSilent] 初始化失败：{e}", exc_info=True)
            return False
    
    def process_cycle(self, shared_memory: Dict) -> List[ZulongEvent]:
        """
        单周期处理 (核心逻辑 - 静默注意 + MediaPipe 检测)
        
        流程:
        1. 从共享内存读取摄像头帧
        2. L1 运动检测 (快速过滤)
        3. L2 语义初筛 (人物检测)
        4. 写入帧缓冲区
        5. L3 意图确认 (MediaPipe 手势/视线/靠近)
        6. 状态比较与事件生成
        7. 动态路由
        
        Args:
            shared_memory: 共享内存
        
        Returns:
            List[ZulongEvent]: 事件列表
        """
        events: List[ZulongEvent] = []
        current_time = time.time()
        
        try:
            # ========== 1. 读取摄像头帧 ==========
            frame = shared_memory.get("camera_frame")
            if frame is None:
                return events
            
            h, w = frame.shape[:2]
            
            # ========== 2. L1 运动检测 (快速过滤) ==========
            if not self._check_motion(frame):
                return events  # 静态场景，跳过
            
            # ========== 3. L2 语义初筛 (人物检测) ==========
            person_bbox = self._detect_person(frame)
            
            # 写入帧缓冲区 (仅存有效帧)
            self._frame_buffer.add_frame(
                frame,
                bbox=person_bbox,
                confidence=0.9 if person_bbox else 0.0,
                timestamp=current_time
            )
            
            if person_bbox:
                self._bbox_history.append(person_bbox)
            
            # ========== 4. L3 意图确认 (MediaPipe) ==========
            intent_type, confidence = self._analyze_intent(frame, person_bbox)
            
            if intent_type:
                # 检查冷却时间
                if current_time - self._last_trigger_time > self._trigger_cooldown:
                    self._last_trigger_time = current_time
                    
                    # 构建事件载荷 (包含 4 帧精选)
                    payload = self._build_event_payload(intent_type, confidence, person_bbox, current_time)
                    
                    if payload:
                        # 动态路由
                        self._route_event(payload, shared_memory)
            
            # ========== 5. 更新共享内存 ==========
            if person_bbox:
                shared_memory["vision.face_detected"] = True
                shared_memory["vision.face_bbox"] = person_bbox
                shared_memory["vision.target_pos"] = self._estimate_depth(person_bbox, (h, w))
            else:
                shared_memory["vision.face_detected"] = False
                shared_memory["vision.face_bbox"] = None
            
            return events
            
        except Exception as e:
            logger.error(f"❌ [VisionSilent] 处理周期失败：{e}", exc_info=True)
            return []
    
    def _check_motion(self, frame: np.ndarray) -> bool:
        """
        L1 运动检测 (快速过滤)
        
        Args:
            frame: 视频帧
        
        Returns:
            bool: 是否检测到运动
        """
        # 简化实现：使用帧差分
        # 实际应使用 motion_detector.py 的光流法
        return True  # 暂时跳过，由外部 motion_detector 处理
    
    def _detect_person(self, frame: np.ndarray) -> Optional[List[float]]:
        """
        L2 人物检测 (语义初筛) - 使用 YOLOv10
        
        Args:
            frame: 视频帧
        
        Returns:
            Optional[List[float]]: 人物边界框 [x1, y1, x2, y2] 或 None
        """
        # 延迟加载 YOLO
        if not _lazy_load_yolo():
            # YOLO 不可用，使用 Mock
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            box_w, box_h = w // 3, h // 2
            return [cx - box_w // 2, cy - box_h // 2, cx + box_w // 2, cy + box_h // 2]
        
        try:
            # 使用 YOLO 检测人物 (class 0)
            results = _yolo_model(frame, classes=[0], verbose=False, conf=0.3)
            
            if not results or len(results[0].boxes) == 0:
                return None
            
            # 获取最大的人物框
            best_box = None
            best_area = 0
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                
                if area > best_area:
                    best_area = area
                    best_box = [float(x1), float(y1), float(x2), float(y2)]
            
            return best_box
            
        except Exception as e:
            logger.debug(f"YOLO 检测失败：{e}")
            return None
    
    def _analyze_intent(self, frame: np.ndarray, bbox: Optional[List[float]]) -> tuple[Optional[str], float]:
        """
        L3 意图确认 (MediaPipe)
        
        分析:
        1. 手势检测 (挥手)
        2. 视线估计 (注视)
        3. 靠近检测 (BBox 变化率)
        
        Args:
            frame: 视频帧
            bbox: 人物边界框
        
        Returns:
            tuple: (意图类型, 置信度) 或 (None, 0.0)
        """
        intent_type = None
        confidence = 0.0
        
        # 确保 MediaPipe 已加载
        if not self._mediapipe_loaded:
            self._mediapipe_loaded = _lazy_load_mediapipe()
            if self._mediapipe_loaded:
                global _mp_hands, _mp_face_mesh
                self._mp_hands = _mp_hands
                self._mp_face_mesh = _mp_face_mesh
        
        # 转换为 RGB
        rgb_frame = frame[:, :, ::-1]  # BGR -> RGB
        
        # 1. 手势识别 (挥手检测)
        if self._mp_hands:
            try:
                # 检测是否是新版 tasks API
                if hasattr(self._mp_hands, 'detect'):
                    # 新版 tasks API
                    import mediapipe as mp
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    hands_res = self._mp_hands.detect(mp_image)
                    
                    if hands_res.hand_landmarks:
                        landmarks = hands_res.hand_landmarks[0]
                        wrist_pos = (landmarks[0].x, landmarks[0].y)
                        self._wrist_history.append(wrist_pos)
                        
                        if self._check_wave_v2(landmarks):
                            intent_type = "WAVE"
                            confidence = 0.92
                else:
                    # 旧版 API
                    hands_res = self._mp_hands.process(rgb_frame)
                    
                    if hands_res.multi_hand_landmarks:
                        landmarks = hands_res.multi_hand_landmarks[0]
                        wrist_pos = (landmarks.landmark[0].x, landmarks.landmark[0].y)
                        self._wrist_history.append(wrist_pos)
                        
                        if self._check_wave():
                            intent_type = "WAVE"
                            confidence = 0.92
                        
            except Exception as e:
                logger.debug(f"手势检测失败：{e}")
        
        # 2. 视线估计 (注视检测)
        if not intent_type and self._mp_face_mesh:
            try:
                # 检测是否是新版 tasks API
                if hasattr(self._mp_face_mesh, 'detect'):
                    # 新版 tasks API
                    import mediapipe as mp
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    face_res = self._mp_face_mesh.detect(mp_image)
                    
                    if face_res.face_landmarks:
                        if self._check_gaze_v2(face_res.face_landmarks[0], frame.shape):
                            intent_type = "GAZE"
                            confidence = 0.88
                else:
                    # 旧版 API
                    face_res = self._mp_face_mesh.process(rgb_frame)
                    
                    if face_res.multi_face_landmarks:
                        if self._check_gaze(face_res.multi_face_landmarks[0], frame.shape):
                            intent_type = "GAZE"
                            confidence = 0.88
                        
            except Exception as e:
                logger.debug(f"视线检测失败：{e}")
        
        # 3. 靠近检测 (BBox 变化率)
        if not intent_type:
            if self._check_approach():
                intent_type = "APPROACH"
                confidence = 0.85
        
        return intent_type, confidence
    
    def _check_wave(self) -> bool:
        """
        检测挥手动作 (旧版 API)
        
        通过分析手腕位置的垂直振荡判断
        
        Returns:
            bool: 是否检测到挥手
        """
        if len(self._wrist_history) < 10:
            return False
        
        ys = [p[1] for p in self._wrist_history]
        oscillation = max(ys) - min(ys)
        
        return oscillation > self._thresholds.get("wave_oscillation", 0.15)
    
    def _check_wave_v2(self, landmarks) -> bool:
        """
        检测挥手动作 (新版 tasks API)
        
        通过分析手腕位置的水平振荡判断
        
        Args:
            landmarks: MediaPipe 手部关键点列表
        
        Returns:
            bool: 是否检测到挥手
        """
        # 获取手腕位置 (索引 0)
        wrist_pos = (landmarks[0].x, landmarks[0].y)
        self._wrist_history.append(wrist_pos)
        
        if len(self._wrist_history) < 10:
            return False
        
        # 分析水平方向的振荡
        xs = [p[0] for p in self._wrist_history]
        
        # 计算方向变化次数
        direction_changes = 0
        prev_direction = None
        
        for i in range(1, len(xs)):
            if xs[i] > xs[i-1] + 0.02:
                current_direction = 'right'
            elif xs[i] < xs[i-1] - 0.02:
                current_direction = 'left'
            else:
                continue
            
            if prev_direction is not None and current_direction != prev_direction:
                direction_changes += 1
            prev_direction = current_direction
        
        # 需要至少 2 次方向变化才算挥手
        if direction_changes >= 2:
            logger.info(f"👋 检测到挥手！方向变化: {direction_changes}")
            return True
        
        return False
    
    def _check_gaze(self, face_landmarks, frame_shape: tuple) -> bool:
        """
        检测注视行为 (旧版 API)
        
        通过分析面部朝向判断是否注视摄像头
        
        Args:
            face_landmarks: MediaPipe 面部关键点
            frame_shape: 帧尺寸
        
        Returns:
            bool: 是否检测到注视
        """
        # 简化实现：检查鼻尖和眼睛中心的相对位置
        # 实际应计算 3D 向量与摄像头光轴的夹角
        try:
            # 获取鼻尖位置 (landmark 1)
            nose = face_landmarks.landmark[1]
            
            # 检查是否在画面中央区域
            center_x = 0.5
            center_y = 0.5
            threshold = 0.2
            
            if abs(nose.x - center_x) < threshold and abs(nose.y - center_y) < threshold:
                return True
                
        except Exception:
            pass
        
        return False
    
    def _check_gaze_v2(self, landmarks, frame_shape: tuple) -> bool:
        """
        检测注视行为 (新版 tasks API)
        
        通过分析面部朝向判断是否注视摄像头
        
        Args:
            landmarks: MediaPipe 面部关键点列表
            frame_shape: 帧尺寸
        
        Returns:
            bool: 是否检测到注视
        """
        try:
            # 获取鼻尖位置 (索引 1)
            nose = landmarks[1]
            
            # 获取眼睛位置
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            
            # 计算眼睛中点
            eye_center_x = (left_eye.x + right_eye.x) / 2
            eye_center_y = (left_eye.y + right_eye.y) / 2
            
            # 计算注视向量
            gaze_x = nose.x - eye_center_x
            gaze_y = nose.y - eye_center_y
            
            # 计算向量大小
            gaze_magnitude = (gaze_x**2 + gaze_y**2) ** 0.5
            
            # 如果向量接近 0，说明正在注视前方
            if gaze_magnitude < 0.05:
                logger.debug(f"👀 检测到注视: magnitude={gaze_magnitude:.3f}")
                return True
                
        except Exception:
            pass
        
        return False
    
    def _check_approach(self) -> bool:
        """
        检测靠近行为
        
        通过分析 BBox 面积变化率判断
        
        Returns:
            bool: 是否检测到靠近
        """
        if len(self._bbox_history) < 5:
            return False
        
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in self._bbox_history if b]
        
        if len(areas) < 5:
            return False
        
        growth_rate = (areas[-1] - areas[-5]) / (areas[-5] + 1e-5)
        
        return growth_rate > self._thresholds.get("approach_growth_rate", 0.5)
    
    def _build_event_payload(
        self, 
        intent_type: str, 
        confidence: float, 
        bbox: Optional[List[float]],
        timestamp: float
    ) -> Optional[Dict[str, Any]]:
        """
        构建事件载荷 (包含 4 帧精选)
        
        Args:
            intent_type: 意图类型
            confidence: 置信度
            bbox: 人物边界框
            timestamp: 时间戳
        
        Returns:
            Dict: 事件载荷
        """
        try:
            # 从缓冲区采样 4 帧
            sampled_frames = self._frame_buffer.sample_frames_for_event(
                num_frames=4,
                crop_to_bbox=True,
                crop_size=(448, 448)
            )
            
            if not sampled_frames:
                logger.warning("⚠️ 无法采样帧，跳过事件构建")
                return None
            
            # 提取 Base64 图像
            visual_sequence_b64 = [f.image_b64 for f in sampled_frames]
            
            # 计算距离 (基于 BBox 大小)
            person_distance = self._estimate_distance(bbox)
            
            return {
                "intent_type": intent_type,
                "intent_confidence": confidence,
                "timestamp": timestamp,
                "visual_sequence_b64": visual_sequence_b64,
                "frame_count": len(visual_sequence_b64),
                "person_distance": person_distance,
                "person_bbox": bbox,
                "route_mode": "pending"  # 将在路由时确定
            }
            
        except Exception as e:
            logger.error(f"❌ 构建事件载荷失败：{e}")
            return None
    
    def _estimate_depth(self, bbox: Optional[List[float]], frame_shape: tuple) -> List[float]:
        """
        估算目标 3D 坐标
        
        Args:
            bbox: 边界框
            frame_shape: 帧尺寸 (height, width)
        
        Returns:
            List[float]: 3D 坐标 [x, y, z]
        """
        if not bbox:
            return [0.0, 0.0, 0.0]
        
        h, w = frame_shape
        x1, y1, x2, y2 = bbox
        
        center_x = (x1 + x2) / 2 / w
        center_y = (y1 + y2) / 2 / h
        size = (x2 - x1 + y2 - y1) / 2
        
        depth = max(0.5, min(3.0, 5.0 / (size + 1e-5)))
        
        return [center_x, center_y, depth]
    
    def _estimate_distance(self, bbox: Optional[List[float]]) -> float:
        """
        估算人物距离
        
        Args:
            bbox: 边界框
        
        Returns:
            float: 距离 (米)
        """
        if not bbox:
            return float('inf')
        
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        
        # 简化估算：面积越大，距离越近
        # 假设 1 米处面积为 100000 像素
        distance = 100000 / (area + 1e-5)
        
        return max(0.3, min(5.0, distance))
    
    def _route_event(self, payload: Dict[str, Any], shared_memory: Dict):
        """
        动态路由事件
        
        根据 L2 状态选择:
        - IDLE: DIRECT_WAKEUP (直连)
        - BUSY/WAITING: INTERACTION_TRIGGER (中断)
        
        Args:
            payload: 事件载荷
            shared_memory: 共享内存
        """
        # 获取 L2 状态
        effective_state = state_manager.get_effective_status()
        l2_status = state_manager.get_l2_status()
        
        # 确定路由模式
        if effective_state == "IDLE":
            route_mode = "direct"
            event_type = EventType.DIRECT_WAKEUP
            priority = EventPriority.HIGH
        else:
            route_mode = "interrupt"
            event_type = EventType.INTERACTION_TRIGGER
            priority = EventPriority.HIGH
        
        payload["route_mode"] = route_mode
        
        # 创建事件
        event = ZulongEvent(
            type=event_type,
            priority=priority,
            source=self.module_id,
            payload=payload
        )
        
        # 发布事件
        event_bus.publish(event)
        
        logger.info(
            f"📡 [VisionSilent] 路由事件：{event_type.value} "
            f"(intent={payload['intent_type']}, L2={l2_status.name}, mode={route_mode})"
        )
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            Dict: 健康状态
        """
        buffer_info = self._frame_buffer.get_buffer_info() if self._frame_buffer else {}
        
        return {
            "status": "healthy",
            "mediapipe_loaded": self._mediapipe_loaded,
            "buffer_info": buffer_info,
            "state_memory": self._state_memory,
            "wrist_history_len": len(self._wrist_history),
            "bbox_history_len": len(self._bbox_history),
            "timestamp": time.time()
        }
    
    def shutdown(self):
        """关闭插件"""
        if self._frame_buffer:
            self._frame_buffer.clear()
        
        # 释放 MediaPipe 资源
        if self._mp_hands:
            self._mp_hands.close()
        if self._mp_face_mesh:
            self._mp_face_mesh.close()
        
        logger.info("🔌 [VisionSilent] 已关闭")


if __name__ == "__main__":
    # 测试代码
    print("🧪 测试 L1-C 视觉静默注意插件...")
    
    plugin = L1C_VisionSilentPlugin()
    shared_memory = {}
    
    if plugin.initialize(shared_memory):
        print("✅ 插件初始化成功")
        
        # 模拟处理周期
        print("\n📊 模拟处理周期...")
        
        # 周期 1: 有帧
        shared_memory["camera_frame"] = np.zeros((480, 640, 3), dtype=np.uint8)
        events = plugin.process_cycle(shared_memory)
        print(f"周期 1: {len(events)} 个事件")
        
        # 查看健康状态
        health = plugin.health_check()
        print(f"\n健康状态: {health}")
        
        print("\n✅ 测试完成!")
    else:
        print("❌ 插件初始化失败")
