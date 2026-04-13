# File: zulong/expert_skills/vision_skill.py
# L3 视觉专家技能 - 物体识别、场景理解、人脸检测

"""
祖龙 (ZULONG) L3 视觉专家技能

对应 TSD v1.7:
- 2.3.2 专家模型层：L3 专家技能池
- 视觉专家：物体识别、场景理解、人脸检测
- Phase 6: 真实模型集成（InternVL-2.5-1B）

功能:
- 物体识别（支持模拟模式和 InternVL 真实模型）
- 场景理解
- 人脸检测与识别
- 视觉注意力管理
- 视觉历史追踪
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import time

# Phase 6: 添加 PIL 支持
try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    """检测到的物体"""
    object_id: str
    label: str  # 物体标签
    confidence: float  # 置信度
    bbox: Tuple[int, int, int, int]  # 边界框 (x1, y1, x2, y2)
    position_3d: Optional[Tuple[float, float, float]] = None  # 3D 位置 (x, y, z)
    timestamp: float = field(default_factory=time.time)


@dataclass
class DetectedFace:
    """检测到的人脸"""
    face_id: str
    person_name: Optional[str] = None  # 如果已识别
    confidence: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    emotions: Dict[str, float] = field(default_factory=dict)  # 情绪分析
    timestamp: float = field(default_factory=time.time)


@dataclass
class SceneUnderstanding:
    """场景理解结果"""
    scene_type: str  # 场景类型（客厅/厨房/办公室等）
    confidence: float
    objects_count: int  # 检测到的物体数量
    people_count: int  # 检测到的人数
    description: str  # 自然语言描述
    timestamp: float = field(default_factory=time.time)


class VisionSkill:
    """L3 视觉专家技能
    
    TSD v1.7 对应规则:
    - L3 专家技能池：视觉专家
    - 物体识别、场景理解、人脸检测
    - Phase 6: 支持 InternVL-2.5-1B 真实模型
    - 支持 L2 调用
    
    功能:
    - 物体检测与识别（模拟模式/InternVL 模式）
    - 人脸检测与识别
    - 场景理解
    - 视觉注意力管理
    - 视觉历史追踪
    """
    
    def __init__(self, 
                 skill_id: str = "vision_expert",
                 detection_threshold: float = 0.5,
                 use_internvl: bool = False,
                 internvl_config: Optional[Any] = None):
        """初始化视觉专家技能
        
        Args:
            skill_id: 技能 ID
            detection_threshold: 检测阈值
            use_internvl: 是否使用 InternVL 真实模型（Phase 6）
            internvl_config: InternVL 配置（可选）
        """
        self.skill_id = skill_id
        self.detection_threshold = detection_threshold
        self.use_internvl = use_internvl
        
        # InternVL 模型（Phase 6）
        self._internvl_model = None
        if use_internvl:
            try:
                from .internvl_model import InternVLModel, InternVLConfig
                config = internvl_config or InternVLConfig()
                self._internvl_model = InternVLModel.get_instance(config)
                logger.info(f"[VisionSkill] InternVL 模型已初始化：{skill_id}")
            except Exception as e:
                logger.warning(f"[VisionSkill] InternVL 初始化失败，降级到模拟模式：{e}")
                self.use_internvl = False
        
        # 视觉注意力
        self.attention_focus: Optional[Tuple[float, float]] = None  # (yaw, pitch)
        
        # 检测结果缓存
        self.detected_objects: Dict[str, DetectedObject] = {}
        self.detected_faces: Dict[str, DetectedFace] = {}
        self.current_scene: Optional[SceneUnderstanding] = None
        
        # 视觉历史
        self.vision_history: List[Dict[str, Any]] = []
        
        # 统计信息
        self.stats = {
            'total_detections': 0,
            'objects_detected': 0,
            'faces_detected': 0,
            'scenes_analyzed': 0,
            'last_detection_time': 0.0,
            'internvl_inferences': 0 if use_internvl else None
        }
        
        logger.info(f"[VisionSkill] 初始化完成：id={skill_id}, "
                   f"threshold={detection_threshold}, internvl={use_internvl}")
    
    # ========== 物体识别 ==========
    
    def detect_objects(self, 
                       image_data: Any,
                       labels: Optional[List[str]] = None) -> List[DetectedObject]:
        """检测物体
        
        Args:
            image_data: 图像数据（可以是 PIL Image、numpy array 等）
            labels: 指定要检测的标签列表（可选）
            
        Returns:
            List[DetectedObject]: 检测到的物体列表
        """
        logger.info(f"[VisionSkill] 物体检测：image_type={type(image_data)}, "
                   f"labels={labels}, internvl={self.use_internvl}")
        
        detected = []
        
        # Phase 6: 使用 InternVL 真实模型
        if self.use_internvl and self._internvl_model is not None:
            try:
                # 确保是 PIL Image
                if not isinstance(image_data, Image.Image):
                    image_data = Image.fromarray(image_data)
                
                # 使用 InternVL 检测
                internvl_objects = self._internvl_model.detect_objects(
                    image=image_data,
                    labels=labels
                )
                
                # 转换结果格式
                for obj_data in internvl_objects:
                    confidence = obj_data.get('confidence', 0.5)
                    if confidence < self.detection_threshold:
                        continue
                    
                    obj = DetectedObject(
                        object_id=f"internvl_{int(time.time())}_{len(detected)}",
                        label=obj_data.get('label', 'unknown'),
                        confidence=confidence,
                        bbox=obj_data.get('bbox', (0, 0, 0, 0)),
                        position_3d=obj_data.get('position_3d'),
                        timestamp=obj_data.get('timestamp', time.time())
                    )
                    
                    detected.append(obj)
                    self.detected_objects[obj.object_id] = obj
                
                # 更新统计
                if self.stats.get('internvl_inferences') is not None:
                    self.stats['internvl_inferences'] += 1
                
                logger.info(f"[VisionSkill] InternVL 检测到 {len(detected)} 个物体")
                
            except Exception as e:
                logger.error(f"[VisionSkill] InternVL 检测失败，降级到模拟模式：{e}")
                # 降级到模拟模式
                detected = self._detect_objects_mock(image_data, labels)
        else:
            # 模拟模式（向后兼容）
            detected = self._detect_objects_mock(image_data, labels)
        
        self.stats['total_detections'] += len(detected)
        self.stats['objects_detected'] += len(detected)
        self.stats['last_detection_time'] = time.time()
        
        logger.info(f"[VisionSkill] 检测到 {len(detected)} 个物体")
        
        return detected
    
    def _detect_objects_mock(self,
                             image_data: Any,
                             labels: Optional[List[str]] = None) -> List[DetectedObject]:
        """模拟物体检测（向后兼容）
        
        Args:
            image_data: 图像数据
            labels: 指定标签列表
            
        Returns:
            List[DetectedObject]: 检测到的物体列表
        """
        detected = []
        
        # 模拟一些检测结果
        mock_objects = [
            ("chair", 0.92, (100, 100, 200, 300)),
            ("table", 0.87, (300, 150, 500, 400)),
            ("person", 0.95, (400, 50, 600, 500)),
        ]
        
        for label, conf, bbox in mock_objects:
            if labels and label not in labels:
                continue
            
            if conf < self.detection_threshold:
                continue
            
            obj = DetectedObject(
                object_id=f"obj_{int(time.time())}_{len(detected)}",
                label=label,
                confidence=conf,
                bbox=bbox
            )
            
            detected.append(obj)
            self.detected_objects[obj.object_id] = obj
        
        return detected
    
    def get_object_by_id(self, object_id: str) -> Optional[DetectedObject]:
        """根据 ID 获取物体"""
        return self.detected_objects.get(object_id)
    
    def get_all_objects(self) -> List[DetectedObject]:
        """获取所有检测到的物体"""
        return list(self.detected_objects.values())
    
    def clear_objects(self):
        """清除物体检测结果"""
        self.detected_objects.clear()
        logger.debug("[VisionSkill] 已清除物体检测结果")
    
    # ========== 人脸检测 ==========
    
    def detect_faces(self, 
                     image_data: Any,
                     recognize: bool = False) -> List[DetectedFace]:
        """检测人脸
        
        Args:
            image_data: 图像数据
            recognize: 是否进行人脸识别
            
        Returns:
            List[DetectedFace]: 检测到的人脸列表
        """
        logger.info(f"[VisionSkill] 人脸检测：recognize={recognize}")
        
        # 简化实现：模拟检测结果
        # 实际应用中应集成 FaceNet、DeepFace 等模型
        detected = []
        
        # 模拟一些检测结果
        mock_faces = [
            ("face_001", "张三", 0.95, (200, 100, 350, 300)),
            ("face_002", None, 0.89, (500, 150, 650, 350)),  # 未知人脸
        ]
        
        for face_id, name, conf, bbox in mock_faces:
            if conf < self.detection_threshold:
                continue
            
            face = DetectedFace(
                face_id=face_id,
                person_name=name if recognize else None,
                confidence=conf,
                bbox=bbox,
                emotions={"happy": 0.8, "neutral": 0.15, "surprised": 0.05}
            )
            
            detected.append(face)
            self.detected_faces[face.face_id] = face
        
        self.stats['total_detections'] += len(detected)
        self.stats['faces_detected'] += len(detected)
        self.stats['last_detection_time'] = time.time()
        
        logger.info(f"[VisionSkill] 检测到 {len(detected)} 张人脸")
        
        return detected
    
    def get_face_by_id(self, face_id: str) -> Optional[DetectedFace]:
        """根据 ID 获取人脸"""
        return self.detected_faces.get(face_id)
    
    def get_all_faces(self) -> List[DetectedFace]:
        """获取所有检测到的人脸"""
        return list(self.detected_faces.values())
    
    def clear_faces(self):
        """清除人脸检测结果"""
        self.detected_faces.clear()
        logger.debug("[VisionSkill] 已清除人脸检测结果")
    
    # ========== 场景理解 ==========
    
    def understand_scene(self, 
                         image_data: Any,
                         objects: Optional[List[DetectedObject]] = None) -> SceneUnderstanding:
        """场景理解
        
        Args:
            image_data: 图像数据
            objects: 已检测的物体列表（可选）
            
        Returns:
            SceneUnderstanding: 场景理解结果
        """
        logger.info(f"[VisionSkill] 场景理解：internvl={self.use_internvl}")
        
        # Phase 6: 使用 InternVL 真实模型
        if self.use_internvl and self._internvl_model is not None:
            try:
                # 确保是 PIL Image
                if not isinstance(image_data, Image.Image):
                    image_data = Image.fromarray(image_data)
                
                # 使用 InternVL 场景理解
                scene_data = self._internvl_model.understand_scene(image=image_data)
                
                # 转换结果格式
                scene = SceneUnderstanding(
                    scene_type=scene_data.get('scene_type', 'unknown'),
                    confidence=scene_data.get('confidence', 0.5),
                    objects_count=scene_data.get('objects_count', 0),
                    people_count=scene_data.get('people_count', 0),
                    description=scene_data.get('description', '')
                )
                
                # 更新统计
                if self.stats.get('internvl_inferences') is not None:
                    self.stats['internvl_inferences'] += 1
                
                self.current_scene = scene
                self.stats['scenes_analyzed'] += 1
                self.stats['last_detection_time'] = time.time()
                
                logger.info(f"[VisionSkill] InternVL 场景理解：{scene.scene_type}")
                
                return scene
                
            except Exception as e:
                logger.error(f"[VisionSkill] InternVL 场景理解失败，降级到模拟模式：{e}")
                # 降级到模拟模式
        
        # 模拟模式（向后兼容）
        return self._understand_scene_mock(image_data, objects)
    
    def _understand_scene_mock(self,
                               image_data: Any,
                               objects: Optional[List[DetectedObject]] = None) -> SceneUnderstanding:
        """模拟场景理解（向后兼容）"""
        if objects is None:
            objects = self.get_all_objects()
        
        # 根据检测到的物体推断场景类型
        object_labels = [obj.label for obj in objects]
        
        if "chair" in object_labels and "table" in object_labels:
            scene_type = "dining_room"
            description = "检测到餐桌和椅子，可能是餐厅或会议室"
        elif "bed" in object_labels:
            scene_type = "bedroom"
            description = "检测到床，可能是卧室"
        elif "desk" in object_labels and "computer" in object_labels:
            scene_type = "office"
            description = "检测到书桌和电脑，可能是办公室"
        else:
            scene_type = "unknown"
            description = f"检测到 {len(objects)} 个物体，场景类型不明确"
        
        # 计算人数
        people_count = sum(1 for obj in objects if obj.label == "person")
        
        scene = SceneUnderstanding(
            scene_type=scene_type,
            confidence=0.85,
            objects_count=len(objects),
            people_count=people_count,
            description=description
        )
        
        self.current_scene = scene
        self.stats['scenes_analyzed'] += 1
        self.stats['last_detection_time'] = time.time()
        
        logger.info(f"[VisionSkill] 场景理解：{scene_type}, {description}")
        
        return scene
    
    def get_current_scene(self) -> Optional[SceneUnderstanding]:
        """获取当前场景"""
        return self.current_scene
    
    # ========== 视觉注意力 ==========
    
    def set_attention(self, yaw: float, pitch: float):
        """设置视觉注意力方向
        
        Args:
            yaw: 水平角度（弧度）
            pitch: 垂直角度（弧度）
        """
        self.attention_focus = (yaw, pitch)
        logger.debug(f"[VisionSkill] 视觉注意力设置：yaw={yaw:.2f}, pitch={pitch:.2f}")
    
    def get_attention(self) -> Optional[Tuple[float, float]]:
        """获取当前视觉注意力方向"""
        return self.attention_focus
    
    def reset_attention(self):
        """重置视觉注意力"""
        self.attention_focus = None
        logger.debug("[VisionSkill] 视觉注意力已重置")
    
    # ========== 视觉历史 ==========
    
    def record_detection(self, 
                         detection_type: str,
                         results: List[Any],
                         metadata: Optional[Dict[str, Any]] = None):
        """记录检测历史
        
        Args:
            detection_type: 检测类型（objects/faces/scene）
            results: 检测结果
            metadata: 元数据
        """
        record = {
            'timestamp': time.time(),
            'type': detection_type,
            'count': len(results),
            'results': results,
            'metadata': metadata or {}
        }
        
        self.vision_history.append(record)
        
        # 保留最近 100 条
        if len(self.vision_history) > 100:
            self.vision_history.pop(0)
        
        logger.debug(f"[VisionSkill] 记录检测历史：type={detection_type}, "
                    f"count={len(results)}")
    
    def get_vision_history(self, 
                           limit: int = 10,
                           detection_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取视觉历史
        
        Args:
            limit: 返回数量
            detection_type: 检测类型过滤（可选）
            
        Returns:
            List[Dict]: 历史记录列表
        """
        history = self.vision_history[-limit * 2:]  # 先获取多一些
        
        if detection_type:
            history = [h for h in history if h['type'] == detection_type]
        
        return history[:limit]
    
    # ========== 统计信息 ==========
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'objects_cached': len(self.detected_objects),
            'faces_cached': len(self.detected_faces),
            'history_count': len(self.vision_history),
            'attention_focus': self.attention_focus
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_detections': 0,
            'objects_detected': 0,
            'faces_detected': 0,
            'scenes_analyzed': 0,
            'last_detection_time': 0.0
        }
        self.vision_history.clear()
        logger.info("[VisionSkill] 统计信息已重置")
