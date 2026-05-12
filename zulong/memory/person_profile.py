# File: zulong/memory/person_profile.py
# 人物画像系统 - 多模态人物识别与画像管理 (TSD v2.5)
#
# 功能：
# - 对话人物画像（从对话中提取人物特征）
# - 人脸特征存储（与视觉系统集成，存储人脸编码）
# - 声音特征分类（与音频系统集成，存储声纹特征）
# - 多模态融合（综合文本/人脸/声音进行人物识别）
# - 与知识图谱集成（自动创建人物实体和关系）

import logging
import json
import time
import os
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构定义
# ============================================================

class IdentitySource(Enum):
    """身份识别来源"""
    DIALOGUE = "dialogue"     # 对话中提取
    FACE = "face"             # 人脸识别
    VOICE = "voice"           # 声纹识别
    MANUAL = "manual"         # 手动标注


class ConfidenceLevel(Enum):
    """置信度等级"""
    HIGH = "high"       # > 0.8
    MEDIUM = "medium"   # 0.5 - 0.8
    LOW = "low"         # < 0.5


@dataclass
class FaceFeature:
    """人脸特征数据"""
    feature_id: str                          # 特征 ID
    encoding: Optional[List[float]] = None   # 人脸编码向量（128维 / 512维）
    bbox: Optional[List[float]] = None       # 检测框 [x1, y1, x2, y2]
    confidence: float = 0.0                  # 检测置信度
    captured_at: float = field(default_factory=time.time)
    source_frame_id: Optional[str] = None    # 来源帧 ID
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "encoding": self.encoding,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "captured_at": self.captured_at,
            "source_frame_id": self.source_frame_id,
        }


@dataclass
class VoiceFeature:
    """声纹特征数据"""
    feature_id: str                          # 特征 ID
    embedding: Optional[List[float]] = None  # 声纹嵌入向量
    duration: float = 0.0                    # 音频片段时长(秒)
    confidence: float = 0.0                  # 识别置信度
    captured_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "embedding": self.embedding,
            "duration": self.duration,
            "confidence": self.confidence,
            "captured_at": self.captured_at,
        }


@dataclass
class PersonProfile:
    """人物画像"""
    person_id: str                           # 唯一人物 ID
    name: Optional[str] = None               # 名字（可能未知）
    nickname: Optional[str] = None           # 昵称
    
    # 基础信息（从对话中提取）
    attributes: Dict[str, Any] = field(default_factory=dict)
    # 示例属性：
    #   age: "30左右"
    #   gender: "male/female/unknown"
    #   occupation: "程序员"
    #   interests: ["编程", "音乐"]
    #   preferences: {"语言风格": "正式", "话题偏好": ["技术"]}
    
    # 多模态特征
    face_features: List[FaceFeature] = field(default_factory=list)
    voice_features: List[VoiceFeature] = field(default_factory=list)
    
    # 对话特征
    dialogue_style: Dict[str, Any] = field(default_factory=dict)
    # 示例：
    #   avg_message_length: 50
    #   common_topics: ["工作", "学习"]
    #   sentiment_tendency: "positive"
    #   language: "zh-CN"
    
    # 交互历史
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    interaction_count: int = 0
    total_turns: int = 0
    
    # 识别状态
    identity_sources: List[str] = field(default_factory=list)
    overall_confidence: float = 0.0
    is_verified: bool = False     # 是否已验证身份
    
    # 与知识图谱的关联
    kg_entity_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "person_id": self.person_id,
            "name": self.name,
            "nickname": self.nickname,
            "attributes": self.attributes,
            "face_features": [f.to_dict() for f in self.face_features],
            "voice_features": [v.to_dict() for v in self.voice_features],
            "dialogue_style": self.dialogue_style,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "interaction_count": self.interaction_count,
            "total_turns": self.total_turns,
            "identity_sources": self.identity_sources,
            "overall_confidence": self.overall_confidence,
            "is_verified": self.is_verified,
            "kg_entity_id": self.kg_entity_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonProfile":
        profile = cls(
            person_id=data["person_id"],
            name=data.get("name"),
            nickname=data.get("nickname"),
            attributes=data.get("attributes", {}),
            dialogue_style=data.get("dialogue_style", {}),
            first_seen=data.get("first_seen", time.time()),
            last_seen=data.get("last_seen", time.time()),
            interaction_count=data.get("interaction_count", 0),
            total_turns=data.get("total_turns", 0),
            identity_sources=data.get("identity_sources", []),
            overall_confidence=data.get("overall_confidence", 0.0),
            is_verified=data.get("is_verified", False),
            kg_entity_id=data.get("kg_entity_id"),
        )
        return profile


# ============================================================
# 人物画像管理器
# ============================================================

class PersonProfileManager:
    """人物画像管理器
    
    TSD v2.5 对应规则:
    - 支持多模态人物识别（对话/人脸/声音）
    - 与知识图谱集成
    - 持久化存储
    - 支持人物合并和拆分
    
    架构:
    - 画像存储：内存 + JSON 持久化
    - 身份识别：基于特征向量的匹配
    - 知识图谱集成：自动创建/更新人物实体
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, persist_path: str = "./data/person_profiles"):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.profiles: Dict[str, PersonProfile] = {}
        self.persist_path = persist_path
        
        # 当前活跃人物（当前正在对话的人）
        self.active_person_id: Optional[str] = None
        
        # 人脸匹配阈值
        self.face_match_threshold = 0.6
        # 声纹匹配阈值
        self.voice_match_threshold = 0.7
        
        os.makedirs(persist_path, exist_ok=True)
        
        # 加载已有数据
        self._load()
        
        self._initialized = True
        logger.info(f"[PersonProfileManager] 初始化完成: {len(self.profiles)} 个人物画像")
    
    # ============================================================
    # 人物画像 CRUD
    # ============================================================
    
    def create_profile(self, name: Optional[str] = None,
                       source: IdentitySource = IdentitySource.DIALOGUE,
                       attributes: Optional[Dict] = None) -> PersonProfile:
        """创建新的人物画像
        
        Args:
            name: 人物名字（可选）
            source: 识别来源
            attributes: 初始属性
            
        Returns:
            新创建的人物画像
        """
        person_id = f"person_{uuid.uuid4().hex[:8]}"
        profile = PersonProfile(
            person_id=person_id,
            name=name,
            attributes=attributes or {},
            identity_sources=[source.value],
            overall_confidence=0.5 if source == IdentitySource.DIALOGUE else 0.7,
        )
        
        self.profiles[person_id] = profile
        
        # 与知识图谱集成
        self._sync_to_knowledge_graph(profile)
        
        logger.info(f"[PersonProfileManager] 创建画像: {name or '未知'} ({person_id})")
        return profile
    
    def get_profile(self, person_id: str) -> Optional[PersonProfile]:
        """获取人物画像"""
        return self.profiles.get(person_id)
    
    def find_by_name(self, name: str) -> List[PersonProfile]:
        """按名字查找人物画像"""
        results = []
        for profile in self.profiles.values():
            if profile.name and (name in profile.name or profile.name in name):
                results.append(profile)
            elif profile.nickname and (name in profile.nickname or profile.nickname in name):
                results.append(profile)
        return results
    
    def update_profile(self, person_id: str, updates: Dict[str, Any]) -> bool:
        """更新人物画像属性
        
        Args:
            person_id: 人物 ID
            updates: 更新内容（键值对）
            
        Returns:
            bool: 是否成功
        """
        profile = self.profiles.get(person_id)
        if not profile:
            return False
        
        for key, value in updates.items():
            if key == "name":
                profile.name = value
            elif key == "nickname":
                profile.nickname = value
            elif key == "attributes":
                profile.attributes.update(value)
            elif key == "dialogue_style":
                profile.dialogue_style.update(value)
        
        profile.last_seen = time.time()
        profile.updated_at = time.time() if hasattr(profile, 'updated_at') else time.time()
        
        # 同步到知识图谱
        self._sync_to_knowledge_graph(profile)
        
        return True
    
    # ============================================================
    # 多模态识别
    # ============================================================
    
    def register_face(self, person_id: str, face_feature: FaceFeature) -> bool:
        """注册人脸特征
        
        Args:
            person_id: 人物 ID
            face_feature: 人脸特征
            
        Returns:
            bool: 是否成功
        """
        profile = self.profiles.get(person_id)
        if not profile:
            return False
        
        profile.face_features.append(face_feature)
        if IdentitySource.FACE.value not in profile.identity_sources:
            profile.identity_sources.append(IdentitySource.FACE.value)
        
        # 更新总体置信度
        self._update_confidence(profile)
        
        logger.info(f"[PersonProfileManager] 注册人脸: {profile.name or person_id}")
        return True
    
    def register_voice(self, person_id: str, voice_feature: VoiceFeature) -> bool:
        """注册声纹特征
        
        Args:
            person_id: 人物 ID
            voice_feature: 声纹特征
            
        Returns:
            bool: 是否成功
        """
        profile = self.profiles.get(person_id)
        if not profile:
            return False
        
        profile.voice_features.append(voice_feature)
        if IdentitySource.VOICE.value not in profile.identity_sources:
            profile.identity_sources.append(IdentitySource.VOICE.value)
        
        self._update_confidence(profile)
        
        logger.info(f"[PersonProfileManager] 注册声纹: {profile.name or person_id}")
        return True
    
    def identify_by_face(self, face_encoding: List[float]) -> Optional[Tuple[str, float]]:
        """通过人脸编码识别人物
        
        Args:
            face_encoding: 人脸编码向量
            
        Returns:
            (person_id, similarity) 或 None
        """
        best_match = None
        best_similarity = 0.0
        
        for person_id, profile in self.profiles.items():
            for face_feat in profile.face_features:
                if face_feat.encoding:
                    similarity = self._cosine_similarity(face_encoding, face_feat.encoding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = person_id
        
        if best_match and best_similarity >= self.face_match_threshold:
            return (best_match, best_similarity)
        return None
    
    def identify_by_voice(self, voice_embedding: List[float]) -> Optional[Tuple[str, float]]:
        """通过声纹嵌入识别人物
        
        Args:
            voice_embedding: 声纹嵌入向量
            
        Returns:
            (person_id, similarity) 或 None
        """
        best_match = None
        best_similarity = 0.0
        
        for person_id, profile in self.profiles.items():
            for voice_feat in profile.voice_features:
                if voice_feat.embedding:
                    similarity = self._cosine_similarity(voice_embedding, voice_feat.embedding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = person_id
        
        if best_match and best_similarity >= self.voice_match_threshold:
            return (best_match, best_similarity)
        return None
    
    # ============================================================
    # 对话分析
    # ============================================================
    
    def update_from_dialogue(self, user_input: str, ai_response: str,
                              person_id: Optional[str] = None) -> PersonProfile:
        """从对话中更新人物画像
        
        Args:
            user_input: 用户输入
            ai_response: AI 回复
            person_id: 人物 ID（如果已知）
            
        Returns:
            更新后的人物画像
        """
        # 如果没有指定人物，使用当前活跃人物或创建新画像
        if person_id is None:
            person_id = self.active_person_id
        if person_id is None:
            profile = self.create_profile(source=IdentitySource.DIALOGUE)
            person_id = profile.person_id
            self.active_person_id = person_id
        
        profile = self.profiles.get(person_id)
        if not profile:
            return self.create_profile(source=IdentitySource.DIALOGUE)
        
        # 更新交互统计
        profile.interaction_count += 1
        profile.total_turns += 1
        profile.last_seen = time.time()
        
        # 提取对话特征
        self._extract_dialogue_features(profile, user_input, ai_response)
        
        # 提取人物信息
        self._extract_person_info(profile, user_input)
        
        return profile
    
    def _extract_dialogue_features(self, profile: PersonProfile, 
                                    user_input: str, ai_response: str):
        """从对话中提取特征"""
        # 更新平均消息长度
        total_msgs = profile.dialogue_style.get("total_messages", 0)
        avg_len = profile.dialogue_style.get("avg_message_length", 0)
        new_avg = (avg_len * total_msgs + len(user_input)) / (total_msgs + 1)
        profile.dialogue_style["avg_message_length"] = round(new_avg, 1)
        profile.dialogue_style["total_messages"] = total_msgs + 1
        
        # 更新常见话题
        topics = profile.dialogue_style.get("common_topics", [])
        topic_keywords = {
            "技术": ["代码", "编程", "开发", "bug", "系统", "程序", "算法"],
            "生活": ["吃饭", "睡觉", "天气", "周末", "放假"],
            "工作": ["工作", "上班", "会议", "项目", "加班", "老板"],
            "学习": ["学习", "课程", "考试", "作业", "论文"],
            "娱乐": ["电影", "游戏", "音乐", "旅游", "运动"],
        }
        for topic, keywords in topic_keywords.items():
            if any(kw in user_input for kw in keywords) and topic not in topics:
                topics.append(topic)
        profile.dialogue_style["common_topics"] = topics[-10:]  # 保留最近10个
        
        # 情感倾向
        positive_words = ["谢谢", "太好了", "开心", "高兴", "不错", "喜欢"]
        negative_words = ["不好", "讨厌", "生气", "难过", "糟糕"]
        pos_count = sum(1 for w in positive_words if w in user_input)
        neg_count = sum(1 for w in negative_words if w in user_input)
        if pos_count > neg_count:
            profile.dialogue_style["sentiment_tendency"] = "positive"
        elif neg_count > pos_count:
            profile.dialogue_style["sentiment_tendency"] = "negative"
    
    def _extract_person_info(self, profile: PersonProfile, user_input: str):
        """从对话中提取人物基本信息"""
        import re
        
        # 提取名字
        name_patterns = [
            r"我叫(\S{2,4})",
            r"我的名字是(\S{2,4})",
            r"叫我(\S{2,4})",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, user_input)
            if match:
                profile.name = match.group(1)
                profile.is_verified = True
        
        # 提取年龄
        age_patterns = [r"我(\d{1,3})岁", r"我今年(\d{1,3})"]
        for pattern in age_patterns:
            match = re.search(pattern, user_input)
            if match:
                profile.attributes["age"] = int(match.group(1))
        
        # 提取职业
        occupation_patterns = [
            r"我是(?:一名|一个)?(\S{2,6}(?:师|员|家|生|手|者))",
            r"我(?:在|做)(\S{2,6}(?:工作|行业))",
        ]
        for pattern in occupation_patterns:
            match = re.search(pattern, user_input)
            if match:
                profile.attributes["occupation"] = match.group(1)
        
        # 提取兴趣爱好
        interest_patterns = [
            r"我(?:喜欢|爱好|热爱|对)(\S{2,6})(?:感兴趣|很感兴趣)?",
        ]
        for pattern in interest_patterns:
            match = re.search(pattern, user_input)
            if match:
                interests = profile.attributes.get("interests", [])
                interest = match.group(1)
                if interest not in interests:
                    interests.append(interest)
                profile.attributes["interests"] = interests[-10:]
    
    # ============================================================
    # 知识图谱集成
    # ============================================================
    
    def _sync_to_knowledge_graph(self, profile: PersonProfile):
        """将人物画像同步到知识图谱"""
        try:
            from zulong.memory.knowledge_graph import (
                get_knowledge_graph, Entity, EntityType
            )
            kg = get_knowledge_graph()
            
            entity_id = profile.kg_entity_id or f"person_{profile.person_id}"
            entity = Entity(
                entity_id=entity_id,
                name=profile.name or f"未知人物_{profile.person_id[:8]}",
                entity_type=EntityType.PERSON,
                attributes={
                    "person_profile_id": profile.person_id,
                    **profile.attributes,
                },
                source="person_profile",
                confidence=profile.overall_confidence,
            )
            kg.add_entity(entity)
            profile.kg_entity_id = entity_id
            
        except Exception as e:
            logger.debug(f"[PersonProfileManager] 知识图谱同步失败(非致命): {e}")
    
    # ============================================================
    # 工具方法
    # ============================================================
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        import math
        
        if len(vec1) != len(vec2):
            return 0.0
        
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def _update_confidence(self, profile: PersonProfile):
        """更新总体置信度（多模态融合）"""
        scores = []
        
        if IdentitySource.DIALOGUE.value in profile.identity_sources:
            scores.append(0.4)
        if IdentitySource.FACE.value in profile.identity_sources:
            avg_face_conf = sum(f.confidence for f in profile.face_features) / max(len(profile.face_features), 1)
            scores.append(avg_face_conf * 0.35)
        if IdentitySource.VOICE.value in profile.identity_sources:
            avg_voice_conf = sum(v.confidence for v in profile.voice_features) / max(len(profile.voice_features), 1)
            scores.append(avg_voice_conf * 0.25)
        if profile.is_verified:
            scores.append(0.3)
        
        profile.overall_confidence = min(sum(scores), 1.0)
    
    # ============================================================
    # 持久化
    # ============================================================
    
    def save(self) -> bool:
        """保存所有画像到磁盘"""
        try:
            filepath = os.path.join(self.persist_path, "person_profiles.json")
            data = {
                "profiles": {pid: p.to_dict() for pid, p in self.profiles.items()},
                "active_person_id": self.active_person_id,
                "saved_at": time.time(),
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[PersonProfileManager] 已保存 {len(self.profiles)} 个人物画像")
            return True
        except Exception as e:
            logger.error(f"[PersonProfileManager] 保存失败: {e}")
            return False
    
    def _load(self) -> bool:
        """从磁盘加载画像"""
        try:
            filepath = os.path.join(self.persist_path, "person_profiles.json")
            if not os.path.exists(filepath):
                return False
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for pid, pdata in data.get("profiles", {}).items():
                self.profiles[pid] = PersonProfile.from_dict(pdata)
            
            self.active_person_id = data.get("active_person_id")
            
            logger.info(f"[PersonProfileManager] 已加载 {len(self.profiles)} 个人物画像")
            return True
        except Exception as e:
            logger.error(f"[PersonProfileManager] 加载失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        verified_count = sum(1 for p in self.profiles.values() if p.is_verified)
        with_face = sum(1 for p in self.profiles.values() if p.face_features)
        with_voice = sum(1 for p in self.profiles.values() if p.voice_features)
        
        return {
            "total_profiles": len(self.profiles),
            "verified_profiles": verified_count,
            "with_face_features": with_face,
            "with_voice_features": with_voice,
            "active_person_id": self.active_person_id,
        }


# ============================================================
# 全局单例
# ============================================================

_person_profile_manager: Optional[PersonProfileManager] = None


def get_person_profile_manager(persist_path: str = "./data/person_profiles") -> PersonProfileManager:
    """获取人物画像管理器单例"""
    global _person_profile_manager
    if _person_profile_manager is None:
        _person_profile_manager = PersonProfileManager(persist_path=persist_path)
    return _person_profile_manager
