# File: zulong/memory/knowledge_graph.py
# 知识图谱模块 - 实体/事件/地点关系定位 (TSD v2.5)
#
# 基于 networkx 实现轻量级知识图谱，用于：
# - 人物关系管理（家人、朋友、同事等）
# - 事件关联追踪（谁在哪里做了什么）
# - 地点空间关系（房间布局、物品位置）
# - 支持与记忆系统（RAG / ShortTermMemory）集成

import logging
import json
import time
import os
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

try:
    import networkx as nx
except ImportError:
    logger.error("networkx 未安装，请执行: pip install networkx")
    raise


# ============================================================
# 数据结构定义
# ============================================================

class EntityType(Enum):
    """实体类型"""
    PERSON = "person"         # 人物
    OBJECT = "object"         # 物品
    LOCATION = "location"     # 地点
    EVENT = "event"           # 事件
    CONCEPT = "concept"       # 抽象概念（如"生日"、"工作"）
    ORGANIZATION = "organization"  # 组织机构


class RelationType(Enum):
    """关系类型"""
    # 人物关系
    FAMILY = "family"             # 家人
    FRIEND = "friend"             # 朋友
    COLLEAGUE = "colleague"       # 同事
    ACQUAINTANCE = "acquaintance" # 认识
    
    # 空间关系
    LOCATED_AT = "located_at"     # 位于
    CONTAINS = "contains"         # 包含
    NEAR = "near"                 # 靠近
    
    # 事件关系
    PARTICIPATED = "participated"  # 参与了
    CAUSED = "caused"              # 导致了
    HAPPENED_AT = "happened_at"    # 发生在
    
    # 所属关系
    OWNS = "owns"                  # 拥有
    BELONGS_TO = "belongs_to"      # 属于
    WORKS_AT = "works_at"          # 工作于
    
    # 通用关系
    RELATED_TO = "related_to"      # 相关


@dataclass
class Entity:
    """知识图谱实体节点"""
    entity_id: str                  # 唯一标识
    name: str                       # 名称
    entity_type: EntityType         # 实体类型
    attributes: Dict[str, Any] = field(default_factory=dict)  # 属性
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    source: str = "dialogue"        # 来源（dialogue/vision/manual）
    confidence: float = 1.0         # 置信度 (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "attributes": self.attributes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source": self.source,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(
            entity_id=data["entity_id"],
            name=data["name"],
            entity_type=EntityType(data["entity_type"]),
            attributes=data.get("attributes", {}),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            source=data.get("source", "dialogue"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class Relation:
    """知识图谱关系边"""
    source_id: str                  # 源实体 ID
    target_id: str                  # 目标实体 ID
    relation_type: RelationType     # 关系类型
    attributes: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0             # 关系强度 (0-1)
    created_at: float = field(default_factory=time.time)
    source: str = "dialogue"        # 来源
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "attributes": self.attributes,
            "weight": self.weight,
            "created_at": self.created_at,
            "source": self.source,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            attributes=data.get("attributes", {}),
            weight=data.get("weight", 1.0),
            created_at=data.get("created_at", time.time()),
            source=data.get("source", "dialogue"),
        )


# ============================================================
# 知识图谱核心引擎
# ============================================================

class KnowledgeGraph:
    """知识图谱管理器
    
    TSD v2.5 对应规则:
    - 支持实体/事件/地点关系定位
    - 与记忆系统集成（通过对话自动提取实体和关系）
    - 支持关系查询和路径发现
    - 持久化到 JSON 文件
    
    架构:
    - 底层：networkx.DiGraph（有向图）
    - 存储：JSON 文件持久化
    - 查询：支持实体检索、关系查询、路径发现、子图提取
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, persist_path: str = "./data/knowledge_graph"):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.graph = nx.DiGraph()
        self.entities: Dict[str, Entity] = {}
        self.persist_path = persist_path
        
        # 确保持久化目录存在
        os.makedirs(persist_path, exist_ok=True)
        
        # 统计信息
        self._stats = {
            "total_entities": 0,
            "total_relations": 0,
            "total_queries": 0,
        }
        
        # 尝试加载已有数据
        self._load()
        
        self._initialized = True
        logger.info(
            f"[KnowledgeGraph] 初始化完成: "
            f"{self._stats['total_entities']} 实体, "
            f"{self._stats['total_relations']} 关系"
        )
    
    # ============================================================
    # 实体操作
    # ============================================================
    
    def add_entity(self, entity: Entity) -> str:
        """添加实体节点
        
        如果实体已存在，则更新属性（合并而非覆盖）
        
        Args:
            entity: 实体对象
            
        Returns:
            str: 实体 ID
        """
        if entity.entity_id in self.entities:
            # 更新已有实体
            existing = self.entities[entity.entity_id]
            existing.attributes.update(entity.attributes)
            existing.updated_at = time.time()
            existing.confidence = max(existing.confidence, entity.confidence)
            # 更新图节点属性
            self.graph.nodes[entity.entity_id].update(entity.to_dict())
            logger.debug(f"[KnowledgeGraph] 更新实体: {entity.name} ({entity.entity_id})")
        else:
            # 添加新实体
            self.entities[entity.entity_id] = entity
            self.graph.add_node(entity.entity_id, **entity.to_dict())
            self._stats["total_entities"] += 1
            logger.info(f"[KnowledgeGraph] 添加实体: {entity.name} ({entity.entity_type.value})")
        
        return entity.entity_id
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        return self.entities.get(entity_id)
    
    def find_entities_by_name(self, name: str, fuzzy: bool = True) -> List[Entity]:
        """按名称查找实体
        
        Args:
            name: 实体名称
            fuzzy: 是否模糊匹配（包含关系）
            
        Returns:
            匹配的实体列表
        """
        results = []
        for entity in self.entities.values():
            if fuzzy:
                if name in entity.name or entity.name in name:
                    results.append(entity)
            else:
                if entity.name == name:
                    results.append(entity)
        return results
    
    def find_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """按类型查找实体"""
        return [e for e in self.entities.values() if e.entity_type == entity_type]
    
    def remove_entity(self, entity_id: str) -> bool:
        """移除实体及其所有关系"""
        if entity_id not in self.entities:
            return False
        
        del self.entities[entity_id]
        self.graph.remove_node(entity_id)
        self._stats["total_entities"] -= 1
        return True
    
    # ============================================================
    # 关系操作
    # ============================================================
    
    def add_relation(self, relation: Relation) -> bool:
        """添加关系边
        
        Args:
            relation: 关系对象
            
        Returns:
            bool: 是否成功
        """
        if relation.source_id not in self.entities:
            logger.warning(f"[KnowledgeGraph] 源实体不存在: {relation.source_id}")
            return False
        if relation.target_id not in self.entities:
            logger.warning(f"[KnowledgeGraph] 目标实体不存在: {relation.target_id}")
            return False
        
        # 检查是否已存在相同关系
        if self.graph.has_edge(relation.source_id, relation.target_id):
            existing_data = self.graph.edges[relation.source_id, relation.target_id]
            existing_type = existing_data.get("relation_type")
            if existing_type == relation.relation_type.value:
                # 更新权重（取较大值）
                self.graph.edges[relation.source_id, relation.target_id]["weight"] = max(
                    existing_data.get("weight", 1.0), relation.weight
                )
                return True
        
        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            **relation.to_dict()
        )
        self._stats["total_relations"] += 1
        
        source_name = self.entities[relation.source_id].name
        target_name = self.entities[relation.target_id].name
        logger.info(
            f"[KnowledgeGraph] 添加关系: {source_name} --[{relation.relation_type.value}]--> {target_name}"
        )
        return True
    
    def get_relations(self, entity_id: str, 
                      direction: str = "both",
                      relation_type: Optional[RelationType] = None) -> List[Dict]:
        """获取实体的关系
        
        Args:
            entity_id: 实体 ID
            direction: "outgoing"(出边) / "incoming"(入边) / "both"(双向)
            relation_type: 过滤特定关系类型
            
        Returns:
            关系列表
        """
        self._stats["total_queries"] += 1
        relations = []
        
        if direction in ("outgoing", "both"):
            for _, target, data in self.graph.out_edges(entity_id, data=True):
                if relation_type and data.get("relation_type") != relation_type.value:
                    continue
                relations.append({
                    "direction": "outgoing",
                    "source": entity_id,
                    "target": target,
                    "source_name": self.entities.get(entity_id, Entity(entity_id, "?", EntityType.CONCEPT)).name,
                    "target_name": self.entities.get(target, Entity(target, "?", EntityType.CONCEPT)).name,
                    **data
                })
        
        if direction in ("incoming", "both"):
            for source, _, data in self.graph.in_edges(entity_id, data=True):
                if relation_type and data.get("relation_type") != relation_type.value:
                    continue
                relations.append({
                    "direction": "incoming",
                    "source": source,
                    "target": entity_id,
                    "source_name": self.entities.get(source, Entity(source, "?", EntityType.CONCEPT)).name,
                    "target_name": self.entities.get(entity_id, Entity(entity_id, "?", EntityType.CONCEPT)).name,
                    **data
                })
        
        return relations
    
    # ============================================================
    # 图谱查询
    # ============================================================
    
    def find_shortest_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """查找两个实体之间的最短路径
        
        Args:
            source_id: 起始实体 ID
            target_id: 目标实体 ID
            
        Returns:
            路径上的实体 ID 列表，无路径返回 None
        """
        try:
            # 转换为无向图查找路径（关系可以双向遍历）
            undirected = self.graph.to_undirected()
            path = nx.shortest_path(undirected, source_id, target_id)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def get_neighbors(self, entity_id: str, depth: int = 1) -> Dict[str, Entity]:
        """获取实体的邻居节点（N 度关系）
        
        Args:
            entity_id: 实体 ID
            depth: 关系深度（1=直接关系, 2=间接关系...）
            
        Returns:
            邻居实体字典 {entity_id: Entity}
        """
        if entity_id not in self.graph:
            return {}
        
        neighbors = {}
        undirected = self.graph.to_undirected()
        
        # BFS 遍历到指定深度
        visited: Set[str] = {entity_id}
        current_level = {entity_id}
        
        for _ in range(depth):
            next_level: Set[str] = set()
            for node in current_level:
                for neighbor in undirected.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
                        if neighbor in self.entities:
                            neighbors[neighbor] = self.entities[neighbor]
            current_level = next_level
        
        return neighbors
    
    def get_subgraph(self, entity_id: str, depth: int = 2) -> Dict[str, Any]:
        """提取以某实体为中心的子图（用于上下文注入）
        
        Args:
            entity_id: 中心实体 ID
            depth: 扩展深度
            
        Returns:
            子图数据 {"entities": [...], "relations": [...], "summary": "..."}
        """
        self._stats["total_queries"] += 1
        
        neighbors = self.get_neighbors(entity_id, depth)
        all_ids = set(neighbors.keys()) | {entity_id}
        
        entities_data = []
        relations_data = []
        
        for eid in all_ids:
            entity = self.entities.get(eid)
            if entity:
                entities_data.append(entity.to_dict())
        
        for u, v, data in self.graph.edges(data=True):
            if u in all_ids and v in all_ids:
                relations_data.append(data)
        
        # 生成可读摘要
        summary_parts = []
        center = self.entities.get(entity_id)
        if center:
            summary_parts.append(f"关于「{center.name}」的关系网络：")
            for rel in self.get_relations(entity_id):
                if rel["direction"] == "outgoing":
                    summary_parts.append(
                        f"  - {rel['source_name']} --[{rel.get('relation_type', '?')}]--> {rel['target_name']}"
                    )
                else:
                    summary_parts.append(
                        f"  - {rel['source_name']} --[{rel.get('relation_type', '?')}]--> {rel['target_name']}"
                    )
        
        return {
            "center": entity_id,
            "entities": entities_data,
            "relations": relations_data,
            "summary": "\n".join(summary_parts),
        }
    
    def query_by_text(self, text: str) -> List[Dict[str, Any]]:
        """基于文本查询相关实体和关系
        
        简单关键词匹配方式，用于从对话中定位图谱信息
        
        Args:
            text: 查询文本
            
        Returns:
            匹配的实体及其关系
        """
        self._stats["total_queries"] += 1
        results = []
        
        for entity in self.entities.values():
            # 名称匹配
            if entity.name in text:
                subgraph = self.get_subgraph(entity.entity_id, depth=1)
                results.append({
                    "entity": entity.to_dict(),
                    "match_type": "name",
                    "relations": self.get_relations(entity.entity_id),
                    "subgraph_summary": subgraph["summary"],
                })
            # 属性匹配
            else:
                for attr_val in entity.attributes.values():
                    if isinstance(attr_val, str) and attr_val in text:
                        results.append({
                            "entity": entity.to_dict(),
                            "match_type": "attribute",
                            "relations": self.get_relations(entity.entity_id),
                        })
                        break
        
        return results
    
    # ============================================================
    # 从对话中提取实体和关系（规则引擎）
    # ============================================================
    
    def extract_from_dialogue(self, user_input: str, ai_response: str, 
                               turn_id: int = 0) -> Dict[str, Any]:
        """从对话中提取实体和关系（基于规则引擎）
        
        使用关键词匹配和模式识别提取实体和关系。
        
        Args:
            user_input: 用户输入
            ai_response: AI 回复
            turn_id: 对话轮次
            
        Returns:
            提取结果 {"entities": [...], "relations": [...]}
        """
        extracted_entities = []
        extracted_relations = []
        combined_text = f"{user_input} {ai_response}"
        
        # 人物关系模式
        person_patterns = {
            "family": ["爸爸", "妈妈", "父亲", "母亲", "哥哥", "姐姐", "弟弟", "妹妹", 
                       "儿子", "女儿", "爷爷", "奶奶", "外公", "外婆", "老公", "老婆",
                       "丈夫", "妻子", "叔叔", "阿姨", "舅舅"],
            "friend": ["朋友", "好友", "闺蜜", "哥们", "兄弟"],
            "colleague": ["同事", "同学", "老板", "领导", "经理", "老师"],
        }
        
        # 地点关键词
        location_keywords = [
            "家", "公司", "学校", "医院", "超市", "商场", "公园", "餐厅",
            "办公室", "客厅", "卧室", "厨房", "卫生间", "阳台", "车库",
        ]
        
        # 提取人物实体
        for rel_type, keywords in person_patterns.items():
            for kw in keywords:
                if kw in combined_text:
                    # 尝试提取名字（关键词前后的词汇）
                    entity = Entity(
                        entity_id=f"person_{kw}_{turn_id}",
                        name=kw,
                        entity_type=EntityType.PERSON,
                        attributes={"role": kw, "mentioned_turn": turn_id},
                        source="dialogue",
                        confidence=0.7,
                    )
                    extracted_entities.append(entity)
                    self.add_entity(entity)
                    
                    # 添加与"用户"的关系
                    user_entity = self._ensure_user_entity()
                    rel = Relation(
                        source_id=user_entity.entity_id,
                        target_id=entity.entity_id,
                        relation_type=RelationType.FAMILY if rel_type == "family" 
                            else RelationType.FRIEND if rel_type == "friend"
                            else RelationType.COLLEAGUE,
                        attributes={"mentioned_turn": turn_id},
                        source="dialogue",
                    )
                    self.add_relation(rel)
                    extracted_relations.append(rel)
        
        # 提取地点实体
        for loc_kw in location_keywords:
            if loc_kw in combined_text:
                entity = Entity(
                    entity_id=f"location_{loc_kw}",
                    name=loc_kw,
                    entity_type=EntityType.LOCATION,
                    attributes={"mentioned_turn": turn_id},
                    source="dialogue",
                    confidence=0.6,
                )
                extracted_entities.append(entity)
                self.add_entity(entity)
        
        # 提取"我叫XX"/"我的名字是XX"模式
        import re
        name_patterns = [
            r"我叫(\S{2,4})",
            r"我的名字是(\S{2,4})",
            r"我是(\S{2,4})",
            r"叫我(\S{2,4})",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, user_input)
            if match:
                name = match.group(1)
                user_entity = self._ensure_user_entity()
                user_entity.name = name
                user_entity.attributes["real_name"] = name
                user_entity.updated_at = time.time()
                self.graph.nodes[user_entity.entity_id].update(user_entity.to_dict())
        
        if extracted_entities:
            logger.info(
                f"[KnowledgeGraph] 从对话提取: "
                f"{len(extracted_entities)} 实体, {len(extracted_relations)} 关系"
            )
        
        return {
            "entities": [e.to_dict() for e in extracted_entities],
            "relations": [r.to_dict() for r in extracted_relations],
        }
    
    def _ensure_user_entity(self) -> Entity:
        """确保用户实体存在"""
        user_id = "user_primary"
        if user_id not in self.entities:
            user_entity = Entity(
                entity_id=user_id,
                name="用户",
                entity_type=EntityType.PERSON,
                attributes={"role": "primary_user"},
                source="system",
                confidence=1.0,
            )
            self.add_entity(user_entity)
        return self.entities[user_id]
    
    # ============================================================
    # 持久化
    # ============================================================
    
    def save(self) -> bool:
        """保存知识图谱到磁盘"""
        try:
            filepath = os.path.join(self.persist_path, "knowledge_graph.json")
            data = {
                "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
                "edges": [
                    {"source": u, "target": v, **d}
                    for u, v, d in self.graph.edges(data=True)
                ],
                "stats": self._stats,
                "saved_at": time.time(),
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[KnowledgeGraph] 已保存: {self._stats['total_entities']} 实体, "
                        f"{self._stats['total_relations']} 关系")
            return True
        except Exception as e:
            logger.error(f"[KnowledgeGraph] 保存失败: {e}")
            return False
    
    def _load(self) -> bool:
        """从磁盘加载知识图谱"""
        try:
            filepath = os.path.join(self.persist_path, "knowledge_graph.json")
            if not os.path.exists(filepath):
                logger.info("[KnowledgeGraph] 无已有数据，从空图谱开始")
                return False
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 恢复实体
            for eid, edata in data.get("entities", {}).items():
                entity = Entity.from_dict(edata)
                self.entities[eid] = entity
                self.graph.add_node(eid, **entity.to_dict())
            
            # 恢复关系边
            for edge in data.get("edges", []):
                source = edge.pop("source")
                target = edge.pop("target")
                self.graph.add_edge(source, target, **edge)
            
            # 恢复统计
            self._stats = data.get("stats", self._stats)
            
            logger.info(
                f"[KnowledgeGraph] 已加载: "
                f"{len(self.entities)} 实体, {self.graph.number_of_edges()} 关系"
            )
            return True
        except Exception as e:
            logger.error(f"[KnowledgeGraph] 加载失败: {e}")
            return False
    
    # ============================================================
    # 统计与调试
    # ============================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        type_counts = {}
        for entity in self.entities.values():
            t = entity.entity_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        
        return {
            "total_entities": len(self.entities),
            "total_relations": self.graph.number_of_edges(),
            "total_queries": self._stats["total_queries"],
            "entity_types": type_counts,
            "graph_density": nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0,
        }


# ============================================================
# 全局单例
# ============================================================

_knowledge_graph_instance: Optional[KnowledgeGraph] = None


def get_knowledge_graph(persist_path: str = "./data/knowledge_graph") -> KnowledgeGraph:
    """获取知识图谱单例"""
    global _knowledge_graph_instance
    if _knowledge_graph_instance is None:
        _knowledge_graph_instance = KnowledgeGraph(persist_path=persist_path)
    return _knowledge_graph_instance
