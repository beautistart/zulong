# File: zulong/memory/memory_evolution.py
# 记忆自进化机制 - 实现记忆的自组织、自优化

import logging
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import time
import math

from .rag_manager import RAGManager
from .base_rag_library import RAGDocument

logger = logging.getLogger(__name__)


@dataclass
class MemoryStrength:
    """🔥 阶段 3：记忆强度模型（支持 L1/L2/L3 层级）
    
    TSD v1.7 对应规则:
    - 记忆强度随时间衰减
    - 重复访问增强强度
    - 情感影响强度
    - 🔥 记忆层级分类（L1/L2/L3）
    """
    initial_strength: float = 1.0  # 初始强度
    current_strength: float = 1.0  # 当前强度
    decay_rate: float = 0.1  # 衰减率（艾宾浩斯曲线）
    last_access_time: float = field(default_factory=time.time)  # 最后访问时间
    access_count: int = 0  # 访问次数
    emotional_weight: float = 1.0  # 情感权重（正面情感增强记忆）
    level: str = "L1"  # 🔥 记忆层级（L1/L2/L3）
    importance_level: str = "normal"  # 🔥 重要性级别（low/medium/high/must_remember）
    
    def decay(self, elapsed_hours: float) -> float:
        """🔥 阶段 3：计算遗忘曲线后的强度（支持访问频率强化）
        
        使用艾宾浩斯遗忘曲线公式：
        R = e^(-t/S)
        其中 R=记忆保留率，t=时间，S=强度系数
        
        Args:
            elapsed_hours: 经过的小时数
            
        Returns:
            float: 衰减后的强度
        """
        # 艾宾浩斯遗忘曲线
        retention = math.exp(-elapsed_hours / (self.initial_strength * 10))
        
        # 🔥 访问频率强化（对数增长）
        frequency_boost = math.log(self.access_count + 1) * 0.1
        
        # 🔥 最终强度 = 保留率 × (1 + 频率强化) × 情感权重
        # 修复：移除 adjusted_retention 中间变量，避免 retention 被平方
        self.current_strength = retention * (1 + frequency_boost) * self.emotional_weight
        
        return self.current_strength
    
    def reinforce(self, boost: float = 0.2) -> None:
        """强化记忆（重复访问）
        
        Args:
            boost: 强化系数（每次访问增加的比例）
        """
        self.access_count += 1
        self.last_access_time = time.time()
        
        # 每次访问增强初始强度
        self.initial_strength *= (1 + boost)
        self.current_strength = self.initial_strength
        
        logger.debug(f"[MemoryStrength] Reinforced: count={self.access_count}, "
                    f"strength={self.initial_strength:.2f}, level={self.level}")
    
    def should_forget(self, threshold: float = 0.1) -> bool:
        """判断是否应该遗忘
        
        Args:
            threshold: 遗忘阈值
            
        Returns:
            bool: 是否应该遗忘
        """
        return self.current_strength < threshold
    
    def should_demote(self) -> bool:
        """🔥 阶段 3：判断是否应该降级（L2→L1）
        
        Returns:
            bool: 是否应该降级
        """
        if self.level == "L2" and self.current_strength < 0.3:
            return True
        return False
    
    def should_promote(self) -> bool:
        """🔥 阶段 3：判断是否应该升级（L2→L3）
        
        Returns:
            bool: 是否应该升级
        """
        if self.level == "L2" and self.current_strength >= 0.6 and self.access_count >= 3:
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "initial_strength": self.initial_strength,
            "current_strength": self.current_strength,
            "decay_rate": self.decay_rate,
            "last_access_time": self.last_access_time,
            "access_count": self.access_count,
            "emotional_weight": self.emotional_weight,
            "level": self.level,  # 🔥 层级
            "importance_level": self.importance_level  # 🔥 重要性级别
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryStrength":
        """从字典加载"""
        return cls(
            initial_strength=data.get("initial_strength", 1.0),
            current_strength=data.get("current_strength", 1.0),
            decay_rate=data.get("decay_rate", 0.1),
            last_access_time=data.get("last_access_time", time.time()),
            access_count=data.get("access_count", 0),
            emotional_weight=data.get("emotional_weight", 1.0),
            level=data.get("level", "L1"),  # 🔥 层级
            importance_level=data.get("importance_level", "normal")  # 🔥 重要性级别
        )


class MemoryConsolidator:
    """记忆巩固器
    
    TSD v1.7 对应规则:
    - 短期记忆→长期记忆转化
    - 基于重要性和重复性
    - 睡眠时批量处理
    """
    
    def __init__(self, rag_manager: RAGManager):
        """初始化记忆巩固器
        
        Args:
            rag_manager: RAG 管理器实例
        """
        self.rag_manager = rag_manager
        
        # 巩固配置
        self.consolidation_threshold = 0.7  # 巩固阈值（强度>0.7 才巩固）
        self.min_access_count = 2  # 最少访问次数
        self.consolidation_interval_hours = 1.0  # 巩固间隔（小时）
        
        # 统计信息
        self.total_consolidations = 0
        self.last_consolidation_time = 0.0
        
        logger.info("[MemoryConsolidator] Initialized")
    
    def consolidate_memories(self, force: bool = False) -> int:
        """执行记忆巩固
        
        Args:
            force: 是否强制执行（忽略冷却时间）
            
        Returns:
            int: 巩固的记忆数量
        """
        # 检查冷却时间
        if not force:
            elapsed = (time.time() - self.last_consolidation_time) / 3600  # 小时
            if elapsed < self.consolidation_interval_hours:
                logger.debug(f"[MemoryConsolidator] Skipping (elapsed={elapsed:.2f}h)")
                return 0
        
        consolidated_count = 0
        
        # 获取记忆 RAG 库
        if "memory" not in self.rag_manager.rag_libraries:
            logger.warning("[MemoryConsolidator] Memory RAG not enabled")
            return 0
        
        memory_rag = self.rag_manager.rag_libraries["memory"]
        
        # 遍历短期记忆
        for doc_id in memory_rag.memory_time_spans.get("short_term", []):
            doc = memory_rag.documents.get(doc_id)
            if not doc:
                continue
            
            # 检查是否应该巩固
            if self._should_consolidate(doc):
                # 转为长期记忆
                doc.metadata["time_span"] = "long_term"
                doc.memorability = "must_remember"
                
                # 更新分类
                memory_rag.memory_time_spans["short_term"].remove(doc_id)
                memory_rag.memory_time_spans["long_term"].append(doc_id)
                
                consolidated_count += 1
                logger.info(f"[MemoryConsolidator] Consolidated: {doc_id}")
        
        self.total_consolidations += consolidated_count
        self.last_consolidation_time = time.time()
        
        logger.info(f"[MemoryConsolidator] Consolidated {consolidated_count} memories")
        return consolidated_count
    
    def _should_consolidate(self, doc: RAGDocument) -> bool:
        """判断记忆是否应该巩固
        
        Args:
            doc: 记忆文档
            
        Returns:
            bool: 是否应该巩固
        """
        # 必须记住的记忆直接巩固
        if doc.memorability == "must_remember":
            return True
        
        # 检查访问次数
        access_count = doc.metadata.get("access_count", 0)
        if access_count >= self.min_access_count:
            return True
        
        # 检查重要性
        if doc.importance == "must_learn":
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_consolidations": self.total_consolidations,
            "last_consolidation_time": self.last_consolidation_time,
            "consolidation_threshold": self.consolidation_threshold,
            "min_access_count": self.min_access_count
        }


class MemoryForgetter:
    """记忆遗忘器
    
    TSD v1.7 对应规则:
    - 自动清理低价值记忆
    - 基于遗忘曲线
    - 保留重要记忆
    """
    
    def __init__(self, rag_manager: RAGManager):
        """初始化记忆遗忘器
        
        Args:
            rag_manager: RAG 管理器实例
        """
        self.rag_manager = rag_manager
        
        # 遗忘配置
        self.forget_threshold = 0.1  # 遗忘阈值（强度<0.1）
        self.check_interval_hours = 6.0  # 检查间隔（小时）
        self.protected_importance = ["must_learn"]  # 保护的重要性级别
        
        # 统计信息
        self.total_forgotten = 0
        self.last_check_time = 0.0
        
        logger.info("[MemoryForgetter] Initialized")
    
    def forget_memories(self, force: bool = False) -> int:
        """执行记忆遗忘
        
        Args:
            force: 是否强制执行（忽略冷却时间）
            
        Returns:
            int: 遗忘的记忆数量
        """
        # 检查冷却时间
        if not force:
            elapsed = (time.time() - self.last_check_time) / 3600
            if elapsed < self.check_interval_hours:
                logger.debug(f"[MemoryForgetter] Skipping (elapsed={elapsed:.2f}h)")
                return 0
        
        forgotten_count = 0
        
        # 遍历所有 RAG 库
        for lib_name, lib in self.rag_manager.rag_libraries.items():
            if lib_name == "knowledge":
                # 知识库不遗忘（知识是永久的）
                continue
            
            # 获取记忆强度元数据
            for doc_id, doc in list(lib.documents.items()):
                # 检查是否受保护
                if doc.importance in self.protected_importance:
                    continue
                
                # 计算记忆强度
                strength = self._calculate_strength(doc)
                
                # 判断是否应该遗忘
                if strength < self.forget_threshold:
                    # 标记为已删除
                    doc.metadata["deleted"] = True
                    doc.metadata["delete_reason"] = "low_strength"
                    doc.metadata["delete_time"] = time.time()
                    
                    forgotten_count += 1
                    logger.info(f"[MemoryForgetter] Forgotten: {doc_id} "
                               f"(strength={strength:.2f})")
        
        self.total_forgotten += forgotten_count
        self.last_check_time = time.time()
        
        logger.info(f"[MemoryForgetter] Forgotten {forgotten_count} memories")
        return forgotten_count
    
    def _calculate_strength(self, doc: RAGDocument) -> float:
        """计算记忆强度
        
        Args:
            doc: 记忆文档
            
        Returns:
            float: 记忆强度（0-1）
        """
        # 获取存储的强度信息
        strength_data = doc.metadata.get("memory_strength", {})
        strength = MemoryStrength.from_dict(strength_data) if strength_data else MemoryStrength()
        
        # 计算经过时间
        elapsed_hours = (time.time() - doc.created_at) / 3600
        
        # 应用遗忘曲线
        current_strength = strength.decay(elapsed_hours)
        
        return current_strength
    
    def cleanup_deleted(self) -> int:
        """清理已删除的记忆（物理删除）
        
        Returns:
            int: 清理的数量
        """
        cleaned_count = 0
        
        for lib_name, lib in self.rag_manager.rag_libraries.items():
            # 找出已删除的文档
            deleted_ids = [
                doc_id for doc_id, doc in lib.documents.items()
                if doc.metadata.get("deleted", False)
            ]
            
            # 物理删除
            for doc_id in deleted_ids:
                del lib.documents[doc_id]
                cleaned_count += 1
        
        logger.info(f"[MemoryForgetter] Cleaned up {cleaned_count} deleted memories")
        return cleaned_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_forgotten": self.total_forgotten,
            "last_check_time": self.last_check_time,
            "forget_threshold": self.forget_threshold,
            "check_interval_hours": self.check_interval_hours
        }


class MemoryEvolutionEngine:
    """记忆自进化引擎
    
    TSD v1.7 对应规则:
    - 2.2.5 基础设施层：记忆自进化
    - 自动优化记忆质量
    - 模拟人类记忆机制
    
    功能:
    - 记忆强度管理
    - 巩固（短期→长期）
    - 遗忘（低价值清理）
    - 强化（重复访问）
    """
    
    def __init__(self, rag_manager: RAGManager):
        """初始化记忆自进化引擎
        
        Args:
            rag_manager: RAG 管理器实例
        """
        self.rag_manager = rag_manager
        
        # 子组件
        self.consolidator = MemoryConsolidator(rag_manager)
        self.forgetter = MemoryForgetter(rag_manager)
        
        # 记忆强度追踪
        self.memory_strengths: Dict[str, MemoryStrength] = {}
        
        # 配置
        self.auto_evolution_enabled = True  # 是否启用自动进化
        self.evolution_interval_minutes = 30.0  # 进化间隔（分钟）
        
        # 统计信息
        self.total_evolution_cycles = 0
        self.last_evolution_time = 0.0
        
        # 异步循环控制
        self._running = False
        self._evolution_task = None
        
        logger.info("[MemoryEvolutionEngine] Initialized")
    
    def track_access(self, doc_id: str, emotional_weight: float = 1.0) -> None:
        """追踪记忆访问
        
        Args:
            doc_id: 文档 ID
            emotional_weight: 情感权重（0.5-2.0）
        """
        if doc_id not in self.memory_strengths:
            self.memory_strengths[doc_id] = MemoryStrength(
                emotional_weight=emotional_weight
            )
        
        # 强化记忆
        self.memory_strengths[doc_id].reinforce()
        
        # 更新文档元数据
        if doc_id in self.rag_manager.rag_libraries.get("memory", {}).documents:
            doc = self.rag_manager.rag_libraries["memory"].documents[doc_id]
            doc.metadata["memory_strength"] = self.memory_strengths[doc_id].to_dict()
            doc.metadata["access_count"] = self.memory_strengths[doc_id].access_count
        
        logger.debug(f"[MemoryEvolutionEngine] Tracked access: {doc_id}")
    
    def evolve(self, force: bool = False) -> Dict[str, int]:
        """执行记忆进化
        
        Args:
            force: 是否强制执行
            
        Returns:
            Dict[str, int]: {巩固数，遗忘数}
        """
        if not self.auto_evolution_enabled and not force:
            logger.debug("[MemoryEvolutionEngine] Auto-evolution disabled")
            return {"consolidated": 0, "forgotten": 0}
        
        # 检查进化间隔
        if not force:
            elapsed = (time.time() - self.last_evolution_time) / 60  # 分钟
            if elapsed < self.evolution_interval_minutes:
                logger.debug(f"[MemoryEvolutionEngine] Skipping (elapsed={elapsed:.2f}min)")
                return {"consolidated": 0, "forgotten": 0}
        
        logger.info("[MemoryEvolutionEngine] Starting evolution cycle")
        
        # 1. 记忆巩固 (RAG库)
        consolidated = self.consolidator.consolidate_memories(force=force)
        
        # 1b. MemoryGraph图节点巩固: 将高访问频次但仍在短期的重要节点写入MemoryRAG
        mg_consolidated = 0
        try:
            from zulong.memory.memory_graph import get_memory_graph, Importance
            mg = get_memory_graph()
            if mg and self.rag_manager:
                memory_rag = self.rag_manager.rag_libraries.get("memory")
                if memory_rag:
                    for node_id, node in list(mg._nodes.items()):
                        if node.access_count >= 3 and not getattr(node, '_consolidated_to_rag', False):
                            imp = mg.get_importance(node_id) or Importance.NORMAL
                            content = getattr(node, 'content', '') or getattr(node, 'label', '')
                            if content and len(content) > 20:
                                try:
                                    from zulong.memory.base_rag_library import RAGDocument
                                    doc = RAGDocument(
                                        id=f"mg_{node_id}",
                                        content=content[:500],
                                        metadata={
                                            "source": "memory_graph",
                                            "node_id": node_id,
                                            "importance": str(imp.value),
                                            "access_count": node.access_count,
                                            "time_span": "long_term",
                                        },
                                    )
                                    memory_rag.add_document(doc)
                                    node._consolidated_to_rag = True
                                    mg_consolidated += 1
                                except Exception:
                                    pass
        except Exception as e:
            logger.debug(f"[MemoryEvolutionEngine] MemoryGraph巩固跳过: {e}")
        
        consolidated += mg_consolidated
        
        # 2. 记忆遗忘
        forgotten = self.forgetter.forget_memories(force=force)
        
        # 3. 清理已删除的记忆
        cleaned = self.forgetter.cleanup_deleted()
        
        self.total_evolution_cycles += 1
        self.last_evolution_time = time.time()
        
        logger.info(f"[MemoryEvolutionEngine] Evolution complete: "
                   f"consolidated={consolidated}, forgotten={forgotten}, "
                   f"cleaned={cleaned}")
        
        return {
            "consolidated": consolidated,
            "forgotten": forgotten,
            "cleaned": cleaned
        }
    
    def get_memory_health(self) -> Dict[str, Any]:
        """获取记忆健康度评估
        
        Returns:
            Dict[str, Any]: 健康度报告
        """
        total_memories = 0
        strong_memories = 0
        weak_memories = 0
        
        for lib_name, lib in self.rag_manager.rag_libraries.items():
            for doc_id, doc in lib.documents.items():
                if doc.metadata.get("deleted", False):
                    continue
                
                total_memories += 1
                
                # 计算强度
                strength = self._get_memory_strength(doc_id, doc)
                
                if strength > 0.7:
                    strong_memories += 1
                elif strength < 0.3:
                    weak_memories += 1
        
        health_score = strong_memories / max(total_memories, 1)
        
        return {
            "total_memories": total_memories,
            "strong_memories": strong_memories,
            "weak_memories": weak_memories,
            "health_score": health_score,
            "health_level": "excellent" if health_score > 0.8 else 
                           "good" if health_score > 0.6 else
                           "fair" if health_score > 0.4 else "poor"
        }
    
    def _get_memory_strength(self, doc_id: str, doc: RAGDocument) -> float:
        """获取记忆强度
        
        Args:
            doc_id: 文档 ID
            doc: 文档对象
            
        Returns:
            float: 记忆强度
        """
        # 使用缓存的强度
        if doc_id in self.memory_strengths:
            elapsed = (time.time() - self.memory_strengths[doc_id].last_access_time) / 3600
            return self.memory_strengths[doc_id].decay(elapsed)
        
        # 计算强度
        strength_data = doc.metadata.get("memory_strength", {})
        if strength_data:
            strength = MemoryStrength.from_dict(strength_data)
            elapsed = (time.time() - doc.created_at) / 3600
            return strength.decay(elapsed)
        
        # 默认强度
        return 1.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_evolution_cycles": self.total_evolution_cycles,
            "last_evolution_time": self.last_evolution_time,
            "auto_evolution_enabled": self.auto_evolution_enabled,
            "evolution_interval_minutes": self.evolution_interval_minutes,
            "consolidator": self.consolidator.get_statistics(),
            "forgetter": self.forgetter.get_statistics(),
            "tracked_memories": len(self.memory_strengths)
        }
    
    def print_status(self):
        """打印状态信息"""
        stats = self.get_statistics()
        health = self.get_memory_health()
        
        print("\n" + "=" * 60)
        print("记忆自进化引擎状态")
        print("=" * 60)
        print(f"进化周期数：{stats['total_evolution_cycles']}")
        print(f"追踪记忆数：{stats['tracked_memories']}")
        print(f"自动进化：{'启用' if stats['auto_evolution_enabled'] else '禁用'}")
        print(f"进化间隔：{stats['evolution_interval_minutes']}分钟")
        
        print(f"\n📊 记忆健康度:")
        print(f"  总记忆数：{health['total_memories']}")
        print(f"  强记忆数：{health['strong_memories']}")
        print(f"  弱记忆数：{health['weak_memories']}")
        print(f"  健康评分：{health['health_score']:.2f}")
        print(f"  健康等级：{health['health_level']}")
        
        print(f"\n🔄 巩固器:")
        cons_stats = stats['consolidator']
        print(f"  总巩固数：{cons_stats['total_consolidations']}")
        
        print(f"\n🗑️ 遗忘器:")
        forg_stats = stats['forgetter']
        print(f"  总遗忘数：{forg_stats['total_forgotten']}")
        
        print("=" * 60 + "\n")

    async def start_evolution_loop(self, interval_seconds: int = 1800):
        """启动异步进化循环 (每30分钟)"""
        if self._running:
            return
        self._running = True

        async def _loop():
            while self._running:
                await asyncio.sleep(interval_seconds)
                try:
                    result = self.evolve()
                    logger.info(
                        f"[MemoryEvolutionEngine] 进化循环完成: "
                        f"consolidated={result.get('consolidated', 0)}, "
                        f"forgotten={result.get('forgotten', 0)}"
                    )
                except Exception as e:
                    logger.error(f"[MemoryEvolutionEngine] 进化循环异常: {e}")

        self._evolution_task = asyncio.create_task(_loop())
        logger.info(f"[MemoryEvolutionEngine] 进化循环已启动 (间隔 {interval_seconds}s)")

    def stop_evolution_loop(self):
        """停止进化循环"""
        self._running = False
        if self._evolution_task and not self._evolution_task.done():
            self._evolution_task.cancel()
        logger.info("[MemoryEvolutionEngine] 进化循环已停止")


# ============================================================
# Singleton 访问器
# ============================================================

_evolution_engine: Optional[MemoryEvolutionEngine] = None


def get_evolution_engine() -> Optional[MemoryEvolutionEngine]:
    """获取全局 MemoryEvolutionEngine 单例"""
    return _evolution_engine


def set_evolution_engine(engine: MemoryEvolutionEngine):
    """设置全局 MemoryEvolutionEngine 单例"""
    global _evolution_engine
    _evolution_engine = engine
