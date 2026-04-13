# File: zulong/memory/short_term_memory.py
# 短期记忆管理 - 基于共享池的对话历史缓存 (TSD v2.5 纯异步版本)
# 对应文档：数据统一共享池化以及增强记忆共享

"""
短期记忆管理器 (TSD v2.5 纯异步版本)

核心特性:
- ✅ 纯异步接口 (async/await)
- ✅ 使用 asyncio.Lock 替代 threading.Lock
- ✅ 与 DataIngestion 完全异步集成
- ✅ 支持从共享池读取多模态上下文

功能:
- 内存中缓存最近 N 轮对话 (通过共享池索引)
- 自动从共享池读取关联的感知上下文
- 支持快速异步读写
- 自动清理过期数据

对应 TSD v2.5 共享池架构
"""

import asyncio
from typing import Dict, List, Optional, Any
import time
import logging
from pathlib import Path
import json
import math

from zulong.infrastructure.shared_memory_pool import (
    ZoneType, DataType, DataEnvelope
)
from zulong.infrastructure.data_ingestion import data_ingestion
try:
    from zulong.memory.memory_evolution import MemoryConsolidator
except ImportError:
    MemoryConsolidator = None
from zulong.memory.rag_manager import RAGManager
from zulong.memory.base_rag_library import RAGDocument

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """
    短期记忆：基于共享池的内存缓存 (TSD v2.5 纯异步版本)
    
    架构:
    - 存储分区：共享池 Memory Zone
    - 接口类型：纯异步 (async/await)
    - 锁机制：asyncio.Lock
    
    数据流:
    1. 用户输入 → await DataIngestion.ingest_text() → Raw Zone
    2. AI 回复 → await DataIngestion.ingest_text() → Raw Zone
    3. 记忆节点 → await ShortTermMemory.store() → Memory Zone
    4. 读取时 → await ShortTermMemory.get_recent() → 从 Memory Zone 获取
    """
    
    _instance = None
    _lock = None
    
    def __new__(cls, max_rounds: int = 100, ttl_seconds: int = 3600):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, max_rounds: int = 100, ttl_seconds: int = 3600):
        """初始化短期记忆（同步版本，兼容旧代码）
        
        ⚠️ 注意：此方法仅用于向后兼容，推荐改用 await get_instance()
        
        Args:
            max_rounds: 🔥 最大保留对话轮数 (默认 100 轮，动态容量管理)
            ttl_seconds: 过期时间 (秒)，默认 1 小时
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        # 🔥 TSD v2.4 动态阈值支持
        try:
            from zulong.l1b.dynamic_threshold_manager import get_dynamic_threshold_manager
            self.threshold_manager = get_dynamic_threshold_manager()
        except ImportError:
            self.threshold_manager = None
        
        # 注册紧急复盘回调
        if self.threshold_manager:
            self.threshold_manager.register_emergency_trigger_callback(self._on_emergency_trigger)
        
        # 🔥 TSD v2.4 新增：语义漂移检测
        try:
            from zulong.memory.semantic_drift_detector import get_semantic_drift_detector
            self.drift_detector = get_semantic_drift_detector()
        except ImportError:
            self.drift_detector = None
        
        # 🔥 TSD v2.4 新增：L2-BACKUP 调度器
        try:
            from zulong.l2.backup_scheduler import get_l2_backup_scheduler
            self.backup_scheduler = get_l2_backup_scheduler()
        except ImportError:
            self.backup_scheduler = None
        
        # 🔥 关键修复：不在 __init__ 中启动异步任务
        # self.backup_scheduler.start()  # 移除这行，由 bootstrap 统一启动
        
        # 注册复盘完成回调
        if self.backup_scheduler:
            self.backup_scheduler.register_completion_callback(self._on_summarization_complete)
        
        # 🔥 阶段 1：动态容量管理（解除 20 条限制）
        self.max_rounds = max_rounds if max_rounds else 100  # 🔥 基础容量 100 轮
        self.soft_limit = 50  # 🔥 软限制（开始淘汰低价值记忆）
        self.hard_limit = 200  # 🔥 硬限制（强制淘汰 LRU）
        self.ttl_seconds = ttl_seconds if ttl_seconds else 3600
        
        # 🔥 TSD v2.4 新增：Token 计数器
        self.token_counter = 0
        self.last_token_check_time = time.time()
        
        # 🔥 TSD v2.4 新增：时间衰减触发
        self.last_user_input_time = time.time()
        self.inactivity_threshold = 180  # 3 分钟无活动触发复盘
        
        # 🔥 TSD v2.4 新增：会话向量缓存（内存内语义缓存）
        from zulong.memory.embedding_manager import EmbeddingModelManager
        self.embedding_manager = EmbeddingModelManager()
        from zulong.memory.vector_cache import SessionVectorCache
        self.vector_cache = SessionVectorCache(
            embedding_manager=self.embedding_manager,
            max_cache_size=50,
            ttl_seconds=3600,
            time_decay_lambda=0.001
        )
        
        # 🔥 关键修复：使用同步方式获取共享池单例
        # 由于 SharedMemoryPool 现在禁止直接实例化，必须使用 get_instance()
        # 但在同步__init__中无法 await，所以需要特殊处理
        import asyncio
        from zulong.infrastructure.shared_memory_pool import SharedMemoryPool
        from zulong.infrastructure.data_ingestion import data_ingestion
        
        # 🔥 关键修复：检查共享池实例是否已存在
        if SharedMemoryPool._instance is not None:
            # 如果已存在，直接使用
            self.pool = SharedMemoryPool._instance
            logger.info(f"[ShortTermMemory] 使用已存在的共享池单例：{id(self.pool)}")
        else:
            # 如果不存在，警告用户应该使用异步方式
            logger.warning(
                "⚠️ [ShortTermMemory] 共享池实例尚未创建！\n"
                "建议：使用 'await ShortTermMemory.get_instance()' 而不是 'ShortTermMemory()'\n"
                "临时方案：系统会自动创建共享池实例，但可能导致数据延迟加载"
            )
            # 临时方案：在同步上下文中创建（不推荐）
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果事件循环正在运行，使用 run_coroutine_threadsafe
                    future = asyncio.run_coroutine_threadsafe(
                        SharedMemoryPool.get_instance(),
                        loop
                    )
                    self.pool = future.result(timeout=10)
                else:
                    self.pool = loop.run_until_complete(SharedMemoryPool.get_instance())
            except RuntimeError:
                # 没有事件循环，创建一个新的
                new_loop = asyncio.new_event_loop()
                self.pool = new_loop.run_until_complete(SharedMemoryPool.get_instance())
                new_loop.close()
            logger.info(f"[ShortTermMemory] 已创建共享池单例：{id(self.pool)}")
        
        # 🔥 关键：让 data_ingestion 使用同一个 pool
        data_ingestion.pool = self.pool
        
        # 记忆索引 (加速读取)
        self._turn_index: Dict[int, str] = {}  # turn_id → trace_id
        self._current_turn = 0
        
        # 🔥 关键修复：延迟初始化 asyncio.Lock
        # 不在 __init__ 中创建锁，而是在第一次使用时在正确的事件循环中创建
        self._lock: Optional[asyncio.Lock] = None
        
        # 🔥 第 1 周优化：记忆巩固配置
        self.consolidator = MemoryConsolidator(RAGManager()) if MemoryConsolidator else None
        self.consolidation_threshold = 0.5  # 🔥 降低阈值 0.7→0.5，基础分即可巩固
        self.last_consolidation_time = time.time()
        self.consolidation_interval = 3600  # 1 小时
        
        # ✅ 第 1 周优化：持久化配置
        self.persistence_enabled = True
        self.persistence_path = Path("./data/short_term_memory")
        self.persistence_path.mkdir(parents=True, exist_ok=True)
        
        # 统计信息 (在_load_index 之前初始化)
        self._stats = {
            "total_writes": 0,
            "total_reads": 0,
            "total_evictions": 0,
            "total_consolidations": 0
        }
        
        self._load_index()  # 启动时加载索引 (现在可以安全访问_stats)
        
        self._initialized = True
        
        logger.info(f"[ShortTermMemory] 初始化完成 (同步版本)")
        logger.info(f"   - 最大轮数：{max_rounds}")
        logger.info(f"   - TTL: {ttl_seconds}s")
        logger.info(f"   - 存储分区：Memory Zone")
        logger.info(f"   - 锁类型：asyncio.Lock")
        logger.info(f"   - 记忆巩固：✅ 已激活 (阈值={self.consolidation_threshold})")
        logger.info(f"   - 持久化：✅ 已启用 (路径={self.persistence_path})")
    
    @classmethod
    async def get_instance(cls, max_rounds: int = 20, ttl_seconds: int = 3600):
        """异步单例模式（推荐用法）"""
        if cls._instance is None:
            cls._instance = cls(max_rounds, ttl_seconds)
            # 🔥 关键：使用异步方式获取共享池单例
            from zulong.infrastructure.shared_memory_pool import SharedMemoryPool
            cls._instance.pool = await SharedMemoryPool.get_instance()
            # 加载索引
            await cls._instance._load_index()
            logger.info("💾 [ShortTermMemory] 已从持久化文件恢复数据")
        return cls._instance
    
    async def _get_lock(self) -> asyncio.Lock:
        """获取或创建 asyncio.Lock（确保在正确的事件循环中）"""
        try:
            # 检查锁是否已创建且绑定到当前事件循环
            if self._lock is not None:
                # 尝试获取锁的循环信息
                loop = asyncio.get_event_loop()
                # 如果锁存在，尝试使用它
                # 如果绑定到错误的循环，会抛出RuntimeError
                await asyncio.wait_for(self._lock.acquire(), timeout=0.001)
                self._lock.release()
                return self._lock
        except (RuntimeError, asyncio.TimeoutError):
            # 锁绑定到错误的循环，需要重新创建
            logger.warning("[ShortTermMemory] 检测到锁绑定到错误的事件循环，正在重新创建...")
            self._lock = None
        
        # 创建新锁
        if self._lock is None:
            self._lock = asyncio.Lock()
            logger.info("[ShortTermMemory] asyncio.Lock 已在当前事件循环中创建")
        return self._lock
    
    async def store(self, user_input: str, ai_response: str, 
                    metadata: Optional[Dict] = None) -> bool:
        """
        存储一轮对话到共享池 (异步版本)
        
        Args:
            user_input: 用户输入
            ai_response: AI 回复
            metadata: 附加元数据 (可包含感知上下文 trace_id)
            
        Returns:
            bool: 存储是否成功
        
        数据流:
        1. await 写入用户消息到 Raw Zone (通过 DataIngestion)
        2. await 写入 AI 消息到 Raw Zone (通过 DataIngestion)
        3. 构建记忆节点，写入 Memory Zone
        
        使用示例:
        ```python
        success = await short_term_memory.store(
            user_input="你好",
            ai_response="你好！有什么可以帮助你的？",
            metadata={"vision_trace_ids": [...]}
        )
        ```
        """
        try:
            # 🔥 使用异步锁（延迟初始化，确保在正确的事件循环中）
            lock = await self._get_lock()
            async with lock:
                timestamp = time.time()
                self._current_turn += 1
                turn_id = self._current_turn
                
                # 1. 🔥 await 写入用户消息到 Raw Zone
                user_trace = await data_ingestion.ingest_text(
                    text=user_input,
                    source="user",
                    timestamp=timestamp,
                    metadata={
                        "turn_id": turn_id,
                        **(metadata or {})
                    }
                )
                
                # 2. 🔥 await 写入 AI 消息到 Raw Zone
                ai_trace = await data_ingestion.ingest_text(
                    text=ai_response,
                    source="assistant",
                    timestamp=timestamp,
                    metadata={
                        "turn_id": turn_id,
                        **(metadata or {})
                    }
                )
                
                # 3. 构建记忆节点
                memory_node = {
                    "turn_id": turn_id,
                    "timestamp": timestamp,
                    "user": {
                        "trace_id": user_trace,
                        "text": user_input
                    },
                    "assistant": {
                        "trace_id": ai_trace,
                        "text": ai_response
                    },
                    "context": {
                        # 从 metadata 中提取感知上下文 trace_id
                        "vision_trace_ids": metadata.get("vision_trace_ids", []) if metadata else [],
                        "audio_trace_ids": metadata.get("audio_trace_ids", []) if metadata else [],
                        "system_state": metadata.get("system_state") if metadata else None
                    }
                }
                
                # 4. 写入 Memory Zone（异步）
                memory_trace_id = f"trace_memory_{turn_id:06d}"
                envelope = DataEnvelope(
                    trace_id=memory_trace_id,
                    timestamp=timestamp,
                    data_type=DataType.MEMORY_NODE,
                    zone=ZoneType.MEMORY,
                    payload=memory_node,
                    metadata={
                        "turn_id": turn_id,
                        "user_trace": user_trace,
                        "ai_trace": ai_trace,
                        "expires_at": timestamp + self.ttl_seconds
                    },
                    related_trace_ids=[user_trace, ai_trace]
                )
                
                # 关联视觉和听觉 trace_id
                if metadata:
                    if "vision_trace_ids" in metadata:
                        envelope.related_trace_ids.extend(metadata["vision_trace_ids"])
                    if "audio_trace_ids" in metadata:
                        envelope.related_trace_ids.extend(metadata["audio_trace_ids"])
                
                await self.pool.write(envelope)
                
                # 🔥 调试：记录 pool 实例 ID
                logger.debug(f"[ShortTermMemory] 写入 pool 实例 ID: {id(self.pool)}")
                logger.debug(f"[ShortTermMemory] Memory Zone 大小：{len(self.pool._memory_zone)}")
                
                # 5. 更新索引
                self._turn_index[turn_id] = memory_trace_id
                
                # 6. 清理过期记忆 (LRU)
                await self._evict_old_memories()
                
                # 7. ✅ 第 1 周优化：检查并执行记忆巩固
                await self._maybe_consolidate(turn_id, user_input, ai_response)
                
                # 8. 🔥 TSD v2.4 新增：延迟向量化（前 2 轮不向量化）
                # 🔥 修复：从第 3 轮开始，每增加一轮就立即向量化前一轮
                if turn_id > 2:
                    # 向量化前一轮（turn_id - 1）
                    vectorization_turn_id = turn_id - 1
                    
                    # 获取需要向量化的历史对话
                    historical_turn = await self.get_turn_by_id(vectorization_turn_id)
                    
                    if historical_turn:
                        await self.vector_cache.store_session_vector(
                            user_text=historical_turn['user']['text'],
                            bot_text=historical_turn['assistant']['text'],
                            turn_id=vectorization_turn_id
                        )
                        logger.info(
                            f"[ShortTermMemory] 🔥 延迟向量化 turn={vectorization_turn_id}, "
                            f"当前 turn={turn_id}"
                        )
                    else:
                        logger.warning(
                            f"[ShortTermMemory] ⚠️ 无法获取 turn={vectorization_turn_id} 的对话内容"
                        )
                else:
                    logger.debug(f"[ShortTermMemory] 前 2 轮不向量化 turn={turn_id}")
                
                # 9. ✅ 第 1 周优化：保存索引到磁盘
                self._save_index()
                
                self._stats["total_writes"] += 1
                
                logger.info(f"[ShortTermMemory] ✅ 存储完成：turn={turn_id}, user={user_input[:30]}...")
                logger.info(f"  - trace_id: {trace_id}")
                logger.info(f"  - 索引大小：{len(self._turn_index)} 轮")
                logger.info(f"  - 当前 turn: {self._current_turn}")
                
                return True
                
        except Exception as e:
            logger.error(f"[ShortTermMemory] 存储失败：{e}", exc_info=True)
            return False
    
    async def get_recent(self, limit: int = 5, 
                        include_context: bool = True) -> List[Dict]:
        """
        获取最近 N 轮对话 (异步版本)
        
        Args:
            limit: 对话轮数限制
            include_context: 是否包含感知上下文 (视觉、听觉)
            
        Returns:
            List[Dict]: 历史对话列表 (包含完整上下文)
        
        使用示例:
        ```python
        recent = await short_term_memory.get_recent(limit=3)
        for turn in recent:
            print(f"用户：{turn['user']['text']}")
            print(f"AI: {turn['assistant']['text']}")
        ```
        """
        try:
            # 🔥 使用异步锁
            async with self._lock:
                self._stats["total_reads"] += 1
                
                # 从索引获取最近 N 轮的 trace_id（异步）
                recent_turns = []
                start_turn = max(1, self._current_turn - limit + 1)
                
                for turn_id in range(start_turn, self._current_turn + 1):
                    trace_id = self._turn_index.get(turn_id)
                    if trace_id:
                        envelope = await self.pool.read_memory(trace_id)
                        if envelope:
                            recent_turns.append(envelope.payload)
                
                # 如果需要包含上下文，从共享池加载感知数据（异步）
                if include_context:
                    for turn in recent_turns:
                        await self._load_perception_context(turn)
                
                logger.debug(f"[ShortTermMemory] 获取 {len(recent_turns)} 轮历史对话")
                
                return recent_turns
                
        except Exception as e:
            logger.error(f"[ShortTermMemory] 读取失败：{e}", exc_info=True)
            return []
    
    async def _load_perception_context(self, turn_data: Dict):
        """
        从共享池加载感知上下文 (异步版本)
        
        Args:
            turn_data: 对话数据
        """
        context = turn_data.get("context", {})
        
        # 🔥 加载视觉上下文（异步）
        vision_trace_ids = context.get("vision_trace_ids", [])
        vision_data = []
        for vision_trace_id in vision_trace_ids:
            vision_envelope = await self.pool.read_feature(vision_trace_id)
            if vision_envelope:
                vision_data.append(vision_envelope.payload)
                logger.debug(f"  📖 加载视觉上下文：{vision_trace_id[:20]}")
        if vision_data:
            context["vision_data"] = vision_data
        
        # 🔥 加载听觉上下文（异步）
        audio_trace_ids = context.get("audio_trace_ids", [])
        audio_data = []
        for audio_trace_id in audio_trace_ids:
            audio_envelope = await self.pool.read_feature(audio_trace_id)
            if audio_envelope:
                audio_data.append(audio_envelope.payload)
                logger.debug(f"  📖 加载听觉上下文：{audio_trace_id[:20]}")
        if audio_data:
            context["audio_data"] = audio_data
        
        # 🔥 加载系统状态（异步）
        system_state = context.get("system_state")
        if system_state:
            system_envelope = await self.pool.read_system(system_state)
            if system_envelope:
                context["system_data"] = system_envelope.payload
                logger.debug(f"  📖 加载系统状态：{system_state}")
    
    async def _evict_old_memories(self):
        """🔥 阶段 1：智能淘汰机制（动态容量管理）"""
        current_size = len(self._turn_index)
        
        # 超过硬限制，强制淘汰 LRU 记忆
        if current_size > self.hard_limit:
            logger.warning(
                f"[ShortTermMemory] 🔴 记忆数量 ({current_size}) 超过硬限制 ({self.hard_limit})，"
                f"强制淘汰 LRU 记忆"
            )
            await self._evict_lru_memories(target_size=self.hard_limit)
        
        # 超过软限制，智能淘汰低价值记忆
        elif current_size > self.soft_limit:
            logger.info(
                f"[ShortTermMemory] 🟡 记忆数量 ({current_size}) 超过软限制 ({self.soft_limit})，"
                f"智能淘汰低价值记忆"
            )
            await self._evict_low_value_memories(target_size=self.soft_limit)
    
    async def _evict_lru_memories(self, target_size: int):
        """🔥 强制淘汰：LRU 策略（最近最少使用）"""
        to_delete = []
        # 按轮次排序，淘汰最旧的记忆
        sorted_turns = sorted(self._turn_index.keys())
        num_to_delete = len(sorted_turns) - target_size
        
        if num_to_delete > 0:
            to_delete = sorted_turns[:num_to_delete]
            
            # 删除旧记忆
            for turn_id in to_delete:
                trace_id = self._turn_index.pop(turn_id)
                # 从共享池删除（标记为过期）
                envelope = await self.pool.read_memory(trace_id)
                if envelope:
                    envelope.expires_at = time.time()  # 立即过期
                    await self.pool.write(envelope)
                
                self._stats["total_evictions"] += 1
            
            logger.warning(
                f"[ShortTermMemory] 🔴 强制淘汰 {len(to_delete)} 条 LRU 记忆"
            )
    
    async def _evict_low_value_memories(self, target_size: int):
        """🔥 智能淘汰：基于多维度评分（重要性 + 访问频率 + 时间）
        
        TSD v2.5: 集成 LLM 审查，在淘汰前让 LLM 做最后把关
        """
        to_delete = []
        memory_scores = []
        
        # 计算每条记忆的价值评分
        for turn_id in self._turn_index:
            score = await self._calculate_memory_value(turn_id)
            memory_scores.append((turn_id, score))
        
        # 按评分排序，淘汰低价值记忆
        memory_scores.sort(key=lambda x: x[1])  # 低分在前
        
        num_to_delete = len(memory_scores) - target_size
        if num_to_delete > 0:
            # 🔥 TSD v2.5: 提交 LLM 审查（异步，不阻塞）
            try:
                from zulong.memory.llm_memory_reviewer import get_llm_memory_reviewer
                reviewer = get_llm_memory_reviewer()
                
                # 收集待淘汰的记忆内容
                evict_candidates = []
                for turn_id, score in memory_scores[:num_to_delete]:
                    mem = await self.get_turn_by_id(turn_id, include_context=False)
                    if mem:
                        evict_candidates.append(mem)
                
                if evict_candidates:
                    usage_ratio = len(self._turn_index) / max(self.hard_limit, 1)
                    await reviewer.review_before_evict(
                        memories=evict_candidates,
                        usage_ratio=usage_ratio,
                        target_free=num_to_delete,
                    )
            except Exception as e:
                logger.debug(f"[ShortTermMemory] LLM 审查提交失败(非阻塞): {e}")
            
            # 先执行淘汰（不等待 LLM 审查结果，审查结果异步回调处理）
            to_delete = [turn_id for turn_id, score in memory_scores[:num_to_delete]]
            
            # 删除低价值记忆
            for turn_id in to_delete:
                trace_id = self._turn_index.pop(turn_id)
                # 从共享池删除（标记为过期）
                envelope = await self.pool.read_memory(trace_id)
                if envelope:
                    envelope.expires_at = time.time()  # 立即过期
                    await self.pool.write(envelope)
                
                self._stats["total_evictions"] += 1
            
            logger.info(
                f"[ShortTermMemory] 🟡 智能淘汰 {len(to_delete)} 条低价值记忆"
            )
    
    async def _calculate_memory_value(self, turn_id: int) -> float:
        """
        🔥 多维度记忆价值评分（0-1）
        
        考虑因素：
        1. 内容重要性（0-0.3）
        2. 访问频率（0-0.3）
        3. 时间新鲜度（0-0.2）
        4. 情感强度（0-0.2）
        
        Args:
            turn_id: 对话轮次 ID
            
        Returns:
            float: 记忆价值评分（0-1）
        """
        score = 0.0
        
        # 获取记忆内容
        memory = await self.get_turn_by_id(turn_id)
        if not memory:
            return 0.0
        
        user_input = memory['user']['text']
        ai_response = memory['assistant']['text']
        
        # 1. 内容重要性（0-0.3）
        importance = self._calculate_content_importance(user_input, ai_response)
        score += importance * 0.3
        
        # 2. 访问频率（0-0.3）
        # 从记忆元数据中获取真实访问次数
        access_count = memory.get('context', {}).get('access_count', 1)
        # 尝试从共享池 envelope 元数据中获取
        trace_id = self._turn_index.get(turn_id)
        if trace_id:
            envelope = await self.pool.read_memory(trace_id)
            if envelope and hasattr(envelope, 'metadata') and envelope.metadata:
                access_count = envelope.metadata.get('access_count', access_count)
        frequency_score = min(access_count / 5, 1.0)
        score += frequency_score * 0.3
        
        # 3. 时间新鲜度（0-0.2）
        # 从记忆时间戳计算真实经过时间
        memory_timestamp = memory.get('timestamp', time.time())
        elapsed_hours = (time.time() - memory_timestamp) / 3600.0
        freshness = math.exp(-elapsed_hours / 2)  # 2 小时半衰期
        score += freshness * 0.2
        
        # 4. 情感强度（0-0.2）
        emotion = self._calculate_emotion(user_input)
        score += emotion * 0.2
        
        return score
    
    def _calculate_content_importance(self, user_input: str, ai_response: str) -> float:
        """
        计算内容重要性（0-1）
        
        Args:
            user_input: 用户输入
            ai_response: AI 回复
            
        Returns:
            float: 重要性评分（0-1）
        """
        score = 0.5  # 基础分
        
        # 因素 1: 用户提问（追问表示感兴趣）
        if any(kw in user_input for kw in ["为什么", "怎么", "什么", "如何", "哪里", "谁"]):
            score += 0.15
        
        # 因素 2: 回复长度（长回复通常更重要）
        if len(ai_response) > 200:
            score += 0.1
        elif len(ai_response) > 100:
            score += 0.05
        
        # 因素 3: 包含关键信息（事实性知识）
        key_info_keywords = ["地址", "电话", "时间", "地点", "姓名", "名字", "叫", "是", "记得", "工作", "公司"]
        if any(kw in ai_response for kw in key_info_keywords):
            score += 0.2
        
        # 因素 4: 用户情感（正面情感增强记忆）
        if any(kw in user_input for kw in ["谢谢", "太好了", "非常", "特别", "很好", "不错"]):
            score += 0.15
        
        # 🔥 因素 5: 任务相关（执行类任务）
        task_keywords = ["帮我", "去做", "执行", "完成", "任务", "工作", "记住", "重要", "必须"]
        if any(kw in user_input for kw in task_keywords):
            score += 0.1
        
        return min(score, 1.0)  # 不超过 1.0
    
    def _calculate_emotion(self, user_input: str) -> float:
        """
        计算用户情感强度（0-1）
        
        Args:
            user_input: 用户输入
            
        Returns:
            float: 情感强度（0-1）
        """
        # 正面情感词
        positive_emotions = ["谢谢", "太好了", "非常", "特别", "很好", "不错", "喜欢", "开心", "高兴"]
        # 负面情感词
        negative_emotions = ["不好", "讨厌", "生气", "失望", "难过", "糟糕", "错误", "问题"]
        
        positive_count = sum(1 for kw in positive_emotions if kw in user_input)
        negative_count = sum(1 for kw in negative_emotions if kw in user_input)
        
        # 情感强度 = 正面情感 - 负面情感（归一化到 0-1）
        emotion = (positive_count - negative_count) / max(len(user_input) / 10, 1)
        return max(0.5, min(1.0, 0.5 + emotion))  # 基础 0.5，上下浮动
    
    async def get_turn_by_id(self, turn_id: int, 
                            include_context: bool = True) -> Optional[Dict]:
        """
        根据轮数获取特定对话 (异步版本，无锁优化)
        
        Args:
            turn_id: 对话轮数
            include_context: 是否包含感知上下文
            
        Returns:
            Optional[Dict]: 对话数据，如果不存在则返回 None
        """
        try:
            # 🔥 优化：移除不必要的锁，读取操作是线程安全的
            trace_id = self._turn_index.get(turn_id)
            if not trace_id:
                logger.debug(f"[get_turn_by_id] 轮次 {turn_id} 索引不存在")
                return None
            
            # 🔥 优化：先快速检查共享池中是否存在（异步，无锁）
            envelope = await self.pool.read_memory(trace_id)
            
            if not envelope:
                logger.debug(f"[get_turn_by_id] 轮次 {turn_id} 数据不存在 (trace_id={trace_id[:8]})")
                return None
            
            turn_data = envelope.payload
            
            # 加载感知上下文（异步）
            if include_context:
                await self._load_perception_context(turn_data)
            
            return turn_data
            
        except Exception as e:
            logger.error(f"[ShortTermMemory] 读取失败：{e}", exc_info=True)
            return None
    
    async def search_similar(
        self, 
        query: str, 
        top_k: int = 3,
        use_vector_cache: bool = True
    ) -> List[Dict]:
        """
        检索相似记忆（TSD v2.4 向量缓存优化版）
        
        Args:
            query: 查询文本
            top_k: 返回 Top-K 相关记忆
            use_vector_cache: 是否使用向量缓存（默认 True）
            
        Returns:
            List[Dict]: 相关记忆列表
        """
        # 🔥 TSD v2.4 新增：优先使用向量缓存检索
        if use_vector_cache and hasattr(self, 'vector_cache'):
            logger.debug(f"[ShortTermMemory] 🔍 使用向量缓存检索 query='{query[:30]}...'")
            
            results = await self.vector_cache.search(
                query_text=query,
                top_k=top_k,
                time_decay=True,
                min_score=0.3
            )
            
            if results:
                logger.info(
                    f"[ShortTermMemory] ✅ 向量缓存检索成功："
                    f"top_k={top_k}, 返回={len(results)}, "
                    f"最高分={results[0]['score']:.4f}"
                )
                return results
        
        # 降级到传统检索（基于共享池）
        logger.debug(f"[ShortTermMemory] 🔍 降级到传统检索")
        # TODO: 实现传统检索逻辑（基于共享池的语义检索）
        return []
    
    def get_vector_cache_stats(self) -> Dict:
        """
        获取向量缓存统计信息
        
        Returns:
            Dict: 统计信息
        """
        if hasattr(self, 'vector_cache'):
            return self.vector_cache.get_cache_stats()
        return {}
    
    def get_current_turn(self) -> int:
        """
        获取当前对话轮数 (同步方法，只读操作)
        
        Returns:
            int: 当前轮数
        """
        return self._current_turn
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息 (同步方法，只读操作)
        
        Returns:
            Dict: 统计信息
        """
        # 🔥 TSD v2.4 新增：获取动态阈值
        thresholds = self.threshold_manager.get_thresholds() if self.threshold_manager else None
        
        # 🔥 TSD v2.4 新增：获取向量缓存统计
        vector_cache_stats = self.get_vector_cache_stats()
        
        return {
            "current_turn": self._current_turn,
            "cached_turns": len(self._turn_index),
            "max_rounds": self.max_rounds,
            "total_writes": self._stats["total_writes"],
            "total_reads": self._stats["total_reads"],
            "total_evictions": self._stats["total_evictions"],
            "total_consolidations": self._stats.get("total_consolidations", 0),
            # 🔥 TSD v2.4 新增
            "token_counter": self.token_counter,
            "hard_token_limit": thresholds.hard_token_limit if thresholds else 0,
            "soft_turn_limit": thresholds.soft_turn_limit if thresholds else 0,
            "is_emergency_mode": thresholds.is_emergency_mode if thresholds else False,
            "vram_usage": thresholds.vram_usage if thresholds else 0,
            # 🔥 TSD v2.4 新增：向量缓存统计
            "vector_cache": vector_cache_stats
        }
    
    def _estimate_tokens(self, text: str) -> int:
        """
        🔥 TSD v2.4 新增：快速估算 Token 数
        
        Args:
            text: 输入文本
            
        Returns:
            int: 估算的 Token 数
        """
        # 简化估算：中文字符 × 1.5 + 英文字符 × 0.75
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        english_chars = sum(1 for c in text if c.isascii())
        
        return int(chinese_chars * 1.5 + english_chars * 0.75)
    
    async def _check_dynamic_thresholds(self, user_input: str, ai_response: str) -> bool:
        """
        🔥 TSD v2.4 增强版：检查动态阈值并决定是否触发复盘（集成语义漂移检测）
        
        Args:
            user_input: 用户输入
            ai_response: AI 回复
            
        Returns:
            bool: True 表示需要触发复盘
        """
        if not self.threshold_manager or not self.drift_detector or not self.backup_scheduler:
            return False

        # 1. 更新 Token 计数
        new_tokens = self._estimate_tokens(user_input + ai_response)
        self.token_counter += new_tokens
        
        # 2. 更新时间衰减
        current_time = time.time()
        time_since_last_input = current_time - self.last_user_input_time
        self.last_user_input_time = current_time
        
        # 3. 🔥 TSD v2.4 新增：语义漂移检测
        is_drift, similarity, drift_reason = await self.drift_detector.detect_drift(user_input)
        logger.info(f"🔍 [语义漂移] 相似度：{similarity:.3f}, 状态：{drift_reason}")
        
        if is_drift:
            logger.info(f"🚨 [语义漂移] 检测到话题转换，触发复盘：{drift_reason}")
            # 提交到 L2-BACKUP 队列
            conversation_turns = await self.get_recent_turns()
            await self.backup_scheduler.submit_summarization_task(
                conversation_turns=conversation_turns,
                priority=0  # 高优先级
            )
            return True
        
        # 4. 检查时间衰减触发（>3 分钟无活动）
        if time_since_last_input > self.inactivity_threshold:
            logger.info(f"⏰ [动态阈值] 检测到长时间无活动 ({time_since_last_input:.1f}秒)，触发复盘")
            return True
        
        # 5. 检查长文本输入
        if self.threshold_manager.check_long_text_input(user_input):
            logger.info("📝 [动态阈值] 检测到长文本输入，触发复盘")
            return True
        
        # 6. 使用动态阈值管理器判断
        should_trigger, reason = self.threshold_manager.should_trigger_summarization(
            current_tokens=self.token_counter,
            current_turns=len(self._turn_index)
        )
        
        if should_trigger:
            logger.info(f"🚨 [动态阈值] 触发复盘：{reason}")
            logger.info(f"  - 当前 Token: {self.token_counter} / {self.threshold_manager.hard_token_limit}")
            logger.info(f"  - 当前轮数：{len(self._turn_index)} / {self.threshold_manager.soft_turn_limit}")
            
            # 提交到 L2-BACKUP 队列
            conversation_turns = await self.get_recent_turns()
            await self.backup_scheduler.submit_summarization_task(
                conversation_turns=conversation_turns,
                priority=1  # 普通优先级
            )
            return True
        
        return False
    
    def _on_emergency_trigger(self, reason: str):
        """
        🔥 TSD v2.4 新增：紧急复盘触发回调
        
        Args:
            reason: 触发原因（如 "vram_emergency"）
        """
        logger.warning(f"🚨 [ShortTermMemory] 收到紧急触发信号：{reason}")
        # 这里可以触发立即复盘，但由于是同步回调，需要异步执行
        # 实际复盘逻辑在 _maybe_consolidate 中处理
    
    async def _on_summarization_complete(self, task):
        """
        🔥 TSD v2.4 新增：L2-BACKUP 复盘完成回调
        
        Args:
            task: 复盘任务
        """
        logger.info(f"✅ [ShortTermMemory] 复盘任务完成：{task.task_id}")
        
        if task.result:
            # 将摘要存储到情景记忆
            from zulong.memory.episodic_memory import EpisodicMemory
            episode_memory = EpisodicMemory()
            
            await episode_memory.add_episode(
                user_input=f"[复盘摘要] {task.result.get('summary', '')}",
                ai_response=f"已处理 {task.result.get('turns_count', 0)} 轮对话",
                metadata={
                    "type": "summary",
                    "task_id": task.task_id,
                    "timestamp": time.time()
                }
            )
            
            logger.info(f"  📝 [ShortTermMemory] 摘要已存储到情景记忆")
    
    async def get_recent_turns(self, limit: int = 10) -> List[Dict[str, str]]:
        """
        🔥 TSD v2.4 新增：获取最近对话轮次（用于提交给 L2-BACKUP）
        
        Args:
            limit: 最大轮次数
            
        Returns:
            List[Dict]: 对话轮次列表
        """
        turns = []
        
        # 从共享池读取最近的对话
        try:
            from zulong.infrastructure.shared_memory_pool import ZoneType, DataEnvelope
            
            # 获取最近的 Memory Zone 数据
            recent_data = await self.pool.get_recent(
                time_window_sec=3600,  # 最近 1 小时
                limit=limit * 2,  # 每轮对话包含 2 条数据（用户+AI）
                zone=ZoneType.MEMORY
            )
            
            # 解析为对话格式
            current_turn = {}
            for envelope in recent_data:
                # 🔥 修复：DataEnvelope 是数据类，不是元组
                if not isinstance(envelope, DataEnvelope):
                    logger.warning(f"[ShortTermMemory] 非 DataEnvelope 对象：{type(envelope)}")
                    continue
                
                data_type = envelope.data_type.value if hasattr(envelope.data_type, 'value') else str(envelope.data_type)
                content = envelope.payload if isinstance(envelope.payload, str) else str(envelope.payload)
                
                if 'user' in data_type.lower():
                    if current_turn:
                        turns.append(current_turn)
                    current_turn = {'user': content}
                elif 'assistant' in data_type.lower():
                    current_turn['assistant'] = content
                    if current_turn:
                        turns.append(current_turn)
                        current_turn = {}
            
            # 添加最后一个
            if current_turn and 'user' in current_turn:
                turns.append(current_turn)
            
            # 限制数量
            return turns[:limit]
            
        except Exception as e:
            logger.error(f"[ShortTermMemory] 获取最近对话失败：{e}")
            return []
    
    async def _maybe_consolidate(self, turn_id: int, user_input: str, ai_response: str):
        """🔥 阶段 2：检查并执行记忆巩固（支持 L2 半固定层）"""
        try:
            # 🔥 TSD v2.4 新增：首先检查动态阈值
            should_trigger_by_threshold = await self._check_dynamic_thresholds(user_input, ai_response)
            
            # 1. 计算重要性分数
            importance = self._calculate_importance(user_input, ai_response)
            
            logger.info(
                f"📊 [记忆巩固检查] turn={turn_id}, "
                f"重要性={importance:.2f}, "
                f"阈值={self.consolidation_threshold:.2f}"
            )
            
            # 2. 🔥 阶段 2：分级固化策略
            # 高重要性（>=0.7）→ 直接固化到 L3（长期记忆）
            if importance >= 0.7:
                logger.info(
                    f"✅ [记忆巩固] 高重要性，固化到 L3：turn_id={turn_id}"
                )
                await self._consolidate_turn(turn_id, level="L3")
            
            # 中等重要性（>=0.5）→ 固化到 L2（半固定记忆）
            elif importance >= self.consolidation_threshold:  # 0.5
                logger.info(
                    f"✅ [记忆巩固] 中等重要性，固化到 L2：turn_id={turn_id}"
                )
                await self._consolidate_turn(turn_id, level="L2")
            
            # 🔥 TSD v2.4 新增：如果动态阈值触发，固化到 L2
            elif should_trigger_by_threshold:
                logger.info(
                    f"✅ [记忆巩固] 动态阈值触发，固化到 L2：turn_id={turn_id}"
                )
                await self._consolidate_turn(turn_id, level="L2")
                # 重置 Token 计数器（复盘后释放空间）
                self.token_counter = max(0, self.token_counter - int(self.token_counter * 0.5))
                logger.info(f"  📊 [记忆巩固] Token 计数器已调整：{self.token_counter}")
            
            # 3. 定期批量巩固（每 1 小时）
            current_time = time.time()
            if current_time - self.last_consolidation_time > self.consolidation_interval:
                logger.info(
                    f"⏰ [记忆巩固] 执行定期记忆巩固...\n"
                    f"  距离上次巩固：{(current_time - self.last_consolidation_time)/3600:.2f} 小时"
                )
                count = self.consolidator.consolidate_memories(force=True)
                if count > 0:
                    logger.info(
                        f"✅ [记忆巩固] 巩固了 {count} 条记忆\n"
                        f"  累计巩固：{self._stats['total_consolidations']}"
                    )
                    self._stats["total_consolidations"] += count
                else:
                    logger.info("ℹ️ [记忆巩固] 没有需要巩固的记忆")
                self.last_consolidation_time = current_time
                
        except Exception as e:
            logger.error(f"[记忆巩固] 失败：{e}", exc_info=True)
    
    def _calculate_importance(self, user_input: str, ai_response: str) -> float:
        """
        计算对话的重要性分数 (0-1)
        
        考虑因素:
        1. 基础分：所有对话都有 0.5 分（🔥 确保普通对话也能巩固）
        2. 用户提问：+0.15
        3. 回复长度：+0.05~0.1
        4. 关键信息：+0.2
        5. 用户情感：+0.15
        6. 🔥 任务相关：+0.1
        """
        score = 0.5  # 🔥 基础分（所有对话都有机会巩固）
        
        # 因素 1: 用户提问（追问表示感兴趣）
        if any(kw in user_input for kw in ["为什么", "怎么", "什么", "如何", "哪里", "谁"]):
            score += 0.15
        
        # 因素 2: 回复长度（长回复通常更重要）
        if len(ai_response) > 200:
            score += 0.1
        elif len(ai_response) > 100:
            score += 0.05
        
        # 因素 3: 包含关键信息（事实性知识）
        key_info_keywords = ["地址", "电话", "时间", "地点", "姓名", "名字", "叫", "是", "记得", "工作", "公司"]
        if any(kw in ai_response for kw in key_info_keywords):
            score += 0.2
        
        # 因素 4: 用户情感（正面情感增强记忆）
        if any(kw in user_input for kw in ["谢谢", "太好了", "非常", "特别", "很好", "不错"]):
            score += 0.15
        
        # 🔥 因素 5: 任务相关（执行类任务）
        task_keywords = ["帮我", "去做", "执行", "完成", "任务", "工作", "记住", "重要", "必须"]
        if any(kw in user_input for kw in task_keywords):
            score += 0.1
        
        return min(score, 1.0)  # 不超过 1.0
    
    async def _consolidate_turn(self, turn_id: int, level: str = "L2"):
        """
        🔥 阶段 2：将指定轮次的对话转为长期记忆（支持 L2/L3 分级）
        
        Args:
            turn_id: 对话轮次 ID
            level: 🔥 记忆层级（"L2"=半固定，"L3"=固定）
        """
        try:
            logger.info(f"🔄 [记忆巩固] 开始巩固 turn_id={turn_id}, level={level}")
            
            # 1. 从共享池读取对话
            trace_id = self._turn_index.get(turn_id)
            if not trace_id:
                logger.warning(f"[记忆巩固] 未找到对话：turn_id={turn_id}")
                return
            
            envelope = await self.pool.read_memory(trace_id)
            if not envelope:
                logger.warning(f"[记忆巩固] 对话已过期：turn_id={turn_id}")
                return
            
            # 2. 提取对话内容
            memory_node = envelope.payload
            user_input = memory_node["user"]["text"]
            ai_response = memory_node["assistant"]["text"]
            importance = self._calculate_importance(user_input, ai_response)
            
            logger.info(
                f"📝 [记忆巩固] 对话内容:\n"
                f"  用户：{user_input[:50]}{'...' if len(user_input) > 50 else ''}\n"
                f"  AI: {ai_response[:50]}{'...' if len(ai_response) > 50 else ''}\n"
                f"  重要性：{importance:.2f}, 层级：{level}"
            )
            
            # 3. 🔥 阶段 2：根据层级选择时间跨度
            if level == "L3":
                time_span = "long_term"
                memorability = "important"
            else:  # L2
                time_span = "medium_term"
                memorability = "normal"
            
            # 4. 添加到长期记忆 RAG
            doc = RAGDocument(
                content=f"用户：{user_input}\nAI: {ai_response}",
                metadata={
                    "turn_id": turn_id,
                    "source": "consolidated_dialogue",
                    "importance": importance,
                    "timestamp": memory_node.get("timestamp", time.time()),
                    "time_span": time_span,  # 🔥 根据层级设置
                    "level": level  # 🔥 标记层级
                }
            )
            
            # 5. 🔥 阶段 2：存储到 Memory RAG（支持 L2/L3）
            self.consolidator.rag_manager.add_memory(
                content=doc.content,
                memory_type="dialogue",
                time_span=time_span,  # 🔥 动态时间跨度
                memorability=memorability
            )
            
            # 6. 更新统计
            self._stats["total_consolidations"] += 1
            
            # 🔥 7. 记录记忆库状态
            if "memory" in self.consolidator.rag_manager.rag_libraries:
                memory_rag = self.consolidator.rag_manager.rag_libraries["memory"]
                
                # 统计各层级数量
                l2_count = len(memory_rag.memory_time_spans.get("medium_term", []))
                l3_count = len(memory_rag.memory_time_spans.get("long_term", []))
                
                logger.info(
                    f"✅ [记忆巩固] 对话已固化：turn_id={turn_id}, level={level}\n"
                    f"  L2 半固定记忆：{l2_count} 条\n"
                    f"  L3 固定记忆：{l3_count} 条"
                )
            else:
                logger.error(f"[记忆巩固] Memory RAG 库不存在！")
            
            
        except Exception as e:
            logger.error(f"[记忆巩固] 巩固对话失败：{e}", exc_info=True)
    
    def _save_index(self):
        """保存索引到磁盘"""
        if not self.persistence_enabled:
            return
        
        try:
            index_data = {
                "turn_index": {str(k): v for k, v in self._turn_index.items()},
                "current_turn": self._current_turn,
                "stats": self._stats,
                "last_saved": time.time()
            }
            
            path = self.persistence_path / "index.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"✅ [持久化] 短期记忆索引已保存")
            
        except Exception as e:
            logger.error(f"[持久化] 保存索引失败：{e}")
    
    def _load_index(self):
        """从磁盘加载索引"""
        if not self.persistence_enabled:
            return
        
        try:
            path = self.persistence_path / "index.json"
            if not path.exists():
                logger.info("ℹ️ [持久化] 未找到短期记忆索引，从空索引启动")
                return
            
            with open(path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # 恢复索引
            self._turn_index = {
                int(k): v for k, v in index_data.get("turn_index", {}).items()
            }
            self._current_turn = index_data.get("current_turn", 0)
            
            # 恢复统计信息
            if "stats" in index_data:
                self._stats.update(index_data["stats"])
            
            logger.info(f"✅ [持久化] 已恢复短期记忆索引：{len(self._turn_index)} 轮")
            
        except Exception as e:
            logger.error(f"[持久化] 加载索引失败：{e}")
    
    async def clear(self):
        """
        清空所有记忆 (异步版本)
        
        使用示例:
        ```python
        await short_term_memory.clear()
        ```
        """
        try:
            async with self._lock:
                # 清空索引
                self._turn_index.clear()
                self._current_turn = 0
                
                logger.info("[ShortTermMemory] 已清空所有记忆")
                
        except Exception as e:
            logger.error(f"[ShortTermMemory] 清空失败：{e}", exc_info=True)
    
# 🔥 全局单例
_short_term_memory_instance: Optional[ShortTermMemory] = None

def get_short_term_memory() -> ShortTermMemory:
    """
    获取 ShortTermMemory 单例
    
    Returns:
        ShortTermMemory: 单例实例
    """
    global _short_term_memory_instance
    if _short_term_memory_instance is None:
        _short_term_memory_instance = ShortTermMemory()
    return _short_term_memory_instance
