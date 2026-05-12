# File: zulong/memory/episodic_memory.py
# 临时记忆管理 - 支持摘要、检索和分级读取

"""
临时记忆 (Episodic Memory)

核心功能:
1. **对话摘要**：为每轮对话生成简短摘要（50-100 字）
2. **基于摘要检索**：快速找到相关对话
3. **分级读取**：
   - Level 1: 摘要（快速浏览）
   - Level 2: 完整对话（按需读取）
4. **时间窗口管理**：支持按时间范围检索

架构设计:
- 存储：SharedMemoryPool (Memory Zone)
- 摘要：自动生成，存储在元数据中
- 检索：基于摘要的语义相似度
- 读取：支持按需读取完整内容
"""

import asyncio
from typing import Dict, List, Optional, Any
import time
import logging
from pathlib import Path

from zulong.infrastructure.shared_memory_pool import (
    ZoneType, DataType, DataEnvelope
)
from zulong.infrastructure.data_ingestion import data_ingestion
from zulong.models.container import ModelContainer
from zulong.models.config import ModelID

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """
    临时记忆管理器
    
    数据流:
    1. 新对话 → 生成摘要 → 存储到共享池
    2. 检索请求 → 基于摘要检索 → 返回 Top-K 摘要
    3. 详细读取 → 根据摘要中的 trace_id → 读取完整对话
    """
    
    _instance = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化临时记忆管理器"""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.pool = None
        self.summary_model = None
        
        # 记忆索引
        self._episode_index: Dict[int, Dict] = {}  # episode_id → metadata (包含摘要)
        self._current_episode = 0
        
        # 🔥 TSD v2.4 动态阈值支持
        from zulong.l1b.dynamic_threshold_manager import get_dynamic_threshold_manager
        self.threshold_manager = get_dynamic_threshold_manager()
        
        # 🔥 第 1 周优化：动态容量管理（替代写死的 max_episodes=50）
        # 配置将在 initialize_async 中根据模型上下文窗口动态计算
        self.max_episodes = 50  # 初始值，会被动态更新
        self.max_tokens_reserved = 0  # 动态计算的 token 预算
        self.estimated_average_turn_tokens = 150  # 估算每轮对话的 token 数
        
        self.summary_max_length = 100  # 摘要最大长度
        self.ttl_seconds = 7200  # 2 小时 TTL
        
        # 🔥 第 1 周优化：异步复盘队列（修复：使用异步队列）
        self._pending_summarization_queue: asyncio.Queue = asyncio.Queue()
        self._summarization_tasks: list = []
        self._num_workers = 2  # 🔥 增加工作者数量，提升处理速度
        
        # 统计信息
        self._stats = {
            "total_writes": 0,
            "total_reads": 0,
            "total_searches": 0,
            "avg_search_latency": 0.0,
            "summarizations_pending": 0,
            "summarizations_completed": 0
        }
        
        # 🔥 关键修复：同步获取共享池（确保初始化完成）
        initialization_success = False
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 事件循环正在运行，同步等待异步初始化
                logger.info("[EpisodicMemory] 事件循环运行中，同步等待初始化...")
                # 使用 asyncio.run_coroutine_threadsafe 在主线程中等待
                future = asyncio.run_coroutine_threadsafe(self.initialize_async(), loop)
                future.result(timeout=60)  # 🔥 增加超时时间：15 秒 -> 60 秒
                logger.info("[EpisodicMemory] 同步初始化完成")
                initialization_success = True
            else:
                # 事件循环未运行，同步初始化
                loop.run_until_complete(self.initialize_async())
                initialization_success = True
        except TimeoutError:
            logger.error("[EpisodicMemory] 初始化超时（60 秒），启动后台重试...")
            self._init_status = "RETRYING"
            self._start_init_retry_loop()
        except RuntimeError:
            # 没有事件循环，创建一个新的并同步初始化
            logger.info("[EpisodicMemory] 创建新事件循环并同步初始化...")
            new_loop = asyncio.new_event_loop()
            try:
                new_loop.run_until_complete(self.initialize_async())
                initialization_success = True
            except Exception as e:
                logger.error(f"[EpisodicMemory] 初始化失败：{e}")
            finally:
                new_loop.close()
        except Exception as e:
            logger.error(f"[EpisodicMemory] 初始化异常：{e}")
        
        if initialization_success:
            self._initialized = True
            self._init_status = "SUCCESS"
            logger.info("[EpisodicMemory] 初始化完成（动态容量 + 异步复盘）")
        else:
            if not getattr(self, '_init_status', None) == "RETRYING":
                self._init_status = "LAZY_LOADING"
                logger.warning("[EpisodicMemory] 初始化未完成，将在首次使用时懒加载")

    def _start_init_retry_loop(self):
        """初始化超时后启动后台异步重试"""
        import threading
        def _retry_worker():
            for attempt in range(3):
                import time as _t
                _t.sleep(10)
                try:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self.initialize_async())
                    loop.close()
                    self._initialized = True
                    self._init_status = "SUCCESS"
                    logger.info(f"[EpisodicMemory] 后台重试初始化成功（第{attempt+1}次）")
                    return
                except Exception as e:
                    logger.warning(f"[EpisodicMemory] 后台重试初始化失败（第{attempt+1}次）: {e}")
            self._init_status = "LAZY_LOADING"
            logger.error("[EpisodicMemory] 后台重试全部失败，降级为懒加载")

        t = threading.Thread(target=_retry_worker, daemon=True)
        t.start()

    async def _ensure_initialized(self):
        """懒加载兜底：若未初始化则执行完整初始化"""
        if not self._initialized and self.pool is None:
            try:
                await self.initialize_async()
                self._initialized = True
                self._init_status = "SUCCESS"
            except Exception as e:
                logger.warning(f"[EpisodicMemory] 懒加载初始化失败: {e}")
    
    async def wait_for_initialization(self, timeout: float = 10.0) -> bool:
        """等待异步初始化完成"""
        start_time = time.time()
        while self.pool is None:
            if time.time() - start_time > timeout:
                logger.error("[EpisodicMemory] 等待初始化超时")
                return False
            await asyncio.sleep(0.1)
        return True
    
    @classmethod
    async def get_instance(cls):
        """异步单例模式（推荐用法）"""
        if cls._instance is None:
            cls._instance = cls()
            # 等待异步初始化完成
            await cls._instance.initialize_async()
        return cls._instance
    
    async def initialize_async(self):
        """异步初始化（推荐）"""
        from zulong.infrastructure.shared_memory_pool import SharedMemoryPool
        
        self.pool = await SharedMemoryPool.get_instance()
        logger.info(f"[EpisodicMemory] 已获取共享池单例：{id(self.pool)}")
        
        # 🔥 使用 L1_SCHEDULER 作为摘要模型（如果可用）
        try:
            container = ModelContainer()
            # 暂时不加载专门的摘要模型，使用规则生成摘要
            # self.summary_model = container.get_model(ModelID.L1B_ATTENTION)
            logger.info("[EpisodicMemory] 摘要模型：使用规则生成（轻量级方案）")
        except Exception as e:
            logger.warning(f"[EpisodicMemory] 摘要模型加载失败，使用规则生成：{e}")
        
        # 🔥 第 1 周优化：动态容量计算
        await self._calculate_dynamic_capacity()
        
        # 加载索引
        await self._load_index()
        
        # 🔥 第 1 周优化：启动异步复盘任务
        asyncio.create_task(self._start_summarization_worker())
        
        logger.info("[EpisodicMemory] 异步初始化完成")
        logger.info(f"  - 动态容量：max_episodes={self.max_episodes}, max_tokens_reserved={self.max_tokens_reserved}")
    
    async def _calculate_dynamic_capacity(self):
        """
        🔥 TSD v2.4 优化：根据模型上下文窗口动态计算记忆容量
        
        策略:
        - 使用 DynamicThresholdManager 统一计算
        - 预留 75% 的上下文用于记忆注入
        - 根据估算的每轮对话 token 数计算最大轮次
        - 适配 4k/8k/128k 不同规格的模型
        """
        try:
            # 🔥 TSD v2.4: 从动态阈值管理器获取配置
            thresholds = self.threshold_manager.get_thresholds()
            
            # 使用硬上限作为记忆 Token 预算
            self.max_tokens_reserved = thresholds.hard_token_limit
            
            # 动态计算最大轮次（使用软上限）
            self.max_episodes = max(
                10,  # 最小保留 10 轮
                thresholds.soft_turn_limit
            )
            
            # 限制最大值（避免内存爆炸）
            self.max_episodes = min(self.max_episodes, 200)
            
            logger.info(f"[EpisodicMemory] 动态容量计算完成（TSD v2.4）:")
            logger.info(f"  - 当前模型：{thresholds.current_model}")
            logger.info(f"  - 安全系数：{thresholds.safety_factor}")
            logger.info(f"  - 速度因子：{thresholds.speed_factor}")
            logger.info(f"  - 记忆 Token 预算：{self.max_tokens_reserved}")
            logger.info(f"  - 最大记忆轮次：{self.max_episodes}")
            
            if thresholds.is_emergency_mode:
                logger.warning(f"🚨 [EpisodicMemory] 当前处于紧急模式！显存使用率：{thresholds.vram_usage*100:.1f}%")
            
        except Exception as e:
            logger.error(f"[EpisodicMemory] 动态容量计算失败：{e}")
            # 降级到默认值
            self.max_episodes = 50
            self.max_tokens_reserved = 3072
    
    async def _start_summarization_worker(self):
        """
        🔥 第 1 周优化：启动多个异步复盘工作者
        
        后台持续监听待摘要队列，利用 L2-BACKUP 空闲资源批量处理
        """
        for i in range(self._num_workers):
            task = asyncio.create_task(self._summarization_worker(i))
            self._summarization_tasks.append(task)
            logger.info(f"[EpisodicMemory] 复盘工作者 #{i} 已启动")
    
    async def _summarization_worker(self, worker_id: int = 0):
        """复盘工作者协程（支持多个工作者）"""
        logger.info(f"[EpisodicMemory] 复盘工作者 #{worker_id} 启动")
        
        while True:
            try:
                # 从队列中获取任务
                episode_data = await self._pending_summarization_queue.get()
                
                if episode_data is None:
                    # 退出信号
                    logger.info(f"[EpisodicMemory] 复盘工作者 #{worker_id} 收到退出信号")
                    break
                
                # 生成摘要
                summary = await self._generate_semantic_summary(
                    episode_data.get('user_input'),
                    episode_data.get('ai_response')
                )
                
                # 更新索引
                episode_id = episode_data.get('episode_id')
                if episode_id in self._episode_index:
                    self._episode_index[episode_id]['summary'] = summary
                    self._episode_index[episode_id]['summary_type'] = 'semantic'
                    
                    logger.info(f"[EpisodicMemory] 复盘完成：Episode {episode_id} (Worker #{worker_id})")
                    self._stats["summarizations_completed"] += 1
                
                # 标记任务完成
                self._pending_summarization_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info(f"[EpisodicMemory] 复盘工作者 #{worker_id} 已停止")
                break
            except Exception as e:
                logger.error(f"[EpisodicMemory] 复盘工作者 #{worker_id} 错误：{e}", exc_info=True)
    
    async def _load_index(self):
        """加载索引"""
        try:
            # 🔥 修复：直接访问共享池的 memory_zone 内部数据
            # 使用 threading.Lock 安全访问
            import threading
            from zulong.infrastructure.shared_memory_pool import ZoneType
            
            pool = self.pool
            
            # 安全获取 memory_zone 的键列表
            memory_keys = []
            with pool._zone_locks[ZoneType.MEMORY]:
                memory_keys = list(pool._memory_zone.keys())
            
            logger.info(f"[EpisodicMemory] Memory Zone 中有 {len(memory_keys)} 条数据")
            
            # 读取所有记忆数据
            for trace_id in memory_keys:
                envelope = await pool.read_memory(trace_id)
                if envelope and envelope.metadata:
                    metadata = envelope.metadata
                    if metadata.get("episode_id"):
                        episode_id = metadata.get("episode_id")
                        self._episode_index[episode_id] = metadata
            
            # 更新当前 episode 编号
            if self._episode_index:
                self._current_episode = max(self._episode_index.keys())
            
            logger.info(f"[EpisodicMemory] 已加载 {len(self._episode_index)} 条记忆索引")
            
        except Exception as e:
            logger.error(f"[EpisodicMemory] 加载索引失败：{e}", exc_info=True)
    
    async def store_episode(self, user_input: str, ai_response: str, 
                           metadata: Optional[Dict] = None) -> Dict:
        """
        存储一轮对话并生成摘要
        
        🔥 第 1 周优化：
        1. 初始摘要：使用基于规则的快速摘要（不阻塞主推理流）
        2. 异步复盘：将任务加入队列，由后台工作线程生成高质量语义摘要
        
        Args:
            user_input: 用户输入
            ai_response: AI 回复
            metadata: 附加元数据（如感知上下文 trace_id）
        
        Returns:
            Dict: 包含 episode_id 和摘要
        """
        try:
            # 🔥 关键修复：等待共享池初始化（支持懒加载）
            if self.pool is None:
                logger.warning("[EpisodicMemory] 共享池未初始化，开始懒加载...")
                
                # 尝试懒加载：检查是否有事件循环
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # 事件循环运行中，异步初始化
                        logger.info("[EpisodicMemory] 事件循环运行中，异步初始化共享池...")
                        await self.initialize_async()
                    else:
                        # 事件循环未运行，同步初始化
                        loop.run_until_complete(self.initialize_async())
                except RuntimeError:
                    # 没有事件循环，创建新的
                    logger.info("[EpisodicMemory] 创建新事件循环并初始化...")
                    new_loop = asyncio.new_event_loop()
                    try:
                        new_loop.run_until_complete(self.initialize_async())
                    finally:
                        new_loop.close()
                
                # 验证初始化结果
                if self.pool is None:
                    logger.error("[EpisodicMemory] 懒加载失败，共享池仍为 None")
                    return {"episode_id": None, "error": "pool_initialization_failed"}
                
                logger.info("[EpisodicMemory] 懒加载成功，共享池已初始化")
            
            self._current_episode += 1
            episode_id = self._current_episode
            
            # 1. 🔥 快速生成初始摘要（基于规则，< 10ms）
            initial_summary = self._generate_quick_summary(user_input, ai_response)
            
            # 2. 存储完整对话到共享池（使用 write 方法支持 zone 参数）
            trace_id = f"trace_episode_{episode_id}_full"
            envelope = DataEnvelope(
                trace_id=trace_id,
                timestamp=time.time(),
                data_type=DataType.TEXT_USER,
                zone=ZoneType.MEMORY,
                payload={
                    "user": user_input,
                    "ai": ai_response,
                    "timestamp": time.time()
                },
                metadata={
                    "episode_id": episode_id,
                    "type": "full_dialogue"
                }
            )
            trace_id = await self.pool.write(envelope)
            
            # 3. 存储摘要到索引（初始摘要）
            episode_metadata = {
                "episode_id": episode_id,
                "summary": initial_summary,
                "summary_type": "quick",  # 标记为快速摘要
                "user_preview": user_input[:50] if user_input else "",
                "ai_preview": ai_response[:50] if isinstance(ai_response, str) else str(ai_response)[:50],
                "trace_id": trace_id,  # 指向完整对话
                "timestamp": time.time(),
                "ttl": self.ttl_seconds
            }
            
            if metadata:
                episode_metadata.update(metadata)
            
            self._episode_index[episode_id] = episode_metadata
            
            logger.info(f"[EpisodicMemory] 存储对话：episode={episode_id}, initial_summary={initial_summary[:30]}...")
            
            # 4. 🔥 关键优化：将摘要任务加入异步队列（不阻塞主推理流）
            # 使用 put_nowait 避免协程未被 await 的问题
            try:
                self._pending_summarization_queue.put_nowait({
                    "episode_id": episode_id,
                    "user_input": user_input,
                    "ai_response": ai_response
                })
                self._stats["summarizations_pending"] += 1
                logger.info(f"[EpisodicMemory] 摘要任务已入队：episode={episode_id}（后台异步处理）")
            except Exception as e:
                logger.warning(f"[EpisodicMemory] 队列操作失败：{e}，跳过异步摘要")
            
            self._stats["total_writes"] += 1
            
            return {
                "episode_id": episode_id,
                "summary": initial_summary,
                "trace_id": trace_id
            }
            
        except Exception as e:
            logger.error(f"[EpisodicMemory] 存储失败：{e}", exc_info=True)
            return {"episode_id": None, "summary": "", "trace_id": None}
    
    def _generate_quick_summary(self, user_input: str, ai_response: str) -> str:
        """
        🔥 第 1 周优化：快速生成初始摘要（基于规则，< 10ms）
        
        用于主推理流中快速返回，不阻塞用户响应
        
        Args:
            user_input: 用户输入
            ai_response: AI 回复
        
        Returns:
            str: 快速摘要
        """
        # 判断问题类型
        if any(kw in user_input for kw in ["是什么", "什么是", "定义"]):
            question_type = "询问定义"
        elif any(kw in user_input for kw in ["怎么", "如何", "怎么做"]):
            question_type = "询问方法"
        elif any(kw in user_input for kw in ["为什么", "为何"]):
            question_type = "询问原因"
        elif any(kw in user_input for kw in ["多少", "价格", "钱"]):
            question_type = "询问数量"
        else:
            question_type = "一般对话"
        
        # 🔥 修复：ai_response 可能是元组，需要转换为字符串
        user_preview = user_input[:30].strip() if user_input else ""
        ai_preview = ai_response[:30].strip() if isinstance(ai_response, str) else str(ai_response)[:30].strip()
        summary = f"{question_type}: {user_preview} → {ai_preview}"
        
        # 限制长度
        if len(summary) > self.summary_max_length:
            summary = summary[:self.summary_max_length-3] + "..."
        
        return summary
    
    async def _generate_semantic_summary(self, user_input: str, ai_response: str) -> str:
        """
        🔥 第 1 周优化：生成高质量语义摘要（异步复盘）
        
        使用轻量级模型生成压缩叙事，包含核心信息
        
        策略:
        - 提取关键实体（谁、什么、何时、何地）
        - 压缩对话为叙事性描述
        - 保留因果逻辑
        
        Args:
            user_input: 用户输入
            ai_response: AI 回复
        
        Returns:
            str: 语义摘要
        """
        try:
            # 🔥 方案 1：使用 L1-B 生成（轻量级，快速）
            if self.summary_model:
                prompt = (
                    f"请用 50-100 字总结以下对话，保留核心信息：\n"
                    f"用户：{user_input}\n"
                    f"AI: {ai_response}\n"
                    f"摘要："
                )
                
                # 调用模型生成摘要
                # 🔥 注意：这里需要根据实际项目架构调整
                # summary = await self.summary_model.generate(prompt)
                # return summary.strip()
                
                # 临时降级：使用规则生成
                return self._generate_quick_summary(user_input, ai_response)
            
            # 🔥 方案 2：降级到规则生成（模型不可用时）
            return self._generate_quick_summary(user_input, ai_response)
            
        except Exception as e:
            logger.error(f"[EpisodicMemory] 语义摘要生成失败：{e}")
            return self._generate_quick_summary(user_input, ai_response)
    
    async def _generate_summary(self, user_input: str, ai_response: str) -> str:
        """
        生成对话摘要
        
        策略:
        - 提取核心信息（谁、什么、何时、何地）
        - 忽略细节和修饰
        - 控制在 100 字以内
        
        Args:
            user_input: 用户输入
            ai_response: AI 回复
        
        Returns:
            str: 对话摘要
        """
        try:
            # 🔥 简单策略：提取关键信息（后续可升级为模型生成）
            # 规则：
            # 1. 用户问题类型（是什么/怎么做/为什么）
            # 2. 核心主题（前 10 个字符）
            # 3. AI 回答类型（事实/建议/解释）
            
            user_preview = user_input[:30].strip() if user_input else ""
            ai_preview = ai_response[:30].strip() if isinstance(ai_response, str) else str(ai_response)[:30].strip()
            
            # 判断问题类型
            if any(kw in user_input for kw in ["是什么", "什么是", "定义"]):
                question_type = "询问定义"
            elif any(kw in user_input for kw in ["怎么", "如何", "怎么做"]):
                question_type = "询问方法"
            elif any(kw in user_input for kw in ["为什么", "为何"]):
                question_type = "询问原因"
            elif any(kw in user_input for kw in ["多少", "价格", "钱"]):
                question_type = "询问数量"
            else:
                question_type = "一般对话"
            
            summary = f"{question_type}: {user_preview} → {ai_preview}"
            
            # 限制长度
            if len(summary) > self.summary_max_length:
                summary = summary[:self.summary_max_length-3] + "..."
            
            return summary
            
        except Exception as e:
            logger.error(f"[EpisodicMemory] 生成摘要失败：{e}")
            return f"{user_input[:30] if user_input else ''}... → {ai_response[:30] if isinstance(ai_response, str) else str(ai_response)[:30]}..."
    
    async def search_by_summary(self, query: str, top_k: int = 5,
                                time_window: Optional[int] = None) -> List[Dict]:
        """
        基于摘要检索相关对话
        
        Args:
            query: 查询文本
            top_k: 返回数量
            time_window: 时间窗口（秒），None 表示不限
        
        Returns:
            List[Dict]: 摘要列表（包含 trace_id，可用于读取完整内容）
        """
        try:
            logger.info(f"[EpisodicMemory] 检索：query='{query}', top_k={top_k}")
            
            # 1. 获取候选摘要
            candidates = []
            current_time = time.time()
            
            for episode_id, metadata in self._episode_index.items():
                # 检查时间窗口
                if time_window:
                    age = current_time - metadata.get("timestamp", 0)
                    if age > time_window:
                        continue
                
                # 计算相似度（基于摘要）
                summary = metadata.get("summary", "")
                similarity = self._calculate_similarity(query, summary)
                
                if similarity > 0.1:  # 阈值
                    candidates.append({
                        "episode_id": episode_id,
                        "summary": summary,
                        "similarity": similarity,
                        "trace_id": metadata.get("trace_id"),
                        "timestamp": metadata.get("timestamp"),
                        "user_preview": metadata.get("user_preview"),
                        "ai_preview": metadata.get("ai_preview")
                    })
            
            # 2. 排序并返回 top_k
            candidates.sort(key=lambda x: x["similarity"], reverse=True)
            
            logger.info(f"  检索到 {len(candidates)} 条，返回 top {min(top_k, len(candidates))}")
            
            return candidates[:top_k]
            
        except Exception as e:
            logger.error(f"[EpisodicMemory] 检索失败：{e}", exc_info=True)
            return []
    
    def _calculate_similarity(self, query: str, text: str) -> float:
        """
        计算文本相似度（字符级别 Jaccard 相似度）
        
        Args:
            query: 查询文本
            text: 待比较文本
        
        Returns:
            float: 相似度分数 (0-1)
        """
        query_chars = set(query)
        text_chars = set(text)
        
        intersection = len(query_chars & text_chars)
        union = len(query_chars | text_chars)
        
        return intersection / union if union > 0 else 0
    
    async def get_full_dialogue(self, episode_id: int) -> Optional[Dict]:
        """
        读取完整对话内容（分级读取 Level 2）
        
        Args:
            episode_id: 对话编号
        
        Returns:
            Optional[Dict]: 完整对话内容
        """
        try:
            metadata = self._episode_index.get(episode_id)
            if not metadata:
                logger.warning(f"[EpisodicMemory] 未找到 episode {episode_id}")
                return None
            
            trace_id = metadata.get("trace_id")
            if not trace_id:
                logger.warning(f"[EpisodicMemory] episode {episode_id} 没有 trace_id")
                return None
            
            # 从共享池读取完整对话（使用 read_memory 方法）
            envelope = await self.pool.read_memory(trace_id)
            full_data = envelope.payload if envelope else None
            
            logger.info(f"[EpisodicMemory] 读取完整对话：episode={episode_id}")
            
            return full_data
            
        except Exception as e:
            logger.error(f"[EpisodicMemory] 读取完整对话失败：{e}", exc_info=True)
            return None
    
    async def get_recent_episodes(self, limit: int = 10) -> List[Dict]:
        """
        获取最近 N 轮对话的摘要
        
        Args:
            limit: 数量限制
        
        Returns:
            List[Dict]: 摘要列表
        """
        try:
            episodes = []
            
            # 按 episode_id 倒序排列
            sorted_episodes = sorted(
                self._episode_index.items(),
                key=lambda x: x[0],
                reverse=True
            )
            
            for episode_id, metadata in sorted_episodes[:limit]:
                episodes.append({
                    "episode_id": episode_id,
                    "summary": metadata.get("summary"),
                    "user_preview": metadata.get("user_preview"),
                    "ai_preview": metadata.get("ai_preview"),
                    "timestamp": metadata.get("timestamp"),
                    "trace_id": metadata.get("trace_id")
                })
            
            return episodes
            
        except Exception as e:
            logger.error(f"[EpisodicMemory] 获取最近对话失败：{e}", exc_info=True)
            return []
    
    async def cleanup_expired(self):
        """清理过期记忆"""
        try:
            current_time = time.time()
            expired_episodes = []
            
            for episode_id, metadata in self._episode_index.items():
                age = current_time - metadata.get("timestamp", 0)
                if age > self.ttl_seconds:
                    expired_episodes.append(episode_id)
            
            for episode_id in expired_episodes:
                # 从索引中移除
                del self._episode_index[episode_id]
                logger.info(f"[EpisodicMemory] 清理过期记忆：episode={episode_id}")
            
            logger.info(f"[EpisodicMemory] 清理了 {len(expired_episodes)} 条过期记忆")
            
        except Exception as e:
            logger.error(f"[EpisodicMemory] 清理失败：{e}")


# 全局单例
_episodic_memory_instance: Optional[EpisodicMemory] = None

def get_episodic_memory() -> EpisodicMemory:
    """获取单例"""
    global _episodic_memory_instance
    if _episodic_memory_instance is None:
        _episodic_memory_instance = EpisodicMemory()
    return _episodic_memory_instance
