# L1-B 调度器 - 记忆管理集成指南

**版本**: v1.0  
**创建时间**: 2026-04-09  
**状态**: ✅ 待实现

---

## 📋 概述

本文档描述如何将**动态资源管理**和**L2-BACKUP 异步复盘**集成到 L1-B 调度器中。

### 核心目标

1. **动态容量管理**：根据 L2 模型规格（4k/8k/128k）自动计算记忆容量
2. **L2-BACKUP 异步复盘**：利用空闲资源批量生成摘要，不阻塞主推理流
3. **智能调度**：在 L2-BACKUP 空闲时触发摘要生成任务

---

## 1. 动态容量管理

### 1.1 L1-B 读取模型配置

```python
# zulong/l1b/scheduler_gatekeeper.py

class L1BScheduler:
    def __init__(self):
        # 从 ModelConfig 读取 L2 模型配置
        self.l2_model_config = self._load_l2_model_config()
        
        # 计算记忆预算
        self.memory_budget = self._calculate_memory_budget()
    
    def _load_l2_model_config(self):
        """加载 L2 模型配置"""
        from zulong.models.config import ModelConfig
        
        config = ModelConfig.get_instance()
        l2_config = config.get_model_config("l2_prime")
        
        return {
            "model_id": l2_config.get("model_id"),
            "max_context": l2_config.get("max_context", 4096),  # 4k/8k/128k
            "type": l2_config.get("type", "vllm")
        }
    
    def _calculate_memory_budget(self):
        """
        计算记忆预算
        
        策略:
        - 预留 75% 的上下文用于记忆注入
        - 估算每轮对话 150 tokens
        - 动态计算最大轮次
        """
        max_context = self.l2_model_config.get("max_context", 4096)
        
        # 75% 用于记忆
        memory_tokens = int(max_context * 0.75)
        
        # 估算每轮对话 token 数（包括用户输入+AI 回复+摘要）
        estimated_turn_tokens = 150
        
        # 动态计算最大轮次
        max_episodes = max(
            10,  # 最小保留 10 轮
            memory_tokens // estimated_turn_tokens
        )
        
        # 限制最大值（避免内存爆炸）
        max_episodes = min(max_episodes, 200)
        
        return {
            "max_context": max_context,
            "memory_tokens": memory_tokens,
            "max_episodes": max_episodes,
            "estimated_turn_tokens": estimated_turn_tokens
        }
    
    def get_memory_config(self):
        """获取记忆配置（供 EpisodicMemory 使用）"""
        return self.memory_budget
```

### 1.2 EpisodicMemory 集成动态配置

```python
# zulong/memory/episodic_memory.py

class EpisodicMemory:
    async def initialize_async(self, scheduler=None):
        """
        异步初始化（支持从 L1-B 获取动态配置）
        
        Args:
            scheduler: L1BScheduler 实例（可选）
        """
        from zulong.infrastructure.shared_memory_pool import SharedMemoryPool
        
        self.pool = await SharedMemoryPool.get_instance()
        
        # 🔥 关键：从 L1-B 获取动态配置
        if scheduler:
            memory_config = scheduler.get_memory_config()
            self.max_episodes = memory_config["max_episodes"]
            self.max_tokens_reserved = memory_config["memory_tokens"]
            
            logger.info(f"[EpisodicMemory] 使用 L1-B 动态配置:")
            logger.info(f"  - max_episodes={self.max_episodes}")
            logger.info(f"  - max_tokens_reserved={self.max_tokens_reserved}")
        else:
            # 降级：使用默认配置
            await self._calculate_dynamic_capacity()
        
        # 加载索引
        await self._load_index()
        
        # 启动异步复盘
        self._start_summarization_worker()
```

---

## 2. L2-BACKUP 异步复盘调度

### 2.1 L1-B 监听新对话

```python
# zulong/l1b/scheduler_gatekeeper.py

class L1BScheduler:
    def __init__(self):
        # 摘要任务队列
        self.summarization_queue = asyncio.Queue()
        
        # L2-BACKUP 状态
        self.l2_backup_status = "IDLE"  # IDLE / BUSY
        
        # 启动调度器
        self._start_backup_scheduler()
    
    async def on_user_text(self, event):
        """
        用户文本事件处理器
        
        流程:
        1. 路由到 L2-PRIME（主推理流）
        2. 监听回复完成
        3. 将摘要任务入队
        """
        # 1. 路由到 L2-PRIME
        response = await self.route_to_l2_prime(event)
        
        # 2. 🔥 监听回复完成，将摘要任务入队
        if response and response.get("success"):
            await self.summarization_queue.put({
                "episode_id": response.get("episode_id"),
                "user_input": event.text,
                "ai_response": response.get("text"),
                "timestamp": time.time()
            })
            
            logger.info(f"[L1-B] 摘要任务已入队：episode={response.get('episode_id')}")
        
        return response
    
    def _start_backup_scheduler(self):
        """启动 L2-BACKUP 调度器"""
        import asyncio
        
        async def backup_scheduler_worker():
            """后台调度工作线程"""
            logger.info("[L1-B] L2-BACKUP 调度器已启动")
            
            while True:
                try:
                    # 等待 L2-BACKUP 空闲
                    if self.l2_backup_status == "IDLE":
                        # 从队列中获取摘要任务
                        try:
                            task = self.summarization_queue.get_nowait()
                            
                            logger.info(f"[L1-B] 调度 L2-BACKUP 处理摘要：episode={task['episode_id']}")
                            
                            # 唤醒 L2-BACKUP
                            await self.wake_l2_backup()
                            
                            # 分配任务
                            await self.dispatch_summarization_task(task)
                            
                            # 标记任务完成
                            self.summarization_queue.task_done()
                            
                        except asyncio.QueueEmpty:
                            # 队列为空，等待
                            await asyncio.sleep(1)
                    else:
                        # L2-BACKUP 忙碌，等待
                        await asyncio.sleep(0.5)
                else:
                    # L2-BACKUP 忙碌，等待
                    await asyncio.sleep(0.5)
                
                except asyncio.CancelledError:
                    logger.info("[L1-B] L2-BACKUP 调度器已停止")
                    break
                except Exception as e:
                    logger.error(f"[L1-B] 调度失败：{e}", exc_info=True)
        
        # 启动后台任务
        self._backup_scheduler_task = asyncio.create_task(backup_scheduler_worker())
    
    async def wake_l2_backup(self):
        """唤醒 L2-BACKUP（如果处于休眠）"""
        # 实现 L2-BACKUP 唤醒逻辑
        logger.info("[L1-B] 唤醒 L2-BACKUP...")
        # ...
    
    async def dispatch_summarization_task(self, task):
        """分发摘要任务到 L2-BACKUP"""
        # 实现任务分发逻辑
        logger.info(f"[L1-B] 分发摘要任务到 L2-BACKUP: episode={task['episode_id']}")
        # ...
```

### 2.2 L2-BACKUP 处理摘要

```python
# zulong/l2/backup_processor.py

class L2BackupProcessor:
    """L2-BACKUP 摘要生成器"""
    
    def __init__(self):
        self.model = None
        self.task_queue = asyncio.Queue()
    
    async def initialize_async(self):
        """初始化 L2-BACKUP"""
        from zulong.models.container import ModelContainer
        from zulong.models.config import ModelID
        
        container = ModelContainer()
        self.model = container.get_model(ModelID.L2_BACKUP)
        
        logger.info("[L2-BACKUP] 初始化完成")
    
    async def process_summarization_task(self, task):
        """
        处理摘要任务
        
        Args:
            task: Dict {episode_id, user_input, ai_response}
        """
        try:
            episode_id = task["episode_id"]
            user_input = task["user_input"]
            ai_response = task["ai_response"]
            
            logger.info(f"[L2-BACKUP] 开始生成摘要：episode={episode_id}")
            
            # 生成高质量语义摘要
            summary = await self._generate_semantic_summary(user_input, ai_response)
            
            # 更新 EpisodicMemory 索引
            from zulong.memory.episodic_memory import get_episodic_memory
            em = get_episodic_memory()
            
            if episode_id in em._episode_index:
                em._episode_index[episode_id]["summary"] = summary
                em._episode_index[episode_id]["summary_type"] = "semantic"
                
                logger.info(f"[L2-BACKUP] 摘要更新完成：episode={episode_id}")
            
            return {"success": True, "episode_id": episode_id, "summary": summary}
            
        except Exception as e:
            logger.error(f"[L2-BACKUP] 摘要生成失败：{e}", exc_info=True)
            return {"success": False, "episode_id": task["episode_id"]}
    
    async def _generate_semantic_summary(self, user_input: str, ai_response: str) -> str:
        """
        生成高质量语义摘要
        
        使用 Map-Reduce 策略处理长对话
        """
        # 1. 如果对话较短（< 500 字），直接生成
        if len(user_input) + len(ai_response) < 500:
            prompt = (
                f"请用 50-100 字总结以下对话，保留核心信息：\n"
                f"用户：{user_input}\n"
                f"AI: {ai_response}\n"
                f"摘要："
            )
            
            summary = await self.model.generate(prompt)
            return summary.strip()
        
        # 2. 🔥 如果对话较长，使用 Map-Reduce
        else:
            # Map: 分段生成摘要
            user_summary = await self._summarize_segment(user_input)
            ai_summary = await self._summarize_segment(ai_response)
            
            # Reduce: 合并为完整摘要
            prompt = (
                f"请合并以下两个摘要为一个连贯的叙事（50-100 字）：\n"
                f"用户部分：{user_summary}\n"
                f"AI 部分：{ai_summary}\n"
                f"合并摘要："
            )
            
            final_summary = await self.model.generate(prompt)
            return final_summary.strip()
    
    async def _summarize_segment(self, text: str) -> str:
        """分段摘要（Map 步骤）"""
        prompt = f"请用 30 字总结：{text[:200]}..."
        summary = await self.model.generate(prompt)
        return summary.strip()
```

---

## 3. 配置管理

### 3.1 配置文件示例

```yaml
# config/memory_config.yaml

memory:
  # 动态容量管理
  dynamic_capacity:
    enabled: true
    max_context_ratio: 0.75  # 75% 用于记忆
    estimated_turn_tokens: 150  # 估算每轮 token 数
    min_episodes: 10  # 最小保留轮次
    max_episodes: 200  # 最大保留轮次
  
  # 异步复盘
  async_summarization:
    enabled: true
    use_l2_backup: true  # 使用 L2-BACKUP
    batch_size: 5  # 批量处理大小
    max_wait_time: 60  # 最大等待时间（秒）
  
  # TTL 管理
  ttl:
    default_seconds: 7200  # 2 小时
    cleanup_interval: 300  # 清理间隔（5 分钟）
  
  # 检索配置
  retrieval:
    default_top_k: 5
    similarity_threshold: 0.1
    time_window_seconds: 7200
```

### 3.2 加载配置

```python
# zulong/memory/config.py

class MemoryConfig:
    """记忆配置管理器"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.config = self._load_config()
        self._initialized = True
    
    def _load_config(self):
        """加载配置文件"""
        import yaml
        from pathlib import Path
        
        config_path = Path("./config/memory_config.yaml")
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # 降级到默认配置
            return self._get_default_config()
    
    def _get_default_config(self):
        """默认配置"""
        return {
            "dynamic_capacity": {
                "enabled": True,
                "max_context_ratio": 0.75,
                "estimated_turn_tokens": 150,
                "min_episodes": 10,
                "max_episodes": 200
            },
            "async_summarization": {
                "enabled": True,
                "use_l2_backup": True,
                "batch_size": 5,
                "max_wait_time": 60
            },
            "ttl": {
                "default_seconds": 7200,
                "cleanup_interval": 300
            },
            "retrieval": {
                "default_top_k": 5,
                "similarity_threshold": 0.1,
                "time_window_seconds": 7200
            }
        }
    
    def get(self, key, default=None):
        """获取配置项"""
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
```

---

## 4. 监控与指标

### 4.1 性能监控

```python
# zulong/memory/monitor.py

class MemoryMonitor:
    """记忆系统监控器"""
    
    def __init__(self):
        self.metrics = {
            "capacity": {
                "current_episodes": 0,
                "max_episodes": 0,
                "memory_tokens_used": 0,
                "memory_tokens_total": 0
            },
            "summarization": {
                "pending_tasks": 0,
                "completed_tasks": 0,
                "avg_latency_ms": 0.0,
                "backup_utilization": 0.0
            },
            "retrieval": {
                "total_searches": 0,
                "avg_search_latency_ms": 0.0,
                "cache_hit_rate": 0.0
            }
        }
    
    async def start_monitoring(self):
        """启动监控"""
        import asyncio
        
        async def monitor_loop():
            while True:
                try:
                    # 收集指标
                    await self._collect_metrics()
                    
                    # 报告指标
                    self._report_metrics()
                    
                    # 每分钟报告一次
                    await asyncio.sleep(60)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"[MemoryMonitor] 监控失败：{e}", exc_info=True)
        
        self._monitor_task = asyncio.create_task(monitor_loop())
    
    async def _collect_metrics(self):
        """收集指标"""
        from zulong.memory.episodic_memory import get_episodic_memory
        
        em = get_episodic_memory()
        
        # 容量指标
        self.metrics["capacity"]["current_episodes"] = len(em._episode_index)
        self.metrics["capacity"]["max_episodes"] = em.max_episodes
        
        # 摘要指标
        self.metrics["summarization"]["pending_tasks"] = em._stats["summarizations_pending"]
        self.metrics["summarization"]["completed_tasks"] = em._stats["summarizations_completed"]
    
    def _report_metrics(self):
        """报告指标"""
        logger.info("=== 记忆系统监控报告 ===")
        logger.info(f"容量：{self.metrics['capacity']['current_episodes']}/{self.metrics['capacity']['max_episodes']} 轮")
        logger.info(f"摘要：{self.metrics['summarization']['pending_tasks']} 待处理，{self.metrics['summarization']['completed_tasks']} 已完成")
        logger.info(f"检索：{self.metrics['retrieval']['total_searches']} 次，平均延迟 {self.metrics['retrieval']['avg_search_latency_ms']:.2f}ms")
```

---

## 5. 测试用例

### 5.1 动态容量测试

```python
import unittest
from zulong.l1b.scheduler_gatekeeper import L1BScheduler

class TestDynamicCapacity(unittest.TestCase):
    
    def test_capacity_calculation_4k_model(self):
        """测试 4k 模型的容量计算"""
        scheduler = L1BScheduler()
        scheduler.l2_model_config = {"max_context": 4096}
        
        budget = scheduler._calculate_memory_budget()
        
        self.assertEqual(budget["memory_tokens"], 3072)  # 4096 * 0.75
        self.assertEqual(budget["max_episodes"], 20)  # 3072 // 150
    
    def test_capacity_calculation_8k_model(self):
        """测试 8k 模型的容量计算"""
        scheduler = L1BScheduler()
        scheduler.l2_model_config = {"max_context": 8192}
        
        budget = scheduler._calculate_memory_budget()
        
        self.assertEqual(budget["memory_tokens"], 6144)  # 8192 * 0.75
        self.assertEqual(budget["max_episodes"], 40)  # 6144 // 150
    
    def test_capacity_calculation_128k_model(self):
        """测试 128k 模型的容量计算"""
        scheduler = L1BScheduler()
        scheduler.l2_model_config = {"max_context": 131072}
        
        budget = scheduler._calculate_memory_budget()
        
        self.assertEqual(budget["memory_tokens"], 98304)  # 131072 * 0.75
        self.assertEqual(budget["max_episodes"], 200)  # 限制到最大值

if __name__ == "__main__":
    unittest.main()
```

### 5.2 异步复盘测试

```python
import unittest
import asyncio
from zulong.memory.episodic_memory import EpisodicMemory

class TestAsyncSummarization(unittest.TestCase):
    
    def setUp(self):
        self.em = EpisodicMemory()
        asyncio.run(self.em.initialize_async())
    
    def test_quick_summary_generation(self):
        """测试快速摘要生成（不阻塞）"""
        import time
        
        start = time.time()
        
        result = asyncio.run(self.em.store_episode(
            "AI MAX 395 是什么？",
            "AI MAX 395 是一款高性能处理器..."
        ))
        
        elapsed = time.time() - start
        
        # 快速摘要应该在 10ms 内完成
        self.assertLess(elapsed, 0.05)  # 50ms
        self.assertEqual(result["summary_type"], "quick")
    
    def test_semantic_summary_update(self):
        """测试语义摘要更新（异步）"""
        # 存储对话
        result = asyncio.run(self.em.store_episode(
            "测试问题",
            "测试回答"
        ))
        
        episode_id = result["episode_id"]
        
        # 等待异步复盘完成
        asyncio.run(asyncio.sleep(2))
        
        # 检查摘要是否更新
        metadata = self.em._episode_index[episode_id]
        
        # 应该从"quick"更新为"semantic"
        self.assertEqual(metadata["summary_type"], "semantic")

if __name__ == "__main__":
    unittest.main()
```

---

## 6. 总结

### 6.1 核心改进

1. ✅ **动态容量管理**：根据模型规格（4k/8k/128k）自动调整记忆容量
2. ✅ **L2-BACKUP 异步复盘**：利用空闲资源批量生成摘要，不阻塞主推理流
3. ✅ **智能调度**：L1-B 监听对话完成，自动调度 L2-BACKUP 处理摘要
4. ✅ **配置化管理**：支持 YAML 配置文件，灵活调整参数

### 6.2 性能预期

| 指标 | 目标值 | 实现方式 |
|------|-------|---------|
| 主推理流延迟 | < 10ms | 快速摘要 + 异步队列 |
| 摘要生成延迟 | < 2s | L2-BACKUP 批量处理 |
| 记忆容量 | 动态适配 | 4k 模型=20 轮，8k=40 轮，128k=200 轮 |
| L2-BACKUP 利用率 | > 60% | 空闲时自动唤醒 |

### 6.3 下一步

1. 实现 L1-B 与 EpisodicMemory 的配置同步
2. 实现 L2-BACKUP 唤醒和任务分发逻辑
3. 添加性能监控和告警机制
4. 优化 Map-Reduce 摘要生成策略
