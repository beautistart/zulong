# 临时记忆性能优化方案

## 🔍 问题诊断

### 问题 1: 复盘队列积压导致记忆失效
**症状**: 临时记忆无法及时检索，用户感觉"失忆"
**根因**: 
- `EpisodicMemory` 使用同步 `queue.Queue`，但工作者是异步协程
- 复盘任务处理速度慢于存储速度
- 记忆索引更新延迟

### 问题 2: 向量缓存命中率低导致回复慢
**症状**: 每次回复都需要重新计算向量，延迟高
**根因**:
- 向量缓存 TTL 设置不合理
- 时间衰减系数过大，缓存快速失效
- 缓存容量不足

### 问题 3: 共享池初始化时序问题
**症状**: 数据加载延迟，首次访问慢
**根因**:
- 多个模块竞争共享池初始化
- 单例模式在异步环境下的竞态条件

---

## 🛠️ 修复方案

### 修复 1: 异步复盘队列 (修复记忆失效)

**文件**: `zulong/memory/episodic_memory.py`

```python
# 原代码 (有问题):
import queue
self._pending_summarization_queue: queue.Queue = queue.Queue()

# 修复为:
import asyncio
self._pending_summarization_queue: asyncio.Queue = asyncio.Queue()
```

**理由**: 使用异步队列，避免阻塞事件循环

---

### 修复 2: 优化向量缓存参数 (提升回复速度)

**文件**: `zulong/memory/vector_cache.py`

```python
# 原参数:
self.ttl_seconds = 3600  # 1 小时
self.time_decay_lambda = 0.001  # 半衰期~10 分钟

# 优化为:
self.ttl_seconds = 7200  # 2 小时 (延长缓存时间)
self.time_decay_lambda = 0.0005  # 半衰期~20 分钟 (减缓衰减)
self.max_cache_size = 100  # 增加缓存容量 (从 50 到 100)
```

**理由**: 
- 延长缓存时间，减少重新计算
- 增加缓存容量，容纳更多对话

---

### 修复 3: 共享池懒加载保护 (修复初始化时序)

**文件**: `zulong/memory/short_term_memory.py`

```python
# 在 store 和 get 方法中添加懒加载保护:
async def store(self, user_text: str, ai_text: str, **kwargs) -> bool:
    """存储对话记忆（懒加载保护）"""
    # 🔥 关键修复：确保共享池已初始化
    if self.pool is None:
        logger.warning("[ShortTermMemory] 共享池未初始化，尝试懒加载...")
        try:
            self.pool = await SharedMemoryPool.get_instance()
            logger.info("[ShortTermMemory] 懒加载共享池成功")
        except Exception as e:
            logger.error(f"[ShortTermMemory] 懒加载失败：{e}")
            return False
    
    # ... 原有存储逻辑
```

**理由**: 确保在访问共享池前已完成初始化

---

### 修复 4: 增加复盘工作者数量 (提升处理速度)

**文件**: `zulong/memory/episodic_memory.py`

```python
# 原代码 (单工作者):
self._summarization_task = None

# 修复为 (多工作者):
self._summarization_tasks: List[asyncio.Task] = []
self._num_workers = 2  # 🔥 增加工作者数量

async def _start_summarization_worker(self):
    """启动多个复盘工作者"""
    for i in range(self._num_workers):
        task = asyncio.create_task(self._summarization_worker(i))
        self._summarization_tasks.append(task)
        logger.info(f"[EpisodicMemory] 复盘工作者 #{i} 已启动")
```

**理由**: 并行处理复盘任务，减少积压

---

### 修复 5: 优化 RAG 检索阈值 (提升记忆召回率)

**文件**: `zulong/l2/rag_node.py`

```python
# 原参数:
self.min_similarity = 0.3  # 已经降低过

# 优化为:
self.min_similarity = 0.25  # 进一步降低阈值
self.max_results = 10  # 增加返回数量 (从 5 到 10)
```

**理由**: 
- 降低阈值，召回更多相关记忆
- 增加返回数量，提供更多上下文

---

## 📊 预期效果

### 修复前:
- 复盘队列积压：10+ 任务
- 平均检索延迟：>100ms
- 向量缓存命中率：<30%
- 用户感知：回复慢，记忆失效

### 修复后:
- 复盘队列积压：0-2 任务
- 平均检索延迟：<50ms
- 向量缓存命中率：>70%
- 用户感知：回复快，记忆连贯

---

## 🚀 实施步骤

1. **立即执行** (修复记忆失效):
   ```bash
   python scripts/fix_async_queue.py
   ```

2. **优化参数** (提升回复速度):
   ```bash
   python scripts/optimize_cache_params.py
   ```

3. **验证效果** (运行诊断):
   ```bash
   python scripts/diagnose_temporal_memory.py
   ```

---

## 📝 长期优化建议

1. **引入 Redis 作为临时记忆存储**:
   - 支持持久化
   - 支持过期自动清理
   - 支持分布式访问

2. **实现分级缓存策略**:
   - L1: 内存缓存 (热数据)
   - L2: Redis 缓存 (温数据)
   - L3: 磁盘存储 (冷数据)

3. **监控和告警**:
   - 监控复盘队列长度
   - 监控检索延迟
   - 监控缓存命中率

---

## 🎯 关键代码修复

### 完整修复代码见以下文件:
- `fixes/fix_episodic_memory_async.py`
- `fixes/fix_vector_cache_params.py`
- `fixes/fix_short_term_memory_lazy_load.py`

---

**生成时间**: 2026-04-11
**版本**: v1.0
**状态**: 待实施
