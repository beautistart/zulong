# 共享池持久化失效诊断报告

**诊断时间**: 2026-04-13 01:15  
**问题**: 重启后没有记忆 - 共享池快照未更新

---

## 🔍 问题现象

### 当前状态
1. ✅ **短期记忆索引**: 已更新到 `current_turn=445`
   - 新增轮次：443, 444, 445（3 轮新对话）
   - 索引文件已保存到磁盘

2. ❌ **共享池快照**: 未更新
   - 最新快照：`snapshot_20260412_192725.json.gz`
   - 更新时间：2026-04-12 19:27:25（昨天）
   - 重启时间：2026-04-13 01:11:42（今天）
   - **时间差**: 5 小时 44 分钟

3. ❌ **重启后记忆丢失**: 
   - 新对话（443-445）只存在于索引文件
   - 共享池加载的是旧快照（157 条数据）
   - 新对话的 trace_id 在共享池中找不到

---

## 🔬 根因分析

### 问题 1: 后台快照任务未执行

**代码位置**: `shared_memory_pool.py` L721

**后台任务**:
```python
async def _snapshot_loop(self):
    """后台快照循环（每 5 分钟保存一次）"""
    while self._running:
        await asyncio.sleep(self.snapshot_interval)
        
        # 检查并保存待处理的快照
        if not self._save_queue.empty():
            logger.debug(f"💾 [SharedMemoryPool] 快照检查点...")
            await self._process_save_queue()
```

**问题**:
- ❌ **快照间隔太长**: 5 分钟（300 秒）
- ❌ **只在有队列请求时保存**: 依赖 `_queue_snapshot_save` 被调用
- ⚠️ **可能的问题**: 后台任务可能没有正确启动

### 问题 2: 保存队列可能未处理

**代码位置**: `shared_memory_pool.py` L731

**保存流程**:
```python
async def _queue_snapshot_save(self):
    # 将保存请求加入队列
    await self._save_queue.put(time.time())
    
    # 如果没有正在进行的保存任务，启动一个
    if self._pending_save_task is None or self._pending_save_task.done():
        logger.debug(f"💾 [SharedMemoryPool] 触发快照保存...")
        self._pending_save_task = asyncio.create_task(self._process_save_queue())
```

**可能问题**:
1. ⚠️ **事件循环问题**: 异步任务可能在错误的事件循环中创建
2. ⚠️ **任务被销毁**: 日志显示有 `Task was destroyed but it is pending!` 的警告
3. ⚠️ **队列阻塞**: 保存任务可能卡住了

### 问题 3: 启动时的日志警告

**观察到的日志**:
```
[2026-04-13 01:11:29.343] [shared_memory_pool] [e92c2f15] ✅ 已恢复快照：snapshot_20260412_192725.json.gz (共 157 条数据)
[2026-04-13 01:11:29.344] [shared_memory_pool] [1026da04] 🚀 [SharedMemoryPool] 后台任务已启动
```

**但随后有警告**:
```
[2026-04-13 01:03:31.201] [asyncio] [efc65dae] Task was destroyed but it is pending!
task: <Task pending name='Task-2' coro=<SharedMemoryPool._cleanup_loop() done...>
[2026-04-13 01:03:31.201] [asyncio] [5c0f19c3] Task was destroyed but it is pending!
task: <Task pending name='Task-3' coro=<SharedMemoryPool._snapshot_loop() done...>
```

**分析**:
- ✅ 后台任务确实启动了
- ❌ 但可能在初始化过程中被销毁了
- ⚠️ 异步任务的生命周期管理可能有问题

---

## 💡 解决方案

### 方案 1: 增强日志输出（立即执行）

**目标**: 确认保存队列是否被调用

**修改位置**: `shared_memory_pool.py` L731-750

**添加日志**:
```python
async def _queue_snapshot_save(self):
    if not self.persistence_enabled:
        return
    
    # 将保存请求加入队列
    await self._save_queue.put(time.time())
    
    # 🔥 新增：详细日志
    logger.info(f"💾 [SharedMemoryPool] 保存请求已加入队列 (队列大小：{self._save_queue.qsize()})")
    
    # 如果没有正在进行的保存任务，启动一个
    if self._pending_save_task is None or self._pending_save_task.done():
        logger.info(f"💾 [SharedMemoryPool] 触发快照保存任务...")
        self._pending_save_task = asyncio.create_task(self._process_save_queue())
    else:
        logger.debug(f"💾 [SharedMemoryPool] 保存任务已在运行，跳过")
```

### 方案 2: 缩短快照间隔（推荐）

**问题**: 5 分钟间隔太长，容易丢失数据

**修改位置**: `shared_memory_pool.py` 初始化配置

**建议**:
```python
# 原配置
self.snapshot_interval = 300  # 5 分钟

# 新配置
self.snapshot_interval = 30  # 30 秒
```

### 方案 3: 强制立即保存（调试用）

**修改位置**: `write_memory` 方法

**临时方案**:
```python
# 在写入后强制立即保存
await self._queue_snapshot_save()

# 🔥 调试用：强制立即处理队列
if self._save_queue.qsize() > 0:
    await self._process_save_queue()
    logger.info(f"💾 [SharedMemoryPool] ✅ 强制保存完成")
```

### 方案 4: 修复后台任务生命周期（长期）

**问题**: 后台任务在初始化时可能被销毁

**修复方向**:
1. 确保后台任务在正确的事件循环中创建
2. 添加任务健康检查
3. 实现任务重启机制

---

## 🛠️ 立即修复步骤

### 步骤 1: 添加详细日志（5 分钟）

**文件**: `zulong/infrastructure/shared_memory_pool.py`

**修改**:
1. 在 `_queue_snapshot_save` 中添加详细日志
2. 在 `_process_save_queue` 中添加开始/结束日志
3. 在 `_save_snapshot` 中添加保存完成日志

### 步骤 2: 测试保存流程（10 分钟）

**测试对话**:
```
用户：测试记忆 1
AI: （回复）

用户：测试记忆 2
AI: （回复）
```

**观察日志**:
```
💾 [SharedMemoryPool] 保存请求已加入队列 (队列大小：1)
💾 [SharedMemoryPool] 触发快照保存任务...
💾 [SharedMemoryPool] 合并保存请求：1 次 → 1 次实际保存
💾 [SharedMemoryPool] ✅ 快照已保存：snapshot_20260413_011500.json.gz
```

### 步骤 3: 验证快照文件（2 分钟）

**检查文件**:
```bash
Get-ChildItem .\data\shared_memory_pool\ | Sort-Object LastWriteTime -Descending
```

**预期**: 看到新的快照文件（时间戳是当前的）

### 步骤 4: 重启验证（5 分钟）

**重启系统**:
```bash
# 停止
Ctrl+C

# 重启
python -m zulong.bootstrap
python -m openclaw_bridge.bootstrap
```

**验证**:
- ✅ 共享池加载最新快照
- ✅ 新对话的 trace_id 可访问
- ✅ AI 记得重启前的对话

---

## 📊 监控指标

### 运行时日志检查点
| 日志 | 预期频率 | 含义 |
|------|---------|------|
| **保存请求已加入队列** | 每次对话 | 写入触发保存 |
| **触发快照保存任务** | 每 N 次对话 | 启动保存任务 |
| **合并保存请求** | 每 0.5 秒 | 合并多次写入 |
| **快照已保存** | 每 0.5 秒 | 保存成功 |

### 文件检查点
| 检查项 | 方法 | 预期结果 |
|--------|------|---------|
| **快照文件时间戳** | 文件列表 | < 1 分钟前 |
| **快照文件大小** | 文件大小 | > 10KB |
| **快照文件数量** | 文件计数 | 持续增长 |

---

## 🎯 成功标准

### 功能标准
- ✅ 每次对话都触发保存请求
- ✅ 保存任务正常执行
- ✅ 快照文件实时更新
- ✅ 重启后记忆完整保留

### 性能标准
- ✅ 保存延迟 < 100ms
- ✅ 快照间隔 < 1 分钟
- ✅ 不影响对话流畅性
- ✅ 日志清晰可读

### 可靠性标准
- ✅ 无数据丢失
- ✅ 后台任务持续运行
- ✅ 异常自动恢复
- ✅ 监控告警正常

---

## 📝 下一步行动

### 立即执行
1. ⏳ **添加详细日志**: 确认保存流程
2. ⏳ **测试对话**: 观察日志输出
3. ⏳ **检查快照**: 验证文件生成
4. ⏳ **重启验证**: 测试记忆保留

### 短期优化
1. ⏳ **缩短快照间隔**: 300s → 30s
2. ⏳ **修复任务生命周期**: 确保后台任务正常运行
3. ⏳ **添加健康检查**: 定期检查后台任务状态

### 长期优化
1. ⏳ **增量保存**: 只保存变化的部分
2. ⏳ **智能间隔**: 根据对话频率动态调整
3. ⏳ **分布式存储**: 支持多实例共享

---

**报告结束**

**下一步**: 立即添加详细日志并测试保存流程
