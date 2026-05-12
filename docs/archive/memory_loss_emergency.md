# 情景记忆失忆问题 - 紧急诊断

**创建时间**: 2026-04-10  
**问题**: AI 忘记用户名字，回答"我叫'您'"  
**状态**: 🔴 **紧急**

---

## 🚨 问题确认

根据对话历史，AI 完全失忆：
- 用户说"我叫小明" → AI 忘记
- 用户问"我叫什么" → AI 回答"我叫'您'"

这是**情景记忆存储或检索完全失效**的表现。

---

## 🔍 诊断结果

### 1. 代码修复状态 ✅

检查 `episodic_memory.py`:
- ✅ 第 306 行：使用 `write()` + `DataEnvelope` (已修复)
- ✅ 第 586 行：使用 `read_memory()` (已修复)

### 2. 测试脚本超时 ❌

运行 `scripts/test_memory_recall.py` 超时：
```
TimeoutError: 等待共享池初始化超时
```

**说明**: 情景记忆的共享池初始化有问题

### 3. 可能原因

#### 原因 A: 共享池未正确初始化 🔴

**检查点**: `episodic_memory.py` 第 99 行

```python
# 在 __init__ 中
if self.pool is None:
    future = asyncio.get_event_loop().run_in_executor(
        None, 
        lambda: asyncio.run(self._init_pool())
    )
    future.result(timeout=15)  # 等待 15 秒
```

**问题**:
- 共享池初始化可能需要更长时间
- 或者初始化过程中出错

#### 原因 B: 存储成功但索引未更新 🟡

**检查点**: `episodic_memory.py` 第 340 行

```python
self._episode_index[episode_id] = episode_metadata
```

**如果这行没有执行**:
- 存储成功但索引为空
- 检索时找不到任何记忆

#### 原因 C: 检索逻辑有问题 🟡

**检查点**: `inference_engine.py` 第 1507-1532 行

```python
relevant_episodes = await self.episodic_memory.search_by_summary(
    query=user_input,
    top_k=5,
    time_window=7200
)
```

**问题**:
- `search_by_summary()` 可能返回空列表
- 时间窗口可能太短（2 小时）

---

## 🛠️ 解决方案

### 方案 1: 增加初始化超时时间

**修改**: `episodic_memory.py` 第 99 行

```python
# 原代码
future.result(timeout=15)  # 等待 15 秒

# 修改为
future.result(timeout=60)  # 等待 60 秒
```

---

### 方案 2: 添加初始化错误处理

**修改**: `episodic_memory.py` 第 95-105 行

```python
# 在 __init__ 中
if self.pool is None:
    try:
        future = asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: asyncio.run(self._init_pool())
        )
        future.result(timeout=60)
        logger.info("[EpisodicMemory] 共享池初始化成功")
    except TimeoutError:
        logger.error("[EpisodicMemory] 共享池初始化超时")
        # 不阻塞，继续使用（懒加载）
    except Exception as e:
        logger.error(f"[EpisodicMemory] 共享池初始化失败：{e}")
        # 不阻塞，继续使用（懒加载）
```

---

### 方案 3: 强制重建索引

**在调试控制台中执行**:

```python
from zulong.memory.episodic_memory import EpisodicMemory

em = EpisodicMemory()

# 手动触发存储
result = await em.store_episode("测试", "测试回复")
print(f"存储结果：{result}")

# 检查索引
print(f"索引数量：{len(em._episode_index)}")
for eid, meta in em._episode_index.items():
    print(f"Episode {eid}: {meta.get('summary')}")
```

---

### 方案 4: 调整检索参数

**修改**: `inference_engine.py` 第 1508-1512 行

```python
# 原代码
relevant_episodes = await self.episodic_memory.search_by_summary(
    query=user_input,
    top_k=5,
    time_window=7200  # 2 小时
)

# 修改为
relevant_episodes = await self.episodic_memory.search_by_summary(
    query=user_input,
    top_k=10,  # 增加检索数量
    time_window=86400  # 24 小时
)
```

---

## 📊 验证步骤

### 步骤 1: 检查共享池状态

**在调试控制台中**:
```python
from zulong.memory.episodic_memory import EpisodicMemory

em = EpisodicMemory()
print(f"共享池状态：{'已初始化' if em.pool else '未初始化'}")
print(f"索引数量：{len(em._episode_index)}")
```

### 步骤 2: 手动存储测试

```python
result = await em.store_episode("我叫小明", "你好小明")
print(f"存储结果：{result}")

# 应该看到
# 存储结果：{'episode_id': 1, 'summary': '...', 'trace_id': '...'}
```

### 步骤 3: 手动检索测试

```python
episodes = await em.search_by_summary("名字", top_k=5)
print(f"检索到：{len(episodes)} 条")

# 应该看到
# 检索到：1 条
```

### 步骤 4: 完整对话测试

1. 在调试控制台中进行对话
2. 问"我叫什么"
3. 检查是否能正确回答

---

## 🎯 紧急修复步骤

### 立即执行：

1. **停止当前系统**
   ```powershell
   # 在终端 33 中按 Ctrl+C
   ```

2. **修改超时时间**
   ```python
   # 修改 episodic_memory.py 第 99 行
   future.result(timeout=60)  # 从 15 秒改为 60 秒
   ```

3. **添加错误处理**
   ```python
   # 在 episodic_memory.py __init__ 中添加 try-except
   ```

4. **重启系统**
   ```powershell
   $env:USE_VLLM_FOR_L2="true"
   cd d:\AI\project\zulong_beta4
   .\zulong_env\Scripts\activate
   python -m zulong.bootstrap
   ```

5. **测试对话**
   ```
   用户：我叫小明
   AI:   你好，小明！
   
   用户：我住在北京
   AI:   北京是个好地方
   
   用户：我叫什么
   AI:   你叫小明  ← 应该正确回答
   ```

---

## 📝 根本原因推测

根据代码分析和测试结果，**最可能的原因**是：

1. **共享池初始化超时** 🔴
   - 测试脚本超时
   - 导致情景记忆无法正常工作

2. **索引未建立** 🟡
   - 如果初始化失败，`_episode_index` 为空
   - 检索时返回空列表

3. **懒加载未触发** 🟡
   - 如果初始化时出错但没有重试
   - 后续操作也无法使用记忆

---

## 📞 后续建议

### 短期（立即）：
- [ ] 增加初始化超时时间
- [ ] 添加错误处理和日志
- [ ] 重启系统测试

### 中期（本周）：
- [ ] 添加记忆监控
- [ ] 添加失败告警
- [ ] 优化检索算法

### 长期（本月）：
- [ ] 重构记忆系统架构
- [ ] 实现真正的懒加载
- [ ] 添加记忆备份机制

---

**诊断人员**: AI Assistant  
**诊断日期**: 2026-04-10  
**优先级**: 🔴 **P0 - 紧急**
