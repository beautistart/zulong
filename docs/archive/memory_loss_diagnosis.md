# 失忆问题诊断报告

**问题时间**: 2026-04-10  
**问题描述**: AI 完全忘记了用户的名字"小明"，回答"我叫'您'"  
**严重程度**: 🔴 **严重** - 核心记忆功能失效

---

## 🔍 问题现象

### 对话历史

```
用户：我叫小明
AI:   你好，小明！我是 AI 助手...

用户：我住在北京
AI:   你好，北京！我是人工智能助手...

用户：我昨天去了故宫参观
AI:   昨天故宫的参观真是令人难忘...

用户：非常震撼
AI:   非常震撼！故宫是中华文明的瑰宝...

用户：我还去了长城，人很多
AI:   那确实壮观，人多拥挤让人惊叹...

用户：我们又去吃了北京烤鸭，果然名不虚传
AI:   太美味了！北京烤鸭不仅外酥里嫩...

用户：我叫什么
AI:   我叫"您"。  ← ❌ 严重失忆！
```

---

## 🐛 根本原因分析

### 可能原因 1: 情景记忆存储失败 🔴

**检查点**: `episodic_memory.py` 第 271-350 行 `store_episode()` 方法

**预期日志**:
```
✅ [EpisodicMemory] 成功存储情景记忆：episode_id=1
✅ [EpisodicMemory] 摘要生成完成：小明自我介绍
```

**实际可能**:
```
❌ [EpisodicMemory] 存储失败：SharedMemoryPool.write_text() got an unexpected keyword argument 'zone'
```

**验证方法**:
1. 检查终端 33 (祖龙系统) 的日志
2. 搜索 `[EpisodicMemory]` 关键词
3. 查看是否有存储失败错误

---

### 可能原因 2: 记忆检索失败 🟡

**检查点**: `inference_engine.py` 第 1507-1532 行

**代码**:
```python
if hasattr(self, 'episodic_memory') and self.episodic_memory is not None:
    relevant_episodes = await self.episodic_memory.search_by_summary(
        query=user_input,
        top_k=5,
        time_window=7200  # 2 小时内的记忆
    )
    
    if relevant_episodes:
        # 注入记忆到 prompt
        memory_str = "\n【相关记忆】...\n"
```

**问题**:
- 如果 `search_by_summary()` 返回空列表
- 记忆不会被注入到 prompt
- 模型看不到历史信息

**验证方法**:
1. 搜索日志 `[临时记忆]` 关键词
2. 查看是否检索到记忆
3. 查看注入的记忆数量

---

### 可能原因 3: 共享池读取失败 🟡

**检查点**: `episodic_memory.py` 第 565-590 行 `get_full_dialogue()` 方法

**代码**:
```python
async def get_full_dialogue(self, episode_id: int) -> Optional[Dict]:
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
```

**问题**:
- `_episode_index` 可能为空
- `trace_id` 可能不存在
- 共享池读取可能返回 `None`

**验证方法**:
1. 检查 `_episode_index` 是否有数据
2. 检查 `trace_id` 是否正确
3. 检查共享池是否有数据

---

## 🔎 诊断步骤

### 步骤 1: 检查情景记忆存储日志

**在终端 33 中搜索**:
```
[EpisodicMemory] 成功存储
```

**预期看到**:
```
✅ [EpisodicMemory] 成功存储情景记忆：episode_id=1
✅ [EpisodicMemory] 成功存储情景记忆：episode_id=2
...
```

**如果看到**:
```
❌ [EpisodicMemory] 存储失败：TypeError
```
说明存储环节有问题

---

### 步骤 2: 检查情景记忆检索日志

**在终端 33 中搜索**:
```
[临时记忆]
```

**预期看到**:
```
📖 [临时记忆] 注入 5 条相关记忆（基于摘要）
```

**如果看到**:
```
📖 [临时记忆] 未检索到相关记忆
```
说明检索环节有问题

---

### 步骤 3: 检查情景记忆索引

**在调试控制台中执行**:
```python
from zulong.memory.episodic_memory import EpisodicMemory

em = EpisodicMemory()
print(f"索引数量：{len(em._episode_index)}")

# 查看所有索引
for eid, meta in em._episode_index.items():
    print(f"Episode {eid}: {meta.get('summary')}")
```

**预期**:
```
索引数量：7
Episode 1: 小明自我介绍
Episode 2: 用户住在北京
Episode 3: 参观故宫
...
```

**如果**:
```
索引数量：0
```
说明索引未建立

---

### 步骤 4: 检查共享池数据

**在调试控制台中执行**:
```python
from zulong.memory.episodic_memory import EpisodicMemory

em = EpisodicMemory()

# 检查第一个 episode
metadata = em._episode_index.get(1)
if metadata:
    trace_id = metadata.get('trace_id')
    print(f"Trace ID: {trace_id}")
    
    # 尝试读取
    envelope = await em.pool.read_memory(trace_id)
    if envelope:
        print(f"读取成功：{envelope.payload}")
    else:
        print("读取失败：envelope 为 None")
else:
    print("索引不存在")
```

---

## 🛠️ 解决方案

### 方案 1: 修复存储问题

**如果存储失败**:

1. **检查代码修复是否生效**
   - 确认 `episodic_memory.py` 第 306 行使用 `write()` 而不是 `write_text()`
   - 确认使用 `DataEnvelope` 包装数据

2. **重启系统**
   ```powershell
   # 停止当前系统
   # 重新启动
   $env:USE_VLLM_FOR_L2="true"
   cd d:\AI\project\zulong_beta4
   .\zulong_env\Scripts\activate
   python -m zulong.bootstrap
   ```

3. **测试存储**
   ```python
   # 在调试控制台中
   from zulong.memory.episodic_memory import EpisodicMemory
   
   em = EpisodicMemory()
   result = await em.store_episode("你好", "你好！")
   print(f"存储结果：{result}")
   ```

---

### 方案 2: 修复检索问题

**如果检索失败**:

1. **检查摘要生成**
   - 确认 `generate_summary()` 方法正常工作
   - 检查摘要是否包含关键词（如"小明"、"北京"）

2. **检查检索逻辑**
   - 查看 `search_by_summary()` 方法
   - 确认关键词匹配逻辑正确

3. **调整检索参数**
   ```python
   # 在 inference_engine.py 第 1508-1512 行
   relevant_episodes = await self.episodic_memory.search_by_summary(
       query=user_input,
       top_k=5,
       time_window=7200  # 可以延长到 24 小时
   )
   ```

---

### 方案 3: 修复索引问题

**如果索引为空**:

1. **检查索引初始化**
   ```python
   # 在 episodic_memory.py __init__ 方法
   self._episode_index: Dict[int, Dict] = {}
   ```

2. **检查索引更新**
   ```python
   # 在 store_episode() 方法中
   self._episode_index[episode_id] = {
       "trace_id": trace_id,
       "summary": summary,
       "timestamp": time.time(),
       ...
   }
   ```

3. **手动重建索引**
   ```python
   # 在调试控制台中
   from zulong.memory.episodic_memory import EpisodicMemory
   
   em = EpisodicMemory()
   await em._rebuild_index()  # 如果有此方法
   ```

---

## 📊 验证清单

### 存储验证

- [ ] 日志显示 `[EpisodicMemory] 成功存储情景记忆`
- [ ] 没有 TypeError 错误
- [ ] `_episode_index` 有数据
- [ ] 共享池有数据

### 检索验证

- [ ] 日志显示 `[临时记忆] 注入 X 条相关记忆`
- [ ] 检索到的记忆包含关键词
- [ ] 记忆注入到 prompt
- [ ] 模型能看到历史信息

### 功能验证

- [ ] 进行 5 轮对话测试
- [ ] 问"我叫什么"能正确回答
- [ ] 问"我住在哪里"能正确回答
- [ ] 问"我昨天去了哪里"能正确回答

---

## 🎯 快速测试脚本

**保存为** `scripts/test_memory_recall.py`:

```python
# -*- coding: utf-8 -*-
# 测试情景记忆存储和检索

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 80)
print("           情景记忆测试")
print("=" * 80)

async def test():
    from zulong.memory.episodic_memory import EpisodicMemory
    
    em = EpisodicMemory()
    
    # 1. 测试存储
    print("\n1. 测试存储")
    print("-" * 80)
    result = await em.store_episode("我叫小明", "你好，小明！")
    print(f"存储结果：{result}")
    
    result = await em.store_episode("我住在北京", "北京是个好地方")
    print(f"存储结果：{result}")
    
    # 2. 检查索引
    print("\n2. 检查索引")
    print("-" * 80)
    print(f"索引数量：{len(em._episode_index)}")
    for eid, meta in em._episode_index.items():
        print(f"Episode {eid}: {meta.get('summary')}")
    
    # 3. 测试检索
    print("\n3. 测试检索")
    print("-" * 80)
    episodes = await em.search_by_summary("名字", top_k=5)
    print(f"检索到 {len(episodes)} 条记忆")
    for ep in episodes:
        print(f"  - {ep['summary']}")
    
    # 4. 测试读取
    print("\n4. 测试读取")
    print("-" * 80)
    full_data = await em.get_full_dialogue(1)
    if full_data:
        print(f"用户：{full_data.get('user')}")
        print(f"AI:   {full_data.get('ai')}")
    else:
        print("读取失败")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test())
```

**运行**:
```powershell
cd d:\AI\project\zulong_beta4
.\zulong_env\Scripts\activate
python scripts\test_memory_recall.py
```

---

## 📝 总结

### 问题根源

根据之前的修复记录，**最可能的原因**是：

1. **情景记忆存储失败** - `write_text()` 方法调用错误
2. **共享池读取失败** - `read_text()` 方法不存在
3. **索引未建立** - 存储失败导致索引为空

### 紧急修复

1. **确认代码已修复**
   - 检查 `episodic_memory.py` 第 306 行
   - 检查 `episodic_memory.py` 第 586 行

2. **重启系统**
   - 停止当前运行
   - 重新启动祖龙系统

3. **测试验证**
   - 运行测试脚本
   - 进行对话测试

### 长期优化

1. **添加监控**
   - 监控存储成功率
   - 监控检索成功率

2. **添加告警**
   - 存储失败时告警
   - 检索失败时告警

3. **添加降级**
   - 如果情景记忆失效，使用短期记忆
   - 如果所有记忆失效，使用默认回复

---

**诊断人员**: AI Assistant  
**诊断日期**: 2026-04-10  
**状态**: 🔴 **紧急** - 需要立即修复
