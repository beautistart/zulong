# 记忆系统测试指南（运行中系统）

**测试时间**: 2026-04-10  
**系统状态**: ✅ 祖龙系统运行中（终端 33）

---

## 🎯 测试步骤

### 步骤 1: 进行对话测试

**在终端 33 的调试控制台中依次输入**:

```
你好，我叫小明
```

等待 AI 回复后，继续：

```
我住在北京
```

继续：

```
我昨天去了故宫参观
```

继续：

```
非常震撼
```

继续：

```
我还去了长城，人很多
```

继续：

```
我们又去吃了北京烤鸭，果然名不虚传
```

最后测试记忆：

```
我叫什么
```

**预期回答**: 你叫小明

---

### 步骤 2: 检查记忆索引

**在终端 33 的调试控制台中输入 Python 代码**:

```python
from zulong.memory.episodic_memory import EpisodicMemory

# 获取情景记忆实例
em = EpisodicMemory()

print("=" * 80)
print("情景记忆状态检查")
print("=" * 80)

# 检查基本状态
print(f"✅ 索引数量：{len(em._episode_index)}")
print(f"✅ 共享池状态：{'已初始化' if em.pool else '未初始化'}")
print(f"✅ 初始化状态：{em._initialized}")

# 显示索引详情
if len(em._episode_index) > 0:
    print(f"\n📋 索引详情 (前 10 条):")
    for eid, meta in list(em._episode_index.items())[:10]:
        summary = meta.get('summary', 'N/A')
        trace_id = meta.get('trace_id', 'N/A')
        print(f"   Episode {eid}: {summary}")
        print(f"      Trace ID: {trace_id}")
else:
    print(f"\n⚠️  索引为空 - 可能没有进行对话或存储失败")
```

---

### 步骤 3: 测试记忆读取

**在终端 33 的调试控制台中输入**:

```python
from zulong.memory.episodic_memory import EpisodicMemory

em = EpisodicMemory()

# 检查是否有索引
if len(em._episode_index) > 0:
    # 读取第一个 episode
    first_id = list(em._episode_index.keys())[0]
    print(f"读取 Episode {first_id}")
    
    full_data = await em.get_full_dialogue(first_id)
    
    if full_data:
        print(f"✅ 读取成功")
        print(f"   用户：{full_data.get('user')}")
        print(f"   AI:   {full_data.get('ai')}")
    else:
        print(f"❌ 读取失败")
else:
    print(f"没有索引")
```

---

### 步骤 4: 测试记忆检索

**在终端 33 的调试控制台中输入**:

```python
from zulong.memory.episodic_memory import EpisodicMemory

em = EpisodicMemory()

# 测试不同关键词
test_queries = ["小明", "北京", "参观", "吃的"]

for query in test_queries:
    print(f"\n🔍 检索关键词：'{query}'")
    episodes = await em.search_by_summary(query, top_k=5)
    print(f"   结果数量：{len(episodes)}")
    
    if episodes:
        for ep in episodes:
            print(f"   - Episode {ep['episode_id']}: {ep['summary']}")
```

---

### 步骤 5: 检查注入模型的上下文

**在终端 33 的调试控制台中输入**:

```python
# 检查推理引擎的记忆注入
from zulong.l2.inference_engine import InferenceEngine

# 获取实例（如果存在）
try:
    # 尝试访问正在使用的推理引擎
    import asyncio
    loop = asyncio.get_event_loop()
    
    # 这里需要访问正在运行的推理引擎实例
    # 但由于是异步对象，可能需要特殊方法
    print("需要访问运行中的推理引擎实例")
    
except Exception as e:
    print(f"无法直接访问：{e}")
```

---

## 📊 预期结果

### 正常情况

```
✅ 索引数量：7
✅ 共享池状态：已初始化
✅ 初始化状态：True

📋 索引详情 (前 10 条):
   Episode 1: 小明自我介绍
   Episode 2: 用户住在北京
   Episode 3: 参观故宫
   Episode 4: 故宫很震撼
   Episode 5: 长城人很多
   Episode 6: 北京烤鸭很美味

✅ 读取成功
   用户：我叫小明
   AI:   你好，小明！很高兴认识你。

🔍 检索关键词：'小明'
   结果数量：1
   - Episode 1: 小明自我介绍
```

### 异常情况

```
❌ 索引数量：0
❌ 共享池状态：未初始化
⚠️  索引为空 - 可能没有进行对话或存储失败
```

---

## 🔧 故障排查

### 问题 1: 索引数量为 0

**可能原因**:
- 对话没有触发记忆存储
- 存储失败
- 懒加载未触发

**检查日志**:
在终端 33 中查看输出，搜索：
```
[EpisodicMemory]
```

---

### 问题 2: 共享池未初始化

**可能原因**:
- 初始化超时
- 资源冲突

**解决**:
等待系统自动懒加载，或手动触发：
```python
from zulong.memory.episodic_memory import EpisodicMemory
em = EpisodicMemory()
await em.initialize_async()
```

---

### 问题 3: 检索不到记忆

**可能原因**:
- 摘要生成失败
- 检索关键词不匹配
- 时间窗口太短

**解决**:
```python
# 调整检索参数
episodes = await em.search_by_summary(
    query="名字",
    top_k=10,
    time_window=86400  # 24 小时
)
```

---

## 📝 快速测试命令

**复制粘贴到终端 33 调试控制台**:

### 测试 1: 基本状态
```python
from zulong.memory.episodic_memory import EpisodicMemory; em = EpisodicMemory(); print(f"索引：{len(em._episode_index)}, 共享池：{'OK' if em.pool else 'NG'}")
```

### 测试 2: 存储测试
```python
from zulong.memory.episodic_memory import EpisodicMemory; em = EpisodicMemory(); result = await em.store_episode("测试", "测试回复"); print(f"存储：{result}")
```

### 测试 3: 检索测试
```python
from zulong.memory.episodic_memory import EpisodicMemory; em = EpisodicMemory(); episodes = await em.search_by_summary("测试", top_k=5); print(f"检索：{len(episodes)}条")
```

---

**测试人员**: AI Assistant  
**测试日期**: 2026-04-10  
**状态**: ⏳ **等待用户在运行中的系统测试**
