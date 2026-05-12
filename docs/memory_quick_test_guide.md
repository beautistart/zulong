# 记忆系统测试快速指南

**版本**: 1.0  
**更新时间**: 2026-04-10  
**适用环境**: 生产环境

---

## 🚀 快速开始

### 方法 1: 使用自动化测试脚本（推荐）

这是最简单快速的测试方法，适合首次验证系统功能。

**步骤**:

1. **打开调试控制台**
   - 在祖龙系统运行的终端（终端 33）中
   - 确保系统正在运行

2. **运行测试脚本**
   ```bash
   python scripts\quick_memory_test.py
   ```

3. **查看结果**
   - 脚本会自动运行 6 个测试
   - 显示每个测试的通过/失败状态
   - 输出详细的测试指标

**预期输出示例**:
```
================================================================================
记忆系统快速测试套件
================================================================================

================================================================================
测试 1: 短期记忆基础功能
================================================================================
[PASS] 存储和读取
      '今天天气真好' -> '是的，阳光明媚，适合外出'
[PASS] 状态查询
      轮数：1

================================================================================
测试 2: 多轮对话（5 轮）
================================================================================
存储 5 轮对话...
  [1/5] 已存储
  [2/5] 已存储
  ...
[PASS] 对话轮数
      5/5

... (更多测试)

================================================================================
测试结果汇总
================================================================================
[PASS] 短期记忆基础
[PASS] 多轮对话
[PASS] 摘要生成
[PASS] 分层读取
[PASS] 上下文注入
[PASS] 容量限制

总计：6/6 通过 (100.0%)

恭喜！所有测试通过，记忆系统运行正常
```

---

### 方法 2: 手动交互式测试

适合深入测试和调试特定功能。

#### 测试 1: 验证短期记忆

**在调试控制台中执行**:

```python
from zulong.memory.short_term_memory import ShortTermMemory
import asyncio

# 获取单例
stm = asyncio.run(ShortTermMemory.get_instance())

# 存储对话
asyncio.run(stm.store("你好，我叫小明", "你好小明，很高兴认识你"))
asyncio.run(stm.store("我今年 25 岁", "25 岁是很好的年纪"))
asyncio.run(stm.store("我住在北京", "北京是个很棒的城市"))

# 读取最近对话
recent = asyncio.run(stm.get_recent(rounds=3))

print(f"当前记忆轮数：{len(recent)}")
for i, conv in enumerate(recent, 1):
    print(f"{i}. 用户：{conv.get('input')}")
    print(f"   AI: {conv.get('output')}")

# 查看状态
status = stm.get_status()
print(f"\n状态：{status}")
```

**预期结果**:
```
当前记忆轮数：3
1. 用户：你好，我叫小明
   AI: 你好小明，很高兴认识你
2. 用户：我今年 25 岁
   AI: 25 岁是很好的年纪
3. 用户：我住在北京
   AI: 北京是个很棒的城市

状态：{'current_round': 3, 'zone': 'memory', ...}
```

---

#### 测试 2: 测试摘要生成

**在调试控制台中执行**:

```python
from zulong.memory.episodic_memory import EpisodicMemory
import asyncio

em = EpisodicMemory()

# 测试对话
dialogue = [
    {"role": "user", "content": "我昨天去了故宫参观"},
    {"role": "assistant", "content": "故宫是北京著名的景点"},
    {"role": "user", "content": "非常震撼，建筑很宏伟"},
    {"role": "assistant", "content": "故宫是中国古代宫殿建筑的精华"}
]

# 生成摘要
summary = asyncio.run(em.generate_summary(dialogue))

print(f"生成的摘要：{summary}")
print(f"摘要长度：{len(summary)} 字符")

# 存储情景记忆
episode_id = asyncio.run(em.store_episode(dialogue, summary, tags=["travel", "beijing"]))
print(f"存储成功，episode_id: {episode_id}")
```

**预期结果**:
```
生成的摘要：用户昨天去了故宫参观，非常震撼，建筑很宏伟
摘要长度：XX 字符
存储成功，episode_id: xxx
```

---

#### 测试 3: 测试分层读取

**在调试控制台中执行**:

```python
from zulong.memory.episodic_memory import EpisodicMemory
import asyncio

em = EpisodicMemory()

# Level 1: 摘要检索
print("Level 1: 摘要检索")
results = asyncio.run(em.retrieve_by_query("故宫", top_k=2))

print(f"检索到 {len(results)} 条结果")
for i, r in enumerate(results, 1):
    print(f"\n结果 {i}:")
    print(f"  摘要：{r.get('summary', 'N/A')}")
    print(f"  分数：{r.get('score', 'N/A')}")
    print(f"  Trace ID: {r.get('trace_id', 'N/A')}")

# Level 2: 完整对话读取
if results and len(results) > 0:
    trace_id = results[0].get('trace_id')
    if trace_id:
        print(f"\nLevel 2: 读取完整对话 (trace_id={trace_id})")
        full_dialogue = asyncio.run(em.read_full_episode(trace_id))
        
        if full_dialogue:
            print(f"完整对话 ({len(full_dialogue)} 条消息):")
            for msg in full_dialogue:
                role = "用户" if msg["role"] == "user" else "AI"
                print(f"{role}: {msg['content']}")
```

**预期结果**:
```
Level 1: 摘要检索
检索到 1 条结果

结果 1:
  摘要：用户昨天去了故宫参观...
  分数：0.85
  Trace ID: xxx

Level 2: 读取完整对话 (trace_id=xxx)
完整对话 (4 条消息):
用户：我昨天去了故宫参观
AI: 故宫是北京著名的景点
用户：非常震撼，建筑很宏伟
AI: 故宫是中国古代宫殿建筑的精华
```

---

#### 测试 4: 测试上下文注入

**在调试控制台中执行**:

```python
from zulong.memory.short_term_memory import ShortTermMemory
import asyncio

stm = asyncio.run(ShortTermMemory.get_instance())

# 清空现有记忆（可选）
# asyncio.run(stm.clear())

# 构建上下文
context = [
    ("我喜欢吃苹果", "苹果是很有营养的水果"),
    ("香蕉也不错", "香蕉富含钾元素"),
    ("橙子也很好", "橙子富含维生素 C")
]

print("构建上下文场景...")
for user_input, ai_output in context:
    asyncio.run(stm.store(user_input, ai_output))
    print(f"  已存储：'{user_input}'")

# 读取上下文并构建 messages
recent = asyncio.run(stm.get_recent(rounds=3))

messages = [{"role": "system", "content": "你是一个助手"}]
for conv in recent:
    if conv.get('input'):
        messages.append({"role": "user", "content": conv['input']})
    if conv.get('output'):
        messages.append({"role": "assistant", "content": conv['output']})

messages.append({"role": "user", "content": "推荐一些水果"})

print(f"\n构建的 messages ({len(messages)} 条):")
for i, msg in enumerate(messages, 1):
    role = msg["role"]
    content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
    print(f"{i}. [{role}] {content}")
```

**预期结果**:
```
构建上下文场景...
  已存储：'我喜欢吃苹果'
  已存储：'香蕉也不错'
  已存储：'橙子也很好'

构建的 messages (7 条):
1. [system] 你是一个助手
2. [user] 我喜欢吃苹果
3. [assistant] 苹果是很有营养的水果
4. [user] 香蕉也不错
5. [assistant] 香蕉富含钾元素
6. [user] 橙子也很好
7. [assistant] 橙子富含维生素 C
8. [user] 推荐一些水果
```

---

## 📋 测试检查清单

完成测试后，请确认以下项目：

### 基础功能测试
- [ ] 短期记忆能够存储对话
- [ ] 短期记忆能够读取对话
- [ ] 状态查询正常工作

### 多轮对话测试
- [ ] 能够存储 5 轮以上对话
- [ ] 能够读取完整的多轮对话
- [ ] 对话内容完整无误

### 摘要功能测试
- [ ] 能够为对话生成摘要
- [ ] 摘要包含关键信息
- [ ] 摘要长度合理（10-200 字符）

### 分层读取测试
- [ ] 能够通过关键词检索摘要
- [ ] 检索结果包含 trace_id
- [ ] 能够通过 trace_id 读取完整对话

### 上下文注入测试
- [ ] 能够正确读取历史对话
- [ ] 能够构建包含上下文的 messages
- [ ] 上下文顺序正确（时间顺序）

### 容量限制测试
- [ ] 短期记忆不超过 20 轮
- [ ] 超过容量后旧记忆被遗忘
- [ ] 新记忆能够正常存储

---

## 🔍 常见问题排查

### Q1: 测试脚本报错 "ImportError"

**解决方法**:
```bash
# 确保已激活虚拟环境
.\zulong_env\Scripts\activate

# 或者使用完整路径
d:\AI\project\zulong_beta4\zulong_env\Scripts\python.exe scripts\quick_memory_test.py
```

### Q2: 短期记忆无法存储

**检查项目**:
1. 共享池是否正常运行
2. 查看日志中是否有 "共享池单例" 相关错误
3. 重启祖龙系统

**调试命令**:
```python
from zulong.infrastructure.shared_memory_pool import SharedMemoryPool
pool = SharedMemoryPool.get_instance()
print(f"共享池状态：{pool}")
```

### Q3: 摘要生成结果为空

**可能原因**:
- 对话内容为空
- 摘要模型未正确初始化

**解决方法**:
```python
from zulong.memory.episodic_memory import EpisodicMemory
em = EpisodicMemory()
print(f"初始化状态：{em._initialized}")
print(f"摘要模型：{em.summary_model}")
```

### Q4: 检索结果为空

**可能原因**:
- 索引未建立
- 关键词不匹配

**解决方法**:
```python
# 检查索引
from zulong.memory.episodic_memory import EpisodicMemory
em = EpisodicMemory()
print(f"索引数量：{len(em._episode_index)}")

# 尝试不同关键词
results = asyncio.run(em.retrieve_by_query("测试", top_k=5))
print(f"检索结果：{len(results)}")
```

---

## 📊 性能基准

以下是正常情况下的性能指标，可用于对比：

| 操作 | 预期时间 | 说明 |
|------|---------|------|
| 存储单轮对话 | < 100ms | 写入共享池 |
| 读取最近对话 | < 50ms | 内存读取 |
| 生成摘要 | < 500ms | 规则提取 |
| 摘要检索 | < 100ms | 内存索引 |
| 完整对话读取 | < 200ms | 从共享池读取 |

---

## 📝 测试报告模板

完成测试后，填写以下报告：

```
测试日期：YYYY-MM-DD HH:MM
测试人员：[姓名]
测试环境：生产环境

自动化测试结果：
- 短期记忆基础：[PASS/FAIL]
- 多轮对话：[PASS/FAIL]
- 摘要生成：[PASS/FAIL]
- 分层读取：[PASS/FAIL]
- 上下文注入：[PASS/FAIL]
- 容量限制：[PASS/FAIL]

总计：X/6 通过 (XX%)

手动测试结果:
- 短期记忆：[PASS/FAIL]
- 摘要生成：[PASS/FAIL]
- 分层读取：[PASS/FAIL]
- 上下文注入：[PASS/FAIL]

发现的问题:
1. [问题描述]
2. [问题描述]

总体评价:
[系统是否正常运行，是否达到生产标准]
```

---

## 📞 技术支持

如遇到问题，请提供以下信息：
1. 测试报告
2. 错误日志
3. 系统版本信息

---

**文档维护**: AI Assistant  
**最后更新**: 2026-04-10
