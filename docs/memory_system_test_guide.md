# 记忆系统生产环境测试流程

**文档版本**: 1.0  
**创建时间**: 2026-04-10  
**适用系统**: Zulong Beta 4  
**测试目标**: 验证记忆摘要生成、分层读取和上下文注入功能

---

## 📋 测试前准备

### 1. 系统状态检查

在开始测试前，请确认以下组件正常运行：

```bash
# 检查 vLLM 服务
curl http://localhost:8000/v1/models
curl http://localhost:8001/v1/models

# 检查祖龙系统
# 查看终端 33 是否显示调试控制台

# 检查 OpenClaw Bridge
# 查看终端 34 是否显示启动成功
```

**预期结果**:
- ✅ L2 CORE vLLM 服务运行在端口 8000
- ✅ L2 BACKUP vLLM 服务运行在端口 8001
- ✅ 祖龙系统调试控制台可用
- ✅ OpenClaw Bridge 已连接

### 2. 访问 Web 界面

打开浏览器访问：`http://localhost:8080`

**检查项目**:
- ✅ 页面正常加载
- ✅ 可以输入消息
- ✅ 能够收到 AI 回复

---

## 🧪 测试流程

### 测试 1: 短期记忆缓存功能

**测试目标**: 验证短期记忆能够正确缓存多轮对话

**测试步骤**:

1. **第一轮对话**
   ```
   用户输入：你好，我叫小明
   ```
   - 观察回复是否正常
   - 记录日志中的 `history_length` 值

2. **第二轮对话**
   ```
   用户输入：我今年 25 岁
   ```
   - 观察 AI 是否记住用户的名字
   - 检查 `history_length` 是否增加到 2

3. **第三轮对话**
   ```
   用户输入：我住在北京
   ```
   - 观察 AI 是否能关联之前的信息
   - 检查 `history_length` 是否增加到 3

4. **第四轮对话**
   ```
   用户输入：今天天气不错
   ```
   - 观察对话连续性
   - 检查 `history_length` 是否增加到 4

**验证标准**:
- ✅ 每轮对话都有正常回复
- ✅ `history_length` 逐轮递增
- ✅ AI 能够记住用户之前提到的信息（名字、年龄、地点）

**日志检查位置**:
```
终端 33 中搜索：history_length
预期看到：history_length: 1, 2, 3, 4...
```

**记录结果**:
```
第 1 轮：history_length = ___ (预期：1)
第 2 轮：history_length = ___ (预期：2)
第 3 轮：history_length = ___ (预期：3)
第 4 轮：history_length = ___ (预期：4)

AI 是否记住用户信息：是 / 否
对话是否连续：是 / 否
```

---

### 测试 2: 记忆摘要生成功能

**测试目标**: 验证系统能够为对话生成摘要

**测试步骤**:

1. **创建有意义的对话场景**
   ```
   用户：我昨天去了故宫参观
   AI:   故宫是北京著名的景点，你觉得怎么样？
   用户：非常震撼，建筑很宏伟
   AI:   是的，故宫是中国古代宫殿建筑的精华
   用户：我还去了长城，人很多
   AI:   长城是世界奇迹之一，值得参观
   ```

2. **等待系统生成摘要**
   - 系统会在后台自动为这段对话生成摘要
   - 查看终端 33 的日志

3. **检查摘要生成日志**
   ```bash
   # 在终端 33 中查找以下关键词：
   - "生成摘要"
   - "摘要模型"
   - "EpisodicMemory"
   ```

**预期日志**:
```
[episodic_memory] 已获取共享池单例：xxx
[episodic_memory] 摘要模型：使用规则生成（轻量级方案）
[episodic_memory] 成功存储情景记忆：episode_id=xxx
```

**验证标准**:
- ✅ 看到摘要模型初始化日志
- ✅ 看到情景记忆存储日志
- ✅ 没有错误信息

**手动验证摘要内容** (可选):
```python
# 在终端 33 的调试控制台中输入：
from zulong.memory.episodic_memory import EpisodicMemory
em = EpisodicMemory()
print(f"当前情景记忆数：{len(em._episode_index)}")
for eid, meta in em._episode_index.items():
    print(f"Episode {eid}: {meta.get('summary', 'N/A')[:100]}")
```

**记录结果**:
```
摘要生成日志：有 / 无
情景记忆数量：___
摘要内容示例：_________________________
```

---

### 测试 3: 分层读取记忆功能

**测试目标**: 验证能够通过摘要检索并读取完整对话

**测试步骤**:

#### Level 1: 摘要检索测试

1. **在调试控制台中执行** (终端 33):
   ```python
   from zulong.memory.episodic_memory import EpisodicMemory
   import asyncio
   
   em = EpisodicMemory()
   
   # 检索与"北京"相关的记忆
   results = asyncio.run(em.retrieve_by_query("北京", top_k=2))
   
   print(f"检索到 {len(results)} 条结果")
   for i, r in enumerate(results, 1):
       print(f"\n结果 {i}:")
       print(f"  摘要：{r.get('summary', 'N/A')}")
       print(f"  分数：{r.get('score', 'N/A')}")
       print(f"  Trace ID: {r.get('trace_id', 'N/A')}")
   ```

2. **记录检索结果**
   ```
   检索到 ___ 条结果
   
   结果 1:
     摘要：____________________
     分数：____________________
   
   结果 2:
     摘要：____________________
   ```

#### Level 2: 完整对话读取测试

1. **使用上一步获取的 trace_id**:
   ```python
   # 假设上一步获取到 trace_id
   trace_id = "从上一步结果中复制"
   
   # 读取完整对话
   full_dialogue = asyncio.run(em.read_full_episode(trace_id))
   
   print(f"\n完整对话内容:")
   for msg in full_dialogue:
       role = "用户" if msg["role"] == "user" else "AI"
       print(f"{role}: {msg['content']}")
   ```

2. **验证读取结果**
   ```
   是否成功读取：是 / 否
   对话内容完整：是 / 否
   对话轮数：___
   ```

**验证标准**:
- ✅ 能够检索到相关摘要
- ✅ 摘要包含关键词"北京"
- ✅ 能够通过 trace_id 读取完整对话
- ✅ 完整对话内容与原始输入一致

**记录结果**:
```
Level 1 检索：成功 / 失败
检索结果数量：___
Level 2 读取：成功 / 失败
对话完整性：完整 / 部分 / 丢失
```

---

### 测试 4: 上下文注入连续性测试

**测试目标**: 验证多轮对话中上下文正确注入

**测试步骤**:

1. **设计连续性对话场景**
   ```
   第 1 轮:
   用户：我喜欢吃苹果
   AI:   苹果是很有营养的水果，富含维生素
   
   第 2 轮:
   用户：香蕉也不错
   AI:   香蕉富含钾元素，对心脏很好
   
   第 3 轮:
   用户：那你喜欢吃什么水果
   AI:   [应该能提到苹果或香蕉，因为之前对话中用户喜欢这些]
   
   第 4 轮:
   用户：我明天想去买水果
   AI:   [应该能结合用户之前的喜好给出建议]
   ```

2. **观察 AI 回复的连贯性**
   - 第 3 轮 AI 是否提到用户之前说过的水果？
   - 第 4 轮 AI 是否记住用户的喜好？

3. **检查日志中的上下文注入**
   ```bash
   # 在终端 33 中查找：
   - "Short-term memory"
   - "get_recent"
   - "history_length"
   ```

**预期日志**:
```
[short_term_memory] 成功读取最近对话：X 轮
[inference_engine] 构建 messages: X 条
[websocket_server] history_length: X
```

**验证标准**:
- ✅ AI 在第 3 轮提到用户之前的喜好
- ✅ AI 在第 4 轮给出个性化建议
- ✅ 日志显示正确的 history_length
- ✅ 对话自然流畅，不突兀

**记录结果**:
```
第 3 轮是否提及历史：是 / 否
第 4 轮是否个性化：是 / 否
history_length 显示：___
上下文注入：成功 / 失败
```

---

### 测试 5: 压力测试 - 超过记忆容量

**测试目标**: 验证短期记忆的容量限制和过期机制

**测试步骤**:

1. **连续进行 25 轮对话** (超过 max_rounds=20):
   ```
   用户：这是第 1 句话
   用户：这是第 2 句话
   ...
   用户：这是第 25 句话
   ```

2. **每 5 轮检查一次 history_length**:
   ```
   第 5 轮后：history_length = ___ (预期：5)
   第 10 轮后：history_length = ___ (预期：10)
   第 15 轮后：history_length = ___ (预期：15)
   第 20 轮后：history_length = ___ (预期：20)
   第 25 轮后：history_length = ___ (预期：20，因为 max=20)
   ```

3. **验证最早的记忆是否被遗忘**:
   ```
   询问 AI："我最开始说的是什么？"
   AI 应该无法准确回答（因为已被遗忘）
   ```

**验证标准**:
- ✅ history_length 不超过 20
- ✅ 超过 20 轮后，最早的记忆被移除
- ✅ 系统运行稳定，没有崩溃

**记录结果**:
```
第 25 轮 history_length: ___ (预期：20)
最早记忆是否被遗忘：是 / 否
系统稳定性：稳定 / 不稳定
```

---

### 测试 6: 记忆持久化测试

**测试目标**: 验证记忆数据是否正确持久化

**测试步骤**:

1. **进行几轮对话并记录内容**:
   ```
   用户：测试持久化功能
   AI:   好的，我明白了
   用户：记住这个测试
   AI:   我会记住的
   ```

2. **重启祖龙系统**:
   ```bash
   # 在终端 33 按 Ctrl+C 停止系统
   # 然后重新启动：
   $env:USE_VLLM_FOR_L2="true"
   cd d:\AI\project\zulong_beta4
   .\zulong_env\Scripts\activate
   python -m zulong.bootstrap
   ```

3. **检查记忆恢复**:
   ```python
   # 在调试控制台中执行：
   from zulong.memory.short_term_memory import ShortTermMemory
   import asyncio
   
   stm = asyncio.run(ShortTermMemory.get_instance())
   recent = asyncio.run(stm.get_recent(rounds=5))
   
   print(f"恢复的对话轮数：{len(recent)}")
   for conv in recent:
       print(f"用户：{conv.get('input', 'N/A')}")
       print(f"AI: {conv.get('output', 'N/A')}")
   ```

**验证标准**:
- ✅ 系统重启后能够恢复部分记忆
- ✅ 持久化文件存在：`data\short_term_memory`
- ⚠️  如果无法恢复，记录错误信息（已知问题：索引恢复功能待完善）

**记录结果**:
```
重启前对话轮数：___
重启后恢复轮数：___
持久化文件：存在 / 不存在
恢复成功：是 / 否
```

---

## 📊 测试结果汇总

### 测试完成情况

| 测试编号 | 测试项目 | 结果 | 备注 |
|---------|---------|------|------|
| 测试 1 | 短期记忆缓存 | ☐ 通过 / ☐ 失败 | |
| 测试 2 | 记忆摘要生成 | ☐ 通过 / ☐ 失败 | |
| 测试 3 | 分层读取记忆 | ☐ 通过 / ☐ 失败 | |
| 测试 4 | 上下文注入 | ☐ 通过 / ☐ 失败 | |
| 测试 5 | 压力测试 | ☐ 通过 / ☐ 失败 | |
| 测试 6 | 持久化测试 | ☐ 通过 / ☐ 失败 | |

### 关键指标记录

| 指标 | 预期值 | 实际值 | 状态 |
|------|--------|--------|------|
| 最大短期记忆轮数 | 20 | ___ | ☐ 正常 / ☐ 异常 |
| 摘要生成速度 | < 1 秒 | ___ 秒 | ☐ 正常 / ☐ 异常 |
| 检索响应时间 | < 500ms | ___ ms | ☐ 正常 / ☐ 异常 |
| 上下文注入成功率 | 100% | ___% | ☐ 正常 / ☐ 异常 |
| 持久化恢复率 | > 80% | ___% | ☐ 正常 / ☐ 异常 |

### 发现的问题

**问题 1**:
```
描述：_______________________________
严重程度：高 / 中 / 低
复现步骤：___________________________
```

**问题 2**:
```
描述：_______________________________
严重程度：高 / 中 / 低
复现步骤：___________________________
```

### 总体评价

```
测试通过率：___/6 (___%)

系统稳定性：⭐⭐⭐⭐⭐ (5 星为最佳)
功能完整性：⭐⭐⭐⭐⭐
性能表现：⭐⭐⭐⭐⭐

是否达到生产标准：是 / 否
```

---

## 🔧 故障排查指南

### 问题：history_length 始终为 0

**可能原因**:
1. 短期记忆未正确初始化
2. 共享池连接失败

**解决方法**:
```bash
# 检查日志中是否有以下错误：
- "Short-term memory initialized"
- "共享池单例"

# 重启系统
```

### 问题：摘要生成失败

**可能原因**:
1. 情景记忆模块未加载
2. 共享池写入失败

**解决方法**:
```bash
# 检查日志：
- "EpisodicMemory initialized"
- "异步复盘工作线程已启动"

# 在调试控制台中手动测试：
from zulong.memory.episodic_memory import EpisodicMemory
em = EpisodicMemory()
print(em._initialized)
```

### 问题：检索结果为空

**可能原因**:
1. 索引未建立
2. 关键词不匹配

**解决方法**:
```bash
# 检查索引：
from zulong.memory.episodic_memory import EpisodicMemory
em = EpisodicMemory()
print(f"索引数量：{len(em._episode_index)}")

# 尝试不同关键词检索
```

---

## 📝 附录

### A. 常用调试命令

```python
# 1. 查看短期记忆状态
from zulong.memory.short_term_memory import ShortTermMemory
import asyncio
stm = asyncio.run(ShortTermMemory.get_instance())
print(stm.get_status())

# 2. 查看情景记忆索引
from zulong.memory.episodic_memory import EpisodicMemory
em = EpisodicMemory()
print(f"情景记忆数：{len(em._episode_index)}")

# 3. 手动生成摘要
dialogue = [
    {"role": "user", "content": "测试内容"},
    {"role": "assistant", "content": "回复内容"}
]
summary = asyncio.run(em.generate_summary(dialogue))
print(f"摘要：{summary}")

# 4. 测试检索
results = asyncio.run(em.retrieve_by_query("测试", top_k=5))
print(f"检索结果：{len(results)}")
```

### B. 日志文件位置

```
终端 33: 祖龙系统运行日志
终端 34: OpenClaw Bridge 日志
data\short_term_memory: 短期记忆持久化文件
```

### C. 联系支持

如遇到问题，请提供以下信息：
1. 测试步骤和结果记录
2. 相关日志片段
3. 系统版本信息

---

**测试人员**: ___________  
**测试日期**: ___________  
**审核人员**: ___________  
**审核日期**: ___________
