# 祖龙系统架构与记忆板块深度分析报告

**文档版本**: v1.0  
**分析日期**: 2026-04-12  
**分析对象**: 祖龙 Beta4 系统  

---

## 📋 目录

1. [系统整体架构](#1-系统整体架构)
2. [记忆系统架构](#2-记忆系统架构)
3. [核心记忆模块详解](#3-核心记忆模块详解)
4. [记忆注入机制](#4-记忆注入机制)
5. [身份认知系统](#5-身份认知系统)
6. [关键修复与优化](#6-关键修复与优化)
7. [总结与建议](#7-总结与建议)

---

## 1. 系统整体架构

### 1.1 分层架构设计

祖龙系统采用**分层式架构**，从底层到高层依次为：

```
┌─────────────────────────────────────────┐
│          L3: 专家系统层                  │
│  (Vision/Motor/TTS Expert Nodes)        │
├─────────────────────────────────────────┤
│          L2: 认知推理层                  │
│  (Inference Engine, RAG, Planning)      │
├─────────────────────────────────────────┤
│          L1: 感知运动层                  │
│  (Vision/Audio/Motor Processing)        │
├─────────────────────────────────────────┤
│          L0: 设备抽象层                  │
│  (Camera, Microphone, Speaker)          │
└─────────────────────────────────────────┘
```

### 1.2 核心基础设施

**数据统一共享池 (Shared Memory Pool)** - TSD v2.5 核心架构

所有数据通过统一的共享池进行管理和交换：

- **RAW Zone**: 原始数据（视频流、音频流、原始文本）
- **FEATURE Zone**: 结构化特征（JSON、提取的特征）
- **SYSTEM Zone**: 系统日志、状态信息
- **MEMORY Zone**: 对话历史、上下文快照

**数据信封 (DataEnvelope)** 结构：
```python
@dataclass
class DataEnvelope:
    trace_id: str                    # 全局唯一追踪 ID
    timestamp: float                 # 纳秒级时间戳
    data_type: DataType              # 数据类型
    zone: ZoneType                   # 存储分区
    payload: Any                     # 原始数据
    metadata: Dict[str, Any]         # 附加信息
```

### 1.3 启动流程

1. **L0 层初始化**: 设备模拟器、传感器、执行器
2. **L1 层初始化**: 感知处理器、运动控制器
3. **L2 层初始化**: 推理引擎、RAG 管理器、记忆系统
4. **WebSocket 服务**: 启动 EventBus (端口 5555)
5. **OpenClaw 桥接器**: 连接外部系统

---

## 2. 记忆系统架构

### 2.1 记忆系统分层模型

祖龙记忆系统采用**三层记忆模型**：

```
┌──────────────────────────────────────┐
│      长期记忆 (Long-term Memory)     │
│  - 经验库 (Experience RAG)           │
│  - 知识库 (Knowledge RAG)            │
│  - 技能库 (Skill Store)              │
├──────────────────────────────────────┤
│      临时记忆 (Episodic Memory)      │
│  - 对话摘要 (50-100 字)              │
│  - 基于摘要检索                       │
│  - 时间窗口管理                       │
├──────────────────────────────────────┤
│      短期记忆 (Short-term Memory)    │
│  - 最近对话缓存 (内存)               │
│  - 向量缓存 (Vector Cache)           │
│  - 快速读写                          │
└──────────────────────────────────────┘
```

### 2.2 三库分立架构 (TSD v2.2)

祖龙记忆系统包含三个独立的 RAG 库：

| 库名称 | 存储内容 | 访问速度 | 持久化 |
|--------|---------|---------|--------|
| **技能库** | 通用能力、系统指令、安全规则 | 0ms (内存) | 代码硬编码 |
| **经验库** | 成功模式、失败教训、用户偏好 | <50ms | 向量数据库 |
| **知识库** | 事实性知识、文档资料 | <100ms | 向量数据库 |

### 2.3 记忆自进化机制

记忆系统具备**自组织、自优化**能力：

**记忆强度模型**：
```python
@dataclass
class MemoryStrength:
    initial_strength: float = 1.0      # 初始强度
    current_strength: float = 1.0      # 当前强度
    decay_rate: float = 0.1            # 衰减率
    last_access_time: float            # 最后访问时间
    access_count: int = 0              # 访问次数
    emotional_weight: float = 1.0      # 情感权重
    level: str = "L1"                  # 记忆层级 (L1/L2/L3)
```

**艾宾浩斯遗忘曲线**：
- 记忆强度随时间衰减：`R = e^(-t/S)`
- 重复访问增强强度：`frequency_boost = log(access_count + 1) * 0.1`
- 最终强度：`strength = retention × (1 + frequency_boost) × emotional_weight`

---

## 3. 核心记忆模块详解

### 3.1 短期记忆 (ShortTermMemory)

**文件**: [`zulong/memory/short_term_memory.py`](d:\AI\project\zulong_beta4\zulong\memory\short_term_memory.py)

**核心功能**：
1. 内存中缓存最近 N 轮对话（默认 100 轮）
2. 自动从共享池读取关联的感知上下文
3. 支持快速异步读写
4. 向量缓存加速检索

**数据结构**：
```python
class ShortTermMemory:
    max_rounds: int = 100              # 最大保留轮数
    ttl_seconds: int = 3600            # 过期时间
    
    conversation_history: List[Dict]   # 对话历史
    vector_cache: VectorCache          # 向量缓存
```

**关键方法**：
- `store(user_input, ai_response)`: 存储对话到共享池
- `get_recent(n)`: 获取最近 N 轮对话
- `search_similar(query, top_k)`: 语义检索相关对话

### 3.2 临时记忆 (EpisodicMemory)

**文件**: [`zulong/memory/episodic_memory.py`](d:\AI\project\zulong_beta4\zulong\memory\episodic_memory.py)

**核心功能**：
1. **对话摘要**: 为每轮对话生成 50-100 字摘要
2. **基于摘要检索**: 快速找到相关对话
3. **分级读取**: 
   - Level 1: 摘要（快速浏览）
   - Level 2: 完整对话（按需读取）
4. **时间窗口管理**: 支持按时间范围检索

**工作流程**：
```
新对话 → 生成摘要 → 存储到共享池
          ↓
检索请求 → 基于摘要检索 → 返回 Top-K 摘要
          ↓
详细读取 → 根据 trace_id → 读取完整对话
```

### 3.3 人物画像系统 (PersonProfile)

**文件**: [`zulong/memory/person_profile.py`](d:\AI\project\zulong_beta4\zulong\memory\person_profile.py)

**多模态人物识别**：
- **对话画像**: 从对话中提取人物特征（名字、年龄、职业等）
- **人脸特征**: 存储人脸编码向量（128 维/512 维）
- **声纹特征**: 存储声纹嵌入向量
- **多模态融合**: 综合文本/人脸/声音进行人物识别

**数据结构**：
```python
@dataclass
class PersonProfile:
    person_id: str                     # 唯一人物 ID
    name: Optional[str]                # 名字
    attributes: Dict[str, Any]         # 基础信息
    face_features: List[FaceFeature]   # 人脸特征
    voice_features: List[VoiceFeature] # 声纹特征
    dialogue_style: Dict[str, Any]     # 对话特征
    interaction_count: int             # 交互次数
```

### 3.4 经验生成器 (ExperienceGenerator)

**文件**: [`zulong/memory/experience_generator.py`](d:\AI\project\zulong_beta4\zulong\memory\experience_generator.py)

**自动经验提取**：
1. 从对话历史中提取成功模式
2. 从错误日志中提取失败教训
3. 从用户反馈中提取偏好
4. 自动分类并添加到经验库

**模式库**：
```python
success_patterns = [
    r"成功.*", r"完成.*", r"解决了.*", r"正确.*",
    r"太好了.*", r"谢谢.*", r"非常好.*", r"完美.*",
]

failure_patterns = [
    r"错误.*", r"失败.*", r"不正确.*", r"有问题.*",
    r"不行.*", r"错误：.*", r"Exception.*", r"Failed.*",
]
```

---

## 4. 记忆注入机制

### 4.1 智能记忆检索策略

**文件**: [`zulong/l2/inference_engine.py`](d:\AI\project\zulong_beta4\zulong\l2\inference_engine.py#L1620-L1680)

**第 3 轮起的双重检索策略**：

```
┌─────────────────────────────────────────┐
│  当前用户输入                            │
└─────────────────────────────────────────┘
           ↓
    ┌──────────────┐
    │ 语义检索     │
    └──────────────┘
           ↓
    ┌──────────────┬──────────────┐
    │              │              │
    ↓              ↓              ↓
【工作记忆】   【前 2 轮检索】  【全局检索】
最近 2 轮对话   Top-1 相关     Top-1 相关
(完整注入)    (向量化)       (向量化)
```

### 4.2 记忆注入流程

**步骤 1: 工作记忆注入**
```python
# 获取最近 2 轮对话
recent_turns = 2
recent_history = conversation_history[-recent_turns * 2:]
# 直接添加到 messages 中
```

**步骤 2: 前 2 轮向量化检索**
```python
# 检索前 2 轮的向量化内容
relevant_recent = await short_term_memory.search_similar(
    query=user_input,
    top_k=1,
    use_vector_cache=True
)
# 如果相关，注入到 system prompt
```

**步骤 3: 全局向量检索**
```python
# 检索全局最相关的对话
relevant_memories = await short_term_memory.search_similar(
    query=user_input,
    top_k=1,
    use_vector_cache=True
)
# 避免重复注入
```

### 4.3 记忆注入示例

**System Prompt 结构**：
```
你是祖龙 (ZULONG) 机器人助手...

【重要身份认知】
- 你的名字是'祖龙'
- 用户是和你对话的人类...

【工作记忆】(最近 2 轮)
用户：你好
AI: 你好呀！

【前 2 轮相关对话】(向量检索)
用户曾问：今天天气如何
你回答：今天天气很好...

【历史对话】(全局向量检索)
用户曾问：你喜欢什么
你回答：我喜欢...

当前用户输入：你还记得我吗？
```

---

## 5. 身份认知系统

### 5.1 身份认知指令

**最新配置** (2026-04-12 更新)：

```python
"**重要身份认知**：",
"- 你的名字是'祖龙'",
"- 用户是和你对话的人类，你要记住用户的信息",
"- 当用户说'我叫 XXX'时，XXX 是用户的名字，不是你的名字",
"**重要：理解对话历史**：",
"- 下面的对话历史中，'user' 是用户说的话，'assistant' 是你（祖龙）说的话",
"- 用户说'我叫小明'，意思是用户的名字是小明",
"- 你说'我叫祖龙'，意思是你（AI）的名字是祖龙",
"- 一定要分清楚'你'（用户）和'我'（祖龙）的指代关系",
```

### 5.2 人称代词理解

**关键指令**：
- 用'你'或'您'称呼用户
- 用'我'称呼自己（祖龙）
- 分清楚'你'（用户）和'我'（祖龙）的指代关系

**示例教学**：
- 用户："我叫小明" → AI："你好小明，我记住了，你叫小明"
- 用户："你叫什么" → AI："我叫祖龙"
- 用户："我 25 岁" → AI："25 岁是很好的年纪，我记住了你今年 25 岁"

### 5.3 身份认知测试

**测试用例**：
1. **基础身份认知**: 问"你叫什么名字？" → 应回答"我叫祖龙"
2. **用户名字记忆**: 说"我叫小明"，问"我叫什么名字？" → 应回答"你叫小明"
3. **人称理解**: 说"我 25 岁"，问"你今年多大？" → AI 没有年龄

---

## 6. 关键修复与优化

### 6.1 身份认知修复 (2026-04-12)

**问题**：
- AI 混淆用户名字和自己的名字
- 人称代词使用错误
- 记忆保持能力差

**修复方案**：
1. 强化身份认知指令
2. 精简指令内容，去除冗余
3. 添加明确的人称代词定义
4. 提供示例教学

**代码变更**：
```python
# 修改前
"- 你的名字是'祖龙'，你永远是祖龙，不会变成其他人"
"- 当用户说'我 XXX 岁'时，XXX 是用户的年龄，不是你的年龄"
"- 当用户说'我住在 XXX'时，XXX 是用户的住址，不是你的住址"

# 修改后
"- 你的名字是'祖龙'"
"- 当用户说'我叫 XXX'时，XXX 是用户的名字，不是你的名字"
```

### 6.2 记忆注入优化

**优化策略**：
1. **减少工作记忆**: 从 3 轮减少到 2 轮，节省 Token
2. **智能检索**: 仅当语义相关时才注入历史对话
3. **避免污染**: 不相关的历史对话不注入
4. **分级读取**: 模型可根据摘要主动读取详细内容

### 6.3 异步化改造

**TSD v2.5 纯异步架构**：
- 使用 `asyncio.Lock` 替代 `threading.Lock`
- 所有记忆操作支持 `async/await`
- 与 `DataIngestion` 完全异步集成

**示例**：
```python
# 异步存储
success = await self.short_term_memory.store(
    user_input=user_input,
    ai_response=ai_response,
    metadata={"source": "inference_engine"}
)

# 异步检索
relevant_memories = await self.short_term_memory.search_similar(
    query=user_input,
    top_k=1,
    use_vector_cache=True
)
```

---

## 7. 总结与建议

### 7.1 系统架构优势

1. **分层清晰**: L0-L3 分层，职责明确
2. **模块化设计**: 各模块独立，易于维护和扩展
3. **共享池架构**: 统一数据管理，避免数据孤岛
4. **异步优先**: 纯异步设计，高性能
5. **记忆进化**: 自组织、自优化，越用越聪明

### 7.2 记忆系统特色

1. **三层记忆模型**: 短期→临时→长期，符合人类记忆规律
2. **三库分立**: 技能/经验/知识独立管理
3. **向量化**: 所有记忆支持语义检索
4. **自进化**: 基于艾宾浩斯曲线的强度管理
5. **多模态**: 支持文本/人脸/声音多模态记忆

### 7.3 当前优化重点

1. **身份认知**: 已精简指令，需测试效果
2. **记忆注入**: 已优化策略，避免上下文污染
3. **异步改造**: 已完成 TSD v2.5 纯异步架构
4. **经验生成**: 自动从对话中提取经验

### 7.4 后续建议

1. **加强测试**: 创建身份认知测试集，定期验证
2. **监控指标**: 添加记忆检索命中率、延迟监控
3. **性能优化**: 向量缓存命中率优化
4. **用户反馈**: 收集用户对记忆能力的反馈
5. **文档完善**: 持续更新架构文档和 API 文档

---

## 📚 附录：核心文件索引

### 记忆系统核心文件
- [`short_term_memory.py`](d:\AI\project\zulong_beta4\zulong\memory\short_term_memory.py) - 短期记忆
- [`episodic_memory.py`](d:\AI\project\zulong_beta4\zulong\memory\episodic_memory.py) - 临时记忆
- [`rag_manager.py`](d:\AI\project\zulong_beta4\zulong\memory\rag_manager.py) - RAG 管理器
- [`three_libraries.py`](d:\AI\project\zulong_beta4\zulong\memory\three_libraries.py) - 三库分立
- [`memory_evolution.py`](d:\AI\project\zulong_beta4\zulong\memory\memory_evolution.py) - 记忆进化
- [`person_profile.py`](d:\AI\project\zulong_beta4\zulong\memory\person_profile.py) - 人物画像
- [`experience_generator.py`](d:\AI\project\zulong_beta4\zulong\memory\experience_generator.py) - 经验生成

### 推理引擎核心文件
- [`inference_engine.py`](d:\AI\project\zulong_beta4\zulong\l2\inference_engine.py) - L2 推理引擎

### 基础设施核心文件
- [`shared_memory_pool.py`](d:\AI\project\zulong_beta4\zulong\infrastructure\shared_memory_pool.py) - 共享内存池
- [`data_ingestion.py`](d:\AI\project\zulong_beta4\zulong\infrastructure\data_ingestion.py) - 数据摄入

---

**文档结束**
