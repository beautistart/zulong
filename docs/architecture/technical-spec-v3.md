# 祖龙 (ZULONG) 系统技术规格说明书
## Technical Specification Document (TSD) v3.0

**版本**: v3.0  
**日期**: 2026-05-11  
**状态**: 发布版 - GitHub 开源准备就绪  
**代码规模**: 82,000+ 行 Python + 946 个 TypeScript/TSX 文件

---

# 📋 修订历史

| 版本 | 日期 | 修订内容 | 作者 |
|------|------|---------|------|
| v2.9 | 2026-04-23 | 记忆系统架构全面同步 + 四大核心模块 + 3D记忆城市 | 架构团队 |
| **v3.0** | **2026-05-11** | **发布版更新**：<br>- 完整技术栈分析<br>- 前后端分离架构详解<br>- WebSocket 通信协议规范<br>- 差异化竞争力评估<br>- 发布策略与商业化路径<br>- 性能基准数据 | 架构团队 |

---

# 第 1 章：项目总览

## 1.1 项目定位

**祖龙 (ZULONG)** 是一个多层自适应智能体认知系统，核心定位：

1. **AI Agent 认知操作系统**：记忆管理 + 注意力调度 + 任务编排
2. **具身机器人大脑**：多模态感知 + 任务规划 + 长期记忆 + 中断恢复
3. **开发者工具**：VS Code Extension + MCP 协议支持

## 1.2 核心差异化

| 能力 | 祖龙 | 竞品现状 | 差异化价值 |
|------|------|---------|-----------|
| **异构记忆图谱** | NetworkX + JSON 持久化 | LangChain 内存图 / MemGPT 单路向量 | **市场独有** |
| **赫布学习** | 共激活边自动增强 | 无竞品实现 | **市场独有** |
| **艾宾浩斯衰减** | exp 衰减 + 突触修剪 | AutoGPT 基于年龄 | **市场独有** |
| **双路径检索** | BFS 热遍历 + FAISS 冷检索 | 单路向量检索 | **领先** |
| **CircuitBreaker** | 6 信号综合熔断 | LangChain 硬限制 | **业界最完善** |
| **任务挂起/恢复** | 跨天级序列化 | 无竞品支持 | **市场独有** |

## 1.3 项目规模

```yaml
代码统计:
  Python 文件: 324 个
  Python 代码行: 30,000+
  TypeScript/TSX 文件: 946 个
  TypeScript 代码行: 52,000+
  总代码行: 82,000+

核心模块:
  - L2 推理引擎: 15,000+ 行
  - 记忆系统: 8,000+ 行
  - IDE 模式: 6,000+ 行
  - 多层感知: 5,000+ 行

文档:
  - 技术文档: 100+ 个 Markdown 文件
  - 总文档量: 2,000+ 页
```

---

# 第 2 章：系统架构

## 2.1 四层推理模型

```
┌─────────────────────────────────────────────────────────────┐
│                  L3 专家层 (Expert Layer)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  DualBrainContainer (7 种专家模型池)                   │  │
│  │  ├─ GENERAL: 通用推理                                 │  │
│  │  ├─ LOGIC: 逻辑推理                                   │  │
│  │  ├─ CREATIVE: 创意生成                                │  │
│  │  ├─ CODE: 代码生成                                    │  │
│  │  ├─ MATH: 数学推理                                    │  │
│  │  ├─ VISION: 视觉理解                                  │  │
│  │  └─ AUDIO: 音频处理                                   │  │
│  │  热切换时间: <10ms                                    │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│              L2 认知层 (Cognitive Layer) - 核心               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  InferenceEngine (5,700+ 行)                          │  │
│  │  ├─ 两阶段推理: 意图分类 → 场景化执行                  │  │
│  │  ├─ FC 循环: Function Calling 自主迭代                │  │
│  │  ├─ 记忆检索: MemoryGraph + RAG 双路径                │  │
│  │  ├─ 任务编排: TaskGraph 无限深度递归树                 │  │
│  │  ├─ 死循环检测: CircuitBreaker 6 信号熔断             │  │
│  │  ├─ 注意力控制: 三模式窗口裁剪                         │  │
│  │  └─ 5 层防护链: CB → RuleGuardian → InfoGap → ...    │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│           L1-B 调度层 (Scheduler Layer)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Gatekeeper: 事件路由与优先级调度                      │  │
│  │  ├─ CRITICAL > HIGH > NORMAL > LOW                    │  │
│  │  └─ 事件打包与去重                                    │  │
│  │  AttentionController: 中断管理与快照                   │  │
│  │  ├─ 三层注意力: L0 采集 → L1 静默 → L2 交互           │  │
│  │  └─ 冻结 → 重组 → 注入                                │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│         L1-A 感知与受控反射层 (Reflex Layer)                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  AudioFusion: 音频融合 + 自反应机制                    │  │
│  │  ├─ ReflexController: 障碍/运动/摔倒/视觉反射        │  │
│  │  └─ 自反射弧: 紧急情况立即响应                         │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│         L1-C 静默视觉注意层 (Vision Layer)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  VisionProcessor: 四层级联视觉处理                     │  │
│  │  ├─ YOLOv10: 人体检测 (语义初筛)                      │  │
│  │  ├─ MediaPipe: 姿态估计 + 手势识别 (10种)              │  │
│  │  ├─ MobileNetV4-TSM: 动作分类 (5类交互意图)            │  │
│  │  └─ 几何规则: 交互意图确认 (挥手/注视/靠近)             │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│         L1-D 听觉层 (Auditory Layer)                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  音频采集 → 预加重+滤波(80Hz) → YAMNet(521类)         │  │
│  │  → VAD → SenseVoice-Small(转录+情感+事件+语种)        │  │
│  │  ├─ 唤醒词检测: "你好", "小紫" (NORMAL)               │  │
│  │  ├─ 紧急呼救: "救命" (CRITICAL, 直达L1-B)             │  │
│  │  └─ Whisper-tiny: ASR备选fallback                     │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│         L1-E 安全层 (Safety Layer)                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  MQ-2烟雾传感器 → 气体浓度检测(阈值500ppm)             │  │
│  │  → CRITICAL事件穿透直达L1-B + 语音报警                │  │
│  │  └─ 报警冷却: 60秒                                    │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│              L0 设备层 (Device Layer)                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  CameraDevice: USB 摄像头驱动                          │  │
│  │  MicrophoneDevice: 麦克风采集 (16kHz PCM)             │  │
│  │  SpeakerDevice: 扬声器输出 (24kHz)                     │  │
│  │  SensorManager: 设备管理 + 热插拔                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 2.2 前后端分离架构

```
┌───────────────────────────────────────────────────────────────┐
│                    VS Code Extension (前端)                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  React + Vite Webview (webview-ui/)                      │  │
│  │  ├─ Chat 组件: 对话界面 + Markdown 渲染                  │  │
│  │  ├─ Settings 组件: Zulong WebSocket 配置                 │  │
│  │  ├─ History 组件: 会话历史                               │  │
│  │  └─ MCP 组件: MCP 服务器管理                             │  │
│  ├─────────────────────────────────────────────────────────┤  │
│  │  TypeScript Extension (src/)                             │  │
│  │  ├─ ZulongHandler: WebSocket API 适配器                  │  │
│  │  ├─ ZulongWebSocketTransport: 传输层                     │  │
│  │  ├─ ToolExecutor: 工具执行引擎                           │  │
│  │  └─ MessageHandler: 消息分发                             │  │
│  └─────────────────────────────────────────────────────────┘  │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     │ WebSocket 双向通信
                     │ ws://127.0.0.1:8090/ide
                     │
                     ↓
┌───────────────────────────────────────────────────────────────┐
│                  Python Backend (后端)                          │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  FastAPI + WebSocket Server (ide_server.py)              │  │
│  │  ├─ IDESession: 会话管理                                 │  │
│  │  │   ├─ outbound_queue: 后端→前端消息队列                │  │
│  │  │   └─ tool_result_queue: 前端→后端结果队列             │  │
│  │  ├─ FC Loop Runner: FC 循环执行器                        │  │
│  │  └─ Tool Executor: 工具分流执行                          │  │
│  ├─────────────────────────────────────────────────────────┤  │
│  │  L2 推理引擎 (l2/)                                       │  │
│  │  ├─ InferenceEngine: 推理决策                            │  │
│  │  ├─ MemoryGraph: 记忆检索                                │  │
│  │  ├─ TaskGraph: 任务编排                                  │  │
│  │  ├─ CircuitBreaker: 死循环检测                           │  │
│  │  └─ AttentionWindow: 注意力控制                          │  │
│  ├─────────────────────────────────────────────────────────┤  │
│  │  语音系统 (l1a/ + audio_handler.py)                      │  │
│  │  ├─ TTS: Kokoro-82M (<0.3s CPU 推理)                     │  │
│  │  ├─ ASR: SenseVoice-Small (244M ONNX INT8)              │  │
│  │  └─ VAD: WebRTC 语音活动检测                             │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

## 2.3 数据流全景图

```
┌───────────────────────────────────────────────────────────────┐
│ 1. 用户输入                                                    │
│    ├─ 文本: 直接进入 L2 推理                                    │
│    └─ 语音: 麦克风 → VAD → ASR → 文本                          │
└────────────────┬──────────────────────────────────────────────┘
                 ↓
┌───────────────────────────────────────────────────────────────┐
│ 2. 会话启动 (ide_server.py)                                    │
│    ├─ 创建 IDESession (session_id, ws, queues)                 │
│    ├─ 启动 FC 循环任务 (asyncio.create_task)                   │
│    └─ 广播 IDE_SESSION_START 到 Web 监控                       │
└────────────────┬──────────────────────────────────────────────┘
                 ↓
┌───────────────────────────────────────────────────────────────┐
│ 3. 意图识别 (ide_fc_runner.py)                                 │
│    ├─ Round 1: 强制 tool_choice 调用 intent_classify           │
│    └─ 输出: CHAT / COMPLEX / RESUME                            │
└────────────────┬──────────────────────────────────────────────┘
                 ↓
┌───────────────────────────────────────────────────────────────┐
│ 4. 记忆检索                                                    │
│    ├─ 热路径: BFS 扩散激活 (<50ms)                              │
│    ├─ 冷路径: FAISS 向量检索 (<200ms)                           │
│    └─ 合并: 热记忆 + 冷记忆 + RAG 上下文                        │
└────────────────┬──────────────────────────────────────────────┘
                 ↓
┌───────────────────────────────────────────────────────────────┐
│ 5. 注意力窗口裁剪 (attention_window.py)                         │
│    ├─ GLOBAL 模式: 全量上下文                                  │
│    ├─ FOCUS 模式: 当前任务链 + 相关记忆                         │
│    └─ SINGLE_CHAIN 模式: 单任务链深度执行                       │
└────────────────┬──────────────────────────────────────────────┘
                 ↓
┌───────────────────────────────────────────────────────────────┐
│ 6. 模型调用 (inference_engine.py)                              │
│    ├─ 构建 Prompt: 系统提示 + 任务图 + 记忆 + 对话历史           │
│    ├─ 调用 LLM: OpenAI SDK (vLLM/Ollama/云端)                  │
│    ├─ 流式解析: text / reasoning / tool_calls                  │
│    └─ 推送: display_text / display_reasoning / tool_request    │
└────────────────┬──────────────────────────────────────────────┘
                 ↓
┌───────────────────────────────────────────────────────────────┐
│ 7. 工具执行                                                    │
│    ├─ 内部工具: 后端直接执行 (task/memory/graph)                │
│    └─ 远程工具: 暂停 → 返回前端 → 等待结果 → 恢复               │
└────────────────┬──────────────────────────────────────────────┘
                 ↓
┌───────────────────────────────────────────────────────────────┐
│ 8. 结果评估 (5 层防护链)                                        │
│    ├─ 1. CB 强制收敛 → 跳过后续                                │
│    ├─ 2. RuleGuardian 过早完成 → 注入纠正                       │
│    ├─ 3. InfoGap 信息缺口 → 继续循环                           │
│    ├─ 4. AutoMark 自动标记 → 推进状态                          │
│    └─ 5. Backfill 节点回填 → 填充内容                          │
└────────────────┬──────────────────────────────────────────────┘
                 ↓
┌───────────────────────────────────────────────────────────────┐
│ 9. 状态更新                                                    │
│    ├─ TaskGraph: 节点状态、进度                                │
│    ├─ MemoryGraph: 创建节点、更新边权重 (赫布学习)               │
│    ├─ CircuitBreaker: 记录工具调用、检测循环                    │
│    └─ 广播: IDE_SESSION_UPDATE 到 Web 监控                     │
└────────────────┬──────────────────────────────────────────────┘
                 ↓
┌───────────────────────────────────────────────────────────────┐
│ 10. 输出生成                                                   │
│     ├─ 文本: display_text → 前端渲染                          │
│     ├─ TTS: text → Kokoro 合成 → 扬声器播放                    │
│     └─ 任务完成: task_complete → 结束会话                       │
└───────────────────────────────────────────────────────────────┘
```

---

# 第 3 章：WebSocket 通信协议

## 3.1 消息类型定义

### 前端 → 后端消息

```typescript
// 1. 会话启动
interface SessionStartMessage {
  type: "session_start"
  payload: {
    task: string              // 任务文本
    cwd: string               // 工作目录
    ide_system_prompt: string // IDE 系统提示
  }
}

// 2. 工具执行结果返回
interface ToolResultMessage {
  type: "tool_result"
  payload: {
    call_id: string
    tool_name: string
    result: string
    is_error: boolean
  }
}

// 3. 用户取消
interface UserCancelMessage {
  type: "user_cancel"
  payload: {}
}

// 4. 音频流 (实时 ASR)
interface AudioChunkMessage {
  type: "audio_chunk"
  payload: {
    audio_data: string  // base64 编码 PCM 16kHz
    is_final: boolean
  }
}
```

### 后端 → 前端消息

```typescript
// 1. 工具请求 (暂停 FC 循环，等待前端执行)
interface ToolRequestMessage {
  type: "tool_request"
  payload: {
    tool_calls: [{
      id: string
      function: {
        name: string
        arguments: string  // JSON 字符串
      }
    }]
    call_ids: string[]
    tool_names: string[]
  }
}

// 2. 文本流式推送
interface DisplayTextMessage {
  type: "display_text"
  payload: {
    text: string
  }
}

// 3. 推理过程推送
interface DisplayReasoningMessage {
  type: "display_reasoning"
  payload: {
    reasoning: string
  }
}

// 4. 任务完成
interface TaskCompleteMessage {
  type: "task_complete"
  payload: {
    result: string
  }
}

// 5. FC 循环状态更新
interface StatusUpdateMessage {
  type: "status_update"
  payload: {
    turn: number       // 当前轮次
    phase: string      // 当前阶段 (planning/executing/evaluating)
  }
}

// 6. 音频转录结果
interface AudioTranscriptMessage {
  type: "audio_transcript"
  payload: {
    text: string
    is_final: boolean
    emotion?: string   // 情感识别结果
  }
}
```

## 3.2 工具分流机制

```python
# ide_fc_runner.py 中的工具分流
async def execute_tool_call(tool_name: str, arguments: dict):
    # 远程工具列表 (前端执行)
    IDE_REMOTE_TOOLS = {
        "read_file", "write_to_file", "replace_in_file", "delete_file",
        "execute_command", "search_files", "list_files",
        "list_code_definition_names", "browser_action",
        "ask_followup_question", "attempt_completion"
    }
    
    if tool_name in IDE_REMOTE_TOOLS:
        # 远程工具: 暂停 FC 循环，返回前端执行
        await session.send_msg("tool_request", {
            "tool_calls": [{
                "id": call_id,
                "function": {"name": tool_name, "arguments": json.dumps(arguments)}
            }],
            "call_ids": [call_id],
            "tool_names": [tool_name]
        })
        
        # 等待前端返回结果
        result = await session.tool_result_queue.get()
        return result
    else:
        # 内部工具: 直接执行
        result = await internal_tool_executor.execute(tool_name, arguments)
        return result
```

---

# 第 4 章：MemoryGraph 异构记忆图谱

## 4.1 核心设计理念

**MemoryGraph 是 L2 认知层的记忆中枢**，采用统一异构图架构，所有记忆子系统投射为节点和边。

**类比**: LLM 是大脑皮层（推理），MemoryGraph 是海马体（记忆索引和联想）。

## 4.2 图结构定义

### 节点类型 (NodeType) - 9 种

```python
class NodeType(Enum):
    TASK = "task"              # 任务节点 (来自 TaskGraph)
    DIALOGUE = "dialogue"      # 对话轮次 (session/round/sub_dialogue)
    KNOWLEDGE = "knowledge"    # 知识图谱实体
    EXPERIENCE = "experience"  # 经验 RAG 文档
    EPISODE = "episode"        # 情节记忆
    FILE = "file"              # 文件锚点
    CONCEPT = "concept"        # 概念节点
    PERSON = "person"          # 人物档案
    DOCUMENT = "document"      # 文档节点
```

### 边类型 (EdgeType) - 7 种

```python
class EdgeType(Enum):
    HIERARCHY = "hierarchy"    # 层级关系 (task → subtask, protected=True)
    DEPENDENCY = "dependency"  # 依赖关系 (task → task, protected=True)
    REFERENCE = "reference"    # 引用关系 (dialogue → file)
    TEMPORAL = "temporal"      # 时间顺序 (dialogue → dialogue, protected=True)
    SEMANTIC = "semantic"      # 语义相似 (自动发现, cos_sim > 0.7)
    CAUSAL = "causal"          # 因果关系 (experience → experience)
    ASSOCIATION = "association" # 赫布学习产生 (共激活 ≥ 3)
```

**protected=True 的边永不修剪**（HIERARCHY, DEPENDENCY, TEMPORAL）。

## 4.3 三维标签系统

| 维度 | 标签 | 作用 | 取值 |
|------|------|------|------|
| **Temperature** | 热度 | 检索路径选择 | HOT (1h内) / WARM (1h-24h) / COLD (>24h) |
| **Importance** | 重要度 | 衰减半衰期 | TRIVIAL (6h) / NORMAL (24h) / IDENTITY (720h) / FACT (360h) / IMPORTANT (168h) / MUST_REMEMBER (∞) |
| **TimeScope** | 时间范围 | 热窗口过滤 | RECENT / NON_RECENT |

**正交组合示例**：
- `HOT × MUST_REMEMBER × RECENT` → 最高优先级检索
- `COLD × TRIVIAL × NON_RECENT` → 候选淘汰节点

## 4.4 核心能力

### 4.4.1 BFS 扩散激活

```python
def compute_activations(
    seed_nodes: List[str],
    max_depth: int = 3,
    decay_factor: float = 0.5
) -> Dict[str, float]:
    """
    从种子节点触发 BFS，逐级扩散激活值
    
    算法:
    1. 初始化: activation[seed] = 1.0
    2. BFS 遍历邻居节点
    3. 激活值计算: act = current_act × edge_weight × decay_factor
    4. 双向传播 (predecessors + successors)
    
    返回: {node_id: activation_value}
    """
```

**性能**: <50ms (热路径，max_depth=3)

### 4.4.2 赫布学习

```python
def hebbian_strengthen(node_a: str, node_b: str):
    """
    共激活节点对边权重增强
    
    算法:
    new_weight = old_weight + 0.1 × (1 - old_weight)
    
    渐近趋向 1.0，模拟神经突触可塑性
    
    触发条件:
    - 同一轮 FC 循环中被连续访问
    - 共激活计数 >= 3
    """
```

**效果**: 频繁共现的节点自动建立强连接。

### 4.4.3 艾宾浩斯衰减

```python
def decay_and_prune():
    """
    艾宾浩斯遗忘曲线衰减 + 弱连接修剪
    
    算法:
    1. 计算经过时间: age_days = (now - last_access) / 86400
    2. 衰减公式: weight *= exp(-age_days × ln(2) / half_life)
    3. 半衰期查表:
       - TRIVIAL: 6h
       - NORMAL: 24h
       - IMPORTANT: 168h (7天)
       - FACT: 360h (15天)
       - IDENTITY: 720h (30天)
       - MUST_REMEMBER: ∞ (不衰减)
    4. 修剪: weight < 0.1 且非 protected 边 → 删除
    """
```

**效果**: 30天后不重要记忆权重降至 5%，自动淘汰。

### 4.4.4 双路径检索

```python
async def retrieve_context(
    query: str,
    session_id: str,
    top_k: int = 10
) -> List[MemoryNode]:
    """
    双路径并行检索
    
    热路径 (asyncio.gather 并行):
    1. 从 session 相关节点作为种子
    2. BFS 扩散激活 (max_depth=3)
    3. 按激活值排序取 top_k
    
    冷路径 (asyncio.gather 并行):
    1. FAISS 向量检索 (query embedding)
    2. 余弦相似度排序取 top_k
    
    合并:
    - 热记忆 + 冷记忆
    - 互斥过滤 (避免重复)
    - 按分数归一化排序
    """
```

**性能**: 热路径 <50ms, 冷路径 <200ms, 并行总耗时 ~200ms

### 4.4.5 语义边自动发现

```python
async def discover_semantic_edges():
    """
    后台持续计算节点间余弦相似度
    
    算法:
    1. 批量获取所有节点 embedding
    2. 计算余弦相似度矩阵
    3. cos_sim > 0.7 且无边 → 创建 SEMANTIC 边
    4. 定期执行 (每 6 小时)
    """
```

**效果**: 自动发现隐含语义关联，无需人工标注。

---

# 第 5 章：CircuitBreaker 死循环检测器

## 5.1 设计动机

LLM 在 FC 循环中可能陷入：
- 重复调用同一工具
- 模式循环（A→B→A→B...）
- 信息增益递减（工具返回重叠）
- 无进度空转（连续信息检索无行动）

**CircuitBreaker** 通过 6 个信号综合检测，实现智能熔断。

## 5.2 六信号机制

```python
class CircuitBreaker:
    """死循环检测器 (6 信号综合熔断)"""
    
    # Signal 1: 相同调用重复检测
    signal_repeated_call: bool
    # 实现: hash(tool_name + params) 在窗口内重复
    
    # Signal 2: 模式循环检测
    signal_pattern_loop: bool
    # 实现: 工具频次统计 + Jaccard 相似度
    
    # Signal 3: 信息增益递减检测
    signal_info_gain_decay: bool
    # 实现: result hash 重叠率 > 0.7
    
    # Signal 4: 上下文压力检测
    signal_context_pressure: bool
    # 实现: token_估算 / 窗口大小 > 0.8
    
    # Signal 5: 经过时间检测
    signal_time_elapsed: bool
    # 实现: elapsed_time > max_time (已禁用，步数为主控)
    
    # Signal 6: 无进度空转检测
    signal_no_progress: bool
    # 实现: 连续信息检索工具无行动工具
```

## 5.3 状态机

```
GREEN (正常)
  ├─ 任何信号触发 → YELLOW
  └─ 继续循环

YELLOW (警告)
  ├─ 注入警告指令到 LLM prompt
  ├─ 再次触发 → RED
  └─ 3 轮无信号 → GREEN

RED (熔断)
  ├─ 强制停止 FC 循环
  ├─ 设置 cb_force_no_tools = True
  └─ 返回累积结果
```

## 5.4 动态放宽模式

```python
# 规划模式放宽
if current_phase == "planning":
    pattern_window = 20  # 默认 6
    max_repeated = 5     # 默认 2

# RESUME 模式放宽
if intent == "RESUME":
    pattern_window = 20
    max_repeated = 5

# 小模型专项补偿
if model_size <= "7B":
    pattern_window *= 2
    max_repeated *= 2
```

**目的**: 小模型逐节点处理行为不应被误判为循环。

---

# 第 6 章：TaskGraph 任务图谱

## 6.1 无限深度递归树

```
TaskGraph 结构:
  ├─ 根节点 (root_task)
  ├─ 子节点 (subtasks)
  │   ├─ 叶子节点 (可执行单元)
  │   └─ 非叶子节点 (状态聚合器)
  └─ 依赖关系 (DEPENDENCY 边)

节点类型:
  ├─ TEMPLATE: 模板节点 (预定义任务)
  └─ DYNAMIC: 动态生成节点 (LLM 创建)
```

## 6.2 节点状态

```python
class TaskStatus(Enum):
    PENDING = "pending"       # 待执行
    IN_PROGRESS = "in_progress"  # 执行中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败
    BLOCKED = "blocked"       # 被阻塞 (依赖未满足)
```

## 6.3 状态聚合规则

```python
def aggregate_status(node: TaskNode) -> TaskStatus:
    """
    非叶子节点状态聚合
    
    规则:
    1. 所有子节点 COMPLETED → 父节点 COMPLETED
    2. 任一子节点 FAILED → 父节点 FAILED
    3. 任一子节点 IN_PROGRESS → 父节点 IN_PROGRESS
    4. 任一子节点 BLOCKED → 父节点 BLOCKED
    5. 其余 → PENDING
    """
```

## 6.4 依赖管理

```python
def check_dependencies(node: TaskNode) -> bool:
    """
    检查依赖是否满足
    
    实现:
    - 遍历 DEPENDENCY 边指向的前置节点
    - 所有前置节点 status == COMPLETED → 返回 True
    - 否则设置 status = BLOCKED，返回 False
    """
```

---

# 第 7 章：5 层防护链

## 7.1 防护链架构

```python
async def eval_response(state: IDEFCState) -> IDEFCState:
    """
    5 层防护链评估 (逐层递进)
    """
    
    # Layer 1: CB 强制收敛检查
    if state.cb_force_no_tools:
        # 跳过后续层，组装工具结果缓冲区
        return assemble_buffer(state)
    
    # Layer 2: RuleGuardian 过早完成拦截
    if rule_guardian.check_premature_completion(state):
        # 注入纠正指令
        inject_correction(state)
        return state
    
    # Layer 3: InfoGap 信息缺口检测
    if info_gap_detector.detect(state):
        # 检测到 NEED_SUBTASK_RESULT / NEED_USER_INPUT
        # 继续循环
        state.gap_continue_count += 1
        return state
    
    # Layer 4: RESUME AutoMark 安全网
    if state.intent == "RESUME" and not tool_called:
        # 模型忘记调 task_mark_status
        # 自动标记当前节点为 completed
        auto_mark_current_node(state)
        state.resume_automark_count += 1
        return state
    
    # Layer 5: COMPLEX Backfill 节点回填
    if state.intent == "COMPLEX":
        # 从回复中匹配节点标签
        # 自动填充节点内容
        backfill_nodes(state)
    
    return state
```

## 7.2 各层职责

| 层级 | 名称 | 触发条件 | 动作 |
|------|------|---------|------|
| 1 | CB 强制收敛 | `cb_force_no_tools=True` | 组装工具结果缓冲区，跳过后续层 |
| 2 | RuleGuardian | TaskGraph 有未完成节点但 LLM 声称完成 | 注入纠正指令，强制继续 |
| 3 | InfoGap | 回复包含 `NEED_SUBTASK_RESULT` 或 `NEED_USER_INPUT` | 继续循环，等待信息补充 |
| 4 | RESUME AutoMark | RESUME 模式下未调用 `task_mark_status` | 自动标记当前节点 completed |
| 5 | COMPLEX Backfill | COMPLEX 模式下回复包含节点标签 | 匹配并填充 TaskGraph 节点内容 |

---

# 第 8 章：性能优化

## 8.1 Schema 缓存

```python
class SchemaCache:
    """工具 Schema 缓存 (SHA-256 哈希校验)"""
    
    _cache: Dict[str, ToolSchema] = {}
    _hash_map: Dict[str, str] = {}  # tool_name -> definition_hash
    
    def get_or_build(tool_name: str, definition: dict) -> ToolSchema:
        # 计算哈希
        definition_hash = hashlib.sha256(
            json.dumps(definition, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        # 检查缓存
        if tool_name in self._cache:
            if self._hash_map[tool_name] == definition_hash:
                self.cache_hits += 1
                return self._cache[tool_name]
        
        # 构建新 Schema
        schema = build_tool_schema(definition)
        self._cache[tool_name] = schema
        self._hash_map[tool_name] = definition_hash
        return schema
```

**效果**: 命中时 Schema 构建时间从 ~100ms 降至 ~5ms。

## 8.2 线程池管理

```python
class ThreadPoolManager:
    """模型调用线程池 (单例模式)"""
    
    _instance = None
    _executor: ThreadPoolExecutor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._executor = ThreadPoolExecutor(max_workers=1)  # 串行化
            atexit.register(cls.graceful_shutdown)
        return cls._instance
    
    async def submit_model_call(self, func, *args, **kwargs):
        """提交模型调用任务"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            functools.partial(func, *args, **kwargs),
        )
```

**效果**: 避免多线程 GPU 资源竞争，提升推理稳定性。

## 8.3 异步消息队列

```python
class IDESession:
    """IDE 会话管理"""
    
    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.ws = websocket
        
        # 异步队列
        self.outbound_queue: asyncio.Queue = asyncio.Queue()
        self.tool_result_queue: asyncio.Queue = asyncio.Queue()
        
        # 启动消息发送协程
        asyncio.create_task(self._send_loop())
    
    async def _send_loop(self):
        """消息发送循环"""
        while True:
            msg = await self.outbound_queue.get()
            await self.ws.send_json(msg)
```

**效果**: 非阻塞消息传递，支持高并发。

---

# 第 9 章：语音交互系统

## 9.1 TTS (文本转语音)

```yaml
引擎: Kokoro-82M
参数: 82M (CPU 推理)
推理时间: <0.3s
音色: zf_xiaoxiao (中文女声)
采样率: 24kHz
支持语言: 中文、英文、日文
```

**性能对比**:
| 引擎 | 参数 | 推理时间 | 自然度 |
|------|------|---------|--------|
| **Kokoro-82M** | 82M | <0.3s | 高 |
| Edge-TTS | 云端 | ~1s | 高 |
| CosyVoice3 | 0.5B | ~0.5s | 极高 |

## 9.2 ASR (语音识别)

```yaml
引擎: SenseVoice-Small
参数: 244M (ONNX INT8 量化)
推理时间: 0.5-1s (5秒音频)
支持语言: 中文、英文、日文、韩文、粤语
情感识别: 支持 (angry/happy/sad/neutral/...)
事件检测: 支持 (BGM/applause/laughter/cry)
```

**能力对比**:
| 引擎 | 参数 | 情感识别 | 事件检测 | 中文准确率 |
|------|------|---------|---------|-----------|
| **SenseVoice-Small** | 244M | ✅ | ✅ | >95% |
| Whisper-small | 244M | ❌ | ❌ | ~90% |

## 9.3 整体延迟

**本地模式延迟**:
```
用户说话
  ↓
WebRTC VAD 检测 (~10ms)
  ↓
音频缓冲 (等待静音)
  ↓
ASR 转录 (0.5-1s)
  ↓
L2 推理 (1-2s)
  ↓
TTS 合成 (<0.3s)
  ↓
播放音频
────────────────
总计: 3-4s
```

**云端 API 调用模式延迟**:
- 云端 LLM 调用增加网络延迟
- **整体延迟**: 3-4s（语音输入 → 语音输出）
- 适用于使用 GPT-4/Claude 等云端模型的场景

---

# 第 9.5 章：L1 感知与具身控制层

## 9.5.1 设计定位

祖龙的 L1 层是连接硬件传感器（L0）与认知大脑（L2）的**桥梁层**，包含五个子层：

```
L2 认知层 (决策)
    ↕ 事件路由
L1-B 调度层 (事件路由 + 中断管理 + 三层注意力)
    ↕ 共享内存
L1-A 反射层 (硬实时安全反射) + L1-C 视觉层 (静默视觉注意) + L1-D 听觉层 (音频感知与意图判断) + L1-E 安全层 (气体/火灾检测)
    ↕ 传感器数据 / 执行器指令
L0 设备层 (硬件驱动)
```

**核心原则**: L1 层实现**无 LLM 开销**的实时感知与安全保障。所有 L1 模块的处理延迟要求 <50ms。

## 9.5.2 L0 设备层规格

| 设备 | 源文件 | 接口规格 | 数据格式 |
|------|--------|----------|----------|
| USB 摄像头 | `l0/devices/camera_device.py` | 640×480, 30fps | BGR 帧 (numpy array) |
| 麦克风 | `l0/devices/microphone_device.py` | 16kHz, 16bit, 单声道 | PCM int16 |
| 扬声器 | `l0/devices/speaker_device.py` | 24kHz, 16bit | PCM int16 |
| 运动检测器 | `l0/motion_detector.py` | Lucas-Kanade 光流 | MotionResult dataclass |
| 执行器模拟器 | `l0/actuator_simulator.py` | 3-DOF 关节 | 位置/速度/扭矩/电流/温度 |

**运动检测器状态机**:

```
IDLE (静止) ──motion_pixels > threshold──→ MOVING (运动中)
  ↑                                              │
  │               motion_pixels < threshold      │
  │               持续 N 帧                       ↓
  └────────────────────────────────────── STOPPED (已停止)
```

**GPU 加速光流**: PyTorch CUDA 实现，RTX 3060 实测 150+ FPS (vs CPU 18 FPS)，加速比 8.28×。通过 `gpu_optical_flow.py` 自动检测 CUDA 可用性并降级。

## 9.5.3 L1 模块化插件架构

所有 L1 功能模块通过统一接口 `IL1Module` 实现为可热插拔插件：

```python
# 核心接口 (zulong/modules/l1/core/interface.py)
class IL1Module(ABC):
    @property
    def module_id(self) -> str: ...
    @property
    def priority(self) -> EventPriority: ...
    def initialize(self, shared_memory: Dict) -> bool: ...
    def process_cycle(self, shared_memory: Dict) -> List[ZulongEvent]: ...
    def health_check(self) -> Dict: ...
    def shutdown(self): ...
```

**插件配置** (`config/l1_plugins.yaml`):

```yaml
plugins:
  - module_id: "L1A/Motor"
    class_path: "zulong.plugins.motor.l1a_motor_plugin.L1A_MotorPlugin"
    enabled: true
    priority: "HIGH"
    config:
      obstacle_threshold: 0.3  # 米
      max_speed: 0.8
      brake_decay: 0.5
      auto_brake: true

  - module_id: "L1C/Vision"
    class_path: "zulong.plugins.vision.l1c_vision_plugin.L1C_VisionPlugin"
    enabled: true
    priority: "NORMAL"
    config:
      fps: 30
      motion_threshold: 500

  - module_id: "L1D/Voice"
    class_path: "zulong.plugins.voice.l1d_voice_plugin.L1D_VoicePlugin"
    enabled: true
    priority: "CRITICAL"
    config:
      wake_words: ["你好", "救命", "小紫"]

  - module_id: "L1E/Gas"
    class_path: "zulong.plugins.gas.l1e_gas_plugin.L1E_GasPlugin"
    enabled: true
    priority: "CRITICAL"
    config:
      threshold: 500  # ppm
```

**共享内存键值规范**:

| 命名空间 | 键名 | 类型 | 说明 |
|----------|------|------|------|
| `motor.*` | `motor.speed`, `motor.target_speed`, `motor.mode` | float/str | 电机状态 |
| `obstacle.*` | `obstacle.distance`, `obstacle.detected` | float/bool | 障碍物检测 |
| `vision.*` | `vision.frame_count`, `vision.motion_detected` | int/bool | 视觉状态 |
| `voice.*` | `voice.wakeup_detected`, `audio.text`, `audio.confidence` | bool/str/float | 语音状态 |
| `gas.*` | `gas.concentration`, `gas.alarm` | int/bool | 气体检测 |

## 9.5.4 L1-A 反射层安全规格

**反射控制器事件响应表**:

| 传感器事件 | 优先级 | 响应指令 | 最大延迟 |
|-----------|--------|----------|----------|
| SENSOR_FALL | CRITICAL | EMERGENCY_STOP | <10ms |
| SENSOR_GAS (>500ppm) | CRITICAL | EMERGENCY_STOP + ALARM | <10ms |
| SENSOR_OBSTACLE (<0.3m) + MOVING | HIGH | CMD_BRAKE | <50ms |
| SENSOR_OBSTACLE (<0.3m) + CHARGING | - | 抑制 (不响应) | - |
| SENSOR_SOUND (>70dB) | NORMAL | ALERT → L2 | <30ms |
| SENSOR_VISION (人体) | NORMAL | 转发 L1-C | <50ms |

**安全约束不可绕过**: L1-A 反射控制器独立于 L2 运行。即使 L2 推理引擎正在执行导航任务，L1-A 仍能在 <50ms 内中断导航并触发紧急停止。

## 9.5.5 L1-C 四层级联视觉

```
输入帧 (640×480)
    │
    ├─ Layer 1: YOLOv10-Nano → 人体检测 (~6ms/帧)
    │   └─ 无人体 → 仅记录，不产生事件 (静默注意)
    │
    ├─ Layer 2: ROI 裁剪 + MediaPipe Pose → 33 个关键点
    │   └─ 姿态估计用于下层动作推理
    │
    ├─ Layer 3: MobileNetV4-TSM → 动作分类
    │   └─ 16 帧滑动窗口，时序特征提取
    │
    └─ Layer 4: MediaPipe Hands → 手势识别
        └─ 挥手 / 指向 / 注视 → 生成 INTERACTION 事件
```

**算力节省**: 无人场景仅运行 Layer 1 (~6ms)，有人场景展开全流水线 (~30ms)。相比始终运行全流水线节省 ~70% 算力。

## 9.5.6 L1-D 听觉层规格

**处理流水线**:

```
音频采集 (L0_SENSOR, 16kHz PCM)
    │
    ├─ Step 1: 预加重 + 低切滤波 (80Hz 高通)
    │
    ├─ Step 2: YAMNet 环境音分类 (L1_SILENT, 521类)
    │   └─ 判断音频场景: 语音/音乐/噪音/环境声
    │
    ├─ Step 3: VAD 语音活动检测
    │   └─ WebRTC VAD (模式3) + 能量阈值 fallback
    │
    ├─ Step 4: SenseVoice-Small 单次推理 (L2_INTERACTIVE)
    │   ├─ 转录文本 (含 ITN 逆文本归一化)
    │   ├─ 情感标签: HAPPY/SAD/ANGRY/NEUTRAL/FRIGHTENED/SURPRISED
    │   ├─ 语音事件: BGM/Applause/Laugh/Cry/Emphasis
    │   └─ 语种标识: zh/en/ja/ko/yue
    │
    ├─ Step 5: 后处理意图判断 (基于事件标签)
    │   ├─ 交互语音 → 发布 AUDIO_SPEECH_END 事件
    │   └─ 非交互 (BGM/Applause等) → 更新静默状态
    │
    └─ Step 6: Whisper-tiny fallback (仅 SenseVoice 不可用时)
```

**唤醒词检测**:

| 唤醒词 | 优先级 | 行为 |
|--------|--------|------|
| "你好" | NORMAL | 唤醒系统，进入交互模式 |
| "小紫" | NORMAL | 唤醒系统，进入交互模式 |
| "救命" | CRITICAL | 紧急呼救，绕过意图识别直达 L1-B |

**三层注意力音频表**:

| 注意力层 | 模块 | 说明 |
|----------|------|------|
| L0_SENSOR | 音频采集 + 预加重 + 低切滤波 | 纯信号处理，无事件 |
| L1_SILENT | YAMNet (521类) + VAD | 环境音分类 + 语音活动检测 |
| L2_INTERACTIVE | SenseVoice-Small + Whisper-tiny | ASR 转录 + 情感/事件检测 |

**与 L1-B 串行协作**:

| 环节 | 层级 | 模型 | 功能 |
|------|------|------|------|
| 粗粒度判断 | L1-D | SenseVoice 事件标签 | 交互/非交互二分 |
| 细粒度分类 | L1-B | ALBERT-tiny | 15类意图分类 |
| 语音模式 | L1-B | ALBERT + 独立分类头 | TEXT_ONLY/AUTO_TTS/FORCED_TTS |

**插件配置**:

```yaml
- module_id: "L1D/Voice"
  class_path: "zulong.plugins.voice.l1d_voice_plugin.L1D_VoicePlugin"
  priority: "CRITICAL"
  config:
    wake_words: ["你好", "救命", "小紫"]
    silent_commands: ["安静", "睡觉", "去休息"]

- module_id: "L1D/Audio"
  class_path: "zulong.plugins.voice.l1d_audio_plugin.L1D_AudioPlugin"
  priority: "CRITICAL"
  config:
    enable_yamnet: true
    enable_whisper: true
```

> **最后更新**: 2026-05-16

## 9.5.7 L1-E 安全层规格

**检测流水线**:

```
MQ-2 烟雾传感器
    │
    ├─ Step 1: 气体浓度读取 (模拟/真实)
    │
    ├─ Step 2: 阈值判定 (500 ppm)
    │
    └─ Step 3: 超阈值 → 发布事件
        ├─ SENSOR_GAS 事件 (CRITICAL)
        └─ ACTION_SPEAK 事件 ("检测到烟雾！请立即疏散！")
```

**CRITICAL 优先级穿透机制**:

| 特性 | 说明 |
|------|------|
| 事件优先级 | CRITICAL (最高) |
| 穿透特性 | 无视 L2 当前任务状态，强制中断 |
| 直达层级 | L1-B AttentionController → EventQueue |
| 响应行为 | 立即暂停 L2 → 处理安全事件 → 语音报警 |
| 报警冷却 | 60秒 (防止重复报警) |

**插件配置**:

```yaml
- module_id: "L1E/Gas"
  class_path: "zulong.plugins.gas.l1e_gas_plugin.L1E_GasPlugin"
  priority: "CRITICAL"
  config:
    threshold: 500
    simulate: true
```

**与 L1-A 协同**: L1-A 处理物理碰撞/跌落反射，L1-E 处理环境危险(气体/火灾)，双重保障。

> **最后更新**: 2026-05-16

## 9.5.8 L3 导航专家

导航专家作为 L3 专家技能池成员，由 L2 通过 FC 工具调用:

**A* 路径规划**: 栅格地图 + 八向搜索 + 动态障碍物更新

**DWA 动态窗口避障**:

| 参数类别 | 参数 | 值 |
|----------|------|-----|
| 运动学 | 最大线速度 / 角速度 | 1.0 m/s / 1.0 rad/s |
| 运动学 | 线加速度 / 角加速度 | 0.5 m/s² / 1.0 rad/s² |
| 安全 | 安全距离 / 停止距离 | 0.5m / 0.3m |
| 采样 | 速度空间采样 | 10(线) × 20(角) |
| 预测 | 轨迹预测 | 2.0s, 步长 0.1s |
| 评估权重 | 朝向/距离/速度 | 0.3 / 0.5 / 0.2 |

**导航-安全协同**: L3 导航输出速度指令 → L1-A 安全约束检查 → L0 执行器执行。L1-A 可随时覆盖 L3 的指令。

## 9.5.9 OpenClaw Bridge

OpenClaw Bridge 是祖龙与实体机器人之间的事件桥接层:

```
祖龙 EventBus ←→ EventBusClient ←→ 适配器/监听器 ←→ 机器人硬件
                   (ZeroMQ/HTTP)
```

**组件**:

| 组件 | 方向 | 功能 |
|------|------|------|
| `MicAdapter` | 机器人→祖龙 | 音频采集 → USER_SPEECH 事件 |
| `VisionReporter` | 机器人→祖龙 | 视觉检测 → SENSOR_VISION 事件 |
| `ExecuteListener` | 祖龙→机器人 | TASK_EXECUTE → 机器人动作指令 |
| `SpeakListener` | 祖龙→机器人 | ACTION_SPEAK → 语音播报 |
| `WebAdapter` | 双向 | HTTP API + Web UI 交互 |

**设计原则**: Bridge 不加载模型，不做决策，仅做事件格式转换。所有智能逻辑保留在祖龙主系统中。

---

# 第 10 章：MCP 协议支持

## 10.1 MCP Server 定义

```python
from mcp import Server, Tool

app = Server("zulong-memory")

@app.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="zulong_memory_search",
            description="【项目级记忆】搜索架构决策、技术选型原因、踩坑教训",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询"},
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        ),
        # ... 共 7 个工具
    ]
```

## 10.2 MCP 工具列表

| 工具名 | 职责 | 适用场景 |
|--------|------|---------|
| `zulong_memory_search` | 项目级记忆搜索 | 修改核心模块前了解历史决策 |
| `zulong_memory_save` | 保存项目记忆 | 记录重要技术决策 |
| `zulong_task_search` | 历史任务搜索 | 复用成功执行路径 |
| `zulong_experience_search` | 经验库搜索 | 查询相似问题解决方案 |
| `zulong_knowledge_query` | 知识库查询 | 查询团队规范/最佳实践 |
| `zulong_graph_query` | 记忆图谱查询 | 探索关联概念 |
| `zulong_entity_link` | 实体关联 | 建立概念间联系 |

**未来规划**: 扩展至 30+ 工具，覆盖更多记忆操作。

---

# 第 11 章：发布策略与商业化

## 11.1 核心竞争力总结

| 能力 | 技术壁垒 | 市场价值 | 商业化潜力 |
|------|---------|---------|-----------|
| **异构记忆图谱** | 高 (无竞品) | 高 | ⭐⭐⭐⭐⭐ |
| **赫布学习** | 中高 | 中高 | ⭐⭐⭐⭐ |
| **艾宾浩斯衰减** | 中 | 中 | ⭐⭐⭐ |
| **CircuitBreaker** | 中高 | 高 | ⭐⭐⭐⭐ |
| **任务挂起/恢复** | 中 | 中 | ⭐⭐⭐ |

## 11.2 许可证策略

**推荐方案**: AGPL-3.0 + 商业例外

| 内容 | 许可证 | 理由 |
|------|--------|------|
| **核心代码** (`zulong/`) | AGPL-3.0 | 保护核心竞争力，SaaS 使用需开源 |
| **VS Code 扩展** (`zulong-ide/`) | MIT | 吸引开发者，降低集成门槛 |
| **文档** (`docs/`) | CC BY-NC-SA 4.0 | 允许传播，禁止商业使用 |

**AGPL-3.0 保护效力**:
1. 企业 SaaS 服务必须开源全部代码
2. 大公司法务看到 AGPL 会放弃直接使用
3. 要么不用，要么来谈商业授权 → **商业化入口**

## 11.3 商业化路径

### 短期 (1-3 个月)
1. **开源 zulong-memory 独立库**
   - 提取 MemoryGraph 为独立包
   - 发布到 PyPI (`pip install zulong-memory`)
   - 文档 + 示例 + 基准测试

2. **内容营销**
   - Twitter/X 英文 thread
   - 知乎技术长文
   - Hacker News Show HN
   - Reddit r/LocalLLaMA

### 中期 (3-6 个月)
1. **商业授权**
   - 企业版闭源记忆系统
   - 定制化集成服务
   - 技术支持订阅

2. **SaaS 服务**
   - 云端记忆托管
   - 多项目记忆同步
   - 团队协作记忆

### 长期 (6-12 个月)
1. **企业级平台**
   - 多租户隔离
   - 审计日志
   - RBAC 权限

2. **生态建设**
   - 插件市场
   - 社区贡献
   - 培训认证

## 11.4 防止大厂 Fork 策略

1. **AGPL-3.0 传染性**: 大厂 fork 后必须开源自己的产品，成本高
2. **商业例外条款**: 明确列出禁止竞争性使用的场景
3. **快速迭代**: 保持技术领先，让 fork 版本永远落后
4. **社区绑定**: 建立 Contributor 体系，核心贡献者签署 CLA
5. **文档壁垒**: 核心设计理念只在付费文档中披露

## 11.5 快速传播策略

### 核心叙事
> "一个室内设计师，用 AI 助手在 2 个月内写出了 82,000 行代码，造了一个有记忆、会遗忘、能自主调整注意力的 AI 大脑。"

### 发布渠道

| 渠道 | 内容形式 | 目标受众 | 优先级 |
|------|---------|---------|--------|
| **Twitter/X** | 英文 thread + 架构图 + demo 视频 | 全球 AI 开发者、投资人 | 最高 |
| **知乎** | 技术长文 | 中文 AI 技术社区 | 高 |
| **Reddit r/LocalLLaMA** | 技术帖，强调本地部署可跑 | 本地模型爱好者 | 高 |
| **Hacker News** | Show HN 帖 | 全球技术圈 | 高 |
| **B站** | demo 录屏讲解 | 中文开发者 + 泛 AI 圈 | 中 |

### 时间线

**第 1 周**:
- [ ] 录制 demo 视频 (复杂任务 + 中断恢复 + 记忆检索)
- [ ] 准备 Twitter thread 草稿
- [ ] 更新 GitHub README

**第 2 周**:
- [ ] 发 Hacker News Show HN
- [ ] 发 Reddit r/LocalLLaMA
- [ ] 知乎更新

**第 3-4 周**:
- [ ] 发布 zulong-memory 到 PyPI
- [ ] 写技术博客系列
- [ ] 准备投资人 Deck

---

# 第 12 章：开发者背景故事

## 12.1 为什么一个室内设计师能做出这个？

**核心原因**:
1. **AI 辅助编程**: 所有代码由 AI 助手生成，人类负责设计决策和方向把控
2. **系统思维**: 室内设计需要理解空间、流线、功能分区，这种系统思维迁移到软件架构
3. **快速学习**: 非程序员身份反而没有思维定势，敢于尝试非传统架构
4. **持续迭代**: 2 个月每天 10+ 小时迭代，累计 82,000 行代码

**技术成长路径**:
```
第 1 周: 学习 Python 基础 + FastAPI
第 2-3 周: 实现基础对话 + 工具调用
第 4-5 周: 添加记忆系统 (简单 RAG)
第 6-7 周: 实现 MemoryGraph 异构图
第 8 周: 添加 CircuitBreaker + TaskGraph
第 9-10 周: 实现语音交互 (TTS + ASR)
第 11-12 周: 集成 VS Code Extension
后续: 持续优化 + 补量测试
```

**对其他非程序员的启示**:
- AI 辅助编程降低了系统级开发的门槛
- 关键是理解系统原理，而非语法细节
- 快速迭代 + 持续验证比完美设计更重要

---

# 附录 A：核心文件索引

| 文件路径 | 行数 | 职责 |
|---------|------|------|
| `zulong/ide/ide_server.py` | 1580+ | WebSocket 服务端 |
| `zulong/ide/ide_fc_runner.py` | 4200+ | FC 循环执行器 |
| `zulong/l2/inference_engine.py` | 5700+ | L2 推理引擎 |
| `zulong/l2/fc_graph.py` | 2000+ | LangGraph FC 循环 |
| `zulong/l2/circuit_breaker.py` | 800+ | 死循环检测器 |
| `zulong/l2/task_graph.py` | 1500+ | 任务图谱 |
| `zulong/l2/attention_window.py` | 1200+ | 注意力窗口 |
| `zulong/memory/memory_graph.py` | 2784 | 异构记忆图谱 |
| `mcp_server.py` | 500+ | MCP Server |
| `zulong-ide/src/core/api/providers/zulong.ts` | 268 | WebSocket API 适配器 |
| `zulong-ide/src/core/api/transport/zulong-websocket.ts` | 500+ | WebSocket 传输层 |

---

# 附录 B：性能基准

| 指标 | 数值 | 测试环境 |
|------|------|---------|
| **热记忆检索** | <50ms | GPU 6GB+ |
| **冷记忆检索** | <200ms | GPU 6GB+ |
| **ASR 推理** | 0.5-1s | CPU, SenseVoice-Small |
| **TTS 推理** | <0.3s | CPU, Kokoro-82M |
| **端到端延迟** | 3-4s | 语音输入 → 语音输出 |
| **内存占用** | ~1.5GB | 全系统运行 |
| **GPU 显存** | ~4GB | vLLM 7B 模型 |

---

**文档结束**

**祖龙 - 让 AI 拥有真正的记忆**
