# Zulong IDE 插件开发修订方案（v2 — 深度集成版）

## Context

祖龙 (Zulong) 是一个多层次自适应智能体架构，拥有异构记忆图谱 (MemoryGraph, 9节点类型+7边类型)、任务图谱 (TaskGraph, 递归树+依赖DAG)、3模式注意力窗口 (AttentionWindow)、6信号熔断器 (CircuitBreaker)、5层FC防护链等核心能力。

**v1 方案的不足**：将祖龙简单视为一个 OpenAI 兼容 LLM API 端点，缺失以下关键需求：
1. **工具数据格式不匹配** — 祖龙内部使用 OpenAI 原生 function calling 格式，而 v1 方案通过 HTTP API 将工具调用转为 XML 文本返回，Cline 再用 XML 解析器处理，存在格式转换损耗和信息丢失
2. **未利用祖龙原生能力** — 所有 IDE 操作必须创建对话节点 (DIALOGUE) 和任务图谱 (TaskGraph)，利用记忆系统、自主注意力机制、上下文监控等原生能力
3. **通信架构错位** — 祖龙已实现 WebSocket 双向协议 (`cline_ide_server.py`)，祖龙作为唯一 Agent 大脑，插件应作为 UI + 工具执行层

**v2 修订目标**：
- 采用 WebSocket 双向通信，祖龙作为大脑，插件作为手脚
- **XML→FC 直接改造** — Cline 原有 native FC 路径（用于 GPT-5/Gemini-3）存在 provider 白名单 + model family 白名单 + variant matcher 三重门槛，之前尝试通过此路径调用祖龙 API 反复出现"工具缺少必需参数"错误（根因：Task 层仍调用 `parseAssistantMessageV2()` 解析 XML，FC tool_calls 与 XML 解析器数据流断裂）。方案改为**直接在 task/index.ts 中为 Zulong provider 开启 FC 路径**（仅 2 行条件判断），不依赖 variant 系统，彻底消除 XML 工具标签
- **后端按需加载/暴露工具** — ClineToolRegistry 按意图（COMPLEX/RESUME）+ CircuitBreaker 状态动态决定暴露给 LLM 的工具集，Cline 侧不管理工具定义
- 每次编程操作自动创建对话节点和任务图谱
- 完全利用祖龙原生的记忆系统、注意力机制、上下文监控

## 架构总览

```
┌──────────────────────────────────────────────────────────────┐
│              VS Code Extension (zulong-ide)                   │
│                                                               │
│  ┌──────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │ZulongHandler  │  │ ZulongTransport  │  │  Webview UI    │  │
│  │(ApiHandler)   │◄─┤ (WebSocket 客户端)│  │ - 聊天面板     │  │
│  │               │  │                  │  │ - 祖龙状态面板  │  │
│  │ createMessage │  │ connect/send/    │  │ - 任务进度     │  │
│  │ → ApiStream   │  │ reconnect        │  │ - 注意力模式   │  │
│  └───────┬───────┘  └────────┬─────────┘  └────────────────┘  │
│          │ yield chunks       │ JSON messages                  │
│          │ (text/tool_calls/  │ (tool_request/display_text/    │
│          │  reasoning/usage)  │  display_reasoning/            │
│          │                    │  status_update/task_complete)   │
└──────────┼────────────────────┼───────────────────────────────┘
           │                    │ WebSocket ws://127.0.0.1:8090/ide
           │                    ▼
┌──────────┼───────────────────────────────────────────────────┐
│          │     祖龙后端 (cline_ide_server.py)                 │
│          │                                                    │
│  ┌───────┴──────┐   ┌───────────────┐  ┌─────────────────┐  │
│  │ IDESession    │   │ClineFCRunner  │  │ClineToolRegistry│  │
│  │ WS↔Queue     │   │run_loop_async │  │internal+remote  │  │
│  └───────┬──────┘   └───────┬───────┘  └─────────────────┘  │
│          │            ┌──────┴──────────────────────┐        │
│          │            │  安全防护网                   │        │
│          │            │  - AttentionWindow (3模式)   │        │
│          │            │  - CircuitBreaker (6信号)    │        │
│          │            │  - RuleGuardian             │        │
│          │            │  - SemanticDriftDetector    │        │
│          │            └────────────────────────────┘        │
│     ┌────┴──────────────────────────────────────────┐       │
│     │ MemoryGraph │ TaskGraph │ DialogueAdapter      │       │
│     │ BFS激活     │ DAG管理   │ 3层对话结构           │       │
│     │ Hebbian学习 │ 状态同步   │ Session→Round→Turn  │       │
│     └───────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

**核心原则**：
1. **祖龙是大脑，插件是手脚** — LLM 调用、任务规划、记忆管理、安全防护均在 Python 后端
2. **FC 直接改造** — 不复用 Cline 现有 native FC 路径（依赖 provider 白名单/variant 匹配，已验证不可用）。直接在 `task/index.ts` 添加 `providerId === "zulong"` 条件：①强制 `enableNativeToolCalls=true` ②强制 `useNativeToolCalls=true`。tool_calls chunk 通过现有 `ToolUseHandler → processNativeToolCalls() → ToolExecutor` 链路处理，**不创建 Zulong Variant**
3. **后端按需加载工具** — `ClineToolRegistry.get_combined_tool_definitions_for_intent(intent)` 按意图/CB 状态动态决定暴露哪些工具给 LLM，Cline 侧不管理工具定义（ZulongHandler 忽略 tools 参数）
4. **渐进式改造** — 保留 Cline 原有 XML/HTTP 路径给其他 provider，Zulong provider 走 FC + WebSocket

---

## Phase 0: 后端补全——DialogueAdapter 集成 + WebSocket 适配

**目标**：修复 ClineFCRunner 的 P0 级缺陷（DialogueAdapter 完全缺失），并适配 WebSocket 通信。

### 0.1 [P0] 补全 DialogueAdapter 对话轮次记录

**问题**：当前 `ClineFCRunner` 和 `fc_graph.py` 中均**零次调用** `DialogueAdapter`，导致 MemoryGraph 中完全没有 DIALOGUE 节点。对话历史链断裂，无法通过对话历史查询、追踪对话分支、恢复到特定轮次。

**文件**: `zulong/cline/cline_fc_runner.py`

在 `_run_loop()` (行 437) 和后续新增的 `run_loop_async()` 中添加：

```python
from zulong.memory.graph_adapters import DialogueAdapter

# 在 _run_loop 开始时（_init_state 之后）：
dialogue_adapter = DialogueAdapter()
mg = get_memory_graph()
round_id = dialogue_adapter.add_round(
    graph=mg,
    user_input=state.user_input_text,
    session_id=self.session.session_id,
    task_graph_id=self._active_task_graph_id,
)

# 在每次工具执行后（_exec_tools 内部或之后）：
dialogue_adapter.add_sub_dialogue(
    graph=mg,
    round_id=round_id,
    turn=state.fc_turn,
    tool_name=tool_call["function"]["name"],
    content=resp_content[:200],
    role="assistant",
)

# 在 _finalize 时：
dialogue_adapter.finalize_round(
    graph=mg,
    round_id=round_id,
    total_turns=state.fc_turn,
    status="completed" if phase == "done" else "interrupted",
)
```

### 0.2 [P1] 恢复 ExperienceGenerator 集成

**问题**：`_auto_save_session_memory()` (行 1380) 绕过了 ExperienceGenerator，直接 `mg.add_node()` 创建 KNOWLEDGE 节点，丢失了重要度评分、domain 分类等智能处理。

**文件**: `zulong/cline/cline_fc_runner.py`

修改 `_auto_save_session_memory()` 改用 ExperienceGenerator：
```python
# 修复前：直接创建节点
node = GraphNode(node_type=NodeType.KNOWLEDGE, ...)
mg.add_node(node)

# 修复后：使用 ExperienceGenerator
from zulong.memory.experience_generator import ExperienceGenerator
exp_gen = ExperienceGenerator()
exp_gen.process_dialogue_batch(mg, dialogue_messages, task_context)
```

### 0.3 实现 `run_loop_async()` 异步 FC 循环

**文件**: `zulong/cline/cline_fc_runner.py` (第 152 行 `run_or_resume` 附近新增)

新增方法签名：
```python
async def run_loop_async(
    self,
    messages: List[Dict],
    send_callback: Callable[[str, Dict], Awaitable[None]],
    tool_result_queue: asyncio.Queue,
    cancel_event: asyncio.Event,
) -> ClineFCResult:
```

**核心设计**（以现有 `_run_loop()` 行 437 为蓝本）：

此方法支持两种运行模式，由 `cline_ide_server.py` 的协议层决定：

**模式 A（暂停/恢复模式，用于保留 Cline 完整管道）**：
- 内部工具直接执行，远程工具触发 `_pause_for_cline()`
- 通过 `send_callback("tool_request", ...)` 发送工具请求
- 通过 `send_callback("display_text", ...)` 实时推送文本
- 当遇到远程工具时，**暂停并返回**（等待下次 `tool_result` 消息触发恢复）
- 每个 FC 周期对应 Cline 一次 `createMessage()` 调用

**模式 B（持续循环模式，用于未来不依赖 Cline UI 的场景）**：
- 远程工具通过 `tool_result_queue` 等待结果
- FC 循环在后端持续运行不暂停

**MVP 首选模式 A**——完美适配 Cline 的递归主循环，保留 100% 编程能力。

### 0.4 扩展 WebSocket 协议

**文件**: `zulong/cline/cline_ide_server.py`

在 `_run_fc_loop` 中丰富 `send_callback` 推送的元数据，并传递 Cline system prompt：
```python
# _run_fc_loop 中修改：接收 Cline 的 system_prompt 并传给 _build_initial_messages
messages = _build_initial_messages(
    engine, task_text, cwd,
    cline_system_prompt=payload.get("system_prompt", ""),
)

# 每轮结束推送祖龙状态
await send_callback("status_update", {
    "turn": state.fc_turn,
    "phase": state.phase,
    "attention_mode": attn_window.current_mode if attn_window else "GLOBAL",
    "memory_hits": last_memory_hit_count,
    "task_progress": {"completed": N, "total": M, "current_node": "..."},
    "circuit_breaker_state": circuit_breaker.state.value if circuit_breaker else "GREEN",
})
```

新增 `session_resume` 消息处理器，支持 WebSocket 断连重连后通过 `session_id` 恢复会话。

---

## Phase 1: 插件端 Zulong Provider + WebSocket Transport

**目标**：创建 `ZulongHandler` (ApiHandler)，通过 WebSocket 与祖龙后端通信，将后端推送消息转译为 Cline 标准 `ApiStream`。

### 1.1 WebSocket Transport 层

**新文件**: `zulong-ide/src/core/api/transport/zulong-websocket.ts`

```typescript
export class ZulongWebSocketTransport {
    private ws: WebSocket | null = null
    private sessionId: string | null = null
    
    // 连接管理
    async connect(url: string): Promise<void>      // 建立连接，等待 session_ack
    disconnect(): void                               // 断开连接
    private reconnect(): Promise<void>               // 指数退避重连
    
    // 消息收发
    send(type: string, payload: object): void        // 发送消息到后端
    onMessage(type: string, callback: Function): void // 注册消息回调
    
    // 状态
    get connected(): boolean
    get sessionId(): string | null
}
```

协议消息（与 `cline_ide_server.py` 完全对齐）：

| 方向 | 类型 | Payload |
|------|------|---------|
| 插件→后端 | `session_start` | `{task, cwd, system_prompt, mode?}` |
| 插件→后端 | `tool_result` | `{call_id, tool_name, result, is_error}` |
| 插件→后端 | `user_cancel` | `{}` |
| 后端→插件 | `session_ack` | `{session_id}` |
| 后端→插件 | `tool_request` | `{call_id, tool_name, arguments}` |
| 后端→插件 | `display_text` | `{content}` |
| 后端→插件 | `display_reasoning` | `{content}` |
| 后端→插件 | `status_update` | `{turn, phase, attention_mode, ...}` |
| 后端→插件 | `task_complete` | `{result}` |
| 后端→插件 | `task_error` | `{error}` |

### 1.2 ZulongHandler 实现

**新文件**: `zulong-ide/src/core/api/providers/zulong.ts`

实现 `ApiHandler` 接口（定义在 `src/core/api/index.ts:53`）：

```typescript
export class ZulongHandler implements ApiHandler {
    private transport: ZulongWebSocketTransport
    
    // 核心：WebSocket 消息 → ApiStream 桥接
    async *createMessage(
        systemPrompt: string,
        messages: ClineStorageMessage[],
        tools?: ClineTool[]
    ): ApiStream {
        // 1. 发送 session_start（仅携带最新用户消息和 cwd）
        // 2. 通过 Promise 队列接收后端推送消息并转译：
        //    display_text    → yield { type: "text", text: content }
        //    display_reasoning → yield { type: "reasoning", reasoning: content }
        //    tool_request    → yield { type: "tool_calls", tool_call: {call_id, function: {name, arguments}} }
        //    task_complete   → return（generator 结束）
        //    task_error      → throw Error
        //    status_update   → 不产出 chunk，通过事件通知 UI
    }
    
    getModel(): ApiHandlerModel { /* 返回 zulong-agent 模型信息 */ }
    abort(): void { /* 发送 user_cancel */ }
    
    // Zulong 专有：回传工具执行结果
    sendToolResult(callId: string, toolName: string, result: string, isError: boolean): void
}
```

**核心设计**：`createMessage()` 是 **WebSocket→ApiStream 的桥接适配器**。Cline 主循环照常从 `ApiStream` 消费 chunks，完全不需感知底层是 HTTP SSE 还是 WebSocket。

### 1.3 XML→FC 直接改造（仅改 task/index.ts 2 行）

**为什么不复用现有 native FC 路径？**

Cline 的 native FC 路径（GPT-5/Gemini-3 等）存在三重门槛：
1. `isNextGenModelProvider()` 硬编码 16 个 provider 白名单（zulong 不在其中）
2. `isNextGenModelFamily()` 要求 model ID 包含 "gpt-5"、"gemini-3" 等关键词
3. 仅 4 个 Variant 有 `use_native_tools:1`，且绑定严格的 matcher
4. **根因**：即使绕过白名单，Task 层仍调用 `parseAssistantMessageV2()` 解析 XML，当后端返回 FC `tool_calls` 时 XML 解析器找不到标签 → `assistantMessageContent` 为空 → ToolValidator 报"缺少必需参数"

之前尝试通过此路径调用祖龙 API 反复失败，正是因为上述第 4 点的数据流断裂。

**直接改造方案**：在 `task/index.ts` 中仅添加 2 行 provider 条件判断，强制 Zulong provider 走 FC 处理路径。**不创建 Zulong Variant，不修改 variant 系统，不注册 getNativeConverter**。

**文件**: `zulong-ide/src/core/task/index.ts`

**改动 1** — enableNativeToolCalls 条件（约行 1987）：
```typescript
enableNativeToolCalls:
    providerInfo.model.info.apiFormat === ApiFormat.OPENAI_RESPONSES ||
    providerInfo.providerId === "zulong" ||  // Zulong 直接启用 FC
    this.stateManager.getGlobalStateKey("nativeToolCallEnabled"),
```

**改动 2** — useNativeToolCalls 覆盖（约行 2001，getSystemPrompt 之后）：
```typescript
const { systemPrompt, tools } = await getSystemPrompt(promptContext)
// Zulong 模式：后端管理工具定义，不需要 Cline 侧提供 tools，但需要启用 FC 处理路径
this.useNativeToolCalls = !!tools?.length || providerInfo.providerId === "zulong"
```

**为什么不需要 Zulong Variant？**
- 工具定义由后端 `ClineToolRegistry` 管理，不需要 Cline 侧的 `getNativeTools()` 提供
- System prompt 由 Cline 默认 variant 生成（会含 XML 工具文档），但后端 `ClinePromptHandler.extract_cline_tool_block()` 自动剥离 XML 文档并注入祖龙增强
- FC 路径通过 `providerId === "zulong"` 直接启用，不依赖 variant 的 `use_native_tools` 标签

**直接改造后的 FC 数据流**：

```
┌─ Zulong 模式：直接 FC 改造，无需 Variant ─────────────────────────┐
│                                                                      │
│  ① getSystemPrompt(context)                                         │
│     默认 variant 生成 systemPrompt（含 XML 工具文档 + 环境上下文）   │
│     tools = undefined（没有 Zulong Variant 提供 native tools）       │
│     但 useNativeToolCalls 被覆盖为 true（改动 2）                    │
│                                                                      │
│  ② createMessage(systemPrompt, messages, tools=undefined)            │
│     ZulongHandler 将 systemPrompt 透传给后端（session_start）        │
│     后端 ClinePromptHandler:                                         │
│       - STRIP: XML 工具文档（extract_cline_tool_block 已实现）       │
│       - KEEP: 环境上下文（workspace roots, .clinerules 等）          │
│       - INJECT: 祖龙增强（memory + task + experience）               │
│     后端 _call_model() 使用 ClineToolRegistry 提供的 tool_defs      │
│                                                                      │
│  ③ 后端按需加载工具                                                  │
│     ClineToolRegistry.get_combined_tool_definitions_for_intent()     │
│     COMPLEX: 全部内部工具 + 全部远程工具                             │
│     RESUME:  排除 task_create_plan/task_add_node + 全部远程工具      │
│     CB RED:  仅保留 recall_memory + attempt_completion               │
│                                                                      │
│  ④ 后端 tool_request → ZulongHandler yield tool_calls chunk          │
│     进入 task/index.ts stream loop case "tool_calls":                │
│       ToolUseHandler.processToolUseDelta() → 积累 FC 参数            │
│       processNativeToolCalls() → 设置 assistantMessageContent        │
│       (标记 isNativeToolCall=true)                                   │
│     parseAssistantMessageV2() 仍运行于 text chunk →                  │
│       找不到 XML 标签 → 返回 text-only 块 → 被 processNativeToolCalls│
│       的 REPLACE 操作覆盖（不是追加，是替换），所以完全无害           │
│                                                                      │
│  ⑤ stream 结束（ZulongHandler 收到 tool_request 后 return）          │
│     processNativeToolCalls(partial=false) → 最终化                   │
│     flushAssistantPresentationOrThrow() → ToolExecutor               │
│     → DiffView / Terminal / Browser 等全部 UI 能力                   │
│                                                                      │
│  ⑥ 工具执行完成 → 结果格式化为 role="tool"（因 isNativeToolCall=true）│
│     → recursivelyMakeClineRequests(userMessageContent)               │
│     → ZulongHandler.createMessage() 检测 tool result → 发送回后端   │
│     → 后端 run_or_resume(tool_results) 恢复 FC 循环                 │
└──────────────────────────────────────────────────────────────────────┘
```

**与现有 native FC 路径的对比**：

| 维度 | 现有 native FC (GPT-5/Gemini-3) | Zulong 直接改造 |
|------|------|------|
| 启用机制 | Variant `use_native_tools:1` + 白名单 | `providerId === "zulong"` 直接判断 |
| 工具定义来源 | Cline `ClineToolSet.getNativeTools()` | 后端 `ClineToolRegistry`（Cline 侧不管） |
| System Prompt | Variant 简化 TOOL_USE 模板（无 XML） | 默认 variant（含 XML，后端自动剥离） |
| Variant 依赖 | 需要创建 + 注册 + matcher | 不需要任何 Variant |
| 改动量 | ~5 个文件（variant/toolset/task/converter） | 1 个文件 2 行代码（task/index.ts） |

### 1.4 Provider 类型注册

**文件**: `zulong-ide/src/shared/api.ts`
- `ApiProvider` 联合类型添加 `"zulong"`，设为 `DEFAULT_API_PROVIDER`
- 模型定义：

```typescript
export const zulongModels = {
  "zulong-agent": {
    maxTokens: 8192,
    contextWindow: 131072,
    supportsImages: true,
    supportsPromptCache: false,
    supportsReasoning: true,    // 后端推送 display_reasoning
    inputPrice: 0,
    outputPrice: 0,
    description: "祖龙多层自适应智能体系统",
  },
} as const satisfies Record<string, ModelInfo>
```

注意：不需要设置 `supportsNativeFunctionCalling` 等字段，因为 FC 路径通过 `task/index.ts` 中的 `providerId === "zulong"` 直接启用（Phase 1.3），不依赖模型 info 属性。

**文件**: `zulong-ide/src/shared/storage/state-keys.ts`
- 添加 `zulongServerUrl: { default: "ws://127.0.0.1:8090/ide" as string | undefined }`
- 不需要 `zulongApiKey`（后端自带 LLM 接入，无需前端传 key）

**文件**: `zulong-ide/src/core/api/index.ts`
- `case "zulong": return new ZulongHandler(options)`

**文件**: `zulong-ide/src/shared/providers/providers.json`
- 列表顶部插入 `{"value": "zulong", "label": "Zulong (祖龙)"}`

### 1.5 设置面板

**新文件**: `zulong-ide/webview-ui/src/components/settings/providers/ZulongProvider.tsx`
- WebSocket URL 输入框（预填 `ws://127.0.0.1:8090/ide`）
- 连接状态指示灯（通过 `/health` HTTP 端点检测）
- 不需要 API Key 输入框

**文件**: `zulong-ide/webview-ui/src/components/settings/ApiOptions.tsx` — 注册组件
**文件**: `zulong-ide/webview-ui/src/utils/validate.ts` — 验证逻辑
**文件**: `zulong-ide/webview-ui/src/components/settings/utils/providerUtils.ts` — normalize/getModels

---

## Phase 2: 复用 Cline 完整编程管道（零丢失方案）

**目标**：让 Zulong 模式 100% 继承 Cline 的所有编程能力——Diff 视图、终端实时输出、浏览器自动化、Checkpoint 系统、ask/say 交互、Plan/Act 模式切换，一个都不丢。

### 2.1 核心设计：适配而非替代

**关键洞察**：Cline 的 `recursivelyMakeClineRequests()` 递归主循环中，70-80% 是 UI/演示逻辑（Diff 视图、终端集成、StreamChunkCoordinator、TaskPresentationScheduler 等）。如果创建独立的 `runZulongMode()` 绕过它，会丢失这些关键能力。

**解决方案**：**不修改 `recursivelyMakeClineRequests()`**。让 `ZulongHandler.createMessage()` 返回的 `ApiStream` 完全适配 Cline 现有的递归循环模式——每次 `createMessage()` 调用对应后端一次 `run_or_resume()` 暂停/恢复周期。

### 2.2 消息流映射

Cline 原有流程：
```
recursivelyMakeClineRequests(userContent)
  → createMessage(systemPrompt, messages, tools) → ApiStream
  → for await (chunk of stream): text/tool_calls/reasoning
  → 工具执行（Diff视图、终端、浏览器...全部UI能力）
  → 收集工具结果到 userMessageContent
  → recursivelyMakeClineRequests(userMessageContent)  [递归]
```

Zulong 模式下**完全保持相同流程**，只是 `createMessage()` 内部实现不同：

```
第1次 createMessage() 调用：
  → 检查 messages 参数：最后是 user 消息 → 新任务
  → 通过 WebSocket 发送 session_start
  → 后端 run_or_resume(new_messages=...) 执行 FC 循环
  → 后端推送 display_text → yield {type:"text"}
  → 后端推送 display_reasoning → yield {type:"reasoning"}
  → 后端遇到远程工具 → _pause_for_cline() → 推送 tool_request
  → yield {type:"tool_calls", tool_call:{...}}
  → stream 结束（后端已暂停等待工具结果）

Cline 主循环正常执行工具（Diff视图、终端、审批对话框...）

第2次 createMessage() 调用：
  → 检查 messages 参数：最后是 tool result → 工具结果回传
  → 通过 WebSocket 发送 tool_result
  → 后端 run_or_resume(tool_results=...) 恢复 FC 循环
  → 重复上述推送过程...
  → 直到后端返回 phase="done" → 推送 task_complete → stream 结束

Cline 主循环正常结束递归。
```

### 2.3 ZulongHandler.createMessage() 实现细节

**文件**: `zulong-ide/src/core/api/providers/zulong.ts`

```typescript
async *createMessage(
    systemPrompt: string,
    messages: ClineStorageMessage[],
    tools?: ClineTool[]
): ApiStream {
    // 判断是新任务还是工具结果回传
    const lastMsg = messages[messages.length - 1]
    const isToolResult = lastMsg?.role === "tool" || this.hasPendingToolResults(messages)
    
    if (isToolResult) {
        // 回传工具结果 → 恢复 FC 循环
        const toolResults = this.extractToolResults(messages)
        this.transport.send("tool_result", toolResults)
    } else {
        // 新任务 → 启动 FC 循环
        const userMessage = this.extractLatestUserMessage(messages)
        this.transport.send("session_start", {
            task: userMessage,
            cwd: this.getCwd(),
            system_prompt: systemPrompt,  // 后端保留环境上下文、替换工具定义、注入祖龙增强
            mode: this.getClineMode?.(),  // 可选：传递 Cline 的 Plan/Act 模式
        })
    }
    
    // 消费后端推送的消息，转译为 ApiStreamChunk
    while (true) {
        const msg = await this.transport.receiveNext()
        
        switch (msg.type) {
            case "display_text":
                yield { type: "text", text: msg.payload.content }
                break
            case "display_reasoning":
                yield { type: "reasoning", reasoning: msg.payload.content }
                break
            case "tool_request":
                yield {
                    type: "tool_calls",
                    tool_call: {
                        call_id: msg.payload.call_id,
                        function: {
                            name: msg.payload.tool_name,
                            arguments: JSON.stringify(msg.payload.arguments),
                        }
                    }
                }
                return  // stream 结束，Cline 开始执行工具
            case "task_complete":
                if (msg.payload.result) {
                    yield { type: "text", text: msg.payload.result }
                }
                return  // 任务完成
            case "task_error":
                throw new Error(msg.payload.error)
            case "status_update":
                this.emitStatusUpdate(msg.payload)  // 通过事件通知 UI
                break
        }
    }
}
```

### 2.4 保留的完整能力清单

因为 **不修改 `recursivelyMakeClineRequests()`**，以下能力 100% 保留：

| 能力 | 状态 | 说明 |
|------|------|------|
| ✅ DiffViewProvider | 保留 | 文件编辑显示 VS Code 原生 Diff 视图 |
| ✅ 终端实时输出 | 保留 | VscodeTerminalProcess + Shell Integration |
| ✅ 浏览器自动化 | 保留 | Puppeteer + BrowserSession |
| ✅ StreamChunkCoordinator | 保留 | 流编排、token 实时更新 |
| ✅ TaskPresentationScheduler | 保留 | 内容节流/合并 |
| ✅ ask/say 交互 | 保留 | 工具审批对话框、用户确认 |
| ✅ Checkpoint 系统 | 保留 | 检查点保存/恢复/Diff 对比 |
| ✅ Plan/Act 模式 | 保留 | 模式切换完整 |
| ✅ Auto-approve | 保留 | 工具审批白名单 |
| ✅ 上下文管理 | 保留 | ContextManager、紧凑化 |
| ✅ 消息状态管理 | 保留 | MessageStateHandler 完整 |
| ✅ Slash 命令 | 保留 | /newrule, /condense 等 |
| ✅ @mentions | 保留 | 文件引用解析 |

### 2.5 Task 主循环——唯一需要的适配

**文件**: `zulong-ide/src/core/task/index.ts`

**无需修改 `recursivelyMakeClineRequests()`**。唯一需要的适配已在 Phase 1.3 中完成：

1. `enableNativeToolCalls` 条件添加 `providerInfo.providerId === "zulong"`
2. `useNativeToolCalls` 覆盖为 `true`（即使 getSystemPrompt 不返回 tools）

由于 `ZulongHandler.createMessage()` yield 的是标准 `ApiStreamToolCallsChunk` 格式，Cline 现有的 `StreamResponseHandler.ToolUseHandler` 和 `processNativeToolCalls()` (行 3448) 会自动处理。`parseAssistantMessageV2()` 对 text chunk 的 XML 解析返回空结果，被 `processNativeToolCalls()` 的 REPLACE 操作覆盖——完全无害。

---

## Phase 2.5: Cline ↔ 祖龙 任务编排协调机制

**目标**：解决完整保留 Cline 编程管道与祖龙后端任务编排之间的 5 大协调冲突，确保"双层架构"（Cline = 编程执行层，祖龙 = 认知决策层）无缝协作。

### 协调原则

```
┌─ Cline 管道（编程执行层）──────────────────────────┐
│  UI 渲染 · Diff 视图 · 终端集成 · 审批对话框       │
│  工具执行 · Checkpoint · Plan/Act · 上下文管理       │
│  职责：看得见、摸得着的一切                          │
└───────────────┬───────────────────────────────────┘
                │ WebSocket（仅传递：task/tool_result/cancel）
                ▼
┌─ 祖龙后端（认知决策层）──────────────────────────────┐
│  LLM 调用 · 任务规划 · 记忆检索 · 安全防护           │
│  AttentionWindow · CircuitBreaker · DialogueAdapter  │
│  职责：想得到、记得住的一切                           │
└──────────────────────────────────────────────────────┘
```

### 2.5.1 System Prompt 权限分工

**冲突**：Cline 的 `getSystemPrompt()` (task/index.ts:2000) **总是生成**完整 system prompt（默认 variant 含 XML 工具定义、环境上下文、rules），`createMessage()` 的第一个参数 `systemPrompt` 是必传的，无法跳过。后端 `ClinePromptHandler` 也构建自己的增强 system prompt。

**解决方案**："前端采集 + 后端增强"分工协议（**无需 Zulong Variant**）

```
Cline getSystemPrompt() ─生成─→ systemPrompt (含环境+工具定义+rules)
    │
    ▼
ZulongHandler.createMessage(systemPrompt, ...) ─传递─→ session_start {system_prompt}
    │
    ▼
后端 _build_initial_messages() ─接收─→ ClinePromptHandler.process_system_prompt()
    │
    ├─ ① DISCARD: Cline 的 XML 工具定义 (extract_cline_tool_block 已实现)
    ├─ ② KEEP: Cline 的环境上下文 (workspace roots, rules, terminal mode)
    └─ ③ INJECT: 祖龙增强 (memory_context + task_context + experience_hints)
```

- 工具定义由后端 `ClineToolRegistry.get_combined_tool_definitions_for_intent()` 独立提供
- Cline 生成的环境信息（cwd, workspace roots, .clinerules, terminal mode 等）被保留，这些是后端无法获取的 IDE 环境感知
- **对 Cline 侧代码：零修改** — system prompt 照常生成，ZulongHandler 只是透传

**后端改动** (`cline_ide_server.py:_build_initial_messages`):
```python
def _build_initial_messages(engine, task_text: str, cwd: str, 
                            cline_system_prompt: str = "") -> list:
    """接收 Cline 的 system prompt，经过增强处理"""
    handler = ClinePromptHandler()
    memory_ctx, task_ctx, exp_hints = handler.retrieve_zulong_context(task_text)
    
    # 复用 Cline 的环境上下文，替换工具定义，注入祖龙增强
    messages = [
        {"role": "system", "content": cline_system_prompt or ""},
        {"role": "user", "content": task_text},
    ]
    return handler.process_system_prompt(
        messages, memory_ctx, task_ctx, exp_hints, intent="complex"
    )
```

### 2.5.2 上下文管理隔离

**冲突**：Cline 的 `ContextManager.getNewContextMessagesAndMetadata()` (ContextManager.ts:227) 在 `createMessage()` 之前截断消息历史；祖龙的 `AttentionWindow.apply_window()` (cline_fc_runner.py:501) 也管理消息可见性。

**解决方案**：两套系统自然隔离，无需协调

```
┌─ Cline 侧 ─────────────────────┐   ┌─ 祖龙后端 ────────────────────┐
│ apiConversationHistory          │   │ state.messages                 │
│   ↓ ContextManager 截断         │   │   ↓ AttentionWindow 过滤       │
│ truncatedConversationHistory    │   │ filtered_messages              │
│   ↓ 传给 createMessage()       │   │   ↓ 传给 LLM API              │
│                                  │   │                                │
│ ZulongHandler 只提取:            │   │ 后端只接收:                    │
│  - 新任务: user 消息文本         │──→│  - session_start {task}        │
│  - 工具结果: tool result 内容    │──→│  - tool_result {results}       │
│ 不转发完整消息历史               │   │ 独立维护自己的消息历史          │
└──────────────────────────────────┘   └────────────────────────────────┘
```

- ZulongHandler **不转发** `truncatedConversationHistory` 全部内容，只提取最新有效信息
- 上下文窗口溢出仅发生在后端 LLM 调用中，由后端 `AttentionWindow` 处理
- Cline 的上下文溢出重试机制 (task/index.ts:2039) 在 Zulong 模式下不触发（因为 `createMessage()` 不做真正的 LLM 调用，不会抛出 context window exceeded 错误）
- **对 Cline 侧代码：零修改**

### 2.5.3 Plan/Act 模式通信

**冲突**：Cline 有 Plan/Act 模式切换（影响 system prompt 变体和 LLM 输出格式）；祖龙有 COMPLEX/RESUME 意图（影响工具过滤和提示词模板）。

**解决方案**：正交关注点，互不干扰

| 维度 | Cline Plan/Act | 祖龙 COMPLEX/RESUME |
|------|---------------|---------------------|
| 作用 | LLM 输出格式（结构化规划 vs 直接行动） | 任务图生命周期（新建 vs 续做） |
| 影响 | system prompt 变体选择 | 内部工具可用性 + 提示词模板 |
| 执行位置 | 插件端 prompt 生成 | 后端 FC 循环内部 |
| 对方可见性 | 后端通过 system_prompt 感知 | Cline 不感知（内部工具透明） |

- 祖龙 RESUME 意图排除 `task_create_plan`、`task_add_node`，强制首轮 `task_view_overview` —— 这些都是内部工具，Cline 完全不感知
- Cline Plan/Act 模式影响的 system prompt 文案，通过 2.5.1 的分工协议保留到后端
- `session_start` 可选传递 `mode: "plan" | "act"` 供后端参考优化提示词
- **对 Cline 侧代码：零修改**

### 2.5.4 错误处理分层

**冲突**：Cline 的 `consecutiveMistakeCount` + auto-retry (max 3, 2s/4s/8s) vs 祖龙的 CircuitBreaker + 6 层安全网。

**解决方案**：传输层 vs 语义层天然分离

| 层级 | 负责方 | 机制 | 对方感知 |
|------|--------|------|---------|
| 传输层 | Cline | WS 断连重连, createMessage 超时重试 | 后端不感知 |
| 语义层 | 祖龙 | CircuitBreaker, 漂移检测, RuleGuardian, InfoGap | Cline 不感知 |

**关键路径分析**：
- CB RED → 后端停止发送 `tool_request`，改发 `display_text` + `task_complete` → Cline 正常渲染文本、结束递归
- 语义漂移 → 后端内部注入纠偏提示，继续 FC 循环 → Cline 只看到 stream 持续 yield `display_text`
- `consecutiveMistakeCount` 在 Zulong 模式下基本不触发 —— 后端总是返回 `tool_request`（产生 tool_calls chunk）或 `task_complete`（正常结束递归）
- Cline auto-retry 仅在 WebSocket 通信层错误时触发（如 receiveNext 超时），与后端语义安全网无交集
- **对 Cline 侧代码：零修改**

### 2.5.5 内部工具可见性

**冲突**：后端内部工具（task_create_plan, recall_memory, navigate_attention 等）在 `_exec_internal()` (cline_fc_runner.py:722) 中静默执行，用户无感知。

**解决方案**：通过 `display_text` 实时推送内部活动摘要

**后端改动** (`cline_fc_runner.py:_exec_internal` 尾部新增，或 `run_loop_async` 中注入):
```python
# 每执行一个内部工具后，推送可读摘要到插件
_TOOL_DISPLAY_NAMES = {
    "task_create_plan": "创建任务规划",
    "task_add_node": "添加子任务",
    "task_mark_status": "更新任务状态",
    "task_view_overview": "查看任务概览",
    "recall_memory": "检索记忆",
    "navigate_attention": "切换注意力焦点",
}
display = _TOOL_DISPLAY_NAMES.get(tn, tn)
await send_callback("display_text", {
    "content": f"[祖龙] {display}: {rt[:100]}",
    "is_internal": True,  # 插件端可据此渲染为灰色/折叠
})
```

- `display_text` 通过 ZulongHandler yield 为 `{type: "text"}` chunk
- Cline 主循环正常渲染为聊天消息文本
- 插件端可通过 `is_internal` 标记做差异化渲染（如折叠、灰色字体）—— Phase 4 状态面板实现
- `status_update` 消息驱动结构化状态面板更新（与 display_text 互补）

### 2.5.6 协调总结：Cline 侧零修改

| 冲突 | 解决方案 | Cline 改动 | 后端改动 |
|------|---------|-----------|---------|
| System Prompt 权限 | 前端采集+后端增强 | 零 | `_build_initial_messages` 接收 Cline prompt |
| 上下文管理 | 自然隔离 | 零 | 零（已有 AttentionWindow） |
| Plan/Act 模式 | 正交关注点 | 零 | 零（已有 COMPLEX/RESUME） |
| 错误处理 | 传输/语义分层 | 零 | 零（CB/安全网已有） |
| 内部工具可见 | display_text 推送 | 零 | `_exec_internal` 尾部加推送 |

**核心结论**：Phase 2.5 的所有协调机制都通过 **WebSocket 协议层** 和 **后端适配** 实现，Cline 的 `recursivelyMakeClineRequests()` 和所有子系统（ContextManager、Plan/Act、error handling）保持原样不变。这印证了 Phase 2 "零丢失"方案的可行性。

---

## Phase 3: 品牌重塑 (Cline → Zulong IDE)

### 3a: 关键身份标识

**文件**: `zulong-ide/package.json`
- `name`: `"claude-dev"` → `"zulong-ide"`
- `displayName`: `"Cline"` → `"Zulong IDE"`
- `description`: 更新为祖龙描述
- `publisher`: 更新
- `contributes.commands`: 所有 title 中 "Cline" → "Zulong"
- `contributes.walkthroughs`: ID 和文案替换

**文件**: `zulong-ide/src/registry.ts`
- prefix 逻辑：`const prefix = name === "zulong-ide" ? "zulong" : name`

**文件**: `zulong-ide/src/core/prompts/system-prompt/components/agent_role.ts`
- `"You are Cline,"` → `"You are Zulong (祖龙),"`

### 3b: UI 界面文案

- `webview-ui/src/components/welcome/WelcomeView.tsx` — 欢迎文案
- `webview-ui/src/assets/` — Logo 组件替换
- `walkthrough/step1-5.md` — 引导文案
- `assets/icons/` — 图标替换

### 3c: 存储路径

**文件**: `zulong-ide/src/core/storage/disk.ts`
- `".cline"` → `".zulong"`
- `"cline_mcp_settings.json"` → `"zulong_mcp_settings.json"`
- `".clinerules"` → `".zulongrules"`

### 3d: 批量文本替换

- `webview-ui/src/components/` — 用户可见字符串中 "Cline" → "Zulong"
- `src/core/prompts/system-prompt/variants/*/overrides.ts` — "You are Cline" 替换
- System Prompt 中所有 Agent 自称

---

## Phase 4: 祖龙原生能力集成

### 4.1 祖龙状态面板 (Webview)

**目标**：在侧边栏聊天面板中嵌入可折叠的"祖龙状态"面板。

**新文件**: `zulong-ide/webview-ui/src/components/chat/ZulongStatusPanel.tsx`

显示内容（数据来源：`status_update` WebSocket 消息）：

| 显示项 | 数据字段 | 可视化 |
|--------|---------|--------|
| FC 轮次 | `turn` | `Turn N` 计数器 |
| 注意力模式 | `attention_mode` | 三色徽标：GLOBAL=蓝, FOCUS=橙, SINGLE_CHAIN=红 |
| 任务进度 | `task_progress` | 进度条 `completed/total` + 当前节点名 |
| 熔断器状态 | `circuit_breaker_state` | GREEN隐藏, YELLOW警告, RED错误 |
| 执行阶段 | `phase` | 文字标签 (initializing/running/...) |

**数据传递路径**：
```
后端 status_update → ZulongTransport.onMessage → 
VS Code postMessage → Webview ZulongStatusPanel 渲染
```

### 4.2 记忆架构完整集成（Phase 0 修复后）

Phase 0 修复 DialogueAdapter 和 ExperienceGenerator 后，祖龙记忆系统在 FC 循环中的覆盖情况：

| 祖龙能力 | 集成状态 | 调用位置 |
|---------|---------|---------|
| **MemoryGraph 记忆检索** | ✅ 完整 | `cline_prompt_handler.py:357` `retrieve_zulong_context()` 三库检索 |
| **DialogueAdapter 对话轮次** | ✅ Phase 0 修复 | `cline_fc_runner.py` `add_round()` / `add_sub_dialogue()` / `finalize_round()` |
| **TaskGraph 同步** | ✅ 完整 | `cline_fc_runner.py:1261` `TaskGraphAdapter().sync()` 每轮触发 |
| **BFS 激活** | ✅ 完整 | `cline_fc_runner.py:1252` `_run_bfs_activation()` 含焦点切换 |
| **AttentionWindow 3模式** | ✅ 完整 | 每轮消息注册 + `observe_tool_call()` 模式转换 |
| **CircuitBreaker 6信号** | ✅ 完整 | `_eval_response()` 中的安全网 |
| **RuleGuardian** | ✅ 完整 | `_eval_response()` 过早完成检测 |
| **InfoGap 检测** | ✅ 完整 | `_eval_response()` 信息缺口检测 |
| **ExperienceGenerator** | ✅ Phase 0 修复 | `_auto_save_session_memory()` 使用智能经验提取 |
| **SemanticDriftDetector** | ✅ 完整 | `cline_fc_runner.py:221` per-runner 实例 |

**插件端需要做的**：确保 `session_start` 消息中传递充分的上下文（task 文本 + cwd），后端自动完成所有记忆架构操作。每次工具执行的结果通过 `tool_result` 回传后，后端的 `_inject_tool_results()` 会自动将结果纳入注意力窗口和对话记录。

### 4.3 System Prompt 适配

**说明**：在 WebSocket 模式下，System Prompt 采用 Phase 2.5.1 的"前端采集+后端增强"分工协议：
1. Cline 使用默认 variant 正常生成 system prompt（含 XML 工具文档 + 环境上下文：workspace roots, .clinerules, terminal mode 等）
2. ZulongHandler 在 `session_start` 中将 Cline 的 system_prompt 原样传递给后端
3. 后端 `ClinePromptHandler.process_system_prompt()` 剥离 XML 工具文档（`extract_cline_tool_block()` 已实现）、保留环境上下文、注入祖龙增强内容（memory, task, experience）
4. 工具定义由后端 `ClineToolRegistry` 独立管理，使用 OpenAI 原生 FC 格式
5. **无需 Zulong Variant** — 默认 variant 生成的 XML 工具文档在后端被自动剥离

插件端 System Prompt（`agent_role.ts`）仅在用户切换到非 Zulong provider 时使用。

---

## Phase 5: HTTP API 修复（降级路径）

**目标**：保留 `cline_api_server.py` 作为非 WebSocket 场景的降级方案，同时修复原生 FC 格式支持。

**文件**: `zulong/cline/cline_api_server.py`

修复 `_stream_response()` (行 786-822)：当模型返回 `tool_calls` 时，使用标准 OpenAI SSE 格式：

```python
# 修复前：所有内容作为 delta.content 文本返回
chunk = {"choices": [{"delta": {"content": chunk_content}}]}

# 修复后：tool_calls 使用原生格式返回
if result.phase == "waiting_cline" and result.xml_tool_calls:
    # 将 xml_tool_calls 转回 OpenAI tool_calls 格式
    tool_calls = ClineFormatTranslator.parse_xml_tool_calls(result.xml_tool_calls)
    chunk = {"choices": [{"delta": {"tool_calls": tool_calls}}]}
```

这样 Cline 原生的 OpenAI provider（配置 baseURL 指向 `http://127.0.0.1:8080/v1`）也能通过标准 FC 路径工作。

---

## Phase 6: 构建与验证

### 构建步骤
```bash
cd zulong-ide
npm install
npm run install:all
npm run protos
npm run check-types
npm run compile
```

### 验证清单

| 验证项 | 验证方法 | 阶段 |
|--------|---------|------|
| **WebSocket 连接** | 启动后端 `cline_ide_server.py` → 插件选择 Zulong provider → 确认连接状态灯为绿色 | P0+P1 |
| **端到端对话** | 发送 "读取当前目录文件列表" → 确认插件执行 `list_files` 工具 → 返回结果 → 后端继续推理 | P0+P1+P2 |
| **工具结果回传** | 观察工具执行后 WebSocket 发送 `tool_result` 消息 → 后端日志显示 `_inject_tool_results` | P0+P2 |
| **多轮工具调用** | 发送复杂任务（如 "重构 utils.py"）→ 确认多次 tool_request/tool_result 循环 | P2 |
| **对话节点创建** | 任务完成后检查 MemoryGraph 中新增 DIALOGUE 节点 | P0 (后端) |
| **任务图谱** | 复杂任务完成后检查 TaskGraph 节点和依赖关系 | P0 (后端) |
| **祖龙状态面板** | 观察侧边栏注意力模式、任务进度、FC 轮次实时更新 | P4 |
| **断连重连** | 手动断开 WebSocket → 确认自动重连 → 确认会话恢复 | P1 |
| **品牌显示** | 侧边栏 "Zulong IDE"，欢迎页 "Hi, I'm Zulong (祖龙)" | P3 |
| **降级路径** | 停止 WebSocket 服务 → 配置 OpenAI provider 指向 HTTP API → 确认仍可对话 | P5 |
| **其他 Provider** | 切换到 Anthropic/OpenAI → 确认原有功能不受影响 | 全局 |
| **FC 直接改造** | 确认 Zulong 模式下 task/index.ts 中 enableNativeToolCalls 和 useNativeToolCalls 均为 true；tool_calls chunk 经 ToolUseHandler → processNativeToolCalls → ToolExecutor 完整链路执行；无"工具缺少必需参数"错误 | P1 |

### 测试命令
```bash
npm run test:unit
npm run check-types
npm run lint
```

---

## 关键文件清单

| 阶段 | 文件路径 | 操作 |
|------|---------|------|
| P0 | `zulong/cline/cline_fc_runner.py` | 修改：[P0] DialogueAdapter 集成 + [P1] ExperienceGenerator 修复 + 新增 `run_loop_async()` |
| P0 | `zulong/cline/cline_ide_server.py` | 修改：扩展 status_update、新增 session_resume |
| P1 | `zulong-ide/src/core/task/index.ts` | 修改：2 行改动——enableNativeToolCalls 条件 + useNativeToolCalls 覆盖（FC 直接改造核心） |
| P1 | `zulong-ide/src/core/api/transport/zulong-websocket.ts` | **新建**：WebSocket Transport |
| P1 | `zulong-ide/src/core/api/providers/zulong.ts` | **新建**：ZulongHandler (ApiHandler)，暂停/恢复模式适配 Cline 递归循环 |
| P1 | `zulong-ide/src/shared/api.ts` | 修改：添加 zulong provider + 模型 |
| P1 | `zulong-ide/src/shared/storage/state-keys.ts` | 修改：添加 zulongServerUrl |
| P1 | `zulong-ide/src/core/api/index.ts` | 修改：case "zulong" |
| P1 | `zulong-ide/src/shared/providers/providers.json` | 修改：列表顶部 |
| P1 | `zulong-ide/webview-ui/src/components/settings/providers/ZulongProvider.tsx` | **新建** |
| P1 | `zulong-ide/webview-ui/src/components/settings/ApiOptions.tsx` | 修改 |
| P1 | `zulong-ide/webview-ui/src/utils/validate.ts` | 修改 |
| P1 | `zulong-ide/webview-ui/src/components/settings/utils/providerUtils.ts` | 修改 |
| P2 | `zulong-ide/src/core/task/index.ts` | **不额外修改**（P1 的 2 行 FC 条件已足够，`recursivelyMakeClineRequests()` 递归主循环完整保留） |
| P2.5 | `zulong/cline/cline_ide_server.py` | 修改：`_build_initial_messages` 接收 Cline system_prompt |
| P2.5 | `zulong/cline/cline_fc_runner.py` | 修改：`_exec_internal` 尾部推送 display_text 内部工具摘要（在 run_loop_async 中） |
| P3 | `zulong-ide/package.json` | 修改：品牌重塑 |
| P3 | `zulong-ide/src/registry.ts` | 修改：命令前缀 |
| P3 | `zulong-ide/src/core/storage/disk.ts` | 修改：存储路径 |
| P3 | `zulong-ide/src/core/prompts/system-prompt/components/agent_role.ts` | 修改 |
| P3 | `zulong-ide/src/core/prompts/system-prompt/variants/*/overrides.ts` (5文件) | 修改 |
| P3 | `zulong-ide/webview-ui/src/components/welcome/WelcomeView.tsx` | 修改 |
| P4 | `zulong-ide/webview-ui/src/components/chat/ZulongStatusPanel.tsx` | **新建** |
| P5 | `zulong/cline/cline_api_server.py` | 修改：修复 _stream_response |

---

## 实施顺序

```
┌─ P0: 后端基础（必须先完成）────────────────────────────────┐
│                                                              │
│  0.1 DialogueAdapter 集成（P0 级缺陷修复）                   │
│  0.2 ExperienceGenerator 恢复                                │
│  0.3 run_loop_async() 实现                                   │
│  0.4 WebSocket 协议扩展                                      │
└──────────────────────────────────────────────────────────────┘
         ↓
┌─ MVP (端到端可运行) ────────────────────────────────────────┐
│                                                              │
│  Phase 1 (WS Transport + ZulongHandler)                      │
│  → Phase 2 (零修改 Cline 主循环)                             │
│  → Phase 2.5 (协调机制: system prompt 分工 + 内部工具可见)   │
│                                                              │
│  完成后：用户发消息→祖龙推理→工具请求→Cline完整管道执行      │
│  （含Diff视图、终端、审批、Checkpoint等全部能力）             │
│  →结果回传→后端继续→记忆/任务/对话全部自动记录               │
│  →内部工具活动实时推送到聊天面板                              │
└──────────────────────────────────────────────────────────────┘
         ↓
┌─ 完善层 (可并行) ──────────────────┐
│  Phase 3: 品牌重塑                  │
│  Phase 4: 祖龙状态面板              │
│  Phase 5: HTTP API 修复（降级路径） │
└─────────────────────────────────────┘
         ↓
┌─ 发布层 ────────────────────────────┐
│  Phase 6: 构建与全面验证             │
└─────────────────────────────────────┘
```

**MVP 目标**：Phase 0 + 1 + 2.5 完成后（Phase 2 不需要修改 Cline 代码），即可实现端到端的完整工作流——用户在 VS Code 中发消息，祖龙后端作为大脑推理，Cline 100% 原有编程能力执行工具，结果回传后端继续，所有操作自动创建对话节点和任务图谱。Cline 侧 `recursivelyMakeClineRequests()`、ContextManager、Plan/Act 模式、错误处理等子系统全部零修改。

---

## 附录：复杂任务协作流程模拟

**场景**：用户在 VS Code 中输入"给 auth 模块添加 JWT 刷新令牌功能，需要修改 token_service.py 和相关的 API 路由"

### 第 1 步：Cline 任务入口

```
用户输入 → recursivelyMakeClineRequests(userContent)
         → getSystemPrompt(promptContext)
           默认 variant 生成 systemPrompt（含 XML 工具文档 + 环境上下文）
           tools = undefined（无 Zulong Variant 提供 native tools）
           但 useNativeToolCalls = true（task/index.ts 直接覆盖）
         → ContextManager 构建 truncatedConversationHistory
         → createMessage(systemPrompt, truncatedHistory, tools=undefined)
           → 进入 ZulongHandler
```

### 第 2 步：ZulongHandler 发送 session_start

```
ZulongHandler.createMessage():
  检测 messages 最后是 user 消息 → 新任务
  ──WebSocket──→
  {"type": "session_start", "payload": {
    "task": "给 auth 模块添加 JWT 刷新令牌功能...",
    "cwd": "d:/projects/myapp",
    "system_prompt": "<默认 variant 生成的 systemPrompt，含 XML 工具文档 + 环境上下文>"
  }}
```

### 第 3 步：后端初始化

```
cline_ide_server._run_fc_loop() 接收:
  ├─ _build_initial_messages(cline_system_prompt=...)
  │   ClinePromptHandler.process_system_prompt():
  │     ① STRIP: Cline 默认 variant 的 XML 工具文档 (extract_cline_tool_block)
  │     ② KEEP: Cline 的环境上下文 (workspace roots, .clinerules)
  │     ③ INJECT: retrieve_zulong_context("给 auth 模块添加 JWT...")
  │        → memory_context: "项目使用 PyJWT 2.8，auth 模块在 src/auth/"
  │        → task_context: "" (新任务，无已有图谱)
  │        → experience_hints: "JWT 刷新令牌建议使用 rotate 策略"
  │     ④ INJECT: 祖龙任务管理规则 (COMPLEX 模板)
  │
  ├─ 创建 ClineFCRunner(engine, session, tool_registry)
  ├─ DialogueAdapter.add_round(user_input="给 auth 模块添加...")
  │   → MemoryGraph 新增 DIALOGUE 节点
  ├─ 初始化 AttentionWindow(GLOBAL 模式)、CircuitBreaker、DriftDetector
  └─ run_loop_async() 开始 FC 循环
```

### 第 4 步：FC Turn 1 — LLM 规划任务

```
_call_model():
  messages: [system(含祖龙增强), user("给 auth 模块添加...")]
  tools: ClineToolRegistry.get_combined_tool_definitions_for_intent("complex")
    → 内部: task_create_plan, task_add_node, task_mark_status,
             task_view_overview, recall_memory, navigate_attention
    → 远程: read_file, write_to_file, execute_command, list_files,
             search_files, replace_in_file, ...

LLM 返回 tool_calls (OpenAI FC 格式):
  ① task_create_plan(title="添加JWT刷新令牌", desc="...")
  ② task_add_node(parent_id="req", label="分析现有 token_service.py")
  ③ task_add_node(parent_id="req", label="实现 refresh_token 生成逻辑")
  ④ task_add_node(parent_id="req", label="添加 /auth/refresh API 路由")
  ⑤ task_add_node(parent_id="req", label="编写单元测试")
```

### 第 5 步：内部工具静默执行 + 推送可见

```
_exec_tools() 分类: 5 个 tool_calls 全部是内部工具
  ├─ _exec_internal(task_create_plan) → TaskGraph 创建
  │   send_callback("display_text", "[祖龙] 创建任务规划: 添加JWT刷新令牌")
  │   DialogueAdapter.add_sub_dialogue(tool_name="task_create_plan")
  │   CircuitBreaker.record_call("task_create_plan", ...)
  │
  ├─ _exec_internal(task_add_node × 4) → 4 个子任务节点
  │   send_callback("display_text", "[祖龙] 添加子任务: 分析现有 token_service.py")
  │   send_callback("display_text", "[祖龙] 添加子任务: 实现 refresh_token 生成逻辑")
  │   send_callback("display_text", "[祖龙] 添加子任务: 添加 /auth/refresh API 路由")
  │   send_callback("display_text", "[祖龙] 添加子任务: 编写单元测试")
  │
  └─ 无远程工具 → 不暂停，继续下一轮

  插件侧 ZulongHandler: 收到 5 个 display_text → yield {type:"text"} × 5
  Cline 聊天面板实时显示祖龙的任务规划过程
```

### 第 6 步：FC Turn 2 — 读取现有代码

```
LLM 返回:
  ① task_mark_status(node_id="n1", status="in_progress")     [内部]
  ② read_file(path="src/auth/token_service.py")               [远程]

_exec_tools():
  ├─ 内部先执行: task_mark_status → send_callback("display_text", "[祖龙] 开始: 分析现有 token_service.py")
  ├─ CircuitBreaker.evaluate() → GREEN（正常）
  └─ 远程: read_file → _pause_for_cline()
       send_callback("tool_request", {
         call_id: "call_001",
         tool_name: "read_file",
         arguments: {"path": "src/auth/token_service.py"}
       })
       FC 循环暂停，等待工具结果

  ← 插件侧 ZulongHandler:
     yield {type: "text", text: "[祖龙] 开始: 分析现有 token_service.py"}
     yield {type: "tool_calls", tool_call: {
       call_id: "call_001",
       function: {name: "read_file", arguments: '{"path":"src/auth/token_service.py"}'}
     }}
     return  ← stream 结束
```

### 第 7 步：Cline 完整管道执行工具

```
Cline recursivelyMakeClineRequests 主循环:
  stream loop case "tool_calls":
    ToolUseHandler.processToolUseDelta() → 积累 FC 参数
    processNativeToolCalls() → 设置 assistantMessageContent
    (isNativeToolCall=true)
  stream 结束 → processNativeToolCalls(partial=false) → 最终化
  flushAssistantPresentationOrThrow() → ToolExecutor:
    ├─ 读取 src/auth/token_service.py
    ├─ 聊天面板显示文件内容
    └─ 收集工具结果: "class TokenService:\n    def create_access_token(...):\n..."

  工具结果 → userMessageContent（格式: role="tool", tool_call_id=...）
  recursivelyMakeClineRequests(userMessageContent)  ← 递归
  → createMessage(systemPrompt, messages, tools) ← 第 2 次调用
```

### 第 8 步：ZulongHandler 回传工具结果

```
ZulongHandler.createMessage() 第 2 次调用:
  检测 messages 最后是 tool result → 工具结果回传
  提取工具结果
  ──WebSocket──→
  {"type": "tool_result", "payload": {
    "call_id": "call_001",
    "tool_name": "read_file",
    "result": "class TokenService:\n    def create_access_token(self, user_id)...",
    "is_error": false
  }}
```

### 第 9 步：后端恢复 FC 循环

```
run_or_resume(tool_results=[...]):
  _inject_tool_results(state, tool_results)
    → state.messages 追加 tool result
    → AttentionWindow.register_message(tool_result)
  _run_bfs_activation() → 记忆图谱激活相关节点
  state.phase = "running" → 继续 FC 循环
```

### 第 10 步：FC Turn 3 — 写入新代码

```
LLM 分析 token_service.py 后返回:
  ① write_to_file(
       path="src/auth/token_service.py",
       content="class TokenService:\n    ...\n    def create_refresh_token(self, user_id):\n        ..."
     )                                                        [远程]

_exec_tools(): 远程 → _pause_for_cline()
  send_callback("tool_request", {
    call_id: "call_002",
    tool_name: "write_to_file",
    arguments: {path: "src/auth/token_service.py", content: "..."}
  })

  ← 插件侧 yield {type: "tool_calls", ...} → return
```

### 第 11 步：Cline DiffView + 用户审批

```
Cline ToolExecutor 执行 write_to_file:
  ├─ ★ DiffViewProvider 打开 VS Code 原生 Diff 编辑器
  │   左侧: 原文件（旧版 token_service.py）
  │   右侧: 新文件（含 create_refresh_token 方法）
  │
  ├─ ★ ask/say 机制弹出审批对话框:
  │   "Zulong wants to edit src/auth/token_service.py"
  │   [Approve] [Reject]
  │
  ├─ 用户点击 [Approve]（或 auto-approve 配置跳过）
  │
  ├─ ★ Checkpoint 系统自动保存检查点
  │   （用户可随时恢复到此状态）
  │
  └─ 文件写入完成，收集结果 → tool_result 回传后端
```

### 第 12 步：FC Turn 4-6 — 继续执行子任务

```
Turn 4: LLM 调用 read_file("src/routes/auth_routes.py") → Cline 读取
Turn 5: LLM 调用 write_to_file("src/routes/auth_routes.py", "...新增 /refresh 路由...")
        → Cline DiffView 显示新增的路由代码，用户审批
Turn 6: LLM 调用 task_mark_status(n2, "completed") [内部]
        + task_mark_status(n3, "in_progress") [内部]
        + execute_command("pytest tests/test_auth.py -v") [远程]
        → Cline 终端实时显示测试输出 ★

  ★ 终端集成:
    VscodeTerminalProcess + Shell Integration
    实时流式显示 pytest 输出:
      tests/test_auth.py::test_create_token PASSED
      tests/test_auth.py::test_refresh_token PASSED
      ======================== 2 passed in 0.3s ========================
```

### 第 13 步：安全网触发 — RuleGuardian 拦截过早完成

```
FC Turn 7: LLM 尝试返回纯文本（无 tool_calls）:
  "JWT 刷新令牌功能已添加完成，包括..."

_eval_response():
  ├─ 安全网 0: SemanticDriftDetector → 无漂移（sim=0.87）
  ├─ ★ 安全网 1: RuleGuardian.check_premature_completion()
  │   检查 TaskGraph: 4 个子任务中只有 2 个 completed
  │   子任务 "编写单元测试" 仍为 pending
  │   → blocked=True, reason="2/4 子任务未完成"
  │
  ├─ 注入纠偏消息到 state.messages:
  │   "你还有 2 个子任务未完成：
  │    - n3: 添加 /auth/refresh API 路由 (in_progress)
  │    - n4: 编写单元测试 (pending)
  │    请继续执行，不要提前结束。"
  │
  └─ 返回 "continue" → FC 循环继续

  （用户在插件侧不感知此拦截——ZulongHandler stream 保持打开，继续 yield display_text）
```

### 第 14 步：FC Turn 8-9 — 完成剩余子任务

```
Turn 8: LLM 调用 write_to_file("tests/test_refresh_token.py", "...") [远程]
        → Cline DiffView 显示新测试文件
        → 用户审批 → 文件创建

Turn 9: LLM 调用 execute_command("pytest tests/ -v") [远程]
        → Cline 终端显示全部测试通过
        → task_mark_status(n4, "completed", result="4 个测试全部通过") [内部]
```

### 第 15 步：任务完成 + 记忆沉淀

```
FC Turn 10: LLM 返回纯文本（无 tool_calls）:
  "JWT 刷新令牌功能已完成。新增了 create_refresh_token()..."

_eval_response():
  ├─ RuleGuardian: 4/4 子任务 completed → 通过
  ├─ InfoGap: 无缺口 → 通过
  └─ 返回 "done"

_finalize():
  ├─ _auto_complete_task() → TaskGraph 所有节点标记完成
  ├─ DialogueAdapter.finalize_round(status="completed", total_turns=10)
  │   → MemoryGraph DIALOGUE 节点完整记录本次对话
  ├─ ExperienceGenerator.process_dialogue_batch()
  │   → 提取经验: "JWT 刷新令牌实现: rotate 策略, 使用 PyJWT..."
  │   → MemoryGraph 新增 KNOWLEDGE 节点（含重要度评分、domain 分类）
  ├─ TaskGraphAdapter.sync() → TaskGraph 全量同步到 MemoryGraph
  └─ send_callback("task_complete", {
       result: "JWT 刷新令牌功能已完成。新增了 create_refresh_token()..."
     })

  ← 插件侧 ZulongHandler:
     yield {type: "text", text: "JWT 刷新令牌功能已完成..."}
     return  ← stream 结束
```

### 第 16 步：Cline 正常结束递归

```
recursivelyMakeClineRequests():
  stream 结束，无 tool_calls → didToolUse = false
  但这是 task_complete 的正常结束（不是错误）
  recursion 自然终止

聊天面板完整显示:
  ┌─────────────────────────────────────┐
  │ You: 给 auth 模块添加 JWT 刷新令牌  │
  │                                      │
  │ [祖龙] 创建任务规划: 添加JWT刷新令牌 │
  │ [祖龙] 添加子任务: 分析现有...       │
  │ [祖龙] 添加子任务: 实现 refresh...   │
  │ [祖龙] 添加子任务: 添加路由...       │
  │ [祖龙] 添加子任务: 编写测试...       │
  │ [祖龙] 开始: 分析现有 token_service  │
  │                                      │
  │ 📄 Read src/auth/token_service.py    │
  │                                      │
  │ ✏️ Edit src/auth/token_service.py    │
  │    [View Diff] [Checkpoint saved]    │
  │                                      │
  │ 📄 Read src/routes/auth_routes.py    │
  │ ✏️ Edit src/routes/auth_routes.py    │
  │                                      │
  │ ▶ pytest tests/test_auth.py -v       │
  │   2 passed in 0.3s                   │
  │                                      │
  │ ✏️ New tests/test_refresh_token.py   │
  │ ▶ pytest tests/ -v                   │
  │   4 passed in 0.5s                   │
  │                                      │
  │ JWT 刷新令牌功能已完成。             │
  │ 新增了 create_refresh_token() 方法...│
  └─────────────────────────────────────┘
```

### 全流程数据流总结

```
                    ┌── Cline 管道（保留 100%）──┐
                    │ DiffView ✅                 │
                    │ Terminal ✅                  │
                    │ ask/say ✅                   │
                    │ Checkpoint ✅                │
                    │ ContextManager ✅            │
                    │ processNativeToolCalls ✅    │
                    └──────────┬──────────────────┘
                               │
  用户 ──→ createMessage() ────┼─── session_start ───→ 后端 FC 循环
                               │                          │
  Cline ←── yield text ────────┼─── display_text ────← 内部工具执行
  Cline ←── yield tool_calls ──┼─── tool_request ────← 远程工具暂停
                               │                          │
  Cline 执行工具 ──────────────┼─── tool_result ─────→ 恢复 FC 循环
  (DiffView/Terminal/审批)     │                          │
                               │                          ↓
                               │              MemoryGraph: DIALOGUE 节点
                               │              TaskGraph: 4 个子任务节点
                               │              Experience: JWT 实现经验
                               │              AttentionWindow: 10 轮消息
                               │
  Cline ←── yield text ────────┼─── task_complete ───← FC 循环结束
  递归终止                     │
                    ┌─────────────────────────────────────┐
                    │ 全程 OpenAI FC 格式                  │
                    │ 无 XML 工具标签                      │
                    │ task/index.ts 仅 2 行条件判断        │
                    │ 不依赖 Variant/白名单/matcher        │
                    │ 后端按需加载工具                      │
                    │ 祖龙记忆/任务/安全全部激活            │
                    └─────────────────────────────────────┘
```
