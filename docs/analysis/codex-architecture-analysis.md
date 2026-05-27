# OpenAI Codex 架构深度分析报告

> 分析仓库: `github.com/openai/codex` (克隆至 `d:/AI/project/openai_codex`)
> 
> 分析日期: 2025-07-08
>
> 代码规模: Rust 主力 (~90 crates), Node.js 入口层 + Python/TypeScript SDK

---

## 目录

1. [整体架构概览](#一整体架构概览)
2. [任务编排 (Task Orchestration)](#二任务编排详解)
3. [IDE 集成逻辑](#三ide-集成逻辑详解)
4. [对祖龙项目的启示](#四对祖龙项目的启示)
5. [附录: 关键文件索引](#五附录-关键文件索引)

---

## 一、整体架构概览

### 1.1 技术栈

| 语言 | 构建系统 | 位置 | 用途 |
|------|---------|------|------|
| **Rust** (主力) | Cargo + Bazel | `codex-rs/` | 90+ crates: CLI, TUI, app-server, core, sandboxing, MCP, auth |
| **JavaScript** | pnpm | `codex-cli/` | npm 入口 — 平台检测 → 启动 Rust 原生二进制 |
| **TypeScript** | pnpm + tsup | `sdk/typescript/` | `@openai/codex-sdk` |
| **Python** | hatchling + uv | `sdk/python/` | `openai-codex` + pydantic app-server v2 protocol |

### 1.2 全局架构图

```
┌─────────────────────────────────────────────────────────────────┐
│  用户终端 / VS Code / IDE                                        │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│  codex-cli/ (Node.js npm 包)                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  bin/codex.js: 平台检测(linux/darwin/win32) → spawn Rust │   │
│  │  映射 target triple → 解析 npm @openai/codex-{platform}  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────┬───────────────────────────────────────────────┘
                  │ spawns native binary
┌─────────────────▼───────────────────────────────────────────────┐
│  codex-rs/cli/ (Rust CLI binary — Clap 解析)                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  约 25 个子命令:                                          │   │
│  │  ├─ (无参数) → run_interactive_tui()                     │   │
│  │  ├─ exec → 非交互式单次执行                                │   │
│  │  ├─ review → 代码审查                                     │   │
│  │  ├─ login/logout → 认证流                                  │   │
│  │  ├─ resume/fork → 会话管理                                 │   │
│  │  ├─ mcp/plugin → 扩展管理                                  │   │
│  │  ├─ app-server → 启动 app server (embedded/daemon)         │   │
│  │  ├─ sandbox → 平台沙箱                                     │   │
│  │  ├─ apply → git apply                                     │   │
│  │  ├─ doctor → 诊断                                          │   │
│  │  └─ update/completion → 更新/补全                           │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     ┌──────────┐ ┌──────────┐ ┌──────────────┐
     │ Embedded │ │  Local   │ │   Remote     │
     │ AppServer│ │  Daemon  │ │  AppServer   │
     │ (同进程) │ │(UDS/WS)  │ │  (WS/WSS)    │
     └────┬─────┘ └────┬─────┘ └──────┬───────┘
          └────────────┼──────────────┘
                       │ JSON-RPC v2 over WebSocket / Unix Domain Socket
┌──────────────────────▼──────────────────────────────────────────┐
│  app-server/ — 核心后端                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Thread CRUD, Bootstrap, Tool Execution, MCP Management  │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     ┌──────────┐ ┌──────────┐ ┌──────────────┐
     │  core/   │ │ codex-   │ │ model-       │
     │  agent   │ │ api/     │ │ provider/    │
     │  skills  │ │ HTTP SSE │ │ 多模型适配    │
     │  hooks   │ │ streaming│ │ OpenAI/OSS/  │
     │  sandbox │ │          │ │ Ollama/LM    │
     └──────────┘ └──────────┘ └──────────────┘
```

### 1.3 核心架构决策

| 决策 | 实现 |
|------|------|
| **Client-Server 分离** | TUI 为客户端, app-server 为后端, JSON-RPC v2 over WS/UDS |
| **Thread 会话模型** | 所有对话组织为 Thread, SQLite + 文件系统持久化, 支持 resume/fork/archive |
| **多平台沙箱** | Linux(Landlock+bwrap), macOS(Seatbelt), Windows(Restricted Tokens) |
| **MCP 双向集成** | 既可作为 MCP Client 消费外部工具, 也可作为 MCP Server 暴露自身能力 |
| **分层配置** | 内置默认 → 系统配置 → 托管配置(MDM) → 用户 config.toml → CLI 覆盖 |
| **多 Agent 协作** | `collaboration-modes` 支持多 agent 工作流, spawn/wait/close 子 agent |

### 1.4 启动链路

```
User: codex [args]
  │
  ▼
[1] codex-cli/bin/codex.js (Node.js ESM)
  ├── 检测 platform + arch → target triple
  ├── 定位 npm @openai/codex-{platform} 原生二进制
  ├── 设置 PATH 环境变量
  ├── spawn Rust 原生二进制为子进程
  └── 转发信号 (SIGINT, SIGTERM, SIGHUP)
      │
      ▼
[2] codex-rs/cli/src/main.rs (Rust Binary)
  ├── arg0_dispatch_or_else() — 多调用二进制支持
  ├── MultitoolCli::parse() via clap
  └── cli_main() → 按子命令分发
      │
      ▼
[3] codex-rs/tui/src/lib.rs → run_main()
  ├── 加载 config.toml (分层合并)
  ├── 解析 app-server 目标 (embedded/local-daemon/remote)
  ├── 初始化 tracing, telemetry, state DB
  ├── 显示 onboarding (login, trust, OSS 选择)
  ├── 处理 resume/fork picker
  └── 启动 App::run() — TUI 主事件循环
```

---

## 二、任务编排详解

### 2.1 四层架构总览

Codex 的任务编排不是单一系统，而是**四层协作架构**：

```
                    ┌─────────────────────────────────────┐
                    │          GOAL (Thread-Level)          │
                    │  持久化目标 + 预算追踪 + 自动延续      │
                    │  6 状态机: Active/Paused/Blocked/     │
                    │  BudgetLimited/UsageLimited/Complete  │
                    └──────────────┬────────────────────────┘
                                   │ 跨回合引导
                    ┌──────────────▼────────────────────────┐
                    │     MULTI-AGENT TREE (Session-Level)   │
                    │  spawn_agent → explorer/worker 角色     │
                    │  send_message / followup_task          │
                    │  AgentPath 层级路由                     │
                    │  Fork 模式: fresh / all / N-turns      │
                    └──────────────┬────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
     ┌────────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
     │   RegularTask    │  │   ReviewTask   │  │   CompactTask  │
     │  (FC 循环)       │  │ (codex_delegate)│  │ (上下文压缩)    │
     │  一次任务         │  │ 子 agent 完成   │  │                │
     │  N 次模型请求     │  │ 后通知父 agent  │  │                │
     └─────────────────┘  └────────────────┘  └────────────────┘
              │
     ┌────────▼────────┐
     │ Agent Jobs       │
     │ (CSV 批量处理)    │
     │ N workers 认领    │
     │ 并报告结果        │
     └─────────────────┘
```

| 层级 | 子系统 | 核心文件 | 作用 | 关键设计 |
|------|--------|----------|------|----------|
| **L1 长期方向** | Goal 系统 | `ext/goal/`, `core/src/goals.rs` | 持久化任务目标 + 预算追踪 | 不拆分任务, 只做引导 |
| **L2 并行委托** | Multi-Agent | `core/src/agent/`, `agent-graph-store/` | 树形 Agent 委托/并行 | 模型驱动 spawn, 无 DAG |
| **L3 回合执行** | SessionTask | `core/src/tasks/` | 回合级执行控制 | 回合循环 + 优雅中断 |
| **L4 批量作业** | Agent Jobs | `core/src/tools/handlers/agent_jobs/` | CSV 驱动的批量处理 | 数据并行模式 |

---

### 2.2 第一层: Goal 系统 — 持久化目标

#### 数据模型

```rust
// codex-rs/state/src/model/thread_goal.rs
pub struct ThreadGoal {
    pub thread_id: ThreadId,
    pub goal_id: String,         // UUID, 乐观并发版本控制
    pub objective: String,       // 用户提供的任务描述
    pub status: ThreadGoalStatus,
    pub token_budget: Option<i64>,
    pub tokens_used: i64,
    pub time_used_seconds: i64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
```

#### 六状态机

```
Active ──────────────────────► Paused (外部手动暂停)
  │                              │
  ├──► Blocked (连续 >=3 轮 blocked)
  ├──► BudgetLimited (tokens_used >= token_budget)
  ├──► UsageLimited (系统强制限制)
  └──► Complete (模型调用 update_goal)
```

- **可恢复态**: `Paused`, `Blocked`, `UsageLimited` — 可回到 `Active`
- **终态**: `BudgetLimited`, `Complete` — 不可逆转

#### 自动延续

当 Goal 处于 Active 状态且回合结束无其他待处理工作时, 系统**自动发起新回合**, 注入隐藏的 `<goal_context>` 延续提示, 包含:
- 目标文本 (html 转义)
- 预算追踪 (已用/总额/剩余)
- 完成审计指令: 必须逐项验证每个需求在当前状态证据下是否已满足
- Blocked 审计: 仅在连续 >=3 轮遇到相同阻塞条件时标记

#### 预算耗尽处理

当 budget 耗尽时, 注入 `budget_limit.md` 引导提示: "不要开始新的实质性工作…尽快完成本回合。"

#### 模型可见工具 (3个)

| 工具 | 参数 | 说明 |
|------|------|------|
| `create_goal` | `{ objective, token_budget }` | 模型可请求创建, 已存在则失败 |
| `get_goal` | 无 | 返回完整状态含剩余 token |
| `update_goal` | `{ status: "complete" \| "blocked" }` | **模型只能设这两种状态** |

> **关键约束**: 模型不能设置 `paused`/`active`/`budget_limited`/`usage_limited` — 这些仅由系统/用户控制。

#### Goal 运行时事件

```rust
// codex-rs/core/src/goals.rs
pub(crate) enum GoalRuntimeEvent<'a> {
    TurnStarted { turn_context, token_usage },
    ToolCompleted { turn_context, tool_name },
    TurnFinished { turn_context, turn_completed },
    MaybeContinueIfIdle,                    // 空闲时自动触发延续
    TaskAborted { turn_context },
    UsageLimitReached { turn_context },
    ExternalSet { external_set },            // 外部设置 Goal
    ExternalClear,                           // 外部清除 Goal
    ThreadResumed,                           // 线程恢复
}
```

`GoalRuntimeState` 管理空闲 Goal 延续: 回合结束且无其他待处理工作 → 自动发起延续回合。

#### SQLite 持久化

```rust
// codex-rs/state/src/runtime/goals.rs
// 单表 thread_goals, UNIQUE(thread_id) — 每个线程最多一个 Goal
// 关键操作:
//   replace_thread_goal() — UPSERT (清空用量计数器)
//   insert_thread_goal() — INSERT ON CONFLICT DO NOTHING
//   update_thread_goal() — 乐观并发: expected_goal_id 防止过期更新
//   account_thread_goal_usage() — 原子递增 + 自动 BudgetLimited 转换
```

#### Goal 会计状态机

```rust
// codex-rs/ext/goal/src/accounting.rs
pub enum GoalAccountingMode {
    ActiveStatusOnly,     // "status = 'active'"
    ActiveOnly,           // "status IN ('active', 'budget_limited')"
    ActiveOrComplete,     // "status IN ('active', 'budget_limited', 'complete')"
    ActiveOrStopped,      // "status IN ('active', 'paused', 'blocked', ...)"
}
```

双会计: Token 会计(每回合 delta) + 墙上时钟会计(`Instant::now()` 基线)。

#### 设计哲学

| 是否 | 说明 |
|------|------|
| **不拆分任务** | Goal 不将目标分解为子任务, 只提供持久声明 + 预算追踪 |
| **模型主导** | 模型决定如何执行, Goal 提供跨回合的持久引导 |
| **系统监控** | 系统自动追踪用量, 预算耗尽时注入限制提示 |
| **乐观并发** | `expected_goal_id` 防止并发写入冲突 |

---

### 2.3 第二层: Multi-Agent 系统 — 树形并行委托

#### Agent 控制面

```rust
// codex-rs/core/src/agent/control.rs
pub(crate) struct AgentControl {
    session_id: SessionId,              // 同一根树的所有 agent 共享
    manager: Weak<ThreadManagerState>,   // 弱引用, 避免循环引用
    state: Arc<AgentRegistry>,           // 内存中 agent 树注册表
}
```

#### Agent 注册表 — 并发限制

```rust
// codex-rs/core/src/agent/registry.rs
pub(crate) struct AgentRegistry {
    active_agents: Mutex<ActiveAgents>,  // HashMap<String, AgentMetadata>
    total_count: AtomicUsize,            // 强制 agent_max_threads 限制
}
```

- `try_increment_spawned()`: 使用 `compare_exchange_weak` 做**无锁有界并发控制**
- `SpawnReservation`: **RAII** 模式 — Drop 自动释放保留槽位

#### Agent 别名

100 个著名科学家/哲学家名字 (`agent_names.txt`), 随机分配去重: "Euclid", "Newton", "Turing"…用完后加 "the 2nd" / "the 3rd" 后缀复位。

#### Agent 拓扑持久化

```rust
// codex-rs/agent-graph-store/src/store.rs
#[async_trait]
pub trait AgentGraphStore: Send + Sync {
    async fn upsert_thread_spawn_edge(parent, child, status);
    async fn list_thread_spawn_descendants(root, status_filter);
    // BFS 按深度排列, 然后是 thread_id
}
```

```rust
// codex-rs/agent-graph-store/src/types.rs
pub enum ThreadSpawnEdgeStatus {
    Open,   // 子 agent 活跃或可恢复
    Closed, // 子 agent 已关闭 — 其子树被跳过
}
```

> 这对恢复时重建 agent 树至关重要。

#### 内置 Agent 角色

| 角色 | 用途 | 配置文件 | 说明 |
|------|------|----------|------|
| `default` | 标准 agent | 无 | 继承调用者配置 |
| `explorer` | 代码库探查 | `explorer.toml` | 快速/权威/可并行 |
| `worker` | 执行具体工作 | 无 | 实现功能/修 bug/重构 |

角色作为**配置层叠加**, 保留调用者的 model/profile, 仅当角色显式覆盖时才修改。

#### Agent 间通信

```rust
// codex-rs/protocol/src/protocol.rs
pub struct InterAgentCommunication {
    pub author: AgentPath,
    pub recipient: AgentPath,
    pub other_recipients: Vec<AgentPath>,  // CC/广播
    pub content: String,
    pub trigger_turn: bool,  // true = 立即唤醒接收者发起回合
}
```

| 模式 | `trigger_turn` | 行为 |
|------|----------------|------|
| `QueueOnly` | `false` | 邮箱投递, 接收者自行取出 |
| `TriggerTurn` | `true` | 立即启动接收者回合 |

#### V2 并行工具

| 工具 | 用途 | 关键参数 |
|------|------|----------|
| `spawn_agent` | 创建子 agent | `message, task_name, agent_type?, fork_turns(none/all/N)?` |
| `send_message` | QueueOnly 投递 | `recipient, content` |
| `followup_task` | TriggerTurn 投递 | `recipient, content` (不可发 root) |

#### Fork 模式

| 模式 | 行为 |
|------|------|
| `"none"` | 全新 agent — 无历史 |
| `"all"` | 完整历史 fork |
| `"N"` | 仅最后 N 个回合 |

#### Orchestrator 提示模板

```
// codex-rs/core/templates/agents/orchestrator.md 核心指令:
"Prefer multiple sub-agents to parallelize your work"
"If sub-agents are running, wait for them before yielding"
"Process plan steps in parallel — one agent per step when possible"
"When you ask a sub-agent to do work, your ONLY role becomes coordination"
```

#### 完成监听器

每个 spawn 附带一个 detached background task (`maybe_start_completion_watcher`), 订阅子 agent 状态。子 agent 达到最终状态时, **自动发送 `InterAgentCommunication`** 回父 agent。V2 路径中作为 **developer-role response item** 出现在父 agent 会话中。

#### 关键设计原则

| 原则 | 实现 |
|------|------|
| **无显式 DAG** | 依赖关系通过模型推理隐式处理, 不构建任务图 |
| **模型驱动并行** | 模型决定何时并行 spawn, 系统不自动拆分 |
| **树形拓扑** | AgentPath 层级路由, parent/child 边持久化到 SQLite |
| **独立线程** | 每个 spawn 创建独立 Thread, 真正并发执行 |
| **协调通信** | InterAgentCommunication + Completion Watcher 模式 |

---

### 2.4 第三层: SessionTask 系统 — 回合级执行

#### Task 类型

```rust
// codex-rs/core/src/state/turn.rs
pub(crate) enum TaskKind {
    Regular,  // 正常对话回合
    Review,   // /review 命令
    Compact,  // 上下文压缩
}
```

#### SessionTask Trait

```rust
// codex-rs/core/src/tasks/mod.rs
pub(crate) trait SessionTask: Send + Sync + 'static {
    fn kind(&self) -> TaskKind;
    fn span_name(&self) -> &'static str;
    fn records_turn_token_usage_on_span(&self) -> bool { false }
    fn run(self: Arc<Self>, session, ctx, input, cancellation_token)
        -> impl Future<Output = Option<String>>;
    fn abort(&self, session, ctx) -> impl Future<Output = ()> { /* default no-op */ }
}
```

#### RunningTask 结构

```rust
pub(crate) struct RunningTask {
    pub(crate) done: Arc<Notify>,               // 完成信号
    pub(crate) kind: TaskKind,
    pub(crate) task: Arc<dyn AnySessionTask>,
    pub(crate) cancellation_token: CancellationToken,
    pub(crate) handle: AbortOnDropHandle<()>,    // Drop 自动 abort!
    pub(crate) turn_context: Arc<TurnContext>,
    pub(crate) turn_extension_data: Arc<ExtensionData>,
    pub(crate) _timer: Option<Timer>,            // 持续时间追踪
}
```

#### ActiveTurn — 多任务容器

```rust
pub(crate) struct ActiveTurn {
    pub(crate) tasks: IndexMap<String, RunningTask>,  // keyed by sub_id
    pub(crate) turn_state: Arc<Mutex<TurnState>>,
}
```

支持 **多个并发任务** (IndexMap), 允许 Session 内并行执行。

#### RegularTask 循环

```rust
// codex-rs/core/src/tasks/regular.rs
loop {
    let last_agent_message = run_turn(sess, ctx, turn_ext, next_input, ...).await;
    if !sess.input_queue.has_pending_input(&sess.active_turn).await {
        return last_agent_message;  // 真正的完成
    }
    next_input = Vec::new();        // 有 pending input（tool results 等）继续
}
```

> **一次 RegularTask = N 次模型请求**: tool call 的 result 回来 → 自动继续循环, 直到 input queue 清空才真正结束。

#### Abort 协议

```
Abort 流程:
1. Cancel CancellationToken
2. 等待 GRACEFUL_INTERRUPTION_TIMEOUT_MS (100ms) 优雅完成
3. 超时 → AbortOnDropHandle 强制 abort
4. 插入 TurnAborted 历史标记 (含中断引导文本)
5. 发出 TurnAbortedEvent
6. 如果原因是 Interrupted → maybe_start_turn_for_pending_work() 处理队列
```

#### CodexDelegate — 子 Agent 路由

```rust
// codex-rs/core/src/codex_delegate.rs
// 两个入口:
//   run_codex_thread_interactive() — 全双工双向通道
//   run_codex_thread_one_shot() — 交互 + 即时输入 + 自动关闭
```

**审批过滤**: 子 agent 的审批事件被路由到父 Session:
- `ExecApprovalRequest` → 父审批系统 (可选 Guardian review)
- `ApplyPatchApprovalRequest` → 父审批系统
- `RequestPermissions` → 父权限管理
- `RequestUserInput` → 父 (MCP tool 审批可选自动 review)

**级联关闭**: 父 cancel → 子 agent 中断 → 排空 → 关闭。

---

### 2.5 第四层: Agent Jobs — CSV 批量处理

#### 数据模型

`spawn_agents_on_csv` 工具 (`core/src/tools/handlers/agent_jobs/spawn_agents_on_csv.rs`):
- 读取 CSV → 每行成为 job item
- 支持 `{column}` 模板变量
- N 个 worker agent 认领 items
- 通过 `report_agent_job_result` 返回结果
- 自动导出结果 CSV

#### 参数

| 参数 | 说明 |
|------|------|
| `max_concurrency` / `max_workers` | 并行限制 |
| `max_runtime_seconds` | 超时 |
| `output_schema` | 结构化结果 schema |
| 错误摘要 | 最 5 条失败 item 详情 |

#### Worker 报告

```rust
// codex-rs/core/src/tools/handlers/agent_jobs/report_agent_job_result.rs
// 返回 JSON 对象, 可选 stop: true 取消剩余工作
```

---

### 2.6 任务编排核心设计哲学总结

| 设计点 | Codex 的做法 | 对比典型实现 |
|--------|-------------|-------------|
| **任务依赖** | **没有显式 DAG** — 通过模型推理隐式处理 | 通常构建依赖图 |
| **并行策略** | **模型驱动** — 模型决定何时 spawn_agent | 系统自动拆分/调度 |
| **任务队列** | **没有队列** — 回合驱动, 非队列驱动 | 通常有任务队列 |
| **状态机** | 6 状态 + 乐观并发 (`expected_goal_id`) | 通常简单 3-4 状态 |
| **任务拆分** | **模型承担 orchestrator** — 系统只提供原语 | 预定义拆分逻辑 |
| **批量处理** | CSV → N workers — 数据并行 | 通常需要单独调度器 |

---

## 三、IDE 集成逻辑详解

### 3.1 IPC 通信桥

**核心文件**: `codex-rs/tui/src/ide_context/ipc.rs`

Codex 通过**本地 Socket/管道**与 IDE 扩展通信:

| 平台 | 通道类型 | 路径 |
|------|----------|------|
| **Linux/macOS** | Unix Domain Socket | `/tmp/codex-ipc/ipc-{uid}.sock` |
| **Windows** | Named Pipe | `\\.\pipe\codex-ipc` |

**Wire Protocol**: JSON-RPC 风格的消息帧
```
[4 bytes LE length prefix][JSON payload]
```

**消息类型**:
| 类型 | 方向 | 用途 |
|------|------|------|
| `request` | 双向 | 请求-响应模式 |
| `response` | 双向 | 对 request 的响应 |
| `broadcast` | IDE → Codex | 编辑器状态变更的主动推送 |
| `client-discovery-request` | Codex → IDE | 握手/发现 |
| `client-discovery-response` | IDE → Codex | 握手/发现响应 |

### 3.2 Session 来源追踪

```rust
// codex-rs/app-server-protocol/src/protocol/v2/thread.rs
pub enum ThreadSourceKind {
    Cli, VsCode, Exec, AppServer,
    SubAgent, SubAgentReview, SubAgentCompact,
    SubAgentThreadSpawn, SubAgentOther, Unknown,
}
```

每个 Thread 标记来源, 使 Codex 能区分来自 VS Code / 终端 / web app / sub-agent 的会话。

### 3.3 IDE 上下文注入

**核心文件**: `codex-rs/tui/src/ide_context/prompt.rs`

通过 `/ide` 斜杠命令或自动注入, 将 IDE 上下文追加到 LLM 提示中:

| 注入内容 | 说明 |
|----------|------|
| 活动文件路径 | 当前焦点文件 |
| 光标位置 | 行 + 列 |
| 选中文本 | 用户在编辑器中高亮的内容 |
| 已打开标签页列表 | 所有打开的文件 |
| 可见代码区域 | VS Code 当前视口内的代码 |

**流程**: TUI 调用 IDE 扩展的 `ide-context` 方法 → 获取编辑器状态 → 注入到 user message。

### 3.4 工具驱动编辑 (apply_patch)

Codex **不使用** `shell` 工具直接写文件 (如 `cat > file` 或 `echo`), 而是使用结构化的 **`apply_patch`** 工具。

#### Patch 格式

```
*** Begin Patch
*** Add File: src/new_component.ts
...完整文件内容...
*** End Patch

*** Begin Patch
*** Update File: src/existing.ts
@@ 15 @@
- old line
+ new line
*** End Patch

*** Begin Patch
*** Delete File: src/deprecated.ts
*** End Patch

*** Begin Patch
*** Move to: src/renamed.ts
...内容...
*** End Patch
```

#### 处理流水线

```
模型输出 patch 文本
     │
     ▼
[Parser] apply-patch/src/parser.rs
  → 解析为 Hunk 枚举: AddFile / DeleteFile / UpdateFile
     │
     ▼
[Verification] apply-patch/src/invocation.rs
  → maybe_parse_apply_patch_verified()
  → 校验 patch 与文件系统是否一致
     │
     ▼
[Application] apply-patch/src/lib.rs
  → apply_patch() → apply_hunks_to_files()
  → seek_sequence 模糊上下文匹配 + Unicode 规范化
  → 生成 unified diff (similar crate)
  → 记录 AppliedPatchDelta (精确度 + 回滚信息)
     │
     ▼
[Approval] core/src/tools/handlers/apply_patch.rs
  → 校验写权限
  → ApplyPatchRuntime (沙箱 + 用户确认)
  → 发出 FileChange 事件 (PatchChangeKind: Add/Delete/Update)
```

### 3.5 终端/Shell 交互

**双后端架构** (`core/src/tools/handlers/shell.rs`):

| 后端 | 特点 | 实现位置 |
|------|------|----------|
| **Classic shell** | 传统进程 spawn, stdout/stderr 捕获 | `core/src/tools/runtimes/shell.rs` |
| **Unified Exec** | PTY 终端模拟, stdin 写入, 颜色/光标 | `core/src/tools/runtimes/unified_exec.rs` |

**Shell 配置** (`tools/src/tool_config.rs`):
- `ShellCommandBackendConfig::Classic` vs `::ZshFork` (Linux zsh fork 模式)
- `UnifiedExecShellMode::Direct` vs `::ZshFork`
- Feature-gated: `Feature::ShellTool`, `Feature::ShellZshFork`, `Feature::UnifiedExec`

**命令安全** (`shell-command/src/command_safety/`):
- `is_dangerous_command()` — 检测 `rm -rf /` 等
- `is_safe_command()` — 安全命令白名单
- 平台区分: Windows (`windows_dangerous_commands.rs`, `powershell_parser.rs`) 和 Unix

**Shell 自检测** (`shell-command/src/shell_detect.rs`):
自动发现用户 shell: bash, zsh, PowerShell, cmd, sh。

### 3.6 分层审批系统

```rust
// codex-rs/protocol/src/approvals.rs
pub enum AskForApproval {
    Never,         // exec 非交互模式: 全部自动拒绝
    OnRequest,     // 交互模式: 弹出用户确认
    OnFailure,     // 失败时确认
    Granular,      // 按类别细粒度控制
}
```

#### 审批请求类型

| 请求类型 | 触发场景 |
|----------|----------|
| `CommandExecutionRequestApproval` | 执行 shell 命令 |
| `FileChangeRequestApproval` | 文件变更 |
| `ApplyPatchApproval` | patch 应用 |
| `PermissionsRequestApproval` | 权限变更 |
| `ToolRequestUserInput` | 需要用户输入 |
| `McpServerElicitationRequest` | MCP 服务器获取 |
| `DynamicToolCall` | 动态工具调用 |

#### Governance 流程

```
Agent 调用工具 (shell/apply_patch)
     │
     ▼
run_exec_like() → 权限检查 + exec policy 评估
     │
     ├── 不需要审批 → 直接执行
     │
     └── 需要审批 → ServerRequest 发给 Client
           │
           ├── exec 模式 → 自动拒绝 ("not supported in exec mode")
           │
           └── 交互模式 → 弹出用户确认 UI
```

### 3.7 完整工具集

```rust
// codex-rs/core/src/tools/handlers/mod.rs
```

| 工具 | 用途 | 命名空间 |
|------|------|----------|
| `apply_patch` | 结构化文件编辑 | 内置 |
| `shell` | 终端命令执行 | 内置 |
| `unified_exec` | PTY 终端执行 | 内置 |
| `write_stdin` | 向运行中的进程发送 stdin | 内置 |
| `plan` (update_plan) | 计划/todo 管理 | 内置 |
| `view_image` | 查看/打开图片 | 内置 |
| `request_user_input` | 向用户请求输入 | 内置 |
| `request_permissions` | 请求扩展沙箱权限 | 内置 |
| `view_image` | 查看/打开图片 | 内置 |
| MCP tools | 外部 MCP 服务器工具 | MCP namespace |
| `spawn_agent` / `send_message` / `followup_task` | 多 Agent 协作 | 内置 |
| `spawn_agents_on_csv` / `report_agent_job_result` | CSV 批量处理 | 内置 |
| `create_goal` / `get_goal` / `update_goal` | Goal 管理 | 内置 |

```rust
// Tool 名称模型 (codex-rs/protocol/src/tool_name.rs)
pub struct ToolName {
    pub name: String,
    pub namespace: Option<String>,  // MCP/extension 共存
}
```

### 3.8 文件系统抽象

```rust
// codex-rs/exec-server/src/local_file_system.rs
// 三层文件系统:
//   DirectFileSystem  → 原生 tokio::fs (512MB 读取上限)
//   UnsandboxedFileSystem → 拒绝沙箱上下文, 委托给 DirectFileSystem
//   LocalFileSystem → 根据上下文选择沙箱/非沙箱
//   SandboxedFileSystem → macOS Seatbelt 或 Linux Landlock 强制执行
```

```rust
// codex-rs/file-system/src/lib.rs
#[async_trait]
pub trait ExecutorFileSystem: Send + Sync {
    async fn read_file(&self, ...) -> Result<String>;
    async fn write_file(&self, ...) -> Result<()>;
    async fn create_directory(&self, ...) -> Result<()>;
    async fn get_metadata(&self, ...) -> Result<FsMetadata>;
    async fn read_directory(&self, ...) -> Result<Vec<DirEntry>>;
    async fn remove(&self, ...) -> Result<()>;
    async fn copy(&self, ...) -> Result<()>;
}
```

### 3.9 文件监听器

```rust
// codex-rs/file-watcher/src/lib.rs
// 基于 notify crate 的多订阅者文件监听器:
//   - 支持递归和非递归监听
//   - 缺失路径回退 (监听祖先目录)
//   - 合并事件爆发
//   - ThrottledWatchReceiver 节流接收器
//   - 引用计数监听效率
```

### 3.10 事件模型

```rust
// codex-rs/exec/src/exec_events.rs
```

| 事件 | 内容 | 用途 |
|------|------|------|
| `thread.started` | 新 Thread 创建 | 会话跟踪 |
| `turn.started/completed/failed` | 回合生命周期 + token 用量 | 执行跟踪 |
| `item.started/updated/completed` | 单个 Item 生命周期 | 流式展示 |
| `AgentMessage` | 模型文本输出 | 内容展示 |
| `Reasoning` | 模型推理摘要 | 透明性 |
| `CommandExecution` | Shell 命令 + exit code/status | 执行追踪 |
| `FileChange` | 文件变更 + PatchApplyStatus | 变更展示 |
| `McpToolCall` | MCP 工具调用结果 | 工具追踪 |
| `CollabToolCall` | 多 Agent 协作事件 | Agent 关系 |
| `WebSearch` | 网页搜索查询 | 信息溯源 |
| `TodoList` | Agent 运行计划 | 进度展示 |
| `ErrorItem` | 非致命错误 | 错误处理 |

### 3.11 MCP 集成

**Codex 作为 MCP Client** (`codex-rs/codex-mcp/`):
- `connection_manager.rs` — 管理 MCP server 连接
- `elicitation.rs` — 处理获取请求 (auto-approve/deny/review)
- `tools.rs` — 转换 MCP tools → Codex tool 格式
- `auth_elicitation.rs` — MCP server OAuth 认证流

**Codex 作为 MCP Server** (`codex-rs/mcp-server/`):
- `codex_tool_config.rs` — 工具配置 (CodexToolCallParam/CodexToolCallReplyParam)
- `codex_tool_runner.rs` — 工具执行
- `exec_approval.rs` — exec 命令审批
- `patch_approval.rs` — apply_patch 审批
- 通信: JSON-RPC over stdin/stdout

### 3.12 沙箱系统

| 平台 | 技术 | 实现位置 |
|------|------|----------|
| **macOS** | Seatbelt (`.sbpl` 策略文件) | `sandboxing/` + `bwrap/` |
| **Linux** | Landlock + bubblewrap | `linux-sandbox/` |
| **Windows** | Restricted Tokens + AppContainer | `windows-sandbox-rs/` |

**沙箱模式**:
```rust
pub enum SandboxMode {
    DangerFullAccess,   // --yolo 无沙箱
    WorkspaceWrite,     // 工作区外只读
    ReadOnly,           // 完全只读
}
```

**远程执行** (`exec-server/`):
- `EnvironmentManager` — 管理和发现远程沙箱
- HTTP + WebSocket 传输
- `RemoteFileSystem` / `RemoteProcess` 统一抽象

### 3.13 IDE 集成关键特征总结

| 方面 | Codex 的做法 | 传统 IDE Agent |
|------|-------------|---------------|
| **文件编辑** | `apply_patch` 自定义 patch 格式 (可审计/可回滚) | 直接 workspace edit |
| **终端** | `shell`/`unified_exec` 双后端 (PTY 终端模拟) | 集成终端 |
| **编辑器上下文** | IPC Socket 双向通信 + context injection | LSP + Editor API |
| **文件监听** | notify crate 多订阅者 | IDE FileWatcher API |
| **审批模型** | 分层策略 (per-tool × per-mode × Granular) | 单一确认弹窗 |
| **沙箱** | Seatbelt/Landlock/Windows 三平台 | 通常无沙箱 |
| **多 Agent** | spawn/wait/close sub-agents 树形委托 | 很少支持 |
| **远程执行** | Remote ExecServer HTTP/WS transport | SSH / Dev Container |
| **MCP** | Client + Server 双模式 | 通常不支持 |

### 3.14 Codex **不使用**的技术

| 技术 | 为什么不使用 | 替代方案 |
|------|-------------|----------|
| **LSP** | 依赖外部基础设施, 启动/配置复杂 | shell 执行 compile/lint, 解析终端输出 |
| **IDE 诊断 API** | 平台绑定限制 | 从编译/测试输出中解析错误 |
| **IDE Code Actions** | 限制灵活性 | `update_plan` + `apply_patch` 工具组合 |
| **传统 Task Queue** | 回合驱动更适合 LLM Agent | RegularTask 循环 + agent spawn 模式 |

---

## 四、对祖龙项目的启示

### 4.1 任务编排层面

| 借鉴点 | Codex 的实现 | 祖龙可参考方向 |
|--------|-------------|---------------|
| **持久化目标** | Goal 6 状态机 + 自动延续 | 增加跨多回合的持久目标追踪, 避免上下文丢失 |
| **树形 Agent** | spawn_agent + AgentPath 层级 | Agent 的树形 fork/join 模式, 支持子 worker 委托 |
| **模型驱动拆分** | 模型自主决定并行, 系统提供原语 | 不要预先规划任务图, 让 LLM 决定何时并行 |
| **乐观并发** | `expected_goal_id` | 多 Agent 共享状态时使用乐观锁模式 |
| **优雅中断** | 100ms 宽限期 + AbortOnDropHandle RAII | 任务中断需要清理逻辑, 避免资源泄漏 |
| **批量作业** | CSV → N workers 数据并行模式 | 批处理任务的数据并行框架 |

### 4.2 IDE 集成层面

| 借鉴点 | Codex 的实现 | 祖龙可参考方向 |
|--------|-------------|---------------|
| **IPC 通道** | Unix Socket / Windows Named Pipe | 建立 IDE ↔ 后端的长连接通道, 不仅仅是文件 |
| **结构化编辑** | apply_patch (非 shell 写文件) | 更安全/可审计的文件编辑方式, 支持回滚 |
| **分层审批** | Never/OnRequest/OnFailure/Granular 四层 | 精细化权限管理, 支持 exec 和 interactive 两种模式 |
| **事件流** | 完整的事件模型 (thread/turn/item/command/...) | 建立完善的事件模型用于前端展示和日志 |
| **沙箱** | 三平台沙箱实现 | 执行 sandbox 参考 (Linux landlock 最简单可入手) |
| **MCP 双向** | Client + Server | 祖龙可对外暴露 MCP server, 也可消费外部 MCP tools |
| **PTY 终端** | Unified Exec 的 PTY 模拟 | 更丰富的终端交互体验 |
| **文件监听** | notify + throttle + 多订阅者 | 实时感知文件变更, 不必轮询 |

### 4.3 架构层面

| 借鉴点 | Codex 的实现 | 祖龙可参考方向 |
|--------|-------------|---------------|
| **Client-Server 分离** | TUI ↔ AppServer JSON-RPC | 前端/后端解耦, app-server 可独立部署 |
| **配置分层** | 5 层配置合并 | 支持 system/managed/user/project/CLI 多层配置 |
| **Feature flag** | Feature-gated 功能 | 功能渐进启用, A/B 测试能力 |
| **Plugin 系统** | core-plugins + extension-api | 支持第三方扩展能力 |
| **Onboarding** | 首次启动引导流 | 改善首次使用体验 |

---

## 五、附录: 关键文件索引

### Goal 系统
| 文件 | 说明 |
|------|------|
| `codex-rs/state/src/model/thread_goal.rs` | Goal 数据模型 + 6 状态定义 |
| `codex-rs/state/src/runtime/goals.rs` | GoalStore SQLite 持久化 |
| `codex-rs/ext/goal/src/accounting.rs` | Token/Time 双层会计 |
| `codex-rs/ext/goal/src/spec.rs` | 3 个模型可见工具定义 |
| `codex-rs/core/src/goals.rs` | GoalRuntimeState + 自动延续 |
| `codex-rs/core/templates/goals/continuation.md` | 延续回合提示模板 |
| `codex-rs/core/templates/goals/budget_limit.md` | 预算耗尽限制提示 |

### Multi-Agent 系统
| 文件 | 说明 |
|------|------|
| `codex-rs/core/src/agent/control.rs` | AgentControl 控制面 |
| `codex-rs/core/src/agent/registry.rs` | AgentRegistry + 无锁并发限制 |
| `codex-rs/core/src/agent/role.rs` | 内置角色: default/explorer/worker |
| `codex-rs/agent-graph-store/src/store.rs` | Agent 拓扑 BFS 持久化 |
| `codex-rs/agent-graph-store/src/types.rs` | ThreadSpawnEdgeStatus: Open/Closed |
| `codex-rs/core/src/tools/handlers/multi_agents_v2/spawn.rs` | spawn_agent 工具 |
| `codex-rs/core/src/tools/handlers/multi_agents_v2/message_tool.rs` | send_message/followup_task |
| `codex-rs/core/templates/agents/orchestrator.md` | Orchestrator 提示模板 |

### SessionTask 系统
| 文件 | 说明 |
|------|------|
| `codex-rs/core/src/tasks/mod.rs` | SessionTask trait + RunningTask + ActiveTurn |
| `codex-rs/core/src/tasks/regular.rs` | RegularTask 循环逻辑 |
| `codex-rs/core/src/tasks/lifecycle.rs` | 扩展生命周期事件 |
| `codex-rs/core/src/codex_delegate.rs` | 子 Agent 路由 + 级联关闭 |

### IDE 集成
| 文件 | 说明 |
|------|------|
| `codex-rs/tui/src/ide_context/ipc.rs` | IPC Socket/Named Pipe 通信桥 |
| `codex-rs/tui/src/ide_context/prompt.rs` | IDE 上下文注入 |
| `codex-rs/apply-patch/src/parser.rs` | Patch 格式解析 |
| `codex-rs/apply-patch/src/lib.rs` | Patch 应用 + unified diff 生成 |
| `codex-rs/core/src/tools/handlers/apply_patch.rs` | apply_patch 审批流程 |
| `codex-rs/protocol/src/approvals.rs` | 分层审批模型 |
| `codex-rs/exec/src/exec_events.rs` | 事件模型 |

### 沙箱
| 文件 | 说明 |
|------|------|
| `codex-rs/sandboxing/` | macOS Seatbelt 沙箱 |
| `codex-rs/linux-sandbox/` | Linux Landlock + bubblewrap |
| `codex-rs/windows-sandbox-rs/` | Windows Restricted Tokens |
| `codex-rs/exec-server/` | 远程沙箱执行 |
