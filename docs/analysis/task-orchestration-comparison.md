# 任务编排系统三方案对比分析 & 祖龙推荐方案

> 分析对象: OpenAI Codex | OpenHands | 祖龙(Zulong)  
> 分析日期: 2025-07-08  
> 祖龙核心约束: **4B 本地模型 (推理能力有限)** | VS Code 插件 | 中文环境 | 重构中途

---

## 一、三系统核心对比

### 1.1 任务编排哲学

| 维度 | Codex | OpenHands | 祖龙(现状) |
|------|-------|-----------|-----------|
| **任务抽象** | Goal = 持久目标, Thread = 会话 | Conversation = 任务 (无独立 Task 对象) | TaskGraph = 递归 DAG (有独立 Task 对象) |
| **编排模式** | 回合驱动 + 模型自主决定下一步 | Plan→Execute 两阶段 | 四路径并行 (FC/Orch/LangGraph/IDE) |
| **任务分解** | 模型通过 spawn_agent 自发分解 | 专门 PLANNING agent 创建 PLAN.md | 模型调用 task_create_plan → task_add_node |
| **状态管理** | 6 状态 Goal + agent 图拓扑 | ConversationExecutionStatus (简单) | 7 种状态类, 7 种节点状态, 碎片化 |
| **并行策略** | spawn_agent 独立 Thread 真并发 | 不支持 | Kahn 算法算层但从未实际并行 |
| **模型依赖** | **强依赖 GPT-4 级推理** | 中等, 但 Planning 和 Execute 分离 | **用 4B 模型做 GPT-4 级任务** |
| **工具数量** | ~15 个 (干净) | SDK 内部定义 | **23 个** (task 15 + CRUD 8, 功能重复) |

### 1.2 代码架构对比

| 维度 | Codex | OpenHands | 祖龙(现状) |
|------|-------|-----------|-----------|
| **语言** | Rust (~90 crates) | Python + TS/React | Python + TypeScript |
| **执行路径** | 1 条 (app-server JSON-RPC) | 1 条 (agent-server → SDK loop) | **4 条** (FC/Orch/LangGraph/IDE) |
| **客户端分离** | TUI ↔ AppServer 完全分离 | app_server 薄代理, SDK 外置 | IDE Server + InferenceEngine 紧耦合 |
| **持久化** | SQLite 统一 (thread-store) | SDK 内部 | **5 种** (SQLite/JSON × 3/内存) |
| **代码质量** | 生产级, 无废弃代码 | 清晰分层 | **废弃代码未清理, 守卫逻辑重复** |
| **提示词系统** | 模板文件 (`.md`) 分离管理 | Skills/Microagents 触发注入 | **~80 行硬编码中文**, 无模板系统 |

---

## 二、祖龙"混乱"的根因

### 2.1 一句话诊断

> **重构从 LangGraph 迁移到纯 Python 循环的中途状态 — 旧代码已删，但新代码分裂成了 4 条并行执行路径、7 种状态类、23 个工具、5 种持久化机制。最关键的是，用一个 4B 模型去承担 GPT-4 级的任务编排推理。**

### 2.2 五大根因

| # | 问题 | 影响 |
|---|------|------|
| 1 | **4 条执行路径各有不同行为** | 修 bug 要改 3-4 个地方，回归测试不可行 |
| 2 | **23 个工具对 4B 模型是灾难** | 模型频繁选错工具、遗漏步骤、幻觉参数 |
| 3 | **7 种状态类互不兼容** | 同一概念 (任务状态) 有 7 种不同表示 |
| 4 | **守卫逻辑在 fc_nodes.py 和 ide_fc_runner.py 中完全重复** | ~200 行重复代码，改一处忘另一处 |
| 5 | **4B 模型做 DAG 分解 = 反模式** | 模型无法正确推理依赖关系、拓扑排序、节点状态传播 |

### 2.3 量化对比

| 指标 | Codex | OpenHands | 祖龙 |
|------|-------|-----------|------|
| 执行路径数 | 1 | 1 | 4 |
| 状态类数量 | 3 (Goal/Agent/Turn) | 1 (ExecutionStatus) | 7 |
| Agent 工具数 | ~15 | SDK 内 (~12-15) | **23** |
| 持久化机制 | 1 (SQLite) | 1 (SDK) | **5** |
| 废弃代码 | 无 | 无 | 存在 (fc_graph.py 等) |
| 重复守卫逻辑 | 无 | 无 | fc_nodes + ide_fc_runner |

---

## 三、推荐方案: 借鉴 OpenHands 的 Plan-then-Execute，融合 Codex 的编辑模式

### 3.1 为什么不是 Codex?

Codex 的任务编排核心依赖是 **GPT-4 级模型的推理能力**:
- 模型自己决定何时 spawn_agent ("Prefer multiple sub-agents to parallelize your work")
- 模型自己决定任务拆分粒度 ("Process plan steps in parallel")
- 模型自己决定 Agent 间通信 ("your ONLY role becomes coordination")

这些指令对于一个 4B 模型来说完全不可行。Codex 的 orchestrator 提示词假定模型具备:
- 多步并行推理
- 资源评估 (什么时候该 spawn, 什么时候不该)
- 协调通信的时序理解

**结论: Codex 模式不适合祖龙。**

### 3.2 为什么是 OpenHands 的 Plan-then-Execute?

OpenHands 的 Plan-then-Execute 模式有 3 个关键特征非常适合小模型:

#### 特征 1: "Planning 和 Execution 使用不同的系统提示词"

```
Plan 阶段:
  "You are a Planning Agent that can ONLY create plans.
   You CANNOT execute code or make changes.
   Your ONLY output is PLAN.md with a structured task list."

Execute 阶段:
  "You are a Code Agent. Refer to PLAN.md for the current task.
   Execute ONE task at a time. Mark it complete in PLAN.md.
   Do NOT modify the plan structure."
```

这解决了 4B 模型的核心问题 — **单次推理能力有限**。把"想"和"做"拆成两个独立的 Agent 调用，每次只需处理一件事。

#### 特征 2: "PLAN.md 作为持久化的状态载体"

PLAN.md 是文件系统中的真实文件，包含:
```markdown
## 计划: XXX功能开发

### 状态: 🟡 进行中

- [x] 任务1: 创建数据模型 (已完成)
- [ ] 任务2: 实现 API 接口 (进行中)
- [ ] 任务3: 编写单元测试
- [ ] 任务4: 更新文档
```

Agent 不依赖内存状态，每次读取 PLAN.md 就知道自己在哪、接下来做什么。这与"7 种状态类"形成鲜明对比。

#### 特征 3: "Skills/Microagents 按需注入"

OpenHands 的 Skills 系统通过 trigger 机制按需注入领域知识:
- `KeywordTrigger` — 当用户消息包含特定关键词时激活
- `TaskTrigger` — 当任务类型匹配时激活
- 每个 Skill 是独立的 `.md` 文件，含 YAML frontmatter

祖龙目前是 ~80 行硬编码中文提示词，没有按需注入能力。

### 3.3 祖龙适配版 Plan-then-Execute 设计

#### 简化后的执行路径 (从 4 条 → 1 条)

```
用户请求
    │
    ▼
[意图检测] chat | plan | execute | resume
    │
    ├── chat → 直接 LLM 回复 (无任务编排)
    │
    ├── plan → PLAN.md Agent (专用 system prompt)
    │   └── 只能创建/修改 PLAN.md, 不能执行代码
    │   └── 输出: 结构化任务列表, 每个任务一行
    │
    ├── execute → Code Agent (专用 system prompt)
    │   └── 读取 PLAN.md → 找第一个未完成任务
    │   └── 执行 → 标记 [x] → 下一个
    │   └── 循环直到全部完成或用户中断
    │
    └── resume → 读取 PLAN.md + 最后状态 → 继续 execute
```

**从 4 条路径收敛到 1 条**，从 7 种状态收敛到 PLAN.md 文件本身即为状态。

#### 工具简化 (从 23 个 → 8 个)

| 保留工具 | 用途 | 替换/合并原工具 |
|----------|------|----------------|
| `read_file` | 读取文件 | (已有) |
| `write_file` | 写入文件 | (已有) |
| `apply_patch` | 结构化编辑 (借鉴 Codex) | 替代 shell 写文件 |
| `shell` | 执行命令 | (已有) |
| `read_plan` | 读取 PLAN.md | 替代 task_view_overview |
| `update_plan_status` | 标记任务 [x] | 替代 task_mark_status |
| `request_user_input` | 请求用户输入 | (新增) |
| `web_search` | 搜索 | (已有) |

删除的 15+ 工具:
- `task_create_plan` / `task_add_node` / `task_add_dependency` / `task_remove_node` / ... → **全部由 PLAN.md Agent 的 system prompt 替代**
- `graph_create_node` / `graph_create_edge` / ... → **不再需要** (PLAN.md 替代 DAG)
- `task_suspend` / `task_list_suspended` / `task_resume_by_address` → **简化为 PLAN.md 行级状态**

#### 状态管理

**过去 (7 种状态类)**:
```
FCLoopState + OrchestratorState + IDEFCState + AgentSession + 
TaskSnapshot + SuspendableTaskState + ZulongState
```

**未来 (PLAN.md 即状态)**:
```
PLAN.md 文件内容 → 无需额外状态类
- [ ] = pending
- [~] = in_progress
- [x] = completed
- [!] = blocked
```

#### PLAN.md 文件格式

```markdown
---
plan_id: abc123def456
created: 2025-07-08T10:30:00
total_tasks: 4
completed_tasks: 1
---

## 计划: 为用户模块添加分页功能

- [x] 1. 在 User 模型中添加 paginate 方法
- [~] 2. 修改 API 端点返回分页数据
- [ ] 3. 更新前端列表组件显示分页控件
- [ ] 4. 编写分页功能的单元测试
```

Agent 执行循环:
1. `read_file("PLAN.md")` → 找到第一个 `[ ]` 或 `[~]` 行
2. 执行对应任务
3. `apply_patch` 将 `[ ]`/`[~]` 改为 `[x]`
4. 重复

---

## 四、迁移路径

### Phase 1: 收敛 (1-2 周)

**目标: 从 4 条路径 → 1 条**

```
[x] 1. 废弃 Orchestrator (路径 2/3) — 删除 orchestrator_graph.py 引用
[x] 2. IDE FC Runner (路径 4) 合并到统一 FC Runner (路径 1)
[x] 3. 删除 fc_graph.py, unified_fc_runner.py 兼容层
[x] 4. 合并 FCLoopState + IDEFCState + OrchestratorState → 1 个 State
[x] 5. 提取守卫逻辑到共享模块 (消除 fc_nodes / ide_fc_runner 重复)
```

### Phase 2: 简化 (2-3 周)

**目标: 实现 Plan-then-Execute**

```
[ ] 6. 实现 PLAN.md Agent (独立 system prompt, 只能创建/修改 PLAN.md)
[ ] 7. 实现 Code Agent (独立 system prompt, 只能执行 PLAN.md 中的任务)
[ ] 8. 删除 23 个 Task/CRUD 工具 → 保留 8 个核心工具
[ ] 9. 实现 read_plan / update_plan_status 工具
[ ] 10. 实现 apply_patch 工具 (借鉴 Codex patch 格式)
```

### Phase 3: 增强 (3-4 周)

**目标: 按需注入 + 持久化统一**

```
[ ] 11. 实现 Skills 按需注入系统 (KeywordTrigger / TaskTrigger)
[ ] 12. 统一持久化: 用 SQLite/JSON 单例管理 PLAN.md + 事件存储
[ ] 13. 前端 FocusChain 适配 PLAN.md 格式
[ ] 14. 测试: 用 10 个真实任务验证端到端流程
```

---

## 五、核心决策速查

| 维度 | 放弃 (祖龙现状) | 采纳 |
|------|----------------|------|
| **编排模式** | TaskGraph DAG (模型无法推理) | **OpenHands Plan-then-Execute** |
| **状态管理** | 7 种状态类 | **PLAN.md 文件即状态** |
| **工具集** | 23 个 task/graph 工具 | **8 个核心工具** |
| **执行路径** | 4 条并行 | **1 条统一路径** |
| **文件编辑** | shell 直接写文件 | **Codex apply_patch 模式** |
| **领域注入** | ~80 行硬编码提示词 | **Skills 按需注入** |
| **持久化** | 5 种分散机制 | **统一 SQLite/JSON** |

---

## 六、三系统可借鉴点总表

| 功能 | Codex | OpenHands | 祖龙推荐采纳 |
|------|-------|-----------|-------------|
| 任务编排 | 回合驱动 GG Goal | **Plan-then-Execute** ✅ | **OpenHands** |
| 状态管理 | Goal 6 状态机 | **PLAN.md 文件** ✅ | **OpenHands** |
| 文件编辑 | **apply_patch** ✅ | SDK 内部 | **Codex** |
| 审批系统 | **分层 4 级** ✅ | 简易确认 | **Codex** |
| 沙箱 | **三平台沙箱** ✅ | Docker sandbox | **Codex** (长期) |
| 领域注入 | Skills + Plugins | **Skills/Microagents** ✅ | **OpenHands** |
| 多 Agent | **spawn_agent 树形** | Sub-conversations | 长期考虑 |
| 事件模型 | **完整事件流** ✅ | 事件驱动 | **Codex** |
| MCP 集成 | **Client + Server** ✅ | 不支持 | **Codex** |
| 配置分层 | 5 层合并 | 用户设置 | **Codex** |
| 批量处理 | Agent Jobs CSV | 不支持 | 特定场景 |
