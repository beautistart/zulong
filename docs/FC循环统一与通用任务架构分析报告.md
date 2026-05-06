# FC 循环统一与通用任务架构分析报告

> **文档定位**: 两套 FC 循环深度对比、记忆/注意力/任务编排模块审查、统一方案设计
> **分析日期**: 2026-05-04
> **分析方法**: 7 个核心模块源码逐行审查 + 函数级对比
> **涉及文件**: fc_graph.py, ide_fc_runner.py, task_graph.py, memory_graph.py, attention_window.py, ide_tool_registry.py, ide_server.py, web_chat_router.py

---

## 1. 核心发现：两套 FC 循环并行运行

### 1.1 两套 FC 循环概述

| 维度 | `fc_graph.py` (主系统 LangGraph) | `IDEFCRunner` (IDE while循环) |
|------|------|------|
| **实现** | LangGraph StateGraph 4 节点有向图 | Python `while True` 异步循环 |
| **入口** | `run_fc_loop()` 被 `InferenceEngine` 调用 | `IDEFCRunner.run_loop_async()` 被 IDE Server 调用 |
| **工具执行** | **全部本地执行** — `engine._execute_tool_call()` → `tool_engine.call_tool()` | **分流执行** — 内部工具本地 + 远程工具 WebSocket 推送 |
| **编程工具** | `exec_write_file`, `exec_run_command` (服务器本地，受限 workspace + 命令白名单) | `read_file`, `write_to_file`, `execute_command` 等 10 个 (IDE 端执行，完整项目上下文) |
| **调用来源** | EventBus → L1-B → L2 (Full模式) 或 Web直连 (IDE模式) | IDE WebSocket `session_start` |
| **代码量** | 1,493 行 | 1,900+ 行 |

### 1.2 函数级对照

| 阶段 | `fc_graph.py` | `IDEFCRunner` | 差异度 |
|------|------|------|------|
| 前置检查 | `check_node()` | `_check()` | **极低** — 相同逻辑 |
| 模型调用 | `call_model_node()` — 同步 `create()` | `_call_model()` — ThreadPoolExecutor + 备用模型 fallback + Qwen3思维链禁用 | **中等** |
| 工具执行 | `exec_tools_node()` — 全部同步本地 | `_exec_tools_async()` — 内部/远程分流 + WebSocket 推送等待 | **高** — 核心分歧点 |
| 回复评估 | `eval_response_node()` — CB降级 + Backfill + InfoGap | `_eval_response()` — 相同逻辑 + 漂移检测 + 对话适配器 | **中等** |
| 终结 | 无显式 finalize | `_finalize()` — 完整生命周期管理 | **高** |

---

## 2. IDEFCRunner 独有的 6 大认知能力

fc_graph.py 路径完全缺失以下能力：

| # | 能力 | IDEFCRunner 实现 | fc_graph.py |
|---|------|-------------------|-------------|
| 1 | **对话轮次跟踪** | `DialogueAdapter` + MemoryGraph DIALOGUE 节点 | 无 |
| 2 | **语义漂移检测** | `SemanticDriftDetector` 监测输出偏离 | 无 |
| 3 | **会话记忆自动保存** | `_auto_save_session_memory()` FC 结束时持久化 | 无 |
| 4 | **注意力窗口持久化/恢复** | `serialize()` / `from_serialized()` 到 session | 无 (每次从零) |
| 5 | **安全网组件隔离** | per-runner 独立 CB/RG + serialize/deserialize | 全局共享 (并发冲突) |
| 6 | **任务自动完成** | `_auto_complete_task()` + `_build_subtask_context()` | 无 |

---

## 3. 各模块深度审查摘要

### 3.1 TaskGraph (task_graph.py, 1,089 行, 生产级 4/5)

- **节点体系**: 5 层深度 + 7 种状态
- **调度算法**: Kahn 拓扑排序 + DFS 环检测
- **MemoryGraph 同步**: `update_node_status()` → `_sync_node_to_memory_graph()` → 激活值映射 (in_progress=0.9, completed=0.1)
- **序列化**: 完整 serialize/deserialize 支持任务恢复

### 3.2 MemoryGraph (memory_graph.py, 3,248 行, 生产级 4.5/5)

- **异构图**: 9 种节点 (TASK/DIALOGUE/KNOWLEDGE/EXPERIENCE/EPISODE/FILE/CONCEPT/PERSON/DOCUMENT) + 7 种边
- **三维标签**: Temperature (HOT/WARM/COLD) × Importance (6级, MUST_REMEMBER永不衰减) × TimeScope
- **核心算法**:
  - BFS 扩散激活: `compute_activations()` max_depth=3, decay=0.5
  - 赫布学习: `new_w = old_w + 0.1 × (1 - old_w)`, 共激活≥3次自动创建ASSOCIATION边
  - 艾宾浩斯衰减: 6级半衰期 (TRIVIAL:6h → MUST_REMEMBER:∞)
- **FAISS 双路径**: 热路径(BFS+关键词) ‖ 冷路径(FAISS向量+关键词融合 0.7:0.3)
- **FC 工具**: 7 个 (recall_memory, read_memory_node, save_memory_note, discover_related, activate_memory_network, list_memory, set_importance)

### 3.3 AttentionWindow (attention_window.py, 948 行, 生产级 4/5)

- **三种模式**:
  - GLOBAL: 大纲权重高, 深度递减 (depth0→×1.2, depth4+→×0.3)
  - FOCUS: 当前节点×3.0, 祖先/依赖×2.0, 兄弟×1.5, 无关×0.5
  - SINGLE_CHAIN: 当前链×5.0, 祖先×3.0, 依赖×2.5, 无关×0.2
- **自动状态机**: recall_memory→FOCUS, exec_write_file→SINGLE_CHAIN, task_view_overview→GLOBAL
- **权重公式**: `base × time_decay(0.95^age) × mode_mult × memory_boost(1.0~1.5)`
- **消息分组**: tool_group 确保 assistant+tool 消息原子性淘汰
- **淘汰处理**: 淘汰摘要持久化到 MemoryGraph + TaskGraph, 提示LLM用recall_memory恢复
- **LLM 工具**: navigate_attention (deeper/broader/jump) + adjust_attention_mode

---

## 4. 两套 FC 循环对认知能力的使用差异

### 完整对比矩阵

| 认知能力 | fc_graph.py (主系统) | IDEFCRunner (IDE) |
|---------|:---:|:---:|
| AttentionWindow 实例化 | 全局共享 engine._attn_window | per-runner 独立 self._attn_window |
| AttentionWindow 持久化/恢复 | 无 | serialize/from_serialized |
| BFS 扩散激活 | exec_tools_node 后调用 | 独立方法, 多处调用 |
| 自动焦点漂移 | 有 (BFS最高激活>0.6时跳转) | 有 (相同逻辑) |
| 对话轮次跟踪 | **无** | DialogueAdapter + DIALOGUE 节点 |
| 语义漂移检测 | **无** | SemanticDriftDetector |
| 会话记忆自动保存 | **无** | _auto_save_session_memory() |
| 子对话记录 | **无** | _record_sub_dialogue() |
| CircuitBreaker 隔离 | 全局共享 | per-runner + serialize |
| RuleGuardian 隔离 | 全局共享 | per-runner + 恢复 |
| 任务自动完成 | **无** | _auto_complete_task() |
| TaskGraph 合成回复 | **无** | _synthesize_from_task_graph() |
| Backfill 任务图 | 有 | 有 |
| Web 监控广播 | **无** | 全程 FC_START/CALLING_MODEL/FC_DONE |

---

## 5. 统一 FC 循环方案

### 5.1 核心原则

以 IDEFCRunner 为基座升级为 `UnifiedFCRunner`，废弃 fc_graph.py。

### 5.2 统一架构

```
                         ┌─────────────────────────────────────────┐
                         │          UnifiedFCRunner                 │
                         │     (现 IDEFCRunner 升级)              │
                         │                                         │
                         │  ┌─ AttentionWindowManager (per-runner)  │
                         │  ├─ CircuitBreaker (per-runner)          │
                         │  ├─ RuleGuardian (per-runner)            │
                         │  ├─ SemanticDriftDetector                │
                         │  ├─ DialogueAdapter (MemoryGraph 跟踪)   │
                         │  └─ ToolExecutor (统一工具执行层) ← 新增  │
                         └───────────────┬─────────────────────────┘
                                         │
               ┌─────────────────────────┼──────────────────────────┐
               │                         │                          │
    ┌──────────▼──────────┐  ┌───────────▼──────────┐  ┌───────────▼──────────┐
    │   Web 前端入口       │  │   IDE 插件入口        │  │   EventBus 入口       │
    │   web_chat_router    │  │   ide_server           │  │   L1-B → L2          │
    │   /ws CHAT_MESSAGE   │  │   /ide session_start  │  │   USER_TEXT event     │
    └──────────────────────┘  └───────────────────────┘  └───────────────────────┘
```

### 5.3 ToolExecutor 统一工具路由

```
ToolExecutor.execute(tool_call)
├─ classify(tool_name)
├─ "internal" → tool_engine.call_tool() (始终本地)
├─ "remote" → IDE 已连接?
│   ├─ YES → WebSocket 推送到 IDE
│   └─ NO → 降级为本地 exec_write_file/exec_run_command
```

### 5.4 各路径改造后的能力获得

| 认知能力 | Web (改造前→后) | EventBus (改造前→后) | IDE |
|---------|:---:|:---:|:---:|
| FC 循环 | 无→**有** | LangGraph→**Unified** | 有 |
| 记忆检索 | 1次注入→**FC自主调用** | FC工具→FC工具 | FC工具 |
| BFS 扩散激活 | 无→**每轮** | 工具后→**每轮** | 每轮 |
| 注意力窗口 | 无→**三模式** | 全局共享→**per-runner** | per-runner |
| 对话记录 | 无→**DialogueAdapter** | 无→**DialogueAdapter** | 有 |
| 语义漂移检测 | 无→**DriftDetector** | 无→**DriftDetector** | 有 |
| 安全网隔离 | 无→**per-runner** | 全局→**per-runner** | per-runner |

### 5.5 废弃清单

| 文件 | 处置 |
|------|------|
| `zulong/l2/fc_graph.py` (1,493行) | **废弃** |
| `zulong/ide/ide_fc_runner.py` (1,900+行) | **升级**为 UnifiedFCRunner, 迁移到 `zulong/l2/` |
| `zulong/ide/ide_tool_registry.py` (307行) | **重构** — 分类逻辑迁到 ToolExecutor |
| `zulong/tools/exec_tools.py` (251行) | **保留** — 无 IDE 时的降级执行后端 |

### 5.6 实施步骤

1. 新建 `ToolExecutor` 类，封装 internal/IDE remote/本地降级 三路由
2. `IDEFCRunner` → `UnifiedFCRunner`，迁移到 `zulong/l2/unified_fc_runner.py`
3. `UnifiedFCRunner._exec_tools` 改用 `ToolExecutor`
4. `InferenceEngine` 替换 `run_fc_loop()` 为 `UnifiedFCRunner.run_loop_async()`
5. `web_chat_router.py` 的 `_chat_via_engine()` 改用 `UnifiedFCRunner`
6. IDE 连接状态广播 — 连接/断开时更新全局注册表
7. 端到端测试：三条入口 × 有/无 IDE 连接 = 6 种场景
8. 废弃 `fc_graph.py`

---

## 6. 与 Cline 的关系

### 现状

Cline v3.82.0 fork → zulong-ide 插件 → 全面重写通信协议 (XML → WebSocket JSON → FC tool_calls) → Cline 降级为"VS Code 工具执行壳"。

### 接口化隔离原则

- 祖龙通过 WebSocket JSON 协议消费 IDE 工具，不在代码级依赖 Cline
- Cline 更新新 VS Code 工具 → 插件端实现 + 祖龙端加一条 schema
- Cline 更新 Agent 编排 → 与祖龙无关
- Cline 更新 UI → 按需 cherry-pick

### 祖龙内部工具可用性（脱离 IDE）

| 类别 | 工具数 | 脱离 IDE |
|------|:---:|:---:|
| 记忆 | 7 | **全部可用** |
| 任务 | 8 | **全部可用** |
| 注意力 | 2 | **全部可用** |
| 搜索 | 2 | **全部可用** |
| 系统 (file/network/command) | 3 | **可用** |
| 代码锚点 | 3 | 部分依赖代码 |
| 执行 (exec_write/run) | 2 | **可用** (本地) |
| IDE 远程 | 10 | **依赖 IDE** |
| **合计脱离 IDE 可用** | **22+** | |

---

## 7. 通用工具验证结果（已完成检查）

### 7.1 工具存在性审计结论

| 需求工具 | 现有等价物 | 位置 | 状态 | 差距说明 |
|----------|-----------|------|------|---------|
| `web_search` | `OpenClawSearchTool.search` | `tools/openclaw_search.py` | **已有** | 依赖本地 OpenClaw API (`http://localhost:3000/api`)，需确保 OpenClaw 服务可用 |
| `web_fetch` | `OpenClawSearchTool.fetch_webpage` | `tools/openclaw_search.py` | **已有** | 专门的 `fetch_webpage` action，通过 OpenClaw API 代理获取；另有 `NetworkTool.get` 作为底层 HTTP GET |
| `ask_user` | `ask_followup_question` | `ide/ide_tool_registry.py` IDE_REMOTE_TOOLS | **仅 IDE 可用** | 只注册为远程工具，Web/EventBus 入口无法使用。需提升为内部工具 |
| `submit_final_answer` | **幽灵工具** | 被 circuit_breaker / attention_window / task_graph 引用但从未注册 | **未实现** | 5 个模块预期它存在 (CB白名单、AW模式切换、TG注释)，但 `tools/` 目录中无定义 |
| `attempt_completion` | `IDE_REMOTE_TOOLS` 中的远程工具 | `ide/ide_tool_registry.py` | **仅 IDE 可用** | 同 ask_user，需提升或统一 |
| `generate_document` | `exec_write_file` / `FileTool.write` | `tools/exec_tools.py` / `tools/system_tools.py` | **已有** | 两处都可生成文件，无需新增 |

### 7.2 关键发现：submit_final_answer 幽灵引用

`submit_final_answer` 在 5 个位置被引用，但从未被实际注册为工具：

| 文件 | 行号 | 引用方式 | 预期行为 |
|------|------|---------|---------|
| `circuit_breaker.py` | L61 | `CB_RETAINED_NAMES` 白名单 | CB RED 状态下仍允许调用 |
| `circuit_breaker.py` | L74 | 终结类工具白名单 | 分类为"终结工具" |
| `attention_window.py` | L89 | GLOBAL_TRIGGER_TOOLS 集合 | 调用后自动切换到 GLOBAL 模式 |
| `attention_window.py` | L519 | 特殊处理分支 | 调用时清除焦点状态 |
| `task_graph.py` | L129 | 注释文档 | 模型通过它完成任务 |

**影响**: `fc_graph.py` 绕过了这个工具，用 `should_terminate: "done"` + 任务节点自动合成代替。但 IDEFCRunner 的 `_get_cb_retained_tools()` (L1039-1041) 显式保留了它。这说明设计意图是让 LLM 主动调用 `submit_final_answer` 来结束任务，但工具从未被实现。

### 7.3 需新增/提升的工具清单

| 优先级 | 工具 | 动作 | 说明 |
|--------|------|------|------|
| **P0** | `submit_final_answer` | **新增内部工具** | 结束 FC 循环，返回最终回复。5 个模块已预期其存在 |
| **P0** | `ask_user` | **新增内部工具** | Web/EventBus 场景下向用户提问。IDE 场景降级为 `ask_followup_question` 远程调用 |
| **P1** | `web_search` | **别名映射** | 将 `openclaw_search` 的 `search` action 暴露为独立 FC 工具 schema |
| **P1** | `web_fetch` | **别名映射** | 将 `openclaw_search` 的 `fetch_webpage` action 暴露为独立 FC 工具 schema |
| **P2** | `generate_image` | 按需新增 | 通用任务 (如设计无人机) 可能需要图像生成能力 |
| **P2** | `structured_output` | 按需新增 | 输出结构化数据 (JSON/表格/图表) |

---

## 8. 外部工具调研与市面方案

### 8.1 代码智能工具评估

#### 8.1.1 code-review-graph

| 维度 | 详情 |
|------|------|
| **功能** | 基于 Tree-sitter 构建代码结构知识图谱 (AST→SQLite), 用于 AI 辅助 Code Review, 减少 5-10x LLM token |
| **GitHub Stars** | ~10,800 |
| **许可证** | **MIT** (完全开放) |
| **架构** | Tree-sitter 解析 12 种语言 + SQLite 本地存储 + 增量更新 (<2s) + MCP 协议暴露工具 |
| **MCP 工具** | `review_pull_request`, `get_symbol_context`, `search_codebase`, `get_file_structure` |
| **集成可行性** | **高** — MCP 原生接口, 零基础设施依赖, 可直接包装为 FC 工具 |
| **风险** | 单人维护 (存在 bug 修复 fork), SQLite 对超大仓库有瓶颈 |
| **推荐** | **推荐集成** — MIT 许可 + 零基础设施 + MCP 原生, 最佳平衡点 |

#### 8.1.2 GitNexus

| 维度 | 详情 |
|------|------|
| **功能** | "零服务器代码智能引擎", 完整知识图谱 + 爆炸半径分析 + 混合搜索 (BM25+语义+RRF) + 跨仓分析 + 污点分析 |
| **GitHub Stars** | ~28,000-28,900 (爆发式增长) |
| **许可证** | **PolyForm Noncommercial** — **禁止商业使用** |
| **架构** | Tree-sitter AST + LadybugDB (嵌入式图数据库) + HuggingFace Transformers.js + WebAssembly 浏览器端 |
| **MCP 工具** | `impact` (爆炸半径), `context`, `query`, `cypher`, `rename`, `detect_changes`, `list_repos` |
| **集成可行性** | 技术可行性**高**, 但许可证**致命** |
| **风险** | **许可证是硬伤** — PolyForm Noncommercial 禁止在商业产品中使用；LadybugDB 成熟度存疑 |
| **推荐** | **不推荐商业集成** — 除非单独谈判商业许可 |

#### 8.1.3 CodeGraphContext

| 维度 | 详情 |
|------|------|
| **功能** | MCP 服务器 + CLI, 将代码索引到 Neo4j 图数据库, 支持 Cypher 查询和图算法 |
| **GitHub Stars** | ~2,765-3,000 |
| **许可证** | **MIT** |
| **架构** | Python AST 解析 + **Neo4j** (外部依赖) + MCP 服务器 |
| **集成可行性** | **中高** — Neo4j 是优势 (成熟、图算法丰富) 也是劣势 (运维开销) |
| **风险** | Neo4j CE 是 GPL (传染性); 当前仅支持 Python; 社区较小 |
| **推荐** | 如果已有 Neo4j 基础设施可考虑, 否则 code-review-graph 更轻量 |

#### 8.1.4 Understand-Anything

| 维度 | 详情 |
|------|------|
| **功能** | Claude Code 插件, 多 Agent 流水线分析代码库 → 生成交互式知识图谱 Web Dashboard (React Flow) |
| **GitHub Stars** | ~3,000-5,000+ |
| **许可证** | **MIT** |
| **架构** | LLM + web-tree-sitter (WASM) + React 18 + Vite + React Flow 可视化 |
| **集成可行性** | **低中** — 为人类理解设计, 非机器消费; 但多 Agent 流水线架构值得学习 |
| **风险** | 紧耦合 Claude Code 生态; 目的不匹配 (可视化 vs 工具调用) |
| **推荐** | **不直接集成**, 但可借鉴其多 Agent 并行分析的设计模式 |

### 8.2 工具集成推荐矩阵

| 工具 | 集成价值 | 许可风险 | 集成成本 | **决策** |
|------|---------|---------|---------|---------|
| code-review-graph | 高 | 无 (MIT) | 低 (MCP 原生) | **推荐集成** |
| GitNexus | 极高 | **致命** (Noncommercial) | 低 (MCP 原生) | **商业禁用** |
| CodeGraphContext | 中高 | 低 (MIT, 但 Neo4j GPL 传染) | 中 (需 Neo4j) | 按需考虑 |
| Understand-Anything | 低 | 无 (MIT) | 高 (重构) | 仅借鉴架构 |

---

## 9. 市面通用复杂任务编排方案全景

### 9.1 主流框架对比

| 框架 | Stars | 许可 | 编程任务 | 通用任务 | 任务分解方式 | 循环防护 |
|------|-------|------|---------|---------|------------|---------|
| **AutoGPT** | ~170k | MIT | 工具级 | 主要 | ReAct 循环 (目标→步骤) | 步数限制 |
| **MetaGPT** | ~45k | MIT | **主要** | 有限 | SOP 流水线 (PM→架构→开发→QA) | **结构性防护** (固定DAG) |
| **CrewAI** | ~25k | MIT | 工具级 | **主要** | 角色+任务依赖 (顺序/层级) | `max_iter` per agent |
| **AutoGen/AG2** | ~35k | MIT/Apache2 | 强 | **强** | 多 Agent 对话 + 嵌套会话 | `max_turns` + `is_termination_msg` |
| **LangGraph** | ~8k | MIT | 图定义 | 图定义 | 有向图 (DAG/cyclic) | `recursion_limit` + 条件边 |
| **OpenHands** | ~45k | MIT | **主要** | 扩展中 | Agent 委托 + 多轮推理 | 轮数+预算限制 |
| **TaskWeaver** | ~5k | MIT | 代码优先 | 数据分析 | Planner→CodeGen→Executor | 轮次+自反思 |
| **CAMEL** | ~5k | Apache2 | 工具级 | 主要 | 角色扮演 + Inception 提示 | 轮数+终止令牌 |

### 9.2 祖龙对标分析

祖龙当前架构 `orchestrator_graph.py` (Plan→Schedule→Execute→Reflect→Synthesize) 最接近 **LangGraph + MetaGPT** 的组合：
- **Plan 阶段** ≈ MetaGPT 的 PM+Architect 角色 (但祖龙由 LLM+FC 工具动态生成，更灵活)
- **Schedule 阶段** ≈ LangGraph 的拓扑排序 + 条件路由 (祖龙用 Kahn 算法)
- **Execute 阶段** ≈ 嵌套 FC 循环 (内层 `run_fc_loop()`)
- **Reflect 阶段** ≈ AutoGen 的 Agent 对话反思 (CONTINUE/REDO/REPLAN 三向路由)
- **Synthesize 阶段** ≈ CrewAI 的 Hierarchical Process 汇总

**祖龙的独特优势**: MemoryGraph (赫布学习+BFS扩散) + AttentionWindow (三模式+原子淘汰) + TaskGraph (5层深度拓扑排序) — 这些认知基础设施在所有对比框架中均无对标。

---

## 10. 编程任务与通用任务的循环封闭分析

### 10.1 问题描述

用户提出的核心问题：复杂通用任务 (如"设计快递无人机") 可能包含代码实现子任务 (如"编写飞控算法"), 而代码实现子任务又可能产生非编程子任务 (如"查询 PID 控制理论"), 形成递归嵌套：

```
通用任务: "设计快递无人机"
├── 子任务: "调研无人机气动布局"        → 通用 (web_search + 文献分析)
├── 子任务: "设计飞控系统"             → 通用 (系统设计)
│   ├── 子子任务: "编写 PID 控制器"    → 编程 (write_file + run_command)
│   │   ├── 子子子任务: "查询 PID 参数经验值" → 通用 (web_search)  ← 循环!
│   │   └── 子子子任务: "编写单元测试"  → 编程
│   └── 子子任务: "选择通信协议"       → 通用 (调研)
└── 子任务: "成本分析"                → 通用 (计算 + 表格)
```

**风险**: 如果通用 FC 循环和编程 FC 循环是独立的编排器, 通用→编程→通用→编程 会形成递归委托, 可能导致无限循环或调用栈溢出。

### 10.2 业界解决方案

学术论文和开源项目提供了 6 种循环防护机制:

| 方案 | 来源 | 原理 | 优势 | 劣势 |
|------|------|------|------|------|
| **固定 DAG 流水线** | MetaGPT SOP | 角色只能前向传递, 不允许回溯 | 结构性保证无循环 | 过于僵硬, 不适合通用任务 |
| **最大委托深度** | ReDel (EMNLP 2024) | `max_delegation_depth` 参数限制递归层数 | 简单有效, 允许有限递归 | 需要选择合理的深度阈值 |
| **同一任务检测** | ReDel | 如果子 Agent 收到与自己完全相同的指令, 阻止委托 | 防止"传话筒"式递归 | 仅防精确重复, 变体可绕过 |
| **扁平禁令** | Goose Framework | 子 Agent 不能再创建子 Agent | 完全杜绝递归 | 牺牲深度分解能力 |
| **预算耗尽** | OpenHands | 全局 token/API 成本追踪, 超限即停 | 不限制结构, 限制资源 | 不保证任务完成 |
| **不可变计划版本** | Graph Harness (2025论文) | 规划/执行/恢复三层分离, 计划一旦制定不可变 | 理论最严谨 | 实现复杂, 灵活性差 |

### 10.3 祖龙现有的循环防护机制

分析 `orchestrator_graph.py` 的现有安全网:

| 安全网 | 位置 | 机制 | 覆盖范围 |
|--------|------|------|---------|
| **全局 FC 步数限制** | `orchestrator_graph.py` L262 | `total_fc_turns >= max_total_fc_turns (默认100)` | 全局 — 所有子任务共享预算 |
| **子任务 FC 预算** | `orchestrator_graph.py` L375 | `effective_budget = min(subtask_budget(30), remaining_global)` | 单任务最多 30 步 |
| **外层迭代限制** | `orchestrator_graph.py` L1171 | `max_iterations = 200` (Plan-Schedule-Execute-Reflect 循环) | 编排器级别 |
| **Replan 次数限制** | `OrchestratorState.replan_count` | 反思阶段 REPLAN 路由有上限检查 | 防止无限重规划 |
| **TaskGraph 深度限制** | `task_graph.py` L406 | `depth > 50` 保护 (但无业务层限制) | 防止无限嵌套子任务 |
| **CircuitBreaker** | `circuit_breaker.py` | 重复工具调用检测 → YELLOW → RED 降级 | 防止工具调用死循环 |
| **LangGraph recursion_limit** | `fc_graph.py` L1325 | `recursion_limit = engine._hard_limit + 10` | FC 循环图遍历安全网 |

### 10.4 祖龙的循环封闭隐患

**现有防护足以应对单层编排**, 但引入通用任务后存在以下隐患:

#### 隐患 1: 缺少任务类型标记

TaskGraph 的 TaskNode.type 由**深度**决定 (`requirement/analysis/outline/task/subtask`), 不区分"编程任务"和"通用任务"。当一个 `subtask` 节点的内容是"编写 PID 控制器"时, 编排器无法判断应该使用哪套工具集。

#### 隐患 2: Execute 阶段的工具集固定

`_EXECUTE_TOOLS` 是静态集合 (L80-86), 包含 `exec_write_file`, `exec_run_command` 等编程工具。通用子任务 (如"调研气动布局") 不需要这些工具, 但需要 `web_search`, `web_fetch`, `ask_user` 等工具。当前无法按子任务类型动态切换工具集。

#### 隐患 3: 编排器无递归深度追踪

如果在 Execute 阶段的 `run_fc_loop()` 内部, LLM 通过 `task_create_plan` 创建了**新的子任务图**, 会在当前 FC 循环内嵌套一个新的编排流程。当前没有追踪这种嵌套深度。

#### 隐患 4: submit_final_answer 的缺失

LLM 没有"我做完了"的显式工具。当通用任务子任务 (如调研报告) 完成时, LLM 只能通过文本回复表达完成, 但编排器依赖 `task_mark_status` 来推进流程。如果 LLM 不主动调用 `task_mark_status`, 可能导致子任务卡死。

### 10.5 循环封闭防护方案 (已实施)

取代原先的五层硬限制设计, 采用**检查点+进度报告+自动继续**的弹性方案:

```
┌─────────────────────────────────────────────────────────────────┐
│ IDEFCRunner._check() 软/硬限制                                   │
│   soft_limit: 注入进度提示到 messages (不终止)                     │
│   hard_limit: 保存 runner state → 返回 "checkpoint" (可恢复)     │
├─────────────────────────────────────────────────────────────────┤
│ Orchestrator schedule_node() 预算耗尽处理                         │
│   1. _save_checkpoint() 保存完整编排状态                           │
│   2. _generate_progress_report() 生成 Markdown 进度报告           │
│   3. 自动延伸预算 50% (min 20 步)                                 │
│   4. 继续执行, 不强制跳转 SYNTHESIZE                              │
├─────────────────────────────────────────────────────────────────┤
│ 任务类型感知路由 (已实现)                                          │
│   TaskNode.task_domain: "code"|"research"|"creative"|"data"|...  │
│   CoreToolManager.DOMAIN_TOOL_MAP → get_domain_tools()            │
├─────────────────────────────────────────────────────────────────┤
│ 语义去重 (已实现)                                                  │
│   TaskGraph.find_duplicate_node() Jaccard 字符集相似度             │
│   相似度 ≥ 0.8 时阻止创建重复子任务                                │
├─────────────────────────────────────────────────────────────────┤
│ CircuitBreaker (已有)                                             │
│   6 信号智能检测 → GREEN/YELLOW/RED 三态降级                       │
└─────────────────────────────────────────────────────────────────┘
```

关键改变: L4/L5 不再硬性终止任务, 而是通过检查点保存进度并自动延伸预算,
让 agent 有机会完成正在进行的工作。进度报告存入 `state["progress_reports"]` 列表,
供后续分析和用户查看。

### 10.6 具体实施方案

#### 方案 A: 统一编排器 + 类型感知工具集 (推荐)

**核心思想**: 不分"编程编排器"和"通用编排器", 用**一个统一的 Orchestrator** 处理所有类型的子任务, 通过**任务类型标签**动态选择工具集。

```python
# TaskNode 扩展
@dataclass
class TaskNode:
    # ... 现有字段 ...
    task_domain: str = "general"  # "coding" | "general" | "hybrid"

# Execute 阶段工具集动态选择
_GENERAL_TOOLS = {
    "task_mark_status", "task_get_detail", "task_view_overview",
    "recall_memory", "read_memory_node", "discover_related",
    "navigate_attention", "search_experience",
    "openclaw_search",  # web_search
    "ask_user",         # 新增
    "submit_final_answer",  # 新增
}

_CODING_TOOLS = _GENERAL_TOOLS | {
    "exec_write_file", "exec_run_command", "exec_read_file",
    # 或 IDE 远程工具 (如果已连接)
}

def get_execute_tools(task_domain: str, ide_connected: bool) -> set:
    if task_domain == "coding":
        tools = _CODING_TOOLS.copy()
        if ide_connected:
            tools |= {"read_file", "write_to_file", "execute_command", ...}
        return tools
    elif task_domain == "general":
        return _GENERAL_TOOLS.copy()
    else:  # hybrid
        return _CODING_TOOLS.copy()  # 给予完整能力
```

#### 方案 B: 递归深度追踪

```python
# 在 OrchestratorState 中新增
class OrchestratorState(TypedDict, total=False):
    # ... 现有字段 ...
    nesting_depth: int              # 当前编排器嵌套深度
    max_nesting_depth: int          # 最大允许嵌套深度 (默认 2)
    parent_orchestrator_id: str     # 父编排器 ID (用于链路追踪)

# 在 execute_node 中检查
def execute_node(state, engine):
    # ... 现有逻辑 ...
    current_depth = state.get("nesting_depth", 0)
    max_depth = state.get("max_nesting_depth", 2)
    
    if current_depth >= max_depth:
        # 禁止在 Execute 内部启动新的 Orchestrator
        # 但仍允许单层 FC 循环执行
        tool_defs = [t for t in exec_tools 
                     if t["function"]["name"] != "task_create_plan"]
```

#### 方案 C: 子任务语义指纹去重

```python
from difflib import SequenceMatcher

def is_duplicate_task(new_desc: str, parent_desc: str, ancestors: List[str]) -> bool:
    """检测子任务是否与父/祖先任务语义重复"""
    for ancestor_desc in [parent_desc] + ancestors:
        ratio = SequenceMatcher(None, new_desc.lower(), ancestor_desc.lower()).ratio()
        if ratio > 0.85:
            return True
    return False
```

### 10.7 推荐实施路径

| 阶段 | 任务 | 依赖 | 效果 |
|------|------|------|------|
| **Phase 1** | 实现 `submit_final_answer` + `ask_user` 内部工具 | 无 | 修复幽灵引用, Web 端获得交互能力 |
| **Phase 2** | TaskNode 新增 `task_domain` 字段 + Execute 工具集动态选择 | Phase 1 | 通用/编程子任务获得各自合适的工具 |
| **Phase 3** | OrchestratorState 新增 `nesting_depth` 追踪 | Phase 2 | 防止递归编排死循环 |
| **Phase 4** | 统一 FC 循环 (UnifiedFCRunner) | Phase 1-3 | 三条入口获得完整认知能力 |
| **Phase 5** | 集成 code-review-graph MCP 工具 | Phase 4 | 编程子任务获得结构化代码理解 |

---

## 11. 总结与下一步行动

### 11.1 核心结论

1. **web_search/web_fetch 已有**: `OpenClawSearchTool` 同时提供搜索和网页获取, 无需重复设计
2. **ask_user/submit_final_answer 缺失**: 是最紧迫的补全项, 5 个核心模块预期 `submit_final_answer` 存在但未实现
3. **code-review-graph 推荐集成**: MIT 许可 + SQLite 零基础设施 + MCP 原生, 是唯一安全可商用的代码智能工具
4. **GitNexus 许可证致命**: PolyForm Noncommercial 禁止商用, 功能最强但不可用
5. **循环封闭可防**: 通过"统一编排器 + 类型感知工具集 + 递归深度追踪 + 语义去重"四层防护解决
6. **祖龙认知基础设施无对标**: MemoryGraph/AttentionWindow/TaskGraph 的组合在市面所有框架中独一无二

### 11.2 技术决策总结

| 决策点 | 结论 | 理由 |
|--------|------|------|
| FC 循环统一 | IDEFCRunner 为基座 → UnifiedFCRunner | 拥有 6 项独有认知能力 |
| 通用 vs 编程任务 | 统一编排器 + 类型感知 | 避免两套编排器互相调用 |
| 循环防护 | 检查点+进度报告+自动继续 | 弹性预算 + CircuitBreaker + 语义去重 |
| 代码智能工具 | code-review-graph | MIT 许可 + 零基础设施 |
| 外部框架借鉴 | LangGraph 模式 + ReDel 深度控制 | 祖龙已部分实现 LangGraph 模式 |

---

*本文档由深度源码审查 + 市场调研 + 架构分析生成*
*覆盖 7 个核心模块 10,000+ 行代码 + 8 个开源框架 + 4 个代码智能工具 + 5 篇学术论文*
*最后更新: 2026-05-04*
