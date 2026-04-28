# MemoryGraph-FC 融合升级规划

> 日期：2026-04-21
> 状态：待审批

## Context

当前 Zulong 系统的 MemoryGraph 拥有 65+ 方法，但模型只能通过 2 个 FC 工具（recall_memory, recall_node_context）访问 3.8% 的能力。更严重的问题：

1. **赫布学习是死代码** — `compute_activations()` 计算了共激活边，但 `hebbian_strengthen()` 从未被调用
2. **思维深度不可见** — `get_focus_path_summary()` 不在模型上下文中，模型不知道自己在图上的位置
3. **记忆检索是手动的** — 模型必须调用 `recall_memory` 才能搜索记忆，系统不会自动注入相关上下文
4. **Orchestrator 持有瞬态** — ScratchPad/Checkpoint 等状态存在 Python 内存而非 MemoryGraph，挂起后丢失

本规划确立纯 PULL 记忆架构：系统只送达用户事件本身，不注入额外上下文；思维深度（GPS）作为系统提示词基线始终伴随；模型通过增强的 FC 工具自主搜索记忆/上下文。并逐步将 Orchestrator 状态迁移到 MemoryGraph。

---

## Phase 1: 修复缺陷 + 确立纯 PULL 记忆架构

> 目标：系统只送达用户事件本身，不注入任何额外上下文。模型通过 FC 工具自主搜索记忆/上下文。
> 设计理念：PUSH（系统注入）→ PULL（模型拉取）。与之前拆除 Gatekeeper 硬编码意图识别一脉相承——系统不替模型做判断。
> 风险：低（Phase 1.1 是单行修复，1.2 是增强 FC 工具能力，1.3 是 prompt 引导，无架构变更）

### 1.1 修复 hebbian_strengthen 死代码

**问题**：`compute_activations()` 填充 `_last_activated_edges`，但无代码调用 `hebbian_strengthen()`。赫布学习是死代码——共激活边永远不被强化。

**修改**：

| 文件 | 位置 | 变更 |
|---|---|---|
| `pipeline/attention_window.py` | `_compute_graph_activations()` (L685) | `compute_activations()` 调用后加 `self._memory_graph.hebbian_strengthen()` |
| `pipeline/orchestrator.py` | 焦点恢复 (L421) | `compute_activations()` 后加 `self._memory_graph.hebbian_strengthen()` |

**验证**：运行多轮任务，日志确认边权重随共激活增长

### 1.2 思维深度作为系统提示词基线（GPS）

**问题**：`get_focus_path_summary()` 只在工具返回值中出现，模型永远看不到自己的图上位置

**设计理念**：焦点路径是模型的位置感知（GPS），属于系统提示词基线，不是"注入的额外上下文"。模型始终知道自己在哪里，但记忆/上下文由模型自己拉取。

**修改**：

| 文件 | 变更 |
|---|---|
| `pipeline/agent_prompt.py` | `build_agent_system_prompt()` 中新增焦点路径段：调用 `mg.get_focus_path_summary()` 注入到系统提示词中（300字符上限） |
| `pipeline/attention_window.py` | 新增 `update_focus_path_message()` 方法：调用 `mg.get_focus_path_summary()`，更新/替换 `tool_name="__focus_path__"` 的 pinned system message |
| `pipeline/attention_window.py` | 在 `observe_tool_call()` 末尾调用 `update_focus_path_message()`（模式切换/焦点变化时自动更新 GPS） |
| `pipeline/orchestrator.py` | 在 `run()` 焦点恢复后（L432后）调用 `attn_window.update_focus_path_message()` 初始注入 |

**与"不注入额外东西"的关系**：GPS ≠ 注入记忆上下文。GPS 是模型对自身位置的基础感知（类似人类知道自己站在哪里），不是系统替模型搜集的信息。模型仍然需要自主调 `recall_memory` 搜索记忆。

**Token 开销**：~100 tokens/轮（300字符 pinned message，不可淘汰）

**验证**：日志确认每轮 LLM 调用的 messages 中始终包含【思维导航】

### 1.3 增强 recall_memory 工具（关键词+向量+混合）

**问题**：当前 `recall_memory` 只有纯关键词子串匹配，`max_results` 硬编码 8，不支持向量搜索。模型用这个工具去"搜集记忆"太弱了。

**修改**：

| 文件 | 变更 |
|---|---|
| `pipeline/tools.py` | `recall_memory` schema 新增 `mode` 参数：`"keyword"`(默认) / `"vector"` / `"hybrid"` |
| `pipeline/tools.py` | `_handle_recall_memory()` 中：mode="keyword" → `search_nodes()`; mode="vector" → `search_summaries()`; mode="hybrid" → `retrieve_context()` |
| `pipeline/tools.py` | `recall_memory` schema 新增 `top_k` 参数（默认 8，上限 20），放开结果数量限制 |
| `pipeline/tools.py` | `recall_memory` schema 新增 `session_id` 参数（可选，用于会话内 boost） |

**效果**：模型搜索记忆的能力从"关键词匹配最多8条"升级为"关键词+向量+混合检索最多20条"

**验证**：测试三种 mode 返回不同结果，确认 hybrid 包含向量搜索发现的冷记忆

### 1.4 系统提示词引导模型主动搜索

**问题**：如果系统不注入任何额外上下文，模型可能不知道自己"不知道什么"。需要在系统提示词中引导模型在关键时刻主动使用 FC 工具搜集上下文。

**修改**：

| 文件 | 变更 |
|---|---|
| `pipeline/agent_prompt.py` | AGENT_SYSTEM_PROMPT 工作规则中新增/修改：<br>• 收到 `[用户新消息]` 时，先判断是否需要上下文：涉及之前的工作/话题 → 先调 `recall_memory(mode="hybrid")` 再回应<br>• 任务开始时，先用 `view_focus_path` 确认当前思维深度位置<br>• 不确定是否需要搜索 → 宁可搜索（搜索成本低，遗漏上下文成本高） |

**设计要点**：不是硬编码"模型必须搜索"，而是提供软引导——模型可以自主判断是否需要。这和"系统不注入额外东西"不矛盾：引导在 system prompt 中，模型有权忽略；而自动注入是系统强制推送，模型无法拒绝。

**验证**：运行任务，观察模型是否在关键时刻主动调用 recall_memory 搜索记忆

### 1.5 （已移除）自动注入记忆上下文

~~原方案：系统在 3 个时机自动注入记忆上下文~~

**新方案**：系统不注入任何记忆上下文。模型通过 `recall_memory(mode="hybrid")` 等工具自主搜索。理由：
- 自动注入 = PUSH，与"模型自主判断"理念矛盾
- 注意力冲突问题自动消失（没有注入就没有挤占）
- 模型已有 `recall_memory`（PULL），增强后能力足够
- 如果模型判断不需要记忆，系统不该强制推送

### 1.6 （已移除）BFS 激活发现注入

~~原方案：自动注入 BFS 发现的节点~~

**新方案**：BFS 激活发现通过 FC 工具 `activate_and_retrieve`（Phase 2.2）由模型主动触发。`_compute_graph_activations()` 仍然运行（用于消息评分），但不自动注入内容。

### 1.7 （已移除）双模搜索自动注入

~~原方案：自动做二次检索~~

**新方案**：`recall_memory(mode="hybrid")` 已经是双模搜索（关键词+向量），模型主动调用时自然获得双模结果，不需要系统自动做二次查询。

---

## Phase 2: 扩展 FC 工具接口

> 目标：让模型能直接导航图、触发激活、管理记忆
> 风险：中（新增工具定义占用 context，需精简）

### 2.1 图导航工具

| 新 FC 工具 | 包装的 MG 方法 | 参数 | 返回 |
|---|---|---|---|
| `navigate_graph` | `update_focus_to_node` + `get_ancestors`/`get_children` | `address`(图地址字符串), `direction`(deeper/broader/jump) | 更新后的焦点路径 + 新焦点邻居 |
| `explore_neighbors` | `get_neighbors` | `node_id`, `edge_types`(可选过滤), `depth`(1-3) | 邻居列表含标签/类型/边元数据 |
| `view_focus_path` | `get_focus_path_summary` | 无 | 当前思维导航树（Phase 1.2 的手动查询版本） |

**文件**：`pipeline/tools.py`（handler + schema）, `pipeline/agent_prompt.py`（工具说明）

**AttentionWindow 联动**：`navigate_graph` 加入 `_FOCUS_TRIGGERS`，`view_focus_path` 加入 `_GLOBAL_FORCE_TRIGGERS`

### 2.2 激活与学习工具

| 新 FC 工具 | 包装的 MG 方法 | 参数 | 返回 |
|---|---|---|---|
| `activate_and_retrieve` | `compute_activations` + `hebbian_strengthen` | `seed_node_ids`(列表), `depth`(1-5), `min_activation` | top-10 激活节点含分数 |
| `strengthen_associations` | `hebbian_strengthen` | 无 | 强化边数量 |

**文件**：`pipeline/tools.py`, `pipeline/agent_prompt.py`

### 2.3 记忆管理工具

| 新 FC 工具 | 包装的 MG 方法 | 参数 | 返回 |
|---|---|---|---|
| `set_memory_importance` | `promote_importance` | `node_id`, `importance`(trivial/normal/important/must_remember) | 成功/失败 |
| `create_memory_node` | `add_node` | `label`, `content`, `node_type`(concept/knowledge/experience), `link_to_node_id`(可选) | 新节点 ID |
| `link_nodes` | `add_edge` | `source_id`, `target_id`, `edge_type`(reference/association/semantic) | 成功/失败 |

**文件**：`pipeline/tools.py`, `pipeline/agent_prompt.py`, `pipeline/guardrails.py`（新增校验规则）

**guardrails 规则**：
- `set_memory_importance` 禁止降级（只允许提升）
- `create_memory_node` 要求 content ≥ 20 字符
- `link_nodes` 防止重复边 + 验证两端节点存在

---

## Phase 3: Orchestrator 瘦身 — 状态迁移到 MemoryGraph

> 目标：Orchestrator 从"状态持有者"变为"薄执行器"，所有持久状态归 MemoryGraph
> 风险：高（架构重构，需逐步推进）

### 3.1 ScratchPad → MemoryGraph 节点

**问题**：ScratchPad 笔记存在 Python 内存，挂起后丢失

**修改**：`pipeline/attention.py`
- `ScratchPad.write()` 同时在 MG 创建 `scratch:{task_id}:{node_id}:{idx}` 节点（CONCEPT 类型, sub_type="scratch_note"）+ REFERENCE 边
- `ScratchPad.read()` 先查本地缓存，再查 MG
- 挂起/恢复时 ScratchPad 状态自动通过 MG 持久化

### 3.2 Checkpoint → MemoryGraph 元数据

**问题**：检查点单独写 JSON 文件，与图节点脱节

**修改**：`pipeline/orchestrator.py`
- `_write_checkpoint()` 在对话轮次节点的 metadata 中写入 checkpoint 信息
- 消息仍然写磁盘（太大不适合 MG），但索引在图中
- `_delete_checkpoint()` 同步清理 MG 元数据

### 3.3 Suspension → MemoryGraph 节点状态

**问题**：挂起状态由 TaskSuspensionManager 独立管理磁盘文件，与 MG 重复

**修改**：`zulong/l2/task_suspension.py`
- `suspend_task()` 在对话轮次节点设 status="suspended" + 存储元数据
- `resume_task()` 从 MG 查询挂起轮次，从关联文件路径加载消息
- `list_suspended_tasks()` 改为 MG 查询
- 旧格式文件添加一次性迁移函数

### 3.4 最终形态：薄 SessionRunner

**前置条件**：3.1 + 3.2 + 3.3 完成

**变更**：`pipeline/orchestrator.py`
- 提取状态管理方法到 `pipeline/session_state.py`（SessionStateManager）
- AgentOrchestrator 保留为向后兼容别名
- SessionRunner 只持有：vllm_client, model_id, memory_graph 引用
- TaskGraph 不再直接持有，通过 TaskGraphAdapter 从 MG 反向加载
- `run()` 精简到 ~300 行（仅 FC 循环 + LLM 调用 + 工具分发）

---

## 依赖关系

```
Phase 1.1 (hebbian fix)
    │
    ▼
Phase 1.2 (焦点路径 GPS) ──► Phase 2.1 (导航工具)
    │
    ▼
Phase 1.3 (增强 recall_memory)
    │
    ▼
Phase 1.4 (prompt 引导)

Phase 1.1 ──► Phase 2.2 (激活工具)

Phase 2.3 (记忆管理工具) — 独立

Phase 3.1 (ScratchPad→MG) — 独立
Phase 3.2 (Checkpoint→MG) — 独立
Phase 3.2 ──► Phase 3.3 (Suspension→MG)
Phase 3.1 + 3.2 + 3.3 ──► Phase 3.4 (薄 SessionRunner)
```

> 注：1.5/1.6/1.7（已移除的自动注入方案）不参与依赖

## Token 预算影响

| Phase | 每次开销 Token | 频率 | 机制 | 模式 |
|---|---|---|---|---|
| 1.2 (焦点路径 GPS) | ~100 | 每轮（pinned） | Pinned system message (300字符) | 系统→模型（基线） |
| 1.3 (recall_memory 增强) | 0 | 模型主动调用 | FC 工具返回值，不走注入 | 模型 PULL |
| 1.4 (prompt 引导) | ~50 | 每轮（system prompt） | 系统提示词中的软引导规则 | 系统→模型（基线） |
| 2.x (工具 schema) | ~800 | 每轮（FC定义） | FC 工具定义（每次 LLM 调用） | 声明式 |
| **Phase 1 合计** | **~150/轮** | 非累积 | 65536 窗口内有 ~58K 可用，余量充足 | — |

> **核心差异**：纯 PULL 模型下，系统不注入记忆上下文。Token 开销仅来自 GPS 基线（~100）和 prompt 引导（~50）。记忆检索的 Token 开销完全由模型自主决定——模型认为需要搜索时才调 FC 工具，返回值作为 tool message 进入上下文，受注意力窗口正常淘汰。

## 关键文件清单

| 文件 | 涉及 Phase |
|---|---|
| `pipeline/attention_window.py` | 1.2, 2.1 |
| `pipeline/orchestrator.py` | 1.1, 1.2, 3.2, 3.3, 3.4 |
| `pipeline/tools.py` | 1.3, 2.1, 2.2, 2.3 |
| `pipeline/agent_prompt.py` | 1.4, 2.1, 2.2, 2.3 |
| `pipeline/guardrails.py` | 2.3 |
| `pipeline/attention.py` | 3.1 |
| `zulong/l2/task_suspension.py` | 3.3 |
| `zulong/memory/memory_graph.py` | 参考（65+方法） |
| `zulong/l1b/scheduler_gatekeeper.py` | 3.4（调用方式变更） |

## 验证方案

### Phase 1 验证
1. 单元测试：`compute_activations()` + `hebbian_strengthen()` → 边权重增长
2. 集成测试：多轮 Agent 任务，日志确认每轮 context 含【思维导航】GPS
3. FC 工具测试：`recall_memory(mode="keyword")` / `recall_memory(mode="vector")` / `recall_memory(mode="hybrid")` 各返回不同结果
4. 行为测试：涉及之前会话的话题时，模型**主动调用** `recall_memory` 搜索记忆后再回应（而非系统注入）
5. 注意力测试：SINGLE_CHAIN 模式下，模型自主搜索返回的 tool message 受正常淘汰，不挤占当前任务上下文

### Phase 2 验证
1. 逐工具 FC dispatch 测试
2. 真实任务中模型自主使用新工具（`navigate_graph` 替代 `recall_memory` 盲搜）
3. `activate_and_retrieve` + `strengthen_associations` 联合测试确认边权重增长

### Phase 3 验证
1. ScratchPad 写入 → MG 节点存在 → 挂起 → 恢复 → 笔记仍在
2. 检查点 → MG 元数据存在 → 崩溃恢复 → 状态完整
3. 挂起 → MG 轮次节点 status="suspended" → 恢复 → 状态完整
4. 最终：进程中断 → 重启 → 完全从 MG 恢复（含 ScratchPad/Checkpoint/Focus）
