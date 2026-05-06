# 祖龙 x LangGraph 深度编排融合方案（修正版）

## Context

### 三份方案共识
千问、Trae、我的方案都同意需要：Plan-Execute-Reflect 三阶段、依赖感知调度、检查点持久化。
- **千问方案**风险最高（替换 FC 循环 = 丢失 7 层安全网）
- **Trae方案**改动量最大（接近重写）
- **我的方案**作为骨架最合理（渐进增强，内层不动），但有三个致命问题需修正

### 三个致命问题

**问题 1：模型无法操作依赖边 —— 依赖调度是空中楼阁**

TaskGraph 底层有 `add_d_edge()`、`remove_node()`、`get_node()` 方法，但**没有暴露为 FC 工具**。模型实际 CRUD 能力：

| 操作 | 底层方法 | FC 工具 | 状态 |
|------|---------|---------|------|
| 创建图 | `__init__` | `task_create_plan` | 有 |
| 添加节点 | `add_node()` | `task_add_node` | 有 |
| 标记状态 | `update_node_status()` | `task_mark_status` | 有 |
| 查看概览 | `to_focused_planning_table()` | `task_view_overview` | 有 |
| **添加依赖** | `add_d_edge()` | **没有** | 缺 |
| **读取节点详情** | `get_node()` | **没有** | 缺 |
| **删除节点** | `remove_node()` | **没有** | 缺 |
| **修改节点内容** | **底层也没有** | **没有** | 缺 |

**问题 2：暴力使用 128K 上下文 —— 超大型项目照样撑爆**

之前方案"注入所有依赖产出"是错误的。一个 50+ 子任务的项目，每个产出 2K token，全部注入 = 100K+ token。即使 128K 也不够。
正确做法：**用已有的 BFS 激活扩散 + AttentionWindow 评分驱逐机制智能选择**，而不是暴力全塞。

**问题 3：Synthesize 阶段不能一次总结所有产出**

50 个子任务的完整产出不可能一次放进上下文。需要**分级汇总**：
- 先按层级分组总结（Tier 0-5 各自总结）
- 再把各层总结汇合为最终报告
- 自然形成：任务标题 → 大纲摘要 → 子目录 → 详细内容的层级结构

---

## 修正后的完整方案

### 架构总览

```
用户输入
   │
   ▼
┌─ Round 1: 意图分类 ────────────────────────────────────┐
│  CHAT → 原路径（不走编排器）                             │
│  COMPLEX/RESUME → 进入编排器                            │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
╔═══════════════════════════════════════════════════════════╗
║  外层编排器 (orchestrator_graph.py)                       ║
║                                                           ║
║  ┌──────┐   ┌──────────┐   ┌─────────┐   ┌─────────┐   ║
║  │ PLAN │──▶│ SCHEDULE │──▶│ EXECUTE │──▶│ REFLECT │   ║
║  └──┬───┘   └────┬─────┘   └────┬────┘   └────┬────┘   ║
║     │            │              │              │         ║
║     │   纯代码逻辑    │    ┌───────────────┐    │         ║
║     │  (拓扑排序)     │    │ 内层FC循环     │    │         ║
║     │  (注意力切换)   │    │ (fc_graph.py)  │    │         ║
║     │               │    │ 7层安全网不动   │    │         ║
║     │               │    └───────────────┘    │         ║
║     │                                         │         ║
║     ◀──── REPLAN ────────────────────────────┘         ║
║                                                         ║
║  ┌───────────┐   全部完成时:                             ║
║  │SYNTHESIZE │←─ 分级汇总，不是一次全塞                  ║
║  └─────┬─────┘                                          ║
╚════════╪════════════════════════════════════════════════╝
         ▼
    最终回答

    全程数据流:
    TaskGraph  ←──双向──→  MemoryGraph
    (增删改查全部节点)      (BFS扩散 + 激活值)
         ↕                      ↕
    AttentionWindow ←── 评分驱逐 + 阶段感知
    (智能选择上下文，不暴力全塞)
```

---

## 阶段 1：补齐 TaskGraph CRUD 工具集（前置基础）

**问题**：模型连依赖边都建不了，谈什么编排？
**方案**：新增 4 个 FC 工具，让模型对任务图谱拥有完整的增删改查能力。

### 1.1 新增 `task_add_dependency` 工具

**文件**: `zulong/tools/task_tools.py`

```python
class TaskAddDependencyTool(BaseTool):
    """添加任务节点间的依赖关系"""
    name = "task_add_dependency"
    parameters = {
        "source_id": {"type": "string", "description": "前置任务节点 ID（先完成的）"},
        "target_id": {"type": "string", "description": "后续任务节点 ID（依赖前者的）"},
        "via": {"type": "string", "description": "依赖内容描述（可选）", "default": ""},
    }
    required = ["source_id", "target_id"]

    def execute(self, request):
        tg = get_active_task_graph()
        # 校验：两个节点都存在
        # 校验：不会形成环（调用 TaskScheduler.validate_dependencies）
        # 调用 tg.add_d_edge(source, target, via)
        # 同步到 MemoryGraph: 添加 DEPENDENCY 边
```

### 1.2 新增 `task_get_detail` 工具

**文件**: `zulong/tools/task_tools.py`

```python
class TaskGetDetailTool(BaseTool):
    """读取指定节点的完整详情（包括 result 产出内容）"""
    name = "task_get_detail"
    parameters = {
        "node_id": {"type": "string", "description": "节点 ID"},
    }
    required = ["node_id"]

    def execute(self, request):
        tg = get_active_task_graph()
        node = tg.get_node(node_id)
        # 返回: id, label, desc, status, result(完整内容),
        #       parent, children, dependencies, dependents
        # 让模型能读取任意节点的完整产出
```

### 1.3 新增 `task_update_node` 工具

**文件**: `zulong/tools/task_tools.py`

```python
class TaskUpdateNodeTool(BaseTool):
    """修改已有节点的标签、描述或产出内容"""
    name = "task_update_node"
    parameters = {
        "node_id": {"type": "string", "description": "节点 ID"},
        "label": {"type": "string", "description": "新标签（可选）"},
        "desc": {"type": "string", "description": "新描述（可选）"},
        "result": {"type": "string", "description": "新产出内容（可选）"},
    }
    required = ["node_id"]

    def execute(self, request):
        tg = get_active_task_graph()
        node = tg.get_node(node_id)
        # 修改 label/desc/result（只改传入的字段）
        # 同步到 MemoryGraph
        # 触发 on_change_callback（推送前端）
```

### 1.4 新增 `task_remove_node` 工具

**文件**: `zulong/tools/task_tools.py`

```python
class TaskRemoveNodeTool(BaseTool):
    """删除节点及其所有后代（不能删根节点）"""
    name = "task_remove_node"
    parameters = {
        "node_id": {"type": "string", "description": "要删除的节点 ID"},
    }
    required = ["node_id"]

    def execute(self, request):
        tg = get_active_task_graph()
        # 校验：不能删 req/analysis
        # 校验：如果有 completed 的依赖节点依赖于此节点，警告
        # 调用 tg.remove_node(node_id)
        # 同步到 MemoryGraph: 标记对应节点为 pruned
```

### 1.5 TaskGraph 底层补充 `update_node_content` 方法

**文件**: `zulong/l2/task_graph.py`

```python
def update_node_content(self, node_id: str, label: str = None,
                        desc: str = None, result: str = None) -> bool:
    """修改节点的标签、描述或产出内容"""
    node = self._nodes.get(node_id)
    if not node:
        return False
    if label is not None:
        node.label = label
    if desc is not None:
        node.desc = desc
    if result is not None:
        node.result = result
    self._sync_node_to_memory_graph(node_id, node.status, result)
    self._mark_dirty()
    return True
```

### 1.6 工具集分配

各阶段的工具集调整：

| 阶段 | 可用工具 |
|------|---------|
| **Plan** | task_add_node, task_add_dependency, task_remove_node, task_update_node, task_view_overview, task_get_detail, recall_memory, discover_related |
| **Execute** | task_mark_status, task_get_detail, task_update_node, task_view_overview, exec_write_file, exec_run_command, recall_memory, navigate_attention |
| **Reflect** | task_view_overview, task_get_detail（轻量 LLM 调用，不用 FC 循环） |

**关键原则**：模型在 Plan 阶段可以自由增删改查任何节点，在 Execute 阶段可以读取任意节点详情并修改当前执行节点的内容。

---

## 阶段 2：智能上下文管理（不暴力使用 128K）

**问题**：即使 128K 也不能暴力全塞。50 个子任务 × 2K 产出 = 100K token。
**方案**：用**已有的 BFS 扩散 + AttentionWindow 评分驱逐**智能选择，而不是手动注入。

### 2.1 核心原则：用 AttentionWindow 管理，不绕过它

当前 AttentionWindow 的评分公式：
```
score = base × time_decay(0.95^age) × mode_mult × memory_boost(BFS activation)
```

它**已经**能做到：
- 高激活值节点的消息权重更高（BFS 扩散决定）
- 低权重消息被驱逐，生成摘要替代
- 按组贪心选择，不超预算

**问题不是 AttentionWindow 不够好，而是依赖产出没有作为消息注册进去。**

### 2.2 依赖产出注册为 AttentionWindow 消息

**文件**: `zulong/l2/orchestrator_graph.py`（execute_node 中）

Execute 阶段开始前，不是暴力注入所有依赖产出，而是：

```python
def execute_node(state):
    subtask_id = state["current_subtask_id"]
    tg = get_active_task_graph()

    # 1. 获取直接依赖节点的产出
    dep_ids = tg.get_dependencies(subtask_id)
    for dep_id in dep_ids:
        dep_node = tg.get_node(dep_id)
        if dep_node.status == "completed" and dep_node.result:
            # 作为系统消息注册到 AttentionWindow
            dep_msg = {
                "role": "system",
                "content": f"[依赖产出 {dep_node.label}] {dep_node.result}"
            }
            messages.append(dep_msg)
            # 注册时标记 node_id，让 BFS 激活值影响其权重
            engine._attn_window.register_message(
                dep_msg,
                turn=current_turn,
                node_id=f"task:{tg.id}/{dep_id}",  # 关联 MemoryGraph 节点
                is_pinned=False,  # 不钉死！让 AttentionWindow 决定保留
            )

    # 2. BFS 扩散自动设置激活值
    #    当前节点 in_progress → activation=0.9
    #    直接依赖（已完成）→ BFS 1-hop → activation ≈ 0.45
    #    间接依赖 → BFS 2-hop → activation ≈ 0.225
    #    无关节点 → activation ≈ 0 → 消息被驱逐

    # 3. AttentionWindow.apply_window() 自动处理
    #    → 高激活的依赖产出保留（score 被 memory_boost 提升）
    #    → 低激活的无关消息驱逐（生成摘要替代）
    #    → 永远不超预算！
```

### 2.3 上下文窗口配置

**文件**: `config/zulong_config.yaml` + `zulong/l2/inference_engine.py`

```yaml
llm:
  ollama:
    num_ctx: 131072     # 128K
```

但**不改 reserved_tokens 的计算逻辑**——AttentionWindow 已经有 `(context - reserved) × 90%` 的安全余量。128K 下预算 ≈ 107K token，AttentionWindow 的贪心驱逐算法保证永远不超。

超大项目时：
- 50 个依赖产出全部注册为消息
- BFS 激活扩散给当前节点 + 直接依赖高激活值
- AttentionWindow 只保留高权重的（可能 5-10 个最相关的）
- 其余 40 个被驱逐，生成简短摘要
- **结果**：模型看到最相关的依赖产出全文 + 其余的一句话摘要

### 2.4 阶段感知注意力（增强而非替代）

**文件**: `zulong/l2/attention_window.py`

新增 `set_phase()` 方法，只调整 **mode** 和 **mode_multiplier**，不绕过评分驱逐：

```python
def set_phase(self, phase: str, subtask_id: str = None):
    """编排器阶段切换时调整注意力模式"""
    if phase == "plan":
        self.mode = AttentionMode.GLOBAL
        # GLOBAL 模式下概览消息权重高，适合规划
    elif phase == "execute":
        self.mode = AttentionMode.FOCUS
        self._current_node_id = subtask_id
        # FOCUS 模式下当前节点 3.0x，依赖 2.0x，其他 0.3x
        # → 依赖产出被 mode_mult 提升 + BFS activation 提升 = 双重保障
    elif phase == "reflect":
        self.mode = AttentionMode.GLOBAL
        # 回到全局视野评估质量
```

**关键**：不新增 `_phase_multipliers` 字典。`set_phase` 只是切换已有的三种 AttentionMode。FOCUS 模式已有的 `_mode_multiplier()` 逻辑（当前节点 3.0x, 祖先 2.0x, 依赖 2.0x, 其他 0.5x）完全够用。不叠加不过度设计。

---

## 阶段 3：分级汇总（Synthesize 不一次全塞）

**问题**：50 个子任务的完整产出不可能一次放进上下文。
**方案**：分层级递归汇总，自然形成 任务→大纲→子目录→详细内容 结构。

### 3.1 分级汇总策略

```
超大项目 (50+ 子任务) 的合成流程:

Step 1: 按拓扑层级分组
  Tier 0: [o1]           → 需求分析
  Tier 1: [o2]           → 架构设计
  Tier 2: [o3, o6]       → 核心功能 + 存储层
  Tier 3: [o4, o5]       → 排序 + 提醒
  Tier 4: [o7]           → CLI集成
  Tier 5: [o8]           → 测试

Step 2: 每个 Tier 独立汇总（一次 LLM 调用，只放该层的产出）
  Tier 0 摘要: "完成需求分析，确定了4大模块..."
  Tier 2 摘要: "核心CRUD和JSON存储已实现..."
  ...

Step 3: 汇总所有 Tier 摘要 → 最终报告
  输入: 5-6 个 Tier 摘要（每个 ~200 token）
  输出: 结构化最终回复

自然形成的层级结构:
  L0: 任务标题 "待办事项管理工具"
  L1: 大纲摘要 "需求分析→架构设计→功能实现→测试"
  L2: Tier 摘要 "核心功能：CRUD + 排序 + 提醒 + 持久化"
  L3: 节点 result "add_todo() 实现了...，包含验证..."
```

### 3.2 小项目（≤10 子任务）直接汇总

不需要分级，一次 LLM 调用 + AttentionWindow 自然保留所有产出即可。

### 3.3 实现

**文件**: `zulong/l2/orchestrator_graph.py`（synthesize_node 中）

```python
def synthesize_node(state):
    tg = get_active_task_graph()
    leaves = tg.get_leaf_nodes()

    if len(leaves) <= 10:
        # 小项目：直接调用 run_fc_loop 汇总
        # AttentionWindow 有足够预算保留所有产出
        return run_fc_loop_for_synthesis(...)

    else:
        # 大项目：分级汇总
        scheduler = TaskScheduler(tg)
        tiers = scheduler.compute_execution_tiers()

        tier_summaries = []
        for tier in tiers:
            # 每个 tier 独立 LLM 汇总
            tier_nodes = [tg.get_node(nid) for nid in tier]
            tier_content = "\n".join(
                f"- {n.label}: {n.result[:300]}" for n in tier_nodes
                if n.status == "completed"
            )
            summary = call_llm_for_summary(
                f"请用 2-3 句话总结以下工作成果:\n{tier_content}"
            )
            tier_summaries.append(summary)

        # 汇总所有 tier 摘要
        final_input = "\n".join(
            f"阶段 {i+1}: {s}" for i, s in enumerate(tier_summaries)
        )
        final_response = call_llm_for_synthesis(
            f"用户需求: {state['user_input_text']}\n\n"
            f"各阶段成果:\n{final_input}\n\n"
            f"请生成完整的最终报告。"
        )
        return {"response": final_response}
```

---

## 阶段 4：TaskScheduler + 编排器

### 4.1 TaskScheduler（拓扑排序）

**文件**: `zulong/l2/task_graph.py`（追加 ~80 行）

```python
class TaskScheduler:
    def __init__(self, task_graph: TaskGraph):
        self.tg = task_graph

    def compute_execution_tiers(self) -> List[List[str]]:
        """Kahn 算法，返回 [[tier0_ids], [tier1_ids], ...]"""
        # 只处理叶节点，忽略中间节点
        # 按 _d_edges 构建入度表
        # 返回分层结果

    def get_next_executable(self) -> List[str]:
        """当前所有前置依赖已完成的 pending/blocked 叶节点"""
        leaves = self.tg.get_leaf_nodes()
        result = []
        for node in leaves:
            if node.status not in ("pending", "blocked"):
                continue
            deps = self.tg.get_dependencies(node.id)
            all_met = all(
                self.tg.get_node(d).status in ("completed", "skipped")
                for d in deps
            )
            if all_met:
                result.append(node.id)
        return result

    def validate_dependencies(self) -> Tuple[bool, str]:
        """DFS 检测依赖图是否有环"""
```

### 4.2 编排器 StateGraph

**文件**: `zulong/l2/orchestrator_graph.py`（新建 ~350 行）

```python
class OrchestratorState(TypedDict, total=False):
    phase: str                          # plan/schedule/execute/reflect/synthesize
    plan_version: int
    replan_count: int
    current_subtask_id: Optional[str]
    completed_results: Dict[str, str]   # node_id → result 摘要（供反思用）
    messages: List[Dict]
    vllm_model_id: str
    tool_definitions: List[Dict]
    user_input_text: str
    total_fc_turns: int
    max_total_fc_turns: int             # 默认 100
    subtask_reflection_count: Dict[str, int]  # node_id → 该节点反思次数
    should_terminate: str
```

5 个节点：

| 节点 | 调用 LLM？ | 核心行为 |
|------|-----------|---------|
| **plan** | 是（run_fc_loop） | 工具集=规划工具，只建图不执行 |
| **schedule** | 否（纯代码） | 拓扑排序找可执行节点 + 注意力切换 + 保存检查点 |
| **execute** | 是（run_fc_loop） | 单子任务，依赖产出注册到 AttentionWindow（不暴力注入） |
| **reflect** | 是（轻量 LLM） | 质量评估 → CONTINUE/REDO/REPLAN |
| **synthesize** | 是 | 小项目直接汇总，大项目分级汇总 |

### 4.3 Reflect 决策逻辑

```
Reflect 输入：当前子任务的 result + 整体进度概览

决策:
  CONTINUE → schedule（下一个节点）
  REDO     → execute（同一节点，注入"上次问题"提示）
             限制: 每个节点最多 redo 3 次
  REPLAN   → plan（保留已完成节点，修改/增删 pending 节点）
             限制: 最多 replan 2 次

安全兜底:
  redo 达上限 → 强制 CONTINUE
  replan 达上限 → 强制 CONTINUE
  total_fc_turns ≥ 100 → 强制 synthesize
```

### 4.4 集成到 inference_engine.py

**文件**: `zulong/l2/inference_engine.py`（~20 行改动）

```python
# _run_round2 方法中:
if intent_type.value in ("complex", "resume") and orch_config.get("enabled", False):
    response, fc_turn = run_orchestrator(engine=self, ...)
else:
    response, fc_turn = run_fc_loop(engine=self, ...)  # 原路径不动
```

### 4.5 配置

**文件**: `config/zulong_config.yaml`

```yaml
llm:
  ollama:
    num_ctx: 131072                     # 128K

l2_inference:
  orchestrator:
    enabled: true                       # 总开关（false 退回扁平循环）
    max_replan_count: 2
    max_redo_per_subtask: 3
    subtask_fc_budget: 30
    reflection_max_tokens: 512
    large_project_threshold: 10         # >10 子任务使用分级汇总
```

---

## 阶段 5：检查点持久化

**文件**: `zulong/l2/checkpoint_manager.py`（新建 ~150 行）

在每个阶段转换时保存快照：
- Plan 完成后 → plan_v{N}
- 每个 Execute 完成后 → exec_{node_id}
- 每个 Reflect 后 → reflect_{node_id}

RESUME 场景：从最近检查点恢复，跳过已完成节点，直接进入 schedule。

---

## 关键文件清单

| 文件 | 操作 | 改动量 |
|------|------|--------|
| `zulong/tools/task_tools.py` | **追加** 4 个 FC 工具 | ~250 行 |
| `zulong/l2/task_graph.py` | **追加** TaskScheduler + update_node_content | ~120 行 |
| `zulong/l2/orchestrator_graph.py` | **新建** 编排器 StateGraph | ~350 行 |
| `zulong/l2/checkpoint_manager.py` | **新建** 检查点管理 | ~150 行 |
| `zulong/l2/inference_engine.py` | **修改** 路由 + 上下文默认值 | ~25 行 |
| `zulong/l2/attention_window.py` | **修改** set_phase() 方法 | ~20 行 |
| `zulong/l2/intent_prompt_builder.py` | **修改** 各阶段提示词 + 工具过滤 | ~80 行 |
| `config/zulong_config.yaml` | **修改** 128K + 编排器配置 | ~15 行 |
| `zulong/l2/fc_graph.py` | **不改** | 0 |
| `zulong/memory/memory_graph.py` | **不改** | 0 |
| `zulong/l2/circuit_breaker.py` | **不改** | 0 |
| `zulong/memory/graph_adapters.py` | **不改** | 0 |

---

## 执行流程图（修正后）

以 "Python 命令行待办工具" 为例，对比修正前后：

```
修正前（暴力注入）:                     修正后（智能选择）:

Execute o7 时:                          Execute o7 时:
┌──────────────────────┐               ┌──────────────────────┐
│ 注入 o1 产出（全文）  │               │ 注册 o1 产出为消息    │
│ 注入 o2 产出（全文）  │               │ 注册 o2 产出为消息    │
│ 注入 o3 产出（全文）  │               │ 注册 o3 产出为消息    │
│ 注入 o4 产出（全文）  │               │ 注册 o4 产出为消息    │
│ 注入 o5 产出（全文）  │               │ 注册 o5 产出为消息    │
│ 注入 o6 产出（全文）  │               │ 注册 o6 产出为消息    │
│                      │               │                      │
│ 总消耗: ~12K token   │               │ BFS 扩散:            │
│ 50个子任务 = 100K!   │               │  o7(当前)=0.9        │
│ 128K 都不够！        │               │  o3(直接依赖)=0.45   │
│                      │               │  o4(直接依赖)=0.45   │
│ 而且无关的早期对话    │               │  o5(直接依赖)=0.45   │
│ 也占着空间           │               │  o6(直接依赖)=0.45   │
└──────────────────────┘               │  o2(间接依赖)=0.225  │
                                       │  o1(间接依赖)=0.1125 │
                                       │                      │
                                       │ AttentionWindow:     │
                                       │  o3,o4,o5,o6: 高权重 │
                                       │  → 保留全文           │
                                       │  o2: 中等权重         │
                                       │  → 保留或摘要替代     │
                                       │  o1: 低权重           │
                                       │  → 驱逐为一句话摘要   │
                                       │  无关消息: 驱逐       │
                                       │                      │
                                       │ 永远不超预算！        │
                                       └──────────────────────┘
```

Synthesize 阶段对比：

```
修正前（一次全塞）:                     修正后（分级汇总）:

50个子任务的完整产出                    Step 1: 按 Tier 分组总结
→ 一次放进上下文                        Tier 0 → 摘要 "需求分析..."
→ 100K token                           Tier 1 → 摘要 "架构设计..."
→ 撑爆！                               Tier 2 → 摘要 "核心功能..."
                                       ...

                                       Step 2: 汇总 Tier 摘要
                                       → 5-6 个摘要(~1K token) → 最终报告
                                       → 绝不超预算

                                       自然形成层级:
                                       L0: 任务标题
                                       L1: 大纲(Tier摘要)
                                       L2: 子目录(节点label)
                                       L3: 详细内容(节点result)
                                       → 每层都保存在TaskGraph中，可追溯可修改
```

---

## 验证计划

1. **新工具单元测试**：task_add_dependency 创建依赖边 → validate_dependencies 无环 → get_next_executable 返回正确节点
2. **CRUD 完整性**：模型可以通过工具 增/删/改/查 任意节点
3. **智能上下文**：构造 20+ 子任务场景，验证 AttentionWindow 不超预算，高激活依赖保留全文
4. **分级汇总**：构造 15+ 子任务场景，验证 synthesize 分 Tier 汇总
5. **回退开关**：`orchestrator.enabled: false` 退回扁平循环
6. **编译验证**：所有修改/新建文件 py_compile
