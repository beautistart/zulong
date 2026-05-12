# 任务编排方案对比与LangGraph集成分析

> 评估日期：2026-04-25
> 基于：zulong_beta4 完整源码分析 + 市场竞品调研

---

## 一、市场复杂任务编排第一梯队

| 排名 | 产品 | 定位 | 核心优势 | 适合祖龙？ |
|------|------|------|---------|-----------|
| **1** | **LangGraph** | AI Agent图编排 | 原生DAG、检查点恢复、动态路由、time-travel调试 | **最适合** |
| **2** | **BehaviorTree.CPP** | 机器人实时编排 | C++高性能、ROS2原生、Tick中断恢复 | 适合L0-L1 |
| **3** | **Temporal** | 分布式持久工作流 | 精确恢复、事件历史、Signal动态修改 | 适合后端 |
| **4** | **HTN规划器** | 层级任务分解 | 天然层级DAG、自动目标分解 | 理念接近 |
| **5** | **AutoGen** | 多Agent对话编排 | 丰富编排模式、40k+ stars | 部分适合 |

**结论：LangGraph 是当前做复杂AI Agent任务编排综合能力最强的开源产品。**

---

## 二、综合对比矩阵

| 框架 | DAG/图结构 | 中断恢复 | 动态重规划 | 状态快照 | 嵌入机器人 | 开源 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LangGraph** | 原生有向图 | 检查点恢复 | 条件路由 | 持久化检查点 | 可嵌入高层 | MIT |
| **Temporal** | 代码定义 | 精确恢复 | Signal/Query | 事件历史 | 需服务端 | MIT |
| **BehaviorTree.CPP** | 树形结构 | Tick机制 | Fallback/Decorator | Blackboard | **原生设计** | MIT |
| **HTN** | 层级分解 | 重新规划 | 失败重分解 | World State | 可嵌入 | 多种 |
| **AutoGen** | 对话驱动 | 有限 | 自适应选择 | 事件驱动 | Token消耗高 | MIT |
| **CrewAI** | 顺序/并行 | 无原生持久化 | 层级委托 | 无检查点 | 不推荐 | MIT |
| **Airflow** | 原生DAG | 任务级重试 | **静态DAG** | 任务状态 | 不适合 | Apache |
| **Prefect** | 隐式DAG | 缓存+重试 | 动态子流 | Result Persistence | 不适合 | Apache |

---

## 三、祖龙当前任务编排的三层拆解

祖龙的"任务编排"包含三层不同层次的东西，不能笼统地说"替换"：

### 第一层：任务数据结构 — TaskGraph (734行)
- 无限深度递归树 + 动态拆解
- 层级边(h_edges) + 依赖边(d_edges) + 并行组(parallel_groups)
- 与MemoryGraph深度绑定（通过TaskGraphAdapter双向同步）
- 7种节点状态 + 非叶子节点状态聚合规则
- 深度自动映射（depth→type: requirement→analysis→outline→task→subtask）

### 第二层：编排引擎 — FC Loop / InferenceEngine (3221行)
- 两阶段意图分类（Round 1: start_session强制分类 → Round 2: 场景化FC循环）
- tool_choice="auto" 自循环最多100轮
- Rule A（过早完成拦截）/ Rule B（用户原话保留）/ Rule C（任务挂起一致性）
- 注意力窗口管理（AttentionWindowManager）

### 第三层：规划算法 — MCTS Planner (349行，原型)
- 蒙特卡洛树搜索探索最优规划路径
- 未生产化，与TaskGraph集成松散

---

## 四、逐层替代性评估

### 第一层 TaskGraph — **保留自研**

不建议替换的原因：
1. 与MemoryGraph深度耦合 — 通过TaskGraphAdapter将任务节点双向同步到图记忆系统，是祖龙最核心的差异化能力
2. 深度自动映射是独有设计 — 根据树深度自动决定节点类型，通用编排框架中不存在
3. 代码量可控 — 734行，结构清晰，维护成本低

### 第二层 FC Loop编排引擎 — **建议用LangGraph替换**

| 维度 | 祖龙FC Loop | LangGraph |
|------|------------|-----------|
| DAG图结构 | 无（纯循环） | 原生有向图，支持环 |
| 状态检查点 | 无（仅任务级序列化） | **内置Checkpointing** |
| 中断恢复 | 手写中断标志 | **原生human-in-the-loop暂停** |
| 动态路由 | tool_choice="auto"（黑盒） | **条件边，显式可控** |
| 调试 | 日志 | **time-travel调试** |
| 流式输出 | 手动实现 | **per-node streaming** |
| 社区 | 单人维护 | 10k+ stars，LangChain全生态 |
| 代码量 | 3221行（含大量手写容错） | 框架处理，业务代码大幅减少 |

### 第三层 MCTS规划 — **可选替换为HTN**

HTN的层级分解更匹配TaskGraph的树形结构，pyhop实现仅~300行Python。

---

## 五、推荐架构方案

```
+--------------------------------------------------+
|           LangGraph StateGraph（编排引擎）          |  <-- 替换FC Loop
|                                                    |
|  classify_intent → fc_loop <-> rule_guardian      |
|       |                         |                  |
|  interrupt_check          memory_sync → END        |
|                                                    |
|  [内置] 检查点 / 中断恢复 / 流式输出 / 调试        |
+--------------------------------------------------+
          |                    |
          v                    v
+------------------+  +------------------+
|   TaskGraph      |  |   MemoryGraph    |  <-- 保留自研
|   (734行，自研)   |  |   (图记忆系统)    |
|   无限深度递归树   |  |   Hebbian强化    |
|   动态拆解        |  |   BFS激活        |
+------------------+  +------------------+
          |
          v
+------------------+
|   HTN / MCTS     |  <-- 可选替换
|   (规划算法)      |
+------------------+
```

分工原则：
- **LangGraph** 管"怎么执行"（编排引擎、状态管理、检查点、中断恢复）
- **TaskGraph** 管"任务长什么样"（数据结构、与记忆图绑定）
- **规划算法** 管"怎么拆任务"（HTN分解 / MCTS搜索）

---

## 六、LangGraph集成示例

```python
from langgraph.graph import StateGraph, END

class ZulongState(TypedDict):
    messages: list
    task_graph: TaskGraph        # 保留自研TaskGraph
    intent: str                  # CHAT / COMPLEX / RESUME
    fc_turn: int
    memory_context: dict         # 从MemoryGraph检索的上下文

graph = StateGraph(ZulongState)

# 节点 = 祖龙的各个处理阶段
graph.add_node("classify_intent", classify_intent_node)
graph.add_node("fc_loop", fc_loop_node)
graph.add_node("rule_guardian", rule_guardian_node)
graph.add_node("memory_sync", memory_sync_node)
graph.add_node("interrupt_check", interrupt_check_node)

# 边 = 显式控制流
graph.add_edge("classify_intent", "fc_loop")
graph.add_conditional_edges("fc_loop", route_after_fc, {
    "tool_call": "fc_loop",
    "guardian_check": "rule_guardian",
    "interrupt": "interrupt_check",
    "done": "memory_sync",
})
graph.add_conditional_edges("rule_guardian", route_after_guard, {
    "retry": "fc_loop",
    "pass": "memory_sync",
})
graph.add_edge("memory_sync", END)

app = graph.compile(checkpointer=SqliteSaver.from_conn_string("zulong.db"))
```

---

## 七、实施步骤

1. 安装LangGraph — `pip install langgraph`，零侵入
2. 定义ZulongState — 将现有state.py中的状态字段迁移
3. 将FC Loop拆为图节点 — classify_intent / fc_step / rule_guardian / memory_sync
4. 用条件边替代循环控制 — MAX_NO_TOOL_TURNS变成显式边
5. 接入Checkpointer — 替代手写的task_suspend序列化
6. 保留TaskGraph和MemoryGraph不动 — 作为State中的自定义字段

预估改动：3221行中约1500-2000行可被LangGraph替代，剩余1200行左右业务逻辑需保留并适配为LangGraph节点。

---

## 八、总结

| 层次 | 现状 | 建议 | 理由 |
|------|------|------|------|
| TaskGraph数据结构 | 734行，成熟 | **保留** | 与MemoryGraph深度耦合，是核心差异化 |
| FC Loop编排引擎 | 3221行，手写容错重 | **替换为LangGraph** | 省约60%代码，自动获得检查点/中断恢复/调试 |
| MCTS规划算法 | 349行，原型 | **可选替换为HTN** | HTN层级分解更匹配树形TaskGraph |
| 注意力/记忆系统 | 独创 | **保留** | 市场无竞品，不可替代 |

**编排引擎不用重复造轮子，用LangGraph；TaskGraph和记忆系统是祖龙的核心壁垒，必须自研。**
