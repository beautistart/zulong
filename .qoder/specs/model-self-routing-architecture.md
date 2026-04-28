# 架构改造：模型自主路由 + 任务图谱作为工具

## Context

当前祖龙系统有一个硬编码的复杂度分类器（`pipeline/complexity_classifier.py`），用 5 个信号加权评分，
超过 0.4 则走 AgentOrchestrator（独立线程，600s 超时），否则走 L2 InferenceEngine（Function Calling 循环）。

这导致两个问题：
1. 规则分类不准确（写作任务被误判为复杂、真正的复杂任务可能被漏判）
2. 两条独立推理路径难以维护，且 AgentOrchestrator 的 600s 超时远超测试的 240s 等待

用户要求：移除硬编码分类器，让模型自主决定何时使用任务图谱（操作手册），
就像一个人听到闲聊就搭话、遇到复杂任务就拿出操作手册照做。

**目标**: 统一推理路径，让 TaskGraph 成为模型可以自愿调用的 FC 工具。

---

## 架构总览

```
改造前:                              改造后:
                                     
L1-B Gatekeeper                      L1-B Gatekeeper
  ├── 复杂度分类器                       │ (无分类器，可选Hint注入)
  │   ├── 简单 → L2 推理                 ▼
  │   └── 复杂 → AgentOrchestrator    L2 InferenceEngine (统一)
  │              (独立线程 600s)          ├── HOT 工具: 搜索/记忆/start_task_plan
  ▼                                      ├── 模型调用 start_task_plan 时:
L2 InferenceEngine                       │   ├── 动态注入 planning 工具
  (Function Calling 循环)                │   ├── 扩展时间预算 → 300s
  (45s 工具预算)                         │   └── 扩展迭代上限 → 200
                                         └── 模型不调用时: 正常 FC 循环 (45s)
```

---

## 实施步骤

### Step 1: 创建 `zulong/tools/planning_tools.py` (新文件)

核心新模块，将 AgentOrchestrator 的能力封装为标准 FC 工具。

**1.1 PlanningSession 上下文类**

```python
class PlanningSession:
    """规划会话状态（挂在 InferenceEngine 上，跨迭代持久化）"""
    task_graph: TaskGraph          # 任务图谱实例
    workspace_dir: str             # 工作目录
    planning_mode: bool = True     # 激活标志
    final_answer: Optional[str]    # 最终答案
```

**1.2 Meta 工具: `start_task_plan`**

- 注册为 HOT 工具（始终对模型可见）
- Schema: `start_task_plan(goal: str)` 
- 执行时:
  1. 创建 TaskGraph（包含 "req" + "analysis" 种子节点）
  2. 创建 workspace_dir
  3. 将 PlanningSession 挂到 InferenceEngine
  4. 动态注入 planning 工具到当前 FC 循环（复用现有 `_pending_tool_schemas` 机制）
  5. 发布 TaskGraph 初始化事件到前端
  6. 返回 "规划模式已激活。已创建任务图谱，请开始分析需求并添加任务节点。"

**1.3 Planning 工具集（Cold，按需加载）**

| 工具名 | 来源 | 描述 |
|--------|------|------|
| `plan_add_node` | 来自 task_graph_pack.AddNodeTool | 添加任务节点 |
| `plan_mark_status` | 来自 task_graph_pack.UpdateNodeStatusTool | 更新节点状态 |
| `plan_add_dependency` | 新增，来自 AgentToolRegistry | 添加依赖边 |
| `view_task_overview` | 来自 AgentToolRegistry._handle_view_overview | 查看图谱概览 |
| `exec_write_file` | 来自 AgentToolRegistry._handle_exec_write_file | 写文件 |
| `submit_final_answer` | 改造现有工具 | 提交最终答案，结束规划 |

每个工具的 `execute()` 方法中发布 EventBus 事件（与旧 orchestrator 格式一致），前端零修改。

**1.4 submit_final_answer 特殊处理**

调用时:
1. 将 `final_answer` 存到 PlanningSession
2. 发布 `pipeline_done` 事件
3. 清除 planning 工具的 hot 提升
4. 标记 `planning_mode = False`
5. InferenceEngine 的 FC 循环检测到 final_answer 后退出循环

---

### Step 2: 修改 `zulong/l2/inference_engine.py`

**2.1 添加 PlanningSession 属性**

```python
# __init__ 中新增:
self._planning_session: Optional[PlanningSession] = None
```

**2.2 修改 `_generate_with_vllm_and_tools` 方法**

在 FC 循环中增加三处适配:

a) **时间预算动态调整** (line ~548):
```python
# 原来: _TOOL_TIME_BUDGET = 45
# 改为:
_base_budget = 45
if self._planning_session and self._planning_session.planning_mode:
    _TOOL_TIME_BUDGET = 300  # 规划模式扩展到 5 分钟
else:
    _TOOL_TIME_BUDGET = _base_budget
```

b) **CircuitBreaker 上限动态调整**:
```python
if self._planning_session and self._planning_session.planning_mode:
    self.circuit_breaker.safety_hard_cap = 200
else:
    self.circuit_breaker.safety_hard_cap = 50
```

c) **final_answer 检测退出** (在工具执行后):
```python
if self._planning_session and self._planning_session.final_answer:
    return self._planning_session.final_answer, links_text_global
```

**2.3 注册 start_task_plan 为 HOT 工具**

在 `__init__` 末尾:
```python
from zulong.tools.planning_tools import StartTaskPlanTool
_plan_tool = StartTaskPlanTool(engine=self)
self.tool_engine.registry.register(_plan_tool)
# 注册为 HOT 工具
if hasattr(self, 'core_tool_manager') and self.core_tool_manager:
    self.core_tool_manager.register_tool(
        tool_name="start_task_plan",
        schema=_plan_tool.get_function_schema(),
        source="builtin",
        force_hot=True,
    )
```

**2.4 移除旧导入和警告**

- 移除 `from pipeline.complexity_classifier import classify as classify_complexity` (line 28)
- 移除 `from pipeline.orchestrator import PipelineOrchestrator` (line 29)
- 移除 `_on_l2_command` 中的复杂度检测警告代码 (lines 1421-1430)

---

### Step 3: 修改 `zulong/l1b/scheduler_gatekeeper.py`

**3.1 移除复杂度路由门** (lines 2284-2288)

```python
# 删除这段:
# if self._is_complex_task(text):
#     logger.info(f"... 检测到复杂任务，启动 AgentOrchestrator")
#     self._start_agent_orchestrator(text, priority, packaged_task)
#     return
```

所有消息统一走 `event_bus.publish(l2_event)` 路径。

**3.2 添加可选 Hint 注入** (替代移除的分类器位置)

```python
# 轻量级 Hint 注入（非硬路由，模型可忽略）
if self._hint_enabled:
    hint = self._generate_task_hint(text)
    if hint:
        packaged_task["task_hint"] = hint
```

**3.3 Hint 生成器（简化版）**

```python
def _generate_task_hint(self, text: str) -> Optional[str]:
    """生成轻量级任务提示（非路由，仅供模型参考）"""
    # 非常简单的信号检测（不做评分，只做存在检测）
    multi_step_signals = ["步骤", "首先", "然后", "实现", "开发", "搭建", "设计"]
    hit = sum(1 for kw in multi_step_signals if kw in text)
    if hit >= 2 and len(text) > 50:
        return "[系统提示: 此请求可能涉及多步骤任务，你可以考虑使用 start_task_plan 工具]"
    return None
```

**3.4 Hint 自动开关（基于模型参数量）**

```python
def _init_hint_mode(self):
    """根据模型参数量自动决定是否启用 Hint"""
    from zulong.models.container import LLM_MODEL_ID
    model_id = (LLM_MODEL_ID or "").lower()
    # 小参数模型 (<=8B): 启用 Hint
    # 大参数模型 (>8B): 关闭 Hint
    small_model_patterns = ["1b", "2b", "3b", "4b", "7b", "8b", 
                            "1.5b", "3.5b", "0.5b"]
    self._hint_enabled = any(p in model_id for p in small_model_patterns)
    logger.info(f"[Gatekeeper] Hint 模式: {'启用' if self._hint_enabled else '关闭'} "
                f"(model={model_id})")
```

**3.5 Hint 注入到 L2 消息**

InferenceEngine 的 `_on_l2_command` 中，如果 payload 包含 `task_hint`，
将其作为一条 system 消息插入到消息列表中（模型可以自行决定是否响应）。

---

### Step 4: 修改 `zulong/l2/circuit_breaker.py`

添加动态预算调整方法:

```python
def escalate_for_planning(self):
    """规划模式: 扩展预算"""
    self.safety_hard_cap = 200
    self._time_yellow_seconds = 180  # 3 分钟开始警告
    self._time_red_seconds = 300     # 5 分钟强制退出

def reset_to_default(self):
    """重置为默认预算"""
    self.safety_hard_cap = 50
    self._time_yellow_seconds = 30
    self._time_red_seconds = 45
```

---

### Step 5: 系统提示词更新

在现有系统提示词中追加「操作手册」说明:

```
## 操作手册工具

你有一个「操作手册」(start_task_plan 工具)。使用时机:
- 用户要求开发项目、系统、应用等需要多步骤规划的复杂任务 → 调用 start_task_plan
- 简单问答、闲聊、翻译、知识查询 → 直接回答，不要使用操作手册

调用 start_task_plan 后，你将获得任务规划工具集，请按流程:
1. 分析需求 → plan_add_node 添加任务节点
2. 逐步执行 → plan_mark_status 标记进度
3. 完成后 → submit_final_answer 提交最终结果
```

---

### Step 6: 清理废弃代码

| 文件 | 操作 |
|------|------|
| `pipeline/complexity_classifier.py` | 删除整个文件 |
| `scheduler_gatekeeper.py` `_is_complex_task()` | 删除方法 |
| `scheduler_gatekeeper.py` `_start_agent_orchestrator()` | 删除方法 |
| `inference_engine.py` 旧 import | 删除 complexity_classifier/PipelineOrchestrator 导入 |
| `inference_engine.py` `_on_l2_command` 中的复杂度检测 | 删除 lines 1421-1430 |

**保留**:
- `pipeline/task_graph.py` — TaskGraph 数据结构（planning 工具依赖）
- `pipeline/orchestrator.py` — 暂时保留作为参考，后续可删除
- `pipeline/tools.py` — 暂时保留作为参考（FC schema 定义）

---

## 关键文件清单

| 文件 | 操作类型 | 说明 |
|------|---------|------|
| `zulong/tools/planning_tools.py` | **新建** | 核心: PlanningSession + 所有规划 FC 工具 |
| `zulong/l2/inference_engine.py` | **修改** | 统一路径 + 动态预算 + planning 工具注册 |
| `zulong/l1b/scheduler_gatekeeper.py` | **修改** | 移除复杂度门 + 添加 Hint 机制 |
| `zulong/l2/circuit_breaker.py` | **修改** | 添加动态预算调整方法 |
| `pipeline/complexity_classifier.py` | **删除** | 硬编码分类器完全移除 |

---

## 验证计划

1. **单元测试**: 验证 `start_task_plan` 工具能正确创建 PlanningSession 并注入工具
2. **集成测试**: 运行 `tests/test_memory_architecture_live.py` 验证:
   - 测试 4-1 (复杂写作任务): 模型直接生成而不触发 planning → 在 120s 内返回 
   - 测试 5-3 (全局上下文): 与之前的 max_history=20 修复配合验证
   - 所有 14 个测试通过率 >= 85%
3. **手动测试**: 
   - 发送 "你好" → 模型直接回答（不触发 start_task_plan）
   - 发送 "帮我设计一个微服务架构迁移方案，包含..." → 模型调用 start_task_plan
   - 验证前端 TaskGraph 可视化事件正常显示
4. **Hint 开关测试**:
   - 使用 qwen3.5:4b → Hint 自动启用
   - 模拟切换到大模型标识 → Hint 自动关闭
