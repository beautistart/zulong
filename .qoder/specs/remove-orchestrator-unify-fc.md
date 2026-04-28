# 删除 Orchestrator，统一 FC 自主决策架构

## Context

祖龙系统当前存在**双路径架构**：普通消息走 L2 单轮推理（无工具能力），复杂/恢复任务走 Orchestrator 硬编码循环（有 FC 工具能力）。这导致 7+ 个恢复流程 Bug、两套工具注册、两套记忆同步逻辑、Gatekeeper 需硬编码关键词路由。

**目标**：删除 Orchestrator，将 FC 能力直接内化到 L2 InferenceEngine。所有用户消息走同一条路径，模型通过 FC 协议自主决定：直接回复、调用工具、检索记忆、切换注意力深度——全部由模型驱动，不由代码框定。

## 架构变更概览

```
当前：
  用户 → Gatekeeper → [硬编码路由]
    ├─ 普通消息 → L2（无工具） → 单轮回复
    └─ 恢复关键词 → Orchestrator（200轮硬循环 + TaskGraph + 护栏 + 检查点）

目标：
  用户 → Gatekeeper（简化） → L2（FC 感知）
    └─ 统一路径：vllm_client.create(messages, tools=tools, tool_choice="auto")
       ├─ 模型不调工具 → 直接回复（等价于原 L2 单轮）
       └─ 模型调工具 → 执行 → 结果回传 → 模型继续（自然 FC 循环）
```

---

## Step 1: L2 InferenceEngine FC 核心升级

**文件**: `zulong/l2/inference_engine.py`

### 1a. 新增 `_collect_tool_definitions()`

遍历 `self.tool_engine.registry.tools`，对每个 `enabled` 的 `BaseTool` 调用 `tool.get_function_schema()`（已有接口，`base.py` L221），返回 `List[Dict]`。

### 1b. 新增 `_execute_tool_call(tc: Dict) -> str`

解析 FC 协议的 `tool_call`（name + arguments JSON），通过 `self.tool_engine.call_tool()` 执行，返回结果 JSON 字符串。错误时返回 `{"error": "..."}` 而非抛异常。

### 1c. 重构 `_process_with_memory` → FC 感知

在现有的上下文构建（RAG、视觉、MemoryGraph 注入）之后，将单次 LLM 调用改为 FC 循环：

```python
tools = self._collect_tool_definitions()
messages = self._build_messages_with_history(user_input, rag_context, visual_context)

for turn in range(MAX_FC_TURNS):  # 安全上限 50
    if time.time() - start > MAX_FC_TIME:  # 安全上限 300s
        break

    response = self.vllm_client.chat.completions.create(
        model=LLM_MODEL_ID, messages=messages,
        tools=tools if tools else NOT_GIVEN,
        tool_choice="auto" if tools else NOT_GIVEN,
        max_tokens=2048, temperature=0.3, ...
    )

    assistant_msg = response.choices[0].message
    messages.append(assistant_msg)  # 含 tool_calls 或纯文本

    if not assistant_msg.tool_calls:
        break  # 模型选择纯文本 → 结束

    for tc in assistant_msg.tool_calls:
        result = self._execute_tool_call(tc)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

final_text = assistant_msg.content or ""
```

后续处理（`</think>` 清理、TTS 清洗、`_update_memory`、`L2_OUTPUT` 发布）保持不变。

**关键**：无工具时（`tools=[]`），等价于原单轮推理，行为完全兼容。

### 1d. 删除已失效代码

- 删除 `_build_tools_description()`（L1066-1101）— 工具描述不再塞进 system prompt 文本，而是通过 FC `tools=` 参数传递
- 删除 system prompt 中的 `【可用工具】` 文本注入和 `【操作手册工具】` 硬编码引导（L892-907）
- 保留 system prompt 中的身份、时间、GPS 焦点路径、MemoryGraph 记忆注入

---

## Step 2: 新增 MemoryGraph FC 工具

**新文件**: `zulong/tools/memory_graph_tools.py`

暴露 MemoryGraph 核心能力为 FC 工具（当前仅 3.8% 能力暴露）：

| 工具名 | 功能 | 包装的 MemoryGraph 方法 |
|--------|------|------------------------|
| `recall_memory` | 混合检索记忆（热/冷） | `retrieve_context(query, top_k)` |
| `read_memory_node` | 读取指定节点详情 | `get_node(id)` + 邻接查询 |
| `save_memory_note` | 写入笔记到图 | `add_node(KNOWLEDGE/TASK)` |
| `discover_related` | 激活扩散发现关联 | `compute_activations(seeds)` |

每个工具继承 `BaseTool`，实现 `execute()` + `_get_parameters_schema()`，由 ToolEngine 自动注册。

---

## Step 3: 迁移任务管理 & 执行工具

**新文件**: `zulong/tools/task_tools.py` + `zulong/tools/exec_tools.py`

从 `pipeline/tools.py` 的 `AgentToolRegistry` 中提取有价值的工具为独立 `BaseTool` 子类：

### 任务管理工具（task_tools.py）

| 工具名 | 迁移自 | 说明 |
|--------|--------|------|
| `task_plan_node` | `plan_add_node` | 在 MemoryGraph 中创建 TASK 节点 |
| `task_mark_status` | `plan_mark_status` | 更新任务节点状态 |
| `task_view_overview` | `view_graph_overview` | 查看任务图谱概览 |
| `task_view_detail` | `view_node_detail` | 查看节点详情 |
| `list_suspended_tasks` | `list_task_workspaces` | 列出挂起任务 |
| `resume_suspended_task` | `resume_at_node` | 恢复挂起任务上下文 |

**TaskGraph 实例管理**：InferenceEngine 持有 `_active_task_graph: Optional[TaskGraph]`。模型首次调用 `task_plan_node` 时自动创建。通过 `ToolEngine.set_context("task_graph", tg)` 注入，工具通过上下文获取。

### 执行工具（exec_tools.py）

| 工具名 | 迁移自 | 说明 |
|--------|--------|------|
| `write_file` | `exec_write_file` | 写文件到工作空间（沙箱） |
| `run_command` | `exec_run_command` | 执行白名单命令（30s 超时） |

安全措施（命令白名单 `_SAFE_COMMAND_PREFIXES`、沙箱目录）完整继承。

---

## Step 4: ToolEngine 上下文注入

**文件**: `zulong/tools/tool_engine.py`

新增运行时上下文机制：
```python
class ToolEngine:
    def __init__(self):
        self._context: Dict[str, Any] = {}

    def set_context(self, key: str, value: Any): ...
    def get_context(self, key: str, default=None) -> Any: ...
    def clear_context(self): ...
```

InferenceEngine 在 FC 循环开始前注入：`task_graph`、`workspace_dir`、`dialogue_round_id`。工具在 execute 时通过 `ToolEngine().get_context("task_graph")` 获取。

---

## Step 5: Gatekeeper 简化

**文件**: `zulong/l1b/scheduler_gatekeeper.py`

### 删除的代码（~800 行）

| 方法/代码块 | 行号 | 理由 |
|------------|------|------|
| 恢复关键词检测 | L530-541 | 模型自主判断，无需硬编码 |
| `_handle_resume_task` | L1624-1712 | L2 FC 自行处理恢复 |
| `_start_agent_orchestrator` | L2013-2222 | Orchestrator 已删除 |
| `_preempt_for_new_task` | L2224-2291 | 中断逻辑移入 InferenceEngine |
| `_resume_agent_orchestrator` | L2480-2789 | Orchestrator 已删除 |
| `_auto_resume` 相关 | L2293-2327 | 模型自主恢复 |
| Orchestrator 引用属性 | 散布各处 | `_active_orchestrator`, `_active_orch_thread` 等 |

### 保留的功能

- `_handle_normal_command` — 仍为用户消息处理入口（打包 + 发布 `SYSTEM_L2_COMMAND`）
- `_ensure_dialogue_node` — MemoryGraph 对话节点创建
- 复盘模式处理、空闲挂起、冷却检测、语音模式检测
- 局部/共享上下文构建

### `_handle_normal_command` 中的简化

- 删除 `【操作手册工具】` 和 `start_task_plan` 引导文本
- 所有消息统一走 `SYSTEM_L2_COMMAND` 事件到 L2

---

## Step 6: 中断处理（替代 Orchestrator 抢占）

**文件**: `zulong/l2/inference_engine.py`

FC 循环进行中收到新用户消息时：
1. Gatekeeper 发布新 `SYSTEM_L2_COMMAND`
2. InferenceEngine 设置中断标志 `_fc_interrupt_requested = True`
3. FC 循环每轮开头检查中断标志
4. 触发时：保存当前状态（通过 `TaskSuspensionManager`），处理新消息
5. 新消息处理完后，模型可通过 `resume_suspended_task` 工具自主恢复

---

## Step 7: pipeline/ 目录清理

### 删除的文件

| 文件 | 大小 |
|------|------|
| `pipeline/orchestrator.py` | 65.4 KB |
| `pipeline/tools.py` | 54.5 KB |
| `pipeline/attention_window.py` | 26.8 KB |
| `pipeline/attention.py` | 13.6 KB |
| `pipeline/agent_prompt.py` | 11.3 KB |
| `pipeline/guardrails.py` | 8.0 KB |
| `pipeline/history_persistence.py` | 11.1 KB |
| `pipeline/task_graph_pack.py` | 14.1 KB |
| `pipeline/prompts.py` | 4.6 KB |
| **合计删除** | **~230 KB** |

### 保留并移动

| 文件 | 移动到 | 理由 |
|------|--------|------|
| `pipeline/task_graph.py` | `zulong/l2/task_graph.py` | 数据结构仍需要，工具层使用 |

### 清理 `pipeline/__init__.py` 中的导出

---

## Step 8: 挂起任务兼容

**文件**: `zulong/l2/task_suspension.py`

- `TaskSuspensionManager` 本身不依赖 Orchestrator，保持不变
- InferenceEngine 在 FC 循环超时/中断时直接调用 `suspend_task()`
- 序列化内容：`messages`（FC 对话历史）、`task_graph`、`metadata`
- 旧版挂起文件（含 Orchestrator 格式）兼容：保留 `TaskGraph.deserialize()` 支持

---

## 关键文件清单

| 文件 | 操作 |
|------|------|
| `zulong/l2/inference_engine.py` | **核心改造** |
| `zulong/tools/memory_graph_tools.py` | **新建** |
| `zulong/tools/task_tools.py` | **新建** |
| `zulong/tools/exec_tools.py` | **新建** |
| `zulong/tools/tool_engine.py` | **小改**（加上下文注入） |
| `zulong/tools/base.py` | **不变**（接口已满足） |
| `zulong/l1b/scheduler_gatekeeper.py` | **大量删除**（~800行） |
| `zulong/l2/task_suspension.py` | **不变** |
| `zulong/memory/memory_graph.py` | **不变**（由工具层包装） |
| `pipeline/task_graph.py` | **移动**到 `zulong/l2/` |
| `pipeline/` 其余文件 | **删除** |

---

## 验证方案

### 1. 单元测试
- FC 循环基础：无工具 → 单轮返回；有工具 → 执行后继续；安全上限触发 → 优雅退出
- 每个迁移工具的 `execute()` 独立测试

### 2. 集成测试
- "你好" → 纯文本回复（无工具调用，不退化）
- "帮我搜索AI市场分析" → 模型自主调用 openclaw_search → 整理结果返回
- "继续之前的任务" → 模型通过 recall_memory + resume_suspended_task 自主恢复
- 中途新消息中断 → 正确挂起 → 可恢复

### 3. 回归测试
- `py_compile` 所有修改文件
- 视觉/语音模式正常
- 复盘模式不受影响
- MemoryGraph 持久化正常
