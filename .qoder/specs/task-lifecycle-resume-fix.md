# Fix: 复杂任务 开始/中断/恢复 全生命周期修复

## Context

用户原始需求："从头梳理复杂任务的开始/中断/重编辑，记忆的注入"。

恢复挂起的复杂任务时存在两个严重 bug（截图确认）：
1. **内容丢失** — 模型说"工作区为空"，不知道之前做了什么
2. **任务图谱重建** — 模型从零重建任务节点，覆盖已有进度

### 根因分析（对照记忆架构图）

架构图将系统分为：**宏空间**（会话节点层级 L1-L3）、**任务空间**（任务节点层级 L4-L7）、**注意力控制**（全局/单链/周期性）、**经验RAG**、**记忆RAG**。

恢复路径（`_resume_agent_orchestrator`）相比新任务启动路径（`_start_agent_orchestrator`）缺失了以下关键初始化：

| 初始化项 | 新任务 | 恢复任务 | 影响 |
|---------|--------|---------|------|
| `state.messages` 注入 | N/A（新任务） | **缺失** | 647+条对话历史丢失，模型无决策记忆 |
| `_ensure_dialogue_node()` | 调用 (L2297) | **跳过** | MemoryGraph 对话链断裂，宏空间 TEMPORAL 边缺失 |
| `_publish_graph_init_event()` | 调用 (L2644) | **缺失** | 前端不知道恢复的图谱状态 |
| `session_id` 传递 | 未传给 Orchestrator | 已传递 | 新任务 bug（已知，非本次重点） |
| `AttentionMode` | GLOBAL 初始化 | **GLOBAL 重置** | 恢复前的 FOCUS/SINGLE_CHAIN 状态丢失 |
| TaskGraph | 新建 | 正确反序列化 | **无问题** |
| 焦点上下文 | 从 MemoryGraph 恢复 | 从 MemoryGraph 恢复 | **无问题**（已有逻辑） |

**代码路径关键发现**：
- `scheduler_gatekeeper.py:2290-2294` — 恢复检测在 `_ensure_dialogue_node()` (L2297) **之前** `return`，完全跳过对话节点创建
- `orchestrator.py:1297` — 挂起时 `messages=[e.msg for e in attn_window.envelopes]` 保存完整历史
- `orchestrator.py:324-354` — 恢复时只创建 system_msg + user_msg（2条），647+条历史被忽略
- `TaskGraphPack.install()` 在恢复路径**不会被调用**（已确认），TaskGraph 正确恢复

## Plan

### Step 1: `AttentionWindowManager` 添加 `bulk_restore_history()`

**File**: `pipeline/attention_window.py` (~line 199 后)

```python
def bulk_restore_history(self, messages: List[Dict], base_turn: int = 0):
    """批量注入挂起任务保存的历史消息。
    
    从保存的 OpenAI 格式消息列表重建 MessageEnvelope，
    恢复 tool_call 分组关系和轮次编号。
    注意力窗口的 apply_window() 评分/淘汰机制会自动管理 token 预算。
    """
```

核心逻辑：
- 遍历 messages，为每条创建 `MessageEnvelope`
- **轮次推断**：每遇到 `assistant` 消息，`turn += 1`
- **分组重建**：`assistant` 消息若含 `tool_calls`，分配新 `group_id`；后续 `tool` 消息（匹配 `tool_call_id`）共享该 `group_id`
- **工具名提取**：从 `assistant.tool_calls` 中反查 `tool_call_id` → `function.name`，赋给对应 tool 消息的 `tool_name`
- **不钉住**：恢复的历史消息全部 `pinned=False`（由评分决定淘汰，新的 system+user 已经 pinned）
- 更新 `self._current_turn`、`self._seq_counter`、`self._group_counter`

### Step 2: `AgentOrchestrator` 支持恢复模式历史注入

**File**: `pipeline/orchestrator.py`

#### 2a. `__init__()` 新增参数 (line ~83)

```python
def __init__(self, ..., resume_messages: Optional[List[Dict]] = None):
    ...
    self._resume_messages = resume_messages
```

#### 2b. `run()` 中注入历史 (line ~354 后，注意力窗口创建之后)

```python
# ── 恢复模式：注入保存的对话历史 ──
if is_resume and self._resume_messages:
    # 跳过原始 system+user（已被新的替代），注入其余历史
    history_to_restore = self._resume_messages[2:]
    if history_to_restore:
        attn_window.bulk_restore_history(history_to_restore, base_turn=1)
        logger.info(
            f"[Agent] 已恢复 {len(history_to_restore)} 条历史消息"
        )
```

#### 2c. `run()` 中调整主循环起始轮次 (line ~432)

```python
# 恢复模式：主循环轮次从历史末尾继续，确保时间衰减评分准确
_start_turn = attn_window._current_turn + 1 if (is_resume and self._resume_messages) else 0

for turn_offset in range(MAX_AGENT_TURNS):
    turn = _start_turn + turn_offset
    self._current_turn = turn
    ...
```

#### 2d. 增强 `_build_resume_context()` 防重建约束 (line ~1493)

在现有返回内容末尾追加：

```python
context += (
    "\n\n**重要约束**:\n"
    "- 任务图谱已从上次保存的状态完整恢复，所有节点和进度都是准确的\n"
    "- 禁止删除或重建已有的任务节点\n"
    "- 不要调用 plan_add_node 重新创建已存在的节点\n"
    "- 你之前的完整对话历史已恢复到上下文中，可以回顾之前的决策和工具调用结果\n"
    "- 直接从第一个 pending 或 in_progress 状态的节点继续执行"
)
```

### Step 3: Gatekeeper 恢复路径补全

**File**: `zulong/l1b/scheduler_gatekeeper.py`

#### 3a. 传递 `state.messages` (line ~3079)

在创建 Orchestrator 时新增 `resume_messages` 参数：

```python
orchestrator = AgentOrchestrator(
    ...
    resume_messages=state.messages,  # 传递保存的对话历史
)
```

#### 3b. 补充 MemoryGraph 对话节点创建 (line ~3076 前)

恢复路径跳过了 `_ensure_dialogue_node()`，需要在 `_resume_and_run()` 中补充等效逻辑：

```python
# ── 为恢复会话创建新的 MemoryGraph 对话轮次节点 ──
_resume_round_id = None
try:
    from zulong.memory.memory_graph import get_memory_graph
    from zulong.memory.graph_adapters import DialogueAdapter
    _mg = get_memory_graph()
    if _mg:
        _adapter = DialogueAdapter()
        _resume_round_id = _adapter.add_round(
            _mg, _effective_rid, state.description[:80],
            prev_round_id=suspended_dialogue_round_id or None,
            session_id=suspended_session_id or session_id,
        )
        # 建立 TEMPORAL 边连接旧轮次 → 新轮次
        if suspended_dialogue_round_id and _mg.has_node(suspended_dialogue_round_id):
            from zulong.memory.memory_graph import EdgeType
            _mg.add_edge(
                suspended_dialogue_round_id, _resume_round_id,
                EdgeType.TEMPORAL, weight=1.0, protected=True,
            )
        # 更新焦点到新轮次
        _mg.update_focus_to_node(_resume_round_id)
        _diag(f"MemoryGraph resume round created: {_resume_round_id}")
except Exception as _mg_err:
    _diag(f"MemoryGraph resume round failed: {_mg_err}")
```

然后将 `_resume_round_id` 传给 Orchestrator：

```python
orchestrator = AgentOrchestrator(
    ...
    dialogue_round_id=_resume_round_id or suspended_dialogue_round_id or ...,
    ...
)
```

#### 3c. 补充图谱初始化事件 (线程启动前 ~line 3217)

```python
# 发布恢复的图谱到前端（与新任务对称）
if task_graph:
    self._publish_graph_init_event(task_graph, {
        'request_id': _effective_rid,
        'session_id': session_id or suspended_session_id,
    })
```

### Step 4: 保存注意力模式到挂起状态（可选优化）

**File**: `pipeline/orchestrator.py`

#### 4a. `_suspend_task()` 中保存注意力模式 (line ~1302)

在 metadata 中新增：

```python
metadata={
    ...
    "attention_mode": attn_window.mode.value if attn_window else "global",
    "current_node_id": getattr(attn_window, '_current_node_id', None),
}
```

#### 4b. `run()` 中恢复注意力模式

在 `bulk_restore_history` 之后：

```python
# 恢复注意力模式
if is_resume and self._resume_messages:
    _saved_mode = (self.task_graph.serialize() if self.task_graph else {})
    # 从 Gatekeeper 传入的 metadata 获取
    # （需要在 resume_messages 之外额外传递，或者从 state.metadata 中提取）
```

> **Note**: 此步骤为可选优化。注意力模式在恢复后的第一轮工具调用时会自动切换回正确状态（由 `observe_tool_call()` 驱动），因此不恢复也不会导致功能性 bug，只是前几轮的消息评分可能不够精确。建议先实施 Step 1-3，验证核心问题解决后再考虑此步。

## Files to Modify

| 文件 | 修改内容 | 优先级 |
|------|---------|--------|
| `pipeline/attention_window.py` | 新增 `bulk_restore_history()` 方法 | P0 |
| `pipeline/orchestrator.py` | 新增 `resume_messages` 参数，`run()` 注入历史，`_build_resume_context()` 防重建，轮次调整 | P0 |
| `zulong/l1b/scheduler_gatekeeper.py` | 传递 `state.messages`，补充 MemoryGraph 对话节点，发布图谱初始化事件 | P0 |

## 不需要修改的文件

- `pipeline/task_graph_pack.py` — 恢复路径不调用 `install()`，TaskGraph 通过直接反序列化正确恢复
- `pipeline/task_graph.py` — 反序列化功能正常
- `zulong/l2/task_suspension.py` — 数据完整性已验证（标准 OpenAI Dict 格式）

## Verification

1. **编译检查**: `python -c "from pipeline.orchestrator import AgentOrchestrator; from pipeline.attention_window import AttentionWindowManager"`
2. **数据完整性**: 使用 `tests/verify_resume_data.py` 验证挂起任务数据
3. **单元测试**: 创建测试验证 `bulk_restore_history()`:
   - 输入样例消息 → 验证 envelopes 数量正确
   - 验证 tool_call 分组重建正确
   - 验证 `apply_window()` 在预算内返回消息
4. **集成验证**: 恢复一个挂起任务，检查:
   - 日志显示 "已恢复 N 条历史消息"
   - 模型**不**重建任务节点
   - 模型从第一个 pending/in_progress 节点继续
   - 前端收到恢复的图谱状态
   - MemoryGraph 中对话链连续（旧 round → 新 round 有 TEMPORAL 边）
