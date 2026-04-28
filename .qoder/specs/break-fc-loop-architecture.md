# 打破 FC 循环 — 从封闭黑箱到开放式步进循环

## Context

FC 循环 (`_generate_with_vllm_and_tools`, inference_engine.py:468-1122) 是祖龙 L2 推理的核心，但它是一个**封闭黑箱**：一旦进入，不检查中断、不刷新上下文、不监听事件、不同步记忆。这是之前分析报告中 11 个协同缺陷 (D1-D11) 的共同根因。

Orchestrator (pipeline/orchestrator.py) 虽然有开放式步进循环的设计，但属于**已弃用的 Pipeline 五阶段流水线架构**，且缺少 RAG/记忆/流式输出等能力。

**方案**: 在 FC 循环内部插入 5 个"检查点切面" (CUT-1~5)，将其从封闭黑箱改造为开放式步进循环，保留所有现有能力。

## 改造文件

- **`zulong/l2/inference_engine.py`** — 主要改造目标 (所有 5 个 CUT)
- **`pipeline/attention_window.py`** — 直接 import 使用 AttentionWindowManager (不修改)
- **`zulong/memory/memory_graph.py`** — 使用现有接口做同步 (不修改)

## 改造架构

```
for iteration in range(hard_cap):
    ┌─ [CUT-1] 中断检查点             ← 新增
    ├─ [CUT-2] 外部事件排空 (每5轮)    ← 新增
    ├─ [CUT-3] 上下文刷新 (每N轮)      ← 新增
    │
    ├─ (保留) 规划模式 final_answer 检查
    ├─ (保留) 时间预算检查
    ├─ (保留) 动态工具注入
    │
    ├─ [CUT-4c] apply_window() 过滤消息 ← 新增
    ├─ (保留) LLM 流式调用
    │
    ├─ if tool_calls:
    │   ├─ [CUT-4a] attn_window 注册 assistant_msg  ← 替代直接 append
    │   ├─ (保留) 执行工具
    │   ├─ [CUT-4b] attn_window 注册 tool_result    ← 替代直接 append
    │   ├─ [CUT-4b] observe_tool_call 驱动模式切换   ← 新增
    │   ├─ [CUT-5]  MemoryGraph 同步                ← 新增
    │   └─ (保留) CircuitBreaker 评估 GREEN/YELLOW/RED
    │
    └─ else: (保留) 文本输出 + InfoGap + 规划拦截
```

## 分阶段实施

### Phase 1: CUT-1 + CUT-2 — 中断响应 + 事件感知 (最小改动)

**目标**: 解决 D6(L2不可中断) 和 D7(排队消息不恢复)

#### 1a. `__init__` 新增事件收件箱

```python
# inference_engine.py __init__ 中，line 148 之后
self._external_event_inbox: list = []
self._event_inbox_lock = threading.Lock()
```

#### 1b. `_on_interrupt` 保持不变 (已设置 `_interrupt_flag`)

#### 1c. 新增 `_on_l2_command` 追加事件到收件箱

在现有 `_on_l2_command` handler 末尾，追加：
```python
# 将事件也存入 inbox 供 FC 循环排空
with self._event_inbox_lock:
    self._external_event_inbox.append(event.payload)
```

#### 1d. 新增辅助方法 `_drain_fc_events`

```python
def _drain_fc_events(self) -> list:
    """原子取出 FC 循环期间积累的外部事件"""
    with self._event_inbox_lock:
        events = list(self._external_event_inbox)
        self._external_event_inbox.clear()
    return events
```

#### 1e. FC 循环内插入 CUT-1 + CUT-2

**位置**: line 608 `for iteration in range(...)` 之后，line 609 之前

```python
for iteration in range(self.circuit_breaker.safety_hard_cap):

    # ══════ CUT-1: 中断检查 ══════
    if self._interrupt_flag:
        self._interrupt_flag = False
        logger.info(f"[vLLM-Tools] FC循环被中断 (iteration={iteration})")
        self._publish_thinking_step("fc_interrupted", iteration, {
            "reason": "external_interrupt",
            "messages_count": len(messages),
        })
        # 如果有规划会话，保存中间状态
        if self._planning_session and self._planning_session.planning_mode:
            logger.info("[vLLM-Tools] 规划会话保持活跃，下次恢复")
        break  # 退出循环，进入 line 1013 的强制生成路径

    # ══════ CUT-2: 外部事件排空 (每5轮) ══════
    if iteration > 0 and iteration % 5 == 0:
        fc_events = self._drain_fc_events()
        for evt in fc_events:
            evt_text = evt.get("text", "")
            if evt_text:
                event_msg = {
                    "role": "system",
                    "content": (
                        f"[外部事件] 用户发送了新消息: \"{evt_text[:200]}\"\n"
                        f"如果与当前任务相关请回应，如果无关请继续当前工作。"
                    ),
                }
                messages.append(event_msg)
                logger.info(f"[vLLM-Tools] 注入外部事件: {evt_text[:60]}...")

    # (existing) 动态预算、final_answer 检查、时间预算...
```

---

### Phase 2: CUT-5 — MemoryGraph 同步

**目标**: 解决 D1(TaskGraph不写入MemoryGraph) 和 D5(历史任务不可搜索)

#### 2a. 新增辅助方法 `_sync_memory_after_tool`

```python
def _sync_memory_after_tool(self, fn_name: str, args_dict: dict,
                            result_content: str, iteration: int):
    """工具调用后同步状态到 MemoryGraph (异常不阻塞主循环)"""
    try:
        from zulong.memory.memory_graph import get_memory_graph
        mg = get_memory_graph()
        if mg is None:
            return

        # 1. 如果工具参数包含 node_id，激活相关节点
        node_id = args_dict.get("node_id") or args_dict.get("outline_id")
        if node_id and mg.has_node(f"task:{node_id}"):
            mg.update_focus_to_node(f"task:{node_id}")

        # 2. 周期性保存焦点上下文 (每15轮)
        if iteration > 0 and iteration % 15 == 0:
            mg.set_last_focus_context(
                dialogue_round_id=getattr(self, '_current_session_id', ''),
                focused_task_node_id=node_id or '',
                active_node_ids=list(mg._active_node_ids) if hasattr(mg, '_active_node_ids') else [],
            )
    except Exception as e:
        logger.debug(f"[vLLM-Tools] MemoryGraph 同步跳过: {e}")
```

#### 2b. 在 FC 循环中插入 CUT-5

**位置**: 工具执行完成后，CB 评估之前 (当前 line 891 `self.circuit_breaker.record_call(...)` 之前)

```python
            # ══════ CUT-5: MemoryGraph 同步 ══════
            self._sync_memory_after_tool(function_name, args_dict,
                                         tool_result_msg.get("content", ""), iteration)

            # (existing) CircuitBreaker 评估
            self.circuit_breaker.record_call(...)
```

---

### Phase 3: CUT-3 — 定期上下文刷新

**目标**: 解决 D2(FC循环不刷新RAG/记忆/导航)

#### 3a. 方法签名变更

```python
# 改前:
async def _generate_with_vllm_and_tools(self, messages: List[Dict[str, str]]) -> str:

# 改后:
async def _generate_with_vllm_and_tools(self, messages: List[Dict[str, str]],
                                         refresh_context: Optional[Dict] = None) -> str:
```

`refresh_context` 结构:
```python
{
    "user_input": str,         # 原始用户输入
    "session_id": str,         # 会话 ID
    "rag_context": str,        # RAG 检索结果 (可选)
    "visual_context": str,     # 视觉上下文 (可选)
}
```

#### 3b. 调用方传入 refresh_context

**位置**: line 2068

```python
# 改前:
response = await self._generate_with_vllm_and_tools(messages)

# 改后:
response = await self._generate_with_vllm_and_tools(messages, refresh_context={
    "user_input": user_input,
    "session_id": session_id or "",
})
```

#### 3c. 新增辅助方法 `_rebuild_system_prompt`

```python
async def _rebuild_system_prompt(self, refresh_context: Dict) -> str:
    """重建 system prompt 内容 (含最新导航/记忆/RAG)"""
    # 复用现有的 _build_messages_with_history_async
    fresh_messages = await self._build_messages_with_history_async(
        user_input=refresh_context.get("user_input", ""),
        session_id=refresh_context.get("session_id", ""),
    )
    # 提取第一条 system 消息的 content
    if fresh_messages and fresh_messages[0].get("role") == "system":
        return fresh_messages[0]["content"]
    return ""
```

#### 3d. FC 循环内插入 CUT-3

**位置**: CUT-2 之后，line 613 (final_answer 检查) 之前

```python
    # ══════ CUT-3: 定期上下文刷新 ══════
    _refresh_interval = 15 if (self._planning_session and
                                self._planning_session.planning_mode) else 8
    if (iteration > 0 and iteration % _refresh_interval == 0
            and refresh_context):
        try:
            new_system_content = await self._rebuild_system_prompt(refresh_context)
            if new_system_content and messages and messages[0].get("role") == "system":
                messages[0]["content"] = new_system_content
                logger.info(f"[vLLM-Tools] 上下文已刷新 (iteration={iteration})")
                # 如果有规划会话，重新注入规划上下文
                if self._planning_session and self._planning_session.planning_mode:
                    tg_ctx = self._planning_session.task_graph.serialize()
                    messages[0]["content"] += (
                        f"\n\n[当前任务图谱状态]\n{tg_ctx[:2000]}"
                    )
        except Exception as e:
            logger.debug(f"[vLLM-Tools] 上下文刷新跳过: {e}")
```

---

### Phase 4: CUT-4 — AttentionWindowManager 集成

**目标**: 解决消息只追加不管理的问题，替代粗暴的 CB RED 压缩

#### 4a. import 和初始化

**文件头部新增 import**:
```python
from pipeline.attention_window import AttentionWindowManager
```

**循环入口前 (line 589 `self.circuit_breaker.reset()` 之后) 新增初始化**:
```python
    # ══════ 初始化注意力窗口 ══════
    _attn_window = None
    try:
        from zulong.memory.memory_graph import get_memory_graph
        _mg = get_memory_graph()
        _attn_window = AttentionWindowManager(
            context_window_size=getattr(
                self.circuit_breaker, '_context_window_size', 65536),
            memory_graph=_mg,
            use_graph_attention=(_mg is not None),
        )
        # 注册初始消息
        for i, msg in enumerate(messages):
            _attn_window.register_message(
                msg, turn=0,
                pinned=(msg.get("role") == "system" or i <= 1),
            )
    except Exception as e:
        logger.warning(f"[vLLM-Tools] AttentionWindow 初始化失败，降级: {e}")
        _attn_window = None
```

#### 4b. LLM 调用前使用窗口 (CUT-4c)

**位置**: line 676 `call_vllm_with_tools_stream` 调用之前

```python
    # ══════ CUT-4c: 使用注意力窗口过滤消息 ══════
    if _attn_window:
        _llm_messages = _attn_window.apply_window()
    else:
        _llm_messages = messages

    # 将 call_vllm_with_tools_stream 内的 messages 参数替换为 _llm_messages
```

#### 4c. 工具调用后注册到窗口 (CUT-4a, CUT-4b)

**位置**: line 811 (assistant_msg append) 和 line 873 (tool_result append)

```python
    # CUT-4a: 注册 assistant 消息
    messages.append(assistant_msg)  # 保留原始 append
    if _attn_window:
        _group_id = _attn_window.new_tool_group()
        _attn_window.register_message(assistant_msg, turn=iteration,
                                       group_id=_group_id)

    # ... 执行工具 ...

    # CUT-4b: 注册 tool result + observe
    messages.append(tool_result_msg)  # 保留原始 append
    if _attn_window:
        _attn_window.register_message(tool_result_msg, turn=iteration,
                                       tool_name=function_name,
                                       group_id=_group_id)
        _attn_window.observe_tool_call(function_name, args_dict)
```

**注意**: 保留 `messages.append()` 确保 CB 评估的 messages 列表完整。LLM 只看 `_attn_window.apply_window()` 的结果。

#### 4d. CB RED 压缩降级

**位置**: line 1016-1058 的上下文压缩逻辑

```python
    # 如果有 attn_window，使用它做最终裁剪，跳过手动压缩
    if _attn_window:
        messages = _attn_window.apply_window()
        logger.info(f"[vLLM-Tools] 使用 AttentionWindow 裁剪 (剩余 {len(messages)} 条)")
    else:
        # (existing) 手动压缩逻辑作为 fallback
        ...
```

## Modification A-E 兼容性

| Modification | 影响 | 处理 |
|---|---|---|
| A: 规划强制引导 (line 875-889) | 无冲突 | 保持不变 |
| B: CB 扩展 (line 592-595) | 无冲突 | CUT-3 的 refresh_interval 按 planning_mode 动态调整 |
| C: 文本拦截 (line 949-993) | 低 | 纠正消息的 append 同步到 _attn_window |
| D: 时间预算 (line 597-638) | 无冲突 | 在 CUT-1~3 之后执行 |
| E: 压缩 (line 1016-1058) | 中 | 有 _attn_window 时跳过手动压缩 |

## 风险控制

| 风险 | 级别 | 缓解策略 |
|---|---|---|
| messages 和 attn_window 双重状态 | 高 | 保留 messages.append + 同步到 window；LLM 用 window 输出，CB 用原始 messages |
| apply_window 淘汰重要消息 | 中 | 最近 3 轮工具结果的 group 自动获得高分不被淘汰 (AWM 已有此机制) |
| 上下文刷新延迟 (FAISS 检索) | 中 | 每 8-15 轮才触发一次；失败不阻塞 (try/except) |
| _attn_window 初始化失败 | 低 | 所有 CUT-4 代码都有 `if _attn_window:` 守卫，降级为原始行为 |

## 验证方法

1. **py_compile**: `python -m py_compile zulong/l2/inference_engine.py`
2. **中断测试**: 在 FC 循环运行期间设置 `_interrupt_flag = True`，验证循环退出
3. **事件注入测试**: 在 FC 循环运行期间通过 EventBus 发布事件，验证事件被排空并注入 messages
4. **上下文刷新测试**: 运行超过 8 轮工具调用的任务，验证 system prompt 被刷新且包含最新记忆
5. **注意力窗口测试**: 运行 20+ 轮任务，验证 LLM 收到的消息经过窗口裁剪，token 数量可控
6. **回归测试**: 验证简单问答、规划模式 (start_task_plan)、信息缺口检测等现有功能不受影响
