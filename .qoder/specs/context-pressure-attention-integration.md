# Context Pressure -> Attention Mode -> BFS Integration

## Context

Zulong IDE agent 的三大机制（上下文压力感知、动态注意力窗口、BFS扩散激活）当前各自独立运行：
- **CircuitBreaker** 有 `_signal_context_pressure()` 检测 75%/90% 阈值，但仅用于强制收敛（停止工具调用）
- **AttentionWindow** 有 `usage_ratio` 属性和三种模式(GLOBAL/FOCUS/SINGLE_CHAIN)，但模式切换只靠 LLM 自己调用工具
- **BFS** 每轮无条件执行，结果不影响 LLM 决策，且污染 `last_accessed`

本次修改的目标是**打通这三个环节**：上下文压力达到阈值时，系统主动注入提示引导 LLM 切换注意力模式，BFS 仅在有意义时执行并将推荐焦点反馈给 LLM。

## Implementation Plan

### Step 1: Fix `last_accessed` Pollution

**File**: `zulong/memory/memory_graph.py` line 1287

删除 BFS 被动激活节点的 `last_accessed` 更新：

```python
# 修改前 (lines 1283-1287):
node = self._nodes.get(nid)
if node:
    node.activation = act_val
    node.last_accessed = time.time()  # ← 删除此行

# 修改后:
node = self._nodes.get(nid)
if node:
    node.activation = act_val
```

---

### Step 2: Add `pressure_force_attention` to IDEFCState

**File**: `zulong/ide/ide_session.py`, 在 `IDEFCState` dataclass 中（line 55 `cb_tool_streak` 之后）添加：

```python
# 压力 RED: 约束工具列表为仅注意力工具
pressure_force_attention: bool = False
```

---

### Step 3: Add BFS Scheduling State to IDEFCRunner

**File**: `zulong/ide/ide_fc_runner.py`, 在 `__init__` 中（~line 164 区域）添加：

```python
# BFS 调度控制
self._last_bfs_seeds_hash: str = ""
self._last_bfs_turn: int = 0
self._last_pressure_tier: str = "green"  # 压力分级跟踪（green/yellow/red）
self._bfs_min_interval: int = 3  # 最小间隔轮次
```

---

### Step 4: Refactor BFS - `_maybe_run_bfs` 替代无条件 BFS

**File**: `zulong/ide/ide_fc_runner.py`

将现有 `_run_bfs_activation` 重构为两个方法：

```python
def _compute_bfs_seeds(self) -> List[str]:
    """收集 BFS 种子（纯计算，无副作用）"""
    # 逻辑同原 _run_bfs_activation 的种子收集部分 (lines 2240-2279)
    # 返回有效种子列表

def _maybe_run_bfs(self, fc_turn: int, trigger: str = "tool_complete") -> Optional[Dict[str, float]]:
    """条件执行 BFS，返回激活结果或 None
    
    trigger: "tool_complete" | "pressure_crossing" | "status_change"
    """
    if fc_turn <= 1:
        return None
    
    seeds = self._compute_bfs_seeds()
    if not seeds:
        return None
    
    # 变更检测
    import hashlib
    seeds_hash = hashlib.md5("|".join(sorted(seeds)).encode()).hexdigest()[:8]
    
    if trigger != "pressure_crossing":
        # 非压力触发：检查种子变更 + 最小间隔
        if seeds_hash == self._last_bfs_seeds_hash:
            return None
        if fc_turn - self._last_bfs_turn < self._bfs_min_interval:
            return None
    
    # 执行 BFS
    mg = get_memory_graph()
    _min_act = 0.05 if len(seeds) > 5 else 0.01
    acts = mg.compute_activations(seeds, max_depth=3, decay=0.5, min_activation=_min_act)
    
    self._last_bfs_seeds_hash = seeds_hash
    self._last_bfs_turn = fc_turn
    
    # 日志
    if acts:
        top_acts = sorted(acts.items(), key=lambda x: -x[1])[:5]
        logger.info(f"[IDEFCRunner][BFS] turn={fc_turn} seeds={len(seeds)}, activated={len(acts)}, top={top_acts}")
    
    return acts
```

**替换 5 个调用点** (lines 181, 548, 570, 1652, 1655)：
- `self._run_bfs_activation(fc)` → `self._maybe_run_bfs(fc, "tool_complete")` （忽略返回值，保持接口兼容）

---

### Step 5: Core Integration - `_apply_pressure_guidance`

**File**: `zulong/ide/ide_fc_runner.py`

在 CircuitBreaker evaluate 之后（async 路径 line 453 后, sync 路径 line 1633 后）调用。

**设计思路**：两级响应，核心区别在于 LLM 的自主度：
- **Yellow (≥75%)**：注入 system 提示 + BFS推荐焦点，引导 LLM 主动调用注意力工具（LLM 仍可选择不调用）
- **Red (≥90%)**：设置 `state.pressure_force_attention = True`，下次 `_call_model` 时工具列表被过滤为**仅注意力工具**，同时注入强制指令。LLM 自行生成调用，但选项被约束。

```python
def _apply_pressure_guidance(self, state: IDEFCState, fc: int) -> None:
    """上下文压力感知 → 注意力引导（两级：yellow 引导 / red 强制选择注意力工具）"""
    if not self._attn_window or state.cb_force_no_tools:
        return  # CB RED 已接管，不重复干预
    
    ratio = self._attn_window.usage_ratio
    
    # 分级（仅两级）
    if ratio >= 0.90:
        tier = "red"
    elif ratio >= 0.75:
        tier = "yellow"
    else:
        tier = "green"
    
    # 仅在跨越阈值时触发（避免每轮重复注入）
    if tier == self._last_pressure_tier:
        return
    
    old_tier = self._last_pressure_tier
    self._last_pressure_tier = tier
    
    if tier == "green":
        return
    
    msgs = state.messages
    
    if tier == "yellow":
        # ── Yellow: 注入引导提示 + BFS 推荐焦点 ──
        acts = self._maybe_run_bfs(fc, trigger="pressure_crossing")
        
        parts = [
            f"[上下文压力 - 注意力引导] 当前上下文使用率已达 {ratio:.0%}。",
            f"建议调用注意力工具收窄关注范围：",
            f"  - adjust_attention_mode(mode='focus') 聚焦当前子任务",
            f"  - navigate_attention(direction='deeper') 深入关键节点",
        ]
        
        # BFS 推荐节点
        if acts:
            seeds_set = set(self._compute_bfs_seeds())
            candidates = [
                (nid, score) for nid, score in acts.items()
                if score > 0.6 and nid not in seeds_set
            ]
            if candidates:
                top_nid, top_score = max(candidates, key=lambda x: x[1])
                parts.append(
                    f"  - navigate_attention(direction='jump', target_node_id='{top_nid}') "
                    f"[BFS推荐，激活分={top_score:.2f}]"
                )
        
        hint = {"role": "system", "content": "\n".join(parts)}
        msgs.append(hint)
        self._attn_window.register_message(hint, turn=fc)
        logger.info(f"[IDEFCRunner][Pressure] YELLOW ({ratio:.0%}): 注入注意力工具引导")
    
    elif tier == "red":
        # ── Red: 约束 LLM 只能调用注意力工具 ──
        state.pressure_force_attention = True
        
        # BFS 推荐焦点
        acts = self._maybe_run_bfs(fc, trigger="pressure_crossing")
        
        parts = [
            f"[注意力强制切换] 上下文使用率达到 {ratio:.0%}（红色警戒）。",
            f"你必须立即调用注意力工具进行焦点切换：",
            f"  - adjust_attention_mode(mode='single_chain') 切换为单链推理模式",
            f"  - navigate_attention(direction='deeper') 深入当前节点",
        ]
        
        if acts:
            seeds_set = set(self._compute_bfs_seeds())
            candidates = [
                (nid, score) for nid, score in acts.items()
                if score > 0.4 and nid not in seeds_set
            ]
            if candidates:
                top_nid, top_score = max(candidates, key=lambda x: x[1])
                parts.append(
                    f"  - navigate_attention(direction='jump', target_node_id='{top_nid}') "
                    f"[推荐焦点，激活分={top_score:.2f}]"
                )
        
        hint = {"role": "system", "content": "\n".join(parts)}
        msgs.append(hint)
        self._attn_window.register_message(hint, turn=fc)
        logger.info(f"[IDEFCRunner][Pressure] RED ({ratio:.0%}): 强制注意力工具选择")
```

**`_call_model` 修改**（line ~1397 区域，在现有 `if state.cb_force_no_tools:` 之前插入）：

```python
if state.cb_force_no_tools:
    # ... existing CB RED logic (unchanged) ...
elif getattr(state, 'pressure_force_attention', False):
    # 压力 RED: 工具列表仅保留注意力工具，强制 LLM 调用
    attn_tools = self._get_attention_only_tools(state.tool_definitions)
    if attn_tools:
        kw["tools"] = attn_tools
        kw["tool_choice"] = "required"  # 强制必须选择一个工具
    logger.info(f"[IDEFCRunner][Pressure] 工具列表约束为注意力工具 ({len(attn_tools)}个)")
elif state.tool_definitions:
    kw["tools"] = state.tool_definitions
    # ... existing logic ...
```

**新增静态方法**：

```python
@staticmethod
def _get_attention_only_tools(tool_definitions: List[Dict]) -> List[Dict]:
    """压力 RED 时仅保留注意力工具，强制 LLM 从中选择"""
    _ATTENTION_TOOL_NAMES = {"navigate_attention", "adjust_attention_mode"}
    return [
        td for td in tool_definitions
        if td.get("function", {}).get("name", "") in _ATTENTION_TOOL_NAMES
    ]
```

**标志位重置**（在 `_exec_internal` 中）：

```python
# 注意力工具执行完毕后，恢复正常工具列表
if tn in ("navigate_attention", "adjust_attention_mode"):
    if getattr(state, 'pressure_force_attention', False):
        state.pressure_force_attention = False
```

---

### Step 6: Wire Into Loop

**Async path** (`ide_fc_runner.py`, line 453 的 CB evaluate try/except 块结束后)：
```python
# 上下文压力感知（在 CB 评估之后）
self._apply_pressure_guidance(state, fc)
```

**Sync path** (`ide_fc_runner.py`, line 1633 的 CB evaluate try/except 块结束后)：
```python
# 上下文压力感知（在 CB 评估之后）
self._apply_pressure_guidance(state, fc)
```

**BFS 调用点替换**（5处）：
- line 181: `self._run_bfs_activation(state.fc_turn)` → `self._maybe_run_bfs(state.fc_turn, "tool_complete")`
- line 548: `self._run_bfs_activation(fc)` → `self._maybe_run_bfs(fc, "tool_complete")`
- line 570: `self._run_bfs_activation(fc)` → `self._maybe_run_bfs(fc, "tool_complete")`
- line 1652: `self._run_bfs_activation(fc)` → `self._maybe_run_bfs(fc, "tool_complete")`
- line 1655: `self._run_bfs_activation(fc)` → `self._maybe_run_bfs(fc, "tool_complete")`

---

### Step 7: Remove TG→MG Sync from BFS (移入独立调用)

现有 `_run_bfs_activation` 中 lines 2247-2251 的 `TaskGraphAdapter().sync(mg, tg)` 逻辑应保留，但改为在 `_maybe_run_bfs` 中仅在 BFS 实际执行时调用（而非每轮调用）。这自然由 `_maybe_run_bfs` 的条件执行保证。

---

### Step 8: Config Parameters

**File**: `config/zulong_config.yaml`, 在 `context_red_ratio: 0.90` (line 192) 之后添加：

```yaml
    # BFS 调度控制
    bfs_min_interval: 3              # BFS 最小执行间隔（轮次）
```

注：压力阈值直接复用已有的 `context_yellow_ratio: 0.75` 和 `context_red_ratio: 0.90`，不新增阈值配置。

---

## Files Modified Summary

| File | Changes |
|------|---------|
| `zulong/memory/memory_graph.py` | 删除 line 1287 的 `node.last_accessed = time.time()` |
| `zulong/ide/ide_session.py` | IDEFCState 添加 `pressure_force_attention: bool = False` 字段 |
| `zulong/ide/ide_fc_runner.py` | 添加 `_apply_pressure_guidance()`、`_maybe_run_bfs()`、`_compute_bfs_seeds()`、`_get_attention_only_tools()`；替换5个BFS调用点；添加实例变量；修改 `_call_model` 和 `_exec_internal` |
| `config/zulong_config.yaml` | 添加 `bfs_min_interval: 3` |

## Verification

1. **启动后端** → 观察日志不再出现每轮 `[BFS]` 输出（仅在种子变更或压力触发时执行）
2. **发送复杂任务**（如"深度分析这个项目"）→ 观察：
   - 前期（<75%）：无压力提示，BFS 仅在种子变更时执行
   - 中期（≥75%）：日志出现 `[Pressure] YELLOW`，消息中注入注意力工具使用建议
   - 后期（≥90%）：日志出现 `[Pressure] RED`，下一次模型调用仅提供注意力工具（`tool_choice: required`），LLM 被迫调用注意力工具
3. **确认 `last_accessed` 修复**：BFS 执行后检查非种子节点的 `last_accessed` 不再被更新
4. **确认 CB 行为不变**：工具重复模式仍触发 YELLOW/RED 收敛（`cb_force_no_tools` 优先级高于 `pressure_force_attention`）
5. **确认标志位重置**：LLM 调用注意力工具后，`pressure_force_attention` 被重置，后续轮次恢复完整工具列表
