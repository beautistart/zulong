# 自主动态注意力与思维深度导航

## Context

祖龙系统的 MemoryGraph 拥有完整的 BFS 扩散激活算法 (`compute_activations`) 和赫布学习算法 (`hebbian_strengthen`)，但这两个核心算法在生产代码中**从未被调用**（仅在单元测试中验证）。这意味着：

- 图中边权始终停留在初始值，无法反映用户实际关注的记忆关联
- 节点 `activation` 值始终为 0.0 或过时数据
- 赫布学习产生的 ASSOCIATION 边在生产中永远不会出现

同时，焦点导航系统 (`update_focus_to_node`, `get_focus_path_summary`) 虽然功能完整，但与检索系统的协作有明显缺口：
- 冷数据 FAISS 检索 (`_retrieve_cold`) 完全无视焦点信息
- 热数据 BFS 种子仅来自关键词匹配，不包含焦点路径节点
- 焦点仅在创建对话轮次/任务时被动更新，话题自然切换时不会自动漂移

本方案的目标是：**让已实现但未激活的核心算法真正运作，并增强焦点路径与检索系统的协作**。

## 设计原则

- **"固化骨架+自由决策"**：所有新增逻辑是纯确定性基础设施代码，不涉及 LLM 调用
- **最小侵入**：利用现有 EventBus + MemoryGraph 基础设施，不创建新文件、不新增事件类型
- **不过度工程**：只做当前有明确价值的改进，明确排除 CoreToolManager 联动等收益不明确的功能

---

## 需修改的文件

| 文件 | 修改范围 | 说明 |
|------|---------|------|
| `zulong/l1b/scheduler_gatekeeper.py` | 新增 ~60 行 | L2_OUTPUT 订阅 + 自主注意力循环 + 焦点漂移 |
| `zulong/memory/memory_graph.py` | 修改 2 处 ~20 行 | `_retrieve_cold` 焦点 boost + `_retrieve_hot` BFS 种子增强 |

---

## Phase 1: 自主注意力循环（激活 compute_activations + hebbian_strengthen）

### 原理

在 Gatekeeper 中订阅 `L2_OUTPUT` 事件（L2 每完成一次回复就触发），以当前焦点路径节点为种子执行一轮 BFS 扩散 + 赫布学习。

**EventBus 路由确认**：`L2_OUTPUT` 事件走优先级队列 → 后台分发线程 `_dispatch_loop()` → `_dispatch_event()` → 调用所有订阅者（无名字过滤）。因此 Gatekeeper 订阅 L2_OUTPUT 只需常规 `event_bus.subscribe()` 即可。

### 实现步骤

**Step 1.1**: 在 `Gatekeeper.__init__()` 中新增实例变量

```python
self._last_focus_drift_time = 0.0   # 焦点漂移防抖时间戳
```

**Step 1.2**: 在 `_register_event_handlers()` 末尾新增一行订阅

```python
event_bus.subscribe(EventType.L2_OUTPUT, self._on_l2_output_attention, "AttentionLoop")
```

注意：订阅者名称用 `"AttentionLoop"`（不含 "L1-B"），因为 L2_OUTPUT 走 `_dispatch_event()` 而非 `_route_to_l1b()`，两者都能正常分发，但语义上更准确。

**Step 1.3**: 新增 `_on_l2_output_attention(self, event)` 方法

位置：放在 `_register_event_handlers()` 方法之后（约行 73 附近）

```python
def _on_l2_output_attention(self, event):
    """L2 回复后自主注意力循环：BFS 扩散 + 赫布学习 + 焦点漂移"""
    try:
        from zulong.memory.memory_graph import get_memory_graph
        mg = get_memory_graph()
        if mg is None:
            return

        # === A. 提取 BFS 种子 ===
        ctx = mg.get_last_focus_context()
        seeds = (ctx or {}).get("focus_path") or []
        if not seeds:
            # 降级：无焦点时不执行
            logger.debug("[AttentionLoop] 无焦点路径，跳过注意力循环")
            return

        # === B. BFS 扩散激活 ===
        activations = mg.compute_activations(
            seed_node_ids=seeds, max_depth=2, decay=0.5
        )

        # === C. 赫布学习 ===
        mg.hebbian_strengthen()
        logger.debug(
            f"[AttentionLoop] 扩散激活完成: "
            f"seeds={len(seeds)}, activated={len(activations)}"
        )

        # === D. 焦点自动漂移（Phase 3 逻辑） ===
        self._maybe_drift_focus(mg, activations, seeds)

    except Exception as e:
        logger.debug(f"[AttentionLoop] 注意力循环异常: {e}")
```

---

## Phase 2: 焦点感知检索增强

### 2A: 冷数据路径焦点 boost

**文件**: `zulong/memory/memory_graph.py`，`_retrieve_cold()` 方法（约行 1905-1917 的循环内）

在现有的 `imp_boost` 计算之后、`results.append()` 之前，新增焦点 boost 逻辑：

```python
# 焦点子树 boost (冷数据路径)
focus_boost = 0.0
if self._last_focus_context and self._last_focus_context.get("focus_path"):
    _cold_fp = set(self._last_focus_context["focus_path"])
    if node_id in _cold_fp:
        focus_boost = 0.15
final_score = faiss_score + imp_boost + focus_boost
```

boost 值 0.15 低于热路径的 0.2，因为冷数据已有较高的 FAISS 相似度基准。

### 2B: 热数据路径 BFS 种子增强

**文件**: `zulong/memory/memory_graph.py`，`_retrieve_hot()` 方法

在 BFS 扩散阶段（行 1868 `if hot_seed_ids:` 之前），注入焦点路径节点作为额外 BFS 种子：

```python
# 焦点路径节点注入 BFS 种子（取最后 3 个，即最接近当前焦点的节点）
if self._last_focus_context and self._last_focus_context.get("focus_path"):
    _focus_seeds = self._last_focus_context["focus_path"][-3:]
    for fid in _focus_seeds:
        if fid not in hot_seed_ids and self.is_recent(fid, window_seconds):
            hot_seed_ids.append(fid)
```

限制最多 3 个额外种子（焦点路径末尾的 3 个节点），避免种子爆炸。仅注入热窗口内的焦点节点。

---

## Phase 3: 对话驱动焦点自动漂移

### 原理

在 Phase 1 的 `compute_activations()` 执行完毕后，检查激活值最高的非焦点路径节点。如果激活值超过阈值且类型为 TASK/DIALOGUE，则自动将焦点漂移到该节点。

### 实现步骤

在 `scheduler_gatekeeper.py` 中新增 `_maybe_drift_focus()` 私有方法：

```python
def _maybe_drift_focus(self, mg, activations, current_seeds):
    """基于激活值的焦点自动漂移（确定性规则，无 LLM）"""
    import time as _time
    now = _time.time()

    # 防抖：两次自动漂移间隔至少 10 秒
    if now - self._last_focus_drift_time < 10.0:
        return

    # 排除当前焦点路径上的节点
    seeds_set = set(current_seeds)
    candidates = [
        (nid, score) for nid, score in activations.items()
        if nid not in seeds_set and score > 0.6
    ]
    if not candidates:
        return

    # 按激活值降序
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_nid, top_score = candidates[0]

    # 仅对 TASK/DIALOGUE 类型节点漂移
    node = mg.get_node(top_nid)
    if not node:
        return
    from zulong.memory.memory_graph import NodeType
    if node.node_type not in (NodeType.TASK, NodeType.DIALOGUE):
        return

    # 执行漂移
    if mg.update_focus_to_node(top_nid):
        self._last_focus_drift_time = now
        logger.info(
            f"[FocusDrift] 焦点自动漂移: → {node.label[:30]} "
            f"(activation={top_score:.2f}, type={node.node_type.value})"
        )
```

### 设计决策

- **阈值 0.6**：`compute_activations` 种子节点初始值为 1.0，2 跳传播后 max_depth=2, decay=0.5 的最大传播值为 0.25。阈值 0.6 意味着只有多条路径汇聚的高关联节点才会触发漂移。
- **防抖 10 秒**：避免快速连续对话中焦点剧烈震荡
- **仅 TASK/DIALOGUE 类型**：排除 KNOWLEDGE/CONCEPT 等辅助节点，焦点应停留在"做什么"而非"知道什么"
- **手动 `navigate_attention` 工具不受防抖限制**：防抖仅限制自动漂移

---

## 明确排除

- **不修改 CoreToolManager**：工具选择与焦点联动收益不明确，留待后续
- **不修改 start_prune_loop**：修剪循环的逻辑和频率保持不变
- **不引入新的 asyncio 任务**：利用 EventBus 同步分发
- **不新增事件类型**：利用现有 L2_OUTPUT
- **不创建新文件**：所有修改在现有 2 个文件中完成

---

## 线程安全分析

`_on_l2_output_attention()` 在 EventBus 的后台分发线程中执行。`compute_activations()` 和 `hebbian_strengthen()` 修改节点的 `activation` 值和边的 `weight`。

与 `_retrieve_hot()`/`_retrieve_cold()`（在 `run_in_executor` 线程池中）可能存在竞争。但：
- `update_node_activation` 是简单属性赋值，Python GIL 保证原子性
- 竞争的最坏结果：读到略过时的激活值（不影响正确性）
- `compute_activations` max_depth=2、种子数 2-4，在 1000 节点图上耗时 < 1ms

---

## 实施顺序

```
Phase 2 (焦点感知检索) → Phase 1 (自主循环) → Phase 3 (焦点漂移)
```

- Phase 2 最先：纯检索增强，改动最小，可立即验证
- Phase 1 其次：注册事件监听 + 调用已有方法
- Phase 3 最后：依赖 Phase 1 的 activations 返回值

---

## 验证方法

### 1. 语法验证
```bash
python -m py_compile zulong/l1b/scheduler_gatekeeper.py
python -m py_compile zulong/memory/memory_graph.py
```

### 2. 单元测试
```bash
python -m pytest tests/test_memory_graph.py -v
```
确认 `compute_activations` 和 `hebbian_strengthen` 的现有测试仍通过。

### 3. 集成验证（Web UI）

**测试 A - 自主注意力循环**：
1. 启动系统，发送任意对话
2. 在日志中搜索 `[AttentionLoop]`，确认每次 L2 回复后都有激活日志
3. 检查 `activated=` 后的数字 > 0

**测试 B - 焦点感知检索**：
1. 创建一个复杂任务（触发 COMPLEX 意图）
2. 在任务进行中，发送一个模糊相关但不包含关键词的查询
3. 观察检索结果中是否包含任务相关的节点（日志搜索 `[MemoryGraph]`）

**测试 C - 焦点自动漂移**：
1. 先讨论话题 A（如"帮我设计一个API"）
2. 再讨论话题 B（如"帮我写测试用例"）
3. 在日志中搜索 `[FocusDrift]`，确认焦点是否自动漂移到新话题相关节点
