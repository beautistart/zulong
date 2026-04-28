# 祖龙系统调试修复计划

## Context

祖龙系统经过大规模重构（图记忆、注意力机制、思维导航、任务恢复等），多个模块间的衔接断裂，导致：
- MemoryGraph 的智能记忆检索在主推理路径中被完全跳过（模型看不到历史对话/任务/知识）
- FAISS 冷记忆索引永远为空（finalize_round 从未调用）
- 记忆图永不衰减修剪（start_prune_loop 未启动）
- 崩溃恢复通知不在启动时触发
- 任务恢复匹配精度低

本计划修复 6 个确认的缺陷，按依赖顺序分 4 个阶段执行。每阶段独立可测试。

---

## Phase 1: 修复 FAISS 写入路径（finalize_round 未调用）

**为什么先修**: FAISS 索引是冷记忆检索的数据源。不先写入数据，后续 Phase 2 修好检索也读不到东西。

**文件**: `zulong/l2/inference_engine.py`

**位置**: `_update_memory()` 方法（第1266行），在第1314行 `add_sub_dialogue()` 调用之后、第1327行 `self._current_dialogue_round_id = None` 之前

**改动**: 插入约 8 行代码

```python
                # 完成对话轮次（索引到 FAISS 摘要侧车，供冷记忆检索）
                try:
                    adapter.finalize_round(
                        mg, round_id,
                        total_turns=1,
                        status="completed",
                    )
                except Exception as e:
                    logger.warning(f"[MemoryGraph] finalize_round 失败: {e}")
```

**验证**: 启动系统发送一条消息，日志中应出现 `[DialogueAdapter] 对话摘要已索引到 FAISS`

---

## Phase 2: 修复记忆检索注入主路径（retrieve_context 被跳过）

**为什么第二修**: Phase 1 保证 FAISS 有数据，现在让主推理路径实际使用它。

**文件**: `zulong/l2/inference_engine.py`

**位置**: `_build_messages_with_history()` 方法（第1166行），在第1249行 RAG 注入块之后、第1251行 `请开始回答` 之前

**改动**: 从异步版 `_build_messages_with_history_async`（第1761-1825行）移植两个代码块到同步版，使用 `_run_async` 桥接异步调用。约 45 行。

**具体步骤**:

1. 在方法内部定义或导入 `_run_async` 桥接函数（复用 `zulong/tools/memory_graph_tools.py:20-33` 的实现）

2. 添加**思维导航注入**（对应异步版第1764-1773行）:
```python
        # 思维导航注入（焦点路径摘要）
        try:
            from zulong.memory.memory_graph import get_memory_graph as _get_mg_nav
            _mg_nav = _get_mg_nav()
            if _mg_nav:
                focus_summary = _mg_nav.get_focus_path_summary()
                if focus_summary:
                    system_parts.append(f"\n{focus_summary}\n")
        except Exception:
            pass
```

3. 添加 **MemoryGraph 记忆注入**（对应异步版第1778-1825行）:
```python
        # MemoryGraph 统一记忆检索与注入
        try:
            from zulong.memory.memory_graph import get_memory_graph
            _mg = get_memory_graph()
            if _mg:
                if not getattr(_mg, '_rag_manager', None) and self.rag_manager:
                    _mg.set_rag_manager(self.rag_manager)
                mg_results = _run_async(
                    _mg.retrieve_context(
                        user_input, top_k=8,
                        session_id=getattr(self, '_current_session_id', ""),
                    )
                )
                # ... 按 node_type 格式化（复用异步版第1789-1823行的逻辑）
        except Exception as e:
            logger.warning(f"[MemoryGraph] 记忆检索失败，降级跳过: {e}")
```

**验证**: 发送消息 A（"我叫张三"），等 1 分钟，发送消息 B（"我叫什么名字？"）。日志应出现 `[MemoryGraph] 注入 X 条记忆到上下文`，LLM 应回答"张三"。

---

## Phase 3: 修复启动初始化 + 恢复匹配

### 3A: bootstrap.py 缺失初始化调用

**文件**: `zulong/bootstrap.py`

**改动 1 - sync_all + RecoveryNotifier**（`initialize()` 方法中，第124行之后）:

```python
        # 全量同步各适配器数据到 MemoryGraph
        if _memory_graph:
            try:
                _memory_graph.sync_all()
            except Exception as e:
                logger.warning(f"[BOOTSTRAP] MemoryGraph sync_all 失败: {e}")

        # 检查可恢复任务（通知用户）
        try:
            from zulong.l2.recovery_notifier import RecoveryNotifier
            RecoveryNotifier.check_and_notify()
        except Exception as e:
            logger.warning(f"[BOOTSTRAP] RecoveryNotifier 失败: {e}")
```

**改动 2 - start_prune_loop**（`_start_camera()` 方法中，第286行 ReviewTrigger 启动之后）:

```python
            # 启动 MemoryGraph 修剪循环
            if _memory_graph:
                asyncio.create_task(_memory_graph.start_prune_loop())
                logger.info("[BOOTSTRAP] MemoryGraph 修剪循环已启动")
```

**验证**: 启动系统，日志应出现:
- `[MemoryGraph] 全量同步完成`
- `[MemoryGraph] 修剪循环已启动 (间隔 1800s)`
- 如有挂起任务: `[RecoveryNotifier] 已通知用户 N 个可恢复任务`

### 3B: Gatekeeper 恢复任务匹配优化

**文件**: `zulong/l1b/scheduler_gatekeeper.py`

**位置**: 第1352-1369行的恢复意图检测块

**改动**: 保留原有按时间排序的兜底逻辑，但优先尝试 `TaskSuspensionManager.find_by_description()` 语义匹配（约 12 行）:

```python
        if is_resume:
            # 优先：语义匹配
            try:
                from zulong.l2.task_suspension import TaskSuspensionManager
                tsm = TaskSuspensionManager()
                matched = _run_async(tsm.find_by_description(text))
                if matched:
                    resume_task_id = matched.get("task_id", "")
            except Exception:
                pass
            # 兜底：按时间排序取最新
            if not resume_task_id:
                <保留原有目录扫描代码>
```

**验证**: 挂起两个不同任务，发送"继续 XXX 项目"，检查匹配到的是正确的任务。

---

## Phase 4: 环境快照最小实现 + 稳健性优化

### 4A: environment_snapshot.py

**文件**: `zulong/l2/environment_snapshot.py`

**位置**: `create_snapshot()` 方法第186-202行的 TODO 块

**改动**: 替换注释为最小可用实现（记录时间戳 + 基础用户状态），约 10 行。不集成视觉/听觉系统。

### 4B: TaskGraph.deserialize 数据校验

**文件**: `zulong/l2/task_graph.py`

**改动**: 在 `deserialize()` 中添加前置字段校验 + 后置悬挂边清理，约 10 行。

---

## 修改文件清单

| 文件 | 阶段 | 改动量 |
|------|------|--------|
| `zulong/l2/inference_engine.py` | Phase 1+2 | ~55 行 |
| `zulong/bootstrap.py` | Phase 3A | ~15 行 |
| `zulong/l1b/scheduler_gatekeeper.py` | Phase 3B | ~12 行 |
| `zulong/l2/environment_snapshot.py` | Phase 4A | ~10 行 |
| `zulong/l2/task_graph.py` | Phase 4B | ~10 行 |

---

## 验证方案

### 单元测试
```bash
python -m pytest tests/ -v
```
确认现有 42/45 测试继续通过（3 个 pipeline 导入失败是预存问题）。

### 集成测试（手动）

1. **启动验证**: 启动系统，检查日志中出现 sync_all、prune_loop、RecoveryNotifier 日志
2. **记忆注入验证**: 发送"我喜欢弹钢琴"，等 1 分钟后问"我有什么爱好？"，LLM 应正确回答
3. **冷记忆验证**: 发送多条不同话题消息，等待超过热窗口（30 分钟或临时调小），查询旧话题看是否通过 FAISS 检索命中
4. **任务恢复验证**: 创建复杂任务后中断，重启系统，发送"继续"，验证恢复通知和匹配
5. **修剪验证**: 等待 30 分钟或临时将 interval 改为 60 秒，确认衰减修剪日志出现

### 关键日志检查点

- `[MemoryGraph] 全量同步完成` (Phase 3A)
- `[MemoryGraph] 修剪循环已启动` (Phase 3A)
- `[DialogueAdapter] 对话摘要已索引到 FAISS` (Phase 1)
- `[MemoryGraph] 注入 X 条记忆到上下文` (Phase 2)
- `[RecoveryNotifier] 已通知用户` (Phase 3A)
