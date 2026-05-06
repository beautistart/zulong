# 确定性任务恢复机制重构

## Context

祖龙系统的任务恢复依赖 5 条启发式规则链（关键词 + 活跃TG + 未完成节点），在进程重启、任务已完成、IDE 新开窗口等场景下频繁失败。本次重构通过前端传递 `graph_id` 实现确定性恢复，利用三级加载（内存 → 磁盘 → MemoryGraph）100% 找到目标图谱，跳过所有启发式规则。

## 修改文件列表

| 文件 | 修改类型 |
|------|----------|
| `zulong/ide/ide_session.py` | 新增字段 |
| `zulong/ide/ide_server.py` | 新增函数 + 修改2个函数签名 |
| `zulong/ide/ide_fc_runner.py` | 修改 `_init_state` |
| `zulong-ide/src/core/api/transport/zulong-websocket.ts` | 修改方法签名 |
| `zulong-ide/src/core/api/providers/zulong.ts` | 新增提取逻辑 |

---

## Step 1: IDEFCState 新增字段

**文件**: `zulong/ide/ide_session.py` (第49行后)

在 `force_first_tool` 字段之后新增：

```python
force_graph_id: str = ""  # 确定性恢复: 前端传入的 graph_id，非空时跳过启发式
```

---

## Step 2: 后端新增确定性加载函数

**文件**: `zulong/ide/ide_server.py` (在 `_handle_session_resume` 函数前，约第184行)

新增函数：

```python
def _load_graph_deterministic(graph_id: str) -> bool:
    """确定性三级加载 TaskGraph: 内存 → 磁盘 → MemoryGraph
    
    Returns: True 表示加载成功并已设置为活跃图
    """
    from zulong.tools.task_tools import (
        get_active_task_graph, set_active_task_graph, load_graph_from_backup,
    )
    
    # Level 1: 内存匹配
    tg = get_active_task_graph()
    if tg and getattr(tg, 'id', '') == graph_id:
        logger.info(f"[ZulongIDE] 确定性恢复 Level 1 (内存): {graph_id}")
        return True
    
    # Level 2: 磁盘备份
    tg = load_graph_from_backup(graph_id)
    if tg:
        set_active_task_graph(tg, graph_id)
        logger.info(f"[ZulongIDE] 确定性恢复 Level 2 (磁盘): {graph_id}")
        return True
    
    # Level 3: MemoryGraph 重建
    try:
        from zulong.memory.memory_graph import get_memory_graph
        from zulong.memory.graph_adapters import rebuild_task_graph_from_memory
        mg = get_memory_graph()
        if mg:
            tg = rebuild_task_graph_from_memory(mg, graph_id)
            if tg:
                set_active_task_graph(tg, graph_id)
                logger.info(f"[ZulongIDE] 确定性恢复 Level 3 (MemoryGraph): {graph_id}")
                return True
    except Exception as e:
        logger.debug(f"[ZulongIDE] Level 3 MemoryGraph 重建失败: {e}")
    
    logger.warning(f"[ZulongIDE] 确定性恢复失败: 三级加载均未找到 {graph_id}")
    return False
```

---

## Step 3: 修改 `_handle_session_resume`

**文件**: `zulong/ide/ide_server.py` (第185-225行)

修改内容：
1. 第189行后新增: `graph_id = payload.get("graph_id", "")`
2. 替换第207-220行的恢复逻辑为新的分支结构：

```python
    # === 恢复活跃 TaskGraph ===
    graph_id = payload.get("graph_id", "")
    
    if graph_id:
        # 确定性恢复路径
        _load_graph_deterministic(graph_id)
    else:
        # 兼容旧逻辑: 从磁盘备份加载最近的图谱
        try:
            from zulong.tools.task_tools import (
                get_active_task_graph, load_latest_backup,
                set_active_task_graph,
            )
            if get_active_task_graph() is None:
                backup_tg, backup_gid = load_latest_backup()
                if backup_tg and backup_gid:
                    set_active_task_graph(backup_tg, backup_gid)
                    logger.info(f"[ZulongIDE] session_resume: 从备份恢复活跃图 {backup_gid}")
        except Exception as e:
            logger.debug(f"[ZulongIDE] session_resume: 备份恢复尝试失败: {e}")
```

3. 修改第224-225行，将 `graph_id` 传递给 `_run_fc_loop`:

```python
    session.fc_task = asyncio.create_task(
        _run_fc_loop(session, resume_text, cwd, ide_system_prompt,
                     force_graph_id=graph_id))
```

---

## Step 4: 修改 `_run_fc_loop` 签名

**文件**: `zulong/ide/ide_server.py` (第347-350行)

签名从：
```python
async def _run_fc_loop(
    session: IDESession, task_text: str, cwd: str,
    ide_system_prompt: str = "",
) -> None:
```

改为：
```python
async def _run_fc_loop(
    session: IDESession, task_text: str, cwd: str,
    ide_system_prompt: str = "",
    force_graph_id: str = "",
) -> None:
```

在第379行 `runner.cwd = cwd` 之后新增：
```python
        runner.force_graph_id = force_graph_id
```

---

## Step 5: 修改 `_init_state` — 确定性恢复快速路径

**文件**: `zulong/ide/ide_fc_runner.py` (第776-777行)

将：
```python
        # ── Layer 1: 意图检测启发式 ──────────────────────────
        intent, has_active_tg = self._detect_ide_intent(user_input)
```

替换为：
```python
        # ── Layer 1: 意图检测 ──────────────────────────────────
        _force_gid = getattr(self, 'force_graph_id', '') or ''
        if _force_gid:
            # 确定性恢复: graph_id 已由 ide_server 加载，跳过启发式
            from zulong.tools.task_tools import get_active_task_graph
            _tg = get_active_task_graph()
            if _tg and getattr(_tg, 'id', '') == _force_gid:
                intent = "resume"
                has_active_tg = True
                self.session.active_task_graph_id = _force_gid
                logger.info(f"[IDEFCRunner] 确定性恢复模式: graph_id={_force_gid}")
            else:
                # 活跃图加载失败(不应发生), 降级到启发式
                logger.warning(f"[IDEFCRunner] 确定性恢复降级: 活跃图不匹配 {_force_gid}")
                intent, has_active_tg = self._detect_ide_intent(user_input)
        else:
            intent, has_active_tg = self._detect_ide_intent(user_input)
```

同时删除第779-787行的旧关联逻辑（已移入上方 if 分支中）：
```python
        # 恢复模式：将全局活跃图 ID 关联到当前 session（跨连接保持关联）
        if intent == "resume" and has_active_tg:
            ...
```

替换为只在非确定性路径时执行的版本：
```python
        # 非确定性路径下，恢复模式关联活跃图到 session
        if not _force_gid and intent == "resume" and has_active_tg:
            from zulong.tools.task_tools import get_active_task_graph
            _tg = get_active_task_graph()
            if _tg and hasattr(_tg, 'id') and not self.session.active_task_graph_id:
                self.session.active_task_graph_id = getattr(_tg, 'id', None)
                logger.info(
                    f"[IDEFCRunner] 恢复模式：关联活跃图 "
                    f"{self.session.active_task_graph_id} 到新 session")
```

---

## Step 6: 前端 WebSocket 传输层增加 graphId

**文件**: `zulong-ide/src/core/api/transport/zulong-websocket.ts` (第158-164行)

从：
```typescript
sendSessionResume(task: string, cwd: string, zulongSystemPrompt?: string): void {
    this.send("session_resume", {
        task,
        cwd,
        ide_system_prompt: zulongSystemPrompt || "",
    })
}
```

改为：
```typescript
sendSessionResume(task: string, cwd: string, zulongSystemPrompt?: string, graphId?: string): void {
    const payload: Record<string, string> = {
        task,
        cwd,
        ide_system_prompt: zulongSystemPrompt || "",
    }
    if (graphId) {
        payload.graph_id = graphId
    }
    this.send("session_resume", payload)
}
```

---

## Step 7: 前端 Provider 提取 graph_id 并传递

**文件**: `zulong-ide/src/core/api/providers/zulong.ts`

在类中新增 private 方法：
```typescript
/**
 * 从历史消息中提取最近的 graph_id (格式: tg_NNNNNNNNNN)
 */
private extractGraphId(messages: ZulongStorageMessage[]): string | undefined {
    const pattern = /\btg_\d{10,13}\b/
    for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i]
        if (msg.role !== "assistant") continue
        const content = msg.content
        if (typeof content === "string") {
            const match = content.match(pattern)
            if (match) return match[0]
        } else if (Array.isArray(content)) {
            for (const block of content) {
                if (typeof block === "object" && block !== null) {
                    const text = (block as any).text || JSON.stringify(block)
                    const match = text.match(pattern)
                    if (match) return match[0]
                }
            }
        }
    }
    return undefined
}
```

修改第186-188行的调用（在 hasHistory 分支中）：
```typescript
if (hasHistory) {
    const graphId = this.extractGraphId(messages)
    Logger.info(`[ZulongHandler] -> session_resume, cwd=${cwd}, graph_id=${graphId || "none"}`)
    this.transport.sendSessionResume(taskText, cwd, systemPrompt, graphId)
}
```

---

## 向后兼容性

| 场景 | 行为 |
|------|------|
| 新前端 + 旧后端 | `graph_id` 字段被 Python `dict.get()` 忽略 |
| 旧前端 + 新后端 | payload 无 `graph_id`，走 fallback 启发式 |
| messages 中无 graph_id 文本 | `extractGraphId` 返回 undefined，不传该字段 |
| 三级加载均失败 | `_load_graph_deterministic` 返回 False，走 load_latest_backup fallback |

---

## 验证方法

1. **Python 语法检查**:
   ```bash
   python -m py_compile zulong/ide/ide_server.py
   python -m py_compile zulong/ide/ide_fc_runner.py
   python -m py_compile zulong/ide/ide_session.py
   ```

2. **TypeScript 检查**:
   ```bash
   cd zulong-ide && npx tsc --noEmit
   ```

3. **功能测试场景**:
   - 正常恢复（messages 中有 graph_id）: 验证日志 `确定性恢复 Level X`
   - 无 graph_id 场景: 验证走 fallback，日志无 `确定性恢复`
   - 进程重启后恢复: 杀后端 → 重启 → 从 IDE 恢复 → Level 2 磁盘加载
   - 已完成任务恢复: 所有节点 completed → 不再新建图谱

4. **日志关键字**:
   - 成功: `确定性恢复 Level 1/2/3`、`确定性恢复模式: graph_id=tg_xxx`
   - 降级: `确定性恢复降级`、`确定性恢复失败`

---

## 实施顺序

1. Step 1 (IDEFCState 新字段) — 无依赖
2. Step 2 (新增 `_load_graph_deterministic`) — 无依赖
3. Step 4 (`_run_fc_loop` 签名) — 依赖 Step 2
4. Step 3 (`_handle_session_resume` 修改) — 依赖 Step 2 + 4
5. Step 5 (`_init_state` 快速路径) — 依赖 Step 1
6. Step 6 (前端传输层) — 无依赖
7. Step 7 (前端 Provider) — 依赖 Step 6
