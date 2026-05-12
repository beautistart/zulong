# 任务文档：基于 ID 的确定性任务恢复机制重构

> 优先级: 高
> 影响范围: IDE 扩展 (TS) + Python 后端 + 前端仪表盘
> 前置依赖: 无（可独立实施）

---

## 一、背景与问题

### 1.1 现状

祖龙系统的任务恢复目前依赖**启发式规则链**，从 IDE 点击"恢复"到 LLM 真正在原图谱上继续，需要经过以下条件全部满足：

```
IDE 发送 session_resume (仅携带 task_text, 无 graph_id)
    → 后端构造 "继续之前的任务：{task_text}"
    → _detect_ide_intent() 启发式检测：
        Rule 0: session.fc_state.phase == "waiting_remote" AND has_active_tg
        Rule 1: get_active_task_graph() != None AND 有未完成叶节点
        Rule 1.5: 用户输入包含 @[label#address] 引用
        Rule 2: 包含"继续/恢复"等关键词 AND has_active_tg
    → _auto_create_task_plan() 判断：
        如果 existing_tg 所有叶节点 completed → 创建新图谱
        如果 is_resume=True → 复用（但前面的 Rule 需先判定为 resume）
```

### 1.2 失败场景（已验证）

| 场景 | 原因 | 结果 |
|------|------|------|
| Python 后端进程重启 | `_active_task_graph` 是内存变量，重启即 None | 所有 Rule 中 has_active_tg=False → COMPLEX → 新建图 |
| 任务已全部完成后恢复 | 所有叶节点 completed | Rule 1 不触发；_auto_create_task_plan 判定"已完成" → 新建 |
| IDE 新开窗口恢复 | 新 session 与旧 session 无关联 | 无法找到原图谱 |
| 多任务并发 | 全局单例 _active_task_graph 只有一个 | 切换任务时前一个丢失 |

### 1.3 生产日志证据

```
备份 tg_1778042526.json: 207节点, 206 completed, 1 in_progress(req根节点)
系统判定: 所有叶节点 completed → "旧图已完成" → 新建了 tg_1778060475
新图标题: "继续之前的任务：[TASK RESUMPTION] This task was interrupted..."
```

### 1.4 市面产品对比

| 产品 | 恢复机制 | 可靠性 |
|------|---------|--------|
| Cline | taskId → 加载 task 目录中的 JSON 文件 | 100% 确定 |
| Cursor | 无内建机制 (.brain/ 社区方案 = Markdown 文件) | 依赖 LLM |
| Windsurf | 本地 Memories + 相关性检索 | 黑盒 |
| **祖龙(当前)** | **5 条启发式规则 AND 链** | **极不稳定** |

---

## 二、设计目标

1. **确定性恢复**：给定 graph_id，任何状态（未完成/已完成/进程重启后）都能 100% 恢复
2. **保持记忆关联**：MemoryGraph 的 BFS/语义发现能力不受影响
3. **向后兼容**：旧的启发式逻辑作为 fallback（IDE 未传 graph_id 时）
4. **多任务支持**：不同 session 可绑定不同 graph_id，不互相覆盖

---

## 三、架构设计

### 3.1 数据模型（利用现有结构）

```
┌─────────────────────────────────────────────────────┐
│                  MemoryGraph                         │
│                                                     │
│  dialogue:session_abc ──REFERENCE──→ task:tg_123    │
│       │(TEMPORAL)                      │(HIERARCHY) │
│       ↓                                ↓            │
│  dialogue:session_def        task:tg_123/task:o1    │
│       │(REFERENCE)                     │            │
│       ↓                                ↓            │
│  task:tg_456              task:tg_123/task:o1_1     │
│                                                     │
└─────────────────────────────────────────────────────┘

持久化层:
  data/graph_backups/tg_123.json  ← 完整 TaskGraph 序列化
  data/graph_backups/tg_456.json
```

### 3.2 恢复主路径（新）

```
IDE "恢复任务" 按钮
    │
    │  payload: { task: "...", cwd: ".", graph_id: "tg_123" }
    │                                    ^^^^^^^^^^^^^^^^
    │                                    确定性锚点（新增）
    ↓
后端 _handle_session_resume(session, payload):
    graph_id = payload.get("graph_id")  # 优先使用
    
    if graph_id:
        # === 确定性恢复路径 ===
        tg = _deterministic_load(graph_id)
        #   Level 1: 内存 get_active_task_graph() 匹配 graph_id
        #   Level 2: 磁盘 load_graph_from_backup(graph_id)
        #   Level 3: MemoryGraph rebuild_task_graph_from_memory(mg, graph_id)
        set_active_task_graph(tg, graph_id)
        → intent = "resume" (强制，不走启发式)
        → force_first_tool = True
        → _auto_create_task_plan 完全跳过
    else:
        # === 兼容旧逻辑（fallback）===
        → 走现有启发式 Rules
```

### 3.3 辅助层：记忆关联发现（不变）

MemoryGraph 的能力完全保留，但**不参与恢复主路径**：

- **BFS 激活扩散**：从当前 session 出发 → TEMPORAL → 发现前序 session → REFERENCE → 发现其历史 task → 注意力注入相关上下文
- **语义检索**：用户说"之前那个 nginx 任务" → 语义索引匹配 → 返回 graph_id → LLM 调用 `task_resume_by_address` 切换
- **节点引用**：`@[label#tg:xxx/task:yyy]` → `_try_activate_from_reference` → 重建并激活

---

## 四、详细实施步骤

### Step 1: IDE 扩展 — sendSessionResume 增加 graph_id

**文件**: `zulong-ide/src/core/api/transport/zulong-websocket.ts`

**现状** (第 158-164 行):
```typescript
sendSessionResume(task: string, cwd: string, zulongSystemPrompt?: string): void {
    this.send("session_resume", {
        task,
        cwd,
        ide_system_prompt: zulongSystemPrompt || "",
    })
}
```

**改为**:
```typescript
sendSessionResume(task: string, cwd: string, zulongSystemPrompt?: string, graphId?: string): void {
    this.send("session_resume", {
        task,
        cwd,
        ide_system_prompt: zulongSystemPrompt || "",
        graph_id: graphId || "",  // 新增: 确定性恢复锚点
    })
}
```

**调用方** 需传入 graphId:

**文件**: `zulong-ide/src/core/api/providers/zulong.ts` (第 186-188 行)

现状:
```typescript
if (hasHistory) {
    this.transport.sendSessionResume(taskText, cwd, systemPrompt)
}
```

改为:
```typescript
if (hasHistory) {
    // 从 task metadata 或最近的 assistant message 中提取 graph_id
    const graphId = this.extractGraphId(messages)
    this.transport.sendSessionResume(taskText, cwd, systemPrompt, graphId)
}
```

**extractGraphId 实现思路**:
1. 查找 messages 中 tool_use name="task_create_plan" 的 response 中的 graph_id
2. 或查找 messages 中 tool_use name="task_view_overview" 的 response 中的 graph_id
3. 兜底: 查找 assistant message 中包含 `graph_id=tg_` 的文本

### Step 2: 后端 — _handle_session_resume 确定性加载

**文件**: `zulong/ide/ide_server.py`

替换 `_handle_session_resume` 中的恢复逻辑:

```python
async def _handle_session_resume(session: IDESession, payload: Dict) -> None:
    """处理会话恢复"""
    task_text = payload.get("task", "")
    cwd = payload.get("cwd", ".")
    ide_system_prompt = payload.get("ide_system_prompt", "")
    graph_id = payload.get("graph_id", "")  # 新增: 确定性锚点

    if not task_text:
        await session.send_msg("task_error", {"error": "resume task 不能为空"})
        return

    # 取消正在运行的 FC 循环
    if session.fc_task and not session.fc_task.done():
        session.cancel_event.set()
        try:
            await asyncio.wait_for(session.fc_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            session.fc_task.cancel()
    session.cancel_event.clear()

    # === 确定性恢复: 通过 graph_id 精确加载 ===
    if graph_id:
        try:
            tg = _deterministic_load_graph(graph_id)
            if tg:
                from zulong.tools.task_tools import set_active_task_graph
                set_active_task_graph(tg, graph_id)
                session.active_task_graph_id = graph_id
                logger.info(f"[ZulongIDE] 确定性恢复: graph_id={graph_id}")
        except Exception as e:
            logger.warning(f"[ZulongIDE] 确定性恢复失败: {graph_id}, {e}")
    else:
        # === 兼容旧逻辑: 尝试从最近备份加载 ===
        try:
            from zulong.tools.task_tools import (
                get_active_task_graph, load_latest_backup, set_active_task_graph
            )
            if get_active_task_graph() is None:
                backup_tg, backup_gid = load_latest_backup()
                if backup_tg and backup_gid:
                    set_active_task_graph(backup_tg, backup_gid)
        except Exception:
            pass

    resume_text = f"继续之前的任务：{task_text}"
    session.fc_task = asyncio.create_task(
        _run_fc_loop(session, resume_text, cwd, ide_system_prompt,
                     force_graph_id=graph_id))  # 传递 graph_id 到 FC 循环
```

**新增辅助函数**:

```python
def _deterministic_load_graph(graph_id: str):
    """确定性三级加载: 内存 → 磁盘 → MemoryGraph"""
    # Level 1: 内存
    from zulong.tools.task_tools import get_active_task_graph
    tg = get_active_task_graph()
    if tg and getattr(tg, 'id', '') == graph_id:
        return tg
    
    # Level 2: 磁盘备份
    from zulong.tools.task_tools import load_graph_from_backup
    tg = load_graph_from_backup(graph_id)
    if tg:
        return tg
    
    # Level 3: MemoryGraph 重建
    from zulong.memory.memory_graph import get_memory_graph
    from zulong.memory.graph_adapters import rebuild_task_graph_from_memory
    mg = get_memory_graph()
    tg = rebuild_task_graph_from_memory(mg, graph_id)
    return tg
```

### Step 3: FC Runner — 有 graph_id 时跳过启发式

**文件**: `zulong/ide/ide_fc_runner.py`

**_init_state 修改**:

```python
def _init_state(self, messages: List[Dict], force_graph_id: str = "") -> IDEFCState:
    # ... extract user_input ...
    
    if force_graph_id:
        # 确定性模式: 跳过所有启发式
        intent = "resume"
        has_active_tg = True
        logger.info(f"[IDEFCRunner] 确定性恢复模式: graph_id={force_graph_id}")
    else:
        # 兼容模式: 走原有启发式
        intent, has_active_tg = self._detect_ide_intent(user_input)
    
    # ... 其余不变 ...
```

**_auto_create_task_plan 修改**:

```python
def _auto_create_task_plan(self, state: IDEFCState) -> None:
    # 如果是确定性恢复（有 force_graph_id），完全跳过自动创建
    if getattr(state, '_force_graph_id', ''):
        return
    # ... 其余现有逻辑 ...
```

### Step 4: IDE 扩展 — 从历史消息提取 graph_id

**文件**: `zulong-ide/src/core/api/providers/zulong.ts`

```typescript
private extractGraphId(messages: any[]): string {
    // 从最近的 assistant 消息中寻找 graph_id
    for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i]
        if (msg.role !== "assistant") continue
        
        // 查找 tool_use 结果中的 graph_id
        const content = typeof msg.content === 'string' 
            ? msg.content 
            : JSON.stringify(msg.content)
        
        // 匹配 "graph_id": "tg_XXXXXXXXXX" 或 graph_id=tg_XXXXXXXXXX
        const match = content.match(/graph_id["\s:=]+["']?(tg_\d+)["']?/)
        if (match) return match[1]
    }
    return ""
}
```

### Step 5: Web 仪表盘 — 恢复按钮传递 graph_id

**文件**: `openclaw_bridge/web/static/index.html`

如果 Web 端也有恢复功能，确保 WebSocket 消息携带 graph_id:

```javascript
function resumeTask(graphId) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'session_resume',
            task: '继续任务',
            cwd: '.',
            graph_id: graphId  // 确定性锚点
        }));
    }
}
```

---

## 五、验证清单

- [ ] IDE 发送 session_resume 携带 graph_id
- [ ] 后端收到 graph_id 后精确加载（不走启发式）
- [ ] 进程重启后恢复：从磁盘 backup 加载
- [ ] 已完成任务恢复：不再新建图谱
- [ ] 无 graph_id 时兼容旧逻辑（fallback）
- [ ] MemoryGraph BFS 仍能发现关联任务
- [ ] task_resume_by_address 工具仍可用
- [ ] Web 仪表盘恢复功能正常
- [ ] 多 session 并发不冲突

---

## 六、风险与注意事项

1. **graph_id 提取失败**：如果 IDE 历史消息中找不到 graph_id（如首次 session），会 fallback 到旧逻辑，不会崩溃
2. **备份文件被删除**：Level 2 失败 → 尝试 Level 3 (MemoryGraph 重建)；都失败 → 报错给用户
3. **IDEFCState 新字段**：需要在 `IDEFCState` dataclass 中增加 `force_graph_id: str = ""` 字段
4. **_run_fc_loop 签名变更**：增加 `force_graph_id` 参数，需要检查所有调用方

---

## 七、与现有修复的关系

本方案是对以下已实施修复的**架构级替代**（实施后可移除）：

| 已实施修复 | 状态 |
|-----------|------|
| load_latest_backup() 启动时加载 | → 降级为 fallback |
| _detect_ide_intent Rule 2 备份恢复 | → 有 graph_id 时跳过 |
| _auto_create_task_plan is_resume 分支 | → 有 graph_id 时完全跳过 |
| bind_session_to_task() 调用 | → 保留（关联发现层仍需要） |
| rebuild_task_graph_from_memory() | → 保留为 Level 3 fallback |
| TaskResumeByAddressTool | → 保留（LLM 自主发现路径） |
| TaskReviseNodeTool | → 保留（修订已完成节点） |

---

## 八、实施顺序建议

1. **Step 2 (后端)** → 最先实施，可以独立测试（graph_id 为空时走旧逻辑）
2. **Step 3 (FC Runner)** → 紧跟 Step 2
3. **Step 1 + Step 4 (IDE 扩展)** → 需要 TypeScript 构建环境
4. **Step 5 (Web 仪表盘)** → 可选，Web 端恢复优先级低
5. **验证** → 按清单逐项测试
