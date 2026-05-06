# 三大问题修复：任务恢复 + 取消机制 + 汇报机制

## Context

生产环境中发现三个严重问题：
1. 任务异常中断(API 429/502)后，重新连接时 LLM 创建全新任务节点，不继续原任务
2. IDE 取消按钮无效（模型在线程池中阻塞），Web 端停止命令无效（无对应 API 端点）
3. 汇报机制不生效（任务仅跑 44 轮就崩溃，远未到 hard_limit=100；且 Web 端无 PROGRESS_REPORT 处理器）

---

## 问题 1: 任务恢复创建新节点

### 根因

WebSocket 断开后重连：
1. `ide_server.py` 为新连接创建**全新** `AgentSession`（`active_task_graph_id = None`）
2. 旧 session 被 `_sessions.pop()` 移除，其 `active_task_graph_id` 丢失
3. 模块级全局 `_active_task_graph` 仍然存活（内存中）
4. `_detect_ide_intent()` 正确检测到活跃图 → 返回 "resume"
5. **但** `_auto_create_task_plan()` 中：
   - 第一个检查 `self.session.active_task_graph_id` 为 None → 跳过
   - 第二个检查 `uncompleted` 节点 → **应该**命中 blocked 节点 → 返回"不覆盖"
6. **可能的失败原因**: 如果 `_mark_unfinished_nodes_blocked` 没有成功标记（如异常），或图对象被 GC

### 修复方案

**文件**: `zulong/ide/ide_fc_runner.py`

在 `_init_state()` 中，当检测到 resume 意图时，主动将全局活跃图的 ID 关联到新 session：

```python
def _init_state(self, messages: List[Dict]) -> IDEFCState:
    ...
    intent, has_active_tg = self._detect_ide_intent(user_input)
    
    # 新增：恢复模式时，将活跃图 ID 关联到当前 session
    if intent == "resume" and has_active_tg:
        from zulong.tools.task_tools import get_active_task_graph
        tg = get_active_task_graph()
        if tg and hasattr(tg, 'id'):
            self.session.active_task_graph_id = getattr(tg, 'id', None)
    ...
```

**文件**: `zulong/ide/ide_fc_runner.py` → `_auto_create_task_plan()`

加强防御：即使 session 没有关联 ID，只要全局图有未完成节点就不覆盖：

```python
if existing_tg:
    # 检查 1: session 已关联（快速路径）
    if (self.session.active_task_graph_id
            and hasattr(existing_tg, 'id')
            and getattr(existing_tg, 'id', '') == self.session.active_task_graph_id):
        return
    # 检查 2: 存在未完成/blocked 节点 → 不覆盖
    leaves = existing_tg.get_leaf_nodes()
    uncompleted = [n for n in leaves
                   if n.status not in ("completed", "skipped")]
    if uncompleted:
        # 主动关联到当前 session
        if hasattr(existing_tg, 'id'):
            self.session.active_task_graph_id = getattr(existing_tg, 'id', '')
        logger.info(f"[IDEFCRunner] 已有活跃任务图（{len(uncompleted)} 未完成），复用")
        return
```

---

## 问题 2: 取消/停止机制失效

### 根因

**IDE 取消**：
- `cancel_event.set()` 正确设置
- 但 `_call_model()` 在线程池中同步阻塞 `future.result(timeout=600s)`
- cancel_event 只在循环顶部检查，模型调用期间无法中断

**Web 停止**：
- `/monitor` WebSocket 只处理 `ping`、`REQUEST_MEMORY_GRAPH`、`EXPAND_NODE`
- **完全没有** stop/cancel 消息处理器
- Web 发停止命令无处可去

### 修复方案

#### 2A: IDE 取消 - 缩短模型阻塞窗口

**文件**: `zulong/ide/ide_fc_runner.py` → `run_loop_async()`

在模型调用时使用短超时轮询 + cancel_event 检查：

```python
# 替换原有的单次 run_in_executor 调用
model_future = loop.run_in_executor(None, self._call_model, state)
while True:
    if cancel_event.is_set():
        model_future.cancel()
        return await loop.run_in_executor(
            None, self._finalize, state, "cancelled")
    try:
        tc_data, resp_content = await asyncio.wait_for(
            asyncio.shield(model_future), timeout=2.0)
        break
    except asyncio.TimeoutError:
        continue  # 每 2 秒检查一次 cancel_event
```

#### 2B: Web 停止 - 添加消息处理器

**文件**: `zulong/ide/ide_server.py` → `/monitor` WebSocket handler

在 `_handle_monitor_ws` 中添加 STOP_TASK 处理：

```python
elif msg_type == "STOP_TASK":
    # 设置所有活跃 session 的 cancel_event
    for sid, sess in _sessions.items():
        if hasattr(sess, 'cancel_event') and sess.cancel_event:
            sess.cancel_event.set()
            logger.info(f"[ZulongIDE] Web停止: session={sid[:12]}")
    # 设置引擎级中断标志（_check 方法会检测）
    if hasattr(engine, '_interrupt_flag'):
        engine._interrupt_flag = True
    await ws.send_json({"type": "STOP_ACK", "payload": {"stopped": True}})
```

**文件**: `openclaw_bridge/web/static/index.html`

Web 端停止按钮发送正确的消息类型：

```javascript
function stopTask() {
    if (monitorWs && monitorWs.readyState === WebSocket.OPEN) {
        monitorWs.send(JSON.stringify({type: "STOP_TASK", payload: {}}));
    }
}
```

---

## 问题 3: 汇报机制不生效

### 根因

两层问题：
1. **任务未到 hard_limit**: 默认 `_hard_limit=100`，任务仅跑 44 轮就 API 错误退出 → 报告条件 `fc >= _hard_limit` 从未满足
2. **Web 无处理器**: 即使触发了 PROGRESS_REPORT 广播，Web 端 `index.html` 的 switch 没有对应 case → 静默丢弃

### 修复方案

#### 3A: 添加周期性汇报（独立于 hard_limit）

**文件**: `zulong/ide/ide_fc_runner.py` → `_check()`

在 `_check()` 中增加独立的周期汇报逻辑（每 N 轮广播一次状态，不影响 FC 循环控制流）：

```python
def _check(self, state: IDEFCState) -> str:
    state.fc_turn += 1
    fc = state.fc_turn
    ...
    # 新增：周期性进度广播（每 _progress_report_interval 轮）
    # 独立于 hard_limit，仅广播不控制循环
    if (fc > 1 
            and fc % self._progress_report_interval == 0
            and fc < self._hard_limit):
        self._broadcast_periodic_progress(state)
    ...
```

新增方法：
```python
def _broadcast_periodic_progress(self, state: IDEFCState) -> None:
    """周期性进度广播（不触发续期，仅通知 Web 端当前状态）"""
    from zulong.tools.task_tools import get_active_task_graph
    tg = get_active_task_graph()
    report = {"turn": state.fc_turn, "type": "periodic"}
    if tg:
        all_nodes = [n for n in tg.nodes.values() if n.id != "req"]
        report["total_nodes"] = len(all_nodes)
        report["completed_count"] = sum(
            1 for n in all_nodes if n.status in ("completed", "skipped"))
        report["in_progress_count"] = sum(
            1 for n in all_nodes if n.status == "in_progress")
        report["pending_count"] = sum(
            1 for n in all_nodes if n.status in ("pending", ""))
    _broadcast_sync("PROGRESS_REPORT", {
        "session_id": self.session.session_id,
        "turn": state.fc_turn,
        "report": report,
        "type": "periodic",
    })
```

#### 3B: Web 端添加 PROGRESS_REPORT 处理器

**文件**: `openclaw_bridge/web/static/index.html`

在 WebSocket 消息 switch 中添加：

```javascript
case 'PROGRESS_REPORT':
    var p = data.payload || {};
    var r = p.report || {};
    var text = '[进度 T' + (p.turn||'?') + '] ' +
        '完成:' + (r.completed_count||0) + ' 进行中:' + (r.in_progress_count||0) +
        ' 待处理:' + (r.pending_count||0) + ' 总计:' + (r.total_nodes||0);
    if (p.auto_continue_count) {
        text += ' (续期#' + p.auto_continue_count + ')';
    }
    addIdeSystemMessage(text);
    break;
```

---

## 关键文件清单

| 文件 | 改动内容 |
|------|---------|
| `zulong/ide/ide_fc_runner.py` | 问题1: _init_state 关联图ID; 问题2: 模型调用轮询取消; 问题3: 周期性广播 |
| `zulong/ide/ide_server.py` | 问题2: /monitor 添加 STOP_TASK 处理 |
| `openclaw_bridge/web/static/index.html` | 问题2: 停止按钮消息; 问题3: PROGRESS_REPORT handler |

---

## 验证方法

### 问题 1 验证
1. 启动任务 → 手动制造 API 错误(断网) → FC 终止
2. 重连 WebSocket → 发新消息
3. 验证日志出现 "已有活跃任务图（N 未完成），复用"
4. 验证模型调用 `task_view_overview` 看到 blocked 节点
5. 验证不出现新的 `tg_` 创建日志

### 问题 2 验证
1. 启动长任务 → 在模型调用中点 IDE 取消
2. 验证 2 秒内日志出现 "cancelled" 终止
3. Web 端点停止 → 验证日志出现 "Web停止"
4. 验证 FC 循环实际停止（无后续 TOOL_CALL 日志）

### 问题 3 验证
1. 启动任务（_progress_report_interval=30）
2. 观察 Web 端：第 30 轮出现进度条/消息
3. 如果任务到 hard_limit(100) → 验证续期报告也出现
4. 确认 api_error 前的 44 轮中至少看到一次周期性广播（第 30 轮）
