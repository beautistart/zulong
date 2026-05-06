# IDE 会话内容转发到 Web 端

## Context

当前祖龙系统中，IDE 插件 (VS Code) 通过 `/ide` WebSocket 运行 FC 循环，产生的对话文本 (`display_text`)、工具调用详情 (`tool_request`)、工具结果等只发送给 IDE，Web 仪表盘 (`/ws`) 无法看到。

虽然 `broadcast_monitor_event` 已经向 Web 端广播了 FC_START、TOOL_CALL、MODEL_RESPONSE 等监控事件，但：
1. Web 前端 `handleMessage` 未处理这些事件类型（都落入 `default: console.log`）
2. `MODEL_RESPONSE` 只包含 200 字符预览，不是完整文本
3. 工具调用详情（参数、结果）未广播
4. `TASK_GRAPH_UPDATE` 事件已广播但前端缺少对应 case 处理

目标：让 Web 端实时展示 IDE 会话的完整内容，包括模型回复、工具调用/结果、任务图谱更新。

## 修改文件

| 文件 | 修改类型 |
|------|----------|
| `zulong/cline/cline_fc_runner.py` | 丰富已有广播 + 新增广播 |
| `zulong/cline/cline_ide_server.py` | 新增会话生命周期广播 |
| `openclaw_bridge/web/static/index.html` | 新增事件处理 + UI 渲染 |

## 一、后端修改：cline_fc_runner.py

### 1.1 丰富 FC_START（~line 214）

在现有 payload 中添加 `user_input`：

```python
await broadcast_monitor_event("FC_START", {
    "session_id": self.session.session_id,
    "max_turns": self._max_fc_turns,
    "intent": getattr(state, "cline_intent", ""),
    "user_input": (state.user_input_text or "")[:500],  # NEW
})
```

### 1.2 丰富 MODEL_RESPONSE（~line 287）

将完整文本（限 5000 字符）加入：

```python
await broadcast_monitor_event("MODEL_RESPONSE", {
    "session_id": self.session.session_id,  # NEW
    "turn": state.fc_turn,
    "text": (resp_content or "")[:5000],    # NEW: 完整文本
    "text_preview": (resp_content or "")[:200],
    "text_length": len(resp_content or ""),
})
```

### 1.3 新增 IDE_TOOL_REQUEST（~line 452 之后）

在 `send_callback("tool_request", ...)` 之后添加：

```python
await broadcast_monitor_event("IDE_TOOL_REQUEST", {
    "session_id": self.session.session_id,
    "turn": fc,
    "tools": [
        {
            "name": tc["function"]["name"],
            "arguments_preview": tc["function"].get("arguments", "")[:300],
            "call_id": tc.get("id", ""),
        }
        for tc in valid_remote
    ],
})
```

### 1.4 新增 IDE_TOOL_RESULT（~line 483 之后，`_inject_tool_results` 之后）

在 `self._inject_tool_results(state, formatted_results)` 之后添加：

```python
await broadcast_monitor_event("IDE_TOOL_RESULT", {
    "session_id": self.session.session_id,
    "turn": fc,
    "results": [
        {
            "tool_name": results[i].get("tool_name", ""),
            "call_id": results[i].get("call_id", ""),
            "result_preview": (results[i].get("result", "") or "")[:500],
            "result_length": len(results[i].get("result", "") or ""),
            "is_error": results[i].get("is_error", False),
        }
        for i in range(len(results))
    ],
})
```

### 1.5 新增内部工具执行广播（~line 371 之后）

在内部工具 `_exec_internal` 执行后添加（由于在 run_in_executor 中执行，使用 `_broadcast_sync`）：

在 `_exec_internal` 方法末尾（~line 1135 附近），添加调用：

```python
_broadcast_sync("IDE_TOOL_EXEC", {
    "session_id": self.session.session_id,
    "turn": fc_turn,
    "tool_name": tool_name,
    "arguments_preview": (args_str or "")[:300],
    "result_preview": (result_str or "")[:500],
    "is_internal": True,
})
```

## 二、后端修改：cline_ide_server.py

### 2.1 新增 IDE_SESSION_START（~line 137，`_handle_session_start` 中创建 fc_task 后）

```python
await broadcast_monitor_event("IDE_SESSION_START", {
    "session_id": session.session_id,
    "task_preview": task_text[:200],
    "cwd": cwd,
})
```

### 2.2 新增 IDE_SESSION_END（~line 357-368，`_run_fc_loop` 的 try/except 分支中）

在正常完成时（line 358-361 之后）：
```python
await broadcast_monitor_event("IDE_SESSION_END", {
    "session_id": session.session_id,
    "result": "completed",
})
```

在异常和取消时也各加一条类似广播。

## 三、前端修改：index.html

### 3.1 在 handleMessage switch 中新增 case（~line 2170 `default` 之前）

```javascript
case 'IDE_SESSION_START':
    handleIdeSessionStart(data);
    break;
case 'FC_START':
    handleFCStart(data);
    break;
case 'CALLING_MODEL':
    handleCallingModel(data);
    break;
case 'TOOL_CALL':
    handleToolCall(data);
    break;
case 'IDE_TOOL_REQUEST':
    handleIdeToolRequest(data);
    break;
case 'IDE_TOOL_RESULT':
    handleIdeToolResult(data);
    break;
case 'IDE_TOOL_EXEC':
    handleIdeToolExec(data);
    break;
case 'MODEL_RESPONSE':
    handleModelResponse(data);
    break;
case 'TURN_COMPLETE':
    handleTurnComplete(data);
    break;
case 'FC_DONE':
    handleFCDone(data);
    break;
case 'IDE_SESSION_END':
    handleIdeSessionEnd(data);
    break;
case 'TASK_GRAPH_UPDATE':
    handleTaskGraphUpdate(data);
    break;
```

### 3.2 新增 CSS 样式（`<style>` 块中）

```css
/* IDE 会话系统消息 */
.message.ide-session {
    background: #f0f4ff;
    border-left: 3px solid #4a7dff;
    padding: 8px 12px;
    margin: 8px 0;
    font-size: 13px;
    color: #555;
}
/* IDE 助手消息（区别于 Web 对话） */
.message.ide-assistant .message-content {
    border-left: 3px solid #4a7dff;
    padding-left: 10px;
}
/* IDE 标识 */
.ide-badge {
    display: inline-block;
    background: #e0e7ff;
    color: #4a7dff;
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 8px;
    margin-right: 6px;
}
/* 工具调用卡片 */
.tool-card {
    background: #1e1e2e;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    font-family: monospace;
    font-size: 12px;
    color: #ccc;
}
.tool-card .tool-name { color: #82aaff; font-weight: bold; }
.tool-card .tool-args { color: #a0a0a0; white-space: pre-wrap; max-height: 100px; overflow: hidden; }
.tool-card .tool-result { color: #c3e88d; white-space: pre-wrap; max-height: 150px; overflow: hidden; }
.tool-card .tool-error { color: #ff5370; }
.tool-card .expand-btn { color: #82aaff; cursor: pointer; font-size: 11px; }
```

### 3.3 新增 JavaScript 函数

**核心渲染函数：**

- `addIdeMessage(html, cssClass)` — 在聊天区域添加 IDE 消息（不保存到会话历史），自带 IDE 标识徽章
- `handleIdeSessionStart(data)` — 显示 "IDE 会话启动" 横幅，含任务摘要
- `handleFCStart(data)` — 显示 FC 循环开始信息（意图、最大轮次）
- `handleCallingModel(data)` — 显示 "正在调用模型 (Turn N)" 轻量提示
- `handleToolCall(data)` — 显示工具调用摘要（工具名列表）
- `handleIdeToolRequest(data)` — 渲染工具卡片：名称 + 参数预览（可展开）
- `handleIdeToolResult(data)` — 渲染工具结果卡片：结果预览（可展开）
- `handleIdeToolExec(data)` — 渲染内部工具执行（单行简洁格式）
- `handleModelResponse(data)` — 显示完整模型回复（Markdown 渲染 + IDE 标识）
- `handleTurnComplete(data)` — 移除 typing indicator
- `handleFCDone(data)` — 显示 "IDE 会话完成 (N 轮)" 横幅
- `handleIdeSessionEnd(data)` — 显示会话结束横幅
- `handleTaskGraphUpdate(data)` — 调用 `addTaskGraph(data.payload.graph)`

### 3.4 WELCOME 增强

在现有 `case 'WELCOME':` 处理中，如果 `data.payload.task_graph` 存在，调用 `addTaskGraph(data.payload.task_graph)`。

## 四、事件格式汇总

| 事件类型 | 来源 | 新增/修改 | 关键 payload 字段 |
|----------|------|-----------|-------------------|
| `IDE_SESSION_START` | ide_server | 新增 | session_id, task_preview, cwd |
| `FC_START` | fc_runner | 修改 | +user_input |
| `CALLING_MODEL` | fc_runner | 不变 | turn, model |
| `TOOL_CALL` | fc_runner | 不变 | turn, tools, count |
| `IDE_TOOL_REQUEST` | fc_runner | 新增 | session_id, turn, tools[{name, arguments_preview, call_id}] |
| `IDE_TOOL_RESULT` | fc_runner | 新增 | session_id, turn, results[{tool_name, result_preview, is_error}] |
| `IDE_TOOL_EXEC` | fc_runner | 新增 | session_id, turn, tool_name, result_preview, is_internal |
| `MODEL_RESPONSE` | fc_runner | 修改 | +session_id, +text(完整) |
| `TURN_COMPLETE` | fc_runner | 不变 | turn, has_tool_calls, tool_names |
| `FC_DONE` | fc_runner | 不变 | session_id, total_turns, reason |
| `IDE_SESSION_END` | ide_server | 新增 | session_id, result |
| `TASK_GRAPH_UPDATE` | ide_server | 不变 | event, detail, graph |

## 五、验证方式

1. 启动祖龙后端 (`python start.py`)
2. 打开 Web 仪表盘 (浏览器访问 localhost:8090)
3. 在 IDE 插件中发起一个任务（如 "帮我创建一个 hello world 文件"）
4. 验证 Web 端：
   - 出现 "IDE 会话启动" 横幅
   - 实时显示模型回复文本
   - 实时显示工具调用卡片（工具名 + 参数）
   - 实时显示工具执行结果
   - 任务图谱面板更新（如果有 TASK_GRAPH_UPDATE）
   - 出现 "IDE 会话完成" 横幅
5. TypeScript 检查不涉及（修改的都是 Python + HTML）
