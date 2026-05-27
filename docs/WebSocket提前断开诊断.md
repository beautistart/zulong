# WebSocket提前断开导致task_complete丢失问题诊断

**诊断时间**：2026-05-18 16:40

---

## 一、问题现象

IDE界面显示：
- ✅ 欢迎语已显示（重复2次）
- ❌ 仍显示"思考中..."
- ❌ 底部无任务完成提示

---

## 二、日志关键证据

```log
16:40:35 [INFO] ✨ [FC] 前端流式推送完成：共 5 个句子
16:40:35 [INFO] display_text推送: streaming=False, complete=True  ← 完成标记
16:40:35 [INFO] TASK_COMPLETE → 3 个监控连接              ← 广播（监控）
16:40:35 [INFO] WebSocket 断开: session=d6eeb60fb839      ← 用户关闭
16:40:36 [WARNING] 兜底WS直发失败: type=task_complete       ← 发送失败
```

**关键时序**：
1. 16:40:35.000 - 流式推送欢迎语完成
2. 16:40:35.XXX - 发送display_text(complete=True)
3. 16:40:35.YYY - WebSocket断开（用户关闭浏览器）
4. 16:40:36.000 - 尝试发送task_complete → **失败**

---

## 三、根因分析

### 3.1 WebSocket断开时机

**证据**：task_complete发送失败，说明WebSocket在FC循环完成前就断开了

**原因推测**：
1. **前端行为**：收到display_text(complete=True)后认为任务完成，关闭WebSocket
2. **或者**：用户手动关闭浏览器/刷新页面
3. **或者**：前端超时机制触发，主动断开

### 3.2 消息发送机制分析

**消息流程**：
```
FC循环 → send_callback("task_complete")
         ↓
    session.send_msg() → outbound_queue.put()
         ↓
    _outbound_sender() → ws.send_json()
         ↓
    前端WebSocket.onmessage
```

**问题**：
- outbound_queue是异步队列，消息先放入队列
- _outbound_sender从队列取出并发送
- **WebSocket断开后，_outbound_sender无法发送，队列中的消息丢失**

---

## 四、为什么修复无效？

### 4.1 P0修复回顾

| 修复项 | 预期效果 | 实际效果 |
|--------|---------|---------|
| disconnected发送error chunk | 前端重置思考状态 | ❌ 未触发（Extension未收到disconnected事件） |
| PendingMessageCache缓存 | 断线后重连恢复 | ❌ 缓存了，但前端未重连 |

### 4.2 为什么disconnected事件未触发？

**Extension端监听**：`zulong.ts:208`
```typescript
this.transport.on("disconnected", (code, reason) => {
    pushChunk({ type: "error", error: `WebSocket断开` })
    pushChunk({ type: "done" })
})
```

**问题**：
- Extension的WebSocket在VS Code侧，不是在webview-ui侧
- 用户关闭的是webview-ui（前端），不是Extension
- Extension的WebSocket连接仍然存活

**架构图**：
```
Python后端 ←WebSocket→ Extension(VS Code侧) ←postMessage→ webview-ui(前端)
                ↑                                         ↑
            仍存活                                   用户关闭这里
```

### 4.3 为什么PendingMessageCache无效？

**缓存逻辑**：
```python
except Exception as e:
    PendingMessageCache.cache(session.session_id, msg)
```

**问题**：
- 消息已缓存（session_id=d6eeb60fb839）
- 但前端未重新连接同一个session_id
- 前端重新打开时，生成新的session_id，无法恢复旧缓存

---

## 五、真正的根因

**三层架构问题**：

1. **Extension与webview-ui分离**
   - Extension运行在VS Code Extension Host
   - webview-ui运行在独立WebView进程
   - 用户关闭webview-ui，不影响Extension WebSocket

2. **前端主动断开时机错误**
   - 前端收到display_text(complete=True)后立即断开
   - 未等待task_complete消息
   - 认为complete=True就代表任务全部完成

3. **消息发送顺序问题**
   - 先发送display_text(complete=True)
   - 后发送task_complete
   - 前端在收到前者后就关闭，后者无法送达

---

## 六、解决方案

### 方案A：合并完成消息（推荐）

**思路**：在display_text(complete=True)中包含task_complete信息

**位置**：`ide_fc_runner.py:500-505`

```python
# 当前代码
await send_callback("display_text", {
    "text": "", 
    "turn": state.fc_turn,
    "streaming": False,
    "complete": True
})

# 改进：合并task_complete
await send_callback("display_text", {
    "text": "", 
    "turn": state.fc_turn,
    "streaming": False,
    "complete": True,
    "task_result": final_text,  # 新增：任务结果
    "task_status": "completed"   # 新增：任务状态
})
```

**前端处理**：收到complete=True且有task_result时，认为任务完成

---

### 方案B：前端等待task_complete（需修改前端）

**思路**：前端不要在complete=True后立即断开，等待task_complete

**位置**：`webview-ui/src/components/chat/ChatView.tsx`

```typescript
// 当前：收到complete=True就认为完成
case "display_text":
    if (payload.complete) {
        // 认为任务完成
    }

// 改进：等待task_complete
case "display_text":
    if (payload.complete) {
        // 仅标记流式推送完成，不认为任务完成
        setIsStreaming(false)
    }

case "task_complete":
    // 真正的任务完成
    setIsThinking(false)
```

---

### 方案C：调整发送顺序（临时方案）

**思路**：先发送task_complete，后发送display_text(complete=True)

**位置**：`ide_fc_runner.py:547`

```python
# 先发送task_complete
await send_callback("task_complete", {"result": _done_text})

# 再发送display_text完成标记
await send_callback("display_text", {
    "text": "", 
    "turn": state.fc_turn,
    "streaming": False,
    "complete": True
})
```

---

## 七、推荐实施方案

**优先级排序**：

| 方案 | 效果 | 难度 | 风险 | 推荐 |
|------|------|------|------|------|
| 方案A：合并消息 | 根本解决 | 低 | 低 | ⭐⭐⭐ |
| 方案B：前端等待 | 根本解决 | 中 | 中 | ⭐⭐ |
| 方案C：调整顺序 | 临时缓解 | 低 | 中 | ⭐ |

**推荐**：实施**方案A**，合并display_text和task_complete消息，确保前端在收到完成标记时同时收到任务结果。

---

## 八、立即行动

### 8.1 快速验证（不改代码）

在前端添加日志，观察WebSocket断开时机：

```typescript
// zulong-websocket.ts
public close() {
    console.trace("[ZulongWS] WebSocket.close() 被调用")
    super.close()
}
```

### 8.2 实施方案A

修改后端发送逻辑，合并消息：

```python
# ide_fc_runner.py:500
await send_callback("display_text", {
    "text": "",
    "turn": state.fc_turn,
    "streaming": False,
    "complete": True,
    # 新增字段
    "task_result": state.last_response_content or "",
    "task_status": "completed"
})
```

---

**报告生成时间**：2026-05-18 16:45
