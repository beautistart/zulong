# Zulong Web 监控前端集成计划

## Context

用户需要一个 Web 前端来实时监控祖龙系统运行状态，包括：
- 会话节点的创建过程
- 任务图谱的创建和状态变更
- 记忆机制的运转（BFS 激活、赫布学习、衰减等）

现有资源：
- `openclaw_bridge/web/static/index.html` (4581行) — 成熟的可视化前端，包含 D3.js 记忆图 + Dagre 任务图 + 思维窗口
- `openclaw_bridge/adapters/web_adapter.py` (820行) — FastAPI WebSocket 广播框架
- `zulong/cline/cline_ide_server.py` — 已运行在 8090 端口的 FastAPI 应用
- `zulong/memory/memory_graph.py` — 已有 `_pending_changes` 增量变更追踪机制
- `zulong/core/event_bus.py` — 事件总线（优先级队列 + 后台分发）

## 架构设计

```
┌─────────────────────┐     ┌───────────────────────┐
│  IDE 插件 (VS Code) │     │  Web 监控前端 (浏览器)│
│  ws://...:8090/ide  │     │  http://...:8090/     │
└────────┬────────────┘     └──────────┬────────────┘
         │                             │
         │ tool_request/result         │ monitor events
         │                             │ (WebSocket /monitor)
         ▼                             ▼
┌──────────────────────────────────────────────────────┐
│            cline_ide_server.py (FastAPI :8090)        │
│                                                      │
│  /ide        — IDE WebSocket (现有，不变)            │
│  /monitor    — Web 监控 WebSocket (新增)             │
│  /health     — 健康检查 (现有)                       │
│  /           — 静态前端 HTML (新增)                   │
│  /static/... — 静态资源 (新增)                       │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│          InferenceEngine / ClineFCRunner              │
│                                                      │
│  事件点:                                             │
│  - FC turn 开始/结束                                 │
│  - 工具调用/完成                                     │
│  - TaskGraph 节点状态变更                            │
│  - MemoryGraph 增量更新 (_pending_changes)           │
│  - AttentionWindow 模式切换                          │
│  - CircuitBreaker 状态变更                           │
└──────────────────────────────────────────────────────┘
```

## 实施步骤

### Phase 1: 最小可用 Web 监控 (MVP)

在 `cline_ide_server.py` 中扩展，不新建文件：

**1.1 添加静态文件服务和根路由**

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 挂载静态前端
app.mount("/static", StaticFiles(directory="openclaw_bridge/web/static"), name="static")

@app.get("/")
async def root():
    return FileResponse("openclaw_bridge/web/static/index.html")
```

**1.2 添加 /monitor WebSocket 端点**

```python
_monitor_connections: Set[WebSocket] = set()

@app.websocket("/monitor")
async def monitor_websocket(ws: WebSocket):
    await ws.accept()
    _monitor_connections.add(ws)
    # 发送当前状态快照
    await ws.send_json({"type": "WELCOME", "active_sessions": list(_sessions.keys())})
    try:
        while True:
            msg = await ws.receive_text()  # 保活 + 命令处理
            data = json.loads(msg)
            if data.get("type") == "SUBSCRIBE":
                pass  # 未来: 细粒度事件订阅
    except WebSocketDisconnect:
        pass
    finally:
        _monitor_connections.discard(ws)
```

**1.3 广播辅助函数**

```python
async def broadcast_monitor_event(event_type: str, payload: dict):
    """向所有 Web 监控连接广播事件"""
    if not _monitor_connections:
        return
    msg = {
        "type": event_type,
        "ts": time.time(),
        "payload": payload,
    }
    dead = set()
    for ws in _monitor_connections:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.add(ws)
    _monitor_connections -= dead
```

### Phase 2: FC 循环事件注入

在 `cline_fc_runner.py` 的关键点调用 `broadcast_monitor_event`：

| 注入点 | 事件类型 | 数据 |
|--------|----------|------|
| `run_loop_async()` 开始 | `fc_start` | session_id, user_input |
| `_call_model()` | `calling_model` | turn, messages_count |
| `_exec_tools_async()` 每个工具 | `tool_call` | tool_name, arguments |
| 工具完成 | `tool_result` | tool_name, status |
| `_check()` turn 递增 | `turn_complete` | turn, phase |
| `_finalize()` | `fc_done` | turn, reason |

### Phase 3: 任务图谱实时推送

利用 TaskGraph 已有的 `to_frontend_dict()` 方法：

```python
# 在 task_tools.py 中状态变更后
async def publish_task_graph_update():
    tg = get_active_task_graph()
    if tg:
        await broadcast_monitor_event("TASK_GRAPH_UPDATE", {
            "graph": tg.to_frontend_dict(),
            "active_node_id": tg.get_active_node_id(),
        })
```

### Phase 4: 记忆图谱实时推送

利用 MemoryGraph 已有的 `_pending_changes` 机制：

```python
# 在 memory_graph.py 的 _auto_save() 中
async def flush_and_broadcast_changes():
    changes = self._pending_changes.copy()
    self._pending_changes.clear()
    if changes:
        await broadcast_monitor_event("MEMORY_GRAPH_UPDATE", {
            "update_type": "incremental",
            "changes": changes,
        })
```

### Phase 5: 前端适配

修改 `openclaw_bridge/web/static/index.html` 的 WebSocket 连接地址：
- 从 `ws://host:port/ws` 改为 `ws://host:8090/monitor`
- 消息类型映射保持兼容（MEMORY_GRAPH_UPDATE、THINKING_STEP 等已定义）

## 需要修改/创建的文件

| 文件 | 操作 | 改动量 |
|------|------|--------|
| `zulong/cline/cline_ide_server.py` | 修改: 添加 /monitor 端点、静态服务、广播函数 | ~80行 |
| `zulong/cline/cline_fc_runner.py` | 修改: 关键点注入事件发布 | ~50行 |
| `openclaw_bridge/web/static/index.html` | 修改: WebSocket URL + 适配消息类型 | ~20行 |
| `config/zulong_config.yaml` | 修改: 添加 monitor 配置节 | ~10行 |

## 前端可视化能力（已实现，可直接复用）

| 组件 | 技术 | 功能 |
|------|------|------|
| 记忆图谱 | D3.js 力导向 | 9种节点×7种边，分层折叠，实时激活动画 |
| 任务图谱 | Dagre 分层 | 节点状态着色，依赖关系虚线，交互选中 |
| 思维窗口 | D3.js 浮动 | 实时推理步骤可视化，节点脉冲效果 |
| 聊天区 | Vanilla JS | 流式响应，会话管理，@节点引用 |

## 验证方式

1. 启动祖龙后端: `python -m zulong.cline.cline_ide_server`
2. 浏览器访问 `http://localhost:8090/` — 应显示可视化前端
3. IDE 插件发起一个任务
4. Web 前端实时显示:
   - FC 循环进度（turn、phase）
   - 任务图谱节点创建和状态变更
   - 记忆图谱节点激活和边关系变化
   - 思考步骤可视化

## 注意事项

- `/ide` 端点保持不变，IDE 插件无需任何修改
- Web 前端是只读监控，不参与工具执行循环
- 广播使用 fire-and-forget 模式，Web 断连不影响 FC 循环
- 首次实现优先保证 MVP 可用，后续再增加细粒度订阅和历史回放
