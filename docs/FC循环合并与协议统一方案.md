# 祖龙FC循环精简合并与通信协议统一方案

> 生成日期: 2026-05-20
> 来源: Qoder 深度分析对话

---

## 一、现状：三套FC循环 + 两套协议

```
┌─────────────────────────────────────────────────────────────────────┐
│  FC循环现状 (3套, ~7800行)                                            │
│                                                                     │
│  ① fc_graph.py (1544行) ───── 半废弃                                  │
│     ├── LangGraph StateGraph构建 ← 已废弃, ~1100行可删               │
│     └── 4个节点工厂函数 ← 仍被 unified_fc_runner 复用                  │
│                                                                     │
│  ② unified_fc_runner.py (221行) ── L2推理使用                         │
│     ├── while循环: check→call_model→exec_tools/eval_response         │
│     └── 完全复用 fc_graph 的4个工厂函数                                │
│                                                                     │
│  ③ ide_fc_runner.py (4061行) ── IDE插件使用                           │
│     ├── while循环: 独立实现全部逻辑（不复用fc_graph）                    │
│     ├── 额外: 流式推送/工具分流/暂停恢复/TTS/4额外安全网                │
│     └── 与 fc_graph 工厂函数有 80% 语义重复                            │
│                                                                     │
│  问题: ide_fc_runner 在没有导入 fc_graph 的情况下，用完全独立的代码     │
│       实现了 ~80% 相同的安全网逻辑。维护两套同等语义但不同代码的安全网。  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  协议现状 (2套)                                                       │
│                                                                     │
│  Web端 ws://.../ws:                                                  │
│    CHAT_MESSAGE/CHAT_RESPONSE/STREAMING_RESPONSE/...                 │
│    无统一帧格式, 命名: 大写+下划线                                     │
│    26种消息类型                                                       │
│                                                                     │
│  IDE端 ws://.../ide:                                                 │
│    session_start/tool_request/task_complete/...                      │
│    统一帧格式: {msg_id, type, session_id, ts, payload}               │
│    命名: 小写+下划线, 12种消息类型                                     │
│                                                                     │
│  问题: 两套命名风格、帧格式、流式策略完全不同                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、FC循环合并方案

### 目标架构

```
合并后: 1套核心 + 2个薄适配层 (~3200行, 减少4600行)

┌──────────────────────────────────────────────────────┐
│  zulong/l2/fc_nodes.py (~1100行, 从fc_graph提取)     │
│  4个节点工厂函数 (纯函数, 无状态)                      │
│  ┌─────────────────────────────────────────┐         │
│  │ _make_check_node      (步数/限制/中断)    │         │
│  │ _make_call_model_node (LLM调用/备用)      │         │
│  │ _make_exec_tools_node (工具执行/CB/扩散)  │         │
│  │ _make_eval_response_node (5层安全网)      │         │
│  └─────────────────────────────────────────┘         │
└──────────────┬───────────────────────────────────────┘
               │ 复用
┌──────────────▼───────────────────────────────────────┐
│  zulong/l2/fc_runner.py (~500行)                       │
│  核心FC循环引擎(纯Python, 不依赖LangGraph)               │
│  ┌─────────────────────────────────────────┐         │
│  │ class FCRunner:                          │         │
│  │   def run(messages, tools, engine):      │         │
│  │     while True:                          │         │
│  │       state = check_fn(state)            │         │
│  │       state = call_model_fn(state)       │         │
│  │       if tool_calls:                     │         │
│  │         state = exec_tools_fn(state)     │         │
│  │       else:                              │         │
│  │         verdict = eval_response_fn(state)│         │
│  │         if verdict == "done": break      │         │
│  │     return result                        │         │
│  └─────────────────────────────────────────┘         │
│  可配置: stream_enabled, stream_handler,              │
│          tool_classifier, pause_handler              │
└──────────────┬───────────────────────────────────────┘
               │ 继承/组合
┌──────────────▼───────────────────────────────────────┐
│  zulong/ide/ide_fc_runner.py (~1600行, 瘦身65%)        │
│  IDE适配层: 流式推送 + 工具分流 + 暂停/恢复              │
│  ┌─────────────────────────────────────────┐         │
│  │ class IDEFCRunner(FCRunner):             │         │
│  │   def _call_model(state):                │         │
│  │     super()._call_model(state)           │         │
│  │     self._stream_to_ws(state)     ← IDE特有│        │
│  │                                          │         │
│  │   def _exec_tools(state):                │         │
│  │     if _is_remote(tool):                 │         │
│  │       self._pause_for_remote(state) ← IDE特有│      │
│  │     else:                                │         │
│  │       super()._exec_tools(state)         │         │
│  └─────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────┘
```

### 合并步骤

```
步骤1: 删除 fc_graph.py 的废弃代码
  删除 LangGraph StateGraph 构建部分 (~1100行)
  重命名为 fc_nodes.py (仅保留4个工厂函数 + 相关类型)

步骤2: 创建 fc_runner.py
  从 unified_fc_runner.py 提取 while 循环
  将工厂函数调用设为可注入 (支持子类覆盖)
  从 ide_fc_runner.py 迁移的核心安全网增量:
    - 重复工具死循环检测 (check阶段)
    - 弹性续期 (check阶段)
    - 进度停滞检测 (check阶段)
    - CB压力感知 (exec_tools阶段)
    - 语义漂移检测 (eval阶段)
    - 响应提前中断检测 (eval阶段)
    - 首轮无效回复拦截 (eval阶段)

步骤3: 简化 ide_fc_runner.py
  继承 FCRunner
  仅保留 IDE 特有逻辑:
    - 流式推送 (call_model 钩子)
    - 工具分流 (exec_tools 钩子)
    - 暂停/恢复 (远程工具)
    - 参数验证/别名映射
    - Web广播
    - 状态持久化/收尾

步骤4: 更新引用
  inference_engine.py: unified_fc_runner → fc_runner
  ide_server.py: ide_fc_runner (API不变, 内部简化)

代码量变化:
  fc_graph.py: 1544行 → fc_nodes.py: ~450行 (保留工厂函数)
  unified_fc_runner.py: 221行 → 删除
  fc_runner.py: ~500行 (新建)
  ide_fc_runner.py: 4061行 → ~1600行
  净减少: 8000行 → 2500行 (-69%)
```

---

## 三、协议统一方案

### 统一帧格式

```json
{
  "msg_id": "a1b2c3d4e5f6",
  "type": "namespace:action",
  "session_id": null,
  "ts": 1716300000.0,
  "payload": {}
}
```

### 统一消息类型

| 统一类型 | 方向 | 说明 | Web旧类型 | IDE旧类型 |
|---------|------|------|----------|----------|
| `handshake` | 上行 | 客户端角色声明 | (无) | (无) |
| `ping`/`pong` | 双向 | 心跳 | 同 | 同 |
| `task:start` | 上行 | 启动任务 | CHAT_MESSAGE | session_start |
| `task:resume` | 上行 | 恢复任务 | (无) | session_resume |
| `task:cancel` | 上行 | 取消任务 | STOP_GENERATION | user_cancel |
| `task:ack` | 下行 | 任务确认 | WELCOME | session_ack |
| `task:complete` | 下行 | 任务完成 | CHAT_RESPONSE | task_complete |
| `task:error` | 下行 | 任务错误 | (无) | task_error |
| `task:progress` | 下行 | 任务进度 | THINKING_STEP | task_progress/status_update |
| `text:stream` | 下行 | 流式文本(增量chunk) | STREAMING_RESPONSE | display_text |
| `text:final` | 下行 | 最终文本 | CHAT_RESPONSE | task_complete |
| `reasoning` | 下行 | 推理过程 | (无) | display_reasoning |
| `tool:request` | 下行 | 请求执行工具 | (无) | tool_request |
| `tool:result` | 上行 | 工具执行结果 | (无) | tool_result |
| `graph:memory:update` | 下行 | 记忆图谱更新 | MEMORY_GRAPH_UPDATE | (无) |
| `graph:memory:expand` | 下行 | 图谱展开 | MEMORY_GRAPH_EXPAND_RESULT | (无) |
| `graph:task:update` | 下行 | 任务图谱更新 | TASK_GRAPH_UPDATE | (无) |
| `session:list` | 下行 | 会话列表 | SESSION_LIST | (无) |
| `session:messages` | 下行 | 会话消息 | SESSION_MESSAGES | (无) |
| `audio:start/end/chunk` | 双向 | 音频流 | 同 | 同 |
| `audio:transcript` | 下行 | ASR转录 | (无) | audio_transcript |

### 端点统一

保留 `ws://127.0.0.1:8090/` 单一入口:
```python
# 客户端连接后先发 handshake
{
  "type": "handshake",
  "payload": {
    "client_type": "dashboard" | "ide_plugin" | "monitor",
    "api_version": "2.0"
  }
}

# 后端根据 client_type 路由到对应的消息处理器
```

### 向后兼容

```
Phase 1: 后端同时处理新旧格式 (version字段区分)
Phase 2: Dashboard前端适配新格式  
Phase 3: IDE插件适配新格式
Phase 4: 移除旧格式支持
```

---

## 四、实施步骤

| 步骤 | 内容 | 风险 | 代码变化 |
|------|------|------|---------|
| 1 | 删除fc_graph废弃代码，重命名为fc_nodes.py | 低 | -1100行 |
| 2 | 创建fc_runner.py核心引擎 | 中 | +500行 |
| 3 | IDEFCRunner继承FCRunner，瘦身 | 高 | -2500行 |
| 4 | 更新inference_engine.py引用 | 中 | -100行 |
| 5 | 后端统一WS帧格式+类型名 | 中 | ~200行改动 |
| 6 | Dashboard前端适配 | 中 | ~300行改动 |
| 7 | IDE前端适配 | 低 | ~50行改动 |

**总计: 减少 ~4600行后端代码, 净减 ~57%**
