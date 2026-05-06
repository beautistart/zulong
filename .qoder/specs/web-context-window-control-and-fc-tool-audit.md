# 计划：Web端上下文窗口控制 + 清理XML转译层死代码

## 背景

祖龙系统两项改进：
1. **Web UI 缺少上下文窗口大小控制** — 用户只能通过编辑 YAML 文件调整，需要在 Web 设置面板添加可视化控件
2. **XML 转译层是死代码** — 经审计确认，祖龙后端与 IDE 插件之间已完全使用 FC（JSON function calling）格式通信。`tool_calls_to_xml()` 等 XML 生成代码仍在执行但产出被 IDE 完全忽略，属于无用开销，需要清理

### 架构现状确认

```
祖龙后端 (Python)                    zulong IDE 插件 (VS Code)
┌─────────────────┐                  ┌────────────────────┐
│ FC 循环         │   WebSocket      │ ZulongHandler      │
│ tool_calls JSON │ ──────────────►  │ 读取 tool_calls    │
│                 │                  │ 执行文件/命令操作   │
│ xml_tool_calls  │ ──────────────►  │ ❌ 完全忽略XML     │
│ (死代码输出)     │                  │                    │
└─────────────────┘                  └────────────────────┘
```

**关键事实**：
- IDE 插件的 `ZulongHandler` 只读取 `req.tool_calls`（FC JSON数组）
- `xml_tool_calls` 字段被生成后从未被任何代码消费
- `tool_calls_to_xml()` 每次远程工具调用都会执行，但输出无人使用

---

## 任务 1：Web端上下文窗口大小控制

### 设计
- 离散预设滑块（9档）：4K / 8K / 16K / 32K / 64K / 128K / 256K / 512K / 1M
- 位置：设置面板顶部（全局设置，在层级卡片上方）
- 交互：滑块 + 当前值显示 + 可点击的预设按钮

### 需要修改的文件

| 文件 | 修改内容 |
|------|----------|
| `zulong/ide/ide_server.py` | 添加 GET/POST `/api/config/context_window` 端点 |
| `openclaw_bridge/web/static/index.html` | 添加滑块组件的 CSS + HTML + JS |

### 后端实现（`ide_server.py`）

在现有模型层路由之后（约第957行）添加两个端点：

**GET `/api/config/context_window`**：
- 读取 `cm.config['l2_inference']['circuit_breaker']['context_window_size']`
- 返回 `{"value": 131072}`

**POST `/api/config/context_window`**：
- 验证：整数，范围 [4096, 1048576]
- 更新目标：
  1. `cm.config['l2_inference']['circuit_breaker']['context_window_size'] = value`
  2. `cm.config['llm'][当前后端]['num_ctx'] = value`
  3. `cm.save()` 持久化到 YAML
  4. `engine._context_window_size = value` 内存生效
  5. `zulong.models.container.LLM_NUM_CTX = value` 模块级全局变量
- 返回 `{"status": "ok", "value": value}`

### 前端实现（`index.html`）

1. **CSS**：`.context-window-section`、`.context-window-slider`、`.context-window-presets`、`.context-window-preset-btn`
2. **HTML**：在 `renderSettingsPanel()` 顶部渲染，位于层级卡片循环之前
3. **JS 函数**：
   - `_CTX_PRESETS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]`
   - `loadContextWindowSetting()` — 面板打开时获取当前值
   - `onContextWindowSliderChange()` — 防抖（300ms）调用 API
   - `_formatContextSize(tokens)` — 显示为 "128K" / "1M"

---

## 任务 2：清理 XML 转译层死代码

### 需要删除的内容

| 位置 | 删除项 |
|------|--------|
| `zulong/ide/ide_format_translator.py` | `tool_calls_to_xml()` 方法、`_single_tool_call_to_xml()` 方法、`_TOOL_PARAM_ORDER` 字典 |
| `zulong/ide/ide_fc_runner.py` | `_pause_for_remote()` 中的 `xml = self.translator.tool_calls_to_xml(remote_calls)` 调用 |
| `zulong/ide/ide_fc_runner.py` | `IDEFCResult` 数据类中的 `xml_tool_calls` 字段 |
| `zulong/ide/ide_server.py` | WebSocket 消息中 `xml_tool_calls` 字段的生成和发送 |
| `zulong-ide/.../zulong-websocket.ts` | `ZulongToolRequest` 接口中的 `xml_tool_calls` 字段 |

### 保留的内容

| 位置 | 保留项 | 原因 |
|------|--------|------|
| `ide_format_translator.py` | `parse_xml_tool_calls()` | 模型偶尔在文本中输出 XML 格式的工具调用，需要回退解析为 FC 格式 |
| `ide_fc_runner.py` | `_strip_xml_tool_tags()` | 清理模型文本中残留的 XML 标签 |
| `ide_format_translator.py` | `parse_ide_tool_results()` | 解析 IDE 返回的工具结果（仍在使用） |

### 具体修改步骤

**步骤 1**：`zulong/ide/ide_format_translator.py`
- 删除 `tool_calls_to_xml()` 静态方法（约第71-126行）
- 删除 `_single_tool_call_to_xml()` 辅助方法
- 删除 `_TOOL_PARAM_ORDER` 字典（约第51-62行）
- 删除 `_escape_closing_tag()` / `_unescape_closing_tag()`（仅被 xml 生成使用时）

**步骤 2**：`zulong/ide/ide_fc_runner.py`
- `IDEFCResult` 数据类：删除 `xml_tool_calls` 字段
- `_pause_for_remote()` 方法：删除 `xml = self.translator.tool_calls_to_xml(remote_calls)` 行
- 修改 `IDEFCResult` 返回值，不再传递 `xml_tool_calls`

**步骤 3**：`zulong/ide/ide_server.py`
- 找到发送 `tool_request` WebSocket 消息的代码
- 从 payload 中移除 `xml_tool_calls` 字段
- 保留 `tool_calls`、`call_ids`、`tool_names`

**步骤 4**：`zulong-ide/src/core/api/transport/zulong-websocket.ts`
- `ZulongToolRequest` 接口：删除 `xml_tool_calls: string` 字段

---

## 执行顺序

1. 任务 2（清理死代码）— 简单、低风险、立即可做
2. 任务 1（上下文窗口控制）— 独立功能，需要前后端配合

---

## 验证方式

### 任务 2 验证（XML清理）：
1. `python -c "import ast; ast.parse(open('zulong/ide/ide_fc_runner.py').read())"` 语法检查
2. `python -c "from zulong.ide.ide_fc_runner import IDEFCRunner; print('OK')"` 导入检查
3. 启动祖龙服务 → IDE 提交任务 → 工具调用正常执行
4. 确认日志中不再出现 XML 相关输出

### 任务 1 验证（上下文窗口）：
1. 启动祖龙服务
2. 打开 Web UI 设置面板
3. 拖动滑块 / 点击预设 → 验证 API 调用成功
4. 检查 `config/zulong_config.yaml` 反映新值
5. 新建 IDE 会话 → 验证 AttentionWindow 日志显示新预算
