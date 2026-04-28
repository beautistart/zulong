# OpenClaw 与 Zulong 集成配置指南

## 概述

本文档说明如何配置 OpenClaw 与 Zulong 系统的集成，确保数据格式正确传递。

## 架构流程

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   OpenClaw      │      │   Zulong API    │      │   Zulong L1-B   │
│   (前端)        │      │   (端口 3928)   │      │   (调度器)      │
└────────┬────────┘      └────────┬────────┘      └────────┬────────┘
         │                        │                        │
         │  Chat Completions      │                        │
         │  POST /v1/chat/        │                        │
         │  completions           │                        │
         │ ──────────────────────>│                        │
         │                        │  消息清洗 & 意图提取    │
         │                        │ ──────────────────────>│
         │                        │                        │
         │                        │  构建 Prompt + 检索    │
         │                        │ <──────────────────────│
         │                        │                        │
         │  返回响应              │                        │
         │ <──────────────────────│                        │
         │                        │                        │
         │                        │                        │
         │  文件上传 Webhook      │                        │
         │  POST /v1/webhook      │                        │
         │ ──────────────────────>│                        │
         │                        │  文件解析 & Ingestion  │
         │                        │ ──────────────────────>│
         │                        │                        │
         │                        │  存入知识库            │
         │                        │ <──────────────────────│
         │                        │                        │
         │  返回处理结果          │                        │
         │ <──────────────────────│                        │
```

## OpenClaw 配置

### 1. 复制配置文件

将 `config/openclaw_config.json` 复制到 OpenClaw 项目目录：

```powershell
copy "D:\AI\project\zulong_beta4\config\openclaw_config.json" "D:\AI\project\openclaw\openclaw_config.json"
```

### 2. 配置说明

```json
{
  "models": {
    "providers": {
      "zulong": {
        "type": "openai",
        "baseUrl": "http://localhost:3928/v1",
        "apiKey": "zulong",
        "defaultHeaders": {
          "X-Zulong-Session-Type": "shared",
          "X-Zulong-Session-Id": "openclaw-shared-session",
          "Authorization": "Bearer zulong"
        }
      }
    }
  }
}
```

**关键字段说明：**

| 字段 | 值 | 说明 |
|------|-----|------|
| `type` | `openai` | 使用 OpenAI 兼容 API |
| `baseUrl` | `http://localhost:3928/v1` | Zulong API 地址 |
| `apiKey` | `zulong` | API 密钥（任意值） |
| `X-Zulong-Session-Type` | `shared` | 会话类型 |
| `X-Zulong-Session-Id` | `openclaw-shared-session` | 会话 ID |

### 3. 安装 Hook

将 `config/zulong_hook.js` 复制到 OpenClaw 的 hooks 目录：

```powershell
# 创建 hooks 目录（如果不存在）
mkdir "D:\AI\project\openclaw\hooks\zulong" -Force

# 复制 hook 文件
copy "D:\AI\project\zulong_beta4\config\zulong_hook.js" "D:\AI\project\openclaw\hooks\zulong\handler.js"
```

## Zulong API 端点

### Chat Completions API

**端点：** `POST /v1/chat/completions`

**请求格式：**

```json
{
  "model": "zulong",
  "messages": [
    {"role": "system", "content": "你是祖龙机器人"},
    {"role": "user", "content": "你好"}
  ],
  "max_tokens": 256,
  "stream": false
}
```

**响应格式：**

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "zulong",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！有什么可以帮助你的吗？"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 15,
    "total_tokens": 35
  }
}
```

### Webhook API

**端点：** `POST /v1/webhook`

**文件上传事件：**

```json
{
  "type": "file_upload",
  "filename": "manual.pdf",
  "filepath": "/path/to/uploads/manual.pdf",
  "user_id": "user_123"
}
```

**用户行为事件：**

```json
{
  "type": "user_action",
  "action": "click_button",
  "user_id": "user_123",
  "metadata": {
    "button_id": "submit"
  }
}
```

### Ingestion API

**端点：** `POST /v1/ingest`

**请求格式：**

```json
{
  "filepath": "/path/to/file.pdf",
  "filename": "document.pdf",
  "domain": "user_docs"
}
```

**响应格式：**

```json
{
  "status": "success",
  "filename": "document.pdf",
  "file_type": "pdf",
  "chunks": 15,
  "knowledge_ids": ["id1", "id2", ...],
  "parse_time_ms": 123.45,
  "total_time_ms": 456.78
}
```

## L1-B 数据处理流程

### 1. 消息清洗

OpenClaw 发送的 messages 数组会被 L1-B 清洗：

```python
# 原始消息
messages = [
    {"role": "system", "content": "你是祖龙机器人"},
    {"role": "user", "content": "查询订单 12345"},
    {"role": "assistant", "content": "好的..."},
    {"role": "user", "content": "状态是什么？"}
]

# 清洗后
cleaned = [
    {"role": "system", "content": "你是祖龙机器人", "type": "instruction"},
    {"role": "user", "content": "查询订单 12345", "intent": "query", "entities": [{"type": "number", "value": "12345"}]},
    {"role": "assistant", "content": "好的...", "action": None, "data": None},
    {"role": "user", "content": "状态是什么？", "intent": "query", "entities": []}
]
```

### 2. 意图识别

L1-B 会自动识别用户意图：

| 意图 | 关键词 |
|------|--------|
| `query` | 是什么, 怎么, 如何, 为什么, 查询, 搜索 |
| `command` | 执行, 运行, 启动, 停止, 打开, 关闭 |
| `upload` | 上传, 发送, 提交, 导入 |
| `confirm` | 确认, 是的, 好的, 可以 |
| `cancel` | 取消, 不要, 不行 |

### 3. 文件解析

支持的文件格式：

| 类型 | 格式 | 处理方式 |
|------|------|---------|
| 文本 | .txt, .md, .json, .csv | 直接读取 |
| Word | .doc, .docx | python-docx 解析 |
| PDF | .pdf | PyMuPDF + OCR |
| 图片 | .jpg, .png | OCR 或视觉模型 |

## 测试验证

### 1. 启动 Zulong API

```powershell
cd D:\AI\project\zulong_beta4
python -m zulong.api.openai_server
```

### 2. 测试 API 连接

```powershell
# 测试健康检查
curl http://localhost:3928/v1/health

# 测试模型列表
curl http://localhost:3928/v1/models

# 测试聊天
curl -X POST http://localhost:3928/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"zulong","messages":[{"role":"user","content":"你好"}]}'
```

### 3. 测试 Webhook

```powershell
curl -X POST http://localhost:3928/v1/webhook `
  -H "Content-Type: application/json" `
  -d '{"type":"file_upload","filename":"test.txt","filepath":"D:\test.txt","user_id":"test"}'
```

## 常见问题

### Q1: 请求超时

**原因：** L2 推理时间过长

**解决：** 
- 减少 `max_tokens` 参数（默认 256）
- 使用更快的模型

### Q2: 文件上传失败

**原因：** 文件路径不存在或不支持的格式

**解决：**
- 检查文件路径是否正确
- 确认文件格式在支持列表中

### Q3: 消息格式错误

**原因：** OpenClaw 发送的消息格式不符合 OpenAI 规范

**解决：**
- 确保每条消息包含 `role` 和 `content` 字段
- 检查 `role` 是否为 `system`、`user` 或 `assistant`

## 下一步

1. 启动 Zulong API 服务
2. 更新 OpenClaw 配置文件
3. 安装 Zulong Hook
4. 重启 OpenClaw
5. 测试对话功能
