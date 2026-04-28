# OpenAI API 集成计划

## 1. 仓库分析

### 1.1 系统架构

* **L0**: 核心层（事件总线、状态管理）

* **L1-A**: 反射层（受控反射）

* **L1-B**: 调度层（电源/守门）

* **L2**: 专家模型层（LangGraph 中枢）

* **L3**: 双脑模型层（热切换）

### 1.2 当前 OpenClaw 集成状态

* ✅ OpenClaw Bridge 已实现

* ✅ Web 适配器运行在 localhost:8080

* ✅ EventBus 通信机制已建立

* ✅ L1-B 事件路由已注册

### 1.3 现有工具

* `vscode_tool.py`: VSCode CLI 集成工具

* `openclaw_bridge`: OpenClaw 桥接器

* 缺少：OpenAI API 服务器实现

## 2. 实现计划

### 2.1 阶段一：OpenAI API 服务器实现

1. **创建 API 服务器目录**：`zulong/api/`
2. **实现 OpenAI 兼容接口**：

   * `POST /v1/chat/completions`

   * `GET /v1/models`

   * `GET /v1/health`
3. **集成 L1-B 调度**：所有请求必须通过 L1-B 调度器
4. **实现请求路由**：

   * 消息清洗与意图提取

   * 构建 Prompt + 检索

   * 响应格式化

### 2.2 阶段二：VSCode 插件集成

1. **配置 Cline 插件**：

   * 设置 OpenAI API 端点为 `http://localhost:3928/v1`

   * 配置 API 密钥为 `zulong`
2. **测试连接**：

   * 验证模型列表

   * 测试列出系统必要文件我要上传代码做审查

   * <br />

   * 验证 L1-B 调度

### 2.3 阶段三：OpenClaw 集成验证

1. **验证 Web 界面**：

   * 确认 web 界面运行在 localhost:8080

   * 测试与 OpenClaw 的通信
2. **验证事件流**：

   * 确认所有请求通过 EventBus 传递

   * 验证 L1-B 正确处理请求

## 3. 技术实现

### 3.1 API 服务器技术栈

* **框架**：FastAPI

* **端口**：3928

* **认证**：简单 API 密钥验证

* **响应**：OpenAI 兼容格式

### 3.2 数据流

```
Cline 插件 → OpenAI API → L1-B 调度 → L2 专家处理 → L3 模型推理 → 响应
```

### 3.3 关键模块

1. **`openai_server.py`**：主 API 服务器
2. **`request_processor.py`**：请求处理与意图提取
3. **`response_formatter.py`**：响应格式化

## 4. 风险与应对

### 4.1 风险

* **性能问题**：L2 推理可能超时

* **兼容性**：OpenAI API 格式差异

* **稳定性**：系统负载过高

### 4.2 应对措施

* **超时处理**：设置合理的超时时间

* **格式验证**：严格验证请求格式

* **负载控制**：实现请求队列

## 5. 测试计划

### 5.1 功能测试

* [ ] API 健康检查

* [ ] 模型列表接口

* [ ] 聊天完成接口

* [ ] VSCode 插件连接

* [ ] OpenClaw 集成

### 5.2 性能测试

* [ ] 响应时间

* [ ] 并发处理

* [ ] 内存占用

## 6. 预期成果

* ✅ OpenAI API 兼容服务器运行在 localhost:3928

* ✅ VSCode Cline 插件成功调用祖龙系统

* ✅ 所有请求通过 L1-B 调度

* ✅ OpenClaw 集成正常工作

* ✅ 系统稳定运行

## 7. 依赖项

* **FastAPI**：API 服务器框架

* **Uvicorn**：ASGI 服务器

* **Pydantic**：数据验证

* **LangGraph**：工作流编排（已有）

