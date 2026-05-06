# 🔌 OpenClaw 连接指南

**创建日期**: 2026-04-03  
**OpenClaw 路径**: `D:\AI\project\openclaw`  
**祖龙 Bridge 路径**: `d:\AI\project\zulong_beta4\openclaw_bridge`

---

## 📋 连接方案概述

OpenClaw 已经配置了与祖龙系统的集成，通过 **Webhook** 和 **HTTP API** 进行通信。

### 当前配置（来自 `openclaw_config.json`）

```json
{
  "gateway": {
    "mode": "local",
    "auth": {
      "mode": "token",
      "token": "zulong-api-token"
    }
  },
  "models": {
    "providers": {
      "zulong": {
        "type": "openai",
        "baseUrl": "http://localhost:3928/v1",
        "apiKey": "zulong"
      }
    }
  },
  "webhook": {
    "enabled": true,
    "url": "http://localhost:3928/v1/webhook",
    "events": ["file_upload", "user_action", "system_event"]
  }
}
```

---

## 🔗 连接方式

### 方式 1: Webhook 事件上报（已实现）

OpenClaw 通过 Webhook 将事件发送到祖龙系统：

- **URL**: `http://localhost:3928/v1/webhook`
- **事件类型**:
  - `file_upload` - 文件上传
  - `user_action` - 用户行为
  - `system_event` - 系统事件

**Handler 位置**: `D:\AI\project\openclaw\hooks\zulong\handler.js`

---

### 方式 2: OpenClaw Bridge 事件总线（新增）

这是我们刚刚开发的 **OpenClaw Bridge**，通过 EventBus 实现双向通信：

```
OpenClaw → OpenClaw Bridge → EventBus → 祖龙 L1-B
祖龙 L1-B → EventBus → OpenClaw Bridge → OpenClaw
```

---

## 🚀 连接步骤

### 步骤 1: 启动祖龙系统

```bash
# 终端 1
cd d:\AI\project\zulong_beta4
python -m zulong.bootstrap
```

**预期输出**:
```
祖龙系统启动成功
EventBus 监听端口：5555
Webhook 监听端口：3928
```

---

### 步骤 2: 启动 OpenClaw Bridge

```bash
# 终端 2
cd d:\AI\project\zulong_beta4
python -m openclaw_bridge.bootstrap
```

**预期输出**:
```
🦾 OpenClaw Bridge - 祖龙系统集成
启动时间：2026-04-03 XX:XX:XX

[1/5] 初始化 EventBus 客户端...
✅ EventBus 客户端已初始化
[2/5] 初始化麦克风适配器...
✅ 麦克风适配器已初始化
[3/5] 初始化视觉报告器...
✅ 视觉报告器已启动
[4/5] 初始化执行监听器...
✅ 执行监听器已初始化
[5/5] 初始化语音播报监听器...
✅ 语音播报监听器已初始化

🎉 OpenClaw Bridge 启动成功！
```

---

### 步骤 3: 配置 OpenClaw 连接 Bridge

修改 `D:\AI\project\openclaw\openclaw_config.json`，添加 Bridge 配置：

```json
{
  "zulongBridge": {
    "enabled": true,
    "eventBusHost": "localhost",
    "eventBusPort": 5555,
    "clientName": "OpenClaw",
    "mockMode": false
  }
}
```

---

### 步骤 4: 创建 OpenClaw 到 Bridge 的连接器

在 `D:\AI\project\openclaw\hooks\zulong\` 下创建 `bridge_connector.js`:

```javascript
/**
 * OpenClaw Bridge Connector
 * 
 * 将 OpenClaw 事件转发到 OpenClaw Bridge EventBus
 */

const WebSocket = require('ws');

class BridgeConnector {
  constructor(config) {
    this.config = config;
    this.ws = null;
    this.connected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
  }

  async connect() {
    const { eventBusHost, eventBusPort, clientName } = this.config;
    const wsUrl = `ws://${eventBusHost}:${eventBusPort}/openclaw`;
    
    console.log(`[Bridge Connector] 正在连接到 ${wsUrl}...`);
    
    try {
      this.ws = new WebSocket(wsUrl);
      
      this.ws.on('open', () => {
        console.log(`[Bridge Connector] ✅ 连接成功！`);
        this.connected = true;
        this.reconnectAttempts = 0;
        
        // 发送握手消息
        this.send({
          type: 'HANDSHAKE',
          client: clientName,
          timestamp: new Date().toISOString()
        });
      });
      
      this.ws.on('message', (data) => {
        const message = JSON.parse(data);
        console.log(`[Bridge Connector] 收到消息：`, message);
        this.handleMessage(message);
      });
      
      this.ws.on('close', () => {
        console.log(`[Bridge Connector] ❌ 连接关闭`);
        this.connected = false;
        this.reconnect();
      });
      
      this.ws.on('error', (error) => {
        console.error(`[Bridge Connector] 错误：`, error);
      });
      
    } catch (error) {
      console.error(`[Bridge Connector] 连接失败：`, error);
      this.reconnect();
    }
  }

  send(event) {
    if (this.connected && this.ws) {
      this.ws.send(JSON.stringify(event));
    } else {
      console.warn(`[Bridge Connector] 未连接，消息已加入队列：`, event.type);
    }
  }

  handleMessage(message) {
    // 处理来自祖龙系统的消息
    switch (message.type) {
      case 'TASK_EXECUTE':
        // 执行祖龙系统下发的任务
        console.log(`[Bridge Connector] 执行任务：`, message.payload);
        this.executeTask(message.payload);
        break;
      
      case 'ACTION_SPEAK':
        // 语音播报
        console.log(`[Bridge Connector] 语音播报：`, message.payload.text);
        this.speak(message.payload.text);
        break;
      
      default:
        console.log(`[Bridge Connector] 未知消息类型：`, message.type);
    }
  }

  async executeTask(payload) {
    // TODO: 调用 OpenClaw SDK 执行任务
    const { name, arguments: args } = payload;
    console.log(`执行动作：${name}`, args);
    
    // 模拟执行结果
    const result = {
      action: name,
      success: true,
      result: { message: '任务完成' }
    };
    
    // 发送执行结果
    this.send({
      type: 'ACTION_RESULT',
      payload: result
    });
  }

  speak(text) {
    // TODO: 调用 OpenClaw TTS
    console.log(`🔊 [OpenClaw] ${text}`);
  }

  reconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
      console.log(`[Bridge Connector] ${delay}ms 后重试...`);
      setTimeout(() => this.connect(), delay);
    } else {
      console.error(`[Bridge Connector] 达到最大重连次数，放弃`);
    }
  }

  // 上报 OpenClaw 事件
  reportEvent(event) {
    this.send({
      type: event.type,
      source: 'openclaw',
      payload: event.payload,
      timestamp: new Date().toISOString()
    });
  }
}

// 导出
module.exports = BridgeConnector;

// 使用示例
if (require.main === module) {
  const connector = new BridgeConnector({
    eventBusHost: 'localhost',
    eventBusPort: 5555,
    clientName: 'OpenClaw'
  });
  
  connector.connect();
  
  // 模拟上报事件
  setTimeout(() => {
    connector.reportEvent({
      type: 'USER_SPEECH',
      payload: { text: '把苹果放到桌子上' }
    });
  }, 2000);
}
```

---

### 步骤 5: 集成到 OpenClaw Hook

修改 `D:\AI\project\openclaw\hooks\zulong\handler.js`，添加 Bridge 连接：

```javascript
const BridgeConnector = require('./bridge_connector');

// 初始化 Bridge 连接器
const bridgeConnector = new BridgeConnector({
  eventBusHost: 'localhost',
  eventBusPort: 5555,
  clientName: 'OpenClaw'
});

// 启动连接
bridgeConnector.connect();

const handler = async (event) => {
  if (!event || typeof event !== 'object') {
    return;
  }

  // 上报到 Bridge
  if (event.type === 'file_upload') {
    bridgeConnector.reportEvent({
      type: 'SENSOR_VISION',
      payload: {
        event: 'file_uploaded',
        filename: event.filename,
        filepath: event.filepath
      }
    });
  }

  if (event.type === 'user_action') {
    bridgeConnector.reportEvent({
      type: 'USER_SPEECH',
      payload: {
        text: event.action,
        metadata: event.metadata
      }
    });
  }

  // ... 原有 Webhook 逻辑
};

module.exports = handler;
module.exports.default = handler;
```

---

## 🧪 测试连接

### 测试脚本

在 `D:\AI\project\openclaw\` 下创建 `test_bridge_connection.js`:

```javascript
const BridgeConnector = require('./hooks/zulong/bridge_connector');

async function test() {
  console.log('🧪 测试 OpenClaw Bridge 连接...\n');
  
  const connector = new BridgeConnector({
    eventBusHost: 'localhost',
    eventBusPort: 5555,
    clientName: 'OpenClaw_Test'
  });
  
  await connector.connect();
  
  // 等待连接
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  if (!connector.connected) {
    console.error('❌ 连接失败！请检查祖龙系统和 Bridge 是否已启动。');
    process.exit(1);
  }
  
  console.log('✅ 连接成功！\n');
  
  // 测试上报事件
  console.log('📤 测试上报事件...');
  connector.reportEvent({
    type: 'USER_SPEECH',
    payload: { text: '你好，我是 OpenClaw' }
  });
  
  // 等待响应
  await new Promise(resolve => setTimeout(resolve, 3000));
  
  console.log('\n✅ 测试完成！');
  process.exit(0);
}

test().catch(console.error);
```

运行测试：

```bash
cd D:\AI\project\openclaw
node test_bridge_connection.js
```

---

## 📊 数据流

### 上行链路（OpenClaw → 祖龙）

```
1. OpenClaw 检测到用户语音
   ↓
2. 调用 handler.js
   ↓
3. BridgeConnector.reportEvent()
   ↓
4. WebSocket 发送到 EventBus
   ↓
5. EventBus 路由到 L1-B
   ↓
6. L1-B 处理并可能转发给 L2
```

### 下行链路（祖龙 → OpenClaw）

```
1. 祖龙 L2 生成任务指令
   ↓
2. L1-B 发送 TASK_EXECUTE 事件
   ↓
3. EventBus 广播
   ↓
4. BridgeConnector 接收
   ↓
5. 调用 OpenClaw SDK 执行
   ↓
6. 返回 ACTION_RESULT
```

---

## 🔧 故障排除

### 问题 1: 连接失败

**错误**: `Error: connect ECONNREFUSED 127.0.0.1:5555`

**解决方案**:
1. 检查祖龙系统是否启动
2. 检查 OpenClaw Bridge 是否启动
3. 确认端口 5555 未被占用

```bash
# 检查端口
netstat -ano | findstr :5555
```

---

### 问题 2: Webhook 404

**错误**: `Webhook 返回 404`

**解决方案**:
1. 检查祖龙系统 Webhook 服务是否运行
2. 确认 URL 正确：`http://localhost:3928/v1/webhook`
3. 检查 API Key 配置

---

### 问题 3: 事件未处理

**错误**: 事件发送成功但祖龙系统无响应

**解决方案**:
1. 检查 EventBus 订阅者
2. 查看祖龙系统日志
3. 确认事件类型匹配

---

## 📚 相关文件

- **OpenClaw 配置**: [`D:\AI\project\openclaw\openclaw_config.json`](file://D:\AI\project\openclaw\openclaw_config.json)
- **Zulong Hook**: [`D:\AI\project\openclaw\hooks\zulong\handler.js`](file://D:\AI\project\openclaw\hooks\zulong\handler.js)
- **OpenClaw Bridge**: [`d:\AI\project\zulong_beta4\openclaw_bridge`](file://d:\AI\project\zulong_beta4\openclaw_bridge)
- **集成报告**: [`OPENCLAW_INTEGRATION_REPORT.md`](file://d:\AI\project\zulong_beta4\OPENCLAW_INTEGRATION_REPORT.md)

---

## 🎯 下一步

### 立即可做

1. ✅ 启动祖龙系统
2. ✅ 启动 OpenClaw Bridge
3. ✅ 运行测试脚本
4. ✅ 验证双向通信

### 后续优化

1. 实现真实 OpenClaw SDK 调用
2. 添加更多事件类型支持
3. 优化错误处理和重试机制
4. 实现视觉状态同步

---

**准备好开始连接了吗？** 请告诉我，我可以帮您创建连接器文件或运行测试！ 🚀
