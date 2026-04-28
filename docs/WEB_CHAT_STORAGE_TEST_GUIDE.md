# 网页端对话持久化存储 - 测试指南

## ✅ 已实现功能

### 核心功能
- ✅ **自动保存**: 每条消息自动保存到 localStorage
- ✅ **自动恢复**: 刷新页面自动恢复历史对话
- ✅ **独立存储**: 与祖龙系统完全隔离
- ✅ **容量控制**: 自动保留最近 100 条消息
- ✅ **手动清空**: 一键删除所有历史记录

### 技术实现
- **存储位置**: 浏览器 localStorage
- **存储键名**: `web_chat_history`
- **数据格式**: JSON
- **作用域**: 仅限当前域名（`localhost:8080`）

---

## 🧪 测试步骤

### 测试 1: 基本保存功能

1. **启动系统**
   ```bash
   # 在 WSL 中启动 vLLM
   bash start_l2_backup_vllm.sh
   
   # 启动祖龙系统
   python -m zulong.bootstrap
   
   # 启动 OpenClaw Bridge
   python -m openclaw_bridge.bootstrap
   ```

2. **打开网页**
   - 访问：`http://localhost:8080`
   - 确认 WebSocket 连接成功（状态显示"已连接"）

3. **发送测试消息**
   ```
   你好，我是测试用户
   ```

4. **检查控制台日志**
   - 按 `F12` 打开开发者工具
   - 查看 Console 标签
   - ✅ 应显示：`[LocalStorage] ✅ 消息已保存`

5. **检查 localStorage**
   - 在开发者工具中，切换到 **Application** 标签
   - 展开 **Local Storage** → `http://localhost:8080`
   - 点击 `web_chat_history` 键
   - ✅ 应看到 JSON 格式的对话数据

---

### 测试 2: 页面刷新恢复

1. **发送多条消息**
   ```
   消息 1: 这是第一条测试消息
   消息 2: 这是第二条测试消息
   消息 3: 这是第三条测试消息
   ```

2. **刷新页面**
   - 按 `F5` 或点击刷新按钮
   - 等待页面加载完成

3. **验证恢复**
   - ✅ 应看到之前发送的 3 条消息
   - ✅ 控制台显示：`[LocalStorage] ✅ 已恢复 3 条消息`
   - ✅ 消息顺序正确（从上到下）
   - ✅ 发送者标识正确（用户/助手）

---

### 测试 3: 与祖龙系统隔离

1. **发送对话到祖龙**
   ```
   我想减肥，请帮我制定一个计划
   ```

2. **等待祖龙回复**
   - ✅ 收到祖龙的详细回复

3. **检查祖龙记忆系统**
   - 查看祖龙系统日志
   - ✅ 确认对话已保存到祖龙短期记忆（Memory Zone）

4. **验证独立性**
   - 清空浏览器 localStorage
   - 刷新页面
   - ✅ Web 端对话清空
   - ✅ 祖龙系统记忆不受影响（仍可通过 API 查询）

---

### 测试 4: 容量限制

1. **批量发送消息**（脚本测试）
   
   打开开发者工具 Console，执行：
   ```javascript
   // 发送 105 条测试消息
   for (let i = 1; i <= 105; i++) {
       saveMessage({
           id: generateId(),
           text: `测试消息 ${i}`,
           sender: i % 2 === 0 ? 'user' : 'assistant',
           timestamp: Date.now()
       });
   }
   
   // 检查存储数量
   const history = loadChatHistory();
   console.log(`当前存储消息数：${history.length}`);
   ```

2. **验证结果**
   - ✅ 应显示：`当前存储消息数：100`
   - ✅ 最早的 5 条消息已自动删除
   - ✅ 保留最近 100 条消息

---

### 测试 5: 清空历史功能

1. **点击清空按钮**
   - 点击页面左上角的 "🗑️ 清空历史" 按钮

2. **验证清空结果**
   - ✅ 聊天窗口清空
   - ✅ localStorage 中的 `web_chat_history` 键被删除
   - ✅ 控制台显示：`[LocalStorage] 🗑️ 历史已清空`

3. **刷新页面**
   - ✅ 聊天窗口保持空白
   - ✅ 无历史对话恢复

---

### 测试 6: 跨会话持久化

1. **发送对话**
   ```
   这是持久化测试消息
   ```

2. **关闭浏览器**
   - 完全关闭浏览器窗口

3. **重新打开浏览器**
   - 访问：`http://localhost:8080`

4. **验证**
   - ✅ 测试消息仍然存在
   - ✅ 数据跨会话保存成功

---

## 📊 预期结果汇总

| 测试项 | 预期结果 | 实际结果 |
|--------|----------|----------|
| 基本保存 | ✅ 消息保存到 localStorage | ⬜ |
| 页面刷新恢复 | ✅ 自动恢复历史对话 | ⬜ |
| 系统隔离 | ✅ Web 与祖龙记忆独立 | ⬜ |
| 容量限制 | ✅ 保留最近 100 条 | ⬜ |
| 手动清空 | ✅ 一键删除所有记录 | ⬜ |
| 跨会话持久化 | ✅ 关闭浏览器后仍存在 | ⬜ |

---

## 🔍 调试技巧

### 问题 1: 消息未保存

**检查点**:
1. 浏览器是否支持 localStorage
2. 是否启用了 Cookie 和网站数据
3. 开发者工具 Console 是否有错误信息

**解决方法**:
```javascript
// 在 Console 中手动测试
localStorage.setItem('test', 'data');
console.log(localStorage.getItem('test'));
// 应输出：data
```

---

### 问题 2: 刷新后未恢复

**检查点**:
1. `restoreChatHistory()` 是否被调用
2. localStorage 中是否有数据
3. `loadChatHistory()` 是否返回空数组

**调试代码**:
```javascript
// 在 Console 中检查
console.log('存储的数据:', localStorage.getItem('web_chat_history'));
console.log('加载的历史:', loadChatHistory());
```

---

### 问题 3: 存储空间不足

**症状**:
- Console 显示错误：`QuotaExceededError`

**解决方法**:
1. 手动清空历史
2. 减少保留消息数量（修改代码中的 `100` 为更小的值）
3. 清理浏览器缓存

---

## 📝 数据格式说明

### 存储结构

```json
{
  "messages": [
    {
      "id": "id-1234567890-abc123def",
      "text": "你好",
      "sender": "user",
      "timestamp": 1713168000000
    },
    {
      "id": "id-1234567891-xyz789ghi",
      "text": "你好！有什么可以帮助你的？",
      "sender": "assistant",
      "timestamp": 1713168001000
    }
  ],
  "lastUpdated": 1713168001000
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 消息唯一标识 |
| `text` | string | 消息文本内容 |
| `sender` | 'user' \| 'assistant' | 发送者类型 |
| `timestamp` | number | Unix 时间戳（毫秒） |
| `lastUpdated` | number | 最后更新时间 |

---

## 🚀 性能优化建议

### 1. 防抖保存（可选）

如果频繁保存影响性能，可以添加防抖：

```javascript
let saveTimeout = null;

function saveMessageDebounced(message) {
    if (saveTimeout) {
        clearTimeout(saveTimeout);
    }
    
    saveTimeout = setTimeout(() => {
        saveMessage(message);
    }, 500); // 500ms 防抖
}
```

### 2. 压缩存储（可选）

如果消息较长，可以压缩：

```javascript
// 使用 LZ-String 库压缩
const compressed = LZString.compress(JSON.stringify(messages));
localStorage.setItem('web_chat_history', compressed);
```

---

## ✅ 完成标志

- [x] 实现 localStorage 存储功能
- [x] 实现页面刷新恢复
- [x] 添加清空历史按钮
- [x] 容量限制（100 条）
- [x] 与祖龙系统隔离
- [x] 编写测试指南
- [ ] 完成所有测试项
- [ ] 性能优化（可选）

---

**测试完成时间**: ⬜  
**测试人员**: ⬜  
**状态**: ⬜ 等待测试
