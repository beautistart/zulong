# 网页端对话持久化存储方案

## 📊 问题分析

### 当前状态
- ❌ 网页端对话内容**仅保存在前端内存**中
- ❌ 刷新页面后**所有对话丢失**
- ❌ **没有后端存储**机制
- ❌ 与祖龙系统的短期记忆**完全隔离**

### 根本原因
网页端作为 OpenClaw 的独立输入源（`openclaw/web_ui`），其对话历史：
1. **前端**: 仅通过 DOM 临时显示，无本地存储
2. **后端**: 通过 EventBus 发送到祖龙系统，但**未保存 Web 专属历史**
3. **隔离**: 与祖龙短期记忆系统**物理隔离**

---

## 💡 解决方案

### 方案选择

#### 方案 A: 前端 localStorage 存储（推荐）
**优点**:
- ✅ 实现简单，无需修改后端
- ✅ 响应速度快，无网络延迟
- ✅ 与祖龙系统完全隔离
- ✅ 每个浏览器独立存储

**缺点**:
- ⚠️ 仅限本地访问，无法跨设备
- ⚠️ 存储空间有限（~5MB）

#### 方案 B: 后端独立数据库
**优点**:
- ✅ 可跨设备同步
- ✅ 存储容量大

**缺点**:
- ❌ 需要数据库支持
- ❌ 增加系统复杂度
- ❌ 违背"隔离"需求

---

## 🎯 推荐方案：前端 localStorage

### 实现原理

```javascript
// 1. 每条消息保存时
localStorage.setItem('web_chat_history', JSON.stringify(messages));

// 2. 页面加载时恢复
const saved = localStorage.getItem('web_chat_history');
const messages = JSON.parse(saved);
```

### 数据结构

```typescript
interface WebChatMessage {
  id: string;          // 消息唯一 ID
  text: string;        // 消息内容
  sender: 'user' | 'assistant';  // 发送者
  timestamp: number;   // 时间戳
}

// 存储格式
{
  "messages": [
    {
      "id": "uuid-1234",
      "text": "你好",
      "sender": "user",
      "timestamp": 1234567890
    },
    {
      "id": "uuid-5678",
      "text": "你好！有什么可以帮助你的？",
      "sender": "assistant",
      "timestamp": 1234567891
    }
  ],
  "lastUpdated": 1234567890
}
```

---

## 🔧 实现步骤

### 步骤 1: 修改 index.html - 添加存储功能

在 `<script>` 部分添加以下函数：

```javascript
// ========== 本地存储管理 ==========

// 从 localStorage 加载历史对话
function loadChatHistory() {
    try {
        const saved = localStorage.getItem('web_chat_history');
        if (saved) {
            const data = JSON.parse(saved);
            return data.messages || [];
        }
    } catch (e) {
        console.error('[LocalStorage] 加载历史失败:', e);
    }
    return [];
}

// 保存单条消息到 localStorage
function saveMessage(message) {
    try {
        const history = loadChatHistory();
        history.push(message);
        
        // 只保留最近 100 条消息，防止存储溢出
        if (history.length > 100) {
            history.shift();
        }
        
        localStorage.setItem('web_chat_history', JSON.stringify({
            messages: history,
            lastUpdated: Date.now()
        }));
        
        console.log('[LocalStorage] ✅ 消息已保存');
    } catch (e) {
        console.error('[LocalStorage] 保存失败:', e);
    }
}

// 清空历史对话
function clearChatHistory() {
    try {
        localStorage.removeItem('web_chat_history');
        console.log('[LocalStorage] 🗑️ 历史已清空');
    } catch (e) {
        console.error('[LocalStorage] 清空失败:', e);
    }
}

// 恢复历史对话到聊天窗口
function restoreChatHistory() {
    const history = loadChatHistory();
    history.forEach(msg => {
        addMessage(msg.text, msg.sender, false); // false = 不重复保存
    });
    console.log(`[LocalStorage] ✅ 已恢复 ${history.length} 条消息`);
}

// 修改 addMessage 函数，增加保存功能
function addMessage(text, sender, shouldSave = true) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text;
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // 滚动到底部
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // 保存到 localStorage
    if (shouldSave) {
        saveMessage({
            id: crypto.randomUUID(),
            text: text,
            sender: sender,
            timestamp: Date.now()
        });
    }
}
```

### 步骤 2: 页面加载时恢复历史

在 `connect()` 函数后添加：

```javascript
// 初始化
connect();
restoreChatHistory();  // 🔥 新增：恢复历史对话
```

### 步骤 3: 添加清空历史按钮（可选）

在 HTML 中添加按钮：

```html
<div class="chat-header">
    <span class="status-indicator disconnected" id="statusIndicator"></span>
    OpenClaw × 祖龙
    <div class="connection-status" id="connectionStatus">连接中...</div>
    <button class="clear-history-btn" onclick="clearChatHistory()" title="清空历史">🗑️</button>
</div>
```

添加样式：

```css
.clear-history-btn {
    position: absolute;
    top: 10px;
    left: 10px;
    background: rgba(255, 255, 255, 0.2);
    border: none;
    border-radius: 5px;
    padding: 5px 10px;
    color: white;
    cursor: pointer;
    font-size: 16px;
    transition: background 0.3s;
}

.clear-history-btn:hover {
    background: rgba(255, 255, 255, 0.3);
}
```

---

## ✅ 实现效果

### 功能清单
- ✅ 每条消息自动保存到 localStorage
- ✅ 刷新页面自动恢复历史对话
- ✅ 仅保存在浏览器本地，与祖龙系统隔离
- ✅ 自动清理旧消息（保留最近 100 条）
- ✅ 支持手动清空历史

### 用户体验
1. **首次访问**: 空聊天窗口
2. **发送消息**: 消息立即显示并保存
3. **收到回复**: 回复显示并保存
4. **刷新页面**: 自动恢复所有历史对话
5. **清空历史**: 一键删除所有记录

---

## 🔒 数据隔离说明

### 数据流向

```
用户输入 (Web)
  ↓
[Web 端 localStorage] ← 独立存储，不共享
  ↓
EventBus (USER_TEXT 事件)
  ↓
祖龙系统短期记忆 (Memory Zone)
  ↓
祖龙系统回复
  ↓
[Web 端 localStorage] ← 独立存储，不共享
```

### 隔离优势
1. **隐私保护**: Web 对话仅限本地访问
2. **性能优化**: 不占用祖龙系统内存
3. **独立性**: 祖龙系统重启不影响 Web 历史
4. **灵活性**: 可单独清空 Web 历史

---

## 📝 注意事项

### localStorage 限制
- **存储容量**: 约 5MB（可存储 ~1000 条短消息）
- **作用域**: 仅限当前域名（`localhost:8080`）
- **持久性**: 除非手动清除，否则永久保存

### 兼容性
- ✅ Chrome/Edge/Safari/Firefox 现代浏览器
- ⚠️ IE11 不支持 `crypto.randomUUID()`，需使用 `Date.now()` 替代

### 替代方案（如果 localStorage 不可用）
```javascript
// 使用 IndexedDB（更复杂，但容量更大）
// 或使用 sessionStorage（仅会话期间有效）
```

---

## 🚀 快速实施

### 立即可用的修改
1. 打开 `openclaw_bridge/web/static/index.html`
2. 在 `<script>` 标签内添加存储函数
3. 修改 `addMessage` 调用保存
4. 页面加载时调用 `restoreChatHistory()`

### 测试步骤
1. 打开网页 `http://localhost:8080`
2. 发送几条测试消息
3. 刷新页面
4. ✅ 确认历史对话已恢复

---

**创建时间**: 2026-04-15  
**状态**: ✅ 方案设计完成，等待实施
