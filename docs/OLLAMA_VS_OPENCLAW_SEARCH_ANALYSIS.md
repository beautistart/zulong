# Ollama 联网搜索 vs OpenClaw 联网搜索对比分析

## 📋 问题背景

用户反馈：**OpenClaw 联网搜索老是遭遇人机验证**，而 **Ollama 自带联网搜索**表现良好。

## 🔍 Ollama 联网搜索实现原理

### 1. 架构设计

Ollama 的联网搜索不是本地实现的，而是通过 **Ollama 官方云服务 API**：

```
用户请求 → 本地模型 (如 Qwen3.5) → 工具调用 → Ollama 云 API → 返回搜索结果
```

### 2. 技术实现

#### 2.1 API 端点

- **Web Search API**: `POST https://ollama.com/api/web_search`
- **Web Fetch API**: `POST https://ollama.com/api/web_fetch`

#### 2.2 认证方式

需要 Ollama 账户 API Key：
```bash
export OLLAMA_API_KEY="your_api_key"
```

#### 2.3 请求示例

```python
import ollama

# 简单搜索
response = ollama.web_search("What is Ollama?")
print(response)

# 带参数搜索
response = ollama.web_search(query="AI news", max_results=5)
```

#### 2.4 响应格式

```json
{
  "results": [
    {
      "title": "网页标题",
      "url": "https://example.com",
      "content": "内容摘要..."
    }
  ]
}
```

### 3. 为什么 Ollama 搜索不会遇到人机验证？

#### ✅ 核心原因

1. **服务端代理**：
   - 搜索请求由 Ollama 官方服务器发起
   - 使用的是 Ollama 的 IP 和域名信誉
   - 不是用户本地 IP 发起的请求

2. **企业级 API**：
   - Ollama 可能购买了商业搜索 API（如 Google Custom Search、Bing Search API）
   - 或者与搜索引擎有合作关系
   - 使用合法的 API 通道，不是爬虫

3. **请求频率控制**：
   - Ollama 云服务有完善的限流机制
   - 免费用户有额度限制
   - 避免触发反爬机制

4. **专业基础设施**：
   - 数据中心 IP 信誉度高
   - 有专门的团队维护搜索服务
   - 能及时处理封禁问题

### 4. Ollama 搜索的优缺点

#### ✅ 优点
- 无需担心人机验证
- 无需配置搜索引擎
- 开箱即用，零配置
- 结果质量稳定
- 有免费额度

#### ❌ 缺点
- 依赖 Ollama 云服务（需要网络连接）
- 免费额度有限
- 无法自定义搜索源
- 国内访问可能不稳定
- 隐私数据经过第三方

---

## 🔍 OpenClaw 联网搜索实现原理

### 1. 架构设计

OpenClaw 使用 **本地 Puppeteer 无头浏览器** 直接访问搜索引擎：

```
用户请求 → OpenClaw Bridge → Node.js 技能 → Puppeteer → 搜索引擎网站 → 解析 HTML → 返回结果
```

### 2. 技术实现

#### 2.1 核心依赖

```javascript
// openclaw/skills/web-search-skill/index_engine_tool.js
const searchEngineTool = require('search-engine-tool');
```

#### 2.2 工作流程

```javascript
async function search(query, engine = 'bing') {
  // 1. 启动 Puppeteer 无头浏览器
  const browser = await puppeteer.launch({ headless: 'new' });
  
  // 2. 打开搜索引擎页面
  const page = await browser.newPage();
  await page.goto(`https://${engine}.com/search?q=${query}`);
  
  // 3. 等待页面加载和 JavaScript 执行
  await page.waitForSelector('.search-result');
  
  // 4. 提取搜索结果
  const results = await page.evaluate(() => {
    return document.querySelectorAll('.search-result').map(el => ({
      title: el.querySelector('h3').textContent,
      url: el.querySelector('a').href,
      snippet: el.querySelector('.snippet').textContent
    }));
  });
  
  return results;
}
```

#### 2.3 支持的搜索引擎

- Bing（默认，对中文友好）
- Google
- DuckDuckGo
- Yahoo

### 3. 为什么 OpenClaw 会遇到人机验证？

#### ❌ 核心原因

1. **直接访问搜索引擎网站**：
   - 请求直接从用户本地 IP 发出
   - 没有中间代理层
   - 被视为普通爬虫

2. **Puppeteer 特征明显**：
   ```javascript
   // 虽然设置了 User-Agent，但仍有其他特征
   navigator.webdriver === true  // 容易被检测
   ```

3. **IP 信誉问题**：
   - 家庭/办公室 IP 信誉度低
   - 频繁请求容易被封
   - 没有域名信誉保护

4. **行为模式异常**：
   - 自动化操作节奏固定
   - 缺少真实用户的随机性
   - 页面停留时间短

5. **搜索引擎反爬升级**：
   - Bing、Google 等不断加强反爬
   - 检测 JavaScript 执行环境
   - 需要完整的浏览器指纹

### 4. OpenClaw 搜索的优缺点

#### ✅ 优点
- 完全本地控制
- 无需第三方服务
- 免费无限制
- 隐私数据不经过第三方
- 可自定义搜索源

#### ❌ 缺点
- 容易触发人机验证
- 速度较慢（需加载完整页面）
- 稳定性差
- 需要维护 Puppeteer 环境
- 并发能力弱

---

## 📊 对比总结

| 特性 | Ollama | OpenClaw |
|------|--------|----------|
| **实现方式** | 云 API 代理 | 本地爬虫 |
| **人机验证** | ❌ 几乎不会 | ✅ 经常遇到 |
| **响应速度** | 快 (1-3 秒) | 慢 (3-10 秒) |
| **稳定性** | 高 | 低 |
| **配置复杂度** | 零配置 | 需要配置环境 |
| **隐私保护** | 中 (经过第三方) | 高 (完全本地) |
| **成本** | 免费额度有限 | 完全免费 |
| **可控性** | 低 | 高 |
| **并发能力** | 强 | 弱 |

---

## 💡 解决方案建议

### 方案 1：集成 Ollama Web Search API（推荐）

**优势**：
- 彻底解决人机验证问题
- 代码改动最小
- 稳定性最高

**实现步骤**：

1. **在 OpenClaw API Server 中添加 Ollama 搜索接口**：

```python
# openclaw_bridge/api_server.py
import ollama

@self.app.post("/api/ollama_search")
async def ollama_search(request: Dict[str, Any]):
    """使用 Ollama 官方搜索 API"""
    query = request.get("query")
    max_results = request.get("max_results", 5)
    
    try:
        response = ollama.web_search(query=query, max_results=max_results)
        return {
            "success": True,
            "results": response.get("results", []),
            "source": "ollama_cloud"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
```

2. **修改 OpenClaw 搜索工具**：

```python
# zulong/tools/openclaw_search.py
def _search(self, query: str, count: int = 3):
    # 优先使用 Ollama API
    try:
        import ollama
        response = ollama.web_search(query=query, max_results=count)
        return {
            "success": True,
            "results": response.get("results", []),
            "source": "ollama"
        }
    except Exception as e:
        # 降级到 Puppeteer 方案
        logger.warning(f"Ollama 搜索失败，降级到本地搜索：{e}")
        return self._search_local(query, count)
```

### 方案 2：使用商业搜索 API

**推荐服务**：
- **Tavily API** (tavily.com) - 专为 AI 设计
- **Serper API** (serper.dev) - Google Search API
- **Bocha API** (博查 AI) - 中文搜索优化

**优势**：
- 稳定可靠
- 有免费额度
- 结果质量高

**示例代码**：

```python
import requests

def search_with_tavily(query: str, num_results: int = 5):
    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TAVILY_API_KEY}"
    }
    payload = {
        "query": query,
        "num_results": num_results
    }
    
    response = requests.post(url, headers=headers, json=payload)
    return response.json()
```

### 方案 3：优化 Puppeteer 反检测（不推荐）

如果必须使用本地搜索，可以优化反检测：

```javascript
const browser = await puppeteer.launch({
  headless: 'new',
  args: [
    '--disable-blink-features=AutomationControlled',
    '--no-sandbox',
    '--disable-dev-shm-usage'
  ]
});

const page = await browser.newPage();

// 注入反检测脚本
await page.evaluateOnNewDocument(() => {
  Object.defineProperty(navigator, 'webdriver', {
    get: () => undefined
  });
});

// 设置真实的 User-Agent
await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64)...');
```

### 方案 4：混合方案（最佳实践）

结合多种方案，实现高可用性：

```python
async def smart_search(query: str, count: int = 5):
    """智能搜索：多策略降级"""
    
    # 策略 1: Ollama 云 API（优先）
    try:
        result = await ollama_search(query, count)
        if result['success']:
            return result
    except Exception as e:
        logger.warning(f"Ollama 失败：{e}")
    
    # 策略 2: Tavily API（备用）
    try:
        result = await tavily_search(query, count)
        if result['success']:
            return result
    except Exception as e:
        logger.warning(f"Tavily 失败：{e}")
    
    # 策略 3: 本地 Puppeteer（最后手段）
    try:
        result = await puppeteer_search(query, count)
        if result['success']:
            return result
    except Exception as e:
        logger.error(f"本地搜索失败：{e}")
    
    # 全部失败
    return {
        "success": False,
        "error": "所有搜索服务均不可用"
    }
```

---

## 🎯 推荐实施方案

### 短期方案（立即实施）

1. **集成 Ollama Web Search API**
   - 修改 `openclaw_bridge/api_server.py`
   - 添加 Ollama 搜索接口
   - 设置 `OLLAMA_API_KEY` 环境变量

2. **保留 Puppeteer 作为降级方案**
   - 当 Ollama API 不可用时自动切换

### 中期方案（1-2 周）

1. **申请 Tavily API 或 Bocha API**
   - 作为第二备用方案
   - 提高系统可靠性

2. **实现智能路由**
   - 根据查询类型选择最优搜索源
   - 中文查询优先 Bocha
   - 英文查询优先 Ollama/Tavily

### 长期方案（1 个月+）

1. **自建搜索索引**
   - 使用 SearXNG 等开源搜索引擎
   - 完全控制搜索质量

2. **多源聚合**
   - 同时调用多个搜索源
   - 去重、排序、融合结果

---

## 📝 实施步骤

### 步骤 1: 获取 Ollama API Key

1. 访问 https://ollama.com/settings/keys
2. 注册/登录 Ollama 账户
3. 创建 API Key
4. 设置环境变量：
   ```bash
   setx OLLAMA_API_KEY "your_api_key_here"
   ```

### 步骤 2: 修改 OpenClaw API Server

编辑 `openclaw_bridge/api_server.py`，添加：

```python
import ollama

async def _perform_search(self, query: str, count: int):
    """优先使用 Ollama API"""
    try:
        # 尝试 Ollama
        response = ollama.web_search(query=query, max_results=count)
        logger.info(f"[OpenClawAPIServer] Ollama 搜索成功：{len(response.get('results', []))} 个结果")
        return response.get('results', [])
    except Exception as e:
        logger.warning(f"[OpenClawAPIServer] Ollama 失败，降级到本地搜索：{e}")
        # 降级到原有 Puppeteer 方案
        return await self._perform_search_local(query, count)
```

### 步骤 3: 测试验证

```powershell
# 测试 Ollama 搜索
Invoke-WebRequest -Uri "http://localhost:3000/api/search" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"query": "AI 最新进展", "count": 5}'
```

---

## 🔗 参考资料

1. [Ollama Web Search API 文档](https://docs.ollama.com/capabilities/web-search)
2. [Ollama Python 库](https://github.com/ollama/ollama-python)
3. [Tavily API 文档](https://tavily.com/)
4. [search-engine-tool NPM](https://www.npmjs.com/package/search-engine-tool)

---

## 📞 联系支持

如有问题，请查看：
- Ollama 官方文档：https://docs.ollama.com
- OpenClaw 配置文档：`openclaw/skills/web-search-skill/README_SEARCH_CONFIG.md`
