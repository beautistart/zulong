# 祖龙集成 Ollama 本地联网搜索方案

## 📋 问题背景

**用户需求**：祖龙能否调用 Ollama 的联网搜索功能，但**不需要 Ollama 云服务的 API Key**？

**答案**：✅ **可以！** 通过自托管 SearXNG 实现完全本地的联网搜索。

---

## 🔍 技术原理

### Ollama 联网搜索的两种模式

#### 模式 1：Ollama 云服务（需要 API Key）

```
用户 → 本地模型 → 工具调用 → Ollama 云 API (https://ollama.com/api/web_search) → 返回结果
```

- ✅ 优点：开箱即用，无需配置
- ❌ 缺点：需要 API Key，依赖云服务

#### 模式 2：自托管 SearXNG（无需 API Key）⭐ 推荐

```
用户 → 本地模型 → 工具调用 → SearXNG (本地) → 多个搜索引擎 → 返回结果
```

- ✅ 优点：完全免费，无需 API Key，隐私保护
- ✅ 优点：支持 70+ 搜索引擎（Google、Bing、DuckDuckGo 等）
- ✅ 优点：完全本地控制，无依赖

---

## 🎯 推荐方案：SearXNG + 祖龙集成

### 方案架构

```
祖龙 L2 模型
    ↓
工具调用 (web_search)
    ↓
OpenClaw API Server (端口 3000)
    ↓
SearXNG API (端口 8101)
    ↓
多个搜索引擎 (Bing、Google、DuckDuckGo...)
```

### 核心组件

1. **SearXNG**：开源元搜索引擎（自托管）
2. **OpenClaw API Server**：祖龙的工具调用接口
3. **祖龙工具系统**：集成搜索工具

---

## 📦 实施步骤

### 步骤 1：部署 SearXNG（Docker 推荐）

#### 1.1 创建 Docker Compose 配置

创建文件 `docker-compose.searxng.yml`：

```yaml
version: '3.8'

services:
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    ports:
      - "8101:8080"
    volumes:
      - ./searxng:/etc/searxng
    environment:
      - SEARXNG_BASE_URL=http://localhost:8101
    restart: unless-stopped
```

#### 1.2 创建 SearXNG 配置目录

```powershell
# 创建配置目录
mkdir searxng
cd searxng

# 生成密钥（Windows PowerShell）
$randomBytes = New-Object byte[] 32
(New-Object Security.Cryptography.RNGCryptoServiceProvider).GetBytes($randomBytes)
$secretKey = -join ($randomBytes | ForEach-Object { "{0:x2}" -f $_ })

# 创建 settings.yml
@"
use_default_settings: true

general:
  debug: false
  instance_name: "SearXNG"
  secret_key: "$secretKey"
  limiter: false

search:
  safe_search: 0
  autocomplete: "google"

engines:
  - name: bing
    engine: bing
    shortcut: b
    disabled: false
  
  - name: duckduckgo
    engine: duckduckgo
    shortcut: ddg
    disabled: false
  
  - name: google
    engine: google
    shortcut: g
    disabled: false

server:
  secret_key: "$secretKey"
  limiter: false
"@ | Out-File -FilePath "settings.yml" -Encoding utf8
```

#### 1.3 启动 SearXNG

```powershell
# 在项目根目录执行
docker-compose -f docker-compose.searxng.yml up -d

# 查看日志
docker-compose -f docker-compose.searxng.yml logs -f

# 验证服务
curl http://localhost:8101/search?q=test
```

访问 Web 界面：http://localhost:8101

---

### 步骤 2：修改 OpenClaw API Server

编辑 [`openclaw_bridge/api_server.py`](file:///d:/AI/project/zulong_beta4/openclaw_bridge/api_server.py)：

#### 2.1 添加 SearXNG 搜索方法

```python
async def _perform_search(self, query: str, count: int) -> List[Dict[str, Any]]:
    """
    执行搜索（优先使用 SearXNG）
    """
    logger.info(f"[OpenClawAPIServer] 执行搜索：{query}, 数量：{count}")
    
    # 方案 1：使用 SearXNG（推荐，无需 API Key）
    try:
        return await self._search_with_searxng(query, count)
    except Exception as e:
        logger.warning(f"[OpenClawAPIServer] SearXNG 失败：{e}")
    
    # 方案 2：降级到 Puppeteer（备用）
    try:
        return await self._search_with_puppeteer(query, count)
    except Exception as e:
        logger.error(f"[OpenClawAPIServer] Puppeteer 也失败：{e}")
        return []

async def _search_with_searxng(self, query: str, count: int) -> List[Dict[str, Any]]:
    """
    使用 SearXNG 进行搜索
    
    Args:
        query: 搜索查询
        count: 结果数量
    
    Returns:
        搜索结果列表
    """
    import aiohttp
    
    searxng_url = "http://localhost:8101/search"
    params = {
        "q": query,
        "format": "json",
        "pageno": 1,
        "language": "zh-CN"  # 中文优先
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(searxng_url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status == 200:
                data = await response.json()
                results = []
                
                for result in data.get("results", [])[:count]:
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("content", ""),
                        "engine": result.get("engine", "searxng")
                    })
                
                logger.info(f"[OpenClawAPIServer] SearXNG 搜索成功：{len(results)} 个结果")
                return results
            else:
                raise Exception(f"SearXNG 返回错误状态码：{response.status}")
```

#### 2.2 添加依赖

编辑 `openclaw_bridge/requirements.txt`（如果不存在则创建）：

```
aiohttp>=3.9.0
```

安装：

```powershell
pip install aiohttp
```

---

### 步骤 3：修改祖龙工具系统

编辑 [`zulong/tools/openclaw_search.py`](file:///d:/AI/project/zulong_beta4/zulong/tools/openclaw_search.py)：

#### 3.1 优化搜索工具

```python
def _search(self, query: str, count: int = 3) -> Dict[str, Any]:
    """执行网络搜索（使用 SearXNG）"""
    try:
        url = f"{self.openclaw_api_url}/search"
        payload = {
            "query": query,
            "count": count,
            "source": "searxng"  # 指定使用 SearXNG
        }
        
        logger.info(f"[OpenClawSearchTool] 执行搜索：{query}, 结果数量：{count}")
        
        response = requests.post(
            url,
            headers=self.headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"[OpenClawSearchTool] 搜索成功，找到 {len(result.get('results', []))} 个结果")
            return {
                "success": True,
                "results": result.get("results", []),
                "source": "searxng"
            }
        else:
            logger.error(f"[OpenClawSearchTool] 搜索失败，状态码：{response.status_code}")
            return {"success": False, "error": f"HTTP 错误：{response.status_code}"}
    except Exception as e:
        logger.error(f"[OpenClawSearchTool] 搜索异常：{e}")
        return {"success": False, "error": str(e)}
```

---

### 步骤 4：测试验证

#### 4.1 测试 SearXNG

```powershell
# 测试 SearXNG API
curl "http://localhost:8101/search?q=AI+技术&format=json"

# 应该返回 JSON 格式的搜索结果
```

#### 4.2 测试 OpenClaw API

```powershell
# 测试搜索接口
$body = @{
    query = "AI 最新进展"
    count = 5
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:3000/api/search" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

#### 4.3 测试祖龙系统

```python
# 通过 WebSocket 发送测试消息
import websocket
import json

ws = websocket.create_connection("ws://localhost:5555")

# 发送测试消息
message = {
    "type": "user_message",
    "content": "帮我搜索最新的 AI 技术进展",
    "complex_task": False
}

ws.send(json.dumps(message))
response = ws.recv()
print(response)
ws.close()
```

---

## 🎨 完整示例

### Docker Compose 配置（一键部署）

创建 `docker-compose.full.yml`：

```yaml
version: '3.8'

services:
  # SearXNG 搜索引擎
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    ports:
      - "8101:8080"
    volumes:
      - ./searxng:/etc/searxng
    environment:
      - SEARXNG_BASE_URL=http://localhost:8101
    restart: unless-stopped
  
  # 祖龙系统（可选，如果已运行）
  zulong:
    build: .
    container_name: zulong
    ports:
      - "5555:5555"  # WebSocket
      - "3000:3000"  # API
    environment:
      - EVENT_BUS_HOST=localhost
      - EVENT_BUS_PORT=5555
      - SEARXNG_URL=http://searxng:8080
    depends_on:
      - searxng
    restart: unless-stopped
```

启动：

```powershell
docker-compose -f docker-compose.full.yml up -d
```

---

## 📊 方案对比

| 特性 | Ollama 云服务 | SearXNG 自托管 | Puppeteer 本地 |
|------|--------------|----------------|----------------|
| **API Key** | ❌ 需要 | ✅ 不需要 | ✅ 不需要 |
| **成本** | 免费额度有限 | ✅ 完全免费 | ✅ 完全免费 |
| **人机验证** | ✅ 不会遇到 | ✅ 不会遇到 | ❌ 经常遇到 |
| **速度** | 快 (1-3 秒) | 快 (1-2 秒) | 慢 (3-10 秒) |
| **稳定性** | 高 | 高 | 低 |
| **隐私** | 中 | ✅ 高 | ✅ 高 |
| **可控性** | 低 | ✅ 高 | ✅ 高 |
| **搜索引擎** | 固定 | ✅ 70+ 可选 | 有限 |

---

## 🔧 高级配置

### SearXNG 优化

编辑 `searxng/settings.yml`：

```yaml
engines:
  # 启用 Bing（对中文友好）
  - name: bing
    engine: bing
    shortcut: b
    disabled: false
    language: zh-CN
  
  # 启用 DuckDuckGo（隐私保护）
  - name: duckduckgo
    engine: duckduckgo
    shortcut: ddg
    disabled: false
  
  # 启用 Google（需要额外配置）
  - name: google
    engine: google
    shortcut: g
    disabled: false
  
  # 禁用质量差的引擎
  - name: yahoo
    disabled: true
```

### 多 SearXNG 实例（提高可靠性）

修改 OpenClaw API Server：

```python
SEARXNG_INSTANCES = [
    "http://localhost:8101",
    "https://searx.be",
    "https://search.inetol.net"
]

async def _search_with_searxng(self, query: str, count: int):
    """使用多个 SearXNG 实例进行故障转移"""
    for instance in SEARXNG_INSTANCES:
        try:
            results = await self._search_single_instance(instance, query, count)
            if results:
                return results
        except Exception as e:
            logger.warning(f"SearXNG 实例 {instance} 失败：{e}")
            continue
    
    raise Exception("所有 SearXNG 实例均失败")
```

---

## 🐛 常见问题

### Q1: SearXNG 返回结果为空？

**解决方案**：
1. 检查 Docker 容器状态：`docker-compose ps`
2. 查看日志：`docker-compose logs searxng`
3. 测试 Web 界面：http://localhost:8101
4. 检查搜索引擎配置，确保至少启用了 Bing 或 DuckDuckGo

### Q2: 搜索速度慢？

**解决方案**：
1. 减少结果数量（count=3-5）
2. 优化 SearXNG 配置，禁用慢的引擎
3. 使用本地 SearXNG 实例（不要用公共实例）

### Q3: 中文搜索结果不准确？

**解决方案**：
1. 在 SearXNG 配置中设置 `language: zh-CN`
2. 优先使用 Bing（对中文支持更好）
3. 在查询中自动添加中文关键词

---

## 📝 总结

### ✅ 方案优势

1. **完全免费**：无需 API Key，无使用限制
2. **完全本地**：数据不经过第三方，隐私保护
3. **高可靠性**：支持 70+ 搜索引擎，自动故障转移
4. **易于维护**：Docker 一键部署，配置简单
5. **无依赖**：不依赖 Ollama 云服务

### 🚀 实施建议

1. **立即实施**：部署 SearXNG（5 分钟）
2. **修改代码**：集成到 OpenClaw API（10 分钟）
3. **测试验证**：发送测试消息（5 分钟）
4. **生产部署**：配置 Docker Compose（可选）

### 📞 支持资源

- SearXNG 官方文档：https://docs.searxng.org
- SearXNG GitHub：https://github.com/searxng/searxng
- 祖龙项目文档：`docs/` 目录

---

**开始实施吧！只需 20 分钟，你就能拥有完全免费、无需 API Key 的联网搜索功能！** 🎉
