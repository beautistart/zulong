# 祖龙系统工具调用分析报告

## 📊 工具系统架构概览

### 核心组件

```
┌─────────────────────────────────────────────────────────┐
│                  L2 Inference Engine                     │
│  (意图识别 → 工具调用决策 → 结果处理)                     │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│                    Tool Engine                           │
│  (统一调度器：注册/执行/监控/错误处理)                    │
│  - ThreadPoolExecutor (max_workers=5)                    │
│  - 并发控制                                                │
│  - 超时管理                                                │
│  - 调用历史记录                                            │
└────────────────┬────────────────────────────────────────┘
                 │
        ┌────────┴────────┬──────────────┬──────────┐
        ▼                 ▼              ▼          ▼
┌──────────────┐  ┌──────────────┐  ┌────────┐  ┌─────────┐
│ Base Tool    │  │ OpenClaw     │  │ Web    │  │ Search  │
│ (工具基类)    │  │ Tools        │  │ Search │  │ Tools   │
└──────────────┘  └──────────────┘  └────────┘  └─────────┘
```

## 📁 工具模块文件结构

```
zulong/tools/
├── base.py              # 工具基类和注册表
├── tool_engine.py       # 工具引擎核心调度器
├── core_tool_manager.py # 核心工具管理器
├── debug_console.py     # 调试控制台工具
├── openclaw_tool.py     # OpenClaw 适配器
├── openclaw_search.py   # OpenClaw 搜索工具
├── openclaw_plugin.py   # OpenClaw 插件适配器
├── search_tools.py      # 搜索工具集合
├── web_search.py        # 网络搜索工具
├── system_tools.py      # 系统工具
└── vscode_tool.py       # VSCode 工具
```

## 🔧 已注册工具清单

### 1. **OpenClaw 系列工具** (外部设备交互)

#### OpenClawTool (`openclaw_tool`)
- **用途**: 与 OpenClaw 机器人通信
- **动作**:
  - `move_to`: 移动到指定位置
  - `grab`: 抓取物体
  - `release`: 释放物体
  - `get_status`: 获取机器人状态
- **调用位置**: `inference_engine.py:600`
- **执行方式**: `tool_engine.call_tool("openclaw_tool", action, params)`

#### OpenClawSearch (`openclaw_search`)
- **用途**: 搜索和获取网页内容
- **动作**:
  - `search`: 网络搜索
  - `fetch_webpage`: 获取网页内容
  - `parse_links`: 解析链接
- **调用位置**: 
  - `inference_engine.py:750` (搜索)
  - `inference_engine.py:796` (获取网页)
  - `inference_engine.py:2271` (本地工具调用)
- **特点**: 支持链接提取和文本摘要

#### OpenClawPlugin (`openclaw_plugin`)
- **用途**: OpenClaw 插件扩展
- **动作**: 动态加载插件功能
- **调用位置**: 通过 ToolEngine 统一调度

### 2. **搜索类工具** (信息检索)

#### WebSearch (`web_search`)
- **用途**: 通用网络搜索
- **动作**:
  - `search`: 执行搜索
  - `get_results`: 获取结果
- **调用位置**: `inference_engine.py:875`
- **集成**: 通过 ToolEngine 调用

#### SearchTools (`search_tools`)
- **用途**: 搜索工具集合
- **动作**:
  - `execute`: 执行搜索
  - `aggregate`: 聚合结果
- **调用位置**: `inference_engine.py:875`

### 3. **技能包工具** (Skill Packs)

#### AutoGPT Planner (`autogpt_planner`)
- **用途**: 任务规划和拆解
- **工具**:
  - `task_decompose`: 任务分解
  - `priority_rank`: 优先级排序
  - `dependency_analyze`: 依赖分析
- **注册位置**: `skill_packs/runtime.py:96`
- **调用方式**: 通过 ToolEngine 统一调度

#### OpenManus Reasoner (`openmanus_reasoner`)
- **用途**: 深度推理
- **工具**:
  - `deep_reasoning`: 深度推理分析
- **注册位置**: `skill_packs/runtime.py:96`

### 4. **系统工具**

#### DebugConsole (`debug_console`)
- **用途**: 系统调试
- **动作**:
  - 执行调试命令
  - 查看系统状态
- **调用位置**: 通过 bootstrap 直接启动

#### SystemTools
- **用途**: 系统管理
- **功能**: 配置管理、状态查询等

## 📈 工具调用流程

### 标准调用链路

```python
# 1. LLM 生成工具调用意图
tool_calls = response.choices[0].message.tool_calls

# 2. InferenceEngine 解析工具调用
for tool_call in tool_calls:
    function_name = tool_call.function.name
    args_dict = json.loads(tool_call.function.arguments)
    
    # 3. 通过通用工具执行器分发
    result = await _execute_tool_call(
        function_name, 
        args_dict, 
        tool_call.id,
        messages,
        links_text_global
    )
    
    # 4. ToolEngine 执行
    tool_result = await asyncio.to_thread(
        self.tool_engine.call_tool,
        function_name,      # 工具名称
        action,            # 动作
        args_dict          # 参数
    )
    
    # 5. 结果处理
    if tool_result.success:
        # 后处理（搜索类提取链接、记忆类读取详情）
        content = _format_tool_result(tool_result)
    else:
        # 错误处理
        content = f"工具执行失败：{tool_result.error}"
```

### 关键代码位置

#### InferenceEngine 中的工具调用

1. **主调用循环** (`inference_engine.py:599`)
```python
tool_result_msg = await self._execute_tool_call(
    function_name, args_dict, tool_call.id, messages, links_text_global
)
```

2. **通用工具执行器** (`inference_engine.py:652-740`)
```python
async def _execute_tool_call(self, function_name, args_dict, tool_call_id, messages, links_text_global):
    # 1. 确定 action
    action = args_dict.pop("action", "execute")
    
    # 2. 通过 ToolEngine 调用
    tool_result = await asyncio.to_thread(
        self.tool_engine.call_tool,
        function_name,
        action,
        args_dict
    )
    
    # 3. 后处理
    if function_name in ["openclaw_search", "web_search", "search_tools"]:
        # 搜索类工具：提取链接和摘要
        content = self._process_search_result(tool_result, links_text_global)
    else:
        # 通用工具：格式化结果
        content = f"工具 '{function_name}' 执行结果：{tool_result.get_data()}"
    
    return content
```

3. **专用搜索调用** (`inference_engine.py:749`)
```python
search_result = await asyncio.to_thread(
    self.tool_engine.call_tool,
    "openclaw_search",
    "search",
    {"query": query, "count": count}
)
```

4. **网页获取** (`inference_engine.py:796`)
```python
webpage_result = await asyncio.to_thread(
    self.tool_engine.call_tool,
    "openclaw_search",
    "fetch_webpage",
    {"url": url}
)
```

## 🎯 ToolEngine 核心功能

### 1. **工具注册管理**

```python
class ToolEngine:
    def __init__(self, max_workers: int = 5):
        self.registry = ToolRegistry()
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 注册内置工具
        self._register_builtin_tools()
    
    def register_tool(self, tool: BaseTool) -> bool:
        """注册工具到注册表"""
        return self.registry.register(tool)
```

### 2. **同步调用**

```python
def call_tool(
    self,
    tool_name: str,
    action: str,
    parameters: Dict[str, Any],
    timeout: float = 30.0,
    priority: int = 5,
    callback: Optional[Callable] = None
) -> ToolResult:
    # 1. 创建调用记录
    call = ToolCall(...)
    
    # 2. 获取工具
    tool = self.registry.get(tool_name)
    
    # 3. 检查状态
    if not tool.enabled:
        return ToolResult(success=False, error="Tool is disabled")
    
    # 4. 执行工具
    result = tool.execute(request)
    
    # 5. 处理 async 结果
    if inspect.iscoroutine(result):
        result = asyncio.run(result)
    
    # 6. 更新统计
    self.total_calls += 1
    if result.success:
        self.successful_calls += 1
    
    # 7. 记录历史
    self._record_call(call)
    
    return result
```

### 3. **异步调用**

```python
def call_tool_async(self, ...) -> Future:
    """异步调用工具，返回 Future 对象"""
    return self.executor.submit(
        self.call_tool,
        tool_name, action, parameters, timeout, priority, callback
    )
```

### 4. **批量调用**

```python
def call_batch(
    self,
    calls: List[Dict[str, Any]],
    parallel: bool = True
) -> List[ToolResult]:
    """批量调用工具"""
    if not parallel:
        # 串行执行
        return [self.call_tool(**call) for call in calls]
    
    # 并行执行
    futures = [self.call_tool_async(**call) for call in calls]
    return [future.result() for future in futures]
```

### 5. **调用历史管理**

```python
def get_call_history(
    self,
    tool_name: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """获取调用历史，支持过滤"""
    history = self.call_history.copy()
    
    if tool_name:
        history = [c for c in history if c.tool_name == tool_name]
    if status:
        history = [c for c in history if c.status == status]
    
    return history[-limit:]
```

## 📊 统计指标

### ToolEngine 统计

```python
class ToolEngine:
    # 统计信息
    self.total_calls = 0      # 总调用次数
    self.successful_calls = 0  # 成功调用次数
    self.failed_calls = 0     # 失败调用次数
    
    # 调用历史
    self.call_history: List[ToolCall]  # 最近 1000 次调用
    self.max_history_size = 1000
```

### ToolCall 记录

```python
@dataclass
class ToolCall:
    call_id: str           # 调用 ID
    tool_name: str         # 工具名称
    action: str            # 动作
    parameters: Dict       # 参数
    start_time: float      # 开始时间
    end_time: float        # 结束时间
    result: ToolResult     # 结果
    status: str            # pending/running/success/failed
    error: Optional[str]   # 错误信息
    
    def to_dict(self) -> Dict:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "action": self.action,
            "duration": self.end_time - self.start_time,
            "status": self.status,
            "error": self.error
        }
```

## 🔍 工具调用示例

### 示例 1: 网络搜索

```python
# InferenceEngine 调用
search_result = await asyncio.to_thread(
    self.tool_engine.call_tool,
    "openclaw_search",
    "search",
    {"query": "今天天气如何", "count": 5}
)

# ToolEngine 执行流程
# 1. 查找 openclaw_search 工具
# 2. 创建 ToolRequest
# 3. 调用 OpenClawSearchTool.execute()
# 4. 返回 ToolResult
# 5. 记录调用历史
```

### 示例 2: 技能包工具调用

```python
# AutoGPT Planner 任务分解
result = await asyncio.to_thread(
    self.tool_engine.call_tool,
    "autogpt_planner",
    "task_decompose",
    {"task": "帮我规划一次旅行", "max_subtasks": 5}
)

# 执行流程
# 1. ToolEngine 查找 autogpt_planner
# 2. 调用 AutoGPTPlannerPack 的工具
# 3. 执行任务分解逻辑
# 4. 返回分解后的子任务列表
```

## 🎛️ 配置参数

### ToolEngine 配置

```yaml
# config/zulong_config.yaml
tools:
  engine:
    max_workers: 5        # 最大并发线程数
    timeout: 60           # 默认超时时间 (秒)
  
  openclaw:
    enabled: true
    api_url: "http://localhost:3000"
    websocket_url: "ws://localhost:5555"
    timeout: 30
  
  web_search:
    enabled: true
    engine: "searxng"
    searxng_url: "http://localhost:8080"
    max_results: 5
  
  skill_packs:
    enabled: true
    enabled_packs:
      - "autogpt_planner"
      - "openmanus_reasoner"
```

### 环境变量

```bash
# 工具引擎配置
ZULONG_TOOLS_MAX_WORKERS=5
ZULONG_TOOLS_TIMEOUT=60

# OpenClaw 配置
ZULONG_OPENCLAW_ENABLED=true
ZULONG_OPENCLAW_API_URL=http://localhost:3000
ZULONG_OPENCLAW_WEBSOCKET_URL=ws://localhost:5555

# 网络搜索配置
ZULONG_WEB_SEARCH_ENABLED=true
ZULONG_WEB_SEARCH_ENGINE=searxng
ZULONG_SEARXNG_URL=http://localhost:8080
```

## 🚨 错误处理机制

### 1. 工具未找到

```python
tool = self.registry.get(tool_name)
if not tool:
    error_msg = f"Tool not found: {tool_name}"
    return ToolResult(success=False, error=error_msg)
```

### 2. 工具被禁用

```python
if not tool.enabled:
    error_msg = f"Tool is disabled: {tool_name}"
    return ToolResult(success=False, error=error_msg)
```

### 3. 执行异常

```python
try:
    result = tool.execute(request)
except Exception as e:
    error_msg = str(e)
    logger.error(f"[ToolEngine] Execute error: {e}")
    return ToolResult(
        success=False,
        error=error_msg,
        execution_time=time.time() - call.start_time
    )
```

### 4. 超时处理

```python
def call_tool(self, ..., timeout: float = 30.0):
    request = ToolRequest(timeout=timeout)
    # 工具执行时检查超时
    if time.time() - start_time > timeout:
        raise TimeoutError(f"Tool execution timeout: {timeout}s")
```

## 📈 性能优化

### 1. 线程池复用

```python
self.executor = ThreadPoolExecutor(max_workers=5)
# 所有工具调用共享线程池，避免频繁创建销毁线程
```

### 2. 调用历史限制

```python
self.max_history_size = 1000
# 超过限制时自动清理旧记录
if len(self.call_history) > self.max_history_size:
    self.call_history = self.call_history[-self.max_history_size:]
```

### 3. 异步执行

```python
# 使用 asyncio.to_thread 避免阻塞主事件循环
tool_result = await asyncio.to_thread(
    self.tool_engine.call_tool,
    tool_name, action, parameters
)
```

## 🔮 扩展方向

### 新增工具类型

1. **继承 BaseTool**
```python
from zulong.tools.base import BaseTool

class MyCustomTool(BaseTool):
    def __init__(self):
        super().__init__("my_tool", "我的自定义工具")
    
    def execute(self, request: ToolRequest) -> ToolResult:
        # 实现工具逻辑
        return ToolResult(success=True, data={"result": "ok"})
```

2. **注册到 ToolEngine**
```python
tool_engine = ToolEngine()
tool_engine.register_tool(MyCustomTool())
```

3. **通过 LLM 调用**
```python
# LLM 会自动发现并使用新工具
# 只需在 prompt 中提供工具描述
```

### 工具 RAG 集成

```python
# 通过 CoreToolManager 实现工具检索
core_tool_manager.register_tool(
    tool_name="my_tool",
    schema=function_schema,
    source="custom",
    category="custom"  # 自动分类为热/冷工具
)
```

## 📝 最佳实践

### 1. 工具设计原则

- ✅ **单一职责**: 每个工具只做一件事
- ✅ **无状态**: 工具不应保存状态
- ✅ **幂等性**: 多次执行结果一致
- ✅ **错误处理**: 返回清晰的错误信息
- ✅ **超时控制**: 设置合理的超时时间

### 2. 调用建议

- ✅ **使用 ToolEngine**: 统一调度和监控
- ✅ **异步调用**: 避免阻塞主线程
- ✅ **批量执行**: 减少调用开销
- ✅ **结果缓存**: 对重复调用缓存结果
- ✅ **日志记录**: 记录关键调用信息

### 3. 性能调优

- ✅ **合理设置 max_workers**: 根据 CPU 核心数调整
- ✅ **工具预加载**: 启动时加载常用工具
- ✅ **连接池**: 对网络工具使用连接池
- ✅ **超时分级**: 不同工具设置不同超时

---

**文档版本**: v1.0  
**最后更新**: 2026-04-16  
**适用版本**: ZULONG v2.0+
