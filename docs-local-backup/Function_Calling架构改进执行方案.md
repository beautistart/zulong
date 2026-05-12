# 祖龙系统 Function Calling 架构改进执行方案

## 问题诊断总结

### 根本原因

系统用**硬编码规则**替代了 LLM 的**语义判断能力**，导致技能包从未被调用。

### 四个病灶

| # | 病灶 | 位置 | 后果 |
|---|------|------|------|
| 1 | 工具执行器只认识 2 个工具 | `inference_engine.py:627-786` 的 if/elif/else | LLM 调用 `task_decompose` 等工具时，返回"工具不支持" |
| 2 | 双层关键词预路由 | `module_router.py` + `inference_engine.py:372-416` | 用关键词匹配代替 LLM 语义判断 |
| 3 | 强制工具指定 | `inference_engine.py:1184-1195` force_tool_call 注入 | 所有"复杂任务"被锁定到 `openclaw_search` |
| 4 | 工具描述不充分 | `_build_tools_description():1455-1491` 硬编码场景 | LLM 不知道何时该用技能包工具 |

### 证据链

- 日志显示 8 个工具已注册：`['openclaw_tool', 'openclaw_search', 'openclaw_plugin', 'web_search', 'task_decompose', 'priority_rank', 'dependency_analyze', 'deep_reasoning']`
- 但只有 `openclaw_search` 被实际调用
- `web_search` 被 LLM 尝试调用后返回"未知工具"
- ExperienceGenerator 提取 0 条经验候选
- `is_complex_task` 在 zulong/ 核心代码中 0 次引用

---

## 执行任务清单

### 任务 1：建立通用工具执行器 [P0 必须]

**目标**：让 inference_engine 能执行 ToolRegistry 中的所有已注册工具，而不是只认识 2 个。

**改动文件**：`zulong/l2/inference_engine.py`

**改动范围**：第 627-786 行

**当前代码结构**（有病的）：
```python
if function_name == "openclaw_search":
    # ~100 行：执行搜索、读网页、格式化结果
elif function_name == "read_memory_detail":
    # ~40 行：读取记忆详情
else:
    # 其他工具一律返回"工具不支持"
    messages.append({"role": "tool", "content": "工具不支持"})
```

**改为**（正常的）：
```python
# 通用工具执行
tool_result = await self._execute_tool_call(function_name, args_dict, tool_call.id)
messages.append(tool_result)
```

**具体步骤**：

1. 新增方法 `_execute_tool_call(self, function_name, args_dict, tool_call_id) -> dict`：
   - 调用 `self.tool_engine.call_tool(function_name, action, args_dict)` 进行通用分发
   - 根据工具类型做后处理：
     - **搜索类**（openclaw_search）：提取链接、智能读取网页（保留现有逻辑）
     - **记忆类**（read_memory_detail）：读取记忆详情（保留现有逻辑）
     - **技能包类**（task_decompose、deep_reasoning、priority_rank、dependency_analyze）：将结果 JSON 序列化后返回
     - **代码类**（read_file、write_file、edit_code 等）：返回执行结果
     - **其他**：通用格式化返回
   - 统一返回 `{"role": "tool", "tool_call_id": ..., "name": ..., "content": ...}` 格式

2. 将现有的 `openclaw_search` 处理逻辑（627-719 行）提取为 `_handle_search_tool()` 私有方法
3. 将 `read_memory_detail` 处理逻辑（727-775 行）提取为 `_handle_memory_tool()` 私有方法
4. 删除 `else: "工具不支持"` 分支

**验证方法**：
- 发送"帮我分析 AI 助手市场竞争格局"，查看日志是否出现 `task_decompose` 或 `deep_reasoning` 的调用
- 确认不再出现"未知工具"日志

---

### 任务 2：移除硬编码预路由 [P0 必须]

**目标**：删除关键词预路由，让 LLM 自主决定是否调用工具。

**改动文件**：`zulong/l2/inference_engine.py`

**步骤**：

1. **删除 `_detect_needs_tools()` 方法**（第 372-416 行）
   - 此方法通过硬编码关键词列表判断是否强制调用工具
   - 删除后，LLM 通过 Function Calling 的 `tool_choice="auto"` 自主判断

2. **删除主流程中的 force_tool_call 逻辑**（第 1184-1195 行）
   ```python
   # 删除以下代码：
   force_tool_call = self._detect_needs_tools(user_input)
   if force_tool_call:
       messages.append({
           "role": "system",
           "content": "【重要提醒】用户的问题需要你调用 openclaw_search 工具..."
       })
   ```

3. **修改 `_generate_with_vllm_and_tools()` 签名**（第 418 行）
   - 移除 `force_tool_call` 参数
   - 第 510 行：`tool_choice` 始终使用 `"auto"`
   ```python
   # 改前：
   current_tool_choice = "required" if (force_tool_call and iteration == 0) else "auto"
   # 改后：
   current_tool_choice = "auto"
   ```

4. **修改调用点**（第 1202 行、1215 行）
   ```python
   # 改前：
   response = await self._generate_with_vllm_and_tools(messages, force_tool_call=force_tool_call)
   # 改后：
   response = await self._generate_with_vllm_and_tools(messages)
   ```

**注意**：`module_router.py` 的 `quick_class()` 虽然也是硬编码预路由，但 grep 确认它在生产代码中未被调用（仅在 `__init__.py` 中导出、在测试中使用），暂不改动，后续清理。

---

### 任务 3A：增强工具 description [P0 必须]

**目标**：让每个工具的 description 足够详细，使 LLM 能准确判断何时调用。

**改动文件**：

1. `zulong/skill_packs/packs/autogpt_planner/tools.py`
   - `TaskDecomposeTool.description` → 增加适用场景、输入输出说明
   - `PriorityRankTool.description` → 同上
   - `DependencyAnalyzeTool.description` → 同上

2. `zulong/skill_packs/packs/openmanus_reasoner/__init__.py`（第 52 行）
   - `DeepReasoningTool.description` → 增加适用场景说明

3. `zulong/skill_packs/packs/cline_coder/__init__.py`
   - `FileReadTool`、`FileWriteTool`、`CodeEditTool`、`TerminalTool`、`CodeSearchTool` 的 description

**description 编写规范**：
每个工具的 description 必须包含：
- **一句话说明**：这个工具干什么
- **适用场景**：什么情况下应该调用（列举 2-3 个典型场景）
- **输入说明**：主要参数是什么
- **输出说明**：返回什么格式的结果

**示例**：

TaskDecomposeTool 当前：
> "任务拆解工具"

改为：
> "将复杂的多步骤任务拆解为可执行的子任务列表。当用户的请求包含多个步骤（如'调研+分析+设计'）、需要先后顺序完成、或涉及多个领域时，必须先调用此工具进行任务规划。输入：goal(任务目标)、context(可选上下文)。输出：结构化子任务列表，含优先级和依赖关系。"

DeepReasoningTool 当前：
> "深度推理工具。适用于算法设计、数学证明、复杂系统架构等高难度任务。"

改为：
> "深度推理工具。当任务需要多步逻辑推理、方案对比分析、优劣势评估、因果关系推导时调用。适用于：市场竞争分析、技术方案评估、产品优缺点对比、战略决策推理。输入：problem(问题描述)、context(可选上下文)。输出：包含分析步骤、假设验证和结论的推理链。"

---

### 任务 3B：改造场景描述生成 [P1 重要]

**目标**：`_build_tools_description()` 不再硬编码使用场景，改为从工具 registry 动态生成。

**改动文件**：`zulong/l2/inference_engine.py`（第 1455-1491 行）

**当前代码**：
```python
# 硬编码场景：只描述了搜索相关场景
description_parts.append("**必须调用工具的场景**：")
description_parts.append("1. 用户需要实时信息（新闻、天气、价格）")
description_parts.append("2. 用户要求搜索、查询、查找信息")
# ... 全是搜索场景
```

**改为**：
```python
# 从工具 registry 动态生成
for tool in tools:
    if tool.get('enabled', True):
        desc = tool['description']
        # description 本身已包含使用场景（任务3A确保了这一点）
        description_parts.append(f"- {tool['name']}: {desc}")

# 通用调用规则（不再列举具体场景）
description_parts.append("")
description_parts.append("**调用规则**：")
description_parts.append("- 根据用户需求自主判断是否需要调用工具")
description_parts.append("- 复杂任务先用 task_decompose 拆解，再逐步执行子任务")
description_parts.append("- 需要搜索信息时用 openclaw_search")
description_parts.append("- 需要深度分析时用 deep_reasoning")
description_parts.append("- 简单闲聊不需要调用工具")
```

---

### 任务 4：增强工具参数 Schema [P1 重要]

**目标**：确保每个技能包工具的 `_get_parameters_schema()` 返回完整的参数描述。

**改动文件**：各技能包工具类

**检查清单**：
- [ ] `TaskDecomposeTool._get_parameters_schema()` - 确认 goal、context 参数有 description
- [ ] `PriorityRankTool._get_parameters_schema()` - 确认 subtasks 参数有 description 和示例
- [ ] `DependencyAnalyzeTool._get_parameters_schema()` - 同上
- [ ] `DeepReasoningTool._get_parameters_schema()` - 已有良好 schema（第 85-95 行），确认无遗漏
- [ ] ClineCoder 的 5 个工具 - 确认参数 schema 完整

**标准**：每个参数需有 `type`、`description`（含示例），`required` 数组准确。

---

### 任务 5：移除前端"复杂任务模式"复选框 [P2 一般]

**目标**：删除未实现且设计方向错误的功能。

**改动文件**：
1. 前端 Web UI：移除"复杂任务模式"复选框
2. `openclaw_bridge/adapters/web_adapter.py`：移除 `is_complex_task` 相关代码

---

### 任务 6：任务编排循环 [P2 自动]

**说明**：无需额外代码。完成任务 1（通用工具执行器）和任务 3（工具描述增强）后，现有的多轮迭代循环（`inference_engine.py` 第 502-816 行，最多 10 次迭代）自然支持以下编排模式：

```
第1轮 LLM 推理 → 调用 task_decompose → 返回子任务列表
第2轮 LLM 推理 → 调用 openclaw_search → 搜索子任务1的信息
第3轮 LLM 推理 → 调用 openclaw_search → 搜索子任务2的信息
第4轮 LLM 推理 → 调用 deep_reasoning → 深度分析汇总
第5轮 LLM 推理 → 不调用工具 → 生成最终报告
```

---

## 依赖关系与执行顺序

```
任务 1（通用工具执行器）──┐
                         ├── 同时开始，互不依赖
任务 2（移除预路由）──────┘
         ↓
任务 3A（增强工具description）──┐
                               ├── 可同时进行
任务 3B（改造场景描述生成）─────┘
         ↓
任务 4（增强参数Schema）
         ↓
任务 5（移除复选框）
         ↓
任务 6（自动生效，无需代码）
```

---

## 验证方案

完成所有任务后，使用 `docs/复杂任务测试方案.md` 中的 5 个测试任务进行验证：

| 测试任务 | 期望调用的工具 | 验证重点 |
|---------|--------------|---------|
| 任务1：市场研究 | task_decompose → openclaw_search → deep_reasoning | 多工具协同 |
| 任务2：技术趋势 | task_decompose → openclaw_search → deep_reasoning | 多轮搜索 |
| 任务3：产品开发 | task_decompose → openclaw_search → deep_reasoning → cline_coder | 跨类型工具 |
| 任务4：新闻分析 | openclaw_search → deep_reasoning | 时效性搜索+分析 |
| 任务5：代码生成 | task_decompose → openclaw_search → cline_coder | 代码工具 |

**关键日志检查点**：
- 不再出现 `🔧 [vLLM-Tools] 未知工具` 日志
- 不再出现 `🔧 [预路由] 匹配搜索关键词` 日志
- 出现 `task_decompose`、`deep_reasoning` 等技能包工具的调用日志
- ExperienceGenerator 能提取到经验候选（>0）

---
---

# 第二阶段：Tool RAG - 工具动态检索与注入机制

## 问题背景

随着系统使用时间增长，注册的技能包和工具会越来越多。当前方案（任务 1-6）把所有工具的 Function Schema 和描述全部放进 prompt，在工具数量较少时（8-15 个）可以工作。但当工具膨胀到 30-50+ 个时：

- **物理溢出**：仅工具 Schema 就可能消耗 20,000-35,000 tokens，超出模型上下文窗口
- **质量下降**：工具太多时 LLM 的选择准确率急剧下降（注意力稀释）
- **启动变慢**：每轮推理都要传送大量工具定义

## 核心思路

把工具检索做成与经验 RAG、记忆 RAG 一样的机制：
1. **工具摘要** → 类似经验文档，每个工具有一张"摘要卡片"
2. **向量索引** → 摘要向量化后存入 FAISS，支持语义搜索
3. **按需检索** → LLM 发现核心工具不够用时，主动调用元工具查找更多工具
4. **动态注入** → 检索到的工具 Schema 临时加入当前轮次的 tools 列表

## 大白话解释

**餐厅比喻**：

餐厅从 8 道菜扩张到 200 道菜。不可能把 200 道菜的菜单全塞给服务员背。解决方案：

1. **常备菜单**（5-8 道最常点的菜）：服务员随身携带
2. **菜品目录**（全部 200 道菜的分类索引）：放在厨房
3. **查菜技能**：服务员有一个特殊能力——当顾客要的菜不在常备菜单上时，去目录里查
4. **临时补菜单**：查到后，把那几道菜的详情临时加到手上的菜单里

## 架构设计

### 分层结构

```
┌──────────────────────────────────────────────────────────┐
│                  ToolRegistry (全量注册)                    │
│  所有工具的实例仍然全部注册在这里，用于执行（不变）          │
│  [30-50 个工具实例]                                       │
└────────────────────────┬─────────────────────────────────┘
                         │
           ┌─────────────┼─────────────┐
           │             │             │
           ▼             ▼             ▼
    ┌───────────┐  ┌──────────┐  ┌──────────────────┐
    │  核心工具集 │  │ ToolRAG  │  │  search_tools    │
    │  (热工具)   │  │ (冷工具) │  │  (元工具)         │
    │  5-8 个     │  │ 向量索引 │  │  LLM 用来查找    │
    │  始终在     │  │ 摘要存储 │  │  更多工具的工具   │
    │  prompt 里  │  │ FAISS    │  │                  │
    └───────────┘  └──────────┘  └──────────────────┘
```

### 推理流程

```
第 0 步：构建 prompt
  tools = 核心工具集(5-8个) + search_tools(1个)
  System Prompt 只包含核心工具描述 + search_tools 描述
  ↓
第 1 轮推理：LLM 判断
  ├─ 核心工具能满足 → 直接调用核心工具（大部分场景）
  └─ 核心工具不够 → 调用 search_tools("CAD 制图")
  ↓
search_tools 执行：
  1. 将 "CAD 制图" 向量化（bge-small-zh-v1.5）
  2. 在 ToolRAG FAISS 索引中搜索 top-3 匹配
  3. 获取匹配工具的完整 Function Schema
  4. 动态注入到当前轮次的 tools 列表
  5. 返回给 LLM："已找到 3 个相关工具：open_autocad, create_drawing, ..."
  ↓
第 2 轮推理：LLM 看到新增的工具
  → 调用 open_autocad(...)  ← 这个工具上一轮还不存在于 tools 列表中
  ↓
后续轮次：正常工具调用循环
```

### 核心组件设计

#### 1. 工具摘要卡片（ToolSummary）

每个工具在 ToolRAG 中存储一张摘要卡片：

```python
@dataclass
class ToolSummary:
    tool_name: str          # "cad_automation"
    display_name: str       # "CAD 自动化工具"
    summary: str            # 200字以内的能力描述+适用场景
    category: str           # "desktop_automation"
    keywords: List[str]     # ["CAD", "AutoCAD", "制图", "绘图"]
    capabilities: List[str] # ["open_autocad", "create_drawing"]
    source_pack: str        # "cad_automation_skill"
    usage_count: int        # 历史调用次数
    success_rate: float     # 成功率
    last_used: float        # 最后使用时间戳
    full_schema: Dict       # 完整 Function Calling schema（不入向量，仅存储）
```

**向量化内容**：`summary` + `keywords` 拼接 → bge-small-zh-v1.5 编码 → 512 维向量 → FAISS 索引。

#### 2. ToolRAG 库

继承现有 BaseRAGLibrary，与 ExperienceRAG / MemoryRAG 平级：

```python
class ToolRAG(BaseRAGLibrary):
    """工具 RAG 库
    
    与 ExperienceRAG 的类比：
    - ExperienceRAG：存储任务执行经验 → 语义搜索匹配相关经验
    - ToolRAG：存储工具摘要卡片 → 语义搜索匹配需要的工具
    """
    
    def add_tool(self, tool_summary: ToolSummary) -> str:
        """注册工具摘要到索引"""
        # 1. 构造 RAGDocument
        # 2. 向量化 summary + keywords
        # 3. 写入 FAISS 索引
        
    def search_tools(self, query: str, top_k: int = 3) -> List[ToolSummary]:
        """语义搜索匹配的工具"""
        # 1. query 向量化
        # 2. FAISS 搜索
        # 3. 返回匹配的 ToolSummary 列表
        
    def remove_tool(self, tool_name: str) -> bool:
        """从索引中移除工具"""
        
    def update_usage_stats(self, tool_name: str, success: bool):
        """更新工具的使用统计"""
```

#### 3. search_tools 元工具

```python
class SearchToolsTool(BaseTool):
    """工具检索元工具
    
    这是一个特殊的工具——它的功能是帮 LLM 查找其他工具。
    当 LLM 发现核心工具不足以完成当前任务时，调用此工具。
    """
    name = "search_tools"
    description = (
        "工具查找器。当你需要的能力不在当前可用工具列表中时调用此工具。"
        "例如：需要 CAD 制图时调用 search_tools(query='CAD制图')，"
        "需要发送邮件时调用 search_tools(query='邮件发送')。"
        "找到的工具会自动添加到你的可用工具中，下一步你就可以直接调用它们。"
    )
    
    def _get_parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "描述你需要的能力，如'CAD制图'、'发送邮件'、'数据库操作'"
                },
                "top_k": {
                    "type": "integer", 
                    "description": "返回最相关的工具数量，默认3"
                }
            },
            "required": ["query"]
        }
    
    def execute(self, request):
        query = request.parameters["query"]
        top_k = request.parameters.get("top_k", 3)
        
        # 1. 在 ToolRAG 中语义搜索
        matched = self.tool_rag.search_tools(query, top_k=top_k)
        
        # 2. 获取完整 schema，注入到推理上下文
        for tool_summary in matched:
            self.inference_context.inject_tool(tool_summary.full_schema)
        
        # 3. 返回摘要给 LLM
        summary_text = "\n".join([
            f"- {t.tool_name}: {t.summary[:100]}"
            for t in matched
        ])
        return self._create_result(
            success=True,
            data=f"已找到 {len(matched)} 个相关工具并激活：\n{summary_text}\n"
                 f"你现在可以直接调用这些工具了。"
        )
```

#### 4. 核心工具集管理

```python
class CoreToolManager:
    """核心工具集管理器
    
    决定哪些工具始终在 prompt 中（热工具），
    哪些工具需要通过 search_tools 检索（冷工具）。
    """
    
    # 初始核心工具集（手动指定）
    INITIAL_CORE_TOOLS = [
        "openclaw_search",    # 联网搜索（最基础）
        "task_decompose",     # 任务拆解（复杂任务必备）
        "deep_reasoning",     # 深度推理（分析必备）
        "search_tools",       # 元工具（查找更多工具，必须在核心集中）
    ]
    
    def get_core_tools(self) -> List[str]:
        """返回当前核心工具名列表"""
        # 初始：使用 INITIAL_CORE_TOOLS
        # 进阶：根据使用频率动态调整
        
    def promote_to_core(self, tool_name: str):
        """将高频工具提升为核心工具"""
        
    def demote_from_core(self, tool_name: str):
        """将低频核心工具降级为冷工具"""
```

#### 5. 动态注入机制

改造 `_generate_with_vllm_and_tools()` 的迭代循环：

```python
class InferenceToolContext:
    """推理过程中的工具上下文（每次推理独立）"""
    
    def __init__(self, core_tools_schemas):
        self.active_tools = list(core_tools_schemas)  # 初始 = 核心工具
        self._injected_names = set()
    
    def inject_tool(self, schema: dict):
        """动态注入工具 schema（由 search_tools 调用）"""
        name = schema["function"]["name"]
        if name not in self._injected_names:
            self.active_tools.append(schema)
            self._injected_names.add(name)
    
    def get_current_tools(self) -> list:
        """返回当前活跃的工具列表"""
        return self.active_tools
```

在迭代循环中：
```python
# 每轮推理使用动态工具列表
for iteration in range(max_tool_iterations):
    current_tools = inference_context.get_current_tools()
    
    response = self.vllm_client.chat.completions.create(
        model=model_id,
        messages=messages,
        tools=current_tools,  # 动态的，可能每轮不同
        tool_choice="auto",
        ...
    )
```

### 与现有 RAG 体系的对齐

| 组件 | 现有 ExperienceRAG | Tool RAG（新增） |
|------|-------------------|-----------------|
| 存什么 | 任务执行经验 | 工具摘要卡片 |
| 向量化什么 | 经验文本内容 | summary + keywords |
| 向量存储 | FAISS（512维） | 复用同一 FAISS 实例 |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 | 复用同一嵌入模型 |
| 去重机制 | SimHash + 向量相似度 | 工具名唯一性 + 向量相似度 |
| 检索触发 | 系统自动（每轮对话前） | LLM 主动（调用 search_tools） |
| 检索结果用途 | 注入 System Prompt 的参考知识 | 注入 tools 列表 |
| 管理入口 | RAGManager.add_experience() | RAGManager.add_tool_summary() |
| 持久化 | FAISS 索引 + JSON | 同上 |

### 工具索引的自动维护

**注册（技能包安装时自动）**：
```
SkillPackRuntime.install_pack()
  → 注册工具到 ToolRegistry（执行用，现有流程不变）
  → 为每个工具生成 ToolSummary（新增）
  → 将 ToolSummary 写入 ToolRAG（新增）
```

**注销（技能包卸载时自动）**：
```
SkillPackRuntime.uninstall_pack()
  → 从 ToolRegistry 注销工具（现有流程不变）
  → 从 ToolRAG 中删除对应摘要（新增）
```

**统计更新（每次工具调用后）**：
```
工具执行完成后：
  → ToolRAG.update_usage_stats(tool_name, success=True/False)
  → 更新 usage_count 和 success_rate
```

### 核心工具集动态调整（进阶，可选）

```
定期评估（如每 100 次任务后）：
  1. 统计所有工具的调用频率
  2. 如果某个冷工具调用频率 > 阈值 → 提升为核心工具
  3. 如果某个核心工具长期未被调用 → 降级为冷工具
  4. 核心工具集始终维持在 5-8 个（含 search_tools）
```

## 改动文件清单

### 新增文件

| 文件 | 内容 | 依赖 |
|------|------|------|
| `zulong/memory/tool_rag.py` | ToolRAG 库，继承 BaseRAGLibrary | base_rag_library.py |
| `zulong/tools/search_tools.py` | search_tools 元工具 | tool_rag.py, base.py |
| `zulong/tools/core_tool_manager.py` | 核心工具集管理器 | tool_rag.py |

### 修改文件

| 文件 | 改动 |
|------|------|
| `zulong/memory/rag_manager.py` | 新增 "tool" RAG 库到 rag_libraries |
| `zulong/skill_packs/runtime.py` | install_pack/uninstall_pack 时同步维护 ToolRAG |
| `zulong/l2/inference_engine.py` | (1) 构建 prompt 时只注入核心工具 (2) 迭代循环支持动态 tools 列表 (3) search_tools 执行后注入新工具 |
| `zulong/tools/tool_engine.py` | 注册 search_tools 为内置工具 |

## 执行顺序

```
第一阶段（任务 1-6）必须先完成：通用工具执行器 + 移除预路由
  ↓  ← 第一阶段验证通过后
第二阶段：Tool RAG
  步骤 1：实现 ToolRAG 库（tool_rag.py）
  步骤 2：实现 search_tools 元工具（search_tools.py）
  步骤 3：改造 SkillPackRuntime 自动维护 ToolRAG
  步骤 4：改造 inference_engine 支持动态 tools 列表
  步骤 5：实现 CoreToolManager
  步骤 6：集成测试
```

## 验证方案

1. **基础验证**：注册 20+ 个模拟工具，确认核心工具集只有 5-8 个在 prompt 中
2. **检索验证**：发送"帮我用 AutoCAD 画一个零件图"，确认 LLM 调用 search_tools 后能找到 CAD 相关工具
3. **端到端验证**：发送复杂任务，确认 LLM 能通过 search_tools → 找到工具 → 调用工具 的完整链路
4. **Token 验证**：对比全量注入 vs Tool RAG 方案的 prompt token 消耗
