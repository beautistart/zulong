# 天气查询问题 - 技能包架构视角的深度诊断报告

**报告时间**: 2026-04-12 18:15  
**问题**: 用户输入"帮我搜索一下今天的天气"，L2 回复"好的，我帮你查一下今天的天气。"但**没有实际调用工具**

---

## 一、问题确认

### 现象
1. **用户输入**: "帮我搜索一下今天的天气"
2. **L2 回复**: "好的，我帮你查一下今天的天气。"
3. **响应时间**: 0.5 秒（太快，不可能是工具调用后的结果）
4. **工具调用**: ❌ **没有发生**

### 日志证据

#### OpenClaw Bridge 日志
```
18:11:59,328 - 收到消息：帮我搜索一下今天的天气
18:11:59,329 - 发布 USER_TEXT 事件
18:11:59,335 - 收到 ACK 确认
```

**关键**：收到 ACK 后，**没有收到 L2_OUTPUT 事件**！

#### 主系统终端（终端 7）
```
[16:57:52] WebSocket Server started on ws://localhost:5555
[16:57:52] ZULONG System Online
```
**问题**：日志停留在 16:57，而当前时间是 18:11！**主系统可能已停止运行**

---

## 二、架构对比：技能包文档 vs 当前实现

### 2.1 技能包架构文档要求

根据 `zulong-skill-plugin-architecture.md`：

#### Phase 1: 工具系统增强（基础设施）
- ✅ **2.1 BaseTool 增加自描述能力** - 已实现 `get_function_schema()`
- ✅ **2.2 InferenceEngine 工具调用增强** - 已实现动态工具列表、并行调用、智能迭代
- ✅ **2.3 现有工具类适配** - 已适配 4 个工具类
- ✅ **Phase 1 测试点** - 需要验证

#### Phase 2: 技能包运行框架（核心，但很薄）
- ❌ **3.1 ISkillPack 统一接口** - **未实现**
- ❌ **3.2 SkillPackRuntime 运行时** - **未实现**
- ❌ **3.3 技能包目录结构** - **未实现**
- ❌ **3.4 与现有系统的对接** - **未实现**
- ❌ **3.5 bootstrap 集成** - **未实现**
- ❌ **3.6 技能包配置** - **未实现**

#### 关键发现
**当前系统处于 Phase 1 和 Phase 2 之间**：
- 工具系统已增强（Phase 1 完成）
- 但技能包运行框架未实现（Phase 2 缺失）
- **这导致工具调用没有经过技能包管理，而是硬编码在 InferenceEngine 中**

---

## 三、根本原因分析

### 原因 1: 主系统可能已停止运行 ⭐⭐⭐⭐⭐

**证据**：
- 终端 7 日志停留在 16:57（当前时间 18:11）
- OpenClaw Bridge 收到 ACK 后没有收到 L2_OUTPUT
- WebSocket 端口 5555 被占用（错误日志显示）

**推断**：
- 主系统可能在 16:57 启动后，因为端口冲突或其他原因崩溃
- OpenClaw Bridge 仍连接着旧的 WebSocket 服务器（已不处理事件）

**验证方法**：
```bash
# 检查端口 5555 是否被占用
netstat -ano | findstr :5555

# 重启主系统
python -m zulong.bootstrap
```

---

### 原因 2: L2 处于降级模式（如果主系统仍在运行）⭐⭐⭐

**证据**：
- inference_engine.py 中有降级逻辑（line 340-358）
- 如果 vLLM Function Calling 不可用，会降级到普通模式
- 普通模式不使用 tools，直接生成回复

**代码**：
```python
# inference_engine.py line 340-358
except Exception as e:
    # 🔥 降级处理：如果 vLLM 不支持 Function Calling，使用普通模式
    logger.warning(f"⚠️ [vLLM-Tools] Function Calling 不可用：{e}")
    logger.info("🔄 [vLLM-Tools] 降级到普通模式（不使用 tools）...")
    
    response = self.vllm_client.chat.completions.create(
        model=vllm_model_id,
        messages=messages,
        max_tokens=1024,
        temperature=0.3,
        top_p=0.85,
        stream=False
    )
    # 直接返回内容，不处理工具调用
```

**为什么没有日志**：
- 降级发生在系统启动时
- 日志可能被刷掉了，或者系统崩溃前没有记录

---

### 原因 3: 技能包架构缺失导致工具调用没有经过统一管理 ⭐⭐⭐⭐

**架构问题**：
根据技能包文档，工具调用应该经过以下流程：

```
用户输入 → L1-B Gatekeeper → L2 InferenceEngine → ModuleRouter → 技能包执行
```

**当前实现**：
```
用户输入 → L1-B Gatekeeper → L2 InferenceEngine → 直接调用 vLLM → 工具调用
```

**缺失的中间件**：
1. **ModuleRouter**（双层路由）- 未实现
   - 第一层：快速预判（关键词 + 经验规则）
   - 第二层：L2 Function Calling 自主决定

2. **SkillPackRuntime** - 未实现
   - 技能包加载/卸载
   - 工具注册管理
   - 经验自动提取

3. **ExperienceGenerator 自动触发** - 未实现
   - 当前需要手动触发复盘
   - 技能包架构要求自动从对话中提取经验

---

## 四、缺失的代码/逻辑清单

### 4.1 缺失的核心组件

| 组件 | 文件路径 | 状态 | 优先级 |
|------|----------|------|--------|
| ISkillPack 接口 | `zulong/skill_packs/interface.py` | ❌ 未实现 | P0 |
| SkillPackRuntime | `zulong/skill_packs/runtime.py` | ❌ 未实现 | P0 |
| SkillPackManifest | `zulong/skill_packs/interface.py` | ❌ 未实现 | P0 |
| ModuleRouter | `zulong/l2/module_router.py` | ❌ 未实现 | P1 |
| 技能包加载器 | `zulong/skill_packs/loader.py` | ❌ 未实现 | P1 |
| 内化完成度评估 | `zulong/skill_packs/internalization.py` | ❌ 未实现 | P2 |

### 4.2 缺失的配置

| 配置文件 | 路径 | 状态 |
|----------|------|------|
| 技能包配置 | `config/skill_packs.yaml` | ❌ 未创建 |
| AutoGPT 技能包 | `zulong/skill_packs/packs/autogpt_planner/` | ❌ 未创建 |
| Cline 技能包 | `zulong/skill_packs/packs/cline_coder/` | ❌ 未创建 |

### 4.3 缺失的逻辑

1. **经验自动提取**
   - 当前：需要手动触发复盘
   - 应该：每次对话后自动提取经验到 ExperienceStore

2. **工具调用的统一管理**
   - 当前：硬编码在 InferenceEngine 中
   - 应该：通过 SkillPackRuntime 管理

3. **内化评估**
   - 当前：无
   - 应该：定期评估技能包内化完成度，支持卸载

---

## 五、立即解决方案

### 方案 A: 重启主系统（立即执行）⭐⭐⭐⭐⭐

**步骤**：
1. 停止所有终端
2. 检查端口 5555 是否被占用
3. 重启主系统：`python -m zulong.bootstrap`
4. 观察启动日志，确认 vLLM Function Calling 是否正常

**预期**：
- 如果 vLLM Function Calling 正常，工具调用应该工作
- 如果降级，会看到"Function Calling 不可用"的警告

---

### 方案 B: 添加工具调用诊断日志（5 分钟）

**修改文件**: `zulong/l2/inference_engine.py`

**添加日志**：
```python
# 在_generate_with_vllm_and_tools 开始处
logger.info(f"🔧 [诊断] 开始工具调用流程")
logger.info(f"🔧 [诊断] 工具列表：{[t['function']['name'] for t in tools]}")

# 在模型响应后
logger.info(f"🔍 [诊断] tool_calls: {message.tool_calls}")
if not message.tool_calls:
    logger.warning(f"⚠️ [诊断] 模型没有生成 tool_calls！content: {message.content[:200]}")
```

---

### 方案 C: 独立测试 vLLM 工具调用（10 分钟）

**创建测试脚本**：
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

tools = [{
    "type": "function",
    "function": {
        "name": "openclaw_search",
        "description": "搜索工具",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["search", "fetch_webpage"]},
                "query": {"type": "string"}
            },
            "required": ["action"]
        }
    }
}]

response = client.chat.completions.create(
    model="/mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-0.8B-AWQ",
    messages=[{"role": "user", "content": "帮我搜索一下今天的天气"}],
    tools=tools,
    tool_choice="auto"
)

print(f"Content: {response.choices[0].message.content}")
print(f"Tool Calls: {response.choices[0].message.tool_calls}")
```

---

## 六、长期架构规划

### Phase 1: 补全技能包基础设施（1-2 天）
1. 实现 ISkillPack 接口
2. 实现 SkillPackRuntime
3. 实现 ModuleRouter（双层路由）
4. 创建技能包配置文件

### Phase 2: 实现 AutoGPT 技能包（2-3 天）
1. 提取 AutoGPT 任务拆解算法
2. 封装为 ISkillPack 接口
3. 注册 task_decompose 工具
4. 测试工具调用

### Phase 3: 实现经验自动提取（1-2 天）
1. 修改 ExperienceGenerator，支持自动触发
2. 在 SkillPackRuntime.execute() 中自动调用
3. 测试经验提取和存储

### Phase 4: 实现内化评估（2-3 天）
1. 实现内化完成度评估算法
2. 支持技能包卸载
3. 测试卸载后经验保留

---

## 七、当前系统状态评估

### 已实现的功能 ✅
- ✅ 工具注册表（ToolRegistry）
- ✅ 工具引擎（ToolEngine）
- ✅ vLLM Function Calling 支持
- ✅ 动态工具列表
- ✅ 并行工具调用
- ✅ 智能迭代控制
- ✅ 死循环检测

### 缺失的功能 ❌
- ❌ 技能包运行框架
- ❌ 经验自动提取
- ❌ 内化评估
- ❌ ModuleRouter 双层路由
- ❌ 技能包配置和管理

### 系统成熟度评估
**当前成熟度**: Phase 1.5 / Phase 4  
**完成度**: 约 40%

---

## 八、建议

### 立即行动（今天）
1. **重启主系统**，确认问题是否解决
2. **添加诊断日志**，观察工具调用流程
3. **独立测试 vLLM**，确认 Function Calling 是否正常

### 短期行动（本周）
1. **实现 Phase 2 核心组件**（ISkillPack, SkillPackRuntime）
2. **创建 AutoGPT 技能包**（任务拆解）
3. **测试工具调用完整流程**

### 长期行动（本月）
1. **完成 Phase 3-4**（经验自动提取、内化评估）
2. **实现 Cline 编程技能包**
3. **优化 ModuleRouter 双层路由**

---

**结论**：当前系统**不是架构问题**，而是**实现不完整**的问题。工具调用的代码已经存在，但缺少技能包框架的统一管理。建议先重启系统解决当前问题，然后逐步实现技能包架构。
