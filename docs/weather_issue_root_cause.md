# 天气查询问题 - 根本原因分析

## 问题确认

**现象**：
- 用户输入："帮我搜索一下今天的天气"
- L2 回复："好的，我帮你查一下今天的天气。"
- **没有实际调用搜索工具**
- 响应时间：0.5 秒（太快，不可能是工具调用后的结果）

## 日志证据

### OpenClaw Bridge 日志
```
17:41:35,359 - 收到消息：帮我搜索一下今天的天气
17:41:35,361 - 发布 USER_TEXT 事件
17:41:35,365 - 收到 ACK 确认
17:41:35,912 - 收到 L2_OUTPUT："好的，我帮你查一下今天的天气。"
```

**关键**：从 USER_TEXT 到 L2_OUTPUT 只有 **0.5 秒**！

### 正常工具调用流程应该需要：
1. L2 接收输入：0.1 秒
2. 模型生成 tool_calls：2-5 秒
3. 执行工具调用：3-10 秒
4. 模型根据工具结果生成回复：2-5 秒
**总计：7-20 秒**

**实际：0.5 秒** → 说明没有调用工具！

---

## 可能的根本原因

### 原因 1: L2 处于降级模式（最可能）⭐⭐⭐⭐⭐

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
- 可能没有看到警告日志，因为日志被刷掉了

---

### 原因 2: 模型没有生成 tool_calls

**可能性**：模型认为不需要调用工具

**原因**：
1. 模型太小（Qwen3.5-0.8B），无法正确识别工具调用场景
2. messages 中的 system prompt 没有正确引导模型使用工具
3. tools 定义格式不正确，模型无法理解

**验证方法**：
查看 L2 日志中是否有：
```
🚀 [vLLM-Tools] 模型响应：
   - content: ...
   - tool_calls: None  # ← 如果是 None，说明模型没有生成 tool_calls
```

---

### 原因 3: vLLM 工具调用配置问题

**检查 vLLM 启动参数**：
```bash
--enable-auto-tool-choice --tool-call-parser qwen3_xml
```

**查看 WSL 终端**（终端 8）的启动命令是否包含这些参数。

---

## 解决方案

### 方案 A: 检查 L2 启动日志（立即执行）

在祖龙主系统终端（终端 7）中查找：
```
🚨 [vLLM-Tools] Function Calling 不可用
🔄 [vLLM-Tools] 降级到普通模式
```

如果有，说明 L2 已经降级到不使用工具的模式。

**解决**：
1. 检查 vLLM 是否支持 Function Calling
2. 检查 tools 定义格式是否正确
3. 可能需要升级 vLLM 或调整 tools 定义

---

### 方案 B: 强制 L2 使用工具（修改代码）

**修改 inference_engine.py**，添加工具调用前的日志：

```python
# 在 call_vllm_with_tools 之前添加
logger.info(f"🔧 [vLLM-Tools] 开始调用 vLLM，使用 tools: {tools[:1]}...")  # 只打印第一个 tool
```

**在模型响应后添加**：
```python
logger.info(f"🔍 [vLLM-Tools] 检查 tool_calls: {message.tool_calls}")
if not message.tool_calls:
    logger.warning(f"⚠️ [vLLM-Tools] 模型没有生成 tool_calls！content: {message.content[:200]}")
```

---

### 方案 C: 测试 vLLM 工具调用（独立测试）

创建测试脚本直接调用 vLLM：

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
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        }
    }
}]

response = client.chat.completions.create(
    model="Qwen3___5-0.8B-AWQ",
    messages=[{"role": "user", "content": "帮我搜索一下今天的天气"}],
    tools=tools,
    tool_choice="auto"
)

print(f"Content: {response.choices[0].message.content}")
print(f"Tool Calls: {response.choices[0].message.tool_calls}")
```

---

## 下一步行动

1. **查看 L2 启动日志**，确认是否降级
2. **创建 vLLM 工具调用测试脚本**，独立测试 vLLM
3. **如果 vLLM 不支持 Function Calling**，考虑：
   - 升级 vLLM 版本
   - 使用更大的模型
   - 改用本地工具调用逻辑

---

**报告生成时间**：2026-04-12 17:45  
**状态**：等待进一步诊断
