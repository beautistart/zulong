# L2 模型生成截断问题分析

**分析时间**: 2026-03-25 23:10  
**问题等级**: 🟠 中等 (影响回复质量)  
**状态**: 🔍 分析中  

---

## 📋 问题现象

**用户输入**: `我在干嘛`

**模型输出**:
```
🔍 [DEBUG] 模型原始输出 (Raw Output): 帮我的吗？...  (仅 5 字符)
🔍 [DEBUG] 原始回复长度：5 字符
💬 生成回复长度：5 字符
💬 回复前 200 字符：'帮我的吗？'
```

**预期输出**: 应该生成完整的回复，例如:
```
"您好！根据我的观察，您正在测试祖龙系统的语音对话功能。"
```

---

## 🔍 可能原因分析

### 原因 1: 默认 max_tokens 过小 ❓

**检查点**: `zulong/models/engine.py` 第 347 行

```python
def generate(self, prompt: str, max_tokens: int = 50, ...)
```

**实际调用**: `inference_engine.py` 第 286 行
```python
response = self.l2_model.generate(
    prompt,
    max_tokens=1024,  # 明确指定 1024
)
```

**结论**: ✅ 调用时已指定 1024，不是默认值 50 的问题

---

### 原因 2: 模型遇到 stop token ❓

**检查点**: 生成配置中是否设置了 stop token

**当前配置**:
```python
generate_kwargs = {
    "max_new_tokens": max_tokens,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1
}
```

**问题**: ❌ 没有设置 stop token，应该不是这个原因

---

### 原因 3: 显存不足导致提前终止 ❓

**检查**: 查看系统显存使用情况

**可能**: 如果显存不足，模型可能会提前终止生成

**验证方法**:
```python
import torch
print(f"显存使用：{torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"显存上限：{torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
```

---

### 原因 4: 模型本身输出质量差 ❓

**最可能的原因**: Qwen3.5-0.8B 模型本身输出就有问题

**分析**:
1. 模型只有 0.8B 参数，能力有限
2. 可能没有正确理解问题
3. 可能输出被截断在思维链之前

**验证**: 查看完整的模型输出 (包括 <think> 标签)

---

## 🧪 调试方案

### 方案 1: 增加日志详细度

**修改**: `inference_engine.py` 第 291 行

```python
# 在提取</think>之前打印完整输出
logger.info(f"🔍 [DEBUG] 模型完整原始输出:\n{raw_response}")
```

**目的**: 查看是否有 <think> 标签被截断

---

### 方案 2: 测试模型直接调用

**脚本**: `test_model_direct.py`

```python
from zulong.models.container import ModelContainer
from zulong.models.config import ModelID

container = ModelContainer()
model = container.get_model(ModelID.L2_CORE)

prompt = """<|im_start|>system
你是一个友好的 AI 助手。<|im_end|>
<|im_start|>user
我在干嘛<|im_end|>
<|im_start|>assistant
"""

response = model.generate(prompt, max_tokens=256)
print(f"完整输出:\n{response}")
```

**目的**: 排除其他代码干扰，直接测试模型

---

### 方案 3: 检查 Prompt 构建

**修改**: 添加 Prompt 打印

```python
logger.info(f"📝 构建的 Prompt:\n{prompt}")
```

**目的**: 确认 Prompt 格式正确

---

## 📊 数据对比

| 项目 | 实际值 | 预期值 | 差异 |
|-----|--------|--------|------|
| Prompt 长度 | 215 tokens | ~200 tokens | ✅ 正常 |
| 生成时间 | ~4.8 秒 | ~3-5 秒 | ✅ 正常 |
| 输出长度 | 5 字符 | ~50-200 字符 | ❌ 异常 |
| 显存使用 | 未知 | ~2GB | ❓ 待确认 |

---

## 🎯 下一步行动

### 立即执行

1. **增加调试日志**: 打印完整模型输出
2. **测试模型直接调用**: 排除代码干扰
3. **检查显存使用**: 确认是否显存不足

### 短期优化

1. **调整生成参数**: 
   ```python
   generate_kwargs = {
       "max_new_tokens": 256,  # 减少到 256
       "do_sample": True,
       "temperature": 0.8,     # 提高温度
       "top_p": 0.95,          # 提高 top_p
       "repetition_penalty": 1.05  # 降低重复惩罚
   }
   ```

2. **优化 Prompt**: 添加明确的回复长度要求
   ```python
   system_prompt = "请用 50-100 字清晰、完整地回答用户问题。"
   ```

3. **模型替换**: 考虑使用更大的模型 (如 1.8B 或 7B)

---

## 📝 临时解决方案

如果确认是模型能力问题，可以:

1. **强制简短回复**:
   ```python
   response = self.l2_model.generate(prompt, max_tokens=128)
   ```

2. **添加回复模板**:
   ```python
   system_prompt += "\n请按照以下格式回复:\n1. 直接回答核心内容\n2. 保持 50-100 字\n3. 使用完整句子"
   ```

3. **后处理增强**: 如果回复过短，重新生成
   ```python
   if len(response) < 20:
       logger.warning("回复过短，尝试重新生成...")
       response = self.l2_model.generate(prompt, max_tokens=256)
   ```

---

**建议**: 先在终端测试模型直接调用，确认是否是模型本身的问题
