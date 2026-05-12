# L2 模型输出优化方案

**分析时间**: 2026-03-25 23:25  
**问题**: Qwen3.5-0.8B 输出过短 (13-38 字符)  
**目标**: 提升到 50-150 字符  

---

## 🎯 优化策略

### 策略 1: System Prompt 强化 (立即执行)

**修改位置**: `inference_engine.py` 或 `vlm_agent.py`

**当前 Prompt**:
```python
system_prompt = "你是祖龙 (ZULONG) 机器人助手。"
```

**优化后**:
```python
system_prompt = """你是祖龙 (ZULONG) 机器人助手，一个友好、专业的 AI 伙伴。

【回复要求】
1. **完整性**: 每个回复必须 50-150 字，确保信息完整
2. **清晰性**: 使用完整句子，避免碎片化表达
3. **友好性**: 语气亲切自然，像朋友对话
4. **具体性**: 提供具体细节，不泛泛而谈

【示例】
❌ 错误： "您好！有什么可以帮您？" (太短)
✅ 正确： "您好！我是祖龙机器人助手。我注意到您正在测试我的对话功能，我会尽力为您提供帮助。无论是查询信息、执行任务还是简单聊天，我都可以协助您。请问现在有什么我可以为您做的吗？" (完整、友好、具体)

现在，请用心回答用户的每一个问题。"""
```

---

### 策略 2: 生成参数优化 (立即执行)

**修改位置**: `zulong/models/engine.py` 第 420 行

**当前参数**:
```python
generate_kwargs = {
    "max_new_tokens": max_tokens,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1
}
```

**优化后**:
```python
generate_kwargs = {
    "max_new_tokens": max(256, max_tokens),  # 至少 256 tokens
    "do_sample": True,
    "temperature": 0.8,  # 提高温度，增加多样性
    "top_p": 0.95,       # 提高 top_p
    "top_k": 50,         # 添加 top_k 采样
    "repetition_penalty": 1.05,  # 降低重复惩罚
    "min_new_tokens": 50,        # 最小生成 50 tokens
    "length_penalty": 1.1        # 长度奖励，鼓励生成长文本
}
```

---

### 策略 3: 后处理重生成 (短期优化)

**修改位置**: `inference_engine.py` 第 300 行后

**添加逻辑**:
```python
# 🔍 检查回复长度，过短则重新生成
if len(response) < 40:
    logger.warning(f"⚠️ 回复过短 ({len(response)} 字符)，尝试重新生成...")
    
    # 构建强化 Prompt
    retry_prompt = f"""用户的问题是：{user_input}

请提供一个详细、完整的回复，至少 80 字。要求：
1. 直接回答核心内容
2. 提供相关背景信息
3. 使用自然流畅的语言
4. 避免过于简短

请开始您的回复："""
    
    response = self.l2_model.generate(retry_prompt, max_tokens=256)
    
    # 再次提取</think>
    if "</think>" in response:
        parts = response.split("</think>")
        if len(parts) > 1:
            response = parts[-1].strip()
    
    logger.info(f"✅ 重新生成后长度：{len(response)} 字符")
```

---

### 策略 4: Few-Shot Examples (中期优化)

**修改位置**: `vlm_agent.py` 的 `_build_messages_with_history` 方法

**添加示例**:
```python
# 添加 few-shot 示例到 system prompt
few_shot_examples = [
    {
        "role": "user",
        "content": "你好"
    },
    {
        "role": "assistant", 
        "content": "您好！我是祖龙机器人助手。很高兴见到您！今天我能为您做些什么呢？无论是查询信息、执行任务、还是简单聊天，我都很乐意协助您。请随时告诉我您的需求，我会尽力为您提供最好的服务！"
    },
    {
        "role": "user",
        "content": "你在干什么"
    },
    {
        "role": "assistant",
        "content": "我正在这里等待您的指令呢！作为祖龙机器人系统，我随时准备协助您处理各种任务。我刚刚完成了系统自检，所有模块都运行正常。现在我的注意力完全集中在您身上，请问有什么我可以为您做的吗？无论是实际问题还是闲聊，我都很期待与您交流！"
    }
]

# 在 messages 中插入示例
messages = [system_message] + few_shot_examples + user_messages
```

---

## 📝 实施计划

### Phase 1: 立即执行 (今天)

- [x] 问题分析完成
- [ ] 修改 System Prompt
- [ ] 优化生成参数
- [ ] 添加重生成逻辑
- [ ] 测试验证

### Phase 2: 短期优化 (明天)

- [ ] 添加 Few-Shot Examples
- [ ] 测试不同参数组合
- [ ] 收集用户反馈

### Phase 3: 中期优化 (本周)

- [ ] 考虑模型替换 (1.8B 或 7B)
- [ ] 微调模型 (如有数据)
- [ ] 建立输出质量监控

---

## 🧪 测试验证

**测试脚本**: `scripts/test_model_direct.py`

**预期结果**:
- 平均回复长度：80-150 字符
- 回复质量：完整、自然、友好
- 生成时间：<10 秒 (GPU)

**验收标准**:
- 无回复 <40 字符
- 90% 回复 >60 字符
- 用户满意度 >80%

---

## ⚠️ 注意事项

1. **显存限制**: 生成长文本会增加显存占用，注意监控
2. **生成时间**: 长文本需要更长时间，用户可能觉得慢
3. **过度啰嗦**: 避免为了长度而凑字数，保持信息密度

---

**建议**: 先实施 Phase 1 的三个优化，然后重新测试
