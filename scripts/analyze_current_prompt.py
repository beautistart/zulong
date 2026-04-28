"""
诊断当前会话提示词内容分析工具

检查：
1. 实际注入到 L2 的提示词内容
2. 各部分 token 数量统计
3. 是否超出模型上下文窗口
4. 提示词结构是否合理
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from zulong.l2.inference_engine import InferenceEngine
from zulong.memory.short_term_memory import ShortTermMemory

def count_tokens(text: str) -> int:
    """估算 token 数量（中文字符≈1.5 tokens，英文字符≈0.25 tokens）"""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    return int(chinese_chars * 1.5 + english_chars * 0.25 + len(text) * 0.1)


async def analyze_prompt_structure():
    """分析提示词结构"""
    print("=" * 80)
    print("🔍 当前会话提示词分析")
    print("=" * 80)
    
    # 1. 初始化组件
    print("\n[1/4] 初始化推理引擎...")
    engine = InferenceEngine()
    
    print("[2/4] 初始化短期记忆...")
    stm = ShortTermMemory()
    
    # 2. 获取当前会话历史
    print("\n[3/4] 读取当前会话状态...")
    print(f"  - 对话历史长度：{len(engine.conversation_history)} 条消息")
    print(f"  - 对话轮数：{len(engine.conversation_history) // 2} 轮")
    
    # 3. 模拟构建提示词
    print("\n[4/4] 构建提示词并分析...")
    print("-" * 80)
    
    test_input = "测试当前提示词结构"
    
    messages = await engine._build_messages_with_history_async(
        user_input=test_input,
        rag_context=None,
        visual_context=None,
        search_context=None
    )
    
    # 4. 分析各部分
    system_prompt = messages[0]['content']
    
    print("\n📊 **提示词结构分析**")
    print("-" * 80)
    
    # 分解 system prompt
    parts = {
        "基础设定": "祖龙机器人助手",
        "时间信息": "当前时间",
        "重要规则": "【重要规则】",
        "工具描述": "【工具功能】",
        "对话原则": "【对话原则】或【核心原则】",
        "参考知识": "【参考知识】",
        "搜索上下文": "【搜索结果】",
        "相关历史": "【相关历史】",
        "历史对话": "【历史对话】",
        "临时记忆": "【临时记忆】",
        "引导语": "请开始回答"
    }
    
    part_lengths = {}
    for part_name, keyword in parts.items():
        if keyword in system_prompt:
            # 估算该部分的长度
            start_idx = system_prompt.find(keyword)
            if start_idx != -1:
                # 找到下一个部分的关键字
                next_keywords = [system_prompt.find(k) for k in parts.values() if k != keyword and system_prompt.find(k) > start_idx]
                next_keywords = [k for k in next_keywords if k > 0]
                end_idx = min(next_keywords) if next_keywords else len(system_prompt)
                
                part_text = system_prompt[start_idx:end_idx]
                part_lengths[part_name] = len(part_text)
    
    # 5. 输出统计
    print(f"\n**System Prompt 总长度**: {len(system_prompt)} 字符")
    print(f"**System Prompt Token 估算**: ~{count_tokens(system_prompt)} tokens")
    
    print(f"\n**各部分长度分布**:")
    for part_name, length in sorted(part_lengths.items(), key=lambda x: x[1], reverse=True):
        percentage = (length / len(system_prompt)) * 100
        print(f"  - {part_name}: {length} 字符 ({percentage:.1f}%)")
    
    # 6. 消息历史统计
    user_messages = [m for m in messages if m['role'] == 'user']
    assistant_messages = [m for m in messages if m['role'] == 'assistant']
    
    total_user_chars = sum(len(m['content']) for m in user_messages)
    total_assistant_chars = sum(len(m['content']) for m in assistant_messages)
    
    print(f"\n**对话历史统计**:")
    print(f"  - 用户消息：{len(user_messages)} 条，{total_user_chars} 字符")
    print(f"  - AI 消息：{len(assistant_messages)} 条，{total_assistant_chars} 字符")
    print(f"  - 历史 Token 估算：~{count_tokens(system_prompt + ''.join(m['content'] for m in messages[1:]))} tokens")
    
    # 7. 检查是否超出上下文窗口
    qwen3_5_0_8b_context = 4096  # Qwen3.5-0.8B 的上下文窗口
    estimated_total = count_tokens(system_prompt + ''.join(m['content'] for m in messages[1:]))
    
    print(f"\n**上下文窗口检查**:")
    print(f"  - 模型上下文窗口：{qwen3_5_0_8b_context} tokens")
    print(f"  - 估算使用量：{estimated_total} tokens")
    print(f"  - 使用率：{(estimated_total / qwen3_5_0_8b_context) * 100:.1f}%")
    
    if estimated_total > qwen3_5_0_8b_context * 0.9:
        print(f"\n⚠️ **警告**: 提示词已接近上下文窗口限制！")
        print(f"   建议：减少记忆注入数量或压缩历史对话")
    
    # 8. 输出实际提示词内容（前 2000 字符）
    print(f"\n**System Prompt 内容预览** (前 2000 字符):")
    print("-" * 80)
    print(system_prompt[:2000])
    if len(system_prompt) > 2000:
        print(f"\n... (还有 {len(system_prompt) - 2000} 字符)")
    
    # 9. 检查记忆注入情况
    print(f"\n**记忆注入检查**:")
    
    if hasattr(engine, 'short_term_memory') and engine.short_term_memory is not None:
        # 检查向量缓存
        if hasattr(engine.short_term_memory, 'vector_cache'):
            cache_size = len(engine.short_term_memory.vector_cache.cache)
            print(f"  - 向量缓存大小：{cache_size} 条")
        
        # 检查临时记忆
        if hasattr(engine, 'episodic_memory'):
            episode_count = len(engine.episodic_memory.episode_cache)
            print(f"  - 临时记忆缓存：{episode_count} 条")
    
    # 10. 问题诊断
    print(f"\n**问题诊断**:")
    print("-" * 80)
    
    issues = []
    
    # 检查 1: System Prompt 是否过长
    if len(system_prompt) > 2000:
        issues.append("❌ System Prompt 过长（>{2000} 字符），可能导致模型注意力分散")
    else:
        issues.append("✅ System Prompt 长度合理")
    
    # 检查 2: 工具描述是否过长
    if hasattr(engine, '_available_tools_description'):
        tool_desc_len = len(engine._available_tools_description)
        if tool_desc_len > 1000:
            issues.append(f"❌ 工具描述过长（{tool_desc_len} 字符），建议精简到 500 字符以内")
        else:
            issues.append(f"✅ 工具描述长度合理（{tool_desc_len} 字符）")
    
    # 检查 3: 记忆注入是否过多
    if "【相关历史】" in system_prompt and "【历史对话】" in system_prompt:
        issues.append("⚠️ 同时注入了多种记忆类型，可能造成上下文污染")
    else:
        issues.append("✅ 记忆注入策略合理")
    
    # 检查 4: 是否有多余的规则复述
    if "必须" in system_prompt and "严禁" in system_prompt and "约束" in system_prompt:
        issues.append("⚠️ 包含大量约束性词汇，可能触发模型复述规则")
    else:
        issues.append("✅ 约束性词汇使用适度")
    
    for issue in issues:
        print(f"  {issue}")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(analyze_prompt_structure())
