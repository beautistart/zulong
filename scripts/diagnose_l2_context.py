"""
L2 上下文注入诊断工具

用于查看 L2 推理引擎在构建 messages 时注入的所有上下文信息
"""

import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from zulong.l2.inference_engine import InferenceEngine

async def diagnose_context_injection():
    """诊断上下文注入情况"""
    
    print("="*80)
    print("🔍 L2 上下文注入诊断工具")
    print("="*80)
    
    # 创建推理引擎实例
    engine = InferenceEngine()
    
    # 模拟用户输入
    test_input = "你好，今天天气怎么样？"
    
    print(f"\n📝 测试输入：{test_input}")
    print("-"*80)
    
    # 构建 messages（不实际调用模型）
    messages = await engine._build_messages_with_history_async(
        user_input=test_input,
        rag_context=None,
        visual_context=None,
        search_context=None
    )
    
    # 分析消息结构
    print(f"\n📊 消息结构分析：")
    print(f"   - 总消息数：{len(messages)}")
    print(f"   - System 消息：1 条")
    
    # 统计各类消息
    user_msgs = [m for m in messages if m['role'] == 'user']
    assistant_msgs = [m for m in messages if m['role'] == 'assistant']
    
    print(f"   - User 消息：{len(user_msgs)} 条")
    print(f"   - Assistant 消息：{len(assistant_msgs)} 条")
    
    # 分析 System Prompt
    system_prompt = messages[0]['content']
    print(f"\n📄 System Prompt 分析：")
    print(f"   - 总长度：{len(system_prompt)} 字符")
    print(f"   - 预估 Token 数：{int(len(system_prompt) * 1.5)}")
    
    # 分解 System Prompt 各部分
    print(f"\n🔍 System Prompt 组成：")
    
    sections = {
        "角色定义": "你是祖龙",
        "时间信息": "当前时间",
        "重要规则": "【重要规则】",
        "工具描述": "🔧",
        "对话原则": "【对话原则】",
        "参考知识": "【参考知识】",
        "相关记忆": "【相关记忆】",
        "历史对话": "【历史对话】",
    }
    
    for name, marker in sections.items():
        if marker in system_prompt:
            # 找到该部分的起始位置
            start = system_prompt.find(marker)
            # 找到下一部分的起始位置（如果有）
            end = len(system_prompt)
            for other_marker in sections.values():
                other_pos = system_prompt.find(other_marker, start + 1)
                if other_pos > start and other_pos < end:
                    end = other_pos
            
            section_text = system_prompt[start:end]
            print(f"\n   [{name}]")
            print(f"   长度：{len(section_text)} 字符，约 {int(len(section_text) * 1.5)} tokens")
            print(f"   内容预览：{section_text[:200]}...")
    
    # 分析工作记忆
    print(f"\n📖 工作记忆分析：")
    if len(messages) > 1:
        for i, msg in enumerate(messages[1:-1], 1):  # 排除 system 和最后一个 user 输入
            print(f"   {i}. {msg['role']}: {len(msg['content'])} 字符")
            if msg['role'] == 'user':
                print(f"      内容：{msg['content'][:100]}...")
            else:
                print(f"      内容：{msg['content'][:100]}...")
    
    # 估算总 token 数
    total_chars = sum(len(m['content']) for m in messages)
    total_tokens_est = int(total_chars * 1.5)
    
    print(f"\n📊 总体评估：")
    print(f"   - 总字符数：{total_chars}")
    print(f"   - 预估 Token 数：{total_tokens_est}")
    print(f"   - Qwen3.5-2B 最佳窗口：1024-2048 tokens")
    
    if total_tokens_est > 2048:
        print(f"   ⚠️ 警告：超出最佳窗口 {total_tokens_est - 2048} tokens")
        print(f"   💡 建议：减少记忆检索数量或截断长文本")
    elif total_tokens_est > 1500:
        print(f"   ⚠️ 注意：接近最佳窗口上限")
    else:
        print(f"   ✅ 良好：在最佳窗口范围内")
    
    print("\n" + "="*80)
    print("诊断完成！")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(diagnose_context_injection())
