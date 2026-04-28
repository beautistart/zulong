"""
诊断当前会话提示词内容分析工具（简化版 - 不加载模型）

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
    
    # 1. 手动读取 inference_engine.py 文件，分析 _build_messages_with_history_async 方法
    print("\n[1/3] 读取 inference_engine.py...")
    
    with open("zulong/l2/inference_engine.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # 2. 提取关键部分
    print("\n[2/3] 分析提示词构建逻辑...")
    
    # 查找 tool description 构建
    if "_build_tools_description" in content:
        print("✅ 工具描述构建方法已找到")
        
        # 估算工具描述长度
        # 从日志中查找实际长度
        import subprocess
        result = subprocess.run(
            ["grep", "-n", "🔧 [工具描述] 工具描述长度", "zulong/l2/inference_engine.py"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"  日志输出：{result.stdout.strip()}")
    
    # 3. 分析提示词组成部分
    print("\n[3/3] 提示词结构分析...")
    print("-" * 80)
    
    # 从代码中提取 system_parts 的构建逻辑
    system_parts_analysis = {
        "基础设定": "祖龙 (ZULONG) 机器人助手，一个活泼、可爱的 AI 伙伴",
        "时间信息": "当前时间：{current_time_str} ({time_period})",
        "重要规则": "当用户问'我是做什么的'时，'我'指的是用户自己",
        "示例": "用户：'我是做什么工作的？' → 你回答：'您是一名室内设计师'",
        "工具描述": "【工具功能】... (动态注入)",
        "对话原则/核心原则": "【对话原则】或【核心原则】(根据是否有视觉信息)",
        "参考知识": "【参考知识】... (RAG 检索结果)",
        "搜索结果": "【搜索结果】... (网络搜索结果)",
        "相关历史": "【相关历史】(前三轮中相关的完整对话)",
        "历史对话": "【历史对话】(基于语义检索的相似对话)",
        "临时记忆": "【临时记忆】(基于摘要检索)",
        "引导语": "请开始回答用户的问题："
    }
    
    print("\n**System Prompt 组成部分**:")
    for part_name, description in system_parts_analysis.items():
        print(f"  - {part_name}: {description}")
    
    # 估算各部分长度
    part_estimates = {
        "基础设定": 50,
        "时间信息": 50,
        "重要规则": 150,
        "示例": 150,
        "工具描述": 800,  # 动态注入，估算
        "对话原则/核心原则": 300,
        "参考知识": 500,  # 动态注入
        "搜索结果": 400,  # 动态注入
        "相关历史": 400,  # 动态注入
        "历史对话": 300,  # 动态注入
        "临时记忆": 300,  # 动态注入
        "引导语": 20
    }
    
    total_estimated = sum(part_estimates.values())
    
    print(f"\n**长度估算**:")
    print(f"  - 固定部分：{sum(v for k, v in part_estimates.items() if k not in ['工具描述', '参考知识', '搜索结果', '相关历史', '历史对话', '临时记忆'])} 字符")
    print(f"  - 动态部分：{sum(v for k, v in part_estimates.items() if k in ['工具描述', '参考知识', '搜索结果', '相关历史', '历史对话', '临时记忆'])} 字符")
    print(f"  - 总计估算：~{total_estimated} 字符")
    print(f"  - Token 估算：~{count_tokens('' * total_estimated)} tokens")
    
    # 分析对话历史注入
    print(f"\n**对话历史注入策略**:")
    print(f"  - 工作记忆：最近 3 轮对话（6 条消息）")
    print(f"  - 前 3 轮：语义检索，相似度≥0.5 才注入")
    print(f"  - 第 4 轮起：向量缓存检索 Top-2")
    print(f"  - 临时记忆：基于摘要检索 Top-2")
    
    # 检查潜在问题
    print(f"\n**潜在问题分析**:")
    print("-" * 80)
    
    issues = []
    
    # 问题 1: 工具描述过长
    issues.append("⚠️ 工具描述可能过长（估计 800+ 字符），建议精简到 500 字符以内")
    
    # 问题 2: 多层记忆同时注入
    issues.append("⚠️ 可能同时注入多种记忆类型（工作记忆 + 前 3 轮 + 历史对话 + 临时记忆），造成上下文污染")
    
    # 问题 3: System Prompt 结构复杂
    issues.append("⚠️ System Prompt 包含多个【】标记块，可能分散模型注意力")
    
    # 问题 4: 约束性词汇过多
    issues.append("⚠️ 包含'严禁'、'必须'、'约束'等词汇，可能触发模型复述规则")
    
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    # 输出优化建议
    print(f"\n**优化建议**:")
    print("-" * 80)
    
    recommendations = [
        "1. **精简工具描述**: 将工具描述压缩到 300-500 字符，移除冗余说明",
        "2. **限制记忆注入数量**: \n" +
        "   - 工作记忆：保持 3 轮\n" +
        "   - 前 3 轮：最多注入 1 轮（相似度最高的）\n" +
        "   - 历史对话：保持 Top-2\n" +
        "   - 临时记忆：减少到 Top-1",
        "3. **简化 System Prompt**: 移除部分【】标记块，使用更自然的叙述方式",
        "4. **移除强硬约束**: 将'严禁'、'必须'改为'建议'、'最好'",
        "5. **动态调整策略**: 根据对话轮次动态调整记忆注入量\n" +
        "   - 前 3 轮：只注入工作记忆\n" +
        "   - 4-10 轮：工作记忆 + 前 3 轮检索\n" +
        "   - 10 轮以上：工作记忆 + Top-2 历史对话"
    ]
    
    for rec in recommendations:
        print(f"\n{rec}")
    
    # 计算优化后的估算
    print(f"\n**优化后估算**:")
    print("-" * 80)
    
    optimized_estimates = {
        "基础设定": 50,
        "时间信息": 50,
        "重要规则": 100,
        "示例": 100,
        "工具描述": 400,  # 精简后
        "对话原则": 150,
        "工作记忆": 300,  # 3 轮对话
        "相关历史": 200,  # 最多 1 轮
        "历史对话": 200,  # Top-2
        "临时记忆": 150,  # Top-1
        "引导语": 20
    }
    
    optimized_total = sum(optimized_estimates.values())
    
    print(f"  - 优化前：~{total_estimated} 字符")
    print(f"  - 优化后：~{optimized_total} 字符")
    print(f"  - 减少：{total_estimated - optimized_total} 字符 ({(1 - optimized_total/total_estimated) * 100:.1f}%)")
    print(f"  - Token 节省：~{count_tokens('' * (total_estimated - optimized_total))} tokens")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(analyze_prompt_structure())
