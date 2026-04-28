# File: zulong/utils/text_cleaner.py
"""
文本清洗工具 - 用于 TTS 输入预处理

功能:
1. 移除 <think>...</think> 思维链标签
2. 移除 Markdown 代码块 (```json ... ```)
3. 解析 JSON 并提取核心字段
4. 移除所有 Emoji 表情
5. 移除 Markdown 格式符号
6. 修复多余的空白

TSD v1.7 对应:
- 4.2 L1-B 调度与意图守门层：TTS 内容清洗
"""

import re
import json


def clean_text_for_tts(text: str) -> str:
    """
    强力清洗 L2 输出以适合 TTS 播放
    
    清洗步骤:
    1. 移除 <think>...</think> 思维链
    2. 移除 Markdown 代码块并尝试解析 JSON
    3. 解析纯 JSON 对象并提取核心字段
    4. 移除 Emoji
    5. 移除 Markdown 格式
    6. 清理多余空白
    7. 🔥 检测并拦截"复述指令"异常输出
    
    Args:
        text: 原始文本 (可能包含思维链、JSON、Emoji、Markdown 等)
    
    Returns:
        str: 清洗后的纯净自然语言文本
    """
    if not text:
        return ""
    
    text = text.strip()
    
    # ========== 步骤 0: 检测"复述指令"异常模式 (新增) ==========
    # 检测是否包含重复的规则复述 (上下文污染)
    anomaly_patterns = [
        (r"必须包含.*关键要素", "规则复述"),
        (r"约束条件.*", "规则复述"),
        (r"版本一：.*版本二：", "多版本输出"),
        (r"^(\d+\..*?\n){3,}", "连续数字列表"),
        (r"5\..*必须.*6\..*必须", "复述约束条件"),
        (r"^\d+\..*必须", "数字列表 + 必须"),  # 新增：检测以"数字 + 必须"开头的句子
    ]
    
    for pattern, anomaly_type in anomaly_patterns:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            # 检测到异常，记录日志并尝试修复
            print(f"[TextCleaner] Warning: Detected model anomaly output ({anomaly_type}), intercepted!")
            
            # 策略 A: 尝试提取正常内容 (找第一个句号后的正常句子)
            # 先移除异常模式本身
            cleaned = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
            
            # 查找正常句子 (包含人称代词或动词)
            normal_patterns = [r'[你我他].*在.*', r'我看到.*', r'你正在.*', r'根据.*']
            for np in normal_patterns:
                match = re.search(np, cleaned)
                if match:
                    # 提取匹配的句子
                    start = match.start()
                    end = cleaned.find('。', start)
                    if end != -1:
                        result = cleaned[start:end+1].strip()
                        if len(result) > 10:
                            print(f"[TextCleaner] Extracted normal content: {result[:50]}...")
                            return result
            
            # 策略 B: 没有正常句子，返回兜底回复
            return "我刚才有点走神，请再说一遍您的问题好吗？"
    
    # ========== 步骤 1: 移除思维链 <think>...</think> ==========
    think_pattern = r'<think>.*?</think>'
    text = re.sub(think_pattern, '', text, flags=re.DOTALL | re.IGNORECASE).strip()
    
    # ========== 步骤 2: 处理 Markdown 代码块 ==========
    markdown_pattern = r'```(?:json|python|text)?\s*(.*?)```'
    code_blocks = re.findall(markdown_pattern, text, flags=re.DOTALL)
    
    if code_blocks:
        # 有代码块，优先处理第一个
        potential_json = code_blocks[0].strip()
        try:
            data = json.loads(potential_json)
            if isinstance(data, dict):
                # 提取核心字段
                for key in ['answer', 'response', 'text', 'content', 'reply', 'message']:
                    if key in data:
                        return str(data[key])
                # 如果没有标准字段，返回序列化字符串
                return potential_json
        except json.JSONDecodeError:
            # 解析失败，使用代码块内的纯文本
            text = potential_json
    else:
        # 没有代码块，清理剩余的 markdown 符号
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # ========== 步骤 3: 处理纯 JSON 对象 ==========
    if text.startswith('{') and text.endswith('}'):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                for key in ['answer', 'response', 'text', 'content', 'reply', 'message']:
                    if key in data:
                        return str(data[key])
                # 返回所有值的拼接
                return " ".join(str(v) for v in data.values())
        except json.JSONDecodeError:
            pass
    
    # ========== 步骤 4: 移除 Emoji ==========
    text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)  # Emoticons
    text = re.sub(r'[\U0001F300-\U0001F5FF]', '', text)  # Symbols & Pictographs
    text = re.sub(r'[\U0001F680-\U0001F6FF]', '', text)  # Transport & Map
    text = re.sub(r'[\U0001F1E0-\U0001F1FF]', '', text)  # Flags
    
    # ========== 步骤 5: 移除 Markdown 格式 ==========
    # 移除粗体标记
    text = re.sub(r'\*\*', '', text)
    # 移除斜体标记
    text = re.sub(r'(?<!\*)\*(?!\*)', '', text)
    # 移除标题标记
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # 移除列表标记
    text = re.sub(r'^[-•]\s+', '', text, flags=re.MULTILINE)
    # 移除数字列表
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    # 移除链接
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    
    # ========== 步骤 6: 清理特殊符号和多余空白 ==========
    text = re.sub(r'[$€£¥→←%©®™]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # 去除首尾标点残留
    text = text.strip('[]{}"\'')
    
    return text


# 测试
if __name__ == "__main__":
    test_cases = [
        # 测试 1: 思维链清洗
        """<think> 用户问的是天气，我需要查询一下...</think> 今天天气晴朗，温度适宜。""",
        
        # 测试 2: JSON 格式清洗
        """{
            "answer": "你好，我是祖龙机器人",
            "confidence": 0.95
        }""",
        
        # 测试 3: Markdown 代码块 JSON
        """```json
        {
            "response": "这是一个测试回复"
        }
        ```""",
        
        # 测试 4: Emoji 清洗
        "今天是 **2026 年 3 月 25 日**。正值下午 16:56！愿一切顺利 🌸💫",
        
        # 测试 5: 混合测试
        """<think> 分析用户问题...</think>
        ```json
        {
            "answer": "你手里拿着一个红色的苹果 🍎"
        }
        ```""",
        
        # 测试 6: 正常文本
        "正常文本：你好，世界！Hello, World! 123",
        
        # 测试 7: 链接和格式
        "## 欢迎使用\n- 功能 1\n- 功能 2\n**重要** 提示 [点击这里](http://test.com)",
    ]
    
    print("=" * 60)
    print("文本清洗工具测试 (增强版)")
    print("=" * 60)
    
    for i, original in enumerate(test_cases, 1):
        cleaned = clean_text_for_tts(original)
        print(f"\n测试 {i}:")
        print(f"  原始：{original[:100]}{'...' if len(original) > 100 else ''}")
        print(f"  清洗：{cleaned}")
        print(f"  长度：{len(original)} → {len(cleaned)}")
