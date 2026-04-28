# -*- coding: utf-8 -*-
"""
直接测试 vLLM 工具调用功能
"""
from openai import OpenAI

# 创建 vLLM 客户端
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "openclaw_search",
            "description": "联网搜索工具。有两种使用方式：\n1. **搜索信息**：当用户询问事实、数据、新闻、产品信息时，使用 search 动作\n2. **读取网页**：当用户提供具体 URL 时，使用 fetch_webpage 动作直接读取",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "动作类型：'search'（搜索）或 'fetch_webpage'（读取网页）",
                        "enum": ["search", "fetch_webpage"]
                    },
                    "query": {
                        "type": "string",
                        "description": "搜索关键词（仅在 action='search' 时需要），例如'AI MAX395 规格参数'"
                    },
                    "count": {
                        "type": "integer",
                        "description": "搜索结果数量（仅在 action='search' 时需要），1-10 之间，默认 5 条",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10
                    },
                    "url": {
                        "type": "string",
                        "description": "网页 URL（仅在 action='fetch_webpage' 时需要），例如'https://example.com'"
                    }
                },
                "required": ["action"]
            }
        }
    }
]

# 测试消息
messages = [
    {"role": "system", "content": "你是祖龙 (ZULONG) 机器人助手。你可以使用 openclaw_search 工具搜索互联网或读取指定网页获取最新、最准确的信息。当用户提供具体 URL 时，直接读取该网页内容。"},
    {"role": "user", "content": "读取这个链接的信息：https://www.chiphell.com/thread-2761306-1-1.html"}
]

print("="*80)
print("测试 vLLM 工具调用功能")
print("="*80)
print("\n用户输入：读取这个链接的信息：https://www.chiphell.com/thread-2761306-1-1.html")
print("\n调用 vLLM API...")

try:
    response = client.chat.completions.create(
        model="/mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-0.8B-AWQ",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        stream=False
    )
    
    print("\n✅ vLLM 响应成功！")
    print(f"\n模型回复内容：{response.choices[0].message.content[:200] if response.choices[0].message.content else 'None'}...")
    print(f"\n工具调用：{response.choices[0].message.tool_calls}")
    
    if response.choices[0].message.tool_calls:
        print("\n✅ 检测到工具调用！")
        for tool_call in response.choices[0].message.tool_calls:
            print(f"\n  工具名称：{tool_call.function.name}")
            print(f"  工具参数：{tool_call.function.arguments}")
    else:
        print("\n⚠️ 没有检测到工具调用")
        
except Exception as e:
    print(f"\n❌ 调用失败：{e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
