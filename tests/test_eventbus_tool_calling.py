# -*- coding: utf-8 -*-
"""
测试完整的工具调用流程（通过 EventBus）
"""
import asyncio
import websockets
import json

async def test_tool_calling():
    """通过 EventBus 发送测试消息"""
    
    # 连接到 EventBus
    async with websockets.connect("ws://localhost:5555/eventbus") as websocket:
        print("="*80)
        print("通过 EventBus 测试工具调用")
        print("="*80)
        
        # 订阅 L2_OUTPUT 事件
        subscribe_msg = {
            "type": "SUBSCRIBE",
            "event_types": ["L2_OUTPUT", "SYSTEM_STATUS"]
        }
        await websocket.send(json.dumps(subscribe_msg))
        print(f"\n✅ 已订阅 L2_OUTPUT 事件")
        
        # 发送测试消息
        test_msg = {
            "type": "PUBLISH",
            "event": {
                "type": "USER_TEXT",
                "source": "test_script",
                "priority": 1,  # 🔥 添加 priority 字段
                "payload": {
                    "text": "读取这个链接的信息：https://www.chiphell.com/thread-2761306-1-1.html",
                    "confidence": 1.0
                }
            }
        }
        
        print(f"\n📤 发送测试消息：读取这个链接的信息：https://www.chiphell.com/thread-2761306-1-1.html")
        await websocket.send(json.dumps(test_msg))
        
        # 监听响应
        print("\n⏳ 等待系统响应...")
        try:
            async with asyncio.timeout(30):  # 30 秒超时
                while True:
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    if data.get("type") == "SUBSCRIBE":
                        event = data.get("event", {})
                        event_type = event.get("type")
                        
                        if event_type == "L2_OUTPUT":
                            payload = event.get("payload", {})
                            text = payload.get("text", "")
                            
                            print(f"\n✅ 收到 L2 输出：")
                            print(f"   {text[:200]}...")
                            
                            # 检查是否有工具调用
                            if "工具" in text or "搜索" in text or "fetch_webpage" in text:
                                print("\n✅ 检测到工具调用相关关键词！")
                            elif "无法" in text and "读取" in text:
                                print("\n⚠️ 系统表示无法读取网页（工具调用可能失败）")
                            else:
                                print("\nℹ️ 未检测到明显的工具调用")
                            
                            break
                            
        except asyncio.TimeoutError:
            print("\n❌ 等待响应超时")
        
        print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(test_tool_calling())
