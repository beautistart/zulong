#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
祖龙系统真实环境交互式测试
- 记忆模块测试（短期、长期、情景、经验）
- RAG模块测试（知识检索、向量相似度）
- 日常闲聊测试
- 复杂任务测试
- 所有测试都在真实系统中进行，可在浏览器中观察
"""

import asyncio
import websockets
import json
import time
from datetime import datetime


class InteractiveZulongTester:
    """交互式祖龙系统测试器"""
    
    def __init__(self, ws_uri="ws://localhost:5555"):
        self.ws_uri = ws_uri
        self.test_log = []
    
    async def send_and_observe(self, message: str, wait_time: float = 5.0, test_name: str = ""):
        """
        发送消息并观察系统响应
        
        Args:
            message: 要发送的消息
            wait_time: 等待响应的时间（秒）
            test_name: 测试名称（用于日志）
        """
        print(f"\n{'='*80}")
        if test_name:
            print(f"📝 [{test_name}]")
        print(f"{'='*80}")
        print(f"\n💬 发送消息: {message}")
        print(f"⏳ 等待 {wait_time} 秒，请在浏览器中观察系统响应...")
        
        try:
            async with websockets.connect(self.ws_uri) as websocket:
                # 发送USER_TEXT事件
                event = {
                    "type": "USER_TEXT",
                    "source": "interactive_test",
                    "payload": {
                        "text": message,
                        "confidence": 1.0
                    },
                    "priority": 5
                }
                
                await websocket.send(json.dumps(event))
                print(f"✅ 消息已发送")
                
                # 等待并接收响应
                print(f"\n👀 正在监听系统响应...")
                response_count = 0
                
                try:
                    while response_count < 3:  # 最多接收3个响应
                        response = await asyncio.wait_for(websocket.recv(), timeout=wait_time)
                        data = json.loads(response)
                        
                        response_count += 1
                        print(f"\n📨 响应 #{response_count}:")
                        print(f"   类型: {data.get('type', 'unknown')}")
                        
                        # 显示关键信息
                        if 'payload' in data:
                            payload = data['payload']
                            if 'text' in payload:
                                text = payload['text']
                                print(f"   内容: {text[:200]}{'...' if len(text) > 200 else ''}")
                            if 'status' in payload:
                                print(f"   状态: {payload['status']}")
                        
                        # 记录日志
                        self.test_log.append({
                            "timestamp": datetime.now().isoformat(),
                            "test_name": test_name,
                            "message": message,
                            "response_type": data.get('type'),
                            "response": data
                        })
                        
                        if response_count >= 3:
                            break
                            
                except asyncio.TimeoutError:
                    print(f"\n⏰ 响应监听超时（{wait_time}秒）")
                    print(f"💡 提示：请查看浏览器中的对话记录")
                
                print(f"\n✅ 本次交互完成，共收到 {response_count} 个响应")
                print(f"💡 请在浏览器 http://localhost:8080 查看完整对话过程")
                
        except Exception as e:
            print(f"\n❌ 发送消息失败: {e}")
            import traceback
            traceback.print_exc()
    
    async def test_short_term_memory(self):
        """测试1: 短期记忆 - 多轮对话上下文"""
        print("\n\n")
        print("╔" + "="*78 + "╗")
        print("║" + " "*20 + "测试模块1: 短期记忆" + " "*35 + "║")
        print("╚" + "="*78 + "╝")
        
        print("\n📋 测试目标: 验证系统能否在多轮对话中保持上下文信息")
        print("💡 观察要点: 系统是否能记住前面提到的信息\n")
        
        await self.send_and_observe(
            "你好！我叫张三，今年28岁，是一名软件工程师，主要做Python开发。",
            wait_time=5.0,
            test_name="短期记忆-第1轮"
        )
        
        await asyncio.sleep(2)
        
        await self.send_and_observe(
            "你还记得我叫什么名字吗？我做什么工作的？",
            wait_time=8.0,
            test_name="短期记忆-第2轮"
        )
        
        await asyncio.sleep(2)
        
        await self.send_and_observe(
            "我今年多大了？我主要用什么编程语言？",
            wait_time=8.0,
            test_name="短期记忆-第3轮"
        )
    
    async def test_long_term_memory(self):
        """测试2: 长期记忆 - 持久化存储和检索"""
        print("\n\n")
        print("╔" + "="*78 + "╗")
        print("║" + " "*20 + "测试模块2: 长期记忆" + " "*35 + "║")
        print("╚" + "="*78 + "╝")
        
        print("\n📋 测试目标: 验证系统能否持久化存储重要信息并在后续检索")
        print("💡 观察要点: 系统是否能记住重要的个人信息\n")
        
        await self.send_and_observe(
            "请记住一个重要的信息：我对花生过敏，不能吃任何含花生的食物。这个很重要！",
            wait_time=6.0,
            test_name="长期记忆-存储"
        )
        
        await asyncio.sleep(3)
        
        await self.send_and_observe(
            "我有什么食物过敏吗？你能告诉我吗？",
            wait_time=8.0,
            test_name="长期记忆-检索"
        )
    
    async def test_rag_knowledge_retrieval(self):
        """测试3: RAG知识检索 - 向量相似度"""
        print("\n\n")
        print("╔" + "="*78 + "╗")
        print("║" + " "*20 + "测试模块3: RAG知识检索" + " "*33 + "║")
        print("╚" + "="*78 + "╝")
        
        print("\n📋 测试目标: 验证RAG系统的知识检索能力")
        print("💡 观察要点: 系统能否通过向量检索找到相关知识\n")
        
        await self.send_and_observe(
            "我最近在学习机器学习，已经了解了线性回归、逻辑回归和决策树算法。",
            wait_time=6.0,
            test_name="RAG-知识存储"
        )
        
        await asyncio.sleep(3)
        
        await self.send_and_observe(
            "我之前学过哪些机器学习算法？",
            wait_time=8.0,
            test_name="RAG-精确检索"
        )
        
        await asyncio.sleep(3)
        
        await self.send_and_observe(
            "我对AI领域有哪些了解？",
            wait_time=8.0,
            test_name="RAG-语义检索"
        )
    
    async def test_episodic_memory(self):
        """测试4: 情景记忆 - 多轮关联和复盘"""
        print("\n\n")
        print("╔" + "="*78 + "╗")
        print("║" + " "*20 + "测试模块4: 情景记忆" + " "*35 + "║")
        print("╚" + "="*78 + "╝")
        
        print("\n📋 测试目标: 验证系统能否关联多个相关记忆并生成总结")
        print("💡 观察要点: 系统是否能综合多个信息进行回答\n")
        
        await self.send_and_observe(
            "我住在北京市海淀区。",
            wait_time=5.0,
            test_name="情景记忆-信息1"
        )
        
        await asyncio.sleep(2)
        
        await self.send_and_observe(
            "我在中关村上班，每天坐地铁4号线。",
            wait_time=5.0,
            test_name="情景记忆-信息2"
        )
        
        await asyncio.sleep(2)
        
        await self.send_and_observe(
            "周末我喜欢去颐和园散步。",
            wait_time=5.0,
            test_name="情景记忆-信息3"
        )
        
        await asyncio.sleep(3)
        
        await self.send_and_observe(
            "你能总结一下我的地理位置和生活习惯吗？",
            wait_time=10.0,
            test_name="情景记忆-综合总结"
        )
    
    async def test_experience_generation(self):
        """测试5: 经验生成 - L2 BACKUP复用L2 CORE"""
        print("\n\n")
        print("╔" + "="*78 + "╗")
        print("║" + " "*20 + "测试模块5: 经验生成" + " "*35 + "║")
        print("╚" + "="*78 + "╝")
        
        print("\n📋 测试目标: 验证经验提取和生成（L2 BACKUP复用L2 CORE）")
        print("💡 观察要点: 系统能否从对话中提取经验并给出建议\n")
        
        await self.send_and_observe(
            "我昨天写代码时遇到了一个bug，花了3个小时才发现是缩进错误。后来我学会了使用linter工具来检查代码。",
            wait_time=6.0,
            test_name="经验生成-场景描述"
        )
        
        await asyncio.sleep(3)
        
        await self.send_and_observe(
            "如果我要避免这种代码错误，你有什么建议吗？",
            wait_time=10.0,
            test_name="经验生成-经验应用"
        )
    
    async def test_casual_chat(self):
        """测试6: 日常闲聊 - 自然对话"""
        print("\n\n")
        print("╔" + "="*78 + "╗")
        print("║" + " "*20 + "测试模块6: 日常闲聊" + " "*35 + "║")
        print("╚" + "="*78 + "╝")
        
        print("\n📋 测试目标: 测试日常对话的自然流畅度")
        print("💡 观察要点: 对话是否自然，回复是否合理\n")
        
        await self.send_and_observe(
            "今天天气怎么样？适合出去走走吗？",
            wait_time=8.0,
            test_name="日常闲聊-天气"
        )
        
        await asyncio.sleep(2)
        
        await self.send_and_observe(
            "你有什么推荐的放松方式吗？",
            wait_time=8.0,
            test_name="日常闲聊-建议"
        )
    
    async def test_complex_task(self):
        """测试7: 复杂任务 - 多步骤任务执行"""
        print("\n\n")
        print("╔" + "="*78 + "╗")
        print("║" + " "*20 + "测试模块7: 复杂任务" + " "*35 + "║")
        print("╚" + "="*78 + "╝")
        
        print("\n📋 测试目标: 测试系统处理复杂多步骤任务的能力")
        print("💡 观察要点: 系统能否分解任务并逐步执行\n")
        
        await self.send_and_observe(
            "请帮我搜索一下Python 3.12的新特性，然后总结一下最重要的3个改进。",
            wait_time=15.0,
            test_name="复杂任务-搜索总结"
        )
        
        await asyncio.sleep(3)
        
        await self.send_and_observe(
            "你能把这些新特性和我之前说的机器学习知识联系起来吗？",
            wait_time=10.0,
            test_name="复杂任务-知识关联"
        )
    
    async def test_review_quick_flow(self):
        """测试8: 快速复盘完整流程"""
        print("\n\n")
        print("╔" + "="*78 + "╗")
        print("║" + " "*18 + "测试模块8: 快速复盘流程" + " "*33 + "║")
        print("╚" + "="*78 + "╝")
        
        print("\n📋 测试目标: 验证复盘机制完整流程（启动→对话→结束→经验提取→确认）")
        print("💡 观察要点: 经验是否成功生成并保存\n")
        
        # 1. 先进行几轮正常对话（积累上下文）
        await self.send_and_observe(
            "我最近在学习Python异步编程，遇到了很多问题，比如事件循环的概念很难理解。",
            wait_time=8.0,
            test_name="复盘-前置对话1"
        )
        
        await asyncio.sleep(3)
        
        await self.send_and_observe(
            "后来我发现用asyncio.run()作为入口点就简单多了，不用手动管理事件循环。",
            wait_time=8.0,
            test_name="复盘-前置对话2"
        )
        
        await asyncio.sleep(3)
        
        # 2. 启动复盘
        await self.send_and_observe(
            "启动复盘",
            wait_time=10.0,
            test_name="复盘-启动"
        )
        
        await asyncio.sleep(3)
        
        # 3. 选择快速复盘
        await self.send_and_observe(
            "快速复盘",
            wait_time=10.0,
            test_name="复盘-选择模式"
        )
        
        await asyncio.sleep(3)
        
        # 4. 复盘中对话
        await self.send_and_observe(
            "我觉得学习异步编程最重要的经验是：先理解同步和异步的区别，然后从简单的例子开始。",
            wait_time=8.0,
            test_name="复盘-对话1"
        )
        
        await asyncio.sleep(3)
        
        await self.send_and_observe(
            "另外一个教训是不要在异步函数里用time.sleep，应该用asyncio.sleep。",
            wait_time=8.0,
            test_name="复盘-对话2"
        )
        
        await asyncio.sleep(3)
        
        # 5. 结束复盘（触发经验提取）
        await self.send_and_observe(
            "结束复盘",
            wait_time=20.0,
            test_name="复盘-结束(经验提取)"
        )
        
        await asyncio.sleep(5)
        
        # 6. 确认经验
        await self.send_and_observe(
            "确认",
            wait_time=10.0,
            test_name="复盘-确认经验保存"
        )
    
    async def test_review_deep_flow(self):
        """测试9: 深度复盘流程"""
        print("\n\n")
        print("╔" + "="*78 + "╗")
        print("║" + " "*18 + "测试模块9: 深度复盘流程" + " "*33 + "║")
        print("╚" + "="*78 + "╝")
        
        print("\n📋 测试目标: 验证深度复盘（分析更详细、生成更多经验）")
        print("💡 观察要点: 深度分析结果是否比快速复盘更丰富\n")
        
        # 1. 正常对话
        await self.send_and_observe(
            "我在做一个机器人项目，设计了五层架构，但是各层之间的通信经常出问题。",
            wait_time=8.0,
            test_name="深度复盘-前置对话"
        )
        
        await asyncio.sleep(3)
        
        # 2. 启动复盘
        await self.send_and_observe(
            "启动复盘",
            wait_time=10.0,
            test_name="深度复盘-启动"
        )
        
        await asyncio.sleep(3)
        
        # 3. 选择深度复盘
        await self.send_and_observe(
            "深度复盘",
            wait_time=10.0,
            test_name="深度复盘-选择模式"
        )
        
        await asyncio.sleep(3)
        
        # 4. 复盘中对话
        await self.send_and_observe(
            "我发现问题出在事件总线的设计上，需要引入优先级队列来解决。",
            wait_time=8.0,
            test_name="深度复盘-对话"
        )
        
        await asyncio.sleep(3)
        
        # 5. 结束复盘
        await self.send_and_observe(
            "结束复盘",
            wait_time=25.0,
            test_name="深度复盘-结束(经验提取)"
        )
        
        await asyncio.sleep(5)
        
        # 6. 确认
        await self.send_and_observe(
            "确认",
            wait_time=10.0,
            test_name="深度复盘-确认经验保存"
        )
    
    def print_test_summary(self):
        """打印测试总结"""
        print("\n\n")
        print("╔" + "="*78 + "╗")
        print("║" + " "*25 + "测试完成总结" + " "*35 + "║")
        print("╚" + "="*78 + "╝")
        
        print(f"\n📊 测试统计:")
        print(f"   - 总测试轮次: {len(self.test_log)}")
        print(f"   - 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n📝 测试覆盖:")
        print(f"   ✅ 短期记忆 - 多轮上下文保持")
        print(f"   ✅ 长期记忆 - 持久化存储和检索")
        print(f"   ✅ RAG知识检索 - 向量相似度匹配")
        print(f"   ✅ 情景记忆 - 多轮关联和复盘")
        print(f"   ✅ 经验生成 - L2 BACKUP复用L2 CORE")
        print(f"   ✅ 日常闲聊 - 自然对话")
        print(f"   ✅ 复杂任务 - 多步骤任务执行")
        print(f"   ✅ 快速复盘 - 完整流程（启动→对话→经验提取→确认）")
        print(f"   ✅ 深度复盘 - 深度分析流程")
        
        print(f"\n🌐 查看方式:")
        print(f"   - 浏览器: http://localhost:8080")
        print(f"   - 可以查看所有对话历史和系统响应")
        
        print(f"\n💡 提示:")
        print(f"   - 在浏览器中可以看到完整的对话过程")
        print(f"   - 可以观察系统的实时响应")
        print(f"   - 可以验证记忆和RAG的实际效果")
        print()


async def main():
    """主测试流程"""
    print("\n" + "="*80)
    print("🧪 祖龙系统真实环境交互式测试")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print("\n📋 测试说明:")
    print("   1. 所有测试都在真实系统中进行")
    print("   2. 可以在浏览器 http://localhost:8080 观察对话过程")
    print("   3. 每个测试之间有等待时间，请观察系统响应")
    print("   4. L2 BACKUP复用L2 CORE模型，共享同一个vLLM实例")
    
    print("\n⚠️  注意事项:")
    print("   - 测试会自动发送消息，请保持系统运行")
    print("   - 可以随时按 Ctrl+C 中断测试")
    print("   - 建议在测试前清空浏览器对话历史")
    
    input("\n按 Enter 键开始测试...")
    
    tester = InteractiveZulongTester(ws_uri="ws://localhost:5555")
    
    try:
        # 运行所有测试模块
        await tester.test_short_term_memory()
        await asyncio.sleep(3)
        
        await tester.test_long_term_memory()
        await asyncio.sleep(3)
        
        await tester.test_rag_knowledge_retrieval()
        await asyncio.sleep(3)
        
        await tester.test_episodic_memory()
        await asyncio.sleep(3)
        
        await tester.test_experience_generation()
        await asyncio.sleep(3)
        
        await tester.test_casual_chat()
        await asyncio.sleep(3)
        
        await tester.test_complex_task()
        await asyncio.sleep(3)
        
        await tester.test_review_quick_flow()
        await asyncio.sleep(3)
        
        await tester.test_review_deep_flow()
        
        # 打印总结
        tester.print_test_summary()
        
        print("\n✅ 所有测试完成！")
        print("💡 请在浏览器中查看完整的对话历史和系统响应\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        tester.print_test_summary()
    except Exception as e:
        print(f"\n\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
