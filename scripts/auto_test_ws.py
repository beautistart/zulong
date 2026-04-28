"""
祖龙系统自动化测试脚本（WebSocket 版本）
测试记忆板块和技能包板块功能
"""

import asyncio
import websockets
import json
import time
from typing import Dict, Any

# 配置
WEBSOCKET_URL = "ws://localhost:5555"

class ZulongWebSocketTester:
    """祖龙系统 WebSocket 自动化测试器"""
    
    def __init__(self):
        self.test_results = []
        self.websocket = None
        self.last_response = None
        
    async def connect(self):
        """连接到祖龙系统 WebSocket"""
        try:
            self.websocket = await websockets.connect(WEBSOCKET_URL)
            print(f"\n✅ 已连接到祖龙系统：{WEBSOCKET_URL}")
            return True
        except Exception as e:
            print(f"\n❌ 连接失败：{e}")
            return False
    
    async def send_message(self, message: str) -> str:
        """发送消息并等待回复"""
        try:
            # 发送消息到 WebSocket（使用正确的事件格式）
            event = {
                "id": str(time.time()),
                "type": "USER_TEXT",
                "priority": 1,
                "source": "openclaw/web_ui",
                "payload": {
                    "text": message
                },
                "timestamp": time.time()
            }
            await self.websocket.send(json.dumps(event))
            
            # 等待回复（最多 30 秒）
            try:
                response = await asyncio.wait_for(self.websocket.recv(), timeout=30)
                data = json.loads(response)
                self.last_response = data.get('content', str(data))
                return self.last_response
            except asyncio.TimeoutError:
                return "[超时] 未收到回复"
                
        except Exception as e:
            return f"[错误] {str(e)}"
    
    async def test_short_term_memory(self):
        """测试短期记忆功能"""
        print("\n" + "="*80)
        print("📝 测试 1：短期记忆测试")
        print("="*80)
        
        # 步骤 1：告诉系统用户信息
        msg1 = "我叫小明，今年 25 岁"
        print(f"\n[测试] 发送：{msg1}")
        result1 = await self.send_message(msg1)
        print(f"[回复] {result1[:200]}...")
        await asyncio.sleep(1)
        
        # 步骤 2：询问用户信息
        msg2 = "你还记得我叫什么名字吗？"
        print(f"\n[测试] 发送：{msg2}")
        result2 = await self.send_message(msg2)
        print(f"[回复] {result2[:200]}...")
        
        # 验证结果
        success = "小明" in result2
        self.test_results.append({
            "test": "短期记忆",
            "success": success,
            "expected": "包含'小明'",
            "actual": result2[:100] + "..."
        })
        
        print(f"\n✅ 测试结果：{'通过' if success else '失败'}")
        return success
    
    async def test_long_term_memory(self):
        """测试长期记忆（RAG）功能"""
        print("\n" + "="*80)
        print("📚 测试 2：长期记忆（RAG）测试")
        print("="*80)
        
        # 步骤 1：存储用户偏好
        msg1 = "我喜欢吃川菜，特别是水煮鱼和麻婆豆腐"
        print(f"\n[测试] 发送：{msg1}")
        result1 = await self.send_message(msg1)
        print(f"[回复] {result1[:200]}...")
        await asyncio.sleep(1)
        
        # 步骤 2：插入其他话题
        msg2 = "今天天气怎么样？"
        print(f"\n[测试] 发送：{msg2}")
        result2 = await self.send_message(msg2)
        print(f"[回复] {result2[:200]}...")
        await asyncio.sleep(1)
        
        # 步骤 3：询问之前的偏好
        msg3 = "我之前说过喜欢吃什么菜？"
        print(f"\n[测试] 发送：{msg3}")
        result3 = await self.send_message(msg3)
        print(f"[回复] {result3[:200]}...")
        
        # 验证结果
        success = "川菜" in result3 or "水煮鱼" in result3
        self.test_results.append({
            "test": "长期记忆 (RAG)",
            "success": success,
            "expected": "包含'川菜'或'水煮鱼'",
            "actual": result3[:100] + "..."
        })
        
        print(f"\n✅ 测试结果：{'通过' if success else '失败'}")
        return success
    
    async def test_experience_learning(self):
        """测试经验学习功能"""
        print("\n" + "="*80)
        print("🧠 测试 3：经验学习测试")
        print("="*80)
        
        # 步骤 1：请求写加法函数
        msg1 = "请帮我写一个 Python 函数，计算两个数的和"
        print(f"\n[测试] 发送：{msg1}")
        result1 = await self.send_message(msg1)
        print(f"[回复] {result1[:200]}...")
        await asyncio.sleep(1)
        
        # 步骤 2：要求改进
        msg2 = "这个函数应该支持浮点数，并且添加类型注解"
        print(f"\n[测试] 发送：{msg2}")
        result2 = await self.send_message(msg2)
        print(f"[回复] {result2[:200]}...")
        await asyncio.sleep(1)
        
        # 步骤 3：请求类似的减法函数
        msg3 = "很好，现在请写一个减法函数，使用类似的风格"
        print(f"\n[测试] 发送：{msg3}")
        result3 = await self.send_message(msg3)
        print(f"[回复] {result3[:200]}...")
        
        # 验证结果
        success = "def" in result3.lower() and ("sub" in result3.lower() or "减" in result3)
        self.test_results.append({
            "test": "经验学习",
            "success": success,
            "expected": "包含函数定义",
            "actual": result3[:100] + "..."
        })
        
        print(f"\n✅ 测试结果：{'通过' if success else '失败'}")
        return success
    
    async def test_skill_pack_architecture(self):
        """测试技能包架构理解"""
        print("\n" + "="*80)
        print("🎯 测试 4：技能包架构理解测试")
        print("="*80)
        
        # 测试对技能包架构的理解
        msg1 = "请解释一下祖龙系统的技能包架构设计理念"
        print(f"\n[测试] 发送：{msg1}")
        result1 = await self.send_message(msg1)
        print(f"[回复] {result1[:300]}...")
        
        # 验证是否理解核心理念
        keywords = ["借用", "学习", "内化", "丢弃", "技能包", "框架"]
        found_keywords = [kw for kw in keywords if kw in result1]
        
        success = len(found_keywords) >= 2
        self.test_results.append({
            "test": "技能包架构理解",
            "success": success,
            "expected": f"包含至少 2 个关键词：{keywords}",
            "actual": f"找到关键词：{found_keywords}"
        })
        
        print(f"\n✅ 测试结果：{'通过' if success else '失败'}")
        print(f"   找到的关键词：{found_keywords}")
        return success
    
    async def test_task_decomposition(self):
        """测试任务拆解能力"""
        print("\n" + "="*80)
        print("📋 测试 5：任务拆解能力测试")
        print("="*80)
        
        # 请求拆解复杂任务
        msg = "请帮我规划一个 Python Web 项目，需要包含用户注册、登录、数据管理功能"
        print(f"\n[测试] 发送：{msg}")
        result = await self.send_message(msg)
        print(f"[回复] {result[:300]}...")
        
        # 验证是否有任务拆解
        decomposition_keywords = ["步骤", "阶段", "首先", "然后", "接下来", "1.", "2.", "3.", "第一", "第二"]
        found_keywords = [kw for kw in decomposition_keywords if kw in result]
        
        success = len(found_keywords) >= 2
        self.test_results.append({
            "test": "任务拆解能力",
            "success": success,
            "expected": f"包含任务拆解关键词：{decomposition_keywords}",
            "actual": f"找到关键词：{found_keywords}"
        })
        
        print(f"\n✅ 测试结果：{'通过' if success else '失败'}")
        return success
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*80)
        print("🚀 祖龙系统自动化测试开始（WebSocket 版本）")
        print("="*80)
        print(f"WebSocket 地址：{WEBSOCKET_URL}")
        
        # 连接系统
        if not await self.connect():
            print("❌ 无法连接到祖龙系统，请确保系统已启动")
            return
        
        try:
            # 运行测试
            await self.test_short_term_memory()
            await asyncio.sleep(2)
            
            await self.test_long_term_memory()
            await asyncio.sleep(2)
            
            await self.test_experience_learning()
            await asyncio.sleep(2)
            
            await self.test_skill_pack_architecture()
            await asyncio.sleep(2)
            
            await self.test_task_decomposition()
            
            # 生成测试报告
            self.generate_report()
            
        finally:
            # 关闭连接
            await self.websocket.close()
            print("\n👋 已断开连接")
    
    def generate_report(self):
        """生成测试报告"""
        print("\n" + "="*80)
        print("📊 测试报告")
        print("="*80)
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["success"])
        failed = total - passed
        
        print(f"\n总测试数：{total}")
        print(f"✅ 通过：{passed}")
        print(f"❌ 失败：{failed}")
        if total > 0:
            print(f"成功率：{passed/total*100:.1f}%")
        
        print("\n详细结果：")
        print("-"*80)
        for i, result in enumerate(self.test_results, 1):
            status = "✅" if result["success"] else "❌"
            print(f"{i}. {status} {result['test']}")
            print(f"   预期：{result['expected']}")
            print(f"   实际：{result['actual']}")
            print()
        
        # 保存测试报告
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": total,
            "passed": passed,
            "failed": failed,
            "success_rate": passed/total*100 if total > 0 else 0,
            "results": self.test_results
        }
        
        with open("test_report_websocket.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📄 测试报告已保存到：test_report_websocket.json")
        print("\n" + "="*80)


async def main():
    tester = ZulongWebSocketTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
