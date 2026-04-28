"""
祖龙系统自动化测试脚本
测试记忆板块和技能包板块功能
"""

import requests
import json
import time
from typing import Dict, Any

# API 配置
API_BASE_URL = "http://localhost:3000"
WEB_BASE_URL = "http://localhost:8080"

class ZulongTester:
    """祖龙系统自动化测试器"""
    
    def __init__(self):
        self.session = requests.Session()
        self.test_results = []
        
    def send_message(self, message: str) -> Dict[str, Any]:
        """发送消息到祖龙系统"""
        try:
            # 通过 Web API 发送消息
            response = self.session.post(
                f"{WEB_BASE_URL}/api/send",
                json={"message": message},
                timeout=30
            )
            return response.json()
        except Exception as e:
            return {"error": str(e), "response": None}
    
    def test_short_term_memory(self):
        """测试短期记忆功能"""
        print("\n" + "="*80)
        print("📝 测试 1：短期记忆测试")
        print("="*80)
        
        # 步骤 1：告诉系统用户信息
        msg1 = "我叫小明，今年 25 岁"
        print(f"\n[测试] 发送：{msg1}")
        result1 = self.send_message(msg1)
        print(f"[回复] {result1.get('response', 'N/A')}")
        time.sleep(1)
        
        # 步骤 2：询问用户信息
        msg2 = "你还记得我叫什么名字吗？"
        print(f"\n[测试] 发送：{msg2}")
        result2 = self.send_message(msg2)
        print(f"[回复] {result2.get('response', 'N/A')}")
        
        # 验证结果
        success = "小明" in result2.get('response', '')
        self.test_results.append({
            "test": "短期记忆",
            "success": success,
            "expected": "包含'小明'",
            "actual": result2.get('response', 'N/A')
        })
        
        print(f"\n✅ 测试结果：{'通过' if success else '失败'}")
        return success
    
    def test_long_term_memory(self):
        """测试长期记忆（RAG）功能"""
        print("\n" + "="*80)
        print("📚 测试 2：长期记忆（RAG）测试")
        print("="*80)
        
        # 步骤 1：存储用户偏好
        msg1 = "我喜欢吃川菜，特别是水煮鱼和麻婆豆腐"
        print(f"\n[测试] 发送：{msg1}")
        result1 = self.send_message(msg1)
        print(f"[回复] {result1.get('response', 'N/A')}")
        time.sleep(1)
        
        # 步骤 2：插入其他话题
        msg2 = "今天天气怎么样？"
        print(f"\n[测试] 发送：{msg2}")
        result2 = self.send_message(msg2)
        print(f"[回复] {result2.get('response', 'N/A')}")
        time.sleep(1)
        
        # 步骤 3：询问之前的偏好
        msg3 = "我之前说过喜欢吃什么菜？"
        print(f"\n[测试] 发送：{msg3}")
        result3 = self.send_message(msg3)
        print(f"[回复] {result3.get('response', 'N/A')}")
        
        # 验证结果
        response_text = result3.get('response', '')
        success = "川菜" in response_text or "水煮鱼" in response_text
        self.test_results.append({
            "test": "长期记忆 (RAG)",
            "success": success,
            "expected": "包含'川菜'或'水煮鱼'",
            "actual": response_text
        })
        
        print(f"\n✅ 测试结果：{'通过' if success else '失败'}")
        return success
    
    def test_experience_learning(self):
        """测试经验学习功能"""
        print("\n" + "="*80)
        print("🧠 测试 3：经验学习测试")
        print("="*80)
        
        # 步骤 1：请求写加法函数
        msg1 = "请帮我写一个 Python 函数，计算两个数的和"
        print(f"\n[测试] 发送：{msg1}")
        result1 = self.send_message(msg1)
        print(f"[回复] {result1.get('response', 'N/A')[:200]}...")
        time.sleep(1)
        
        # 步骤 2：要求改进
        msg2 = "这个函数应该支持浮点数，并且添加类型注解"
        print(f"\n[测试] 发送：{msg2}")
        result2 = self.send_message(msg2)
        print(f"[回复] {result2.get('response', 'N/A')[:200]}...")
        time.sleep(1)
        
        # 步骤 3：请求类似的减法函数
        msg3 = "很好，现在请写一个减法函数，使用类似的风格"
        print(f"\n[测试] 发送：{msg3}")
        result3 = self.send_message(msg3)
        print(f"[回复] {result3.get('response', 'N/A')[:200]}...")
        
        # 验证结果
        response_text = result3.get('response', '')
        success = "def" in response_text and "sub" in response_text.lower()
        self.test_results.append({
            "test": "经验学习",
            "success": success,
            "expected": "包含函数定义",
            "actual": response_text[:100] + "..."
        })
        
        print(f"\n✅ 测试结果：{'通过' if success else '失败'}")
        return success
    
    def test_skill_pack_architecture(self):
        """测试技能包架构理解"""
        print("\n" + "="*80)
        print("🎯 测试 4：技能包架构理解测试")
        print("="*80)
        
        # 测试对技能包架构的理解
        msg1 = "请解释一下祖龙系统的技能包架构设计理念"
        print(f"\n[测试] 发送：{msg1}")
        result1 = self.send_message(msg1)
        response_text = result1.get('response', '')
        print(f"[回复] {response_text[:300]}...")
        
        # 验证是否理解核心理念
        keywords = ["借用", "学习", "内化", "丢弃", "技能包"]
        found_keywords = [kw for kw in keywords if kw in response_text]
        
        success = len(found_keywords) >= 3
        self.test_results.append({
            "test": "技能包架构理解",
            "success": success,
            "expected": f"包含至少 3 个关键词：{keywords}",
            "actual": f"找到关键词：{found_keywords}"
        })
        
        print(f"\n✅ 测试结果：{'通过' if success else '失败'}")
        print(f"   找到的关键词：{found_keywords}")
        return success
    
    def test_task_decomposition(self):
        """测试任务拆解能力"""
        print("\n" + "="*80)
        print("📋 测试 5：任务拆解能力测试")
        print("="*80)
        
        # 请求拆解复杂任务
        msg = "请帮我规划一个 Python Web 项目，需要包含用户注册、登录、数据管理功能"
        print(f"\n[测试] 发送：{msg}")
        result = self.send_message(msg)
        response_text = result.get('response', '')
        print(f"[回复] {response_text[:300]}...")
        
        # 验证是否有任务拆解
        decomposition_keywords = ["步骤", "阶段", "首先", "然后", "接下来", "1.", "2.", "3."]
        found_keywords = [kw for kw in decomposition_keywords if kw in response_text]
        
        success = len(found_keywords) >= 2
        self.test_results.append({
            "test": "任务拆解能力",
            "success": success,
            "expected": f"包含任务拆解关键词：{decomposition_keywords}",
            "actual": f"找到关键词：{found_keywords}"
        })
        
        print(f"\n✅ 测试结果：{'通过' if success else '失败'}")
        return success
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*80)
        print("🚀 祖龙系统自动化测试开始")
        print("="*80)
        print(f"API 地址：{API_BASE_URL}")
        print(f"Web 地址：{WEB_BASE_URL}")
        
        # 检查系统是否在线
        try:
            response = requests.get(WEB_BASE_URL, timeout=5)
            print(f"\n✅ 系统状态：在线 (HTTP {response.status_code})")
        except Exception as e:
            print(f"\n❌ 系统状态：离线 ({e})")
            print("请确保祖龙系统和 OpenClaw Bridge 已启动")
            return
        
        # 运行测试
        self.test_short_term_memory()
        time.sleep(2)
        
        self.test_long_term_memory()
        time.sleep(2)
        
        self.test_experience_learning()
        time.sleep(2)
        
        self.test_skill_pack_architecture()
        time.sleep(2)
        
        self.test_task_decomposition()
        
        # 生成测试报告
        self.generate_report()
    
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
            "success_rate": passed/total*100,
            "results": self.test_results
        }
        
        with open("test_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📄 测试报告已保存到：test_report.json")
        print("\n" + "="*80)


if __name__ == "__main__":
    tester = ZulongTester()
    tester.run_all_tests()
