# File: tests/test_zulong_search_integration.py
# 测试祖龙系统能否在对话中调用 openclaw_search 工具

import sys
import os
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zulong.config.config_manager import get_config, get_llm_config
from zulong.tools.tool_engine import ToolEngine
from zulong.tools.openclaw_search import OpenClawSearchTool
from zulong.l2.inference_engine import InferenceEngine


class ZulongSearchIntegrationTester:
    """祖龙系统搜索集成测试器"""
    
    def __init__(self):
        """初始化测试器"""
        print("=" * 80)
        print("  祖龙系统 - OpenClaw Search 集成测试")
        print("=" * 80)
        
        # 初始化配置
        self.llm_config = get_llm_config()
        print(f"\n✅ LLM 配置已加载")
        print(f"   后端：{self.llm_config['backend']}")
        print(f"   模型：{self.llm_config['model_id']}")
        
        # 初始化推理引擎
        print("\n🔧 初始化推理引擎...")
        self.inference_engine = InferenceEngine()
        print(f"   ✅ InferenceEngine 已初始化")
        
        # 初始化工具引擎
        print("\n🔧 初始化工具引擎...")
        self.tool_engine = ToolEngine(max_workers=5)
        print(f"   ✅ ToolEngine 已初始化")
        
        # 注册搜索工具
        print("\n🔧 注册 OpenClaw Search 工具...")
        try:
            search_tool = OpenClawSearchTool()
            if search_tool.initialize():
                self.tool_engine.registry.register(search_tool)
                print(f"   ✅ OpenClaw Search 工具已注册")
            else:
                print(f"   ❌ OpenClaw Search 工具初始化失败")
        except Exception as e:
            print(f"   ❌ 工具注册异常：{e}")
        
        print("\n" + "=" * 80)
    
    def test_search_tool_call(self, query: str):
        """测试搜索工具调用"""
        print(f"\n🔍 测试查询：{query}")
        print("-" * 80)
        
        try:
            # 直接使用工具引擎执行搜索
            tool_result = self.tool_engine.call_tool(
                tool_name="openclaw_search",
                action="search",
                parameters={"query": query},
                timeout=30.0
            )
            
            if tool_result and hasattr(tool_result, 'result') and tool_result.result:
                results = tool_result.result.get("results", [])
                print(f"   ✅ 搜索成功！返回 {len(results)} 条结果")
                
                # 显示前 3 条结果
                for i, result in enumerate(results[:3], 1):
                    print(f"\n   📊 结果 {i}:")
                    print(f"      标题：{result.get('title', 'N/A')}")
                    print(f"      URL: {result.get('url', 'N/A')}")
                    print(f"      摘要：{result.get('content', 'N/A')[:100]}...")
                
                return True
            else:
                print(f"   ❌ 搜索返回空结果")
                return False
                
        except Exception as e:
            print(f"   ❌ 搜索失败：{e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_tests(self):
        """运行所有测试"""
        print("\n🚀 开始执行集成测试...\n")
        
        # 测试用例
        test_queries = [
            "2026 年最新的 AI 技术突破",
            "量子计算最新进展",
            "SpaceX 星舰发射时间",
        ]
        
        results = []
        for query in test_queries:
            success = self.test_search_tool_call(query)
            results.append((query, success))
            time.sleep(1)  # 避免请求过快
        
        # 总结
        print("\n" + "=" * 80)
        print("  测试总结")
        print("=" * 80)
        
        success_count = sum(1 for _, success in results if success)
        total_count = len(results)
        
        print(f"\n📊 测试统计:")
        print(f"   总测试数：{total_count}")
        print(f"   成功：{success_count}")
        print(f"   失败：{total_count - success_count}")
        print(f"   成功率：{success_count/total_count*100:.1f}%")
        
        print(f"\n📝 详细结果:")
        for query, success in results:
            status = "✅" if success else "❌"
            print(f"   {status} {query}")
        
        if success_count == total_count:
            print("\n" + "=" * 80)
            print("  🎉 所有测试通过！祖龙系统可以正常调用搜索工具")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("  ⚠️ 部分测试失败，请检查日志")
            print("=" * 80)


def main():
    """主函数"""
    try:
        tester = ZulongSearchIntegrationTester()
        tester.run_tests()
    except Exception as e:
        print(f"\n❌ 测试异常：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
