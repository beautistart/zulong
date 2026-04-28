# File: tests/test_openclaw_search.py
# 测试 OpenClaw Search 工具通过 Docker SearxNG 访问网络

import sys
import os
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zulong.config.config_manager import get_config, get_llm_config
from zulong.tools.tool_engine import ToolEngine
from zulong.tools.openclaw_search import OpenClawSearchTool


class OpenClawSearchTester:
    """OpenClaw Search 测试器"""
    
    def __init__(self):
        """初始化测试器"""
        print("=" * 80)
        print("  OpenClaw Search - SearxNG 网络访问测试")
        print("=" * 80)
        
        # 初始化配置
        self.llm_config = get_llm_config()
        print(f"\n✅ LLM 配置已加载")
        print(f"   后端：{self.llm_config['backend']}")
        print(f"   模型：{self.llm_config['model_id']}")
        
        # 初始化工具引擎
        print("\n🔧 初始化工具引擎...")
        self.tool_engine = ToolEngine(max_workers=5)
        print(f"   ✅ ToolEngine 已初始化 (max_workers=5)")
        
        # 注册 OpenClaw Search 工具
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
        
        # 测试结果统计
        self.test_results = []
        self.total_tests = 0
        self.successful_tests = 0
        self.failed_tests = 0
        
        print("\n" + "=" * 80)
    
    def test_searxng_connectivity(self):
        """测试 1: SearxNG 连接性测试"""
        print("\n" + "=" * 80)
        print("  测试 1: SearxNG 连接性测试 🔌")
        print("=" * 80)
        
        # 检查 SearxNG Docker 容器
        print("\n📋 检查 SearxNG Docker 容器状态...")
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=searxng", "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and "searxng" in result.stdout.lower():
                print(f"   ✅ SearxNG 容器正在运行")
                print(f"   状态：{result.stdout.strip()}")
            else:
                print(f"   ⚠️ SearxNG 容器未运行")
                print(f"   提示：运行 'docker start searxng' 启动容器")
                self._record_result("SearxNG 连接性", False)
                return
        except Exception as e:
            print(f"   ❌ 检查失败：{e}")
            self._record_result("SearxNG 连接性", False)
            return
        
        # 测试 SearxNG API 可访问性
        print("\n📋 测试 SearxNG API 可访问性...")
        try:
            import requests
            searxng_url = "http://localhost:8101"
            
            response = requests.get(searxng_url, timeout=5)
            
            if response.status_code == 200:
                print(f"   ✅ SearxNG API 可访问 (状态码：{response.status_code})")
                self._record_result("SearxNG API 访问", True)
            else:
                print(f"   ⚠️ SearxNG API 返回异常状态码：{response.status_code}")
                self._record_result("SearxNG API 访问", False)
        except requests.exceptions.ConnectionError:
            print(f"   ❌ 无法连接到 SearxNG (http://localhost:8101)")
            print(f"   提示：检查 Docker 容器端口映射")
            self._record_result("SearxNG API 访问", False)
        except Exception as e:
            print(f"   ❌ 测试失败：{e}")
            self._record_result("SearxNG API 访问", False)
    
    def test_search_functionality(self):
        """测试 2: 搜索功能测试"""
        print("\n" + "=" * 80)
        print("  测试 2: 搜索功能测试 🔍")
        print("=" * 80)
        
        test_queries = [
            "2026 年 AI 技术发展趋势",
            "Python 编程教程",
            "北京天气预报"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📋 测试查询 {i}: {query}")
            
            try:
                result = self.tool_engine.call_tool(
                    tool_name="openclaw_search",
                    action="search",
                    parameters={
                        "query": query,
                        "count": 3
                    },
                    timeout=30.0
                )
                
                if result.success:
                    print(f"   ✅ 搜索成功")
                    
                    # 检查结果格式
                    if hasattr(result, 'data') and result.data:
                        data = result.data
                        if isinstance(data, dict) and 'results' in data:
                            results = data['results']
                            print(f"   📊 返回 {len(results)} 条结果")
                            
                            if results:
                                print(f"   📝 第一条结果:")
                                first = results[0]
                                print(f"      标题：{first.get('title', 'N/A')}")
                                print(f"      URL: {first.get('url', 'N/A')}")
                                print(f"      摘要：{first.get('snippet', 'N/A')[:100]}...")
                            
                            self._record_result(f"搜索测试-{i}", True)
                        else:
                            print(f"   ⚠️ 返回数据格式异常：{data}")
                            self._record_result(f"搜索测试-{i}", False)
                    else:
                        print(f"   ⚠️ 返回结果为空")
                        self._record_result(f"搜索测试-{i}", False)
                else:
                    print(f"   ❌ 搜索失败：{result.error}")
                    self._record_result(f"搜索测试-{i}", False)
                    
            except Exception as e:
                print(f"   ❌ 异常：{e}")
                self._record_result(f"搜索测试-{i}", False)
            
            # 等待一下，避免请求过快
            time.sleep(1)
    
    def test_webpage_fetch(self):
        """测试 3: 网页获取测试"""
        print("\n" + "=" * 80)
        print("  测试 3: 网页获取测试 🌐")
        print("=" * 80)
        
        test_urls = [
            "https://www.example.com",
            "https://httpbin.org/html"
        ]
        
        for i, url in enumerate(test_urls, 1):
            print(f"\n📋 测试 URL {i}: {url}")
            
            try:
                result = self.tool_engine.call_tool(
                    tool_name="openclaw_search",
                    action="fetch_webpage",
                    parameters={
                        "url": url
                    },
                    timeout=30.0
                )
                
                if result.success:
                    print(f"   ✅ 网页获取成功")
                    
                    if hasattr(result, 'data') and result.data:
                        data = result.data
                        if isinstance(data, dict):
                            content = data.get('content', '')
                            print(f"   📊 网页内容长度：{len(content)} 字符")
                            print(f"   📝 内容预览：{content[:200]}...")
                            self._record_result(f"网页获取-{i}", True)
                        else:
                            print(f"   ⚠️ 返回数据格式异常")
                            self._record_result(f"网页获取-{i}", False)
                    else:
                        print(f"   ⚠️ 返回结果为空")
                        self._record_result(f"网页获取-{i}", False)
                else:
                    print(f"   ❌ 网页获取失败：{result.error}")
                    self._record_result(f"网页获取-{i}", False)
                    
            except Exception as e:
                print(f"   ❌ 异常：{e}")
                self._record_result(f"网页获取-{i}", False)
            
            time.sleep(1)
    
    def _record_result(self, test_name: str, success: bool):
        """记录测试结果"""
        self.total_tests += 1
        if success:
            self.successful_tests += 1
        else:
            self.failed_tests += 1
        
        self.test_results.append({
            "test_name": test_name,
            "success": success,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 80)
        print("  测试总结")
        print("=" * 80)
        
        success_rate = (self.successful_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"\n📊 测试统计:")
        print(f"   总测试数：{self.total_tests}")
        print(f"   成功：{self.successful_tests}")
        print(f"   失败：{self.failed_tests}")
        print(f"   成功率：{success_rate:.1f}%")
        
        print(f"\n📝 详细结果:")
        for result in self.test_results:
            status = "✅" if result["success"] else "❌"
            print(f"   {status} {result['test_name']} - {result['timestamp']}")
        
        print("\n" + "=" * 80)
        
        if success_rate >= 80:
            print("  🎉 测试通过！OpenClaw Search 可以正常通过 SearxNG 访问网络")
        elif success_rate >= 50:
            print("  ⚠️ 部分测试通过，建议检查配置")
        else:
            print("  ❌ 测试失败，请检查 SearxNG 配置和网络连接")
        
        print("=" * 80)
        print("  测试完成!")
        print("=" * 80)


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("  OpenClaw Search - SearxNG 网络访问测试")
    print("  测试通过 Docker SearxNG 容器访问网络的能力")
    print("=" * 80)
    
    # 创建测试器
    tester = OpenClawSearchTester()
    
    # 执行测试
    print("\n🚀 开始执行测试...\n")
    
    # 测试 1: SearxNG 连接性
    tester.test_searxng_connectivity()
    
    # 测试 2: 搜索功能
    tester.test_search_functionality()
    
    # 测试 3: 网页获取
    tester.test_webpage_fetch()
    
    # 打印总结
    tester.print_summary()
    
    # 返回退出码
    return 0 if tester.failed_tests == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
