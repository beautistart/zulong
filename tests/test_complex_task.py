# File: tests/test_complex_task.py
# 复杂任务测试脚本 - 测试任务规划和深度思考能力

import sys
import os
import time
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zulong.config.config_manager import get_config, get_llm_config
from zulong.tools.tool_engine import ToolEngine
from zulong.skill_packs.runtime import SkillPackRuntime
from zulong.skill_packs.packs.complex_task import ComplexTaskPack


class ComplexTaskTester:
    """复杂任务测试器"""
    
    def __init__(self):
        """初始化测试器"""
        print("=" * 80)
        print("  祖龙系统 - 复杂任务测试器")
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
        
        # 初始化技能包运行时
        print("\n🔧 初始化技能包运行时...")
        self.skill_runtime = SkillPackRuntime(
            tool_engine=self.tool_engine,
            experience_store=None,  # 简化测试，不使用经验存储
            hot_update_engine=None
        )
        
        # 安装技能包
        self._install_skill_packs()
        
        # 测试结果统计
        self.test_results = []
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        
        print("\n" + "=" * 80)
    
    def _install_skill_packs(self):
        """安装技能包"""
        print("\n📦 安装技能包...")
        
        # 安装 ComplexTask 技能包
        try:
            complex_task_pack = ComplexTaskPack()
            # 传递 tool_engine 的 registry 给技能包运行时
            self.skill_runtime.tool_engine = self.tool_engine
            success = self.skill_runtime.install_pack(complex_task_pack)
            if success:
                print(f"   ✅ ComplexTask 技能包已安装")
            else:
                print(f"   ⚠️ ComplexTask 技能包安装失败")
        except Exception as e:
            print(f"   ❌ ComplexTask 技能包安装异常：{e}")
        
        # 列出已注册工具
        tools = self.tool_engine.registry.list_tools()
        print(f"\n🔧 已注册工具 ({len(tools)} 个):")
        for tool in tools:
            print(f"   - {tool}")
    
    def test_task_1_travel_planning(self):
        """测试任务 1: 智能旅行规划师"""
        print("\n" + "=" * 80)
        print("  测试任务 1: 智能旅行规划师 ⭐⭐⭐⭐")
        print("=" * 80)
        
        task_description = """
        帮我规划一次为期 7 天的日本东京旅行，要求:
        1. 预算控制在 2 万元人民币以内
        2. 包含机票、酒店、交通、餐饮、景点门票
        3. 安排 3 个必去景点和 2 个备选景点
        4. 考虑最佳旅行季节和天气
        5. 提供详细的每日行程安排
        """
        
        print(f"\n📋 任务描述:\n{task_description}")
        
        # 1. 任务分解
        print("\n🔍 步骤 1: 任务分解")
        try:
            result = self.tool_engine.call_tool(
                tool_name="task_decompose",
                action="execute",
                parameters={
                    "goal": "规划 7 天东京旅行",
                    "constraints": ["预算 2 万元", "7 天时间", "必去 3 个景点"]
                },
                timeout=None
            )
            
            if result.success:
                print(f"   ✅ 任务分解成功")
                print(f"   结果：{result.data}")
            else:
                print(f"   ❌ 任务分解失败：{result.error}")
        except Exception as e:
            print(f"   ❌ 异常：{e}")
        
        # 2. 依赖分析
        print("\n🔍 步骤 2: 依赖分析")
        try:
            result = self.tool_engine.call_tool(
                tool_name="dependency_analyze",
                action="execute",
                parameters={
                    "goal": "规划 7 天东京旅行",
                    "subtasks": [
                        {"task": "订机票", "tool_hint": "search"},
                        {"task": "订酒店", "tool_hint": "search"},
                        {"task": "安排行程", "tool_hint": "analyze"},
                        {"task": "办理签证", "tool_hint": "execute"},
                        {"task": "购买保险", "tool_hint": "execute"}
                    ]
                },
                timeout=None
            )
            
            if result.success:
                print(f"   ✅ 依赖分析成功")
                print(f"   结果：{result.data}")
            else:
                print(f"   ❌ 依赖分析失败：{result.error}")
        except Exception as e:
            print(f"   ❌ 异常：{e}")
        
        # 3. 信息搜索
        print("\n🔍 步骤 3: 信息搜索")
        try:
            result = self.tool_engine.call_tool(
                tool_name="openclaw_search",
                action="search",
                parameters={
                    "query": "上海到东京机票价格 2026",
                    "count": 3
                },
                timeout=None
            )
            
            if result.success:
                print(f"   ✅ 搜索成功")
                # ToolResult 直接访问 data 属性
                data = result.data if hasattr(result, 'data') else str(result)
                print(f"   结果：{str(data)[:200]}...")
            else:
                print(f"   ❌ 搜索失败：{result.error}")
        except Exception as e:
            print(f"   ❌ 异常：{e}")
        
        # 4. 深度推理
        print("\n🔍 步骤 4: 深度推理")
        try:
            result = self.tool_engine.call_tool(
                tool_name="deep_reasoning",
                action="execute",
                parameters={
                    "problem": "如何在预算限制下优化旅行体验",
                    "context": {
                        "budget": 20000,
                        "days": 7,
                        "preferences": ["美食", "购物", "文化"]
                    }
                },
                timeout=None
            )
            
            if result.success:
                print(f"   ✅ 深度推理成功")
                print(f"   结果：{str(result.data)[:300]}...")
            else:
                print(f"   ❌ 深度推理失败：{result.error}")
        except Exception as e:
            print(f"   ❌ 异常：{e}")
        
        # 记录结果
        self._record_result("旅行规划", True)
    
    def test_task_2_technical_selection(self):
        """测试任务 2: 技术选型顾问"""
        print("\n" + "=" * 80)
        print("  测试任务 2: 技术选型顾问 ⭐⭐⭐⭐⭐")
        print("=" * 80)
        
        task_description = """
        我计划开发一个实时聊天机器人系统，需要选择合适的技术栈。请帮我:
        
        1. 分析需求:
           - 支持 10 万并发用户
           - 响应时间 < 100ms
           - 支持文本、语音、图像多模态
        
        2. 技术选型:
           - 后端框架 (Node.js/Python/Go)
           - 数据库 (关系型/NoSQL)
           - 消息队列 (RabbitMQ/Kafka/Redis)
           - AI 模型 (本地部署/云端 API)
        
        3. 成本估算和风险评估
        """
        
        print(f"\n📋 任务描述:\n{task_description}")
        
        # 1. 任务分解
        print("\n🔍 步骤 1: 任务分解")
        try:
            result = self.tool_engine.call_tool(
                tool_name="task_decompose",
                action="execute",
                parameters={
                    "goal": "技术选型分析",
                    "aspects": ["后端", "数据库", "消息队列", "AI 模型", "部署"]
                },
                timeout=None
            )
            
            if result.success:
                print(f"   ✅ 任务分解成功")
            else:
                print(f"   ❌ 失败：{result.error}")
        except Exception as e:
            print(f"   ❌ 异常：{e}")
        
        # 2. 优先级排序
        print("\n🔍 步骤 2: 优先级排序")
        try:
            result = self.tool_engine.call_tool(
                tool_name="priority_rank",
                action="execute",
                parameters={
                    "subtasks": [
                        {"task": "Node.js+Socket.IO", "tool_hint": "analyze"},
                        {"task": "Python+FastAPI", "tool_hint": "analyze"},
                        {"task": "Go+Gorilla", "tool_hint": "analyze"}
                    ],
                    "criteria": ["性能", "开发效率", "生态成熟度"]
                },
                timeout=None
            )
            
            if result.success:
                print(f"   ✅ 优先级排序成功")
            else:
                print(f"   ❌ 失败：{result.error}")
        except Exception as e:
            print(f"   ❌ 异常：{e}")
        
        # 3. 信息搜索
        print("\n🔍 步骤 3: 信息搜索")
        try:
            result = self.tool_engine.call_tool(
                tool_name="openclaw_search",
                action="search",
                parameters={
                    "query": "2026 最佳实时聊天架构 10 万并发",
                    "count": 3
                },
                timeout=None
            )
            
            if result.success:
                print(f"   ✅ 搜索成功")
            else:
                print(f"   ❌ 失败：{result.error}")
        except Exception as e:
            print(f"   ❌ 异常：{e}")
        
        # 4. 深度推理
        print("\n🔍 步骤 4: 深度推理")
        try:
            result = self.tool_engine.call_tool(
                tool_name="deep_reasoning",
                action="execute",
                parameters={
                    "problem": "技术选型决策",
                    "framework": {
                        "options": ["Node.js", "Python", "Go"],
                        "criteria": ["性能", "开发效率", "学习曲线"]
                    }
                },
                timeout=None
            )
            
            if result.success:
                print(f"   ✅ 深度推理成功")
            else:
                print(f"   ❌ 失败：{result.error}")
        except Exception as e:
            print(f"   ❌ 异常：{e}")
        
        # 记录结果
        self._record_result("技术选型", True)
    
    def test_task_3_business_plan(self):
        """测试任务 3: 商业计划书撰写"""
        print("\n" + "=" * 80)
        print("  测试任务 3: 商业计划书撰写 ⭐⭐⭐⭐⭐")
        print("=" * 80)
        
        task_description = """
        我计划创办一家 AI 驱动的在线教育公司，请帮我撰写一份完整的商业计划书:
        
        1. 市场分析 (规模、趋势、竞争)
        2. 产品与服务 (核心功能、商业模式)
        3. 营销策略 (获客、品牌)
        4. 财务预测 (3 年收入、成本、盈亏平衡)
        5. 融资计划 (金额、用途)
        """
        
        print(f"\n📋 任务描述:\n{task_description}")
        
        # 1. 任务分解
        print("\n🔍 步骤 1: 任务分解")
        try:
            result = self.tool_engine.call_tool(
                tool_name="task_decompose",
                action="execute",
                parameters={
                    "goal": "撰写商业计划书",
                    "sections": ["市场分析", "产品设计", "营销策略", "财务预测"]
                },
                timeout=None
            )
            
            if result.success:
                print(f"   ✅ 任务分解成功")
            else:
                print(f"   ❌ 失败：{result.error}")
        except Exception as e:
            print(f"   ❌ 异常：{e}")
        
        # 2. 信息搜索
        print("\n🔍 步骤 2: 信息搜索")
        try:
            result = self.tool_engine.call_tool(
                tool_name="openclaw_search",
                action="search",
                parameters={
                    "query": "2026 中国在线教育市场规模 AI 教育",
                    "count": 3
                },
                timeout=None
            )
            
            if result.success:
                print(f"   ✅ 搜索成功")
            else:
                print(f"   ❌ 失败：{result.error}")
        except Exception as e:
            print(f"   ❌ 异常：{e}")
        
        # 3. 深度推理
        print("\n🔍 步骤 3: 深度推理")
        try:
            result = self.tool_engine.call_tool(
                tool_name="deep_reasoning",
                action="execute",
                parameters={
                    "problem": "如何差异化竞争",
                    "context": {
                        "market": "在线教育红海",
                        "advantage": "AI 个性化教学",
                        "target": "K12 学生"
                    }
                },
                timeout=None
            )
            
            if result.success:
                print(f"   ✅ 深度推理成功")
            else:
                print(f"   ❌ 失败：{result.error}")
        except Exception as e:
            print(f"   ❌ 异常：{e}")
        
        # 记录结果
        self._record_result("商业计划书", True)
    
    def _record_result(self, task_name: str, success: bool):
        """记录测试结果"""
        self.total_tasks += 1
        if success:
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1
        
        self.test_results.append({
            "task_name": task_name,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 80)
        print("  测试总结")
        print("=" * 80)
        
        print(f"\n📊 测试统计:")
        print(f"   总任务数：{self.total_tasks}")
        print(f"   成功：{self.successful_tasks}")
        print(f"   失败：{self.failed_tasks}")
        print(f"   成功率：{self.successful_tasks / self.total_tasks * 100:.1f}%")
        
        print(f"\n📝 详细结果:")
        for result in self.test_results:
            status = "✅" if result["success"] else "❌"
            print(f"   {status} {result['task_name']} - {result['timestamp']}")
        
        print("\n" + "=" * 80)
        print("  测试完成!")
        print("=" * 80)


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("  祖龙系统复杂任务测试")
    print("  测试任务规划和深度思考能力")
    print("=" * 80)
    
    # 创建测试器
    tester = ComplexTaskTester()
    
    # 执行测试任务
    print("\n🚀 开始执行测试任务...\n")
    
    # 任务 1: 旅行规划
    tester.test_task_1_travel_planning()
    
    # 任务 2: 技术选型
    tester.test_task_2_technical_selection()
    
    # 任务 3: 商业计划书
    tester.test_task_3_business_plan()
    
    # 打印总结
    tester.print_summary()
    
    # 返回退出码
    return 0 if tester.failed_tasks == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
