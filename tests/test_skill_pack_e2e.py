#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
技能包系统端到端联合验证测试

验证完整的技能包生命周期：
1. 技能包加载 (loader.py)
2. 技能包安装 (runtime.py)
3. 工具注册 (ToolRegistry)
4. 技能包执行 (execute_capability)
5. 工具调用 (ToolEngine)
6. 经验记录 (ExperienceGenerator)
7. 技能包卸载 (uninstall_pack)
8. 内化评估 (check_internalization)

测试所有技能包：
- AutoGPT Planner
- OpenManus Reasoner
- Cline Coder
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSkillPackEndToEnd(unittest.TestCase):
    """技能包系统端到端测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试前准备"""
        print("\n" + "=" * 80)
        print("技能包系统端到端联合验证")
        print("=" * 80)
    
    def setUp(self):
        """每个测试前准备"""
        # 创建 Mock 依赖
        self.mock_tool_engine = Mock()
        self.mock_tool_engine.registry = Mock()
        self.mock_tool_engine.registry.tools = {}
        self.mock_tool_engine.register_tool = Mock()
        
        # Mock experience_store 的方法返回具体值
        self.mock_experience_store = Mock()
        self.mock_experience_store.add_experience = Mock()
        self.mock_experience_store.get_success_count = Mock(return_value=0)
        self.mock_experience_store.get_total_count = Mock(return_value=0)
        
        self.mock_hot_update_engine = Mock()
        self.mock_hot_update_engine.apply_patch = Mock()
        
        # 清理 ToolRegistry 单例污染
        from zulong.tools.base import ToolRegistry
        ToolRegistry._instance = None
        ToolRegistry()._initialized = False
        ToolRegistry().tools = {}
        ToolRegistry().categories = {cat: [] for cat in ToolRegistry().categories}
    
    def test_01_autogpt_planner_full_lifecycle(self):
        """测试 1: AutoGPT Planner 完整生命周期
        
        流程：加载 -> 安装 -> 注册工具 -> 执行任务拆解 -> 卸载
        """
        print("\n[Test 1] AutoGPT Planner 完整生命周期")
        
        from zulong.skill_packs.runtime import SkillPackRuntime
        from zulong.skill_packs.packs.autogpt_planner import AutoGPTPlannerPack
        
        # 创建运行时
        runtime = SkillPackRuntime(
            tool_engine=self.mock_tool_engine,
            experience_store=self.mock_experience_store,
            hot_update_engine=self.mock_hot_update_engine
        )
        
        # 1. 创建技能包
        pack = AutoGPTPlannerPack()
        manifest = pack.get_manifest()
        self.assertEqual(manifest.pack_id, "autogpt_planner")
        print(f"  [PASS] 技能包创建成功: {manifest.name} v{manifest.version}")
        
        # 2. 安装技能包
        install_result = runtime.install_pack(pack, {"max_subtasks": 10})
        self.assertTrue(install_result, "安装失败")
        print(f"  [PASS] 技能包安装成功")
        
        # 3. 验证工具已注册
        tools = pack.get_tools()
        self.assertGreater(len(tools), 0, "技能包没有提供工具")
        print(f"  [PASS] 工具列表: {[t.name for t in tools]}")
        
        # 4. 执行任务拆解
        result = runtime.execute_capability(
            "autogpt_planner",
            "task_decompose",
            {"user_request": "搜索机器人新闻并写总结"}
        )
        self.assertTrue(result.get("success"), f"任务拆解失败: {result}")
        self.assertIn("subtasks", result)
        print(f"  [PASS] 任务拆解成功，子任务数: {len(result.get('subtasks', []))}")
        
        # 5. 检查内化完成度
        internalization = runtime.check_internalization("autogpt_planner")
        self.assertIsInstance(internalization, float)
        print(f"  [PASS] 内化完成度: {internalization:.2f}")
        
        # 6. 卸载技能包
        uninstall_result = runtime.uninstall_pack("autogpt_planner")
        self.assertTrue(uninstall_result, "卸载失败")
        print(f"  [PASS] 技能包卸载成功")
        
        print("  [DONE] AutoGPT Planner 完整生命周期测试通过")
    
    def test_02_openmanus_reasoner_full_lifecycle(self):
        """测试 2: OpenManus Reasoner 完整生命周期
        
        流程：加载 -> 安装 -> 深度推理 -> 验证推理链 -> 卸载
        """
        print("\n[Test 2] OpenManus Reasoner 完整生命周期")
        
        from zulong.skill_packs.runtime import SkillPackRuntime
        from zulong.skill_packs.packs.openmanus_reasoner import OpenManusReasonerPack
        
        # 创建运行时
        runtime = SkillPackRuntime(
            tool_engine=self.mock_tool_engine,
            experience_store=self.mock_experience_store,
            hot_update_engine=self.mock_hot_update_engine
        )
        
        # 1. 创建技能包
        pack = OpenManusReasonerPack()
        manifest = pack.get_manifest()
        self.assertEqual(manifest.pack_id, "openmanus_reasoner")
        print(f"  [PASS] 技能包创建成功: {manifest.name} v{manifest.version}")
        
        # 2. 安装技能包
        install_result = runtime.install_pack(pack, {"reasoning_depth": 3})
        self.assertTrue(install_result, "安装失败")
        print(f"  [PASS] 技能包安装成功")
        
        # 3. 执行深度推理
        result = runtime.execute_capability(
            "openmanus_reasoner",
            "deep_reasoning",
            {
                "problem": "设计一个高效的排序算法",
                "context": "用于大规模数据处理"
            }
        )
        self.assertTrue(result.get("success"), f"推理失败: {result}")
        self.assertIn("reasoning_chain", result)
        self.assertIn("conclusion", result)
        self.assertIn("confidence", result)
        self.assertGreater(len(result["reasoning_chain"]), 0, "推理链为空")
        print(f"  [PASS] 深度推理成功，推理步骤数: {len(result['reasoning_chain'])}")
        print(f"  [INFO] 置信度: {result['confidence']:.2f}")
        
        # 4. 卸载技能包
        uninstall_result = runtime.uninstall_pack("openmanus_reasoner")
        self.assertTrue(uninstall_result, "卸载失败")
        print(f"  [PASS] 技能包卸载成功")
        
        print("  [DONE] OpenManus Reasoner 完整生命周期测试通过")
    
    def test_03_cline_coder_full_lifecycle(self):
        """测试 3: Cline Coder 完整生命周期
        
        流程：加载 -> 安装 -> 文件操作 -> 代码编辑 -> 命令执行 -> 卸载
        """
        print("\n[Test 3] Cline Coder 完整生命周期")
        
        from zulong.skill_packs.runtime import SkillPackRuntime
        from zulong.skill_packs.packs.cline_coder import ClineCoderPack
        
        # 创建临时工作目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建运行时
            runtime = SkillPackRuntime(
                tool_engine=self.mock_tool_engine,
                experience_store=self.mock_experience_store,
                hot_update_engine=self.mock_hot_update_engine
            )
            
            # 1. 创建技能包
            pack = ClineCoderPack()
            manifest = pack.get_manifest()
            self.assertEqual(manifest.pack_id, "cline_coder")
            print(f"  [PASS] 技能包创建成功: {manifest.name} v{manifest.version}")
            
            # 2. 安装技能包
            install_result = runtime.install_pack(pack, {"workspace": temp_dir})
            self.assertTrue(install_result, "安装失败")
            print(f"  [PASS] 技能包安装成功")
            
            # 3. 验证工具已注册
            tools = pack.get_tools()
            self.assertEqual(len(tools), 5, "ClineCoder 应该有 5 个工具")
            tool_names = [t.name for t in tools]
            expected_names = ["read_file", "write_file", "edit_code", "run_command", "search_code"]
            for name in expected_names:
                self.assertIn(name, tool_names, f"缺少工具: {name}")
            print(f"  [PASS] 工具列表: {tool_names}")
            
            # 4. 测试文件写入
            test_file = os.path.join(temp_dir, "test.py")
            write_result = runtime.execute_capability(
                "cline_coder",
                "write_file",
                {
                    "file_path": test_file,
                    "content": "# 测试文件\nprint('Hello, World!')\n",
                    "mode": "write"
                }
            )
            self.assertTrue(write_result.get("success"), f"文件写入失败: {write_result}")
            print(f"  [PASS] 文件写入成功")
            
            # 5. 测试文件读取
            read_result = runtime.execute_capability(
                "cline_coder",
                "read_file",
                {"file_path": test_file}
            )
            self.assertTrue(read_result.get("success"), f"文件读取失败: {read_result}")
            self.assertIn("Hello, World!", read_result.get("data", {}).get("content", ""))
            print(f"  [PASS] 文件读取成功")
            
            # 6. 测试代码编辑
            edit_result = runtime.execute_capability(
                "cline_coder",
                "edit_code",
                {
                    "file_path": test_file,
                    "old_str": "print('Hello, World!')",
                    "new_str": "print('Hello, ZULONG!')"
                }
            )
            self.assertTrue(edit_result.get("success"), f"代码编辑失败: {edit_result}")
            print(f"  [PASS] 代码编辑成功")
            
            # 7. 验证编辑结果
            read_result2 = runtime.execute_capability(
                "cline_coder",
                "read_file",
                {"file_path": test_file}
            )
            self.assertIn("Hello, ZULONG!", read_result2.get("data", {}).get("content", ""))
            print(f"  [PASS] 编辑结果验证成功")
            
            # 8. 测试代码搜索
            search_result = runtime.execute_capability(
                "cline_coder",
                "search_code",
                {
                    "pattern": "Hello",
                    "directory": temp_dir,
                    "file_pattern": "*.py"
                }
            )
            self.assertTrue(search_result.get("success"), f"代码搜索失败: {search_result}")
            print(f"  [PASS] 代码搜索成功")
            
            # 9. 卸载技能包
            uninstall_result = runtime.uninstall_pack("cline_coder")
            self.assertTrue(uninstall_result, "卸载失败")
            print(f"  [PASS] 技能包卸载成功")
        
        print("  [DONE] Cline Coder 完整生命周期测试通过")
    
    def test_04_multiple_packs_coexistence(self):
        """测试 4: 多个技能包共存
        
        验证多个技能包可以同时安装和工作
        """
        print("\n[Test 4] 多个技能包共存")
        
        from zulong.skill_packs.runtime import SkillPackRuntime
        from zulong.skill_packs.packs.autogpt_planner import AutoGPTPlannerPack
        from zulong.skill_packs.packs.openmanus_reasoner import OpenManusReasonerPack
        from zulong.skill_packs.packs.cline_coder import ClineCoderPack
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建运行时
            runtime = SkillPackRuntime(
                tool_engine=self.mock_tool_engine,
                experience_store=self.mock_experience_store,
                hot_update_engine=self.mock_hot_update_engine
            )
            
            # 安装所有技能包
            packs = [
                (AutoGPTPlannerPack(), {"max_subtasks": 10}),
                (OpenManusReasonerPack(), {"reasoning_depth": 3}),
                (ClineCoderPack(), {"workspace": temp_dir})
            ]
            
            for pack, config in packs:
                result = runtime.install_pack(pack, config)
                self.assertTrue(result, f"{pack.get_manifest().pack_id} 安装失败")
            
            print(f"  [PASS] 3 个技能包全部安装成功")
            
            # 验证所有技能包都在运行
            packs_info = runtime.list_packs()
            self.assertEqual(len(packs_info), 3, f"应该有 3 个技能包，实际: {len(packs_info)}")
            
            pack_ids = [p['pack_id'] for p in packs_info]
            self.assertIn('autogpt_planner', pack_ids)
            self.assertIn('openmanus_reasoner', pack_ids)
            self.assertIn('cline_coder', pack_ids)
            print(f"  [PASS] 技能包列表: {pack_ids}")
            
            # 测试各技能包独立工作
            # AutoGPT
            autogpt_result = runtime.execute_capability(
                "autogpt_planner",
                "task_decompose",
                {"user_request": "搜索并总结新闻"}
            )
            self.assertTrue(autogpt_result.get("success"))
            print(f"  [PASS] AutoGPT 独立工作正常")
            
            # OpenManus
            openmanus_result = runtime.execute_capability(
                "openmanus_reasoner",
                "deep_reasoning",
                {"problem": "设计排序算法"}
            )
            self.assertTrue(openmanus_result.get("success"))
            print(f"  [PASS] OpenManus 独立工作正常")
            
            # 逐个卸载
            for pack_id in ['autogpt_planner', 'openmanus_reasoner', 'cline_coder']:
                result = runtime.uninstall_pack(pack_id)
                self.assertTrue(result, f"{pack_id} 卸载失败")
            
            print(f"  [PASS] 3 个技能包全部卸载成功")
        
        print("  [DONE] 多个技能包共存测试通过")
    
    def test_05_loader_from_config(self):
        """测试 5: 从 YAML 配置加载技能包
        
        验证 loader.py 的 load_from_config 功能
        """
        print("\n[Test 5] 从 YAML 配置加载技能包")
        
        import yaml
        import tempfile
        from zulong.skill_packs.runtime import SkillPackRuntime
        from zulong.skill_packs.loader import SkillPackLoader
        
        # 创建临时配置文件
        config_content = {
            'skill_packs': [
                {
                    'pack_id': 'autogpt_planner',
                    'enabled': True,
                    'path': 'zulong.skill_packs.packs.autogpt_planner',
                    'config': {'max_subtasks': 10}
                },
                {
                    'pack_id': 'cline_coder',
                    'enabled': False,  # 禁用
                    'path': 'zulong.skill_packs.packs.cline_coder',
                    'config': {}
                }
            ],
            'internalization': {
                'min_experience_count': 50,
                'min_success_rate': 0.9
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, 'test_skill_packs.yaml')
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_content, f, allow_unicode=True)
            
            # 创建运行时和加载器
            runtime = SkillPackRuntime(
                tool_engine=self.mock_tool_engine,
                experience_store=self.mock_experience_store,
                hot_update_engine=self.mock_hot_update_engine
            )
            loader = SkillPackLoader(runtime)
            
            # 从配置加载
            loaded_count = loader.load_from_config(config_path)
            
            # 应该只加载启用的技能包
            self.assertEqual(loaded_count, 1, f"应该加载 1 个技能包，实际: {loaded_count}")
            print(f"  [PASS] 从配置加载成功，加载了 {loaded_count} 个技能包")
            
            # 验证只有启用的技能包被加载
            packs_info = runtime.list_packs()
            pack_ids = [p['pack_id'] for p in packs_info]
            self.assertIn('autogpt_planner', pack_ids)
            self.assertNotIn('cline_coder', pack_ids)  # 禁用的不应加载
            print(f"  [PASS] 只有启用的技能包被加载: {pack_ids}")
        
        print("  [DONE] 从 YAML 配置加载测试通过")
    
    def test_06_shared_memory_pool_task_queue(self):
        """测试 6: SharedMemoryPool 任务队列
        
        验证任务队列的读写、更新、列表、删除功能
        """
        print("\n[Test 6] SharedMemoryPool 任务队列")
        
        import asyncio
        from zulong.infrastructure.shared_memory_pool import SharedMemoryPool
        
        async def run_test():
            # 获取共享池实例
            pool = await SharedMemoryPool.get_instance()
            
            # 1. 写入任务
            task_data = {
                "subtasks": [
                    {"id": "t1", "description": "搜索新闻"},
                    {"id": "t2", "description": "分析趋势"},
                    {"id": "t3", "description": "写报告"}
                ],
                "dependencies": {"t2": ["t1"], "t3": ["t2"]},
                "parallel_groups": [["t1"], ["t2"], ["t3"]]
            }
            
            trace_id = await pool.write_task_queue("test_task_001", task_data)
            self.assertIsNotNone(trace_id)
            print(f"  [PASS] 任务写入队列: {trace_id}")
            
            # 2. 读取任务
            read_data = await pool.read_task_queue("test_task_001")
            self.assertIsNotNone(read_data)
            self.assertEqual(read_data['task_id'], 'test_task_001')
            self.assertEqual(len(read_data['data']['subtasks']), 3)
            print(f"  [PASS] 任务读取成功，子任务数: {len(read_data['data']['subtasks'])}")
            
            # 3. 更新任务状态
            update_result = await pool.update_task_queue_status(
                "test_task_001",
                "EXECUTING",
                {"t1": "completed"}
            )
            self.assertTrue(update_result)
            print(f"  [PASS] 任务状态更新为 EXECUTING")
            
            # 4. 列出任务队列
            task_list = await pool.list_task_queue()
            self.assertGreater(len(task_list), 0)
            print(f"  [PASS] 任务队列列表: {len(task_list)} 个任务")
            
            # 5. 删除任务
            delete_result = await pool.delete_task_queue_item("test_task_001")
            self.assertTrue(delete_result)
            print(f"  [PASS] 任务删除成功")
            
            # 6. 验证删除
            read_after_delete = await pool.read_task_queue("test_task_001")
            self.assertIsNone(read_after_delete)
            print(f"  [PASS] 删除验证成功")
        
        asyncio.get_event_loop().run_until_complete(run_test())
        print("  [DONE] SharedMemoryPool 任务队列测试通过")
    
    def test_07_tool_registry_integration(self):
        """测试 7: 工具注册表集成
        
        验证技能包的工具能正确注册到 ToolRegistry
        """
        print("\n[Test 7] 工具注册表集成")
        
        from zulong.tools.base import ToolRegistry
        from zulong.skill_packs.packs.autogpt_planner import AutoGPTPlannerPack
        
        # 获取单例
        registry = ToolRegistry()
        initial_count = len(registry.tools)
        
        # 创建并安装技能包
        pack = AutoGPTPlannerPack()
        pack.install(registry, {})
        
        # 验证工具已注册
        tools_after = len(registry.tools)
        self.assertGreater(tools_after, initial_count, "工具数量没有增加")
        print(f"  [PASS] 工具注册成功，新增 {tools_after - initial_count} 个工具")
        
        # 验证可以通过 registry 获取工具
        tools = pack.get_tools()
        for tool in tools:
            registered_tool = registry.get(tool.name)
            self.assertIsNotNone(registered_tool, f"工具 {tool.name} 未在 registry 中找到")
        print(f"  [PASS] 所有工具都可在 registry 中找到")
        
        # 清理
        pack.uninstall()
    
    def test_08_module_router_integration(self):
        """测试 8: ModuleRouter 集成
        
        验证 ModuleRouter 的双层路由功能
        """
        print("\n[Test 8] ModuleRouter 集成")
        
        from zulong.skill_packs.module_router import quick_class, classify_with_timing
        
        # 测试快速分类
        test_cases = [
            ("你好", "chitchat"),
            ("帮我搜索新闻", "task"),
            ("今天天气怎么样", "chitchat"),
            ("搜索并总结机器人新闻", "task"),
        ]
        
        for text, expected_type in test_cases:
            result = quick_class(text)
            self.assertIn(result, ["chitchat", "task"], f"无效的分类结果: {result}")
            print(f"  [INFO] '{text}' -> {result}")
        
        print(f"  [PASS] 快速分类测试通过")
        
        # 测试带计时的分类
        result, timing = classify_with_timing("帮我搜索最新新闻")
        self.assertIn(result, ["chitchat", "task"])
        self.assertIn("first_stage_ms", timing)
        self.assertIn("total_ms", timing)
        print(f"  [PASS] 带计时分类测试通过，耗时: {timing['total_ms']:.2f}ms")
        
        print("  [DONE] ModuleRouter 集成测试通过")


def run_all_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSkillPackEndToEnd)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"总测试数: {result.testsRun}")
    print(f"通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ 所有测试通过！技能包系统端到端验证成功！")
    else:
        print("\n❌ 部分测试失败，请检查错误信息")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
