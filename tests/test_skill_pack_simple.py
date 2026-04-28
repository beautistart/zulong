#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
技能包系统核心流程验证 - 简化版
验证: 安装 -> 执行 -> 卸载 完整生命周期
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_core_flow():
    print("=" * 80)
    print("Skill Pack System Core Flow Verification")
    print("=" * 80)
    
    # Setup mocks
    from unittest.mock import Mock
    mock_tool_engine = Mock()
    mock_tool_engine.registry = Mock()
    mock_tool_engine.registry.tools = {}
    mock_tool_engine.register_tool = Mock()
    
    mock_exp_store = Mock()
    mock_exp_store.get_success_count = Mock(return_value=0)
    mock_exp_store.get_total_count = Mock(return_value=0)
    
    mock_hue = Mock()
    
    # Clean ToolRegistry
    from zulong.tools.base import ToolRegistry
    ToolRegistry._instance = None
    
    print("\n[Test 1] AutoGPT Planner")
    from zulong.skill_packs.runtime import SkillPackRuntime
    from zulong.skill_packs.packs.autogpt_planner import AutoGPTPlannerPack
    
    runtime = SkillPackRuntime(mock_tool_engine, mock_exp_store, mock_hue)
    pack1 = AutoGPTPlannerPack()
    assert runtime.install_pack(pack1, {})
    print("  [OK] Installed")
    
    result = runtime.execute_capability("autogpt_planner", "task_decompose", 
                                       {"user_request": "Search and summarize news"})
    assert result.get("success"), f"Failed: {result}"
    print(f"  [OK] Executed, subtasks: {len(result.get('subtasks', []))}")
    
    assert runtime.uninstall_pack("autogpt_planner")
    print("  [OK] Uninstalled")
    
    print("\n[Test 2] OpenManus Reasoner")
    ToolRegistry._instance = None
    from zulong.skill_packs.packs.openmanus_reasoner import OpenManusReasonerPack
    
    pack2 = OpenManusReasonerPack()
    assert runtime.install_pack(pack2, {})
    print("  [OK] Installed")
    
    result = runtime.execute_capability("openmanus_reasoner", "deep_reasoning",
                                       {"problem": "Design efficient sorting algorithm"})
    assert result.get("success"), f"Failed: {result}"
    print(f"  [OK] Reasoned, steps: {len(result['reasoning_chain'])}, confidence: {result['confidence']:.2f}")
    
    assert runtime.uninstall_pack("openmanus_reasoner")
    print("  [OK] Uninstalled")
    
    print("\n[Test 3] Cline Coder")
    ToolRegistry._instance = None
    from zulong.skill_packs.packs.cline_coder import ClineCoderPack
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pack3 = ClineCoderPack()
        assert runtime.install_pack(pack3, {"workspace": tmpdir})
        print("  [OK] Installed")
        
        # Write file
        test_file = os.path.join(tmpdir, "test.py")
        result = runtime.execute_capability("cline_coder", "write_file",
                                           {"file_path": test_file, "content": "print('hello')\n"})
        assert result.get("success"), f"Write failed: {result}"
        print("  [OK] File written")
        
        # Read file
        result = runtime.execute_capability("cline_coder", "read_file", {"file_path": test_file})
        assert result.get("success"), f"Read failed: {result}"
        assert "hello" in result["data"]["content"]
        print("  [OK] File read")
        
        # Edit code
        result = runtime.execute_capability("cline_coder", "edit_code",
                                           {"file_path": test_file, "old_str": "print('hello')", "new_str": "print('zulong')"})
        assert result.get("success"), f"Edit failed: {result}"
        print("  [OK] Code edited")
        
        # Search
        result = runtime.execute_capability("cline_coder", "search_code",
                                           {"pattern": "zulong", "directory": tmpdir})
        assert result.get("success"), f"Search failed: {result}"
        print("  [OK] Code searched")
        
        assert runtime.uninstall_pack("cline_coder")
        print("  [OK] Uninstalled")
    
    print("\n[Test 4] Loader from YAML config")
    ToolRegistry._instance = None
    import yaml
    from zulong.skill_packs.loader import SkillPackLoader
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'skill_packs': [
                {'pack_id': 'autogpt_planner', 'enabled': True, 
                 'path': 'zulong.skill_packs.packs.autogpt_planner', 'config': {}},
                {'pack_id': 'cline_coder', 'enabled': False,
                 'path': 'zulong.skill_packs.packs.cline_coder', 'config': {}}
            ]
        }
        config_path = os.path.join(tmpdir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        runtime2 = SkillPackRuntime(mock_tool_engine, mock_exp_store, mock_hue)
        loader = SkillPackLoader(runtime2)
        count = loader.load_from_config(config_path)
        assert count == 1, f"Expected 1, got {count}"
        print(f"  [OK] Loaded {count} pack from YAML")
    
    print("\n[Test 5] SharedMemoryPool Task Queue")
    import asyncio
    from zulong.infrastructure.shared_memory_pool import SharedMemoryPool
    
    async def test_pool():
        pool = await SharedMemoryPool.get_instance()
        task_id = "test_001"
        
        # Write
        trace = await pool.write_task_queue(task_id, {"subtasks": [{"id": "t1"}]})
        assert trace
        print("  [OK] Task written")
        
        # Read
        data = await pool.read_task_queue(task_id)
        assert data
        print("  [OK] Task read")
        
        # Update
        assert await pool.update_task_queue_status(task_id, "EXECUTING")
        print("  [OK] Status updated")
        
        # Delete
        assert await pool.delete_task_queue_item(task_id)
        print("  [OK] Task deleted")
    
    asyncio.get_event_loop().run_until_complete(test_pool())
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED - Skill Pack System Core Flow Verified")
    print("=" * 80)

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    test_core_flow()
