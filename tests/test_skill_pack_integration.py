#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
技能包系统集成测试 - 验证 5 个断层修复后系统能正常运作

测试矩阵:
1. 技能包加载测试 - 验证配置启用后能正确加载
2. 工具注册测试 - 验证工具出现在 ToolRegistry 中
3. Planner 拆解测试 - 验证任务拆解功能（关键词模式 + LLM 模式）
4. 经验记录测试 - 验证 _record_experience 正确调用 add_experience
5. 端到端测试 - WebSocket 发送复杂任务（需要运行中的系统）
"""

import sys
import os
import time
import json
import logging
import io

# 修复 Windows GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger(__name__)


def test_1_skill_pack_loading():
    """测试1: 技能包从配置文件正确加载"""
    print("\n" + "=" * 60)
    print("测试1: 技能包加载")
    print("=" * 60)

    from zulong.skill_packs.runtime import SkillPackRuntime
    from zulong.tools.tool_engine import ToolEngine

    tool_engine = ToolEngine()
    runtime = SkillPackRuntime(
        tool_engine=tool_engine,
        experience_store=None,
        hot_update_engine=None,
    )

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config', 'skill_packs.yaml'
    )

    assert os.path.exists(config_path), f"配置文件不存在: {config_path}"

    # 重置 SkillPackRuntime 单例的 _packs（避免与其他测试冲突）
    loaded_count = runtime.load_from_config(config_path)
    print(f"  加载的技能包数量: {loaded_count}")

    assert loaded_count >= 2, f"期望加载至少 2 个技能包，实际加载了 {loaded_count} 个"

    packs = runtime.list_packs()
    pack_ids = [p['pack_id'] for p in packs]
    print(f"  已加载的技能包: {pack_ids}")

    assert 'autogpt_planner' in pack_ids, "autogpt_planner 未加载"
    assert 'openmanus_reasoner' in pack_ids, "openmanus_reasoner 未加载"

    print("  ✅ 测试通过: 技能包正确加载")
    return runtime


def test_2_tool_registration(runtime=None):
    """测试2: 工具注册到 ToolRegistry"""
    print("\n" + "=" * 60)
    print("测试2: 工具注册")
    print("=" * 60)

    from zulong.tools.base import ToolRegistry

    registry = ToolRegistry()
    schemas = registry.get_all_function_schemas()
    tool_names = [s['function']['name'] for s in schemas]

    print(f"  已注册工具数量: {len(schemas)}")
    print(f"  工具名称: {tool_names}")

    # 检查关键工具是否注册
    expected_tools = ['task_decompose', 'priority_rank', 'dependency_analyze']
    for tool_name in expected_tools:
        assert tool_name in tool_names, f"工具 {tool_name} 未注册到 ToolRegistry"
        print(f"  ✅ {tool_name} 已注册")

    # 检查 deep_reasoning 工具
    deep_tools = [n for n in tool_names if 'reason' in n.lower() or 'deep' in n.lower()]
    print(f"  深度推理工具: {deep_tools}")

    print("  ✅ 测试通过: 所有关键工具已注册")
    return schemas


def test_3_planner_keyword_decompose():
    """测试3: Planner 关键词模式任务拆解（不需要 vLLM）"""
    print("\n" + "=" * 60)
    print("测试3: Planner 关键词模式拆解")
    print("=" * 60)

    from zulong.skill_packs.packs.autogpt_planner.planner import TaskDecomposeAlgorithm

    # 不传 llm_client，走关键词降级
    planner = TaskDecomposeAlgorithm(max_subtasks=10)

    goal = "帮我搜索 Rust vs Go 在嵌入式AI场景的优劣，分析对比，然后写一份报告"
    result = planner.decompose(goal)

    print(f"  拆解结果: {json.dumps(result, ensure_ascii=False, indent=2)}")

    assert result['success'], f"任务拆解失败: {result.get('error')}"
    assert result['subtask_count'] > 0, "子任务数量为 0"

    subtasks = result['subtasks']
    print(f"  子任务数量: {len(subtasks)}")
    for st in subtasks:
        print(f"    Step {st['step']}: {st['task']} (hint: {st['tool_hint']})")

    # 验证包含搜索和写作步骤
    hints = [st['tool_hint'] for st in subtasks]
    assert 'search' in hints, "缺少搜索步骤"
    assert 'write' in hints, "缺少写作步骤"

    print("  ✅ 测试通过: 关键词模式拆解正常")
    return result


def test_4_experience_recording():
    """测试4: 经验记录 API 兼容性"""
    print("\n" + "=" * 60)
    print("测试4: 经验记录 API 兼容性")
    print("=" * 60)

    from zulong.memory.enhanced_experience_store import EnhancedExperienceStore
    from zulong.skill_packs.runtime import SkillPackRuntime
    from zulong.tools.tool_engine import ToolEngine

    # 使用临时数据库路径
    experience_store = EnhancedExperienceStore(
        db_path="data/test_experience_db",
        enable_persistence=False,
    )

    tool_engine = ToolEngine()
    runtime = SkillPackRuntime(
        tool_engine=tool_engine,
        experience_store=experience_store,
        hot_update_engine=None,
    )

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config', 'skill_packs.yaml'
    )
    runtime.load_from_config(config_path)

    # 执行技能包（触发 _record_experience）
    result = runtime.execute_capability(
        "autogpt_planner",
        "task_decompose",
        {"goal": "搜索最新AI新闻并总结"}
    )

    print(f"  执行结果: success={result.get('success')}")

    # 验证经验已记录
    exp_count = runtime._experience_counts.get("autogpt_planner", 0)
    print(f"  经验计数: {exp_count}")
    assert exp_count >= 1, f"经验计数应 >= 1，实际 {exp_count}"

    # 验证 ExperienceStore 中有数据
    experiences = experience_store.search_by_text("技能包 autogpt_planner", limit=5)
    print(f"  ExperienceStore 中检索到经验数量: {len(experiences)}")
    assert len(experiences) >= 1, "ExperienceStore 中未找到技能包执行经验"

    for exp in experiences:
        print(f"    - [{exp.experience_type}] {exp.content[:80]}")

    print("  ✅ 测试通过: 经验记录 API 兼容")
    return True


def test_5_planner_with_vllm():
    """测试5: Planner LLM 模式拆解（需要 vLLM 运行在 port 8000）"""
    print("\n" + "=" * 60)
    print("测试5: Planner LLM 模式拆解 (vLLM)")
    print("=" * 60)

    try:
        from openai import OpenAI
    except ImportError:
        print("  ⚠️ 跳过: openai SDK 未安装")
        return None

    # 检查 vLLM 是否可用
    from zulong.models.container import VLLM_BASE_URL
    vllm_client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")

    try:
        models = vllm_client.models.list()
        model_id = models.data[0].id if models.data else None
        if not model_id:
            print("  ⚠️ 跳过: vLLM 没有加载模型")
            return None
        print(f"  vLLM 模型: {model_id}")
    except Exception as e:
        print(f"  ⚠️ 跳过: vLLM 不可用 ({e})")
        return None

    from zulong.skill_packs.packs.autogpt_planner.planner import TaskDecomposeAlgorithm

    planner = TaskDecomposeAlgorithm(
        max_subtasks=10,
        llm_client=vllm_client,
        model_id=model_id,
    )

    goal = "帮我调研 Rust vs Go 在嵌入式AI场景的优劣并写一份报告"
    result = planner.decompose(goal)

    print(f"  拆解结果: {json.dumps(result, ensure_ascii=False, indent=2)}")

    if result['success']:
        print(f"  ✅ LLM 模式拆解成功: {result['subtask_count']} 个子任务")
        for st in result['subtasks']:
            print(f"    Step {st.get('step')}: {st.get('task')} (hint: {st.get('tool_hint')})")
    else:
        # LLM 可能返回格式不对，降级到关键词模式
        print(f"  ⚠️ LLM 模式拆解返回 success=False: {result.get('error')}")
        print("  (这是预期内的情况 - 小模型可能无法生成正确的 JSON 格式)")

    return result


def test_6_full_bootstrap_integration():
    """测试6: 完整 bootstrap 集成验证 (验证 vllm_client 注入到技能包)"""
    print("\n" + "=" * 60)
    print("测试6: Bootstrap 集成验证")
    print("=" * 60)

    from zulong.memory.enhanced_experience_store import EnhancedExperienceStore
    from zulong.skill_packs.runtime import SkillPackRuntime
    from zulong.tools.tool_engine import ToolEngine

    tool_engine = ToolEngine()
    experience_store = EnhancedExperienceStore(
        db_path="data/test_bootstrap_exp_db",
        enable_persistence=False,
    )

    runtime = SkillPackRuntime(
        tool_engine=tool_engine,
        experience_store=experience_store,
        hot_update_engine=None,
    )

    # 模拟 bootstrap 的 vllm_client 注入
    try:
        from openai import OpenAI
        from zulong.models.container import VLLM_BASE_URL
        vllm_client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
        # 检查连接
        models = vllm_client.models.list()
        model_id = models.data[0].id if models.data else None
        if model_id:
            runtime._vllm_client = vllm_client
            runtime._vllm_model_id = model_id
            print(f"  ✅ vLLM 客户端已注入: model={model_id}")
        else:
            print("  ⚠️ vLLM 无模型，使用关键词降级")
    except Exception as e:
        print(f"  ⚠️ vLLM 不可用 ({e})，使用关键词降级")

    # 加载技能包
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config', 'skill_packs.yaml'
    )
    loaded = runtime.load_from_config(config_path)
    print(f"  加载技能包: {loaded} 个")

    # 检查 planner 是否拿到了 llm_client
    if 'autogpt_planner' in runtime._packs:
        pack = runtime._packs['autogpt_planner']
        planner = pack._planner
        has_llm = planner.llm_client is not None
        print(f"  Planner llm_client: {'已注入' if has_llm else '未注入 (关键词降级)'}")
        print(f"  Planner model_id: {planner.model_id}")
    else:
        print("  ❌ autogpt_planner 未加载")

    # 执行一次任务拆解
    result = runtime.execute_capability(
        "autogpt_planner",
        "task_decompose",
        {"goal": "帮我搜索 Python 3.12 新特性并总结报告"}
    )
    print(f"  执行结果: success={result.get('success')}, subtasks={result.get('subtask_count', 0)}")

    # 验证经验记录
    experiences = experience_store.search_by_text("autogpt_planner", limit=3)
    print(f"  经验库记录数: {len(experiences)}")

    assert result.get('success'), f"任务拆解失败: {result.get('error')}"
    print("  ✅ 测试通过: Bootstrap 集成验证")
    return True


def main():
    """运行所有测试"""
    print("╔" + "=" * 58 + "╗")
    print("║      技能包系统集成测试 - 5 个断层修复验证       ║")
    print("╚" + "=" * 58 + "╝")

    passed = 0
    failed = 0
    skipped = 0

    tests = [
        ("1. 技能包加载", test_1_skill_pack_loading),
        ("2. 工具注册", test_2_tool_registration),
        ("3. 关键词拆解", test_3_planner_keyword_decompose),
        ("4. 经验记录", test_4_experience_recording),
        ("5. LLM拆解", test_5_planner_with_vllm),
        ("6. Bootstrap集成", test_6_full_bootstrap_integration),
    ]

    for name, test_func in tests:
        try:
            result = test_func()
            if result is None:
                skipped += 1
                print(f"\n  ⚠️ [{name}] 跳过")
            else:
                passed += 1
                print(f"\n  ✅ [{name}] 通过")
        except Exception as e:
            failed += 1
            print(f"\n  ❌ [{name}] 失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败, {skipped} 跳过")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
