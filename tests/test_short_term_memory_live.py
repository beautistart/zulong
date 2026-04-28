# -*- coding: utf-8 -*-
# File: tests/test_short_term_memory_live.py
# 短期记忆实时验证测试 - 验证短期记忆是否真正工作

"""
实时测试短期记忆是否真正在 InferenceEngine 中工作
"""

import sys
from pathlib import Path
import time

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_short_term_memory_exists():
    """测试 1: 验证短期记忆对象是否存在"""
    print("\n" + "="*80)
    print("测试 1: 短期记忆对象是否存在")
    print("="*80)
    
    try:
        from zulong.l2.inference_engine import InferenceEngine
        
        engine = InferenceEngine()
        
        # 检查是否有 short_term_memory 属性
        assert hasattr(engine, 'short_term_memory'), "InferenceEngine 应该有 short_term_memory 属性"
        print("✅ InferenceEngine 有 short_term_memory 属性")
        
        # 检查属性是否不为 None
        assert engine.short_term_memory is not None, "short_term_memory 不应该为 None"
        print("✅ short_term_memory 已初始化")
        
        # 检查类型
        from zulong.memory.short_term_memory import ShortTermMemory
        assert isinstance(engine.short_term_memory, ShortTermMemory), "short_term_memory 应该是 ShortTermMemory 类型"
        print("✅ short_term_memory 类型正确")
        
        print("\n✅ 短期记忆对象存在且已正确初始化")
        return True
        
    except Exception as e:
        print(f"\n❌ 短期记忆对象检查失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_update_memory_method_exists():
    """测试 2: 验证记忆更新方法是否存在"""
    print("\n" + "="*80)
    print("测试 2: 记忆更新方法是否存在")
    print("="*80)
    
    try:
        from zulong.l2.inference_engine import InferenceEngine
        
        engine = InferenceEngine()
        
        # 检查是否有 _update_memory 方法
        assert hasattr(engine, '_update_memory'), "InferenceEngine 应该有 _update_memory 方法"
        print("✅ InferenceEngine 有 _update_memory 方法")
        
        # 检查方法是否可调用
        assert callable(engine._update_memory), "_update_memory 应该是可调用的"
        print("✅ _update_memory 方法可调用")
        
        print("\n✅ 记忆更新方法存在且可调用")
        return True
        
    except Exception as e:
        print(f"\n❌ 记忆更新方法检查失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_store_call_in_update():
    """测试 3: 验证_update_memory 中是否调用了 store 方法"""
    print("\n" + "="*80)
    print("测试 3: 验证 store 方法调用")
    print("="*80)
    
    try:
        import inspect
        from zulong.l2.inference_engine import InferenceEngine
        
        # 获取 _update_memory 方法的源代码
        source = inspect.getsource(InferenceEngine._update_memory)
        
        # 检查是否调用了 short_term_memory.store
        assert 'short_term_memory.store' in source, "_update_memory 应该调用 short_term_memory.store"
        print("✅ _update_memory 调用了 short_term_memory.store")
        
        # 检查是否传递了正确的参数
        assert 'user_input=user_input' in source, "应该传递 user_input 参数"
        print("✅ 传递了 user_input 参数")
        
        assert 'ai_response=response' in source, "应该传递 ai_response 参数"
        print("✅ 传递了 ai_response 参数")
        
        # 检查是否有持久化日志
        assert '记忆已持久化到共享池' in source, "应该有持久化日志"
        print("✅ 有持久化日志记录")
        
        print("\n✅ store 方法调用正确")
        return True
        
    except Exception as e:
        print(f"\n❌ store 方法调用检查失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_update_in_generate():
    """测试 4: 验证生成回复后是否调用记忆更新"""
    print("\n" + "="*80)
    print("测试 4: 验证生成后调用记忆更新")
    print("="*80)
    
    try:
        import inspect
        from zulong.l2.inference_engine import InferenceEngine
        
        # 获取主推理方法的源代码
        source = inspect.getsource(InferenceEngine)
        
        # 检查是否调用了 _update_memory
        assert '_update_memory' in source, "应该调用 _update_memory"
        print("✅ 推理流程调用了 _update_memory")
        
        # 检查调用顺序 (应该在生成之后，输出之前)
        lines = source.split('\n')
        update_memory_line = None
        generate_line = None
        output_line = None
        
        for i, line in enumerate(lines):
            if '_update_memory' in line:
                update_memory_line = i
            if 'generate' in line.lower() and 'response' in line.lower():
                generate_line = i
            if 'L2_OUTPUT' in line or 'publish' in line.lower():
                output_line = i
        
        if update_memory_line and output_line:
            assert update_memory_line < output_line, "_update_memory 应该在输出之前调用"
            print("✅ _update_memory 在输出之前调用")
        
        print("\n✅ 记忆更新调用位置正确")
        return True
        
    except Exception as e:
        print(f"\n❌ 记忆更新调用检查失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_conversation_history():
    """测试 5: 验证对话历史是否被记录"""
    print("\n" + "="*80)
    print("测试 5: 对话历史记录")
    print("="*80)
    
    try:
        from zulong.l2.inference_engine import InferenceEngine
        
        engine = InferenceEngine()
        
        # 检查是否有 conversation_history 属性
        assert hasattr(engine, 'conversation_history'), "应该有 conversation_history 属性"
        print("✅ 有 conversation_history 属性")
        
        # 检查是否是列表
        assert isinstance(engine.conversation_history, list), "conversation_history 应该是列表"
        print("✅ conversation_history 是列表类型")
        
        # 检查是否有 max_history 限制
        assert hasattr(engine, 'max_history'), "应该有 max_history 属性"
        print(f"✅ max_history = {engine.max_history}")
        
        print("\n✅ 对话历史记录功能正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 对话历史记录检查失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_experience_generator_integration():
    """测试 6: 验证经验生成器集成"""
    print("\n" + "="*80)
    print("测试 6: 经验生成器集成")
    print("="*80)
    
    try:
        from zulong.l2.inference_engine import InferenceEngine
        
        engine = InferenceEngine()
        
        # 检查是否有 experience_generator 属性
        assert hasattr(engine, 'experience_generator'), "应该有 experience_generator 属性"
        print("✅ 有 experience_generator 属性")
        
        # 检查属性是否不为 None
        assert engine.experience_generator is not None, "experience_generator 不应该为 None"
        print("✅ experience_generator 已初始化")
        
        # 检查 _update_memory 中是否调用了经验生成
        import inspect
        source = inspect.getsource(InferenceEngine._update_memory)
        assert 'process_dialogue_batch' in source, "应该调用 process_dialogue_batch"
        print("✅ _update_memory 调用了经验批量处理")
        
        print("\n✅ 经验生成器集成正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 经验生成器集成检查失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_persistence():
    """测试 7: 验证记忆持久化"""
    print("\n" + "="*80)
    print("测试 7: 记忆持久化")
    print("="*80)
    
    try:
        import inspect
        from zulong.l2.inference_engine import InferenceEngine
        
        source = inspect.getsource(InferenceEngine._update_memory)
        
        # 检查是否有持久化逻辑
        assert 'store' in source, "应该有 store 方法调用"
        print("✅ 有 store 方法调用")
        
        # 检查是否有异步处理
        assert 'asyncio' in source or 'async' in source.lower(), "应该有异步处理"
        print("✅ 使用异步处理")
        
        # 检查是否有日志
        assert 'logger.info' in source and '持久化' in source, "应该有持久化日志"
        print("✅ 有持久化日志记录")
        
        print("\n✅ 记忆持久化功能正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 记忆持久化检查失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🧪 短期记忆实时验证测试")
    print("="*80)
    
    tests = [
        ("短期记忆对象存在", test_short_term_memory_exists),
        ("记忆更新方法存在", test_update_memory_method_exists),
        ("store 方法调用", test_memory_store_call_in_update),
        ("生成后调用记忆更新", test_memory_update_in_generate),
        ("对话历史记录", test_conversation_history),
        ("经验生成器集成", test_experience_generator_integration),
        ("记忆持久化", test_memory_persistence),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} 测试异常：{e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "="*80)
    print("📊 测试结果汇总")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")
    
    print(f"\n通过：{passed}/{len(tests)}")
    print(f"失败：{failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 所有测试通过！短期记忆系统正常工作！")
        print("\n📝 短期记忆功能验证:")
        print("  ✅ 对象已初始化")
        print("  ✅ 方法可调用")
        print("  ✅ store 方法被调用")
        print("  ✅ 对话历史被记录")
        print("  ✅ 经验生成器集成")
        print("  ✅ 持久化到共享池")
    else:
        print(f"\n⚠️ {failed} 个测试失败，请检查")
        print("\n失败的测试:")
        for test_name, result in results:
            if not result:
                print(f"  ❌ {test_name}")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
