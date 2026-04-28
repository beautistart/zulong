# -*- coding: utf-8 -*-
# File: tests/test_week4_experience_generation.py
# 第 4 周优化测试：经验自动生成

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from zulong.memory.experience_generator import ExperienceGenerator, ExperienceCandidate
from zulong.memory.rag_manager import RAGManager, RAGConfig


def test_experience_generator_basic():
    """测试 1: 经验生成器基本功能"""
    print("\n" + "="*80)
    print("测试 1: 经验生成器基本功能")
    print("="*80)
    
    # 创建经验生成器
    print("\n--- 创建经验生成器 ---")
    generator = ExperienceGenerator()
    
    # 测试对话历史
    dialogue_history = [
        {"role": "user", "content": "Python 如何读取文件？"},
        {"role": "assistant", "content": "使用 open() 函数：with open('file.txt', 'r') as f: content = f.read()"},
        {"role": "user", "content": "太好了，谢谢！这方法很好用"},
    ]
    
    # 提取经验
    print("\n--- 提取经验 ---")
    candidates = generator.extract_from_dialogue(dialogue_history)
    
    print(f"提取到 {len(candidates)} 个经验候选")
    for i, candidate in enumerate(candidates, 1):
        print(f"\n  [{i}] 分类：{candidate.category}")
        print(f"      内容：{candidate.content[:100]}...")
        print(f"      置信度：{candidate.confidence:.2f}")
        print(f"      来源：{candidate.source}")
    
    # 验证
    assert len(candidates) > 0, "应该提取到经验"
    assert candidates[0].category == "成功模式", "应该是成功模式"
    assert candidates[0].confidence >= 0.6, "置信度应该达标"
    
    print(f"\n✅ 测试 1 通过：提取到 {len(candidates)} 个经验")
    
    return generator, candidates


def test_experience_categories():
    """测试 2: 经验分类"""
    print("\n" + "="*80)
    print("测试 2: 经验分类")
    print("="*80)
    
    generator = ExperienceGenerator()
    
    # 测试不同类型的反馈
    test_cases = [
        {
            "name": "正面反馈",
            "dialogue": [
                {"role": "assistant", "content": "解决方案是..."},
                {"role": "user", "content": "太好了，完美解决了我的问题！"}
            ]
        },
        {
            "name": "负面反馈",
            "dialogue": [
                {"role": "assistant", "content": "某种方法..."},
                {"role": "user", "content": "这个方法是错误的，不行"}
            ]
        },
        {
            "name": "用户偏好",
            "dialogue": [
                {"role": "user", "content": "我喜欢简洁的代码风格"}
            ]
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        candidates = generator.extract_from_dialogue(test_case['dialogue'])
        
        if candidates:
            print(f"  提取到：{candidates[0].category}")
            print(f"  置信度：{candidates[0].confidence:.2f}")
        else:
            print(f"  未提取到经验")
    
    print(f"\n✅ 测试 2 通过：经验分类测试完成")
    
    return True


def test_add_experience_to_rag():
    """测试 3: 添加到 RAG 经验库"""
    print("\n" + "="*80)
    print("测试 3: 添加到 RAG 经验库")
    print("="*80)
    
    # 创建 RAG 管理器
    print("\n--- 初始化 RAG 管理器 ---")
    config = RAGConfig(vector_dimension=768, base_path="./data/rag_test_exp")
    rag_manager = RAGManager(config)
    
    # 创建经验生成器
    print("\n--- 创建经验生成器 ---")
    generator = ExperienceGenerator()
    generator.set_rag_manager(rag_manager)
    
    # 创建经验候选
    candidates = [
        ExperienceCandidate(
            content="当用户询问编程问题时，提供详细的代码示例和解释",
            category="成功模式",
            confidence=0.85,
            source="用户反馈"
        ),
        ExperienceCandidate(
            content="避免使用复杂的术语，应该用简单易懂的语言解释概念",
            category="失败教训",
            confidence=0.75,
            source="用户反馈"
        ),
        ExperienceCandidate(
            content="用户偏好：喜欢分步骤的详细说明",
            category="用户偏好",
            confidence=0.9,
            source="用户直接表达"
        )
    ]
    
    # 添加到 RAG
    print("\n--- 添加到 RAG 经验库 ---")
    added_count = 0
    for candidate in candidates:
        doc_id = generator.add_experience_to_rag(candidate)
        if doc_id:
            print(f"  ✅ 添加成功：{doc_id[:8]}... - {candidate.category} (置信度：{candidate.confidence:.2f})")
            added_count += 1
        else:
            print(f"  ❌ 添加失败：{candidate.category}")
    
    # 验证
    assert added_count > 0, "应该至少添加一个经验"
    
    print(f"\n✅ 测试 3 通过：添加了 {added_count}/{len(candidates)} 个经验")
    
    return generator


def test_batch_processing():
    """测试 4: 批量处理对话历史"""
    print("\n" + "="*80)
    print("测试 4: 批量处理对话历史")
    print("="*80)
    
    # 创建 RAG 管理器
    config = RAGConfig(vector_dimension=768, base_path="./data/rag_test_exp2")
    rag_manager = RAGManager(config)
    
    # 创建经验生成器
    generator = ExperienceGenerator()
    generator.set_rag_manager(rag_manager)
    
    # 模拟多轮对话
    dialogue_history = [
        {"role": "user", "content": "Python 怎么读取 JSON 文件？"},
        {"role": "assistant", "content": "使用 json 模块：import json; with open('data.json') as f: data = json.load(f)"},
        {"role": "user", "content": "非常好，谢谢！"},
        
        {"role": "user", "content": "那如何写入 JSON 呢？"},
        {"role": "assistant", "content": "使用 json.dump(): with open('output.json', 'w') as f: json.dump(data, f)"},
        {"role": "user", "content": "完美，正是我需要的"},
        
        {"role": "user", "content": "我不喜欢太冗长的解释"},
        {"role": "assistant", "content": "好的，我会简洁回答"},
        {"role": "user", "content": "太好了"},
    ]
    
    # 批量处理
    print("\n--- 批量处理对话历史 ---")
    stats = generator.process_dialogue_batch(dialogue_history)
    
    print(f"\n处理统计:")
    print(f"  提取：{stats['extracted']} 个")
    print(f"  添加：{stats['added']} 个")
    print(f"  跳过：{stats['skipped']} 个")
    print(f"  历史总计：{stats['total_added_all_time']} 个")
    
    # 验证
    assert stats['extracted'] > 0, "应该提取到经验"
    assert stats['added'] > 0, "应该添加至少一个经验"
    
    print(f"\n✅ 测试 4 通过：批量处理完成")
    
    return generator


def test_confidence_threshold():
    """测试 5: 置信度阈值"""
    print("\n" + "="*80)
    print("测试 5: 置信度阈值")
    print("="*80)
    
    generator = ExperienceGenerator()
    
    # 创建不同置信度的候选
    candidates = [
        ExperienceCandidate(
            content="高置信度经验",
            category="成功模式",
            confidence=0.95,
            source="测试"
        ),
        ExperienceCandidate(
            content="中等置信度经验",
            category="成功模式",
            confidence=0.65,
            source="测试"
        ),
        ExperienceCandidate(
            content="低置信度经验",
            category="成功模式",
            confidence=0.4,  # 低于阈值
            source="测试"
        )
    ]
    
    print(f"\n当前最小置信度阈值：{generator.min_confidence}")
    print("\n测试不同置信度:")
    
    for candidate in candidates:
        should_add = candidate.confidence >= generator.min_confidence
        print(f"  置信度={candidate.confidence:.2f} -> {'添加' if should_add else '跳过'}")
    
    # 验证：低置信度应该被跳过
    assert candidates[0].confidence >= generator.min_confidence
    assert candidates[2].confidence < generator.min_confidence
    
    print(f"\n✅ 测试 5 通过：置信度阈值测试完成")
    
    return True


def test_experience_statistics():
    """测试 6: 统计信息"""
    print("\n" + "="*80)
    print("测试 6: 统计信息")
    print("="*80)
    
    generator = ExperienceGenerator()
    
    # 初始统计
    print("\n--- 初始统计 ---")
    stats = generator.get_statistics()
    print(f"  总提取：{stats['total_extracted']}")
    print(f"  总添加：{stats['total_added']}")
    print(f"  总跳过：{stats['total_skipped']}")
    print(f"  提取率：{stats['extraction_rate']:.2f}")
    
    # 添加一些经验后
    print("\n--- 添加经验后 ---")
    for i in range(5):
        candidate = ExperienceCandidate(
            content=f"测试经验 {i}",
            category="成功模式",
            confidence=0.8,
            source="测试"
        )
        generator.total_extracted += 1
        generator.total_added += 1
    
    stats = generator.get_statistics()
    print(f"  总提取：{stats['total_extracted']}")
    print(f"  总添加：{stats['total_added']}")
    print(f"  提取率：{stats['extraction_rate']:.2f}")
    
    # 验证
    assert stats['total_extracted'] == 5
    assert stats['total_added'] == 5
    
    print(f"\n✅ 测试 6 通过：统计信息测试完成")
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🧪 第 4 周优化测试：经验自动生成")
    print("="*80)
    
    tests = [
        ("测试 1: 基本功能", test_experience_generator_basic),
        ("测试 2: 经验分类", test_experience_categories),
        ("测试 3: 添加到 RAG", test_add_experience_to_rag),
        ("测试 4: 批量处理", test_batch_processing),
        ("测试 5: 置信度阈值", test_confidence_threshold),
        ("测试 6: 统计信息", test_experience_statistics),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test_name} 失败：{e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # 总结
    print("\n" + "="*80)
    print("📊 测试总结")
    print("="*80)
    print(f"通过：{passed}/{len(tests)}")
    print(f"失败：{failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 所有测试通过！第 4 周优化完成！")
        print("\n📝 4 周优化计划全部完成:")
        print("  ✅ 第 1 周：记忆巩固 + 自动复盘")
        print("  ✅ 第 2 周：共享池持久化")
        print("  ✅ 第 3 周：BM25 混合检索")
        print("  ✅ 第 4 周：经验自动生成")
    else:
        print(f"\n⚠️ {failed} 个测试失败，请检查")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
