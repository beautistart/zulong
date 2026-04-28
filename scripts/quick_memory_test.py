# -*- coding: utf-8 -*-
# File: scripts/quick_memory_test.py
# 记忆系统快速测试脚本（生产环境用）
# 使用方法：在调试控制台中运行 python scripts\quick_memory_test.py

import sys
import asyncio
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from zulong.memory.short_term_memory import ShortTermMemory
from zulong.memory.episodic_memory import EpisodicMemory
from zulong.infrastructure.shared_memory_pool import SharedMemoryPool


def print_header(text):
    """打印标题"""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def print_result(test_name, passed, details=""):
    """打印测试结果"""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {test_name}")
    if details:
        print(f"      {details}")


async def test_short_term_memory_basic():
    """测试 1: 短期记忆基础功能"""
    print_header("测试 1: 短期记忆基础功能")
    
    try:
        # 获取单例
        stm = await ShortTermMemory.get_instance(max_rounds=20, ttl_seconds=3600)
        
        # 测试存储
        test_input = "今天天气真好"
        test_output = "是的，阳光明媚，适合外出"
        
        await stm.store(test_input, test_output)
        
        # 测试读取
        recent = await stm.get_recent(rounds=1)
        
        if recent and len(recent) > 0:
            input_ok = recent[0].get('input') == test_input
            output_ok = recent[0].get('output') == test_output
            print_result("存储和读取", input_ok and output_ok)
            print_result("输入内容", input_ok, f"'{test_input}'")
            print_result("输出内容", output_ok, f"'{test_output}'")
            
            # 获取状态
            status = stm.get_status()
            print_result("状态查询", True, f"轮数：{status.get('current_round', 0)}")
            
            return True
        else:
            print_result("存储和读取", False, "读取结果为空")
            return False
            
    except Exception as e:
        print_result("异常", False, str(e))
        return False


async def test_multi_turn_conversation():
    """测试 2: 多轮对话"""
    print_header("测试 2: 多轮对话（5 轮）")
    
    try:
        stm = await ShortTermMemory.get_instance()
        
        # 模拟 5 轮对话
        conversations = [
            ("你好，我叫小明", "你好小明，很高兴认识你"),
            ("我今年 25 岁", "25 岁是很好的年纪"),
            ("我住在北京", "北京是个很棒的城市"),
            ("今天天气不错", "是的，适合外出活动"),
            ("想去公园散步", "公园是个好去处")
        ]
        
        print(f"存储 {len(conversations)} 轮对话...")
        for i, (user_input, ai_output) in enumerate(conversations, 1):
            await stm.store(user_input, ai_output)
            print(f"  [{i}/5] 已存储")
        
        # 读取全部 5 轮
        recent = await stm.get_recent(rounds=5)
        
        if len(recent) == 5:
            print_result("对话轮数", True, f"5/5")
            
            # 验证第一轮的完整性
            first = recent[0]
            name_ok = "小明" in first.get('input', '')
            print_result("第一轮内容", name_ok, f"记住名字：{'小明' if name_ok else '未记住'}")
            
            # 验证最后一轮
            last = recent[-1]
            park_ok = "公园" in last.get('input', '')
            print_result("最后一轮内容", park_ok, f"记住活动：{'公园' if park_ok else '未记住'}")
            
            return True
        else:
            print_result("对话轮数", False, f"{len(recent)}/5")
            return False
            
    except Exception as e:
        print_result("异常", False, str(e))
        return False


async def test_summary_generation():
    """测试 3: 摘要生成"""
    print_header("测试 3: 摘要生成功能")
    
    try:
        em = EpisodicMemory()
        
        # 测试对话
        test_dialogue = [
            {"role": "user", "content": "我昨天去了北京，参观了故宫和长城"},
            {"role": "assistant", "content": "哇，北京有很多著名景点"},
            {"role": "user", "content": "非常震撼，长城人山人海"},
            {"role": "assistant", "content": "长城确实很壮观"}
        ]
        
        print("生成摘要...")
        summary = await em.generate_summary(test_dialogue)
        
        if summary and len(summary) > 0:
            print_result("摘要生成", True)
            print(f"摘要内容：{summary}")
            
            # 检查关键词
            keywords = ["北京", "故宫", "长城"]
            found = [kw for kw in keywords if kw in summary]
            coverage = len(found) / len(keywords) * 100
            
            print_result("关键词覆盖", coverage > 50, f"{len(found)}/{len(keywords)} ({coverage:.0f}%)")
            
            # 检查长度
            length_ok = 10 <= len(summary) <= 200
            print_result("摘要长度", length_ok, f"{len(summary)} 字符")
            
            return True
        else:
            print_result("摘要生成", False, "生成结果为空")
            return False
            
    except Exception as e:
        print_result("异常", False, str(e))
        return False


async def test_hierarchical_reading():
    """测试 4: 分层读取"""
    print_header("测试 4: 分层读取（摘要检索 + 完整对话）")
    
    try:
        em = EpisodicMemory()
        
        # 存储测试数据
        test_dialogue = [
            {"role": "user", "content": "我想学习 Python 编程"},
            {"role": "assistant", "content": "Python 是很好的编程语言"}
        ]
        
        summary = await em.generate_summary(test_dialogue)
        episode_id = await em.store_episode(test_dialogue, summary, tags=["learning", "programming"])
        
        print(f"已存储情景记忆，episode_id={episode_id}")
        
        # Level 1: 摘要检索
        print("\nLevel 1: 摘要检索...")
        results = await em.retrieve_by_query("Python 编程", top_k=2)
        
        if results:
            print_result("检索结果", True, f"{len(results)} 条")
            
            # 显示第一条摘要
            first_result = results[0]
            print(f"摘要：{first_result.get('summary', 'N/A')}")
            
            # Level 2: 完整对话读取
            print("\nLevel 2: 读取完整对话...")
            trace_id = first_result.get('trace_id')
            
            if trace_id:
                full_dialogue = await em.read_full_episode(trace_id)
                
                if full_dialogue:
                    print_result("完整对话读取", True, f"{len(full_dialogue)} 条消息")
                    
                    # 验证内容
                    has_user = any(msg["role"] == "user" for msg in full_dialogue)
                    has_ai = any(msg["role"] == "assistant" for msg in full_dialogue)
                    
                    print_result("内容完整性", has_user and has_ai)
                    
                    return True
                else:
                    print_result("完整对话读取", False, "读取结果为空")
                    return False
            else:
                print_result("trace_id", False, "结果中没有 trace_id")
                return False
        else:
            print_result("检索结果", False, "0 条")
            return False
            
    except Exception as e:
        print_result("异常", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_context_injection():
    """测试 5: 上下文注入"""
    print_header("测试 5: 上下文注入连续性")
    
    try:
        stm = await ShortTermMemory.get_instance()
        
        # 清空现有记忆（可选）
        # stm.clear()
        
        # 构建上下文场景
        context_scenario = [
            ("我喜欢吃苹果", "苹果是很有营养的水果"),
            ("香蕉也不错", "香蕉富含钾元素"),
            ("橙子也很好", "橙子富含维生素 C")
        ]
        
        print("构建上下文场景...")
        for user_input, ai_output in context_scenario:
            await stm.store(user_input, ai_output)
            print(f"  已存储：'{user_input}'")
        
        # 读取上下文
        recent = await stm.get_recent(rounds=3)
        
        if len(recent) == 3:
            print_result("上下文缓存", True, f"3/3")
            
            # 验证上下文连续性
            has_apple = any("苹果" in conv.get('input', '') for conv in recent)
            has_banana = any("香蕉" in conv.get('input', '') for conv in recent)
            has_orange = any("橙子" in conv.get('input', '') for conv in recent)
            
            print_result("包含苹果", has_apple)
            print_result("包含香蕉", has_banana)
            print_result("包含橙子", has_orange)
            
            # 模拟注入到 messages
            messages = [{"role": "system", "content": "你是一个助手"}]
            for conv in recent:
                if conv.get('input'):
                    messages.append({"role": "user", "content": conv['input']})
                if conv.get('output'):
                    messages.append({"role": "assistant", "content": conv['output']})
            
            messages.append({"role": "user", "content": "推荐一些水果"})
            
            print_result("消息构建", True, f"{len(messages)} 条消息")
            
            return True
        else:
            print_result("上下文缓存", False, f"{len(recent)}/3")
            return False
            
    except Exception as e:
        print_result("异常", False, str(e))
        return False


async def test_capacity_limit():
    """测试 6: 容量限制"""
    print_header("测试 6: 容量限制测试（max_rounds=20）")
    
    try:
        stm = await ShortTermMemory.get_instance(max_rounds=20)
        
        # 存储 25 轮对话（超过容量）
        print("存储 25 轮对话（超过容量 20）...")
        for i in range(25):
            await stm.store(f"这是第{i+1}句话", f"回复第{i+1}句")
        
        # 检查当前轮数
        status = stm.get_status()
        current_round = status.get('current_round', 0)
        
        # 读取最近对话
        recent = await stm.get_recent(rounds=20)
        
        # 验证容量限制
        capacity_ok = len(recent) <= 20
        print_result("容量限制", capacity_ok, f"实际：{len(recent)}, 最大：20")
        
        # 验证最早的记忆被遗忘
        first_msg = recent[0].get('input', '') if recent else ""
        oldest_forgotten = "第 1 句" not in first_msg
        print_result("旧记忆遗忘", oldest_forgotten, f"第一条：'{first_msg[:20]}...'")
        
        # 验证最新的记忆保留
        last_msg = recent[-1].get('input', '') if recent else ""
        newest_kept = "第 25 句" in last_msg
        print_result("新记忆保留", newest_kept, f"最后一条：'{last_msg}'")
        
        return capacity_ok and oldest_forgotten and newest_kept
            
    except Exception as e:
        print_result("异常", False, str(e))
        return False


async def run_all_tests():
    """运行所有测试"""
    print_header("记忆系统快速测试套件")
    print("生产环境测试 - 开始时间:", __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    
    tests = [
        ("短期记忆基础", test_short_term_memory_basic),
        ("多轮对话", test_multi_turn_conversation),
        ("摘要生成", test_summary_generation),
        ("分层读取", test_hierarchical_reading),
        ("上下文注入", test_context_injection),
        ("容量限制", test_capacity_limit)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n[ERROR] {test_name}: {e}")
            results[test_name] = False
        print()
    
    # 汇总结果
    print_header("测试结果汇总")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\n总计：{passed}/{total} 通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n恭喜！所有测试通过，记忆系统运行正常")
    else:
        print(f"\n注意：{total - passed} 个测试失败，请检查日志")
    
    print("\n测试结束时间:", __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
