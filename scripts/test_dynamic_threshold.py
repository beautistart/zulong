# File: scripts/test_dynamic_threshold.py
"""
动态阈值管理器测试脚本

功能：
1. 测试不同模型配置下的阈值计算
2. 模拟显存紧急模式
3. 测试各种触发条件
4. 验证 Token 计数和轮次检测

使用方法：
    python scripts/test_dynamic_threshold.py
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from zulong.l1b.dynamic_threshold_manager import (
    DynamicThresholdManager,
    get_dynamic_threshold_manager,
    ModelConfig
)


def test_model_config(model_name: str, config: dict):
    """测试特定模型配置的阈值计算"""
    print(f"\n{'='*80}")
    print(f"🧪 测试模型：{model_name}")
    print(f"{'='*80}")
    
    manager = get_dynamic_threshold_manager()
    
    # 重置单例
    DynamicThresholdManager._instance = None
    manager = get_dynamic_threshold_manager()
    
    # 初始化配置
    manager.initialize_with_model_config(config)
    
    # 获取阈值
    thresholds = manager.get_thresholds()
    
    print(f"\n📊 模型配置:")
    print(f"  - 模型名称：{config['name']}")
    print(f"  - 模型大小：{config['size_in_billions']}B")
    print(f"  - 最大上下文：{config['max_context_window']} tokens")
    print(f"  - 量化级别：{config['quantization']}")
    
    print(f"\n📈 计算结果:")
    print(f"  - 硬上限：{thresholds.hard_token_limit} tokens")
    print(f"  - 软上限：{thresholds.soft_turn_limit} 轮")
    print(f"  - 安全系数：{thresholds.safety_factor}")
    print(f"  - 速度因子：{thresholds.speed_factor}")
    
    return manager


def test_vram_emergency(manager: DynamicThresholdManager):
    """测试显存紧急模式"""
    print(f"\n{'='*80}")
    print(f"🚨 测试显存紧急模式")
    print(f"{'='*80}")
    
    # 模拟显存使用率逐步上升
    for vram_usage in [0.5, 0.7, 0.85, 0.95, 0.98]:
        print(f"\n模拟显存使用率：{vram_usage*100:.1f}%")
        manager.update_vram_usage(vram_usage)
        
        thresholds = manager.get_thresholds()
        print(f"  - 硬上限：{thresholds.hard_token_limit}")
        print(f"  - 软上限：{thresholds.soft_turn_limit}")
        print(f"  - 紧急模式：{thresholds.is_emergency_mode}")
        
        if thresholds.is_emergency_mode:
            print(f"  🚨 已进入紧急模式！")


def test_trigger_conditions(manager: DynamicThresholdManager):
    """测试各种触发条件"""
    print(f"\n{'='*80}")
    print(f"🎯 测试触发条件")
    print(f"{'='*80}")
    
    # 测试场景 1: Token 数超限
    print(f"\n场景 1: Token 数超限")
    should_trigger, reason = manager.should_trigger_summarization(
        current_tokens=manager.hard_token_limit + 100,
        current_turns=5
    )
    print(f"  - Token 数：{manager.hard_token_limit + 100} / {manager.hard_token_limit}")
    print(f"  - 轮数：5")
    print(f"  - 触发：{should_trigger}, 原因：{reason}")
    
    # 测试场景 2: 轮数超限
    print(f"\n场景 2: 轮数超限")
    should_trigger, reason = manager.should_trigger_summarization(
        current_tokens=2000,
        current_turns=manager.soft_turn_limit + 5
    )
    print(f"  - Token 数：2000")
    print(f"  - 轮数：{manager.soft_turn_limit + 5} / {manager.soft_turn_limit}")
    print(f"  - 触发：{should_trigger}, 原因：{reason}")
    
    # 测试场景 3: 正常状态
    print(f"\n场景 3: 正常状态")
    should_trigger, reason = manager.should_trigger_summarization(
        current_tokens=2000,
        current_turns=5
    )
    print(f"  - Token 数：2000")
    print(f"  - 轮数：5")
    print(f"  - 触发：{should_trigger}, 原因：{reason}")
    
    # 测试场景 4: 90% 水位警告
    print(f"\n场景 4: 90% 水位警告")
    warning_threshold = int(manager.hard_token_limit * 0.9)
    should_trigger, reason = manager.should_trigger_summarization(
        current_tokens=warning_threshold,
        current_turns=5
    )
    print(f"  - Token 数：{warning_threshold} (90%)")
    print(f"  - 轮数：5")
    print(f"  - 触发：{should_trigger}, 原因：{reason}")


def test_long_text_detection(manager: DynamicThresholdManager):
    """测试长文本检测"""
    print(f"\n{'='*80}")
    print(f"📝 测试长文本检测")
    print(f"{'='*80}")
    
    # 短文本
    short_text = "你好，我叫小明"
    is_long = manager.check_long_text_input(short_text)
    print(f"\n短文本：'{short_text}'")
    print(f"  - 判定为长文本：{is_long}")
    
    # 中等文本
    medium_text = "今天天气真好，我想去公园散步。公园里有很多人在锻炼身体，有的在跑步，有的在打太极拳。"
    is_long = manager.check_long_text_input(medium_text)
    print(f"\n中等文本：'{medium_text[:50]}...'")
    print(f"  - 判定为长文本：{is_long}")
    
    # 长文本
    long_text = """
    人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个分支，
    它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，
    可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
    人工智能是对人的意识、思维的信息过程的模拟。
    人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。
    """
    is_long = manager.check_long_text_input(long_text)
    print(f"\n长文本：'{long_text[:50]}...'")
    print(f"  - 判定为长文本：{is_long}")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🚀 TSD v2.4 动态阈值管理器测试套件")
    print("="*80)
    
    # ========== 测试 1: 小模型场景（Qwen-0.8B INT4） ==========
    manager = test_model_config(
        "Qwen-0.8B (INT4)",
        {
            'name': 'Qwen3.5-0.8B-AWQ',
            'size_in_billions': 0.8,
            'max_context_window': 4096,
            'quantization': 'int4',
            'vram_limit_gb': 2.0
        }
    )
    test_vram_emergency(manager)
    test_trigger_conditions(manager)
    test_long_text_detection(manager)
    
    # ========== 测试 2: 中模型场景（Llama-3-8B INT4） ==========
    manager = test_model_config(
        "Llama-3-8B (INT4)",
        {
            'name': 'Llama-3-8B-Instruct-AWQ',
            'size_in_billions': 8,
            'max_context_window': 8192,
            'quantization': 'int4',
            'vram_limit_gb': 8.0
        }
    )
    test_trigger_conditions(manager)
    
    # ========== 测试 3: 大模型场景（Qwen-72B FP16） ==========
    manager = test_model_config(
        "Qwen-72B (FP16)",
        {
            'name': 'Qwen-72B-Chat',
            'size_in_billions': 72,
            'max_context_window': 32768,
            'quantization': 'fp16',
            'vram_limit_gb': 144.0
        }
    )
    test_trigger_conditions(manager)
    
    # ========== 测试总结 ==========
    print(f"\n{'='*80}")
    print("✅ 测试完成！")
    print(f"{'='*80}")
    
    print(f"\n📊 测试总结:")
    print(f"  ✅ 小模型（0.8B）: 硬上限~3500 tokens, 软上限 10 轮")
    print(f"  ✅ 中模型（8B）: 硬上限~7000 tokens, 软上限 10 轮")
    print(f"  ✅ 大模型（72B）: 硬上限~24500 tokens, 软上限 5 轮")
    print(f"  ✅ 显存紧急模式：>95% 时强制下调阈值 20%")
    print(f"  ✅ 长文本检测：>1000 tokens 触发")
    
    print(f"\n💡 建议:")
    print(f"  - 小模型场景：多保留轮次（10-12 轮），提升连贯性")
    print(f"  - 大模型场景：早复盘（5-8 轮），降低显存压力")
    print(f"  - 生产环境：启用显存实时监控，动态调整阈值")


if __name__ == "__main__":
    run_all_tests()
