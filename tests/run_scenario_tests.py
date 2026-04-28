# File: tests/run_scenario_tests.py
# 运行所有场景测试

import sys
from pathlib import Path
import logging

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from tests.scenario_tests import (
    HomeCompanionTest,
    OfficeAssistantTest,
    EducationTutorTest,
    EmergencyResponseTest
)


def main():
    """主函数"""
    print("=" * 70)
    print("  🎬 祖龙 (ZULONG) 场景测试套件")
    print("=" * 70)
    print()
    
    # 创建测试实例
    tests = [
        HomeCompanionTest(),
        OfficeAssistantTest(),
        EducationTutorTest(),
        EmergencyResponseTest()
    ]
    
    results = []
    
    # 运行测试
    for test in tests:
        print()
        test.print_summary()
        success = test.execute()
        results.append((test.name, success))
        print()
    
    # 统计结果
    print("=" * 70)
    print("📊 测试结果总结")
    print("=" * 70)
    
    success_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {name}")
    
    print()
    print(f"📈 成功率：{success_count}/{total_count}")
    print("=" * 70)
    
    if success_count == total_count:
        print("\n🎉 所有场景测试通过！")
        return True
    else:
        print(f"\n⚠️  有 {total_count - success_count} 个场景测试失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
