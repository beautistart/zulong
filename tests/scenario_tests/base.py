# File: tests/scenario_tests/base.py
# 场景测试基类

import time
import logging
from typing import Dict, Any, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ScenarioTest(ABC):
    """场景测试基类"""
    
    def __init__(self, name: str, description: str):
        """初始化
        
        Args:
            name: 场景名称
            description: 场景描述
        """
        self.name = name
        self.description = description
        self.results: List[Dict[str, Any]] = []
        self.start_time = 0.0
        self.end_time = 0.0
    
    @abstractmethod
    def setup(self):
        """设置测试环境"""
        pass
    
    @abstractmethod
    def run(self) -> bool:
        """运行测试
        
        Returns:
            bool: 测试是否通过
        """
        pass
    
    @abstractmethod
    def teardown(self):
        """清理测试环境"""
        pass
    
    def execute(self) -> bool:
        """执行完整测试流程
        
        Returns:
            bool: 测试是否通过
        """
        logger.info(f"🎬 开始场景测试：{self.name}")
        self.start_time = time.time()
        
        try:
            # 设置
            logger.info("📦 设置测试环境...")
            self.setup()
            
            # 运行
            logger.info("🏃 运行测试...")
            result = self.run()
            
            # 清理
            logger.info("🧹 清理测试环境...")
            self.teardown()
            
            self.end_time = time.time()
            elapsed = self.end_time - self.start_time
            
            # 记录结果
            self.results.append({
                "scenario": self.name,
                "success": result,
                "elapsed_time": elapsed,
                "timestamp": time.time()
            })
            
            logger.info(f"{'✅' if result else '❌'} 场景测试完成，耗时：{elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 场景测试失败：{e}")
            import traceback
            traceback.print_exc()
            
            self.end_time = time.time()
            self.results.append({
                "scenario": self.name,
                "success": False,
                "error": str(e),
                "elapsed_time": self.end_time - self.start_time,
                "timestamp": time.time()
            })
            return False
    
    def get_report(self) -> Dict[str, Any]:
        """获取测试报告
        
        Returns:
            Dict[str, Any]: 测试报告
        """
        return {
            "name": self.name,
            "description": self.description,
            "results": self.results,
            "total_time": sum(r.get("elapsed_time", 0) for r in self.results)
        }
    
    def print_summary(self):
        """打印测试摘要"""
        print("\n" + "=" * 70)
        print(f"场景测试：{self.name}")
        print("=" * 70)
        print(f"描述：{self.description}")
        
        if self.results:
            last_result = self.results[-1]
            status = "✅ 通过" if last_result.get("success") else "❌ 失败"
            print(f"状态：{status}")
            print(f"耗时：{last_result.get('elapsed_time', 0):.2f}s")
        
        print("=" * 70 + "\n")
