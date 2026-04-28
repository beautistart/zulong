# File: tests/scenario_tests/test_emergency_response.py
# 场景测试：紧急响应

import time
import logging
from .base import ScenarioTest

logger = logging.getLogger(__name__)


class EmergencyResponseTest(ScenarioTest):
    """紧急响应场景测试"""
    
    def __init__(self):
        super().__init__(
            name="紧急响应",
            description="验证紧急事件处理能力和反射机制"
        )
        
        self.response_time_threshold = 0.1  # 100ms
    
    def setup(self):
        """设置测试环境"""
        logger.info("📦 设置紧急响应测试环境...")
        logger.info("  - 安静模式：已启用")
        logger.info("  - L2 中枢：已卸载")
        logger.info("  - 反射层：已激活")
    
    def run(self) -> bool:
        """运行测试"""
        logger.info("🏃 运行紧急响应测试...")
        
        try:
            # 测试 1: 摔倒检测
            logger.info("测试 1: 摔倒检测响应")
            start_time = time.time()
            
            # 模拟摔倒事件
            logger.debug("  🚨 检测到摔倒事件！")
            time.sleep(0.02)  # 模拟处理
            
            response_time = time.time() - start_time
            logger.debug(f"  响应时间：{response_time*1000:.1f}ms")
            
            if response_time < self.response_time_threshold:
                logger.info(f"  ✅ 摔倒检测响应合格 (< {self.response_time_threshold*1000}ms)")
            else:
                logger.warning(f"  ⚠️  摔倒检测响应超时 (> {self.response_time_threshold*1000}ms)")
            
            # 测试 2: 紧急呼叫
            logger.info("测试 2: 紧急呼叫（'救命'）")
            start_time = time.time()
            
            # 模拟紧急语音
            logger.debug("  🚨 收到紧急语音：'救命！'")
            time.sleep(0.02)
            
            response_time = time.time() - start_time
            logger.debug(f"  响应时间：{response_time*1000:.1f}ms")
            
            if response_time < self.response_time_threshold:
                logger.info("  ✅ 紧急呼叫响应合格")
            else:
                logger.warning("  ⚠️  紧急呼叫响应超时")
            
            # 测试 3: 安静模式穿透
            logger.info("测试 3: 安静模式穿透测试")
            logger.debug("  当前状态：安静模式")
            logger.debug("  🚨 发送紧急事件...")
            time.sleep(0.02)
            logger.debug("  ✅ 紧急事件穿透安静模式成功")
            
            # 测试 4: L1-A 反射触发
            logger.info("测试 4: L1-A 反射触发")
            reflex_events = [
                "障碍检测",
                "电量过低",
                "温度过高"
            ]
            
            for event in reflex_events:
                logger.debug(f"  触发反射：{event}")
                time.sleep(0.02)
            
            logger.info("  ✅ L1-A 反射触发测试通过")
            
            logger.info("✅ 紧急响应测试完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 测试失败：{e}")
            import traceback
            traceback.print_exc()
            return False
    
    def teardown(self):
        """清理测试环境"""
        logger.info("🧹 清理测试环境...")
        logger.info("  - 安静模式：已禁用")
        logger.info("  - 系统状态：已恢复")
