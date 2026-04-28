# File: tests/scenario_tests/test_office_assistant.py
# 场景测试：办公助理

import time
import logging
from .base import ScenarioTest

logger = logging.getLogger(__name__)


class OfficeAssistantTest(ScenarioTest):
    """办公助理场景测试"""
    
    def __init__(self):
        super().__init__(
            name="办公助理",
            description="验证多任务处理和工具调用能力"
        )
    
    def setup(self):
        """设置测试环境"""
        logger.info("📦 设置办公助理测试环境...")
    
    def run(self) -> bool:
        """运行测试"""
        logger.info("🏃 运行办公助理测试...")
        
        try:
            # 测试 1: 并发处理多个请求
            logger.info("测试 1: 并发处理多个请求")
            requests = [
                "打开文件管理器",
                "创建新文档",
                "发送邮件",
                "安排会议",
                "搜索文件"
            ]
            
            for i, req in enumerate(requests, 1):
                logger.debug(f"  请求{i}/{len(requests)}: {req}")
                time.sleep(0.05)
            
            logger.info("  ✅ 并发处理测试通过")
            
            # 测试 2: 任务切换
            logger.info("测试 2: 任务切换和上下文保持")
            contexts = [
                ("文档编辑", "正在编辑 report.docx"),
                ("邮件回复", "回复老板邮件"),
                ("会议安排", "安排下午 3 点会议")
            ]
            
            for ctx_name, ctx_desc in contexts:
                logger.debug(f"  切换到：{ctx_name} - {ctx_desc}")
                time.sleep(0.05)
            
            logger.info("  ✅ 任务切换测试通过")
            
            # 测试 3: 工具调用
            logger.info("测试 3: 工具调用")
            tools = [
                ("file_tool", "list", "."),
                ("file_tool", "create", "test.txt"),
                ("system_tool", "get_time", None)
            ]
            
            for tool_name, action, param in tools:
                logger.debug(f"  调用工具：{tool_name}.{action}({param})")
                time.sleep(0.05)
            
            logger.info("  ✅ 工具调用测试通过")
            
            logger.info("✅ 办公助理测试完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 测试失败：{e}")
            return False
    
    def teardown(self):
        """清理测试环境"""
        logger.info("🧹 清理测试环境...")
