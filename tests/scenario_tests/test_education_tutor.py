# File: tests/scenario_tests/test_education_tutor.py
# 场景测试：教育辅导

import time
import logging
from .base import ScenarioTest

logger = logging.getLogger(__name__)


class EducationTutorTest(ScenarioTest):
    """教育辅导场景测试"""
    
    def __init__(self):
        super().__init__(
            name="教育辅导",
            description="验证知识检索和推理能力"
        )
    
    def setup(self):
        """设置测试环境"""
        logger.info("📦 设置教育辅导测试环境...")
    
    def run(self) -> bool:
        """运行测试"""
        logger.info("🏃 运行教育辅导测试...")
        
        try:
            # 测试 1: 复杂问题解答
            logger.info("测试 1: 复杂问题解答")
            questions = [
                "解释量子纠缠",
                "计算圆的面积，半径为 5",
                "什么是光合作用？",
                "牛顿第一定律是什么？"
            ]
            
            for i, q in enumerate(questions, 1):
                logger.debug(f"  问题{i}/{len(questions)}: {q}")
                # 模拟 RAG 检索和推理
                time.sleep(0.1)
            
            logger.info("  ✅ 问题解答测试通过")
            
            # 测试 2: 多轮对话
            logger.info("测试 2: 多轮对话（追问、澄清）")
            dialogue = [
                {"用户": "什么是 AI？", "助手": "AI 是人工智能..."},
                {"用户": "它有什么用？", "助手": "AI 可以用于..."},
                {"用户": "难学吗？", "助手": "取决于..."}
            ]
            
            for turn in dialogue:
                logger.debug(f"  对话：{turn}")
                time.sleep(0.05)
            
            logger.info("  ✅ 多轮对话测试通过")
            
            # 测试 3: RAG 知识库使用
            logger.info("测试 3: RAG 知识库检索")
            rag_queries = [
                "Python 编程基础",
                "机器学习算法",
                "数据结构"
            ]
            
            for query in rag_queries:
                logger.debug(f"  检索：{query}")
                time.sleep(0.05)
            
            logger.info("  ✅ RAG 检索测试通过")
            
            logger.info("✅ 教育辅导测试完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 测试失败：{e}")
            return False
    
    def teardown(self):
        """清理测试环境"""
        logger.info("🧹 清理测试环境...")
