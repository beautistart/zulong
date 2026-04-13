# File: zulong/l3/base_expert_node.py
# L3 专家节点基类 - 定义所有专家节点的公共接口

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)


class ExpertExecutionError(Exception):
    """专家执行异常"""
    pass


class BaseExpertNode(ABC):
    """专家节点基类
    
    所有专家节点必须继承此类并实现抽象方法。
    
    TSD v1.7 对应规则:
    - 2.2.4 L3: 专家技能池 - 提供专用领域能力
    - 3.2 智能路由逻辑 - 专家调用协议
    """
    
    def __init__(self, expert_type: str):
        """初始化专家节点
        
        Args:
            expert_type: 专家类型标识 (如 "EXPERT_NAV", "EXPERT_VISION")
        """
        self.expert_type = expert_type
        self.is_loaded = False
        self.last_access_time = time.time()
        logger.info(f"[BaseExpertNode] 初始化专家节点：{expert_type}")
    
    @abstractmethod
    def execute(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """执行专家任务（抽象方法，子类必须实现）
        
        Args:
            task_payload: 任务载荷，包含任务描述、上下文等信息
            
        Returns:
            Dict[str, Any]: 执行结果，必须包含 status 字段
            
        Raises:
            ExpertExecutionError: 当专家执行失败时
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """获取专家能力描述（抽象方法，子类必须实现）
        
        Returns:
            Dict[str, Any]: 能力描述字典
        """
        pass
    
    def validate_payload(self, task_payload: Dict[str, Any]) -> bool:
        """验证任务载荷（可选重写）
        
        Args:
            task_payload: 任务载荷
            
        Returns:
            bool: 载荷是否有效
        """
        # 基础验证：必须包含 task_description
        if not isinstance(task_payload, dict):
            logger.error(f"[{self.expert_type}] 任务载荷必须是字典")
            return False
        
        if "task_description" not in task_payload:
            logger.error(f"[{self.expert_type}] 任务载荷缺少 task_description 字段")
            return False
        
        return True
    
    def run(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """运行专家节点（模板方法）
        
        此方法实现了标准的专家执行流程：
        1. 验证载荷
        2. 更新访问时间
        3. 执行专家任务（带异常捕获）
        4. 返回标准化结果
        
        Args:
            task_payload: 任务载荷
            
        Returns:
            Dict[str, Any]: 标准化执行结果
        """
        # 步骤 1: 验证载荷
        if not self.validate_payload(task_payload):
            return {
                "status": "error",
                "error_message": "Invalid task payload",
                "expert_type": self.expert_type
            }
        
        # 步骤 2: 更新访问时间
        self.last_access_time = time.time()
        
        # 步骤 3: 执行专家任务（带异常捕获）
        try:
            logger.info(f"[{self.expert_type}] 开始执行任务：{task_payload.get('task_description', 'Unknown')}")
            
            start_time = time.time()
            result = self.execute(task_payload)
            elapsed_time = time.time() - start_time
            
            logger.info(f"[{self.expert_type}] 任务完成，耗时：{elapsed_time:.3f}s")
            
            # 确保结果包含必要字段
            if "status" not in result:
                result["status"] = "success"
            if "expert_type" not in result:
                result["expert_type"] = self.expert_type
                
            return result
            
        except ExpertExecutionError as e:
            logger.error(f"[{self.expert_type}] 执行失败：{e}")
            return {
                "status": "error",
                "error_message": str(e),
                "expert_type": self.expert_type
            }
        except Exception as e:
            logger.error(f"[{self.expert_type}] 未知错误：{e}", exc_info=True)
            return {
                "status": "error",
                "error_message": f"专家执行异常：{str(e)}",
                "expert_type": self.expert_type
            }
    
    def load(self) -> bool:
        """加载专家模型（可选重写）
        
        Returns:
            bool: 加载是否成功
        """
        logger.info(f"[{self.expert_type}] 加载专家模型")
        self.is_loaded = True
        return True
    
    def unload(self) -> bool:
        """卸载专家模型（可选重写）
        
        Returns:
            bool: 卸载是否成功
        """
        logger.info(f"[{self.expert_type}] 卸载专家模型")
        self.is_loaded = False
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """获取专家状态
        
        Returns:
            Dict[str, Any]: 状态信息
        """
        return {
            "expert_type": self.expert_type,
            "is_loaded": self.is_loaded,
            "last_access_time": self.last_access_time
        }
