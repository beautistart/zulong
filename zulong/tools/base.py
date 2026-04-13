# File: zulong/tools/base.py
# 工具/技能基础接口定义

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)


class ToolStatus(Enum):
    """工具状态"""
    READY = "ready"          # 就绪
    RUNNING = "running"      # 运行中
    SUCCESS = "success"      # 成功
    FAILED = "failed"        # 失败
    TIMEOUT = "timeout"      # 超时


class ToolCategory(Enum):
    """工具分类"""
    SYSTEM = "system"        # 系统工具（文件、进程等）
    NETWORK = "network"      # 网络工具（HTTP、API 等）
    CODE = "code"           # 代码工具（VSCode、执行等）
    ROBOT = "robot"         # 机器人工具（运动、感知等）
    CUSTOM = "custom"       # 自定义工具


@dataclass
class ToolRequest:
    """工具请求
    
    TSD v1.7 对应规则:
    - 结构化请求
    - 支持超时控制
    - 支持优先级
    """
    tool_name: str                    # 工具名称
    action: str                       # 动作
    parameters: Dict[str, Any]       # 参数
    timeout: float = 30.0            # 超时时间（秒）
    priority: int = 5                # 优先级（1-10，10 最高）
    request_id: str = ""             # 请求 ID
    callback: Optional[Callable] = None  # 回调函数
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"req_{int(time.time() * 1000)}"


@dataclass
class ToolResult:
    """工具执行结果
    
    TSD v1.7 对应规则:
    - 统一返回格式
    - 包含状态码
    - 支持错误信息
    """
    success: bool                     # 是否成功
    data: Any = None                 # 返回数据
    error: Optional[str] = None      # 错误信息
    status_code: int = 0             # 状态码
    execution_time: float = 0.0      # 执行时间（秒）
    request_id: str = ""             # 请求 ID
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "status_code": self.status_code,
            "execution_time": self.execution_time,
            "request_id": self.request_id
        }


class BaseTool(ABC):
    """工具基类
    
    TSD v1.7 对应规则:
    - 统一接口
    - 支持异步执行
    - 支持超时控制
    - 支持错误恢复
    
    所有工具必须继承此类并实现:
    - initialize(): 初始化
    - execute(): 执行
    - cleanup(): 清理
    """
    
    def __init__(self, name: str, category: ToolCategory = ToolCategory.CUSTOM):
        """初始化工具
        
        Args:
            name: 工具名称
            category: 工具分类
        """
        self.name = name
        self.category = category
        self.status = ToolStatus.READY
        self.description = ""
        self.version = "1.0.0"
        
        # 统计信息
        self.total_calls = 0
        self.success_calls = 0
        self.failed_calls = 0
        self.total_execution_time = 0.0
        
        # 配置
        self.enabled = True
        self.max_timeout = 300.0  # 最大超时（秒）
        
        logger.info(f"[BaseTool] {name} initialized (category={category.value})")
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化工具
        
        Returns:
            bool: 是否初始化成功
        """
        pass
    
    @abstractmethod
    def execute(self, request: ToolRequest) -> ToolResult:
        """执行工具
        
        Args:
            request: 工具请求
            
        Returns:
            ToolResult: 执行结果
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """清理工具资源"""
        pass
    
    def validate(self, request: ToolRequest) -> bool:
        """验证请求
        
        Args:
            request: 工具请求
            
        Returns:
            bool: 是否有效
        """
        if not self.enabled:
            logger.warning(f"[{self.name}] Tool is disabled")
            return False
        
        if request.timeout > self.max_timeout:
            logger.warning(f"[{self.name}] Timeout too large: {request.timeout}")
            return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "name": self.name,
            "category": self.category.value,
            "status": self.status.value,
            "total_calls": self.total_calls,
            "success_calls": self.success_calls,
            "failed_calls": self.failed_calls,
            "avg_execution_time": (
                self.total_execution_time / max(self.total_calls, 1)
            ),
            "enabled": self.enabled
        }
    
    def _create_result(
        self,
        success: bool,
        data: Any = None,
        error: Optional[str] = None,
        status_code: int = 0,
        execution_time: float = 0.0,
        request_id: str = ""
    ) -> ToolResult:
        """创建执行结果
        
        Args:
            success: 是否成功
            data: 返回数据
            error: 错误信息
            status_code: 状态码
            execution_time: 执行时间
            request_id: 请求 ID
            
        Returns:
            ToolResult: 执行结果
        """
        # 更新统计
        self.total_calls += 1
        if success:
            self.success_calls += 1
        else:
            self.failed_calls += 1
        self.total_execution_time += execution_time
        
        return ToolResult(
            success=success,
            data=data,
            error=error,
            status_code=status_code,
            execution_time=execution_time,
            request_id=request_id
        )
    
    def get_function_schema(self) -> Dict[str, Any]:
        """返回 OpenAI Function Calling 格式的工具描述
        
        用于动态构建 tools 参数，使模型能够了解并调用此工具。
        子类应实现 _get_parameters_schema() 来描述参数。
        
        Returns:
            OpenAI Function Calling 格式的工具描述
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._get_parameters_schema()
            }
        }
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """返回工具的参数 Schema（OpenAI Function Calling 格式）
        
        子类应重写此方法来描述工具接受的参数。
        默认返回空 schema（无参数）。
        
        Returns:
            JSON Schema 格式的参数描述
        """
        return {
            "type": "object",
            "properties": {},
            "required": []
        }


class ToolRegistry:
    """工具注册表
    
    TSD v1.7 对应规则:
    - 全局单例
    - 支持动态注册/注销
    - 支持按分类查询
    
    功能:
    - 工具注册
    - 工具查找
    - 工具列表
    - 批量初始化
    """
    
    _instance: Optional["ToolRegistry"] = None
    
    def __new__(cls) -> "ToolRegistry":
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.tools: Dict[str, BaseTool] = {}
        self.categories: Dict[ToolCategory, List[str]] = {
            cat: [] for cat in ToolCategory
        }
        self._initialized = True
        
        logger.info("[ToolRegistry] Initialized (singleton)")
    
    def register(self, tool: BaseTool) -> bool:
        """注册工具
        
        Args:
            tool: 工具实例
            
        Returns:
            bool: 是否注册成功
        """
        if tool.name in self.tools:
            logger.warning(f"[ToolRegistry] Tool already exists: {tool.name}")
            return False
        
        self.tools[tool.name] = tool
        self.categories[tool.category].append(tool.name)
        
        logger.info(f"[ToolRegistry] Registered: {tool.name} ({tool.category.value})")
        return True
    
    def unregister(self, tool_name: str) -> bool:
        """注销工具
        
        Args:
            tool_name: 工具名称
            
        Returns:
            bool: 是否注销成功
        """
        if tool_name not in self.tools:
            logger.warning(f"[ToolRegistry] Tool not found: {tool_name}")
            return False
        
        tool = self.tools.pop(tool_name)
        self.categories[tool.category].remove(tool_name)
        
        logger.info(f"[ToolRegistry] Unregistered: {tool_name}")
        return True
    
    def get(self, tool_name: str) -> Optional[BaseTool]:
        """获取工具
        
        Args:
            tool_name: 工具名称
            
        Returns:
            Optional[BaseTool]: 工具实例，不存在则返回 None
        """
        return self.tools.get(tool_name)
    
    def get_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """按分类获取工具
        
        Args:
            category: 工具分类
            
        Returns:
            List[BaseTool]: 工具列表
        """
        return [
            self.tools[name]
            for name in self.categories.get(category, [])
        ]
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有工具
        
        Returns:
            List[Dict]: 工具信息列表
        """
        return [
            {
                "name": tool.name,
                "category": tool.category.value,
                "description": tool.description,
                "version": tool.version,
                "enabled": tool.enabled
            }
            for tool in self.tools.values()
        ]
    
    def initialize_all(self) -> int:
        """初始化所有工具
        
        Returns:
            int: 成功初始化的数量
        """
        success_count = 0
        
        for tool in self.tools.values():
            try:
                if tool.initialize():
                    success_count += 1
                    logger.info(f"[ToolRegistry] Initialized: {tool.name}")
                else:
                    logger.error(f"[ToolRegistry] Init failed: {tool.name}")
            except Exception as e:
                logger.error(f"[ToolRegistry] Init error: {tool.name} - {e}")
        
        return success_count
    
    def cleanup_all(self) -> None:
        """清理所有工具"""
        for tool in self.tools.values():
            try:
                tool.cleanup()
                logger.info(f"[ToolRegistry] Cleaned up: {tool.name}")
            except Exception as e:
                logger.error(f"[ToolRegistry] Cleanup error: {tool.name} - {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_tools": len(self.tools),
            "tools_by_category": {
                cat.value: len(names)
                for cat, names in self.categories.items()
            },
            "tools": {
                name: tool.get_statistics()
                for name, tool in self.tools.items()
            }
        }
    
    def get_all_function_schemas(self) -> List[Dict[str, Any]]:
        """聚合所有已注册工具的 Function Calling Schema
        
        用于 InferenceEngine 动态加载工具列表。
        
        Returns:
            OpenAI Function Calling 格式的 tools 列表
        """
        schemas = []
        for tool in self.tools.values():
            try:
                schema = tool.get_function_schema()
                schemas.append(schema)
            except Exception as e:
                logger.warning(f"[ToolRegistry] Failed to get schema for {tool.name}: {e}")
        return schemas
