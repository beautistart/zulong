# File: zulong/tools/tool_engine.py
# 工具调用引擎 - 工具系统的核心调度器

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future

from .base import (
    BaseTool, ToolRegistry, ToolRequest, ToolResult,
    ToolStatus, ToolCategory
)
from .web_search import WebSearchTool
from .openclaw_tool import OpenClawToolAdapter
from .openclaw_search import OpenClawSearchTool
from .openclaw_plugin import OpenClawPluginAdapter

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """工具调用记录
    
    TSD v1.7 对应规则:
    - 记录所有工具调用
    - 支持性能分析
    - 支持错误追踪
    """
    call_id: str
    tool_name: str
    action: str
    parameters: Dict[str, Any]
    start_time: float
    end_time: float = 0.0
    result: Optional[ToolResult] = None
    status: str = "pending"  # pending, running, success, failed
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "action": self.action,
            "parameters": self.parameters,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time if self.end_time > 0 else 0,
            "status": self.status,
            "error": self.error
        }


class ToolEngine:
    """工具调用引擎
    
    TSD v1.7 对应规则:
    - 统一工具调用接口
    - 支持并发执行
    - 支持优先级调度
    - 支持超时控制
    - 支持错误恢复
    - 支持调用历史
    
    功能:
    - 工具注册管理
    - 请求解析与验证
    - 并发调度
    - 结果聚合
    - 性能监控
    """
    
    def __init__(self, max_workers: int = 5):
        """初始化工具引擎
        
        Args:
            max_workers: 最大并发工作线程数
        """
        self.registry = ToolRegistry()
        self.max_workers = max_workers
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 调用历史
        self.call_history: List[ToolCall] = []
        self.max_history_size = 1000
        
        # 统计信息
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        
        # 状态
        self.running = True
        self._lock = threading.Lock()
        
        # 注册内置工具
        self._register_builtin_tools()
        
        logger.info(f"[ToolEngine] Initialized (max_workers={max_workers})")
    
    def shutdown(self, wait: bool = True) -> None:
        """关闭引擎
        
        Args:
            wait: 是否等待未完成的任务
        """
        self.running = False
        self.executor.shutdown(wait=wait)
        self.registry.cleanup_all()
        logger.info("[ToolEngine] Shutdown complete")
    
    def register_tool(self, tool: BaseTool) -> bool:
        """注册工具
        
        Args:
            tool: 工具实例
            
        Returns:
            bool: 是否注册成功
        """
        return self.registry.register(tool)
    
    def unregister_tool(self, tool_name: str) -> bool:
        """注销工具
        
        Args:
            tool_name: 工具名称
            
        Returns:
            bool: 是否注销成功
        """
        return self.registry.unregister(tool_name)
    
    def call_tool(
        self,
        tool_name: str,
        action: str,
        parameters: Dict[str, Any],
        timeout: float = 30.0,
        priority: int = 5,
        callback: Optional[Callable[[ToolResult], None]] = None
    ) -> ToolResult:
        """同步调用工具
        
        Args:
            tool_name: 工具名称
            action: 动作
            parameters: 参数
            timeout: 超时时间（秒）
            priority: 优先级（1-10）
            callback: 回调函数
            
        Returns:
            ToolResult: 执行结果
        """
        # 创建调用记录
        call_id = f"call_{int(time.time() * 1000)}"
        call = ToolCall(
            call_id=call_id,
            tool_name=tool_name,
            action=action,
            parameters=parameters,
            start_time=time.time()
        )
        
        # 获取工具
        tool = self.registry.get(tool_name)
        if not tool:
            error_msg = f"Tool not found: {tool_name}"
            logger.error(f"[ToolEngine] {error_msg}")
            
            call.status = "failed"
            call.error = error_msg
            call.end_time = time.time()
            self._record_call(call)
            
            return ToolResult(
                success=False,
                error=error_msg,
                execution_time=0.0,
                request_id=call_id
            )
        
        # 检查工具状态
        if not tool.enabled:
            error_msg = f"Tool is disabled: {tool_name}"
            logger.error(f"[ToolEngine] {error_msg}")
            
            call.status = "failed"
            call.error = error_msg
            call.end_time = time.time()
            self._record_call(call)
            
            return ToolResult(
                success=False,
                error=error_msg,
                execution_time=0.0,
                request_id=call_id
            )
        
        # 创建请求
        request = ToolRequest(
            tool_name=tool_name,
            action=action,
            parameters=parameters,
            timeout=timeout,
            priority=priority,
            request_id=call_id,
            callback=callback
        )
        
        # 执行
        call.status = "running"
        logger.debug(f"[ToolEngine] Calling {tool_name}.{action}")
        
        try:
            result = tool.execute(request)
            
            # 更新调用记录
            call.status = "success" if result.success else "failed"
            call.result = result
            call.error = result.error
            call.end_time = time.time()
            
            # 更新统计
            self.total_calls += 1
            if result.success:
                self.successful_calls += 1
            else:
                self.failed_calls += 1
            
            # 回调
            if callback and result.success:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"[ToolEngine] Callback error: {e}")
            
            logger.debug(f"[ToolEngine] Completed {tool_name}.{action} "
                        f"in {result.execution_time:.3f}s")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[ToolEngine] Execute error: {e}")
            
            call.status = "failed"
            call.error = error_msg
            call.end_time = time.time()
            
            result = ToolResult(
                success=False,
                error=error_msg,
                execution_time=time.time() - call.start_time,
                request_id=call_id
            )
        
        # 记录
        self._record_call(call)
        
        return result
    
    def call_tool_async(
        self,
        tool_name: str,
        action: str,
        parameters: Dict[str, Any],
        timeout: float = 30.0,
        priority: int = 5,
        callback: Optional[Callable[[ToolResult], None]] = None
    ) -> Future:
        """异步调用工具
        
        Args:
            tool_name: 工具名称
            action: 动作
            parameters: 参数
            timeout: 超时时间
            priority: 优先级
            callback: 回调函数
            
        Returns:
            Future: 未来对象
        """
        future = self.executor.submit(
            self.call_tool,
            tool_name,
            action,
            parameters,
            timeout,
            priority,
            callback
        )
        
        logger.debug(f"[ToolEngine] Async call {tool_name}.{action}")
        
        return future
    
    def call_batch(
        self,
        calls: List[Dict[str, Any]],
        parallel: bool = True
    ) -> List[ToolResult]:
        """批量调用工具
        
        Args:
            calls: 调用列表，每项包含 tool_name, action, parameters 等
            parallel: 是否并行执行
            
        Returns:
            List[ToolResult]: 结果列表
        """
        if not parallel:
            # 串行执行
            results = []
            for call in calls:
                result = self.call_tool(**call)
                results.append(result)
            return results
        
        # 并行执行
        futures = []
        for call in calls:
            future = self.call_tool_async(**call)
            futures.append(future)
        
        # 等待所有结果
        results = []
        for future in futures:
            try:
                result = future.result(timeout=call.get('timeout', 30))
                results.append(result)
            except Exception as e:
                results.append(ToolResult(
                    success=False,
                    error=str(e),
                    execution_time=0.0
                ))
        
        return results
    
    def _record_call(self, call: ToolCall) -> None:
        """记录调用历史
        
        Args:
            call: 调用记录
        """
        with self._lock:
            self.call_history.append(call)
            
            # 限制历史记录大小
            if len(self.call_history) > self.max_history_size:
                self.call_history = self.call_history[-self.max_history_size:]
    
    def get_call_history(
        self,
        tool_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取调用历史
        
        Args:
            tool_name: 工具名称过滤
            status: 状态过滤
            limit: 返回数量限制
            
        Returns:
            List[Dict]: 调用历史
        """
        with self._lock:
            history = self.call_history.copy()
        
        # 过滤
        if tool_name:
            history = [c for c in history if c.tool_name == tool_name]
        if status:
            history = [c for c in history if c.status == status]
        
        # 限制数量
        history = history[-limit:]
        
        # 转换为字典
        return [c.to_dict() for c in history]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            history_size = len(self.call_history)
        
        # 计算平均执行时间
        if history_size > 0:
            total_time = sum(
                c.end_time - c.start_time
                for c in self.call_history
                if c.end_time > 0
            )
            avg_time = total_time / history_size
        else:
            avg_time = 0.0
        
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": (
                self.successful_calls / max(self.total_calls, 1)
            ),
            "avg_execution_time": avg_time,
            "history_size": history_size,
            "max_history_size": self.max_history_size,
            "max_workers": self.max_workers,
            "tools_registered": len(self.registry.tools)
        }
    
    def list_available_tools(self) -> List[Dict[str, Any]]:
        """列出可用工具
        
        Returns:
            List[Dict]: 工具信息列表
        """
        return self.registry.list_tools()
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """获取工具信息
        
        Args:
            tool_name: 工具名称
            
        Returns:
            Optional[Dict]: 工具信息
        """
        tool = self.registry.get(tool_name)
        if tool:
            return {
                "name": tool.name,
                "category": tool.category.value,
                "description": tool.description,
                "version": tool.version,
                "enabled": tool.enabled,
                "status": tool.status.value,
                "statistics": tool.get_statistics()
            }
        return None
    
    def enable_tool(self, tool_name: str) -> bool:
        """启用工具"""
        tool = self.registry.get(tool_name)
        if tool:
            tool.enabled = True
            logger.info(f"[ToolEngine] Enabled tool: {tool_name}")
            return True
        return False
    
    def disable_tool(self, tool_name: str) -> bool:
        """禁用工具"""
        tool = self.registry.get(tool_name)
        if tool:
            tool.enabled = False
            logger.info(f"[ToolEngine] Disabled tool: {tool_name}")
            return True
        return False
    
    def _register_builtin_tools(self):
        """注册内置工具"""
        try:
            # 注册网络搜索工具
            web_search_tool = WebSearchTool()
            if self.register_tool(web_search_tool):
                logger.info("[ToolEngine] 已注册网络搜索工具")
            else:
                logger.debug("[ToolEngine] 网络搜索工具已存在，跳过注册")
            
            # 注册 OpenClaw 工具适配器
            openclaw_tool = OpenClawToolAdapter()
            if self.register_tool(openclaw_tool):
                logger.info("[ToolEngine] 已注册 OpenClaw 工具适配器")
            else:
                logger.debug("[ToolEngine] OpenClaw 工具适配器已存在，跳过注册")
            
            # 注册 OpenClaw 搜索工具
            openclaw_search_tool = OpenClawSearchTool()
            if self.register_tool(openclaw_search_tool):
                logger.info("[ToolEngine] 已注册 OpenClaw 搜索工具")
            else:
                logger.debug("[ToolEngine] OpenClaw 搜索工具已存在，跳过注册")
            
            # 注册 OpenClaw 插件适配器
            openclaw_plugin_tool = OpenClawPluginAdapter()
            if self.register_tool(openclaw_plugin_tool):
                logger.info("[ToolEngine] 已注册 OpenClaw 插件适配器")
            else:
                logger.debug("[ToolEngine] OpenClaw 插件适配器已存在，跳过注册")
        except Exception as e:
            logger.error(f"[ToolEngine] 注册内置工具失败：{e}")
    
    def print_status(self):
        """打印状态信息"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("工具调用引擎状态")
        print("=" * 60)
        print(f"总调用数：{stats['total_calls']}")
        print(f"成功调用：{stats['successful_calls']}")
        print(f"失败调用：{stats['failed_calls']}")
        print(f"成功率：{stats['success_rate']:.2%}")
        print(f"平均执行时间：{stats['avg_execution_time']:.3f}s")
        print(f"历史记录：{stats['history_size']}/{stats['max_history_size']}")
        print(f"注册工具：{stats['tools_registered']}")
        print(f"最大并发：{stats['max_workers']}")
        print("=" * 60 + "\n")
