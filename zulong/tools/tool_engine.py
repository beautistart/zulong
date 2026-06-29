# File: zulong/tools/tool_engine.py
# 工具调用引擎 - 工具系统的核心调度器

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import time
import threading
import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor, Future

from .base import (
    BaseTool, ToolRegistry, ToolRequest, ToolResult,
    ToolStatus, ToolCategory
)
from .web_search import WebSearchTool

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
        
        # 运行时上下文（InferenceEngine 在 FC 循环前注入，工具可读取）
        self._context: Dict[str, Any] = {}
        
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
        parameters = dict(parameters or {})

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
        
        # TaskGraph 继承会话节点地址，绑定后不可被其他会话改绑。
        # 这里统一补入运行时会话身份，并在工具执行前做 owner 守卫。
        if tool_name.startswith("task_") or tool_name == "submit_final_answer":
            try:
                conversation_id = (
                    parameters.get("conversation_id")
                    or parameters.get("session_id")
                    or self.get_context("conversation_id")
                    or self.get_context("session_id")
                    or ""
                )
                session_node_id = (
                    parameters.get("session_node_id")
                    or parameters.get("dialogue_session_id")
                    or self.get_context("session_node_id")
                    or self.get_context("dialogue_session_id")
                    or ""
                )
                if conversation_id:
                    parameters.setdefault("conversation_id", conversation_id)
                    parameters.setdefault("session_id", conversation_id)
                if conversation_id or session_node_id:
                    from zulong.tools.task_tools import (
                        ensure_task_graph_owner_for_request,
                        get_active_task_graph,
                        infer_task_graph_owner_session_node_id,
                    )

                    owner_node_id = infer_task_graph_owner_session_node_id(
                        conversation_id,
                        session_node_id,
                    )
                    if owner_node_id:
                        parameters.setdefault("session_node_id", owner_node_id)
                        parameters.setdefault("dialogue_session_id", owner_node_id)
                    owner_guard_exempt = {
                        "task_create_plan",
                        "task_list_suspended",
                        "task_resume_by_address",
                    }
                    if tool_name not in owner_guard_exempt:
                        active_task_graph = get_active_task_graph()
                        if active_task_graph is not None:
                            ok, guard_error = ensure_task_graph_owner_for_request(
                                active_task_graph,
                                parameters,
                                operation=tool_name,
                            )
                            if not ok:
                                call.status = "failed"
                                call.error = guard_error
                                call.end_time = time.time()
                                self._record_call(call)
                                return ToolResult(
                                    success=False,
                                    error=guard_error,
                                    execution_time=call.end_time - call.start_time,
                                    request_id=call_id,
                                )
            except Exception as exc:
                logger.debug("[ToolEngine] TaskGraph owner guard skipped: %s", exc)

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
        runtime_workspace_token = None
        runtime_workspace_reset = None
        try:
            runtime_workspace = (
                parameters.get("workspace_path")
                or parameters.get("workspace_dir")
                or parameters.get("cwd")
                or (
                    self.get_context("workspace_path")
                    if tool_name == "task_create_plan"
                    else ""
                )
                or (
                    self.get_context("workspace_dir")
                    if tool_name == "task_create_plan"
                    else ""
                )
                or ""
            )
            if tool_name == "task_create_plan" and runtime_workspace:
                parameters.setdefault("workspace_path", runtime_workspace)
                parameters.setdefault("workspace_dir", runtime_workspace)
            if runtime_workspace:
                from zulong.tools.task_tools import (
                    get_active_workspace_dir,
                    reset_runtime_workspace_dir,
                    set_runtime_workspace_dir,
                )
                from zulong.tools.workspace_access import (
                    normalize_workspace_path,
                    require_folder_access_authorization,
                )

                import os

                normalized_workspace = normalize_workspace_path(runtime_workspace)
                current_workspace = get_active_workspace_dir() or os.getcwd()
                same_workspace = False
                try:
                    same_workspace = bool(
                        current_workspace
                        and os.path.normcase(normalize_workspace_path(current_workspace))
                        == os.path.normcase(normalized_workspace)
                    )
                except Exception:
                    same_workspace = bool(current_workspace and current_workspace == normalized_workspace)
                if not same_workspace:
                    access = require_folder_access_authorization(
                        normalized_workspace,
                        current_workspace=current_workspace,
                        tool_name=tool_name,
                        action_summary=f"允许祖龙访问文件夹：{normalized_workspace}",
                        conversation_id=(
                            parameters.get("conversation_id")
                            or parameters.get("session_id")
                            or self.get_context("conversation_id")
                            or self.get_context("session_id")
                            or ""
                        ),
                        session_id=parameters.get("session_id") or self.get_context("session_id") or "",
                        request_id=parameters.get("request_id") or self.get_context("request_id") or call_id,
                        timeout=float(parameters.get("approval_timeout") or 180.0),
                    )
                    if not access.approved:
                        call.status = "failed"
                        call.error = access.message
                        call.end_time = time.time()
                        result = ToolResult(
                            success=False,
                            data=access.to_payload(),
                            error=access.message,
                            status_code=403,
                            execution_time=call.end_time - call.start_time,
                            request_id=call_id,
                        )
                        self.total_calls += 1
                        self.failed_calls += 1
                        self._record_call(call)
                        return result
                runtime_workspace_token = set_runtime_workspace_dir(normalized_workspace)
                runtime_workspace_reset = reset_runtime_workspace_dir
        except Exception as exc:
            logger.debug("[ToolEngine] Runtime workspace context skipped: %s", exc)
        
        try:
            result = tool.execute(request)
            
            # 防御性检测：如果工具返回了 coroutine（async def 未被 await），在此处运行
            if inspect.iscoroutine(result):
                logger.warning(f"[ToolEngine] {tool_name}.execute returned coroutine, running in event loop")
                try:
                    result = asyncio.run(result)
                except RuntimeError:
                    # 已有事件循环运行时，使用新线程
                    import concurrent.futures as cf
                    with cf.ThreadPoolExecutor(max_workers=1) as pool:
                        result = pool.submit(asyncio.run, result).result(timeout=30)
            
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
        finally:
            if runtime_workspace_token is not None and runtime_workspace_reset:
                runtime_workspace_reset(runtime_workspace_token)
        
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
            # 注册 Web 搜索工具（SearXNG 直连）
            web_search_tool = WebSearchTool()
            if self.register_tool(web_search_tool):
                logger.info("[ToolEngine] 已注册 Web 搜索工具 (SearXNG)")
            else:
                logger.debug("[ToolEngine] Web 搜索工具已存在，跳过注册")

            # 注册 search_tools 元工具（Tool RAG 入口）
            try:
                from zulong.tools.search_tools import SearchToolsTool, RequestToolSupplementTool
                search_tools_tool = SearchToolsTool()
                if self.register_tool(search_tools_tool):
                    logger.info("[ToolEngine] 已注册 search_tools 元工具")
                else:
                    logger.debug("[ToolEngine] search_tools 元工具已存在，跳过注册")
                supplement_tool = RequestToolSupplementTool(self.registry)
                if self.register_tool(supplement_tool):
                    logger.info("[ToolEngine] 已注册 request_tool_supplement 工具")
                else:
                    logger.debug("[ToolEngine] request_tool_supplement 工具已存在，跳过注册")
            except ImportError:
                logger.debug("[ToolEngine] search_tools 模块未找到，跳过注册")
            
            # 注册 search_experience 工具（ExperienceRAG 被动检索）
            try:
                from zulong.tools.experience_tool import SearchExperienceTool
                experience_tool = SearchExperienceTool()
                if self.register_tool(experience_tool):
                    logger.info("[ToolEngine] 已注册 search_experience 工具")
                else:
                    logger.debug("[ToolEngine] search_experience 工具已存在，跳过注册")
            except ImportError:
                logger.debug("[ToolEngine] experience_tool 模块未找到，跳过注册")

            # 注册 IDE 后台桥工具（供 L2 主动打开/切换 VS Code 工作区）
            # 文件写入统一由 exec_write_file 处理，并在内部按需路由到编辑器桥。
            try:
                from zulong.tools.ide_bridge_tools import (
                    IdeOpenWorkspaceTool,
                )
                ide_open_workspace_tool = IdeOpenWorkspaceTool()
                if self.register_tool(ide_open_workspace_tool):
                    logger.info("[ToolEngine] 已注册 ide_open_workspace 工具")
                else:
                    logger.debug("[ToolEngine] ide_open_workspace 工具已存在，跳过注册")
            except ImportError:
                logger.debug("[ToolEngine] ide_bridge_tools 模块未找到，跳过注册")
            
            # 注册 navigate_attention 工具（思维深度索引 — 注意力导航）
            try:
                from zulong.tools.attention_tool import NavigateAttentionTool, AdjustAttentionModeTool
                attention_tool = NavigateAttentionTool()
                if self.register_tool(attention_tool):
                    logger.info("[ToolEngine] 已注册 navigate_attention 工具")
                else:
                    logger.debug("[ToolEngine] navigate_attention 工具已存在，跳过注册")
                # P2-1: 注册直接模式切换工具
                attn_mode_tool = AdjustAttentionModeTool()
                if self.register_tool(attn_mode_tool):
                    logger.info("[ToolEngine] 已注册 adjust_attention_mode 工具")
                else:
                    logger.debug("[ToolEngine] adjust_attention_mode 工具已存在，跳过注册")
            except ImportError:
                logger.debug("[ToolEngine] attention_tool 模块未找到，跳过注册")
            
            # 注册 MemoryGraph FC 工具集
            try:
                from zulong.tools.memory_graph_tools import (
                    RecallMemoryTool, ReadMemoryNodeTool,
                    SaveMemoryNoteTool, DiscoverRelatedTool,
                    ActivateMemoryNetworkTool, ListMemoryTool,
                    SetImportanceTool, DeleteMemoryNodeTool,
                    DeleteMemoryEdgeTool,
                )
                for tool_cls in [RecallMemoryTool, ReadMemoryNodeTool,
                                 SaveMemoryNoteTool, DiscoverRelatedTool,
                                 ActivateMemoryNetworkTool, ListMemoryTool,
                                 SetImportanceTool, DeleteMemoryNodeTool,
                                 DeleteMemoryEdgeTool]:
                    tool_inst = tool_cls()
                    if self.register_tool(tool_inst):
                        logger.info(f"[ToolEngine] 已注册 {tool_inst.name} 工具")
                    else:
                        logger.debug(f"[ToolEngine] {tool_inst.name} 工具已存在，跳过注册")
            except ImportError:
                logger.debug("[ToolEngine] memory_graph_tools 模块未找到，跳过注册")
            
            # 注册 CodeAnchor FC 工具集
            try:
                from zulong.tools.code_anchor_tools import (
                    MemoryWriteWithCodeTool, CodeQueryTool, TaskLinkCodeTool,
                )
                for tool_cls in [MemoryWriteWithCodeTool, CodeQueryTool, TaskLinkCodeTool]:
                    tool_inst = tool_cls()
                    if self.register_tool(tool_inst):
                        logger.info(f"[ToolEngine] 已注册 {tool_inst.name} 工具")
                    else:
                        logger.debug(f"[ToolEngine] {tool_inst.name} 工具已存在，跳过注册")
            except ImportError:
                logger.debug("[ToolEngine] code_anchor_tools 模块未找到，跳过注册")
            
            # 注册任务管理 FC 工具集
            try:
                from zulong.tools.task_tools import (
                    TaskCreatePlanTool, TaskAddNodeTool,
                    TaskMarkStatusTool,
                    TaskViewOverviewTool,
                    TaskSuspendTool, TaskListSuspendedTool,
                    TaskAddDependencyTool, TaskGetDetailTool,
                    TaskUpdateNodeTool, TaskRemoveNodeTool,
                    TaskUpdateContentTool, TaskAttachFileTool,
                    SubmitFinalAnswerTool,
                    TaskResumeByAddressTool, TaskReviseNodeTool,
                )
                for tool_cls in [TaskCreatePlanTool, TaskAddNodeTool,
                                 TaskMarkStatusTool,
                                 TaskViewOverviewTool,
                                 TaskSuspendTool, TaskListSuspendedTool,
                                 TaskAddDependencyTool, TaskGetDetailTool,
                                 TaskUpdateNodeTool, TaskRemoveNodeTool,
                                 TaskUpdateContentTool, TaskAttachFileTool,
                                 SubmitFinalAnswerTool,
                                 TaskResumeByAddressTool, TaskReviseNodeTool]:
                    tool_inst = tool_cls()
                    if self.register_tool(tool_inst):
                        logger.info(f"[ToolEngine] 已注册 {tool_inst.name} 工具")
                    else:
                        logger.debug(f"[ToolEngine] {tool_inst.name} 工具已存在，跳过注册")
            except ImportError:
                logger.debug("[ToolEngine] task_tools 模块未找到，跳过注册")
            
            # 注册执行 FC 工具集
            try:
                from zulong.tools.exec_tools import (
                    ExecWriteFileTool, ExecRunCommandTool,
                )
                for tool_cls in [ExecWriteFileTool, ExecRunCommandTool]:
                    tool_inst = tool_cls()
                    if self.register_tool(tool_inst):
                        logger.info(f"[ToolEngine] 已注册 {tool_inst.name} 工具")
                    else:
                        logger.debug(f"[ToolEngine] {tool_inst.name} 工具已存在，跳过注册")
            except ImportError:
                logger.debug("[ToolEngine] exec_tools 模块未找到，跳过注册")
            
            # 注册会话交互工具
            try:
                from zulong.tools.session_tools import AskUserTool
                ask_user_tool = AskUserTool()
                if self.register_tool(ask_user_tool):
                    logger.info("[ToolEngine] 已注册 ask_user 工具")
                else:
                    logger.debug("[ToolEngine] ask_user 工具已存在，跳过注册")
            except ImportError:
                logger.debug("[ToolEngine] session_tools 模块未找到，跳过注册")

            # 注册代码智能工具
            try:
                from zulong.tools.code_tools import (
                    SearchCodeSymbolsTool, GetSymbolContextTool,
                    GetImpactAnalysisTool, IndexCodeFileTool,
                    IndexProjectTool, AnalyzeModuleTool,
                )
                from zulong.tools.read_file_tool import ReadFileTool
                read_file_tool = ReadFileTool()
                if self.register_tool(read_file_tool):
                    logger.info(f"[ToolEngine] 已注册 {read_file_tool.name} 工具")
                for code_tool_cls in (
                    SearchCodeSymbolsTool, GetSymbolContextTool,
                    GetImpactAnalysisTool, IndexCodeFileTool,
                    IndexProjectTool, AnalyzeModuleTool,
                ):
                    code_tool = code_tool_cls()
                    if self.register_tool(code_tool):
                        logger.info(f"[ToolEngine] 已注册 {code_tool.name} 工具")
            except ImportError:
                logger.debug("[ToolEngine] code_tools 模块未找到，跳过注册")
        except Exception as e:
            logger.error(f"[ToolEngine] 注册内置工具失败：{e}")
    
    def set_context(self, **kwargs) -> None:
        """注入运行时上下文（由 InferenceEngine 在 FC 循环前调用）
        
        Args:
            **kwargs: 上下文键值对，如 user_input, voice_mode, task_graph 等
        """
        with self._lock:
            self._context.update(kwargs)

    def get_context(self, key: str = None, default=None):
        """获取运行时上下文
        
        Args:
            key: 上下文键名。为 None 时返回整个上下文字典。
            default: 键不存在时的默认值
            
        Returns:
            上下文值或整个上下文字典
        """
        with self._lock:
            if key is None:
                return self._context.copy()
            return self._context.get(key, default)

    def clear_context(self) -> None:
        """清除运行时上下文（由 InferenceEngine 在 FC 循环结束后调用）"""
        with self._lock:
            self._context.clear()

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
