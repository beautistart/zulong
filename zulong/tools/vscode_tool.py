# File: zulong/tools/vscode_tool.py
# VSCode 集成工具 - CLI 方案

import logging
import subprocess
import shutil
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base import BaseTool, ToolRequest, ToolResult, ToolCategory, ToolStatus

logger = logging.getLogger(__name__)


class VSCodeTool(BaseTool):
    """VSCode CLI 工具
    
    TSD v1.7 对应规则:
    - 方案 A：CLI 工具（快速验证，功能有限）
    - 使用 `code` 命令行工具
    - 支持打开文件、工作区
    - 支持执行命令
    
    功能:
    - 打开文件
    - 打开工作区
    - 新建文件
    - 执行终端命令
    - 检查安装状态
    """
    
    def __init__(self):
        super().__init__("vscode_tool", ToolCategory.CODE)
        self.description = "VSCode CLI integration tool"
        self.version = "1.0.0"
        
        # 配置
        self.code_command = "code"  # VSCode CLI 命令
        self.timeout = 30.0  # 默认超时
        
        # 状态
        self.vscode_installed = False
        self.vscode_path: Optional[str] = None
    
    def initialize(self) -> bool:
        """初始化 VSCode 工具
        
        Returns:
            bool: 是否初始化成功
        """
        try:
            # 检查 code 命令是否存在
            code_path = shutil.which(self.code_command)
            
            if code_path:
                self.vscode_installed = True
                self.vscode_path = code_path
                logger.info(f"[VSCodeTool] VSCode CLI found at: {code_path}")
                
                # 获取版本信息
                version = self._get_version()
                logger.info(f"[VSCodeTool] Version: {version}")
            else:
                self.vscode_installed = False
                logger.warning("[VSCodeTool] VSCode CLI not found in PATH")
                logger.warning("[VSCodeTool] Please install VSCode and add 'code' to PATH")
            
            self.status = ToolStatus.READY
            return self.vscode_installed
            
        except Exception as e:
            logger.error(f"[VSCodeTool] Initialization error: {e}")
            self.status = ToolStatus.FAILED
            return False
    
    def execute(self, request: ToolRequest) -> ToolResult:
        """执行 VSCode 工具
        
        Args:
            request: 工具请求
            
        Returns:
            ToolResult: 执行结果
        """
        import time
        start_time = time.time()
        
        # 验证请求
        if not self.validate(request):
            return self._create_result(
                success=False,
                error="Tool validation failed",
                execution_time=time.time() - start_time,
                request_id=request.request_id
            )
        
        # 检查 VSCode 是否安装
        if not self.vscode_installed:
            return self._create_result(
                success=False,
                error="VSCode CLI not installed",
                execution_time=time.time() - start_time,
                request_id=request.request_id
            )
        
        try:
            # 根据动作执行
            action = request.action
            params = request.parameters
            
            if action == "open_file":
                result = self._open_file(
                    params.get("file_path", ""),
                    params.get("line", None),
                    params.get("column", None)
                )
            elif action == "open_folder":
                result = self._open_folder(params.get("folder_path", ""))
            elif action == "open_workspace":
                result = self._open_workspace(params.get("workspace_path", ""))
            elif action == "new_file":
                result = self._new_file(
                    params.get("file_path", ""),
                    params.get("content", "")
                )
            elif action == "run_command":
                result = self._run_command(
                    params.get("command", ""),
                    params.get("cwd", None)
                )
            elif action == "install_extension":
                result = self._install_extension(params.get("extension_id", ""))
            elif action == "get_info":
                result = self._get_info()
            else:
                result = {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }
            
            execution_time = time.time() - start_time
            
            return self._create_result(
                success=result.get("success", False),
                data=result.get("data"),
                error=result.get("error"),
                execution_time=execution_time,
                request_id=request.request_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[VSCodeTool] Execute error: {e}")
            return self._create_result(
                success=False,
                error=str(e),
                execution_time=execution_time,
                request_id=request.request_id
            )
    
    def cleanup(self) -> None:
        """清理 VSCode 工具资源"""
        logger.info("[VSCodeTool] Cleanup complete")
    
    def _open_file(
        self,
        file_path: str,
        line: Optional[int] = None,
        column: Optional[int] = None
    ) -> Dict[str, Any]:
        """打开文件
        
        Args:
            file_path: 文件路径
            line: 行号（可选）
            column: 列号（可选）
            
        Returns:
            Dict: 执行结果
        """
        try:
            # 构建命令
            cmd = [self.code_command]
            
            # 添加行号/列号
            if line is not None and column is not None:
                cmd.extend(["-g", f"{file_path}:{line}:{column}"])
            elif line is not None:
                cmd.extend(["-g", f"{file_path}:{line}"])
            else:
                cmd.append(file_path)
            
            # 执行
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                logger.info(f"[VSCodeTool] Opened file: {file_path}")
                return {
                    "success": True,
                    "data": {
                        "file_path": file_path,
                        "line": line,
                        "column": column
                    }
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Timeout after {self.timeout}s"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _open_folder(self, folder_path: str) -> Dict[str, Any]:
        """打开文件夹/工作区
        
        Args:
            folder_path: 文件夹路径
            
        Returns:
            Dict: 执行结果
        """
        try:
            cmd = [self.code_command, folder_path]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                logger.info(f"[VSCodeTool] Opened folder: {folder_path}")
                return {
                    "success": True,
                    "data": {"folder_path": folder_path}
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _open_workspace(self, workspace_path: str) -> Dict[str, Any]:
        """打开工作区文件 (.code-workspace)
        
        Args:
            workspace_path: 工作区文件路径
            
        Returns:
            Dict: 执行结果
        """
        try:
            cmd = [self.code_command, workspace_path]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                logger.info(f"[VSCodeTool] Opened workspace: {workspace_path}")
                return {
                    "success": True,
                    "data": {"workspace_path": workspace_path}
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _new_file(self, file_path: str, content: str = "") -> Dict[str, Any]:
        """新建文件并打开
        
        Args:
            file_path: 文件路径
            content: 文件内容（可选）
            
        Returns:
            Dict: 执行结果
        """
        try:
            # 创建父目录
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 写入内容
            if content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # 打开文件
            return self._open_file(file_path)
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_command(
        self,
        command: str,
        cwd: Optional[str] = None
    ) -> Dict[str, Any]:
        """在 VSCode 终端运行命令（通过集成终端）
        
        注意：这实际上是运行系统命令，不是 VSCode 特有功能
        
        Args:
            command: 命令
            cwd: 工作目录
            
        Returns:
            Dict: 执行结果
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=cwd
            )
            
            return {
                "success": result.returncode == 0,
                "data": {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Timeout after {self.timeout}s"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _install_extension(self, extension_id: str) -> Dict[str, Any]:
        """安装 VSCode 扩展
        
        Args:
            extension_id: 扩展 ID (e.g., "ms-python.python")
            
        Returns:
            Dict: 执行结果
        """
        try:
            cmd = [
                self.code_command,
                "--install-extension",
                extension_id
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout * 2  # 安装可能需要更长时间
            )
            
            if result.returncode == 0:
                logger.info(f"[VSCodeTool] Installed extension: {extension_id}")
                return {
                    "success": True,
                    "data": {"extension_id": extension_id}
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_info(self) -> Dict[str, Any]:
        """获取 VSCode 信息
        
        Returns:
            Dict: VSCode 信息
        """
        return {
            "success": True,
            "data": {
                "installed": self.vscode_installed,
                "path": self.vscode_path,
                "version": self._get_version(),
                "command": self.code_command
            }
        }
    
    def _get_version(self) -> Optional[str]:
        """获取 VSCode 版本
        
        Returns:
            Optional[str]: 版本号
        """
        if not self.vscode_installed:
            return None
        
        try:
            cmd = [self.code_command, "--version"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5.0
            )
            
            if result.returncode == 0:
                # 第一行是版本号
                version = result.stdout.strip().split('\n')[0]
                return version
            else:
                return None
                
        except Exception:
            return None
    
    def is_available(self) -> bool:
        """检查 VSCode 是否可用
        
        Returns:
            bool: 是否可用
        """
        return self.vscode_installed and self.status == ToolStatus.READY
