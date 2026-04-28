# File: zulong/tools/system_tools.py
# 系统工具集 - 文件、网络、系统命令

import logging
import subprocess
import shutil
import os
from pathlib import Path
from typing import Dict, Any, Optional
import time

import requests

from .base import BaseTool, ToolRequest, ToolResult, ToolCategory, ToolStatus

logger = logging.getLogger(__name__)


class FileTool(BaseTool):
    """文件操作工具
    
    功能:
    - 读取文件
    - 写入文件
    - 删除文件
    - 复制/移动文件
    - 列出目录
    - 创建/删除目录
    """
    
    def __init__(self):
        super().__init__("file_tool", ToolCategory.SYSTEM)
        self.description = "File operation tool"
        self.version = "1.0.0"
    
    def initialize(self) -> bool:
        """初始化"""
        self.status = ToolStatus.READY
        logger.info("[FileTool] Initialized")
        return True
    
    def execute(self, request: ToolRequest) -> ToolResult:
        """执行"""
        start_time = time.time()
        
        if not self.validate(request):
            return self._create_result(
                success=False,
                error="Validation failed",
                execution_time=time.time() - start_time,
                request_id=request.request_id
            )
        
        try:
            action = request.action
            params = request.parameters
            
            if action == "read":
                result = self._read_file(params.get("path", ""))
            elif action == "write":
                result = self._write_file(
                    params.get("path", ""),
                    params.get("content", "")
                )
            elif action == "delete":
                result = self._delete_file(params.get("path", ""))
            elif action == "copy":
                result = self._copy_file(
                    params.get("src", ""),
                    params.get("dst", "")
                )
            elif action == "move":
                result = self._move_file(
                    params.get("src", ""),
                    params.get("dst", "")
                )
            elif action == "list_dir":
                result = self._list_dir(params.get("path", ""))
            elif action == "create_dir":
                result = self._create_dir(params.get("path", ""))
            elif action == "delete_dir":
                result = self._delete_dir(params.get("path", ""))
            elif action == "exists":
                result = self._exists(params.get("path", ""))
            else:
                result = {"success": False, "error": f"Unknown action: {action}"}
            
            return self._create_result(
                success=result.get("success", False),
                data=result.get("data"),
                error=result.get("error"),
                execution_time=time.time() - start_time,
                request_id=request.request_id
            )
            
        except Exception as e:
            logger.error(f"[FileTool] Execute error: {e}")
            return self._create_result(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
                request_id=request.request_id
            )
    
    def cleanup(self) -> None:
        """清理"""
        pass
    
    def _read_file(self, path: str) -> Dict[str, Any]:
        """读取文件"""
        try:
            file_path = Path(path)
            if not file_path.exists():
                return {"success": False, "error": f"File not found: {path}"}
            
            content = file_path.read_text(encoding='utf-8')
            return {
                "success": True,
                "data": {
                    "content": content,
                    "size": len(content),
                    "lines": content.count('\n') + 1
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _write_file(self, path: str, content: str) -> Dict[str, Any]:
        """写入文件"""
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
            
            return {
                "success": True,
                "data": {
                    "path": str(path),
                    "size": len(content)
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _delete_file(self, path: str) -> Dict[str, Any]:
        """删除文件"""
        try:
            file_path = Path(path)
            if file_path.exists():
                file_path.unlink()
                return {"success": True, "data": {"deleted": str(path)}}
            else:
                return {"success": False, "error": f"File not found: {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _copy_file(self, src: str, dst: str) -> Dict[str, Any]:
        """复制文件"""
        try:
            src_path = Path(src)
            dst_path = Path(dst)
            
            if not src_path.exists():
                return {"success": False, "error": f"Source not found: {src}"}
            
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            
            return {
                "success": True,
                "data": {"src": str(src), "dst": str(dst)}
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _move_file(self, src: str, dst: str) -> Dict[str, Any]:
        """移动文件"""
        try:
            src_path = Path(src)
            dst_path = Path(dst)
            
            if not src_path.exists():
                return {"success": False, "error": f"Source not found: {src}"}
            
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))
            
            return {
                "success": True,
                "data": {"src": str(src), "dst": str(dst)}
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _list_dir(self, path: str) -> Dict[str, Any]:
        """列出目录"""
        try:
            dir_path = Path(path)
            if not dir_path.exists() or not dir_path.is_dir():
                return {"success": False, "error": f"Directory not found: {path}"}
            
            items = []
            for item in dir_path.iterdir():
                items.append({
                    "name": item.name,
                    "path": str(item),
                    "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else 0
                })
            
            return {
                "success": True,
                "data": {
                    "path": str(path),
                    "count": len(items),
                    "items": items
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _create_dir(self, path: str) -> Dict[str, Any]:
        """创建目录"""
        try:
            dir_path = Path(path)
            dir_path.mkdir(parents=True, exist_ok=True)
            
            return {
                "success": True,
                "data": {"path": str(path)}
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _delete_dir(self, path: str) -> Dict[str, Any]:
        """删除目录"""
        try:
            dir_path = Path(path)
            if dir_path.exists() and dir_path.is_dir():
                shutil.rmtree(dir_path)
                return {"success": True, "data": {"deleted": str(path)}}
            else:
                return {"success": False, "error": f"Directory not found: {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _exists(self, path: str) -> Dict[str, Any]:
        """检查路径是否存在"""
        try:
            path_obj = Path(path)
            return {
                "success": True,
                "data": {
                    "exists": path_obj.exists(),
                    "is_file": path_obj.is_file(),
                    "is_dir": path_obj.is_dir()
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class NetworkTool(BaseTool):
    """网络工具
    
    功能:
    - HTTP GET/POST 请求
    - 下载文件
    - 检查网络连接
    """
    
    def __init__(self):
        super().__init__("network_tool", ToolCategory.NETWORK)
        self.description = "Network operation tool"
        self.version = "1.0.0"
        self.session = requests.Session()
    
    def initialize(self) -> bool:
        """初始化"""
        self.status = ToolStatus.READY
        logger.info("[NetworkTool] Initialized")
        return True
    
    def execute(self, request: ToolRequest) -> ToolResult:
        """执行"""
        start_time = time.time()
        
        if not self.validate(request):
            return self._create_result(
                success=False,
                error="Validation failed",
                execution_time=time.time() - start_time,
                request_id=request.request_id
            )
        
        try:
            action = request.action
            params = request.parameters
            
            if action == "get":
                result = self._http_get(
                    params.get("url", ""),
                    params.get("headers", {}),
                    params.get("timeout", 30)
                )
            elif action == "post":
                result = self._http_post(
                    params.get("url", ""),
                    params.get("data", {}),
                    params.get("headers", {}),
                    params.get("timeout", 30)
                )
            elif action == "download":
                result = self._download(
                    params.get("url", ""),
                    params.get("save_path", "")
                )
            elif action == "check_connection":
                result = self._check_connection()
            else:
                result = {"success": False, "error": f"Unknown action: {action}"}
            
            return self._create_result(
                success=result.get("success", False),
                data=result.get("data"),
                error=result.get("error"),
                execution_time=time.time() - start_time,
                request_id=request.request_id
            )
            
        except Exception as e:
            logger.error(f"[NetworkTool] Execute error: {e}")
            return self._create_result(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
                request_id=request.request_id
            )
    
    def cleanup(self) -> None:
        """清理"""
        self.session.close()
    
    def _http_get(
        self,
        url: str,
        headers: Dict = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """HTTP GET 请求"""
        try:
            response = self.session.get(
                url,
                headers=headers,
                timeout=timeout
            )
            
            return {
                "success": True,
                "data": {
                    "status_code": response.status_code,
                    "content": response.text,
                    "headers": dict(response.headers)
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _http_post(
        self,
        url: str,
        data: Dict = None,
        headers: Dict = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """HTTP POST 请求"""
        try:
            response = self.session.post(
                url,
                json=data,
                headers=headers,
                timeout=timeout
            )
            
            return {
                "success": True,
                "data": {
                    "status_code": response.status_code,
                    "content": response.text,
                    "headers": dict(response.headers)
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _download(self, url: str, save_path: str) -> Dict[str, Any]:
        """下载文件"""
        try:
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path_obj, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return {
                "success": True,
                "data": {
                    "url": url,
                    "save_path": str(save_path),
                    "size": save_path_obj.stat().st_size
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _check_connection(self) -> Dict[str, Any]:
        """检查网络连接"""
        try:
            # 尝试访问 Google DNS
            self.session.get("https://8.8.8.8", timeout=5)
            return {
                "success": True,
                "data": {"connected": True}
            }
        except Exception:
            return {
                "success": True,
                "data": {"connected": False}
            }


class SystemCommandTool(BaseTool):
    """系统命令工具
    
    功能:
    - 执行 Shell 命令
    - 获取系统信息
    - 进程管理
    """
    
    def __init__(self):
        super().__init__("system_command_tool", ToolCategory.SYSTEM)
        self.description = "System command execution tool"
        self.version = "1.0.0"
    
    def initialize(self) -> bool:
        """初始化"""
        self.status = ToolStatus.READY
        logger.info("[SystemCommandTool] Initialized")
        return True
    
    def execute(self, request: ToolRequest) -> ToolResult:
        """执行"""
        start_time = time.time()
        
        if not self.validate(request):
            return self._create_result(
                success=False,
                error="Validation failed",
                execution_time=time.time() - start_time,
                request_id=request.request_id
            )
        
        try:
            action = request.action
            params = request.parameters
            
            if action == "run":
                result = self._run_command(
                    params.get("command", ""),
                    params.get("cwd", None),
                    params.get("timeout", 30)
                )
            elif action == "get_system_info":
                result = self._get_system_info()
            elif action == "list_processes":
                result = self._list_processes()
            else:
                result = {"success": False, "error": f"Unknown action: {action}"}
            
            return self._create_result(
                success=result.get("success", False),
                data=result.get("data"),
                error=result.get("error"),
                execution_time=time.time() - start_time,
                request_id=request.request_id
            )
            
        except Exception as e:
            logger.error(f"[SystemCommandTool] Execute error: {e}")
            return self._create_result(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
                request_id=request.request_id
            )
    
    def cleanup(self) -> None:
        """清理"""
        pass
    
    def _run_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """执行系统命令"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
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
                "error": f"Command timeout after {timeout}s"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        try:
            import platform
            
            return {
                "success": True,
                "data": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                    "python_version": platform.python_version()
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _list_processes(self) -> Dict[str, Any]:
        """列出进程（简化版）"""
        try:
            # Windows
            if os.name == 'nt':
                result = subprocess.run(
                    "tasklist",
                    capture_output=True,
                    text=True
                )
            else:
                # Linux/Mac
                result = subprocess.run(
                    "ps aux",
                    capture_output=True,
                    text=True
                )
            
            return {
                "success": result.returncode == 0,
                "data": {
                    "output": result.stdout
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
