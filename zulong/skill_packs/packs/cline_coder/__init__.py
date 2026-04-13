# File: zulong/skill_packs/packs/cline_coder/__init__.py
"""
Cline 编程技能包（基础版）

从 Cline 开源编程助手提取的核心能力：
- 文件读写操作
- 代码生成与修改
- 终端命令执行（基础）
- 代码搜索（基础）

安全策略：
- 所有文件操作限制在 workspace 目录内
- 终端命令执行有白名单限制
- 禁止执行 rm -rf、format 等危险命令
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import os
import subprocess
import time

from zulong.skill_packs.interface import ISkillPack, SkillPackManifest
from zulong.tools.base import BaseTool, ToolCategory, ToolRequest, ToolResult

logger = logging.getLogger(__name__)


class FileReadTool(BaseTool):
    """文件读取工具"""
    
    def __init__(self):
        super().__init__(name="read_file", category=ToolCategory.CODE)
        self.description = "读取文件内容。参数：file_path(文件路径), max_lines(最大行数，默认100)"
    
    def initialize(self):
        return True
    
    def cleanup(self):
        pass
    
    def execute(self, request: ToolRequest) -> ToolResult:
        file_path = request.parameters.get("file_path", "")
        max_lines = request.parameters.get("max_lines", 100)
        
        if not file_path:
            return self._create_result(success=False, error="缺少文件路径", request_id=request.request_id)
        
        try:
            path = Path(file_path)
            if not path.exists():
                return self._create_result(success=False, error=f"文件不存在: {file_path}", request_id=request.request_id)
            
            if not path.is_file():
                return self._create_result(success=False, error=f"不是文件: {file_path}", request_id=request.request_id)
            
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 限制读取行数
            if len(lines) > max_lines:
                content = ''.join(lines[:max_lines])
                content += f"\n\n... (文件还有 {len(lines) - max_lines} 行，使用 max_lines 参数读取更多)"
            else:
                content = ''.join(lines)
            
            return self._create_result(
                success=True,
                data={
                    "file_path": str(path),
                    "content": content,
                    "total_lines": len(lines),
                    "size_bytes": path.stat().st_size
                },
                request_id=request.request_id
            )
        except Exception as e:
            return self._create_result(success=False, error=str(e), request_id=request.request_id)
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "文件绝对或相对路径"},
                "max_lines": {"type": "integer", "description": "最大读取行数，默认 100"}
            },
            "required": ["file_path"]
        }


class FileWriteTool(BaseTool):
    """文件写入工具"""
    
    def __init__(self, workspace: str = "."):
        super().__init__(name="write_file", category=ToolCategory.CODE)
        self.description = "写入文件内容。参数：file_path(文件路径), content(内容), mode(模式: write/append，默认 write)"
        self.workspace = Path(workspace).resolve()
    
    def initialize(self):
        self.workspace.mkdir(parents=True, exist_ok=True)
        return True
    
    def cleanup(self):
        pass
    
    def execute(self, request: ToolRequest) -> ToolResult:
        file_path = request.parameters.get("file_path", "")
        content = request.parameters.get("content", "")
        mode = request.parameters.get("mode", "write")
        
        if not file_path:
            return self._create_result(success=False, error="缺少文件路径", request_id=request.request_id)
        
        try:
            # 安全检查：限制在工作目录内
            path = Path(file_path).resolve()
            if not str(path).startswith(str(self.workspace)):
                return self._create_result(
                    success=False,
                    error=f"安全限制：文件路径必须在工作目录内: {self.workspace}",
                    request_id=request.request_id
                )
            
            # 确保父目录存在
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # 写入文件
            write_mode = 'a' if mode == 'append' else 'w'
            with open(path, write_mode, encoding='utf-8') as f:
                f.write(content)
            
            return self._create_result(
                success=True,
                data={
                    "file_path": str(path),
                    "bytes_written": len(content.encode('utf-8')),
                    "mode": mode
                },
                request_id=request.request_id
            )
        except Exception as e:
            return self._create_result(success=False, error=str(e), request_id=request.request_id)
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "文件绝对或相对路径"},
                "content": {"type": "string", "description": "要写入的内容"},
                "mode": {"type": "string", "description": "写入模式：write(覆盖) 或 append(追加)", "enum": ["write", "append"]}
            },
            "required": ["file_path", "content"]
        }


class CodeEditTool(BaseTool):
    """代码编辑工具"""
    
    def __init__(self, workspace: str = "."):
        super().__init__(name="edit_code", category=ToolCategory.CODE)
        self.description = "精确编辑代码文件。参数：file_path(文件路径), old_str(要替换的代码), new_str(新代码)"
        self.workspace = Path(workspace).resolve()
    
    def initialize(self):
        return True
    
    def cleanup(self):
        pass
    
    def execute(self, request: ToolRequest) -> ToolResult:
        file_path = request.parameters.get("file_path", "")
        old_str = request.parameters.get("old_str", "")
        new_str = request.parameters.get("new_str", "")
        
        if not file_path or not old_str:
            return self._create_result(success=False, error="缺少必要参数", request_id=request.request_id)
        
        try:
            path = Path(file_path).resolve()
            
            # 安全检查
            if not str(path).startswith(str(self.workspace)):
                return self._create_result(
                    success=False,
                    error=f"安全限制：文件路径必须在工作目录内: {self.workspace}",
                    request_id=request.request_id
                )
            
            if not path.exists():
                return self._create_result(success=False, error=f"文件不存在: {file_path}", request_id=request.request_id)
            
            # 读取文件
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 替换代码
            if old_str not in content:
                return self._create_result(
                    success=False,
                    error="未找到要替换的代码字符串",
                    data={"hint": "请确保 old_str 与文件中的内容完全匹配（包括缩进和换行）"},
                    request_id=request.request_id
                )
            
            new_content = content.replace(old_str, new_str, 1)
            
            # 写回文件
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return self._create_result(
                success=True,
                data={
                    "file_path": str(path),
                    "replacements": 1,
                    "old_length": len(old_str),
                    "new_length": len(new_str)
                },
                request_id=request.request_id
            )
        except Exception as e:
            return self._create_result(success=False, error=str(e), request_id=request.request_id)
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "要编辑的文件路径"},
                "old_str": {"type": "string", "description": "要替换的代码字符串（必须完全匹配）"},
                "new_str": {"type": "string", "description": "新的代码字符串"}
            },
            "required": ["file_path", "old_str", "new_str"]
        }


class TerminalTool(BaseTool):
    """终端命令执行工具"""
    
    # 命令白名单
    ALLOWED_COMMANDS = [
        "python", "python3", "pip", "pip3",
        "ls", "dir", "pwd", "cd",
        "cat", "type", "head", "tail",
        "grep", "find",
        "echo",
        "git",
        "node", "npm", "npx",
        "pytest", "unittest",
        "make", "cmake",
    ]
    
    def __init__(self):
        super().__init__(name="run_command", category=ToolCategory.CODE)
        self.description = "执行终端命令。参数：command(命令), args(参数列表), timeout(超时秒数，默认 30)"
    
    def initialize(self):
        return True
    
    def cleanup(self):
        pass
    
    def execute(self, request: ToolRequest) -> ToolResult:
        command = request.parameters.get("command", "")
        args = request.parameters.get("args", [])
        timeout = request.parameters.get("timeout", 30)
        
        if not command:
            return self._create_result(success=False, error="缺少命令", request_id=request.request_id)
        
        # 安全检查：命令白名单
        if command not in self.ALLOWED_COMMANDS:
            return self._create_result(
                success=False,
                error=f"命令不在白名单: {command}。允许的命令: {', '.join(self.ALLOWED_COMMANDS)}",
                request_id=request.request_id
            )
        
        try:
            # 构建完整命令
            cmd = [command] + (args if isinstance(args, list) else [args])
            
            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8'
            )
            
            return self._create_result(
                success=True,
                data={
                    "command": ' '.join(cmd),
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                    "duration": timeout
                },
                request_id=request.request_id
            )
        except subprocess.TimeoutExpired:
            return self._create_result(success=False, error=f"命令执行超时 ({timeout}秒)", request_id=request.request_id)
        except Exception as e:
            return self._create_result(success=False, error=str(e), request_id=request.request_id)
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的命令（必须在白名单内）"},
                "args": {"type": "array", "description": "命令参数列表", "items": {"type": "string"}},
                "timeout": {"type": "integer", "description": "超时时间（秒），默认 30"}
            },
            "required": ["command"]
        }


class CodeSearchTool(BaseTool):
    """代码搜索工具"""
    
    def __init__(self, workspace: str = "."):
        super().__init__(name="search_code", category=ToolCategory.CODE)
        self.description = "在代码文件中搜索内容。参数：pattern(搜索模式), directory(目录，默认工作目录), file_pattern(文件模式，如 *.py)"
        self.workspace = Path(workspace).resolve()
    
    def initialize(self):
        return True
    
    def cleanup(self):
        pass
    
    def execute(self, request: ToolRequest) -> ToolResult:
        pattern = request.parameters.get("pattern", "")
        directory = request.parameters.get("directory", str(self.workspace))
        file_pattern = request.parameters.get("file_pattern", "*.py")
        
        if not pattern:
            return self._create_result(success=False, error="缺少搜索模式", request_id=request.request_id)
        
        try:
            search_dir = Path(directory).resolve()
            if not search_dir.exists():
                return self._create_result(success=False, error=f"目录不存在: {directory}", request_id=request.request_id)
            
            # 查找匹配的文件
            matches = []
            for file_path in search_dir.rglob(file_pattern):
                if file_path.is_file():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if pattern in content:
                            # 找到匹配的行
                            matching_lines = []
                            for i, line in enumerate(content.split('\n'), 1):
                                if pattern in line:
                                    matching_lines.append({
                                        "line_number": i,
                                        "content": line.strip()
                                    })
                            
                            matches.append({
                                "file": str(file_path),
                                "matches": matching_lines[:20]  # 限制每个文件 20 条
                            })
                    except Exception:
                        pass
            
            return self._create_result(
                success=True,
                data={
                    "pattern": pattern,
                    "file_pattern": file_pattern,
                    "directory": str(search_dir),
                    "total_files_matched": len(matches),
                    "matches": matches[:10]  # 限制返回 10 个文件
                },
                request_id=request.request_id
            )
        except Exception as e:
            return self._create_result(success=False, error=str(e), request_id=request.request_id)
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "要搜索的文本模式"},
                "directory": {"type": "string", "description": "搜索目录，默认工作目录"},
                "file_pattern": {"type": "string", "description": "文件匹配模式，如 *.py"}
            },
            "required": ["pattern"]
        }


class ClineCoderPack(ISkillPack):
    """Cline 编程技能包
    
    提供完整的代码操作能力：
    - 文件读写
    - 代码编辑
    - 终端命令执行
    - 代码搜索
    """
    
    def __init__(self):
        self._manifest = None
        self._workspace = "."
        self._tools = []
        logger.info("[ClineCoderPack] Initialized")
    
    def get_manifest(self) -> SkillPackManifest:
        """返回技能包清单"""
        if self._manifest is None:
            self._manifest = SkillPackManifest(
                pack_id="cline_coder",
                name="Cline编程",
                version="1.0.0",
                description="基于 Cline 的编程助手能力，支持文件操作、代码编辑、命令执行",
                capabilities=["read_file", "write_file", "edit_code", "run_command", "search_code"],
                dependencies=[],
                resource_requirements={"cpu_mb": 128, "gpu_mb": 0},
                learning_objectives=["代码生成模式", "错误修复策略", "文件操作规范"],
                source="cline"
            )
        return self._manifest
    
    def install(self, tool_registry, config: Dict[str, Any] = None) -> bool:
        """安装技能包，注册工具"""
        try:
            self._workspace = config.get("workspace", ".") if config else "."
            
            # 创建工具
            self._tools = [
                FileReadTool(),
                FileWriteTool(self._workspace),
                CodeEditTool(self._workspace),
                TerminalTool(),
                CodeSearchTool(self._workspace)
            ]
            
            # 注册工具
            for tool in self._tools:
                try:
                    tool_registry.register(tool)
                except Exception as e:
                    logger.warning(f"[ClineCoderPack] 工具注册失败: {e}")
            
            logger.info(f"[ClineCoderPack] 已注册 {len(self._tools)} 个工具")
            return True
        except Exception as e:
            logger.error(f"[ClineCoderPack] Install failed: {e}", exc_info=True)
            return False
    
    def execute(self, capability: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行技能包能力"""
        capability_map = {
            "read_file": (FileReadTool, {}),
            "write_file": (FileWriteTool, {"workspace": self._workspace}),
            "edit_code": (CodeEditTool, {"workspace": self._workspace}),
            "run_command": (TerminalTool, {}),
            "search_code": (CodeSearchTool, {"workspace": self._workspace}),
        }
        
        if capability not in capability_map:
            return {
                "success": False,
                "error": f"Unknown capability: {capability}"
            }
        
        tool_class, kwargs = capability_map[capability]
        tool = tool_class(**kwargs)
        tool.initialize()
        
        # 模拟工具请求
        request = ToolRequest(
            tool_name=tool.name,
            action=capability,
            parameters=params
        )
        
        result = tool.execute(request)
        return result.to_dict()
    
    def get_tools(self):
        """返回提供的工具列表"""
        return self._tools
    
    def uninstall(self) -> bool:
        """卸载技能包"""
        self._tools = []
        logger.info("[ClineCoderPack] Uninstalled")
        return True
