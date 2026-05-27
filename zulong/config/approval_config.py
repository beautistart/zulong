"""
TSD v1.7 §23.4.3: 审批白名单配置加载器
从 config/approval_whitelist.yaml 加载白名单规则并提供匹配检查
"""
import re
import os
from pathlib import Path
from typing import List, Optional

import yaml


class ApprovalWhitelist:
    """审批白名单管理器"""

    def __init__(self, config_path: Optional[Path] = None):
        self.directories: List[str] = []
        self.commands: List[str] = []
        self.tools: List[str] = []
        self.patterns: List[str] = []
        self._loaded = False

        if config_path is None:
            # 默认路径: 项目根目录/config/approval_whitelist.yaml
            project_root = Path(os.environ.get("ZULONG_HOME", Path(__file__).parent.parent.parent))
            config_path = project_root / "config" / "approval_whitelist.yaml"

        self._config_path = config_path
        self.load()

    def load(self) -> None:
        """从 YAML 文件加载白名单"""
        if not self._config_path.exists():
            return

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            self.directories = data.get("directories", [])
            self.commands = data.get("commands", [])
            self.tools = data.get("tools", [])
            self.patterns = data.get("patterns", [])
            self._loaded = True
        except Exception:
            # 加载失败时使用空白名单（保守策略: 全部手动审批）
            pass

    def reload(self) -> None:
        """热加载白名单配置"""
        self.load()

    def is_tool_whitelisted(self, tool_name: str) -> bool:
        """检查工具是否在白名单中"""
        if not self._loaded:
            return False
        return tool_name in self.tools

    def is_command_whitelisted(self, command: str) -> bool:
        """检查命令是否在白名单中"""
        if not self._loaded:
            return False
        if command in self.commands:
            return True
        for pattern in self.patterns:
            if re.match(pattern, command):
                return True
        return False

    def is_directory_whitelisted(self, path: str) -> bool:
        """检查路径是否在白名单目录下"""
        if not self._loaded:
            return False
        for directory in self.directories:
            if path.startswith(directory):
                return True
        return False

    def should_auto_approve(self, tool_name: str, tool_args: Optional[dict] = None) -> bool:
        """综合判断工具调用是否应该自动放行

        检查优先级:
        1. 工具名在白名单 → 自动放行
        2. 路径在白名单目录 → 自动放行
        3. 命令在白名单 → 自动放行
        """
        if not self._loaded:
            return False

        if self.is_tool_whitelisted(tool_name):
            return True

        if tool_args:
            path = tool_args.get("path", tool_args.get("filePath", ""))
            if path and self.is_directory_whitelisted(str(path)):
                return True

            command = tool_args.get("command", "")
            if command and self.is_command_whitelisted(str(command)):
                return True

        return False


# 全局单例
_whitelist_instance: Optional[ApprovalWhitelist] = None


def get_approval_whitelist() -> ApprovalWhitelist:
    """获取白名单单例"""
    global _whitelist_instance
    if _whitelist_instance is None:
        _whitelist_instance = ApprovalWhitelist()
    return _whitelist_instance


def reload_approval_whitelist() -> None:
    """热加载白名单"""
    get_approval_whitelist().reload()
