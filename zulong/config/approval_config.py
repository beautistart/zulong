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

    def add_rule(self, rule: str) -> Optional[dict]:
        """Add a whitelist rule from Web approval UI.

        Supported rule formats:
        - dir:<path>
        - tool:<tool_name>
        - command:<command>
        - pattern:<regex>
        """
        parsed = _parse_whitelist_rule(rule)
        if not parsed:
            return None

        key, value = parsed
        current = list(getattr(self, key))
        if value in current:
            self._loaded = True
            return {"kind": key, "value": value, "changed": False}

        current.append(value)
        setattr(self, key, current)
        self._loaded = True
        self._save()
        return {"kind": key, "value": value, "changed": True}

    def _save(self) -> None:
        """Persist current whitelist rules to YAML."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "directories": self.directories,
            "commands": self.commands,
            "tools": self.tools,
            "patterns": self.patterns,
        }
        with open(self._config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _normalize_directory_rule(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    if text.endswith("/"):
        return text
    if "/" in text:
        leaf = text.rsplit("/", 1)[-1]
        if "." in leaf:
            return text.rsplit("/", 1)[0] + "/"
        return text + "/"
    return text


def _parse_whitelist_rule(rule: str) -> Optional[tuple]:
    text = str(rule or "").strip()
    if not text or ":" not in text:
        return None
    kind, value = text.split(":", 1)
    kind = kind.strip().lower()
    value = value.strip()
    if not value:
        return None
    if kind in {"dir", "directory", "path"}:
        value = _normalize_directory_rule(value)
        return ("directories", value) if value else None
    if kind == "tool":
        return ("tools", value)
    if kind in {"cmd", "command"}:
        return ("commands", value)
    if kind in {"pattern", "regex"}:
        return ("patterns", value)
    return None


# 全局单例
_whitelist_instance: Optional[ApprovalWhitelist] = None
_VALID_APPROVAL_MODES = {"full_auto", "whitelist", "manual", "popup"}
_runtime_approval_mode = os.environ.get("ZULONG_APPROVAL_MODE", "manual")
if _runtime_approval_mode not in _VALID_APPROVAL_MODES:
    _runtime_approval_mode = "manual"


def get_approval_whitelist() -> ApprovalWhitelist:
    """获取白名单单例"""
    global _whitelist_instance
    if _whitelist_instance is None:
        _whitelist_instance = ApprovalWhitelist()
    return _whitelist_instance


def reload_approval_whitelist() -> None:
    """热加载白名单"""
    get_approval_whitelist().reload()


def add_approval_whitelist_rule(rule: str) -> Optional[dict]:
    """Add one Web approval whitelist rule and persist it."""
    return get_approval_whitelist().add_rule(rule)


def set_runtime_approval_mode(mode: str) -> str:
    """Set Web runtime approval mode defined by TSD §23.4."""
    global _runtime_approval_mode
    normalized = str(mode or "").strip().lower()
    if normalized not in _VALID_APPROVAL_MODES:
        normalized = "manual"
    _runtime_approval_mode = normalized
    return _runtime_approval_mode


def get_runtime_approval_mode() -> str:
    """Return current Web runtime approval mode."""
    return _runtime_approval_mode


def should_runtime_auto_approve(
    tool_name: str,
    tool_args: Optional[dict] = None,
    *,
    risk_level: str = "",
) -> bool:
    """Decide whether an IDE approval request should be auto-approved.

    TSD §23.4 exposes four modes: full_auto, whitelist, manual and popup.
    The Web frontend owns the current mode, while the IDE bridge enforces it
    when a concrete approval request arrives.
    """
    mode = get_runtime_approval_mode()
    if mode == "full_auto":
        return True
    if mode != "whitelist":
        return False
    risk = str(risk_level or "").lower()
    if risk in {"critical", "danger", "popup"}:
        return False
    return get_approval_whitelist().should_auto_approve(tool_name, tool_args or {})
