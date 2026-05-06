"""
模块基类和状态枚举

每个功能模块继承 Module 基类，声明依赖和模式标签，
由 ModuleManager 按拓扑序编排启动。
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


class ModuleState(Enum):
    """模块生命周期状态"""
    UNLOADED = "unloaded"   # 已注册但未启动
    STARTING = "starting"   # 正在启动
    RUNNING  = "running"    # 正常运行
    STOPPING = "stopping"   # 正在停止
    STOPPED  = "stopped"    # 已停止
    ERROR    = "error"      # 出错


class Module(ABC):
    """功能模块抽象基类"""

    # 子类必须设置
    name: str = ""                          # 唯一标识
    display_name: str = ""                  # 中文显示名
    dependencies: list = []                 # 依赖的模块 name 列表
    mode_tags: Set[str] = set()             # {"core"} / {"full"} / {"optional"}

    def __init__(self):
        self.state = ModuleState.UNLOADED
        self.error_message: str = ""
        self.progress_message: str = ""
        # 共享上下文（由 ModuleManager 注入）
        self._context: Dict[str, Any] = {}

    def set_context(self, ctx: Dict[str, Any]) -> None:
        self._context = ctx

    @abstractmethod
    async def start(self) -> None:
        """初始化并启动模块。成功后应设置 self.state = RUNNING"""
        ...

    async def stop(self) -> None:
        """优雅停止。默认实现仅更新状态。"""
        self.state = ModuleState.STOPPED

    def to_status(self) -> Dict[str, Any]:
        """返回当前状态快照（供 API 使用）"""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "state": self.state.value,
            "optional": "optional" in self.mode_tags,
            "error": self.error_message or None,
            "message": self.progress_message or None,
        }
