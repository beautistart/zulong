# File: zulong/skill_packs/interface.py
"""
技能包统一接口定义

所有技能包必须实现 ISkillPack 接口，确保：
- 统一的安装/执行/卸载流程
- 工具自动注册到 ToolEngine
- 经验自动提取到 ExperienceStore
- 内化完成度可评估
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional

from zulong.tools.base import BaseTool, ToolRegistry


class SkillPackStatus(Enum):
    """技能包状态"""
    AVAILABLE = "available"        # 可安装（已下载但未加载）
    INSTALLED = "installed"        # 已安装（已加载，工具已注册）
    LEARNING = "learning"          # 学习中（正在积累经验）
    INTERNALIZED = "internalized"  # 已内化（经验充足，可卸载）
    UNINSTALLED = "uninstalled"    # 已卸载（经验保留）


@dataclass
class SkillPackManifest:
    """技能包清单
    
    描述技能包的基本信息、能力、依赖和资源需求。
    """
    pack_id: str                                    # 唯一标识，如 "autogpt_planner"
    name: str                                       # 显示名称，如 "AutoGPT任务拆解"
    version: str = "1.0.0"                         # 版本号
    description: str = ""                           # 描述
    capabilities: List[str] = field(default_factory=list)  # 能力列表，如 ["task_decompose", "priority_rank"]
    dependencies: List[str] = field(default_factory=list)  # 依赖的Python包
    resource_requirements: Dict[str, int] = field(default_factory=lambda: {"cpu_mb": 512, "gpu_mb": 0})
    learning_objectives: List[str] = field(default_factory=list)  # 学习目标，用于内化评估
    source: str = "custom"                          # 来源: "autogpt" / "openmanus" / "cline" / "custom"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "pack_id": self.pack_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": self.capabilities,
            "dependencies": self.dependencies,
            "resource_requirements": self.resource_requirements,
            "learning_objectives": self.learning_objectives,
            "source": self.source,
        }


class ISkillPack(ABC):
    """技能包接口
    
    所有技能包必须实现此接口。
    技能包是"借用→学习→内化→丢弃"生命周期的核心载体。
    """
    
    @abstractmethod
    def get_manifest(self) -> SkillPackManifest:
        """返回技能包清单
        
        描述技能包的能力、依赖、资源需求等。
        在 install() 之前调用，用于验证是否满足安装条件。
        """
        pass
    
    @abstractmethod
    def install(self, tool_registry: ToolRegistry, config: Optional[Dict[str, Any]] = None) -> bool:
        """安装技能包
        
        Args:
            tool_registry: 工具注册表，技能包的工具将注册到这里
            config: 可选的配置参数
        
        Returns:
            bool: 是否安装成功
        """
        pass
    
    @abstractmethod
    def execute(self, capability: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行技能包提供的某个能力
        
        Args:
            capability: 能力名称（来自 manifest.capabilities）
            params: 执行参数
        
        Returns:
            执行结果字典
        """
        pass
    
    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """返回此技能包提供的工具列表
        
        这些工具将自动注册到 ToolEngine，
        使得 InferenceEngine 的 Function Calling 可以调用它们。
        """
        pass
    
    @abstractmethod
    def uninstall(self) -> bool:
        """卸载技能包
        
        注销工具、释放资源。
        注意：经验数据不删除，继续保留在 ExperienceStore 中。
        
        Returns:
            bool: 是否卸载成功
        """
        pass
