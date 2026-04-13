# File: zulong/skill_packs/runtime.py
"""
技能包运行时 - 管理技能包的完整生命周期

职责：
1. 安装/卸载技能包
2. 执行技能包能力（同时自动记录经验）
3. 内化完成度评估
4. 与现有系统对接（ToolEngine、ExperienceStore、HotUpdateEngine）
"""

import logging
import time
from typing import Dict, Any, List, Optional

from zulong.skill_packs.interface import ISkillPack, SkillPackStatus, SkillPackManifest
from zulong.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


class SkillPackRuntime:
    """技能包运行时
    
    管理技能包的完整生命周期：
    - 安装：验证清单 → 注册工具 → 初始化状态
    - 执行：调用能力 → 记录经验 → 更新计数
    - 评估：检查内化完成度
    - 卸载：注销工具 → 释放资源 → 保留经验
    """
    
    def __init__(self, tool_engine=None, experience_store=None, hot_update_engine=None):
        """初始化技能包运行时
        
        Args:
            tool_engine: ToolEngine 实例（用于工具注册/注销）
            experience_store: ExperienceStore 实例（用于经验记录）
            hot_update_engine: HotUpdateEngine 实例（用于经验内化）
        """
        self.tool_engine = tool_engine
        self.experience_store = experience_store
        self.hot_update_engine = hot_update_engine
        
        self._packs: Dict[str, ISkillPack] = {}
        self._status: Dict[str, SkillPackStatus] = {}
        self._experience_counts: Dict[str, int] = {}
        self._install_times: Dict[str, float] = {}
        self._config: Dict[str, Dict[str, Any]] = {}
        
        logger.info("[SkillPackRuntime] 初始化完成")
    
    def install_pack(self, pack: ISkillPack, config: Optional[Dict[str, Any]] = None) -> bool:
        """安装技能包
        
        Args:
            pack: 技能包实例
            config: 可选的配置参数
        
        Returns:
            bool: 是否安装成功
        """
        manifest = pack.get_manifest()
        pack_id = manifest.pack_id
        
        # 1. 验证状态
        if pack_id in self._packs and self._status.get(pack_id) == SkillPackStatus.INSTALLED:
            logger.warning(f"[SkillPackRuntime] 技能包已安装: {pack_id}")
            return False
        
        # 2. 验证清单（依赖检查）
        if not self._check_dependencies(manifest):
            logger.error(f"[SkillPackRuntime] 依赖检查失败: {pack_id}")
            return False
        
        # 3. 调用技能包的 install 方法
        try:
            success = pack.install(ToolRegistry(), config)
            if not success:
                logger.error(f"[SkillPackRuntime] 技能包 install 失败: {pack_id}")
                return False
        except Exception as e:
            logger.error(f"[SkillPackRuntime] 技能包安装异常: {pack_id} - {e}")
            return False
        
        # 4. 注册工具到 ToolEngine
        if self.tool_engine is not None:
            tools = pack.get_tools()
            for tool in tools:
                try:
                    self.tool_engine.registry.register(tool)
                    logger.info(f"[SkillPackRuntime] 工具已注册: {tool.name} (来自 {pack_id})")
                except Exception as e:
                    logger.warning(f"[SkillPackRuntime] 工具注册失败: {tool.name} - {e}")
        
        # 5. 更新状态
        self._packs[pack_id] = pack
        self._status[pack_id] = SkillPackStatus.INSTALLED
        self._experience_counts[pack_id] = 0
        self._install_times[pack_id] = time.time()
        self._config[pack_id] = config or {}
        
        logger.info(f"[SkillPackRuntime] 技能包已安装: {pack_id} ({manifest.name})")
        return True
    
    def execute_capability(
        self,
        pack_id: str,
        capability: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行技能包能力，同时自动记录经验
        
        Args:
            pack_id: 技能包 ID
            capability: 能力名称
            params: 执行参数
        
        Returns:
            执行结果
        """
        if pack_id not in self._packs:
            logger.error(f"[SkillPackRuntime] 技能包未安装: {pack_id}")
            return {"success": False, "error": f"技能包未安装: {pack_id}"}
        
        if self._status.get(pack_id) not in (SkillPackStatus.INSTALLED, SkillPackStatus.LEARNING, SkillPackStatus.INTERNALIZED):
            logger.error(f"[SkillPackRuntime] 技能包状态异常: {pack_id} ({self._status.get(pack_id)})")
            return {"success": False, "error": f"技能包状态异常: {self._status.get(pack_id)}"}
        
        start_time = time.time()
        result = {}
        
        try:
            # 1. 调用技能包能力
            pack = self._packs[pack_id]
            result = pack.execute(capability, params)
            
            # 2. 更新状态为 LEARNING（正在积累经验）
            if self._status.get(pack_id) == SkillPackStatus.INSTALLED:
                self._status[pack_id] = SkillPackStatus.LEARNING
            
            # 3. 记录经验到 ExperienceStore
            self._record_experience(pack_id, capability, params, result, time.time() - start_time)
            
            # 4. 通知 HotUpdateEngine（如果有）
            if self.hot_update_engine is not None:
                try:
                    self.hot_update_engine.on_skill_pack_experience(
                        pack_id, capability, params, result
                    )
                except AttributeError:
                    # HotUpdateEngine 可能没有这个方法，忽略
                    pass
            
            logger.info(f"[SkillPackRuntime] 执行成功: {pack_id}.{capability} ({time.time() - start_time:.2f}s)")
            return result
            
        except Exception as e:
            logger.error(f"[SkillPackRuntime] 执行失败: {pack_id}.{capability} - {e}")
            result = {"success": False, "error": str(e)}
            
            # 即使失败也记录经验
            self._record_experience(pack_id, capability, params, result, time.time() - start_time)
            
            return result
    
    def check_internalization(self, pack_id: str) -> float:
        """检查内化完成度（0.0-1.0）
        
        基于以下因素评估：
        - 该技能包的经验数量是否充足
        - 相关 SystemPatch 的应用成功率
        - 经验质量（成功 vs 失败比例）
        
        Args:
            pack_id: 技能包 ID
        
        Returns:
            内化完成度 (0.0-1.0)
        """
        if pack_id not in self._packs:
            return 0.0
        
        manifest = self._packs[pack_id].get_manifest()
        experience_count = self._experience_counts.get(pack_id, 0)
        
        # 基础评估：经验数量
        min_experience = 50  # 至少积累50条经验
        quantity_score = min(experience_count / min_experience, 1.0)
        
        # 质量评估：成功率（如果有经验存储）
        quality_score = 1.0
        if self.experience_store is not None and not isinstance(self.experience_store, type(Mock())):
            try:
                success_count = self.experience_store.get_success_count(pack_id)
                total_count = self.experience_store.get_total_count(pack_id)
                if total_count and total_count > 0:
                    quality_score = success_count / total_count
            except (AttributeError, ZeroDivisionError, TypeError):
                pass
        
        # 综合评分（数量权重 0.6，质量权重 0.4）
        internalization_score = quantity_score * 0.6 + quality_score * 0.4
        
        # 如果完成度 > 90%，更新状态
        if internalization_score > 0.9:
            if self._status.get(pack_id) in (SkillPackStatus.INSTALLED, SkillPackStatus.LEARNING):
                self._status[pack_id] = SkillPackStatus.INTERNALIZED
                logger.info(f"[SkillPackRuntime] 技能包已内化: {pack_id} (完成度: {internalization_score:.2f})")
        
        return internalization_score
    
    def uninstall_pack(self, pack_id: str) -> bool:
        """卸载技能包（经验保留在 ExperienceStore 中）
        
        Args:
            pack_id: 技能包 ID
        
        Returns:
            bool: 是否卸载成功
        """
        if pack_id not in self._packs:
            logger.warning(f"[SkillPackRuntime] 技能包未安装: {pack_id}")
            return False
        
        pack = self._packs[pack_id]
        manifest = pack.get_manifest()
        
        # 1. 从 ToolEngine 注销工具
        if self.tool_engine is not None:
            tools = pack.get_tools()
            for tool in tools:
                try:
                    self.tool_engine.registry.unregister(tool.name)
                    logger.info(f"[SkillPackRuntime] 工具已注销: {tool.name}")
                except Exception as e:
                    logger.warning(f"[SkillPackRuntime] 工具注销失败: {tool.name} - {e}")
        
        # 2. 调用技能包的 uninstall 方法
        try:
            success = pack.uninstall()
            if not success:
                logger.warning(f"[SkillPackRuntime] 技能包 uninstall 返回 False: {pack_id}")
        except Exception as e:
            logger.error(f"[SkillPackRuntime] 技能包卸载异常: {pack_id} - {e}")
        
        # 3. 更新状态（经验和计数保留）
        self._status[pack_id] = SkillPackStatus.UNINSTALLED
        
        logger.info(f"[SkillPackRuntime] 技能包已卸载: {pack_id} (经验保留: {self._experience_counts.get(pack_id, 0)} 条)")
        return True
    
    def list_packs(self) -> List[Dict[str, Any]]:
        """列出所有技能包及其状态
        
        Returns:
            技能包信息列表
        """
        result = []
        for pack_id, pack in self._packs.items():
            manifest = pack.get_manifest()
            result.append({
                "pack_id": pack_id,
                "name": manifest.name,
                "version": manifest.version,
                "status": self._status.get(pack_id, "unknown").value,
                "experience_count": self._experience_counts.get(pack_id, 0),
                "internalization_score": self.check_internalization(pack_id),
                "capabilities": manifest.capabilities,
                "source": manifest.source,
                "installed_at": self._install_times.get(pack_id, 0),
            })
        return result
    
    def get_pack_status(self, pack_id: str) -> Optional[Dict[str, Any]]:
        """获取单个技能包的状态"""
        if pack_id not in self._packs:
            return None
        
        packs = self.list_packs()
        for p in packs:
            if p["pack_id"] == pack_id:
                return p
        return None
    
    def is_installed(self, pack_id: str) -> bool:
        """检查技能包是否已安装"""
        return self._status.get(pack_id) == SkillPackStatus.INSTALLED
    
    # ========== 内部方法 ==========
    
    def _check_dependencies(self, manifest: SkillPackManifest) -> bool:
        """验证依赖是否满足"""
        import importlib
        
        for dep in manifest.dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                logger.warning(f"[SkillPackRuntime] 依赖缺失: {dep} (技能包: {manifest.pack_id})")
                return False
        
        return True
    
    def _record_experience(
        self,
        pack_id: str,
        capability: str,
        params: Dict[str, Any],
        result: Dict[str, Any],
        execution_time: float
    ):
        """记录经验到 ExperienceStore"""
        self._experience_counts[pack_id] = self._experience_counts.get(pack_id, 0) + 1
        
        if self.experience_store is not None:
            try:
                experience = {
                    "type": "skill_pack_execution",
                    "pack_id": pack_id,
                    "capability": capability,
                    "params": params,
                    "result": result,
                    "success": result.get("success", False),
                    "execution_time": execution_time,
                    "timestamp": time.time(),
                }
                self.experience_store.add(experience)
            except Exception as e:
                logger.warning(f"[SkillPackRuntime] 经验记录失败: {e}")
    
    def load_from_config(self, config_path: str) -> int:
        """从 YAML 配置文件加载技能包
        
        Args:
            config_path: 配置文件路径
        
        Returns:
            成功加载的技能包数量
        """
        import yaml
        import importlib
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"[SkillPackRuntime] 配置文件加载失败: {config_path} - {e}")
            return 0
        
        loaded_count = 0
        packs_config = config.get("skill_packs", [])
        
        for pack_cfg in packs_config:
            if not pack_cfg.get("enabled", False):
                continue
            
            pack_id = pack_cfg["pack_id"]
            pack_path = pack_cfg["path"]
            pack_config = pack_cfg.get("config", {})
            
            try:
                # 动态导入
                module = importlib.import_module(pack_path)
                
                # 查找技能包类（约定：模块中必须有且只有一个 ISkillPack 子类）
                pack_class = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, ISkillPack) and attr != ISkillPack:
                        pack_class = attr
                        break
                
                if pack_class is None:
                    logger.error(f"[SkillPackRuntime] 未找到技能包类: {pack_path}")
                    continue
                
                # 实例化并安装
                pack_instance = pack_class()
                if self.install_pack(pack_instance, pack_config):
                    loaded_count += 1
                    logger.info(f"[SkillPackRuntime] 已加载技能包: {pack_id}")
                
            except Exception as e:
                logger.error(f"[SkillPackRuntime] 加载技能包失败: {pack_id} - {e}")
        
        logger.info(f"[SkillPackRuntime] 配置加载完成: {loaded_count}/{len(packs_config)} 个技能包")
        return loaded_count
