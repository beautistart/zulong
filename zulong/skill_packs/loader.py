# File: zulong/skill_packs/loader.py
"""
技能包加载器

从 YAML 配置文件或目录结构自动加载技能包。
支持动态导入、依赖检查、资源验证。
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import importlib

import yaml

from zulong.skill_packs.interface import ISkillPack, SkillPackManifest
from zulong.skill_packs.runtime import SkillPackRuntime

logger = logging.getLogger(__name__)


class SkillPackLoader:
    """技能包加载器
    
    功能:
    - 从 YAML 配置文件加载技能包列表
    - 动态导入技能包模块
    - 验证依赖和资源需求
    - 批量安装到 SkillPackRuntime
    """
    
    def __init__(self, runtime: SkillPackRuntime):
        """初始化加载器
        
        Args:
            runtime: SkillPackRuntime 实例
        """
        self.runtime = runtime
        self._loaded_packs: Dict[str, ISkillPack] = {}
        logger.info("[SkillPackLoader] Initialized")
    
    def load_from_config(self, config_path: str) -> int:
        """从 YAML 配置文件加载技能包
        
        Args:
            config_path: YAML 配置文件路径
            
        Returns:
            成功加载的技能包数量
        """
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"[SkillPackLoader] 配置文件不存在: {config_path}")
            return 0
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"[SkillPackLoader] 加载配置文件失败: {e}")
            return 0
        
        skill_packs = config.get('skill_packs', [])
        if not skill_packs:
            logger.info("[SkillPackLoader] 配置文件中没有技能包")
            return 0
        
        success_count = 0
        for pack_config in skill_packs:
            pack_id = pack_config.get('pack_id')
            enabled = pack_config.get('enabled', False)
            pack_path = pack_config.get('path')
            pack_config_options = pack_config.get('config', {})
            
            if not enabled:
                logger.info(f"[SkillPackLoader] 跳过未启用的技能包: {pack_id}")
                continue
            
            if not pack_path:
                logger.warning(f"[SkillPackLoader] 技能包 {pack_id} 缺少 path 配置")
                continue
            
            try:
                pack_instance = self._load_pack(pack_id, pack_path, pack_config_options)
                if pack_instance:
                    success_count += 1
            except Exception as e:
                logger.error(f"[SkillPackLoader] 加载技能包 {pack_id} 失败: {e}")
        
        logger.info(f"[SkillPackLoader] 从配置加载完成: {success_count}/{len(skill_packs)} 个技能包")
        return success_count
    
    def _load_pack(self, pack_id: str, pack_path: str, config: Dict[str, Any]) -> Optional[ISkillPack]:
        """动态加载单个技能包
        
        Args:
            pack_id: 技能包 ID
            pack_path: Python 模块路径 (e.g. "zulong.skill_packs.packs.autogpt_planner")
            config: 技能包配置参数
            
        Returns:
            技能包实例，失败返回 None
        """
        try:
            # 动态导入模块
            module = importlib.import_module(pack_path)
            logger.info(f"[SkillPackLoader] 已导入模块: {pack_path}")
            
            # 查找 ISkillPack 实现
            pack_class = self._find_pack_class(module, pack_id)
            if not pack_class:
                logger.error(f"[SkillPackLoader] 模块 {pack_path} 中未找到 ISkillPack 实现")
                return None
            
            # 实例化
            pack_instance = pack_class()
            if not isinstance(pack_instance, ISkillPack):
                logger.error(f"[SkillPackLoader] {pack_class.__name__} 未实现 ISkillPack 接口")
                return None
            
            # 验证清单
            manifest = pack_instance.get_manifest()
            if manifest.pack_id != pack_id:
                logger.warning(f"[SkillPackLoader] pack_id 不匹配: {manifest.pack_id} vs {pack_id}")
            
            # 检查依赖
            if not self._check_dependencies(manifest):
                logger.warning(f"[SkillPackLoader] 技能包 {pack_id} 依赖检查失败")
                return None
            
            # 安装到运行时
            install_success = self.runtime.install_pack(pack_instance, config)
            if not install_success:
                logger.error(f"[SkillPackLoader] 技能包 {pack_id} 安装失败")
                return None
            
            self._loaded_packs[pack_id] = pack_instance
            logger.info(f"[SkillPackLoader] 技能包 {pack_id} 加载并安装成功")
            return pack_instance
            
        except ImportError as e:
            logger.error(f"[SkillPackLoader] 导入模块失败 {pack_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"[SkillPackLoader] 加载技能包 {pack_id} 异常: {e}", exc_info=True)
            return None
    
    def _find_pack_class(self, module, pack_id: str) -> Optional[type]:
        """在模块中查找 ISkillPack 实现类
        
        Args:
            module: Python 模块
            pack_id: 技能包 ID
            
        Returns:
            ISkillPack 实现类，未找到返回 None
        """
        # 策略 1: 查找模块中所有 ISkillPack 子类
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, ISkillPack) and attr != ISkillPack:
                return attr
        
        # 策略 2: 按命名约定查找 (e.g. AutoGPTPlanner for autogpt_planner)
        class_name = ''.join(part.capitalize() for part in pack_id.split('_'))
        if hasattr(module, class_name):
            cls = getattr(module, class_name)
            if isinstance(cls, type) and issubclass(cls, ISkillPack):
                return cls
        
        return None
    
    def _check_dependencies(self, manifest: SkillPackManifest) -> bool:
        """检查技能包依赖
        
        Args:
            manifest: 技能包清单
            
        Returns:
            依赖是否满足
        """
        if not manifest.dependencies:
            return True
        
        import importlib.util
        for dep in manifest.dependencies:
            if importlib.util.find_spec(dep) is None:
                logger.error(f"[SkillPackLoader] 缺少依赖: {dep}")
                return False
        
        return True
    
    def load_from_directory(self, packs_dir: str) -> int:
        """从目录加载所有技能包
        
        Args:
            packs_dir: 技能包目录路径
            
        Returns:
            成功加载的技能包数量
        """
        packs_path = Path(packs_dir)
        if not packs_path.exists():
            logger.warning(f"[SkillPackLoader] 技能包目录不存在: {packs_path}")
            return 0
        
        success_count = 0
        for item in packs_path.iterdir():
            if not item.is_dir():
                continue
            
            # 跳过 __pycache__ 等隐藏目录
            if item.name.startswith('_'):
                continue
            
            # 尝试加载 __init__.py
            init_file = item / '__init__.py'
            if not init_file.exists():
                continue
            
            pack_id = item.name
            
            # 构建模块路径
            module_path = f"zulong.skill_packs.packs.{pack_id}"
            
            try:
                pack_instance = self._load_pack(pack_id, module_path, {})
                if pack_instance:
                    success_count += 1
            except Exception as e:
                logger.error(f"[SkillPackLoader] 从目录加载 {pack_id} 失败: {e}")
        
        logger.info(f"[SkillPackLoader] 从目录加载完成: {success_count} 个技能包")
        return success_count
    
    def unload_pack(self, pack_id: str) -> bool:
        """卸载已加载的技能包
        
        Args:
            pack_id: 技能包 ID
            
        Returns:
            是否卸载成功
        """
        if pack_id not in self._loaded_packs:
            logger.warning(f"[SkillPackLoader] 技能包未加载: {pack_id}")
            return False
        
        success = self.runtime.uninstall_pack(pack_id)
        if success:
            del self._loaded_packs[pack_id]
            logger.info(f"[SkillPackLoader] 技能包 {pack_id} 已卸载")
        
        return success
    
    def list_loaded_packs(self) -> List[Dict[str, Any]]:
        """列出已加载的技能包
        
        Returns:
            已加载技能包信息列表
        """
        result = []
        for pack_id, pack_instance in self._loaded_packs.items():
            manifest = pack_instance.get_manifest()
            result.append({
                'pack_id': pack_id,
                'name': manifest.name,
                'version': manifest.version,
                'capabilities': manifest.capabilities,
                'status': 'loaded'
            })
        return result
