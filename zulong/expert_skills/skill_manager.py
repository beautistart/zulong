# 专家技能管理器

"""
功能:
- 专家技能生命周期管理
- LRU 内存管理（自动卸载）
- 技能注册与发现
- 内存使用监控

对应 TSD v2.3 第 14.3 节
"""

import logging
from typing import Dict, Optional, List, Any
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


class ExpertSkillManager:
    """专家技能管理器
    
    功能:
    - 单例模式
    - LRU 淘汰策略
    - 内存限制
    - 懒加载
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式（线程安全）"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, 
                 max_memory_mb: int = 2048,
                 max_skills: int = 10):
        """初始化专家技能管理器
        
        Args:
            max_memory_mb: 最大内存限制（MB）
            max_skills: 最大技能数量
        """
        if hasattr(self, '_initialized'):
            return
        
        self.max_memory_mb = max_memory_mb
        self.max_skills = max_skills
        
        # LRU 缓存（OrderedDict 保持访问顺序）
        self._skills: OrderedDict[str, Any] = OrderedDict()
        self._skill_metadata: Dict[str, Dict] = {}
        
        # 内存跟踪
        self._memory_usage_mb = 0.0
        
        # 技能工厂函数注册表
        self._skill_factories: Dict[str, callable] = {}
        
        self._initialized = True
        
        logger.info(f"[ExpertSkillManager] 初始化完成："
                   f"max_memory={max_memory_mb}MB, max_skills={max_skills}")
    
    def register_skill_factory(self, 
                               skill_type: str, 
                               factory_func: callable):
        """注册技能工厂函数
        
        Args:
            skill_type: 技能类型
            factory_func: 工厂函数
        """
        self._skill_factories[skill_type] = factory_func
        logger.debug(f"[ExpertSkillManager] 注册技能工厂：{skill_type}")
    
    async def get_skill(self, 
                        skill_id: str,
                        skill_type: str,
                        **kwargs) -> Optional[Any]:
        """获取技能实例（懒加载 + LRU）
        
        Args:
            skill_id: 技能 ID
            skill_type: 技能类型
            **kwargs: 初始化参数
            
        Returns:
            Any: 技能实例，失败返回 None
        """
        # 检查是否已加载
        if skill_id in self._skills:
            # 移动到最近使用
            self._skills.move_to_end(skill_id)
            logger.debug(f"[ExpertSkillManager] 技能已加载：{skill_id}")
            return self._skills[skill_id]
        
        # 检查工厂函数
        if skill_type not in self._skill_factories:
            logger.error(f"[ExpertSkillManager] 未知技能类型：{skill_type}")
            return None
        
        # 检查限制并触发 LRU 淘汰
        while not self._can_load_skill():
            logger.warning(f"[ExpertSkillManager] 达到限制，触发 LRU 淘汰")
            if not self._skills:  # 没有技能可淘汰
                break
            self._evict_lru()
        
        # 加载技能
        try:
            factory = self._skill_factories[skill_type]
            skill = factory(skill_id=skill_id, **kwargs)
            
            # 添加到 LRU 缓存
            self._skills[skill_id] = skill
            self._skill_metadata[skill_id] = {
                'type': skill_type,
                'loaded_at': self._get_timestamp(),
                'access_count': 1
            }
            
            # 更新内存使用（估算）
            self._update_memory_usage()
            
            logger.info(f"[ExpertSkillManager] 技能已加载：{skill_id} ({skill_type})")
            
            return skill
            
        except Exception as e:
            logger.error(f"[ExpertSkillManager] 加载技能失败：{skill_id}, 错误：{e}")
            return None
    
    def unload_skill(self, skill_id: str) -> bool:
        """卸载技能
        
        Args:
            skill_id: 技能 ID
            
        Returns:
            bool: 是否成功卸载
        """
        if skill_id not in self._skills:
            return False
        
        # 调用技能的 clear 方法（如果有）
        skill = self._skills[skill_id]
        if hasattr(skill, 'clear'):
            skill.clear()
        
        # 从缓存中移除
        del self._skills[skill_id]
        if skill_id in self._skill_metadata:
            del self._skill_metadata[skill_id]
        
        # 更新内存使用
        self._update_memory_usage()
        
        logger.info(f"[ExpertSkillManager] 技能已卸载：{skill_id}")
        
        return True
    
    def unload_all_skills(self):
        """卸载所有技能（用于系统关闭）"""
        for skill_id in list(self._skills.keys()):
            self.unload_skill(skill_id)
        
        logger.info(f"[ExpertSkillManager] 所有技能已卸载")
    
    def get_loaded_skills(self) -> List[str]:
        """获取已加载的技能 ID 列表
        
        Returns:
            List[str]: 技能 ID 列表
        """
        return list(self._skills.keys())
    
    def get_skill_stats(self) -> Dict[str, Any]:
        """获取技能统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = {
            'loaded_skills': len(self._skills),
            'max_skills': self.max_skills,
            'memory_usage_mb': self._memory_usage_mb,
            'max_memory_mb': self.max_memory_mb,
            'skills': []
        }
        
        for skill_id, metadata in self._skill_metadata.items():
            skill_info = {
                'id': skill_id,
                'type': metadata['type'],
                'loaded_at': metadata['loaded_at'],
                'access_count': metadata['access_count']
            }
            
            # 获取技能自身的统计（如果有）
            if skill_id in self._skills:
                skill = self._skills[skill_id]
                if hasattr(skill, 'get_stats'):
                    skill_info['stats'] = skill.get_stats()
            
            stats['skills'].append(skill_info)
        
        return stats
    
    def _can_load_skill(self) -> bool:
        """检查是否可以加载新技能
        
        Returns:
            bool: 是否可以加载
        """
        # 检查数量限制
        if len(self._skills) >= self.max_skills:
            return False
        
        # 检查内存限制（保守估计）
        if self._memory_usage_mb >= self.max_memory_mb:
            return False
        
        return True
    
    def _evict_lru(self):
        """淘汰最近最少使用的技能（LRU）"""
        if not self._skills:
            return
        
        # OrderedDict 的第一个元素是最久未使用的
        oldest_skill_id = next(iter(self._skills))
        
        logger.info(f"[ExpertSkillManager] LRU 淘汰：{oldest_skill_id}")
        
        self.unload_skill(oldest_skill_id)
    
    def _update_memory_usage(self):
        """更新内存使用估算（简化实现）"""
        # 简单估算：每个技能 ~50MB
        self._memory_usage_mb = len(self._skills) * 50.0
        
        logger.debug(f"[ExpertSkillManager] 内存使用：{self._memory_usage_mb:.1f}MB")
    
    def _get_timestamp(self) -> str:
        """获取时间戳字符串"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 工厂函数
def get_expert_skill_manager(
    max_memory_mb: int = 2048,
    max_skills: int = 10
) -> ExpertSkillManager:
    """获取专家技能管理器单例
    
    Args:
        max_memory_mb: 最大内存限制（MB）
        max_skills: 最大技能数量
        
    Returns:
        ExpertSkillManager: 单例实例
    """
    return ExpertSkillManager(
        max_memory_mb=max_memory_mb,
        max_skills=max_skills
    )
