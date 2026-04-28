# File: zulong/expert_skills/skill_pool.py
# L3 专家技能池管理器 - 统一调度、资源管理、技能编排

"""
祖龙 (ZULONG) L3 专家技能池管理器

对应 TSD v1.7:
- 2.3.2 专家模型层：L3 专家技能池
- 技能编排与资源管理

功能:
- 技能注册与发现
- 技能生命周期管理
- 资源调度（GPU/CPU 分配）
- 技能编排（多技能协作）
- 性能监控
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Type
from dataclasses import dataclass, field
import time
import threading
from enum import Enum

logger = logging.getLogger(__name__)


class SkillStatus(Enum):
    """技能状态"""
    IDLE = "idle"  # 空闲
    BUSY = "busy"  # 忙碌
    LOADING = "loading"  # 加载中
    ERROR = "error"  # 错误
    UNLOADED = "unloaded"  # 已卸载


@dataclass
class SkillMetadata:
    """技能元数据"""
    skill_id: str
    skill_type: str  # navigation/vision/rag/tool
    status: SkillStatus = SkillStatus.UNLOADED
    loaded_at: Optional[float] = None
    last_used_at: Optional[float] = None
    usage_count: int = 0
    gpu_memory_mb: int = 0  # GPU 显存占用
    cpu_memory_mb: int = 0  # CPU 内存占用
    priority: int = 0  # 优先级（0-10，越高越优先）


@dataclass
class SkillCallResult:
    """技能调用结果"""
    skill_id: str
    success: bool
    result: Any
    execution_time: float
    error: Optional[str] = None


class SkillPool:
    """L3 专家技能池管理器
    
    TSD v1.7 对应规则:
    - L3 专家技能池统一管理
    - 支持懒加载和 LRU 淘汰
    - 资源监控和调度
    - 技能编排
    
    功能:
    - 技能注册
    - 懒加载/卸载
    - 调用编排
    - 资源监控
    """
    
    def __init__(self, 
                 max_gpu_memory_mb: int = 4096,  # RTX 3060 6GB，预留 2GB
                 max_cpu_memory_mb: int = 8192):
        """初始化技能池
        
        Args:
            max_gpu_memory_mb: 最大 GPU 显存（MB）
            max_cpu_memory_mb: 最大 CPU 内存（MB）
        """
        self.max_gpu_memory_mb = max_gpu_memory_mb
        self.max_cpu_memory_mb = max_cpu_memory_mb
        
        # 技能工厂函数注册表
        self._skill_factories: Dict[str, Callable] = {}
        
        # 已加载的技能实例
        self._loaded_skills: Dict[str, Any] = {}
        
        # 技能元数据
        self._metadata: Dict[str, SkillMetadata] = {}
        
        # 资源跟踪
        self.current_gpu_usage_mb = 0
        self.current_cpu_usage_mb = 0
        
        # 调用历史
        self.call_history: List[Dict[str, Any]] = []
        
        # 锁
        self._lock = threading.RLock()
        
        logger.info(f"[SkillPool] 初始化完成："
                   f"GPU={max_gpu_memory_mb}MB, CPU={max_cpu_memory_mb}MB")
    
    # ========== 技能注册 ==========
    
    def register_skill(self,
                       skill_type: str,
                       factory_func: Callable,
                       gpu_memory_mb: int = 0,
                       cpu_memory_mb: int = 512,
                       priority: int = 5):
        """注册技能
        
        Args:
            skill_type: 技能类型
            factory_func: 工厂函数（返回技能实例）
            gpu_memory_mb: 预估 GPU 显存占用
            cpu_memory_mb: 预估 CPU 内存占用
            priority: 优先级
        """
        with self._lock:
            skill_id = f"{skill_type}_expert"
            
            self._skill_factories[skill_type] = factory_func
            
            self._metadata[skill_id] = SkillMetadata(
                skill_id=skill_id,
                skill_type=skill_type,
                status=SkillStatus.UNLOADED,
                gpu_memory_mb=gpu_memory_mb,
                cpu_memory_mb=cpu_memory_mb,
                priority=priority
            )
            
            logger.info(f"[SkillPool] 注册技能：{skill_id}, "
                       f"GPU={gpu_memory_mb}MB, CPU={cpu_memory_mb}MB")
    
    # ========== 技能获取（懒加载） ==========
    
    def get_skill(self, skill_type: str) -> Optional[Any]:
        """获取技能实例（懒加载）
        
        Args:
            skill_type: 技能类型
            
        Returns:
            Any: 技能实例，失败返回 None
        """
        with self._lock:
            skill_id = f"{skill_type}_expert"
            
            # 检查是否已加载
            if skill_id in self._loaded_skills:
                # 更新使用时间和计数
                self._metadata[skill_id].last_used_at = time.time()
                self._metadata[skill_id].usage_count += 1
                return self._loaded_skills[skill_id]
            
            # 检查是否已注册
            if skill_type not in self._skill_factories:
                logger.error(f"[SkillPool] 技能未注册：{skill_type}")
                return None
            
            # 检查资源是否足够
            metadata = self._metadata[skill_id]
            if not self._check_resources(metadata):
                logger.warning(f"[SkillPool] 资源不足，尝试卸载低优先级技能")
                self._evict_low_priority_skills()
                
                # 再次检查
                if not self._check_resources(metadata):
                    logger.error(f"[SkillPool] 资源仍然不足，无法加载 {skill_id}")
                    return None
            
            # 加载技能
            logger.info(f"[SkillPool] 懒加载技能：{skill_id}")
            metadata.status = SkillStatus.LOADING
            
            try:
                # 调用工厂函数创建实例
                skill_instance = self._skill_factories[skill_type]()
                
                # 保存实例
                self._loaded_skills[skill_id] = skill_instance
                
                # 更新元数据
                metadata.status = SkillStatus.IDLE
                metadata.loaded_at = time.time()
                metadata.last_used_at = time.time()
                metadata.usage_count = 1
                
                # 更新资源使用
                self.current_gpu_usage_mb += metadata.gpu_memory_mb
                self.current_cpu_usage_mb += metadata.cpu_memory_mb
                
                logger.info(f"[SkillPool] 技能加载成功：{skill_id}, "
                           f"GPU={self.current_gpu_usage_mb}MB, "
                           f"CPU={self.current_cpu_usage_mb}MB")
                
                return skill_instance
                
            except Exception as e:
                logger.error(f"[SkillPool] 技能加载失败：{skill_id}, 错误：{e}")
                metadata.status = SkillStatus.ERROR
                return None
    
    def _check_resources(self, metadata: SkillMetadata) -> bool:
        """检查资源是否足够"""
        gpu_available = (self.current_gpu_usage_mb + metadata.gpu_memory_mb 
                        <= self.max_gpu_memory_mb)
        cpu_available = (self.current_cpu_usage_mb + metadata.cpu_memory_mb 
                        <= self.max_cpu_memory_mb)
        return gpu_available and cpu_available
    
    def _evict_low_priority_skills(self):
        """淘汰低优先级技能（LRU + 优先级）"""
        with self._lock:
            # 按优先级和使用时间排序
            skills_to_evict = []
            
            for skill_id, metadata in self._metadata.items():
                if skill_id in self._loaded_skills and metadata.status == SkillStatus.IDLE:
                    skills_to_evict.append((
                        metadata.priority,
                        metadata.last_used_at or 0,
                        skill_id
                    ))
            
            # 排序（优先级低、长时间未使用的优先淘汰）
            skills_to_evict.sort(key=lambda x: (x[0], x[1]))
            
            # 淘汰直到资源足够
            for _, _, skill_id in skills_to_evict:
                if self._unload_skill(skill_id):
                    logger.info(f"[SkillPool] 淘汰技能：{skill_id}")
                    break
    
    def _unload_skill(self, skill_id: str) -> bool:
        """卸载技能"""
        try:
            if skill_id not in self._loaded_skills:
                return False
            
            # 删除实例
            del self._loaded_skills[skill_id]
            
            # 更新元数据
            metadata = self._metadata[skill_id]
            self.current_gpu_usage_mb -= metadata.gpu_memory_mb
            self.current_cpu_usage_mb -= metadata.cpu_memory_mb
            metadata.status = SkillStatus.UNLOADED
            metadata.loaded_at = None
            metadata.last_used_at = None
            
            logger.debug(f"[SkillPool] 技能已卸载：{skill_id}")
            return True
            
        except Exception as e:
            logger.error(f"[SkillPool] 技能卸载失败：{skill_id}, 错误：{e}")
            return False
    
    # ========== 技能调用 ==========
    
    def call_skill(self,
                    skill_type: str,
                    method: str,
                    *args,
                    **kwargs) -> SkillCallResult:
        """调用技能方法
        
        Args:
            skill_type: 技能类型
            method: 方法名
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            SkillCallResult: 调用结果
        """
        start_time = time.time()
        
        # 获取技能实例
        skill = self.get_skill(skill_type)
        
        if skill is None:
            return SkillCallResult(
                skill_id=f"{skill_type}_expert",
                success=False,
                result=None,
                execution_time=0.0,
                error=f"技能不可用：{skill_type}"
            )
        
        # 更新状态
        with self._lock:
            skill_id = f"{skill_type}_expert"
            self._metadata[skill_id].status = SkillStatus.BUSY
        
        try:
            # 调用方法
            method_func = getattr(skill, method)
            result = method_func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            # 记录历史
            self._record_call(skill_id, method, True, execution_time)
            
            # 恢复状态
            with self._lock:
                self._metadata[skill_id].status = SkillStatus.IDLE
                self._metadata[skill_id].last_used_at = time.time()
                self._metadata[skill_id].usage_count += 1
            
            return SkillCallResult(
                skill_id=skill_id,
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            # 记录历史
            self._record_call(skill_id, method, False, execution_time, error_msg)
            
            # 更新状态
            with self._lock:
                self._metadata[skill_id].status = SkillStatus.ERROR
            
            logger.error(f"[SkillPool] 技能调用失败：{skill_id}.{method}, 错误：{e}")
            
            return SkillCallResult(
                skill_id=skill_id,
                success=False,
                result=None,
                execution_time=execution_time,
                error=error_msg
            )
    
    def _record_call(self, 
                     skill_id: str, 
                     method: str, 
                     success: bool,
                     execution_time: float,
                     error: Optional[str] = None):
        """记录调用历史"""
        record = {
            'timestamp': time.time(),
            'skill_id': skill_id,
            'method': method,
            'success': success,
            'execution_time': execution_time,
            'error': error
        }
        
        self.call_history.append(record)
        
        # 保留最近 1000 条
        if len(self.call_history) > 1000:
            self.call_history.pop(0)
    
    # ========== 技能编排 ==========
    
    def orchestrate(self, 
                    workflow: List[Dict[str, Any]]) -> List[SkillCallResult]:
        """技能编排（工作流）
        
        Args:
            workflow: 工作流定义
                     [{"skill": "vision", "method": "detect_objects", "args": [...]},
                      {"skill": "navigation", "method": "plan_path", "args": [...]}]
        
        Returns:
            List[SkillCallResult]: 每个步骤的结果
        """
        results = []
        
        logger.info(f"[SkillPool] 开始执行工作流：{len(workflow)} 步骤")
        
        for i, step in enumerate(workflow):
            skill_type = step.get('skill')
            method = step.get('method')
            args = step.get('args', [])
            kwargs = step.get('kwargs', {})
            
            if not skill_type or not method:
                logger.error(f"[SkillPool] 工作流步骤 {i} 缺少 skill 或 method")
                continue
            
            logger.debug(f"[SkillPool] 执行步骤 {i+1}: {skill_type}.{method}")
            
            result = self.call_skill(skill_type, method, *args, **kwargs)
            results.append(result)
            
            # 如果失败且是关键步骤，可以选择中止
            if not result.success and step.get('critical', False):
                logger.error(f"[SkillPool] 关键步骤失败，中止工作流")
                break
        
        logger.info(f"[SkillPool] 工作流执行完成：成功 {sum(1 for r in results if r.success)}/{len(results)}")
        
        return results
    
    # ========== 监控与管理 ==========
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            loaded_count = len(self._loaded_skills)
            total_count = len(self._skill_factories)
            
            return {
                'loaded_skills': loaded_count,
                'total_skills': total_count,
                'gpu_usage_mb': self.current_gpu_usage_mb,
                'gpu_max_mb': self.max_gpu_memory_mb,
                'cpu_usage_mb': self.current_cpu_usage_mb,
                'cpu_max_mb': self.max_cpu_memory_mb,
                'total_calls': len(self.call_history),
                'skills': {
                    skill_id: {
                        'status': metadata.status.value,
                        'usage_count': metadata.usage_count,
                        'loaded_at': metadata.loaded_at
                    }
                    for skill_id, metadata in self._metadata.items()
                }
            }
    
    def list_skills(self) -> List[Dict[str, Any]]:
        """列出所有技能"""
        with self._lock:
            return [
                {
                    'skill_id': metadata.skill_id,
                    'skill_type': metadata.skill_type,
                    'status': metadata.status.value,
                    'loaded': metadata.skill_id in self._loaded_skills,
                    'gpu_memory_mb': metadata.gpu_memory_mb,
                    'cpu_memory_mb': metadata.cpu_memory_mb,
                    'priority': metadata.priority,
                    'usage_count': metadata.usage_count
                }
                for metadata in self._metadata.values()
            ]
    
    def unload_all(self):
        """卸载所有技能"""
        with self._lock:
            for skill_id in list(self._loaded_skills.keys()):
                self._unload_skill(skill_id)
            
            logger.info("[SkillPool] 所有技能已卸载")
    
    def get_skill_info(self, skill_type: str) -> Optional[Dict[str, Any]]:
        """获取技能详细信息"""
        with self._lock:
            skill_id = f"{skill_type}_expert"
            
            if skill_id not in self._metadata:
                return None
            
            metadata = self._metadata[skill_id]
            
            return {
                'skill_id': metadata.skill_id,
                'skill_type': metadata.skill_type,
                'status': metadata.status.value,
                'loaded': skill_id in self._loaded_skills,
                'loaded_at': metadata.loaded_at,
                'last_used_at': metadata.last_used_at,
                'usage_count': metadata.usage_count,
                'gpu_memory_mb': metadata.gpu_memory_mb,
                'cpu_memory_mb': metadata.cpu_memory_mb,
                'priority': metadata.priority
            }
