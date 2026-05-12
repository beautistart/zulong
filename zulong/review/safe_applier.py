# File: zulong/review/safe_applier.py
# 安全应用器 - L1-B 执行写入操作

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SafeExperienceApplier:
    """安全经验应用器
    
    职责（L1-B 执行层）：
    - 验证 L2 生成的数据结构
    - 执行数据库写入操作
    - 合并临时缓冲区到主记忆池
    - 清理临时资源
    """
    
    def __init__(self):
        """初始化应用器"""
        logger.info("[SafeExperienceApplier] 初始化完成")
    
    def apply_experiences(self, experiences: List[Dict[str, Any]], session_id: str) -> Dict[str, Any]:
        """应用经验到数据库
        
        Args:
            experiences: 经验列表
            session_id: 会话 ID
            
        Returns:
            Dict: 应用结果
        """
        result = {
            'success': True,
            'applied_count': 0,
            'failed_count': 0,
            'errors': []
        }
        
        try:
            # 1. 验证每条经验
            validated_experiences = self._validate_experiences(experiences)
            
            # 2. 批量写入数据库
            for exp in validated_experiences:
                try:
                    self._write_experience_to_db(exp, session_id)
                    result['applied_count'] += 1
                except Exception as e:
                    logger.error(f"[SafeExperienceApplier] 写入经验失败：{e}")
                    result['failed_count'] += 1
                    result['errors'].append(str(e))
            
            if result['failed_count'] > 0:
                result['success'] = False
            
            logger.info(f"[SafeExperienceApplier] 应用完成：{result['applied_count']}成功，{result['failed_count']}失败")
            
        except Exception as e:
            logger.error(f"[SafeExperienceApplier] 应用经验失败：{e}", exc_info=True)
            result['success'] = False
            result['errors'].append(str(e))
        
        return result
    
    def _validate_experiences(self, experiences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证经验数据
        
        Args:
            experiences: 经验列表
            
        Returns:
            List[Dict]: 验证后的经验列表
        """
        validated = []
        
        for exp in experiences:
            # 白名单校验
            if not self._is_experience_valid(exp):
                logger.warning(f"[SafeExperienceApplier] 经验验证失败：{exp.get('content', '')[:50]}")
                continue
            
            # 添加系统元数据
            exp['_validated_at'] = datetime.now().isoformat()
            exp['_source'] = 'review_session'
            
            validated.append(exp)
        
        return validated
    
    def _is_experience_valid(self, exp: Dict[str, Any]) -> bool:
        """单条经验有效性检查（白名单机制）
        
        Args:
            exp: 经验数据
            
        Returns:
            bool: 是否有效
        """
        # 1. 必须包含内容
        if 'content' not in exp or not exp['content']:
            return False
        
        # 2. 内容长度限制（防止注入攻击）
        if len(exp['content']) > 1000:
            logger.warning("经验内容过长")
            return False
        
        # 3. 类型白名单
        allowed_types = ['decision', 'improvement', 'lesson', 'best_practice', 'general']
        exp_type = exp.get('type', 'general')
        if exp_type not in allowed_types:
            logger.warning(f"未知的经验类型：{exp_type}")
            return False
        
        # 4. 置信度范围检查
        confidence = exp.get('confidence', 0.5)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            logger.warning(f"无效的置信度：{confidence}")
            return False
        
        # 5. 标签白名单（可选）
        tags = exp.get('tags', [])
        if not isinstance(tags, list):
            return False
        
        # 6. 安全检查：不允许包含文件路径、系统命令等
        content = exp['content']
        dangerous_keywords = ['/etc/', 'sudo', 'rm -rf', 'DROP TABLE', 'DELETE FROM']
        for keyword in dangerous_keywords:
            if keyword in content:
                logger.warning(f"经验包含危险关键词：{keyword}")
                return False
        
        return True
    
    def _write_experience_to_db(self, exp: Dict[str, Any], session_id: str):
        """写入经验到数据库
        
        Args:
            exp: 经验数据
            session_id: 会话 ID
        """
        try:
            # 🔥 TODO: 实际写入数据库
            # 这里调用 memory/experience_generator.py 的安全接口
            
            logger.info(f"[SafeExperienceApplier] 写入经验：{exp['content'][:50]}...")
            
            # 模拟数据库写入
            # 实际应该：
            # from zulong.memory.experience_generator import ExperienceGenerator
            # generator = ExperienceGenerator()
            # generator.add_experience(exp, session_id=session_id)
            
        except Exception as e:
            logger.error(f"[SafeExperienceApplier] 数据库写入失败：{e}")
            raise
    
    def merge_buffer_to_memory(self, buffer_data: Dict[str, Any], confirmed_experiences: List[Dict[str, Any]]):
        """合并临时缓冲区到主记忆池
        
        Args:
            buffer_data: 缓冲区数据
            confirmed_experiences: 已确认的经验
        """
        try:
            logger.info(f"[SafeExperienceApplier] 合并缓冲区到主记忆池...")
            
            # 1. 提取精华对话（与已确认经验相关的对话）
            key_conversations = self._extract_key_conversations(buffer_data, confirmed_experiences)
            
            # 2. 高权重写入主记忆池
            self._write_to_shared_memory_pool(key_conversations, weight=1.5)
            
            # 3. 更新记忆索引
            self._update_memory_index(confirmed_experiences)
            
            logger.info("[SafeExperienceApplier] 缓冲区合并完成")
            
        except Exception as e:
            logger.error(f"[SafeExperienceApplier] 合并缓冲区失败：{e}", exc_info=True)
    
    def _extract_key_conversations(self, buffer_data: Dict[str, Any], experiences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取关键对话
        
        Args:
            buffer_data: 缓冲区数据
            experiences: 经验列表
            
        Returns:
            List[Dict]: 关键对话列表
        """
        # 简单实现：返回所有对话
        # 优化实现：基于证据字段筛选相关对话
        return buffer_data.get('conversations', [])
    
    def _write_to_shared_memory_pool(self, conversations: List[Dict[str, Any]], weight: float = 1.0):
        """写入共享记忆池
        
        Args:
            conversations: 对话列表
            weight: 权重系数
        """
        try:
            # 🔥 TODO: 实际写入共享记忆池
            from zulong.infrastructure.shared_memory_pool import get_shared_memory_pool
            
            pool = get_shared_memory_pool()
            
            for conv in conversations:
                # 封装成 DataEnvelope
                # pool.write(...)
                pass
            
            logger.info(f"[SafeExperienceApplier] 写入 {len(conversations)} 条对话到记忆池，权重：{weight}")
            
        except Exception as e:
            logger.error(f"[SafeExperienceApplier] 写入记忆池失败：{e}")
    
    def _update_memory_index(self, experiences: List[Dict[str, Any]]):
        """更新记忆索引
        
        Args:
            experiences: 经验列表
        """
        try:
            # 🔥 TODO: 更新记忆索引
            logger.info(f"[SafeExperienceApplier] 更新记忆索引，{len(experiences)} 条经验")
        except Exception as e:
            logger.error(f"[SafeExperienceApplier] 更新索引失败：{e}")
    
    def cleanup(self, session_id: str):
        """清理临时资源
        
        Args:
            session_id: 会话 ID
        """
        try:
            # 1. 销毁临时缓冲区
            from zulong.review.temp_buffer import get_review_buffer_manager
            buffer_manager = get_review_buffer_manager()
            buffer_manager.destroy_buffer()
            
            # 2. 清除全局状态
            from zulong.core.state_manager import state_manager
            state_manager.set_context('review_mode', False)
            state_manager.set_context('review_session_id', None)
            
            logger.info(f"[SafeExperienceApplier] 已清理会话 {session_id} 的资源")
            
        except Exception as e:
            logger.error(f"[SafeExperienceApplier] 清理资源失败：{e}")


# 全局单例
safe_applier = SafeExperienceApplier()


def get_safe_applier() -> SafeExperienceApplier:
    """获取安全应用器实例
    
    Returns:
        SafeExperienceApplier: 实例
    """
    return safe_applier
