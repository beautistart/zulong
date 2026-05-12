# 复盘机制：成功经验提炼器

"""
功能:
- 任务描述提取
- 关键步骤识别
- 成功因素分析
- 结构化经验生成

对应 TSD v2.3 第 11.2.1 节
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class SuccessExperience:
    """成功经验数据类"""
    
    def __init__(self,
                 experience_id: str,
                 task_description: str,
                 key_steps: List[str],
                 success_factors: List[str],
                 context: Dict[str, Any],
                 metadata: Dict[str, Any]):
        self.experience_id = experience_id
        self.task_description = task_description
        self.key_steps = key_steps
        self.success_factors = success_factors
        self.context = context
        self.metadata = metadata
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'experience_id': self.experience_id,
            'task_description': self.task_description,
            'key_steps': self.key_steps,
            'success_factors': self.success_factors,
            'context': self.context,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


class SuccessExperienceExtractor:
    """成功经验提炼器"""
    
    def __init__(self,
                 embedding_model=None,
                 experience_store=None):
        """初始化提炼器
        
        Args:
            embedding_model: Embedding 模型 (可选)
            experience_store: 经验库实例 (可选)
        """
        self.embedding_model = embedding_model
        self.experience_store = experience_store
        
        # 提取配置
        self.config = {
            'min_step_length': 10,
            'max_steps': 10,
            'key_indicators': [
                '成功', '完成', '解决', '搞定', '好了', 'ok', 'yes'
            ],
            'noise_patterns': [
                r'^嗯',
                r'^那个',
                r'^让我',
                r'^我想',
                r'思考.*中',
                r'正在.*',
            ]
        }
        
        logger.info("[SuccessExtractor] 初始化完成")
    
    def extract_from_dialog(self,
                            dialog_history: List[Dict[str, Any]],
                            success_marker: Optional[str] = None) -> Optional[SuccessExperience]:
        """从对话历史中提取成功经验
        
        Args:
            dialog_history: 对话历史
            success_marker: 成功标记 (用户明确说"成功了"等)
            
        Returns:
            Optional[SuccessExperience]: 提炼的经验，失败返回 None
        """
        try:
            # 1. 识别任务描述
            task_description = self._extract_task_description(dialog_history)
            
            if not task_description:
                logger.debug("[SuccessExtractor] 未识别到任务描述")
                return None
            
            # 2. 提取关键步骤
            key_steps = self._extract_key_steps(dialog_history)
            
            if not key_steps:
                logger.debug("[SuccessExtractor] 未提取到关键步骤")
                return None
            
            # 3. 识别成功因素
            success_factors = self._identify_success_factors(dialog_history)
            
            # 4. 提取上下文
            context = self._extract_context(dialog_history)
            
            # 5. 生成经验
            experience_id = f"success_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            experience = SuccessExperience(
                experience_id=experience_id,
                task_description=task_description,
                key_steps=key_steps,
                success_factors=success_factors,
                context=context,
                metadata={
                    'source': 'dialog',
                    'dialog_turns': len(dialog_history),
                    'extraction_method': 'rule_based'
                }
            )
            
            logger.info(f"[SuccessExtractor] 成功经验已提炼：{experience_id}")
            logger.debug(f"[SuccessExtractor] 任务：{task_description}")
            logger.debug(f"[SuccessExtractor] 步骤：{len(key_steps)} 个")
            
            return experience
            
        except Exception as e:
            logger.error(f"[SuccessExtractor] 提炼失败：{e}")
            return None
    
    def _extract_task_description(self,
                                  dialog_history: List[Dict[str, Any]]) -> Optional[str]:
        """提取任务描述
        
        Args:
            dialog_history: 对话历史
            
        Returns:
            Optional[str]: 任务描述
        """
        # 查找第一个用户消息
        for msg in dialog_history:
            if msg.get('role') == 'user':
                content = msg.get('content', '').strip()
                if len(content) > 5:
                    return content
        
        return None
    
    def _extract_key_steps(self,
                           dialog_history: List[Dict[str, Any]]) -> List[str]:
        """提取关键步骤
        
        Args:
            dialog_history: 对话历史
            
        Returns:
            List[str]: 关键步骤列表
        """
        steps = []
        
        # 查找 assistant 消息
        for msg in dialog_history:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                
                # 按行分割
                lines = content.split('\n')
                
                for line in lines:
                    line = line.strip()
                    
                    # 检查是否包含步骤标记
                    if any(marker in line for marker in ['1.', '2.', '3.', '第一步', '第二步', '第三步']):
                        # 过滤噪声
                        is_noise = False
                        for pattern in self.config['noise_patterns']:
                            import re
                            if re.search(pattern, line):
                                is_noise = True
                                break
                        
                        if not is_noise and len(line) >= self.config['min_step_length']:
                            steps.append(line)
                
                # 限制步骤数量
                if len(steps) >= self.config['max_steps']:
                    break
        
        return steps[:self.config['max_steps']]
    
    def _identify_success_factors(self,
                                   dialog_history: List[Dict[str, Any]]) -> List[str]:
        """识别成功因素
        
        Args:
            dialog_history: 对话历史
            
        Returns:
            List[str]: 成功因素列表
        """
        factors = []
        
        # 查找成功相关的消息
        for msg in dialog_history:
            content = msg.get('content', '')
            
            # 检查成功关键词
            for indicator in self.config['key_indicators']:
                if indicator.lower() in content.lower():
                    factors.append(f"用户确认：{indicator}")
                    break
        
        # 去重
        return list(set(factors))
    
    def _extract_context(self,
                         dialog_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """提取上下文信息
        
        Args:
            dialog_history: 对话历史
            
        Returns:
            Dict[str, Any]: 上下文信息
        """
        context = {
            'timestamp': datetime.utcnow().isoformat(),
            'dialog_length': len(dialog_history)
        }
        
        # 提取 metadata
        for msg in dialog_history:
            if 'metadata' in msg:
                context.update(msg['metadata'])
                break
        
        return context
    
    def save_to_experience_store(self,
                                  experience: SuccessExperience):
        """保存经验到经验库
        
        Args:
            experience: 成功经验
        """
        if not self.experience_store:
            logger.warning("[SuccessExtractor] 经验库未初始化")
            return
        
        try:
            # 构建内容
            content = f"""
任务：{experience.task_description}

关键步骤:
{chr(10).join(f"- {step}" for step in experience.key_steps)}

成功因素:
{chr(10).join(f"- {factor}" for factor in experience.success_factors)}
"""
            
            # 添加到经验库
            self.experience_store.add_experience(
                content=content,
                experience_type="success",
                tags=["success", "experience"],
                metadata={
                    'experience_id': experience.experience_id,
                    'source': 'dialog'
                }
            )
            
            logger.info(f"[SuccessExtractor] 经验已保存到经验库")
            
        except Exception as e:
            logger.error(f"[SuccessExtractor] 保存失败：{e}")


# 全局单例
_success_extractor_instance = None


def get_success_extractor(
    embedding_model=None,
    experience_store=None
) -> SuccessExperienceExtractor:
    """获取成功经验提炼器单例
    
    Args:
        embedding_model: Embedding 模型
        experience_store: 经验库实例
        
    Returns:
        SuccessExperienceExtractor: 单例实例
    """
    global _success_extractor_instance
    
    if _success_extractor_instance is None:
        _success_extractor_instance = SuccessExperienceExtractor(
            embedding_model=embedding_model,
            experience_store=experience_store
        )
    
    return _success_extractor_instance
