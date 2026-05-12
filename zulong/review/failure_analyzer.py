# 复盘机制：失败案例分析

"""
功能:
- 错误归因分析 (能力不足/环境限制/指令错误)
- 避坑指南生成
- 权重策略 (1.5 倍)
- 失败模式识别

对应 TSD v2.3 第 11.2.2 节
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import re

logger = logging.getLogger(__name__)


class FailureCase:
    """失败案例数据类"""
    
    def __init__(self,
                 case_id: str,
                 task_description: str,
                 error_type: str,
                 root_cause: str,
                 avoidance_guide: str,
                 severity: str,
                 context: Dict[str, Any],
                 metadata: Dict[str, Any]):
        self.case_id = case_id
        self.task_description = task_description
        self.error_type = error_type
        self.root_cause = root_cause
        self.avoidance_guide = avoidance_guide
        self.severity = severity
        self.context = context
        self.metadata = metadata
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'case_id': self.case_id,
            'task_description': self.task_description,
            'error_type': self.error_type,
            'root_cause': self.root_cause,
            'avoidance_guide': self.avoidance_guide,
            'severity': self.severity,
            'context': self.context,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


class FailureAnalyzer:
    """失败案例分析器"""
    
    def __init__(self,
                 experience_store=None):
        """初始化分析器
        
        Args:
            experience_store: 经验库实例
        """
        self.experience_store = experience_store
        
        # 错误类型定义
        self.error_types = {
            'capability_limit': '能力不足',
            'environment_limit': '环境限制',
            'instruction_error': '指令错误',
            'timeout': '超时',
            'resource_error': '资源错误',
            'unknown': '未知错误'
        }
        
        # 严重性级别
        self.severity_levels = {
            'critical': '严重',
            'major': '主要',
            'minor': '次要',
            'trivial': '轻微'
        }
        
        # 错误模式匹配
        self.error_patterns = {
            'capability_limit': [
                r'无法完成', '做不到', '不支持', '能力有限',
                r'超出.*范围', '无法处理'
            ],
            'environment_limit': [
                r'网络错误', '连接失败', '超时', '资源不足',
                r'权限不足', '访问被拒绝', '文件不存在'
            ],
            'instruction_error': [
                r'指令错误', '参数错误', '格式错误', '无效输入',
                r'缺少.*参数', '不识别'
            ],
            'timeout': [
                r'超时', 'timeout', '响应超时', '连接超时'
            ],
            'resource_error': [
                r'内存不足', '显存不足', '磁盘空间不足',
                r'资源耗尽', '超出配额'
            ]
        }
        
        logger.info("[FailureAnalyzer] 初始化完成")
    
    def analyze_from_error(self,
                           error_message: str,
                           task_description: str,
                           context: Optional[Dict] = None) -> Optional[FailureCase]:
        """从错误信息分析失败案例
        
        Args:
            error_message: 错误信息
            task_description: 任务描述
            context: 上下文信息
            
        Returns:
            Optional[FailureCase]: 失败案例，失败返回 None
        """
        try:
            # 1. 识别错误类型
            error_type = self._identify_error_type(error_message)
            
            # 2. 分析根本原因
            root_cause = self._analyze_root_cause(error_message, error_type)
            
            # 3. 生成避坑指南
            avoidance_guide = self._generate_avoidance_guide(error_type, root_cause)
            
            # 4. 评估严重性
            severity = self._assess_severity(error_type, error_message)
            
            # 5. 生成案例
            case_id = f"failure_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            case = FailureCase(
                case_id=case_id,
                task_description=task_description,
                error_type=error_type,
                root_cause=root_cause,
                avoidance_guide=avoidance_guide,
                severity=severity,
                context=context or {},
                metadata={
                    'source': 'error_analysis',
                    'error_message': error_message,
                    'analysis_method': 'pattern_matching'
                }
            )
            
            logger.info(f"[FailureAnalyzer] 失败案例已分析：{case_id}")
            logger.debug(f"[FailureAnalyzer] 错误类型：{error_type}")
            logger.debug(f"[FailureAnalyzer] 根本原因：{root_cause}")
            
            return case
            
        except Exception as e:
            logger.error(f"[FailureAnalyzer] 分析失败：{e}")
            return None
    
    def _identify_error_type(self, error_message: str) -> str:
        """识别错误类型
        
        Args:
            error_message: 错误信息
            
        Returns:
            str: 错误类型
        """
        error_lower = error_message.lower()
        
        # 遍历错误模式
        for error_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, error_lower):
                    logger.debug(f"[FailureAnalyzer] 匹配错误类型：{error_type}")
                    return error_type
        
        # 默认未知
        return 'unknown'
    
    def _analyze_root_cause(self,
                            error_message: str,
                            error_type: str) -> str:
        """分析根本原因
        
        Args:
            error_message: 错误信息
            error_type: 错误类型
            
        Returns:
            str: 根本原因
        """
        # 根据错误类型生成原因模板
        cause_templates = {
            'capability_limit': "系统当前能力不足以处理该任务",
            'environment_limit': "外部环境条件不满足 (网络/资源/权限等)",
            'instruction_error': "用户指令存在错误或歧义",
            'timeout': "操作执行时间超过限制",
            'resource_error': "系统资源 (内存/显存/磁盘) 不足",
            'unknown': "未知错误，需要进一步分析"
        }
        
        base_cause = cause_templates.get(error_type, cause_templates['unknown'])
        
        # 添加具体错误信息
        specific_cause = f"{base_cause}. 错误信息：{error_message[:100]}"
        
        return specific_cause
    
    def _generate_avoidance_guide(self,
                                   error_type: str,
                                   root_cause: str) -> str:
        """生成避坑指南
        
        Args:
            error_type: 错误类型
            root_cause: 根本原因
            
        Returns:
            str: 避坑指南
        """
        guides = {
            'capability_limit': [
                "明确系统能力边界",
                "在能力范围内选择替代方案",
                "考虑升级系统或寻求外部帮助"
            ],
            'environment_limit': [
                "检查网络连接",
                "确认资源充足",
                "验证权限配置",
                "准备备用方案"
            ],
            'instruction_error': [
                "仔细阅读任务要求",
                "确认指令格式正确",
                "提供充足的上下文信息",
                "避免歧义表达"
            ],
            'timeout': [
                "优化任务执行流程",
                "增加超时时间配置",
                "分阶段执行长任务",
                "添加进度监控"
            ],
            'resource_error': [
                "监控资源使用情况",
                "及时释放无用资源",
                "优化资源使用策略",
                "考虑扩容"
            ],
            'unknown': [
                "详细记录错误信息",
                "逐步排查可能原因",
                "寻求技术支持"
            ]
        }
        
        guide_list = guides.get(error_type, guides['unknown'])
        
        # 组合指南
        guide = "避免此问题的建议:\n" + "\n".join(f"- {item}" for item in guide_list)
        
        return guide
    
    def _assess_severity(self,
                         error_type: str,
                         error_message: str) -> str:
        """评估严重性
        
        Args:
            error_type: 错误类型
            error_message: 错误信息
            
        Returns:
            str: 严重性级别
        """
        # 默认中等严重性
        severity = 'minor'
        
        # 根据错误类型调整
        if error_type in ['capability_limit', 'resource_error']:
            severity = 'major'
        elif error_type == 'environment_limit':
            severity = 'minor'
        elif error_type == 'timeout':
            severity = 'minor'
        
        # 检查关键词提升严重性
        critical_keywords = ['崩溃', 'crash', 'fatal', 'critical', '系统错误']
        for keyword in critical_keywords:
            if keyword.lower() in error_message.lower():
                severity = 'critical'
                break
        
        return severity
    
    def save_to_experience_store(self,
                                  case: FailureCase,
                                  weight_multiplier: float = 1.5):
        """保存案例到经验库 (失败案例权重 1.5 倍)
        
        Args:
            case: 失败案例
            weight_multiplier: 权重倍数
        """
        if not self.experience_store:
            logger.warning("[FailureAnalyzer] 经验库未初始化")
            return
        
        try:
            # 构建经验内容
            content = f"""
任务：{case.task_description}

错误类型：{case.error_type} ({self.error_types.get(case.error_type, '未知')})

根本原因:
{case.root_cause}

避坑指南:
{case.avoidance_guide}

严重性：{case.severity} ({self.severity_levels.get(case.severity, '未知')})
"""
            
            # 添加到经验库 (失败案例权重更高)
            self.experience_store.add_experience(
                content=content,
                experience_type="failure_case",
                tags=["failure", "case_study", case.error_type],
                metadata={
                    'weight_multiplier': weight_multiplier,
                    'severity': case.severity,
                    'case_id': case.case_id
                }
            )
            
            logger.info(f"[FailureAnalyzer] 失败案例已保存到经验库 (权重 x{weight_multiplier})")
            
        except Exception as e:
            logger.error(f"[FailureAnalyzer] 保存失败：{e}")
    
    def get_similar_failures(self,
                              error_message: str,
                              limit: int = 5) -> List[Dict]:
        """获取相似失败案例
        
        Args:
            error_message: 当前错误信息
            limit: 返回数量限制
            
        Returns:
            List[Dict]: 相似失败案例列表
        """
        if not self.experience_store:
            return []
        
        try:
            # 搜索经验库
            results = self.experience_store.search(
                query=error_message,
                filter={'experience_type': 'failure_case'},
                limit=limit
            )
            
            return results
            
        except Exception as e:
            logger.error(f"[FailureAnalyzer] 搜索失败案例失败：{e}")
            return []


# 全局单例
_failure_analyzer_instance = None


def get_failure_analyzer(
    experience_store=None
) -> FailureAnalyzer:
    """获取失败分析器单例
    
    Args:
        experience_store: 经验库实例
        
    Returns:
        FailureAnalyzer: 单例实例
    """
    global _failure_analyzer_instance
    
    if _failure_analyzer_instance is None:
        _failure_analyzer_instance = FailureAnalyzer(
            experience_store=experience_store
        )
    
    return _failure_analyzer_instance
