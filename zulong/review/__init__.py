# 复盘机制模块

"""
复盘机制系统:
- 三重触发机制（用户主动/安静模式/夜间定时）
- 经验分类处理（成功提炼/失败分析）
- 三重防重复机制（事件级/内容级/时间级）

对应 TSD v2.3 第 11 章
"""

from .trigger import (
    ReviewTrigger,
    TriggerType,
    TriggerPriority,
    get_review_trigger
)

from .integration import (
    ReplayIntegration,
    get_replay_integration
)

from .success_extractor import (
    SuccessExperienceExtractor,
    SuccessExperience,
    get_success_extractor
)

from .failure_analyzer import (
    FailureAnalyzer,
    FailureCase,
    get_failure_analyzer
)

from .deduplication import (
    DeduplicationFilter,
    get_dedup_filter
)

__all__ = [
    # 触发器
    'ReviewTrigger',
    'TriggerType',
    'TriggerPriority',
    'get_review_trigger',
    
    # 集成器
    'ReplayIntegration',
    'get_replay_integration',
    
    # 成功经验
    'SuccessExperienceExtractor',
    'SuccessExperience',
    'get_success_extractor',
    
    # 失败分析
    'FailureAnalyzer',
    'FailureCase',
    'get_failure_analyzer',
    
    # 防重复
    'DeduplicationFilter',
    'get_dedup_filter',
]
