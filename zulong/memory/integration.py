# 经验库集成模块

"""
功能:
- 集成复盘机制到经验库
- 集成时间标签到检索排序
- 统一的经验管理接口

对应 TSD v2.3 第 13 章
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from .enhanced_experience_store import EnhancedExperienceStore, Experience
from .time_tags import TimeTags, TimeTagManager, get_time_tag_manager
from .rollback import RollbackManager, get_rollback_manager, RollbackResult
from ..review import (
    get_review_trigger,
    get_success_extractor,
    get_failure_analyzer,
    get_dedup_filter,
    TriggerType
)

logger = logging.getLogger(__name__)


class IntegratedExperienceStore:
    """集成版经验库
    
    整合功能:
    1. ✅ 增强版经验存储 (向量 +BM25)
    2. ✅ 时间标签管理
    3. ✅ 复盘机制集成
    4. ✅ 自动降智回滚
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self,
                 db_path: Optional[str] = None,
                 enable_persistence: bool = True,
                 enable_smart_tagging: bool = True,
                 enable_review: bool = True,
                 enable_time_tags: bool = True):
        """初始化集成经验库
        
        Args:
            db_path: 数据库路径
            enable_persistence: 是否启用持久化
            enable_smart_tagging: 是否启用智能打标
            enable_review: 是否启用复盘机制
            enable_time_tags: 是否启用时间标签
        """
        if not hasattr(self, '_initialized'):
            # 0. 初始化 Embedding 模型（在 EnhancedExperienceStore 之前）
            from zulong.memory import get_embedding_manager
            self.embedding_manager = get_embedding_manager()
            
            # 1. 初始化基础经验库
            self.store = EnhancedExperienceStore(
                db_path=db_path,
                enable_persistence=enable_persistence,
                enable_smart_tagging=enable_smart_tagging
            )
            
            # 设置 Embedding 模型到 EnhancedExperienceStore
            self.store.set_embedding_model(self.embedding_manager)
            
            # 2. 初始化时间标签管理
            self.enable_time_tags = enable_time_tags
            if enable_time_tags:
                self.time_tag_manager = get_time_tag_manager()
                self.rollback_manager = get_rollback_manager(self.time_tag_manager)
                
                # 注册回滚回调
                self.rollback_manager.register_callback(self._on_rollback)
            else:
                self.time_tag_manager = None
                self.rollback_manager = None
            
            # 3. 初始化复盘机制
            self.enable_review = enable_review
            if enable_review:
                self.review_trigger = get_review_trigger()
                self.success_extractor = get_success_extractor(
                    experience_store=self.store
                )
                self.failure_analyzer = get_failure_analyzer(
                    experience_store=self.store
                )
                self.dedup_filter = get_dedup_filter()
                
                # 注册复盘回调
                self._setup_review_callbacks()
            else:
                self.review_trigger = None
                self.success_extractor = None
                self.failure_analyzer = None
                self.dedup_filter = None
            
            # 4. 统计信息
            self.stats = {
                'total_experiences': 0,
                'reviews_triggered': 0,
                'rollbacks_executed': 0
            }
            
            self._initialized = True
            
            logger.info(f"[IntegratedExperienceStore] 初始化完成："
                       f"review={enable_review}, time_tags={enable_time_tags}")
    
    def _setup_review_callbacks(self):
        """设置复盘回调"""
        if not self.review_trigger:
            return
        
        # 注册复盘处理回调
        async def handle_review(request):
            await self._process_review(request)
        
        self.review_trigger.register_callback(
            TriggerType.USER_ACTIVE,
            handle_review
        )
        self.review_trigger.register_callback(
            TriggerType.QUIET_MODE,
            handle_review
        )
        self.review_trigger.register_callback(
            TriggerType.NIGHT_SCHEDULE,
            handle_review
        )
        
        logger.info("[IntegratedExperienceStore] 复盘回调已注册")
    
    async def _process_review(self, request: Dict[str, Any]):
        """处理复盘请求
        
        Args:
            request: 复盘请求
        """
        try:
            context = request.get('context', {})
            event_type = context.get('type', 'failure')
            event_data = context.get('data', {})
            
            # 1. 防重复检查
            if self.dedup_filter:
                should_review, reason = self.dedup_filter.should_review(
                    event_data, event_type
                )
                
                if not should_review:
                    logger.info(f"[Review] 跳过复盘：{reason}")
                    return
            
            # 2. 经验处理
            if event_type == 'success':
                await self._process_success(event_data)
            else:
                await self._process_failure(event_data)
            
            self.stats['reviews_triggered'] += 1
            
        except Exception as e:
            logger.error(f"[Review] 处理失败：{e}")
    
    async def _process_success(self, event_data: Dict[str, Any]):
        """处理成功经验
        
        Args:
            event_data: 事件数据
        """
        if not self.success_extractor:
            return
        
        dialog = event_data.get('dialog', [])
        success_marker = event_data.get('success_marker', '成功')
        
        # 提炼经验
        experience = self.success_extractor.extract_from_dialog(
            dialog_history=dialog,
            success_marker=success_marker
        )
        
        if experience:
            # 保存到经验库
            self.success_extractor.save_to_experience_store(experience)
            
            logger.info(f"[Success] 经验已保存：{experience.experience_id}")
    
    async def _process_failure(self, event_data: Dict[str, Any]):
        """处理失败案例
        
        Args:
            event_data: 事件数据
        """
        if not self.failure_analyzer:
            return
        
        error_message = event_data.get('error', '')
        task_description = event_data.get('task', '')
        
        # 分析失败
        case = self.failure_analyzer.analyze_from_error(
            error_message=error_message,
            task_description=task_description
        )
        
        if case:
            # 保存到经验库
            self.failure_analyzer.save_to_experience_store(case)
            
            logger.info(f"[Failure] 案例已保存：{case.case_id}")
    
    def add_experience(self,
                       content: str,
                       experience_type: str,
                       tags: Optional[List[str]] = None,
                       metadata: Optional[Dict] = None) -> str:
        """添加经验（带时间标签）
        
        Args:
            content: 经验内容
            experience_type: 经验类型
            tags: 标签列表
            metadata: 元数据
            
        Returns:
            str: 经验 ID
        """
        # 1. 创建时间标签
        time_tags = None
        if self.enable_time_tags and self.time_tag_manager:
            time_tags = self.time_tag_manager.create_time_tags()
        
        # 2. 添加到基础经验库（会生成实际 ID）
        exp_id = self.store.add_experience(
            content=content,
            experience_type=experience_type,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # 3. 更新时间标签到元数据
        if time_tags and exp_id in self.store._experiences:
            self.store._experiences[exp_id].metadata['time_tags'] = time_tags.to_dict()
        
        self.stats['total_experiences'] += 1
        
        logger.info(f"[IntegratedExperienceStore] 经验已添加：{exp_id}")
        
        return exp_id
    
    def search(self,
               query: str,
               filter: Optional[Dict] = None,
               limit: int = 10,
               use_time_decay: bool = True) -> List[Dict]:
        """搜索经验（带时间衰减）
        
        Args:
            query: 查询文本
            filter: 过滤条件 (tags, types 等)
            limit: 返回数量
            use_time_decay: 是否应用时间衰减
            
        Returns:
            List[Dict]: 搜索结果
        """
        # 1. 生成查询向量
        query_vector = self.store._get_embedding(query)
        
        # 2. 解析过滤条件
        filter_types = filter.get('types') if filter else None
        filter_tags = filter.get('tags') if filter else None
        
        # 3. 基础搜索
        results = self.store.search(
            query_vector=query_vector,
            query_text=query,
            filter_types=filter_types,
            filter_tags=filter_tags,
            use_hybrid=True,
            apply_time_decay=use_time_decay,
            limit=limit
        )
        
        # 转换为字典列表
        dict_results = []
        for score, exp in results:
            result_dict = {
                'id': exp.id,
                'content': exp.content,
                'experience_type': exp.experience_type,
                'score': score,
                'metadata': exp.metadata,
                'tags': exp.tags
            }
            
            # 4. 应用时间衰减（如果启用）
            if use_time_decay and self.enable_time_tags and self.time_tag_manager:
                time_tags_data = exp.metadata.get('time_tags')
                
                if time_tags_data:
                    time_tags = TimeTags.from_dict(time_tags_data)
                    evaluation = self.time_tag_manager.evaluate_experience(
                        time_tags,
                        exp.access_count
                    )
                    
                    # 更新分数
                    original_score = score
                    time_weight = evaluation['time_weight']
                    result_dict['score'] = original_score * time_weight
                    result_dict['time_evaluation'] = evaluation
            
            dict_results.append(result_dict)
        
        # 5. 排序
        dict_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return dict_results
    
    async def evaluate_and_rollback(self,
                                     experience_ids: Optional[List[str]] = None) -> Dict:
        """评估并执行回滚
        
        Args:
            experience_ids: 经验 ID 列表，None 表示评估全部
            
        Returns:
            Dict: 回滚结果摘要
        """
        if not self.enable_time_tags or not self.rollback_manager:
            return {'error': '时间标签未启用'}
        
        # 1. 获取经验列表
        if experience_ids:
            experiences = [
                {
                    'id': exp_id,
                    'time_tags': self.store._experiences[exp_id].metadata.get('time_tags'),
                    'usage_count': self.store._experiences[exp_id].access_count
                }
                for exp_id in experience_ids
                if exp_id in self.store._experiences
            ]
        else:
            experiences = [
                {
                    'id': exp_id,
                    'time_tags': exp.metadata.get('time_tags'),
                    'usage_count': exp.access_count
                }
                for exp_id, exp in self.store._experiences.items()
            ]
        
        # 2. 批量评估
        results = await self.rollback_manager.evaluate_batch(experiences)
        
        # 3. 执行回滚
        executed = 0
        for result in results:
            if result.action.value != 'none':
                success = self.rollback_manager.execute_rollback(
                    result,
                    self.store
                )
                if success:
                    executed += 1
        
        # 4. 获取摘要
        summary = self.rollback_manager.get_rollback_summary(results)
        summary['executed_count'] = executed
        
        self.stats['rollbacks_executed'] += executed
        
        logger.info(f"[IntegratedExperienceStore] 回滚完成："
                   f"评估={len(results)}, 执行={executed}")
        
        return summary
    
    def update_usage(self, experience_id: str):
        """更新经验使用记录
        
        Args:
            experience_id: 经验 ID
        """
        if experience_id not in self.store._experiences:
            logger.warning(f"[IntegratedExperienceStore] 经验不存在：{experience_id}")
            return
        
        exp = self.store._experiences[experience_id]
        exp.access_count += 1
        exp.last_accessed = datetime.utcnow().timestamp()
        
        # 更新时间标签
        if self.enable_time_tags and self.time_tag_manager:
            time_tags_data = exp.metadata.get('time_tags')
            
            if time_tags_data:
                time_tags = TimeTags.from_dict(time_tags_data)
                self.time_tag_manager.update_usage(time_tags)
                exp.metadata['time_tags'] = time_tags.to_dict()
        
        logger.debug(f"[IntegratedExperienceStore] 使用记录已更新：{experience_id}")
    
    async def _on_rollback(self, result: RollbackResult):
        """回滚回调
        
        Args:
            result: 回滚结果
        """
        logger.info(f"[Rollback] {result.experience_id}: {result.action.value} - {result.reason}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        store_stats = {
            'total_experiences': len(self.store._experiences),
            'bm25_docs': len(self.store.bm25_index.documents)
        }
        
        if self.time_tag_manager:
            store_stats['time_tags_enabled'] = True
        
        if self.rollback_manager:
            store_stats['rollback_stats'] = self.rollback_manager.get_stats()
        
        if self.review_trigger:
            store_stats['review_stats'] = self.review_trigger.get_stats()
        
        return {
            **self.stats,
            **store_stats
        }
    
    async def start(self):
        """启动集成经验库"""
        logger.info("[IntegratedExperienceStore] 启动中...")
        
        # 启动复盘触发器
        if self.review_trigger:
            await self.review_trigger.start()
            logger.info("[IntegratedExperienceStore] 复盘触发器已启动")
        
        logger.info("[IntegratedExperienceStore] 已启动")
    
    async def stop(self):
        """停止集成经验库"""
        logger.info("[IntegratedExperienceStore] 停止中...")
        
        # 停止复盘触发器
        if self.review_trigger:
            await self.review_trigger.stop()
            logger.info("[IntegratedExperienceStore] 复盘触发器已停止")
        
        logger.info("[IntegratedExperienceStore] 已停止")


# 全局单例
_integrated_store_instance = None


def get_integrated_experience_store(
    db_path: Optional[str] = None,
    enable_persistence: bool = True,
    enable_smart_tagging: bool = True,
    enable_review: bool = True,
    enable_time_tags: bool = True
) -> IntegratedExperienceStore:
    """获取集成经验库单例
    
    Args:
        db_path: 数据库路径
        enable_persistence: 是否启用持久化
        enable_smart_tagging: 是否启用智能打标
        enable_review: 是否启用复盘机制
        enable_time_tags: 是否启用时间标签
        
    Returns:
        IntegratedExperienceStore: 单例实例
    """
    global _integrated_store_instance
    
    if _integrated_store_instance is None:
        _integrated_store_instance = IntegratedExperienceStore(
            db_path=db_path,
            enable_persistence=enable_persistence,
            enable_smart_tagging=enable_smart_tagging,
            enable_review=enable_review,
            enable_time_tags=enable_time_tags
        )
    
    return _integrated_store_instance
