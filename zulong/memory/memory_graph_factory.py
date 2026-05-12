"""
MemoryGraph 工厂方法 - 支持切换 NetworkX / Hybrid 存储

根据配置自动选择存储后端：
- enabled: false → NetworkX + JSON (默认)
- enabled: true  → igraph + LMDB (高性能)
"""

import logging
from typing import Optional, Any, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


def create_memory_graph(persist_path: str = "./data/memory_graph", config: Optional[Dict] = None):
    """
    根据配置创建 MemoryGraph 实例
    
    Args:
        persist_path: 持久化路径
        config: 可选的配置字典（用于测试覆盖）
    
    Returns:
        MemoryGraph 或 MemoryGraphHybrid 实例
    """
    try:
        from zulong.config.config_manager import get_config
    except ImportError:
        get_config = lambda key, default=None: default
    
    hybrid_enabled = config.get('enabled') if config else get_config('memory.hybrid_storage.enabled', False)
    
    if not hybrid_enabled:
        logger.info("[MemoryGraphFactory] 使用 NetworkX 后端")
        from .memory_graph import MemoryGraph
        return MemoryGraph(persist_path=persist_path)
    
    logger.info("[MemoryGraphFactory] 使用 Hybrid 存储后端 (igraph + LMDB)")
    
    try:
        from .storage_hybrid import ShardedMemoryGraph, MemoryGraphHybrid, ShardStrategy
    except ImportError as e:
        logger.error(f"[MemoryGraphFactory] 无法导入 hybrid 存储: {e}")
        logger.warning("[MemoryGraphFactory] 回退到 NetworkX 后端")
        from .memory_graph import MemoryGraph
        return MemoryGraph(persist_path=persist_path)
    
    data_dir = config.get('data_dir') if config else get_config('memory.hybrid_storage.data_dir', './data/memory_graph_hybrid')
    map_size_gb = config.get('map_size_gb') if config else get_config('memory.hybrid_storage.map_size_gb', 10)
    shard_strategy = config.get('shard_strategy') if config else get_config('memory.hybrid_storage.shard_strategy', 'month')
    max_active_shards = config.get('max_active_shards') if config else get_config('memory.hybrid_storage.max_active_shards', 3)
    enable_vector_index = config.get('enable_vector_index') if config else get_config('memory.hybrid_storage.enable_vector_index', False)
    use_sharding = config.get('use_sharding') if config else get_config('memory.hybrid_storage.use_sharding', False)
    
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    if use_sharding:
        logger.info(f"[MemoryGraphFactory] 启用分片策略: {shard_strategy}")
        strategy_map = {
            'month': ShardStrategy.MONTHLY,
            'week': ShardStrategy.WEEKLY,
            'day': ShardStrategy.DAILY,
        }
        return ShardedMemoryGraph(
            base_dir=data_dir,
            shard_strategy=strategy_map.get(shard_strategy, ShardStrategy.MONTHLY),
            max_active_shards=max_active_shards,
            map_size_gb=map_size_gb,
            enable_vector_index=enable_vector_index
        )
    
    return MemoryGraphHybrid(
        data_dir=data_dir,
        shard_id="default",
        map_size_gb=map_size_gb,
        enable_vector_index=enable_vector_index
    )


def get_memory_graph_type(graph_instance) -> str:
    """获取图谱实例的类型"""
    class_name = graph_instance.__class__.__name__
    if 'Hybrid' in class_name or 'Sharded' in class_name:
        return "hybrid"
    return "networkx"


def get_memory_graph_stats(graph_instance) -> Dict[str, Any]:
    """获取存储统计信息"""
    result = {
        "type": get_memory_graph_type(graph_instance),
        "node_count": len(graph_instance) if hasattr(graph_instance, '__len__') else 0,
    }
    
    if hasattr(graph_instance, 'get_stats'):
        result.update(graph_instance.get_stats())
    
    return result
