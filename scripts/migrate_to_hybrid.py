#!/usr/bin/env python3
"""
从 NetworkX MemoryGraph 迁移到 Hybrid 存储 (igraph + LMDB)

使用方法:
    python scripts/migrate_to_hybrid.py
    python scripts/migrate_to_hybrid.py --sharding  # 启用分片模式
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_memory_graph(
    source_path: str = "./data/memory_graph",
    target_dir: str = "./data/memory_graph_hybrid",
    use_sharding: bool = False
):
    """
    迁移 MemoryGraph 数据到混合存储
    
    Args:
        source_path: NetworkX MemoryGraph 数据路径
        target_dir: Hybrid 存储目标目录
        use_sharding: 是否使用分片模式
    """
    logger.info(f"=" * 60)
    logger.info(f"开始迁移: {source_path} → {target_dir}")
    logger.info(f"=" * 60)
    
    source_json = os.path.join(source_path, "memory_graph.json")
    if not os.path.exists(source_json):
        logger.error(f"源数据文件不存在: {source_json}")
        return False
    
    from zulong.memory.memory_graph import MemoryGraph
    
    logger.info("加载源数据 (NetworkX)...")
    source_graph = MemoryGraph(persist_path=source_path)
    node_count = len(source_graph.graph.nodes)
    edge_count = len(source_graph.graph.edges)
    logger.info(f"源数据加载完成: {node_count} 节点, {edge_count} 边")
    
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("初始化目标存储 (Hybrid)...")
    
    try:
        from zulong.memory.storage_hybrid import (
            ShardedMemoryGraph, 
            MemoryGraphHybrid,
            ShardStrategy
        )
    except ImportError as e:
        logger.error(f"无法导入 hybrid 存储: {e}")
        logger.error("请确保已安装依赖: pip install python-igraph lmdb msgspec")
        return False
    
    if use_sharding:
        target_graph = ShardedMemoryGraph(
            base_dir=target_dir,
            shard_strategy=ShardStrategy.MONTHLY,
            max_active_shards=3,
            map_size_gb=10
        )
    else:
        target_graph = MemoryGraphHybrid(
            data_dir=target_dir,
            shard_id="migrated",
            map_size_gb=10
        )
    
    logger.info("迁移节点...")
    migrated_nodes = 0
    start_time = time.time()
    
    for node_id, node_data in source_graph.graph.nodes(data=True):
        node_type = node_data.get("node_type", "unknown")
        label = node_data.get("label", node_id)
        
        try:
            success = target_graph.add_node(
                node_id=node_id,
                node_type=node_type,
                label=label,
                content=node_data.get("content"),
                importance=node_data.get("importance", "normal"),
                backend_ref=node_data.get("backend_ref", ""),
                metadata=node_data
            )
            
            if success:
                migrated_nodes += 1
                
        except Exception as e:
            logger.warning(f"节点迁移失败 {node_id}: {e}")
        
        if migrated_nodes > 0 and migrated_nodes % 1000 == 0:
            logger.info(f"已迁移 {migrated_nodes}/{node_count} 节点...")
    
    logger.info("迁移边...")
    migrated_edges = 0
    
    for src_id, dst_id, edge_data in source_graph.graph.edges(data=True):
        edge_type = edge_data.get("edge_type", "association")
        weight = edge_data.get("weight", 1.0)
        
        try:
            success = target_graph.add_edge(
                src_id=src_id,
                dst_id=dst_id,
                edge_type=edge_type,
                weight=weight,
                protected=edge_data.get("protected", False),
                metadata=edge_data
            )
            
            if success:
                migrated_edges += 1
                
        except Exception as e:
            logger.warning(f"边迁移失败 {src_id}→{dst_id}: {e}")
    
    logger.info("保存目标存储...")
    target_graph.save()
    
    elapsed = time.time() - start_time
    
    logger.info(f"=" * 60)
    logger.info(f"迁移完成!")
    logger.info(f"  节点: {migrated_nodes}/{node_count}")
    logger.info(f"  边: {migrated_edges}/{edge_count}")
    logger.info(f"  耗时: {elapsed:.2f}s")
    logger.info(f"=" * 60)
    
    if hasattr(target_graph, 'get_stats'):
        stats = target_graph.get_stats()
        logger.info(f"目标存储统计: {stats}")
    
    backup_path = source_json + ".bak"
    if os.path.exists(backup_path):
        os.remove(backup_path)
    os.rename(source_json, backup_path)
    logger.info(f"源数据已备份到: {backup_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="迁移 MemoryGraph 到 Hybrid 存储")
    parser.add_argument(
        "--source", 
        default="./data/memory_graph",
        help="源数据路径 (默认: ./data/memory_graph)"
    )
    parser.add_argument(
        "--target",
        default="./data/memory_graph_hybrid",
        help="目标路径 (默认: ./data/memory_graph_hybrid)"
    )
    parser.add_argument(
        "--sharding",
        action="store_true",
        help="启用分片模式 (适合大数据量)"
    )
    
    args = parser.parse_args()
    
    success = migrate_memory_graph(
        source_path=args.source,
        target_dir=args.target,
        use_sharding=args.sharding
    )
    
    if success:
        print("\n✅ 迁移成功!")
        print("\n下一步:")
        print("1. 确认 config/zulong_config.yaml 中 memory.hybrid_storage.enabled: true")
        print("2. 重启系统")
    else:
        print("\n❌ 迁移失败，请检查日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
