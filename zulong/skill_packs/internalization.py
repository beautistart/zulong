# File: zulong/skill_packs/internalization.py
"""
内化完成度评估模块

评估祖龙系统是否已经"学会"了某个技能包的能力，
从而可以安全地卸载该技能包而不影响功能。

评估维度：
1. 经验数量：该技能包积累的执行经验数量
2. 成功率：经验中成功执行的比例
3. 一致性：最近N次执行结果的稳定性
4. 补丁效果：相关 SystemPatch 的应用成功率
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InternalizationConfig:
    """内化评估配置"""
    min_experience_count: int = 50       # 至少积累50条经验
    min_success_rate: float = 0.9         # 成功率>90%
    min_consistency: float = 0.85         # 一致性>85%
    evaluation_window: int = 20           # 最近N次执行的一致性检查窗口
    weight_experience: float = 0.4       # 经验数量权重
    weight_success: float = 0.35         # 成功率权重
    weight_consistency: float = 0.25     # 一致性权重


def evaluate_internalization(
    pack_id: str,
    experience_store=None,
    hot_update_engine=None,
    config: Optional[InternalizationConfig] = None
) -> Dict[str, Any]:
    """评估技能包内化完成度
    
    Args:
        pack_id: 技能包 ID
        experience_store: 经验存储实例
        hot_update_engine: 热更新引擎
        config: 评估配置
    
    Returns:
        评估结果：{
            "score": 0.0-1.0,          # 综合完成度
            "is_ready": bool,           # 是否可以卸载
            "experience_count": int,    # 经验数量
            "success_rate": float,      # 成功率
            "consistency": float,       # 一致性
            "detail": str               # 详细说明
        }
    """
    config = config or InternalizationConfig()
    
    result = {
        "pack_id": pack_id,
        "score": 0.0,
        "is_ready": False,
        "experience_count": 0,
        "success_rate": 0.0,
        "consistency": 0.0,
        "detail": "",
    }
    
    # 1. 获取经验数量
    experience_count = 0
    if experience_store is not None:
        try:
            experience_count = experience_store.get_count_by_type(
                "skill_pack_execution",
                pack_id
            )
        except (AttributeError, TypeError):
            # 降级：尝试直接获取总数
            try:
                all_experiences = experience_store.get_all()
                experience_count = sum(
                    1 for e in all_experiences
                    if e.get("type") == "skill_pack_execution" and e.get("pack_id") == pack_id
                )
            except Exception:
                experience_count = 0
    
    result["experience_count"] = experience_count
    
    # 2. 获取成功率
    success_rate = 0.0
    if experience_store is not None and experience_count > 0:
        try:
            experiences = experience_store.get_by_type("skill_pack_execution", pack_id)
            if experiences:
                success_count = sum(1 for e in experiences if e.get("success", False))
                success_rate = success_count / len(experiences)
        except (AttributeError, TypeError):
            success_rate = 0.5  # 默认值（信息不足时保守估计）
    
    result["success_rate"] = success_rate
    
    # 3. 计算一致性（最近N次执行结果的方差）
    consistency = 0.0
    if experience_store is not None and experience_count >= config.evaluation_window:
        try:
            experiences = experience_store.get_by_type("skill_pack_execution", pack_id)
            recent = experiences[-config.evaluation_window:]
            
            # 计算成功率的稳定性
            if len(recent) > 0:
                success_values = [1 if e.get("success", False) else 0 for e in recent]
                avg = sum(success_values) / len(success_values)
                # 一致性 = 1 - 方差（方差越小越一致）
                variance = sum((v - avg) ** 2 for v in success_values) / len(success_values)
                consistency = max(0, 1 - variance)  # 归一化到 0-1
        except Exception:
            consistency = 0.5
    
    result["consistency"] = consistency
    
    # 4. 计算综合评分
    experience_score = min(experience_count / config.min_experience_count, 1.0)
    
    result["score"] = (
        experience_score * config.weight_experience
        + success_rate * config.weight_success
        + consistency * config.weight_consistency
    )
    
    # 5. 判断是否可以卸载
    result["is_ready"] = (
        experience_count >= config.min_experience_count
        and success_rate >= config.min_success_rate
        and consistency >= config.min_consistency
        and result["score"] >= 0.9
    )
    
    # 6. 生成详细说明
    result["detail"] = _generate_detail(result, config)
    
    return result


def _generate_detail(result: Dict[str, Any], config: InternalizationConfig) -> str:
    """生成人类可读的评估说明"""
    parts = []
    
    # 经验数量
    exp_count = result["experience_count"]
    exp_target = config.min_experience_count
    if exp_count >= exp_target:
        parts.append(f"经验数量充足 ({exp_count}/{exp_target})")
    else:
        parts.append(f"经验数量不足 ({exp_count}/{exp_target})")
    
    # 成功率
    sr = result["success_rate"]
    sr_target = config.min_success_rate
    if sr >= sr_target:
        parts.append(f"成功率达标 ({sr:.1%})")
    else:
        parts.append(f"成功率未达标 ({sr:.1%} < {sr_target:.1%})")
    
    # 一致性
    con = result["consistency"]
    con_target = config.min_consistency
    if con >= con_target:
        parts.append(f"执行一致性良好 ({con:.1%})")
    else:
        parts.append(f"执行一致性不足 ({con:.1%} < {con_target:.1%})")
    
    # 综合
    score = result["score"]
    is_ready = result["is_ready"]
    
    if is_ready:
        summary = f"技能包已内化，可以安全卸载。综合完成度: {score:.1%}"
    else:
        summary = f"技能包尚未内化。综合完成度: {score:.1%} (需达90%)"
    
    return summary + " | " + "，".join(parts)
