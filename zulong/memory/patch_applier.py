# File: zulong/memory/patch_applier.py
# 祖龙 (ZULONG) 补丁应用器 - 将热补丁应用到各层

"""
补丁应用器 - 将经验转换为实际的参数调整

功能:
1. 应用到 L0 执行器（原子动作参数）
2. 应用到 L1-A 反射层（反射规则）
3. 应用到 L1-B 调度层（调度策略）
4. 参数验证和安全性检查

对应 TSD v2.3 第 10.3.2 节：参数热更新
"""

import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import asyncio

from .hot_update_engine import SystemPatch, PatchType

logger = logging.getLogger(__name__)


@dataclass
class ParameterConfig:
    """参数配置"""
    name: str
    value: Any
    min_value: Optional[Any] = None  # 最小值
    max_value: Optional[Any] = None  # 最大值
    default: Any = None
    description: str = ""


class PatchApplier:
    """补丁应用器"""
    
    def __init__(self):
        """初始化应用器"""
        # L0 层参数注册表
        self._l0_parameters: Dict[str, ParameterConfig] = {}
        
        # L1-A 层规则注册表
        self._l1a_rules: Dict[str, Dict[str, Any]] = {}
        
        # L1-B 层策略注册表
        self._l1b_strategies: Dict[str, Dict[str, Any]] = {}
        
        # 参数验证器
        self._validators: Dict[str, Callable] = {}
        
        # 应用历史
        self._apply_history = []
        
        logger.info("[PatchApplier] 初始化完成")
    
    def register_l0_parameter(
        self,
        name: str,
        default: Any,
        min_value: Optional[Any] = None,
        max_value: Optional[Any] = None,
        description: str = ""
    ):
        """
        注册 L0 层参数
        
        Args:
            name: 参数名称
            default: 默认值
            min_value: 最小值
            max_value: 最大值
            description: 描述
        """
        self._l0_parameters[name] = ParameterConfig(
            name=name,
            value=default,
            min_value=min_value,
            max_value=max_value,
            default=default,
            description=description
        )
        
        logger.info(f"[PatchApplier] 已注册 L0 参数：{name} = {default}")
    
    def register_l1a_rule(
        self,
        rule_id: str,
        rule_data: Dict[str, Any]
    ):
        """
        注册 L1-A 层规则
        
        Args:
            rule_id: 规则 ID
            rule_data: 规则数据
        """
        self._l1a_rules[rule_id] = rule_data.copy()
        logger.info(f"[PatchApplier] 已注册 L1-A 规则：{rule_id}")
    
    def register_l1b_strategy(
        self,
        strategy_id: str,
        strategy_data: Dict[str, Any]
    ):
        """
        注册 L1-B 层策略
        
        Args:
            strategy_id: 策略 ID
            strategy_data: 策略数据
        """
        self._l1b_strategies[strategy_id] = strategy_data.copy()
        logger.info(f"[PatchApplier] 已注册 L1-B 策略：{strategy_id}")
    
    def register_validator(
        self,
        parameter_name: str,
        validator: Callable[[Any], bool]
    ):
        """
        注册参数验证器
        
        Args:
            parameter_name: 参数名称
            validator: 验证函数
        """
        self._validators[parameter_name] = validator
        logger.info(f"[PatchApplier] 已注册验证器：{parameter_name}")
    
    async def apply_to_l0(self, patch: SystemPatch) -> bool:
        """
        应用补丁到 L0 执行器
        
        Args:
            patch: 补丁对象
        
        Returns:
            bool: 是否成功应用
        """
        try:
            logger.info(f"[PatchApplier] 应用 L0 补丁：{patch.patch_id}")
            
            # 1. 验证补丁类型
            if patch.patch_type not in [PatchType.PARAMETER, PatchType.THRESHOLD]:
                logger.warning(f"[PatchApplier] L0 补丁类型不匹配：{patch.patch_type}")
                return False
            
            # 2. 验证调整内容
            adjustments = patch.adjustment
            for param_name, new_value in adjustments.items():
                if not self._validate_parameter(param_name, new_value):
                    logger.error(f"[PatchApplier] 参数验证失败：{param_name} = {new_value}")
                    return False
            
            # 3. 应用调整
            for param_name, new_value in adjustments.items():
                if param_name in self._l0_parameters:
                    old_value = self._l0_parameters[param_name].value
                    self._l0_parameters[param_name].value = new_value
                    
                    logger.info(
                        f"[PatchApplier] L0 参数更新：{param_name} "
                        f"{old_value} → {new_value}"
                    )
                    
                    # 记录历史
                    self._record_change(
                        layer="l0",
                        param_name=param_name,
                        old_value=old_value,
                        new_value=new_value,
                        patch_id=patch.patch_id
                    )
                else:
                    logger.warning(f"[PatchApplier] 参数未注册：{param_name}")
            
            logger.info(f"[PatchApplier] L0 补丁应用完成：{patch.patch_id}")
            return True
        
        except Exception as e:
            logger.error(f"[PatchApplier] L0 补丁应用失败：{e}")
            return False
    
    async def apply_to_l1a(self, patch: SystemPatch) -> bool:
        """
        应用补丁到 L1-A 反射层
        
        Args:
            patch: 补丁对象
        
        Returns:
            bool: 是否成功应用
        """
        try:
            logger.info(f"[PatchApplier] 应用 L1-A 补丁：{patch.patch_id}")
            
            # 1. 验证补丁类型
            if patch.patch_type not in [PatchType.RULE, PatchType.STRATEGY]:
                logger.warning(f"[PatchApplier] L1-A 补丁类型不匹配：{patch.patch_type}")
                return False
            
            # 2. 应用规则更新
            adjustments = patch.adjustment
            
            if 'rules' in adjustments:
                for rule_id, rule_data in adjustments['rules'].items():
                    if rule_id in self._l1a_rules:
                        old_data = self._l1a_rules[rule_id].copy()
                        self._l1a_rules[rule_id].update(rule_data)
                        
                        logger.info(
                            f"[PatchApplier] L1-A 规则更新：{rule_id}"
                        )
                        
                        self._record_change(
                            layer="l1a",
                            param_name=rule_id,
                            old_value=old_data,
                            new_value=self._l1a_rules[rule_id],
                            patch_id=patch.patch_id
                        )
                    else:
                        logger.warning(f"[PatchApplier] 规则未注册：{rule_id}")
            
            logger.info(f"[PatchApplier] L1-A 补丁应用完成：{patch.patch_id}")
            return True
        
        except Exception as e:
            logger.error(f"[PatchApplier] L1-A 补丁应用失败：{e}")
            return False
    
    async def apply_to_l1b(self, patch: SystemPatch) -> bool:
        """
        应用补丁到 L1-B 调度层
        
        Args:
            patch: 补丁对象
        
        Returns:
            bool: 是否成功应用
        """
        try:
            logger.info(f"[PatchApplier] 应用 L1-B 补丁：{patch.patch_id}")
            
            # 1. 验证补丁类型
            if patch.patch_type not in [PatchType.STRATEGY, PatchType.PARAMETER]:
                logger.warning(f"[PatchApplier] L1-B 补丁类型不匹配：{patch.patch_type}")
                return False
            
            # 2. 应用策略调整
            adjustments = patch.adjustment
            
            if 'strategies' in adjustments:
                for strategy_id, strategy_data in adjustments['strategies'].items():
                    if strategy_id in self._l1b_strategies:
                        old_data = self._l1b_strategies[strategy_id].copy()
                        self._l1b_strategies[strategy_id].update(strategy_data)
                        
                        logger.info(
                            f"[PatchApplier] L1-B 策略更新：{strategy_id}"
                        )
                        
                        self._record_change(
                            layer="l1b",
                            param_name=strategy_id,
                            old_value=old_data,
                            new_value=self._l1b_strategies[strategy_id],
                            patch_id=patch.patch_id
                        )
                    else:
                        logger.warning(f"[PatchApplier] 策略未注册：{strategy_id}")
            
            logger.info(f"[PatchApplier] L1-B 补丁应用完成：{patch.patch_id}")
            return True
        
        except Exception as e:
            logger.error(f"[PatchApplier] L1-B 补丁应用失败：{e}")
            return False
    
    def _validate_parameter(self, param_name: str, value: Any) -> bool:
        """
        验证参数
        
        Args:
            param_name: 参数名称
            value: 参数值
        
        Returns:
            bool: 是否有效
        """
        # 1. 检查参数是否注册
        if param_name not in self._l0_parameters:
            logger.warning(f"[PatchApplier] 参数未注册：{param_name}")
            # 未注册的参数也允许（可能是动态参数）
        
        # 2. 使用自定义验证器
        if param_name in self._validators:
            validator = self._validators[param_name]
            if not validator(value):
                logger.error(f"[PatchApplier] 自定义验证失败：{param_name}")
                return False
        
        # 3. 检查范围
        if param_name in self._l0_parameters:
            config = self._l0_parameters[param_name]
            
            if config.min_value is not None and value < config.min_value:
                logger.error(f"[PatchApplier] 参数值小于最小值：{param_name} = {value}")
                return False
            
            if config.max_value is not None and value > config.max_value:
                logger.error(f"[PatchApplier] 参数值大于最大值：{param_name} = {value}")
                return False
        
        return True
    
    def _record_change(
        self,
        layer: str,
        param_name: str,
        old_value: Any,
        new_value: Any,
        patch_id: str
    ):
        """
        记录变更历史
        
        Args:
            layer: 层级
            param_name: 参数名
            old_value: 旧值
            new_value: 新值
            patch_id: 补丁 ID
        """
        from datetime import datetime
        
        self._apply_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'layer': layer,
            'param_name': param_name,
            'old_value': old_value,
            'new_value': new_value,
            'patch_id': patch_id
        })
        
        # 限制历史记录大小
        if len(self._apply_history) > 1000:
            self._apply_history = self._apply_history[-500:]
    
    def get_parameter(self, param_name: str) -> Any:
        """
        获取参数值
        
        Args:
            param_name: 参数名称
        
        Returns:
            参数值
        """
        if param_name in self._l0_parameters:
            return self._l0_parameters[param_name].value
        return None
    
    def get_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """
        获取规则
        
        Args:
            rule_id: 规则 ID
        
        Returns:
            规则数据
        """
        return self._l1a_rules.get(rule_id)
    
    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        获取策略
        
        Args:
            strategy_id: 策略 ID
        
        Returns:
            策略数据
        """
        return self._l1b_strategies.get(strategy_id)
    
    def get_apply_history(self, limit: int = 100) -> list:
        """
        获取应用历史
        
        Args:
            limit: 返回数量
        
        Returns:
            历史记录列表
        """
        return self._apply_history[-limit:]


# 单例模式
_patch_applier: Optional[PatchApplier] = None


def get_patch_applier() -> PatchApplier:
    """
    获取补丁应用器单例
    
    Returns:
        PatchApplier 实例
    """
    global _patch_applier
    
    if _patch_applier is None:
        _patch_applier = PatchApplier()
    
    return _patch_applier
