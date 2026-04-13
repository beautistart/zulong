"""
智能打标系统：多层渐进式打标

架构：
- Layer 1: 增强规则匹配（关键词 + 正则 + 权重）
- Layer 2: 语义相似度匹配（领域原型向量）
- Layer 3: 轻量级分类器（可选，FastText）

特点：
- 三层融合，置信度加权
- 默认标签兜底（general/unknown）
- 支持否定词检测
- 支持多标签输出

对应 TSD v2.2 第 9.5.2 节：经验库自动打标优化
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EnhancedRuleMatcher:
    """Layer 1: 增强版规则匹配器"""
    
    def __init__(self):
        # 1. 扩展关键词词典（带权重）
        self.domain_keywords = {
            "network": {
                "high": ["网络", "WiFi", "路由器", "网速", "DNS", "IP 地址", "局域网", "宽带"],
                "medium": ["网卡", "流量", "带宽", "网关", "子网", "信号", "连接"],
                "low": ["在线", "离线", "上传", "下载", "ping"]
            },
            "navigation": {
                "high": ["导航", "路径", "避障", "移动", "定位", "GPS", "地图", "路线"],
                "medium": ["方向", "坐标", "位置", "轨迹", "行驶", "行走"],
                "low": ["前进", "后退", "转弯", "停止", "启动"]
            },
            "manipulation": {
                "high": ["抓取", "操作", "物体", "机械臂", "夹持", "夹爪", "搬运"],
                "medium": ["放置", "拾取", "释放", "力度", "校准", "精度"],
                "low": ["拿", "放", "捏", "夹", "推", "拉"]
            },
            "vision": {
                "high": ["视觉", "图像", "识别", "检测", "摄像头", "相机", "画面"],
                "medium": ["像素", "颜色", "形状", "轮廓", "特征", "匹配"],
                "low": ["看", "看见", "显示", "出现", "消失"]
            },
            "dialog": {
                "high": ["对话", "聊天", "回复", "回答", "问题", "询问", "语音"],
                "medium": ["文本", "语义", "意图", "理解", "表达"],
                "low": ["说", "问", "讲", "听", "读"]
            },
            "planning": {
                "high": ["规划", "计划", "调度", "协调", "决策", "优化"],
                "medium": ["任务", "步骤", "顺序", "优先级"],
                "low": ["先", "后", "然后", "接着"]
            }
        }
        
        # 2. 正则表达式模式（高置信度）
        self.patterns = {
            "network": [
                r"网络\s*(慢 | 卡 | 断 | 差 | 不好)",
                r"WiFi\s*(连不上 | 信号弱 | 断开)",
                r"路由器\s*(重启 | 设置 | 配置 | 故障)",
                r"网速\s*(快 | 慢 | 卡 | 提升)",
                r"DNS\s*(配置 | 错误 | 设置)"
            ],
            "navigation": [
                r"导航\s*(失败 | 错误 | 规划 | 路线)",
                r"路径\s*(规划 | 优化 | 调整 | 计算)",
                r"避障\s*(成功 | 失败 | 检测)",
                r"定位\s*(校准 | 精度 | 丢失)"
            ],
            "manipulation": [
                r"抓取\s*(失败 | 成功 | 力度 | 校准)",
                r"机械臂\s*(控制 | 校准 | 操作 | 运动)",
                r"夹爪\s*(力度 | 校准 | 故障)",
                r"搬运\s*(任务 | 成功 | 失败)"
            ],
            "vision": [
                r"识别\s*(失败 | 准确 | 速度 | 率)",
                r"摄像头\s*(模糊 | 遮挡 | 校准 | 故障)",
                r"图像\s*(清晰 | 处理 | 分析)",
                r"检测\s*(到 | 失败 | 成功 | 结果)"
            ],
            "dialog": [
                r"回答\s*(错误 | 正确 | 满意 | 不上)",
                r"问题\s*(理解 | 识别 | 意图)",
                r"对话\s*(系统 | 流畅 | 自然)",
                r"回复\s*(太慢 | 太快 | 合适)"
            ]
        }
        
        # 3. 否定词列表（避免误判）
        self.negations = ["不", "没", "无", "非", "别", "莫", "未", "勿"]
        
        # 4. 领域别名映射
        self.domain_aliases = {
            "网络": "network",
            "导航": "navigation",
            "操作": "manipulation",
            "视觉": "vision",
            "对话": "dialog",
            "规划": "planning"
        }
    
    def match(self, text: str) -> Dict[str, float]:
        """匹配文本，返回领域置信度
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[str, float]: {领域：置信度 (0-1)}
        """
        scores = {}
        
        for domain, config in self.domain_keywords.items():
            score = 0.0
            
            # 1. 关键词匹配（带权重）
            for weight_level, weight in [
                ("high", 1.0),
                ("medium", 0.7),
                ("low", 0.4)
            ]:
                keywords = config.get(weight_level, [])
                for keyword in keywords:
                    if keyword.lower() in text.lower():
                        # 检查是否有否定词
                        if not self._has_negation(text, keyword):
                            score += weight * 0.15
                        else:
                            # 否定词存在，降低分数
                            score += weight * 0.05
            
            # 2. 正则匹配（高置信度）
            if domain in self.patterns:
                for pattern in self.patterns[domain]:
                    if re.search(pattern, text, re.IGNORECASE):
                        score += 0.4
            
            scores[domain] = min(1.0, score)
        
        return scores
    
    def _has_negation(self, text: str, keyword: str) -> bool:
        """检查关键词前是否有否定词
        
        Args:
            text: 完整文本
            keyword: 关键词
            
        Returns:
            bool: 是否有否定词
        """
        idx = text.find(keyword)
        if idx == -1:
            return False
        
        # 检查前 3 个字
        start = max(0, idx - 3)
        prefix = text[start:idx]
        
        return any(neg in prefix for neg in self.negations)


class SemanticSimilarityMatcher:
    """Layer 2: 语义相似度匹配器"""
    
    def __init__(self, embedding_model=None):
        self._embedding_model = embedding_model
        
        # 领域原型向量（每个领域的典型文本）
        self.domain_prototypes = {
            "network": [
                "网络速度慢，需要检查路由器设置并重启",
                "WiFi 信号弱，建议调整路由器位置或更换天线",
                "DNS 配置错误，导致无法上网，需要修改 DNS 服务器",
                "网速卡顿，可能是带宽不足或网络拥堵"
            ],
            "navigation": [
                "导航路径规划失败，需要重新计算路线并避开障碍物",
                "机器人避障功能正常，可以安全移动到目标位置",
                "定位系统校准完成，精度提升到厘米级",
                "路径优化成功，行驶时间缩短了 30%"
            ],
            "manipulation": [
                "机械臂抓取物体时力度控制很重要，需要根据物体材质调整",
                "夹爪校准完成，抓取精度提升到 0.1mm",
                "搬运任务执行成功，物体放置到指定位置",
                "操作力度过大，导致物体滑落，需要减小力度"
            ],
            "vision": [
                "视觉识别系统检测到物体位置，坐标为 (x=100, y=200)",
                "摄像头图像清晰，识别准确率达到 95%",
                "颜色检测功能正常，可以区分红、绿、蓝三种颜色",
                "图像识别失败，原因是摄像头被遮挡"
            ],
            "dialog": [
                "对话系统理解用户意图，给出了合适的回答",
                "聊天机器人回复自然，用户满意度高",
                "问题识别准确，提供了有用的信息和建议",
                "语音识别失败，原因是环境噪音太大"
            ],
            "planning": [
                "任务规划完成，确定了最优执行顺序",
                "调度系统协调多个机器人协同工作",
                "决策优化成功，整体效率提升了 25%",
                "计划调整，优先处理紧急任务"
            ]
        }
        
        # 预计算原型向量
        self._prototype_embeddings = {}
        if embedding_model is not None:
            self._compute_prototype_embeddings()
    
    def set_embedding_model(self, model):
        """设置 Embedding 模型"""
        self._embedding_model = model
        self._compute_prototype_embeddings()
    
    def _compute_prototype_embeddings(self):
        """预计算领域原型向量"""
        if self._embedding_model is None:
            return
        
        for domain, texts in self.domain_prototypes.items():
            try:
                embeddings = []
                for text in texts:
                    emb = self._embedding_model.encode([text])[0]
                    embeddings.append(emb)
                
                # 平均向量作为原型
                prototype = np.mean(embeddings, axis=0)
                # 归一化
                self._prototype_embeddings[domain] = prototype / np.linalg.norm(prototype)
                
            except Exception as e:
                logger.warning(f"[SemanticMatcher] 计算{domain}原型向量失败：{e}")
    
    def match(self, text: str, threshold: float = 0.55) -> Dict[str, float]:
        """计算文本与各领域的语义相似度
        
        Args:
            text: 输入文本
            threshold: 相似度阈值（低于此值不返回）
            
        Returns:
            Dict[str, float]: {领域：相似度 (0-1)}
        """
        if self._embedding_model is None or not self._prototype_embeddings:
            return {}
        
        try:
            # 计算文本向量
            text_embedding = self._embedding_model.encode([text])[0]
            text_embedding = text_embedding / np.linalg.norm(text_embedding)
            
            scores = {}
            for domain, prototype in self._prototype_embeddings.items():
                # 余弦相似度
                similarity = np.dot(text_embedding, prototype)
                # 归一化到 0-1（余弦相似度范围是 [-1, 1]）
                similarity = (similarity + 1) / 2
                
                # 阈值过滤
                if similarity >= threshold:
                    scores[domain] = similarity
            
            return scores
            
        except Exception as e:
            logger.warning(f"[SemanticMatcher] 计算语义相似度失败：{e}")
            return {}


class MultiLayerTagger:
    """多层智能打标器（三层融合）"""
    
    def __init__(self, embedding_model=None):
        # Layer 1: 规则匹配
        self.rule_matcher = EnhancedRuleMatcher()
        
        # Layer 2: 语义相似度
        self.semantic_matcher = SemanticSimilarityMatcher(embedding_model)
        
        # 配置
        self.use_semantic = embedding_model is not None
        
        # 置信度阈值
        self.thresholds = {
            "rule_high": 0.3,      # 规则匹配高置信度（降低阈值）
            "semantic_high": 0.65, # 语义匹配高置信度
            "min_confidence": 0.25  # 最低置信度
        }
        
        # 权重配置
        self.weights = {
            "rule": 0.5,      # 规则权重 50%
            "semantic": 0.5   # 语义权重 50%
        }
    
    def set_embedding_model(self, model):
        """设置 Embedding 模型"""
        self.semantic_matcher.set_embedding_model(model)
        self.use_semantic = True
    
    def tag(self, text: str, 
            use_default: bool = True,
            default_tag: str = "general",
            min_confidence: float = 0.3) -> List[Tuple[str, float]]:
        """智能打标
        
        Args:
            text: 输入文本
            use_default: 是否使用默认标签
            default_tag: 默认标签名称
            min_confidence: 最低置信度阈值
            
        Returns:
            List[Tuple[str, float]]: [(领域，置信度), ...] 按置信度降序
        """
        all_scores = {}
        
        # ========== Layer 1: 规则匹配 ==========
        rule_scores = self.rule_matcher.match(text)
        
        # 检查是否有高置信度匹配
        high_confidence_domains = [
            domain for domain, score in rule_scores.items()
            if score >= self.thresholds["rule_high"]
        ]
        
        if high_confidence_domains:
            # 高置信度直接返回（不再进行后续计算）
            final_scores = {
                domain: score for domain, score in rule_scores.items()
                if score >= self.thresholds["rule_high"]
            }
            return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 融合规则分数
        for domain, score in rule_scores.items():
            all_scores[domain] = score * self.weights["rule"]
        
        # ========== Layer 2: 语义相似度 ==========
        if self.use_semantic:
            semantic_scores = self.semantic_matcher.match(text)
            
            # 检查是否有高置信度匹配
            high_confidence_domains = [
                domain for domain, score in semantic_scores.items()
                if score >= self.thresholds["semantic_high"]
            ]
            
            if high_confidence_domains:
                # 高置信度直接返回
                return sorted(semantic_scores.items(), 
                            key=lambda x: x[1], reverse=True)
            
            # 融合语义分数
            for domain, score in semantic_scores.items():
                if domain in all_scores:
                    all_scores[domain] += score * self.weights["semantic"]
                else:
                    all_scores[domain] = score * self.weights["semantic"]
        
        # ========== 融合策略 ==========
        # 1. 过滤低置信度
        final_scores = {
            domain: score for domain, score in all_scores.items()
            if score >= min_confidence
        }
        
        # 2. 归一化（缩放到 0-1）
        if final_scores:
            max_score = max(final_scores.values())
            if max_score > 0:
                final_scores = {
                    domain: min(1.0, score / max_score) 
                    for domain, score in final_scores.items()
                }
        
        # 3. 默认标签
        if not final_scores and use_default:
            final_scores[default_tag] = 1.0
        
        # 4. 排序返回
        return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    
    def tag_with_explanation(self, text: str, **kwargs) -> Dict:
        """智能打标（带解释）
        
        Args:
            text: 输入文本
            **kwargs: 传递给 tag() 的参数
            
        Returns:
            Dict: {
                "tags": [(领域，置信度), ...],
                "explanation": "打标原因说明"
            }
        """
        tags = self.tag(text, **kwargs)
        
        # 生成解释
        if not tags:
            explanation = "未识别出明确领域，使用默认标签"
        elif len(tags) == 1 and tags[0][1] >= 0.8:
            explanation = f"高置信度识别为 {tags[0][0]} 领域"
        elif len(tags) == 1:
            explanation = f"识别为 {tags[0][0]} 领域（置信度：{tags[0][1]:.2f}）"
        else:
            domains = [tag[0] for tag in tags[:3]]
            explanation = f"多标签识别：{', '.join(domains)}"
        
        return {
            "tags": tags,
            "explanation": explanation
        }


# 全局单例
_tagger_instance = None


def get_smart_tagger(embedding_model=None) -> MultiLayerTagger:
    """获取智能打标器单例
    
    Args:
        embedding_model: Embedding 模型（可选）
        
    Returns:
        MultiLayerTagger: 单例实例
    """
    global _tagger_instance
    if _tagger_instance is None:
        _tagger_instance = MultiLayerTagger(embedding_model)
    elif embedding_model is not None and not _tagger_instance.use_semantic:
        # 如果已有实例但没有语义模型，更新它
        _tagger_instance.set_embedding_model(embedding_model)
    
    return _tagger_instance
