# 增强版经验库架构设计文档

**版本**: v1.0  
**日期**: 2026-03-29  
**作者**: ZULONG 架构团队

---

## 📋 概述

本文档描述祖龙系统经验库的四大增强功能实现与集成方案。

### 四大核心功能

1. ✅ **真实 Embedding 模型集成** - 支持 sentence-transformers、LangChain 等真实 Embedding 模型
2. ✅ **混合检索（向量 + BM25）** - 结合语义理解和关键词匹配的优势
3. ✅ **时间衰减因子** - 新经验权重更高，老经验自动衰减
4. ✅ **多标签组合过滤** - 支持 OR/AND 逻辑的灵活标签过滤

---

## 🏗️ 架构设计

### 核心组件

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  EnhancedExperienceStore (增强版经验库)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────┐     ┌──────────────────────┐                     │
│  │  Embedding Manager   │     │   BM25 Index Engine  │                     │
│  │  - sentence-trans.   │     │   - 中文分词         │                     │
│  │  - LangChain         │     │   - IDF 计算          │                     │
│  │  - 降级方案          │     │   - 词频统计         │                     │
│  └──────────────────────┘     └──────────────────────┘                     │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              Hybrid Search Engine (混合检索引擎)                      │  │
│  │                                                                       │  │
│  │  输入：query_text → query_vector + query_tokens                      │  │
│  │              ↓                                                        │  │
│  │  ┌──────────────────┐    ┌──────────────────┐                        │  │
│  │  │ Vector Search    │    │ BM25 Search      │                        │  │
│  │  │ (余弦相似度)      │    │ (关键词匹配)      │                        │  │
│  │  │ score_v          │    │ score_b          │                        │  │
│  │  └──────────────────┘    └──────────────────┘                        │  │
│  │              ↓                      ↓                                 │  │
│  │         合并：final_score = alpha * score_v + (1-alpha) * score_b    │  │
│  │              ↓                                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────┐     │  │
│  │  │ 时间衰减 × 重要性权重 × 热度补偿                             │     │  │
│  │  │ final_score *= exp(-decay * age) * importance * heat_factor │     │  │
│  │  └─────────────────────────────────────────────────────────────┘     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              Multi-Tag Filter Engine (多标签过滤引擎)                 │  │
│  │                                                                       │  │
│  │  filter_types: ["logic", "success"]  → 类型过滤                      │  │
│  │  filter_tags: ["network", "hardware"] → 标签过滤                     │  │
│  │  tag_logic: OR / AND → 过滤逻辑                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 数据结构增强

### Experience 数据类（增强版）

```python
@dataclass
class Experience:
    # 基础字段
    id: str
    content: str
    experience_type: str  # logic/failure/success/preference
    task_id: Optional[str]
    success: bool
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray]
    timestamp: float
    
    # 新增字段（增强功能）
    keywords: List[str] = field(default_factory=list)      # 关键词（BM25）
    tags: List[str] = field(default_factory=list)         # 多标签（过滤）
    importance_score: float = 1.0                         # 重要性权重
    access_count: int = 0                                 # 访问次数（热度）
    last_accessed: float = field(default_factory=time.time)
```

---

## 🔧 功能详解

### 1️⃣ 真实 Embedding 模型集成

#### 支持的模型接口

```python
# 1. sentence-transformers 风格
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
embedding = model.encode([text])[0]

# 2. LangChain 风格
from langchain.embeddings import HuggingFaceEmbeddings
model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
embedding = np.array(model.embed_query(text))

# 3. 通用接口
class CustomEmbeddingModel:
    def encode(self, texts):
        # 自定义实现
        return embeddings
```

#### 降级方案

```python
def _get_embedding(self, text: str) -> np.ndarray:
    if self._embedding_model is None:
        # 降级：使用模拟向量
        logger.warning("Embedding 模型未加载，使用模拟向量")
        return np.random.rand(768).astype(np.float32)
    
    try:
        # 尝试真实 Embedding
        if hasattr(self._embedding_model, 'encode'):
            embedding = self._embedding_model.encode([text])[0]
        elif hasattr(self._embedding_model, 'embed_query'):
            embedding = np.array(self._embedding_model.embed_query(text))
        
        # 归一化
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Embedding 失败：{e}")
        return np.random.rand(768).astype(np.float32)  # 降级
```

---

### 2️⃣ 混合检索（向量 + BM25）

#### 检索策略对比

| 检索类型 | 优势 | 劣势 | 适用场景 |
|---------|------|------|---------|
| **向量检索** | 语义理解<br>同义词匹配<br>拼写容错 | 计算量大<br>需要 Embedding<br>关键词不敏感 | 语义查询<br>模糊匹配<br>概念检索 |
| **BM25 检索** | 关键词精确匹配<br>计算快速<br>可解释性强 | 无法理解语义<br>同义词无法匹配<br>拼写敏感 | 精确查询<br>术语检索<br>专有名词 |
| **混合检索** | 结合两者优势<br>可调节权重<br>适应性强 | 实现复杂<br>需要调优参数 | 通用场景<br>复杂查询 |

#### 混合得分计算

```python
# 1. 向量相似度（余弦）
vector_score = cosine_similarity(query_vector, exp_embedding)

# 2. BM25 关键词得分
bm25_score = bm25_index.search(query_text)

# 3. 加权合并
combined_score = (
    alpha * vector_score +           # 向量权重（默认 0.7）
    (1 - alpha) * bm25_score         # BM25 权重（默认 0.3）
)

# 4. 应用时间衰减
time_factor = exp(-decay_factor * age_days)
final_score = combined_score * time_factor

# 5. 应用重要性权重
final_score *= importance_score

# 6. 应用热度补偿
heat_factor = log(access_count + 1) / log(100)
final_score *= (1 + 0.1 * heat_factor)
```

#### 参数配置建议

```python
store.configure_hybrid_search(
    alpha=0.7,          # 向量权重 70%，BM25 权重 30%
    time_decay=0.05,    # 每天衰减 5%
    max_age_days=30     # 最大保留 30 天
)
```

**不同场景的 alpha 值建议**:

| 场景 | alpha 建议 | 说明 |
|------|----------|------|
| 技术文档检索 | 0.5-0.6 | 关键词重要，语义也重要 |
| 对话系统 | 0.7-0.8 | 语义理解更重要 |
| 法律/医疗检索 | 0.4-0.5 | 精确关键词匹配 |
| 通用场景 | 0.7 | 平衡语义和关键词 |

---

### 3️⃣ 时间衰减因子

#### 衰减公式

```
final_score = base_score × exp(-decay_factor × age_days)
```

**衰减曲线** (decay=0.05):

```
年龄 (天) | 衰减因子 | 得分保留
---------|---------|----------
0        | 1.00    | 100%
1        | 0.95    | 95%
3        | 0.86    | 86%
7        | 0.70    | 70%
14       | 0.50    | 50%
30       | 0.22    | 22%
>30      | 0.00    | 淘汰
```

#### 实现代码

```python
def apply_time_decay(self, base_score: float, timestamp: float) -> float:
    current_time = time.time()
    age_days = (current_time - timestamp) / (24 * 3600)
    
    # 超过最大年龄，直接淘汰
    if age_days > self.max_age_days:
        return 0.0
    
    # 指数衰减
    time_factor = np.exp(-self.time_decay_factor * age_days)
    return base_score * time_factor
```

#### 热度补偿

为了防止有价值的老经验被过度衰减，引入热度补偿：

```python
heat_factor = np.log(access_count + 1) / np.log(100)  # 归一化到 0-1
final_score *= (1 + 0.1 * heat_factor)  # 最多提升 10%
```

**效果**:
- 从未访问：heat_factor = 0，无补偿
- 访问 10 次：heat_factor ≈ 0.5，提升 5%
- 访问 100 次：heat_factor = 1.0，提升 10%

---

### 4️⃣ 多标签组合过滤

#### 过滤逻辑

```python
# 1. 类型过滤（支持多类型）
if filter_types and exp.experience_type not in filter_types:
    continue

# 2. 标签过滤
if filter_tags:
    if tag_logic == "AND":
        # 必须包含所有标签
        if not all(tag in exp.tags for tag in filter_tags):
            continue
    else:  # OR
        # 至少包含一个标签
        if not any(tag in exp.tags for tag in filter_tags):
            continue
```

#### 使用示例

```python
# 示例 1: OR 逻辑（宽松）
results = store.search_by_text(
    query="网络问题",
    filter_tags=["network", "hardware"],
    tag_logic="OR"  # 包含 network 或 hardware 即可
)

# 返回：
# - 经验 A (标签：network, troubleshooting) ✓
# - 经验 B (标签：hardware, failure) ✓
# - 经验 C (标签：network, optimization) ✓

# 示例 2: AND 逻辑（严格）
results = store.search_by_text(
    query="网络硬件问题",
    filter_tags=["network", "hardware"],
    tag_logic="AND"  # 必须同时包含 network 和 hardware
)

# 返回：
# - 经验 D (标签：network, hardware, failure) ✓
# - 经验 A (标签：network, troubleshooting) ✗ (缺少 hardware)
# - 经验 B (标签：hardware, failure) ✗ (缺少 network)

# 示例 3: 类型 + 标签组合过滤
results = store.search_by_text(
    query="网络优化",
    filter_types=["logic", "success"],  # 只检索 logic 或 success 类型
    filter_tags=["network"],
    tag_logic="OR"
)
```

#### 自动打标

系统支持基于内容自动识别领域标签：

```python
def _extract_tags(self, content: str, experience_type: str) -> List[str]:
    tags = set()
    tags.add(experience_type)  # 添加类型标签
    
    # 领域识别（基于关键词）
    domain_keywords = {
        "network": ["网络", "WiFi", "路由器", "网速", "DNS"],
        "navigation": ["导航", "路径", "避障", "移动", "定位"],
        "manipulation": ["抓取", "操作", "物体", "机械臂", "夹持"],
        "vision": ["视觉", "图像", "识别", "检测", "摄像头"],
        "dialog": ["对话", "聊天", "回复", "回答", "问题"]
    }
    
    for domain, keywords in domain_keywords.items():
        for keyword in keywords:
            if keyword.lower() in content.lower():
                tags.add(domain)
                break
    
    return list(tags)
```

---

## 🔌 集成方案

### 方案 A: 完全替换（推荐）

**步骤**:

1. **备份现有经验库**
```python
# 备份旧版本数据
old_store = get_experience_store()
backup_data = old_store._experiences
```

2. **初始化增强版**
```python
from zulong.memory.enhanced_experience_store import (
    get_enhanced_experience_store
)

new_store = get_enhanced_experience_store()
```

3. **配置参数**
```python
new_store.configure_hybrid_search(
    alpha=0.7,
    time_decay=0.05,
    max_age_days=30
)
```

4. **设置 Embedding 模型**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
new_store.set_embedding_model(model)
```

5. **迁移数据**
```python
for exp_id, exp in backup_data.items():
    new_store.add_experience(
        content=exp.content,
        experience_type=exp.experience_type,
        success=exp.success,
        metadata=exp.metadata,
        importance_score=1.0
    )
```

6. **更新 ThreeLibraryManager**
```python
# 在 three_libraries.py 中
class ThreeLibraryManager:
    async def retrieve_all(self, query: str, ...):
        # 使用增强版经验库
        experiences = self.enhanced_experience_store.search_by_text(
            query, 
            filter_types=experience_type,
            use_hybrid=True,
            apply_time_decay=True,
            limit=experience_limit
        )
```

---

### 方案 B: 并行运行（保守）

**步骤**:

1. **保留旧版本**
```python
from zulong.memory.three_libraries import get_experience_store
from zulong.memory.enhanced_experience_store import get_enhanced_experience_store

old_store = get_experience_store()
new_store = get_enhanced_experience_store()
```

2. **A/B 测试**
```python
# 50% 流量使用新版
if random.random() < 0.5:
    results = new_store.search_by_text(query, ...)
else:
    results = old_store.search_by_text(query, ...)
```

3. **对比效果**
```python
# 记录点击率、满意度等指标
track_metrics(store_version, results, user_feedback)
```

4. **逐步切换**
```python
# 根据效果逐步提高新版比例
ab_test_ratio = 0.5  # → 0.7 → 0.9 → 1.0
```

---

## 📝 API 使用示例

### 基础使用

```python
from zulong.memory.enhanced_experience_store import get_enhanced_experience_store

# 1. 获取单例
store = get_enhanced_experience_store()

# 2. 配置
store.configure_hybrid_search(
    alpha=0.7,
    time_decay=0.05,
    max_age_days=30
)

# 3. 设置 Embedding 模型
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
store.set_embedding_model(model)

# 4. 添加经验
exp_id = store.add_experience(
    content="当用户抱怨网络慢时，应引导其检查路由器是否过热",
    experience_type="logic",
    success=True,
    tags=["network", "troubleshooting", "hardware"],
    importance_score=0.9
)

# 5. 检索
results = store.search_by_text(
    query="网络慢怎么办",
    filter_types=["logic", "success"],
    filter_tags=["network"],
    tag_logic="OR",
    use_hybrid=True,
    apply_time_decay=True,
    limit=5
)

# 6. 生成 Prompt
prompt_context = store.to_prompt_context(results)
print(prompt_context)
```

### 高级使用

```python
# 1. 多标签 AND 过滤
results = store.search_by_text(
    query="网络硬件故障",
    filter_tags=["network", "hardware"],
    tag_logic="AND",  # 必须同时包含
    limit=5
)

# 2. 纯关键词检索（alpha=0.0）
store.configure_hybrid_search(alpha=0.0)
results = store.search_by_text("路由器重启", use_hybrid=True)

# 3. 纯向量检索（alpha=1.0）
store.configure_hybrid_search(alpha=1.0)
results = store.search_by_text("网速卡", use_hybrid=True)

# 4. 不应用时间衰减
results = store.search_by_text(
    query="历史经验",
    apply_time_decay=False,
    limit=10
)

# 5. 获取统计信息
stats = store.get_statistics()
print(f"总经验数：{stats['total']}")
print(f"类型分布：{stats['type_distribution']}")
print(f"Top 标签：{stats['top_tags']}")
```

---

## 🎯 性能优化建议

### 1. Embedding 模型选择

| 模型 | 维度 | 速度 | 精度 | 推荐场景 |
|------|------|------|------|---------|
| BAAI/bge-small-zh-v1.5 | 512 | ⚡⚡⚡ | ⭐⭐⭐ | 资源受限 |
| BAAI/bge-base-zh-v1.5 | 768 | ⚡⚡ | ⭐⭐⭐⭐ | 通用场景 |
| BAAI/bge-large-zh-v1.5 | 1024 | ⚡ | ⭐⭐⭐⭐⭐ | 高精度需求 |

### 2. BM25 分词优化

```python
# 当前：简单字符分割（不推荐）
tokens = list(text.replace(' ', ''))

# 推荐：集成 jieba 分词
import jieba
tokens = list(jieba.cut(text))

# 更优：自定义词典
jieba.load_userdict("zulong_dictionary.txt")
tokens = list(jieba.cut(text))
```

### 3. 缓存策略

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding_cached(text: str) -> np.ndarray:
    return store._get_embedding(text)

# 热门查询缓存
cache_hits = 0
cache_misses = 0
```

### 4. 批量检索

```python
# 不推荐：逐条检索
for query in queries:
    results = store.search_by_text(query)

# 推荐：批量检索
query_vectors = [store._get_embedding(q) for q in queries]
results = store.batch_search(query_vectors, limit=5)
```

---

## 📊 测试验证

### 单元测试

```bash
# 运行测试脚本
python tests/test_enhanced_experience_store.py
```

### 测试覆盖

- ✅ Embedding 模型集成测试
- ✅ 混合检索权重测试
- ✅ 时间衰减曲线测试
- ✅ 多标签过滤逻辑测试
- ✅ Prompt 生成测试
- ✅ 统计信息测试

---

## 🔮 未来规划

### Phase 1 (已完成)
- ✅ 真实 Embedding 模型支持
- ✅ BM25 关键词检索
- ✅ 时间衰减因子
- ✅ 多标签过滤

### Phase 2 (进行中)
- ⏳ 集成 jieba 分词
- ⏳ 持久化存储（SQLite/FAISS）
- ⏳ 增量更新（支持经验更新/删除）

### Phase 3 (规划中)
- 🔮 自动标签（基于 LLM）
- 🔮 经验质量评估
- 🔮 用户反馈机制
- 🔮 可视化调试工具

---

## 📚 参考资料

### Embedding 模型
- [BAAI/bge 系列](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [sentence-transformers 文档](https://www.sbert.net/)
- [LangChain Embeddings](https://python.langchain.com/docs/integrations/text_embedding/)

### BM25 算法
- [BM25 维基百科](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Rasch 分词](https://github.com/jieba-fenci/jieba)

### 时间衰减
- [指数衰减模型](https://en.wikipedia.org/wiki/Exponential_decay)

---

**文档完成时间**: 2026-03-29  
**测试脚本**: [`tests/test_enhanced_experience_store.py`](file://d:\AI\project\zulong_beta4\tests\test_enhanced_experience_store.py)  
**实现代码**: [`zulong/memory/enhanced_experience_store.py`](file://d:\AI\project\zulong_beta4\zulong\memory\enhanced_experience_store.py)
