# Embedding 模型使用指南

## 概述

本指南介绍如何在祖龙 (ZULONG) 系统中使用 Embedding 模型进行文本向量化。

**对应 TSD**: v2.3 第 14.1 节  
**版本**: 1.0  
**日期**: 2026-03-29

---

## 快速开始

### 1. 基本使用

```python
from zulong.memory import get_embedding_manager

# 获取单例（默认配置）
manager = get_embedding_manager()

# 编码单个文本
text = "如何连接网络？"
embedding = manager.encode(text)

print(f"向量维度：{embedding.shape}")  # (768,)
print(f"向量范数：{np.linalg.norm(embedding):.4f}")  # ~1.0
```

### 2. 查询优化编码

```python
# 查询编码（针对搜索优化）
query = "网络问题"
query_emb = manager.encode_query(query)

# 文档编码（针对存储优化）
doc = "WiFi 连接经验分享"
doc_emb = manager.encode_document(doc)

# 计算相似度
similarity = np.dot(query_emb, doc_emb)
print(f"相似度：{similarity:.4f}")
```

### 3. 批量编码

```python
texts = ["文本 1", "文本 2", "文本 3"]
embeddings = manager.encode(texts)

print(f"批量向量形状：{embeddings.shape}")  # (3, 768)
```

---

## 高级配置

### 1. 自定义模型配置

```python
from zulong.memory import EmbeddingModelManager

# 自定义配置
manager = EmbeddingModelManager(
    model_name="BAAI/bge-small-zh-v1.5",  # 模型名称
    use_cpu=True,                          # CPU 模式（节省显存）
    quantize=True,                         # 4bit 量化
    cache_dir="data/models"                # 模型缓存目录
)
```

### 2. GPU 加速模式

```python
# GPU 模式（需要 CUDA 支持）
manager = get_embedding_manager(
    use_cpu=False,  # 使用 GPU
    quantize=True   # 仍建议量化以节省显存
)
```

### 3. 模型卸载

```python
# 手动释放显存
manager.unload()

# 重新加载
manager.load()
```

---

## 最佳实践

### 1. 显存优化

**推荐配置**（RTX 3060 6GB）:
```python
manager = get_embedding_manager(
    use_cpu=True,   # CPU 模式
    quantize=True   # 4bit 量化
)
```

**优势**:
- 不占用 GPU 显存
- 推理速度 ~50ms/文本
- 适合批量处理

### 2. 查询 - 文档匹配

**正确做法**:
```python
# 查询使用 encode_query
query_emb = manager.encode_query("如何连接网络")

# 文档使用 encode_document
docs = [
    "WiFi 连接经验",
    "网络故障排查",
    "路由器设置指南"
]
doc_embs = [manager.encode_document(doc) for doc in docs]

# 计算相似度
similarities = [np.dot(query_emb, doc_emb) for doc_emb in doc_embs]
```

**错误做法**:
```python
# ❌ 都使用 encode
query_emb = manager.encode("如何连接网络")
doc_emb = manager.encode("WiFi 连接经验")
# 相似度可能不准确
```

### 3. 向量归一化

```python
from sklearn.preprocessing import normalize

# 批量归一化
embeddings = manager.encode(["文本 1", "文本 2"])
normalized = normalize(embeddings)

# 验证
norms = np.linalg.norm(normalized, axis=1)
print(f"归一化范数：{norms}")  # [1.0, 1.0]
```

---

## 性能优化

### 1. 懒加载

```python
# 创建时不加载模型
manager = get_embedding_manager()

# 首次使用时自动加载
embedding = manager.encode("文本")  # 触发加载
```

### 2. 缓存策略

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding_cached(text: str):
    manager = get_embedding_manager()
    return manager.encode_query(text)

# 重复查询直接使用缓存
emb1 = get_embedding_cached("网络")
emb2 = get_embedding_cached("网络")  # 缓存命中
```

### 3. 批量处理

```python
# 推荐：批量编码
texts = [f"文本{i}" for i in range(100)]
embeddings = manager.encode(texts)  # 一次调用

# 不推荐：循环编码
embeddings = [manager.encode(text) for text in texts]  # 100 次调用
```

---

## 故障排查

### 1. 模型加载失败

**现象**:
```
WARNING: sentence-transformers 未安装
```

**原因**: 网络问题导致安装失败

**解决方案**:
```bash
# 手动安装
pip install sentence-transformers

# 或使用镜像
pip install sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**临时方案**: 使用 Mock 模式（随机向量）
```python
manager = get_embedding_manager(mock=True)
```

### 2. 显存不足

**现象**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
```python
# 切换到 CPU 模式
manager = get_embedding_manager(use_cpu=True)

# 或减小批量大小
embeddings = manager.encode(texts[:10])  # 分批处理
```

### 3. 向量维度不匹配

**现象**:
```
ValueError: shapes (768,) and (384,) not aligned
```

**原因**: 混用了不同模型的向量

**解决方案**:
```python
# 统一使用同一模型
manager = get_embedding_manager(
    model_name="BAAI/bge-small-zh-v1.5"
)

# 检查维度
print(f"维度：{embedding.shape[-1]}")  # 应为 768
```

---

## API 参考

### EmbeddingModelManager

#### 方法

- **`encode(texts: Union[str, List[str]]) -> np.ndarray`**
  - 编码文本为向量
  - 支持单个文本或批量文本
  - 返回：(dim,) 或 (n, dim)

- **`encode_query(text: str) -> np.ndarray`**
  - 编码查询文本（BGE 优化）
  - 添加查询前缀
  - 返回：(dim,)

- **`encode_document(text: str) -> np.ndarray`**
  - 编码文档（BGE 优化）
  - 添加文档前缀
  - 返回：(dim,)

- **`load() -> None`**
  - 加载模型到内存
  - 懒加载，首次使用时自动调用

- **`unload() -> None`**
  - 卸载模型，释放显存
  - 可重新调用 load() 加载

- **`get_dimension() -> int`**
  - 获取向量维度
  - 默认：768

### get_embedding_manager

```python
def get_embedding_manager(
    model_name: str = "BAAI/bge-small-zh-v1.5",
    use_cpu: bool = True,
    quantize: bool = True,
    cache_dir: str = "data/models",
    mock: bool = False
) -> EmbeddingModelManager
```

**参数**:
- `model_name`: 模型名称
- `use_cpu`: 是否使用 CPU（默认 True）
- `quantize`: 是否 4bit 量化（默认 True）
- `cache_dir`: 模型缓存目录
- `mock`: Mock 模式（随机向量，用于测试）

**返回**: EmbeddingModelManager 单例

---

## 示例代码

### 1. 经验库检索

```python
from zulong.memory import get_embedding_manager, EnhancedExperienceStore

# 初始化
manager = get_embedding_manager()
store = EnhancedExperienceStore()

# 添加经验
exp_id = store.add_experience(
    query="网络问题",
    solution="重启路由器",
    tags=["网络", "故障排查"]
)

# 编码查询
query_emb = manager.encode_query("WiFi 连不上")

# 检索
results = store.search(
    query_vector=query_emb,
    top_k=5
)
```

### 2. 混合检索

```python
from zulong.memory import (
    get_embedding_manager,
    HybridSearchConfig,
    get_balanced_config
)

# 配置
config = get_balanced_config()
manager = get_embedding_manager()

# 编码
query_emb = manager.encode_query("网络")

# 检索（向量 + BM25）
results = store.search(
    query_vector=query_emb,
    query_text="网络",  # BM25 使用
    config=config,
    top_k=10
)
```

### 3. 相似度计算

```python
from sklearn.metrics.pairwise import cosine_similarity

# 编码
texts = ["文本 1", "文本 2", "文本 3"]
embeddings = manager.encode(texts)

# 计算相似度矩阵
similarity_matrix = cosine_similarity(embeddings)

print(f"相似度矩阵:\n{similarity_matrix}")
```

---

## 参考资料

- TSD v2.3 第 14.1 节：Embedding 模型管理
- [BGE 模型官方文档](https://github.com/FlagOpen/FlagEmbedding)
- [Sentence Transformers 文档](https://sbert.net/)
