# 祖龙 (ZULONG) Phase 7 API 参考文档

**版本**: v1.0  
**发布日期**: 2026-03-30  
**阶段**: Phase 7 - 硬件验证与性能优化

---

## 📚 目录

1. [专家技能模块](#1-专家技能模块)
   - [InternVL 视觉模型](#11-internvl-视觉模型)
   - [DWA 路径规划](#12-dwa-路径规划)
2. [工具模块](#2-工具模块)
   - [内存管理器](#21-内存管理器)
   - [结构化日志](#22-结构化日志)
   - [监控指标](#23-监控指标)
3. [使用指南](#3-使用指南)
4. [最佳实践](#4-最佳实践)
5. [常见问题](#5-常见问题)

---

## 1. 专家技能模块

### 1.1 InternVL 视觉模型

**模块路径**: `zulong.expert_skills.internvl_model_optimized`

**功能**:
- 物体检测
- 场景理解
- 视觉问答（VQA）
- 图像描述生成

**优化特性**:
- ✅ 异步推理（async/await）
- ✅ 结果缓存（LRU，100 条）
- ✅ 并发控制（semaphore）
- ✅ 4bit 量化支持
- ✅ 智能预加载

#### 1.1.1 快速开始

```python
from zulong.expert_skills import InternVLOptimized, InternVLConfig

# 创建配置
config = InternVLConfig(
    model_name="OpenGVLab/InternVL2-1B",
    use_cpu=True,  # CPU 运行
    load_in_4bit=True,  # 4bit 量化
    cache_size=100,  # 缓存大小
    enable_async=True,  # 启用异步
    preload=False  # 不预加载
)

# 获取单例实例
model = InternVLOptimized.get_instance(config)

# 同步方法（自动转为异步）
from PIL import Image
image = Image.open("test.jpg")

# 物体检测
objects = await model.detect_objects_async(image, labels=["桌子", "椅子"])

# 场景理解
scene = await model.understand_scene_async(image)

# 视觉问答
answer = await model.visual_qa_async(image, "这是什么场景？")

# 获取统计信息
stats = model.get_stats()
print(f"推理次数：{stats['total_inferences']}")
print(f"缓存命中率：{stats['cache_hit_rate']*100:.1f}%")
```

#### 1.1.2 API 参考

**类**: `InternVLOptimized`

**构造函数**:
```python
def __init__(self, config: Optional[InternVLConfig] = None)
```

**方法**:

| 方法 | 参数 | 返回值 | 描述 |
|------|------|--------|------|
| `get_instance(config)` | `config: InternVLConfig` | `InternVLOptimized` | 获取单例实例 |
| `detect_objects_async(image, labels)` | `image: Image`, `labels: List[str]` | `List[Dict]` | 异步物体检测 |
| `understand_scene_async(image)` | `image: Image` | `Dict[str, Any]` | 异步场景理解 |
| `visual_qa_async(image, question)` | `image: Image`, `question: str` | `str` | 异步视觉问答 |
| `get_stats()` | - | `Dict[str, Any]` | 获取统计信息 |
| `clear_cache()` | - | `None` | 清空缓存 |
| `preload_model()` | - | `None` | 后台预加载模型 |

**配置类**: `InternVLConfig`

```python
@dataclass
class InternVLConfig:
    model_name: str = "OpenGVLab/InternVL2-1B"
    use_cpu: bool = True
    load_in_4bit: bool = True
    cache_size: int = 100
    enable_batch: bool = True
    batch_size: int = 4
    enable_async: bool = True
    preload: bool = False
```

#### 1.1.3 统计信息

`get_stats()` 返回的字段:

```python
{
    'total_inferences': 10,           # 总推理次数
    'cache_hits': 5,                  # 缓存命中次数
    'cache_misses': 5,                # 缓存未命中次数
    'batch_inferences': 2,            # 批处理推理次数
    'async_inferences': 8,            # 异步推理次数
    'total_objects_detected': 25,     # 检测到的物体总数
    'total_scenes_understood': 3,     # 场景理解总数
    'total_vqa_queries': 2,           # VQA 查询总数
    'avg_inference_time_ms': 150.5,   # 平均推理时间
    'last_inference_time': 145.2,     # 上次推理时间
    'cache_hit_rate': 0.5,            # 缓存命中率
    'cache_size': 5,                  # 当前缓存大小
    'cache_capacity': 100,            # 缓存容量
    'is_loaded': True,                # 模型是否已加载
    'load_time_s': 2.5                # 加载时间
}
```

---

### 1.2 DWA 路径规划

**模块路径**: `zulong.expert_skills.dwa_planner_optimized`

**功能**:
- 动态窗口路径规划
- 障碍物避障
- 实时轨迹评估
- 自适应采样

**优化特性**:
- ✅ 并行轨迹评估（ThreadPoolExecutor）
- ✅ 自适应采样（减少 30-50% 样本）
- ✅ 轨迹缓存（避免重复计算）
- ✅ 早期终止（碰撞检测）
- ✅ 动态分辨率调整

#### 1.2.1 快速开始

```python
from zulong.expert_skills import DWAOptimized, DWAConfig
import numpy as np

# 创建配置
config = DWAConfig(
    v_min=0.0,
    v_max=1.0,
    w_min=-1.0,
    w_max=1.0,
    n_v_samples=10,  # 速度样本数
    n_w_samples=10,  # 角速度样本数
    enable_parallel=True,  # 启用并行
    n_workers=4,  # 并行工作线程数
    adaptive_sampling=True,  # 自适应采样
    cache_size=50,  # 轨迹缓存大小
    early_termination=True  # 早期终止
)

# 创建规划器
planner = DWAOptimized(config)

# 规划路径
current_pos = np.array([0.0, 0.0])
target_pos = np.array([10.0, 10.0])
current_v = 0.5
current_w = 0.0

obstacles = [
    {'x': 5.0, 'y': 5.0, 'angle': 0.5},
    {'x': 7.0, 'y': 3.0, 'angle': 0.3},
]

# 异步规划
v, w = await planner.plan_async(
    current_pos, target_pos, current_v, current_w, obstacles
)

print(f"最优速度：v={v:.2f}, w={w:.2f}")

# 获取统计
stats = planner.get_stats()
print(f"规划次数：{stats['total_plans']}")
print(f"平均规划时间：{stats['avg_planning_time_ms']:.2f}ms")
```

#### 1.2.2 API 参考

**类**: `DWAOptimized`

**构造函数**:
```python
def __init__(self, config: Optional[DWAConfig] = None)
```

**方法**:

| 方法 | 参数 | 返回值 | 描述 |
|------|------|--------|------|
| `plan_async(...)` | 见上 | `Tuple[float, float]` | 异步路径规划 |
| `get_stats()` | - | `Dict[str, Any]` | 获取统计信息 |
| `clear_cache()` | - | `None` | 清空轨迹缓存 |
| `shutdown()` | - | `None` | 关闭线程池 |

**配置类**: `DWAConfig`

```python
@dataclass
class DWAConfig:
    v_min: float = 0.0
    v_max: float = 1.0
    w_min: float = -1.0
    w_max: float = 1.0
    a_v_max: float = 0.5
    a_w_max: float = 1.0
    n_v_samples: int = 10
    n_w_samples: int = 10
    adaptive_sampling: bool = True
    enable_parallel: bool = True
    n_workers: int = 4
    cache_size: int = 50
    early_termination: bool = True
```

#### 1.2.3 统计信息

`get_stats()` 返回的字段:

```python
{
    'total_plans': 100,                    # 总规划次数
    'cache_hits': 20,                      # 缓存命中次数
    'cache_misses': 80,                    # 缓存未命中次数
    'total_trajectories_evaluated': 1000,  # 评估的轨迹总数
    'avg_planning_time_ms': 15.5,          # 平均规划时间
    'last_planning_time': 14.2,            # 上次规划时间
    'parallel_evaluations': 50,            # 并行评估次数
    'adaptive_reductions': 30,             # 自适应优化次数
    'early_terminations': 10,              # 早期终止次数
    'cache_size': 25,                      # 当前缓存大小
    'cache_capacity': 50                   # 缓存容量
}
```

---

## 2. 工具模块

### 2.1 内存管理器

**模块路径**: `zulong.utils.memory_manager`

**功能**:
- 统一内存监控
- 懒加载容器
- 智能预加载
- LRU 驱逐策略
- 内存压力检测

#### 2.1.1 快速开始

```python
from zulong.utils import MemoryManager, LazyLoader, LRUCacheManager

# 获取全局内存管理器
manager = MemoryManager.get_instance()

# 注册懒加载器
def create_expensive_model():
    # 加载模型的代码
    return model

manager.register_lazy_loader(
    name="vision_model",
    factory=create_expensive_model,
    preload=False,  # 不预加载
    auto_unload_timeout=300.0  # 5 分钟未使用自动卸载
)

# 注册缓存
manager.register_cache(
    name="trajectory_cache",
    max_capacity=100
)

# 获取内存统计
stats = manager.get_stats()
print(f"CPU 内存：{stats['cpu']['used_gb']:.2f} GB")
print(f"GPU 显存：{stats['gpu']['allocated_gb']:.2f} GB")
print(f"内存压力：{stats['pressure']:.2f}")

# 优化内存
manager.optimize()

# 启动监控
manager.start_monitoring(interval=5.0)  # 每 5 秒检查一次
```

#### 2.1.2 API 参考

**类**: `MemoryManager` (单例)

| 方法 | 参数 | 返回值 | 描述 |
|------|------|--------|------|
| `get_instance()` | - | `MemoryManager` | 获取单例实例 |
| `register_lazy_loader(...)` | 见上 | `None` | 注册懒加载器 |
| `register_cache(...)` | 见上 | `None` | 注册缓存 |
| `get_memory_stats()` | - | `MemoryStats` | 获取内存统计 |
| `get_memory_pressure()` | - | `float` | 获取内存压力 (0.0-1.0) |
| `optimize()` | - | `None` | 执行内存优化 |
| `start_monitoring(interval)` | `interval: float` | `None` | 启动监控 |
| `stop_monitoring()` | - | `None` | 停止监控 |
| `get_stats()` | - | `Dict[str, Any]` | 获取详细统计 |

**类**: `LazyLoader[T]`

```python
loader = LazyLoader(
    name="my_object",
    factory=lambda: create_expensive_object(),
    preload=False,
    auto_unload_timeout=300.0
)

# 首次访问自动加载
obj = loader._instance

# 手动卸载
loader.unload()

# 检查状态
is_loaded = loader.is_loaded()
```

**类**: `LRUCacheManager`

```python
cache = LRUCacheManager(max_capacity=100)

# 添加缓存（带优先级）
cache.put("key", value, priority=5)

# 获取缓存
value = cache.get("key")

# 清空缓存
cache.clear()

# 获取统计
stats = cache.get_stats()
```

---

### 2.2 结构化日志

**模块路径**: `zulong.utils.structured_logging`

**功能**:
- JSON 格式日志
- 性能追踪
- 异常捕获
- 上下文日志

#### 2.2.1 快速开始

```python
from zulong.utils import get_structured_logger
import logging

# 创建日志器
logger = get_structured_logger(
    name="my_module",
    log_file="logs/my_module.log",
    level=logging.INFO,
    enable_json=True
)

# 各级别日志
logger.debug("Debug message", test_id="1")
logger.info("Info message", user_id="user_001")
logger.warning("Warning message", category="performance")
logger.error("Error occurred", exc_info=True)

# 性能日志
logger.performance("inference", 150.5, status="success")

# 上下文日志
logger.context({
    "user_id": "user_001",
    "session_id": "session_123",
    "action": "detect_objects"
})

# 获取性能统计
stats = logger.get_performance_stats("inference")
print(f"平均耗时：{stats['avg_ms']:.2f}ms")
```

---

### 2.3 监控指标

**模块路径**: `zulong.utils.metrics`

**功能**:
- Prometheus 指标
- Counter（计数器）
- Gauge（仪表盘）
- Histogram（直方图）

#### 2.3.1 快速开始

```python
from zulong.utils import MetricsRegistry

# 获取注册表
registry = MetricsRegistry()

# 注册计数器
inference_counter = registry.register_counter(
    name="inference_total",
    description="Total number of inferences",
    labels=["model", "status"]
)

# 注册仪表盘
memory_gauge = registry.register_gauge(
    name="memory_usage_bytes",
    description="Memory usage in bytes",
    labels=["type"]
)

# 注册直方图
latency_histogram = registry.register_histogram(
    name="inference_latency_ms",
    description="Inference latency in milliseconds",
    labels=["model"],
    buckets=[50, 100, 200, 500, 1000]
)

# 使用指标
inference_counter.labels(model="internvl", status="success").inc()
memory_gauge.labels(type="gpu").set(2 * 1024**3)
latency_histogram.labels(model="internvl").observe(150.5)

# 导出指标
metrics_data = registry.collect()
```

---

## 3. 使用指南

### 3.1 完整工作流示例

```python
import asyncio
import numpy as np
from PIL import Image
from zulong.expert_skills import InternVLOptimized, DWAOptimized
from zulong.utils import MemoryManager, get_structured_logger

async def main():
    # 初始化
    logger = get_structured_logger("workflow")
    manager = MemoryManager.get_instance()
    
    # 创建模型
    vision_config = InternVLConfig(cache_size=100, enable_async=True)
    vision = InternVLOptimized.get_instance(vision_config)
    
    dwa_config = DWAConfig(enable_parallel=True, n_workers=4)
    navigator = DWAOptimized(dwa_config)
    
    # 加载图像
    image = Image.open("scene.jpg")
    
    # 视觉感知
    logger.info("开始视觉感知...")
    objects = await vision.detect_objects_async(image, labels=["障碍物"])
    scene = await vision.understand_scene_async(image)
    
    # 路径规划
    logger.info("开始路径规划...")
    current_pos = np.array([0.0, 0.0])
    target_pos = np.array([10.0, 10.0])
    
    obstacles = [
        {'x': obj['x'], 'y': obj['y'], 'angle': obj.get('angle', 0)}
        for obj in objects
    ]
    
    v, w = await navigator.plan_async(
        current_pos, target_pos, 0.5, 0.0, obstacles
    )
    
    # 输出结果
    logger.info(f"规划完成：v={v:.2f}, w={w:.2f}")
    
    # 获取统计
    vision_stats = vision.get_stats()
    navigator_stats = navigator.get_stats()
    memory_stats = manager.get_stats()
    
    print(f"视觉统计：{vision_stats}")
    print(f"导航统计：{navigator_stats}")
    print(f"内存统计：{memory_stats}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4. 最佳实践

### 4.1 性能优化

**1. 启用缓存**
```python
config = InternVLConfig(cache_size=100)  # 推荐 100
```

**2. 使用异步**
```python
# 推荐
result = await model.detect_objects_async(image)

# 不推荐（阻塞）
result = model.detect_objects(image)
```

**3. 并行评估**
```python
config = DWAConfig(enable_parallel=True, n_workers=4)
```

**4. 智能预加载**
```python
# 启动时预加载
config = InternVLConfig(preload=True)
model = InternVLOptimized.get_instance(config)
```

### 4.2 内存管理

**1. 懒加载**
```python
manager.register_lazy_loader("model", factory, preload=False)
```

**2. 自动卸载**
```python
manager.register_lazy_loader("model", factory, auto_unload_timeout=300.0)
```

**3. 监控压力**
```python
manager.start_monitoring(interval=5.0)
```

### 4.3 错误处理

```python
try:
    result = await model.detect_objects_async(image)
except Exception as e:
    logger.error(f"检测失败：{e}", exc_info=True)
    # 降级处理
    result = []
```

---

## 5. 常见问题

### Q1: 如何降低显存占用？

**A**: 使用 CPU 运行 + 4bit 量化
```python
config = InternVLConfig(
    use_cpu=True,
    load_in_4bit=True
)
```

### Q2: 如何提高推理速度？

**A**: 启用缓存 + 异步推理
```python
config = InternVLConfig(
    cache_size=100,
    enable_async=True
)
```

### Q3: 如何处理多模型并发？

**A**: 使用热切换机制
```python
class ModelContainer:
    def switch_to(self, model_type):
        # 卸载当前模型
        # 加载目标模型
```

### Q4: 内存泄漏怎么办？

**A**: 启用自动优化
```python
manager.start_monitoring(interval=5.0)
```

### Q5: 如何监控性能？

**A**: 使用统计信息 + Prometheus
```python
stats = model.get_stats()
registry = MetricsRegistry()
```

---

**文档版本**: v1.0  
**最后更新**: 2026-03-30  
**维护者**: 祖龙 (ZULONG) 系统架构组
