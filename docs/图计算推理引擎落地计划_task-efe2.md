# 祖龙图计算推理引擎 — 可行性分析与落地任务规划

## 背景

祖龙当前的MemoryGraph是"被动记忆库"——节点只存储数据，BFS扩散只做检索，记忆以文本方式拼接到prompt供LLM消费。通过系列架构研讨，确定演进方向为"图计算推理引擎"：每个记忆节点携带参数化投影矩阵（算子），BFS扩散从"检索"升级为"推理"（信号经节点算子变换后传播），LLM逐步退化为语言接口角色。

---

## 一、可行性分析

### 1.1 技术可行性：可行，分阶段渐进

| 维度 | 评估 | 依据 |
|------|------|------|
| **核心概念** | 可行 | GNN/GCN已有成熟的"节点参数+消息传递"范式，本方案在其基础上增加动态图结构+事件驱动传播 |
| **矩阵运算** | 可行 | numpy>=1.24已在requirements-core.txt中，64x64 FP32矩阵乘法<0.01ms/节点 |
| **存储扩展** | 可行 | PropertyStore(LMDB) + NodeProperties.metadata 提供无schema变更的扩展点；独立LMDB DB存储算子参数 |
| **BFS复用** | 可行 | TopologyIndex已有igraph C后端BFS，discover_related_nodes_weighted()可直接扩展为算子扩散 |
| **分片复用** | 可行 | ShardedMemoryGraph的LRU分片缓存(max_active_shards)可直接用于算子分片管理 |
| **向后兼容** | 可行 | 每个阶段新增模块独立于现有代码，通过可选接口渐进集成 |

### 1.2 性能可行性：稀疏激活是关键

| 规模 | 活跃节点 | 算子参数量 | 推理延迟(CPU) | 推理延迟(GPU) |
|------|---------|-----------|--------------|--------------|
| 核心子图(8000节点) | ~500 | 8MB | ~10ms | ~1ms |
| 中等图谱(50万节点) | ~500 | 8MB + IO | ~50ms | ~5ms |
| 大规模图谱(5000万节点) | ~500 | 8MB + IO | ~150ms | ~10ms |

关键洞察：每次推理只激活约500个节点(稀疏激活)，计算总量远小于LLM的全量参数计算。性能瓶颈在IO（加载未缓存算子）而非计算。

### 1.3 硬件可行性

| 平台 | 内存 | 存储 | 图规模上限 | 端到端延迟 | 成本 |
|------|------|------|-----------|-----------|------|
| x86 PC + GPU | 16GB+ | 2TB SSD | 5000万节点 | 30-50ms + 0.5s(语言) | ~5000元 |
| RK3588 ARM | 16GB | 1TB NVMe | 500万节点 | 144ms + 3-5s(语言) | ~1500元 |
| Orin NX ARM | 16GB | 1TB NVMe | 500万节点 | 32ms + 1s(语言) | ~5000元 |
| ESP32-S3 | 8.5MB | 16MB Flash | 3200节点 | 15ms(无反射) | ~30元 |

### 1.4 风险总评估

| 风险 | 等级 | 缓解策略 |
|------|------|----------|
| 64x64矩阵编码能力不足 | 高 | Phase 1验证编码能力；不足时升级到128x128或多层级联 |
| BFS信号发散/消失 | 中 | 引入信号归一化(LayerNorm)，每层扩散后标准化 |
| LLM蒸馏成本高 | ~~高~~→已解决 | Phase 3已改为LLM引导式白盒构建，仅需API调用(~¥30)，不需要本地GPU |
| 跨分片算子传播延迟 | 中 | 分片100MB(约5400节点)，桥接边预取，跨分片限1跳 |
| 与现有FC循环冲突 | 低 | 图推理作为可选推理路径，与LLM推理并行，通过融合层选择 |
| 高精度数值计算不可靠 | 高 | ComputeDelegateLayer内部委托SymPy/numpy/pint，图引擎仅负责推理规划 |

### 1.5 64×64维度能力边界与计算委托机制

#### 64×64算子在物理/数学推理中的有效编码能力分析

| 推理类型 | 64×64是否够用 | 原因 |
|----------|-------------|------|
| 符号推理（用什么公式、什么顺序） | ✅ 够用 | 每个节点专注单一概念，64维全部编码该概念 |
| 线性关系（F=ma, V=IR, E=mc²） | ✅ 够用 | 线性变换是64×64矩阵的原生能力 |
| 简单非线性（E=½mv², F=Gm₁m₂/r²） | ✅ 够用 | 残差初始化可编码非线性修正 |
| 守恒律判断/单位换算 | ✅ 够用 | 分类/映射任务，维度需求低 |
| 多步公式联立（3-5跳BFS） | ✅ 够用 | 多节点串联，等效320维编码空间 |
| **高精度数值计算**（247.38×891.56） | ❌ 不够 | 64维FP32无法编码10位以上数值精度 |
| **复杂符号运算**（不定积分、微分方程求解） | ❌ 不够 | 需数十步符号变换，超出单节点容量 |
| **大规模线性代数**（100×100矩阵特征值） | ❌ 不够 | 64维无法表达100维向量 |

**关键洞察**：LLM用4096-8192维做数学，但其中仅~50维有效编码数学信息（其余被语法/风格/世界知识瓜分）。图引擎64维全部用于数学，有效编码反而更高。LLM数学能力的瓶颈不在维度，而在分布式表示不擅长精确计算——图引擎同样如此，**解决方案也相同：委托专用计算工具**。

#### 三层分离架构：推理层 vs 计算层 vs 语言层

```
图引擎的推理 = "想"（用什么公式、什么顺序、什么条件）→ 64×64足够
精确计算     = "算"（具体数值运算）                     → 委托计算器
语言表达     = "说"（生成自然语言文本）                 → 语言子图/LLM

类比: 人类物理学家解物理题:
  大脑: 决定用牛顿第二定律、分析受力、列方程 → "推理"（图引擎64×64）
  计算器: 计算 9.8×5.3×sin(30°) = 25.97     → "计算"（委托工具）
  嘴巴: 向别人解释解题过程                    → "表达"（语言子图）
```

#### 计算委托触发机制

```
BFS推理过程中，节点算子检测到以下信号时触发委托:

信号1 - 数值精度需求:
  节点content中标记 has_numerical_computation=True
  例: "牛顿第二定律"节点 → 公式F=ma是符号推理(不委托)
       但"计算F=10.5×9.81"是数值计算(委托)

信号2 - 输入信号中的数值密度:
  signal中包含连续数值特征(多个维度的绝对值>NUMERIC_THRESHOLD)
  → 说明当前步骤涉及具体数值而非纯符号推理

信号3 - 节点案例匹配:
  节点的cases中包含"需要计算"的历史案例
  → 从案例中学习到"这种情况需要委托"

触发后:
  1. 节点算子输出: {公式骨架, 变量绑定, 计算表达式}
  2. 计算委托层: 解析表达式 → 调用计算器 → 返回精确结果
  3. 结果注入: 精确数值注入回BFS信号 → 继续推理
```

#### 委托架构设计

```
BFS推理流:
  ... → 节点A(公式选择) → 节点B(变量绑定) → [计算委托点] → 节点C(结果验证) → ...
                                                    ↓
                                            ComputeDelegateLayer
                                              ↓            ↓
                                          符号运算         数值运算
                                        (SymPy本地)    (Python eval/math)
                                              ↓            ↓
                                            精确结果 → 注入回BFS信号

ComputeDelegateLayer:
  - 符号运算: 本地SymPy(不定积分/微分/方程求解)
  - 数值运算: Python math/numpy(浮点运算)
  - 单位换算: 本地pint库(物理量单位自动转换)
  - 线性代数: numpy.linalg(矩阵运算)

延迟预算:
  符号运算(SymPy): ~10-100ms(取决于复杂度)
  数值运算(eval):  ~0.1ms
  单位换算(pint):  ~1ms
  → 委托延迟可控，不显著影响端到端推理延迟
```

#### 内部计算委托 vs 外部工具调用的边界

```
图引擎推理过程中触发计算委托时:

内部计算委托(ComputeDelegateLayer):
  graph_compute_engine内部持有ComputeDelegateLayer实例
  BFS到委托点 → 直接调用SymPy/numpy/pint → 结果注入 → 继续BFS
  特点: 低延迟，不走FC循环，图引擎自用
  本质: 推理基础设施的一部分，如同CPU的浮点运算单元

外部工具调用(系统层职责):
  网络搜索、代码执行、文件读写、数据库查询等
  由祖龙系统的FC循环/工具注册表提供
  系统机制负责调用并将结果汇总注入回图引擎
  图引擎不感知、不管理外部工具

严格区分:
  内部计算 = 推理的必要能力(符号/数值/单位) → 图引擎内部解决
  外部工具 = 信息获取/交互能力(搜索/代码/文件) → 系统层FC循环提供
  两者不混淆，ComputeDelegateLayer不是"工具调用"
```

#### 验证标准（计算委托）

- 触发正确性: 需要精确计算的步骤100%触发委托，纯符号推理步骤0%误触发
- 委托延迟: 数值运算<1ms，符号运算<100ms，单位换算<5ms
- 结果精度: 委托计算结果与Python精确计算一致(float64精度)
- 端到端影响: 含委托的推理总延迟 < 无委托推理延迟 + 200ms

---

## 二、七阶段实施计划

### 阶段总览与依赖关系

```
Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 3.5 ──┐
算子MVP     BFS引擎    LLM白盒构建   蒸馏框架        │
2-3周/中    3-4周/中   2-3周/低-中   3-4周/中       │
                                              ↓
              Phase 4 ──→ Phase 5 ──→ Phase 6 ──→ Phase 7
              三级跃迁    混合推理     分片优化      ARM部署
              3-4周/中    4-5周/高    3-4周/中     2-3周/中
```

Phase 3与Phase 3.5可并行开发，Phase 4与Phase 3.5也可并行。

---

### Phase 1: 算子原语验证（MVP）
**目标**：在200节点测试子图上验证"节点+投影矩阵+BFS变换"概念可行
**前置**：无 | **风险**：中 | **工作量**：2-3周

#### 任务清单

| 任务 | 描述 | 工时 |
|------|------|------|
| 1.1 OperatorNode数据结构 | dataclass: node_id, projection_matrix(64x64 FP32), bias_vector(64), activation_fn, frozen标志 | 0.5天 |
| 1.2 算子变换函数 | `apply_operator(node, signal) -> W@x + b + activation` | 1天 |
| 1.3 信号BFS扩散引擎 | GraphSignalPropagation类：种子→BFS扩散→逐节点算子变换→边权重衰减 | 2天 |
| 1.4 测试子图构建 | 用MemoryGraphHybrid创建200节点图+随机投影矩阵 | 1天 |
| 1.5 性能基准测试 | 50/100/200/500节点规模的延迟和内存测量 | 1天 |
| 1.6 输出区分度验证 | 同一输入在不同拓扑下产出可区分输出（余弦相似度<0.8） | 1天 |
| 1.7 集成冒烟 | import到祖龙进程，确认不破坏现有功能 | 0.5天 |
| 1.8 多跳信号保留率基准 | 3/5/10跳信号保留率测量(cos(signal_0, signal_N))，验证残差+LayerNorm稳定性 | 1天 |
| 1.9 计算委托触发基准 | 构建含数值计算场景的20节点测试子图，验证委托触发的精确性(需触发100%，误触发0%) | 1天 |

#### 涉及文件

新建：
- `zulong/memory/graph_compute/__init__.py`
- `zulong/memory/graph_compute/operator_node.py` — OperatorNode + apply_operator
- `zulong/memory/graph_compute/signal_propagation.py` — GraphSignalPropagation
- `zulong/memory/graph_compute/compute_delegate.py` — ComputeDelegateLayer(计算委托层)
- `tests/test_graph_compute_mvp.py`
- `tests/test_compute_delegate.py`

#### 验证标准
- 200节点/3层BFS扩散 < 50ms
- 500个OperatorNode总内存 < 100MB
- 现有`pytest tests/`全部通过
- **3跳信号保留率 >90%（cos(signal_0, signal_3)）**
- **5跳信号保留率 >75%**
- **10跳信号保留率 >50%**
- **计算委托触发正确率100%（需委托场景），误触发率0%（纯符号推理场景）**

---

### Phase 2: 算子BFS推理引擎
**目标**：将MVP引擎集成到MemoryGraphHybrid，BFS扩散升级为图计算推理
**前置**：Phase 1 | **风险**：中 | **工作量**：3-4周

#### 任务清单

| 任务 | 描述 | 工时 |
|------|------|------|
| 2.1 OperatorStore持久化 | 独立LMDB DB存储投影矩阵，msgpack序列化numpy数组 | 2天 |
| 2.2 算子懒加载+LRU缓存 | 按需加载，LRU淘汰，复用分片缓存机制 | 2天 |
| 2.3 MemoryGraphHybrid算子层 | get_operator()/set_operator()/has_operator()方法 | 1天 |
| 2.4 多探针注入引擎 | 输入语义分解为4-5个探针→热区FAISS搜索(仅top-5%节点)→入口节点→BFS扩散(自然触达冷节点) | 3天 |
| 2.5 GraphComputeEngine核心类 | 多源入口+探针标签 → 并行BFS算子扩散(结构推理) → 交叉激活放大 → 输出答案向量 | 3天 |
| 2.6 输入编码器 | 复用bge-small-zh embedding → 线性投影到64维信号向量 | 1.5天 |
| 2.7 输出解码器 | 答案向量 → top-k最近节点+分数的结构化结果 | 1天 |
| 2.8 compute_activations适配 | compute_activations_with_operators()兼容现有接口 | 1天 |
| 2.9 ShardedMemoryGraph算子适配 | 跨分片算子传播、分片级算子缓存 | 2天 |
| 2.10 批量矩阵运算优化 | 同层节点并行np.matmul | 1天 |
| 2.11 热区FAISS索引 | 仅热节点(top-5%)embedding构建IVF_PQ索引(~1.5GB)，定期更新热区组成 | 1.5天 |
| 2.12 会话相关性预过滤 | BFS前用会话上下文embedding过滤入口节点, effective_activation=base×relevance, 砍掉80%无关入口 | 2天 |
| 2.13 侧抑制竞争 | BFS每跳后执行侧抑制: 强节点抑制相邻弱节点, 弱者归零, 强者更强 | 2天 |
| 2.14 相关性优先预算剪枝 | 每跳按“链路相关性优先级+有界激活增益+交叉激活奖励”排序, 预算随任务复杂度/延迟水位自适应; 不固定限制激活链路条数, 其余ECHO(仅记录)或PRUNED(丢弃) | 1天 |
| 2.15 任务门控 | BFS结束后用探针方向作为PFC目标信号, 调节存活节点有效激活值(音量旋钮非开关) | 1天 |
| 2.16 计算委托集成 | 节点算子检测委托触发信号→ComputeDelegateLayer执行精确计算→结果注入BFS信号；支持符号运算(SymPy)/数值运算(numpy)/单位换算(pint)三类委托 | 2天 |
| 2.17 集成测试 | 覆盖探针注入+预过滤+BFS侧抑制+算力剪枝+任务门控+计算委托全流程 | 2天 |

#### 涉及文件

新建：
- `zulong/memory/graph_compute/operator_store.py` — OperatorStore(LMDB)
- `zulong/memory/graph_compute/compute_engine.py` — GraphComputeEngine
- `zulong/memory/graph_compute/probe_injector.py` — 多探针注入引擎(语义分解+FAISS全局搜索+入口节点选择)
- `zulong/memory/graph_compute/faiss_index.py` — 热区FAISS索引管理(构建/更新/查询/热区轮换)
- `zulong/memory/graph_compute/session_filter.py` — 会话相关性预过滤+BFS内动态权重调节
- `zulong/memory/graph_compute/lateral_inhibitor.py` — 侧抑制竞争+算力预算剪枝+任务门控
- `zulong/memory/graph_compute/compute_delegate.py` — ComputeDelegateLayer(计算委托层: SymPy符号+numpy数值+pint单位)
- `zulong/memory/graph_compute/signal_encoder.py` — SignalEncoder
- `zulong/memory/graph_compute/signal_decoder.py` — SignalDecoder
- `tests/test_graph_compute_engine.py`
- `tests/test_compute_delegate_integration.py` — 计算委托集成测试

修改：
- `zulong/memory/storage_hybrid/memory_graph_hybrid.py` — 新增算子层方法
- `zulong/memory/storage_hybrid/sharded_memory_graph.py` — 跨分片算子传播
- `zulong/memory/storage_hybrid/property_store.py` — 新增operator_params_db

#### 核心设计决策

**算子存储：**
- 算子参数存储在独立LMDB DB（`operator_params_db`），不塞进NodeProperties.metadata
- 16KB/节点对LMDB友好，mmap零拷贝读取延迟<1ms
- 无算子的节点在BFS扩散中被跳过，完全向后兼容

**多探针注入机制（热区FAISS稀疏加载+BFS冷数据发现）：**
```
输入文本 → 语义分解为4-5个独立探针(每个探针=一个语义维度)
  ↓
每个探针 → 热区FAISS索引搜索(仅含top-5%热节点embedding):
  热区内存: ~500万节点embedding × PQ压缩 ≈ 1.5GB
  探针向量 vs 热区节点embedding → 每探针top-3入口节点
  ↓
合并去重 → 8-10个入口节点，每个标记探针来源
  ↓
会话相关性预过滤: effective_activation = base_activation × cosine(node_emb, session_emb)
  → 砍掉80%无关入口，仅保留会话高度相关的节点
  ↓
从存活入口节点并行启动BFS算子扩散(结构推理，非向量检索)
  ↓
BFS每跳后执行五信号渐进收敛:
  ① 激活值衰减: 信号传到下一跳 × 衰减因子(0.6-0.8)
  ② 交叉激活放大: 多探针汇合节点超线性增强
  ③ 相关链路优先控制: path_relevance作为第一优先级, 相关链路可压过无关高激活新节点
  ④ 侧抑制竞争: 强节点抑制相邻弱节点，弱者归零
  ⑤ 算力预算剪枝: 按"相关性优先级+有界激活"排序保留预算内存活链路，其余ECHO/PRUNED
  → 每跳淘汰60-80%低价值候选，实际完整计算量由任务复杂度/延迟预算自适应决定
  ↓
遇到计算委托点时:
  节点算子检测数值计算信号 → ComputeDelegateLayer执行精确计算
  精确结果注入回BFS信号 → 继续推理
  ↓
BFS结束后执行任务门控(PFC-like):
  用探针原始方向作为目标信号 → 调节存活节点有效激活值
  最低保留30%信号(不完全抑制任何节点)
  ↓
收集输出信号 → 加权收敛 → 解码为答案骨架
```

**计算委托层(ComputeDelegateLayer)：**
```python
class ComputeDelegateLayer:
    """BFS推理过程中的高精度计算委托层"""
    def __init__(self):
        self.sympy_engine = SymPyEngine()    # 符号运算(积分/微分/方程)
        self.numeric_engine = NumericEngine() # 数值运算(numpy/math)
        self.unit_engine = UnitEngine()       # 单位换算(pint)
    
    def should_delegate(self, node, signal) -> DelegateDecision:
        if node.has_numerical_computation:        # 信号1: 节点标记
            return DelegateDecision.NEEDED
        if count_high_magnitude_dims(signal) > THRESHOLD:  # 信号2: 数值密度
            return DelegateDecision.NEEDED
        if any(c.requires_computation for c in node.match_cases(signal)):  # 信号3: 案例
            return DelegateDecision.NEEDED
        return DelegateDecision.NOT_NEEDED
    
    def execute(self, req) -> ComputeResult:
        if req.type == SYMBOLIC:   return self.sympy_engine.solve(req.expression)
        if req.type == NUMERIC:    return self.numeric_engine.evaluate(req.expression)
        if req.type == UNIT:       return self.unit_engine.convert(req.value, req.target_unit)
```

**四类控制信号渐进收敛架构（质量+效率兼顾）：**
```
信号一 - 会话相关性预过滤(BFS前, <0.1ms):
  作用: 在BFS开始前就砍掉无关入口
  效果: 40个候选入口 → 12个存活
  原理: effective_weight = base_weight × cosine(node_emb, session_emb)
  → 新节点base_weight再高，relevance≈0时有效权重也≈0
  → 旧能力在相关会话中始终保持高权重，不会断崖式下降

信号二 - 侧抑制竞争(BFS每跳, <0.05ms/跳):
  作用: 强节点抑制相邻弱节点，消除“所有节点都差不多”的模糊状态
  效果: 96个候选 → 30个存活 + 20个ECHO + 46个PRUNED
  原理: 皮层侧抑制(WTA)的图计算实现

信号三 - 相关激活链路控制(BFS每跳, <0.05ms/跳):
  作用: 让真正贴近当前问题的激活链路拥有最高路由优先级
  原理: route_priority = relevance_bucket + bounded_activation_gain + cross_probe_bonus
  约束: 相关性是排序优先级，不是无上限权重；增益必须clamp，避免形成新的放大失控源
  效果: 新节点base_weight再高，只要与当前问题/路径无关，就不能挤占相关旧链路的传播预算

信号四 - 算力预算剪枝(BFS每跳, <0.01ms/跳):
  作用: 硬上限保护，确保总计算量不超预算
  效果: 按相关性优先级和有界激活排序保留预算内候选，预算由compute_budget/latency_budget共同决定
  原理: 简单问题预算小，复杂问题预算大，自适应；不使用固定链路条数作为生存上限

四类控制协同:
  预过滤: 砍掉80%无关入口(零计算浪费)
  侧抑制: 每跳淘汰60-80%(只让强者继续)
  相关链路控制: 相关性优先，平衡/覆盖无关新节点高权重
  算力剪枝: 自适应预算保护(永不超载)
  → 总计算量: ~100次矩阵运算, <1ms CPU
```

**设计原理：稀疏加载+BFS发现（类脑机制）**
- 人脑每次只激活极小比例神经元，大部分皮层处于静息状态
- 热区FAISS仅存储top-5%热节点embedding（1.5GB），而非全量（30GB）
- 冷节点不在FAISS中，但通过图的边与热节点相连 → BFS扩散自然触达
- 图的边就是“关联记忆”的物理实现——即使冷数据不被搜索，也能被关联发现
- 孤立冷节点（无边连接热区）→ 定期检测+LLM补充关联边

**热区FAISS索引规格：**
| 热节点规模 | 索引类型 | 内存 | 搜索延迟 |
|------------|----------|------|----------|
| 50万(5%) | IVF512,PQ96 | ~300MB | <0.3ms |
| 500万(5%) | IVF2048,PQ96 | ~1.5GB | <0.5ms |
| 5000万(5%) | IVF8192,PQ96 | ~8GB | <1ms |

- 热区组成：按访问频率统计，定期（每小时）更新
- 冷区无FAISS索引：100%节点属性+算子存储在硬盘分片中，BFS触及时按需加载
- 索引与分片存储正交：FAISS管定位(热区embedding)，分片管存储(全量属性+算子)

**交叉激活放大规则：**
```
if 节点被N个不同探针标签的信号同时激活:
    activation = (Σ signal_i) ^ (1 + 0.3*log(N))   # N≥2时超线性放大
    # 例: 2个探针汇合 → 放大1.21倍, 3个 → 放大1.36倍
```

**激活链路控制原则（已采纳）：**
```
route_priority =
    relevance_bucket                 # 第一优先级：当前问题/路径越相关越靠前
  + bounded_activation_gain           # 第二优先级：原始激活值只做有界增益
  + cross_probe_bonus                 # 多探针汇合奖励，同样有上限

effective_activation = base_activation × (1 + clamp(λ × path_relevance, 0, max_gain))
```

- 相关链路权重最高，含义是“路由/排序优先级最高”，不是把权重无限放大。
- 越相关，优先级越高；但所有放大项都必须有上限，避免相关链路本身变成新的失控源。
- 旧能力/沉淀链路不因新节点加入而降权；无关新节点即使base_weight很高，也不能覆盖当前问题的相关旧链路。
- 激活链路控制不固定限制“链路条数”。不能把`beam_width=8`理解为最多只允许8条激活链路生存。
- `8`只可作为路径审计/可视化默认展示数量（例如展示top-8证据路径），不是BFS传播的硬性生存上限。
- 传播阶段使用自适应预算：由问题复杂度、当前延迟、节点稀疏度、GPU/CPU负载共同决定存活候选数量。

#### 验证标准
- 算子参数写入→重载：numpy.allclose(atol=1e-6)
- 2000节点/3层BFS < 200ms
- 跨分片传播可达
- 渐进收敛正确性: 会话预过滤后正确路径节点不被误删(相关度>阈值)
- 侧抑制正确性: 正确路径节点(多探针交叉激活)在侧抑制后保持高激活
- 算力预算: 5跳BFS总完整计算量≤算力预算上限, 延迟<5ms
- 新节点干扰测试: 新建100个高权重无关节点后, 旧能力推理质量不下降
- 激活链路控制测试: 相关旧链路在低原始激活下仍能压过无关高权重新节点; case_drops=0, pass_fail_regressions=0
- 自适应预算测试: 不设置固定链路条数上限; 展示/审计top-8路径时，不影响BFS内部候选生存数量
- 计算委托触发正确性: 需要精确计算的步骤100%触发委托，纯符号推理步骤0%误触发
- 计算委托延迟: 数值运算<1ms，符号运算<100ms，单位换算<5ms
- 计算委托精度: 委托计算结果与Python精确计算一致(float64精度)
- 端到端影响: 含委托的推理总延迟 < 无委托推理延迟 + 200ms

---

### Phase 3: LLM自动化教学管线（白盒构建，替代蒸馏）
**目标**：TeacherOrchestrator全自动调度LLM构建核心认知图，节点数由LLM规划+测试驱动动态确定
**前置**：Phase 2 | **风险**：低-中 | **工作量**：2-3周
**核心优势**：绕过蒸馏训练限制，不需要本地GPU算力，仅需LLM API调用，8天全自动运行

#### 核心思路

传统蒸馏是“黑盒→黑盒”（LLM权重→训练数据→拟合矩阵），本方案是“白盒→白盒”：
1. TeacherOrchestrator主控调度全流程，无需人工介入
2. 节点数不固定：LLM规划初始规模 → 测试驱动动态增长 → 达标后自动停止
3. 节点的投影矩阵从其语义内容自动推导（embedding投影），无需训练
4. 整个图结构完全可解释——每个节点代表什么概念、连接什么其他节点都清晰可查

#### 自动化教学管线（TeacherOrchestrator）

```
┌─────────────────────────────────────────────────┐
│              TeacherOrchestrator (主控)            │
│  负责：调度所有阶段、监控质量、决定何时停止          │
└────┬──────┬──────┬──────┬──────┬──────┬─────────┘
     │      │      │      │      │      │
     ▼      ▼      ▼      ▼      ▼      ▼
  Stage1  Stage2  Stage3  Stage4  Stage5  Stage6
  规划    填充    建边    审计    测试    修复
  ↓      ↓      ↓      ↓      ↓      ↓
  LLM    LLM    LLM    LLM    BFS    LLM
  生成   生成   生成   校验   推理   补漏
```

| Stage | 操作 | 自动化策略 | 耗时 |
|------|------|------------|------|
| **S1:动态规划** | LLM规划维度→子领域→概念，节点数由LLM决定 | 并发去重(embedding相似度>0.92自动合并) | 2h |
| **S2:批量填充** | 并发度10调用LLM API，每节点填充结构化知识 | JSON Schema校验+空字段重试+投影矩阵自动推导 | 30min |
| **S3:批量建边** | 每批30节点(混合维度)送LLM分析关系 | HIERARCHY边≤10%自动检查+连通性校验+Hub识别 | 1h |
| **S4:交叉审计** | 同/不同LLM审计概念覆盖+关系覆盖+推理链 | 自动定位断裂链+触发补充 | 2h |
| **S5:测试驱动** | 自动生成测试题→BFS推理→LLM判分→失败点回溯 | 循环:测试→分类差距(4种类型)→自动修复→再测试 | 4h×N轮 |
| **S6:多模型补充** | 不同LLM审计薄弱维度→增量填充 | 按模型特长分配维度(数学/逻辑/社会等) | 2h |

**动态节点数确定策略：**
```
S1阶段: LLM规划初始规模（由LLM根据领域知识决定）
  → 通常产出300-800个初始概念
  → 自动去重后可能缩减到250-600个

S5阶段: 测试驱动动态增长
  → 每轮测试定位缺失概念 → LLM自动补充新节点
  → 补充后重新测试，直到通过率≥目标阈值
  → 最终节点数由“知识完整性”决定，不是预先设定的

停止条件: 测试通过率 ≥ 80% 且 连续2轮新增节点 < 总数5%
```

#### 任务清单

| 任务 | 描述 | 工时 |
|------|------|------|
| 3.1 TeacherOrchestrator主控 | 调度全流程+状态管理+错误恢复+进度日志 | 2天 |
| 3.2 动态规划引擎 | LLM规划概念清单→embedding去重→规模自适应 | 1天 |
| 3.3 并发填充管线 | asyncio并发度10→JSON Schema校验→投影矩阵推导 | 1.5天 |
| 3.4 混合批次建边 | 每批30节点(混合维度)→LLM分析关系→拓扑自动校验 | 1.5天 |
| 3.5 自动审计模块 | LLM审计Prompt→断裂链检测→自动触发补充 | 1天 |
| 3.6 测试驱动修复循环 | 自动生题→BFS推理→LLM判分→4种差距分类→自动修复 | 2天 |
| 3.7 多模型补充接口 | 按模型特长分配维度→增量填充→算子重推导 | 1天 |
| 3.8 投影矩阵推导 | W = I + α·outer(e₆₄, e₆₄)，e₆₄=Linear(768,64)@embedding | 0.5天 |
| 3.9 构建报告生成 | 自动输出:节点数/边数/拓扑指标/测试通过率/API成本 | 0.5天 |

#### 涉及文件

新建：
- `zulong/memory/graph_compute/teacher_orchestrator.py` — TeacherOrchestrator主控
- `zulong/memory/graph_compute/llm_graph_planner.py` — 动态规划+节点填充+建边
- `zulong/memory/graph_compute/auto_auditor.py` — 自动审计+断裂链检测
- `zulong/memory/graph_compute/test_driven_fixer.py` — 测试驱动修复循环
- `zulong/memory/graph_compute/multi_teacher.py` — 多模型补充接口
- `zulong/memory/graph_compute/semantic_operator_init.py` — 投影矩阵推导
- `scripts/build_core_graph.py` — 构建执行脚本(一键启动)
- `tests/test_teacher_orchestrator.py`

修改：
- `memory_graph_hybrid.py` — 新增NodeType.COMPUTE_NODE
- `config/zulong_config.yaml` — LLM API端点、并发度、目标通过率配置

#### 核心设计决策

**节点初始化策略：**
```
S1: LLM动态规划概念清单(不预设数量)
S2: 创建空白节点
  - projection_matrix = I₆₄ (单位矩阵)
  - bias_vector = zeros(64)
S3: LLM逐个填充结构化知识
  - content, key_facts[], reasoning_patterns[], conclusion_template
S4: 投影矩阵推导
  - e₆₄ = Linear(768, 64) @ embedding
  - W = I + α * outer(e₆₄, e₆₄) (残差连接)
S5: LLM批量建边(遵循小世界网络拓扑规范)
  - 先离散后关联：概念平铺列出，不做层级分类
  - 强制跨维度建边：每批混合不同维度节点
  - Hub节点策略：识别元认知Hub各连50+节点
  - 边类型比例: SEMANTIC(30%)+ASSOCIATION(20%)+CAUSAL(15%)+REFERENCE(15%)+SIMILAR_TO(10%)+HIERARCHY(10%)
  - 禁止树状层级结构: HIERARCHY边≤总量10%
```

**图拓扑组织规范（小世界网络，禁止树状结构）：**

| 特性 | 要求 | 验证方法 |
|------|------|----------|
| 小世界 | 平均最短路径 ≤ 5跳 | BFS全对最短路径统计 |
| 高聚类 | 局部聚类系数 ≥ 0.3 | NetworkX clustering_coefficient |
| 无标度 | 度分布近似幂律(γ≈2-3) | 度分布直方图 + 幂律拟合 |
| Hub节点 | 度≥50的节点 | 度排序统计 |
| 语义社区 | Louvain自动发现 | Louvain社区检测 |
| HIERARCHY边 | ≤ 总边数10% | 边类型统计 |

**LLM API成本估算：**
- 节点填充: ~500-800次API调用 × 约2000 token/次 ≈ 1-1.6M tokens
- 连接建立: ~20-30批 × 约5000 token/次 ≈ 150K tokens
- 测试+审计+多模型: ~2000次调用 ≈ 2M tokens
- 使用DeepSeek-V3等低成本API: 总计约 ¥20-50
- 无需本地GPU，纯API调用

**增量扩展路径（测试驱动，不预设规模）：**
```
初始构建: LLM规划 → 填充 → 建边 → 审计 → 测试循环
  → 节点数由测试通过率决定(通常300-800个)
  ↓ 测试驱动动态增长
每轮测试自动发现缺失概念 → LLM补充新节点 → 重新测试
  → 停止条件: 通过率≥80% 且 连续2轮新增<5%
  ↓ + Phase 3.5蒸馏补充
蒸馏框架补充深层能力 → 可能增至1000-2000个
  ↓ + Phase 4三级跃迁
持续增长 → 临时记忆自动升级为计算节点
  → 节点数可无限增长，不受初始规模限制
```

#### 验证标准
- 所有节点填充且content非空
- 图最大连通分量 > 70%节点
- 每个节点平均连接数 > 3（图不过于稀疏）
- 平均最短路径 ≤ 5跳（小世界特性）
- HIERARCHY边占比 ≤ 10%（禁止树状层级结构）
- 基准测试通过率 > 80%（每维度不低于70%）
- TeacherOrchestrator全流程无人工介入
- LLM API总成本 < ¥100
- 不需要本地GPU

---

### Phase 3.5: 云端模型完整蒸馏 + 增量蒸馏框架
**目标**：从云端模型（27B/70B）蒸馏更深层能力；支持后续多模型增量蒸馏，补短板+增长处+保旧能力
**前置**：Phase 3 | **风险**：中 | **工作量**：3-4周

#### 完整蒸馏三步法（首次蒸馏一个云端模型）

| 步骤 | 操作 | 说明 |
|------|------|------|
| **行为采样** | 给云端模型发送10000+条测试输入（覆盖各领域），记录(input, output)对 | 建立"老师"的行为基准 |
| **差距分析** | 同样输入给图推理引擎回答，对比差异，分类为4种差距类型 | 定位图的短板 |
| **针对性补充** | 根据差距类型分别处理 | 增量修复，不重建 |

**四种差距类型及处理方式：**

| 差距类型 | 表现 | 处理方式 | 示例 |
|----------|------|----------|------|
| 节点缺失 | BFS路径无法到达关键概念 | LLM提取新概念→新增节点+建边 | 云端提到"复利"，图中无此节点 |
| 连接缺失 | 两个相关节点间无边 | LLM分析关系→新增边 | "归纳推理"和"统计推断"未连接 |
| 算子不准 | 节点存在但变换方向不对 | 收集样本对→岭回归重拟合W | CPU计算，64x64矩阵，<1ms/节点 |
| 深度不足 | BFS跳数不够到达目标 | 添加中间节点或增加max_depth | 3跳到达→添加中间节点延长至5跳 |

#### 增量蒸馏五步法（蒸馏第N+1个模型）

| 步骤 | 操作 | 说明 |
|------|------|------|
| **1.能力对标** | 分领域测试集(5维度×100题)对比图引擎与新模型 | 识别图强势/短板领域 |
| **2.短板定位** | 对短板领域深入分析失败案例 | 精确定位缺失节点/边/算子 |
| **3.定向补充** | 新模型生成短板领域的新节点+新边 | 增量添加，不修改现有结构 |
| **4.alpha混合** | 对旧节点算子做选择性更新：W=αW_new+(1-α)W_old | 强势领域α=0.05~0.1(保护)，短板领域α=0.2~0.4(更新) |
| **5.回归验证** | 全量500题重测：短板应提升，强势不应下降 | 下降则回滚对应节点的alpha参数 |

**alpha混合保护机制：**
```
if 该节点属于图强势领域:
    alpha = 0.05~0.1   # 几乎不动，保护旧能力
elif 该节点属于图短板领域:
    alpha = 0.2~0.4    # 较大幅度更新，学习新模型长处
elif 新旧输出冲突:
    先验证谁正确，再决定alpha
```

#### 任务清单

| 任务 | 描述 | 工时 |
|------|------|------|
| 3.5.1 行为采样框架 | 批量调用云端模型API，10000+条input→output采样+存储 | 2天 |
| 3.5.2 差距分析引擎 | 图推理vs云端模型输出对比，自动分类4种差距类型 | 3天 |
| 3.5.3 针对性补充管线 | 根据差距类型自动触发：新增节点/新增边/重拟合算子/添加中间节点 | 3天 |
| 3.5.4 能力对标测试集 | 5维度×100题=500题标准测试集+自动评分 | 2天 |
| 3.5.5 增量蒸馏引擎 | alpha混合更新+强势领域保护+回滚机制 | 2天 |
| 3.5.6 回归验证框架 | 全量重测+分数对比+自动回滚 | 1天 |
| 3.5.7 蒸馏审计日志 | 记录每次蒸馏的增量：新增节点/边/更新算子来源和效果 | 1天 |
| 3.5.8 自动扩轮编排 | 课程规划→教师缓存缺口检查→批量补采→严格缓存覆盖后再回放蒸馏；禁止目标未覆盖时伪完成 | 1天 |
| 3.5.9 教师缓存身份隔离 | 缓存持久复用，但按case_id+prompt指纹+教师配置指纹校验；同编号prompt漂移或换教师版本时不误命中 | 0.5天 |
| 3.5.10 守护式分批蒸馏应用 | 按case/小批次应用补丁并即时回归验证；失败批次局部回滚，已通过批次保留，避免“一处回归、全量收益丢失” | 1天 |

#### 涉及文件

新建：
- `zulong/memory/graph_compute/distillation/behavior_sampler.py` — 行为采样
- `zulong/memory/graph_compute/distillation/gap_analyzer.py` — 差距分析(4种类型)
- `zulong/memory/graph_compute/distillation/targeted_patcher.py` — 针对性补充
- `zulong/memory/graph_compute/distillation/incremental_distiller.py` — 增量蒸馏引擎(alpha混合)
- `zulong/memory/graph_compute/distillation/regression_validator.py` — 回归验证
- `zulong/memory/graph_compute/distillation/benchmark_suite.py` — 500题标准测试集
- `zulong/memory/graph_compute/distillation/teacher_cache.py` — 教师缓存持久化+prompt/provider指纹隔离
- `zulong/memory/graph_compute/distillation/teacher_provider.py` — 本地/缓存/云端教师Provider
- `zulong/memory/graph_compute/distillation/pipeline.py` — 蒸馏编排+守护式分批应用+回归触发回滚
- `scripts/distill_cloud_model.py` — 蒸馏执行脚本
- `scripts/fill_teacher_cache.py` — 教师输出缓存补采脚本
- `scripts/run_graph_compute_auto_distillation.py` — 自动扩轮编排脚本（安全规划/补缓存/严格回放）
- `tests/test_distillation.py`
- `tests/test_graph_compute_auto_distillation_script.py`

修改：
- `config/zulong_config.yaml` — 云端模型API配置、alpha参数

#### 多次增量蒸馏的长期演进

```
初始: 500核心节点 (Phase 3 LLM白盒构建, 6轮迭代)
  ↓ +Qwen3.6-27B完整蒸馏
700节点 (补齐基础短板, 算子精度提升)
  ↓ +DeepSeek-V3增量蒸馏
900节点 (补齐数学/代码短板, 增量多语言能力)
  ↓ +GPT-5增量蒸馏(未来)
1200节点 (补齐创意推理短板, 增量多模态理解)
  ↓ + Phase 4三级跃迁自动增长
持续增长 (使用中积累的经验自动固化为计算节点)
```

每次蒸馏增量越来越小（图能力逐渐逼近云端模型），成本递减。

#### 与传统微调的本质区别

| 维度 | LLM微调 | 图计算增量蒸馏 |
|------|---------|---------------|
| 旧知识 | 灾难性遗忘 | alpha混合保护不衰减 |
| 新知识 | 混入参数不可区分 | 新增节点/边可追溯来源 |
| 可解释性 | 黑盒 | 每个节点有概念标签 |
| 回滚 | 无法回滚单知识点 | 可回滚单个节点/边 |
| 算力 | 需GPU训练 | LLM API + CPU岭回归 |

#### 验证标准
- 首次蒸馏后：250题基准测试通过率从70%提升到85%+
- 增量蒸馏后：短板领域分数提升10%+，强势领域分数不降
- 增量蒸馏成本：每次 < ¥100 API费用 + < 1小时CPU计算
- 课程扩轮验证：1100→2200时必须先补齐2200条教师缓存；`--require-full-curriculum-cache`发现缺口时必须阻断蒸馏并报告缺失数量/领域分布
- 自动蒸馏编排验证：默认dry-run只规划不联网；只有显式`--fill-missing --allow-network-teacher`才采样；只有`ready_to_distill=true`才允许执行`distill_cloud_model.py`
- 教师缓存复用验证：缓存不是一次性产物，可跨轮训练复用；命中必须校验prompt指纹，provider/model/prompt版本变更时可通过教师配置指纹隔离，避免旧答案串用
- 守护式分批验证：整批蒸馏触发单case回归时不能整体丢弃全部安全收益；开启`--guarded-batches`后，坏批次局部回滚、好批次保留，最终`case_drops=0`且`pass_fail_regressions=0`

---

### Phase 4: 记忆三级跃迁与衰减机制
**目标**：实现记忆节点自动分级：临时记忆→沉淀记忆→计算节点；新记忆有衰减+赫布学习，沉淀后永久不衰减
**前置**：Phase 2 | **风险**：中 | **工作量**：3-4周

#### 核心设计：衰减与沉淀的双轨机制

**新记忆（临时级）——有衰减+赫布学习：**
```
新记忆节点创建时:
  memory_tier = EPHEMERAL
  decay_rate = 默认衰减率（艾宾浩斯曲线）
  hebbian_weight = 初始连接权重

赫布学习规则（激活时强化）:
  if 节点被BFS激活:
    hebbian_weight += α * (激活信号强度)   # 越用越强
    decay_rate *= 0.9                       # 每次激活降低衰减率

自然衰减（未被激活时）:
  if 节点长期未被BFS激活:
    importance *= decay_rate                # 重要性随时间下降
    if importance < DROP_THRESHOLD:
      标记为待遗忘（降级/归档）

赫布学习的意义：模拟人脑“经常一起被激活的神经元连接更强”
  → 频繁共现的概念对，边权重自动增强
  → 增强后的边使BFS更容易沿该路径传播
```

**沉淀记忆（永久级）——无衰减，永久保存：**
```
当临时节点通过六维指标评估后沉淀为永久记忆:
  memory_tier = REINFORCED (或 COMPUTATIONAL)
  decay_rate = 0                            # 永久不衰减
  importance = PROTECTED                    # 受保护，不会被遗忘

沉淀的意义：
  - 数字数据不似生物突触，无物理降解
  - 沉淀后的数据是经验/知识/重要内容的固化
  - 即使长期不被激活，数据仍完整保留
  - 但温度(temperature)仍影响缓存优先级（HOT/WARM/COLD仅控制内存分配）
```

#### 任务清单

| 任务 | 描述 | 工时 |
|------|------|------|
| 4.1 三级状态模型 | NodeProperties新增memory_tier字段(ephemeral/reinforced/computational)+decay_rate+hebbian_weight | 1天 |
| 4.2 赫布学习引擎 | 节点被BFS激活时自动增强边权重+降低衰减率(Hebbian LTP) | 2天 |
| 4.3 衰减计算引擎 | 临时节点未被激活时importance随时间衰减(艾宾浩斯曲线) | 1.5天 |
| 4.4 六维沉淀评估器 | 使用频率+关联密度+推理贡献度+时间稳定性+领域核心度+新颖性→加权评分 | 2天 |
| 4.5 沉淀跃迁 | 临时→沉淀：decay_rate=0+importance=PROTECTED+边权重固化 | 1天 |
| 4.6 沉淀→计算节点跃迁 | 从 content和关联结构编码初始算子参数(W=I+α·outer(e₆₄,e₆₄)) | 2天 |
| 4.7 后台跃迁引擎 | 周期性扫描赫布权重+衰减状态+六维指标→触发跃迁/遗忘 | 2天 |
| 4.8 遗忘机制 | 临时节点importance<DROP_THRESHOLD→降级归档(沉淀节点不受影响) | 1天 |
| 4.9 审计日志 | 记录每次跃迁/遗忘详情 | 1天 |

| 4.10 实时自动建边 | 新节点创建时通过embedding相似度+FAISS搜索自动建弱边(本地bge-small, <50ms) | 2天 |
| 4.11 BFS共激活建边 | BFS推理后统计共激活节点对, 达到阈值后自动建边(赫布图级应用) | 2天 |
| 4.12 路径贡献度反馈 | BFS推理后根据答案质量增强/衰减被遍历边的权重 | 2天 |
| 4.13 死边剪枝+桥接发现 | 仅两个端点均为EPHEMERAL的边可剪枝; 沉淀节点相关边永久保护; 高频跨社区节点标记为桥接节点 | 1天 |
| 4.14 LLM边渐进替换 | 冷启动LLM边(权重0.7)+图自建边(0.3)共存, 按路径贡献度逐步替换 | 1天 |
| 4.15 自验证测试引擎 | 历史QA对自动重测, 准确率下降时定位并衰减问题边 | 2天 |
| 4.16 对话输入节点创建 | 从用户对话中提取实体/事实，本地ALBERT实体识别+bge-small embedding，自动创建EPHEMERAL节点 | 2天 |
| 4.17 推理路径抽象节点 | 统计高频BFS路径(≥50次), 将反复出现的推理链浓缩为抽象节点(直接COMPUTATIONAL) | 2天 |
| 4.18 纠错性桥梁节点 | 自验证发现错误路径时, 创建桥梁节点修复缺失关联(仅当embedding相似度不够自动建边时) | 2天 |
| 4.19 跨实例知识同步 | 对拷线同步其他实例的节点/边, 导入后遵循本地沉淀机制 | 2天 |

#### 涉及文件

新建：
- `zulong/memory/graph_compute/tier_promotion.py` — 六维沉淀评估+跃迁评估
- `zulong/memory/graph_compute/hebbian_engine.py` — 赫布学习(LTP)+衰减计算
- `zulong/memory/graph_compute/promotion_engine.py` — 后台跃迁引擎
- `zulong/memory/graph_compute/edge_builder.py` — 多信号融合建边(实时+BFS共激活+路径贡献)
- `zulong/memory/graph_compute/edge_auditor.py` — 死边剪枝+桥接发现+自验证测试
- `zulong/memory/graph_compute/node_factory.py` — 节点自创建主控(对话输入+路径抽象+纠错桥梁)
- `zulong/memory/graph_compute/entity_extractor.py` — 本地实体/事实提取(ALBERT-tiny)
- `zulong/memory/graph_compute/path_abstraction.py` — 高频推理路径浓缩为抽象节点
- `tests/test_tier_promotion.py`
- `tests/test_edge_quality.py`
- `tests/test_node_creation.py`

修改：
- `memory_graph_hybrid.py` — memory_tier字段支持
- `event_bus.py` / `types.py` — 新增TIER_PROMOTED/TIER_DEMOTED事件

#### 六维沉淀评估指标

```
promotion_score = w1*frequency        # 使用频率（被BFS激活的次数）
                + w2*density          # 关联密度（连接数/同类平均连接数）
                + w3*contribution     # 推理贡献度（BFS输出中该节点信号占比）
                + w4*stability        # 时间稳定性（创建以来经过的天数）
                + w5*centrality       # 领域核心度（所在社区的PageRank）
                + w6*hebbian_weight   # 赫布权重累积（边权重增强总量）

当promotion_score > PROMOTE_THRESHOLD时:
  EPHEMERAL → REINFORCED: decay_rate=0, importance=PROTECTED
  REINFORCED → COMPUTATIONAL: 额外编码算子参数(W=I+α·outer(e₆₄,e₆₄))
```

**衰减与赫布的协作关系：**
```
新记忆诞生 → EPHEMERAL(decay_rate>0, hebbian_weight=初始值)
  ↓ 赫布学习：每次被BFS激活 → hebbian_weight↑ + decay_rate↓
  ↓ 自然衰减：长期不被激活 → importance↓
  ↓
六维评估 → 分数达标?
  是 → 沉淀为REINFORCED(decay_rate=0, 永久保存)
  否 → 继续衰减，最终归档遗忘
  ↓
沉淀后继续累积激活 → 分数再次达标?
  是 → 升级为COMPUTATIONAL(获得算子参数)
  否 → 保持REINFORCED(永久保存但不参与计算)
```

**沉淀节点保护规则（绝对不可逆）：**
```
规则: 一旦沉淀，永不可逆。所有衰减/剪枝/降级机制均不适用于沉淀节点。

1. 节点保护:
   - REINFORCED节点: 永不遗忘，永不降级回EPHEMERAL
   - COMPUTATIONAL节点: 永不遗忘，永不降级回REINFORCED，永不回收算子
   - 仅EPHEMERAL节点可被遗忘机制处理

2. 边保护:
   - 沉淀节点之间的边: 永不剪枝(即使1000次未遍历)
   - 沉淀节点与其他节点的边: 永不剪枝(保护沉淀节点的关联结构)
   - 仅两个端点均为EPHEMERAL的边可被死边剪枝处理

3. 边权重保护:
   - 路径贡献度反馈: 可增强沉淀节点相关边的权重，不可衰减
   - 自验证测试: 定位问题边时，沉淀节点相关边仅可“标记警告”，不可自动衰减

4. 设计依据:
   沉淀代表“已验证的知识/经验”，类似于人脑的“长期记忆固化”
   一旦固化，即使暂时不用，也不应被删除——它是系统积累的认知资本
   如果未来发现沉淀节点有问题，应通过“新增修正节点+新边”而非“删除旧节点”解决
```

#### 验证标准
- 赫布学习正确性：频繁激活节点边权重自动增强
- 衰减正确性：30天不活跃临时节点importance自动下降
- 沉淀正确性：六维评分达标后decay_rate=0，不再衰减
- 沉淀保护正确性：REINFORCED/COMPUTATIONAL节点及其边不受任何衰减/剪枝/降级影响
- 遗忘正确性：仅EPHEMERAL节点被遗忘，沉淀节点永久保留
- 不影响现有MemoryGraph操作
- 自建边质量：运行1000次推理后，自建边路径贡献度 > 残留LLM边
- 自验证测试：历史QA重测准确率不下降（下降时自动修复）
- 闭环验证：断开LLM后图引擎独立运行，推理质量不降

---

## 四、闭环运行与自建边机制

### 核心原则：LLM仅参与冷启动，日常运行零依赖

```
冷启动阶段(Phase 3): LLM作为教师构建初始图 → 一次性依赖
日常运行阶段:         图引擎自主增长+建边+巩固+质检 → 完全闭环
```

### 三种自建边机制（无LLM依赖）

```
机制A - 实时自动边（对话中，<50ms）:
  新节点创建 → 本地bge-small计算embedding → FAISS搜索top-5相似节点 → 建弱边
  纯本地计算，bge-small是本地embedding模型，非LLM

机制B - BFS共激活建边（推理后，<10ms）:
  BFS结束 → 统计本轮共激活节点对 → 累计达到阈值(3次)后建边
  本质是赫布学习的图级应用: 同时被激活的节点应该连接

机制C - 路径贡献度反馈（推理后，<5ms）:
  评估答案质量(置信度/用户反馈) → 正确则增强遍历边，错误则衰减遍历边
  图引擎独有的质量保障——LLM无法知道"我建的边后来有没有帮上忙"
```

### 边质量多信号融合评分

```
edge_score = w1 × embedding相似度      # 语义近邻关系
           + w2 × BFS共激活频率          # 实际使用关系
           + w3 × 时间邻近性              # 同时创建/使用
           + w4 × 路径贡献度              # 对正确答案的贡献(核心信号)

各信号互补盲区:
  embedding相似: 无法发现因果/推理关系
  共激活频率: 初期样本不足
  路径贡献度: 需要累积推理样本
  融合后: 全方位覆盖
```

### LLM边渐进替换策略

```
冷启动后:  LLM边(权重0.7) + 图自建边(权重0.3) 共存
1个月后:  LLM边中60%被路径贡献验证 → 保留增强
          30%未被使用 → 自然衰减
          10%引导错误 → 删除
3个月后:  图自建边占比 > 70%，平均质量 > 残留LLM边
6个月后:  LLM边几乎全部被替换，图引擎完全自主
```

### 自验证测试引擎（图引擎自我质检）

```
周期性测试(不依赖LLM):
  1. 从历史对话提取QA对作为测试集
  2. 图引擎推理，对比答案与历史正确答案
  3. 准确率下降 → 定位问题边(哪些边被增强后反而降低了准确率)
  4. 自动修复: 衰减问题边权重(沉淀节点相关边仅标记警告不自动衰减)

闭环核心: 图引擎自己发现并修复自己的错误边
沉淀保护: 沉淀节点相关边不可被自动修复机制衰减，仅可标记警告供人工审查
```

### 节点自创建机制（冷启动后无LLM依赖）

```
冷启动(Phase 3 LLM教师): 一次性构建知识骨架 → 几百到几千个高质量节点

运行中(4种自创建机制, 零LLM依赖):

机制A - 对话输入创建(本地ALBERT实体识别+bge-small):
  用户对话 → 提取实体/事实 → 创建EPHEMERAL节点
  例: "我家金毛叫旺财" → 节点{旺财是金毛犬, EPHEMERAL}
  纯本地计算，不需要LLM

机制B - 推理路径抽象(统计驱动):
  统计BFS高频路径(≥50次走相同节点序列)
  → 将推理链浓缩为抽象节点(直接COMPUTATIONAL)
  例: "速度"→"加速度"→"距离"→"时间" → 抽象节点"运动学四量关系"
  相当于人脑的"自动化回路"——反复做同类推理后形成快捷通路

机制C - 纠错性桥梁节点(自验证驱动):
  自验证发现某推理链总是给错误答案 → 分析缺失环节
  若embedding相似度不够自动建边 → 创建桥梁节点修复关联
  例: "紧急刹车"与"制动距离"无边 → 创建"紧急刹车需要计算制动距离"桥梁节点

机制D - 跨实例知识同步(对拷线):
  其他祖龙实例的成熟节点/边同步过来
  导入后遵循本地沉淀机制(EPHEMERAL起步, 自然筛选)

→ 自创建节点与LLM创建节点遵循完全相同的生命周期:
  EPHEMERAL → 赫布学习 → 六维评估 → REINFORCED → COMPUTATIONAL
  不达标则自然衰减遗忘
```

### 自建边为何优于LLM边

| 维度 | LLM建边 | 图引擎自建边 |
|------|---------|------------|
| 依据 | LLM先验知识(认为什么相关) | 实际使用模式(确实什么相关) |
| 个性化 | 通用(所有用户相同) | 个性化(反映用户实际思维) |
| 反馈机制 | 无(建完即固化) | 路径贡献度持续反馈 |
| 自我修复 | 无 | 自验证测试+自动衰减问题边 |
| 时效性 | 训练时快照 | 实时反映最新知识结构 |
| 依赖 | 需要LLM API | 纯本地计算 |

---

### Phase 5: 图计算直出文本 + 语言子图（终极形态）
**目标**：图计算节点直接输出结果（语义插槽拼接→模板填充→文本），LLM退化为可选润色器；远期构建语言子图实现图=LLM
**前置**：Phase 2+3+4 | **风险**：中(近期)+高(语言子图) | **工作量**：5-6周

#### 近期方案：语义插槽直出（Phase 5a）

| 任务 | 描述 | 工时 |
|------|------|------|
| 5.1 语义插槽结构 | 节点content升级为结构化SemanticSlots: subject/relationship/factors/conclusion_template | 2天 |
| 5.2 激活结果收集器 | BFS结束后收集被激活节点的SemanticSlots，按激活值排序 | 1天 |
| 5.3 模板填充引擎 | 用激活节点的fact数据填充conclusion_template中的插槽({speed}/{distance}等) | 2天 |
| 5.4 文本拼接器 | 按激活值排序拼接各节点的填充后文本，生成原始答案 | 1天 |
| 5.5 轻量LLM润色(可选) | 仅加连接词(“因此”“但是”“综合来看”)，不改语义内容 | 1天 |
| 5.6 融合门控 | 图置信度高→直出；低→fallback到LLM；中间→润色模式 | 2天 |
| 5.7 A/B评估框架 | 直出 vs 图+LLM润色 vs 纯LLM，自动对比质量 | 1天 |

#### 远期方案：语言子图（Phase 5b，终极形态：图=LLM）

| 任务 | 描述 | 工时 |
|------|------|------|
| 5.8 语言子图设计 | 全量加载不分片，含字/词/语法模式节点，共现/搭配/映射边 | 3天 |
| 5.9 LLM冷启动初始化 | 从LLM提取embedding层+attention模式+句式模板→初始化语言子图(达到源LLM80-90%水平) | 3天 |
| 5.10 维度升级(768维) | 语言子图节点使用768×768算子，匹配embedding维度 | 2天 |
| 5.11 自回归BFS循环 | BFS→解码token→token嵌入再注入→BFS→循环生成句子 | 3天 |
| 5.12 残差连接 | 加深BFS到10-20跳，引入层间残差连接 | 2天 |
| 5.13 知识→语言桥接 | 知识子图(64维)通过Linear(64,768)投影到语言子图(768维) | 1天 |
| 5.14 语言有机增长引擎 | 用户认可的表达沉淀为新语言节点+赫布强化常用句式 | 2天 |

**输出链路演进路线：**
```
当前:  图推理 → 答案骨架(节点列表) → LLM生成全文 → TTS
近期:  图推理 → SemanticSlots模板填充 → 直接文本 → TTS (LLM仅润色)
远期:  图推理 → 语言子图自回归解码 → token序列 → TTS (完全无LLM)
```

**语言子图的自回归生成机制（与LLM原理相同）：**
```
第1轮: 知识子图BFS结果 → Linear(64,768)投影 → 语言子图入口节点激活
       → BFS解码出第1个token: “在”
第2轮: token“在”的embedding注入语言子图 → BFS解码出第2个token: “120”
第3轮: token“120”embedding注入 → 解码: “km”
...
第N轮: 累积上下文注入 → 解码: “<eos>” → 生成结束

这与LLM的自回归生成在数学上等价：
  LLM:    token_n = softmax(W_vocab @ h_n)， h_n = Transformer(h_{n-1}, x_n)
  图引擎: token_n = softmax(语言子图BFS输出)， BFS接收上轮token的embedding
```

**语言子图冷启动初始化（从LLM提取）：**
```
Step 1: 提取embedding层 → 初始化词汇节点(~5万个token)
  LLM的词嵌入矩阵(V×768)直接作为语言节点的初始向量
Step 2: 提取attention模式 → 初始化语法边
  分析LLM在不同句式下的attention pattern → 转化为语言子图的边
Step 3: 行为采样 → 初始化句式模板节点
  给LLM发10000个句子，记录生成过程 → 提取常见token序列模式
  → 初始化后语言子图达到源LLM的80-90%文本生成水平
```

**语言能力有机增长曲线：**
```
初始化:  源LLM的80-90%质量(词汇+语法+句式模板)
成长期:  接近源LLM(积累领域术语+表达模式)
成熟期:  超越源LLM(领域特化+用户偏好+风格积累)
  → LLM能力在训练后固定，图引擎语言能力持续增长
  → 这是LLM永远做不到的增长曲线
```

#### 涉及文件

新建：
- `zulong/memory/graph_compute/semantic_slots.py` — SemanticSlots数据结构
- `zulong/memory/graph_compute/template_filler.py` — 模板填充引擎
- `zulong/memory/graph_compute/text_assembler.py` — 文本拼接器
- `zulong/l2/hybrid_reasoning.py` — 融合门控+fallback
- `zulong/memory/graph_compute/language_subgraph.py` — 语言子图(远期)
- `zulong/memory/graph_compute/autoregressive_decoder.py` — 自回归BFS解码(远期)
- `tests/test_text_output.py`

修改：
- `zulong/l2/inference_engine.py` — 增加图推理直出路径
- `zulong/l2/intent_prompt_builder.py` — SemanticSlots注入替代纯文本记忆注入

#### 验证标准
- 语义插槽直出质量 ≥ LLM生成质量的80%（图擅长领域）
- 直出延迟 < 50ms（无需LLM时）
- LLM润色后质量 ≥ 纯LLM质量
- 融合门控正确性：高置信度时直出，低置信度时fallback
- (远期) 语言子图自回归生成质量 ≈ 轻量LLM水平

---

### Phase 6: 分片算子优化
**目标**：100MB分片级算子批量加载、缓存管理、BFS预取
**前置**：Phase 2+5 | **风险**：中 | **工作量**：3-4周

#### 任务清单

| 任务 | 描述 | 工时 |
|------|------|------|
| 6.1 语义社区分片策略 | 替换时间分片为语义社区分片(community detection)，每片~5400节点~100MB | 3天 |
| 6.2 桥接边索引 | 每分片头部存储桥接边信息(目标分片+目标节点)，用于跨分片预取 | 2天 |
| 6.3 分片级算子批量加载 | NVMe顺序读取整个分片的算子参数(100MB/10ms) | 2天 |
| 6.4 分片缓存管理 | LRU + 加权淘汰(巩固节点多的分片保留) + 核心分片钉住 | 2天 |
| 6.5 BFS预取引擎 | 当前分片BFS进行中，异步预取桥接边指向的下一分片 | 2天 |
| 6.6 存储布局优化 | 按社区结构重排SSD上的节点存储位置 | 2天 |
| 6.7 性能验证 | 冷启动/热路径/跨分片延迟基准测试 | 1天 |

#### 涉及文件

新建：
- `zulong/memory/graph_compute/shard_optimizer.py` — 语义社区分片+布局优化
- `zulong/memory/graph_compute/prefetch_engine.py` — BFS预取引擎
- `tests/test_shard_optimization.py`

修改：
- `sharded_memory_graph.py` — 从时间分片策略升级为语义社区分片
- `operator_store.py` — 分片级批量加载
- `global_index.py` — 桥接边索引

#### 验证标准
- 冷启动延迟：单分片加载 < 15ms(NVMe SSD)
- 热路径推理：分片内BFS < 5ms
- 跨分片预取命中率 > 80%
- 10个热分片缓存内存占用 < 1.2GB

---

### Phase 7: ARM部署适配
**目标**：RK3588 16GB平台最小可行部署
**前置**：Phase 5+6 | **风险**：中 | **工作量**：2-3周

#### 任务清单

| 任务 | 描述 | 工时 |
|------|------|------|
| 7.1 内存预算配置 | 16GB分配：语言子图5.9GB(全量FP16)+知识热区80MB+知识温区2GB+工作区200MB+桥接192KB+系统2GB | 1天 |
| 7.2 算子精度适配 | FP32→FP16可选(节点级精度控制)，核心节点保持FP32 | 2天 |
| 7.3 NEON SIMD优化 | ARM NEON指令加速矩阵运算，benchmark对比 | 2天 |
| 7.4 语言子图适配 | 语言子图FP16全量加载(5.9GB)+桥接矩阵Linear(64,768) | 1天 |
| 7.5 端到端部署验证 | RK3588上完整推理流程测试 | 2天 |

#### 涉及文件

新建：
- `zulong/memory/graph_compute/arm_optimizer.py` — NEON加速+FP16适配
- `scripts/deploy_arm.sh` — ARM平台部署脚本
- `tests/test_arm_deployment.py`

修改：
- `config/zulong_config.yaml` — ARM平台内存预算配置
- `compute_engine.py` — FP16可选路径

#### 验证标准
- RK3588上图推理延迟 < 200ms
- 端到端(含语言渲染) < 5s
- 内存占用 < 14GB(语言子图5.9GB全量+知识子图按需+系统2GB，留2GB安全余量)

---

## 三、双维度混合架构（已确定）

### 架构决策：推理64×64稀疏 + 语言768×768全量

```
知识子图(64×64) — 推理层，稀疏加载:
  - 核心认知/推理/经验节点，规模可无限增长
  - 5000节点算子仅80MB，推理<0.5ms
  - 负责“思考”：BFS沿知识链推导答案
  - 热区FAISS仅索引top-5%节点，冷节点由BFS发现

语言子图(768×768) — 表达层，全量加载，不分片:
  - 字/词/语法模式节点，规模~5000节点
  - FP16量化: 5000×2.36MB/2 = ~5.9GB
  - 启动时全量加载到内存，运行期间不卸载
  - 不分片：语言子图规模有限、访问密集、无需分片管理
  - 负责“表达”：将知识子图的推理结果转化为文本/语音
  - 自回归BFS解码生成token序列
  - 可有机增长：用户认可的表达沉淀为新节点

桥接: Linear(64, 768)投影矩阵 — 常驻:
  - 知识子图BFS输出(64维) → Linear(64,768) → 语言子图入口(768维)
  - 仅192KB，常驻内存
```

### 硬件资源分配

| 平台 | 知识子图 | 语言子图(全量FP16) | 系统+其他 | 是否可行 |
|------|----------|------------|----------|----------|
| x86 PC 32GB | 80MB(热区)+按需加载 | ~5.9GB | 充足 | ✅ 轻松 |
| RK3588 16GB | 80MB(热区)+按需加载 | ~5.9GB | ~5GB | ✅ 可行 |
| ESP32-S3 | 不可行 | 不可行 | - | ❌ 不支持 |

### 加载策略说明
- 语言子图启动时全量加载(~5.9GB FP16)，运行期间不卸载、不分片
- 知识子图稀疏加载（热区FAISS+LRU分片缓存），可无限增长
- 两者通过桥接矩阵协同：推理完成 → 投影 → 语言子图生成文本
- 语言子图可有机增长：用户认可的表达沉淀为新语言节点

---

## 五、总体时间线

```
Month 1-2:   Phase 1 (MVP) + Phase 2 (算子BFS引擎)
Month 2-3:   Phase 3 (LLM白盒构建6轮迭代) + Phase 4 (三级跃迁+闭环建边+节点自创建)  [可并行]
Month 3-4:   Phase 3.5 (蒸馏框架) + Phase 5a (语义插槽直出)  [可并行]
Month 4-5:   Phase 6 (分片算子优化)
Month 5-6:   Phase 7 (ARM部署)
Month 6+:    Phase 5b (语言子图+自回归生成) [远期，可与后续迭代并行]
```

**总工作量估算**：24-29周（约6-7个月）。
Phase 3(2-3周) + Phase 3.5(3-4周) + Phase 4(5-6周，含闭环建边+节点自创建) 三者可部分并行，缩短实际日历时间。

---

## 六、关键文件地图

### 新建文件（核心）
```
zulong/memory/graph_compute/
├── __init__.py
├── operator_node.py          # Phase 1: OperatorNode数据结构
├── signal_propagation.py     # Phase 1: BFS信号扩散引擎
├── compute_delegate.py       # Phase 1: 计算委托层(SymPy+numpy+pint)
├── operator_store.py         # Phase 2: LMDB算子持久化
├── compute_engine.py         # Phase 2: GraphComputeEngine核心
├── signal_encoder.py         # Phase 2: 输入编码(text→64dim)
├── signal_decoder.py         # Phase 2: 输出解码(向量→结构化结果)
├── sparse_ops.py             # Phase 2: 稀疏矩阵运算
├── probe_injector.py         # Phase 2: 多探针注入引擎
├── faiss_index.py            # Phase 2: 热区FAISS索引管理
├── session_filter.py         # Phase 2: 会话相关性预过滤+动态权重调节
├── lateral_inhibitor.py      # Phase 2: 侧抑制+算力预算剪枝+任务门控
├── core_graph_builder.py     # Phase 3: 核心子图构建主流程
├── llm_graph_planner.py      # Phase 3: LLM知识规划+节点数据化+连接建立
├── semantic_operator_init.py # Phase 3: embedding→投影矩阵推导
├── tier_promotion.py         # Phase 4: 六维指标+跃迁评估
├── promotion_engine.py       # Phase 4: 后台跃迁引擎
├── edge_builder.py           # Phase 4: 多信号融合建边(实时+共激活+路径贡献)
├── edge_auditor.py           # Phase 4: 死边剪枝+桥接发现+自验证测试
├── node_factory.py           # Phase 4: 节点自创建主控(对话+抽象+纠错+同步)
├── entity_extractor.py       # Phase 4: 本地实体/事实提取(ALBERT-tiny)
├── path_abstraction.py       # Phase 4: 高频推理路径浓缩为抽象节点
├── shard_optimizer.py        # Phase 6: 语义社区分片
├── prefetch_engine.py        # Phase 6: BFS预取
├── arm_optimizer.py          # Phase 7: NEON加速
└── distillation/             # Phase 3.5: 蒸馏框架
    ├── __init__.py
    ├── behavior_sampler.py     # 行为采样(10000+输入输出对)
    ├── gap_analyzer.py         # 差距分析(4种类型)
    ├── targeted_patcher.py     # 针对性补充(新增节点/边/算子)
    ├── incremental_distiller.py # 增量蒸馏(alpha混合+保护+回滚)
    ├── regression_validator.py # 回归验证(500题重测)
    └── benchmark_suite.py      # 500题标准测试集
```

### 修改文件（现有）
```
zulong/memory/storage_hybrid/memory_graph_hybrid.py  # 算子层接口
zulong/memory/storage_hybrid/sharded_memory_graph.py  # 分片算子管理
zulong/memory/storage_hybrid/property_store.py        # operator_params_db
zulong/l2/inference_engine.py                         # 图推理路径
zulong/l2/intent_prompt_builder.py                    # 答案骨架注入
zulong/l2/fc_runner.py                                # 图推理结果支持
zulong/core/event_bus.py / types.py                   # 新增事件类型
config/zulong_config.yaml                             # 新配置项
```

---

## 七、验证策略

### 每阶段验证
- Phase 1: 单元测试 + 性能基准(200节点<50ms) + 计算委托触发基准(委托100%，误触发0%)
- Phase 2: 集成测试 + 跨分片验证 + 渐进收敛验证 + 新节点干扰测试 + 计算委托集成验证(精度/延迟)
- Phase 3: 图连通性 + 推理链验证 + 250题基准测试>70% + API成本<¥100
- Phase 3.5: 蒸馏后250题提升85%+ + 增量蒸馏旧能力不降 + 成本<¥100/次
- Phase 4: 模拟1000次使用的自动跃迁验证 + 自建边质量 > LLM边 + 闭环测试(断LLM不降质)
- Phase 5: A/B对比(图+LLM vs 纯LLM)
- Phase 6: 冷启动/热路径延迟基准
- Phase 7: RK3588端到端部署测试

### 端到端验证
完成全部7个阶段后，执行以下验收：
1. 同一问题集(200题)对比：纯LLM vs 图+LLM，质量评分和延迟对比
2. 增量蒸馏验证：Qwen3.6→DeepSeek-V3→GPT-5依次蒸馏，每次验证旧能力不降
3. ARM平台部署验证：RK3588上完整推理流程
4. 长期运行验证：连续运行7天，观察图谱增长和推理质量变化

---

## 八、对照可行性论证的补充项（建议纳入）

### 8.1 主要遗漏与加强点

| 补充项 | 现状 | 建议补充到 |
|------|------|------------|
| 动态路径地址 | 计划中已有BFS与路径推理，但未明确“地址是遍历属性” | Phase 2 / Phase 4 |
| 路径级Beam Search | 有多探针与FAISS入口选择，但缺少路径爆炸控制 | Phase 2 |
| 残差 + LayerNorm 稳定性 | 有残差初始化，但缺少多跳传播稳定约束 | Phase 1 / Phase 2 |
| 交叉激活融合 | 有多探针合并思路，但缺少稳定的融合/放大公式 | Phase 2 |
| 路径沉淀与快捷边 | 有赫布建边与路径贡献度，但未显式定义“高频路径固化” | Phase 4 |
| 逐层隔离测试 | 有阶段验证，但缺少“可独立验算”的隔离基准 | 全阶段 |
| 风险矩阵与阈值实验 | 已有风险点，但未全部转成可测阈值 | Phase 1 / 3 / 4 |
| 白盒等价原则 | 已有白盒构建，但未单独列出Transformer等价映射 | 总则 |

### 8.2 建议新增任务

| 任务 | 建议内容 |
|------|----------|
| 1.x | 增加 3/5/10 跳信号保留率基准，验证残差传播稳定性 |
| 2.x | 增加动态路径上下文编码器，输出 path_signature / path_context |
| 2.x | 增加路径级候选优先队列/自适应Beam，按相关性优先+有界激活做质量评分；Beam用于预算控制和审计抽样，不作为固定链路条数上限 |
| 2.x | 增加交叉激活融合器，统一处理多探针汇合与放大 |
| 4.x | 增加路径沉淀机制，将高频路径固化为快捷边/快捷通路 |
| 4.x | 增加路径级自验证，记录“哪条路径真正贡献了答案” |
| 全局 | 增加兼容性守卫，确保对现有 L2 / FC / MemoryGraph 零破坏 |

### 8.3 建议新增文件

```
zulong/memory/graph_compute/
├── path_context.py          # 动态路径上下文与路径签名
├── path_beam_search.py      # 路径级Beam Search
├── propagation_stability.py # 残差 + LayerNorm + 多跳稳定性
├── shortcut_path_engine.py  # 路径沉淀与快捷边固化
└── tests/test_path_stability.py
```

### 8.4 建议新增验收指标

- 3/5/10 跳信号保留率分别满足基线阈值
- 同一节点在不同路径上下文下输出可区分
- 活跃路径数受自适应预算/延迟水位约束，不出现路径爆炸；审计展示可默认top-8证据路径，但不限制BFS内部激活链路生存数量
- 高频路径可自动沉淀为快捷通路，且不影响其他路径可达性
- 新增机制不破坏现有 L2、FC、MemoryGraph 接口

### 8.5 建议补充的总则

1. 白盒等价原则：把 Transformer 的残差、注意力、位置编码、归一化映射为图算子。
2. 动态路径原则：节点不固定地址，路径在运行时生成语义地址。
3. 残差保护原则：原始信号直传保留，局部变换只做增量修正。
4. 稀疏激活原则：只激活相关子图，避免全图计算。
5. 渐进替代原则：LLM 只做冷启动与校验，不进入日常主链路。
6. 沉淀不可逆原则：已验证知识与稳定路径不再自动衰减。
7. 增量扩展原则：新增节点/边/算子，不重构旧结构。
8. 相关链路优先原则：当前问题越相关，激活链路路由优先级越高；所有权重增益有界。
9. 非固定链路数原则：激活链路控制不以固定条数为上限，`8`仅可作为证据路径展示/审计默认值。
10. 三层分离原则：推理(64×64图引擎) / 计算(内部委托SymPy/numpy/pint，不走FC循环) / 表达(语言子图)，图引擎不感知外部工具。
