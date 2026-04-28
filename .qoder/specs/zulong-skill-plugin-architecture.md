# 祖龙(ZULONG) 技能包系统 - 技术规划

## Context

祖龙系统的核心理念是"借用 -> 学习 -> 内化 -> 丢弃"。
不是自己从零写一套任务拆解/编程能力的代码，而是：

1. 把AutoGPT的任务拆解算法**打包成技能包**，祖龙加载后获得任务拆解能力
2. 把Cline的编程逻辑**打包成技能包**，祖龙加载后获得编程能力
3. 祖龙在使用这些技能包的过程中，通过ExperienceStore+HotUpdateEngine自动积累经验
4. 当经验足够丰富（"学会了"），可以卸载技能包，祖龙靠内化的经验独立运行

**祖龙系统本身只需要建一个薄薄的"技能包运行框架"**，具体能力都从技能包中来。

### 与之前方案的根本区别
```
之前的方案（路线A - 已否决）:
  从零写 TaskPlanner/TaskExecutor 代码 → 永久焊死在祖龙里
  问题: 重复造轮子，AutoGPT花了几年做的东西要自己重写

现在的方案（路线B - 用户想法）:
  祖龙 = 技能包运行框架（薄，永久内置）
       + AutoGPT任务拆解技能包（临时借用，学完可卸载）
       + Cline编程技能包（临时借用，学完可卸载）
       + 更多技能包...
  优势: 站在巨人肩膀上，不重复造轮子，学会后可以"毕业"
```

### 现有基础设施（已可复用）
- `PluginManager` -- 已有YAML驱动的插件加载/卸载/异常隔离机制
- `ExperienceGenerator` -- 已能从对话/执行中自动提取经验
- `HotUpdateEngine` -- 已能将经验转化为SystemPatch并实时应用
- `ReplayIntegration` -- 已有复盘触发→经验提取→用户确认→保存应用流程
- `SkillPool` -- 已有LRU+优先级的技能管理和orchestrate()编排
- `ToolEngine` -- 已有并行执行(call_batch)和工具注册表
- `SharedMemoryPool` -- 已有分区存储(Raw/Feature/System/Memory Zone)、异步单例、可作为任务队列 (`zulong/infrastructure/shared_memory_pool.py`)
- `scheduler_with_shared_pool` -- 已有TaskItem状态机(PENDING→READY→EXECUTING→COMPLETED)和异步任务处理循环 (`zulong/l1b/scheduler_with_shared_pool.py`)
- `IntentRecognitionNode` -- 已有意图识别框架，但目前仅支持运动控制类意图 (`zulong/l2/intent_recognition_node.py`)

### 法律风险策略（来源：技能插件理论分析）

核心原则：**"参考架构，重写接口"** -- 借鉴思想（不受版权保护），代码用祖龙风格自己写

| 项目 | 许可证 | 风险等级 | 策略 |
|------|--------|----------|------|
| AutoGPT | MIT | 低 | 最安全，允许商用/闭源/修改。提取Planner算法后用祖龙风格重写 |
| OpenManus | Apache 2.0 | 低 | 安全，保留版权声明即可。提取Reasoning链后封装为技能包 |
| Cline | MIT | 低 | 安全，提取文件操作/代码编辑逻辑后重写 |
| OpenClaw/MCP | 部分MIT/私有 | 中 | 只参考协议思想，不直接使用私有部分代码 |

**实操要点**：不直接git clone整个项目进来，而是阅读其核心模块代码→理解算法→在祖龙中重新实现

---

## 一、总体架构

### 通俗解释
把祖龙想象成一个**刚入职的新员工**：
- **技能包** = 老师傅带的教材和工具箱。比如"项目管理教材"(AutoGPT)、"编程教材"(Cline)
- **技能包运行框架** = 新员工的学习能力。能打开教材、照着做、记录心得、最终融会贯通
- **ExperienceStore** = 新员工的笔记本。每次跟老师傅学到东西就记下来
- **HotUpdateEngine** = 把笔记变成自己的肌肉记忆。经验内化后不再需要翻教材
- **卸载技能包** = 把教材还回去。因为已经学会了，不需要教材也能干活

### 架构图

```
                     用户输入
                        |
                        v
+----------------------------------------------------------+
|  L1-B Gatekeeper (现有，不改)                              |
|  所有事件路由到 L2                                         |
+---------------------------+------------------------------+
                            |
                            v
+----------------------------------------------------------+
|  L2 InferenceEngine (增强工具调用能力)                      |
|  - 动态工具列表（技能包可注册新工具）                       |
|  - 并行工具调用（不再只处理第一个）                         |
|  - 智能迭代（不再固定3次）                                 |
|                                                          |
|  +----------------------------------------------------+  |
|  | ModuleRouter (新增，双层路由)                        |  |
|  | 第一层: 快速预判（关键词+经验规则）                   |  |
|  |   → 简单对话: L2直接处理                             |  |
|  |   → 疑似复杂任务: 进入第二层                          |  |
|  | 第二层: L2 Function Calling自主决定                   |  |
|  |   → 需要任务拆解: 调用 task_decompose 工具            |  |
|  |   → 需要深度推理: 调用 deep_reasoning 工具            |  |
|  |   → 不需要: 正常回复                                  |  |
|  +----------------------------------------------------+  |
+---------------------------+------------------------------+
                            |
              需要技能包能力时调用
                            |
                            v
+----------------------------------------------------------+
|  技能包运行框架 SkillPackRuntime (新增，薄层)               |
|                                                          |
|  +-----------+  +------------+  +-----------+            |
|  | AutoGPT   |  | Cline      |  | OpenManus |  ← 可装卸  |
|  | 任务拆解包|  | 编程技能包 |  | 深度推理包|            |
|  +-----------+  +------------+  +-----------+            |
|       |               |              |                   |
|       v               v              v                   |
|  [ISkillPack统一接口]                                     |
|  - install()   安装技能包                                 |
|  - execute()   执行技能包提供的能力                        |
|  - get_tools() 获取技能包提供的工具                        |
|  - uninstall() 卸载技能包                                 |
+---------------------------+------------------------------+
                            |
           简单任务: L2直接Function Calling执行
           复杂任务(子任务>3): 任务列表交给L1-B调度
                            |
                            v
+----------------------------------------------------------+
|  执行权分层 (新增逻辑)                                     |
|                                                          |
|  简单任务（子任务≤3个）:                                   |
|    L2 InferenceEngine 通过 Function Calling 循环完成       |
|                                                          |
|  复杂任务（子任务>3个）:                                   |
|    任务列表写入 SharedMemoryPool (Memory Zone)             |
|    L1-B scheduler_with_shared_pool 接管执行:               |
|    → TaskItem(PENDING→READY→EXECUTING→COMPLETED)          |
|    → 每完成一步回报L2，L2决定下一步                        |
|    → 前端可展示任务进度                                    |
|    → 断线重连可恢复任务                                    |
+---------------------------+------------------------------+
                            |
                    执行过程中自动记录
                            |
                            v
+----------------------------------------------------------+
|  经验学习闭环 (现有系统，增强)                              |
|                                                          |
|  ExperienceGenerator → ExperienceStore → HotUpdateEngine |
|       提取经验              存储经验         生成补丁      |
|                                                ↓         |
|                                         SystemPatch      |
|                                         参数/规则优化     |
|                                                ↓         |
|                                    [内化完成度评估]        |
|                                    置信度>90% → 可卸载    |
+----------------------------------------------------------+
```

### 三层技能包策略（来源：技能插件理论分析）

不同任务需要不同的"专家"，祖龙根据任务类型动态加载不同的技能包：

| 层级 | 技能包 | 擅长什么 | 类比 |
|------|--------|----------|------|
| 地基层（工具层） | MCP生态适配 | 标准化工具调用 | 给祖龙装"万能插座" |
| 中间层（执行层） | AutoGPT任务拆解 | 长流程、多步骤任务 | 给祖龙请"项目经理" |
| 顶层（推理层） | OpenManus深度推理 | 复杂逻辑、科研计算 | 给祖龙请"数学教授" |

**日常编程**: Cline编程技能包（跨层，既有工具也有推理）

---

## 二、Phase 1: 工具系统增强（基础设施，技能包需要这个才能注册工具）

### 为什么先做这个？
技能包需要向祖龙注册自己提供的工具。比如AutoGPT技能包要注册"任务拆解"工具，
Cline技能包要注册"代码编辑"工具。当前工具系统只有2个硬编码工具，必须先打开。

### 2.1 BaseTool增加自描述能力

**修改文件**: `zulong/tools/base.py`

```python
# BaseTool新增方法
def get_function_schema(self) -> Dict[str, Any]:
    """返回OpenAI Function Calling格式的工具描述，技能包注册工具时自动调用"""
    return {
        "type": "function",
        "function": {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters_schema()
        }
    }

@abstractmethod
def _get_parameters_schema(self) -> Dict[str, Any]:
    """子类实现：描述这个工具接受什么参数"""
    pass
```

ToolRegistry新增：
```python
def get_all_function_schemas(self) -> List[Dict]:
    """聚合所有已注册工具的描述，供InferenceEngine动态加载"""
```

### 2.2 InferenceEngine工具调用增强

**修改文件**: `zulong/l2/inference_engine.py`

**解决3次迭代限制问题**:
```
当前问题:
- 硬编码2个工具 (第244-295行)
- 最多循环3次 (第303行 max_tool_iterations=3)
- 每次只处理第一个tool_call (第375行)
- eval()安全隐患 (第386行)

优化为:
1. 动态工具列表: 从ToolRegistry获取（技能包注册的工具自动可用）
2. 并行工具调用: 模型返回多个tool_calls时全部并行执行
3. 智能迭代控制: 默认最多10次，支持以下保护：
   - 模型不再请求工具 → 正常结束
   - token接近上下文窗口80% → 强制结束
   - 连续2次相同工具+相同参数 → 死循环检测，强制结束
   - 单次工具超过30秒 → 跳过继续
4. 移除eval(): 改用json.loads()+异常处理
```

### 2.3 现有4个工具类适配

为每个工具添加`_get_parameters_schema()`：
- `zulong/tools/web_search.py`
- `zulong/tools/openclaw_tool.py`
- `zulong/tools/openclaw_search.py`
- `zulong/tools/openclaw_plugin.py`

### Phase 1 测试点
- [ ] 编写CalculatorTool测试工具，验证动态注册
- [ ] InferenceEngine多轮Function Calling对话正常
- [ ] 并行tool_calls正确执行
- [ ] eval()已移除，异常JSON有错误日志
- [ ] 迭代控制：死循环检测、超时保护

---

## 三、Phase 2: 技能包运行框架（核心，但很薄）

### 通俗解释
这一步不是写具体能力，而是建一个"教材管理系统"：
- 能装载教材（安装技能包）
- 能翻开教材照着做（执行技能包）
- 能记录学习笔记（经验记录）
- 能评估是否学会了（内化评估）
- 能归还教材（卸载技能包）

### 3.1 ISkillPack统一接口

**新增文件**: `zulong/skill_packs/interface.py`

```python
class SkillPackStatus(Enum):
    AVAILABLE = "available"        # 可安装（已下载但未加载）
    INSTALLED = "installed"        # 已安装（已加载，工具已注册）
    LEARNING = "learning"          # 学习中（正在积累经验）
    INTERNALIZED = "internalized"  # 已内化（经验充足，可卸载）
    UNINSTALLED = "uninstalled"    # 已卸载（经验保留）

class ISkillPack(ABC):
    """所有技能包必须实现的接口"""

    @abstractmethod
    def get_manifest(self) -> SkillPackManifest:
        """返回技能包清单：名称、版本、能力列表、依赖、资源需求"""
        pass

    @abstractmethod
    def install(self, tool_registry: ToolRegistry, config: Dict) -> bool:
        """安装技能包：注册工具到ToolRegistry，初始化内部状态"""
        pass

    @abstractmethod
    def execute(self, capability: str, params: Dict) -> Dict:
        """执行技能包提供的某个能力"""
        pass

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """返回此技能包提供的工具列表（将注册到ToolEngine）"""
        pass

    @abstractmethod
    def uninstall(self) -> bool:
        """卸载技能包：注销工具，释放资源"""
        pass

@dataclass
class SkillPackManifest:
    """技能包清单"""
    pack_id: str                    # "autogpt_planner"
    name: str                       # "AutoGPT任务拆解"
    version: str                    # "1.0.0"
    description: str                # 技能包描述
    capabilities: List[str]         # ["task_decompose", "priority_rank"]
    dependencies: List[str]         # 依赖的Python包
    resource_requirements: Dict     # {"cpu_mb": 512, "gpu_mb": 0}
    learning_objectives: List[str]  # ["任务拆解模式", "优先级判断"]
    source: str                     # "autogpt" / "cline" / "custom"
```

### 3.2 SkillPackRuntime（技能包运行时）

**新增文件**: `zulong/skill_packs/runtime.py`

```python
class SkillPackRuntime:
    """技能包运行时 - 管理技能包的完整生命周期"""

    def __init__(self, tool_engine, experience_store, hot_update_engine):
        self.tool_engine = tool_engine
        self.experience_store = experience_store
        self.hot_update_engine = hot_update_engine
        self._packs: Dict[str, ISkillPack] = {}
        self._status: Dict[str, SkillPackStatus] = {}
        self._experience_counts: Dict[str, int] = {}  # 每个包积累的经验数

    def install_pack(self, pack: ISkillPack, config: Dict = None) -> bool:
        """安装技能包"""
        # 1. 验证清单（依赖检查、资源检查）
        # 2. 调用 pack.install(self.tool_engine.registry, config)
        # 3. 将技能包的工具注册到 ToolEngine（InferenceEngine自动可见）
        # 4. 状态设为 INSTALLED

    def execute_capability(self, pack_id: str, capability: str, params: Dict) -> Dict:
        """执行技能包能力，同时自动记录经验"""
        # 1. 调用 pack.execute(capability, params)
        # 2. 将执行过程和结果发送给 ExperienceGenerator 提取经验
        # 3. 经验自动存入 ExperienceStore
        # 4. HotUpdateEngine 自动将经验转化为 SystemPatch
        # 5. 更新经验计数

    def check_internalization(self, pack_id: str) -> float:
        """检查内化完成度（0.0-1.0）"""
        # 基于:
        # - 该技能包的经验数量是否充足
        # - 相关SystemPatch的应用成功率
        # - 最近N次不使用技能包时的任务成功率（如果有的话）

    def uninstall_pack(self, pack_id: str) -> bool:
        """卸载技能包（经验保留在ExperienceStore中）"""
        # 1. 从 ToolEngine 注销该包的工具
        # 2. 调用 pack.uninstall()
        # 3. 状态设为 UNINSTALLED
        # 4. 经验和SystemPatch不删除，继续生效

    def list_packs(self) -> List[Dict]:
        """列出所有技能包及其状态"""
```

### 3.3 技能包目录结构

```
zulong/skill_packs/
  __init__.py
  interface.py           # ISkillPack接口 + SkillPackManifest
  runtime.py             # SkillPackRuntime运行时
  loader.py              # 从目录/YAML加载技能包
  internalization.py     # 内化完成度评估逻辑
  packs/                 # 具体的技能包（Phase 3实现）
    __init__.py
    autogpt_planner/     # AutoGPT任务拆解技能包
    cline_coder/         # Cline编程技能包
```

### 3.4 与现有系统的对接

```
技能包安装时:
  pack.get_tools() → 注册到 ToolEngine.registry
                   → InferenceEngine 的动态工具列表自动包含（Phase 1已实现）

技能包执行时:
  execute() 的结果 → ExperienceGenerator.extract_from_dialogue() (现有)
                   → ExperienceStore.add() (现有)
                   → HotUpdateEngine.on_experience_added() (现有)
                   → SystemPatch 自动生成并应用 (现有)

技能包卸载后:
  经验和补丁保留 → 祖龙靠积累的经验独立运行
```

### 3.5 bootstrap集成

**修改文件**: `zulong/bootstrap.py`

在 SystemBootstrap.initialize() 中增加：
```python
# 初始化技能包运行时
self.skill_pack_runtime = SkillPackRuntime(
    tool_engine=self.tool_engine,
    experience_store=self.experience_store,
    hot_update_engine=self.hot_update_engine
)

# 自动加载 config/skill_packs.yaml 中启用的技能包
self.skill_pack_runtime.load_from_config("config/skill_packs.yaml")
```

### 3.6 技能包配置

**新增文件**: `config/skill_packs.yaml`

```yaml
# 技能包配置
skill_packs:
  - pack_id: "autogpt_planner"
    enabled: true
    path: "zulong.skill_packs.packs.autogpt_planner"
    config:
      max_subtasks: 10
      planning_model: "default"  # 使用系统默认L2模型

  - pack_id: "cline_coder"
    enabled: false  # 暂未实现
    path: "zulong.skill_packs.packs.cline_coder"
    config:
      workspace: "/workspace"
      allowed_languages: ["python", "javascript"]

# 内化评估参数
internalization:
  min_experience_count: 50      # 至少积累50条经验才评估
  min_success_rate: 0.9         # 成功率>90%才认为学会了
  evaluation_interval_hours: 24 # 每24小时评估一次
```

### Phase 2 测试点
- [ ] 编写MockSkillPack测试包，验证install/execute/uninstall流程
- [ ] 验证技能包的工具自动注册到ToolEngine
- [ ] 验证InferenceEngine能通过Function Calling调用技能包注册的工具
- [ ] 验证execute()过程中经验自动提取到ExperienceStore
- [ ] 验证卸载后经验和SystemPatch保留

---

## 四、Phase 3: 具体技能包实现

### 通俗解释
Phase 2建好了"教材管理系统"，Phase 3就是把具体的"教材"做出来。
每本教材就是对一个开源项目的"外科手术式"提取和包装。

### 4.1 AutoGPT任务拆解技能包

**新增目录**: `zulong/skill_packs/packs/autogpt_planner/`

**不是整体引入AutoGPT**，而是提取其核心的任务分解算法：
- 从AutoGPT源码中提取 Planner/TaskDecomposition 模块的**核心算法逻辑**
- 包装成 ISkillPack 接口
- 注册一个 `task_decompose` 工具到 ToolEngine

```python
class AutoGPTPlanner(ISkillPack):
    """AutoGPT任务拆解技能包"""

    def get_manifest(self):
        return SkillPackManifest(
            pack_id="autogpt_planner",
            name="AutoGPT任务拆解",
            capabilities=["task_decompose", "priority_rank", "dependency_analyze"],
            learning_objectives=["将复杂请求拆解为子任务", "判断子任务优先级和依赖关系"],
            source="autogpt"
        )

    def install(self, tool_registry, config):
        # 注册任务拆解工具
        tool_registry.register(TaskDecomposeTool(self))
        # 注册优先级排序工具
        tool_registry.register(PriorityRankTool(self))

    def execute(self, capability, params):
        if capability == "task_decompose":
            return self._decompose_task(params["user_request"])
        elif capability == "priority_rank":
            return self._rank_priorities(params["subtasks"])

    def _decompose_task(self, user_request: str) -> Dict:
        """核心算法：从AutoGPT提取的任务拆解逻辑"""
        # 1. 调用L2推理引擎，使用特定的任务拆解prompt
        # 2. 解析返回的结构化子任务列表
        # 3. 分析子任务之间的依赖关系
        # 4. 标记可并行的任务组
        return {"subtasks": [...], "dependencies": {...}, "parallel_groups": [...]}

    def get_tools(self):
        return [TaskDecomposeTool(self), PriorityRankTool(self)]

    def uninstall(self):
        # 清理资源，但经验保留
        pass
```

**提供的工具（自动注册到ToolEngine，InferenceEngine可通过Function Calling调用）**:
- `task_decompose`: 将复杂请求拆解为子任务列表
- `priority_rank`: 对子任务排序
- `dependency_analyze`: 分析子任务依赖关系

**数据流（用户说"帮我搜索机器人新闻并写总结"时）**:
```
1. 用户输入 → L2 InferenceEngine
2. L2判断需要拆解 → Function Calling调用 task_decompose 工具
3. AutoGPT技能包执行拆解 → 返回子任务列表
4. L2按子任务依次执行（调用搜索工具、推理汇总等）
5. ExperienceGenerator自动记录"这次拆解效果好/不好"
6. 多次积累后 → HotUpdateEngine内化拆解模式
7. 内化完成 → 卸载AutoGPT技能包 → L2靠经验自己拆解
```

### 4.2 OpenManus深度推理技能包（来源：技能插件理论分析）

**新增目录**: `zulong/skill_packs/packs/openmanus_reasoner/`

**为什么需要这个？**
AutoGPT擅长"流程化拆解"（步骤1→步骤2→步骤3），但遇到需要深度逻辑推理的任务
（如"设计飞行器气动布局"、"证明数学定理"），它的拆解质量不够。
OpenManus的推理链（Reasoning Chain）能在这类场景下提供更高质量的思考。

**通俗比喻**: AutoGPT是"项目经理"（擅长拆活分活），OpenManus是"数学教授"（擅长深度思考）

```python
class OpenManusReasoner(ISkillPack):
    """OpenManus深度推理技能包"""

    def get_manifest(self):
        return SkillPackManifest(
            pack_id="openmanus_reasoner",
            name="OpenManus深度推理",
            capabilities=["deep_reasoning", "logic_chain", "problem_decompose"],
            learning_objectives=["复杂逻辑推理模式", "多步推理链构建", "假设验证策略"],
            source="openmanus"
        )

    def install(self, tool_registry, config):
        tool_registry.register(DeepReasoningTool(self))

    def execute(self, capability, params):
        if capability == "deep_reasoning":
            return self._deep_reason(params["problem"], params.get("context", ""))

    def _deep_reason(self, problem: str, context: str) -> Dict:
        """核心算法：从OpenManus提取的推理链逻辑"""
        # 1. 问题分析（Reasoning）：先思考问题本质
        # 2. 假设生成：提出多个可能的解决路径
        # 3. 假设验证：逐一检验每条路径的可行性
        # 4. 方案选择：选出最优路径并输出详细步骤
        return {"reasoning_chain": [...], "conclusion": "...", "confidence": 0.85}

    def get_tools(self):
        return [DeepReasoningTool(self)]

    def uninstall(self):
        pass
```

**提供的工具**:
- `deep_reasoning`: 对复杂问题进行深度推理分析

**与AutoGPT的分工**:
- ModuleRouter第一层判断→疑似复杂任务→L2 Function Calling决定：
  - 流程化任务（"搜索+分析+总结"） → 调用 `task_decompose`（AutoGPT）
  - 深度逻辑任务（"设计算法"、"数学证明"） → 调用 `deep_reasoning`（OpenManus）
  - 两者都需要（"设计飞行器并写代码"） → 先 `deep_reasoning` 再 `task_decompose`

### 4.3 Cline编程技能包（后续实现）

**新增目录**: `zulong/skill_packs/packs/cline_coder/`

从Cline(开源编程助手)中提取核心能力：
- 文件读写操作
- 代码生成与修改
- 终端命令执行
- 代码搜索与分析

```python
class ClineCoder(ISkillPack):
    """Cline编程技能包"""

    def get_manifest(self):
        return SkillPackManifest(
            pack_id="cline_coder",
            capabilities=["read_file", "write_file", "edit_code",
                          "run_command", "search_code"],
            learning_objectives=["代码生成模式", "错误修复策略", "文件操作规范"],
            source="cline"
        )

    def get_tools(self):
        return [
            FileReadTool(self),      # 读文件
            FileWriteTool(self),     # 写文件
            CodeEditTool(self),      # 编辑代码
            TerminalTool(self),      # 执行命令
            CodeSearchTool(self),    # 搜索代码
        ]
```

### 4.4 闲聊 vs 任务的自动判断

**这个判断逻辑本身也可以从技能包中学到**。

初始阶段：使用两级快速判断（内置，不需要技能包）：
- 第一级: 关键词预过滤(0延迟) -- 短句/问候→闲聊，动作指令词→疑似任务
- 第二级: L2轻量确认(max_tokens=10) -- 仅对疑似任务调用

随着AutoGPT技能包使用积累，ExperienceStore会记录"哪些请求被拆解了、效果如何"，
HotUpdateEngine会生成更精准的判断规则（SystemPatch），逐步替代简单的关键词匹配。

**修改文件**: `zulong/l2/inference_engine.py`

在现有的 `_generate_with_vllm_and_tools` 流程中，当AutoGPT技能包已安装时：
- InferenceEngine的tools列表自动包含 `task_decompose` 工具
- L2模型自主决定是否调用（不需要额外的Gatekeeper路由逻辑）
- 如果模型认为需要拆解，它会自己调用 task_decompose → 拿到子任务 → 逐步执行

**这意味着不需要修改Gatekeeper**。闲聊vs任务的判断由L2模型通过Function Calling自主完成。

### Phase 3 测试点
- [ ] AutoGPT技能包安装后，`task_decompose`工具出现在Function Calling可用列表
- [ ] OpenManus技能包安装后，`deep_reasoning`工具出现在Function Calling可用列表
- [ ] 用户说"帮我搜索并分析XXX"时，L2自动调用task_decompose
- [ ] 用户说"设计一个算法解决XXX"时，L2自动调用deep_reasoning
- [ ] 拆解结果合理（子任务、依赖关系、并行组）
- [ ] 子任务>3个时，任务列表正确写入SharedMemoryPool
- [ ] L1-B正确接管复杂任务执行，TaskItem状态正常流转(PENDING→EXECUTING→COMPLETED)
- [ ] 执行过程中经验自动提取
- [ ] 简单对话"你好"时，ModuleRouter第一层直接过滤，L2不调用任何技能包工具
- [ ] ModuleRouter路由判断不增加明显延迟（第一层<5ms）

---

## 五、Phase 4: MCP生态接入 + 端到端验证

### 5.1 MCP作为工具扩展机制

MCP(Model Context Protocol)是Anthropic提出的工具标准协议。
和技能包的关系：MCP提供的是**工具**，技能包提供的是**能力+工具**。

**新增文件**: `zulong/tools/mcp_client_adapter.py`

- MCPToolBridge继承BaseTool
- 连接外部MCP Server，将MCP工具映射为BaseTool注册到ToolRegistry
- InferenceEngine自动可见（Phase 1已实现动态工具列表）

**新增文件**: `config/mcp_servers.yaml`
```yaml
servers:
  filesystem:
    command: "npx"
    args: ["-y", "@anthropic/mcp-server-filesystem", "/workspace"]
    description: "文件系统操作"
```

### 5.2 MCP Server暴露（让外部调用祖龙）

**新增文件**: `zulong/tools/mcp_server.py`

将ToolRegistry中的工具通过MCP协议暴露，让Claude/Cursor可以调用祖龙的机器人工具。

### 5.3 生态选择
- **利用Anthropic MCP现有生态**，不自建
- 社区已有数千个MCP Server可直接使用
- 祖龙作为MCP Client接入 = 瞬间获得工具生态

### Phase 4 测试点
- [ ] 通过MCP接入filesystem Server
- [ ] InferenceEngine能通过Function Calling调用MCP工具
- [ ] 端到端: 复杂请求→技能包拆解→工具调用→经验记录→结果汇总

---

## 六、完整数据流

### 场景A: 简单对话（"你好"/"今天天气怎么样"）

```
用户: "你好"
   |
   v
L2 InferenceEngine
   | ModuleRouter第一层: 短句+问候 → 闲聊
   | → 不进入Function Calling工具调用
   | → L2直接生成回复
   v
回复: "你好！有什么我能帮你的吗？"
```

### 场景B: 简单任务（子任务≤3个，L2直接执行）

```
用户: "帮我搜索最新的机器人新闻"
   |
   v
L2 InferenceEngine (工具列表中包含 task_decompose/web_search/...)
   |
   | ModuleRouter第一层: 包含"搜索"指令词 → 疑似任务
   | ModuleRouter第二层: L2 Function Calling自主判断
   | L2判断: 这个任务简单，直接调用web_search即可，不需要拆解
   | → Function Calling: web_search("最新机器人新闻")
   |
   v
返回搜索结果 → L2生成总结回复给用户
   |
   v (后台)
ExperienceGenerator 记录执行模式
```

### 场景C: 复杂任务（子任务>3个，L1-B接管执行）

```
用户: "帮我搜索最新的机器人新闻，分析趋势，写一份报告，发送到我的邮箱"
   |
   v
L2 InferenceEngine
   |
   | ModuleRouter第一层: 多动作指令 → 疑似复杂任务
   | ModuleRouter第二层: L2 Function Calling
   | L2判断: 这个请求需要拆解
   | → Function Calling: task_decompose(user_request="...")
   |
   v
AutoGPT技能包.execute("task_decompose", ...)
   |
   | 返回: {subtasks: ["搜索新闻", "分析趋势", "写报告", "发送邮件"],
   |        subtask_count: 4,   ← 超过3个
   |        dependencies: {"分析趋势": ["搜索新闻"], ...}}
   |
   v
L2发现子任务>3个 → 任务列表写入 SharedMemoryPool (Memory Zone)
   |
   v
L1-B scheduler_with_shared_pool 接管:
   | TaskItem(id=1, prompt="搜索新闻", status=PENDING)
   | TaskItem(id=2, prompt="分析趋势", status=PENDING, depends=[1])
   | TaskItem(id=3, prompt="写报告", status=PENDING, depends=[2])
   | TaskItem(id=4, prompt="发送邮件", status=PENDING, depends=[3])
   |
   | 按依赖顺序逐步执行:
   | TaskItem 1: PENDING → EXECUTING → COMPLETED
   |   └→ 调用 web_search → 结果存入SharedMemoryPool
   | TaskItem 2: PENDING → READY → EXECUTING → COMPLETED
   |   └→ L2分析搜索结果 → 趋势分析存入SharedMemoryPool
   | ...依此类推
   |
   v
全部完成 → 汇总结果回复用户
   |
   v (后台自动)
ExperienceGenerator 记录:
  - "task_decompose 拆解效果: 好/差"
  - "搜索+分析+总结+发送 的执行模式"
  - "用户满意度反馈"
   |
   v
HotUpdateEngine 生成 SystemPatch:
  - "遇到'搜索+分析+总结'类请求时的最优拆解模式"
   |
   v
下次遇到类似请求，L2靠经验(SystemPatch)自己就能拆解
当内化完成度>90%时，AutoGPT技能包可以卸载
```

### 场景D: 高难度任务（需要深度推理）

```
用户: "设计一个飞行器气动布局分析工具"
   |
   v
L2 InferenceEngine
   | ModuleRouter第一层: "设计"+"分析" → 疑似复杂任务
   | ModuleRouter第二层: L2 Function Calling
   | L2判断: 这个问题需要深度思考
   | → Function Calling: deep_reasoning(problem="飞行器气动布局分析")
   |
   v
OpenManus技能包.execute("deep_reasoning", ...)
   | 返回推理链 + 高层方案
   |
   v
L2拿到方案后 → 调用 task_decompose 拆解为具体步骤
   | → L1-B接管执行（子任务>3个时）
   |
   v
（后续流程同场景C）
```

---

## 七、关键设计决策

| 决策项 | 选择 | 通俗解释 |
|--------|------|----------|
| 架构理念 | 技能包借用，不从零写 | 借教材学习，不自己编教材 |
| 路由判断 | ModuleRouter双层：快速预判+L2 Function Calling | 先快速筛选，再让AI精确判断 |
| 执行权分层 | 简单任务L2直接执行，复杂任务(>3步)L1-B接管 | 小事自己做，大事交给调度中心 |
| 任务队列 | 复杂任务子任务写入SharedMemoryPool | 利用现有共享池，支持进度追踪和断线恢复 |
| 工具迭代 | 从固定3次→智能动态(最多10次) | "干完了才停"，不再"数到3就停" |
| AutoGPT集成 | 外科手术提取Planner算法→包装成技能包 | 只拿需要的模块，不整体引入 |
| OpenManus集成 | 提取Reasoning链→包装成深度推理技能包 | 给祖龙请"数学教授"处理高难度问题 |
| Cline集成 | 提取文件操作/代码编辑逻辑→包装成技能包 | 给祖龙请"编程教练" |
| MCP | 利用Anthropic现有生态 | 不自建，直接"插USB"用别人的工具 |
| 内化判断 | 经验数>50 + 成功率>90% | 学够了、做对了，才能"毕业" |
| 卸载策略 | 卸载技能包但保留经验 | 还了教材但笔记留着 |
| 法律策略 | "参考架构，重写接口" | 借鉴思想（不受版权保护），代码用祖龙风格自己写 |

---

## 八、需要修改/新增的文件清单

### 修改现有文件
| 文件 | 改动内容 |
|------|----------|
| `zulong/tools/base.py` | BaseTool增加`get_function_schema()`；ToolRegistry增加`get_all_function_schemas()` |
| `zulong/l2/inference_engine.py` | 动态工具列表、并行tool_calls、移除eval()、智能迭代控制、集成ModuleRouter |
| `zulong/tools/web_search.py` | 添加`_get_parameters_schema()` |
| `zulong/tools/openclaw_tool.py` | 添加`_get_parameters_schema()` |
| `zulong/tools/openclaw_search.py` | 添加`_get_parameters_schema()` |
| `zulong/tools/openclaw_plugin.py` | 添加`_get_parameters_schema()` |
| `zulong/l1b/scheduler_with_shared_pool.py` | 增加复杂任务接管逻辑：接收L2下发的子任务列表，按依赖顺序执行（新增） |
| `zulong/infrastructure/shared_memory_pool.py` | Memory Zone增加任务队列读写方法（新增） |
| `zulong/bootstrap.py` | 初始化SkillPackRuntime，加载技能包配置 |
| `requirements.txt` | 增加`langgraph`（已使用未列入）和`mcp`依赖 |

### 新增文件
| 文件 | 用途 |
|------|------|
| `zulong/skill_packs/__init__.py` | 技能包模块 |
| `zulong/skill_packs/interface.py` | ISkillPack接口 + SkillPackManifest |
| `zulong/skill_packs/runtime.py` | SkillPackRuntime运行时 |
| `zulong/skill_packs/loader.py` | 从目录/YAML加载技能包 |
| `zulong/skill_packs/internalization.py` | 内化完成度评估 |
| `zulong/skill_packs/module_router.py` | ModuleRouter双层路由（新增） |
| `zulong/skill_packs/packs/__init__.py` | 具体技能包目录 |
| `zulong/skill_packs/packs/autogpt_planner/__init__.py` | AutoGPT任务拆解技能包 |
| `zulong/skill_packs/packs/autogpt_planner/planner.py` | 核心拆解算法 |
| `zulong/skill_packs/packs/autogpt_planner/tools.py` | 注册的工具类 |
| `zulong/skill_packs/packs/openmanus_reasoner/__init__.py` | OpenManus深度推理技能包（新增） |
| `zulong/skill_packs/packs/openmanus_reasoner/reasoner.py` | 核心推理链算法（新增） |
| `zulong/skill_packs/packs/openmanus_reasoner/tools.py` | 注册的工具类（新增） |
| `zulong/skill_packs/packs/cline_coder/__init__.py` | Cline编程技能包(后续) |
| `zulong/tools/mcp_client_adapter.py` | MCP Client适配器 |
| `zulong/tools/mcp_server.py` | MCP Server暴露器 |
| `config/skill_packs.yaml` | 技能包配置 |
| `config/mcp_servers.yaml` | MCP服务器配置 |

---

## 九、实施顺序（分步实现，每步测试后再继续）

**Phase 1: 工具系统增强** (基础设施)
1. BaseTool增加get_function_schema()
2. ToolRegistry增加get_all_function_schemas()
3. 4个现有工具类添加_get_parameters_schema()
4. InferenceEngine: 动态工具列表+并行调用+移除eval()+智能迭代
→ 测试通过后继续

**Phase 2: 技能包运行框架** (核心框架)
1. 定义ISkillPack接口和SkillPackManifest
2. 实现SkillPackRuntime（安装/执行/卸载）
3. 实现技能包加载器(从YAML配置加载)
4. 实现内化完成度评估
5. 实现ModuleRouter双层路由
6. 对接ExperienceGenerator+HotUpdateEngine
7. bootstrap集成
→ 用MockSkillPack测试通过后继续

**Phase 3: 具体技能包 + 执行权分层**
1. AutoGPT任务拆解技能包（提取Planner算法+包装）
2. OpenManus深度推理技能包（提取Reasoning链+包装）
3. 复杂任务L1-B执行接管（子任务>3个时，任务列表写入SharedMemoryPool，L1-B逐步执行）
4. 闲聊vs任务判断（ModuleRouter预判+L2 Function Calling自主决定）
5. Cline编程技能包（提取文件操作/代码编辑+包装）
→ 端到端测试: 复杂请求→拆解→执行→经验记录

**Phase 4: MCP生态接入**
1. MCP Client适配器
2. MCP Server暴露器
3. 完整端到端验证
→ 全流程测试通过

---

## 十、验证方案

### 总体验证策略
1. **单元级**: 每个Phase编写独立测试
2. **集成级**: Phase 2完成后用MockSkillPack测试完整生命周期
3. **系统级**: Phase 3完成后测试"安装AutoGPT包→执行任务拆解→积累经验→评估内化→卸载"
4. **回归级**: 每个Phase后验证日常对话不受影响

### 关键验收标准
- 日常闲聊响应不受技能包系统影响（ModuleRouter正确过滤）
- 工具调用从固定3次提升到智能动态控制
- 技能包可安装/执行/卸载，工具自动注册/注销
- 执行过程中经验自动提取到ExperienceStore
- 卸载技能包后经验保留，祖龙靠经验继续运行
- 复杂任务(>3步)正确由L1-B接管执行，任务进度可追踪
- OpenManus深度推理与AutoGPT任务拆解可协同工作
- MCP接入后可调用外部工具生态

### 与技能插件理论分析.txt的对齐验证
本规划已融合理论文档的核心洞见：
- [x] "借用→学习→内化→丢弃"生命周期 → ISkillPack + 内化评估 + 卸载
- [x] "外科手术式"提取 → 不整体引入，提取核心算法重写
- [x] Windows类比 → 祖龙=OS，技能包=可卸载软件
- [x] ModuleRouter动态路由 → 双层路由（快速预判+Function Calling）
- [x] 三层集成策略 → MCP(工具层)+AutoGPT(执行层)+OpenManus(推理层)
- [x] 执行权在L1-B → 复杂任务L1-B接管，简单任务L2直接执行
- [x] SharedMemoryPool任务队列 → 复杂任务子任务列表持久化
- [x] 法律风险策略 → "参考架构，重写接口"
- [x] "深度集成能力，而非代码" → 提取算法逻辑，用祖龙风格重写
