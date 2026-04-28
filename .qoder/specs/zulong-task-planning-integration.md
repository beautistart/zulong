# 技能包任务规划系统：修复断层让现有架构跑通

## Context

祖龙系统的任务规划/拆分功能已通过技能包体系实现，架构设计如下：

```
用户输入 → Gatekeeper → SYSTEM_L2_COMMAND → L2 InferenceEngine
  → _generate_with_vllm_and_tools(messages)
    → ToolRegistry().get_all_function_schemas()  ← 技能包工具应在此出现
    → vLLM 通过 Function Calling 自主决定是否调用 task_decompose 等工具
    → ToolEngine 执行工具 → planner.decompose() → 返回子任务
    → L2 根据子任务结果生成最终回复
```

**这条链路的设计是完整的**——`module_router.py` 提供第一层分类，L2 的 Function Calling 负责第二层自主决策。ToolRegistry 是单例，技能包注册的工具可以被 L2 的 vLLM 看到。

**但链路上有 5 个断层，导致系统完全无法运作：**

| # | 断层 | 位置 | 后果 |
|---|------|------|------|
| 1 | 所有技能包 `enabled: false` | `config/skill_packs.yaml` | 工具从未注册到 ToolRegistry，L2 看不到 |
| 2 | `experience_store=None` | `bootstrap.py:151` | 经验永远无法持久化 |
| 3 | `_record_experience()` 调用 `.add()` | `runtime.py:327` | EnhancedExperienceStore 的 API 是 `.add_experience()`，会崩溃 |
| 4 | planner 没有 `llm_client` | `autogpt_planner/__init__.py:52` | 任务拆解只能走关键词降级，无法用 vLLM 智能拆解 |
| 5 | planner 的 `model_id="default"` | `autogpt_planner/__init__.py:50` | 即使注入 llm_client，model_id 与 vLLM 不匹配 |

## 修复步骤

### Step 1: 启用技能包配置

**文件**: `config/skill_packs.yaml`

```yaml
skill_packs:
  - pack_id: "autogpt_planner"
    enabled: true   # false → true
    path: "zulong.skill_packs.packs.autogpt_planner"
    config:
      max_subtasks: 10
      planning_model: "default"

  - pack_id: "openmanus_reasoner"
    enabled: true   # false → true
    path: "zulong.skill_packs.packs.openmanus_reasoner"
    config:
      reasoning_depth: 3
      max_hypotheses: 5

  - pack_id: "cline_coder"
    enabled: false  # 保持禁用，需要文件系统权限
    ...
```

启用后，`SkillPackLoader.load_from_config()` 会加载这些包 → `install_pack()` 注册工具到 ToolRegistry → L2 的 Function Calling 能看到 `task_decompose`、`deep_reasoning` 等工具。

---

### Step 2: 修复 bootstrap.py — 注入 experience_store

**文件**: `zulong/bootstrap.py` (第 149-153 行)

```python
# 修复前:
self.skill_pack_runtime = SkillPackRuntime(
    tool_engine=inference_engine.tool_engine,
    experience_store=None,  # ← BUG
    hot_update_engine=get_hot_update_engine()
)

# 修复后:
from zulong.memory.enhanced_experience_store import EnhancedExperienceStore
experience_store = EnhancedExperienceStore()

self.skill_pack_runtime = SkillPackRuntime(
    tool_engine=inference_engine.tool_engine,
    experience_store=experience_store,
    hot_update_engine=get_hot_update_engine()
)
```

---

### Step 3: 修复 runtime.py — _record_experience API 不匹配

**文件**: `zulong/skill_packs/runtime.py` (第 315-329 行)

当前代码调用 `self.experience_store.add(experience)` 但 `EnhancedExperienceStore` 没有 `.add()` 方法，只有 `.add_experience(content, experience_type, ...)`。

```python
# 修复前 (327行):
self.experience_store.add(experience)

# 修复后:
content = (f"技能包 {pack_id} 执行 {capability}: "
           f"{'成功' if result.get('success', False) else '失败'}, "
           f"耗时 {execution_time:.2f}s")
self.experience_store.add_experience(
    content=content,
    experience_type="skill_pack_execution",
    task_id=pack_id,
    success=result.get("success", False),
    metadata={
        "pack_id": pack_id,
        "capability": capability,
        "params": str(params)[:500],
        "execution_time": execution_time,
    },
    tags=[pack_id, capability],
)
```

---

### Step 4: 修复 AutoGPTPlannerPack — 注入 llm_client 和正确的 model_id

**文件 A**: `zulong/bootstrap.py` (第 149-169 行)

在技能包加载配置中注入 vllm_client 和 model_id：

```python
# 在 skill_pack_runtime 创建之后、load_from_config 之前
# 将 vllm_client 和 model_id 注入到 config 中供技能包使用
self.skill_pack_runtime._vllm_client = getattr(inference_engine, 'vllm_client', None)
self.skill_pack_runtime._vllm_model_id = "/mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-0.8B-AWQ"
```

**文件 B**: `zulong/skill_packs/packs/autogpt_planner/__init__.py` (第 47-55 行)

修改 `install()` 方法，从 config 或 runtime 获取 llm_client：

```python
def install(self, tool_registry, config=None):
    config = config or {}
    max_subtasks = config.get("max_subtasks", 10)

    # 尝试获取 vLLM 客户端
    llm_client = config.get("llm_client", None)
    model_id = config.get("model_id", "default")

    self._planner = TaskDecomposeAlgorithm(
        max_subtasks=max_subtasks,
        llm_client=llm_client,
        model_id=model_id,
    )
    ...
```

**文件 C**: `zulong/skill_packs/runtime.py` — `install_pack()` 方法 (第 76 行)

在调用 `pack.install()` 时，将 vllm_client 注入 config：

```python
# 修复前 (76-77行):
success = pack.install(ToolRegistry(), config)

# 修复后:
install_config = dict(config or {})
if hasattr(self, '_vllm_client') and self._vllm_client:
    install_config['llm_client'] = self._vllm_client
if hasattr(self, '_vllm_model_id'):
    install_config['model_id'] = self._vllm_model_id
success = pack.install(ToolRegistry(), install_config)
```

---

### Step 5: 创建集成测试并验证

**新文件**: `tests/test_skill_pack_integration.py`

测试内容：

1. **技能包加载测试**: 验证 `load_from_config()` 正确加载 autogpt_planner 和 openmanus_reasoner
2. **工具注册测试**: 验证 `task_decompose`、`deep_reasoning` 等工具出现在 `ToolRegistry().get_all_function_schemas()` 中
3. **Planner LLM 拆解测试**: 直接调用 `execute_capability("autogpt_planner", "task_decompose", {"goal": "..."})` 验证 vLLM 返回结构化子任务
4. **经验记录测试**: 执行后验证 `EnhancedExperienceStore` 中存入了经验
5. **端到端测试**: WebSocket 发送 "帮我调研 Rust vs Go 在嵌入式AI场景的优劣并写一份报告"，验证 L2 通过 Function Calling 调用 task_decompose 工具

---

## 关键文件清单

| 操作 | 文件路径 | 修改点 |
|------|----------|--------|
| 修改 | `config/skill_packs.yaml` | enabled: false → true |
| 修改 | `zulong/bootstrap.py` | experience_store 注入 + vllm_client 传递 |
| 修改 | `zulong/skill_packs/runtime.py` | _record_experience API + install_pack config 注入 |
| 修改 | `zulong/skill_packs/packs/autogpt_planner/__init__.py` | install() 接收 llm_client |
| 新建 | `tests/test_skill_pack_integration.py` | 集成验证测试 |

## 验证方式

1. **单元验证**: 运行 `tests/test_skill_pack_integration.py`
2. **日志验证**: 启动系统后检查日志中是否出现：
   - `[SkillPackRuntime] 已加载技能包: autogpt_planner`
   - `[SkillPackRuntime] 工具已注册: task_decompose`
   - `[SkillPackRuntime] 工具已注册: deep_reasoning`
3. **端到端验证**: WebSocket 发送复杂任务，检查 L2 日志是否出现 Function Calling 调用 task_decompose
4. **经验闭环验证**: 执行任务后检查 ExperienceStore 是否存入经验

## 为什么不需要新建 TaskOrchestrator

现有架构已经具备完整的任务规划能力：

- **L2 Function Calling** = 自动路由（vLLM 自主决定是否调用 task_decompose）
- **ToolRegistry 单例** = 技能包工具全局可见
- **SkillPackRuntime.execute_capability()** = 执行 + 自动记录经验
- **EnhancedExperienceStore** = 经验持久化 + BM25 检索
- **module_router.quick_class()** = 闲聊/任务预判（可选优化）

只需修复 5 个断层，现有系统即可正常运作。
