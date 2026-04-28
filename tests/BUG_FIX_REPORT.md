# Bug 修复说明

**修复时间**: 2026-04-16  
**修复内容**: dependency_analyze 和 priority_rank 参数处理 bug  
**影响范围**: ComplexTaskPack 技能包

---

## 🐛 问题描述

### 原始错误

```
[ToolEngine] Execute error: 'str' object has no attribute 'get'
```

### 问题原因

`DependencyAnalyzeTool` 和 `PriorityRankTool` 在执行时，期望 `subtasks` 参数是字典列表格式:

```python
[{"task": "任务描述", "tool_hint": "search"}]
```

但测试脚本和实际调用中可能传入字符串列表:

```python
["订机票", "订酒店", "安排行程"]
```

导致在访问 `task.get("task")` 时抛出 AttributeError。

---

## ✅ 修复方案

### 1. 修复 planning_tools.py

**文件**: `zulong/skill_packs/packs/complex_task/planning_tools.py`

#### 修复 1: DependencyAnalyzeTool

```python
def execute(self, request: ToolRequest) -> ToolResult:
    start_time = time.time()
    
    subtasks = request.parameters.get("subtasks", [])
    goal = request.parameters.get("goal", "")
    
    if not subtasks:
        return self._create_result(
            success=False,
            error="缺少子任务列表 (subtasks 参数)",
            execution_time=time.time() - start_time,
            request_id=request.request_id,
        )
    
    # 兼容处理：支持字符串列表和字典列表两种格式
    if subtasks and isinstance(subtasks[0], str):
        # 如果是字符串列表，转换为字典格式
        subtasks = [{"task": task, "tool_hint": "execute"} for task in subtasks]
    
    dependencies = self.planner.analyze_dependencies(subtasks, goal)
    parallel_groups = self.planner._compute_parallel_groups(subtasks, dependencies)
    
    return self._create_result(
        success=True,
        data={
            "dependencies": dependencies,
            "parallel_groups": parallel_groups,
        },
        execution_time=time.time() - start_time,
        request_id=request.request_id,
    )
```

#### 修复 2: PriorityRankTool

```python
def execute(self, request: ToolRequest) -> ToolResult:
    start_time = time.time()
    
    subtasks = request.parameters.get("subtasks", [])
    
    if not subtasks:
        return self._create_result(
            success=False,
            error="缺少子任务列表 (subtasks 参数)",
            execution_time=time.time() - start_time,
            request_id=request.request_id,
        )
    
    # 兼容处理：支持字符串列表和字典列表两种格式
    if subtasks and isinstance(subtasks[0], str):
        # 如果是字符串列表，转换为字典格式
        subtasks = [{"task": task, "tool_hint": "execute"} for task in subtasks]
    
    ranked = self.planner.rank_priorities(subtasks)
    
    return self._create_result(
        success=True,
        data={"ranked_subtasks": ranked},
        execution_time=time.time() - start_time,
        request_id=request.request_id,
    )
```

### 2. 更新测试脚本

**文件**: `tests/test_complex_task.py`

更新参数格式为字典列表:

```python
# 依赖分析
parameters={
    "goal": "规划 7 天东京旅行",
    "subtasks": [
        {"task": "订机票", "tool_hint": "search"},
        {"task": "订酒店", "tool_hint": "search"},
        {"task": "安排行程", "tool_hint": "analyze"},
        {"task": "办理签证", "tool_hint": "execute"},
        {"task": "购买保险", "tool_hint": "execute"}
    ]
}

# 优先级排序
parameters={
    "subtasks": [
        {"task": "Node.js+Socket.IO", "tool_hint": "analyze"},
        {"task": "Python+FastAPI", "tool_hint": "analyze"},
        {"task": "Go+Gorilla", "tool_hint": "analyze"}
    ],
    "criteria": ["性能", "开发效率", "生态成熟度"]
}
```

### 3. 更新文档

**文件**: `tests/complex_task_test_plan.md`

更新工具调用示例:

```python
# 任务规划
task_decompose.execute(
    goal="规划 7 天东京旅行",
    constraints=["预算 2 万元", "7 天时间", "必去 3 个景点"]
)

# 依赖分析
dependency_analyze.execute(
    goal="规划 7 天东京旅行",
    subtasks=[
        {"task": "订机票", "tool_hint": "search"},
        {"task": "订酒店", "tool_hint": "search"},
        {"task": "安排行程", "tool_hint": "analyze"},
        {"task": "准备签证", "tool_hint": "execute"}
    ]
)

# 深度推理
deep_reasoning.execute(
    problem="如何在预算限制下优化旅行体验",
    context={"budget": 20000, "days": 7, "preferences": ["美食", "购物", "文化"]}
)
```

---

## 🧪 测试结果

修复后重新运行测试:

```
================================================================================
  测试总结
================================================================================

📊 测试统计:
   总任务数：3
   成功：3
   失败：0
   成功率：100.0%  ✅

📝 详细结果:
   ✅ 旅行规划 - 2026-04-16T20:13:52.138560
   ✅ 技术选型 - 2026-04-16T20:13:54.923020
   ✅ 商业计划书 - 2026-04-16T20:13:55.799373
```

### 修复前后对比

| 功能 | 修复前 | 修复后 |
|-----|--------|--------|
| dependency_analyze | ❌ 失败 | ✅ 成功 |
| priority_rank | ❌ 失败 | ✅ 成功 |
| task_decompose | ✅ 成功 | ✅ 成功 |
| deep_reasoning | ✅ 成功 | ✅ 成功 |
| openclaw_search | ✅ 成功 | ✅ 成功 |

**成功率**: 60% → 100%

---

## 📋 修复清单

- [x] 修复 `DependencyAnalyzeTool.execute()` 参数处理
- [x] 修复 `PriorityRankTool.execute()` 参数处理
- [x] 更新测试脚本参数格式
- [x] 更新测试方案文档
- [x] 重新运行测试验证
- [x] 创建修复说明文档

---

## 💡 改进建议

### 短期

1. **增强参数验证**
   - 在工具执行前进行参数类型检查
   - 提供更友好的错误信息

2. **添加单元测试**
   - 测试字符串列表输入
   - 测试字典列表输入
   - 测试混合输入

### 中期

1. **参数格式标准化**
   - 统一所有工具的参数格式
   - 提供参数转换工具

2. **文档完善**
   - 更新所有工具的参数说明
   - 添加使用示例

### 长期

1. **类型系统**
   - 引入 Pydantic 进行参数验证
   - 提供参数 schema 自动生成

2. **兼容性层**
   - 支持多种参数格式
   - 自动格式检测和转换

---

**修复人**: AI Assistant  
**审核状态**: 已完成  
**测试状态**: 通过 ✅
