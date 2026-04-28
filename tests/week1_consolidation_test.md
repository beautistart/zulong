# 第 1 周优化验证测试

**测试日期**: 2026-04-04  
**测试内容**: 记忆巩固 + 自动复盘 + 持久化  
**状态**: 待测试

---

## 📋 测试清单

### 1. 记忆巩固机制测试

#### 测试场景 1: 重要对话立即巩固

**测试代码**:
```python
import asyncio
from zulong.memory.short_term_memory import get_short_term_memory

async def test_consolidation():
    stm = get_short_term_memory()
    
    # 测试 1: 包含关键信息的对话（应该巩固）
    print("\n=== 测试 1: 重要对话 ===")
    await stm.store(
        user_input="我的名字是张三",
        ai_response="好的，我记住了您叫张三。很高兴认识您！"
    )
    
    # 等待巩固完成
    await asyncio.sleep(1)
    
    # 测试 2: 普通对话（不应该立即巩固）
    print("\n=== 测试 2: 普通对话 ===")
    await stm.store(
        user_input="你好",
        ai_response="你好！有什么可以帮助你的？"
    )
    
    # 查看统计
    stats = stm.get_stats()
    print(f"\n📊 统计信息：{stats}")
    
    # 验证：应该有 1 次巩固
    assert stats["total_consolidations"] >= 1, "重要对话应该被巩固"
    print("✅ 测试通过：重要对话已巩固")

# 运行测试
asyncio.run(test_consolidation())
```

**预期结果**:
- ✅ 日志显示"重要性评分：turn=1, importance=0.85"
- ✅ 日志显示"重要对话，立即巩固：turn_id=1"
- ✅ 日志显示"对话已转为长期记忆：turn_id=1"
- ✅ 统计信息显示 total_consolidations >= 1

---

### 2. 持久化机制测试

#### 测试场景 2: 系统重启后恢复索引

**测试代码**:
```python
import asyncio
from zulong.memory.short_term_memory import ShortTermMemory

async def test_persistence():
    print("\n=== 测试：持久化 ===")
    
    # 第 1 次启动：存储对话
    print("\n--- 第 1 次启动 ---")
    stm1 = ShortTermMemory()
    await stm1.store(
        user_input="测试持久化",
        ai_response="这是测试回复"
    )
    
    current_turn = stm1.get_current_turn()
    print(f"当前轮数：{current_turn}")
    
    # 模拟系统重启（创建新实例）
    print("\n--- 第 2 次启动（模拟重启）---")
    stm2 = ShortTermMemory()
    
    # 验证：索引已恢复
    restored_turn = stm2.get_current_turn()
    print(f"恢复后轮数：{restored_turn}")
    
    assert restored_turn == current_turn, f"索引应该恢复：期望{current_turn}, 实际{restored_turn}"
    print("✅ 测试通过：索引已成功恢复")

# 运行测试
asyncio.run(test_persistence())
```

**预期结果**:
- ✅ 第 1 次启动：存储对话，保存索引到磁盘
- ✅ 第 2 次启动：从磁盘加载索引
- ✅ 日志显示"已恢复短期记忆索引：1 轮"
- ✅ 轮数一致：restored_turn == current_turn

---

### 3. 定期巩固测试

#### 测试场景 3: 每小时自动巩固

**测试代码**:
```python
import asyncio
from zulong.memory.short_term_memory import get_short_term_memory
import time

async def test_periodic_consolidation():
    stm = get_short_term_memory()
    
    print("\n=== 测试：定期巩固 ===")
    
    # 修改巩固间隔为 5 秒（便于测试）
    stm.consolidation_interval = 5
    stm.last_consolidation_time = time.time() - 10  # 模拟已过 10 秒
    
    # 存储对话
    await stm.store(
        user_input="测试定期巩固",
        ai_response="这是测试回复"
    )
    
    # 等待定期巩固触发
    print("等待定期巩固...")
    await asyncio.sleep(2)
    
    stats = stm.get_stats()
    print(f"📊 统计信息：{stats}")
    
    print("✅ 测试完成")

# 运行测试
asyncio.run(test_periodic_consolidation())
```

**预期结果**:
- ✅ 日志显示"执行定期记忆巩固..."
- ✅ 日志显示"巩固了 X 条记忆"
- ✅ 统计信息显示 total_consolidations 增加

---

## 📊 重要性评分测试

### 测试场景 4: 重要性计算

**测试代码**:
```python
from zulong.memory.short_term_memory import get_short_term_memory

def test_importance_calculation():
    stm = get_short_term_memory()
    
    print("\n=== 测试：重要性计算 ===")
    
    # 测试用例
    test_cases = [
        {
            "user": "我的名字是张三",
            "ai": "好的，我记住了您叫张三",
            "expected_min": 0.7  # 包含关键信息
        },
        {
            "user": "为什么天是蓝的？",
            "ai": "因为瑞利散射...",
            "expected_min": 0.65  # 用户追问
        },
        {
            "user": "谢谢！",
            "ai": "不客气~",
            "expected_min": 0.65  # 用户情感
        },
        {
            "user": "你好",
            "ai": "你好",
            "expected_min": 0.5  # 普通对话
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        importance = stm._calculate_importance(case["user"], case["ai"])
        print(f"\n测试{i}: importance={importance:.2f} (期望>={case['expected_min']})")
        print(f"  用户：{case['user']}")
        print(f"  AI: {case['ai']}")
        
        assert importance >= case["expected_min"], f"重要性评分过低：{importance}"
    
    print("\n✅ 所有重要性测试通过")

# 运行测试
test_importance_calculation()
```

**预期结果**:
- ✅ 包含关键信息的对话：importance >= 0.7
- ✅ 用户追问：importance >= 0.65
- ✅ 用户情感：importance >= 0.65
- ✅ 普通对话：importance = 0.5

---

## 🎯 验收标准

### 第 1 周优化完成标准

- [x] ✅ 记忆巩固机制已激活
- [x] ✅ 重要性评分算法已实现
- [x] ✅ 高重要性对话立即巩固
- [x] ✅ 定期批量巩固（每小时）
- [x] ✅ 持久化机制已启用
- [x] ✅ 系统重启后索引恢复
- [ ] ⏳ 所有测试用例通过

---

## 🔍 调试日志示例

### 正常流程日志

```
[ShortTermMemory] 初始化完成 (纯异步版本 + 记忆巩固)
  - 最大轮数：20
  - TTL: 3600s
  - 记忆巩固：✅ 已激活 (阈值=0.7)
  - 持久化：✅ 已启用 (路径=./data/short_term_memory)

[ShortTermMemory] 存储对话：turn=1, 我的名字是张三...
📊 [记忆巩固] 重要性评分：turn=1, importance=0.85
✅ [记忆巩固] 重要对话，立即巩固：turn_id=1
✅ [记忆巩固] 对话已转为长期记忆：turn_id=1
✅ [持久化] 短期记忆索引已保存

✅ 测试通过：重要对话已巩固
```

### 重启恢复日志

```
--- 第 1 次启动 ---
[ShortTermMemory] 初始化完成 (纯异步版本 + 记忆巩固)
[ShortTermMemory] 存储对话：turn=1, 测试持久化...
✅ [持久化] 短期记忆索引已保存
当前轮数：1

--- 第 2 次启动（模拟重启）---
✅ [持久化] 已恢复短期记忆索引：1 轮
恢复后轮数：1
✅ 测试通过：索引已成功恢复
```

---

**测试执行者**: ___________  
**测试日期**: ___________  
**测试结果**: [ ] 通过 [ ] 失败  
**备注**: ___________
