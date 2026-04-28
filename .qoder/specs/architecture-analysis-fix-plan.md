# 祖龙系统架构修复方案 - 图记忆、对话节点、任务恢复

## Context

测试报告发现祖龙系统在普通对话和复杂任务流程中存在 5 个架构级问题，导致：
- MemoryGraph 中每条消息产生 2 个冗余 round 节点
- 每条消息都创建新 session，对话上下文完全割裂
- "继续之前的任务"无法归回原 session，恢复流程形同虚设
- FC 工具虽然完整，但模型缺少恢复引导，不主动调用 task_list_suspended

本方案通过 3 批共 11 处精确修改，解决所有问题。

---

## 架构数据流全景

```
用户消息
  |
  v
Gatekeeper._handle_normal_command()     [scheduler_gatekeeper.py:1635]
  |-- 1. 恢复意图检测 (新增)              --> is_resume, resume_task_id
  |-- 2. 打包 packaged_task               --> {text, is_resume, resume_task_id, ...}
  |-- 3. _ensure_dialogue_node()          --> 创建 round 骨架 + 确定 session
  |-- 4. 发布 SYSTEM_L2_COMMAND 事件
  v
InferenceEngine._on_l2_command()         [inference_engine.py:466]
  |-- 缓存 dialogue_round_id, session_id, is_resume
  v
InferenceEngine._process_with_memory()   [inference_engine.py:659]
  |-- RAG 检索 + 构建 messages
  |-- FC 循环 (行765-847): tool_choice="auto", max 10 轮
  |   |-- 模型决定: 直接回复 or 调用工具
  |   |-- (恢复场景) system prompt 注入引导 --> 模型调用 task_list_suspended
  |-- _update_memory(): 复用 GK 的 round 节点，补充 bot 回复 (不再新建)
  v
MemoryGraph: Session --> Round --> SubDialogue  (单一 round, 完整内容)
```

---

## 第一批修改: 解决双重 Round 节点 [最高优先级]

### 核心决策: Gatekeeper 创建骨架，InferenceEngine 复用并补充

**理由**:
- GK 在入口即创建 round，保证时间线完整性（即使 FC 循环耗时数十秒或被中断）
- IE 有完整的 bot 回复，负责补充内容
- Session 归属由 GK 一次性确定，IE 不再重复调用 ensure_session

### 修改 1: InferenceEngine._on_l2_command() - 缓存 GK 传来的节点信息

**文件**: `zulong/l2/inference_engine.py`  
**位置**: 行 474-475 之后，行 528 之前

**修改**: 在提取现有字段后，新增缓存逻辑：

```python
# 行 475 之后新增:
self._current_dialogue_round_id = event.payload.get("dialogue_round_id", None)
self._current_session_id = event.payload.get("session_id", "")
self._is_resume = is_resume
self._resume_task_id = event.payload.get("resume_task_id", "")
```

### 修改 2: InferenceEngine._update_memory() - 复用 round 而非新建

**文件**: `zulong/l2/inference_engine.py`  
**方法**: `_update_memory()` (行 1061-1119)

**将行 1073-1107 替换为**:

```python
try:
    from zulong.memory.memory_graph import get_memory_graph
    from zulong.memory.graph_adapters import DialogueAdapter
    
    mg = get_memory_graph()
    if mg:
        adapter = DialogueAdapter()
        
        # 优先复用 Gatekeeper 已创建的 round 节点
        gk_round_id = getattr(self, '_current_dialogue_round_id', None)
        
        if gk_round_id and mg.get_node(gk_round_id):
            # --- 复用路径: 补充内容到 GK 创建的 round ---
            round_id = gk_round_id
            session_id = getattr(self, '_current_session_id', "")
        else:
            # --- 降级路径: GK round 不可用，走原有逻辑 ---
            session_id = adapter.ensure_session(
                mg, user_input,
                is_resume=getattr(self, '_is_resume', False),
                resume_task_id=getattr(self, '_resume_task_id', ""),
            )
            prev_round_id = getattr(self, '_last_round_id', None)
            request_id = str(int(time.time() * 1000))
            round_id = adapter.add_round(
                mg, request_id, user_input,
                prev_round_id=prev_round_id,
                session_id=session_id,
            )
        
        # 添加 bot 回复为 sub_dialogue
        adapter.add_sub_dialogue(
            mg, round_id, turn=1,
            content=response, role="assistant",
        )
        
        self._last_round_id = round_id
        
        # 更新 round 节点内容（无论复用还是新建）
        round_node = mg.get_node(round_id)
        if round_node:
            round_node.metadata["content"] = f"用户：{user_input}\n回答：{response[:200]}"
            round_node.metadata["user_text"] = user_input
            round_node.metadata["bot_text"] = response
        
        # 消耗一次性引用
        self._current_dialogue_round_id = None
        
        logger.info(f"[MemoryGraph] 对话已写入: {round_id} (session={session_id})")
except Exception as e:
    logger.warning(f"[MemoryGraph] 记忆写入失败: {e}")
```

### 修改 3: InferenceEngine._update_memory_async() - 同上逻辑

**文件**: `zulong/l2/inference_engine.py`  
**方法**: `_update_memory_async()` (行 1620-1687)

**行 1633-1674 应用与修改2 完全相同的逻辑**: 先检查 `_current_dialogue_round_id`，有则复用，无则降级走原逻辑。

---

## 第二批修改: 打通恢复流程

### 修改 4: Gatekeeper._handle_normal_command() - 恢复意图检测

**文件**: `zulong/l1b/scheduler_gatekeeper.py`  
**方法**: `_handle_normal_command()` (行 1737 之前)

**在行 1736（voice_mode 检测完成）之后、行 1738（打包 packaged_task）之前，新增**:

```python
# 恢复意图检测
_RESUME_KEYWORDS = ['继续', '接着做', '恢复任务', '回到之前', '上次的', '接着', '继续之前', '上次那个']
text_lower = text.lower()
is_resume = any(kw in text_lower for kw in _RESUME_KEYWORDS)
resume_task_id = ""
task_graph_id = ""

if is_resume:
    try:
        from zulong.l2.task_suspension import TaskSuspensionManager
        mgr = TaskSuspensionManager()
        import asyncio
        tasks = asyncio.get_event_loop().run_until_complete(mgr.list_suspended_tasks())
        if tasks:
            resume_task_id = tasks[0].get("task_id", "")
            task_graph_id = tasks[0].get("metadata", {}).get("graph_id", "")
        logger.info(f"[Gatekeeper] 恢复意图检测: resume_task_id={resume_task_id}")
    except Exception as e:
        logger.debug(f"[Gatekeeper] 恢复意图检测失败: {e}")
```

> **注意**: `run_until_complete` 在已有事件循环中可能失败。需要检查 Gatekeeper 是否在异步上下文中运行。如果是，改用同步文件列表扫描 `data/suspended_tasks/` 目录获取最近的 task_id。

### 修改 5: Gatekeeper._handle_normal_command() - packaged_task 增加恢复字段

**文件**: `zulong/l1b/scheduler_gatekeeper.py`  
**位置**: 行 1738-1743

**修改**: 在 packaged_task 字典中增加 3 个字段:

```python
packaged_task = {
    "text": text,
    "local_context": local_context,
    "shared_context_snapshot": shared_context_snapshot,
    "voice_mode": voice_mode,
    "is_resume": is_resume,              # 新增
    "resume_task_id": resume_task_id,    # 新增
    "task_graph_id": task_graph_id,      # 新增
}
```

### 修改 6: InferenceEngine._process_with_memory() - 恢复引导注入

**文件**: `zulong/l2/inference_engine.py`  
**方法**: `_process_with_memory()` (行 659)

**修改签名** (行 659):
```python
def _process_with_memory(self, user_input: str, priority: EventPriority, 
                          voice_mode: str = "TEXT_ONLY", is_resume: bool = False):
```

**在构建 messages 后 (行 735)、FC 循环前 (行 745)，新增**:
```python
# 恢复场景: 注入引导提示，让模型主动调用 task_list_suspended
if is_resume:
    messages.append({
        "role": "system",
        "content": (
            "[系统提示] 用户正在尝试恢复之前的任务。"
            "请立即使用 task_list_suspended 工具查看挂起的任务列表，"
            "然后恢复最相关的任务并继续执行。"
        )
    })
```

### 修改 7: InferenceEngine._on_l2_command() - 传递 is_resume

**文件**: `zulong/l2/inference_engine.py`  
**位置**: 行 528

**修改**:
```python
# 原: self._process_with_memory(text, event.priority, voice_mode)
self._process_with_memory(text, event.priority, voice_mode, is_resume=is_resume)
```

---

## 第三批修改: Session 话题检测改进

### 修改 8: DialogueAdapter._is_same_topic() - 多策略融合

**文件**: `zulong/memory/graph_adapters.py`  
**方法**: `_is_same_topic()` (行 539-575)

**替换为**:

```python
def _is_same_topic(
    self, graph: MemoryGraph, session: GraphNode, new_text: str,
) -> bool:
    import re, time as _time

    # 策略1: 绑定任务的 session 默认延续（已有逻辑，保持）
    bound_task = session.metadata.get("bound_task_id", "")
    if bound_task:
        return True

    # 策略2 (新增): 时间窗口 - 5分钟内的消息默认延续同一 session
    last_active = session.metadata.get("last_active_at", session.created_at)
    if (_time.time() - last_active) < 300:  # 5分钟
        return True

    # 策略3: 关键词交集（阈值从 >=2 降为 >=1）
    latest_goal = self._get_latest_round_goal(graph, session.node_id)
    if not latest_goal:
        latest_goal = session.metadata.get("topic_summary", "")
    if not latest_goal:
        return False

    def extract_keywords(text: str) -> set:
        segments = re.split(r'[，、。！？\s,.\-:;/\\()\[\]{}""\'\']+', text.lower())
        return {seg for seg in segments if len(seg) >= 2 and seg not in self._STOPWORDS}

    old_kw = extract_keywords(latest_goal)
    new_kw = extract_keywords(new_text)
    overlap = old_kw & new_kw

    return len(overlap) >= 1  # 原为 >= 2
```

### 修改 9: DialogueAdapter.add_round() - 更新 session 活跃时间

**文件**: `zulong/memory/graph_adapters.py`  
**方法**: `add_round()` 

**在创建 round 节点并建立 HIERARCHY 边之后**（session round_count 更新附近），新增:

```python
# 更新 session 最后活跃时间（供 _is_same_topic 时间窗口策略使用）
if session_id:
    sess_node = graph.get_node(session_id)
    if sess_node:
        sess_node.metadata["last_active_at"] = time.time()
```

### 修改 10: DialogueAdapter.ensure_session() - 增加 force_session_id

**文件**: `zulong/memory/graph_adapters.py`  
**方法**: `ensure_session()` (行 409-414)

**修改签名**, 新增参数:

```python
def ensure_session(
    self, graph: MemoryGraph, text: str,
    is_resume: bool = False,
    resume_task_id: Optional[str] = None,
    task_graph_id: Optional[str] = None,
    force_session_id: Optional[str] = None,   # 新增
) -> str:
```

**在方法体最前面**（行 433 之前）新增:

```python
# 优先使用外部指定的 session_id（如 Web API 传入的会话 ID）
if force_session_id and graph.has_node(force_session_id):
    logger.debug(f"[DialogueAdapter] 使用指定 session: {force_session_id}")
    return force_session_id
```

### 修改 11: DialogueAdapter._STOPWORDS - 扩展停用词

**文件**: `zulong/memory/graph_adapters.py`  
**位置**: `_STOPWORDS` 定义处

**增加**: `'继续', '之前', '任务', '上次', '恢复', '接着', '那个', '这个', '什么', '怎么', '可以', '帮我'`

---

## 三种场景数据流验证

### 场景1: 普通对话 "今天天气怎么样？"

```
GK: is_resume=False
GK: ensure_session() --> 时间窗口5分钟内 --> 延续 session_abc
GK: add_round("gk_xxx") --> round_gk_xxx (仅有 goal)
GK: packaged_task = {dialogue_round_id: "dialogue:round_gk_xxx", session_id: "session_abc"}

IE: 缓存 _current_dialogue_round_id = "dialogue:round_gk_xxx"
IE: FC循环 --> 模型直接回复
IE: _update_memory() --> mg.get_node("dialogue:round_gk_xxx") 存在 --> 复用
IE: 补充 bot_text, user_text, sub_dialogue

结果: 1个 round 节点, 同一 session, 内容完整
```

### 场景2: 复杂任务 "帮我搜索AI市场动态并分析"

```
GK: is_resume=False, 新建 session_def (与前无关联)
GK: add_round --> round_gk_yyy

IE: FC循环多轮:
  Turn1: task_create_plan("AI市场分析")
  Turn2: task_add_node(...)
  Turn3: openclaw_search("AI市场 2025")
  Turn4: 模型综合分析, 直接回复
IE: _update_memory() --> 复用 round_gk_yyy, 补充完整内容

结果: 1个 round + TaskGraph, session_def 可通过 bound_task_id 绑定任务
```

### 场景3: 退出后恢复 "继续之前的任务"

```
GK: 恢复关键词检测 --> is_resume=True, resume_task_id="tg_xxx"
GK: ensure_session(is_resume=True, resume_task_id="tg_xxx")
    --> 策略2: 找到 session_abc (bound_task_id="tg_xxx") --> 归回原 session
GK: add_round --> round_gk_zzz (在 session_abc 下)

IE: is_resume=True --> system prompt 注入恢复引导
IE: FC循环:
  Turn1: 模型调用 task_list_suspended(query="之前的任务")
         --> 恢复 TaskGraph, set_active_task_graph()
  Turn2: task_view_overview() 查看进度
  Turn3: 继续执行未完成节点
  TurnN: 综合回复

结果: 归回原 session, TaskGraph 恢复, 上下文连贯
```

---

## 涉及的关键文件

| 文件 | 修改项 | 行号范围 |
|------|--------|----------|
| `zulong/l2/inference_engine.py` | 修改1,2,3,6,7 | 474, 528, 659, 1061-1119, 1620-1687 |
| `zulong/l1b/scheduler_gatekeeper.py` | 修改4,5 | 1736-1743 |
| `zulong/memory/graph_adapters.py` | 修改8,9,10,11 | 409-414, 539-575, add_round内, _STOPWORDS |

---

## 验证计划

### 步骤1: 重启祖龙系统
```bash
# 在独立终端重启
```

### 步骤2: 普通对话测试
- 发送 "你好祖龙"
- 发送 "今天有什么新闻"
- **验证**: MemoryGraph 中这2条消息在同一 session 下，各只有 1 个 round 节点

### 步骤3: 复杂任务测试
- 发送 "帮我搜索2025年AI助手市场动态，分析主流产品的优劣势"
- 发送追问 "哪个产品最有潜力？"
- **验证**: 
  - 2条消息在同一 session（时间窗口内）
  - FC 循环调用了搜索工具（SearXNG 现已可用）
  - 追问能引用到前一轮的分析结果

### 步骤4: 恢复测试
- 关闭客户端连接
- 重新连接
- 发送 "继续之前的任务"
- **验证**:
  - 恢复消息归入原 session（bound_task_id 匹配）
  - FC 循环调用了 task_list_suspended
  - 模型回复中引用了之前的任务上下文

### 步骤5: MemoryGraph 分析
- 运行 analyze_mg.py 脚本分析 memory_graph.json
- 确认: 无双重 round 节点、session 数量合理、恢复消息归入原 session

---

## 风险与降级

| 风险 | 降级策略 |
|------|---------|
| GK 的 round_id 在 IE 中找不到 | _update_memory 有降级路径，回退到原有的完整创建逻辑 |
| 恢复意图误检（"请继续说" 误判） | 最终由 FC 循环中模型决定是否调用 task_list_suspended，误检无害 |
| 时间窗口5分钟导致不同话题合并 | 5分钟阈值可配置化；远好于当前每条消息都断裂的问题 |
| Gatekeeper 中 asyncio 事件循环冲突 | 改用同步目录扫描 `data/suspended_tasks/` 获取最近 task_id |
