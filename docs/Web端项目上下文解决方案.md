# Web端项目上下文解决方案

## 问题定义

**IDE端**: 天然有项目上下文（cwd自动注入）
**Web端**: 缺失项目上下文，模型无法准确判断是否是编程任务以及目标项目

## 解决方案

### 方案1: 显式项目声明 (推荐)

**用户语法**:
```
@project:zulong_beta4 分析记忆模块的实现
@project:/path/to/project 修改配置文件
```

**实现逻辑**:
```python
# zulong/launcher/web_chat_router.py

def parse_project_context(user_message: str) -> Tuple[str, str]:
    """解析项目上下文
    
    Returns:
        (project_path, cleaned_message)
    """
    import re
    
    # 匹配 @project:name 或 @project:/path
    pattern = r'@project:([^\s]+)\s*'
    match = re.search(pattern, user_message)
    
    if match:
        project_ref = match.group(1)
        cleaned_msg = re.sub(pattern, '', user_message, count=1)
        
        # 路径解析
        if project_ref.startswith('/') or project_ref.startswith('\\'):
            # 绝对路径
            project_path = project_ref
        else:
            # 项目名称，查找已知项目
            project_path = find_project_by_name(project_ref)
        
        return project_path, cleaned_msg
    
    return None, user_message

def find_project_by_name(project_name: str) -> Optional[str]:
    """从已知项目列表查找"""
    # 从配置或历史会话中查找
    known_projects = load_known_projects()
    return known_projects.get(project_name)
```

**优点**:
- ✅ 100%准确
- ✅ 用户明确意图
- ✅ 无需复杂判断逻辑

**缺点**:
- ❌ 用户需要学习语法
- ❌ 增加输入负担

---

### 方案2: 智能上下文检测

**检测信号**:

1. **编程任务关键词**:
```python
PROGRAMMING_KEYWORDS = {
    "分析代码", "修改文件", "执行命令", "搜索文件",
    "创建文件", "删除文件", "重构", "调试",
    "read_file", "write_to_file", "execute_command",
    "实现功能", "修复bug", "添加特性", "优化性能",
    "查看代码", "理解代码", "代码结构", "模块分析",
}
```

2. **历史会话上下文**:
```python
def get_recent_project_context(user_id: str) -> Optional[str]:
    """从用户历史会话推断当前工作项目"""
    # 查询最近N次会话中涉及的项目
    recent_sessions = query_recent_sessions(user_id, limit=5)
    
    # 统计项目出现频率
    project_counts = Counter()
    for session in recent_sessions:
        if session.project_path:
            project_counts[session.project_path] += 1
    
    # 返回最频繁的项目
    if project_counts:
        return project_counts.most_common(1)[0][0]
    return None
```

3. **文件路径检测**:
```python
def detect_project_from_paths(user_message: str) -> Optional[str]:
    """从消息中提取文件路径推断项目"""
    import re
    
    # 匹配文件路径模式
    path_patterns = [
        r'([a-zA-Z]:\\[^\s]+)',  # Windows路径
        r'(/[/\w\-./]+)',         # Unix路径
    ]
    
    for pattern in path_patterns:
        matches = re.findall(pattern, user_message)
        for path in matches:
            # 向上查找项目根目录（包含.git/package.json等）
            project_root = find_project_root(path)
            if project_root:
                return project_root
    
    return None
```

**综合判断**:
```python
def infer_project_context(
    user_message: str,
    user_id: str,
    session_history: List[Session]
) -> Optional[str]:
    """综合推断项目上下文"""
    
    # 信号1: 显式声明 (权重最高)
    explicit_project, cleaned_msg = parse_project_context(user_message)
    if explicit_project:
        return explicit_project
    
    # 信号2: 编程任务关键词
    has_programming_intent = any(
        kw in user_message for kw in PROGRAMMING_KEYWORDS
    )
    
    if not has_programming_intent:
        # 不是编程任务，无需项目上下文
        return None
    
    # 信号3: 文件路径检测
    path_project = detect_project_from_paths(user_message)
    if path_project:
        return path_project
    
    # 信号4: 历史会话上下文
    history_project = get_recent_project_context(user_id)
    if history_project:
        return history_project
    
    # 无法确定，触发追问
    return None
```

**优点**:
- ✅ 对用户透明
- ✅ 多信号综合判断

**缺点**:
- ❌ 准确率<100%
- ❌ 需要复杂逻辑
- ❌ 可能误判

---

### 方案3: 工作区概念 (推荐)

**设计**:
```
Web端维护"当前工作区"概念，类似IDE workspace
```

**数据结构**:
```python
@dataclass
class UserWorkspace:
    user_id: str
    active_project: Optional[str]  # 当前激活项目
    recent_projects: List[str]      # 最近使用项目
    last_activity: datetime
```

**前端UI**:
```
┌─────────────────────────────────────┐
│ 当前项目: zulong_beta4    [切换]   │
├─────────────────────────────────────┤
│ 💬 分析记忆模块的实现              │
│    (模型知道上下文是zulong_beta4)  │
└─────────────────────────────────────┘
```

**实现**:
```python
# zulong/launcher/web_chat_router.py

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_id = request.user_id
    message = request.message
    
    # 获取用户工作区
    workspace = await get_user_workspace(user_id)
    
    # 注入项目上下文到system prompt
    if workspace.active_project:
        project_context = build_project_context(workspace.active_project)
        system_prompt = f"{project_context}\n\n{original_system_prompt}"
    
    # 调用FC循环
    response = await run_fc_loop(messages, system_prompt, ...)
    
    return response

@router.post("/workspace/set")
async def set_workspace(request: SetWorkspaceRequest):
    """设置当前工作区"""
    user_id = request.user_id
    project_path = request.project_path
    
    # 验证项目路径
    if not is_valid_project(project_path):
        return {"error": "无效项目路径"}
    
    # 更新工作区
    await update_user_workspace(user_id, active_project=project_path)
    
    # 构建项目索引（CRG）
    await build_project_index(project_path)
    
    return {"success": True, "project": project_path}

@router.get("/workspace/recent")
async def get_recent_projects(user_id: str):
    """获取最近使用项目列表"""
    workspace = await get_user_workspace(user_id)
    return {"projects": workspace.recent_projects}
```

**优点**:
- ✅ 明确的项目上下文
- ✅ 符合IDE使用习惯
- ✅ 支持快速切换
- ✅ 无需每次声明

**缺点**:
- ❌ 需要前端UI支持
- ❌ 需要工作区持久化

---

### 方案4: 混合方案 (最终方案)

**综合以上方案，分层判断**:

```
Layer 1: 显式声明 → @project:xxx
Layer 2: 工作区上下文 → 用户当前激活项目
Layer 3: 智能检测 → 关键词+路径+历史
Layer 4: 追问确认 → 无法确定时询问用户
```

**实现**:
```python
async def resolve_project_context(
    user_message: str,
    user_id: str
) -> Tuple[Optional[str], str]:
    """解析项目上下文
    
    Returns:
        (project_path, cleaned_message)
    """
    
    # Layer 1: 显式声明
    explicit_project, cleaned_msg = parse_project_context(user_message)
    if explicit_project:
        return explicit_project, cleaned_msg
    
    # Layer 2: 工作区上下文
    workspace = await get_user_workspace(user_id)
    if workspace.active_project:
        # 检查是否是编程任务
        if is_programming_task(user_message):
            return workspace.active_project, user_message
    
    # Layer 3: 智能检测
    detected_project = infer_project_context(
        user_message, user_id, workspace.recent_projects
    )
    if detected_project:
        return detected_project, user_message
    
    # Layer 4: 追问确认
    if is_likely_programming_task(user_message):
        # 触发追问工具
        return None, user_message  # FC循环中调用ask_followup_question
    
    # 不是编程任务
    return None, user_message
```

## 项目上下文注入

**获取项目上下文后，注入到system prompt**:

```python
def build_project_context(project_path: str) -> str:
    """构建项目上下文信息"""
    
    # 读取项目元信息
    project_name = os.path.basename(project_path)
    
    # 读取AGENTS.md
    agents_md = read_file_safe(f"{project_path}/AGENTS.md")
    
    # 读取package.json/pyproject.toml等
    dependencies = read_dependencies(project_path)
    
    # 读取README
    readme = read_file_safe(f"{project_path}/README.md")
    
    # 查询CRG索引
    crg_summary = query_crg_summary(project_path)
    
    return f"""
【项目上下文】
项目名称: {project_name}
项目路径: {project_path}

{agents_md if agents_md else ''}

【项目结构】
{crg_summary if crg_summary else '索引构建中...'}

【依赖】
{dependencies}
"""
```

## IDE端 vs Web端对比

| 维度 | IDE端 | Web端 |
|------|-------|-------|
| 项目上下文来源 | cwd自动注入 | 显式声明/工作区/智能检测 |
| 确定性 | 100% | 分层判断，最终追问确认 |
| 用户负担 | 无 | 可选（显式声明/工作区切换） |
| 首次使用 | 打开项目即可 | 需要设置工作区或声明 |
| 后续使用 | 无需重复 | 工作区记住，无需重复 |

## 实施建议

### 短期（快速实现）
1. 实现方案1（显式声明）- 最简单准确
2. Web端提示用户使用`@project:xxx`语法

### 中期（提升体验）
1. 实现方案3（工作区概念）
2. 前端增加项目切换UI
3. 历史会话自动关联项目

### 长期（智能化）
1. 实现方案4（混合方案）
2. 智能检测+追问确认
3. 用户习惯学习

## 总结

**核心思路**: 
- IDE端天然有上下文，无需处理
- Web端通过**显式声明 + 工作区 + 智能检测 + 追问确认**四层机制，确保100%准确判断

**推荐方案**: 
- 先实现方案1（显式声明）快速上线
- 逐步演进到方案4（混合方案）提升体验
