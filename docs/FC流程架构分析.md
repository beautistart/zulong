# FC流程架构分析

## 当前三套FC流程

### 1. IDEFCRunner (3748行)
**使用场景**: IDE WebSocket长连接
**工具执行**:
- 内部工具: 服务端执行 (task/memory/attention)
- 远程工具: 客户端执行 (read_file/write_to_file/execute_command)

**核心特性**:
- WebSocket长连接推送 (display_text/tool_request/task_complete)
- 工具分流机制 (内部/远程)
- 跨HTTP请求暂停/恢复
- 客户端环境感知 (Git Bash/PowerShell/CMD)
- 权限确认机制 (自动批准)

**调用位置**:
```python
# zulong/ide/ide_server.py:567
from zulong.ide.ide_fc_runner import IDEFCRunner
```

### 2. UnifiedFCRunner (221行)
**使用场景**: Web端/其他服务端场景
**工具执行**: 全部服务端执行

**核心特性**:
- 复用fc_graph.py节点工厂函数
- while循环驱动 (无LangGraph依赖)
- 完整认知能力 (AttentionWindow/CircuitBreaker/RuleGuardian)

**调用位置**:
```python
# zulong/l2/inference_engine.py:29
from zulong.l2.unified_fc_runner import run_fc_loop
```

### 3. fc_graph.py (1544行) [已废弃]
**状态**: LangGraph StateGraph入口已废弃
**保留**: 节点工厂函数被复用
- _make_check_node
- _make_call_model_node
- _make_exec_tools_node
- _make_eval_response_node

## 核心差异分析

### IDE模式独特需求

**远程工具定义**:
```python
IDE_REMOTE_TOOLS = {
    "read_file",           # 读取客户端本地文件
    "write_to_file",       # 写入客户端本地文件
    "replace_in_file",     # 替换文件内容
    "delete_file",         # 删除文件
    "execute_command",     # 在客户端环境执行命令
    "search_files",        # 搜索客户端文件系统
    "list_files",          # 列出文件
    "list_code_definition_names",  # 列出代码定义
    "browser_action",      # 浏览器操作
    "ask_followup_question",  # 追问用户
    "attempt_completion",  # 尝试完成
}
```

**差异原因**:
1. **文件路径**: `d:\AI\project\zulong_beta4\...` 是客户端路径
2. **终端环境**: Git Bash/PowerShell/CMD在客户端运行
3. **权限控制**: IDE需要用户确认（自动批准机制）
4. **实时推送**: WebSocket长连接推送display_text/tool_request

### Web模式特点

- 所有工具服务端执行
- 无需客户端环境感知
- 无权限确认机制
- HTTP请求/响应模式

## 统一方案

### 目标架构

```
Web端发送任务
  ↓
祖龙后端接收并分析
  ↓
任务类型判断:
  ├─ 普通对话 → UnifiedFCRunner (服务端执行)
  └─ 编程任务 → UnifiedFCRunner + 客户端工具代理
       ├─ 自动打开IDE
       ├─ WebSocket推送tool_request
       ├─ IDE客户端执行read_file/write_to_file/execute_command
       └─ 返回结果继续FC循环
  ↓
任务完成
```

### 重构建议

**统一核心循环**:
```python
class UnifiedFCRunner:
    def __init__(self, engine, execution_mode="server"):
        self.execution_mode = execution_mode  # "server" | "client"
        
    async def run_loop_async(self, ...):
        # 统一的FC循环逻辑
        while True:
            tc_data, resp = await self._call_model(...)
            
            if tc_data:
                # 工具分流执行
                internal, remote = self._classify_tools(tc_data)
                
                # 内部工具: 服务端执行
                if internal:
                    await self._exec_tools_internal(internal)
                
                # 远程工具: 根据模式选择执行位置
                if remote:
                    if self.execution_mode == "client":
                        await self._exec_tools_remote(remote)  # 推送到客户端
                    else:
                        await self._exec_tools_local(remote)   # 服务端执行
```

**需要保留的差异**:
- 工具执行器 (服务端 vs 客户端)
- WebSocket推送机制
- 权限确认机制
- 客户端环境感知

## 重构收益

1. ✅ **减少代码重复**: 核心逻辑统一，差异仅在于工具执行位置
2. ✅ **保留必要差异**: 工具执行、权限控制、环境感知
3. ✅ **统一入口**: Web端可以发送所有任务类型
4. ✅ **自动IDE**: 祖龙可以自动打开IDE执行编程任务
5. ✅ **维护性提升**: 核心逻辑只需维护一处

## 实施步骤

### Phase 1: 分析当前差异
- [x] 梳理三套FC流程
- [x] 识别核心差异点
- [x] 设计统一架构

### Phase 2: 提取公共逻辑
- [ ] 提取FC核心循环到基类
- [ ] 工具执行器抽象为接口
- [ ] 状态管理统一

### Phase 3: 重构IDE模式
- [ ] IDEFCRunner继承统一基类
- [ ] 实现客户端工具执行器
- [ ] 保留WebSocket推送机制

### Phase 4: 测试验证
- [ ] Web端普通对话测试
- [ ] Web端编程任务测试 (自动打开IDE)
- [ ] IDE端编程任务测试
- [ ] 混合场景测试

## 注意事项

1. **向后兼容**: 保持现有API接口不变
2. **渐进迁移**: 分阶段重构，每阶段可独立测试
3. **性能影响**: 确保统一后性能不下降
4. **错误处理**: 统一异常处理机制
