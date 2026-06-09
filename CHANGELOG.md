# 祖龙 (ZULONG) - 更新日志

所有重要的变更都将记录在此文件中。

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 计划中
- 多Agent协作支持
- 插件系统和插件市场
- 更多IDE集成（JetBrains, Vim等）
- 性能优化（关键路径Rust重写）

---

## [2.0.0] - 2026-05-27

### 重大架构升级 — 任务编排重构与交互体验革新

#### 新增 (Added)
- ✅ FC 循环合并与协议统一 — 统一 IDE 与 Web 端 FC 执行路径，消除双轨维护
- ✅ 交互式任务卡片系统 — ApprovalCard、InteractionCard、StartupCard、SummaryCard
- ✅ 审批白名单机制 — 操作权限分级审批，每步操作人工确认与实时进度跟踪
- ✅ VS Code 执行桥接 — 代码在安全环境执行，支持受控文件系统访问
- ✅ 工具袋系统 (ToolBag) — 工具智能路由与预测加载，减少工具选择延迟
- ✅ 对话编排器 (ConversationOrchestrator) — 统一的多轮对话与任务编排流程
- ✅ 记忆镜像系统 (MemoryMirror) — 会话窗口与记忆节点实时绑定
- ✅ 记忆图谱可视化面板 — BFS 动画、注意力视图、交互式记忆浏览
- ✅ 事件存储 (EventStore) — 持久化事件流，支持回放与审计
- ✅ 统一协议层 — 规范化交互载荷与跨层通信协议
- ✅ `navigate_attention` 注意力导航工具 — 替代 `focus_on_chain`
- ✅ `request_tool_supplement` 工具补充工具 — 动态扩展工具能力
- ✅ `search_experience` 经验库检索工具 — 被动历史经验检索

#### 变更 (Changed)
- ⚠️ 废弃会话意图分类 (CHAT/COMPLEX/RESUME 分类已移除)
- L1-B 不再输出意图分类标签，改为输出工具预判与上下文信号
- L2 统一主链负责推理、回复生成与工具执行决策
- FC 循环节点化重构 (fc_nodes.py + fc_runner.py)
- `focus_on_chain` → `navigate_attention`

#### 移除 (Removed)
- 移除 graph.py / state.py / types.py / session_tool.py 等旧代码
- 移除 openclaw_plugin.py / openclaw_tool.py
- 移除 l3_openclaw_skill.yaml 配置文件

---

## [1.0.0] - 2026-05-12

### 首次正式发布

#### 新增 (Added)
- ✅ 完整的四层推理架构（L0/L1/L2/L3）
- ✅ MemoryGraph 异构记忆图谱
  - 9种节点类型
  - 7种边类型
  - 赫布学习引擎
  - 艾宾浩斯衰减
  - BFS扩散激活
  - 双路径检索（热路径BFS + 冷路径FAISS）
- ✅ CircuitBreaker 死循环检测器
  - 6信号综合检测
  - 状态机（GREEN→YELLOW→RED）
  - 动态放宽模式
- ✅ TaskGraph 任务图谱
  - 无限深度递归树
  - 状态聚合
  - 依赖管理
- ✅ 5层防护链
  - CB强制收敛
  - RuleGuardian过早完成拦截
  - InfoGap信息缺口检测
  - RESUME AutoMark安全网
  - COMPLEX Backfill节点回填
- ✅ 两阶段意图分类（CHAT/COMPLEX/RESUME）
- ✅ 注意力窗口三模式（GLOBAL/FOCUS/SINGLE_CHAIN）
- ✅ 跨天级任务挂起/恢复
- ✅ TTS语音合成（Kokoro-82M）
- ✅ ASR语音识别（SenseVoice-Small）
- ✅ VS Code扩展完整前端
- ✅ WebSocket实时通信
- ✅ MCP协议支持（7个工具）

#### 核心 (Core)
- `zulong/l2/inference_engine.py` - L2推理引擎（190KB, 5700+行）
- `zulong/memory/memory_graph.py` - 异构记忆图谱（148KB, 2784行）
- `zulong/l2/circuit_breaker.py` - 死循环检测（23KB）
- `zulong/ide/ide_server.py` - WebSocket服务器（62KB）
- `zulong/ide/ide_fc_runner.py` - IDE FC循环（167KB）

#### 文档 (Documentation)
- ✅ 技术规格说明书 v3.0
- ✅ 系统深度技术分析报告
- ✅ IDE使用指南
- ✅ 配置指南
- ✅ 快速启动指南

#### 许可证 (License)
- 核心代码：AGPL-3.0（保护核心竞争力）
- 接口前端：MIT（鼓励集成使用）
- 文档：CC BY-NC-SA 4.0（允许传播，禁止商业）

---

## 版本说明

- **主版本号（Major）**: 不兼容的API变更
- **次版本号（Minor）**: 向后兼容的功能新增
- **修订号（Patch）**: 向后兼容的问题修复

---

祖龙 - 让AI拥有真正的记忆
