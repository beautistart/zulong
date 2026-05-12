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
