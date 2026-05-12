# 祖龙 (Zulong) 技术文档

> 祖龙 - 多层自适应智能体认知系统
> 让 AI 拥有真正的记忆

---

## 📖 文档导航

### 系统架构

- [系统深度技术分析报告](architecture/system-overview.md) - 完整架构分析、竞品对比
- [技术规格说明书 v3.0](architecture/technical-spec-v3.md) - 系统技术规格定义
- [四层架构设计](architecture/four-layer-design.md) - L0/L1/L2/L3 分层架构
- [核心模块说明](architecture/core-modules.md) - 各核心模块职责
- [L1 插件架构指南](architecture/l1-plugin-guide.md) - L1 层插件开发

### 记忆系统

- [记忆系统架构](memory/architecture.md) - 完整记忆系统设计
- [三级记忆检索架构](memory/three-level-retrieval.md) - 热/温/冷三层检索
- [异构图记忆图谱](memory_graph/) - MemoryGraph 设计与实现
  - [架构设计](memory_graph/01-architecture.md)
  - [分类与标签](memory_graph/02-classification-tags.md)
  - [检索机制](memory_graph/03-retrieval.md)
  - [注意力机制](memory_graph/04-attention.md)
  - [任务编排](memory_graph/05-task-orchestration.md)
  - [完整指南](memory_graph/COMPLETE_GUIDE.md)
- [三层注意力机制](memory/three-layer-attention.md)
- [L1-B 经验检索注入流程](memory/l1b-experience-flow.md)

### 任务系统

- [FC 循环与通用任务架构](task-system/fc-loop.md) - Function Calling 循环统一架构
- [任务图谱与分卷存储](task-system/suspend-resume.md) - 无限深度任务图谱
- [任务恢复机制](task-system/task-recovery.md) - 确定性任务恢复
- [Pipeline 与 FC Loop 对比](task-system/pipeline-vs-fc.md) - 任务编排方案对比
- [Function Calling 架构改进](task-system/fc-architecture.md)

### 功能特性

- [智能标签系统](features/smart-tagging.md) - 三维标签系统
- [经验存储](features/experience-store.md) - 经验生成与存储
- [审查机制](features/review-mechanism.md) - 记忆审查与衰减
- [混合搜索](features/hybrid-search.md) - 混合检索指南
- [工具调用分析](features/tool-call-analysis.md)

### 使用指南

- [快速启动指南](guides/quick-start.md) - 3 步安装与启动
- [配置指南](guides/configuration.md) - 系统配置说明
- [Docker 部署指南](guides/docker-deployment.md)
- [CI/CD 配置](guides/cicd.md)
- [监控配置](guides/monitoring.md)
- [IDE 使用指南](guides/ide-usage.md) - VS Code 扩展使用

### 其他

- [我的故事](MY_STORY.md) - 室内设计师的 AI 之旅
- [技术对比](TECH_COMPARISON.md) - 与竞品技术对比

---

## 💡 如何贡献文档

1. Fork 本仓库
2. 创建分支 `docs/your-doc-name`
3. 编写文档（遵循项目格式规范）
4. 提交 PR 并描述变更

---

## 📄 许可证

本目录下的所有文档采用 **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License)

您可以自由分享和传播这些文档，但必须：
- 署名
- 非商业性使用
- 相同方式分享

---

<p align="center">
  <strong>祖龙 - 让 AI 拥有真正的记忆</strong><br>
  <em>Built with ❤️ by an Interior Designer</em>
</p>
