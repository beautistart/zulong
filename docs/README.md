# 祖龙 (Zulong) 技术文档

> 祖龙 - 多层自适应智能体认知系统
> 让 AI 拥有真正的记忆

---

## ✨ 祖龙的核心特点与优势

**祖龙是什么？** 祖龙是一个让 AI 拥有**真正生物学记忆机制**的智能体认知系统。它不是一个简单的对话工具，而是一个**能够记住你、理解你、随时间越来越懂你**的 AI 伴侣。

### 直观效果

当你使用祖龙时，你会体验到：

- **无限上下文**：祖龙通过三级记忆架构（热/温/冷），让 AI 突破模型上下文窗口限制。即使经过数月、跨年级的对话，它依然能想起你曾经说过的话、做过的事。

- **记忆关联发现**：祖龙会自动发现记忆之间的关联。就像人类的"联想"，当你提到"上次那个项目"时，它会自动联想到相关的背景、人员和细节，而不需要你重复说明。

- **跨年级的完整记忆**：不同于其他 AI 的"摘要式"记忆（只保留模糊概括），祖龙保存的是**完整的对话、任务、经验细节**。一年后你问"去年那个装修项目用的什么颜色方案？"，它能给出具体答案。

### 核心技术能力

| 能力 | 效果 | 竞品对比 |
|------|------|----------|
| 异构记忆图谱 | 9 种节点 + 7 种边，模拟人脑记忆网络 | 市场最完整实现 |
| 赫布学习 + 艾宾浩斯衰减 | 经常使用的记忆加强，不重要的自然遗忘 | 认知科学级实现 |
| FC 循环 5 层防护 | 自动检测死循环、信息缺口，确保任务完成 | 全市场独有 |
| 跨天级任务恢复 | 任务挂起后第二天完整恢复，状态不丢失 | 竞品不支持 |
| 三维标签系统 | Temperature × Importance × TimeScope | 全市场独有 |

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
