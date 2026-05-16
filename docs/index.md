# 祖龙 (Zulong) 文档导航

> 快速查找你需要的技术文档

---

## 📖 快速入口

| 文档 | 描述 | 适合人群 |
|------|------|----------|
| [README.md](./README.md) | 项目概述、文档导航 | 所有人 |
| [我的故事](./MY_STORY.md) | 室内设计师独立开发的故事 | 媒体/非技术人群 |
| [快速启动指南](./guides/quick-start.md) | 3步安装与启动 | 新用户 |

---

## 🏗️ 系统架构

### 整体架构

- [系统深度技术分析报告](./architecture/system-overview.md) - 完整的系统架构分析、竞品对比、技术亮点
- [技术规格说明书 v3.0](./architecture/technical-spec-v3.md) - 系统技术规格定义
- [四层架构设计](./architecture/four-layer-design.md) - L0/L1/L2/L3 分层架构说明
- [核心模块说明](./architecture/core-modules.md) - 各核心模块职责
- [L1 插件架构指南](./architecture/l1-plugin-guide.md) - L1 层插件开发指南

### 具身机器人

- [L1 层深度技术分析](./architecture/system-overview.md#39-l0l1-感知与具身控制层-深度技术分析) - L0/L1 各子层详细分析、安全架构、OpenClaw Bridge
- [L1 感知与具身控制层规格](./architecture/technical-spec-v3.md#第-95-章l1-感知与具身控制层) - TSD v3.0 中的 L1 层技术规格
- [L1 插件开发指南](./architecture/l1-plugin-guide.md) - 自定义 L1 插件开发入门
- [具身机器人 OS 竞品对标](./architecture/system-overview.md#391-具身机器人-os-市场竞品对标-v40) - 与 ROS 2/NVIDIA Isaac/AimRT 对比

---

## 🧠 记忆系统

### 核心架构

- [记忆系统架构](./memory/architecture.md) - 完整记忆系统设计
- [三级记忆检索架构](./memory/three-level-retrieval.md) - 热/温/冷三层检索机制
- [三层注意力机制](./memory/three-layer-attention.md) - 注意力机制实现
- [L1-B 经验检索注入流程](./memory/l1b-experience-flow.md) - 经验检索与注入

### 异构图记忆 (MemoryGraph)

详见 [memory_graph/](./memory_graph/) 目录：

- [架构设计](./memory_graph/01-architecture.md) - 异构图记忆核心设计
- [分类与标签](./memory_graph/02-classification-tags.md) - 节点分类与三维标签
- [检索机制](./memory_graph/03-retrieval.md) - 双路径检索 (BFS + FAISS)
- [注意力机制](./memory_graph/04-attention.md) - 赫布学习与衰减
- [任务编排](./memory_graph/05-task-orchestration.md) - 图与任务编排
- [完整指南](./memory_graph/COMPLETE_GUIDE.md) - MemoryGraph 完整文档

---

## ⚙️ 任务系统

### 核心机制

- [FC 循环与通用任务架构](./task-system/fc-loop.md) - Function Calling 循环统一架构
- [任务图谱与分卷存储](./task-system/suspend-resume.md) - 无限深度任务图谱解决方案
- [任务恢复机制](./task-system/task-recovery.md) - 确定性任务恢复机制
- [Pipeline 与 FC Loop 对比](./task-system/pipeline-vs-fc.md) - 任务编排方案对比
- [Function Calling 架构改进](./task-system/fc-architecture.md)

---

## 🎯 功能特性

- [智能标签系统](./features/smart-tagging.md) - 三维标签系统使用指南
- [经验存储](./features/experience-store.md) - 经验生成与存储指南
- [审查机制](./features/review-mechanism.md) - 记忆审查与衰减机制
- [混合搜索](./features/hybrid-search.md) - 混合检索指南
- [工具调用分析](./features/tool-call-analysis.md)

---

## 🚀 使用指南

### 快速开始

- [快速启动指南](./guides/quick-start.md) - 3步安装与启动
- [配置指南](./guides/configuration.md) - 系统配置说明
- [IDE 使用指南](./guides/ide-usage.md) - VS Code 扩展使用

### 部署运维

- [Docker 部署指南](./guides/docker-deployment.md) - Docker 容器化部署
- [监控配置](./guides/monitoring.md) - 系统监控与告警
- [CI/CD 配置](./guides/cicd.md) - 持续集成与部署

---

## 📊 技术对比

- [技术对比分析](./TECH_COMPARISON.md) - 与竞品技术对比

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
