# 祖龙 (Zulong) 文档导航

> 快速查找你需要的文档

---

## 📖 快速入口 | Quick Links

| 文档 | 描述 | 适合人群 |
|------|------|----------|
| [README.md](../README.md) | 项目概述、快速开始、核心特性 | 所有人 |
| [README (English)](./README_EN.md) | English version of README | 英文用户 |
| [我的故事](./MY_STORY.md) | 室内设计师独立开发的故事 | 媒体/非技术人群 |
| [快速启动指南](./快速启动指南.md) | 3步安装与启动 | 新用户 |

---

## 🏗️ 技术架构文档 | Architecture Docs

### 系统整体架构

- [系统深度技术分析报告](./祖龙系统深度技术分析报告.md) - 完整的系统架构分析、竞品对比、技术亮点
- [技术规格说明书 v3.0](./TSD_v3.0.md) - 系统技术规格定义
- [四层架构设计](./ARCHITECTURE_UPGRADE_SUMMARY.md) - L0/L1/L2/L3 分层架构说明

### 核心模块

- [异构图记忆系统详解](./MemoryGraph_Architecture.md) - MemoryGraph 设计与实现
- [熔断器设计文档](./CircuitBreaker_Design.md) - 6信号死循环检测机制
- [注意力窗口机制](./AttentionWindow_Mechanism.md) - 3模式注意力控制
- [任务图谱设计](./TaskGraph_Design.md) - 无限深度递归树
- [核心模块说明](./CORE_MANAGER_MODULES.md) - 各核心模块职责

### L1 层架构

- [L1 插件架构指南](./L1_PLUGIN_ARCHITECTURE_GUIDE.md) - L1 层插件开发
- [L1-B 经验检索注入流程](./L1B_EXPERIENCE_RETRIEVAL_INJECTION_FLOW.md) - 经验检索与注入
- [三层注意力机制实现](./三层注意力机制实现总结.md) - 注意力机制详解

---

## 🚀 使用指南 | User Guides

### 快速开始

- [快速启动指南](./快速启动指南.md) - 3步安装与启动
- [配置指南](./Configuration_Guide.md) - 系统配置说明
- [IDE 使用指南](./Zulong_IDE使用指南.md) - VS Code 扩展使用

### 功能使用

- [智能标签系统](./SMART_TAGGING_SYSTEM.md) - 三维标签系统使用
- [记忆存储系统](./STORAGE_QUICKSTART.md) - 记忆存储快速上手
- [集成经验存储](./INTEGRATED_EXPERIENCE_STORE.md) - 经验存储指南
- [审查机制](./REVIEW_MECHANISM.md) - 记忆审查与衰减

### 部署运维

- [Docker 部署指南](./DOCKER_DEPLOYMENT.md) - Docker 容器化部署
- [监控配置](./MONITORING_SETUP.md) - 系统监控与告警
- [CI/CD 配置](./CICD_SETUP.md) - 持续集成与部署

---

## 🔧 开发文档 | Development Docs

### 贡献指南

- [贡献指南](../CONTRIBUTING.md) - 如何贡献代码
- [更新日志](../CHANGELOG.md) - 版本更新记录
- [AGENTS.md](../AGENTS.md) - Qoder 项目上下文

### 开发阶段记录

- [Phase 1 完成总结](./PHASE1_COMPLETION_SUMMARY.md)
- [Phase 2 完成总结](./PHASE2_COMPLETION_SUMMARY.md)
- [Phase 2.5 完成总结](./PHASE2_5_COMPLETION_SUMMARY.md)
- [Phase 3 完成报告](./PHASE3_COMPLETION_REPORT.md)
- [Phase 6 完成报告](./PHASE6_COMPLETE_REPORT.md)
- [Phase 7 完成报告](./PHASE7_COMPLETE_REPORT.md)
- [Phase 8 完成报告](./PHASE8_COMPLETE_REPORT.md)

### 技术实现细节

- [增强经验存储架构](./ENHANCED_EXPERIENCE_STORE_ARCHITECTURE.md)
- [混合搜索指南](./HYBRID_SEARCH_GUIDE.md)
- [Embedding 使用指南](./EMBEDDING_GUIDE.md)
- [工具引擎评估](./TOOL_ENGINE_EVALUATION.md)
- [自学习分析](./SELF_LEARNING_ANALYSIS.md)
- [Phase 7 API 参考](./PHASE7_API_REFERENCE.md)

---

## 📊 测试与验证 | Testing & Validation

### 测试报告

- [功能检查报告](./archive/FUNCTION_CHECK_REPORT.md)
- [真实环境测试报告](./archive/REAL_ROBOT_TEST_REPORT.md)
- [API 响应格式报告](./archive/API_RESPONSE_FORMAT_REPORT.md)
- [RAG 存储验证报告](./archive/RAG_STORAGE_VERIFICATION_REPORT.md)

### 测试指南

- [视觉系统测试指南](./视觉系统测试指南.md)
- [三层注意力测试](./REAL_SENSORS_ATTENTION_TEST_GUIDE.md)
- [光流法测试指南](./光流法测试指南.md)
- [阈值优化指南](./阈值优化指南.md)

---

## 🎯 专题文档 | Topic Docs

### 语音交互

- [语音策略实现](./archive/voice_strategy_implementation.md)
- [文本清洗修复报告](./archive/TEXT_CLEANING_FIX_REPORT.md)
- [模型截断分析](./archive/MODEL_TRUNCATION_ANALYSIS.md)

### 视觉系统

- [视觉检测逻辑梳理](./视觉检测逻辑梳理.md)
- [视觉系统测试报告](./archive/视觉系统测试报告.md)
- [视觉首帧修复](./archive/vision_first_frame_fix.md)

### 系统优化

- [L2 输出优化](./archive/L2_OUTPUT_OPTIMIZATION.md)
- [上下文污染修复](./archive/CONTEXT_POLLUTION_FIX_REPORT.md)
- [保存逻辑冲突修复](./archive/保存逻辑冲突修复报告.md)

---

## 📚 外部资源 | External Resources

### 相关项目

- [Cline](https://github.com/cline/cline) - VS Code Agent 基础框架
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent 编排框架
- [Kokoro TTS](https://github.com/hexgrad/kokoro) - 82M 参数 TTS 模型
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) - 多语言语音识别

### 技术社区

- [GitHub Issues](https://github.com/beautystart/zulong/issues) - 报告问题
- [GitHub Discussions](https://github.com/beautystart/zulong/discussions) - 功能建议
- [Discord](https://discord.gg/zulong) - 加入社区 (建设中)

---

## 🔍 文档查找索引 | Full Index

<details>
<summary><strong>点击展开完整文档列表</strong></summary>

### 根目录文档
- `README.md` - 项目概述
- `CHANGELOG.md` - 更新日志
- `CONTRIBUTING.md` - 贡献指南
- `AGENTS.md` - Qoder 上下文
- `LICENSE` - 许可证
- `COMMERCIAL_LICENSE.md` - 商业许可

### 技术报告 (docs/)
- `祖龙系统深度技术分析报告.md`
- `TSD_v3.0.md`
- `ARCHITECTURE_UPGRADE_SUMMARY.md`
- `CORE_MANAGER_MODULES.md`
- `L1_PLUGIN_ARCHITECTURE_GUIDE.md`
- `L1B_EXPERIENCE_RETRIEVAL_INJECTION_FLOW.md`
- `SMART_TAGGING_SYSTEM.md`
- `REVIEW_MECHANISM.md`
- `INTEGRATED_EXPERIENCE_STORE.md`
- `ENHANCED_EXPERIENCE_STORE_ARCHITECTURE.md`
- `HYBRID_SEARCH_GUIDE.md`
- `EMBEDDING_GUIDE.md`
- `TOOL_ENGINE_EVALUATION.md`
- `SELF_LEARNING_ANALYSIS.md`
- `SELF_LEARNING_LOGIC_CLARIFICATION.md`
- `PHASE7_API_REFERENCE.md`

### 使用指南 (docs/)
- `快速启动指南.md`
- `Zulong_IDE使用指南.md`
- `DOCKER_DEPLOYMENT.md`
- `MONITORING_SETUP.md`
- `CICD_SETUP.md`
- `STORAGE_QUICKSTART.md`

### 开发阶段 (docs/)
- `PHASE1_COMPLETION_SUMMARY.md`
- `PHASE2_COMPLETION_SUMMARY.md`
- `PHASE2_5_COMPLETION_SUMMARY.md`
- `PHASE3_COMPLETION_REPORT.md`
- `PHASE6_COMPLETE_REPORT.md`
- `PHASE7_COMPLETE_REPORT.md`
- `PHASE8_COMPLETE_REPORT.md`

### 测试文档 (docs/)
- `视觉系统测试指南.md`
- `REAL_SENSORS_ATTENTION_TEST_GUIDE.md`
- `光流法测试指南.md`
- `阈值优化指南.md`
- `阈值配置说明_v3.md`

### 归档文档 (docs/archive/)
所有历史报告和修复记录

</details>

---

## 💡 如何贡献文档

1. Fork 本仓库
2. 创建分支 `docs/your-doc-name`
3. 编写文档（遵循项目格式规范）
4. 提交 PR 并描述变更

详见 [贡献指南](../CONTRIBUTING.md)

---

<p align="center">
  <strong>祖龙 - 让 AI 拥有真正的记忆</strong><br>
  <em>Built with ❤️ by an Interior Designer</em>
</p>
