# 系统必要文件清单

## 1. 核心源代码

### 1.1 核心层 (L0)
- `zulong/core/event_bus.py` - 事件总线
- `zulong/core/state.py` - 状态管理
- `zulong/core/state_manager.py` - 状态管理器
- `zulong/core/types.py` - 类型定义
- `zulong/core/websocket_server.py` - WebSocket 服务器

### 1.2 反射层 (L1)
- `zulong/l1a/reflex_controller.py` - 反射控制器
- `zulong/l1a/vision_processor.py` - 视觉处理器
- `zulong/l1b/scheduler_gatekeeper.py` - 调度器
- `zulong/l1b/intent_filter.py` - 意图过滤器
- `zulong/l1c/action_classifier.py` - 动作分类器

### 1.3 专家层 (L2)
- `zulong/l2/expert_invoker.py` - 专家调用器
- `zulong/l2/event_handler.py` - 事件处理器
- `zulong/l2/rag_node.py` - RAG 集成节点
- `zulong/l2/inference_engine.py` - 推理引擎
- `zulong/l2/intent_recognition_node.py` - 意图识别节点

### 1.4 双脑层 (L3)
- `zulong/l3/dual_brain_container.py` - 双脑容器
- `zulong/l3/model_switcher.py` - 模型切换器
- `zulong/l3/expert_container.py` - 专家容器

### 1.5 记忆系统
- `zulong/memory/rag_manager.py` - RAG 管理器
- `zulong/memory/embedding_manager.py` - Embedding 管理器
- `zulong/memory/short_term_memory.py` - 短期记忆
- `zulong/memory/three_libraries.py` - 三库管理

### 1.6 工具系统
- `zulong/tools/base.py` - 工具基类
- `zulong/tools/tool_engine.py` - 工具引擎
- `zulong/tools/vscode_tool.py` - VSCode 工具

### 1.7 专家技能
- `zulong/expert_skills/skill_pool.py` - 技能池
- `zulong/expert_skills/rag_skill.py` - RAG 技能
- `zulong/expert_skills/vision_skill.py` - 视觉技能
- `zulong/expert_skills/dwa_planner.py` - DWA 避障

### 1.8 模型管理
- `zulong/models/container.py` - 模型容器
- `zulong/models/engine.py` - 模型引擎
- `zulong/models/model_configs.py` - 模型配置

### 1.9 启动文件
- `zulong/bootstrap.py` - 系统引导器
- `zulong/state.py` - 全局状态

## 2. 配置文件

### 2.1 系统配置
- `config/calibration.json` - 校准配置
- `config/l1_plugins.yaml` - L1 插件配置
- `config/production.yml` - 生产配置
- `config.yaml` - 主配置文件

### 2.2 OpenClaw 配置
- `openclaw/openclaw_config.json` - OpenClaw 配置
- `openclaw/hooks/zulong/handler.js` - 祖龙 Hook

## 3. 文档文件

### 3.1 架构文档
- `README.md` - 项目说明
- `ARCHITECTURE_SOLUTION_ANALYSIS.md` - 架构分析
- `docs/ARCHITECTURE_UPGRADE_SUMMARY.md` - 架构升级摘要
- `docs/INTEGRATION_QUICKSTART.md` - 集成快速开始

### 3.2 OpenClaw 集成
- `docs/openclaw_integration.md` - OpenClaw 集成指南
- `docs/OPENCLAW_CONNECTION_GUIDE.md` - OpenClaw 连接指南

### 3.3 技术规格
- `docs/TSD_v1.7.txt` - 技术规格说明书
- `docs/TSD_v1.8_CHANGELOG.md` - 变更日志

## 4. OpenClaw Bridge

### 4.1 核心文件
- `openclaw_bridge/bootstrap.py` - Bridge 启动脚本
- `openclaw_bridge/event_bus_client.py` - EventBus 客户端
- `openclaw_bridge/types.py` - 类型定义

### 4.2 Web 适配器
- `openclaw_bridge/web/static/index.html` - Web 界面
- `openclaw_bridge/requirements_web.txt` - Web 依赖

## 5. 依赖文件
- `requirements.txt` - Python 依赖
- `docker-compose.yml` - Docker 配置
- `Dockerfile` - Docker 构建文件

## 6. 排除文件

以下文件不需要包含在代码审查中：

### 6.1 脚本文件
- `scripts/` 目录下的所有文件

### 6.2 测试文件
- `tests/` 目录下的所有文件

### 6.3 模型文件
- `models/` 目录下的所有模型文件

### 6.4 虚拟环境
- `zulong_env/` 目录

### 6.5 备份文件
- `backups/` 目录
- `safetensorsvers/` 目录

### 6.6 资料文件
- `资料/` 目录

## 7. 总结

以上文件覆盖了祖龙系统的核心功能，包括：
- 系统架构和核心组件
- OpenClaw 集成功能
- 记忆系统和工具系统
- 模型管理和专家技能
- 配置和文档

这些文件足以进行代码审查和系统理解，不包含不必要的大型文件或临时文件。