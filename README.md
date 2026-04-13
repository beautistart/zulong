<div align="center">

# ZULONG - Robot Operating System

**祖龙 -- 机器人操作系统**

![Version](https://img.shields.io/badge/version-beta4-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![License](https://img.shields.io/badge/license-AGPL--3.0-red.svg)
![Lines](https://img.shields.io/badge/code-82K%2B%20lines-blueviolet.svg)

事件驱动 | 分层架构 | 插件生态 | RAG 记忆 | 消费级硬件

[English](#architecture) | [架构](#架构) | [快速开始](#快速开始) | [插件开发](#插件开发) | [许可证](#许可证)

</div>

---

## 这个项目是什么

ZULONG（祖龙）是一个完整的机器人操作系统。不是一个 demo，不是一个概念验证，是一个跑在真实硬件上的、经过实测的操作系统 -- 82,000 行 Python，230 个模块，四层架构，从传感器输入到语音输出的完整链路。

L2 单实例可在一张 RTX 3060 6GB 显卡上运行，双实例需要 8GB 以上显存。

## 谁做的

一个室内设计师。不会编程。

没有团队，没有计算机科学背景，没有任何软件工程训练。我的工具是通义千问桌面版、Qoder 和 Trae -- 三个 AI 编程助手。我告诉它们我想要什么，它们帮我写代码，我测试、调整、再测试。

从第一行代码到现在的 82K 行，每一行都是这样"搓"出来的。

我不觉得这是什么值得炫耀的事。我只是想说明一个事实：**AI 正在改变软件的生产方式。** 领域专家可以直接把自己的想法变成可运行的系统，不需要先花几年学编程。设计师可以写操作系统，机械工程师可以写控制算法，医生可以写诊断工具。

门槛消失了。剩下的只有想法。

## 为什么开源

1. **时间戳** -- 公开发布建立优先权
2. **生态** -- 一个人做不了所有传感器和硬件的适配，需要插件开发者
3. **信任** -- 开源代码可审计，比闭源承诺更有说服力
4. **商业化** -- Open Core 模式：社区版免费，企业版收费

---

## 架构

```
+-----------------------------------------------------------+
|                L3: 专家技能层                                |
|   专家节点 | 模型切换 | TTS 语音合成                         |
+-----------------------------------------------------------+
                          ^
                          v
+-----------------------------------------------------------+
|            L2: 专家推理层 (LangGraph)                       |
|   专家调度 | 事件处理 | RAG 检索增强                         |
+-----------------------------------------------------------+
                          ^
                          v
+-----------------------------------------------------------+
|                 L1: 反射层                                  |
|   L1-A 反射引擎 | L1-B 调度器 | L1-C 视觉注意力              |
+-----------------------------------------------------------+
                          ^
                          v
+-----------------------------------------------------------+
|                 L0: 核心服务层                               |
|   EventBus 事件总线 | 状态管理 | 电源管理                     |
+-----------------------------------------------------------+
```

**设计理念：**

- **L0** -- 硬件抽象和事件总线，所有模块通过事件通信，零轮询
- **L1-A** -- 反射层，处理紧急事件（有人靠近、异常声音），毫秒级响应
- **L1-B** -- 调度器，根据电源状态和负载动态分配算力
- **L1-C** -- 视觉注意力，安静时持续观察环境变化
- **L2** -- 专家推理，需要深度思考时调用大模型 + RAG 检索记忆
- **L3** -- 技能执行，模型推理、语音合成、工具调用

**数据流：**
```
传感器输入 -> L1-A 反射检查 -> L1-B 调度 -> L2 专家处理 -> L3 模型推理
                  |                |              |
              紧急事件?         负载管理?       需要记忆?
                  |                |              |
            直接响应          动态调度        RAG 检索
```

### 目录结构

```
zulong/
├── core/               # L0: EventBus 事件总线、类型定义、状态管理
├── l0/                 # 硬件抽象层（摄像头、音频、执行器）
├── l1a/                # L1-A: 反射引擎（视觉、音频预处理）
├── l1b/                # L1-B: 调度器、注意力控制
├── l1c/                # L1-C: 安静视觉注意力
├── l2/                 # L2: 专家调度、RAG 节点、事件处理
├── l3/                 # L3: 专家节点、TTS、模型切换
├── memory/             # 记忆系统：RAG 管理、向量嵌入、混合搜索
├── models/             # 模型配置与加载
├── modules/l1/core/    # 插件接口 (IL1Module) & 插件管理器
├── plugins/            # 内置插件（视觉、语音、电机、气体检测）
├── tools/              # 工具引擎、网络搜索、调试控制台
├── skill_packs/        # 技能包系统（加载、运行、路由）
├── emotion/            # 情感检测（文本、语音）
├── mcp/                # MCP 客户端 & 服务端
└── utils/              # 日志、指标、监控
```

---

## 功能列表

### 社区版 vs 企业版

本仓库是 **社区版**，基于 AGPL-3.0 开源。

| 功能 | 社区版 | 企业版 |
|------|:------:|:------:|
| EventBus 事件总线 & 插件框架 | Y | Y |
| 基础 RAG & 向量嵌入管道 | Y | Y |
| L1-A 反射引擎 | Y | Y |
| L2 专家调度 & RAG 节点 | Y | Y |
| L3 专家技能池 | Y | Y |
| TTS 语音合成 (CosyVoice) | Y | Y |
| 插件开发接口 (MIT 许可) | Y | Y |
| MCP 客户端/服务端 | Y | Y |
| 记忆进化 & 知识图谱 | - | Y |
| 双脑 KV Cache 热切换 | - | Y |
| 推理引擎 | - | Y |
| 短期记忆 & 人物画像 | - | Y |
| 优先技术支持 & SLA | - | Y |

详见 [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md)。

---

## 快速开始

### 硬件要求

| 场景 | GPU | CPU | 内存 | 存储 |
|------|-----|-----|------|------|
| **L2 单实例**（基础运行） | NVIDIA RTX 3060 6GB | Intel i5-10400 | 16GB DDR4 | 50GB SSD |
| **L2 双实例**（双脑并行） | NVIDIA 8GB+ 显存 | Intel i7-12700 | 32GB DDR4 | 100GB SSD |
| **推荐配置** | AMD AI MAX 395 / NVLink 多卡 | - | 64GB+ | 1TB NVMe SSD |

- **Python**: 3.10+（推荐 3.11+）
- L2 单实例：一个大模型实例处理推理，适合开发和基础部署
- L2 双实例：两个大模型实例并行（如 KV Cache 热交换），需要至少 8GB 显存
- 推荐 AMD AI MAX 395 统一内存架构主机，或 NVLink 多卡主机用于生产环境

### 安装

```bash
# 克隆仓库
git clone https://github.com/beautistart/zulong.git
cd zulong

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或: venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 复制并编辑配置
cp config.yaml.example config.yaml
cp .env.example .env
# 编辑 config.yaml 填入模型路径
```

### 模型下载

ZULONG 使用量化模型，6GB 显存即可运行：

| 模型 | 大小 | 用途 | 运行设备 |
|------|------|------|---------|
| Qwen3.5-0.8B-INT4 | ~500MB | 反射 & 调度 | GPU |
| Qwen3.5-2B | ~1.5GB | 专家推理 | GPU |
| CosyVoice3-0.5B | ~1GB | 语音合成 | CPU |
| bge-small-zh-v1.5 | ~100MB | 向量嵌入 | CPU |

将模型放在 `models/` 目录下，路径在 `config.yaml` 中配置。

### 运行

```bash
# 启动系统
python zulong/bootstrap.py

# 带 Web 界面启动
python openclaw_bridge/bootstrap.py
```

---

## 插件开发

ZULONG 的插件系统允许你在不修改框架代码的情况下扩展功能。使用插件接口开发的插件 **不受 AGPL 约束**（见 LICENSE 中的插件例外条款）。

### 插件接口

所有插件实现 `IL1Module` 抽象基类：

```python
from zulong.modules.l1.core.interface import IL1Module, EventPriority
from zulong.core.types import EventType

class MyPlugin(IL1Module):
    @property
    def module_id(self) -> str:
        return "my_custom_plugin"

    @property
    def priority(self) -> EventPriority:
        return EventPriority.NORMAL

    async def on_event(self, event) -> None:
        if event.event_type == EventType.USER_SPEECH:
            # 处理用户语音事件
            pass

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass
```

### 插件许可

使用插件接口（IL1Module、EventType、ZulongEvent）开发的插件可以使用 **任何许可证**，包括闭源商业许可。详见 [LICENSE](LICENSE) 中的插件例外条款。

---

## 配置

复制示例文件并根据环境修改：

```bash
cp config.yaml.example config.yaml       # 系统配置
cp docker-compose.yml.example docker-compose.yml  # Docker 部署
cp .env.example .env                      # 环境变量
```

---

## 开发状态

### 实机测试通过

- 核心 EventBus & 插件框架
- 短期记忆 / 长期记忆 / 跨会话记忆持久化
- 双实例 KV Cache 热交换
- 对话/任务中断与恢复 -- 毫秒级打断，插入新对话/新任务后可恢复旧任务，不丢失上下文，多任务并行计算
- 语音回复 (CosyVoice TTS)
- L1-A 反射引擎

### 部分通过 / 待进一步测试

- 复盘机制 -- 代码完成，部分实机测试未通过，持续调试中
- 增强记忆库 -- 代码完成，待进一步实机测试
- 技能包系统 -- 代码完成，待进一步实机测试

---

## 许可证

### 框架代码 (AGPL-3.0)

本项目基于 **GNU Affero General Public License v3.0** 开源，附带插件例外条款。详见 [LICENSE](LICENSE)。

### 插件接口 (MIT)

插件接口文件（`zulong/modules/l1/core/interface.py`、`zulong/core/types.py`）按 MIT 许可设计。仅使用这些接口的插件 **不是** ZULONG 的衍生作品，不受 AGPL 约束。

### 商业许可

如需闭源部署、SaaS 服务、或企业版模块，请联系获取商业许可。详见 [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md)。

---

## 贡献

欢迎贡献代码和插件。

1. 框架代码贡献遵循 AGPL-3.0
2. 插件开发可用任何许可证
3. 提交 PR 即表示同意以 AGPL-3.0 许可你的贡献

---

## 联系

- **Issues**: [GitHub Issues](https://github.com/beautistart/zulong/issues)
- **Email**: q1550567472@qq.com
- **商业合作**: 见 [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md)

---

<div align="center">

*一个设计师，三个 AI，82000 行代码，一个机器人操作系统。*

</div>
