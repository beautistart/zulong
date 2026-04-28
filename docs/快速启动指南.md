# 祖龙 (ZULONG) 系统 - 快速启动指南

**版本**: v1.7  
**更新日期**: 2026-03-21  
**适用环境**: Windows + NVIDIA GPU (RTX 3060 6GB 或更高)

---

## 🚀 快速启动

### 1. 环境准备

#### 1.1 安装依赖
```bash
cd d:\AI\project\zulong_beta4
pip install -r requirements.txt
```

#### 1.2 检查硬件
确保以下硬件可用:
- ✅ Windows 系统
- ✅ NVIDIA GPU (RTX 3060 6GB 或更高)
- ✅ 麦克风设备
- ✅ 扬声器/耳机设备

#### 1.3 检查模型
确保已下载以下模型:
```bash
# 检查模型文件
python scripts/verify_new_models.py
```

所需模型:
- `models/Qwen3.5-0.8B-Base` (L1-A VL 模型)
- `models/Qwen3.5-0.8B` (L1-B 音频理解)
- `models/CosyVoice3-0.5B` (TTS 专家)

---

## 🎯 启动模式

### 模式 1: 真实环境启动 (推荐)

**用途**: 使用真实硬件 (麦克风、扬声器) 进行交互

```bash
# 快速启动 (默认模式)
python -m scripts.run_real_test

# 或者使用 bootstrap.py
python -m zulong.bootstrap
```

**启动后可用功能**:
- 🎤 语音输入 (通过麦克风)
- 🔊 语音输出 (通过扬声器)
- 💬 文本对话 (通过调试控制台)
- 🎭 事件驱动 (自动处理传感器事件)

---

### 模式 2: 带传感器模拟启动

**用途**: 测试/演示模式，模拟传感器数据

```bash
python -m scripts.run_real_test --mock-sensors
```

**模拟的传感器**:
- 📡 障碍传感器 (30% 概率触发)
- 🏃 运动传感器 (20% 概率触发)
- 🔊 声音传感器 (40% 概率触发)
- 🤕 摔倒传感器 (5% 概率触发，CRITICAL 优先级)

---

### 模式 3: 自动测试模式

**用途**: 自动发送测试消息，验证系统功能

```bash
python -m scripts.run_real_test --auto-test
```

**自动测试场景**:
- 每 10 秒发送一次测试消息
- 测试事件路由和响应
- 验证系统稳定性

---

### 模式 4: 无交互模式

**用途**: 后台运行，适合集成到其他系统

```bash
python -m scripts.run_real_test --no-interactive
```

**特点**:
- 不启动调试控制台
- 系统后台运行
- 按 Ctrl+C 退出

---

## 💡 调试控制台使用指南

### 启动调试控制台

```bash
python -m scripts.run_real_test
```

### 可用命令

| 命令 | 功能 | 示例 |
|------|------|------|
| `/help` | 显示帮助信息 | `/help` |
| `/status` | 查看系统状态 | `/status` |
| `/freeze` | 手动冻结当前任务 | `/freeze` |
| `/resume <id>` | 恢复指定任务 | `/resume task_123` |
| `/inject <type> <payload>` | 手动注入事件 | `/inject USER_SPEECH {"text": "你好"}` |
| `/clear` | 清屏 | `/clear` |
| `/quit` | 退出系统 | `/quit` |

### 对话示例

```
==================================================
  🐉 ZULONG Debug Console
==================================================
Type your message or '/help' for commands.

[You] 你好祖龙
[System] 正在处理...
[L2] 你好！有什么可以帮你的吗？

[You] 现在几点了？
[System] 正在处理...
[L2] 现在是下午 3 点 30 分。

[You] /status
[System] 当前状态：ACTIVE, L2Status: IDLE
```

---

## 🧪 运行测试

### 完整集成测试

```bash
# 运行真实环境测试
python tests/test_real_environment_av.py
```

**测试内容**:
1. ✅ 麦克风功能测试
2. ✅ 扬声器功能测试
3. ✅ VL 模型音频理解测试
4. ✅ 音频理解模型测试
5. ✅ 端到端语音对话测试

### 单元测试

```bash
# 运行架构验证测试
python tests/test_architecture_validation.py

# 运行事件路由测试
python tests/test_event_routing.py

# 运行 L1 反射测试
python tests/test_l1_reflex.py

# 运行 L2 中断测试
python tests/test_l2_interrupt_resume.py
```

---

## 📊 系统监控

### 查看日志

系统日志会输出到:
- **控制台**: 实时日志
- **文件**: `zulong_real_test.log`

### 性能监控

系统会自动监控以下指标:
- ⚡ 事件处理延迟
- 💾 显存占用
- 🔄 任务执行时间
- 📈 系统吞吐量

### 查看系统状态

```bash
# 在调试控制台中
/status
```

**输出示例**:
```
系统状态:
- PowerState: ACTIVE
- L2Status: IDLE
- 活跃任务：无
- 任务栈深度：0
```

---

## 🛠️ 故障排查

### 问题 1: 模型加载失败

**现象**: `ModuleNotFoundError: No module named 'torch'`

**解决方案**:
```bash
pip install -r requirements.txt
```

---

### 问题 2: 麦克风/扬声器未找到

**现象**: `未找到可用的麦克风设备`

**解决方案**:
1. 检查硬件连接
2. 检查 Windows 音频驱动
3. 运行以下命令查看可用设备:
```bash
python -c "from zulong.l0.devices.microphone_device import MicrophoneDevice; mic = MicrophoneDevice(); print(mic.list_devices())"
```

---

### 问题 3: 显存不足

**现象**: `CUDA out of memory`

**解决方案**:
1. 关闭其他占用显存的程序
2. 使用 4bit 量化模型 (已默认启用)
3. 减少同时加载的模型数量

---

### 问题 4: 系统启动后无响应

**现象**: 系统启动成功，但输入无响应

**解决方案**:
1. 检查 EventBus 是否正常初始化
2. 查看日志中的事件路由信息
3. 尝试重启系统:
```bash
# 停止系统 (Ctrl+C)
# 重新启动
python -m scripts.run_real_test
```

---

## 📚 参考文档

### 核心文档

- [祖龙 (ZULONG) 机器人系统技术规格说明书 (TSD) v1.7](../资料/祖龙%20(ZULONG)%20机器人系统技术规格说明书%20(TSD)1.7.txt)
- [多模态感知层实现任务规划_v4.md](../资料/多模态感知层实现任务规划_v4.md)
- [L3 专家层实现总结.md](../资料/L3 专家层实现总结.md)

### 测试报告

- [真实环境测试报告_音频视频语音.md](./真实环境测试报告_音频视频语音.md)
- [L3 专家层集成验证报告_v1.md](../资料/L3 专家层集成验证报告_v1.md)

### 开发文档

- [开发完成总结_v1.0.txt](../资料/开发完成总结_v1.0.txt)
- [模型分配与加载总结_v2.md](../资料/模型分配与加载总结_v2.md)

---

## 🎉 下一步

### 初学者

1. ✅ 完成快速启动
2. ✅ 运行自动测试模式
3. ✅ 尝试调试控制台对话
4. ✅ 阅读 TSD v1.7 规范

### 开发者

1. 📖 阅读架构文档
2. 🔧 修改和扩展功能
3. 🧪 编写单元测试
4. 📝 提交代码和文档

### 高级用户

1. 🚀 集成真实硬件
2. 🎯 优化性能
3. 🧠 训练自定义模型
4. 🔌 开发专家技能

---

## 📞 支持与反馈

如有问题或建议，请:
1. 查看文档和日志
2. 运行诊断测试
3. 联系开发团队

---

**最后更新**: 2026-03-21  
**维护者**: 祖龙 (ZULONG) 系统架构团队
