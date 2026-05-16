# L1 层模块化架构使用指南

## 📋 概述

祖龙系统 L1 层现在采用**"模块化软插接"架构**,所有功能模块实现为独立插件，通过统一接口通信。

### 架构优势

- ✅ **松耦合**: 插件间通过 shared_memory 通信
- ✅ **热插拔**: 动态加载/卸载，无需重启系统
- ✅ **异常隔离**: 单个插件崩溃不影响其他插件
- ✅ **优先级调度**: CRITICAL > HIGH > NORMAL > LOW
- ✅ **配置化**: YAML 文件管理所有插件

## 🏗️ 架构层次

```
┌─────────────────────────────────────────┐
│     L1-B 调度与意图守门层               │
│  (IntentFilter - ALBERT 15类意图分类)   │
│  (AttentionController - 中断管理)       │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│     L1-A 感知与受控反射层               │
│  (L1A_MotorPlugin - 电机控制/障碍反射)  │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│     L1-C 静默视觉注意层                 │
│  (L1C_VisionPlugin - 视觉分析/动作分类) │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│     L1-D 听觉层                         │
│  (L1D_VoicePlugin - 语音唤醒)           │
│  (L1D_AudioPlugin - 三层注意力音频)     │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│     L1-E 安全层                         │
│  (L1E_GasPlugin - 气体检测/火灾报警)    │
└─────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 基本使用

```python
from zulong.modules.l1.core.plugin_manager import PluginManager

# 创建管理器
manager = PluginManager()

# 加载配置
manager.load_plugins("config/l1_plugins.yaml")

# 启动
manager.start()

# 主循环
while True:
    events = manager.run_cycle(shared_memory={})
    for event in events:
        event_bus.publish(event)
```

### 2. 配置文件 (config/l1_plugins.yaml)

```yaml
plugins:
  - module_id: "L1A/Motor"
    class_path: "zulong.plugins.motor.l1a_motor_plugin.L1A_MotorPlugin"
    enabled: true
    priority: "HIGH"
    config:
      obstacle_threshold: 0.3
      max_speed: 0.8
  
  - module_id: "L1C/Vision"
    class_path: "zulong.plugins.vision.l1c_vision_plugin.L1C_VisionPlugin"
    enabled: true
    priority: "NORMAL"
    config:
      fps: 30
      motion_threshold: 500
```

## 🔌 开发自定义插件

### 步骤 1: 继承 L1PluginBase

```python
from zulong.modules.l1.core.interface import L1PluginBase, ZulongEvent, EventPriority

class MyCustomPlugin(L1PluginBase):
    """我的自定义插件"""
    
    @property
    def module_id(self) -> str:
        return "L1X/MyCustom"
    
    @property
    def priority(self) -> EventPriority:
        return EventPriority.NORMAL
    
    def initialize(self, shared_memory: Dict) -> bool:
        """初始化"""
        shared_memory["my.value"] = 0
        return True
    
    def process_cycle(self, shared_memory: Dict):
        """单周期处理"""
        events = []
        
        # 读取共享内存
        value = shared_memory.get("my.value", 0)
        
        # 业务逻辑
        if value > 100:
            event = ZulongEvent(
                type=EventType.SYSTEM_STATE_CHANGE,
                priority=EventPriority.HIGH,
                source=self.module_id,
                payload={"alert": "Value too high!"}
            )
            events.append(event)
        
        return events
    
    def health_check(self):
        """健康检查"""
        return {"status": "OK", "details": {}, "last_update": time.time()}
```

### 步骤 2: 添加到配置

```yaml
plugins:
  - module_id: "L1X/MyCustom"
    class_path: "zulong.plugins.my_custom.MyCustomPlugin"
    enabled: true
    priority: "NORMAL"
    config:
      threshold: 100
```

## 📊 共享内存键值规范

### 电机插件 (L1A/Motor)

| 键 | 类型 | 描述 | 示例 |
|----|------|------|------|
| `motor.speed` | float | 当前速度 | 0.5 |
| `motor.target_speed` | float | 目标速度 | 0.8 |
| `motor.mode` | str | 模式 | "auto", "charging" |
| `obstacle.distance` | float | 障碍距离 | 0.35 |
| `obstacle.detected` | bool | 障碍标志 | True |

### 视觉插件 (L1C/Vision)

| 键 | 类型 | 描述 | 示例 |
|----|------|------|------|
| `vision.frame_count` | int | 帧计数 | 1500 |
| `vision.motion_detected` | bool | 运动标志 | False |
| `vision.request` | bool | 请求标志 | True |
| `vision.query` | str | 查询文本 | "描述场景" |

### 语音插件 (L1D/Voice)

| 键 | 类型 | 描述 | 示例 |
|----|------|------|------|
| `voice.wakeup_detected` | bool | 唤醒标志 | True |
| `voice.silent_mode` | bool | 安静模式 | False |
| `audio.text` | str | 识别文本 | "你好" |
| `audio.confidence` | float | 置信度 | 0.95 |

### 气体插件 (L1E/Gas)

| 键 | 类型 | 描述 | 示例 |
|----|------|------|------|
| `gas.concentration` | int | 浓度 (ppm) | 450 |
| `gas.alarm` | bool | 报警标志 | False |
| `gas.threshold` | int | 阈值 | 500 |

## 🔧 管理命令

### 查看插件状态

```python
manager = PluginManager()
manager.load_plugins("config/l1_plugins.yaml")

plugin_list = manager.get_plugin_list()
for plugin in plugin_list:
    print(f"{plugin['module_id']}: {plugin['status']}")
    print(f"  优先级：{plugin['priority']}")
    print(f"  周期时间：{plugin['cycle_time_ms']:.2f}ms")
    print(f"  错误数：{plugin['error_count']}")
```

### 动态加载/卸载

```python
# 加载单个插件
manager._load_single_plugin({
    "module_id": "L1X/New",
    "class_path": "zulong.plugins.new.NewPlugin",
    "priority": "HIGH"
})

# 卸载插件
manager.unload_plugin("L1X/New")

# 卸载所有
manager.unload_all()
```

### 性能监控

```python
stats = manager.get_statistics()
print(f"总插件数：{stats['total_plugins']}")
print(f"总周期数：{stats['cycle_count']}")
print(f"平均周期时间：{stats['avg_cycle_time_ms']:.2f}ms")
```

## 🧪 测试

### 运行单元测试

```bash
pytest tests/test_l1_plugin_architecture.py -v
```

### 测试覆盖

- ✅ 接口定义验证
- ✅ 插件加载/卸载
- ✅ 优先级调度
- ✅ 异常隔离
- ✅ 共享内存通信
- ✅ 性能测试 (<33ms)

## 📚 相关文件

### 核心接口

- `zulong/modules/l1/core/interface.py`: 标准接口定义
- `zulong/modules/l1/core/plugin_manager.py`: 插件管理器

### 插件实现

- `zulong/plugins/motor/l1a_motor_plugin.py`: 电机控制
- `zulong/plugins/vision/l1c_vision_plugin.py`: 视觉分析
- `zulong/plugins/voice/l1d_voice_plugin.py`: 语音唤醒
- `zulong/plugins/gas/l1e_gas_plugin.py`: 气体检测

### 配置与测试

- `config/l1_plugins.yaml`: 插件配置
- `tests/test_l1_plugin_architecture.py`: 集成测试

## 🎯 最佳实践

### 1. 插件设计原则

- ✅ **单一职责**: 一个插件只做一件事
- ✅ **快速执行**: process_cycle < 10ms
- ✅ **异常隔离**: 永远不抛出异常
- ✅ **无状态**: 通过 shared_memory 交换数据

### 2. 优先级选择

| 优先级 | 使用场景 | 示例 |
|--------|---------|------|
| CRITICAL | 安全相关 | 气体检测、紧急停止 |
| HIGH | 快速响应 | 障碍检测、运动触发 |
| NORMAL | 普通任务 | 视觉分析、日志记录 |
| LOW | 后台任务 | 数据同步、清理 |

### 3. 共享内存使用

```python
# ✅ 正确：使用唯一键名
shared_memory["motor.speed"] = 0.5

# ❌ 错误：键名冲突
shared_memory["speed"] = 0.5  # 可能与其他模块冲突

# ✅ 正确：嵌套结构
shared_memory["motor"]["speed"] = 0.5
```

### 4. 事件产生

```python
# ✅ 正确：使用辅助函数
from zulong.modules.l1.core.interface import create_event

event = create_event(
    EventType.SENSOR_OBSTACLE,
    EventPriority.HIGH,
    source=self.module_id,
    distance=0.5
)

# ❌ 错误：直接构造 (容易遗漏字段)
event = ZulongEvent(...)
```

## 🔍 调试技巧

### 1. 启用详细日志

```yaml
global:
  log_level: "DEBUG"
```

### 2. 监控性能

```python
import time

for i in range(1000):
    start = time.time()
    events = manager.run_cycle({})
    
    if i % 100 == 0:
        print(f"周期 #{i}: {len(events)} 事件")
```

### 3. 查看插件健康

```python
for plugin in manager._plugins.values():
    health = plugin.health_check()
    print(f"{plugin.module_id}: {health['status']}")
```

## ✅ 成功标志

当你看到以下日志时，说明架构运行正常:

```
🔌 开始加载 L1 插件 (共 4 个)...
✅ 插件加载成功：L1A/Motor
✅ 插件加载成功：L1C/Vision
✅ 插件加载成功：L1D/Voice
✅ 插件加载成功：L1E/Gas
✅ L1 插件加载完成：4/4
🚀 插件管理器已启动
📊 周期 #100 完成 | 插件数：4 | 事件数：12 | 耗时：15.23ms
```

---

**创建时间**: 2026-03-25  
**版本**: v1.0  
**状态**: ✅ 已完成
