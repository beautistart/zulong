# 摄像头配置指南

## 📋 概述

祖龙系统支持多个摄像头设备，默认为**USB Video 摄像头** (索引 1)。

## 🔧 配置方法

### 方法 1: 修改配置文件 (推荐)

编辑 `zulong/config/camera_config.py`:

```python
# 摄像头设备索引
CAMERA_DEVICE_INDEX = 1  # 0=内置，1=USB Video
```

### 方法 2: 直接修改 bootstrap.py

编辑 `zulong/bootstrap.py` 第 152 行:

```python
self._camera_device = CameraDevice(device_index=1)  # 改为需要的索引
```

## 🔍 检测可用摄像头

### 1. 列出所有摄像头

```bash
python test_list_cameras.py
```

**输出示例:**
```
✅ 发现摄像头 #0:
   - 名称：Camera 0
   - 分辨率：640 x 480
   - FPS: 30.00

✅ 发现摄像头 #1:
   - 名称：USB Video
   - 分辨率：640 x 480
   - FPS: 30.00
```

### 2. 识别 USB 摄像头

```bash
python test_identify_usb_camera.py
```

**输出示例:**
```
测试摄像头 #0
   - 初始亮度：89.24
   - 最终亮度：89.30
✅ 摄像头 #0 工作正常

测试摄像头 #1
   - 初始亮度：23.87
   - 最终亮度：23.87
⚠️ 摄像头 #1 亮度过低
```

## 📊 摄像头索引说明

| 索引 | 设备类型 | 说明 |
|------|----------|------|
| 0 | 内置摄像头 | 笔记本内置摄像头或第一个检测到的设备 |
| 1 | USB Video | 外接 USB 摄像头 (推荐) |
| 2+ | 其他 | 其他摄像头设备 |

## ⚙️ 高级配置

编辑 `zulong/config/camera_config.py`:

```python
# 分辨率
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# 帧率
CAMERA_FPS = 30

# 预热帧数 (等待自动曝光)
CAMERA_WARMUP_FRAMES = 60

# 最小可用亮度
MIN_BRIGHTNESS = 30
```

## 🐛 故障排除

### 问题 1: 摄像头无法打开

**症状:**
```
❌ 无法打开摄像头 0
```

**解决方案:**
1. 检查摄像头是否已连接
2. 检查摄像头驱动是否已安装
3. 关闭其他使用摄像头的程序
4. 运行 `python test_list_cameras.py` 检测可用设备

### 问题 2: 画面过暗

**症状:**
```
⚠️ 预热完成但亮度仍然过低 (23.87)
```

**解决方案:**
1. 打开环境灯光
2. 打开摄像头镜头盖
3. 调整摄像头角度
4. 尝试使用另一个摄像头 (修改 CAMERA_DEVICE_INDEX)

### 问题 3: 画面卡顿

**症状:**
- 帧率低于 15 FPS
- 画面延迟严重

**解决方案:**
1. 降低分辨率: `CAMERA_WIDTH = 320`, `CAMERA_HEIGHT = 240`
2. 降低帧率：`CAMERA_FPS = 15`
3. 关闭其他占用 CPU 的程序

## 📝 测试步骤

1. **检测摄像头:**
   ```bash
   python test_list_cameras.py
   ```

2. **识别 USB 摄像头:**
   ```bash
   python test_identify_usb_camera.py
   ```

3. **修改配置:**
   编辑 `zulong/config/camera_config.py`,设置正确的 `CAMERA_DEVICE_INDEX`

4. **启动系统:**
   ```bash
   python -m zulong.bootstrap
   ```

5. **验证:**
   查看日志中的摄像头信息:
   ```
   📷 Initializing camera device (index=1)...
   ✅ Camera device initialized
   ```

## 💡 最佳实践

1. **固定使用 USB 摄像头**: 设置 `CAMERA_DEVICE_INDEX = 1`
2. **避免频繁切换**: 启动前确定好摄像头
3. **定期检查**: 确保摄像头镜头清洁
4. **环境光线**: 保证充足的环境照明

## 📖 相关文件

- `zulong/config/camera_config.py` - 摄像头配置文件
- `zulong/l0/devices/camera_device.py` - 摄像头驱动
- `test_list_cameras.py` - 摄像头检测工具
- `test_identify_usb_camera.py` - USB 摄像头识别工具
