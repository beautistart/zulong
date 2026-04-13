# File: zulong/config/camera_config.py
# 摄像头配置文件

# 🎯 摄像头设备索引配置
# 
# 说明:
# - 0: 通常是内置摄像头或第一个检测到的摄像头
# - 1: 通常是外接 USB 摄像头 (USB Video)
# - 2+: 其他摄像头设备
#
# 使用方法:
# 1. 运行测试脚本识别摄像头:
#    python test_list_cameras.py
#    python test_identify_usb_camera.py
#
# 2. 根据测试结果修改 CAMERA_DEVICE_INDEX
#
# 3. 重启祖龙系统:
#    python -m zulong.bootstrap

# 摄像头设备索引
CAMERA_DEVICE_INDEX = 0  # 内置摄像头

# 🎯 启用/禁用摄像头
ENABLE_CAMERA = False  # 设置为 False 禁用摄像头（物理断开时使用）
