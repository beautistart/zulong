"""
系统状态诊断脚本

用途：
1. 检查 Python 进程状态
2. 检查关键组件是否运行
3. 测试事件总线连接
4. 发送测试消息

使用方法：
python diagnose_system.py
"""

import asyncio
import os
import shutil
import socket
import subprocess
import sys
import time
import logging
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_processes():
    """检查 Python 进程"""
    logger.info("=" * 80)
    logger.info("📊 步骤 1: 检查 Python 进程")
    logger.info("=" * 80)

    try:
        processes = _list_python_processes()
        if not processes:
            logger.warning("⚠️ 警告：未发现 Python 进程")
            return False

        for process in processes[:20]:
            logger.info(process)
        if len(processes) > 20:
            logger.info(f"... 另有 {len(processes) - 20} 个 Python 进程未展示")
        logger.info(f"✅ 发现 {len(processes)} 个 Python 进程")
        if len(processes) < 3:
            logger.warning("⚠️ Python 进程数量少于 3 个，系统可能只启动了部分组件")
        return True

    except Exception as e:
        logger.error(f"❌ 检查失败：{e}")
        return False


def _list_python_processes():
    """跨平台列出 Python 进程，优先使用 psutil，降级到系统命令。"""
    try:
        import psutil  # type: ignore

        rows = []
        for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent"]):
            info = proc.info
            name = str(info.get("name") or "")
            cmdline = " ".join(info.get("cmdline") or [])
            if "python" in name.lower() or "python" in cmdline.lower():
                rows.append(
                    f"pid={info.get('pid')} name={name} cpu={info.get('cpu_percent')} cmd={cmdline[:180]}"
                )
        return rows
    except Exception:
        pass

    if os.name == "nt":
        command = ["tasklist", "/FO", "CSV", "/NH"]
    else:
        command = ["ps", "-eo", "pid=,comm=,pcpu=,args="]

    if not shutil.which(command[0]):
        return []

    result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    output = result.stdout or ""
    return [line.strip() for line in output.splitlines() if "python" in line.lower()]


def check_listening_ports():
    """检查关键端口是否监听"""
    logger.info("=" * 80)
    logger.info("📊 步骤 2: 检查网络端口")
    logger.info("=" * 80)

    ports_to_check = [
        (8090, "IDE/WebSocket Backend"),
        (5555, "EventBus 旧版/可选"),
        (8765, "Web Server 旧版/可选"),
    ]

    for port, name in ports_to_check:
        if _is_port_open("127.0.0.1", port):
            logger.info(f"✅ {name} 端口 {port} 正在监听")
        else:
            logger.warning(f"⚠️ {name} 端口 {port} 未监听")


def _is_port_open(host: str, port: int, timeout: float = 0.8) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


async def test_event_bus():
    """测试事件总线"""
    logger.info("=" * 80)
    logger.info("📊 步骤 3: 测试事件总线")
    logger.info("=" * 80)
    
    try:
        from zulong.core.event_bus import event_bus
        from zulong.core.types import EventType, EventPriority, ZulongEvent
        
        logger.info("✅ EventBus 模块导入成功")
        
        # 创建一个测试事件
        test_event = ZulongEvent(
            type=EventType.USER_TEXT,
            source="DiagnosticScript",
            payload={"text": "系统诊断测试"},
            priority=EventPriority.LOW
        )
        
        logger.info(f"📤 发送测试事件：{test_event.type}")
        event_bus.publish(test_event)
        logger.info("✅ 测试事件已发布")
        
        # 等待响应（2 秒）
        logger.info("⏳ 等待 2 秒观察响应...")
        await asyncio.sleep(2)
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ EventBus 模块导入失败：{e}")
        logger.error("💡 提示：确保在正确的虚拟环境中运行")
        return False
    except Exception as e:
        logger.error(f"❌ 事件总线测试失败：{e}")
        return False


def check_memory_files():
    """检查记忆文件"""
    import os
    from pathlib import Path
    
    logger.info("=" * 80)
    logger.info("📊 步骤 4: 检查记忆文件")
    logger.info("=" * 80)
    
    ZULONG_HOME = Path(os.environ.get('ZULONG_HOME', Path(__file__).resolve().parent.parent))
    base_path = ZULONG_HOME / "data"
    
    files_to_check = [
        "short_term_memory/index.json",
        "experience_db/experiences.db",
    ]
    
    for file_path in files_to_check:
        full_path = base_path / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            logger.info(f"✅ {file_path} 存在 (大小：{size} 字节)")
        else:
            logger.warning(f"⚠️ {file_path} 不存在")


def send_test_counting():
    """发送数数测试"""
    logger.info("=" * 80)
    logger.info("📊 步骤 5: 发送数数测试")
    logger.info("=" * 80)
    
    try:
        from zulong.core.event_bus import event_bus
        from zulong.core.types import EventType, EventPriority, ZulongEvent
        
        for i in range(1, 4):
            logger.info(f"\n📤 发送数字：{i}")
            
            event = ZulongEvent(
                type=EventType.USER_TEXT,
                source="DiagnosticScript",
                payload={"text": str(i)},
                priority=EventPriority.NORMAL
            )
            
            event_bus.publish(event)
            logger.info(f"✅ 事件 {i} 已发布")
            
            # 等待 3 秒（避免冷却时间）
            time.sleep(3)
        
        logger.info("\n🎯 数数测试完成！请检查系统响应")
        
    except Exception as e:
        logger.error(f"❌ 数数测试失败：{e}")


async def main():
    """主诊断流程"""
    logger.info("\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 20 + "祖龙系统诊断工具" + " " * 35 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info(f"诊断时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\n")
    
    # 1. 检查进程
    processes_ok = check_python_processes()
    
    # 2. 检查端口
    check_listening_ports()
    
    # 3. 检查记忆文件
    check_memory_files()
    
    # 4. 测试事件总线
    if processes_ok:
        eventbus_ok = await test_event_bus()
        
        # 5. 发送数数测试
        if eventbus_ok:
            logger.info("\n⏳ 等待 5 秒后开始数数测试...")
            await asyncio.sleep(5)
            send_test_counting()
    
    # 总结
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("📋 诊断总结")
    logger.info("=" * 80)
    logger.info("✅ 诊断完成！请查看上方的日志输出")
    logger.info("\n💡 建议:")
    logger.info("1. 如果后端端口未监听，请按 docs/三端安装与诊断指南.md 启动对应平台服务")
    logger.info("2. 如果事件总线测试失败，请检查当前 Python 虚拟环境和项目根目录")
    logger.info("3. 如果数数测试无响应，请查看详细诊断报告:")
    logger.info("   diagnostics\\Counting_No_Response_Deep_Diagnosis.md")
    logger.info("\n")


if __name__ == "__main__":
    asyncio.run(main())
