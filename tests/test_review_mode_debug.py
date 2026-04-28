#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""精简测试：仅发送 启动复盘 + 快速复盘，用于调试 review_mode 状态"""

import asyncio
import websockets
import json
import sys
import io
from datetime import datetime

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


async def send_and_collect(ws, message, wait_time=10.0):
    """发送消息并收集响应"""
    event = {
        "type": "PUBLISH",
        "event": {
            "type": "USER_TEXT",
            "source": "debug_test",
            "payload": {"text": message, "confidence": 1.0},
            "priority": "NORMAL"
        }
    }
    print(f"\n>>> 发送: {message}")
    await ws.send(json.dumps(event))
    
    responses = []
    try:
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=wait_time)
            data = json.loads(raw)
            resp_type = data.get('type', 'unknown')
            if resp_type == 'ACK':
                print(f"  <- ACK")
            elif resp_type == 'SUBSCRIBE':
                inner = data.get('event', {})
                evt_type = inner.get('type', 'unknown')
                payload = inner.get('payload', {})
                text = payload.get('text', '')
                if isinstance(text, (list, tuple)):
                    text = text[0] if text else ''
                text = str(text)[:200]
                print(f"  <- [EVENT:{evt_type}] {text}")
            else:
                print(f"  <- [{resp_type}] {json.dumps(data, ensure_ascii=False)[:200]}")
            responses.append(data)
    except asyncio.TimeoutError:
        pass
    
    print(f"  共 {len(responses)} 个响应")
    return responses


async def main():
    print(f"=== review_mode 调试测试 ({datetime.now()}) ===\n")
    
    uri = "ws://127.0.0.1:5555/eventbus"
    print(f"连接到 {uri} ...")
    
    async with websockets.connect(uri, ping_interval=None) as ws:
        print("已连接\n")
        
        # Step 1: 启动复盘
        await send_and_collect(ws, "启动复盘", wait_time=10.0)
        
        # 等 3 秒
        print("\n--- 等待 3 秒 ---")
        await asyncio.sleep(3)
        
        # Step 2: 快速复盘
        await send_and_collect(ws, "快速复盘", wait_time=15.0)
        
        # 等 3 秒
        print("\n--- 等待 3 秒 ---")
        await asyncio.sleep(3)
        
        # Step 3: 结束复盘 (经验提取含vLLM推理，需要更长等待)
        await send_and_collect(ws, "结束复盘", wait_time=45.0)
        
        # 等 3 秒
        print("\n--- 等待 3 秒 ---")
        await asyncio.sleep(3)
        
        # Step 4: 确认经验
        await send_and_collect(ws, "确认", wait_time=15.0)
    
    print("\n=== 测试完成 ===")
    print("请检查服务器日志中的 [StateManager] review_mode 变更 记录")


if __name__ == "__main__":
    asyncio.run(main())
