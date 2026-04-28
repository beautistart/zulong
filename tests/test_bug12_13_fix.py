#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BUG-12/13 修复验证测试

验证：
1. BUG-12: experience_confirming 阶段"确认"能正确到达 ReplayIntegration
2. BUG-13: 确认/取消经验后 ReviewStateManager 正确退出，后续消息不再被拦截
"""

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
            "source": "bug12_13_test",
            "payload": {"text": message, "confidence": 1.0},
            "priority": "NORMAL"
        }
    }
    print(f"\n>>> 发送: {message}")
    await ws.send(json.dumps(event))
    
    responses = []
    l2_texts = []
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
                text = str(text)[:300]
                print(f"  <- [EVENT:{evt_type}] {text[:200]}")
                if evt_type == 'L2_OUTPUT':
                    l2_texts.append(text)
            else:
                print(f"  <- [{resp_type}] {json.dumps(data, ensure_ascii=False)[:200]}")
            responses.append(data)
    except asyncio.TimeoutError:
        pass
    
    print(f"  共 {len(responses)} 个响应 (L2: {len(l2_texts)})")
    return responses, l2_texts


async def main():
    print(f"=== BUG-12/13 修复验证测试 ({datetime.now().strftime('%H:%M:%S')}) ===\n")
    
    uri = "ws://127.0.0.1:5555/eventbus"
    print(f"连接到 {uri} ...")
    
    results = {}
    
    async with websockets.connect(uri, ping_interval=None) as ws:
        print("已连接\n")
        
        # Step 1: 启动复盘
        resp, texts = await send_and_collect(ws, "启动复盘", wait_time=10.0)
        results['启动复盘'] = len(texts) > 0
        
        print("\n--- 等待 3 秒 ---")
        await asyncio.sleep(3)
        
        # Step 2: 快速复盘 (进入 review_active 阶段)
        resp, texts = await send_and_collect(ws, "快速复盘", wait_time=15.0)
        results['快速复盘'] = len(texts) > 0
        
        print("\n--- 等待 3 秒 ---")
        await asyncio.sleep(3)
        
        # Step 3: 结束复盘 (触发经验提取，进入 experience_confirming 阶段)
        resp, texts = await send_and_collect(ws, "结束复盘", wait_time=60.0)
        results['结束复盘'] = len(texts) > 0
        has_experience_prompt = any('确认' in t or '经验' in t or '保存' in t for t in texts)
        results['经验提取'] = has_experience_prompt
        
        print("\n--- 等待 3 秒 ---")
        await asyncio.sleep(3)
        
        # Step 4: 确认 (BUG-12 验证：应该触发经验保存，而非"请选择操作")
        resp, texts = await send_and_collect(ws, "确认", wait_time=15.0)
        bug12_fixed = not any('请选择操作' in t for t in texts)
        confirm_ok = any('保存' in t or '应用' in t or '已退出' in t or '确认' in t for t in texts)
        results['BUG-12_确认处理'] = bug12_fixed and len(texts) > 0
        
        print("\n--- 等待 3 秒 ---")
        await asyncio.sleep(3)
        
        # Step 5: 正常对话 (BUG-13 验证：应该正常响应，不被复盘模式拦截)
        resp, texts = await send_and_collect(ws, "你好，今天天气怎么样？", wait_time=15.0)
        bug13_fixed = not any('请选择操作' in t or '复盘模式' in t or '快速复盘' in t for t in texts)
        results['BUG-13_正常退出'] = bug13_fixed
    
    # 结果汇总
    print("\n" + "=" * 60)
    print("=== 测试结果汇总 ===")
    print("=" * 60)
    
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")
    
    print("=" * 60)
    if all_pass:
        print("[PASS] 所有测试通过！BUG-12 和 BUG-13 已修复。")
    else:
        print("[FAIL] 部分测试失败，需要进一步调查。")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
