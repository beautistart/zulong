# 祖龙系统修复验证方案

## 修复内容总结

### 1. 模型预加载功能
- 新增 `zulong/utils/model_preloader.py`
- 支持 Ollama、LM Studio、vLLM 等后端
- 后台启动，不阻塞系统

### 2. 超时配置优化
- 核心模型超时：30 → 300 秒
- FC 循环超时：120 → 300 秒
- 备用模型超时：30 → 60 秒

### 3. 配置变量修复
- 修复 `l2_inference.core_model` 跨路径引用问题
- 直接使用具体模型 ID

### 4. Bootstrap 集成
- 在 SharedMemoryPool 后添加模型预加载

## 验证步骤

### 步骤 1：重启系统
```bash
# 停止现有系统（Ctrl+C）
# 重新启动
python -m zulong.bootstrap
```

### 步骤 2：检查启动日志
应该看到：
- `🔥 [BOOTSTRAP] 启动模型预加载（后台）...`
- `✅ [BOOTSTRAP] 模型预加载已在后台启动`
- `⏱️ [L2] 超时配置: core=300s, backup=60s, fc_loop=300s`

### 步骤 3：测试简单对话
```python
import asyncio
import websockets
import json

async def test_simple():
    uri = "ws://localhost:5555/eventbus"
    async with websockets.connect(uri) as ws:
        # 发送消息
        msg = {"type": "USER_TEXT", "payload": {"text": "你好，1+1等于几？"}}
        await ws.send(json.dumps(msg))
        
        # 等待响应（最多 60 秒）
        for _ in range(60):
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=1.0)
                data = json.loads(resp)
                if data.get("type") == "L2_OUTPUT":
                    print(f"✅ 收到响应: {data['payload']['text']}")
                    return True
            except:
                pass
        print("❌ 未收到响应")
        return False

asyncio.run(test_simple())
```

### 步骤 4：测试复杂任务
发送复杂任务请求（如"帮我制定一个学习计划"），观察：
- 日志中是否显示工具调用
- 是否在 300 秒内完成
- 是否收到 "抱歉，我当前响应较慢" 的错误

### 步骤 5：检查模型是否已预加载
```bash
# 检查 Ollama 是否已加载模型
curl http://localhost:11434/api/ps
```
应该看到 `deepseek-v3.1:671b-cloud` 在运行中

## 预期结果

1. ✅ 系统启动时自动预加载模型
2. ✅ 第一次对话不需要等待模型冷启动
3. ✅ 复杂任务有足够处理时间（5 分钟）
4. ✅ 配置变量正确解析

## 故障排查

如果预加载失败：
- 检查 Ollama 服务是否运行：`ollama list`
- 检查模型是否存在：`ollama list | grep deepseek`
- 检查 API 端点：`curl http://localhost:11434/api/tags`

如果仍然超时：
- 查看日志中的超时时间是否正确
- 检查网络延迟
- 考虑增加 `fc_loop` 超时值
