# Web 页面模型选择器 + 运行时热切换

## Context

当前 Zulong 系统的 LLM 模型在启动时从配置冻结，无法运行时切换。用户在 `http://127.0.0.1:8090/` 的 Web 监控页面上无法直观看到或切换当前使用的模型。刚刚排查到 production 环境错误覆盖导致打了云端 API，说明模型配置可视化和切换的需求很迫切。

**目标**：在 Web 页面顶部添加模型选择器，支持运行时热切换后端（ollama/default/...）和模型 ID，无需重启后端。

## Implementation Plan

### Step 1: 后端 API — 模型配置读取与热切换

**File**: `zulong/ide/ide_server.py`

添加两个 REST API 端点：

```python
@ide_router.get("/api/llm/config")
async def get_llm_config_api():
    """获取当前 LLM 配置和可用后端列表"""
    from zulong.config.config_manager import get_config_manager
    cm = get_config_manager()
    
    # 当前活跃配置
    current_backend = cm.get('llm.backend', 'ollama')
    current_config = cm.get_dict(f'llm.{current_backend}', {})
    
    # 所有可用后端
    backends = {}
    for name in ['ollama', 'default', 'vllm', 'sglang', 'llamaccp', 'lmstudio', 'openai']:
        cfg = cm.get_dict(f'llm.{name}', {})
        if cfg:
            backends[name] = {
                'base_url': cfg.get('base_url', ''),
                'model_id': cfg.get('model_id', ''),
            }
    
    return {
        "current_backend": current_backend,
        "current_model_id": current_config.get('model_id', ''),
        "current_base_url": current_config.get('base_url', ''),
        "backends": backends,
    }


@ide_router.post("/api/llm/switch")
async def switch_llm(data: dict):
    """热切换 LLM 后端和/或模型 ID"""
    backend = data.get("backend")       # 可选：切换后端
    model_id = data.get("model_id")     # 可选：覆盖模型 ID
    
    engine = _get_engine()
    if not engine:
        return {"status": "error", "message": "Engine 未初始化"}
    
    success, msg = engine.hot_switch_llm(backend=backend, model_id=model_id)
    return {"status": "ok" if success else "error", "message": msg}
```

---

### Step 2: InferenceEngine 热切换方法

**File**: `zulong/l2/inference_engine.py`

添加 `hot_switch_llm` 方法：

```python
def hot_switch_llm(self, backend: str = None, model_id: str = None) -> tuple:
    """运行时热切换 LLM 客户端
    
    Args:
        backend: 新后端名称（如 "ollama", "default"）。None 表示保持当前后端。
        model_id: 覆盖模型 ID。None 表示使用后端默认 model_id。
    
    Returns:
        (success: bool, message: str)
    """
    from zulong.config.config_manager import get_config_manager
    cm = get_config_manager()
    
    # 确定目标后端
    target_backend = backend or cm.get('llm.backend', 'ollama')
    target_config = cm.get_dict(f'llm.{target_backend}', {})
    if not target_config:
        return False, f"后端 '{target_backend}' 未配置"
    
    target_base_url = target_config.get('base_url', '')
    target_model_id = model_id or target_config.get('model_id', '')
    target_api_key = target_config.get('api_key', 'EMPTY')
    target_num_ctx = int(target_config.get('num_ctx', 0))
    
    # 尝试创建新客户端
    try:
        from openai import OpenAI
        new_client = OpenAI(base_url=target_base_url, api_key=target_api_key)
    except Exception as e:
        return False, f"客户端创建失败: {e}"
    
    # 原子替换
    self.vllm_client = new_client
    
    # 更新全局变量（供其他模块引用）
    import zulong.models.container as _mc
    _mc.LLM_BACKEND = target_backend
    _mc.LLM_BASE_URL = target_base_url
    _mc.LLM_MODEL_ID = target_model_id
    _mc.LLM_API_KEY = target_api_key
    _mc.VLLM_BASE_URL = target_base_url
    _mc.LLM_NUM_CTX = target_num_ctx
    
    # 更新 context window size
    if target_num_ctx > 0:
        self._context_window_size = target_num_ctx
    
    # 更新配置文件（持久化）
    cm.config['llm']['backend'] = target_backend
    if model_id:
        cm.config['llm'][target_backend]['model_id'] = model_id
    cm.save()
    
    logger.info(f"[LLM] 热切换完成: {target_backend} / {target_model_id} @ {target_base_url}")
    return True, f"已切换到 {target_backend} / {target_model_id}"
```

---

### Step 3: Web 前端 — 模型选择器 UI

**File**: `openclaw_bridge/web/static/index.html`

在 `chat-header` 中添加模型选择器（在 "模块" 按钮之前）：

```html
<!-- 模型选择器（在 chat-header 中） -->
<div class="model-selector" id="modelSelector">
    <select id="backendSelect" onchange="onBackendChange(this.value)" title="后端">
        <option value="ollama">Ollama</option>
    </select>
    <input id="modelIdInput" type="text" placeholder="model_id" title="模型ID" />
    <button onclick="applyModelSwitch()" title="应用">✓</button>
</div>
```

**CSS**：紧凑的内联选择器样式，与现有 header 按钮风格一致。

**JavaScript**：
- 页面加载时 fetch `/api/llm/config` 填充下拉选项和当前值
- `onBackendChange()`: 切换后端时更新 model_id 输入框的 placeholder
- `applyModelSwitch()`: POST `/api/llm/switch` 执行切换，显示结果 toast

---

### Step 4: ConfigManager save() — 已确认可用

**File**: `zulong/config/config_manager.py` (line 251)

`save()` 方法已存在，使用 `yaml.dump` 写入。无需修改。注意：保存后 YAML 注释会丢失，但功能不受影响。

---

## Files Modified Summary

| File | Changes |
|------|---------|
| `zulong/ide/ide_server.py` | 添加 `/api/llm/config` (GET) 和 `/api/llm/switch` (POST) 端点 |
| `zulong/l2/inference_engine.py` | 添加 `hot_switch_llm()` 方法 (~40行) |
| `openclaw_bridge/web/static/index.html` | 在 header 添加模型选择器 UI + CSS + JS |
| `zulong/config/config_manager.py` | 确认/补充 `save()` 方法 |

## Verification

1. 启动后端 → 打开 `http://127.0.0.1:8090/`
2. 页面 header 显示当前后端和模型 ID
3. 下拉切换到 "ollama" → 点击确认 → 观察日志出现 `[LLM] 热切换完成`
4. 发送一条消息 → 验证请求打到新的 base_url（查看日志中 HTTP 请求目标）
5. 刷新页面 → 选择器仍显示切换后的值（配置已持久化）
6. 切换到不存在的后端 → 返回错误提示（不影响当前连接）
