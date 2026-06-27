# File: zulong/models/container.py
# 实现祖龙系统的模型单例容器
# 🔥 已修改：使用统一配置系统

import time
import os
from typing import Dict
from zulong.models.config import ModelID, MODEL_CONFIGS, BASE_VRAM_USAGE, SAFE_VRAM_LIMIT_GB
from zulong.models.engine import RealModelLoader

import logging
logger = logging.getLogger(__name__)


def _load_project_dotenv() -> None:
    """Load local .env values before config reads environment variables."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for env_path in (
        os.path.join(root_dir, "config", ".env"),
        os.path.join(root_dir, ".env"),
    ):
        if not os.path.exists(env_path):
            continue
        try:
            with open(env_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    raw = line.strip()
                    if not raw or raw.startswith("#") or "=" not in raw:
                        continue
                    if raw.startswith("export "):
                        raw = raw[len("export "):].strip()
                    key, value = raw.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except Exception as exc:
            logger.debug("[ModelContainer] 本地 .env 加载跳过: %s", exc)


_load_project_dotenv()

# ================================================================================
# 🔥 配置系统集成
# --------------------------------------------------------------------------------
# 注意：LLM 配置不再从此处读取，统一由 models.registry 提供
# （见 InferenceEngine._apply_registry_llm_config）。此处仅保留日志可用性检测。
# ================================================================================
try:
    import zulong.config.config_manager  # noqa: F401  (确认配置系统可导入)
    logger.info("✅ [ModelContainer] 使用统一配置系统")
except ImportError:
    logger.info("⚠️ [ModelContainer] 配置系统不可用 (降级模式)")

# 🔥 关键：手动注册 qwen3_5 架构（解决 Transformers 识别问题）
def register_qwen35_architecture():
    """
    手动注册 qwen3_5 架构到 Transformers
    
    问题：Transformers 4.57.6 可能还未正式支持 qwen3_5 架构
    解决：使用 Qwen2Config 作为基础配置进行注册
    
    注意：必须在导入 transformers 后立即执行
    """
    try:
        from transformers.models.auto import CONFIG_MAPPING
        
        # 检查是否已注册
        if "qwen3_5" in CONFIG_MAPPING:
            print("[ModelContainer] [OK] qwen3_5 架构已注册，跳过")
            return True
        
        # 🔥 关键：使用 Qwen2Config 作为基础（架构相似）
        from transformers import Qwen2Config
        
        # 注册 qwen3_5 架构
        CONFIG_MAPPING.register("qwen3_5", Qwen2Config)
        
        print("[ModelContainer] [OK] qwen3_5 架构注册成功（使用 Qwen2Config）")
        print(f"[ModelContainer]   已注册的架构数：{len(CONFIG_MAPPING)}")
        return True
        
    except Exception as e:
        import traceback
        print(f"[ModelContainer] [WARN] qwen3_5 架构注册失败：{e}")
        print(f"[ModelContainer]   将使用 trust_remote_code=True 加载模型")
        print(f"[ModelContainer]   错误详情：{traceback.format_exc()}")
        return False

# 🔥 关键：在模块加载时立即注册
print("[ModelContainer] 开始注册 qwen3_5 架构...")
register_qwen35_architecture()

# ================================================================================
# 🔥 LLM 后端配置
# --------------------------------------------------------------------------------
# 唯一权威来源是 models.registry（由 Web 端 /api/models/registry/* 维护）。
# 此处模块级全局变量初始化为空；运行时由 InferenceEngine._apply_registry_llm_config()
# 在启动时从 registry 注入真实值（或保持空 —— 此时发消息会自然报错）。
# 不再从 llm.* 配置段 / 环境变量 / 字面量默认值读取，杜绝硬编码与静默兜底。
# ================================================================================
LLM_BACKEND = ""          # 后端类型，如 openai_compatible / openai / anthropic / ollama
LLM_BASE_URL = ""         # API 地址
LLM_MODEL_ID = ""         # 模型 ID
LLM_API_KEY = ""          # API 密钥
LLM_NUM_CTX = 0           # 上下文窗口大小（0 = 未设置）

# 🔥 L2 BACKUP 备用模型配置（同主模型，registry 为唯一来源，初始为空）
LLM_MODEL_ID_BACKUP = ""
LLM_BASE_URL_BACKUP = ""
LLM_API_KEY_BACKUP = ""

# 向后兼容：保留 USE_VLLM_FOR_L2 和 VLLM_BASE_URL
USE_VLLM_FOR_L2 = os.environ.get("USE_VLLM_FOR_L2", "false").lower() == "true"
VLLM_BASE_URL = LLM_BASE_URL  # 向后兼容别名

# API 格式（litellm 网关用）：chat_completions / anthropic_messages / openai_responses / ollama
LLM_API_FORMAT = "chat_completions"
LLM_API_FORMAT_BACKUP = "chat_completions"

print(f"[ModelContainer] [LLM] 后端: {LLM_BACKEND or '(未配置，等待 registry 注入)'}")
print(f"[ModelContainer] [LLM] API 地址: {LLM_BASE_URL or '(空)'}")
print(f"[ModelContainer] [LLM] 模型 ID (CORE): {LLM_MODEL_ID or '(空)'}")
print(f"[ModelContainer] [LLM] 模型 ID (BACKUP): {LLM_MODEL_ID_BACKUP or '(空)'}")
print(f"[ModelContainer] [LLM] API Key: {'***' if LLM_API_KEY else '(空)'}")
print(f"[ModelContainer] [LLM] USE_VLLM_FOR_L2 = {USE_VLLM_FOR_L2}")


class ModelContainer:
    """模型单例容器，管理模型的加载、卸载和访问"""
    _instance = None
    
    def __new__(cls):
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super(ModelContainer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化模型容器"""
        # 避免重复初始化
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        print("[ModelContainer] 初始化模型容器...")
        
        # 存储常驻模型（is_expert=False）
        self.resident_models: Dict[ModelID, object] = {}
        
        # 存储已加载的专家模型及最后访问时间
        self.active_experts: Dict[ModelID, Dict] = {}
        
        # 当前显存使用量
        self.current_vram_usage = 0.0
        
        # 加载常驻模型
        self._load_resident_models()
        
        self._initialized = True
        print(f"[ModelContainer] 初始化完成，当前显存使用：{self.current_vram_usage:.2f}/{SAFE_VRAM_LIMIT_GB}GB")
    
    def _load_resident_models(self):
        """加载所有常驻模型（is_expert=False 且 enabled=True）"""
        print("[ModelContainer] 加载常驻模型...")
        
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        for model_id, config in MODEL_CONFIGS.items():
            # 跳过专家模型
            if config.is_expert:
                continue
            
            # 跳过禁用模型
            if not config.enabled:
                print(f"[ModelContainer] [SKIP] 跳过禁用模型：{model_id.value}")
                continue
            
            print(f"[ModelContainer] 加载常驻模型：{model_id.value}")
            
            # 根据模型 ID 选择正确的本地模型路径
            # 动态路由架构：已移除 L1_PERCEPTION (InternVL)
            if model_id == ModelID.L1_SCHEDULER:
                # 🔥 L1-B 调度器不加载模型，只创建占位符
                # L1-B 只负责事件路由和上下文打包，不调用本地模型
                print(f"[ModelContainer] [INFO] L1_SCHEDULER 创建占位符（不加载模型）")
                self.resident_models[model_id] = {'type': 'placeholder', 'role': 'scheduler'}
                print(f"[ModelContainer] [OK] L1_SCHEDULER 占位符注册成功")
                continue  # 跳过后续加载逻辑
            elif model_id == ModelID.L2_CORE:
                # L2 CORE: 云端模型 (deepseek-v3.1:671b-cloud) 或本地 Qwen3.5-0.8B
                # 🔥 vLLM 支持：如果使用 vLLM，跳过本地加载
                # 🔥 Ollama 支持：如果使用 Ollama，跳过本地加载（通过 API 调用云端/本地模型）
                if USE_VLLM_FOR_L2:
                    print(f"[ModelContainer] [vLLM] L2_CORE 将使用 vLLM OpenAI API，跳过本地加载")
                    print(f"[ModelContainer] [INFO] 使用 0.8B AWQ 量化模型（端口 8000）")
                    # 创建一个占位对象，表示模型已通过 vLLM 加载
                    self.resident_models[model_id] = {'path': 'vllm', 'type': 'remote', 'endpoint': 'http://localhost:8000/v1', 'model_name': 'Qwen3___5-0.8B-AWQ'}
                    print(f"[ModelContainer] [OK] L2_CORE vLLM 占位符注册成功")
                    continue  # 跳过后续加载逻辑
                else:
                    # 🔥 Ollama 模式：L2_CORE 通过 Ollama API 调用，不需要本地加载
                    # 模型配置在 inference_engine.py 中通过 LLM_MODEL_ID 和 LLM_BASE_URL 指定
                    print(f"[ModelContainer] [Ollama] L2_CORE 将通过 Ollama API 调用，跳过本地加载")
                    print(f"[ModelContainer] [INFO] 模型：{os.environ.get('LLM_MODEL_ID', 'deepseek-v3.1:671b-cloud')} @ {os.environ.get('LLM_BASE_URL', 'http://localhost:11434/v1')}")
                    # 创建一个占位对象，表示模型已通过 Ollama 加载
                    self.resident_models[model_id] = {'path': 'ollama', 'type': 'remote', 'endpoint': os.environ.get('LLM_BASE_URL', 'http://localhost:11434/v1'), 'model_name': os.environ.get('LLM_MODEL_ID', 'deepseek-v3.1:671b-cloud')}
                    print(f"[ModelContainer] [OK] L2_CORE Ollama 占位符注册成功")
                    continue  # 跳过后续加载逻辑
            elif model_id == ModelID.L2_BACKUP:
                # L2 BACKUP: 本地 Qwen3.5:4b 模型（通过 Ollama API）或 vLLM 实例
                # 🔥 检查是否有 L2_BACKUP 的 vLLM 配置
                USE_VLLM_FOR_L2_BACKUP = os.environ.get("USE_VLLM_FOR_L2_BACKUP", "true").lower() == "true"
                
                if USE_VLLM_FOR_L2_BACKUP:
                    print(f"[ModelContainer] [vLLM] L2_BACKUP 将独立启动 vLLM 实例（端口 8001）")
                    print(f"[ModelContainer] [INFO] L2_BACKUP 独立运行，不与 L2_CORE 共享")
                    print(f"[ModelContainer] [INFO] 模型路径：Qwen3___5-0.8B-AWQ")
                    print(f"[ModelContainer] [INFO] 量化格式：AWQ 4bit")
                    
                    # 创建一个占位对象，表示模型已通过 vLLM 加载
                    # 🔥 关键：L2_BACKUP 使用独立的 vLLM 实例（端口 8001）
                    self.resident_models[model_id] = {
                        'path': 'vllm', 
                        'type': 'remote', 
                        'endpoint': 'http://localhost:8001/v1',
                        'model_name': 'Qwen3___5-0.8B-AWQ',
                        'quantization': 'awq',
                        'shared_with': None  # 独立运行
                    }
                    print(f"[ModelContainer] [OK] L2_BACKUP vLLM 占位符注册成功（独立实例，端口 8001）")
                    continue  # 跳过后续加载逻辑
                else:
                    # 🔥 Ollama 模式：L2_BACKUP 通过 Ollama API 调用，不需要本地加载
                    print(f"[ModelContainer] [Ollama] L2_BACKUP 将通过 Ollama API 调用，跳过本地加载")
                    print(f"[ModelContainer] [INFO] 模型：{os.environ.get('LLM_MODEL_ID_BACKUP', 'qwen3.5:4b')} @ {os.environ.get('LLM_BASE_URL_BACKUP', 'http://localhost:11434/v1')}")
                    # 创建一个占位对象，表示模型已通过 Ollama 加载
                    self.resident_models[model_id] = {'path': 'ollama', 'type': 'remote', 'endpoint': os.environ.get('LLM_BASE_URL_BACKUP', 'http://localhost:11434/v1'), 'model_name': os.environ.get('LLM_MODEL_ID_BACKUP', 'qwen3.5:4b')}
                    print(f"[ModelContainer] [OK] L2_BACKUP Ollama 占位符注册成功")
                    continue  # 跳过后续加载逻辑
                    
                    continue  # 跳过后续通用加载逻辑
            elif model_id == ModelID.EMBEDDING:
                # Embedding: BAAI/bge-small-zh-v1.5 (CPU)
                model_name = os.path.join(base_dir, "models", "BAAI", "bge-small-zh-v1.5")
            elif model_id == ModelID.VISION_YOLO:
                # Layer 1: YOLOv10-Nano 人体检测 (GPU)
                # YOLO 模型使用 ultralytics 加载，不通过 RealModelLoader
                model_name = os.path.join(base_dir, "yolov10n.pt")
                print(f"[ModelContainer] [WARN] YOLO 模型使用 ultralytics 加载，跳过 RealModelLoader")
                # 创建模拟 loader (实际使用时需要集成 ultralytics)
                self.resident_models[model_id] = {'path': model_name, 'type': 'yolo'}
                self.current_vram_usage += config.estimated_vram_gb
                print(f"[ModelContainer] [OK] YOLO 模型注册成功：{model_name}")
                continue  # 跳过后续加载逻辑
            elif model_id == ModelID.VISION_ACTION:
                # Layer 3: MobileNetV4-TSM 动作分类 (GPU)
                # MobileNetV4 使用 torchvision 加载，不通过 RealModelLoader
                model_name = os.path.join(base_dir, "models", "jaiwei98", "MobileNetV4")
                print(f"[ModelContainer] [WARN] MobileNetV4 使用 torchvision 加载，跳过 RealModelLoader")
                # 创建模拟 loader (实际使用时需要集成 torchvision)
                self.resident_models[model_id] = {'path': model_name, 'type': 'mobilenetv4'}
                self.current_vram_usage += config.estimated_vram_gb
                print(f"[ModelContainer] [OK] MobileNetV4 模型注册成功：{model_name}")
                continue  # 跳过后续加载逻辑
            elif model_id == ModelID.VISION_GESTURE:
                # Layer 4: EfficientNet-B0 手势识别 (GPU)
                # EfficientNet 使用 torchvision 加载，不通过 RealModelLoader
                model_name = os.path.join(base_dir, "models", "google", "efficientnet-b0")
                print(f"[ModelContainer] [WARN] EfficientNet 使用 torchvision 加载，跳过 RealModelLoader")
                # 创建模拟 loader (实际使用时需要集成 torchvision)
                self.resident_models[model_id] = {'path': model_name, 'type': 'efficientnet'}
                self.current_vram_usage += config.estimated_vram_gb
                print(f"[ModelContainer] [OK] EfficientNet 模型注册成功：{model_name}")
                continue  # 跳过后续加载逻辑
            else:
                model_name = os.path.join(base_dir, "models", config.repo_id.replace("/", "_"))
            
            # 检查本地路径是否存在
            if not os.path.exists(model_name):
                print(f"[ModelContainer] [ERROR] 本地模型不存在：{model_name}")
                raise RuntimeError(f"[ModelContainer] 加载失败：{model_id.value}，本地路径不存在：{model_name}")
            
            print(f"[ModelContainer] [OK] 使用本地模型：{model_name}")
            
            # 获取设备配置
            device = config.device
            use_int4 = config.use_int4
            print(f"[ModelContainer] 目标设备：{device.upper()}")
            print(f"[ModelContainer] 使用 INT4 量化：{use_int4}")
            
            # 创建 RealModelLoader 实例
            loader = RealModelLoader(model_name=model_name, device=device, use_int4=use_int4)
            if loader.load_model():
                self.resident_models[model_id] = loader
                self.current_vram_usage += config.estimated_vram_gb
                print(f"[ModelContainer] [OK] 加载完成：{model_id.value}")
            else:
                raise RuntimeError(f"[ModelContainer] 加载失败：{model_id.value}")
    
    def load_expert(self, model_id: ModelID):
        """加载专家模型"""
        if not MODEL_CONFIGS[model_id].is_expert:
            print(f"[ModelContainer] {model_id.value} 不是专家模型，无需加载")
            return
        
        # 检查是否已加载
        if model_id in self.active_experts:
            print(f"[ModelContainer] {model_id.value} 已加载，更新访问时间")
            self.active_experts[model_id]['last_access_time'] = time.time()
            return
        
        model_size = MODEL_CONFIGS[model_id].estimated_vram_gb
        
        # 检查显存是否足够
        while self.current_vram_usage + model_size > SAFE_VRAM_LIMIT_GB:
            print(f"[ModelContainer] 显存不足，当前使用：{self.current_vram_usage:.2f}GB, 需加载：{model_size}GB")
            if not self._evict_lru_expert():
                raise RuntimeError(f"[ModelContainer] 无法加载专家模型 {model_id.value}，无专家模型可驱逐")
        
        # 模拟加载过程
        print(f"[ModelContainer] 加载专家模型：{model_id.value}")
        time.sleep(1.5)  # 模拟从磁盘读取的耗时
        
        # 更新状态
        self.active_experts[model_id] = {
            'model': f"{model_id.value}_model",
            'last_access_time': time.time()
        }
        self.current_vram_usage += model_size
        
        # 记录日志
        print(f"[Memory] Loaded Expert: {model_id.value}, VRAM: {self.current_vram_usage:.2f}/{SAFE_VRAM_LIMIT_GB}GB")
    
    def _evict_lru_expert(self) -> bool:
        """驱逐最近最少使用的专家模型"""
        if not self.active_experts:
            return False
        
        # 找到最近最少使用的模型（last_access_time 最小）
        lru_model_id = min(
            self.active_experts.keys(),
            key=lambda mid: self.active_experts[mid]['last_access_time']
        )
        
        # 释放内存
        model_size = MODEL_CONFIGS[lru_model_id].estimated_vram_gb
        self.current_vram_usage -= model_size
        
        # 移除模型
        del self.active_experts[lru_model_id]
        
        # 记录日志
        print(f"[Memory] VRAM Full. Evicting LRU: {lru_model_id.value}")
        print(f"[Memory] VRAM After Eviction: {self.current_vram_usage:.2f}/{SAFE_VRAM_LIMIT_GB}GB")
        
        return True
    
    def get_model(self, model_id: ModelID) -> object:
        """获取模型，如果是专家模型且未加载则自动加载"""
        # 检查是否为常驻模型
        if model_id in self.resident_models:
            return self.resident_models[model_id]
        
        # 如果是专家模型，尝试加载
        if MODEL_CONFIGS[model_id].is_expert:
            self.load_expert(model_id)
            return self.active_experts[model_id]['model']
        
        raise KeyError(f"[ModelContainer] 模型未找到：{model_id.value}")
    
    def unload_model(self, model_id: ModelID):
        """卸载指定模型"""
        if model_id in self.resident_models:
            # 常驻模型不卸载，只释放显存
            print(f"[ModelContainer] 常驻模型 {model_id.value} 不卸载，仅释放显存")
            return
        
        if model_id in self.active_experts:
            model_size = MODEL_CONFIGS[model_id].estimated_vram_gb
            self.current_vram_usage -= model_size
            del self.active_experts[model_id]
            print(f"[ModelContainer] 已卸载专家模型：{model_id.value}")
        else:
            print(f"[ModelContainer] 模型未加载：{model_id.value}")
