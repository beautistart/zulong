# File: zulong/config/config_manager.py
# 祖龙系统统一配置管理器
# 支持多环境、多后端配置，环境变量覆盖

import os
import platform
import re
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List
from functools import lru_cache

import logging
logger = logging.getLogger(__name__)

_local_env_loaded = False


def _load_local_env_files_once() -> None:
    """Load ignored local env files before YAML variable substitution."""
    global _local_env_loaded
    if _local_env_loaded:
        return
    _local_env_loaded = True

    if os.environ.get("ZULONG_SKIP_LOCAL_ENV", "false").lower() in {"1", "true", "yes"}:
        return

    candidates: List[Path] = []
    explicit_env = os.environ.get("ZULONG_ENV_FILE")
    if explicit_env:
        candidates.append(Path(explicit_env))
    project_root = Path(__file__).resolve().parent.parent.parent
    candidates.append(project_root / "config" / ".env")

    for env_path in candidates:
        if not env_path.exists():
            continue
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[len("export "):].strip()
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    if not key or re.search(r"\s", key):
                        continue
                    value = value.strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                        value = value[1:-1]
                    os.environ.setdefault(key, os.path.expandvars(value))
            logger.info(f"📄 已加载本地环境变量文件：{env_path}")
        except Exception as e:
            logger.warning(f"⚠️ 本地环境变量文件加载失败 [{env_path}]: {e}")


class ConfigManager:
    """
    配置管理器
    
    功能:
    1. 加载 YAML 配置文件
    2. 支持环境变量替换
    3. 支持配置继承和覆盖
    4. 提供类型安全的配置访问
    5. 支持热重载
    """
    
    _instance = None
    _config_cache: Dict[str, Any] = {}
    
    def __new__(cls, config_path: Optional[str] = None):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为 None 则使用默认路径
        """
        if hasattr(self, '_initialized') and self._initialized:
            return

        _load_local_env_files_once()
        
        self.config_path = config_path or self._find_config_file()
        self.config: Dict[str, Any] = {}
        self.environment = os.environ.get("ZULONG_ENV", "production")
        self.platform_profile = os.environ.get("ZULONG_PLATFORM_PROFILE") or self._detect_platform_profile()
        self._load_config()
        self._initialized = True
        
        logger.info(f"✅ ConfigManager 已初始化: {self.config_path}")
        logger.info(f"   环境：{self.environment}")
        logger.info(f"   平台Profile：{self.platform_profile}")
    
    def _find_config_file(self) -> str:
        """查找配置文件"""
        # 优先级：环境变量 > 项目根目录 > 默认路径
        if os.environ.get("ZULONG_CONFIG"):
            return os.environ["ZULONG_CONFIG"]
        
        # 尝试常见路径
        possible_paths = [
            "config/zulong_config.yaml",
            "./config/zulong_config.yaml",
            "../config/zulong_config.yaml",
            str(Path(__file__).parent.parent.parent / "config" / "zulong_config.yaml"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # 如果都没找到，返回默认路径
        return "config/zulong_config.yaml"

    def _detect_platform_profile(self) -> str:
        """检测当前平台 profile 名称。"""
        system = platform.system().lower()
        if system.startswith("win"):
            return "windows"
        if system == "darwin":
            return "macos"
        if system == "linux":
            return "linux"
        return "default"

    def _profile_path(self) -> Optional[Path]:
        """返回当前平台 profile 路径；不存在则返回 None。"""
        if not self.platform_profile or self.platform_profile == "default":
            return None
        config_dir = Path(self.config_path).resolve().parent
        profile_path = config_dir / "profiles" / f"{self.platform_profile}.yaml"
        if profile_path.exists():
            return profile_path
        return None
    
    def _load_config(self) -> None:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            
            # 替换环境变量
            self.config = self._substitute_env_variables(raw_config)
            
            # 应用环境特定配置覆盖
            self._apply_environment_overrides()

            # 应用平台配置覆盖
            self._apply_platform_profile()
            
            # 缓存配置
            ConfigManager._config_cache = self.config
            
            logger.info(f"📄 配置文件加载成功：{self.config_path}")
            
        except FileNotFoundError:
            logger.warning(f"⚠️ 配置文件未找到：{self.config_path}，使用默认配置")
            self.config = self._get_default_config()
        except Exception as e:
            logger.error(f"❌ 配置文件加载失败：{e}")
            raise
    
    def _substitute_env_variables(self, config: Any) -> Any:
        """
        递归替换配置中的环境变量
        
        支持格式:
        - ${ENV_VAR}
        - ${ENV_VAR:default_value}
        
        Args:
            config: 配置对象 (可以是 dict, list, 或基本类型)
            
        Returns:
            替换后的配置对象
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_variables(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_variables(item) for item in config]
        elif isinstance(config, str):
            # 匹配 ${VAR} 或 ${VAR:default}
            pattern = r'\$\{([^}:]+)(?::([^}]+))?\}'
            
            def replace(match):
                env_var = match.group(1)
                default_value = match.group(2)
                value = os.environ.get(env_var, default_value)
                if value is None:
                    return match.group(0)  # 保留原样
                return value
            
            substituted = re.sub(pattern, replace, config)
            return os.path.expanduser(os.path.expandvars(substituted))
        else:
            return config
    
    def _apply_environment_overrides(self) -> None:
        """应用环境特定的配置覆盖"""
        if self.environment not in self.config.get('environments', {}):
            return
        
        env_config = self.config['environments'][self.environment]
        self._merge_config(self.config, env_config)
        
        logger.info(f"🔧 已应用 [{self.environment}] 环境配置覆盖")

    def _apply_platform_profile(self) -> None:
        """应用 Windows/Linux/macOS 平台 profile 覆盖。"""
        profile_path = self._profile_path()
        if not profile_path:
            logger.info(f"ℹ️ 未找到平台 profile，跳过: {self.platform_profile}")
            return

        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                raw_profile = yaml.safe_load(f) or {}
            profile_config = self._substitute_env_variables(raw_profile)
            self._merge_config(self.config, profile_config)
            logger.info(f"🔧 已应用平台 profile [{self.platform_profile}]: {profile_path}")
        except Exception as e:
            logger.error(f"❌ 平台 profile 加载失败 [{profile_path}]: {e}")
            raise
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        递归合并配置
        
        Args:
            base: 基础配置 (会被修改)
            override: 覆盖配置
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _get_default_config(self) -> Dict[str, Any]:
        """返回默认配置"""
        return {
            'system': {
                'name': 'ZULONG',
                'version': '2.0.0',
                'environment': 'production',
                'debug_mode': False,
                'log_level': 'INFO',
            },
            'llm': {
                'backend': 'ollama',
                'ollama': {
                    'base_url': 'http://localhost:11434/v1',
                    'model_id': 'qwen3.5:4b',
                }
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键 (支持点号分隔，如 "llm.ollama.model_id")
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            logger.warning(f"⚠️ 配置键未找到：{key}")
            return None
    
    def get_int(self, key: str, default: int = 0) -> int:
        """获取整数配置值"""
        value = self.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """获取浮点数配置值"""
        value = self.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """获取布尔配置值"""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', '1', 'on')
        return bool(value)
    
    def get_list(self, key: str, default: Optional[List] = None) -> List:
        """获取列表配置值"""
        value = self.get(key, default)
        if isinstance(value, list):
            return value
        return default or []
    
    def get_dict(self, key: str, default: Optional[Dict] = None) -> Dict:
        """获取字典配置值"""
        value = self.get(key, default)
        if isinstance(value, dict):
            return value
        return default or {}
    
    def reload(self) -> None:
        """重新加载配置"""
        logger.info("🔄 重新加载配置...")
        self._load_config()
        logger.info("✅ 配置重新加载完成")
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return self.config.copy()
    
    def save(self, path: Optional[str] = None) -> None:
        """保存配置到文件"""
        save_path = path or self.config_path
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
        logger.info(f"💾 配置已保存到：{save_path}")


# 全局配置实例
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    获取全局配置管理器实例
    
    Args:
        config_path: 配置文件路径 (可选)
        
    Returns:
        ConfigManager 实例
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """
    便捷函数：获取配置值
    
    Args:
        key: 配置键
        default: 默认值
        
    Returns:
        配置值
    """
    return get_config_manager().get(key, default)


def get_llm_config(backend: Optional[str] = None) -> Dict[str, Any]:
    """
    获取 LLM 配置
    
    Args:
        backend: 后端类型 (可选，如果不指定则使用配置的默认值)
        
    Returns:
        LLM 配置字典
    """
    config_manager = get_config_manager()
    from zulong.adapters.backend_resolver import resolve_llm_backend

    resolution = resolve_llm_backend(config_manager.config, backend)
    for warning in resolution.warnings:
        logger.warning(f"⚠️ [LLM] {warning}")
    return resolution.to_config()


def get_l2_inference_config() -> Dict[str, Any]:
    """
    获取 L2 推理引擎配置
    
    Returns:
        L2 推理配置字典
    """
    config_manager = get_config_manager()
    
    return {
        'core_model': config_manager.get('l2_inference.core_model', 'qwen3.5:4b'),
        'backup_model': config_manager.get('l2_inference.backup_model', 'qwen3.5:0.8b'),
        'generation': config_manager.get_dict('l2_inference.generation'),
        'timeout': config_manager.get_dict('l2_inference.timeout'),
        'retry': config_manager.get_dict('l2_inference.retry'),
        'request_interval': config_manager.get('l2_inference.request_interval', 1.0),
        'visual_keywords': (
            config_manager.get_list('l2_inference.visual_keywords_strong')
            + config_manager.get_list('l2_inference.visual_keywords_weak')
        ),
        'orchestrator': config_manager.get_dict('l2_inference.orchestrator'),
        'fc_loop': config_manager.get_dict('l2_inference.fc_loop'),
    }


def get_memory_config() -> Dict[str, Any]:
    """
    获取记忆系统配置
    
    Returns:
        记忆系统配置字典
    """
    config_manager = get_config_manager()
    
    return {
        'short_term': config_manager.get_dict('memory.short_term'),
        'episodic': config_manager.get_dict('memory.episodic'),
        'rag': config_manager.get_dict('memory.rag'),
        'experience': config_manager.get_dict('memory.experience'),
    }


def get_vision_config() -> Dict[str, Any]:
    """
    获取视觉系统配置
    
    Returns:
        视觉系统配置字典
    """
    config_manager = get_config_manager()
    
    return {
        'camera': config_manager.get_dict('vision.camera'),
        'yolo': config_manager.get_dict('vision.yolo'),
        'mediapipe': config_manager.get_dict('vision.mediapipe'),
        'analysis': config_manager.get_dict('vision.analysis'),
    }


def get_audio_config() -> Dict[str, Any]:
    """
    获取音频系统配置
    
    Returns:
        音频系统配置字典
    """
    config_manager = get_config_manager()
    
    return {
        'microphone': config_manager.get_dict('audio.microphone'),
        'speaker': config_manager.get_dict('audio.speaker'),
        'tts': config_manager.get_dict('audio.tts'),
    }


# 模块级快捷函数
def init_config(config_path: Optional[str] = None) -> ConfigManager:
    """初始化配置系统"""
    return get_config_manager(config_path)


# 自动初始化
if os.environ.get("ZULONG_AUTO_INIT_CONFIG", "true").lower() == "true":
    try:
        _ = get_config_manager()
    except Exception as e:
        logger.warning(f"⚠️ 配置自动初始化失败：{e}，将使用默认配置")
