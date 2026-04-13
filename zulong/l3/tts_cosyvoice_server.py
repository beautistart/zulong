# File: zulong/l3/tts_cosyvoice_server.py
"""
CosyVoice TTS 服务进程
模型常驻内存，通过 socket 接收合成请求
"""

import os
import sys
import json
import socket
import threading
import traceback
from typing import Optional
import logging
import time

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 51001


class CosyVoiceServer:
    """
    CosyVoice TTS 服务
    
    模型常驻内存，通过 socket 接收合成请求
    """
    
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        model_dir: str = ""  # TODO: Set your CosyVoice model directory,
        code_path: str = ""  # TODO: Set your CosyVoice code path,
        default_prompt_audio: str = ""  # TODO: Set your prompt audio path,
        default_prompt_text: str = "希望你以后能够做的比我还好呦。"
    ):
        self.host = host
        self.port = port
        self.model_dir = model_dir
        self.code_path = code_path
        self.default_prompt_audio = default_prompt_audio
        self.default_prompt_text = default_prompt_text
        
        self.cosy = None
        self.sample_rate = 24000
        self.running = False
        self.server_socket = None
        
    def load_model(self):
        """加载模型"""
        print(f"加载 CosyVoice2 模型...")
        print(f"  模型目录: {self.model_dir}")
        
        sys.path.insert(0, self.code_path)
        sys.path.insert(0, os.path.join(self.code_path, "third_party", "Matcha-TTS"))
        
        import torch
        import torchaudio
        from cosyvoice.cli.cosyvoice import CosyVoice2
        from cosyvoice.utils.file_utils import load_wav
        
        self.torch = torch
        self.torchaudio = torchaudio
        self.load_wav = load_wav
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  设备: {device}")
        if device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        start_time = time.time()
        self.cosy = CosyVoice2(self.model_dir, load_jit=False, load_trt=False, fp16=False)
        self.sample_rate = self.cosy.sample_rate
        
        elapsed = time.time() - start_time
        print(f"  ✓ 模型加载完成 ({elapsed:.1f}s)")
        print(f"  采样率: {self.sample_rate}")
    
    def synthesize(self, text: str, output_path: str, prompt_text: Optional[str] = None, prompt_audio_path: Optional[str] = None) -> dict:
        """合成语音"""
        if self.cosy is None:
            return {"success": False, "error": "模型未加载"}
        
        try:
            if prompt_text is None:
                prompt_text = self.default_prompt_text
            if prompt_audio_path is None:
                prompt_audio_path = self.default_prompt_audio
            
            if not os.path.exists(prompt_audio_path):
                return {"success": False, "error": f"提示音频不存在: {prompt_audio_path}"}
            
            start_time = time.time()
            
            prompt_speech_16k = self.load_wav(prompt_audio_path, 16000)
            
            for i, j in enumerate(self.cosy.inference_zero_shot(text, prompt_text, prompt_speech_16k)):
                tts_speech = j["tts_speech"]
                if tts_speech.is_cuda:
                    tts_speech = tts_speech.cpu()
                self.torchaudio.save(output_path, tts_speech, self.sample_rate)
                break
            
            elapsed = time.time() - start_time
            size = os.path.getsize(output_path)
            
            return {
                "success": True,
                "output_path": output_path,
                "size": size,
                "elapsed": elapsed
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def handle_client(self, client_socket: socket.socket, address: tuple):
        """处理客户端请求"""
        print(f"客户端连接: {address}")
        
        try:
            data = b""
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n\n" in data:
                    break
            
            if not data:
                return
            
            request = json.loads(data.decode('utf-8').strip())
            command = request.get("command")
            
            if command == "synthesize":
                result = self.synthesize(
                    text=request.get("text", ""),
                    output_path=request.get("output_path", ""),
                    prompt_text=request.get("prompt_text"),
                    prompt_audio_path=request.get("prompt_audio_path")
                )
                response = json.dumps(result, ensure_ascii=False) + "\n\n"
                client_socket.send(response.encode('utf-8'))
            
            elif command == "ping":
                response = json.dumps({"status": "ok", "model_loaded": self.cosy is not None}) + "\n\n"
                client_socket.send(response.encode('utf-8'))
            
            elif command == "shutdown":
                response = json.dumps({"status": "shutting_down"}) + "\n\n"
                client_socket.send(response.encode('utf-8'))
                self.running = False
            
        except Exception as e:
            error_response = json.dumps({"success": False, "error": str(e)}) + "\n\n"
            try:
                client_socket.send(error_response.encode('utf-8'))
            except:
                pass
        
        finally:
            client_socket.close()
            print(f"客户端断开: {address}")
    
    def start(self):
        """启动服务"""
        print(f"\n{'='*60}")
        print(f"CosyVoice TTS 服务")
        print(f"{'='*60}")
        
        self.load_model()
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        
        self.running = True
        print(f"\n服务已启动: {self.host}:{self.port}")
        print(f"等待客户端连接...")
        
        while self.running:
            try:
                self.server_socket.settimeout(1.0)
                try:
                    client_socket, address = self.server_socket.accept()
                    thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    thread.start()
                except socket.timeout:
                    continue
                    
            except Exception as e:
                if self.running:
                    print(f"服务错误: {e}")
        
        self.server_socket.close()
        print("服务已停止")


class CosyVoiceClient:
    """CosyVoice TTS 客户端"""
    
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: int = 60):
        self.host = host
        self.port = port
        self.timeout = timeout
    
    def ping(self) -> bool:
        """检查服务是否可用"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self.host, self.port))
            
            request = json.dumps({"command": "ping"}) + "\n\n"
            sock.send(request.encode('utf-8'))
            
            response = b""
            while b"\n\n" not in response:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
            
            sock.close()
            result = json.loads(response.decode('utf-8').strip())
            return result.get("status") == "ok"
            
        except:
            return False
    
    def synthesize(
        self,
        text: str,
        output_path: str,
        prompt_text: Optional[str] = None,
        prompt_audio_path: Optional[str] = None
    ) -> dict:
        """合成语音"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            
            request = {
                "command": "synthesize",
                "text": text,
                "output_path": output_path,
                "prompt_text": prompt_text,
                "prompt_audio_path": prompt_audio_path
            }
            
            sock.send((json.dumps(request, ensure_ascii=False) + "\n\n").encode('utf-8'))
            
            response = b""
            while b"\n\n" not in response:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
            
            sock.close()
            return json.loads(response.decode('utf-8').strip())
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def shutdown(self):
        """关闭服务"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self.host, self.port))
            
            request = json.dumps({"command": "shutdown"}) + "\n\n"
            sock.send(request.encode('utf-8'))
            sock.close()
            
        except:
            pass


def run_server():
    """运行 TTS 服务"""
    integrated_python = ""  # TODO: Set your Python executable path
    
    if sys.executable != integrated_python:
        print(f"切换到整合包 Python 环境...")
        import subprocess
        subprocess.run([integrated_python] + sys.argv)
        return
    
    server = CosyVoiceServer()
    server.start()


def test_client():
    """测试客户端"""
    print("\n" + "="*60)
    print("测试 CosyVoice TTS 客户端")
    print("="*60)
    
    client = CosyVoiceClient()
    
    print("\n检查服务...")
    if not client.ping():
        print("✗ 服务未运行")
        print(f"  请先启动服务: python -m zulong.l3.tts_cosyvoice_server")
        return False
    
    print("✓ 服务已运行")
    
    output_path = r"d:\AI\project\zulong_beta4\tests\output_cosyvoice_server.wav"
    
    print("\n测试语音合成...")
    result = client.synthesize(
        text="你好，我是祖龙机器人。",
        output_path=output_path
    )
    
    if result.get("success"):
        print(f"✓ 合成成功")
        print(f"  输出: {result.get('output_path')}")
        print(f"  大小: {result.get('size')} bytes")
        print(f"  耗时: {result.get('elapsed'):.2f}s")
    else:
        print(f"✗ 合成失败: {result.get('error')}")
    
    return result.get("success", False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        test_client()
    else:
        run_server()
