# File: openclaw_bridge/api_server.py
"""
OpenClaw API 服务器

提供工具调用的 API 接口：
- POST /api/search - 网络搜索
- POST /api/weather - 天气查询
- 其他工具 API...

这个服务器运行在 3000 端口，专门用于工具调用
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import threading
import requests

logger = logging.getLogger(__name__)


class OpenClawAPIServer:
    """
    OpenClaw API 服务器
    
    提供 REST API 接口供祖龙工具调用
    """
    
    def __init__(self, host: str = "localhost", port: int = 3000):
        """
        初始化 API 服务器
        
        Args:
            host: 监听地址
            port: 监听端口
        """
        self.host = host
        self.port = port
        self._running = False
        
        # 创建 FastAPI 应用
        self.app = FastAPI(title="OpenClaw API")
        
        # 设置路由
        self._setup_routes()
        
        logger.info(f"[OpenClawAPIServer] 初始化完成，监听 {host}:{port}")
    
    def _setup_routes(self):
        """设置 API 路由"""
        
        @self.app.post("/api/search")
        async def search(query: Dict[str, Any]):
            """
            网络搜索接口
            
            Args:
                query: {"query": "搜索关键词", "count": 结果数量}
            
            Returns:
                {"success": bool, "results": [...], "error": "错误信息"}
            """
            try:
                search_query = query.get("query")
                count = query.get("count", 3)
                
                if not search_query:
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "error": "缺少搜索查询"}
                    )
                
                # 调用搜索服务（这里使用 Mock 模式，实际应该调用真实的搜索 API）
                results = await self._perform_search(search_query, count)
                
                return {
                    "success": True,
                    "results": results,
                    "query": search_query,
                    "count": len(results)
                }
            except Exception as e:
                logger.error(f"[OpenClawAPIServer] 搜索失败：{e}")
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "error": str(e)}
                )
        
        @self.app.post("/api/fetch_webpage")
        async def fetch_webpage(request: Dict[str, Any]):
            """
            获取网页内容接口
            
            Args:
                request: {"url": "网页 URL"}
            
            Returns:
                {"success": bool, "content": "网页内容", "error": "错误信息"}
            """
            try:
                url = request.get("url")
                
                if not url:
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "error": "缺少 URL 参数"}
                    )
                
                # 调用网页抓取服务
                result = await self._fetch_webpage_content(url)
                
                return {
                    "success": result.get("success", False),
                    "content": result.get("content", ""),
                    "url": url,
                    "length": len(result.get("content", ""))
                }
            except Exception as e:
                logger.error(f"[OpenClawAPIServer] 获取网页失败：{e}")
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "error": str(e)}
                )
        
        logger.info("[OpenClawAPIServer] API 路由已设置")
    
    async def _perform_search(self, query: str, count: int) -> List[Dict[str, Any]]:
        """
        执行搜索
        
        Args:
            query: 搜索查询
            count: 结果数量
        
        Returns:
            搜索结果列表
        """
        logger.info(f"[OpenClawAPIServer] 执行搜索：{query}, 数量：{count}")
        
        try:
            import subprocess
            import json
            import os
            
            # OpenClaw web-search-skill 的路径
            openclaw_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "openclaw")
            skill_dir = os.path.join(openclaw_dir, "skills", "web-search-skill")
            skill_path = os.path.join(skill_dir, "index.js")
            
            logger.info(f"[OpenClawAPIServer] 技能路径：{skill_path}")
            
            # 创建一个临时脚本来调用 skill
            temp_script = f"""
const skill = require('{skill_path.replace(chr(92), "/")}');
skill('web-search', {{ query: '{query}', numResults: {count} }})
    .then(result => {{
        console.log(JSON.stringify(result));
    }})
    .catch(error => {{
        console.error(JSON.stringify({{ success: false, error: error.message }}));
        process.exit(1);
    }});
"""
            
            # 🔥 设置 Puppeteer 使用合适的 Chromium
            env = os.environ.copy()
            import sys
            if sys.platform == 'win32':
                # Windows 平台：不设置 PUPPETEER_EXECUTABLE_PATH，让 Puppeteer 使用自带的 Chromium
                # 如果已安装 Chromium，puppeteer 会自动下载并使用
                logger.info("[OpenClawAPIServer] Windows 平台，使用 Puppeteer 自带的 Chromium")
            else:
                # Linux/WSL 平台：使用系统安装的 Chromium
                env['PUPPETEER_EXECUTABLE_PATH'] = '/usr/bin/chromium-browser'
                logger.info("[OpenClawAPIServer] Linux/WSL 平台，使用系统 Chromium")
            
            # 执行 Node.js 脚本（在技能目录下执行）
            result = subprocess.run(
                ['node', '-e', temp_script],
                capture_output=True,
                text=True,
                timeout=60,  # 🔥 增加到 60 秒，给 Puppeteer 足够时间启动和搜索
                encoding='utf-8',
                cwd=skill_dir,  # 在技能目录下执行
                env=env
            )
            
            logger.debug(f"[OpenClawAPIServer] Node.js 输出：{result.stdout}")
            if result.stderr:
                logger.debug(f"[OpenClawAPIServer] Node.js 错误：{result.stderr}")
            
            if result.returncode == 0:
                # 提取 JSON 输出（忽略日志行）
                # Node.js 可能输出多行，我们只需要最后一个 JSON 对象
                lines = result.stdout.strip().split('\n')
                json_output = None
                
                # 从后往前找，找到第一个有效的 JSON
                for line in reversed(lines):
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        json_output = line
                        break
                
                # 如果没有找到，尝试从整个输出中提取
                if not json_output:
                    json_output = result.stdout.strip()
                    # 找到最后一个 '{' 开始的位置（JSON 开始）
                    json_start = json_output.rfind('{')
                    if json_start >= 0:
                        json_output = json_output[json_start:]
                
                logger.debug(f"[OpenClawAPIServer] JSON 输出：{json_output}")
                
                try:
                    search_result = json.loads(json_output)
                except json.JSONDecodeError as e:
                    logger.error(f"[OpenClawAPIServer] JSON 解析失败：{e}")
                    logger.error(f"[OpenClawAPIServer] 原始输出：{result.stdout}")
                    return []
                
                if search_result.get('success'):
                    results = search_result.get('results', [])
                    logger.info(f"[OpenClawAPIServer] 搜索成功，找到 {len(results)} 个结果")
                    return results
                else:
                    logger.warning(f"[OpenClawAPIServer] 搜索失败：{search_result.get('error', 'Unknown error')}")
                    return []
            else:
                logger.error(f"[OpenClawAPIServer] Node.js 执行失败：{result.stderr}")
                return []
                
        except subprocess.TimeoutExpired:
            logger.error("[OpenClawAPIServer] 搜索超时")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"[OpenClawAPIServer] JSON 解析错误：{e}")
            logger.error(f"[OpenClawAPIServer] 原始输出：{result.stdout if 'result' in locals() else 'N/A'}")
            return []
        except Exception as e:
            logger.error(f"[OpenClawAPIServer] 搜索异常：{e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def _fetch_webpage_content(self, url: str) -> Dict[str, Any]:
        """
        获取网页内容
        
        Args:
            url: 网页 URL
            
        Returns:
            网页内容
        """
        logger.info(f"[OpenClawAPIServer] 获取网页内容：{url}")
        
        try:
            import subprocess
            import json
            import os
            
            # OpenClaw web-search-skill 的路径
            openclaw_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "openclaw")
            skill_dir = os.path.join(openclaw_dir, "skills", "web-search-skill")
            skill_path = os.path.join(skill_dir, "index.js")
            
            logger.info(f"[OpenClawAPIServer] 技能路径：{skill_path}")
            
            # 创建一个临时脚本来调用 skill
            temp_script = f"""
const skill = require('{skill_path.replace(chr(92), "/")}');
skill('fetch-webpage', {{ url: '{url}' }})
    .then(result => {{
        console.log(JSON.stringify(result));
    }})
    .catch(error => {{
        console.error(JSON.stringify({{ success: false, error: error.message }}));
        process.exit(1);
    }});
"""
            
            # 🔥 设置 Puppeteer 使用合适的 Chromium
            env = os.environ.copy()
            import sys
            if sys.platform == 'win32':
                # Windows 平台：不设置 PUPPETEER_EXECUTABLE_PATH，让 Puppeteer 使用自带的 Chromium
                logger.info("[OpenClawAPIServer] Windows 平台，使用 Puppeteer 自带的 Chromium")
            else:
                # Linux/WSL 平台：使用系统安装的 Chromium
                env['PUPPETEER_EXECUTABLE_PATH'] = '/usr/bin/chromium-browser'
                logger.info("[OpenClawAPIServer] Linux/WSL 平台，使用系统 Chromium")
            
            # 执行 Node.js 脚本（在技能目录下执行）
            result = subprocess.run(
                ['node', '-e', temp_script],
                capture_output=True,
                text=True,
                timeout=30,  # 网页抓取超时 30 秒
                encoding='utf-8',
                cwd=skill_dir,
                env=env
            )
            
            logger.debug(f"[OpenClawAPIServer] Node.js 输出：{result.stdout}")
            if result.stderr:
                logger.debug(f"[OpenClawAPIServer] Node.js 错误：{result.stderr}")
            
            if result.returncode == 0:
                # 提取 JSON 输出
                lines = result.stdout.strip().split('\n')
                json_output = None
                
                for line in reversed(lines):
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        json_output = line
                        break
                
                if not json_output:
                    json_output = result.stdout.strip()
                    json_start = json_output.rfind('{')
                    if json_start >= 0:
                        json_output = json_output[json_start:]
                
                logger.debug(f"[OpenClawAPIServer] JSON 输出：{json_output}")
                
                try:
                    webpage_result = json.loads(json_output)
                except json.JSONDecodeError as e:
                    logger.error(f"[OpenClawAPIServer] JSON 解析失败：{e}")
                    logger.error(f"[OpenClawAPIServer] 原始输出：{result.stdout}")
                    return {"success": False, "error": "JSON 解析失败"}
                
                if webpage_result.get('success'):
                    content = webpage_result.get('content', '')
                    logger.info(f"[OpenClawAPIServer] 成功获取网页内容，长度：{len(content)}")
                    return webpage_result
                else:
                    logger.warning(f"[OpenClawAPIServer] 获取网页失败：{webpage_result.get('error', 'Unknown error')}")
                    return webpage_result
            else:
                logger.error(f"[OpenClawAPIServer] Node.js 执行失败：{result.stderr}")
                return {"success": False, "error": "Node.js 执行失败"}
                
        except subprocess.TimeoutExpired:
            logger.error("[OpenClawAPIServer] 获取网页超时")
            return {"success": False, "error": "获取网页超时"}
        except Exception as e:
            logger.error(f"[OpenClawAPIServer] 获取网页异常：{e}")
            return {"success": False, "error": str(e)}
    
    async def start(self):
        """启动 API 服务器"""
        logger.info("[OpenClawAPIServer] 启动 API 服务器...")
        self._running = True
        
        # 配置 uvicorn
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        # 在后台线程中运行
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        
        # 等待服务器启动
        import time
        time.sleep(1)
        
        logger.info(f"[OpenClawAPIServer] ✅ API 服务器已启动于 http://{self.host}:{self.port}")
    
    async def stop(self):
        """停止 API 服务器"""
        logger.info("[OpenClawAPIServer] 停止 API 服务器...")
        self._running = False
        logger.info("[OpenClawAPIServer] ✅ API 服务器已停止")
