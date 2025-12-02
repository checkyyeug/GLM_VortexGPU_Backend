#!/usr/bin/env python3
"""
Vortex GPU Audio Backend - 改进的服务器启动脚本
支持优雅停止和更好的信号处理
"""

import sys
import signal
import time
import threading
import subprocess
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import hashlib
import datetime
from urllib.parse import urlparse, parse_qs
import base64

# ===== 改进的API处理器 =====
class VortexAPIHandler(BaseHTTPRequestHandler):
    """支持优雅停止的API处理器"""

    # 类变量，用于控制服务器状态
    server_should_stop = False

    def log_message(self, format, *args):
        """自定义日志格式"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {format % args}")

    def do_GET(self):
        """处理GET请求"""
        if self.server_should_stop:
            self.send_error(503, "Server is shutting down")
            return

        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query_params = parse_qs(parsed_path.query)

        # CORS headers
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()

        try:
            if path == '/api/health':
                response = {
                    "status": "healthy",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "server": "Vortex GPU Audio Backend",
                    "version": "1.0.0",
                    "shutdown_mode": "enabled"  # 表示支持优雅停止
                }
                self.wfile.write(json.dumps(response, ensure_ascii=False, indent=2).encode('utf-8'))

            elif path == '/api/stop':
                # 优雅停止API
                response = {
                    "status": "shutting_down",
                    "message": "服务器正在优雅停止...",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                self.wfile.write(json.dumps(response, ensure_ascii=False, indent=2).encode('utf-8'))
                VortexAPIHandler.server_should_stop = True

            elif path == '/api/status':
                # 获取详细状态
                import psutil
                process = psutil.Process()

                response = {
                    "server": {
                        "status": "running" if not self.server_should_stop else "shutting_down",
                        "version": "1.0.0",
                        "uptime": "1小时32分钟",  # 模拟运行时间
                        "python_version": sys.version.split()[0]
                    },
                    "gpu": {
                        "status": "active",
                        "utilization": 78,
                        "memory_used": "6.2GB",
                        "memory_total": "12GB",
                        "temperature": "72°C"
                    },
                    "audio": {
                        "sample_rate": 384000,
                        "bit_depth": 32,
                        "channels": 2,
                        "buffer_size": 256,
                        "format": "PCM32"
                    },
                    "processing": {
                        "equalizer_bands": 512,
                        "convolution_length": "16M",
                        "dsd_support": True,
                        "dsd_rate": 45158400,
                        "latency_ms": 4.2
                    },
                    "system": {
                        "cpu_usage": process.cpu_percent(),
                        "memory_usage": process.memory_info().rss / 1024 / 1024,  # MB
                        "thread_count": process.num_threads()
                    }
                }
                self.wfile.write(json.dumps(response, ensure_ascii=False, indent=2).encode('utf-8'))

            elif path == '/api/spectrum':
                # 生成频谱数据
                import numpy as np
                freqs = np.logspace(np.log10(20), np.log10(20000), 256).tolist()
                magnitudes = (80 * np.exp(-((np.array(range(256)) - 128) ** 2) / 2000) +
                              np.random.normal(0, 2, 256)).tolist()
                phases = np.random.uniform(0, 2 * np.pi, 256).tolist()

                response = {
                    "frequencies": freqs,
                    "magnitudes": magnitudes,
                    "phases": phases,
                    "sample_rate": 384000,
                    "fft_size": 65536,
                    "window": "hann",
                    "overlap": 0.75
                }
                self.wfile.write(json.dumps(response, ensure_ascii=False, indent=2).encode('utf-8'))

            else:
                # 静态文件服务
                self.serve_static_file(path)

        except Exception as e:
            self.send_response(500)
            self.end_headers()
            error_response = {
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
            self.wfile.write(json.dumps(error_response, ensure_ascii=False, indent=2).encode('utf-8'))

    def serve_static_file(self, path):
        """提供静态文件服务"""
        if path == '/':
            path = '/index.html'

        # 简化的静态文件内容
        static_content = {
            '/index.html': '''<!DOCTYPE html>
<html>
<head>
    <title>Vortex GPU Audio Backend</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 40px; }
        .controls { display: flex; gap: 20px; margin-bottom: 30px; justify-content: center; }
        .btn { padding: 12px 24px; background: #007acc; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
        .btn:hover { background: #005a9e; }
        .btn.danger { background: #dc3545; }
        .btn.danger:hover { background: #c82333; }
        .status { background: #2d2d2d; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .spectrum { height: 300px; background: #2d2d2d; border-radius: 8px; position: relative; }
        .spectrum-bar { position: absolute; bottom: 0; width: 2px; background: linear-gradient(to top, #007acc, #00ff88); }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎵 Vortex GPU Audio Backend</h1>
            <p>专业级GPU音频处理演示</p>
        </div>

        <div class="controls">
            <button class="btn" onclick="checkHealth()">检查健康状态</button>
            <button class="btn" onclick="getStatus()">获取详细状态</button>
            <button class="btn danger" onclick="stopServer()">停止服务器</button>
        </div>

        <div id="status" class="status">
            <h3>服务器状态</h3>
            <div id="status-content">点击"检查健康状态"开始...</div>
        </div>

        <div class="spectrum" id="spectrum">
            <div style="position: absolute; top: 10px; left: 10px;">实时频谱分析</div>
        </div>
    </div>

    <script>
        async function checkHealth() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                document.getElementById('status-content').innerHTML =
                    '<strong>状态:</strong> ' + data.status + '<br>' +
                    '<strong>版本:</strong> ' + data.version + '<br>' +
                    '<strong>时间:</strong> ' + data.timestamp;
            } catch (error) {
                document.getElementById('status-content').innerHTML = '连接失败: ' + error.message;
            }
        }

        async function getStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                document.getElementById('status-content').innerHTML =
                    '<strong>服务器:</strong> ' + data.server.status + '<br>' +
                    '<strong>GPU使用率:</strong> ' + data.gpu.utilization + '%<br>' +
                    '<strong>采样率:</strong> ' + data.audio.sample_rate + ' Hz<br>' +
                    '<strong>延迟:</strong> ' + data.processing.latency_ms + ' ms';
            } catch (error) {
                document.getElementById('status-content').innerHTML = '获取状态失败: ' + error.message;
            }
        }

        async function stopServer() {
            if (confirm('确定要停止服务器吗？')) {
                try {
                    const response = await fetch('/api/stop');
                    const data = await response.json();
                    document.getElementById('status-content').innerHTML =
                        '<strong style="color: orange;">' + data.message + '</strong>';
                    setTimeout(() => {
                        window.location.reload();
                    }, 2000);
                } catch (error) {
                    document.getElementById('status-content').innerHTML = '停止失败: ' + error.message;
                }
            }
        }

        // 自动更新状态
        setInterval(checkHealth, 5000);
        checkHealth();

        // 绘制频谱
        async function drawSpectrum() {
            try {
                const response = await fetch('/api/spectrum');
                const data = await response.json();
                const spectrum = document.getElementById('spectrum');

                // 清除旧的频谱条
                const oldBars = spectrum.querySelectorAll('.spectrum-bar');
                oldBars.forEach(bar => bar.remove());

                // 绘制新的频谱条
                const barCount = 64;
                const spectrumWidth = spectrum.offsetWidth;
                const barWidth = spectrumWidth / barCount - 2;

                for (let i = 0; i < barCount; i++) {
                    const bar = document.createElement('div');
                    bar.className = 'spectrum-bar';
                    bar.style.left = (i * (barWidth + 2)) + 'px';

                    // 从完整频谱数据中采样
                    const dataIndex = Math.floor(i * data.magnitudes.length / barCount);
                    const height = (data.magnitudes[dataIndex] / 100) * 250;
                    bar.style.height = Math.max(1, height) + 'px';
                    bar.style.width = barWidth + 'px';

                    spectrum.appendChild(bar);
                }
            } catch (error) {
                console.log('频谱更新失败:', error);
            }
        }

        // 定期更新频谱
        setInterval(drawSpectrum, 100);
        drawSpectrum();
    </script>
</body>
</html>'''
        }

        if path in static_content:
            content_type = 'text/html' if path.endswith('.html') else 'text/plain'
            self.send_response(200)
            self.send_header('Content-Type', content_type + '; charset=utf-8')
            self.end_headers()
            self.wfile.write(static_content[path].encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        """处理CORS预检请求"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

# ===== 改进的服务器类 =====
class VortexHTTPServer(HTTPServer):
    """支持优雅停止的HTTP服务器"""

    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        self._stop_event = threading.Event()

    def serve_forever(self, poll_interval=0.5):
        """改进的serve_forever方法"""
        print(f"🚀 Vortex GPU Audio Backend 启动成功!")
        print(f"📍 监听地址: http://{self.server_address[0]}:{self.server_address[1]}")
        print(f"🌐 Web界面: http://localhost:{self.server_address[1]}")
        print(f"⏰ 启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💡 提示: 按 Ctrl+C 或访问 http://localhost:{self.server_address[1]}/api/stop 来停止服务器")
        print("")
        print("=" * 70)
        print("🎵 VORTEX GPU AUDIO BACKEND - 专业音频处理演示")
        print("=" * 70)
        print("")
        print("📊 可用API端点:")
        print(f"  • 健康检查: http://localhost:{self.server_address[1]}/api/health")
        print(f"  • 详细状态: http://localhost:{self.server_address[1]}/api/status")
        print(f"  • 频谱数据: http://localhost:{self.server_address[1]}/api/spectrum")
        print(f"  • 停止服务: http://localhost:{self.server_address[1]}/api/stop")
        print("")
        print("🛠️ 停止方法:")
        print("  • 按 Ctrl+C (优雅停止)")
        print("  • 访问 /api/stop (API停止)")
        print("  • 运行: python stop_server.py")
        print("")
        print("=" * 70)
        print("🖥️  Web界面功能:")
        print("  • 实时频谱分析显示")
        print("  • GPU状态监控")
        print("  • 服务器健康检查")
        print("  • 一键停止控制")
        print("")

        try:
            while not self._stop_event.is_set():
                self.handle_request()

                # 检查是否需要停止
                if VortexAPIHandler.server_should_stop:
                    print("📡 收到停止请求，正在优雅关闭...")
                    break

        except KeyboardInterrupt:
            print("\n⚠️  收到中断信号，正在停止...")
        finally:
            self.server_close()

# ===== 信号处理 =====
def signal_handler(signum, frame):
    """处理系统信号"""
    print(f"\n🛑 收到信号 {signum}，准备停止服务器...")
    VortexAPIHandler.server_should_stop = True

# ===== 主函数 =====
def main():
    """主函数"""
    print("=" * 70)
    print("🎵 VORTEX GPU AUDIO BACKEND - 启动中...")
    print("=" * 70)
    print("")

    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 检查依赖
        try:
            import psutil
            import numpy
            print("✅ 依赖库检查通过")
        except ImportError as e:
            print(f"❌ 缺少依赖库: {e}")
            print("📦 正在安装依赖...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil", "numpy"])
            print("✅ 依赖库安装完成")

        # 检查端口
        port = 8080
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) == 0:
                print(f"⚠️  端口 {port} 已被占用")
                print(f"🔍 检查占用进程...")
                try:
                    import psutil
                    for conn in psutil.net_connections():
                        if conn.laddr.port == port:
                            process = psutil.Process(conn.pid)
                            print(f"  进程: {process.name()} (PID: {conn.pid})")
                            print(f"  命令行: {' '.join(process.cmdline())}")
                            break
                except:
                    pass

                response = input("是否继续使用此端口? (y/N): ").lower()
                if response != 'y':
                    print("❌ 启动取消")
                    return

        # 创建服务器
        server_address = ('', port)
        httpd = VortexHTTPServer(server_address, VortexAPIHandler)

        # 启动服务器
        httpd.serve_forever()

        print("🎉 服务器已优雅停止")
        print("=" * 70)

    except Exception as e:
        print(f"❌ 服务器启动失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()