#!/usr/bin/env python3
"""
Vortex GPU Audio Backend - 简单演示服务器
不需要完整构建环境，展示项目API和功能
"""

import json
import time
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import socket
import math
import random

class VortexAPIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """处理GET请求"""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/':
            self.send_html_response(self.get_main_page())
        elif parsed_path.path == '/api/health':
            self.send_json_response(self.get_health_status())
        elif parsed_path.path == '/api/status':
            self.send_json_response(self.get_system_status())
        elif parsed_path.path == '/api/capabilities':
            self.send_json_response(self.get_capabilities())
        elif parsed_path.path == '/api/spectrum':
            self.send_json_response(self.get_spectrum_data())
        elif parsed_path.path == '/api/equalizer':
            self.send_json_response(self.get_equalizer_status())
        elif parsed_path.path == '/api/devices':
            self.send_json_response(self.get_output_devices())
        else:
            self.send_404()

    def do_POST(self):
        """处理POST请求"""
        parsed_path = urlparse(self.path)
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode('utf-8'))
        except:
            data = {}

        if parsed_path.path == '/api/audio/upload':
            self.send_json_response(self.handle_audio_upload(data))
        elif parsed_path.path == '/api/audio/process':
            self.send_json_response(self.handle_audio_process(data))
        elif parsed_path.path == '/api/equalizer':
            self.send_json_response(self.handle_equalizer_update(data))
        elif parsed_path.path == '/api/volume':
            self.send_json_response(self.handle_volume_update(data))
        else:
            self.send_404()

    def send_html_response(self, html_content):
        """发送HTML响应"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))

    def send_json_response(self, data):
        """发送JSON响应"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = json.dumps(data, indent=2, ensure_ascii=False)
        self.wfile.write(response.encode('utf-8'))

    def send_404(self):
        """发送404错误"""
        self.send_response(404)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        error = {"error": "Not found", "path": self.path}
        self.wfile.write(json.dumps(error).encode('utf-8'))

    def log_message(self, format, *args):
        """自定义日志格式"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {format % args}")

    def get_main_page(self):
        """返回主页面"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Vortex GPU Audio Backend</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 40px; }
        .status-card { background: #2a2a2a; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #00ff88; }
        .api-section { background: #333; padding: 20px; margin: 20px 0; border-radius: 8px; }
        .endpoint { background: #444; padding: 10px; margin: 10px 0; border-radius: 4px; font-family: monospace; }
        .button { background: #00ff88; color: #000; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin: 5px; }
        .button:hover { background: #00cc66; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
        .stat-item { background: #3a3a3a; padding: 15px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 24px; font-weight: bold; color: #00ff88; }
        .spectrum { height: 100px; background: linear-gradient(to right, #00ff88, #0088ff, #ff0088); border-radius: 4px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎵 Vortex GPU Audio Backend</h1>
            <h2>专业GPU加速音频处理系统</h2>
            <p>512频段均衡器 | 16M点卷积 | 实时频谱分析 | 多设备输出</p>
        </div>

        <div class="status-card">
            <h3>🔥 系统状态</h3>
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value" id="sample-rate">48000</div>
                    <div>采样率 (Hz)</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="bit-depth">32</div>
                    <div>位深 (bit)</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="channels">2</div>
                    <div>通道数</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="latency">12</div>
                    <div>延迟 (ms)</div>
                </div>
            </div>
        </div>

        <div class="status-card">
            <h3>📊 实时频谱</h3>
            <div class="spectrum" id="spectrum"></div>
            <button class="button" onclick="updateSpectrum()">更新频谱</button>
        </div>

        <div class="api-section">
            <h3>🔌 API接口</h3>
            <div class="endpoint">GET /api/health - 健康检查</div>
            <div class="endpoint">GET /api/status - 系统状态</div>
            <div class="endpoint">GET /api/capabilities - 功能特性</div>
            <div class="endpoint">GET /api/spectrum - 实时频谱数据</div>
            <div class="endpoint">POST /api/equalizer - 均衡器控制</div>
            <div class="endpoint">GET /api/devices - 输出设备</div>
            <button class="button" onclick="testAPI()">测试API</button>
        </div>

        <div class="status-card">
            <h3>⚡ 处理能力</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
                <div>
                    <h4>均衡器</h4>
                    <p>✅ 512频段图形均衡器</p>
                    <p>✅ GPU加速实时处理</p>
                    <p>✅ 多种滤波器类型</p>
                </div>
                <div>
                    <h4>卷积处理器</h4>
                    <p>✅ 16,000,000点最大长度</p>
                    <p>✅ 多FFT算法支持</p>
                    <p>✅ 零延迟优化</p>
                </div>
                <div>
                    <h4>输出设备</h4>
                    <p>✅ Roon Bridge集成</p>
                    <p>✅ HQPlayer NAA支持</p>
                    <p>✅ UPnP/DLNA渲染器</p>
                </div>
                <div>
                    <h4>高级特性</h4>
                    <p>✅ DSD1024支持</p>
                    <p>✅ 实时自动化</p>
                    <p>✅ 多线程处理</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        function updateSpectrum() {
            const spectrum = document.getElementById('spectrum');
            const bars = 50;
            let html = '';
            for (let i = 0; i < bars; i++) {
                const height = Math.random() * 100;
                html += `<div style="display: inline-block; width: 2%; background: #00ff88; height: ${height}%; margin: 0; vertical-align: bottom;"></div>`;
            }
            spectrum.innerHTML = html;
        }

        function testAPI() {
            fetch('/api/health')
                .then(response => response.json())
                .then(data => {
                    alert('API测试成功!\\n' + JSON.stringify(data, null, 2));
                })
                .catch(error => {
                    alert('API测试失败: ' + error);
                });
        }

        // 自动更新
        setInterval(updateSpectrum, 1000);
        updateSpectrum();
    </script>
</body>
</html>
        """

    def get_health_status(self):
        """健康检查状态"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime": "2 hours 34 minutes",
            "version": "1.0.0",
            "gpu_acceleration": True,
            "gpu_type": "NVIDIA CUDA",
            "memory_usage": "1.2GB / 8GB",
            "cpu_usage": "15%"
        }

    def get_system_status(self):
        """系统状态"""
        return {
            "audio": {
                "sample_rate": 48000,
                "bit_depth": 32,
                "channels": 2,
                "buffer_size": 512,
                "latency_ms": 12,
                "processing_mode": "REAL_TIME"
            },
            "gpu": {
                "acceleration_enabled": True,
                "backend": "CUDA",
                "memory_allocated": "1.2GB",
                "memory_total": "8GB",
                "compute_capability": "8.6",
                "cuda_version": "12.8"
            },
            "processing": {
                "equalizer": {
                    "bands": 512,
                    "enabled": True,
                    "gpu_accelerated": True
                },
                "convolution": {
                    "max_length": 16777216,
                    "current_length": 1048576,
                    "enabled": True
                },
                "spectrum_analyzer": {
                    "fft_size": 2048,
                    "update_rate": 60,
                    "enabled": True
                }
            },
            "outputs": {
                "roon_bridge": {"enabled": True, "connected": False},
                "hqplayer_naa": {"enabled": True, "connected": False},
                "upnp_renderer": {"enabled": True, "discovered_devices": 3}
            }
        }

    def get_capabilities(self):
        """功能特性"""
        return {
            "audio_formats": [
                {"format": "PCM", "max_sample_rate": 768000, "max_bit_depth": 32, "max_channels": 32},
                {"format": "DSD", "max_sample_rate": 45158000, "max_bit_depth": 1, "max_channels": 8},
                {"format": "DXD", "max_sample_rate": 352800, "max_bit_depth": 24, "max_channels": 8}
            ],
            "processing_features": {
                "equalizer": {
                    "bands": 512,
                    "filter_types": ["PEAK", "LOW_SHELF", "HIGH_SHELF", "BELL"],
                    "gpu_accelerated": True
                },
                "convolution": {
                    "max_length": 16777216,
                    "fft_methods": ["FFTW", "KISS_FFT", "OOURA"],
                    "real_time": True
                },
                "spectrum_analysis": {
                    "fft_size": 2048,
                    "frequency_resolution": 23.4,
                    "update_rate": 60
                }
            },
            "output_devices": {
                "roon_bridge": {"protocol": "RAAT", "airplay": True, "http_control": True},
                "hqplayer_naa": {"protocol": "TCP/UDP", "max_sample_rate": 1536000, "bit_depth": 64},
                "upnp_renderer": {"protocol": "DLNA/UPnP", "open_home": True, "media_server": True}
            }
        }

    def get_spectrum_data(self):
        """生成模拟频谱数据"""
        frequencies = []
        magnitudes = []

        # 生成20Hz到20kHz的对数频谱
        for i in range(512):
            freq = 20 * math.pow(1000, i / 511)  # 20Hz to 20kHz log scale
            # 模拟音乐频谱 - 低频较强，中频适中，高频较弱
            if freq < 200:
                magnitude = 0.8 + 0.2 * math.sin(i * 0.1) + random.uniform(-0.1, 0.1)
            elif freq < 2000:
                magnitude = 0.6 + 0.3 * math.sin(i * 0.05) + random.uniform(-0.05, 0.05)
            elif freq < 8000:
                magnitude = 0.4 + 0.2 * math.sin(i * 0.02) + random.uniform(-0.03, 0.03)
            else:
                magnitude = 0.2 + 0.1 * random.uniform(0, 1)

            frequencies.append(round(freq, 2))
            magnitudes.append(max(0, min(1, magnitude)))  # Clamp to [0, 1]

        return {
            "timestamp": datetime.now().isoformat(),
            "sample_rate": 48000,
            "fft_size": 2048,
            "frequencies": frequencies,
            "magnitudes": magnitudes,
            "peak_frequency": frequencies[magnitudes.index(max(magnitudes))],
            "rms_level": sum(m ** 2 for m in magnitudes) / len(magnitudes)
        }

    def get_equalizer_status(self):
        """均衡器状态"""
        bands = []
        for i in range(512):
            freq = 20 * math.pow(1000, i / 511)
            gain = random.uniform(-6, 6)  # 模拟均衡器设置
            bands.append({
                "band": i,
                "frequency": round(freq, 2),
                "gain": round(gain, 2),
                "q": 1.0,
                "filter_type": "PEAK",
                "enabled": True
            })

        return {
            "enabled": True,
            "bypass": False,
            "master_gain": 0.0,
            "bands": bands,
            "presets": ["flat", "rock", "jazz", "classical", "electronic"],
            "current_preset": "custom",
            "gpu_accelerated": True
        }

    def get_output_devices(self):
        """输出设备状态"""
        return {
            "devices": [
                {
                    "id": "default",
                    "name": "默认音频设备",
                    "type": "DIRECT_SOUND",
                    "max_channels": 8,
                    "max_sample_rate": 192000,
                    "latency_ms": 5,
                    "enabled": True,
                    "connected": True
                },
                {
                    "id": "roon-bridge-1",
                    "name": "Vortex Roon Bridge",
                    "type": "ROON_BRIDGE",
                    "max_channels": 8,
                    "max_sample_rate": 192000,
                    "latency_ms": 12,
                    "enabled": True,
                    "connected": False
                },
                {
                    "id": "hqplayer-naa-1",
                    "name": "Vortex HQPlayer NAA",
                    "type": "HQPLAYER_NAA",
                    "max_channels": 8,
                    "max_sample_rate": 1536000,
                    "latency_ms": 8,
                    "enabled": True,
                    "connected": False
                }
            ],
            "active_device": "default",
            "volume": 0.75,
            "muted": False
        }

    def handle_audio_upload(self, data):
        """处理音频上传"""
        return {
            "status": "success",
            "message": "音频文件上传成功",
            "file_id": f"audio_{int(time.time())}",
            "file_size": len(str(data)),
            "format": "auto-detected",
            "duration": "3:45",
            "sample_rate": 44100,
            "bit_depth": 16,
            "channels": 2
        }

    def handle_audio_process(self, data):
        """处理音频处理请求"""
        return {
            "status": "processing",
            "job_id": f"job_{int(time.time())}",
            "progress": 45,
            "stages": [
                {"name": "解码", "status": "completed", "time": "0.2s"},
                {"name": "均衡器", "status": "processing", "time": "0.1s"},
                {"name": "卷积", "status": "pending", "time": "0.0s"},
                {"name": "输出", "status": "pending", "time": "0.0s"}
            ],
            "estimated_total_time": "2.5s"
        }

    def handle_equalizer_update(self, data):
        """处理均衡器更新"""
        return {
            "status": "success",
            "message": "均衡器设置已更新",
            "bands_updated": len(data.get('bands', [])),
            "master_gain": data.get('master_gain', 0.0),
            "preset": data.get('preset', 'custom'),
            "processing_time": "0.001s"
        }

    def handle_volume_update(self, data):
        """处理音量更新"""
        volume = data.get('volume', 0.5)
        return {
            "status": "success",
            "volume": volume,
            "db_level": round(20 * math.log10(max(0.001, volume)), 2),
            "muted": data.get('muted', False),
            "device": data.get('device', 'default')
        }

def find_free_port():
    """找到可用端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def main():
    """主函数"""
    # 查找可用端口
    port = 8080
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', port))
    except OSError:
        port = find_free_port()

    print("🎵 Vortex GPU Audio Backend - 演示服务器")
    print("=" * 50)
    print(f"🚀 启动服务器在端口 {port}")
    print(f"🌐 访问地址: http://localhost:{port}")
    print(f"🔌 API地址: http://localhost:{port}/api")
    print()
    print("📋 可用的API端点:")
    print("  GET  /api/health - 健康检查")
    print("  GET  /api/status - 系统状态")
    print("  GET  /api/capabilities - 功能特性")
    print("  GET  /api/spectrum - 实时频谱数据")
    print("  GET  /api/equalizer - 均衡器状态")
    print("  GET  /api/devices - 输出设备")
    print("  POST /api/equalizer - 更新均衡器")
    print("  POST /api/volume - 设置音量")
    print()
    print("🔥 演示功能:")
    print("  ✅ 512频段GPU均衡器")
    print("  ✅ 实时频谱分析")
    print("  ✅ 多设备输出管理")
    print("  ✅ 专业音频API")
    print()
    print("按 Ctrl+C 停止服务器")
    print("=" * 50)

    try:
        server = HTTPServer(('localhost', port), VortexAPIHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
        server.shutdown()

if __name__ == '__main__':
    main()