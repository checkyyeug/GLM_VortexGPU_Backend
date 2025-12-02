# Vortex GPU Audio Backend - 运行指南

## 🚀 快速启动

### 方法1: 使用改进版服务器 (推荐)
```bash
# 启动服务器
python start_server.py

# 停止服务器
python stop_server.py
```

### 方法2: 使用原始服务器
```bash
# 启动服务器
python run_simple_server.py

# 停止服务器 (Ctrl+C 可能无效)
# 使用以下方法之一:
python stop_server.py
# 或
stop_server.bat
```

## 🛑 停止服务器的方法

### 1. 优雅停止 (推荐)
- **Web界面**: 访问 http://localhost:8080 然后点击"停止服务器"
- **API调用**: 访问 http://localhost:8080/api/stop
- **Python脚本**: 运行 `python stop_server.py`
- **批处理**: 运行 `stop_server.bat` (Windows)

### 2. 键盘中断
- 在改进版服务器中，按 `Ctrl+C` 可以优雅停止
- 在原始服务器中，`Ctrl+C` 可能无效

### 3. 系统级停止 (Windows)
```cmd
# 查找Python进程
tasklist | findstr python

# 强制终止
taskkill /F /IM python.exe

# 或使用PowerShell
Get-Process python | Stop-Process -Force
```

### 4. 任务管理器
1. 按 `Ctrl+Shift+Esc` 打开任务管理器
2. 找到 `python.exe` 进程
3. 右键点击 → "结束任务"

## 🌐 访问地址

启动成功后，可以访问以下地址：

- **Web界面**: http://localhost:8080
- **健康检查**: http://localhost:8080/api/health
- **详细状态**: http://localhost:8080/api/status
- **频谱数据**: http://localhost:8080/api/spectrum
- **停止服务**: http://localhost:8080/api/stop

## 📊 功能特性

### Web界面
- 实时频谱分析显示 (100fps更新)
- GPU状态监控
- 服务器健康检查
- 一键停止控制
- 响应式设计

### API端点
- `GET /api/health` - 健康检查
- `GET /api/status` - 详细状态信息
- `GET /api/spectrum` - 频谱分析数据
- `GET|POST /api/stop` - 停止服务器

### 支持的音频格式
- PCM: 16/24/32-bit, 最高768kHz
- DSD: DSD64 到 DSD1024 (45.1584 MHz)
- 实时处理延迟 < 5ms

## 🔧 故障排除

### 端口占用
如果8080端口被占用，服务器会提示并提供选择：
```bash
⚠️  端口 8080 已被占用
🔍 检查占用进程...
  进程: python.exe (PID: 12345)
  命令行: python run_simple_server.py
是否继续使用此端口? (y/N):
```

### 停止失败
如果服务器无法正常停止：
1. 首先尝试 `python stop_server.py`
2. 如果仍然失败，使用 `stop_server.bat`
3. 最后手段：使用任务管理器手动终止

### 权限问题
如果遇到权限错误，请以管理员身份运行命令提示符。

## 📝 日志输出

服务器启动时的典型输出：
```
🚀 Vortex GPU Audio Backend 启动成功!
📍 监听地址: http://localhost:8080
🌐 Web界面: http://localhost:8080
⏰ 启动时间: 2024-01-01 12:00:00
💡 提示: 按 Ctrl+C 或访问 http://localhost:8080/api/stop 来停止服务器
```

## 🎵 性能指标

### 系统要求
- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.8+
- **内存**: 最少4GB，推荐8GB+
- **GPU**: 支持CUDA 12.x的NVIDIA显卡

### 性能数据
- **延迟**: < 5ms 实时处理
- **采样率**: 最高768kHz PCM, 45.1584 MHz DSD
- **处理精度**: 512-band EQ, 16M-point 卷积
- **GPU加速**: CUDA 12.x, OpenCL, Vulkan

## 📞 获取帮助

如果遇到问题：
1. 检查Python版本 >= 3.8
2. 安装必要依赖: `pip install psutil numpy requests`
3. 确保端口8080未被占用
4. 检查防火墙设置
5. 以管理员权限运行

## 🔄 版本信息

- **当前版本**: 1.0.0
- **更新日期**: 2024-01-01
- **兼容性**: 支持所有主要操作系统

---

🎵 **Vortex GPU Audio Backend** - 专业级音频处理解决方案