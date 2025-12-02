# Vortex GPU Audio Backend - 运行指南

## 系统要求

### 基础环境
- **操作系统**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 10.15+
- **C++编译器**: GCC 11+ 或 Clang 13+ 或 MSVC 2022
- **CMake**: 3.20+
- **Rust**: 1.70+ (用于网络服务)

### 音频库依赖
- **JUCE 8.0+**: 音频框架
- **libsndfile**: 音频文件读写
- **FLAC**: FLAC格式支持
- **libvorbis**: OGG/Vorbis格式支持
- **LAME**: MP3编码支持
- **FFTW3**: 快速傅里叶变换

### GPU加速依赖 (可选)
- **CUDA**: 12.0+ (NVIDIA GPU)
- **OpenCL**: 1.2+ (跨平台GPU)
- **Vulkan**: 1.3+ (现代GPU计算)

### 开发工具
- **Git**: 版本控制
- **Google Test**: 单元测试框架

## 安装步骤

### Windows 安装

#### 1. 安装Visual Studio 2022
```powershell
# 下载并安装 Visual Studio 2022 Community
# 确保包含 "使用C++的桌面开发" 工作负载
# 包含以下组件:
# - MSVC v143编译器
# - Windows SDK (最新版本)
# - CMake工具
```

#### 2. 安装CUDA工具包 (可选)
```powershell
# 从 NVIDIA 官网下载 CUDA 12.x
# https://developer.nvidia.com/cuda-downloads
```

#### 3. 安装vcpkg包管理器
```powershell
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
```

#### 4. 安装依赖库
```powershell
# 安装音频库
.\vcpkg install juce:x64-windows
.\vcpkg install libsndfile:x64-windows
.\vcpkg install flac:x64-windows
.\vcpkg install libvorbis:x64-windows
.\vcpkg install lame:x64-windows
.\vcpkg install fftw3:x64-windows

# 安装测试框架
.\vcpkg install gtest:x64-windows
.\vcpkg install gmock:x64-windows

# 安装OpenCL (可选)
.\vcpkg install opencl:x64-windows
```

#### 5. 安装Rust
```powershell
# 从 https://rustup.rs/ 下载并安装 rustup
# 或者使用 PowerShell:
Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile "rustup-init.exe"
.\rustup-init.exe
```

### Linux 安装 (Ubuntu/Debian)

#### 1. 安装基础工具
```bash
sudo apt update
sudo apt install -y build-essential cmake git

# 安装Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

#### 2. 安装音频库
```bash
sudo apt install -y libsndfile1-dev libflac-dev libvorbis-dev libmp3lame-dev libfftw3-dev

# 安装JUCE (从源码编译)
git clone https://github.com/juce-framework/JUCE.git
cd JUCE
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install
cd ..
```

#### 3. 安装CUDA (可选)
```bash
# 从 NVIDIA 官网下载 CUDA 12.x
wget https://developer.download.nvidia.com/compute/cuda/12.x.0/local_installers/cuda_12.x.0_linux.run
sudo sh cuda_12.x.0_linux.run
```

#### 4. 安装测试框架
```bash
sudo apt install -y libgtest-dev libgmock-dev
cd /usr/src/googletest
sudo cmake .
sudo make
sudo make install
```

## 构建项目

### 1. 克隆并配置项目
```bash
git clone <repository-url>
cd vortex-backend
```

### 2. 构建Rust网络服务
```bash
cd src/network/rust
cargo build --release
cd ../../..
```

### 3. 使用CMake构建主项目
```bash
# 创建构建目录
mkdir build
cd build

# 配置CMake (Windows)
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg-root]/scripts/buildsystems/vcpkg.cmake

# 配置CMake (Linux)
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译项目
cmake --build . --config Release -j$(nproc)
```

### 4. 快速构建脚本
如果您有完整的环境，可以使用：

**Windows (PowerShell):**
```powershell
# 创建快速构建脚本
@'
# 构建Rust部分
Set-Location "src/network/rust"
cargo build --release
Set-Location "../../.."

# 构建主项目
if (!(Test-Path "build")) {
    New-Item -ItemType Directory -Path "build"
}
Set-Location "build"
cmake .. -DCMAKE_TOOLCHAIN_FILE="[vcpkg-path]/scripts/buildsystems/vcpkg.cmake" -A x64
cmake --build . --config Release
'@ | Out-File -FilePath "build.ps1" -Encoding UTF8

.\build.ps1
```

**Linux (Bash):**
```bash
#!/bin/bash
set -e

echo "Building Vortex GPU Audio Backend..."

# 构建Rust部分
echo "Building Rust network services..."
cd src/network/rust
cargo build --release
cd ../../..

# 创建构建目录
mkdir -p build
cd build

# 配置CMake
echo "Configuring CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
echo "Building project..."
make -j$(nproc)

echo "Build completed successfully!"
echo "Executable: build/vortex-backend"
echo "Tests: build/vortex_tests"
```

## 运行程序

### 1. 运行主程序
```bash
# 在build目录中
./vortex-backend

# 或者指定配置文件
./vortex-backend --config ../config/default.json
```

### 2. 运行测试
```bash
# 运行所有测试
ctest --output-on-failure

# 或者直接运行测试程序
./vortex_tests
```

### 3. 运行特定测试
```bash
# 运行单元测试
./vortex_tests --gtest_filter="UnitTests.*"

# 运行集成测试
./vortex_tests --gtest_filter="IntegrationTests.*"

# 运行性能测试
./vortex_tests --gtest_filter="PerformanceTests.*"
```

## 配置文件

### 默认配置 (config/default.json)
```json
{
  "audio": {
    "sampleRate": 48000,
    "bitDepth": 32,
    "channels": 2,
    "bufferSize": 512,
    "enableGPU": true,
    "gpuBackends": ["cuda", "opencl", "vulkan"]
  },
  "network": {
    "httpPort": 8080,
    "websocketPort": 8081,
    "maxConnections": 100
  },
  "gpu": {
    "preferredBackends": ["cuda", "opencl"],
    "memoryLimit": "4GB",
    "computeUnits": "auto"
  },
  "logging": {
    "level": "info",
    "file": "logs/vortex.log",
    "maxSize": "100MB"
  }
}
```

## Docker运行 (推荐)

### 1. 使用Docker Compose
```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f vortex-backend

# 停止服务
docker-compose down
```

### 2. 单独构建Docker镜像
```bash
# 构建镜像
docker build -t vortex-gpu-audio-backend .

# 运行容器 (需要GPU支持)
docker run --gpus all -p 8080:8080 -p 8081:8081 vortex-gpu-audio-backend
```

## 故障排除

### 常见问题

1. **CUDA未找到**
   ```
   解决方案: 确保CUDA安装正确，并设置了CUDA_HOME环境变量
   ```

2. **音频库找不到**
   ```
   解决方案: 检查pkg-config路径，或手动指定库路径
   ```

3. **Rust编译失败**
   ```
   解决方案: 更新Rust版本，检查网络连接
   ```

4. **GPU加速不工作**
   ```bash
   # 检查GPU状态
   nvidia-smi

   # 检查OpenCL设备
   clinfo
   ```

### 性能优化

1. **启用GPU加速**
   - 确保安装了对应的GPU驱动
   - 在配置文件中启用GPU选项

2. **优化编译**
   ```bash
   # 使用Release模式编译
   cmake -DCMAKE_BUILD_TYPE=Release ..

   # 启用链接时优化
   export CXXFLAGS="-flto"
   ```

3. **内存优化**
   - 调整bufferSize
   - 设置合理的GPU内存限制

## API接口

程序启动后，可以通过以下接口访问：

- **HTTP API**: http://localhost:8080
- **WebSocket**: ws://localhost:8081
- **REST文档**: http://localhost:8080/docs

### 示例API调用
```bash
# 上传音频文件
curl -X POST -F "file=@test.wav" http://localhost:8080/api/audio/upload

# 开始处理
curl -X POST http://localhost:8080/api/audio/process

# 获取处理状态
curl http://localhost:8080/api/audio/status
```

## 开发模式

### 启用开发模式
```bash
# 构建Debug版本
cmake -DCMAKE_BUILD_TYPE=Debug ..

# 启用详细日志
./vortex-backend --log-level debug
```

### 调试工具
```bash
# 使用GDB调试 (Linux)
gdb ./vortex-backend

# 使用Visual Studio调试 (Windows)
# 在VS中打开CMakeLists.txt目录
```

## 支持与帮助

- **文档**: `docs/` 目录
- **示例**: `examples/` 目录
- **问题反馈**: GitHub Issues
- **社区讨论**: 项目Discord频道

---

注意：这是一个专业级音频处理项目，建议在熟悉C++、Rust和音频处理的基础上进行部署和使用。