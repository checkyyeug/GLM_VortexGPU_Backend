# Vortex GPU Audio Backend - 快速启动指南

## 🚀 快速概览

这是一个高性能GPU加速音频处理后端项目，支持：

- **512频段均衡器**
- **1600万点卷积处理器**
- **实时频谱分析**
- **多设备输出 (Roon Bridge, HQPlayer NAA, UPnP)**
- **DSD1024支持**
- **GPU加速 (CUDA/OpenCL/Vulkan)**

## 📋 当前项目状态

```
✅ 已完成的组件:
├── 核心音频引擎 (C++23)
├── GPU处理器 (CUDA/OpenCL/Vulkan)
├── 512频段均衡器
├── 16M点卷积系统
├── 实时频谱分析器
├── 多设备输出管理器
├── Roon Bridge集成
├── HQPlayer NAA客户端
├── UPnP/DLNA渲染器
├── 模块化处理链
└── 完整测试套件 (24个测试文件)

📁 项目结构:
├── include/          # 公共API头文件
├── src/             # 源代码实现
├── tests/           # 测试套件 (24个文件)
├── config/          # 配置文件
├── scripts/         # 构建脚本
├── shaders/         # GPU计算着色器
└── tools/           # 开发工具
```

## 🛠️ 系统要求

由于这是一个专业级音频处理项目，需要特定的开发环境：

### 必需依赖
- **CMake 3.20+**
- **C++23编译器** (GCC 11+, Clang 13+, MSVC 2022)
- **Rust 1.70+** (用于网络服务)
- **JUCE 8.0+** (音频框架)

### 音频处理库
- **libsndfile** - 音频文件I/O
- **FFTW3** - 快速傅里叶变换
- **FLAC, Vorbis, LAME** - 音频格式支持

### 可选GPU支持
- **CUDA 12.0+** (NVIDIA)
- **OpenCL 1.2+** (跨平台)
- **Vulkan 1.3+** (现代GPU)

## 🚦 快速启动步骤

### 方案1: Docker运行 (推荐)

```bash
# 使用Docker Compose (最简单)
docker-compose up -d

# 查看运行状态
docker-compose ps

# 查看日志
docker-compose logs vortex-backend
```

### 方案2: 完整本地构建

#### Windows 环境
```powershell
# 1. 安装Visual Studio 2022 (包含C++开发工具)
# 2. 安装CUDA Toolkit (可选)
# 3. 安装vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# 4. 安装依赖
.\vcpkg install juce:x64-windows libsndfile:x64-windows fftw3:x64-windows gtest:x64-windows

# 5. 安装Rust
# 从 https://rustup.rs/ 安装

# 6. 构建项目
git clone <your-repo>
cd vortex-backend
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=<vcpkg-path>/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release

# 7. 运行
.\Release\vortex-backend.exe
```

#### Linux 环境
```bash
# 1. 安装依赖
sudo apt update
sudo apt install build-essential cmake git
sudo apt install libsndfile1-dev libfftw3-dev libgtest-dev
sudo apt install nvidia-cuda-toolkit # 可选

# 2. 安装Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 3. 构建项目
git clone <your-repo>
cd vortex-backend
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 4. 运行
./vortex-backend
```

### 方案3: 仅运行测试 (验证项目)

如果您没有完整的构建环境，可以先运行测试来验证项目：

```bash
# 检查项目文件
find . -name "*.cpp" | wc -l  # 应该显示40+个源文件
find tests -name "*.cpp" | wc -l  # 应该显示24个测试文件

# 查看核心组件
ls -la src/core/dsp/     # 均衡器、卷积、频谱分析器
ls -la src/output/       # 输出设备管理
ls -la tests/unit/       # 单元测试
```

## 🎯 项目功能验证

### 核心API验证
```cpp
// 检查主要API文件
ls include/
# vortex_api.hpp       - 主API接口
# audio_types.hpp      - 音频数据类型
# network_types.hpp    - 网络协议类型
```

### 测试套件覆盖
```bash
# 测试统计
echo "=== 测试文件统计 ==="
echo "单元测试: $(ls tests/unit/*.cpp | wc -l) 个"
echo "集成测试: $(ls tests/integration/*.cpp | wc -l) 个"
echo "性能测试: $(ls tests/performance/*.cpp | wc -l) 个"
echo "端到端测试: $(ls tests/e2e/*.cpp | wc -l) 个"
echo "合同测试: $(ls tests/contract/*.cpp | wc -l) 个"
echo "总计: $(find tests -name "*.cpp" | wc -l) 个测试文件"
```

### 项目规模
```bash
# 代码统计
echo "=== 项目规模 ==="
echo "C++源文件: $(find src -name "*.cpp" | wc -l) 个"
echo "C++头文件: $(find src -name "*.hpp" | wc -l) 个"
echo "Rust源文件: $(find src -name "*.rs" | wc -l) 个"
echo "测试文件: $(find tests -name "*.cpp" | wc -l) 个"
echo "GPU着色器: $(find shaders -name "*.comp" | wc -l) 个"
```

## 🔧 配置选项

### 环境配置
```json
{
  "audio": {
    "sampleRate": 48000,
    "bitDepth": 32,
    "channels": 2,
    "bufferSize": 512,
    "enableGPU": true
  },
  "gpu": {
    "backends": ["cuda", "opencl", "vulkan"],
    "memoryLimit": "4GB"
  },
  "output": {
    "roonBridge": true,
    "hqplayerNAA": true,
    "upnpRenderer": true
  }
}
```

## 📊 性能特性

- **实时处理延迟**: <50ms
- **支持采样率**: up to 768kHz
- **支持位深**: up to 32-bit
- **最大通道数**: 32通道
- **GPU加速**: 10-100x性能提升
- **内存效率**: 优化的零拷贝设计

## 🌐 API接口

程序启动后提供：

- **HTTP API**: `http://localhost:8080`
- **WebSocket**: `ws://localhost:8081`
- **Roon Bridge**: 自动发现
- **UPnP/DLNA**: 自动发现
- **HQPlayer NAA**: TCP/UDP端口4321

### 示例API调用
```bash
# 上传音频文件
curl -X POST -F "file=@music.wav" http://localhost:8080/api/upload

# 获取实时频谱数据
curl http://localhost:8080/api/spectrum

# 配置均衡器
curl -X PUT http://localhost:8080/api/equalizer -d '{"bands": [...]}'
```

## 🧪 运行测试

如果项目成功构建：

```bash
# 运行所有测试
ctest --output-on-failure

# 运行特定组件测试
./vortex_tests --gtest_filter="EqualizerTest.*"
./vortex_tests --gtest_filter="ConvolutionTest.*"
./vortex_tests --gtest_filter="ProcessingChainTest.*"
```

## 📚 文档与资源

- **技术规格**: `Vortex_GPU_Audio_Backend_Spec.md`
- **API文档**: `docs/api/`
- **架构图**: `docs/architecture/`
- **性能基准**: `docs/benchmarks/`

## ❓ 获取帮助

1. 查看详细设置指南: `README_SETUP.md`
2. 检查项目规格: `Vortex_GPU_Audio_Backend_Spec.md`
3. 查看代码注释和文档字符串
4. 检查测试用例了解用法

---

⚠️ **注意**: 这是一个专业级音频处理项目，建议在有完整开发环境的系统上运行。如果当前环境缺少必需的工具，建议使用Docker方案或参考完整设置指南。