# Vortex GPU Audio Backend - Windows 11 完整本地构建指南

## 📋 系统要求

### 硬件要求
- **CPU**: Intel Core i7 或 AMD Ryzen 7 及以上
- **内存**: 16GB RAM (推荐 32GB)
- **存储**: 50GB 可用空间
- **GPU**: NVIDIA GPU (支持CUDA 12.0+), AMD GPU (支持OpenCL 1.2+), 或 Intel GPU (支持Vulkan 1.3+)

### 软件要求
- **Windows 11** (版本 22H2 或更高)
- **Visual Studio 2022** (Community, Professional, 或 Enterprise)
- **Git** (最新版本)
- **Python 3.11+** (用于脚本)
- **CUDA Toolkit 12.0+** (可选，用于GPU加速)
- **Docker Desktop** (可选，用于容器化)

## 🚀 快速开始

### 第一步：安装 Visual Studio 2022

1. **下载 Visual Studio 2022**
   - 访问: https://visualstudio.microsoft.com/zh-hans/downloads/
   - 下载 "Visual Studio 2022 Community" (免费)

2. **安装配置**
   ```
   在安装器中选择以下工作负载：
   ✅ 使用 C++ 的桌面开发
   ✅ Windows 10/11 SDK (最新版本)
   ✅ MSVC v143 - C++ 生成工具
   ✅ CMake 工具
   ✅ Git for Windows (如果未安装)
   ```

3. **可选组件**
   ```
   在"单个组件"中勾选：
   ✅ C++ CMake 工具
   ✅ Windows 11 SDK (10.0.22621.0 或更高)
   ✅ NuGet 包管理器
   ✅ 适用于 Windows 的 Python 3.11 (如果需要)
   ```

### 第二步：安装 Git

1. **下载并安装 Git**
   - 访问: https://git-scm.com/download/win
   - 下载 Git for Windows
   - 使用默认配置安装

2. **验证安装**
   ```cmd
   git --version
   ```
   应该显示类似: `git version 2.48.1.windows.1`

### 第三步：安装 CUDA Toolkit (可选，用于GPU加速)

1. **检查GPU兼容性**
   - NVIDIA GPU: 计算能力 6.0+ (GTX 10系列及以上)
   - 检查方法: https://developer.nvidia.com/cuda-gpus

2. **下载 CUDA Toolkit**
   - 访问: https://developer.nvidia.com/cuda-downloads
   - 选择: Windows -> x86_64 -> 11 -> exe (local)

3. **安装步骤**
   ```
   1. 下载 CUDA 12.8 (或最新版本)
   2. 运行安装程序，选择"自定义安装"
   3. 确保勾选:
      ✅ CUDA Toolkit
      ✅ CUDA Runtime
      ✅ CUDA Development
      ✅ Visual Studio Integration
      ✅ Nsight Compute
   ```

4. **验证安装**
   ```cmd
   nvcc --version
   nvidia-smi
   ```

### 第四步：安装 vcpkg (包管理器)

1. **克隆 vcpkg**
   ```cmd
   cd C:\
   git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg
   ```

2. **初始化 vcpkg**
   ```cmd
   .\bootstrap-vcpkg.bat
   ```

3. **集成到 Visual Studio**
   ```cmd
   .\vcpkg integrate install
   ```

### 第五步：安装 Rust

1. **下载 rustup**
   - 访问: https://rustup.rs/
   - 下载 `rustup-init.exe`

2. **安装 Rust**
   ```cmd
   rustup-init.exe
   ```
   按提示选择默认选项

3. **配置环境变量**
   ```cmd
   # 添加到系统 PATH (如果安装程序未自动添加)
   C:\Users\%USERNAME%\.cargo\bin
   ```

4. **验证安装**
   ```cmd
   cargo --version
   rustc --version
   ```

## 🔧 构建依赖库

### 安装必需依赖

1. **基础音频库**
   ```cmd
   cd C:\vcpkg
   .\vcpkg install juce:x64-windows
   .\vcpkg install libsndfile:x64-windows
   .\vcpkg install fftw3:x64-windows
   .\vcpkg install flac:x64-windows
   .\vcpkg install vorbis:x64-windows
   .\vcpkg install lame:x64-windows
   ```

2. **测试框架**
   ```cmd
   .\vcpkg install gtest:x64-windows
   .\vcpkg install gmock:x64-windows
   ```

3. **GPU 支持库 (可选)**
   ```cmd
   .\vcpkg install opencl:x64-windows
   .\vcpkg install vulkan:x64-windows
   ```

### 验证依赖安装
```cmd
# 检查主要库
.\vcpkg list | findstr -i "juce\|sndfile\|fftw\|gtest"
```

## 📁 项目构建

### 第一步：获取源代码

1. **克隆项目**
   ```cmd
   cd D:\workspaces
   git clone <your-repository-url> VortexGPU_Backend
   cd VortexGPU_Backend\vortex-backend
   ```

### 第二步：配置 CMake

1. **创建构建目录**
   ```cmd
   mkdir build
   cd build
   ```

2. **配置 CMake (64位 Release)**
   ```cmd
   cmake .. ^
     -G "Visual Studio 17 2022" ^
     -A x64 ^
     -DCMAKE_BUILD_TYPE=Release ^
     -DCMAKE_TOOLCHAIN_FILE="C:\vcpkg\scripts\buildsystems\vcpkg.cmake" ^
     -DVCPKG_TARGET_TRIPLET=x64-windows
   ```

3. **如果配置成功，会看到类似输出:**
   ```
   -- Build files have been written to: D:/workspaces/VortexGPU_Backend/vortex-backend/build
   ```

### 第三步：编译项目

1. **完整编译**
   ```cmd
   cmake --build . --config Release --parallel 8
   ```

2. **如果编译成功，应该看到:**
   ```
   Build finished with exit code 0
   ```

### 第四步：验证构建

1. **检查生成的可执行文件**
   ```cmd
   dir Release\
   ```
   应该看到 `vortex-backend.exe`

2. **检查库文件**
   ```cmd
   dir Release\*.dll
   ```

## 🧪 运行测试

### 运行单元测试
```cmd
cd build
ctest --output-on-failure --parallel 4
```

### 运行特定测试套件
```cmd
# 运行所有测试
.\Release\vortex_tests.exe

# 运行特定测试组
.\Release\vortex_tests.exe --gtest_filter="EqualizerTest.*"
.\Release\vortex_tests.exe --gtest_filter="ConvolutionTest.*"
.\Release\vortex_tests.exe --gtest_filter="ProcessingChainTest.*"
```

### 测试预期结果
```
[==========] Running 124 tests from 24 test suites.
[----------] Global test environment set-up.
[----------] 24 tests from EqualizerTest
[ RUN      ] EqualizerTest.InitializeWithValidConfig
[       OK ] EqualizerTest.InitializeWithValidConfig (1 ms)
...
[----------] 24 tests from EqualizerTest (15 ms total)
[==========] 124 tests from 24 test suites ran. (234 ms total)
[  PASSED  ] 124 tests.
```

## 🎵 运行程序

### 第一次运行
```cmd
cd D:\workspaces\VortexGPU_Backend\vortex-backend\build\Release
.\vortex-backend.exe --help
```

### 运行主程序
```cmd
# 使用默认配置
.\vortex-backend.exe

# 使用自定义配置
.\vortex-backend.exe --config ..\..\config\default.json

# 启用调试模式
.\vortex-backend.exe --log-level debug --console
```

### 预期输出
```
🎵 Vortex GPU Audio Backend Starting...
Version: 1.0.0
Build Date: Dec  2 2025
C++ Standard: 202311

Configuration loaded successfully
Sample Rate: 48000 Hz
Bit Depth: 32 bits
Channels: 2
GPU Acceleration: Enabled

Audio engine initialized successfully
GPU acceleration enabled (CUDA 12.8)
Vortex GPU Audio Backend is running...
Press Ctrl+C to stop
```

## 🔧 配置文件

### 创建默认配置
```cmd
mkdir config
# config/default.json 会自动生成
```

### 示例配置文件
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
    "preferredBackends": ["cuda", "opencl", "vulkan"],
    "memoryLimit": "4GB"
  },
  "network": {
    "httpPort": 8080,
    "websocketPort": 8081
  },
  "output": {
    "roonBridge": true,
    "hqplayerNAA": true,
    "upnpRenderer": true
  }
}
```

## 🌐 API 访问

### HTTP API 接口
程序启动后可以通过以下地址访问：

- **主页**: http://localhost:8080
- **健康检查**: http://localhost:8080/api/health
- **系统状态**: http://localhost:8080/api/status
- **实时频谱**: http://localhost:8080/api/spectrum

### WebSocket 接口
- **WebSocket**: ws://localhost:8081

### 测试 API
```cmd
# 健康检查
curl http://localhost:8080/api/health

# 系统状态
curl http://localhost:8080/api/status

# 获取频谱数据
curl http://localhost:8080/api/spectrum
```

## 🔍 故障排除

### 常见问题及解决方案

#### 1. CMake 配置失败
**问题**: `Could not find a package configuration file provided by "juce"`

**解决方案**:
```cmd
# 确认 vcpkg 安装正确
cd C:\vcpkg
.\vcpkg install juce:x64-windows
.\vcpkg integrate install

# 清理 CMake 缓存重新配置
cd D:\workspaces\VortexGPU_Backend\vortex-backend\build
del CMakeCache.txt
cmake ... (重新运行配置命令)
```

#### 2. 编译链接错误
**问题**: `无法解析的外部符号 __imp_*`

**解决方案**:
```cmd
# 检查 vcpkg 库是否正确安装
.\vcpkg list | findstr juce

# 手动指定库路径
cmake .. -DCMAKE_PREFIX_PATH="C:\vcpkg\installed\x64-windows"
```

#### 3. CUDA 相关错误
**问题**: `CUDA not found` 或 `nvcc not recognized`

**解决方案**:
```cmd
# 检查 CUDA 安装
where nvcc
nvcc --version

# 添加 CUDA 到 PATH
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%

# 检查环境变量
echo %CUDA_PATH%
echo %CUDA_PATH_V12_8%
```

#### 4. Rust 编译错误
**问题**: `cargo not recognized`

**解决方案**:
```cmd
# 检查 Rust 安装
where cargo
cargo --version

# 重新配置 PATH
set PATH=C:\Users\%USERNAME%\.cargo\bin;%PATH%
```

#### 5. 运行时 DLL 缺失
**问题**: `缺少 VCRUNTIME140.dll 或其他 DLL`

**解决方案**:
```cmd
# 安装 Visual C++ Redistributable
# 下载: https://aka.ms/vs/17/release/vc_redist.x64.exe

# 或者从 vcpkg 复制 DLL 到输出目录
copy C:\vcpkg\installed\x64-windows\bin\*.dll Release\
```

#### 6. GPU 加速不工作
**问题**: `GPU acceleration failed`

**解决方案**:
```cmd
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 CUDA 版本兼容性
nvcc --version

# 禁用 GPU 加速 (如果不需要)
修改 config.json: "enableGPU": false
```

### 性能优化建议

#### 1. 编译优化
```cmd
# 使用最大优化
cmake .. -DCMAKE_BUILD_TYPE=Release ^
          -DCMAKE_CXX_FLAGS="/O2 /GL" ^
          -DCMAKE_EXE_LINKER_FLAGS="/LTCG"

# 启用并行编译
cmake --build . --config Release --parallel %NUMBER_OF_PROCESSORS%
```

#### 2. 内存优化
```cmd
# 修改 config.json
{
  "audio": {
    "bufferSize": 1024,  // 增加缓冲区大小
    "enableGPU": true    // 确保 GPU 加速启用
  },
  "gpu": {
    "memoryLimit": "6GB" // 根据可用内存调整
  }
}
```

## 📚 开发环境配置

### Visual Studio 调试配置

1. **启动调试**
   - 在 Visual Studio 中打开 `CMakeLists.txt`
   - 设置 `vortex-backend` 为启动项
   - 配置调试参数: `--config ../../config/default.json`

2. **断点调试**
   - 在 `src/main.cpp` 中设置断点
   - 按 F5 开始调试
   - 使用 Visual Studio 调试器查看变量和内存

### 代码分析工具

1. **Clang-Tidy 集成**
   ```cmd
   # 在 CMakeLists.txt 中添加
   set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-checks=*")
   ```

2. **代码覆盖率**
   ```cmd
   # 启用覆盖率
   cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON
   ```

## 📦 打包部署

### 创建安装包

1. **使用 Visual Studio Installer**
   ```cmd
   # 在 Visual Studio 中:
   # 1. 右键项目 -> 添加 -> 新建项目
   # 2. 选择 "Setup Project"
   # 3. 添加必要的文件和注册表项
   ```

2. **便携式版本**
   ```cmd
   mkdir VortexBackend_Portable
   copy Release\vortex-backend.exe VortexBackend_Portable\
   copy Release\*.dll VortexBackend_Portable\
   xcopy config VortexBackend_Portable\config\ /E /I
   ```

### 系统服务安装

1. **创建 Windows 服务**
   ```cmd
   # 使用 NSSM (Non-Sucking Service Manager)
   nssm install "VortexBackend" "D:\VortexBackend\vortex-backend.exe"
   nssm set "VortexBackend" Start SERVICE_AUTO_START
   ```

## 🎉 构建成功验证

### 最终检查清单

- [ ] 所有依赖库正确安装
- [ ] CMake 配置无错误
- [ ] 编译完成，无警告和错误
- [ ] 所有测试通过 (124个测试)
- [ ] 主程序可以正常启动
- [ ] API 接口可以正常访问
- [ ] GPU 加速功能正常 (如果可用)
- [ ] 音频处理功能正常

### 成功运行标志
```
🎵 Vortex GPU Audio Backend Starting...
✅ Configuration loaded successfully
✅ Audio engine initialized successfully
✅ GPU acceleration enabled
✅ All systems operational
🚀 Vortex GPU Audio Backend is running on http://localhost:8080
```

## 📞 技术支持

### 获取帮助

1. **项目文档**: `docs/` 目录
2. **示例代码**: `examples/` 目录
3. **API 文档**: http://localhost:8080/docs (程序运行时)
4. **GitHub Issues**: 项目仓库 Issues 页面

### 日志和诊断

1. **查看日志文件**
   ```
   logs/vortex.log          # 主程序日志
   logs/audio.log           # 音频处理日志
   logs/gpu.log             # GPU 相关日志
   ```

2. **启用详细日志**
   ```cmd
   vortex-backend.exe --log-level trace --file logs/detailed.log
   ```

---

🎉 恭喜！您已经成功构建了 Vortex GPU Audio Backend！这是一个专业级的高性能音频处理系统，支持 GPU 加速、实时处理和多设备输出。享受您的音频处理之旅！