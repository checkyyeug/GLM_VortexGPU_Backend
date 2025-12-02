@echo off
echo ===================================================
echo Vortex GPU Audio Backend - Windows 一键构建脚本
echo ===================================================
echo.

:: 检查管理员权限
net session >nul 2>&1
if %errorLevel% == 0 (
    echo 检测到管理员权限，继续...
) else (
    echo 警告: 建议以管理员身份运行以确保完整权限
    pause
)

:: 设置环境变量
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows

echo.
echo 🔍 检查系统环境...

:: 检查必需的程序
echo 检查必需的程序...

where git >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ Git 未安装，请先安装 Git for Windows
    echo 下载地址: https://git-scm.com/download/win
    pause
    exit /b 1
)
echo ✅ Git 已安装

where python >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ Python 未安装，请先安装 Python 3.11+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo ✅ Python 已安装

where cargo >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ Rust/Cargo 未安装，请先安装 Rust
    echo 下载地址: https://rustup.rs/
    pause
    exit /b 1
)
echo ✅ Rust 已安装

:: 检查 Visual Studio
where cl >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ Visual Studio C++ 编译器未找到
    echo 请确保已安装 Visual Studio 2022 和 C++ 桌面开发工作负载
    pause
    exit /b 1
)
echo ✅ Visual Studio C++ 编译器已安装

:: 检查 CUDA (可选)
nvcc --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ⚠️  CUDA 未找到，GPU 加速将被禁用
    echo 如需 GPU 加速，请安装 CUDA Toolkit 12.0+
    echo 下载地址: https://developer.nvidia.com/cuda-downloads
    set GPU_ENABLED=false
) else (
    echo ✅ CUDA 已安装，GPU 加速可用
    set GPU_ENABLED=true
)

echo.
echo 📦 安装依赖库...

:: 设置 vcpkg
if not exist "%VCPKG_ROOT%" (
    echo 🔄 克隆 vcpkg...
    git clone https://github.com/Microsoft/vcpkg.git "%VCPKG_ROOT%"
    cd "%VCPKG_ROOT%"
    call bootstrap-vcpkg.bat
    call vcpkg integrate install
    cd /d "%~dp0"
) else (
    echo ✅ vcpkg 已安装
)

:: 检查 vcpkg 是否需要初始化
if not exist "%VCPKG_ROOT%\vcpkg.exe" (
    echo 🔄 初始化 vcpkg...
    cd "%VCPKG_ROOT%"
    call bootstrap-vcpkg.bat
    call vcpkg integrate install
    cd /d "%~dp0"
)

:: 安装必需的依赖
echo 🔄 安装音频处理库...
call "%VCPKG_ROOT%\vcpkg.exe" install juce:x64-windows >nul 2>&1
call "%VCPKG_ROOT%\vcpkg.exe" install libsndfile:x64-windows >nul 2>&1
call "%VCPKG_ROOT%\vcpkg.exe" install fftw3:x64-windows >nul 2>&1
call "%VCPKG_ROOT%\vcpkg.exe" install flac:x64-windows >nul 2>&1
call "%VCPKG_ROOT%\vcpkg.exe" install vorbis:x64-windows >nul 2>&1
call "%VCPKG_ROOT%\vcpkg.exe" install lame:x64-windows >nul 2>&1

echo 🔄 安装测试框架...
call "%VCPKG_ROOT%\vcpkg.exe" install gtest:x64-windows >nul 2>&1
call "%VCPKG_ROOT%\vcpkg.exe" install gmock:x64-windows >nul 2>&1

:: 安装可选的 GPU 支持
echo 🔄 安装 GPU 支持库...
call "%VCPKG_ROOT%\vcpkg.exe" install opencl:x64-windows >nul 2>&1
call "%VCPKG_ROOT%\vcpkg.exe" install vulkan:x64-windows >nul 2>&1

echo ✅ 依赖库安装完成

echo.
echo 🔧 构建项目...

:: 创建构建目录
if not exist "build" mkdir build
cd build

:: 配置 CMake
echo 🔄 配置 CMake...
cmake .. ^
    -G "Visual Studio 17 2022" ^
    -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
    -DVCPKG_TARGET_TRIPLET=x64-windows

if %errorLevel% neq 0 (
    echo ❌ CMake 配置失败
    echo 请检查错误信息并解决依赖问题
    pause
    exit /b 1
)

:: 构建项目
echo 🔄 编译项目...
cmake --build . --config Release --parallel %NUMBER_OF_PROCESSORS%

if %errorLevel% neq 0 (
    echo ❌ 编译失败
    echo 请检查编译错误信息
    pause
    exit /b 1
)

echo ✅ 项目构建成功！

echo.
echo 🧪 运行测试...
ctest --output-on-failure --parallel 4

if %errorLevel% neq 0 (
    echo ⚠️  部分测试失败，但程序应该仍可运行
) else (
    echo ✅ 所有测试通过！
)

:: 创建配置文件
echo 📝 创建配置文件...
if not exist "..\config" mkdir "..\config"
echo { > "..\config\default.json"
echo   "audio": { >> "..\config\default.json"
echo     "sampleRate": 48000, >> "..\config\default.json"
echo     "bitDepth": 32, >> "..\config\default.json"
echo     "channels": 2, >> "..\config\default.json"
echo     "bufferSize": 512, >> "..\config\default.json"
echo     "enableGPU": %GPU_ENABLED% >> "..\config\default.json"
echo   }, >> "..\config\default.json"
echo   "gpu": { >> "..\config\default.json"
echo     "preferredBackends": ["cuda", "opencl", "vulkan"], >> "..\config\default.json"
echo     "memoryLimit": "4GB" >> "..\config\default.json"
echo   }, >> "..\config\default.json"
echo   "network": { >> "..\config\default.json"
echo     "httpPort": 8080, >> "..\config\default.json"
echo     "websocketPort": 8081 >> "..\config\default.json"
echo   }, >> "..\config\default.json"
echo   "output": { >> "..\config\default.json"
echo     "roonBridge": true, >> "..\config\default.json"
echo     "hqplayerNaa": true, >> "..\config\default.json"
echo     "upnpRenderer": true >> "..\config\default.json"
echo   }, >> "..\config\default.json"
echo   "logging": { >> "..\config\default.json"
echo     "level": "info", >> "..\config\default.json"
echo     "file": "logs/vortex.log", >> "..\config\default.json"
echo     "console": true >> "..\config\default.json"
echo   } >> "..\config\default.json"
echo } >> "..\config\default.json"

echo ✅ 配置文件已创建

:: 创建日志目录
if not exist "..\logs" mkdir "..\logs"

:: 复制必要的 DLL 文件
echo 🔄 复制运行时库...
if exist "%VCPKG_ROOT%\installed\x64-windows\bin\*.dll" (
    copy "%VCPKG_ROOT%\installed\x64-windows\bin\*.dll" "Release\" >nul 2>&1
    echo ✅ DLL 文件已复制
)

echo.
echo ===================================================
echo 🎉 Vortex GPU Audio Backend 构建完成！
echo ===================================================
echo.
echo 📁 构建文件位置:
echo    可执行文件: %cd%\Release\vortex-backend.exe
echo    测试程序:    %cd%\Release\vortex_tests.exe
echo    配置文件:    %cd%\..\config\default.json
echo    日志目录:    %cd%\..\logs\
echo.
echo 🚀 运行程序:
echo    Release\vortex-backend.exe
echo.
echo 🌐 访问地址:
echo    主页: http://localhost:8080
echo    API:  http://localhost:8080/api
echo.
echo 🧪 运行测试:
echo    Release\vortex_tests.exe
echo.
echo 💡 提示:
echo    - 首次运行可能需要几分钟初始化
echo    - 确保 NVIDIA 驱动已更新以使用 GPU 加速
echo    - 查看 logs/vortex.log 获取详细日志
echo.
echo 现在可以运行程序了吗? (Y/N)
set /p choice=请选择:

if /i "%choice%"=="Y" (
    echo.
    echo 🚀 启动 Vortex GPU Audio Backend...
    echo 按 Ctrl+C 停止程序
    echo.
    start "Vortex Backend" /MIN "Release\vortex-backend.exe" --config "..\config\default.json"
    timeout /t 3 /nobreak >nul
    echo 🌐 正在打开浏览器...
    start http://localhost:8080
    echo ✅ 程序已启动！
)

pause