@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo Vortex GPU Audio Backend - Windows 环境检查
echo ===================================================
echo.

:: 检查 Windows 版本
echo 📋 系统信息:
for /f "tokens=*" %%i in ('ver') do echo   Windows: %%i
for /f "tokens=2 delims==" %%i in ('wmic os get TotalVisibleMemorySize /value ^| find "="') do (
    set /a ram_gb=%%i/1024/1024
    echo   内存: !ram_gb! GB
)
echo.

:: 检查必需的软件
echo 🔍 软件检查:

:: Git
where git >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=3" %%i in ('git --version') do echo   ✅ Git: %%i
) else (
    echo   ❌ Git: 未安装
)

:: Python
where python >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do echo   ✅ Python: %%i
) else (
    echo   ❌ Python: 未安装
)

:: Visual Studio
where cl >nul 2>&1
if %errorLevel% equ 0 (
    echo   ✅ Visual Studio C++: 已安装
) else (
    echo   ❌ Visual Studio C++: 未安装
)

:: CMake
where cmake >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=3" %%i in ('cmake --version ^| find "cmake version"') do echo   ✅ CMake: %%i
) else (
    echo   ❌ CMake: 未安装
)

:: Rust
where cargo >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=2" %%i in ('cargo --version') do echo   ✅ Rust: %%i
) else (
    echo   ❌ Rust: 未安装
)

echo.

:: 检查硬件信息
echo 💻 硬件检查:

:: CPU 信息
wmic cpu get name /value | find "Name=" > cpu_info.txt
set /p cpu_name=<cpu_info.txt
echo   CPU: %cpu_name:~6%
del cpu_info.txt

:: GPU 信息
wmic path win32_videocontroller get name /value | find "Name=" > gpu_info.txt
set /p gpu_name=<gpu_info.txt
echo   GPU: %gpu_name:~6%
del gpu_info.txt

:: NVIDIA GPU 检查
nvidia-smi >nul 2>&1
if %errorLevel% equ 0 (
    echo   ✅ NVIDIA 驱动: 已安装
    for /f "tokens=2" %%i in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader') do echo   NVIDIA 驱动版本: %%i
) else (
    echo   ❌ NVIDIA 驱动: 未安装或不可用
)

:: CUDA 检查
nvcc --version >nul 2>&1
if %errorLevel% equ 0 (
    echo   ✅ CUDA: 已安装
    for /f "tokens=4" %%i in ('nvcc --version ^| find "release"') do echo   CUDA 版本: %%i
) else (
    echo   ❌ CUDA: 未安装
)

echo.

:: 检查 vcpkg
echo 📦 包管理器检查:
if exist "C:\vcpkg\vcpkg.exe" (
    echo   ✅ vcpkg: 已安装
    for /f "tokens=*" %%i in ('C:\vcpkg\vcpkg.exe version') do echo   版本: %%i
) else (
    echo   ❌ vcpkg: 未安装
)

:: 检查项目文件
echo 📁 项目文件检查:
if exist "CMakeLists.txt" (
    echo   ✅ CMakeLists.txt: 存在
) else (
    echo   ❌ CMakeLists.txt: 不存在
)

if exist "Cargo.toml" (
    echo   ✅ Cargo.toml: 存在
) else (
    echo   ❌ Cargo.toml: 不存在
)

if exist "src\main.cpp" (
    echo   ✅ src\main.cpp: 存在
) else (
    echo   ❌ src\main.cpp: 不存在
)

:: 统计源文件
set cpp_count=0
set hpp_count=0
set test_count=0

for /r %%f in (*.cpp) do set /a cpp_count+=1
for /r %%f in (*.hpp) do set /a hpp_count+=1
for /r tests\%%f in (*.cpp) do set /a test_count+=1

echo   C++ 源文件: %cpp_count% 个
echo   C++ 头文件: %hpp_count% 个
echo   测试文件: %test_count% 个

echo.

:: 检查磁盘空间
for /f "tokens=3" %%i in ('dir /-c "%~dp0" ^| find "bytes free"') do set free_space=%%i
set /a free_gb=%free_space:~0,-9%
echo 💾 磁盘空间:
echo   可用空间: %free_gb% GB

if %free_gb% LSS 10 (
    echo   ⚠️  警告: 磁盘空间不足 10GB，可能影响构建
) else (
    echo   ✅ 磁盘空间充足
)

echo.

:: 计算环境准备度
set ready_score=0
set total_score=10

:: 基本工具
where git >nul 2>&1 && set /a ready_score+=1
where python >nul 2>&1 && set /a ready_score+=1
where cl >nul 2>&1 && set /a ready_score+=1
where cmake >nul 2>&1 && set /a ready_score+=1
where cargo >nul 2>&1 && set /a ready_score+=1

:: 硬件支持
nvidia-smi >nul 2>&1 && set /a ready_score+=1
nvcc --version >nul 2>&1 && set /a ready_score+=1

:: 项目文件
if exist "CMakeLists.txt" set /a ready_score+=1
if exist "src\main.cpp" set /a ready_score+=1
if exist "C:\vcpkg\vcpkg.exe" set /a ready_score+=1

:: 计算百分比
set /a readiness=ready_score*100/total_score

echo 📊 环境准备度: %readiness%%%

:: 给出建议
echo.
echo 💡 建议:

if %readiness% LSS 50 (
    echo   🔴 环境准备不足，需要安装多个依赖
    echo   建议运行 build_windows.bat 自动安装
) else if %readiness% LSS 80 (
    echo   🟡 环境基本准备就绪，可能需要完善一些配置
    echo   可以尝试运行 build_windows.bat 进行构建
) else (
    echo   🟢 环境准备充分，可以开始构建项目
    echo   运行 build_windows.bat 开始构建
)

echo.
echo 🛠️  推荐的下一步操作:
echo   1. 运行 build_windows.bat 进行完整构建
echo   2. 或参考 BUILD_WINDOWS.md 手动构建
echo   3. 查看 quick_start.md 了解快速启动选项
echo.

:: 检查是否可以立即运行
if %readistance% GEQ 70 (
    echo 🚀 环境良好，可以开始构建！
    echo.
    echo 是否现在运行构建脚本? (Y/N)
    set /p choice=请选择:

    if /i "%choice%"=="Y" (
        echo.
        echo 🔄 启动构建脚本...
        call build_windows.bat
    )
) else (
    echo 📝 请先完善环境后再运行构建
    echo.
    echo 查看 BUILD_WINDOWS.md 获取详细的安装指南
)

pause