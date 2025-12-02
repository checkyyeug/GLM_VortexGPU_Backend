#!/usr/bin/env python3
"""
Vortex GPU Audio Backend - 环境检查工具
检查系统是否满足运行要求
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path

def run_command(cmd, capture_output=True):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output,
                              text=True, timeout=10)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "Command timeout"
    except Exception as e:
        return False, "", str(e)

def check_program(name, version_cmd=None):
    """检查程序是否安装"""
    success, output, error = run_command(f"which {name}" if platform.system() != "Windows" else f"where {name}")
    if success:
        if version_cmd:
            success, version, _ = run_command(version_cmd)
            return True, version if success else "Unknown version"
        return True, "Installed"
    return False, "Not found"

def check_library(name, check_cmd=None):
    """检查库是否可用"""
    if check_cmd:
        success, output, error = run_command(check_cmd)
        return success, output if success else error
    return False, "Not found"

def check_gpu_support():
    """检查GPU支持"""
    gpu_info = {}

    # 检查NVIDIA GPU
    success, output, _ = run_command("nvidia-smi")
    if success:
        gpu_info["nvidia"] = True
        # 解析GPU信息
        lines = output.split('\n')
        for line in lines:
            if "NVIDIA" in line and "Driver Version" in line:
                gpu_info["nvidia_driver"] = line.strip()
                break
    else:
        gpu_info["nvidia"] = False

    # 检查CUDA
    success, output, _ = run_command("nvcc --version")
    if success:
        gpu_info["cuda"] = True
        for line in output.split('\n'):
            if "release" in line.lower():
                gpu_info["cuda_version"] = line.strip()
    else:
        gpu_info["cuda"] = False

    # 检查OpenCL
    if platform.system() == "Linux":
        success, output, _ = run_command("clinfo")
        gpu_info["opencl"] = success
    else:
        gpu_info["opencl"] = "Unknown"

    return gpu_info

def check_project_structure():
    """检查项目结构"""
    project_root = Path(".")
    required_dirs = [
        "src/core",
        "src/dsp",
        "src/output",
        "src/network",
        "tests/unit",
        "tests/integration",
        "include",
        "config"
    ]

    required_files = [
        "CMakeLists.txt",
        "Cargo.toml",
        "src/main.cpp",
        "include/vortex_api.hpp"
    ]

    structure_info = {}

    # 检查目录
    structure_info["directories"] = {}
    for dir_path in required_dirs:
        exists = project_root / dir_path
        structure_info["directories"][dir_path] = exists.is_dir()

    # 检查文件
    structure_info["files"] = {}
    for file_path in required_files:
        exists = project_root / file_path
        structure_info["files"][file_path] = exists.is_file()

    # 统计代码文件
    cpp_files = list(project_root.rglob("*.cpp"))
    hpp_files = list(project_root.rglob("*.hpp"))
    test_files = list(project_root.rglob("tests/*.cpp"))

    structure_info["stats"] = {
        "cpp_files": len(cpp_files),
        "hpp_files": len(hpp_files),
        "test_files": len(test_files),
        "total_source_files": len(cpp_files) + len(hpp_files)
    }

    return structure_info

def main():
    """主检查函数"""
    print("🔍 Vortex GPU Audio Backend - 环境检查")
    print("=" * 50)
    print(f"系统: {platform.system()} {platform.release()}")
    print(f"架构: {platform.machine()}")
    print(f"Python: {sys.version}")
    print()

    # 检查基本工具
    print("🛠️  基本工具检查:")
    tools = {
        "Git": ("git --version"),
        "CMake": ("cmake --version"),
        "C++ Compiler": ("g++ --version" if platform.system() != "Windows" else "cl"),
        "Rust/Cargo": ("cargo --version"),
        "Python": ("python --version")
    }

    for tool, version_cmd in tools.items():
        installed, version = check_program(tool.lower().replace(" ", "").replace("/", ""), version_cmd)
        status = "✅" if installed else "❌"
        print(f"  {status} {tool}: {version}")
    print()

    # 检查音频库
    print("🎵 音频库检查:")
    audio_libs = {
        "JUCE": None,  # 需要特殊检查
        "libsndfile": ("pkg-config --modversion sndfile"),
        "FFTW3": ("pkg-config --modversion fftw3f"),
        "FLAC": ("pkg-config --modversion flac"),
        "Google Test": ("pkg-config --modversion gtest")
    }

    for lib, check_cmd in audio_libs.items():
        if lib == "JUCE":
            # 简单检查JUCE头文件
            juce_paths = [
                "/usr/local/include/JuceHeader.h",
                "/usr/include/JuceHeader.h",
                "C:/JUCE/modules/juce_audio_basics/juce_audio_basics.h"
            ]
            found = any(Path(p).exists() for p in juce_paths)
            status = "✅" if found else "❌"
            version = "Found" if found else "Not found"
        else:
            success, version = check_library(lib, check_cmd)
            status = "✅" if success else "❌"

        print(f"  {status} {lib}: {version}")
    print()

    # 检查GPU支持
    print("🎮 GPU支持检查:")
    gpu_info = check_gpu_support()

    for gpu_type, status in gpu_info.items():
        if isinstance(status, bool):
            icon = "✅" if status else "❌"
            text = "Available" if status else "Not available"
        else:
            icon = "ℹ️"
            text = status
        print(f"  {icon} {gpu_type.upper()}: {text}")
    print()

    # 检查项目结构
    print("📁 项目结构检查:")
    structure = check_project_structure()

    # 检查必需目录
    missing_dirs = [d for d, exists in structure["directories"].items() if not exists]
    if missing_dirs:
        print("  ❌ 缺失目录:")
        for d in missing_dirs:
            print(f"    - {d}")
    else:
        print("  ✅ 所有必需目录都存在")

    # 检查必需文件
    missing_files = [f for f, exists in structure["files"].items() if not exists]
    if missing_files:
        print("  ❌ 缺失文件:")
        for f in missing_files:
            print(f"    - {f}")
    else:
        print("  ✅ 所有必需文件都存在")

    # 显示代码统计
    stats = structure["stats"]
    print(f"  📊 代码统计:")
    print(f"    C++源文件: {stats['cpp_files']}")
    print(f"    C++头文件: {stats['hpp_files']}")
    print(f"    测试文件: {stats['test_files']}")
    print(f"    总源文件: {stats['total_source_files']}")
    print()

    # 总结和建议
    print("📋 总结:")

    # 评估环境准备度
    ready_score = 0
    total_checks = 0

    # 基本工具
    basic_tools_ready = all([
        "git" in tools and check_program("git")[0],
        "cmake" in tools and check_program("cmake")[0],
        "rust" in tools and check_program("cargo")[0]
    ])

    if basic_tools_ready:
        ready_score += 3
    total_checks += 3

    # 编译器
    compiler_ready = check_program("g++")[0] or check_program("cl")[0]
    if compiler_ready:
        ready_score += 1
    total_checks += 1

    # 音频库
    audio_libs_ready = len([lib for lib in audio_libs if lib != "JUCE" and check_library(lib, audio_libs[lib])[0]]) >= 2
    if audio_libs_ready:
        ready_score += 1
    total_checks += 1

    # 项目结构
    project_ready = len(missing_dirs) == 0 and len(missing_files) == 0
    if project_ready:
        ready_score += 1
    total_checks += 1

    readiness = (ready_score / total_checks) * 100

    print(f"  环境准备度: {readiness:.0f}%")
    print(f"  代码完整性: {'✅ 完整' if project_ready else '❌ 不完整'}")

    # 给出建议
    print("\n💡 建议:")

    if readiness < 50:
        print("  🔴 环境准备不足，需要安装更多依赖")
    elif readiness < 80:
        print("  🟡 环境基本准备好，建议完善一些配置")
    else:
        print("  🟢 环境准备良好，可以开始构建项目")

    if not basic_tools_ready:
        print("  📌 安装基本开发工具: Git, CMake, Rust")

    if not compiler_ready:
        print("  📌 安装C++编译器 (GCC/Clang/MSVC)")

    if not audio_libs_ready:
        print("  📌 安装音频处理库 (libsndfile, FFTW3等)")

    if not project_ready:
        print("  📌 确保项目文件完整")

    # 运行选项
    print(f"\n🚀 运行选项:")

    if readiness >= 80:
        print("  ✅ 可以尝试本地构建")
        print("     ./quick_start.md 中的构建步骤")

    print("  🐳 使用Docker (推荐)")
    print("     docker-compose up -d")

    print("  🧪 检查项目完整性")
    print("     python check_environment.py")

    print(f"\n检查完成! 时间戳: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}")

if __name__ == "__main__":
    main()