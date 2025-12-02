#!/usr/bin/env python3
"""
Vortex GPU Audio Backend - 服务器停止脚本
提供多种优雅停止方法
"""

import sys
import requests
import time
import subprocess
import signal
import psutil
import socket

def print_banner():
    """打印横幅"""
    print("=" * 70)
    print("🛑 VORTEX GPU AUDIO BACKEND - 服务器停止工具")
    print("=" * 70)
    print("")

def check_server_running(port=8080):
    """检查服务器是否运行"""
    try:
        # 尝试连接到服务器
        with socket.create_connection(('localhost', port), timeout=3):
            return True
    except:
        return False

def stop_via_api(port=8080):
    """通过API停止服务器"""
    print("🌐 尝试通过API优雅停止服务器...")
    try:
        response = requests.post(f'http://localhost:{port}/api/stop', timeout=5)
        if response.status_code == 200:
            print("✅ API停止请求已发送")
            return True
        else:
            print(f"❌ API停止请求失败: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器API")
        return False
    except Exception as e:
        print(f"❌ API停止失败: {e}")
        return False

def find_python_processes_on_port(port=8080):
    """查找占用端口的Python进程"""
    target_processes = []

    try:
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.status == 'LISTEN':
                try:
                    process = psutil.Process(conn.pid)
                    if 'python' in process.name().lower():
                        target_processes.append(process)
                        print(f"  🔍 找到Python进程: {process.name()} (PID: {conn.pid})")
                        print(f"     命令行: {' '.join(process.cmdline())}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except Exception as e:
        print(f"❌ 查找进程失败: {e}")

    return target_processes

def stop_processes_gracefully(processes):
    """优雅停止进程"""
    print("🤝 尝试优雅停止进程...")
    for process in processes:
        try:
            print(f"  🔄 向PID {process.pid} 发送SIGTERM信号...")
            process.terminate()

            # 等待进程退出
            try:
                process.wait(timeout=5)
                print(f"  ✅ PID {process.pid} 已优雅停止")
            except psutil.TimeoutExpired:
                print(f"  ⚠️  PID {process.pid} 未在5秒内退出，将强制终止")
                process.kill()
                print(f"  🔨 PID {process.pid} 已强制终止")

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"  ❌ 无法停止PID {process.pid}: {e}")

def stop_processes_forcefully(processes):
    """强制停止进程"""
    print("🔨 尝试强制停止进程...")
    for process in processes:
        try:
            print(f"  🔨 强制终止PID {process.pid}...")
            process.kill()
            time.sleep(1)
            if not process.is_running():
                print(f"  ✅ PID {process.pid} 已强制终止")
            else:
                print(f"  ⚠️  PID {process.pid} 仍在运行")
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"  ❌ 无法强制终止PID {process.pid}: {e}")

def stop_via_taskkill(port=8080):
    """使用系统命令停止进程"""
    print("🔧 尝试使用系统命令停止...")

    # Windows系统
    if sys.platform == 'win32':
        try:
            # 查找占用端口的进程
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True,
                text=True,
                timeout=10
            )

            pids = []
            for line in result.stdout.split('\n'):
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        if pid.isdigit():
                            pids.append(pid)

            # 终止进程
            for pid in set(pids):  # 去重
                print(f"  🔨 使用taskkill终止PID: {pid}")
                subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True)
                print(f"  ✅ PID {pid} 终止命令已发送")

        except Exception as e:
            print(f"❌ taskkill失败: {e}")

    # Linux/Mac系统
    else:
        try:
            # 使用lsof查找进程
            result = subprocess.run(
                ['lsof', '-ti', f':{port}'],
                capture_output=True,
                text=True,
                timeout=10
            )

            pids = result.stdout.strip().split('\n')
            pids = [pid for pid in pids if pid.isdigit()]

            # 终止进程
            for pid in pids:
                print(f"  🔄 向PID {pid} 发送SIGTERM...")
                subprocess.run(['kill', pid], capture_output=True)
                time.sleep(2)

                # 检查是否还在运行
                try:
                    result = subprocess.run(['kill', '-0', pid], capture_output=True)
                    if result.returncode == 0:
                        print(f"  🔨 向PID {pid} 发送SIGKILL...")
                        subprocess.run(['kill', '-9', pid], capture_output=True)
                except:
                    pass

        except Exception as e:
            print(f"❌ kill命令失败: {e}")

def verify_server_stopped(port=8080, max_wait=10):
    """验证服务器是否已停止"""
    print(f"⏳ 验证服务器状态 (最多等待{max_wait}秒)...")

    for i in range(max_wait):
        if not check_server_running(port):
            print("✅ 服务器已确认停止")
            return True
        time.sleep(1)
        print(f"  等待中... ({i+1}/{max_wait})")

    print("❌ 服务器仍在运行")
    return False

def main():
    """主函数"""
    print_banner()

    port = 8080

    # 检查服务器是否运行
    if not check_server_running(port):
        print("ℹ️  服务器未运行")
        return

    print(f"🔍 检测到端口 {port} 上有服务运行")
    print("")

    # 方法1: API优雅停止
    if stop_via_api(port):
        time.sleep(2)
        if verify_server_stopped(port):
            print("🎉 服务器已成功停止 (API方式)")
            return

    print("")

    # 方法2: 优雅停止进程
    processes = find_python_processes_on_port(port)
    if processes:
        stop_processes_gracefully(processes)
        time.sleep(2)
        if verify_server_stopped(port):
            print("🎉 服务器已成功停止 (优雅方式)")
            return

    print("")

    # 方法3: 强制停止进程
    if processes:
        stop_processes_forcefully(processes)
        time.sleep(2)
        if verify_server_stopped(port):
            print("🎉 服务器已成功停止 (强制方式)")
            return

    print("")

    # 方法4: 系统命令停止
    stop_via_taskkill(port)
    time.sleep(3)
    if verify_server_stopped(port):
        print("🎉 服务器已成功停止 (系统命令)")
        return

    print("")
    print("❌ 所有停止方法都失败了")
    print("💡 建议:")
    print("  1. 手动打开任务管理器查找Python进程")
    print("  2. 重启计算机")
    print("  3. 检查是否有其他服务占用端口")

if __name__ == '__main__':
    main()