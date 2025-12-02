@echo off
echo ========================================
echo 验证本地Docker连接状态
echo ========================================
echo.

echo 1. 检查Docker守护进程:
docker version
echo.

echo 2. 检查Docker Engine状态:
docker info
echo.

echo 3. 检查本地镜像缓存:
echo 本地镜像数量:
docker images | find /c /v " "
echo.
echo 主要镜像:
docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}" | head -10
echo.

echo 4. 测试本地Docker功能:
echo 创建测试容器...
docker run --rm hello-world >nul 2>&1
if %errorLevel% equ 0 (
    echo ✅ 本地Docker功能正常
    echo hello-world 镜像缓存: ✓
) else (
    echo ❌ 本地Docker功能异常
)
echo.

echo 5. 检查Docker Hub连接:
echo 尝试拉取小镜像...
timeout 10 docker pull alpine:latest >nul 2>&1
if %errorLevel% equ 0 (
    echo ✅ Docker Hub连接正常
    docker rmi alpine:latest
) else (
    echo ❌ Docker Hub连接异常
    echo 这就是构建失败的根本原因
)
echo.

echo 6. 诊断网络问题:
echo 当前系统网络测试:
echo - ping registry-1.docker.io
echo - ping github.com
echo - ping google.com
echo.

echo ========================================
echo 结论分析
echo ========================================
echo.
echo 如果显示:
echo ✅ Docker版本正常
echo ✅ Docker守护进程正常
echo ✅ 本地镜像存在
echo ✅ hello-world 正常运行
echo ❌ Docker Hub连接失败
echo.
echo 则说明:
echo - 本地Docker Desktop 完全正常
echo - 问题在于外网连接Docker Hub
echo - 这是网络环境限制，不是Docker本身问题
echo.
echo 解决方案:
echo 1. 配置镜像加速器
echo 2. 使用本地Python演示服务器 (已运行)
echo 3. 等待网络环境改善
echo.

pause