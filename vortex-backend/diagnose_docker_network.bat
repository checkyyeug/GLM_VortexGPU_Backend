@echo off
echo ========================================
echo Docker网络问题诊断工具
echo ========================================
echo.

echo 1. 检查Docker服务状态:
docker --version
docker info
echo.

echo 2. 检查网络连接:
ping -n 4 registry-1.docker.io
ping -n 4 registry-2.docker.io
ping -n 4 hub.docker.io
echo.

echo 3. 检查DNS解析:
nslookup registry-1.docker.io
nslookup registry-2.docker.io
echo.

echo 4. 检查代理设置:
echo HTTP_PROXY: %HTTP_PROXY%
echo HTTPS_PROXY: %HTTPS_PROXY%
echo NO_PROXY: %NO_PROXY%
echo.

echo 5. 检查防火墙状态:
netsh advfirewall show allprofiles
echo.

echo 6. 测试Docker Hub连接:
docker pull hello-world 2>&1
if %errorLevel% equ 0 (
    echo ✅ Docker Hub连接正常
    docker rmi hello-world:latest
) else (
    echo ❌ Docker Hub连接失败
    echo 建议配置镜像加速器
)
echo.

echo 7. 检查Docker配置文件:
if exist "%USERPROFILE%\.docker\config.json" (
    echo ✅ Docker配置文件存在
    type "%USERPROFILE%\.docker\config.json"
) else (
    echo ❌ Docker配置文件不存在
)
echo.

echo 8. 建议的解决方案:
echo   1. 配置Docker镜像加速器
echo   2. 使用国内镜像源
echo   3. 检查网络代理设置
echo   4. 临时使用Python演示服务器
echo.

echo 当前运行的演示服务器:
curl -s http://localhost:8080/api/health 2>nul && echo ✅ Python演示服务器正常运行 || echo ❌ 演示服务器未运行

echo.
pause