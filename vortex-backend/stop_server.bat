@echo off
echo ========================================
echo 停止 Vortex GPU Audio Backend 服务器
echo ========================================
echo.

echo 1. 查找运行中的Python进程...
tasklist | findstr python.exe > temp_processes.txt

echo 2. 显示找到的进程:
type temp_processes.txt
echo.

echo 3. 尝试安全终止所有Python进程...
for /f "tokens=2" %%i in ('type temp_processes.txt ^| findstr python') do (
    echo 终止进程 PID: %%i
    taskkill /PID %%i
)

echo 4. 清理临时文件
del temp_processes.txt

echo.
echo 5. 检查端口8080是否仍被占用:
netstat -ano | findstr :8080
if %errorLevel% equ 0 (
    echo ❌ 端口8080仍被占用，尝试强制终止...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do (
        setlocal pid=%%d
        echo 强制终止PID: !pid!
        taskkill /F /PID !pid!
    )
) else (
    echo ✅ 端口8080已释放
)

echo.
echo 6. 验证服务已停止:
curl -s http://localhost:8080/api/health 2>nul
if %errorLevel% equ 0 (
    echo ❌ 服务仍在运行
    echo 使用 taskmgr 手动查找并终止进程
    start taskmgr
) else (
    echo ✅ 服务已成功停止
)

echo.
echo ========================================
echo 停止操作完成
echo ========================================
pause