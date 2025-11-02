@echo off
chcp 65001 > nul
echo ============================================================
echo AI Pose Match - 系统托盘一键启动程序
echo ============================================================
echo.

REM Check if Python is available
python --version > nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.9或更高版本
    pause
    exit /b 1
)

echo [1/3] Python环境检查完成
echo.

REM Check if virtual environment exists
if exist ".venv\Scripts\activate.bat" (
    echo [2/3] 激活虚拟环境...
    call .venv\Scripts\activate.bat
) else (
    echo [2/3] 未找到虚拟环境，使用系统Python
)

echo [3/3] 启动程序...
echo.
echo ============================================================
echo.

REM Run the program
python run_with_tray.py

echo.
echo ============================================================
echo 程序已退出
echo ============================================================
pause

