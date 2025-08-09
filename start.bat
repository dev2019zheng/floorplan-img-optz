@echo off
echo 正在启动户型图透视矫正应用...
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python环境
    echo 请先安装Python 3.7或更高版本
    pause
    exit /b 1
)

REM 安装依赖
echo 正在安装依赖包...
pip install -r requirements.txt

REM 启动应用
echo.
echo 正在启动应用服务器...
echo 请在浏览器中访问: http://localhost:5000
echo.
python app.py
