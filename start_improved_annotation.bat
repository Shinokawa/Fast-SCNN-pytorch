@echo off
title 智能车道线标注工具 - 改进版
echo ================================
echo 智能车道线标注工具 - 改进版
echo ================================
echo.
echo 正在启动服务器...
echo.
echo 新功能:
echo - 自动车道检测
echo - 区域填充工具
echo - 批量处理操作
echo - 撤销/重做功能
echo - 图片去重处理
echo - 优化的前端界面
echo.

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python环境
    echo 请确保已安装Python并添加到PATH
    pause
    exit /b 1
)

REM 检查必要的包
python -c "import flask, cv2, numpy, PIL" >nul 2>&1
if errorlevel 1 (
    echo 正在安装必要的依赖包...
    pip install flask flask-cors opencv-python numpy pillow
)

REM 创建必要的目录
if not exist "data\custom\images" mkdir "data\custom\images"
if not exist "data\custom\masks" mkdir "data\custom\masks"
if not exist "static\uploads" mkdir "static\uploads"
if not exist "static\temp" mkdir "static\temp"

echo 目录结构已准备完成
echo.

REM 启动改进的Web服务器
echo 启动改进的Web标注服务器...
python improved_web_annotation.py

echo.
echo 服务器已关闭
pause
