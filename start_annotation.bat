@echo off
title 车道线标注Web工具
color 0A

echo.
echo ================================================================
echo                    车道线标注Web工具
echo                   支持iPad + Apple Pencil
echo ================================================================
echo.

cd /d "%~dp0"

echo 正在启动服务器...
python start_web_annotation.py

pause
