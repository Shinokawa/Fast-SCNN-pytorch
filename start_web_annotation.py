#!/usr/bin/env python3
"""
车道线标注Web工具启动脚本
支持iPad + Apple Pencil标注
"""

import os
import sys
import webbrowser
import time
from threading import Timer

def open_browser():
    """延迟打开浏览器"""
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("🚀 启动车道线标注Web工具...")
    print("=" * 50)
    
    # 检查必要的目录
    input_dir = 'data/custom/images'
    output_dir = 'data/custom/masks'
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
        print(f"📁 已创建图片目录: {input_dir}")
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 已创建掩码目录: {output_dir}")
    
    print(f"📷 图片目录: {os.path.abspath(input_dir)}")
    print(f"🎭 掩码目录: {os.path.abspath(output_dir)}")
    print()
    
    # 延迟3秒后打开浏览器
    Timer(3.0, open_browser).start()
    
    # 启动Flask服务器
    from web_annotation_server import app
    import socket
    
    # 获取本机IP地址
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("🌐 服务器访问地址:")
    print(f"   本机访问: http://localhost:5000")
    print(f"   网络访问: http://{local_ip}:5000")
    print(f"   iPad访问: http://{local_ip}:5000")
    print()
    print("📱 使用说明:")
    print("   1. 将640x480的图片放入images文件夹")
    print("   2. 在iPad上打开上述网址")
    print("   3. 使用Apple Pencil绘制车道线")
    print("   4. 完成后点击保存，继续下一张")
    print()
    print("⌨️  快捷键:")
    print("   Ctrl+S: 保存当前标注")
    print("   Ctrl+Z: 撤销")
    print()
    print("🛑 按 Ctrl+C 停止服务器")
    print("=" * 50)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
        sys.exit(0)
