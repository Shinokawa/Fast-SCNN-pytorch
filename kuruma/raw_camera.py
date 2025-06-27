import os
import cv2
import time
import numpy as np
from datetime import datetime
from flask import Flask, Response, render_template_string, request, jsonify
from threading import Thread, Lock
import json

# --- Flask App 和全局变量 ---
app = Flask(__name__)
current_frame = None  # 保存当前原始帧
frame_lock = Lock()   # 线程锁，保护帧数据

# --- 可配置常量 ---
CAMERA_INDEX = 0

# 数据保存目录
SAVE_DIR = "./raw_collected_data"
RAW_DIR = os.path.join(SAVE_DIR, "raw_images")
METADATA_FILE = os.path.join(SAVE_DIR, "metadata.json")

# 创建保存目录
os.makedirs(RAW_DIR, exist_ok=True)

# --- 摄像头读取循环 ---
def camera_loop():
    global current_frame
    
    # 使用 cv2.CAP_V4L2 后端可能会更稳定，但通常不是必须的
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: Cannot open camera at index {CAMERA_INDEX}.")
        return

    # 1. 设置视频编码格式为 MJPEG (最关键的一步！)
    # 'M', 'J', 'P', 'G' 构成了 FourCC (四字符代码)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    # 2. 设置期望的分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    # 3. 设置期望的帧率
    cap.set(cv2.CAP_PROP_FPS, 30)

    # --- 验证实际生效的参数 ---
    # 增加这部分代码来确认设置是否成功
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取实际的FOURCC并转换成字符串
    fourcc_get = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_get >> 8 * i) & 0xFF) for i in range(4)])
    
    print("\n--- 摄像头实际参数 ---")
    print(f"分辨率: {width}x{height}")
    print(f"帧率: {fps}")
    print(f"视频格式: {fourcc_str}")
    print("------------------------\n")
    
    # 检查帧率是否符合预期
    if fps < 25: # 小于25帧都算不符合预期
        print("警告: 摄像头帧率远低于预期！请检查硬件或驱动。")


    while True:
        ret, frame = cap.read()
        if not ret: 
            print("无法读取摄像头画面")
            time.sleep(1) # 等待一秒再试
            continue
        
        with frame_lock:
            current_frame = frame.copy()

# --- 数据保存函数 ---
def save_captured_data(description=""):
    global current_frame
    
    if current_frame is None:
        return False, "No frame available"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
    with frame_lock:
        raw_frame = current_frame.copy()
    
    # 保存原始图像
    raw_filename = f"raw_{timestamp}.jpg"
    raw_path = os.path.join(RAW_DIR, raw_filename)
    
    # 保存图像
    cv2.imwrite(raw_path, raw_frame)
    
    # 保存元数据
    metadata = {
        "timestamp": timestamp,
        "raw_image": raw_filename,
        "description": description,
        "capture_time": datetime.now().isoformat(),
        "image_size": {
            "width": raw_frame.shape[1],
            "height": raw_frame.shape[0],
            "channels": raw_frame.shape[2]
        }
    }
    
    # 读取现有元数据或创建新的
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            all_metadata = json.load(f)
    else:
        all_metadata = []
    
    all_metadata.append(metadata)
    
    # 保存更新后的元数据
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    
    return True, f"已保存: {raw_filename}"

# --- Web服务器部分 ---
def generate():
    global current_frame
    while True:
        if current_frame is None:
            continue
        
        with frame_lock:
            frame_to_encode = current_frame.copy()
        
        # 将帧编码为JPEG格式
        (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode)
        if not flag:
            continue
        # 以字节流形式产生输出
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>原始摄像头数据收集</title>
    <meta charset="UTF-8">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .video-container {
            text-align: center;
            margin: 20px 0;
        }
        #videoStream {
            max-width: 100%;
            border: 2px solid #ddd;
            border-radius: 5px;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        .btn {
            background-color: #007bff;
            color: white;
            padding: 15px 32px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 10px;
        }
        .btn:hover {
            background-color: #0056b3;
        }
        .btn:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .btn.capture {
            background-color: #28a745;
        }
        .btn.capture:hover {
            background-color: #218838;
        }
        .description-input {
            width: 300px;
            padding: 10px;
            margin: 0 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }
        .status {
            text-align: center;
            margin: 20px 0;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        .status.success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .stat-number {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
        .stat-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        .info-panel {
            background-color: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .info-panel h3 {
            margin-top: 0;
            color: #495057;
        }
        .info-panel p {
            margin: 5px 0;
            color: #6c757d;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📹 原始摄像头数据收集系统</h1>
        
        <div class="info-panel">
            <h3>🔍 系统信息</h3>
            <p>📊 <strong>模式:</strong> 原始摄像头数据（无AI推理）</p>
            <p>📂 <strong>保存路径:</strong> ./raw_collected_data/raw_images/</p>
            <p>⌨️ <strong>快捷键:</strong> 空格键快速拍照</p>
        </div>
        
        <div class="video-container">
            <img id="videoStream" src="{{ url_for('video_feed') }}" alt="Raw Camera Stream">
        </div>
        
        <div class="controls">
            <input type="text" id="description" class="description-input" placeholder="添加描述信息（可选）">
            <button id="captureBtn" class="btn capture" onclick="captureImage()">📸 拍照保存</button>
            <button id="refreshStats" class="btn" onclick="refreshStats()">🔄 刷新统计</button>
        </div>
        
        <div id="status" class="status" style="display: none;"></div>
        
        <div class="stats">
            <div class="stat-card">
                <div id="totalImages" class="stat-number">-</div>
                <div class="stat-label">总收集图像数</div>
            </div>
            <div class="stat-card">
                <div id="todayImages" class="stat-number">-</div>
                <div class="stat-label">今日收集数</div>
            </div>
            <div class="stat-card">
                <div id="lastCapture" class="stat-number">-</div>
                <div class="stat-label">最后拍照时间</div>
            </div>
        </div>
    </div>

    <script>
        function showStatus(message, isSuccess) {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
            statusDiv.className = isSuccess ? 'status success' : 'status error';
            statusDiv.style.display = 'block';
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 3000);
        }

        function captureImage() {
            const captureBtn = document.getElementById('captureBtn');
            const description = document.getElementById('description').value;
            
            captureBtn.disabled = true;
            captureBtn.textContent = '📸 正在保存...';
            
            fetch('/capture', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    description: description
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus(data.message, true);
                    document.getElementById('description').value = '';
                    refreshStats();
                } else {
                    showStatus(data.message, false);
                }
            })
            .catch(error => {
                showStatus('拍照失败: ' + error.message, false);
            })
            .finally(() => {
                captureBtn.disabled = false;
                captureBtn.textContent = '📸 拍照保存';
            });
        }

        function refreshStats() {
            fetch('/stats')
            .then(response => response.json())
            .then(data => {
                document.getElementById('totalImages').textContent = data.total_images;
                document.getElementById('todayImages').textContent = data.today_images;
                document.getElementById('lastCapture').textContent = data.last_capture || '无';
            })
            .catch(error => {
                console.error('获取统计信息失败:', error);
            });
        }

        // 页面加载时获取统计信息
        window.onload = function() {
            refreshStats();
        };

        // 键盘快捷键：空格键拍照
        document.addEventListener('keydown', function(event) {
            if (event.code === 'Space' && event.target.tagName !== 'INPUT') {
                event.preventDefault();
                captureImage();
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/capture", methods=['POST'])
def capture():
    try:
        data = request.get_json()
        description = data.get('description', '') if data else ''
        
        success, message = save_captured_data(description)
        
        return jsonify({
            'success': success,
            'message': message
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'拍照失败: {str(e)}'
        })

@app.route("/stats")
def stats():
    try:
        total_images = 0
        today_images = 0
        last_capture = None
        
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                all_metadata = json.load(f)
            
            total_images = len(all_metadata)
            
            today = datetime.now().strftime("%Y%m%d")
            today_images = sum(1 for item in all_metadata 
                             if item['timestamp'].startswith(today))
            
            if all_metadata:
                last_item = all_metadata[-1]
                last_capture = last_item['capture_time'][:19].replace('T', ' ')
        
        return jsonify({
            'total_images': total_images,
            'today_images': today_images,
            'last_capture': last_capture
        })
    except Exception as e:
        return jsonify({
            'total_images': 0,
            'today_images': 0,
            'last_capture': None
        })

# --- 主程序 ---
if __name__ == '__main__':
    print("正在启动原始摄像头数据收集系统...")
    
    # 在一个后台线程中启动摄像头循环
    t = Thread(target=camera_loop)
    t.daemon = True
    t.start()
    
    # 启动Flask Web服务器
    print("Web服务器启动中...")
    print("原始摄像头数据收集系统已启动！")
    print("请在浏览器中访问: http://<Your_Atlas_IP>:8002")
    print("快捷键: 空格键快速拍照")
    print(f"数据保存目录: {SAVE_DIR}")
    
    app.run(host='0.0.0.0', port=8002, threaded=True)
