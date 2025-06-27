import os
import cv2
import time
import numpy as np
from datetime import datetime
from ais_bench.infer.interface import InferSession
from flask import Flask, Response, render_template_string, request, jsonify
from threading import Thread, Lock
import json

# --- Flask App 和全局变量 ---
app = Flask(__name__)
output_frame = None
current_frame = None  # 保存当前原始帧，用于拍照
frame_lock = Lock()   # 线程锁，保护帧数据

# --- 可配置常量 ---
DEVICE_ID = 0
MODEL_PATH = "./weights/fast_scnn_tusimple_bs1.om"
MODEL_WIDTH = 1024
MODEL_HEIGHT = 768
CAMERA_INDEX = 0

# 数据保存目录
SAVE_DIR = "./collected_data"
RAW_DIR = os.path.join(SAVE_DIR, "raw_images")
PROCESSED_DIR = os.path.join(SAVE_DIR, "processed_images")
METADATA_FILE = os.path.join(SAVE_DIR, "metadata.json")

# 创建保存目录
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --- 预处理和后处理函数 ---
def preprocess(img_bgr):
    img_resized = cv2.resize(img_bgr, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_tensor - mean) / std
    img_transposed = img_normalized.transpose(2, 0, 1)
    input_data = np.ascontiguousarray(img_transposed[np.newaxis, :, :, :], dtype=np.float32)
    return input_data

def postprocess(output_tensor, original_width, original_height):
    pred_mask = np.argmax(output_tensor, axis=1).squeeze()
    vis_mask = (pred_mask * 255).astype(np.uint8)
    vis_mask_resized = cv2.resize(vis_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    return vis_mask_resized

# --- 核心推理循环 ---
def inference_loop():
    global output_frame, current_frame
    model = InferSession(DEVICE_ID, MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Cannot open camera at index {CAMERA_INDEX}.")
        return

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        with frame_lock:
            current_frame = frame.copy()  # 保存原始帧
        
        cam_height, cam_width = frame.shape[:2]
        input_data = preprocess(frame)
        outputs = model.infer([input_data])
        lane_mask = postprocess(outputs[0], cam_width, cam_height)
        
        lane_mask_color = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
        green_overlay = np.zeros_like(lane_mask_color)
        green_overlay[lane_mask > 0] = [0, 255, 0]
        result_frame = cv2.addWeighted(frame, 1.0, green_overlay, 0.5, 0)
        
        # 将处理好的帧存入全局变量
        with frame_lock:
            output_frame = result_frame.copy()

# --- 数据保存函数 ---
def save_captured_data(description=""):
    global current_frame, output_frame
    
    if current_frame is None or output_frame is None:
        return False, "No frame available"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
    with frame_lock:
        raw_frame = current_frame.copy()
        processed_frame = output_frame.copy()
    
    # 保存原始图像
    raw_filename = f"raw_{timestamp}.jpg"
    processed_filename = f"processed_{timestamp}.jpg"
    
    raw_path = os.path.join(RAW_DIR, raw_filename)
    processed_path = os.path.join(PROCESSED_DIR, processed_filename)
    
    # 保存图像
    cv2.imwrite(raw_path, raw_frame)
    cv2.imwrite(processed_path, processed_frame)
    
    # 保存元数据
    metadata = {
        "timestamp": timestamp,
        "raw_image": raw_filename,
        "processed_image": processed_filename,
        "description": description,
        "capture_time": datetime.now().isoformat()
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
    
    return True, f"Captured: {raw_filename}"

# --- Web服务器部分 ---
def generate():
    global output_frame
    while True:
        if output_frame is None:
            continue
        
        with frame_lock:
            frame_to_encode = output_frame.copy()
        
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
    <title>车道线检测数据收集</title>
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
            background-color: #4CAF50;
            color: white;
            padding: 15px 32px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 10px;
        }
        .btn:hover {
            background-color: #45a049;
        }
        .btn:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 车道线检测数据收集系统</h1>
        
        <div class="video-container">
            <img id="videoStream" src="{{ url_for('video_feed') }}" alt="Video Stream">
        </div>
        
        <div class="controls">
            <input type="text" id="description" class="description-input" placeholder="添加描述信息（可选）">
            <button id="captureBtn" class="btn" onclick="captureImage()">📸 拍照保存</button>
            <button id="refreshStats" class="btn" onclick="refreshStats()" style="background-color: #007bff;">🔄 刷新统计</button>
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
    print("正在启动车道线检测数据收集系统...")
    
    # 在一个后台线程中启动推理循环
    t = Thread(target=inference_loop)
    t.daemon = True
    t.start()
    
    # 启动Flask Web服务器
    print("Web服务器启动中...")
    print("数据收集系统已启动！")
    print("请在浏览器中访问: http://<Your_Atlas_IP>:8001")
    print("快捷键: 空格键快速拍照")
    print(f"数据保存目录: {SAVE_DIR}")
    
    app.run(host='0.0.0.0', port=8001, threaded=True)
