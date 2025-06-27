"""
车道线检测与PID控制集成脚本 (端到端FP16版本)

功能:
- 使用端到端ONNX模型进行实时车道线检测 (FP16输入优化)。
- 根据检测结果计算车辆横向偏移。
- 使用PID控制器计算转向角度。
- 根据偏移量动态调整速度。
- 提供一个带有实时PID参数调试功能的Web UI。
"""
import os
import cv2
import time
import numpy as np
from threading import Thread, Lock
import queue
import psutil
import subprocess
import re

from ais_bench.infer.interface import InferSession
from flask import Flask, Response, render_template_string, jsonify, request

# --- PID控制器 ---
class PIDController:
    """一个简单的PID控制器"""
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

    def update(self, current_value):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: return 0

        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        self.prev_error = error
        self.last_time = current_time
        return output

    def reset(self):
        self.integral = 0
        self.prev_error = 0
        self.last_time = time.time()

# --- Flask App 和全局共享数据 ---
app = Flask(__name__)
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)

# 使用一个字典来统一管理所有共享数据
shared_data = {
    # 性能统计
    "fps": "0.0", "pipeline_latency": "0.0", "inference_time": "0.0",
    "cpu_percent": "0.0", "mem_percent": "0.0", "npu_util": "N/A", "npu_mem": "N/A",
    # 控制信号
    "steering_angle": 0.0, "speed": 0.0, "error": 0.0,
    # PID参数
    "pid_p": 0.4, "pid_i": 0.0, "pid_d": 0.2
}
data_lock = Lock()

# --- 可配置常量 ---
DEVICE_ID = 0
# 注意：此路径应指向经过 --input_fp16_nodes 参数转换后的FP16输入模型
MODEL_PATH = "./weights/fast_scnn_480x640_e2e_fp16_input.om"
MODEL_WIDTH = 640
MODEL_HEIGHT = 480
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
# FP16模型需要输入np.float16类型的数据
MODEL_INPUT_DTYPE = np.float16
NPU_SMI_PATH = "/usr/local/Ascend/driver/tools/npu-smi"

# --- 初始化PID控制器 ---
pid_steer = PIDController(shared_data["pid_p"], shared_data["pid_i"], shared_data["pid_d"])

# --- 预处理函数 ---
def preprocess_end_to_end(img_bgr, dtype=np.float32):
    """端到端模型的极简预处理"""
    if img_bgr.shape[1] != MODEL_WIDTH or img_bgr.shape[0] != MODEL_HEIGHT:
        img_resized = cv2.resize(img_bgr, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img_bgr
    
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_transposed = np.transpose(img_rgb, (2, 0, 1))
    input_data = np.ascontiguousarray(img_transposed[np.newaxis, :, :, :], dtype=dtype)
    return input_data

# --- 后处理函数 ---
def postprocess(output_tensor):
    """从模型输出中提取车道线掩码"""
    pred_mask = np.argmax(output_tensor, axis=1).squeeze()
    return (pred_mask * 255).astype(np.uint8)

# --- 核心控制逻辑 ---
def calculate_control_signals(lane_mask):
    """根据车道线掩码计算转向和速度"""
    h, w = lane_mask.shape
    
    # 1. 定义一个ROI (Region of Interest)，我们更关心车辆前方的车道位置
    roi_y_start = h * 3 // 4 
    roi = lane_mask[roi_y_start:, :]

    # 2. 计算ROI内车道线质心
    M = cv2.moments(roi)
    if M["m00"] == 0:
        # 在ROI内未检测到车道线，保持上一状态或直行慢速
        return 0.0, 20.0, 0, w // 2, w // 2 # 修正：返回5个值，补上error=0

    lane_center_x = int(M["m10"] / M["m00"])
    vehicle_center_x = w // 2
    
    # 3. 计算横向误差
    error = vehicle_center_x - lane_center_x
    
    # 4. 使用PID控制器计算转向角
    steering_angle = pid_steer.update(error)
    steering_angle = np.clip(steering_angle, -25.0, 25.0) # 限制最大转向角

    # 5. 简单的速度控制逻辑
    max_speed = 50.0
    min_speed = 20.0
    # 误差越大，速度越慢
    speed_reduction = (abs(error) / (w / 2)) * (max_speed - min_speed)
    speed = max_speed - speed_reduction
    speed = np.clip(speed, min_speed, max_speed)

    return steering_angle, speed, error, lane_center_x, vehicle_center_x

# --- 摄像头抓取线程 ---
def camera_capture_thread():
    print("正在打开摄像头...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头索引 {CAMERA_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"摄像头配置: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass # 如果队列已满，则丢弃旧帧
    cap.release()

# --- 推理与控制线程 ---
def inference_thread():
    global shared_data, data_lock
    print("正在加载端到端模型...")
    model = InferSession(DEVICE_ID, MODEL_PATH)
    print("端到端模型加载完成。")

    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        # 1. 预处理 (注意：使用FP16数据类型)
        input_data = preprocess_end_to_end(frame, dtype=MODEL_INPUT_DTYPE)
        
        # 2. NPU端到端推理
        infer_start_time = time.time()
        outputs = model.infer([input_data])
        inference_time_ms = (time.time() - infer_start_time) * 1000
        
        # 3. 后处理
        lane_mask = postprocess(outputs[0])
        
        # 4. 计算控制信号
        steering, speed, error, lane_center, vehicle_center = calculate_control_signals(lane_mask)
        
        # 5. 更新共享数据
        with data_lock:
            shared_data["inference_time"] = f"{inference_time_ms:.1f}"
            shared_data["steering_angle"] = f"{steering:.1f}"
            shared_data["speed"] = f"{speed:.1f}"
            shared_data["error"] = f"{error:.1f}"

        # 6. 将结果放入队列，供Web UI显示
        try:
            result_queue.put_nowait({
                "frame": frame, "mask": lane_mask,
                "steering": steering, "speed": speed, "error": error,
                "lane_center": lane_center, "vehicle_center": vehicle_center
            })
        except queue.Full:
            pass

# --- 系统监控线程 ---
def system_monitor_loop():
    # (此部分与lane_dashboard_e2e.py基本一致，此处省略以保持简洁)
    pass # 在实际运行时，应将原代码粘贴于此

# --- Web UI 和 Flask路由 ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>车道线检测与PID控制</title>
    <style>
        body { font-family: sans-serif; margin: 20px; background: #f0f2f5; }
        .container { display: flex; gap: 20px; }
        .main { flex: 3; }
        .sidebar { flex: 1; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        img { width: 100%; border-radius: 8px; background: #ddd; }
        h1, h2 { color: #333; }
        .pid-controls label, .stats-card label { display: block; margin-bottom: 5px; color: #555; }
        .pid-controls input { width: 80px; }
        .pid-controls .value { font-weight: bold; margin-left: 10px; }
        .stats-card { margin-bottom: 15px; }
        .stats-card .value { font-size: 1.2em; font-weight: bold; color: #1a73e8; }
        .control-display .value { color: #e84393; }
    </style>
</head>
<body>
    <h1>🚀 车道线检测与PID控制 (端到端FP16)</h1>
    <div class="container">
        <div class="main">
            <img id="videoStream" src="/video_feed">
        </div>
        <div class="sidebar">
            <h2>⚙️ PID控制器调试</h2>
            <div class="pid-controls">
                <div class="stats-card">
                    <label for="p">P (比例): <span id="pValue">0.4</span></label>
                    <input type="range" id="p" min="0" max="2" step="0.05" value="0.4" oninput="updatePID()">
                </div>
                <div class="stats-card">
                    <label for="i">I (积分): <span id="iValue">0.0</span></label>
                    <input type="range" id="i" min="0" max="0.5" step="0.01" value="0.0" oninput="updatePID()">
                </div>
                <div class="stats-card">
                    <label for="d">D (微分): <span id="dValue">0.2</span></label>
                    <input type="range" id="d" min="0" max="1" step="0.05" value="0.2" oninput="updatePID()">
                </div>
            </div>
            <h2>🚗 车辆状态</h2>
            <div class="control-display">
                <div class="stats-card">
                    <label>转向角 (度)</label>
                    <span id="steering" class="value">0.0</span>
                </div>
                <div class="stats-card">
                    <label>目标速度 (km/h)</label>
                    <span id="speed" class="value">0.0</span>
                </div>
                 <div class="stats-card">
                    <label>横向误差 (像素)</label>
                    <span id="error" class="value">0.0</span>
                </div>
            </div>
             <h2>📊 性能监控</h2>
            <div class="perf-stats">
                 <div class="stats-card">
                    <label>推理耗时 (ms)</label>
                    <span id="inference_time" class="value">--</span>
                </div>
            </div>
        </div>
    </div>
<script>
    function updatePID() {
        const p = document.getElementById('p').value;
        const i = document.getElementById('i').value;
        const d = document.getElementById('d').value;
        document.getElementById('pValue').textContent = p;
        document.getElementById('iValue').textContent = i;
        document.getElementById('dValue').textContent = d;
        fetch('/update_pid', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({p: p, i: i, d: d})
        });
    }
    function refreshStats() {
        fetch('/stats')
        .then(response => response.json())
        .then(data => {
            document.getElementById('steering').textContent = data.steering_angle;
            document.getElementById('speed').textContent = data.speed;
            document.getElementById('error').textContent = data.error;
            document.getElementById('inference_time').textContent = data.inference_time;
        });
    }
    setInterval(refreshStats, 500);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/stats")
def stats():
    with data_lock:
        return jsonify(shared_data)

@app.route("/update_pid", methods=['POST'])
def update_pid_route():
    global pid_steer
    data = request.json
    with data_lock:
        shared_data['pid_p'] = float(data.get('p', shared_data['pid_p']))
        shared_data['pid_i'] = float(data.get('i', shared_data['pid_i']))
        shared_data['pid_d'] = float(data.get('d', shared_data['pid_d']))
        pid_steer.Kp = shared_data['pid_p']
        pid_steer.Ki = shared_data['pid_i']
        pid_steer.Kd = shared_data['pid_d']
        pid_steer.reset()
    return jsonify(success=True)

@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            try:
                result = result_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            frame = result["frame"]
            h, w, _ = frame.shape
            
            # 绘制车道线覆盖
            green_overlay = np.zeros_like(frame, dtype=np.uint8)
            green_overlay[result["mask"] > 0] = [0, 255, 0]
            vis_frame = cv2.addWeighted(frame, 1.0, green_overlay, 0.5, 0)
            
            # 绘制控制辅助线
            roi_y_start = h * 3 // 4
            cv2.line(vis_frame, (result["lane_center"], roi_y_start), (result["lane_center"], h), (255, 0, 0), 2) # 车道中心线
            cv2.line(vis_frame, (result["vehicle_center"], roi_y_start), (result["vehicle_center"], h), (0, 0, 255), 2) # 车辆中心线
            
            # 显示控制信息
            steer_text = f"Steer: {result['steering']:.1f} deg"
            speed_text = f"Speed: {result['speed']:.1f} km/h"
            cv2.putText(vis_frame, steer_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(vis_frame, speed_text, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            (flag, encodedImage) = cv2.imencode(".jpg", vis_frame)
            if not flag: continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encodedImage) + b'\r\n')
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    print("🚀 启动车道线检测与PID控制系统...")
    # 启动后台线程
    Thread(target=camera_capture_thread, daemon=True).start()
    Thread(target=inference_thread, daemon=True).start()
    # Thread(target=system_monitor_loop, daemon=True).start() # 可选
    
    print("✅ 系统初始化完成。")
    print(f"模型路径: {MODEL_PATH}")
    print(f"模型输入类型: {MODEL_INPUT_DTYPE}")
    print("请在浏览器中访问: http://<Your_Atlas_IP>:8000")
    app.run(host='0.0.0.0', port=8000, threaded=True)
