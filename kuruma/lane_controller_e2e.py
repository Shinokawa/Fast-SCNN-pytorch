"""
车道线检测与高级控制集成脚本 (端到端FP16 + 鸟瞰图)

功能:
- 使用端到端ONNX模型进行实时车道线检测 (FP16输入优化)。
- 将分割结果进行透视变换，生成鸟瞰图 (BEV)。
- 在鸟瞰图上使用滑动窗口和多项式拟合来寻找和跟踪车道线。
- 计算车道曲率和车辆横向偏移（物理单位）。
- 使用PID控制器根据物理偏移计算转向角度。
- 提供一个带有双视频流（原始+鸟瞰图）和实时参数调试功能的Web UI。
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
from collections import deque

try:
    from ais_bench.infer.interface import InferSession
except ImportError:
    print("警告：无法导入 ais_bench，将使用模拟推理。这仅用于PC端UI调试。")
    # 创建一个模拟的InferSession类，以便在没有Atlas环境的PC上进行UI开发和测试
    class InferSession:
        def __init__(self, device_id, model_path):
            print(f"[模拟模式] 初始化模型: {model_path}")
            self.model_path = model_path
            # 模拟一个与模型输出类似的形状
            self.dummy_output_shape = (1, 2, MODEL_HEIGHT, MODEL_WIDTH)

        def infer(self, inputs):
            print("[模拟模式] 正在执行推理...")
            # 返回一个随机的模拟输出，模拟分割掩码
            dummy_output = np.random.rand(*self.dummy_output_shape).astype(np.float32)
            time.sleep(0.05) # 模拟推理耗时
            return [dummy_output]

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
    "offset_m": "0.0", "curvature_m": "inf",
    # PID参数 (针对物理偏移量，需要重新调整)
    "pid_p": 1.5, "pid_i": 0.01, "pid_d": 0.5
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

# --- 透视变换和物理单位转换 ---
# 假设与 process_static_image.py 中的标定参数一致
# 原始图像中的四个点
SRC_POINTS = np.float32([[200, 480], [440, 480], [5, 300], [635, 300]])
# 变换后在鸟瞰图中的对应点
DST_POINTS = np.float32([[100, 480], [540, 480], [100, 0], [540, 0]])

# 像素到米的转换关系 (需要根据实际摄像头和场景标定)
YM_PER_PIX = 30 / 480  # y方向: 30米对应480像素
XM_PER_PIX = 3.7 / 440 # x方向: 3.7米对应440像素

M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
M_INV = cv2.getPerspectiveTransform(DST_POINTS, SRC_POINTS)

# --- 初始化PID控制器 ---
# 注意：现在的PID输入是米，Kp, Ki, Kd需要重新调整
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

# --- 高级车道线查找器 ---
class LaneFinder:
    def __init__(self):
        # 初始化时赋予有效的默认值，防止第一帧出错
        self.left_fit = np.array([0., 0., 0.])
        self.right_fit = np.array([0., 0., 0.])
        self.left_fit_m = np.array([0., 0., 0.])
        self.right_fit_m = np.array([0., 0., 0.])
        self.left_lane_inds = None
        self.right_lane_inds = None
        self.left_fitx = None
        self.right_fitx = None
        self.ploty = None
        self.detected = False
        # 使用deque来平滑多帧的拟合结果
        self.recent_left_fits = deque(maxlen=10)
        self.recent_right_fits = deque(maxlen=10)

    def find_lane_pixels(self, warped_mask):
        # ... (此处省略 find_lane_pixels 的具体实现, 与 process_static_image.py 类似)
        # 它会返回 leftx, lefty, rightx, righty, out_img
        histogram = np.sum(warped_mask[warped_mask.shape[0] // 2:, :], axis=0)
        out_img = np.dstack((warped_mask, warped_mask, warped_mask)) * 255
        midpoint = np.int32(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        window_height = np.int32(warped_mask.shape[0] / nwindows)
        nonzero = warped_mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 100
        minpix = 50
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = warped_mask.shape[0] - (window + 1) * window_height
            win_y_high = warped_mask.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def fit_polynomial(self, leftx, lefty, rightx, righty):
        # 重写以增强鲁棒性
        try:
            # 只有在有足够点的情况下才进行新的拟合
            if len(lefty) > 50 and len(leftx) > 50:
                left_fit = np.polyfit(lefty, leftx, 2)
                self.recent_left_fits.append(left_fit)
            if len(righty) > 50 and len(rightx) > 50:
                right_fit = np.polyfit(righty, rightx, 2)
                self.recent_right_fits.append(right_fit)
        except TypeError:
            # 拟合失败，忽略此帧，继续使用历史数据
            pass

        # 只有在有历史数据时才进行平滑
        if len(self.recent_left_fits) > 0:
            self.left_fit = np.mean(self.recent_left_fits, axis=0)
        if len(self.recent_right_fits) > 0:
            self.right_fit = np.mean(self.recent_right_fits, axis=0)

        # 总是生成绘图点，即使是基于旧的/默认的拟合
        ploty = np.linspace(0, MODEL_HEIGHT - 1, MODEL_HEIGHT)
        self.ploty = ploty
        self.left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        # 转换为米，同样需要检查
        if len(lefty) > 50 and len(leftx) > 50:
            self.left_fit_m = np.polyfit(lefty * YM_PER_PIX, leftx * XM_PER_PIX, 2)
        if len(righty) > 50 and len(rightx) > 50:
            self.right_fit_m = np.polyfit(righty * YM_PER_PIX, rightx * XM_PER_PIX, 2)

    def measure_curvature_offset(self):
        # ... (此处省略 measure_curvature_offset 的具体实现)
        # 返回 left_curverad, right_curverad, offset
        y_eval = (MODEL_HEIGHT - 1) * YM_PER_PIX # 在图像底部计算
        
        # 增加一个极小值(epsilon)以防止除以零
        left_curverad = ((1 + (2 * self.left_fit_m[0] * y_eval + self.left_fit_m[1]) ** 2) ** 1.5) / (np.absolute(2 * self.left_fit_m[0]) + 1e-6)
        right_curverad = ((1 + (2 * self.right_fit_m[0] * y_eval + self.right_fit_m[1]) ** 2) ** 1.5) / (np.absolute(2 * self.right_fit_m[0]) + 1e-6)
        
        # 计算车辆中心偏移
        # 检查self.left_fitx是否有效
        if self.left_fitx is not None and self.right_fitx is not None and len(self.left_fitx) > 0 and len(self.right_fitx) > 0:
            lane_center_pos = (self.left_fitx[-1] + self.right_fitx[-1]) / 2
        else: # 如果拟合无效，则假设车道在中心
            lane_center_pos = MODEL_WIDTH / 2

        car_center_pos = MODEL_WIDTH / 2
        offset = (car_center_pos - lane_center_pos) * XM_PER_PIX
        
        return left_curverad, right_curverad, offset

    def process(self, lane_mask, original_image):
        warped_mask = cv2.warpPerspective(lane_mask, M, (MODEL_WIDTH, MODEL_HEIGHT), flags=cv2.INTER_LINEAR)
        
        if not self.detected:
            leftx, lefty, rightx, righty, self.out_img = self.find_lane_pixels(warped_mask)
            if len(leftx) > 0 and len(rightx) > 0:
                 self.detected = True
        else: # 如果上一帧检测到了，就从上一帧的拟合位置附近开始搜索
            nonzero = warped_mask.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            margin = 100
            left_lane_inds = ((nonzerox > (self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] + margin)))
            right_lane_inds = ((nonzerox > (self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] + margin)))
            leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
            rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
            self.out_img = np.dstack((warped_mask, warped_mask, warped_mask)) * 255
            if len(leftx) < 50 or len(rightx) < 50: # 如果点太少，重新进行滑动窗口搜索
                self.detected = False

        self.fit_polynomial(leftx, lefty, rightx, righty)
        
        left_curverad, right_curverad, offset = self.measure_curvature_offset()
        curvature = (left_curverad + right_curverad) / 2

        # --- 可视化 ---
        # 绘制拟合线到鸟瞰图
        self.out_img[lefty, leftx] = [255, 0, 0]
        self.out_img[righty, rightx] = [0, 0, 255]
        
        # 绘制车道区域
        warp_zero = np.zeros_like(warped_mask).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        
        # 将鸟瞰图上的车道区域反向投射回原始图像
        new_warp = cv2.warpPerspective(color_warp, M_INV, (original_image.shape[1], original_image.shape[0]))
        result_img = cv2.addWeighted(original_image, 1, new_warp, 0.3, 0)
        
        # 在图像上显示曲率和偏移
        cv2.putText(result_img, f'Curvature: {curvature:.0f}m', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result_img, f'Offset: {offset:.2f}m', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return offset, curvature, result_img, self.out_img

# --- 核心控制逻辑 (现在由LaneFinder处理) ---
def calculate_control_signals(offset, curvature):
    """根据物理偏移和曲率计算转向和速度"""
    # 1. 使用PID控制器计算转向角 (输入为米)
    steering_angle = pid_steer.update(-offset) # 注意：误差方向与偏移量相反
    steering_angle = np.clip(steering_angle, -25.0, 25.0)

    # 2. 简单的速度控制逻辑
    max_speed = 50.0
    min_speed = 20.0
    # 偏移越大，速度越慢
    speed_reduction = (abs(offset) / 0.5) * (max_speed - min_speed) # 假设0.5米为较大偏移
    speed = max_speed - speed_reduction
    speed = np.clip(speed, min_speed, max_speed)

    return steering_angle, speed

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
    lane_finder = LaneFinder() # 初始化高级车道线查找器
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
        
        # 4. 高级车道线处理
        offset, curvature, final_frame, bev_frame = lane_finder.process(lane_mask, frame)

        # 5. 计算控制信号
        steering, speed = calculate_control_signals(offset, curvature)
        
        # 6. 更新共享数据
        with data_lock:
            shared_data["inference_time"] = f"{inference_time_ms:.1f}"
            shared_data["steering_angle"] = f"{steering:.1f}"
            shared_data["speed"] = f"{speed:.1f}"
            shared_data["error"] = f"{offset:.3f}" # 现在error是物理偏移
            shared_data["offset_m"] = f"{offset:.3f}"
            shared_data["curvature_m"] = f"{curvature:.0f}"

        # 7. 将结果放入队列，供Web UI显示
        try:
            result_queue.put_nowait({
                "final_frame": final_frame, 
                "bev_frame": bev_frame
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
    <title>高级车道线检测与控制</title>
    <style>
        body { font-family: sans-serif; margin: 20px; background: #f0f2f5; }
        .container { display: flex; flex-wrap: wrap; gap: 20px; }
        .main-view { flex: 2; min-width: 640px; }
        .sidebar { flex: 1; min-width: 320px; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        img { width: 100%; border-radius: 8px; background: #ddd; }
        h1, h2 { color: #333; }
        .video-container { display: flex; gap: 20px; margin-bottom: 20px;}
        .video-box { flex: 1; }
        .video-box h3 { text-align: center; color: #555; }
        .pid-controls label, .stats-card label { display: block; margin-bottom: 5px; color: #555; }
        .pid-controls input { width: 80px; }
        .pid-controls .value { font-weight: bold; margin-left: 10px; }
        .stats-card { margin-bottom: 15px; }
        .stats-card .value { font-size: 1.2em; font-weight: bold; color: #1a73e8; }
        .control-display .value { color: #e84393; }
    </style>
</head>
<body>
    <h1>🚀 高级车道线检测与PID控制 (鸟瞰图+物理模型)</h1>
    <div class="container">
        <div class="main-view">
            <div class="video-container">
                <div class="video-box">
                    <h3>原始视角 + 拟合结果</h3>
                    <img id="videoStream" src="/video_feed">
                </div>
                <div class="video-box">
                    <h3>鸟瞰图 (BEV) + 调试信息</h3>
                    <img id="bevStream" src="/bev_feed">
                </div>
            </div>
        </div>
        <div class="sidebar">
            <h2>⚙️ PID控制器调试 (物理单位)</h2>
            <div class="pid-controls">
                <div class="stats-card">
                    <label for="p">P (比例): <span id="pValue">1.5</span></label>
                    <input type="range" id="p" min="0" max="5" step="0.1" value="1.5" oninput="updatePID()">
                </div>
                <div class="stats-card">
                    <label for="i">I (积分): <span id="iValue">0.01</span></label>
                    <input type="range" id="i" min="0" max="0.5" step="0.01" value="0.01" oninput="updatePID()">
                </div>
                <div class="stats-card">
                    <label for="d">D (微分): <span id="dValue">0.5</span></label>
                    <input type="range" id="d" min="0" max="2" step="0.1" value="0.5" oninput="updatePID()">
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
                    <label>横向偏移 (米)</label>
                    <span id="offset_m" class="value">0.0</span>
                </div>
                 <div class="stats-card">
                    <label>曲率半径 (米)</label>
                    <span id="curvature_m" class="value">--</span>
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
            document.getElementById('offset_m').textContent = data.offset_m;
            document.getElementById('curvature_m').textContent = data.curvature_m;
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
            
            frame = result["final_frame"]
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if not flag: continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encodedImage) + b'\r\n')
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/bev_feed")
def bev_feed():
    def generate():
        while True:
            try:
                result = result_queue.get(timeout=1)
            except queue.Empty:
                # 如果没有新帧，可以发送一个占位符图像或等待
                continue
            
            bev_frame = result["bev_frame"]
            (flag, encodedImage) = cv2.imencode(".jpg", bev_frame)
            if not flag: continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encodedImage) + b'\r\n')
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    print("🚀 启动高级车道线检测与PID控制系统...")
    # 启动后台线程
    Thread(target=camera_capture_thread, daemon=True).start()
    Thread(target=inference_thread, daemon=True).start()
    # Thread(target=system_monitor_loop, daemon=True).start() # 可选
    
    print("✅ 系统初始化完成。")
    print(f"模型路径: {MODEL_PATH}")
    print(f"模型输入类型: {MODEL_INPUT_DTYPE}")
    print("请在浏览器中访问: http://<Your_Atlas_IP>:8000")
    app.run(host='0.0.0.0', port=8000, threaded=True)
