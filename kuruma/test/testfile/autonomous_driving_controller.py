#!/usr/bin/env python3
"""
🚗 智能小车自动驾驶控制系统
融合端到端车道线检测 + PID控制 + 实时调试界面
"""
import os
import cv2
import time
import numpy as np
import math
from threading import Thread, Lock
import queue
import psutil
import subprocess
import re
from collections import deque

from ais_bench.infer.interface import InferSession
from flask import Flask, Response, render_template_string, jsonify, request

# =============================================================================
# --- 🎯 全局配置和变量 ---
# =============================================================================

app = Flask(__name__)

# --- 队列和锁 ---
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
control_queue = queue.Queue(maxsize=1)
data_lock = Lock()

# --- 性能统计 ---
stats_data = {
    "fps": "0.0", "pipeline_latency": "0.0", "inference_time": "0.0",
    "preprocess_time": "0.0", "postprocess_time": "0.0", "control_time": "0.0",
    "cpu_percent": "0.0", "mem_percent": "0.0", "npu_util": "N/A", "npu_mem": "N/A"
}

# --- 控制状态 ---
control_data = {
    "steering_angle": 0.0,      # 转向角度 (-100 到 +100)
    "throttle": 0.0,            # 油门 (0 到 100)
    "brake": 0.0,               # 刹车 (0 到 100)
    "target_speed": 30.0,       # 目标速度 km/h
    "current_speed": 0.0,       # 当前速度 km/h
    "curvature": 0.0,           # 道路曲率 (m)
    "center_offset": 0.0,       # 中心偏移 (m)
    "lane_detected": False,     # 车道线检测状态
    "autonomous_mode": True     # 自动驾驶模式开关
}

# --- PID参数 ---
pid_params = {
    # 转向控制PID
    "steering_kp": 2.5,
    "steering_ki": 0.1,
    "steering_kd": 0.8,
    
    # 速度控制PID
    "speed_kp": 1.0,
    "speed_ki": 0.2,
    "speed_kd": 0.1,
    
    # 其他参数
    "max_steering": 45.0,       # 最大转向角度
    "min_speed": 10.0,          # 最小速度
    "max_speed": 60.0,          # 最大速度
    "curve_speed_factor": 0.7   # 弯道减速系数
}

# --- 硬件配置 ---
DEVICE_ID = 0
MODEL_PATH = "./weights/fast_scnn_480x640_e2e_uint8_input.om"
MODEL_WIDTH = 640
MODEL_HEIGHT = 480
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
NPU_SMI_PATH = "/usr/local/Ascend/driver/tools/npu-smi"

# --- 透视变换参数 (请根据您的摄像头标定) ---
SRC_POINTS = np.float32([[47, 590], [629, 572], [458, 421], [212, 434]])
DST_POINTS = np.float32([
    [CAMERA_WIDTH * 0.15, CAMERA_HEIGHT], # 左下
    [CAMERA_WIDTH * 0.85, CAMERA_HEIGHT], # 右下
    [CAMERA_WIDTH * 0.85, 0],            # 右上
    [CAMERA_WIDTH * 0.15, 0]             # 左上
])

# --- 真实世界转换系数 ---
YM_PER_PIX = 30 / CAMERA_HEIGHT  # 垂直方向米/像素
XM_PER_PIX = 3.7 / (DST_POINTS[1][0] - DST_POINTS[0][0])  # 水平方向米/像素

# =============================================================================
# --- 🔧 核心处理函数 ---
# =============================================================================

def preprocess_uint8(img_bgr):
    """端到端模型预处理：uint8输入"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_transposed = np.transpose(img_rgb, (2, 0, 1))
    return np.ascontiguousarray(img_transposed[np.newaxis, :, :, :], dtype=np.uint8)

def postprocess_segmentation(output_tensor, original_width, original_height):
    """分割后处理"""
    pred_mask = np.argmax(output_tensor, axis=1).squeeze()
    vis_mask = (pred_mask > 0).astype(np.uint8) * 255
    return cv2.resize(vis_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

def find_lane_pixels_and_fit(warped_mask):
    """车道线像素检测和多项式拟合"""
    histogram = np.sum(warped_mask[warped_mask.shape[0]//2:, :], axis=0)
    midpoint = np.int32(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint]) if np.max(histogram[:midpoint]) > 50 else None
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint if np.max(histogram[midpoint:]) > 50 else None

    if leftx_base is None and rightx_base is None:
        return None, None, ([], []), ([], [])

    nwindows = 9
    margin = 100
    minpix = 50
    window_height = np.int32(warped_mask.shape[0]//nwindows)

    nonzero = warped_mask.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds, right_lane_inds = [], []
    leftx_current = leftx_base if leftx_base is not None else midpoint//2
    rightx_current = rightx_base if rightx_base is not None else midpoint + midpoint//2

    for window in range(nwindows):
        win_y_low = warped_mask.shape[0] - (window+1)*window_height
        win_y_high = warped_mask.shape[0] - window*window_height
        
        # 左车道窗口
        if leftx_base is not None:
            win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                             (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        
        # 右车道窗口
        if rightx_base is not None:
            win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            right_lane_inds.append(good_right_inds)
            if len(good_right_inds) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    # 合并所有像素
    left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([])
    right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([])

    leftx, lefty = (nonzerox[left_lane_inds], nonzeroy[left_lane_inds]) if len(left_lane_inds) > 0 else ([], [])
    rightx, righty = (nonzerox[right_lane_inds], nonzeroy[right_lane_inds]) if len(right_lane_inds) > 0 else ([], [])

    # 多项式拟合
    left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 10 else None
    right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 10 else None
    
    return left_fit, right_fit, (leftx, lefty), (rightx, righty)

def calculate_control_metrics(left_fit, right_fit, img_shape):
    """计算控制所需的关键指标"""
    h, w = img_shape
    y_eval = h - 1  # 评估点在图像底部
    
    curvature = 0.0
    center_offset = 0.0
    lane_detected = False
    
    if left_fit is not None and right_fit is not None:
        lane_detected = True
        
        # 计算曲率 (转换为真实世界坐标)
        left_fit_cr = np.polyfit(np.arange(h) * YM_PER_PIX, 
                                (left_fit[0] * np.arange(h)**2 + left_fit[1] * np.arange(h) + left_fit[2]) * XM_PER_PIX, 2)
        right_fit_cr = np.polyfit(np.arange(h) * YM_PER_PIX,
                                 (right_fit[0] * np.arange(h)**2 + right_fit[1] * np.arange(h) + right_fit[2]) * XM_PER_PIX, 2)
        
        y_eval_m = y_eval * YM_PER_PIX
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_m + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0] + 1e-6)
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_m + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0] + 1e-6)
        curvature = (left_curverad + right_curverad) / 2
        
        # 计算中心偏移
        left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
        right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
        lane_center_x = (left_x + right_x) / 2
        car_center_x = w / 2
        center_offset = (car_center_x - lane_center_x) * XM_PER_PIX
    
    elif left_fit is not None:
        # 只有左车道线
        lane_detected = True
        left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
        # 假设标准车道宽度3.7米
        estimated_right_x = left_x + (3.7 / XM_PER_PIX)
        lane_center_x = (left_x + estimated_right_x) / 2
        car_center_x = w / 2
        center_offset = (car_center_x - lane_center_x) * XM_PER_PIX
        
        # 简化曲率计算
        left_fit_cr = np.polyfit(np.arange(h) * YM_PER_PIX,
                                (left_fit[0] * np.arange(h)**2 + left_fit[1] * np.arange(h) + left_fit[2]) * XM_PER_PIX, 2)
        y_eval_m = y_eval * YM_PER_PIX
        curvature = ((1 + (2*left_fit_cr[0]*y_eval_m + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0] + 1e-6)
    
    elif right_fit is not None:
        # 只有右车道线
        lane_detected = True
        right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
        estimated_left_x = right_x - (3.7 / XM_PER_PIX)
        lane_center_x = (estimated_left_x + right_x) / 2
        car_center_x = w / 2
        center_offset = (car_center_x - lane_center_x) * XM_PER_PIX
        
        # 简化曲率计算
        right_fit_cr = np.polyfit(np.arange(h) * YM_PER_PIX,
                                 (right_fit[0] * np.arange(h)**2 + right_fit[1] * np.arange(h) + right_fit[2]) * XM_PER_PIX, 2)
        y_eval_m = y_eval * YM_PER_PIX
        curvature = ((1 + (2*right_fit_cr[0]*y_eval_m + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0] + 1e-6)
    
    return curvature, center_offset, lane_detected

# =============================================================================
# --- 🎮 PID控制器类 ---
# =============================================================================

class PIDController:
    def __init__(self, kp, ki, kd, output_limits=(-100, 100)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        self.error_history = deque(maxlen=10)
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
        
    def update_gains(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
    def compute(self, setpoint, measured_value):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 0.01
            
        error = setpoint - measured_value
        self.error_history.append(error)
        
        # P项
        proportional = self.kp * error
        
        # I项
        self.integral += error * dt
        integral = self.ki * self.integral
        
        # D项
        derivative = self.kd * (error - self.last_error) / dt
        
        # PID输出
        output = proportional + integral + derivative
        
        # 限制输出范围
        output = max(min(output, self.output_limits[1]), self.output_limits[0])
        
        self.last_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        self.error_history.clear()
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()

# 初始化PID控制器
steering_pid = PIDController(
    pid_params["steering_kp"], 
    pid_params["steering_ki"], 
    pid_params["steering_kd"], 
    output_limits=(-pid_params["max_steering"], pid_params["max_steering"])
)

speed_pid = PIDController(
    pid_params["speed_kp"],
    pid_params["speed_ki"], 
    pid_params["speed_kd"],
    output_limits=(-100, 100)
)

# =============================================================================
# --- 🚗 控制逻辑 ---
# =============================================================================

def compute_vehicle_control(curvature, center_offset, lane_detected, target_speed):
    """计算车辆控制指令"""
    global control_data, pid_params
    
    control_start = time.time()
    
    if not control_data["autonomous_mode"]:
        # 手动模式
        steering_angle = 0.0
        throttle = 0.0
        brake = 0.0
        current_speed = 0.0
    elif not lane_detected:
        # 未检测到车道线 - 紧急制动
        steering_angle = 0.0
        throttle = 0.0
        brake = 80.0
        current_speed = 0.0
    else:
        # 自动驾驶模式
        
        # 1. 转向控制 (基于中心偏移)
        steering_angle = steering_pid.compute(0.0, center_offset)
        
        # 2. 基于曲率的自适应速度
        if curvature > 0:
            curve_factor = min(1.0, 500.0 / curvature)  # 曲率越大，速度越慢
        else:
            curve_factor = 1.0
        
        adaptive_target_speed = target_speed * curve_factor * pid_params["curve_speed_factor"]
        adaptive_target_speed = max(pid_params["min_speed"], 
                                   min(pid_params["max_speed"], adaptive_target_speed))
        
        # 3. 速度控制
        current_speed = control_data["current_speed"]  # 这里应该从实际传感器获取
        speed_error = adaptive_target_speed - current_speed
        speed_output = speed_pid.compute(adaptive_target_speed, current_speed)
        
        if speed_output > 0:
            throttle = speed_output
            brake = 0.0
        else:
            throttle = 0.0
            brake = -speed_output
        
        # 4. 安全限制
        max_steering = pid_params["max_steering"]
        steering_angle = max(-max_steering, min(max_steering, steering_angle))
        throttle = max(0, min(100, throttle))
        brake = max(0, min(100, brake))
    
    control_time = (time.time() - control_start) * 1000
    
    # 更新控制状态
    with data_lock:
        control_data.update({
            "steering_angle": steering_angle,
            "throttle": throttle,
            "brake": brake,
            "curvature": curvature,
            "center_offset": center_offset,
            "lane_detected": lane_detected,
            "current_speed": current_speed  # 实际应该从车辆传感器读取
        })
        stats_data["control_time"] = f"{control_time:.1f}"
    
    return steering_angle, throttle, brake

# =============================================================================
# --- 🎥 摄像头和推理线程 ---
# =============================================================================

def camera_capture_thread():
    """摄像头抓取线程"""
    print("正在打开摄像头...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头索引 {CAMERA_INDEX}.")
        return

    # 设置摄像头参数
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"摄像头启动成功: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        try:
            if frame_queue.empty():
                frame_queue.put_nowait(frame)
        except queue.Full:
            pass
    cap.release()

def autonomous_control_thread():
    """自动驾驶主控制线程"""
    global stats_data, control_data, data_lock
    
    print(f"正在加载端到端模型: {MODEL_PATH}")
    model = InferSession(DEVICE_ID, MODEL_PATH)
    print("模型加载完成。开始自动驾驶控制循环...")
    
    # 透视变换矩阵
    M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
    Minv = cv2.getPerspectiveTransform(DST_POINTS, SRC_POINTS)
    
    frame_count = 0
    total_times = {"preprocess": 0, "inference": 0, "postprocess": 0, "control": 0, "pipeline": 0}

    print("\n=== 🚗 自动驾驶控制系统启动 ===")
    print("每20帧输出一次性能分析...")

    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        loop_start_time = time.time()
        cam_height, cam_width = frame.shape[:2]
        
        # 1. 预处理
        preprocess_start = time.time()
        input_data = preprocess_uint8(frame)
        preprocess_time_ms = (time.time() - preprocess_start) * 1000
        
        # 2. NPU推理
        infer_start_time = time.time()
        outputs = model.infer([input_data])
        inference_time_ms = (time.time() - infer_start_time) * 1000
        
        # 3. 分割后处理
        postprocess_start = time.time()
        lane_mask = postprocess_segmentation(outputs[0], cam_width, cam_height)
        
        # 4. 透视变换到鸟瞰图
        warped_mask = cv2.warpPerspective(lane_mask, M, (cam_width, cam_height), flags=cv2.INTER_NEAREST)
        
        # 5. 车道线拟合
        left_fit, right_fit, left_pixels, right_pixels = find_lane_pixels_and_fit(warped_mask)
        
        # 6. 计算控制指标
        curvature, center_offset, lane_detected = calculate_control_metrics(left_fit, right_fit, (cam_height, cam_width))
        postprocess_time_ms = (time.time() - postprocess_start) * 1000
        
        # 7. 计算车辆控制指令
        target_speed = control_data["target_speed"]
        steering_angle, throttle, brake = compute_vehicle_control(curvature, center_offset, lane_detected, target_speed)
        
        pipeline_latency_ms = (time.time() - loop_start_time) * 1000
        
        # 统计
        frame_count += 1
        total_times["preprocess"] += preprocess_time_ms
        total_times["inference"] += inference_time_ms
        total_times["postprocess"] += postprocess_time_ms
        total_times["pipeline"] += pipeline_latency_ms
        
        if frame_count % 20 == 0:
            avg_pipeline = total_times["pipeline"] / frame_count
            print(f"\n--- 🚗 第{frame_count}帧控制分析 ---")
            print(f"车道检测: {'✅' if lane_detected else '❌'} | 曲率: {curvature:.1f}m | 偏移: {center_offset:.2f}m")
            print(f"控制指令: 转向={steering_angle:.1f}° | 油门={throttle:.1f}% | 刹车={brake:.1f}%")
            print(f"平均延迟: {avg_pipeline:.1f}ms (目标FPS: {1000/avg_pipeline:.1f})")
            print("-" * 50)

        # 发送结果
        try:
            if result_queue.empty():
                result_queue.put_nowait({
                    "frame": frame, 
                    "mask": lane_mask,
                    "warped_mask": warped_mask,
                    "left_fit": left_fit,
                    "right_fit": right_fit,
                    "latency": pipeline_latency_ms, 
                    "inference_time": inference_time_ms,
                    "preprocess_time": preprocess_time_ms, 
                    "postprocess_time": postprocess_time_ms
                })
        except queue.Full:
            pass
        
        with data_lock:
            stats_data.update({
                "pipeline_latency": f"{pipeline_latency_ms:.1f}",
                "inference_time": f"{inference_time_ms:.1f}",
                "preprocess_time": f"{preprocess_time_ms:.1f}",
                "postprocess_time": f"{postprocess_time_ms:.1f}"
            })

def system_monitor_loop():
    """系统监控线程"""
    global stats_data, data_lock
    npu_error_printed = False
    
    while True:
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        with data_lock:
            stats_data["cpu_percent"] = f"{cpu:.1f}"
            stats_data["mem_percent"] = f"{mem:.1f}"

        try:
            result = subprocess.run([NPU_SMI_PATH, 'info'], capture_output=True, text=True, check=True, timeout=2)
            output = result.stdout
            util_match = re.search(r'NPU\s+Utilization\s*:\s*(\d+)\s*%', output)
            mem_match = re.search(r'Memory\s+Usage\s*:\s*([\d\.]+)\s*MiB\s*/\s*([\d\.]+)\s*MiB', output)
            npu_util = util_match.group(1) if util_match else "N/A"
            npu_mem = f"{float(mem_match.group(1)):.0f} / {float(mem_match.group(2)):.0f} MiB" if mem_match else "N/A"
            with data_lock:
                stats_data["npu_util"] = npu_util
                stats_data["npu_mem"] = npu_mem
            if npu_error_printed:
                print("✅ NPU监控已恢复正常")
                npu_error_printed = False
        except Exception:
             if not npu_error_printed:
                 print(f"❌ NPU监控失败，请检查路径或权限: {NPU_SMI_PATH}")
                 npu_error_printed = True
             with data_lock:
                 stats_data["npu_util"] = "Error"
                 stats_data["npu_mem"] = "Error"
        time.sleep(1)

# =============================================================================
# --- 🌐 Web界面 ---
# =============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🚗 智能小车自动驾驶控制系统</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; background-color: #f0f2f5; color: #333; }
        .container { max-width: 1600px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #1a73e8; margin-bottom: 30px; }
        .main-content { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        .video-section { display: flex; flex-direction: column; gap: 15px; }
        #videoStream { width: 100%; border-radius: 10px; box-shadow: 0 3px 15px rgba(0,0,0,0.1); background-color: #eee; }
        .control-panel { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 15px; }
        .control-section { background-color: rgba(255,255,255,0.1); margin-bottom: 20px; padding: 20px; border-radius: 10px; }
        .control-section h3 { margin: 0 0 15px 0; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 10px; }
        .status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px; }
        .status-item { background-color: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px; text-align: center; }
        .status-value { font-size: 24px; font-weight: bold; margin-top: 5px; }
        .control-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .control-item { display: flex; flex-direction: column; margin-bottom: 15px; }
        .control-item label { font-size: 14px; margin-bottom: 5px; opacity: 0.9; }
        .control-item input, .control-item select { padding: 8px; border: none; border-radius: 5px; background-color: rgba(255,255,255,0.9); }
        .toggle-button { background-color: #e74c3c; color: white; border: none; padding: 12px 24px; border-radius: 25px; font-size: 16px; font-weight: bold; cursor: pointer; transition: all 0.3s ease; }
        .toggle-button.active { background-color: #27ae60; }
        .toggle-button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .stat-item { background-color: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px; font-size: 12px; }
        .emergency-stop { background-color: #e74c3c !important; font-size: 18px; padding: 15px 30px; }
        .lane-status { padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; margin-bottom: 15px; }
        .lane-status.detected { background-color: #27ae60; }
        .lane-status.not-detected { background-color: #e74c3c; }
        .pid-section { background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .pid-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
        .pid-input { padding: 5px; border: none; border-radius: 3px; background-color: rgba(255,255,255,0.9); font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 智能小车自动驾驶控制系统</h1>
        <div class="main-content">
            <div class="video-section">
                <img id="videoStream" src="{{ url_for('video_feed') }}" alt="Live Stream">
            </div>
            <div class="control-panel">
                <!-- 系统状态 -->
                <div class="control-section">
                    <h3>🚦 系统状态</h3>
                    <div id="laneStatus" class="lane-status not-detected">车道线检测: 未检测</div>
                    <div class="status-grid">
                        <div class="status-item">
                            <div>转向角度</div>
                            <div id="steeringAngle" class="status-value">0°</div>
                        </div>
                        <div class="status-item">
                            <div>油门/刹车</div>
                            <div id="throttleBrake" class="status-value">0%</div>
                        </div>
                        <div class="status-item">
                            <div>道路曲率</div>
                            <div id="curvature" class="status-value">0m</div>
                        </div>
                        <div class="status-item">
                            <div>中心偏移</div>
                            <div id="centerOffset" class="status-value">0.0m</div>
                        </div>
                    </div>
                </div>

                <!-- 控制开关 -->
                <div class="control-section">
                    <h3>🎮 控制开关</h3>
                    <button id="autonomousToggle" class="toggle-button active" onclick="toggleAutonomous()">
                        自动驾驶: 开启
                    </button>
                    <br><br>
                    <button class="toggle-button emergency-stop" onclick="emergencyStop()">
                        🚨 紧急停车
                    </button>
                </div>

                <!-- 目标参数 -->
                <div class="control-section">
                    <h3>🎯 目标参数</h3>
                    <div class="control-item">
                        <label>目标速度 (km/h)</label>
                        <input type="range" id="targetSpeed" min="10" max="60" value="30" oninput="updateTargetSpeed(this.value)">
                        <div style="text-align: center; margin-top: 5px;" id="targetSpeedDisplay">30 km/h</div>
                    </div>
                </div>

                <!-- PID参数调试 -->
                <div class="control-section">
                    <h3>🔧 PID参数调试</h3>
                    <div class="pid-section">
                        <h4 style="margin: 0 0 10px 0;">转向控制PID</h4>
                        <div class="pid-grid">
                            <div>
                                <label>Kp</label>
                                <input type="number" class="pid-input" id="steeringKp" value="2.5" step="0.1" onchange="updatePID()">
                            </div>
                            <div>
                                <label>Ki</label>
                                <input type="number" class="pid-input" id="steeringKi" value="0.1" step="0.01" onchange="updatePID()">
                            </div>
                            <div>
                                <label>Kd</label>
                                <input type="number" class="pid-input" id="steeringKd" value="0.8" step="0.1" onchange="updatePID()">
                            </div>
                        </div>
                    </div>
                    <div class="pid-section">
                        <h4 style="margin: 0 0 10px 0;">速度控制PID</h4>
                        <div class="pid-grid">
                            <div>
                                <label>Kp</label>
                                <input type="number" class="pid-input" id="speedKp" value="1.0" step="0.1" onchange="updatePID()">
                            </div>
                            <div>
                                <label>Ki</label>
                                <input type="number" class="pid-input" id="speedKi" value="0.2" step="0.01" onchange="updatePID()">
                            </div>
                            <div>
                                <label>Kd</label>
                                <input type="number" class="pid-input" id="speedKd" value="0.1" step="0.01" onchange="updatePID()">
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 性能监控 -->
                <div class="control-section">
                    <h3>📊 性能监控</h3>
                    <div class="stats-grid">
                        <div class="stat-item">FPS: <span id="fps">--</span></div>
                        <div class="stat-item">延迟: <span id="latency">--</span>ms</div>
                        <div class="stat-item">推理: <span id="inference">--</span>ms</div>
                        <div class="stat-item">控制: <span id="control">--</span>ms</div>
                        <div class="stat-item">CPU: <span id="cpu">--</span>%</div>
                        <div class="stat-item">NPU: <span id="npu">--</span>%</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function refreshStatus() {
            fetch('/control_status')
                .then(response => response.json())
                .then(data => {
                    // 更新状态显示
                    document.getElementById('steeringAngle').textContent = parseFloat(data.steering_angle).toFixed(1) + '°';
                    
                    let throttleBrakeText = '';
                    if (data.throttle > 0) {
                        throttleBrakeText = '🚀 ' + parseFloat(data.throttle).toFixed(0) + '%';
                    } else if (data.brake > 0) {
                        throttleBrakeText = '🛑 ' + parseFloat(data.brake).toFixed(0) + '%';
                    } else {
                        throttleBrakeText = '⏸️ 0%';
                    }
                    document.getElementById('throttleBrake').textContent = throttleBrakeText;
                    
                    document.getElementById('curvature').textContent = parseFloat(data.curvature).toFixed(0) + 'm';
                    document.getElementById('centerOffset').textContent = parseFloat(data.center_offset).toFixed(2) + 'm';
                    
                    // 车道线检测状态
                    const laneStatus = document.getElementById('laneStatus');
                    if (data.lane_detected) {
                        laneStatus.textContent = '车道线检测: ✅ 已检测';
                        laneStatus.className = 'lane-status detected';
                    } else {
                        laneStatus.textContent = '车道线检测: ❌ 未检测';
                        laneStatus.className = 'lane-status not-detected';
                    }
                    
                    // 自动驾驶状态
                    const autoToggle = document.getElementById('autonomousToggle');
                    if (data.autonomous_mode) {
                        autoToggle.textContent = '自动驾驶: 开启';
                        autoToggle.className = 'toggle-button active';
                    } else {
                        autoToggle.textContent = '自动驾驶: 关闭';
                        autoToggle.className = 'toggle-button';
                    }
                })
                .catch(error => console.error('获取控制状态失败:', error));
            
            // 更新性能统计
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps;
                    document.getElementById('latency').textContent = data.pipeline_latency;
                    document.getElementById('inference').textContent = data.inference_time;
                    document.getElementById('control').textContent = data.control_time;
                    document.getElementById('cpu').textContent = data.cpu_percent;
                    document.getElementById('npu').textContent = data.npu_util;
                })
                .catch(error => console.error('获取性能统计失败:', error));
        }
        
        function toggleAutonomous() {
            fetch('/toggle_autonomous', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('自动驾驶状态已切换:', data.autonomous_mode);
                });
        }
        
        function emergencyStop() {
            fetch('/emergency_stop', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('紧急停车已激活');
                });
        }
        
        function updateTargetSpeed(value) {
            document.getElementById('targetSpeedDisplay').textContent = value + ' km/h';
            fetch('/update_target_speed', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ target_speed: parseFloat(value) })
            });
        }
        
        function updatePID() {
            const pidData = {
                steering_kp: parseFloat(document.getElementById('steeringKp').value),
                steering_ki: parseFloat(document.getElementById('steeringKi').value),
                steering_kd: parseFloat(document.getElementById('steeringKd').value),
                speed_kp: parseFloat(document.getElementById('speedKp').value),
                speed_ki: parseFloat(document.getElementById('speedKi').value),
                speed_kd: parseFloat(document.getElementById('speedKd').value)
            };
            
            fetch('/update_pid', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(pidData)
            });
        }
        
        window.onload = function() {
            refreshStatus();
            setInterval(refreshStatus, 500); // 500ms更新一次，保证控制响应性
        };
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/video_feed")
def video_feed():
    def generate():
        global stats_data, data_lock
        frame_count = 0
        start_time = time.time()
        
        while True:
            try:
                result = result_queue.get(timeout=1)
            except queue.Empty:
                continue
                
            frame = result["frame"]
            mask = result["mask"]
            
            # 创建可视化帧
            green_overlay = np.zeros_like(frame, dtype=np.uint8)
            green_overlay[mask > 0] = [0, 255, 0]
            vis_frame = cv2.addWeighted(frame, 0.7, green_overlay, 0.3, 0)
            
            # 绘制车道线拟合结果
            if result.get("left_fit") is not None or result.get("right_fit") is not None:
                h, w = frame.shape[:2]
                ploty = np.linspace(0, h-1, h)
                
                if result.get("left_fit") is not None:
                    left_fit = result["left_fit"]
                    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
                    left_points = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                    cv2.polylines(vis_frame, [left_points.astype(np.int32)], False, (255, 0, 0), 3)
                
                if result.get("right_fit") is not None:
                    right_fit = result["right_fit"]
                    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
                    right_points = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
                    cv2.polylines(vis_frame, [right_points.astype(np.int32)], False, (0, 0, 255), 3)
            
            # 显示控制信息
            with data_lock:
                steering = control_data["steering_angle"]
                throttle = control_data["throttle"]
                brake = control_data["brake"]
                fps_text = stats_data.get("fps", "0")
                latency_text = stats_data.get("pipeline_latency", "0")
            
            # 文本叠加
            cv2.putText(vis_frame, f"FPS: {fps_text}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Latency: {latency_text}ms", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Steering: {steering:.1f}°", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if throttle > 0:
                cv2.putText(vis_frame, f"Throttle: {throttle:.0f}%", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif brake > 0:
                cv2.putText(vis_frame, f"Brake: {brake:.0f}%", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 计算FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                with data_lock:
                    stats_data["fps"] = f"{fps:.1f}"
                frame_count = 0
                start_time = time.time()
            
            # 编码并发送
            (flag, encodedImage) = cv2.imencode(".jpg", vis_frame)
            if not flag:
                continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encodedImage) + b'\r\n')
    
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stats")
def stats():
    with data_lock:
        return jsonify(stats_data)

@app.route("/control_status")
def control_status():
    with data_lock:
        return jsonify(control_data)

@app.route("/toggle_autonomous", methods=['POST'])
def toggle_autonomous():
    global control_data
    with data_lock:
        control_data["autonomous_mode"] = not control_data["autonomous_mode"]
        
        # 重置PID控制器
        if control_data["autonomous_mode"]:
            steering_pid.reset()
            speed_pid.reset()
        
        return jsonify({"autonomous_mode": control_data["autonomous_mode"]})

@app.route("/emergency_stop", methods=['POST'])
def emergency_stop():
    global control_data
    with data_lock:
        control_data["autonomous_mode"] = False
        control_data["steering_angle"] = 0.0
        control_data["throttle"] = 0.0
        control_data["brake"] = 100.0
        
    # 重置PID控制器
    steering_pid.reset()
    speed_pid.reset()
    
    return jsonify({"status": "emergency_stop_activated"})

@app.route("/update_target_speed", methods=['POST'])
def update_target_speed():
    global control_data
    data = request.get_json()
    target_speed = data.get("target_speed", 30.0)
    
    with data_lock:
        control_data["target_speed"] = max(10.0, min(60.0, target_speed))
    
    return jsonify({"target_speed": control_data["target_speed"]})

@app.route("/update_pid", methods=['POST'])
def update_pid():
    global pid_params, steering_pid, speed_pid
    data = request.get_json()
    
    # 更新PID参数
    pid_params.update({
        "steering_kp": data.get("steering_kp", pid_params["steering_kp"]),
        "steering_ki": data.get("steering_ki", pid_params["steering_ki"]),
        "steering_kd": data.get("steering_kd", pid_params["steering_kd"]),
        "speed_kp": data.get("speed_kp", pid_params["speed_kp"]),
        "speed_ki": data.get("speed_ki", pid_params["speed_ki"]),
        "speed_kd": data.get("speed_kd", pid_params["speed_kd"])
    })
    
    # 更新控制器参数
    steering_pid.update_gains(
        pid_params["steering_kp"],
        pid_params["steering_ki"],
        pid_params["steering_kd"]
    )
    speed_pid.update_gains(
        pid_params["speed_kp"],
        pid_params["speed_ki"],
        pid_params["speed_kd"]
    )
    
    return jsonify({"status": "pid_updated", "params": pid_params})

# =============================================================================
# --- 🚀 主程序 ---
# =============================================================================

if __name__ == '__main__':
    print("🚗 智能小车自动驾驶控制系统启动")
    print("=" * 60)
    print(f"🧠 模型: {MODEL_PATH}")
    print(f"🎯 输入尺寸: {MODEL_WIDTH}x{MODEL_HEIGHT}")
    print(f"🎮 控制模式: PID转向 + 自适应速度")
    print(f"🔧 调试界面: Web控制面板")
    print("=" * 60)
    
    # 启动所有线程
    Thread(target=camera_capture_thread, daemon=True).start()
    Thread(target=autonomous_control_thread, daemon=True).start()
    Thread(target=system_monitor_loop, daemon=True).start()
    
    print("\n🌐 Web控制面板已启动: http://<Your_Atlas_IP>:8000")
    print("🎮 功能包括:")
    print("  - 实时车道线检测和控制")
    print("  - PID参数实时调试")
    print("  - 速度/转向控制")
    print("  - 紧急停车功能")
    print("  - 性能监控")
    print("\n开始自动驾驶控制循环...")
    
    app.run(host='0.0.0.0', port=8000, threaded=True)
