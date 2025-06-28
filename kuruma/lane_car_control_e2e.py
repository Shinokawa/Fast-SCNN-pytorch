import os
import cv2
import time
import numpy as np
from threading import Thread, Lock
import queue
import psutil
import subprocess
import re
import math
from collections import deque

from ais_bench.infer.interface import InferSession
from flask import Flask, Response, render_template_string, jsonify

# --- Flask App 和全局变量 ---
app = Flask(__name__)
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
stats_data = {
    "fps": "0.0", "pipeline_latency": "0.0", "inference_time": "0.0",
    "preprocess_time": "0.0", "postprocess_time": "0.0",
    "cpu_percent": "0.0", "mem_percent": "0.0", "npu_util": "N/A", "npu_mem": "N/A",
    # 新增车辆控制状态
    "steering_angle": "0.0", "lane_deviation": "0.0", "curve_radius": "0.0",
    "control_mode": "AUTO", "speed": "0.0", "lane_status": "DETECTING"
}
data_lock = Lock()

# --- 【关键】可配置常量 ---
DEVICE_ID = 0
MODEL_PATH = "./weights/fast_scnn_480x640_e2e_fp16_input.om" 
MODEL_WIDTH = 640
MODEL_HEIGHT = 480
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
NPU_SMI_PATH = "/usr/local/Ascend/driver/tools/npu-smi"

# --- 小车控制参数 ---
CAR_CONTROL = {
    "max_steering_angle": 30.0,    # 最大转向角度(度)
    "base_speed": 20.0,            # 基础速度
    "min_speed": 10.0,             # 最小速度
    "max_speed": 40.0,             # 最大速度
    "kp": 1.2,                     # PID比例系数
    "ki": 0.1,                     # PID积分系数  
    "kd": 0.8,                     # PID微分系数
    "lookahead_distance": 100,     # 前瞻距离(像素)
    "curve_speed_factor": 0.6,     # 弯道减速系数
    "emergency_stop_speed": 5.0,   # 紧急停车速度
}

# PID控制器状态
pid_state = {
    "previous_error": 0.0,
    "integral": 0.0,
    "error_history": deque(maxlen=10),
    "last_valid_steering": 0.0,    # 上一次有效的转向角度
    "no_lane_counter": 0           # 连续检测不到车道线的帧数
}

# ---------------------------------------------------------------------------------
# --- 🚀🚀🚀 极简预处理 (分辨率匹配) 🚀🚀🚀 ---
# ---------------------------------------------------------------------------------

def preprocess_matched_resolution(img_bgr, dtype=np.float16):
    """当模型输入分辨率与摄像头输出完全匹配时，预处理开销最小化。"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_typed = img_rgb.astype(dtype)
    img_transposed = np.transpose(img_typed, (2, 0, 1))
    return np.ascontiguousarray(img_transposed[np.newaxis, :, :, :])

# ---------------------------------------------------------------------------------
# --- 🚗 增强的智能车道线检测与控制算法 🚗 ---
# ---------------------------------------------------------------------------------

def extract_lane_points(mask, roi_height_ratio=0.7):
    """
    从分割掩码中提取车道线点，专门针对小车场景优化
    处理弯曲车道和近距离视角
    """
    height, width = mask.shape
    
    # 定义ROI区域 - 小车主要关注前方较近的区域
    roi_top = int(height * (1 - roi_height_ratio))
    roi_bottom = height
    roi_mask = mask[roi_top:roi_bottom, :]
    
    # 提取车道线点
    lane_points = []
    y_offset = roi_top
    
    # 从下往上扫描，每隔几行提取一次（减少计算量）
    scan_step = max(1, int(roi_height_ratio * height // 20))  # 大约20个扫描线
    
    for y in range(roi_mask.shape[0] - 1, 0, -scan_step):
        row = roi_mask[y, :]
        lane_pixels = np.where(row > 0)[0]
        
        if len(lane_pixels) > 0:
            # 处理多段车道线（左右车道线）
            if len(lane_pixels) > width * 0.1:  # 如果检测到的点太多，进行聚类
                # 简单的左右分离
                left_pixels = lane_pixels[lane_pixels < width // 2]
                right_pixels = lane_pixels[lane_pixels >= width // 2]
                
                if len(left_pixels) > 0:
                    lane_points.append((np.mean(left_pixels), y + y_offset, 'left'))
                if len(right_pixels) > 0:
                    lane_points.append((np.mean(right_pixels), y + y_offset, 'right'))
            else:
                # 单一车道线或稀疏点
                center_x = np.mean(lane_pixels)
                side = 'left' if center_x < width // 2 else 'right'
                lane_points.append((center_x, y + y_offset, side))
    
    return lane_points

def fit_lane_curves(lane_points, image_width, image_height):
    """
    拟合车道线曲线，支持弯曲道路
    返回车道线参数和可视化点，增强鲁棒性
    """
    if len(lane_points) < 3:
        return None, None, []  # 返回空列表而不是None
    
    # 分离左右车道线
    left_points = [(x, y) for x, y, side in lane_points if side == 'left']
    right_points = [(x, y) for x, y, side in lane_points if side == 'right']
    
    left_curve = None
    right_curve = None
    
    # 拟合左车道线
    if len(left_points) >= 3:
        left_x = np.array([p[0] for p in left_points])
        left_y = np.array([p[1] for p in left_points])
        try:
            # 使用二次多项式拟合弯曲车道
            left_curve = np.polyfit(left_y, left_x, 2)
        except np.RankWarning:
            try:
                # 如果二次拟合失败，尝试一次拟合
                left_curve = np.polyfit(left_y, left_x, 1)
                left_curve = np.append([0], left_curve)  # 补零成为二次多项式格式
            except:
                left_curve = None
        except:
            left_curve = None
    
    # 拟合右车道线
    if len(right_points) >= 3:
        right_x = np.array([p[0] for p in right_points])
        right_y = np.array([p[1] for p in right_points])
        try:
            right_curve = np.polyfit(right_y, right_x, 2)
        except np.RankWarning:
            try:
                right_curve = np.polyfit(right_y, right_x, 1)
                right_curve = np.append([0], right_curve)
            except:
                right_curve = None
        except:
            right_curve = None
    
    # 生成可视化点
    vis_points = []
    y_range = np.linspace(image_height * 0.3, image_height - 1, 50)
    
    if left_curve is not None:
        try:
            left_x_vals = np.polyval(left_curve, y_range)
            for x, y in zip(left_x_vals, y_range):
                if 0 <= x < image_width:
                    vis_points.append((int(x), int(y), 'left'))
        except:
            pass  # 忽略计算错误
    
    if right_curve is not None:
        try:
            right_x_vals = np.polyval(right_curve, y_range)
            for x, y in zip(right_x_vals, y_range):
                if 0 <= x < image_width:
                    vis_points.append((int(x), int(y), 'right'))
        except:
            pass  # 忽略计算错误
    
    return left_curve, right_curve, vis_points

def calculate_steering_control(left_curve, right_curve, image_width, image_height):
    """
    基于车道线拟合结果计算转向控制，增强鲁棒性
    """
    global pid_state
    
    # 车辆在图像中的位置（底部中央）
    car_x = image_width // 2
    car_y = image_height - 1
    
    # 前瞻点（车辆前方的目标点）
    lookahead_y = car_y - CAR_CONTROL["lookahead_distance"]
    lookahead_y = max(lookahead_y, image_height * 0.3)
    
    # 计算车道中心
    lane_center_x = None
    curve_radius = float('inf')
    lane_status = "NO_LANE"
    
    try:
        if left_curve is not None and right_curve is not None:
            # 双车道线情况
            left_x = np.polyval(left_curve, lookahead_y)
            right_x = np.polyval(right_curve, lookahead_y)
            
            # 检查计算结果是否合理
            if (0 <= left_x < image_width and 0 <= right_x < image_width and 
                abs(right_x - left_x) > image_width * 0.1):  # 车道线间距合理
                lane_center_x = (left_x + right_x) / 2
                lane_status = "BOTH_LANES"
                
                # 计算曲率半径
                if len(left_curve) >= 3:
                    a, b, c = left_curve
                    curvature = abs(2 * a) / max((1 + (2 * a * lookahead_y + b) ** 2) ** 1.5, 1e-6)
                    if curvature > 1e-6:
                        curve_radius = min(1 / curvature, 9999)
        
        elif left_curve is not None:
            # 仅有左车道线
            left_x = np.polyval(left_curve, lookahead_y)
            if 0 <= left_x < image_width * 0.8:  # 合理范围内
                estimated_lane_width = image_width * 0.3
                lane_center_x = left_x + estimated_lane_width / 2
                lane_status = "LEFT_ONLY"
                
        elif right_curve is not None:
            # 仅有右车道线
            right_x = np.polyval(right_curve, lookahead_y)
            if image_width * 0.2 <= right_x < image_width:  # 合理范围内
                estimated_lane_width = image_width * 0.3
                lane_center_x = right_x - estimated_lane_width / 2
                lane_status = "RIGHT_ONLY"
    
    except Exception as e:
        print(f"车道线计算错误: {e}")
        lane_center_x = None
        lane_status = "CALCULATION_ERROR"
    
    # 鲁棒性处理：没有检测到车道线的情况
    if lane_center_x is None:
        pid_state["no_lane_counter"] += 1
        
        if pid_state["no_lane_counter"] < 10:  # 短期丢失，保持上次转向
            steering_angle = pid_state["last_valid_steering"] * 0.8  # 逐渐减弱
            lateral_error = steering_angle * image_width / 60.0  # 反向估算偏差
            lane_status = "LOST_TEMPORARY"
        else:  # 长期丢失，紧急停车
            steering_angle = 0.0
            lateral_error = 0.0
            lane_status = "LOST_EMERGENCY"
        
        return steering_angle, lateral_error, curve_radius, lane_status
    
    # 重置丢失计数器
    pid_state["no_lane_counter"] = 0
    
    # 计算横向偏差（像素）
    lateral_error = lane_center_x - car_x
    
    # 转换为角度偏差（简化模型）
    angle_per_pixel = 60.0 / image_width
    error_angle = lateral_error * angle_per_pixel
    
    # PID控制器
    pid_state["integral"] += error_angle
    derivative = error_angle - pid_state["previous_error"]
    
    # 限制积分项防止饱和
    max_integral = 10.0
    pid_state["integral"] = np.clip(pid_state["integral"], -max_integral, max_integral)
    
    # PID输出
    steering_angle = (CAR_CONTROL["kp"] * error_angle + 
                     CAR_CONTROL["ki"] * pid_state["integral"] + 
                     CAR_CONTROL["kd"] * derivative)
    
    # 限制转向角度
    steering_angle = np.clip(steering_angle, -CAR_CONTROL["max_steering_angle"], 
                            CAR_CONTROL["max_steering_angle"])
    
    # 更新状态
    pid_state["previous_error"] = error_angle
    pid_state["error_history"].append(abs(error_angle))
    pid_state["last_valid_steering"] = steering_angle
    
    return steering_angle, lateral_error, curve_radius, lane_status

def calculate_speed_control(steering_angle, curve_radius, lane_status):
    """
    基于转向角度、曲率和车道状态计算速度
    """
    base_speed = CAR_CONTROL["base_speed"]
    
    # 根据车道检测状态调整速度
    if lane_status == "LOST_EMERGENCY":
        return CAR_CONTROL["emergency_stop_speed"]
    elif lane_status in ["LOST_TEMPORARY", "CALCULATION_ERROR"]:
        return CAR_CONTROL["min_speed"]
    elif lane_status in ["LEFT_ONLY", "RIGHT_ONLY"]:
        base_speed *= 0.7  # 单车道线时减速
    
    # 根据转向角度调整速度
    steering_factor = 1.0 - (abs(steering_angle) / CAR_CONTROL["max_steering_angle"]) * 0.5
    
    # 根据曲率调整速度
    if curve_radius < 100:  # 急弯
        curve_factor = CAR_CONTROL["curve_speed_factor"]
    elif curve_radius < 300:  # 缓弯
        curve_factor = 0.8
    else:  # 直道
        curve_factor = 1.0
    
    target_speed = base_speed * steering_factor * curve_factor
    target_speed = np.clip(target_speed, CAR_CONTROL["min_speed"], CAR_CONTROL["max_speed"])
    
    return target_speed

def draw_lane_info_robust(frame, left_curve, right_curve, vis_points, steering_angle, 
                         lateral_error, curve_radius, target_speed, lane_status):
    """
    在图像上绘制车道线信息和控制参数（增强鲁棒性的轻量级绘制）
    """
    height, width = frame.shape[:2]
    
    # 安全地绘制拟合的车道线点
    if vis_points and len(vis_points) > 0:
        for i, (x, y, side) in enumerate(vis_points):
            if i % 3 == 0:  # 每3个点绘制一个，减少绘制量
                # 确保坐标在有效范围内
                if 0 <= x < width and 0 <= y < height:
                    color = (0, 255, 0) if side == 'left' else (0, 0, 255)
                    cv2.circle(frame, (x, y), 2, color, -1)
    
    # 绘制车辆位置和方向
    car_x, car_y = width // 2, height - 1
    cv2.circle(frame, (car_x, car_y), 8, (255, 255, 255), -1)
    cv2.circle(frame, (car_x, car_y), 6, (0, 255, 255), -1)
    
    # 绘制前瞻点
    try:
        lookahead_y = max(car_y - CAR_CONTROL["lookahead_distance"], height * 0.3)
        cv2.circle(frame, (car_x, int(lookahead_y)), 5, (255, 0, 255), -1)
    except:
        pass
    
    # 绘制车道状态指示器
    status_color = {
        "BOTH_LANES": (0, 255, 0),      # 绿色：双车道线
        "LEFT_ONLY": (0, 255, 255),     # 黄色：仅左车道线
        "RIGHT_ONLY": (0, 255, 255),    # 黄色：仅右车道线
        "LOST_TEMPORARY": (0, 165, 255), # 橙色：临时丢失
        "LOST_EMERGENCY": (0, 0, 255),   # 红色：紧急状态
        "NO_LANE": (0, 0, 255),          # 红色：无车道线
        "CALCULATION_ERROR": (128, 0, 128) # 紫色：计算错误
    }.get(lane_status, (128, 128, 128))
    
    cv2.circle(frame, (width - 30, 30), 15, status_color, -1)
    
    # 绘制控制信息（精简版本）
    info_y = 30
    line_height = 25
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # 根据状态调整显示的信息
    if lane_status == "LOST_EMERGENCY":
        texts = [
            f"EMERGENCY STOP!",
            f"Status: {lane_status}",
            f"Speed: {target_speed:.1f}",
            f"No Lane: {pid_state['no_lane_counter']} frames"
        ]
    else:
        texts = [
            f"Steering: {steering_angle:.1f}°",
            f"Deviation: {lateral_error:.1f}px", 
            f"Speed: {target_speed:.1f}",
            f"Status: {lane_status}",
            f"Curve: {curve_radius:.0f}" if curve_radius != float('inf') else "Curve: Straight"
        ]
    
    for i, text in enumerate(texts):
        try:
            # 根据状态选择颜色
            text_color = (0, 0, 255) if "EMERGENCY" in text or "LOST" in lane_status else (0, 255, 0)
            cv2.putText(frame, text, (10, info_y + i * line_height), 
                       font, font_scale, text_color, thickness)
        except:
            pass  # 忽略绘制错误
    
    return frame

# ---------------------------------------------------------------------------------
# --- 极简后处理 (无需裁剪) ---
# ---------------------------------------------------------------------------------

def postprocess_matched_resolution(output_tensor, original_width, original_height):
    """由于输入和模型尺寸匹配，输出也直接对应原始图像，无需裁剪。"""
    pred_mask = np.argmax(output_tensor, axis=1).squeeze()
    vis_mask = (pred_mask * 255).astype(np.uint8)
    return cv2.resize(vis_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

# --- 摄像头抓取线程 ---
def camera_capture_thread():
    print("正在打开摄像头...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头索引 {CAMERA_INDEX}.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n--- 摄像头实际参数 ---")
    print(f"分辨率: {actual_w}x{actual_h}, 帧率: {actual_fps}")
    print("---------------------------\n")

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

# --- 增强的推理和控制线程 ---
def inference_thread():
    global stats_data, data_lock
    print(f"正在加载智能小车控制模型: {MODEL_PATH}")
    model = InferSession(DEVICE_ID, MODEL_PATH)
    print("模型加载完成。")
    
    frame_count = 0
    total_times = {"preprocess": 0, "inference": 0, "postprocess": 0, "control": 0, "pipeline": 0}

    print("\n=== 🚗 鲁棒智能小车车道线控制系统 🚗 ===")
    print("💡 特性: 弯道适应 + PID控制 + 异常处理 + 紧急制动")
    print("每20帧输出一次性能分析...")

    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        loop_start_time = time.time()
        cam_height, cam_width = frame.shape[:2]
        
        # --- 极简CPU预处理 ---
        preprocess_start = time.time()
        input_data = preprocess_matched_resolution(frame, dtype=np.float16)
        preprocess_time_ms = (time.time() - preprocess_start) * 1000
        
        # --- NPU推理 ---
        infer_start_time = time.time()
        outputs = model.infer([input_data])
        inference_time_ms = (time.time() - infer_start_time) * 1000
        
        # --- 极简CPU后处理 ---
        postprocess_start = time.time()
        lane_mask = postprocess_matched_resolution(outputs[0], cam_width, cam_height)
        postprocess_time_ms = (time.time() - postprocess_start) * 1000
        
        # --- 🚗 鲁棒智能车控制逻辑 ---
        control_start = time.time()
        
        try:
            # 提取车道线点
            lane_points = extract_lane_points(lane_mask)
            
            # 拟合车道线曲线
            left_curve, right_curve, vis_points = fit_lane_curves(lane_points, cam_width, cam_height)
            
            # 计算转向控制
            steering_angle, lateral_error, curve_radius, lane_status = calculate_steering_control(
                left_curve, right_curve, cam_width, cam_height)
            
            # 计算速度控制
            target_speed = calculate_speed_control(steering_angle, curve_radius, lane_status)
            
        except Exception as e:
            print(f"控制算法异常: {e}")
            # 故障安全模式
            steering_angle, lateral_error, curve_radius = 0.0, 0.0, float('inf')
            lane_status = "SYSTEM_ERROR"
            target_speed = CAR_CONTROL["emergency_stop_speed"]
            vis_points = []
            left_curve, right_curve = None, None
        
        control_time_ms = (time.time() - control_start) * 1000
        pipeline_latency_ms = (time.time() - loop_start_time) * 1000
        
        # 统计和打印
        frame_count += 1
        total_times["preprocess"] += preprocess_time_ms
        total_times["inference"] += inference_time_ms
        total_times["postprocess"] += postprocess_time_ms
        total_times["control"] += control_time_ms
        total_times["pipeline"] += pipeline_latency_ms
        
        if frame_count % 20 == 0:
            avg_preprocess = total_times["preprocess"] / frame_count
            avg_inference = total_times["inference"] / frame_count
            avg_postprocess = total_times["postprocess"] / frame_count
            avg_control = total_times["control"] / frame_count
            avg_pipeline = total_times["pipeline"] / frame_count
            
            print(f"\n--- 🚗 第{frame_count}帧鲁棒控制分析 ---")
            print(f"【CPU预处理】: {preprocess_time_ms:.1f}ms (平均: {avg_preprocess:.1f}ms)")
            print(f"【NPU 推理】: {inference_time_ms:.1f}ms (平均: {avg_inference:.1f}ms)")
            print(f"【CPU后处理】: {postprocess_time_ms:.1f}ms (平均: {avg_postprocess:.1f}ms)")
            print(f"【控制算法】: {control_time_ms:.1f}ms (平均: {avg_control:.1f}ms)")
            print(f"【转向角度】: {steering_angle:.1f}° | 【目标速度】: {target_speed:.1f}")
            print(f"【车道状态】: {lane_status} | 【偏差】: {lateral_error:.1f}px")
            print(f"【曲率半径】: {curve_radius:.0f} | 【丢失计数】: {pid_state['no_lane_counter']}")
            print(f"【总延迟】: {pipeline_latency_ms:.1f}ms (FPS: {1000/pipeline_latency_ms:.1f})")
            print("=" * 60)

        # 准备结果数据
        result_data = {
            "frame": frame.copy(),
            "mask": lane_mask,
            "left_curve": left_curve,
            "right_curve": right_curve,
            "vis_points": vis_points if vis_points else [],  # 确保不为None
            "steering_angle": steering_angle,
            "lateral_error": lateral_error,
            "curve_radius": curve_radius,
            "target_speed": target_speed,
            "lane_status": lane_status,
            "latency": pipeline_latency_ms,
            "inference_time": inference_time_ms,
            "preprocess_time": preprocess_time_ms,
            "postprocess_time": postprocess_time_ms,
            "control_time": control_time_ms
        }
        
        try:
            if result_queue.empty():
                result_queue.put_nowait(result_data)
        except queue.Full:
            pass
        
        # 更新统计数据
        with data_lock:
            stats_data["pipeline_latency"] = f"{pipeline_latency_ms:.1f}"
            stats_data["inference_time"] = f"{inference_time_ms:.1f}"
            stats_data["preprocess_time"] = f"{preprocess_time_ms:.1f}"
            stats_data["postprocess_time"] = f"{postprocess_time_ms:.1f}"
            stats_data["steering_angle"] = f"{steering_angle:.1f}"
            stats_data["lane_deviation"] = f"{lateral_error:.1f}"
            stats_data["curve_radius"] = f"{curve_radius:.0f}" if curve_radius != float('inf') else "∞"
            stats_data["speed"] = f"{target_speed:.1f}"
            stats_data["lane_status"] = lane_status

# --- 系统监控线程 ---
def system_monitor_loop():
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
                 print(f"❌ NPU监控失败: {NPU_SMI_PATH}")
                 npu_error_printed = True
             with data_lock:
                 stats_data["npu_util"] = "Error"
                 stats_data["npu_mem"] = "Error"
        time.sleep(1)

# --- 增强的HTML模板 ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🚗 鲁棒智能小车控制系统</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f7f6; color: #333; }
        .container { max-width: 1400px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
        h1 { text-align: center; color: #1a73e8; }
        .main-content { display: flex; flex-wrap: wrap; gap: 20px; }
        .video-container { flex: 3; min-width: 600px; }
        #videoStream { width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); background-color: #eee; }
        .stats-container { flex: 1; min-width: 300px; background-color: #f8f9fa; padding: 20px; border-radius: 8px; }
        .stats-container h2 { margin-top: 0; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; color: #3c4043; }
        .stat-grid { display: grid; grid-template-columns: 1fr; gap: 15px; }
        .stat-card { background-color: #fff; padding: 15px; border-radius: 5px; border-left: 5px solid #1a73e8; box-shadow: 0 1px 3px rgba(0,0,0,0.08); display: flex; justify-content: space-between; align-items: center; }
        .stat-card.control { border-left-color: #ff6b35; }
        .stat-card.npu { border-left-color: #34a853; }
        .stat-card.cpu { border-left-color: #fbbc05; }
        .stat-card.e2e { border-left-color: #ea4335; }
        .stat-card.status { border-left-color: #9c27b0; }
        .stat-label { font-size: 14px; color: #5f6368; }
        .stat-value { font-size: 18px; font-weight: 600; color: #202124; }
        .control-badge { background: linear-gradient(45deg, #ff6b35, #f7931e); color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold; margin-left: 10px; }
        .status-indicator { width: 12px; height: 12px; border-radius: 50%; margin-left: 10px; }
        .status-good { background-color: #4caf50; }
        .status-warning { background-color: #ff9800; }
        .status-error { background-color: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 鲁棒智能小车控制系统 <span class="control-badge">故障安全设计</span></h1>
        <div class="main-content">
            <div class="video-container">
                <img id="videoStream" src="{{ url_for('video_feed') }}" alt="Live Stream">
            </div>
            <div class="stats-container">
                <h2>📊 系统状态监控</h2>
                <div class="stat-grid">
                    <div class="stat-card"><span class="stat-label">显示帧率 (FPS)</span><span id="fps" class="stat-value">--</span></div>
                    <div class="stat-card e2e"><span class="stat-label">系统总延迟 (ms)</span><span id="pipeline_latency" class="stat-value">--</span></div>
                    <div class="stat-card e2e"><span class="stat-label">NPU 推理 (ms)</span><span id="inference_time" class="stat-value">--</span></div>
                    <div class="stat-card status"><span class="stat-label">车道检测状态</span><span id="lane_status" class="stat-value">--</span><div id="status_indicator" class="status-indicator"></div></div>
                    <div class="stat-card control"><span class="stat-label">转向角度 (°)</span><span id="steering_angle" class="stat-value">--</span></div>
                    <div class="stat-card control"><span class="stat-label">车道偏差 (px)</span><span id="lane_deviation" class="stat-value">--</span></div>
                    <div class="stat-card control"><span class="stat-label">曲率半径</span><span id="curve_radius" class="stat-value">--</span></div>
                    <div class="stat-card control"><span class="stat-label">目标速度</span><span id="speed" class="stat-value">--</span></div>
                    <div class="stat-card npu"><span class="stat-label">NPU 利用率 (%)</span><span id="npu_util" class="stat-value">--</span></div>
                    <div class="stat-card npu"><span class="stat-label">NPU 内存占用</span><span id="npu_mem" class="stat-value">--</span></div>
                    <div class="stat-card cpu"><span class="stat-label">CPU 利用率 (%)</span><span id="cpu_percent" class="stat-value">--</span></div>
                    <div class="stat-card cpu"><span class="stat-label">系统内存占用 (%)</span><span id="mem_percent" class="stat-value">--</span></div>
                </div>
            </div>
        </div>
    </div>
    <script>
        function updateStatusIndicator(status) {
            const indicator = document.getElementById('status_indicator');
            if (status.includes('BOTH') || status.includes('DETECTING')) {
                indicator.className = 'status-indicator status-good';
            } else if (status.includes('ONLY') || status.includes('TEMPORARY')) {
                indicator.className = 'status-indicator status-warning';
            } else {
                indicator.className = 'status-indicator status-error';
            }
        }
        
        function refreshStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps;
                    document.getElementById('pipeline_latency').textContent = data.pipeline_latency + ' ms';
                    document.getElementById('inference_time').textContent = data.inference_time + ' ms';
                    document.getElementById('lane_status').textContent = data.lane_status || 'DETECTING';
                    document.getElementById('steering_angle').textContent = data.steering_angle + '°';
                    document.getElementById('lane_deviation').textContent = data.lane_deviation + 'px';
                    document.getElementById('curve_radius').textContent = data.curve_radius;
                    document.getElementById('speed').textContent = data.speed;
                    document.getElementById('npu_util').textContent = data.npu_util;
                    document.getElementById('npu_mem').textContent = data.npu_mem;
                    document.getElementById('cpu_percent').textContent = data.cpu_percent + ' %';
                    document.getElementById('mem_percent').textContent = data.mem_percent + ' %';
                    updateStatusIndicator(data.lane_status || 'DETECTING');
                })
                .catch(error => console.error('获取统计信息失败:', error));
        }
        window.onload = function() {
            refreshStats();
            setInterval(refreshStats, 1000);
        };
    </script>
</body>
</html>
"""

# --- Flask路由 ---
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
            
            # 鲁棒地绘制车道线信息
            try:
                vis_frame = draw_lane_info_robust(
                    frame, 
                    result.get("left_curve"), 
                    result.get("right_curve"),
                    result.get("vis_points", []),  # 提供默认空列表
                    result.get("steering_angle", 0.0),
                    result.get("lateral_error", 0.0),
                    result.get("curve_radius", float('inf')),
                    result.get("target_speed", 0.0),
                    result.get("lane_status", "UNKNOWN")
                )
            except Exception as e:
                print(f"绘制错误: {e}")
                vis_frame = frame  # 使用原始帧作为备选
            
            # 计算显示帧率
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                with data_lock:
                    stats_data["fps"] = f"{fps:.1f}"
                frame_count = 0
                start_time = time.time()
            
            # 编码图像
            try:
                (flag, encodedImage) = cv2.imencode(".jpg", vis_frame)
                if not flag:
                    continue
                    
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                      bytearray(encodedImage) + b'\r\n')
            except Exception as e:
                print(f"图像编码错误: {e}")
                continue
    
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stats")
def stats():
    with data_lock:
        return jsonify(stats_data)

if __name__ == '__main__':
    print("🚗 鲁棒智能小车车道线控制系统启动")
    print("=============================================================")
    print(f"🧠 模型: {MODEL_PATH}")
    print(f"🎯 输入尺寸: {MODEL_WIDTH}x{MODEL_HEIGHT}")
    print(f"⚡ 安全特性: 故障安全 + 紧急制动 + 异常恢复")
    print(f"🔧 PID参数: Kp={CAR_CONTROL['kp']}, Ki={CAR_CONTROL['ki']}, Kd={CAR_CONTROL['kd']}")
    print("=============================================================")
    
    Thread(target=camera_capture_thread, daemon=True).start()
    Thread(target=inference_thread, daemon=True).start()
    Thread(target=system_monitor_loop, daemon=True).start()
    
    print("\nWeb服务器已启动。请在浏览器中访问: http://<Your_Atlas_IP>:8000")
    print("鲁棒控制参数将在终端实时显示...")
    app.run(host='0.0.0.0', port=8000, threaded=True)