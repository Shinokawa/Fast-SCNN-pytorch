import os
import cv2
import time
import numpy as np
from threading import Thread, Lock
import queue # 引入队列模块
import psutil
import subprocess
import re

from ais_bench.infer.interface import InferSession
from flask import Flask, Response, render_template_string, jsonify

# --- Flask App 和全局变量 ---
app = Flask(__name__)
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
stats_data = {
    "fps": "0.0", "pipeline_latency": "0.0", "inference_time": "0.0",
    "preprocess_time": "0.0", "postprocess_time": "0.0",
    "cpu_percent": "0.0", "mem_percent": "0.0", "npu_util": "N/A", "npu_mem": "N/A"
}
data_lock = Lock()

# --- 可配置常量 ---
DEVICE_ID = 0
MODEL_PATH = "./weights/fast_scnn_tusimple_bs1.om"
MODEL_WIDTH = 640
MODEL_HEIGHT = 640
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 640
NPU_SMI_PATH = "/usr/local/Ascend/driver/tools/npu-smi"

# --------------------------------------------------------------------------
# --- ⚡⚡⚡ 新增和修改的核心部分：预处理函数 ⚡⚡⚡ ---
# --------------------------------------------------------------------------

# 预计算常数（保留给原始优化版对比用）
NORM_MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
NORM_STD_NP = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
NORM_SCALE = (1.0 / 255.0) / NORM_STD_NP
NORM_OFFSET = -NORM_MEAN_NP / NORM_STD_NP

def preprocess_optimized_numpy(img_bgr, dtype=np.float32):
    """你原来的优化版，基于NumPy，用于对比。增加了dtype参数。"""
    img_resized = cv2.resize(img_bgr, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_normalized = img_rgb * NORM_SCALE + NORM_OFFSET
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    input_data = np.ascontiguousarray(img_transposed[np.newaxis, :, :, :], dtype=dtype)
    return input_data

# --- 这是我们的新王牌：结合 cv2.dnn.blobFromImage 和 FP16 ---
# 为 blobFromImage 准备的常数
BLOB_MEAN_FOR_SUBTRACTION = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
BLOB_STD_FOR_DIVISION = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

def preprocess_ultimate_optimized(img_bgr, dtype=np.float16):
    """
    终极优化预处理函数:
    1. 使用 cv2.dnn.blobFromImage 进行高速 缩放, BGR->RGB, /255.0。
    2. 使用 NumPy 广播操作快速完成减均值和除以标准差。
    3. 最终转换为指定数据类型 (默认为FP16)。
    """
    # 步骤1: 用blobFromImage高效完成 缩放, BGR->RGB转换, 并归一化到[0,1]
    # scalefactor=1.0/255.0: 将像素值从0-255缩放到0-1
    # swapRB=True: 将BGR图像转换为RGB，因为模型通常需要RGB
    blob = cv2.dnn.blobFromImage(img_bgr, 
                                 scalefactor=1.0/255.0, 
                                 size=(MODEL_WIDTH, MODEL_HEIGHT),
                                 swapRB=True, 
                                 crop=False) # blob的shape是(1, 3, H, W)

    # 步骤2: 手动进行减均值和除以标准差
    # (blob - mean) / std
    blob -= BLOB_MEAN_FOR_SUBTRACTION
    blob /= BLOB_STD_FOR_DIVISION
    
    # 步骤3: 转换为目标数据类型并确保内存连续性
    # 如果你的模型需要FP16，这一步至关重要
    return blob.astype(dtype)

# --- 选择使用的预处理函数 ---
# 切换到新的终极优化版本！
preprocess = preprocess_ultimate_optimized
# 如果需要对比，可以切换回旧版
# preprocess = lambda img: preprocess_optimized_numpy(img, dtype=np.float16)


def postprocess(output_tensor, original_width, original_height):
    """后处理函数保持不变"""
    pred_mask = np.argmax(output_tensor, axis=1).squeeze()
    vis_mask = (pred_mask * 255).astype(np.uint8)
    vis_mask_resized = cv2.resize(vis_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    return vis_mask_resized

# --- 摄像头抓取线程 (无修改) ---
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
    print("\n--- 摄像头实际参数 ---")
    print(f"分辨率: {actual_w}x{actual_h}, 帧率: {actual_fps}")
    print("---------------------------\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass
    cap.release()

# --- 推理线程 (修改以使用新预处理函数并打印数据类型) ---
def inference_thread():
    global stats_data, data_lock
    print("正在加载模型...")
    model = InferSession(DEVICE_ID, MODEL_PATH)
    print("模型加载完成。")
    
    frame_count = 0
    total_times = {"preprocess": 0, "inference": 0, "postprocess": 0, "pipeline": 0}
    
    # 增加一个变量来确认数据类型
    input_dtype_str = "N/A"

    print("\n=== 开始性能监控 (终极优化版: FP16 + blobFromImage) ===")
    print("每20帧输出一次详细性能分析...")

    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        loop_start_time = time.time()
        cam_height, cam_width = frame.shape[:2]
        
        # --- 预处理计时 ---
        preprocess_start = time.time()
        # 调用我们选择的预处理函数
        # 默认使用 preprocess_ultimate_optimized, 并传入 np.float16
        input_data = preprocess(frame, dtype=np.float16) 
        preprocess_time_ms = (time.time() - preprocess_start) * 1000
        
        # 记录数据类型，只记录一次
        if frame_count == 0:
            input_dtype_str = str(input_data.dtype)

        # --- 推理计时 ---
        infer_start_time = time.time()
        outputs = model.infer([input_data])
        inference_time_ms = (time.time() - infer_start_time) * 1000
        
        # --- 后处理计时 ---
        postprocess_start = time.time()
        lane_mask = postprocess(outputs[0], cam_width, cam_height)
        postprocess_time_ms = (time.time() - postprocess_start) * 1000
        
        pipeline_latency_ms = (time.time() - loop_start_time) * 1000
        
        # 累计统计
        frame_count += 1
        total_times["preprocess"] += preprocess_time_ms
        total_times["inference"] += inference_time_ms
        total_times["postprocess"] += postprocess_time_ms
        total_times["pipeline"] += pipeline_latency_ms
        
        if frame_count % 20 == 0:
            avg_preprocess = total_times["preprocess"] / frame_count
            avg_inference = total_times["inference"] / frame_count
            avg_postprocess = total_times["postprocess"] / frame_count
            avg_pipeline = total_times["pipeline"] / frame_count
            
            print(f"\n--- 第{frame_count}帧性能分析 (终极优化版) ---")
            print(f"输入帧: {cam_width}x{cam_height} -> 模型输入: {MODEL_WIDTH}x{MODEL_HEIGHT}")
            print(f"🔥 模型输入数据类型: {input_dtype_str}")
            print(f"")
            print(f"【预处理】: {preprocess_time_ms:.1f}ms   (平均: {avg_preprocess:.1f}ms) ⚡️")
            print(f"【推  理】: {inference_time_ms:.1f}ms   (平均: {avg_inference:.1f}ms) ⚡️")
            print(f"【后处理】: {postprocess_time_ms:.1f}ms   (平均: {avg_postprocess:.1f}ms)")
            print(f"--------------------------------------------------")
            print(f"【流水线总延迟】: {pipeline_latency_ms:.1f}ms (理论FPS: {1000/pipeline_latency_ms:.1f})")
            print(f"【平均总延迟】  : {avg_pipeline:.1f}ms (平均FPS: {1000/avg_pipeline:.1f})")
            print(f"")

            # 瓶颈分析
            max_time = max(avg_preprocess, avg_inference, avg_postprocess)
            if max_time == avg_preprocess:
                print("🔴 当前瓶颈: 预处理 (CPU)")
                print("   🤔 优化建议: 如果仍然是瓶颈，请考虑使用AIPP进行硬件预处理。")
            elif max_time == avg_inference:
                print("🟡 当前瓶颈: NPU推理")
                print("   ✅ 优化状态: 已使用FP16，若需更快可考虑INT8量化或更轻量模型。")
            else:
                print("🟢 当前瓶颈: 后处理")
                print("   ✅ 优化状态: 后处理通常不是主要瓶颈。")
            print("=" * 60)

        try:
            result_queue.put_nowait({
                "frame": frame, "mask": lane_mask,
                "latency": pipeline_latency_ms, "inference_time": inference_time_ms,
                "preprocess_time": preprocess_time_ms, "postprocess_time": postprocess_time_ms
            })
        except queue.Full:
            pass
        
        with data_lock:
            stats_data["pipeline_latency"] = f"{pipeline_latency_ms:.1f}"
            stats_data["inference_time"] = f"{inference_time_ms:.1f}"
            stats_data["preprocess_time"] = f"{preprocess_time_ms:.1f}"
            stats_data["postprocess_time"] = f"{postprocess_time_ms:.1f}"


# --- NPU/CPU 监控线程 和 Flask部分 (无修改) ---
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
                 print(f"❌ NPU监控失败，请检查路径: {NPU_SMI_PATH}")
                 npu_error_printed = True
             with data_lock:
                 stats_data["npu_util"] = "Error"
                 stats_data["npu_mem"] = "Error"
        time.sleep(1)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>车道线检测实时推理</title>
    <meta charset="UTF-8">
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
        .stat-card.npu { border-left-color: #34a853; }
        .stat-card.cpu { border-left-color: #fbbc05; }
        .stat-label { font-size: 14px; color: #5f6368; }
        .stat-value { font-size: 18px; font-weight: 600; color: #202124; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 车道线检测实时推理监控 (终极优化版: FP16 + blobFromImage)</h1>
        <div class="main-content">
            <div class="video-container">
                <img id="videoStream" src="{{ url_for('video_feed') }}" alt="Live Inference Stream">
            </div>
            <div class="stats-container">
                <h2>📊 性能监控</h2>
                <div class="stat-grid">
                    <div class="stat-card">
                        <span class="stat-label">显示帧率 (FPS)</span>
                        <span id="fps" class="stat-value">--</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-label">处理流水线延迟 (ms)</span>
                        <span id="pipeline_latency" class="stat-value">--</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-label">模型推理耗时 (ms)</span>
                        <span id="inference_time" class="stat-value">--</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-label">预处理耗时 (ms)</span>
                        <span id="preprocess_time" class="stat-value">--</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-label">后处理耗时 (ms)</span>
                        <span id="postprocess_time" class="stat-value">--</span>
                    </div>
                    <div class="stat-card npu">
                        <span class="stat-label">NPU 利用率 (%)</span>
                        <span id="npu_util" class="stat-value">--</span>
                    </div>
                    <div class="stat-card npu">
                        <span class="stat-label">NPU 内存占用</span>
                        <span id="npu_mem" class="stat-value">--</span>
                    </div>
                    <div class="stat-card cpu">
                        <span class="stat-label">CPU 利用率 (%)</span>
                        <span id="cpu_percent" class="stat-value">--</span>
                    </div>
                    <div class="stat-card cpu">
                        <span class="stat-label">系统内存占用 (%)</span>
                        <span id="mem_percent" class="stat-value">--</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        function refreshStats() {
            fetch('/stats')
            .then(response => response.json())
            .then(data => {
                document.getElementById('fps').textContent = data.fps;
                document.getElementById('pipeline_latency').textContent = data.pipeline_latency + ' ms';
                document.getElementById('inference_time').textContent = data.inference_time + ' ms';
                document.getElementById('preprocess_time').textContent = data.preprocess_time + ' ms';
                document.getElementById('postprocess_time').textContent = data.postprocess_time + ' ms';
                document.getElementById('npu_util').textContent = data.npu_util + ' %';
                document.getElementById('npu_mem').textContent = data.npu_mem;
                document.getElementById('cpu_percent').textContent = data.cpu_percent + ' %';
                document.getElementById('mem_percent').textContent = data.mem_percent + ' %';
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
            green_overlay = np.zeros_like(frame, dtype=np.uint8)
            green_overlay[mask > 0] = [0, 255, 0]
            vis_frame = cv2.addWeighted(frame, 1.0, green_overlay, 0.5, 0)
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                with data_lock:
                    stats_data["fps"] = f"{fps:.1f}"
                frame_count = 0
                start_time = time.time()
            with data_lock:
                fps_text = f"Display FPS: {stats_data['fps']}"
                latency_text = f"Latency: {stats_data['pipeline_latency']} ms"
            cv2.putText(vis_frame, fps_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(vis_frame, latency_text, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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

if __name__ == '__main__':
    print("🚀 车道线检测实时推理系统启动 (终极优化版: FP16 + blobFromImage)")
    print("=" * 60)
    print(f"📷 摄像头配置: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_INDEX}")
    print(f"🧠 模型配置: {MODEL_PATH}")
    print(f"   - 输入尺寸: {MODEL_WIDTH}x{MODEL_HEIGHT}, 设备ID: {DEVICE_ID}")
    print(f"⚡ 优化特性:")
    print(f"   - ✅ 预处理: cv2.dnn.blobFromImage (高速CPU处理)")
    print(f"   - ✅ 数据类型: FP16 (加速数据传输和NPU推理)")
    print(f"💻 系统信息:")
    print(f"   - CPU核心: {psutil.cpu_count()}, 内存: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"   - NPU工具路径: {NPU_SMI_PATH}")
    print("=" * 60)
    
    Thread(target=camera_capture_thread, daemon=True).start()
    Thread(target=inference_thread, daemon=True).start()
    Thread(target=system_monitor_loop, daemon=True).start()
    
    print("Web服务器已启动。请在浏览器中访问: http://<Your_Atlas_IP>:8000")
    print("终端将显示详细的性能分析信息...")
    app.run(host='0.0.0.0', port=8000, threaded=True)