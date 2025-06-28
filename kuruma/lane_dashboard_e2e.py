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

# --- 【关键】可配置常量 ---
DEVICE_ID = 0
# 使用您的新模型路径
MODEL_PATH = "./weights/fast_scnn_480x640_e2e_fp16_input.om" 
# 模型输入尺寸与摄像头输出完全匹配
MODEL_WIDTH = 640
MODEL_HEIGHT = 480
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
NPU_SMI_PATH = "/usr/local/Ascend/driver/tools/npu-smi"

# ---------------------------------------------------------------------------------
# --- 🚀🚀🚀 极简预处理 (分辨率匹配) 🚀🚀🚀 ---
# ---------------------------------------------------------------------------------

def preprocess_matched_resolution(img_bgr, dtype=np.float16):
    """
    当模型输入分辨率与摄像头输出完全匹配时，预处理开销最小化。
    CPU只负责最基本的数据格式和类型转换。
    """
    # 1. 转换颜色通道 (BGR -> RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. 转换数据类型 (uint8 -> float16)
    img_typed = img_rgb.astype(dtype)
    
    # 3. 转换为CHW格式并添加batch维度
    img_transposed = np.transpose(img_typed, (2, 0, 1))
    return np.ascontiguousarray(img_transposed[np.newaxis, :, :, :])

# ---------------------------------------------------------------------------------
# --- 🚀🚀🚀 极简后处理 (无需裁剪) 🚀🚀🚀 ---
# ---------------------------------------------------------------------------------

def postprocess_matched_resolution(output_tensor, original_width, original_height):
    """
    由于输入和模型尺寸匹配，输出也直接对应原始图像，无需裁剪。
    """
    # 1. Argmax获取分割掩码
    pred_mask = np.argmax(output_tensor, axis=1).squeeze()
    
    # 2. 转换为可视化格式
    vis_mask = (pred_mask * 255).astype(np.uint8)
    
    # 3. （可选）确保尺寸完全一致，对于尺寸匹配的情况，这一步几乎无开销
    return cv2.resize(vis_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)


# --- 摄像头抓取线程 (无修改) ---
def camera_capture_thread():
    print("正在打开摄像头...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头索引 {CAMERA_INDEX}.")
        return

    # 设置摄像头参数，确保输出是我们期望的尺寸
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 再次确认摄像头的实际输出参数
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print("\n--- 摄像头实际参数 ---")
    print(f"分辨率: {actual_w}x{actual_h}, 帧率: {actual_fps}")
    print("---------------------------\n")

    if actual_w != CAMERA_WIDTH or actual_h != CAMERA_HEIGHT:
        print(f"⚠️ 警告: 摄像头实际输出 ({actual_w}x{actual_h}) 与期望 ({CAMERA_WIDTH}x{CAMERA_HEIGHT}) 不符!")

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

# --- 推理线程 (使用极简化的预处理和后处理) ---
def inference_thread():
    global stats_data, data_lock
    print(f"正在加载[分辨率匹配]的模型: {MODEL_PATH}")
    model = InferSession(DEVICE_ID, MODEL_PATH)
    print("模型加载完成。")
    
    frame_count = 0
    total_times = {"preprocess": 0, "inference": 0, "postprocess": 0, "pipeline": 0}

    print("\n=== 🚀 分辨率匹配-极致性能监控 🚀 ===")
    print("💡 预处理: 无Resize/Padding，仅格式转换，CPU开销最小化！")
    print("每20帧输出一次详细性能分析...")

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
        
        pipeline_latency_ms = (time.time() - loop_start_time) * 1000
        
        # 统计和打印
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
            
            print(f"\n--- ⚡ 第{frame_count}帧性能分析 (分辨率匹配) ---")
            print(f"输入 -> 模型: {cam_width}x{cam_height} -> {MODEL_WIDTH}x{MODEL_HEIGHT} (完美匹配)")
            print(f"🎯 数据类型: {str(input_data.dtype).upper()}")
            print(f"【CPU预处理】: {preprocess_time_ms:.1f}ms (平均: {avg_preprocess:.1f}ms) ⚡")
            print(f"【NPU 推理】: {inference_time_ms:.1f}ms (平均: {avg_inference:.1f}ms) 🚀")
            print(f"【CPU后处理】: {postprocess_time_ms:.1f}ms (平均: {avg_postprocess:.1f}ms)")
            print(f"--------------------------------------------------")
            print(f"【流水线总延迟】: {pipeline_latency_ms:.1f}ms (理论FPS: {1000/pipeline_latency_ms:.1f})")
            print(f"【平均总延迟】  : {avg_pipeline:.1f}ms (平均FPS: {1000/avg_pipeline:.1f})")
            print("=" * 60)

        # 结果队列和数据锁
        try:
            if result_queue.empty():
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


# --- NPU/CPU 监控线程, Flask路由和主函数 (保持不变) ---
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
                 print(f"❌ NPU监控失败，请检查路径或权限: {NPU_SMI_PATH}")
                 npu_error_printed = True
             with data_lock:
                 stats_data["npu_util"] = "Error"
                 stats_data["npu_mem"] = "Error"
        time.sleep(1)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>车道线检测 (分辨率匹配优化版)</title>
    <!-- ... 此处省略CSS样式，与之前相同 ... -->
    <style>body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f7f6; color: #333; } .container { max-width: 1400px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); } h1 { text-align: center; color: #1a73e8; } .main-content { display: flex; flex-wrap: wrap; gap: 20px; } .video-container { flex: 3; min-width: 600px; } #videoStream { width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); background-color: #eee; } .stats-container { flex: 1; min-width: 300px; background-color: #f8f9fa; padding: 20px; border-radius: 8px; } .stats-container h2 { margin-top: 0; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; color: #3c4043; } .stat-grid { display: grid; grid-template-columns: 1fr; gap: 15px; } .stat-card { background-color: #fff; padding: 15px; border-radius: 5px; border-left: 5px solid #1a73e8; box-shadow: 0 1px 3px rgba(0,0,0,0.08); display: flex; justify-content: space-between; align-items: center; } .stat-card.npu { border-left-color: #34a853; } .stat-card.cpu { border-left-color: #fbbc05; } .stat-card.e2e { border-left-color: #ea4335; } .stat-label { font-size: 14px; color: #5f6368; } .stat-value { font-size: 18px; font-weight: 600; color: #202124; } .optimization-badge { background: linear-gradient(45deg, #34A853, #4285F4); color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold; margin-left: 10px; }</style>
</head>
<body>
    <div class="container">
        <h1>🚀 车道线检测 <span class="optimization-badge">分辨率匹配-性能最优</span></h1>
        <!-- ... 此处省略HTML结构，与之前相同 ... -->
        <div class="main-content"><div class="video-container"><img id="videoStream" src="{{ url_for('video_feed') }}" alt="Live Stream"></div><div class="stats-container"><h2>📊 性能监控</h2><div class="stat-grid"><div class="stat-card"><span class="stat-label">显示帧率 (FPS)</span><span id="fps" class="stat-value">--</span></div><div class="stat-card e2e"><span class="stat-label">端到端流水线延迟 (ms)</span><span id="pipeline_latency" class="stat-value">--</span></div><div class="stat-card e2e"><span class="stat-label">NPU 推理 (ms)</span><span id="inference_time" class="stat-value">--</span></div><div class="stat-card"><span class="stat-label">CPU预处理 (ms)</span><span id="preprocess_time" class="stat-value">--</span></div><div class="stat-card"><span class="stat-label">后处理耗时 (ms)</span><span id="postprocess_time" class="stat-value">--</span></div><div class="stat-card npu"><span class="stat-label">NPU 利用率 (%)</span><span id="npu_util" class="stat-value">--</span></div><div class="stat-card npu"><span class="stat-label">NPU 内存占用</span><span id="npu_mem" class="stat-value">--</span></div><div class="stat-card cpu"><span class="stat-label">CPU 利用率 (%)</span><span id="cpu_percent" class="stat-value">--</span></div><div class="stat-card cpu"><span class="stat-label">系统内存占用 (%)</span><span id="mem_percent" class="stat-value">--</span></div></div></div></div>
    </div>
    <!-- ... 此处省略JS脚本，与之前相同 ... -->
    <script>function refreshStats() { fetch('/stats').then(response => response.json()).then(data => { document.getElementById('fps').textContent = data.fps; document.getElementById('pipeline_latency').textContent = data.pipeline_latency + ' ms'; document.getElementById('inference_time').textContent = data.inference_time + ' ms'; document.getElementById('preprocess_time').textContent = data.preprocess_time + ' ms'; document.getElementById('postprocess_time').textContent = data.postprocess_time + ' ms'; document.getElementById('npu_util').textContent = data.npu_util; document.getElementById('npu_mem').textContent = data.npu_mem; document.getElementById('cpu_percent').textContent = data.cpu_percent + ' %'; document.getElementById('mem_percent').textContent = data.mem_percent + ' %'; }).catch(error => console.error('获取统计信息失败:', error)); } window.onload = function() { refreshStats(); setInterval(refreshStats, 1000); };</script>
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
    print("🚀 车道线检测 [分辨率匹配] 实时推理系统启动")
    print("=============================================================")
    print(f"🧠 模型: {MODEL_PATH}")
    print(f"🎯 输入尺寸: {MODEL_WIDTH}x{MODEL_HEIGHT} (与摄像头匹配)")
    print(f"⚡ 优化: 无需Resize/Padding，CPU预处理开销已降至最低！")
    print("=============================================================")
    
    Thread(target=camera_capture_thread, daemon=True).start()
    Thread(target=inference_thread, daemon=True).start()
    Thread(target=system_monitor_loop, daemon=True).start()
    
    print("\nWeb服务器已启动。请在浏览器中访问: http://<Your_Atlas_IP>:8000")
    print("终端将显示性能分析...")
    app.run(host='0.0.0.0', port=8000, threaded=True)