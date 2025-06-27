import os
import cv2
import time
import numpy as np
from threading import Thread, Lock
import queue # 引入队列模块
import psutil
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# ⚡ 启用NumPy多线程优化
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())

from ais_bench.infer.interface import InferSession
from flask import Flask, Response, render_template_string, jsonify

# --- Flask App 和全局变量 (多线程优化版) ---
app = Flask(__name__)
# 使用更大的队列来支持多线程处理
frame_queue = queue.Queue(maxsize=3)           # 原始帧队列
processed_queue = queue.Queue(maxsize=3)       # 预处理后的数据队列
result_queue = queue.Queue(maxsize=1)          # 最终结果队列

# 多线程预处理相关队列
preprocess_input_queue = queue.Queue(maxsize=5)
preprocess_output_queue = queue.Queue(maxsize=5)

stats_data = { # 初始化所有统计数据 (包含详细计时)
    "fps": "0.0", "pipeline_latency": "0.0", "inference_time": "0.0",
    "preprocess_time": "0.0", "postprocess_time": "0.0", "preprocess_threads": "0",
    "cpu_percent": "0.0", "mem_percent": "0.0", "npu_util": "N/A", "npu_mem": "N/A"
}
data_lock = Lock() # 仅用于保护 stats_data 的写入

# --- 可配置常量 (多线程优化版本) ---
DEVICE_ID = 0
MODEL_PATH = "./weights/fast_scnn_tusimple_bs1.om"
MODEL_WIDTH = 1024
MODEL_HEIGHT = 768
CAMERA_INDEX = 0

# 📊 摄像头分辨率配置
CAMERA_WIDTH = 1280   # 实测最佳平衡点
CAMERA_HEIGHT = 720

# ⚡ 多线程和批处理优化配置
PREPROCESS_THREADS = 2      # 预处理线程数量
BATCH_SIZE = 1              # 暂时保持1，避免延迟增加
ENABLE_NUMPY_THREADING = True  # 启用NumPy多线程

NPU_SMI_PATH = "/usr/local/Ascend/driver/tools/npu-smi"

# 📊 摄像头分辨率配置 (根据性能需求选择)
# 根据v4l2-ctl输出，您的摄像头支持以下分辨率@帧率:
# 
# 🟢 推荐配置 (性能优先):
# CAMERA_WIDTH = 640    # 测试结果: 反而更慢，缩放开销大
# CAMERA_HEIGHT = 480
#
# 🟡 平衡配置 (当前最优):
CAMERA_WIDTH = 1280   # 实测最佳平衡点
CAMERA_HEIGHT = 720
#
# 🔴 高质量配置 (质量优先):
# CAMERA_WIDTH = 1920   # 最高质量，预处理时间 ~150ms
# CAMERA_HEIGHT = 1080
#
# 💡 结论: 1280x720是最佳平衡点，现在专注优化归一化计算

NPU_SMI_PATH = "/usr/local/Ascend/driver/tools/npu-smi" # <--- 在这里填入你 'which npu-smi' 找到的路径

# --- 预处理和后处理函数 (多线程优化版本) ---

# 预计算归一化常数以避免重复计算
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
# 预计算组合常数: (x/255 - mean) / std = x * scale + offset
NORM_SCALE = (1.0 / 255.0) / NORM_STD
NORM_OFFSET = -NORM_MEAN / NORM_STD

def preprocess_single_optimized(img_bgr):
    """单帧优化预处理，用于多线程调用"""
    # 步骤1: 图像缩放
    img_resized = cv2.resize(img_bgr, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    # 步骤2: 颜色转换 + 数据类型转换（合并操作）
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    
    # 步骤3: 超级优化的归一化 - 一次性完成所有计算
    img_normalized = img_rgb * NORM_SCALE + NORM_OFFSET
    
    # 步骤4: 转置（不添加batch维度，在主线程中处理）
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    return img_transposed

def preprocess_worker():
    """预处理工作线程函数"""
    thread_id = multiprocessing.current_process().name
    print(f"预处理线程 {thread_id} 启动")
    
    while True:
        try:
            # 获取待处理的帧
            item = preprocess_input_queue.get(timeout=1)
            if item is None:  # 退出信号
                break
                
            frame_id, frame, timestamp = item
            
            # 执行预处理
            start_time = time.time()
            processed_data = preprocess_single_optimized(frame)
            process_time = (time.time() - start_time) * 1000
            
            # 返回结果
            preprocess_output_queue.put((frame_id, processed_data, process_time, timestamp))
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"预处理线程 {thread_id} 错误: {e}")

def preprocess_batch(img_list):
    """批处理预处理函数（如果需要批处理）"""
    if len(img_list) == 1:
        processed = preprocess_single_optimized(img_list[0])
        return np.ascontiguousarray(processed[np.newaxis, :, :, :], dtype=np.float32)
    
    # 批处理逻辑（暂时保留单个处理）
    batch_data = []
    for img in img_list:
        processed = preprocess_single_optimized(img)
        batch_data.append(processed)
    
    return np.ascontiguousarray(np.stack(batch_data, axis=0), dtype=np.float32)

def preprocess_optimized(img_bgr):
    """高度优化的预处理函数，专门优化归一化计算"""
    # 步骤1: 图像缩放
    img_resized = cv2.resize(img_bgr, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    # 步骤2: 颜色转换 + 数据类型转换（合并操作）
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    
    # 步骤3: 超级优化的归一化 - 一次性完成所有计算
    # 原来: (img/255 - mean) / std
    # 优化: img * scale + offset (其中 scale = 1/(255*std), offset = -mean/std)
    img_normalized = img_rgb * NORM_SCALE + NORM_OFFSET
    
    # 步骤4: 转置和内存整理（使用更快的方法）
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    input_data = np.ascontiguousarray(img_transposed[np.newaxis, :, :, :], dtype=np.float32)
    return input_data

def preprocess_original(img_bgr):
    """原始预处理函数，用于对比"""
    # 优化1: 如果输入分辨率已经接近目标分辨率，使用更快的插值方法
    img_resized = cv2.resize(img_bgr, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    # 优化2: 合并颜色转换和归一化步骤
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    
    # 优化3: 使用向量化操作
    img_normalized = (img_rgb / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # 优化4: 直接转置并确保内存连续
    input_data = np.ascontiguousarray(img_normalized.transpose(2, 0, 1)[np.newaxis, :, :, :], dtype=np.float32)
    return input_data

# 选择使用哪个版本（可以切换测试）
preprocess = preprocess_optimized  # 使用优化版本
# preprocess = preprocess_original  # 使用原始版本

def postprocess(output_tensor, original_width, original_height):
    pred_mask = np.argmax(output_tensor, axis=1).squeeze()
    vis_mask = (pred_mask * 255).astype(np.uint8)
    vis_mask_resized = cv2.resize(vis_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    return vis_mask_resized

# --- 摄像头抓取线程 (优化版本) ---
def camera_capture_thread():
    print("正在打开摄像头...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头索引 {CAMERA_INDEX}.")
        return

    # 设置MJPG格式以获得更好的性能
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 优化缓冲区设置
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区延迟

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print("\n--- 摄像头实际参数 (优化版) ---")
    print(f"分辨率: {actual_w}x{actual_h}, 帧率: {actual_fps}")
    print(f"理论预处理数据量减少: {((1920*1080)/(actual_w*actual_h)):.1f}x")
    print("------------------------------------\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        try:
            # 尝试放入队列，如果队列满了（因为推理线程慢），就直接丢弃旧的
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass # 队列已满，忽略这一帧，等待下一次循环
    cap.release()


# --- 多线程推理系统 ---
def inference_thread_v2():
    """多线程优化的推理线程"""
    global stats_data, data_lock
    print("正在加载模型...")
    model = InferSession(DEVICE_ID, MODEL_PATH)
    print("模型加载完成。")
    
    # 启动预处理工作线程池
    preprocess_executor = ThreadPoolExecutor(max_workers=PREPROCESS_THREADS, thread_name_prefix="Preprocess")
    
    # 性能统计变量
    frame_count = 0
    total_times = {
        "preprocess": 0,
        "inference": 0, 
        "postprocess": 0,
        "pipeline": 0
    }
    
    active_preprocess_tasks = {}  # 跟踪活跃的预处理任务
    frame_id_counter = 0
    
    print(f"\n=== 开始多线程性能监控 (预处理线程数: {PREPROCESS_THREADS}) ===")
    print("每10帧输出一次详细性能分析...")

    while True:
        try:
            # 获取原始帧
            frame = frame_queue.get(timeout=0.1)
            pipeline_start_time = time.time()
            
            cam_height, cam_width = frame.shape[:2]
            frame_id = frame_id_counter
            frame_id_counter += 1
            
            # 提交预处理任务到线程池
            preprocess_start_time = time.time()
            future = preprocess_executor.submit(preprocess_single_optimized, frame)
            active_preprocess_tasks[frame_id] = (future, preprocess_start_time, pipeline_start_time, cam_width, cam_height)
            
            # 检查已完成的预处理任务
            completed_tasks = []
            for fid, (fut, prep_start, pipe_start, w, h) in active_preprocess_tasks.items():
                if fut.done():
                    try:
                        processed_data = fut.result()
                        preprocess_time_ms = (time.time() - prep_start) * 1000
                        
                        # 添加batch维度并执行推理
                        input_data = np.ascontiguousarray(processed_data[np.newaxis, :, :, :], dtype=np.float32)
                        
                        # 推理
                        infer_start_time = time.time()
                        outputs = model.infer([input_data])
                        inference_time_ms = (time.time() - infer_start_time) * 1000
                        
                        # 后处理
                        postprocess_start = time.time()
                        pred_mask = np.argmax(outputs[0], axis=1).squeeze()
                        vis_mask = (pred_mask * 255).astype(np.uint8)
                        lane_mask = cv2.resize(vis_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        postprocess_time_ms = (time.time() - postprocess_start) * 1000
                        
                        pipeline_latency_ms = (time.time() - pipe_start) * 1000
                        
                        # 统计
                        frame_count += 1
                        total_times["preprocess"] += preprocess_time_ms
                        total_times["inference"] += inference_time_ms
                        total_times["postprocess"] += postprocess_time_ms
                        total_times["pipeline"] += pipeline_latency_ms
                        
                        # 输出结果
                        try:
                            result_queue.put_nowait({
                                "frame": frame,  # 使用原始帧
                                "mask": lane_mask,
                                "latency": pipeline_latency_ms,
                                "inference_time": inference_time_ms,
                                "preprocess_time": preprocess_time_ms,
                                "postprocess_time": postprocess_time_ms
                            })
                        except queue.Full:
                            pass
                        
                        # 更新统计
                        with data_lock:
                            stats_data["pipeline_latency"] = f"{pipeline_latency_ms:.1f}"
                            stats_data["inference_time"] = f"{inference_time_ms:.1f}"
                            stats_data["preprocess_time"] = f"{preprocess_time_ms:.1f}"
                            stats_data["postprocess_time"] = f"{postprocess_time_ms:.1f}"
                            stats_data["preprocess_threads"] = str(len(active_preprocess_tasks))
                        
                        # 每10帧输出性能分析
                        if frame_count % 10 == 0:
                            avg_preprocess = total_times["preprocess"] / frame_count
                            avg_inference = total_times["inference"] / frame_count
                            avg_postprocess = total_times["postprocess"] / frame_count
                            avg_pipeline = total_times["pipeline"] / frame_count
                            
                            print(f"\n--- 第{frame_count}帧性能分析 (多线程优化版) ---")
                            print(f"输入帧尺寸: {w}x{h}")
                            print(f"模型输入尺寸: {MODEL_WIDTH}x{MODEL_HEIGHT}")
                            print(f"活跃预处理线程: {len(active_preprocess_tasks)}/{PREPROCESS_THREADS}")
                            print(f"")
                            print(f"【预处理】: {preprocess_time_ms:.1f}ms (多线程并行)")
                            print(f"【推理时间】: {inference_time_ms:.1f}ms")
                            print(f"【后处理】: {postprocess_time_ms:.1f}ms")
                            print(f"【流水线总延迟】: {pipeline_latency_ms:.1f}ms")
                            print(f"【理论最大FPS】: {1000/pipeline_latency_ms:.1f}")
                            print(f"")
                            print(f"【平均性能(最近{frame_count}帧)】")
                            print(f"  预处理: {avg_preprocess:.1f}ms")
                            print(f"  推理: {avg_inference:.1f}ms") 
                            print(f"  后处理: {avg_postprocess:.1f}ms")
                            print(f"  总延迟: {avg_pipeline:.1f}ms")
                            print(f"")
                            
                            # 多线程效果评估
                            theoretical_single_thread = avg_preprocess * PREPROCESS_THREADS
                            if theoretical_single_thread > avg_preprocess:
                                speedup = theoretical_single_thread / avg_preprocess
                                print(f"🚀 多线程优化效果:")
                                print(f"   理论加速比: {speedup:.1f}x")
                                print(f"   预处理并行效率: {(speedup/PREPROCESS_THREADS)*100:.1f}%")
                            
                            # CPU利用率分析
                            cpu_util = psutil.cpu_percent()
                            print(f"� CPU利用率分析:")
                            print(f"   当前CPU使用: {cpu_util:.1f}%")
                            if cpu_util < 80:
                                print(f"   建议: CPU还有余量，可考虑增加预处理线程")
                            else:
                                print(f"   建议: CPU使用接近饱和")
                            
                            print("=" * 60)
                        
                        completed_tasks.append(fid)
                        
                    except Exception as e:
                        print(f"处理帧 {fid} 时出错: {e}")
                        completed_tasks.append(fid)
            
            # 清理已完成的任务
            for fid in completed_tasks:
                del active_preprocess_tasks[fid]
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"推理线程错误: {e}")
            time.sleep(0.1)


# --- NPU/CPU 监控线程 (增强错误诊断) ---
def system_monitor_loop():
    global stats_data, data_lock
    npu_error_printed = False
    
    while True:
        # 更新 CPU 和内存
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        with data_lock:
            stats_data["cpu_percent"] = f"{cpu:.1f}"
            stats_data["mem_percent"] = f"{mem:.1f}"

        # 更新 NPU - 增强错误诊断
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
                
            # 如果之前有错误，现在恢复了，打印恢复信息
            if npu_error_printed:
                print("✅ NPU监控已恢复正常")
                npu_error_printed = False
                
        except subprocess.TimeoutExpired:
            if not npu_error_printed:
                print("⚠️  NPU监控超时 - npu-smi命令响应缓慢")
                print(f"   使用的NPU_SMI_PATH: {NPU_SMI_PATH}")
                print("   建议检查NPU驱动状态")
                npu_error_printed = True
            with data_lock:
                stats_data["npu_util"] = "Timeout"
                stats_data["npu_mem"] = "Timeout"
        except subprocess.CalledProcessError as e:
            if not npu_error_printed:
                print(f"❌ NPU命令执行失败:")
                print(f"   命令: {NPU_SMI_PATH} info")
                print(f"   返回码: {e.returncode}")
                print(f"   错误输出: {e.stderr}")
                print("   可能原因:")
                print("   1. NPU_SMI_PATH路径不正确")
                print("   2. 权限不足")
                print("   3. NPU驱动未正确安装")
                npu_error_printed = True
            with data_lock:
                stats_data["npu_util"] = "Error"
                stats_data["npu_mem"] = "Error"
        except FileNotFoundError:
            if not npu_error_printed:
                print(f"❌ 找不到npu-smi工具:")
                print(f"   配置的路径: {NPU_SMI_PATH}")
                print("   请执行 'which npu-smi' 或 'find /usr -name npu-smi' 找到正确路径")
                print("   然后更新代码中的 NPU_SMI_PATH 变量")
                npu_error_printed = True
            with data_lock:
                stats_data["npu_util"] = "NotFound"
                stats_data["npu_mem"] = "NotFound"
        except Exception as e:
            if not npu_error_printed:
                print(f"❌ 获取NPU信息时发生未知错误: {type(e).__name__}: {e}")
                npu_error_printed = True
            with data_lock:
                stats_data["npu_util"] = "Unknown"
                stats_data["npu_mem"] = "Unknown"
        
        time.sleep(1) # 每秒更新一次系统状态

# --- HTML 模板 (与之前相同) ---
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
        <h1>🚗 车道线检测实时推理监控 (性能优化版)</h1>
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
                        <span class="stat-label">活跃预处理线程</span>
                        <span id="preprocess_threads" class="stat-value">--</span>
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
                document.getElementById('preprocess_threads').textContent = data.preprocess_threads + ' / 2';
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
            
            # --- 可视化 ---
            green_overlay = np.zeros_like(frame, dtype=np.uint8)
            green_overlay[mask > 0] = [0, 255, 0]
            vis_frame = cv2.addWeighted(frame, 1.0, green_overlay, 0.5, 0)
            
            # 计算并更新显示FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                with data_lock:
                    stats_data["fps"] = f"{fps:.1f}"
                frame_count = 0
                start_time = time.time()
                
            # 绘制文字
            with data_lock:
                fps_text = f"Display FPS: {stats_data['fps']}"
                latency_text = f"Latency: {stats_data['pipeline_latency']} ms"
            cv2.putText(vis_frame, fps_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(vis_frame, latency_text, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
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

if __name__ == '__main__':
    # 启动时的系统信息
    print("🚀 车道线检测实时推理系统启动 (多线程超级优化版)")
    print("=" * 70)
    print(f"📷 摄像头配置:")
    print(f"   - 目标分辨率: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"   - 摄像头索引: {CAMERA_INDEX}")
    print(f"🧠 模型配置:")
    print(f"   - 模型路径: {MODEL_PATH}")
    print(f"   - 输入尺寸: {MODEL_WIDTH}x{MODEL_HEIGHT}")
    print(f"   - 设备ID: {DEVICE_ID}")
    print(f"⚡ 多线程优化特性:")
    print(f"   - NumPy多线程: 启用 (CPU核心: {multiprocessing.cpu_count()})")
    print(f"   - 预处理线程数: {PREPROCESS_THREADS}")
    print(f"   - 批处理大小: {BATCH_SIZE}")
    print(f"   - 预计算归一化常数")
    print(f"   - 异步预处理流水线")
    print(f"💻 系统信息:")
    print(f"   - CPU核心: {psutil.cpu_count()}")
    print(f"   - 物理内存: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"   - NPU工具路径: {NPU_SMI_PATH}")
    print("=" * 70)
    
    # 启动摄像头抓取线程
    cam_thread = Thread(target=camera_capture_thread, daemon=True)
    cam_thread.start()
    
    # 启动多线程推理线程
    inf_thread = Thread(target=inference_thread_v2, daemon=True)
    inf_thread.start()

    # 启动系统监控线程
    monitor_thread = Thread(target=system_monitor_loop, daemon=True)
    monitor_thread.start()
    
    print("Web服务器已启动。请在浏览器中访问: http://<Your_Atlas_IP>:8000")
    print("终端将显示详细的多线程性能分析信息...")
    app.run(host='0.0.0.0', port=8000, threaded=True)