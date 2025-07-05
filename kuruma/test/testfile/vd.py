import os
import cv2
import time
import numpy as np
from ais_bench.infer.interface import InferSession
from flask import Flask, Response
from threading import Thread

# --- Flask App 和全局变量 ---
app = Flask(__name__)
output_frame = None # 用于在线程间传递最终图像帧

# --- 可配置常量 (与之前相同) ---
DEVICE_ID = 0
MODEL_PATH = "./weights/fast_scnn_tusimple_bs1.om"
MODEL_WIDTH = 1024
MODEL_HEIGHT = 768
CAMERA_INDEX = 0

# --- 预处理和后处理函数 (与之前相同) ---
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
    global output_frame
    model = InferSession(DEVICE_ID, MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Cannot open camera at index {CAMERA_INDEX}.")
        return

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        cam_height, cam_width = frame.shape[:2]
        input_data = preprocess(frame)
        outputs = model.infer([input_data])
        lane_mask = postprocess(outputs[0], cam_width, cam_height)
        
        lane_mask_color = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
        green_overlay = np.zeros_like(lane_mask_color)
        green_overlay[lane_mask > 0] = [0, 255, 0]
        result_frame = cv2.addWeighted(frame, 1.0, green_overlay, 0.5, 0)
        
        # 将处理好的帧存入全局变量
        output_frame = result_frame.copy()

# --- Web服务器部分 ---
def generate():
    global output_frame
    while True:
        if output_frame is None:
            continue
        # 将帧编码为JPEG格式
        (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
        if not flag:
            continue
        # 以字节流形式产生输出
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

@app.route("/")
def video_feed():
    # 返回视频流的响应
    return Response(generate(),
                    mimetype = "multipart/x-mixed-replace; boundary=frame")

# --- 主程序 ---
if __name__ == '__main__':
    # 在一个后台线程中启动推理循环
    t = Thread(target=inference_loop)
    t.daemon = True
    t.start()
    
    # 启动Flask Web服务器
    # host='0.0.0.0' 表示允许任何IP地址访问
    print("Web stream started. Open your browser and navigate to http://<Your_Atlas_IP>:8000")
    app.run(host='0.0.0.0', port=8000, threaded=True)