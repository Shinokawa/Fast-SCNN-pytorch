"""
Atlas NPU端到端推理示例
展示如何使用集成预处理的ONNX模型进行高效车道线检测
"""
import cv2
import numpy as np
import time
import argparse
from pathlib import Path

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ onnxruntime not available. This is a demo script.")

class AtlasNPULaneDetector:
    """Atlas NPU车道线检测器（端到端版本）"""
    
    def __init__(self, model_path, providers=None):
        """
        初始化检测器
        Args:
            model_path: ONNX模型路径
            providers: 推理提供者列表
        """
        if not ONNX_AVAILABLE:
            print("⚠️ This is a demo class. Install onnxruntime for actual usage.")
            return
            
        if providers is None:
            # Atlas NPU: 优先使用CUDA (NPU mapped to CUDA in some setups)
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 获取模型输入输出尺寸
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.num_classes = output_shape[1]
        
        print(f"✅ Atlas NPU模型加载成功!")
        print(f"   📊 输入尺寸: {self.input_height}x{self.input_width}")
        print(f"   📊 输出类别: {self.num_classes}")
        print(f"   🖥️ 推理提供者: {self.session.get_providers()}")
    
    def preprocess_frame(self, frame):
        """
        预处理摄像头帧 (简化版本，大部分预处理已在模型内部)
        Args:
            frame: OpenCV图像 (H, W, 3) BGR格式
        Returns:
            input_tensor: 模型输入张量 (1, 3, H, W)
        """
        # 转换BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 调整尺寸 (如果需要)
        if frame_rgb.shape[:2] != (self.input_height, self.input_width):
            frame_rgb = cv2.resize(frame_rgb, (self.input_width, self.input_height))
        
        # 转换为张量格式: (H, W, 3) → (1, 3, H, W)
        input_tensor = frame_rgb.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        
        return input_tensor
    
    def postprocess_output(self, output):
        """
        后处理模型输出
        Args:
            output: 模型输出 (1, num_classes, H, W)
        Returns:
            lane_mask: 车道线掩码 (H, W) [0-255]
            confidence: 检测置信度
        """
        # 获取最大概率类别
        pred_mask = np.argmax(output[0], axis=0)  # (H, W)
        max_probs = np.max(output[0], axis=0)     # (H, W)
        
        # 转换为可视化格式
        lane_mask = np.zeros_like(pred_mask, dtype=np.uint8)
        lane_mask[pred_mask > 0] = 255  # 非背景区域设为255
        
        # 计算置信度
        confidence = np.mean(max_probs)
        
        return lane_mask, confidence
    
    def detect_lanes(self, frame):
        """
        单帧车道线检测
        Args:
            frame: OpenCV图像
        Returns:
            lane_mask: 车道线掩码
            confidence: 检测置信度
            inference_time: 推理时间(ms)
        """
        if not ONNX_AVAILABLE:
            # 返回示例结果
            h, w = frame.shape[:2]
            lane_mask = np.zeros((h, w), dtype=np.uint8)
            return lane_mask, 0.5, 10.0
        
        # 预处理
        input_tensor = self.preprocess_frame(frame)
        
        # NPU推理
        start_time = time.time()
        output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        inference_time = (time.time() - start_time) * 1000
        
        # 后处理
        lane_mask, confidence = self.postprocess_output(output)
        
        return lane_mask, confidence, inference_time
    
    def detect_batch(self, frames):
        """
        批量检测（如果NPU支持）
        Args:
            frames: 帧列表
        Returns:
            results: 检测结果列表
        """
        # 批量预处理
        batch_input = np.stack([self.preprocess_frame(frame)[0] for frame in frames])
        
        # 批量推理
        start_time = time.time()
        batch_output = self.session.run([self.output_name], {self.input_name: batch_input})[0]
        total_time = (time.time() - start_time) * 1000
        
        # 批量后处理
        results = []
        for i in range(len(frames)):
            lane_mask, confidence = self.postprocess_output(batch_output[i:i+1])
            results.append((lane_mask, confidence))
        
        avg_time = total_time / len(frames)
        return results, avg_time

def visualize_lanes(frame, lane_mask, confidence, inference_time):
    """
    可视化车道线检测结果
    Args:
        frame: 原始图像
        lane_mask: 车道线掩码
        confidence: 检测置信度
        inference_time: 推理时间
    Returns:
        vis_frame: 可视化结果
    """
    vis_frame = frame.copy()
    
    # 调整掩码尺寸到原图尺寸
    if lane_mask.shape != frame.shape[:2]:
        lane_mask = cv2.resize(lane_mask, (frame.shape[1], frame.shape[0]))
    
    # 叠加车道线（绿色）
    lane_overlay = np.zeros_like(frame)
    lane_overlay[lane_mask > 0] = [0, 255, 0]  # 绿色车道线
    vis_frame = cv2.addWeighted(vis_frame, 0.7, lane_overlay, 0.3, 0)
    
    # 添加信息文本
    cv2.putText(vis_frame, f"Confidence: {confidence:.3f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_frame, f"Time: {inference_time:.1f}ms", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_frame, f"FPS: {1000/inference_time:.1f}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return vis_frame

def test_image_inference(detector, image_path, output_path=None):
    """测试单张图像推理"""
    print(f"\n🖼️ Testing image inference: {image_path}")
    
    # 读取图像
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"   📊 Image size: {frame.shape}")
    
    # 检测车道线
    lane_mask, confidence, inference_time = detector.detect_lanes(frame)
    
    print(f"✅ Detection completed!")
    print(f"   ⚡ Inference time: {inference_time:.2f}ms")
    print(f"   🎯 Confidence: {confidence:.3f}")
    print(f"   🚀 Theoretical FPS: {1000/inference_time:.1f}")
    
    # 可视化结果
    vis_frame = visualize_lanes(frame, lane_mask, confidence, inference_time)
    
    # 保存结果
    if output_path:
        cv2.imwrite(str(output_path), vis_frame)
        print(f"   💾 Result saved to: {output_path}")
    
    return vis_frame

def test_video_inference(detector, video_path, output_path=None, max_frames=None):
    """测试视频推理"""
    print(f"\n🎥 Testing video inference: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   📊 Video info: {width}x{height}, {fps:.1f}FPS, {total_frames} frames")
    
    # 设置输出视频
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # 处理视频帧
    frame_count = 0
    total_time = 0
    confidences = []
    
    print("🔄 Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测车道线
        lane_mask, confidence, inference_time = detector.detect_lanes(frame)
        
        # 统计
        total_time += inference_time
        confidences.append(confidence)
        frame_count += 1
        
        # 可视化
        vis_frame = visualize_lanes(frame, lane_mask, confidence, inference_time)
        
        # 保存帧
        if out:
            out.write(vis_frame)
        
        # 进度显示
        if frame_count % 30 == 0:
            avg_time = total_time / frame_count
            print(f"   📊 Processed {frame_count}/{total_frames} frames, "
                  f"Avg time: {avg_time:.1f}ms, Avg FPS: {1000/avg_time:.1f}")
        
        # 限制处理帧数
        if max_frames and frame_count >= max_frames:
            break
    
    # 清理资源
    cap.release()
    if out:
        out.release()
    
    # 统计结果
    avg_time = total_time / frame_count
    avg_confidence = np.mean(confidences)
    
    print(f"✅ Video processing completed!")
    print(f"   📊 Processed frames: {frame_count}")
    print(f"   ⚡ Average inference time: {avg_time:.2f}ms")
    print(f"   🚀 Average FPS: {1000/avg_time:.1f}")
    print(f"   🎯 Average confidence: {avg_confidence:.3f}")
    
    if output_path:
        print(f"   💾 Result video saved to: {output_path}")

def test_camera_inference(detector, camera_id=0, display=True):
    """测试摄像头实时推理"""
    print(f"\n📷 Testing camera inference (Camera ID: {camera_id})")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Failed to open camera: {camera_id}")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("🔄 Starting real-time inference... Press 'q' to quit")
    
    frame_count = 0
    total_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 检测车道线
        lane_mask, confidence, inference_time = detector.detect_lanes(frame)
        
        # 统计
        total_time += inference_time
        frame_count += 1
        
        # 可视化
        vis_frame = visualize_lanes(frame, lane_mask, confidence, inference_time)
        
        # 显示
        if display:
            cv2.imshow('Atlas NPU Lane Detection', vis_frame)
            
            # 按键控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # 每100帧输出一次统计
        if frame_count % 100 == 0:
            avg_time = total_time / frame_count
            print(f"   📊 Frame {frame_count}: Avg time {avg_time:.1f}ms, Avg FPS {1000/avg_time:.1f}")
    
    # 清理
    cap.release()
    if display:
        cv2.destroyAllWindows()
    
    avg_time = total_time / frame_count
    print(f"✅ Camera inference completed!")
    print(f"   📊 Total frames: {frame_count}")
    print(f"   ⚡ Average inference time: {avg_time:.2f}ms")
    print(f"   🚀 Average FPS: {1000/avg_time:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Atlas NPU端到端车道线检测示例')
    parser.add_argument('--model', type=str, required=True,
                        help='端到端ONNX模型路径')
    parser.add_argument('--mode', type=str, choices=['image', 'video', 'camera'], 
                        default='image', help='测试模式')
    parser.add_argument('--input', type=str, help='输入文件路径（图像或视频）')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--camera-id', type=int, default=0, help='摄像头ID')
    parser.add_argument('--max-frames', type=int, help='最大处理帧数（视频模式）')
    parser.add_argument('--no-display', action='store_true', help='不显示结果')
    
    args = parser.parse_args()
    
    print("🚀 Atlas NPU端到端车道线检测测试")
    print("=" * 50)
    
    # 检查模型文件
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    # 初始化检测器
    print(f"📦 Loading model: {model_path.name}")
    detector = AtlasNPULaneDetector(str(model_path))
    
    # 根据模式运行测试
    if args.mode == 'image':
        if not args.input:
            print("❌ Image mode requires --input parameter")
            return
        test_image_inference(detector, args.input, args.output)
        
    elif args.mode == 'video':
        if not args.input:
            print("❌ Video mode requires --input parameter")
            return
        test_video_inference(detector, args.input, args.output, args.max_frames)
        
    elif args.mode == 'camera':
        test_camera_inference(detector, args.camera_id, not args.no_display)
    
    print("\n🎉 Test completed!")

if __name__ == '__main__':
    main()
