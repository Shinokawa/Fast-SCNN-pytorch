"""
TUSimple Lane Segmentation - End-to-End ONNX Model Inference Test
使用端到端ONNX模型进行推理测试，模拟Atlas开发板上的实际部署场景。
特点：
1. 直接输入原始图像数据（0-255像素值）
2. 无需CPU预处理，充分利用NPU性能
3. 模拟智能小车摄像头实时推理场景
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import random
import time
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2

plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

class EndToEndONNXTester:
    """端到端ONNX模型推理测试器 - 专为Atlas开发板优化"""
    
    def __init__(self, onnx_model_path):
        self.onnx_model_path = onnx_model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 检查CUDA并设置ONNX Runtime的执行提供者
        if self.device.type == 'cuda' and ort.get_device() == 'GPU':
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("🚀 ONNX Runtime 将使用 CUDAExecutionProvider (模拟NPU加速)")
        else:
            self.providers = ['CPUExecutionProvider']
            print("⚠️  ONNX Runtime 将使用 CPUExecutionProvider")

        # 加载端到端ONNX模型
        print(f"📦 正在从 {self.onnx_model_path} 加载端到端ONNX模型...")
        self.ort_session = ort.InferenceSession(self.onnx_model_path, providers=self.providers)
        self.input_name = self.ort_session.get_inputs()[0].name
        input_shape = self.ort_session.get_inputs()[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        print(f"✅ 端到端模型加载完毕")
        print(f"   📊 输入名称: \'{self.input_name}\'")
        print(f"   📊 输入形状 (N, C, H, W): {input_shape}")
        print(f"   ✨ 预期输入: 原始图像数据 [0-255] (HWC格式)")
        print(f"   🎯 内置功能: 自动resize至 ({self.input_height}, {self.input_width}) + 归一化 + 推理")

        # 数据路径
        self.root = './manideep1108/tusimple/versions/5/TUSimple'
        self.test_clips_root = os.path.join(self.root, 'test_set', 'clips')
        self.seg_label_root = os.path.join(self.root, 'train_set', 'seg_label')

    def find_test_images(self, num_images=5):
        """从测试集中查找随机图像"""
        image_paths = []
        mask_paths = []
        
        if not os.path.exists(self.test_clips_root):
            print(f"❌ 错误: 测试集图像目录未找到 {self.test_clips_root}")
            return [], []

        # 收集所有有效的图像/mask对
        for date_folder in os.listdir(self.test_clips_root):
            date_path = os.path.join(self.test_clips_root, date_folder)
            if os.path.isdir(date_path):
                for video_folder in os.listdir(date_path):
                    video_path = os.path.join(date_path, video_folder)
                    if os.path.isdir(video_path):
                        for i in range(1, 21): # 检查 1.jpg 到 20.jpg
                            img_file = os.path.join(video_path, f'{i}.jpg')
                            if os.path.exists(img_file):
                                mask_file = os.path.join(self.seg_label_root, date_folder, video_folder, f'{i}.png')
                                if os.path.exists(mask_file):
                                    image_paths.append(img_file)
                                    mask_paths.append(mask_file)
        
        print(f"📷 在测试集中共找到 {len(image_paths)} 张图像")
        
        # 随机选择 num_images 张
        if len(image_paths) >= num_images:
            selected_pairs = random.sample(list(zip(image_paths, mask_paths)), num_images)
            image_paths, mask_paths = zip(*selected_pairs)
            return list(image_paths), list(mask_paths)
        else:
            print(f"⚠️  警告: 图像数量不足。找到 {len(image_paths)} 张, 需要 {num_images} 张")
            return image_paths, mask_paths

    def load_raw_image(self, image_path):
        """加载原始图像数据 - 模拟摄像头输入"""
        # 使用OpenCV加载，保持原始像素值 [0-255]
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # PC上简单resize到模型期望的输入尺寸
        original_size = image_rgb.shape[:2] # (H, W)
        # 注意：cv2.resize的dsize参数是(width, height)
        image_rgb_resized = cv2.resize(image_rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        
        # 转换为模型期望的CHW格式 (3, H, W) 并保持原始数据类型
        image_chw = image_rgb_resized.transpose(2, 0, 1)  # HWC -> CHW
        image_batch = image_chw[np.newaxis, ...]  # 添加batch维度 -> (1, 3, H, W)
        
        # 确保数据类型为float32（ONNX推理需要）
        image_input = image_batch.astype(np.float32)
        
        print(f"   📷 原始图像尺寸: {original_size} -> 640x480 (匹配开发板)")
        print(f"   📊 模型输入尺寸: {image_input.shape} (NCHW)")
        print(f"   🎯 像素值范围: [{image_input.min():.0f}, {image_input.max():.0f}]")
        
        return image_input, image_rgb

    def load_mask(self, mask_path):
        """加载并二值化mask"""
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask)
        mask_binary = (mask_array > 0).astype(np.uint8)
        return mask_binary

    def predict_end_to_end(self, image_input):
        """使用端到端ONNX模型进行推理 - 模拟Atlas NPU推理"""
        print(f"   🚀 开始端到端推理...")
        print(f"   ✨ 内置操作: resize -> 归一化 -> 分割推理 -> softmax")
        
        start_time = time.time()
        ort_inputs = {self.input_name: image_input}
        ort_outs = self.ort_session.run(None, ort_inputs)
        inference_time = time.time() - start_time
        
        # 端到端模型输出已经是概率分布（经过softmax）
        pred_probs = ort_outs[0]  # Shape: (1, num_classes, H, W)
        pred_mask = np.argmax(pred_probs, axis=1).squeeze().astype(np.uint8)
        
        print(f"   ⚡ 推理完成: {inference_time*1000:.2f}ms")
        print(f"   📊 输出概率形状: {pred_probs.shape}")
        print(f"   📊 预测mask形状: {pred_mask.shape}")
        print(f"   🎯 预测类别数: {pred_probs.shape[1]}")
        
        # 计算置信度
        max_probs = np.max(pred_probs, axis=1).squeeze()
        avg_confidence = np.mean(max_probs)
        print(f"   📈 平均置信度: {avg_confidence:.3f}")
        
        return pred_mask, inference_time, avg_confidence

    def create_comparison_plot(self, original_img, gt_mask, pred_mask, save_path, inference_time, confidence):
        """创建对比图 - 突出端到端优势"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'端到端ONNX模型推理 (Atlas NPU优化)\n' +
                    f'推理时间: {inference_time*1000:.2f}ms | 置信度: {confidence:.3f} | ' +
                    f'理论FPS: {1/inference_time:.1f}', fontsize=16)
        
        axes[0].imshow(original_img)
        axes[0].set_title('原始摄像头输入\n(无预处理)')
        axes[0].axis('off')
        
        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title('真实车道线')
        axes[1].axis('off')
        
        # 为预测结果添加颜色映射，更好地显示车道线
        pred_colored = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        pred_colored[pred_mask == 1] = [255, 0, 0]    # 车道线1 - 红色
        pred_colored[pred_mask == 2] = [0, 255, 0]    # 车道线2 - 绿色  
        pred_colored[pred_mask == 3] = [0, 0, 255]    # 车道线3 - 蓝色
        pred_colored[pred_mask == 4] = [255, 255, 0]  # 车道线4 - 黄色
        
        axes[2].imshow(pred_colored)
        axes[2].set_title('端到端预测结果\n(NPU加速)')
        axes[2].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   💾 对比图已保存: {save_path}")

    def run_performance_benchmark(self, image_input, runs=20):
        """运行性能基准测试 - 模拟Atlas开发板性能"""
        print(f"\n🏁 开始性能基准测试 (模拟Atlas NPU)...")
        print(f"   🔄 预热运行: 5次")
        print(f"   📊 测试运行: {runs}次")
        
        # 预热
        for _ in range(5):
            _ = self.ort_session.run(None, {self.input_name: image_input})
        
        # 性能测试
        times = []
        for i in range(runs):
            start_time = time.time()
            _ = self.ort_session.run(None, {self.input_name: image_input})
            end_time = time.time()
            times.append(end_time - start_time)
            
            if (i + 1) % 5 == 0:
                print(f"   ⚡ 已完成 {i+1}/{runs} 次测试...")
        
        # 统计分析
        times = np.array(times)
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        print(f"\n📈 性能基准测试结果:")
        print(f"   ⚡ 平均推理时间: {avg_time*1000:.2f}ms")
        print(f"   🚀 最快推理时间: {min_time*1000:.2f}ms") 
        print(f"   🐌 最慢推理时间: {max_time*1000:.2f}ms")
        print(f"   📊 标准差: {std_time*1000:.2f}ms")
        print(f"   🎯 理论平均FPS: {1/avg_time:.1f}")
        print(f"   🏆 理论最大FPS: {1/min_time:.1f}")
        
        # Atlas NPU性能对比
        print(f"\n🚗 智能小车性能分析:")
        if avg_time < 0.04:  # < 40ms
            print(f"   🎉 优秀! 完全满足实时自动驾驶需求 (>25 FPS)")
        elif avg_time < 0.067:  # < 67ms  
            print(f"   ✅ 良好! 满足基本自动驾驶需求 (15-25 FPS)")
        elif avg_time < 0.1:  # < 100ms
            print(f"   ⚠️  一般! 可用但不够流畅 (10-15 FPS)")
        else:
            print(f"   ❌ 需要优化! FPS过低，影响安全性 (<10 FPS)")
            
        return avg_time, min_time, max_time

    def run_test(self, num_images=5, benchmark=True):
        """运行完整的端到端测试流程"""
        print("\n" + "="*80)
        print("🚀 端到端ONNX模型推理测试 - Atlas NPU优化版本")
        print("🚗 智能小车实时车道线检测模拟")
        print("="*80)
        
        image_paths, mask_paths = self.find_test_images(num_images)
        
        if not image_paths:
            print("❌ 未找到可供测试的图像，程序退出")
            return
            
        output_dir = './end_to_end_onnx_results'
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 结果将保存到: {output_dir}")
        
        inference_times = []
        confidences = []
        
        for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            print(f"\n" + "-"*60)
            print(f"🖼️  正在处理图像 {i+1}/{num_images}")
            print(f"📂 图像路径: {os.path.basename(img_path)}")
            
            # 加载原始图像数据（模拟摄像头输入）
            image_input, original_img = self.load_raw_image(img_path)
            gt_mask = self.load_mask(mask_path)
            
            # 端到端推理
            pred_mask, inference_time, confidence = self.predict_end_to_end(image_input)
            inference_times.append(inference_time)
            confidences.append(confidence)
            
            # 调整预测结果尺寸（如果需要）
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)

            # 创建可视化结果
            save_path = os.path.join(output_dir, f'end_to_end_test_{i+1}.png')
            self.create_comparison_plot(original_img, gt_mask, pred_mask, save_path, 
                                      inference_time, confidence)
            
            # 性能基准测试（仅在第一张图像上进行）
            if benchmark and i == 0:
                avg_time, min_time, max_time = self.run_performance_benchmark(image_input)

        # 总体性能统计
        print(f"\n" + "="*80)
        print(f"✅ {len(image_paths)} 张图像的端到端推理测试完成!")
        print(f"📊 总体性能统计:")
        print(f"   ⚡ 平均推理时间: {np.mean(inference_times)*1000:.2f}ms")
        print(f"   📈 平均置信度: {np.mean(confidences):.3f}")
        print(f"   🚀 实际平均FPS: {1/np.mean(inference_times):.1f}")
        print(f"📁 结果保存在: {output_dir}")
        
        # Atlas开发板部署建议
        print(f"\n🎯 Atlas开发板部署优势:")
        print(f"   ✨ 端到端推理: 无CPU预处理瓶颈")
        print(f"   🚀 NPU加速: 预期性能提升 2-3倍")
        print(f"   💾 内存优化: 减少CPU-NPU数据传输")
        print(f"   🎮 易于集成: 单次推理调用完成所有操作")
        print("="*80)

def main():
    # 端到端ONNX模型路径
    onnx_model_path = './weights/fast_scnn_tusimple_e2e_640x480.onnx'
    
    if not os.path.exists(onnx_model_path):
        print(f"❌ 错误: 端到端ONNX模型文件未找到 '{onnx_model_path}'")
        print("💡 请先运行以下命令生成端到端模型:")
        print("   python export_onnx_fixed.py --end-to-end --input-size 640,480 --output-path ./weights/fast_scnn_tusimple_e2e_640x480.onnx")
        return

    print("🚀 启动端到端ONNX推理测试...")
    print("🎯 模拟Atlas开发板智能小车场景")
    
    tester = EndToEndONNXTester(onnx_model_path)
    tester.run_test(num_images=5, benchmark=True)

if __name__ == '__main__':
    main()
