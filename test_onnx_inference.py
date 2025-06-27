"""
TUSimple Lane Segmentation - ONNX Model Inference Test
使用导出的ONNX模型在测试集上进行推理并验证结果。
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import random
import time
import numpy as np
import onnxruntime as ort
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2

plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

class ONNXInferenceTester:
    def __init__(self, onnx_model_path):
        self.onnx_model_path = onnx_model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 检查CUDA并设置ONNX Runtime的执行提供者
        if self.device.type == 'cuda' and ort.get_device() == 'GPU':
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("ONNX Runtime 将使用 CUDAExecutionProvider。")
        else:
            self.providers = ['CPUExecutionProvider']
            print("ONNX Runtime 将使用 CPUExecutionProvider。")

        # 加载ONNX模型
        print(f"正在从 {self.onnx_model_path} 加载ONNX模型...")
        self.ort_session = ort.InferenceSession(self.onnx_model_path, providers=self.providers)
        self.input_name = self.ort_session.get_inputs()[0].name
        input_shape = self.ort_session.get_inputs()[0].shape
        print(f"模型加载完毕。输入名称: '{self.input_name}', 输入形状: {input_shape}")

        # 图像变换
        self.input_transform = transforms.Compose([
            transforms.Resize((640, 640), interpolation=transforms.InterpolationMode.BILINEAR), # 调整为模型输入尺寸
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        # 数据路径
        self.root = './manideep1108/tusimple/versions/5/TUSimple'
        self.test_clips_root = os.path.join(self.root, 'test_set', 'clips')
        self.seg_label_root = os.path.join(self.root, 'train_set', 'seg_label')

    def find_test_images(self, num_images=5):
        """从测试集中查找随机图像"""
        image_paths = []
        mask_paths = []
        
        if not os.path.exists(self.test_clips_root):
            print(f"错误: 测试集图像目录未找到 {self.test_clips_root}")
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
        
        print(f"在测试集中共找到 {len(image_paths)} 张图像。")
        
        # 随机选择 num_images 张
        if len(image_paths) >= num_images:
            selected_pairs = random.sample(list(zip(image_paths, mask_paths)), num_images)
            image_paths, mask_paths = zip(*selected_pairs)
            return list(image_paths), list(mask_paths)
        else:
            print(f"警告: 图像数量不足。找到 {len(image_paths)} 张, 需要 {num_images} 张。")
            return image_paths, mask_paths

    def load_image(self, image_path):
        """加载并预处理图像"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.input_transform(image).unsqueeze(0)
        return image_tensor, image

    def load_mask(self, mask_path):
        """加载并二值化mask"""
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask)
        mask_binary = (mask_array > 0).astype(np.uint8)
        return mask_binary

    def predict(self, image_tensor):
        """使用ONNX模型进行推理"""
        image_np = image_tensor.numpy()
        
        start_time = time.time()
        ort_inputs = {self.input_name: image_np}
        ort_outs = self.ort_session.run(None, ort_inputs)
        inference_time = time.time() - start_time
        
        pred_logits = ort_outs[0]
        pred_mask = np.argmax(pred_logits, axis=1).squeeze().astype(np.uint8)
        
        return pred_mask, inference_time

    def create_comparison_plot(self, original_img, gt_mask, pred_mask, save_path, inference_time):
        """创建对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'ONNX 模型推理\n推理时间: {inference_time*1000:.2f} ms', fontsize=16)
        
        axes[0].imshow(original_img)
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title('真实Mask')
        axes[1].axis('off')
        
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title('ONNX 预测Mask')
        axes[2].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"对比图已保存至 {save_path}")

    def run_test(self, num_images=5):
        """运行完整的测试流程"""
        print("\n🚀 开始 ONNX 推理测试...")
        
        image_paths, mask_paths = self.find_test_images(num_images)
        
        if not image_paths:
            print("❌ 未找到可供测试的图像，程序退出。")
            return
            
        output_dir = './onnx_inference_results'
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            print(f"\n--- 正在处理图像 {i+1}/{num_images}: {img_path} ---")
            
            # 加载数据
            image_tensor, original_img = self.load_image(img_path)
            gt_mask = self.load_mask(mask_path)
            
            # 预测
            pred_mask, inference_time = self.predict(image_tensor)
            
            # 如果需要，将预测结果尺寸调整为与真实mask一致
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            # 可视化
            save_path = os.path.join(output_dir, f'onnx_test_{i+1}.png')
            self.create_comparison_plot(original_img, gt_mask, pred_mask, save_path, inference_time)

        print(f"\n✅ {len(image_paths)} 张图像的 ONNX 推理测试完成。")
        print(f"结果保存在: {output_dir}")

def main():
    # 用户要求测试未简化的模型
    onnx_model_path = './weights/fast_scnn_tusimple_fixed.onnx'
    
    if not os.path.exists(onnx_model_path):
        print(f"❌ 错误: ONNX 模型文件未找到 '{onnx_model_path}'")
        print("请先运行 `export_onnx.py` 来生成模型。")
        return

    tester = ONNXInferenceTester(onnx_model_path)
    tester.run_test(num_images=5)

if __name__ == '__main__':
    main()
