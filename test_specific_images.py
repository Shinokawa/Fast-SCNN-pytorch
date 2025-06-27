"""
TUSimple Lane Segmentation - Test Specific Images (20.jpg)
预测每个文件夹中的20.jpg并与真实mask对比分析 - 随机选择50张
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torchvision import transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from models.fast_scnn import get_fast_scnn
from utils.metric import SegmentationMetric


class TUSimpleSpecificTester:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Image transforms (原始尺寸，不resize)
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        
        # Load model
        self.model = get_fast_scnn(dataset='tusimple', aux=False)
        model_path = './weights/fast_scnn_tusimple_best_model.pth'
        if os.path.isfile(model_path):
            print(f'Loading model from {model_path}...')
            state_dict = torch.load(model_path, map_location='cpu')
            # Handle DataParallel models
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Metrics
        self.metric = SegmentationMetric(2)  # 2 classes
        
        # 数据路径
        self.root = './manideep1108/tusimple/versions/5/TUSimple'
        self.test_clips_root = os.path.join(self.root, 'test_set', 'clips')
        self.seg_label_root = os.path.join(self.root, 'train_set', 'seg_label')
        
        print(f"Device: {self.device}")
        print(f"Model loaded successfully")
    
    def find_20jpg_images(self):
        """查找所有包含20.jpg的文件夹"""
        image_paths = []
        mask_paths = []
        
        # 遍历test_set/clips下的所有日期文件夹
        if os.path.exists(self.test_clips_root):
            for date_folder in os.listdir(self.test_clips_root):
                date_path = os.path.join(self.test_clips_root, date_folder)
                if os.path.isdir(date_path):
                    # 遍历每个日期下的视频文件夹
                    for video_folder in os.listdir(date_path):
                        video_path = os.path.join(date_path, video_folder)
                        if os.path.isdir(video_path):
                            # 检查是否存在20.jpg
                            img_file = os.path.join(video_path, '20.jpg')
                            if os.path.exists(img_file):
                                # 构建对应的mask路径
                                mask_file = os.path.join(self.seg_label_root, date_folder, 
                                                       video_folder, '20.png')
                                if os.path.exists(mask_file):
                                    image_paths.append(img_file)
                                    mask_paths.append(mask_file)
                                    print(f"Found pair: {video_folder}/20.jpg")
        
        print(f"Total found {len(image_paths)} valid 20.jpg images with masks")
        
        # 随机选择50张图像
        if len(image_paths) > 50:
            paired_data = list(zip(image_paths, mask_paths))
            random.shuffle(paired_data)
            selected_pairs = paired_data[:50]
            image_paths, mask_paths = zip(*selected_pairs)
            image_paths, mask_paths = list(image_paths), list(mask_paths)
            print(f"Randomly selected 50 images from {len(paired_data)} available")
        
        return image_paths, mask_paths
    
    def load_image(self, image_path):
        """加载并预处理图像"""
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # 应用transforms
        image_tensor = self.input_transform(image).unsqueeze(0)
        return image_tensor, image, original_size
    
    def load_mask(self, mask_path):
        """加载并处理mask"""
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask)
        
        # 二值化：非零值设为1（车道线），零值设为0（背景）
        mask_binary = (mask_array > 0).astype(np.uint8)
        return mask_binary
    
    def predict_image(self, image_tensor):
        """对单张图像进行预测"""
        with torch.no_grad():
            image_input = image_tensor.to(self.device)
            
            start_time = time.time()
            with autocast(device_type=self.device.type):
                outputs = self.model(image_input)
            inference_time = time.time() - start_time
            
            # 获取预测结果
            if isinstance(outputs, tuple):
                pred = outputs[0]
            else:
                pred = outputs
            
            pred = torch.argmax(pred, dim=1)
            pred_np = pred.squeeze().cpu().numpy()
            
            return pred_np, inference_time
    
    def calculate_metrics(self, pred, gt):
        """计算单张图像的指标"""
        # 确保pred和gt都是numpy数组
        pred = np.array(pred)
        gt = np.array(gt)
        
        # 计算像素准确率
        correct = (pred == gt).sum()
        total = pred.size
        pixel_acc = correct / total
        
        # 计算IoU for each class
        ious = []
        for class_id in range(2):  # 0: background, 1: lane
            pred_mask = (pred == class_id)
            gt_mask = (gt == class_id)
            
            intersection = (pred_mask & gt_mask).sum()
            union = (pred_mask | gt_mask).sum()
            
            if union == 0:
                iou = 1.0 if intersection == 0 else 0.0
            else:
                iou = intersection / union
            ious.append(iou)
        
        miou = np.mean(ious)
        
        return pixel_acc, miou, ious
    
    def create_comparison_plot(self, original_img, gt_mask, pred_mask, metrics, save_path):
        """创建对比可视化图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Lane Segmentation Results\nPixAcc: {metrics["pixel_acc"]*100:.2f}% | mIoU: {metrics["miou"]*100:.2f}%', 
                     fontsize=14, fontweight='bold')
        
        # 原始图像
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 真实mask
        axes[0, 1].imshow(gt_mask, cmap='gray')
        axes[0, 1].set_title('Ground Truth Mask')
        axes[0, 1].axis('off')
        
        # 预测mask
        axes[0, 2].imshow(pred_mask, cmap='gray')
        axes[0, 2].set_title('Predicted Mask')
        axes[0, 2].axis('off')
        
        # 原图+真实mask overlay
        gt_overlay = np.array(original_img).copy()
        gt_colored = np.zeros_like(gt_overlay)
        gt_colored[:, :, 1] = gt_mask * 255  # 绿色表示真实车道线
        gt_result = cv2.addWeighted(gt_overlay, 0.7, gt_colored, 0.3, 0)
        axes[1, 0].imshow(gt_result)
        axes[1, 0].set_title('GT Overlay (Green)')
        axes[1, 0].axis('off')
        
        # 原图+预测mask overlay
        pred_overlay = np.array(original_img).copy()
        pred_colored = np.zeros_like(pred_overlay)
        pred_colored[:, :, 0] = pred_mask * 255  # 红色表示预测车道线
        pred_result = cv2.addWeighted(pred_overlay, 0.7, pred_colored, 0.3, 0)
        axes[1, 1].imshow(pred_result)
        axes[1, 1].set_title('Prediction Overlay (Red)')
        axes[1, 1].axis('off')
        
        # 误差分析
        error_map = np.zeros_like(gt_mask)
        # True Positive: 绿色 (预测对的车道线)
        # False Positive: 红色 (错误预测的车道线) 
        # False Negative: 蓝色 (漏检的车道线)
        
        tp = (pred_mask == 1) & (gt_mask == 1)  # 正确预测的车道线
        fp = (pred_mask == 1) & (gt_mask == 0)  # 错误预测的车道线
        fn = (pred_mask == 0) & (gt_mask == 1)  # 漏检的车道线
        
        error_rgb = np.zeros((*error_map.shape, 3), dtype=np.uint8)
        error_rgb[tp] = [0, 255, 0]   # TP: 绿色
        error_rgb[fp] = [255, 0, 0]   # FP: 红色  
        error_rgb[fn] = [0, 0, 255]   # FN: 蓝色
        
        axes[1, 2].imshow(error_rgb)
        axes[1, 2].set_title('Error Analysis\nGreen:TP, Red:FP, Blue:FN')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def test_specific_images(self):
        """测试所有20.jpg图像"""
        print("\n🔍 Starting specific image testing...")
        
        # 查找所有20.jpg图像
        image_paths, mask_paths = self.find_20jpg_images()
        
        if len(image_paths) == 0:
            print("❌ No valid 20.jpg images found!")
            return
        
        # 创建输出目录
        output_dir = './test_20jpg_random50_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # 统计信息
        all_metrics = []
        total_inference_time = 0
        
        print(f"\n📊 Processing {len(image_paths)} images...")
        
        for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            print(f"\nProcessing [{i+1}/{len(image_paths)}]: {os.path.basename(os.path.dirname(img_path))}/20.jpg")
            
            try:
                # 加载图像和mask
                image_tensor, original_img, original_size = self.load_image(img_path)
                gt_mask = self.load_mask(mask_path)
                
                # 预测
                pred_mask, inference_time = self.predict_image(image_tensor)
                total_inference_time += inference_time
                
                # 调整预测结果尺寸到原始尺寸
                if pred_mask.shape != gt_mask.shape:
                    pred_mask = cv2.resize(pred_mask.astype(np.uint8), 
                                         (gt_mask.shape[1], gt_mask.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
                
                # 计算指标
                pixel_acc, miou, ious = self.calculate_metrics(pred_mask, gt_mask)
                
                metrics = {
                    'pixel_acc': pixel_acc,
                    'miou': miou,
                    'iou_bg': ious[0],
                    'iou_lane': ious[1],
                    'inference_time': inference_time
                }
                all_metrics.append(metrics)
                
                # 创建可视化
                folder_name = os.path.basename(os.path.dirname(img_path))
                save_path = os.path.join(output_dir, f'{folder_name}_20jpg_result.png')
                self.create_comparison_plot(original_img, gt_mask, pred_mask, metrics, save_path)
                
                print(f"   PixAcc: {pixel_acc*100:.2f}% | mIoU: {miou*100:.2f}% | Lane IoU: {ious[1]*100:.2f}% | Time: {inference_time*1000:.1f}ms")
                
            except Exception as e:
                print(f"   ❌ Error processing {img_path}: {str(e)}")
                continue
        
        # 生成统计报告
        self.generate_report(all_metrics, output_dir)
        
        print(f"\n🎉 Testing completed! Results saved to: {output_dir}")
    
    def generate_report(self, all_metrics, output_dir):
        """生成详细的测试报告"""
        if not all_metrics:
            print("❌ No valid results to generate report")
            return
        
        # 计算统计数据
        pixel_accs = [m['pixel_acc'] for m in all_metrics]
        mious = [m['miou'] for m in all_metrics]
        lane_ious = [m['iou_lane'] for m in all_metrics]
        bg_ious = [m['iou_bg'] for m in all_metrics]
        inference_times = [m['inference_time'] for m in all_metrics]
        
        # 统计结果
        stats = {
            'num_samples': len(all_metrics),
            'avg_pixel_acc': np.mean(pixel_accs),
            'std_pixel_acc': np.std(pixel_accs),
            'avg_miou': np.mean(mious),
            'std_miou': np.std(mious),
            'avg_lane_iou': np.mean(lane_ious),
            'std_lane_iou': np.std(lane_ious),
            'avg_bg_iou': np.mean(bg_ious),
            'std_bg_iou': np.std(bg_ious),
            'avg_inference_time': np.mean(inference_times),
            'total_inference_time': np.sum(inference_times),
            'fps': 1.0 / np.mean(inference_times)
        }
        
        # 创建统计图表
        self.create_statistics_plot(stats, pixel_accs, mious, lane_ious, inference_times, output_dir)
        
        # 生成文本报告
        report_path = os.path.join(output_dir, 'test_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# TUSimple Lane Segmentation Test Report (Random 50 x 20.jpg)\n\n")
            f.write(f"## 测试概述\n")
            f.write(f"- 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- 测试样本数: {stats['num_samples']} (随机选择)\n")
            f.write(f"- 模型: Fast-SCNN\n")
            f.write(f"- 数据集: TUSimple (20.jpg specific frames)\n")
            f.write(f"- 随机种子: 42 (可重复结果)\n\n")
            
            f.write(f"## 性能指标\n")
            f.write(f"### 分割精度\n")
            f.write(f"- **平均像素准确率**: {stats['avg_pixel_acc']*100:.3f}% ± {stats['std_pixel_acc']*100:.3f}%\n")
            f.write(f"- **平均mIoU**: {stats['avg_miou']*100:.3f}% ± {stats['std_miou']*100:.3f}%\n")
            f.write(f"- **车道线IoU**: {stats['avg_lane_iou']*100:.3f}% ± {stats['std_lane_iou']*100:.3f}%\n")
            f.write(f"- **背景IoU**: {stats['avg_bg_iou']*100:.3f}% ± {stats['std_bg_iou']*100:.3f}%\n\n")
            
            f.write(f"### 推理性能\n")
            f.write(f"- **平均推理时间**: {stats['avg_inference_time']*1000:.2f}ms\n")
            f.write(f"- **推理FPS**: {stats['fps']:.1f}\n")
            f.write(f"- **总推理时间**: {stats['total_inference_time']:.2f}s\n\n")
            
            f.write(f"## 详细结果\n")
            f.write(f"| 序号 | 像素准确率(%) | mIoU(%) | 车道线IoU(%) | 推理时间(ms) |\n")
            f.write(f"|------|---------------|---------|--------------|-------------|\n")
            
            for i, m in enumerate(all_metrics):
                f.write(f"| {i+1:2d} | {m['pixel_acc']*100:11.2f} | {m['miou']*100:7.2f} | {m['iou_lane']*100:10.2f} | {m['inference_time']*1000:9.1f} |\n")
            
            f.write(f"\n## 分析结论\n")
            if stats['avg_miou'] > 0.7:
                conclusion = "优秀"
            elif stats['avg_miou'] > 0.6:
                conclusion = "良好"
            else:
                conclusion = "一般"
            
            f.write(f"- **总体评价**: {conclusion} (mIoU: {stats['avg_miou']*100:.2f}%)\n")
            f.write(f"- **车道线检测效果**: {'优秀' if stats['avg_lane_iou'] > 0.6 else '良好' if stats['avg_lane_iou'] > 0.4 else '一般'}\n")
            f.write(f"- **实时性**: {'满足实时要求' if stats['fps'] > 10 else '接近实时' if stats['fps'] > 5 else '需要优化'}\n")
            f.write(f"- **稳定性**: {'稳定' if stats['std_miou'] < 0.1 else '中等' if stats['std_miou'] < 0.2 else '不稳定'}\n")
        
        print(f"\n📊 Test Statistics:")
        print(f"   Samples: {stats['num_samples']}")
        print(f"   Avg PixAcc: {stats['avg_pixel_acc']*100:.3f}% ± {stats['std_pixel_acc']*100:.3f}%")
        print(f"   Avg mIoU: {stats['avg_miou']*100:.3f}% ± {stats['std_miou']*100:.3f}%")
        print(f"   Avg Lane IoU: {stats['avg_lane_iou']*100:.3f}% ± {stats['std_lane_iou']*100:.3f}%")
        print(f"   Avg Inference: {stats['avg_inference_time']*1000:.2f}ms ({stats['fps']:.1f} FPS)")
        print(f"   Report saved to: {report_path}")
    
    def create_statistics_plot(self, stats, pixel_accs, mious, lane_ious, inference_times, output_dir):
        """创建统计图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('TUSimple Random 50x20.jpg Test Results Statistics', fontsize=16, fontweight='bold')
        
        # 指标分布直方图
        ax1.hist(np.array(mious)*100, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(stats['avg_miou']*100, color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["avg_miou"]*100:.2f}%')
        ax1.set_xlabel('mIoU (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('mIoU Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 像素准确率分布
        ax2.hist(np.array(pixel_accs)*100, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.axvline(stats['avg_pixel_acc']*100, color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["avg_pixel_acc"]*100:.2f}%')
        ax2.set_xlabel('Pixel Accuracy (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Pixel Accuracy Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 车道线IoU分布
        ax3.hist(np.array(lane_ious)*100, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.axvline(stats['avg_lane_iou']*100, color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["avg_lane_iou"]*100:.2f}%')
        ax3.set_xlabel('Lane IoU (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Lane IoU Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 推理时间分布
        ax4.hist(np.array(inference_times)*1000, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax4.axvline(stats['avg_inference_time']*1000, color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["avg_inference_time"]*1000:.1f}ms')
        ax4.set_xlabel('Inference Time (ms)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Inference Time Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        stats_plot_path = os.path.join(output_dir, 'statistics_plot.png')
        plt.savefig(stats_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Statistics plot saved to: {stats_plot_path}")


def main():
    print("🚗 TUSimple Lane Segmentation - Random 50 Images Test (20.jpg)")
    print("=" * 65)
    
    # 设置随机种子以获得可重复的结果
    random.seed(42)
    
    tester = TUSimpleSpecificTester()
    tester.test_specific_images()


if __name__ == '__main__':
    main()
