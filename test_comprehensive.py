#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Test Inference Script
在TUSimple测试集上进行全面的推理测试，生成详细的性能分析报告
"""

import os
import sys
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from models.fast_scnn import get_fast_scnn

def load_model(model_path, device='cuda'):
    """加载训练好的模型"""
    print(f"Loading model from: {model_path}")
    model = get_fast_scnn(dataset='tusimple', num_classes=2, pretrained=False)
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        best_miou = checkpoint.get('best_miou', 0)
        print(f"✓ Model loaded from epoch {epoch}, best mIoU: {best_miou:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("✓ Model loaded (no training info available)")
    
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, device='cuda'):
    """预处理图像，保持原始尺寸"""
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    h, w = original_image.shape[:2]
    
    # 转换为tensor并归一化
    image_array = original_image.astype(np.float32) / 255.0
    
    # ImageNet标准化
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        image_array[:, :, i] = (image_array[:, :, i] - mean[i]) / std[i]
    
    # 转换为tensor: H x W x C -> C x H x W
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    return image_tensor.to(device), original_image

def inference_single_image(model, image_path, device='cuda'):
    """对单张图像进行推理"""
    image_tensor, original_image = preprocess_image(image_path, device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image_tensor)
        pred = outputs[0]
        pred_softmax = F.softmax(pred, dim=1)
        pred_mask = torch.argmax(pred_softmax, dim=1).cpu().numpy()[0]
        lane_prob = pred_softmax[0, 1].cpu().numpy()
    
    inference_time = time.time() - start_time
    
    return pred_mask, lane_prob, original_image, inference_time

def create_visualization(original_image, pred_mask, lane_prob, save_path=None):
    """创建四合一可视化结果"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 原始图像
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # 预测mask
    axes[0, 1].imshow(pred_mask, cmap='gray')
    axes[0, 1].set_title('Lane Mask', fontsize=14)
    axes[0, 1].axis('off')
    
    # 概率热图
    im = axes[1, 0].imshow(lane_prob, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Lane Probability', fontsize=14)
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # 叠加效果
    overlay = original_image.copy()
    lane_pixels = pred_mask == 1
    overlay[lane_pixels] = [255, 0, 0]  # 红色叠加
    result = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)
    
    axes[1, 1].imshow(result)
    axes[1, 1].set_title('Lane Overlay', fontsize=14)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig

def find_test_images(max_images=50):
    """查找测试图像"""
    test_root = Path("manideep1108/tusimple/versions/5/TUSimple/test_set/clips")
    test_images = []
    
    if test_root.exists():
        print(f"Searching for test images in: {test_root}")
        
        for date_dir in ["0530", "0531", "0601"]:
            date_path = test_root / date_dir
            if date_path.exists():
                clip_dirs = [d for d in date_path.iterdir() if d.is_dir()]
                
                for clip_dir in clip_dirs:
                    jpg_files = list(clip_dir.glob("*.jpg"))
                    test_images.extend([str(f) for f in jpg_files])
                    
                    if len(test_images) >= max_images:
                        break
                
                if len(test_images) >= max_images:
                    break
    
    return test_images[:max_images]

def analyze_results(results):
    """分析推理结果"""
    if not results:
        return {}
    
    # 基本统计
    total_images = len(results)
    inference_times = [r['inference_time'] for r in results]
    lane_ratios = [r['lane_ratio'] for r in results]
    max_probs = [r['max_prob'] for r in results]
    mean_probs = [r['mean_prob'] for r in results]
    image_sizes = [r['image_size'] for r in results]
    
    # 计算统计指标
    stats = {
        'total_images': total_images,
        'inference_time': {
            'mean': np.mean(inference_times),
            'std': np.std(inference_times),
            'min': np.min(inference_times),
            'max': np.max(inference_times),
            'fps': 1.0 / np.mean(inference_times)
        },
        'lane_ratio': {
            'mean': np.mean(lane_ratios),
            'std': np.std(lane_ratios),
            'min': np.min(lane_ratios),
            'max': np.max(lane_ratios)
        },
        'max_probability': {
            'mean': np.mean(max_probs),
            'std': np.std(max_probs),
            'min': np.min(max_probs),
            'max': np.max(max_probs)
        },
        'mean_probability': {
            'mean': np.mean(mean_probs),
            'std': np.std(mean_probs),
            'min': np.min(mean_probs),
            'max': np.max(mean_probs)
        },
        'image_sizes': {
            'unique_sizes': list(set(image_sizes)),
            'size_counts': {str(size): image_sizes.count(size) for size in set(image_sizes)}
        }
    }
    
    return stats

def generate_report(stats, output_dir):
    """生成测试报告"""
    report_path = os.path.join(output_dir, 'test_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("TUSimple Lane Detection - Test Inference Report\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total processed images: {stats['total_images']}\n\n")
        
        f.write("📊 Inference Performance:\n")
        f.write(f"  Average inference time: {stats['inference_time']['mean']:.3f} ± {stats['inference_time']['std']:.3f} seconds\n")
        f.write(f"  Min/Max inference time: {stats['inference_time']['min']:.3f} / {stats['inference_time']['max']:.3f} seconds\n")
        f.write(f"  Average FPS: {stats['inference_time']['fps']:.1f}\n\n")
        
        f.write("🛣️ Lane Detection Statistics:\n")
        f.write(f"  Average lane pixel ratio: {stats['lane_ratio']['mean']:.3f} ± {stats['lane_ratio']['std']:.3f}\n")
        f.write(f"  Min/Max lane ratio: {stats['lane_ratio']['min']:.3f} / {stats['lane_ratio']['max']:.3f}\n\n")
        
        f.write("🎯 Prediction Confidence:\n")
        f.write(f"  Average max probability: {stats['max_probability']['mean']:.3f} ± {stats['max_probability']['std']:.3f}\n")
        f.write(f"  Average mean probability: {stats['mean_probability']['mean']:.3f} ± {stats['mean_probability']['std']:.3f}\n\n")
        
        f.write("📐 Image Size Analysis:\n")
        f.write(f"  Unique image sizes: {len(stats['image_sizes']['unique_sizes'])}\n")
        for size, count in stats['image_sizes']['size_counts'].items():
            f.write(f"    {size}: {count} images\n")
    
    print(f"📄 Detailed report saved to: {report_path}")

def run_comprehensive_test():
    """运行全面测试"""
    print("="*60)
    print("🚗 TUSimple Lane Detection - Comprehensive Test")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 检查模型
    model_path = "weights/fast_scnn_tusimple_best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # 加载模型
    model = load_model(model_path, device)
    
    # 查找测试图像
    test_images = find_test_images(max_images=30)  # 处理30张图像
    if not test_images:
        print("❌ No test images found!")
        return
    
    print(f"✓ Found {len(test_images)} test images")
    
    # 创建输出目录
    output_dir = "comprehensive_test_results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    
    print(f"✓ Output directory: {output_dir}")
    print("\n🔄 Running inference...")
    
    # 批量推理
    results = []
    failed_count = 0
    
    for i, image_path in enumerate(tqdm(test_images, desc="Processing")):
        try:
            # 推理
            pred_mask, lane_prob, original_image, inference_time = inference_single_image(
                model, image_path, device
            )
            
            # 计算统计信息
            lane_pixels = np.sum(pred_mask == 1)
            total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
            lane_ratio = lane_pixels / total_pixels if total_pixels > 0 else 0
            max_prob = np.max(lane_prob)
            mean_prob = np.mean(lane_prob[pred_mask == 1]) if lane_pixels > 0 else 0
            
            # 保存结果
            base_name = f"test_{i:03d}_{Path(image_path).stem}"
            
            # 保存mask
            mask_path = os.path.join(output_dir, "masks", f"{base_name}_mask.png")
            Image.fromarray((pred_mask * 255).astype(np.uint8)).save(mask_path)
            
            # 保存可视化（只保存前10张）
            if i < 10:
                viz_path = os.path.join(output_dir, "visualizations", f"{base_name}_viz.png")
                create_visualization(original_image, pred_mask, lane_prob, viz_path)
            
            # 记录结果
            results.append({
                'image_path': image_path,
                'image_name': Path(image_path).name,
                'inference_time': inference_time,
                'lane_ratio': lane_ratio,
                'max_prob': max_prob,
                'mean_prob': mean_prob,
                'image_size': (original_image.shape[1], original_image.shape[0]),  # (width, height)
                'lane_pixels': int(lane_pixels),
                'total_pixels': int(total_pixels)
            })
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {str(e)}")
            failed_count += 1
            continue
    
    # 分析结果
    print(f"\n📊 Processing Summary:")
    print(f"✓ Successfully processed: {len(results)}/{len(test_images)} images")
    print(f"❌ Failed: {failed_count} images")
    
    if results:
        stats = analyze_results(results)
        
        # 打印关键统计信息
        print(f"\n📈 Key Statistics:")
        print(f"⏱️  Average inference time: {stats['inference_time']['mean']:.3f}s ({stats['inference_time']['fps']:.1f} FPS)")
        print(f"🛣️  Average lane ratio: {stats['lane_ratio']['mean']:.3f}")
        print(f"🎯 Average max confidence: {stats['max_probability']['mean']:.3f}")
        print(f"📐 Processed {len(stats['image_sizes']['unique_sizes'])} different image sizes")
        
        # 保存详细结果
        results_path = os.path.join(output_dir, 'detailed_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        stats_path = os.path.join(output_dir, 'statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 生成报告
        generate_report(stats, output_dir)
        
        print(f"\n💾 Results saved to:")
        print(f"  📁 {output_dir}/")
        print(f"    📊 statistics.json - Statistical analysis")
        print(f"    📝 detailed_results.json - Detailed per-image results")
        print(f"    📄 test_report.txt - Human-readable report")
        print(f"    🖼️  visualizations/ - Sample visualizations (first 10 images)")
        print(f"    🎭 masks/ - All prediction masks")
        
        return True
    else:
        print("❌ No images were processed successfully")
        return False

if __name__ == "__main__":
    # 确保安装所需包
    try:
        import cv2
    except ImportError:
        print("Installing opencv-python...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
        import cv2
    
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
