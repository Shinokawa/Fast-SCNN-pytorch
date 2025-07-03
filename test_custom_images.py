#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对自定义图片进行Fast-SCNN推理和可视化
使用训练好的BDD100K二分类模型对data/custom/images下的图片进行推理
生成可视化结果并保存到output目录
"""

import os
import argparse
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# 添加模型路径
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.fast_scnn import get_fast_scnn


def load_model(weights_path, device, num_classes=2):
    """加载训练好的模型"""
    print(f"Loading model from {weights_path}")
    
    # 创建模型
    model = get_fast_scnn(
        dataset='bdd100k',
        pretrained=False,
        root='.',
        map_cpu=True,
        num_class=num_classes
    )
    
    # 加载权重
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded successfully")
    else:
        print(f"Warning: Weight file {weights_path} not found, using random weights")
    
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, target_size=(1024, 512)):
    """预处理图片"""
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # BGR转RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]  # (H, W)
    
    # 调整大小
    image_resized = cv2.resize(image, target_size)
    
    # 归一化
    image_norm = image_resized.astype(np.float32) / 255.0
    
    # 标准化 (ImageNet标准)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_norm - mean) / std
    
    # 转换为torch tensor并添加batch维度
    image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).unsqueeze(0).float()
    
    return image_tensor, image, original_size


def postprocess_prediction(pred, original_size, target_size=(1024, 512)):
    """后处理预测结果"""
    # Fast-SCNN返回的是tuple，取第一个元素（主要输出）
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    
    # 如果输出大小不是目标大小，调整到目标大小
    if pred.shape[-2:] != target_size:
        pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=True)
    
    # 获取预测类别
    pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
    
    # 调整回原始大小
    if original_size != target_size:
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), 
                              (original_size[1], original_size[0]), 
                              interpolation=cv2.INTER_NEAREST)
    
    return pred_mask


def create_visualization(original_image, pred_mask, save_path=None):
    """创建可视化结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图片
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 预测mask (二分类: 0=不可驾驶, 1=可驾驶)
    # 使用颜色映射: 黑色=不可驾驶, 红色=可驾驶
    mask_colored = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    mask_colored[pred_mask == 1] = [255, 0, 0]  # 红色表示可驾驶区域
    
    axes[1].imshow(mask_colored)
    axes[1].set_title('Predicted Drivable Area')
    axes[1].axis('off')
    
    # 叠加显示
    overlay = original_image.copy()
    # 添加半透明的红色遮罩到可驾驶区域
    alpha = 0.4
    overlay[pred_mask == 1] = (1 - alpha) * overlay[pred_mask == 1] + alpha * np.array([255, 0, 0])
    
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.close()
    return mask_colored, overlay


def calculate_statistics(pred_mask, image_name):
    """计算推理统计信息"""
    total_pixels = pred_mask.size
    drivable_pixels = np.sum(pred_mask == 1)
    non_drivable_pixels = np.sum(pred_mask == 0)
    
    drivable_ratio = drivable_pixels / total_pixels * 100
    
    stats = {
        'image': image_name,
        'total_pixels': total_pixels,
        'drivable_pixels': drivable_pixels,
        'non_drivable_pixels': non_drivable_pixels,
        'drivable_ratio': drivable_ratio
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Fast-SCNN Custom Images Inference')
    parser.add_argument('--weights', type=str, 
                       default='weights/fast_scnn_bdd100k_best_model.pth',
                       help='Path to model weights')
    parser.add_argument('--images_dir', type=str, 
                       default='data/custom/images',
                       help='Directory containing custom images')
    parser.add_argument('--output_dir', type=str, 
                       default='custom_inference_results',
                       help='Directory to save results')
    parser.add_argument('--input_size', type=str, default='1024,512',
                       help='Input size as "width,height"')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--max_images', type=int, default=20,
                       help='Maximum number of images to process')
    parser.add_argument('--save_masks', action='store_true',
                       help='Save individual mask files')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 解析输入尺寸
    input_size = tuple(map(int, args.input_size.split(',')))
    print(f"Input size: {input_size}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 加载模型
    model = load_model(args.weights, device)
    
    # 获取图片列表
    images_dir = Path(args.images_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
        image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))
    
    image_files = sorted(image_files)[:args.max_images]
    print(f"Found {len(image_files)} images to process")
    
    if not image_files:
        print("No images found!")
        return
    
    # 推理统计
    all_stats = []
    total_time = 0
    
    print("\nStarting inference...")
    for i, image_path in enumerate(image_files):
        print(f"Processing [{i+1}/{len(image_files)}]: {image_path.name}")
        
        try:
            # 预处理
            start_time = time.time()
            image_tensor, original_image, original_size = preprocess_image(str(image_path), input_size)
            image_tensor = image_tensor.to(device)
            
            # 推理
            with torch.no_grad():
                pred = model(image_tensor)
            
            # 后处理
            pred_mask = postprocess_prediction(pred, original_size, input_size)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # 创建可视化
            vis_path = output_dir / f"{image_path.stem}_inference.png"
            mask_colored, overlay = create_visualization(original_image, pred_mask, str(vis_path))
            
            # 保存单独的mask (如果需要)
            if args.save_masks:
                mask_path = output_dir / f"{image_path.stem}_mask.png"
                cv2.imwrite(str(mask_path), mask_colored)
            
            # 计算统计信息
            stats = calculate_statistics(pred_mask, image_path.name)
            stats['inference_time'] = inference_time
            all_stats.append(stats)
            
            print(f"  - Inference time: {inference_time:.3f}s")
            print(f"  - Drivable area: {stats['drivable_ratio']:.1f}%")
            
        except Exception as e:
            print(f"  - Error processing {image_path.name}: {e}")
            continue
    
    # 生成总结报告
    if all_stats:
        avg_time = total_time / len(all_stats)
        avg_drivable = np.mean([s['drivable_ratio'] for s in all_stats])
        
        print(f"\n{'='*50}")
        print("INFERENCE SUMMARY")
        print(f"{'='*50}")
        print(f"Total images processed: {len(all_stats)}")
        print(f"Average inference time: {avg_time:.3f}s")
        print(f"Average FPS: {1/avg_time:.1f}")
        print(f"Average drivable area ratio: {avg_drivable:.1f}%")
        print(f"Results saved to: {output_dir}")
        
        # 保存详细统计
        stats_file = output_dir / 'inference_stats.txt'
        with open(stats_file, 'w') as f:
            f.write("Custom Images Inference Statistics\n")
            f.write("="*50 + "\n\n")
            f.write(f"Model: {args.weights}\n")
            f.write(f"Input size: {input_size}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Total images: {len(all_stats)}\n")
            f.write(f"Average inference time: {avg_time:.3f}s\n")
            f.write(f"Average FPS: {1/avg_time:.1f}\n")
            f.write(f"Average drivable area ratio: {avg_drivable:.1f}%\n\n")
            
            f.write("Individual Results:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Image':<30} {'Time(s)':<10} {'Drivable%':<12} {'Drivable Pixels':<15}\n")
            f.write("-" * 80 + "\n")
            
            for stats in all_stats:
                f.write(f"{stats['image']:<30} {stats['inference_time']:<10.3f} "
                       f"{stats['drivable_ratio']:<12.1f} {stats['drivable_pixels']:<15}\n")
        
        print(f"Detailed statistics saved to: {stats_file}")
        
        # 创建统计图表
        create_summary_plots(all_stats, output_dir)
    
    print("\nInference completed!")


def create_summary_plots(all_stats, output_dir):
    """创建汇总统计图表"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 推理时间分布
    times = [s['inference_time'] for s in all_stats]
    axes[0, 0].hist(times, bins=20, alpha=0.7, color='blue')
    axes[0, 0].set_title('Inference Time Distribution')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(np.mean(times), color='red', linestyle='--', label=f'Mean: {np.mean(times):.3f}s')
    axes[0, 0].legend()
    
    # 可驾驶区域比例分布
    ratios = [s['drivable_ratio'] for s in all_stats]
    axes[0, 1].hist(ratios, bins=20, alpha=0.7, color='green')
    axes[0, 1].set_title('Drivable Area Ratio Distribution')
    axes[0, 1].set_xlabel('Drivable Area (%)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].axvline(np.mean(ratios), color='red', linestyle='--', label=f'Mean: {np.mean(ratios):.1f}%')
    axes[0, 1].legend()
    
    # 推理时间趋势
    indices = range(len(times))
    axes[1, 0].plot(indices, times, 'b-', alpha=0.7)
    axes[1, 0].set_title('Inference Time Trend')
    axes[1, 0].set_xlabel('Image Index')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].axhline(np.mean(times), color='red', linestyle='--', label=f'Mean: {np.mean(times):.3f}s')
    axes[1, 0].legend()
    
    # 可驾驶区域比例趋势
    axes[1, 1].plot(indices, ratios, 'g-', alpha=0.7)
    axes[1, 1].set_title('Drivable Area Ratio Trend')
    axes[1, 1].set_xlabel('Image Index')
    axes[1, 1].set_ylabel('Drivable Area (%)')
    axes[1, 1].axhline(np.mean(ratios), color='red', linestyle='--', label=f'Mean: {np.mean(ratios):.1f}%')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plots saved to: {output_dir / 'inference_summary.png'}")


if __name__ == '__main__':
    main()
