#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TUSimple Test Set Inference Script
在test集上进行推理，支持原始尺寸输入，生成分割mask和可视化结果
"""

import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

from models.fast_scnn import get_fast_scnn
from data_loader.tusimple import TUSimpleDataset

def load_model(model_path, num_classes=2, device='cuda'):
    """加载训练好的模型"""
    model = get_fast_scnn(dataset='tusimple', num_classes=num_classes, pretrained=False)
    
    # 加载模型权重
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Best mIoU: {checkpoint.get('best_miou', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded from: {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, device='cuda'):
    """预处理单张图像，保持原始尺寸"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    # 转换为tensor并归一化
    image_array = np.array(image).astype(np.float32) / 255.0
    
    # 标准化 (ImageNet标准)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    for i in range(3):
        image_array[:, :, i] = (image_array[:, :, i] - mean[i]) / std[i]
    
    # 转换为tensor: H x W x C -> C x H x W
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    return image_tensor, original_size, np.array(image)

def inference_single_image(model, image_path, device='cuda'):
    """对单张图像进行推理"""
    image_tensor, original_size, original_image = preprocess_image(image_path, device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        pred = outputs[0]  # 主要输出
        
        # 应用softmax并获取预测类别
        pred_softmax = F.softmax(pred, dim=1)
        pred_class = torch.argmax(pred_softmax, dim=1).cpu().numpy()[0]
        pred_prob = pred_softmax.cpu().numpy()[0]
    
    return pred_class, pred_prob, original_image

def create_visualization(original_image, pred_mask, pred_prob, save_path=None):
    """创建可视化结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 原始图像
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 预测mask (二值化)
    axes[0, 1].imshow(pred_mask, cmap='gray')
    axes[0, 1].set_title('Predicted Lane Mask')
    axes[0, 1].axis('off')
    
    # 车道线概率图
    lane_prob = pred_prob[1]  # 车道线类别的概率
    axes[1, 0].imshow(lane_prob, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Lane Probability Map')
    axes[1, 0].axis('off')
    
    # 叠加显示
    overlay = original_image.copy()
    # 将车道线区域用红色叠加
    red_mask = np.zeros_like(overlay)
    red_mask[:, :, 0] = pred_mask * 255  # 红色通道
    overlay = cv2.addWeighted(overlay.astype(np.uint8), 0.7, red_mask.astype(np.uint8), 0.3, 0)
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Original + Lane Overlay')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    return fig

def run_test_inference():
    """在test集上运行推理"""
    # 配置
    model_path = "weights/best_model_tusimple.pth"  # 最佳模型路径
    test_data_root = "manideep1108/tusimple/versions/5/TUSimple"
    output_dir = "test_results"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # 加载模型
    print("Loading model...")
    model = load_model(model_path, num_classes=2, device=device)
    
    # 创建test数据集 (不应用变换，使用原始尺寸)
    print("Loading test dataset...")
    test_dataset = TUSimpleDataset(
        root=test_data_root,
        split='test',
        transform=None,  # 不使用数据增强
        mode='test'
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 推理统计
    total_time = 0
    num_samples = min(50, len(test_dataset))  # 只处理前50张图像作为演示
    
    print(f"Running inference on {num_samples} test images...")
    
    results = []
    
    for i in tqdm(range(num_samples), desc="Inference"):
        try:
            # 获取测试图像路径
            item = test_dataset[i]
            image_path = item['image_path'] if isinstance(item, dict) else test_dataset.images[i]
            
            # 确保路径正确
            if not os.path.isabs(image_path):
                image_path = os.path.join(test_data_root, image_path)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # 推理
            start_time = time.time()
            pred_mask, pred_prob, original_image = inference_single_image(model, image_path, device)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # 保存结果
            image_name = os.path.basename(image_path).replace('.jpg', '')
            
            # 保存mask
            mask_path = os.path.join(output_dir, "masks", f"{image_name}_mask.png")
            Image.fromarray((pred_mask * 255).astype(np.uint8)).save(mask_path)
            
            # 创建并保存可视化
            viz_path = os.path.join(output_dir, "visualizations", f"{image_name}_viz.png")
            fig = create_visualization(original_image, pred_mask, pred_prob, viz_path)
            plt.close(fig)
            
            # 记录结果
            lane_pixels = np.sum(pred_mask == 1)
            total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
            lane_ratio = lane_pixels / total_pixels
            
            results.append({
                'image_path': image_path,
                'inference_time': inference_time,
                'lane_pixel_ratio': lane_ratio,
                'image_size': original_image.shape[:2]
            })
            
            # 每10张图像打印一次进度
            if (i + 1) % 10 == 0:
                avg_time = total_time / (i + 1)
                print(f"Processed {i + 1}/{num_samples}, Avg time: {avg_time:.3f}s/image")
        
        except Exception as e:
            print(f"Error processing image {i}: {str(e)}")
            continue
    
    # 统计结果
    if results:
        avg_inference_time = total_time / len(results)
        avg_lane_ratio = np.mean([r['lane_pixel_ratio'] for r in results])
        
        print(f"\n=== Inference Results ===")
        print(f"Processed images: {len(results)}")
        print(f"Average inference time: {avg_inference_time:.3f} seconds/image")
        print(f"Average lane pixel ratio: {avg_lane_ratio:.3f}")
        print(f"FPS: {1.0/avg_inference_time:.1f}")
        
        # 保存统计结果
        stats = {
            'total_images': len(results),
            'average_inference_time': avg_inference_time,
            'average_lane_ratio': avg_lane_ratio,
            'fps': 1.0/avg_inference_time,
            'device': str(device),
            'model_path': model_path
        }
        
        with open(os.path.join(output_dir, 'inference_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
        print("- masks/: Binary lane masks")
        print("- visualizations/: Visualization results")
        print("- inference_stats.json: Performance statistics")
    
    return results

if __name__ == "__main__":
    # 需要安装cv2
    try:
        import cv2
    except ImportError:
        print("Installing opencv-python...")
        os.system("pip install opencv-python")
        import cv2
    
    results = run_test_inference()
