#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试自定义数据集微调模型的推理效果
支持单张图片和批量推理，提供详细的性能分析和可视化
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from models.fast_scnn import get_fast_scnn
from data_loader.custom import CustomDataset

def visualize_prediction(image, mask_true, mask_pred, save_path, filename):
    """可视化预测结果"""
    # 确保mask是numpy数组
    if torch.is_tensor(mask_true):
        mask_true = mask_true.cpu().numpy()
    if torch.is_tensor(mask_pred):
        mask_pred = mask_pred.cpu().numpy()
    
    # 创建图像
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始图像
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    # 真实标签
    axes[0, 1].imshow(mask_true, cmap='RdYlBu_r', vmin=0, vmax=1)
    axes[0, 1].set_title('真实标签 (红=可驾驶, 蓝=不可驾驶)')
    axes[0, 1].axis('off')
    
    # 预测结果
    axes[1, 0].imshow(mask_pred, cmap='RdYlBu_r', vmin=0, vmax=1)
    axes[1, 0].set_title('预测结果 (红=可驾驶, 蓝=不可驾驶)')
    axes[1, 0].axis('off')
    
    # 差异图
    diff = np.abs(mask_true.astype(float) - mask_pred.astype(float))
    axes[1, 1].imshow(diff, cmap='Reds', vmin=0, vmax=1)
    axes[1, 1].set_title('预测差异 (白=正确, 红=错误)')
    axes[1, 1].axis('off')
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(color='red', label='可驾驶区域'),
        mpatches.Patch(color='blue', label='不可驾驶区域')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.95))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(save_path, f'{filename}_prediction.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()

def calculate_metrics(mask_true, mask_pred):
    """计算分割指标"""
    if torch.is_tensor(mask_true):
        mask_true = mask_true.cpu().numpy()
    if torch.is_tensor(mask_pred):
        mask_pred = mask_pred.cpu().numpy()
    
    # 像素准确率
    correct = (mask_true == mask_pred).sum()
    total = mask_true.size
    pixel_acc = correct / total
    
    # 计算IoU for each class
    ious = []
    for cls in range(2):  # 0: 不可驾驶, 1: 可驾驶
        pred_cls = (mask_pred == cls)
        true_cls = (mask_true == cls)
        
        intersection = (pred_cls & true_cls).sum()
        union = (pred_cls | true_cls).sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0  # 如果该类别不存在，IoU为1
        ious.append(iou)
    
    mean_iou = np.mean(ious)
    
    return {
        'pixel_accuracy': pixel_acc,
        'mean_iou': mean_iou,
        'class_ious': ious
    }

def load_model(model_path, device):
    """加载模型"""
    print(f"加载模型: {model_path}")
    
    # 创建模型
    model = get_fast_scnn(dataset='custom', aux=False).to(device)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("模型加载完成")
    return model

def test_single_image(model, image_path, mask_path, device, save_path, filename):
    """测试单张图片"""
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # 加载真实标签
    mask_true = Image.open(mask_path)
    mask_true_array = np.array(mask_true)
    mask_true_binary = (mask_true_array > 0).astype(np.uint8)
    
    # 预处理图片
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array.transpose((2, 0, 1))).unsqueeze(0).to(device)
    
    # 推理
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image_tensor)
        if isinstance(outputs, (list, tuple)):
            output = outputs[0]
        else:
            output = outputs
        
        pred = F.softmax(output, dim=1)
        pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
    
    inference_time = time.time() - start_time
    
    # 调整预测结果大小以匹配原始图像
    if pred.shape != mask_true_binary.shape:
        pred_pil = Image.fromarray(pred.astype(np.uint8))
        pred_pil = pred_pil.resize((original_size[0], original_size[1]), Image.NEAREST)
        pred = np.array(pred_pil)
    
    # 计算指标
    metrics = calculate_metrics(mask_true_binary, pred)
    
    # 可视化
    visualize_prediction(image, mask_true_binary, pred, save_path, filename)
    
    print(f"图片: {filename}")
    print(f"  推理时间: {inference_time*1000:.2f}ms")
    print(f"  像素准确率: {metrics['pixel_accuracy']:.4f}")
    print(f"  平均IoU: {metrics['mean_iou']:.4f}")
    print(f"  不可驾驶IoU: {metrics['class_ious'][0]:.4f}")
    print(f"  可驾驶IoU: {metrics['class_ious'][1]:.4f}")
    print("-" * 50)
    
    return metrics, inference_time

def test_dataset(model, dataset, device, save_path, max_samples=None):
    """测试整个数据集"""
    all_metrics = []
    all_times = []
    
    if max_samples is None:
        max_samples = len(dataset)
    else:
        max_samples = min(max_samples, len(dataset))
    
    print(f"测试 {max_samples} 个样本...")
    
    for i in range(max_samples):
        # 获取数据
        image_tensor, mask_true = dataset[i]
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # 推理
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image_tensor)
            if isinstance(outputs, (list, tuple)):
                output = outputs[0]
            else:
                output = outputs
            
            pred = F.softmax(output, dim=1)
            pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
        
        inference_time = time.time() - start_time
        all_times.append(inference_time)
        
        # 计算指标
        metrics = calculate_metrics(mask_true.numpy(), pred)
        all_metrics.append(metrics)
        
        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1}/{max_samples} 个样本")
    
    # 计算平均指标
    avg_metrics = {
        'pixel_accuracy': np.mean([m['pixel_accuracy'] for m in all_metrics]),
        'mean_iou': np.mean([m['mean_iou'] for m in all_metrics]),
        'class_ious': [
            np.mean([m['class_ious'][0] for m in all_metrics]),
            np.mean([m['class_ious'][1] for m in all_metrics])
        ]
    }
    
    avg_time = np.mean(all_times)
    
    return avg_metrics, avg_time, all_metrics, all_times

def save_results_summary(metrics, inference_time, save_path, model_name):
    """保存测试结果摘要"""
    summary_path = os.path.join(save_path, 'test_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"自定义数据集微调模型测试结果\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"模型: {model_name}\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n性能指标:\n")
        f.write(f"  平均推理时间: {inference_time*1000:.2f}ms\n")
        f.write(f"  FPS: {1/inference_time:.2f}\n")
        f.write(f"  像素准确率: {metrics['pixel_accuracy']:.4f}\n")
        f.write(f"  平均IoU: {metrics['mean_iou']:.4f}\n")
        f.write(f"  不可驾驶IoU: {metrics['class_ious'][0]:.4f}\n")
        f.write(f"  可驾驶IoU: {metrics['class_ious'][1]:.4f}\n")
    
    print(f"测试结果摘要已保存到: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='测试自定义数据集微调模型')
    parser.add_argument('--model-path', type=str, 
                       default='weights/custom_finetune/best_model.pth',
                       help='微调模型路径')
    parser.add_argument('--data-folder', type=str, default='data/custom',
                       help='测试数据集路径')
    parser.add_argument('--output-folder', type=str, 
                       default='custom_finetune_results',
                       help='结果保存路径')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最大测试样本数')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    parser.add_argument('--test-mode', type=str, default='dataset',
                       choices=['dataset', 'single'],
                       help='测试模式: dataset (整个数据集) 或 single (单张图片)')
    parser.add_argument('--image-path', type=str, default=None,
                       help='单张图片路径 (仅当test-mode=single时使用)')
    parser.add_argument('--mask-path', type=str, default=None,
                       help='单张图片对应的标签路径 (仅当test-mode=single时使用)')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_folder, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(args.model_path, device)
    
    print("=" * 50)
    print("自定义数据集微调模型测试")
    print("=" * 50)
    
    if args.test_mode == 'single':
        # 单张图片测试
        if not args.image_path or not args.mask_path:
            print("错误: 单张图片测试需要指定 --image-path 和 --mask-path")
            return
        
        if not os.path.exists(args.image_path) or not os.path.exists(args.mask_path):
            print("错误: 指定的图片或标签文件不存在")
            return
        
        filename = os.path.splitext(os.path.basename(args.image_path))[0]
        metrics, inference_time = test_single_image(
            model, args.image_path, args.mask_path, device, 
            args.output_folder, filename
        )
        
        save_results_summary(metrics, inference_time, args.output_folder, args.model_path)
        
    else:
        # 数据集测试
        print(f"加载测试数据集: {args.data_folder}")
        
        test_dataset_obj = CustomDataset(
            root=args.data_folder,
            split='val',
            mode='val',
            original_size=True  # 使用原始尺寸进行测试
        )
        
        print(f"测试集大小: {len(test_dataset_obj)}")
        
        avg_metrics, avg_time, all_metrics, all_times = test_dataset(
            model, test_dataset_obj, device, args.output_folder, args.max_samples
        )
        
        print("\n" + "=" * 50)
        print("测试完成!")
        print("=" * 50)
        print(f"平均推理时间: {avg_time*1000:.2f}ms")
        print(f"平均FPS: {1/avg_time:.2f}")
        print(f"平均像素准确率: {avg_metrics['pixel_accuracy']:.4f}")
        print(f"平均IoU: {avg_metrics['mean_iou']:.4f}")
        print(f"不可驾驶IoU: {avg_metrics['class_ious'][0]:.4f}")
        print(f"可驾驶IoU: {avg_metrics['class_ious'][1]:.4f}")
        
        save_results_summary(avg_metrics, avg_time, args.output_folder, args.model_path)
        
        # 绘制性能分布图
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist([m['pixel_accuracy'] for m in all_metrics], bins=20, alpha=0.7)
        plt.title('像素准确率分布')
        plt.xlabel('像素准确率')
        plt.ylabel('频次')
        
        plt.subplot(1, 3, 2)
        plt.hist([m['mean_iou'] for m in all_metrics], bins=20, alpha=0.7)
        plt.title('平均IoU分布')
        plt.xlabel('平均IoU')
        plt.ylabel('频次')
        
        plt.subplot(1, 3, 3)
        plt.hist([t*1000 for t in all_times], bins=20, alpha=0.7)
        plt.title('推理时间分布')
        plt.xlabel('推理时间 (ms)')
        plt.ylabel('频次')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_folder, 'performance_distribution.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"结果已保存到: {args.output_folder}")


if __name__ == '__main__':
    main()
