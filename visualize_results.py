#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成自定义数据集微调模型的可视化效果对比
对比原图、真实标签、预测结果和误差图
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import cv2
    HAS_CV2 = True
    print("✅ 检测到opencv-python，将生成叠加效果图")
except ImportError:
    HAS_CV2 = False
    print("⚠️ 未安装opencv-python，将使用简化版可视化")

from models.fast_scnn import get_fast_scnn
from data_loader.custom import CustomDataset

def load_model(model_path, device):
    """加载微调模型"""
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
    print("✅ 模型加载完成")
    return model

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

def visualize_comparison(image, mask_true, mask_pred, save_path, filename, metrics=None):
    """生成详细的可视化对比图"""
    if HAS_CV2:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 原始图像
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('原始图像', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 真实标签
        im1 = axes[0, 1].imshow(mask_true, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[0, 1].set_title('真实标签', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # 预测结果
        im2 = axes[0, 2].imshow(mask_pred, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[0, 2].set_title('预测结果', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # 叠加效果 - 原图 + 真实标签
        overlay_true = np.array(image * 255).astype(np.uint8)
        mask_true_colored = np.zeros_like(overlay_true)
        mask_true_colored[mask_true == 1] = [0, 255, 0]  # 绿色表示可驾驶
        overlay_true = cv2.addWeighted(overlay_true, 0.7, mask_true_colored, 0.3, 0)
        axes[1, 0].imshow(overlay_true)
        axes[1, 0].set_title('原图 + 真实标签', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 叠加效果 - 原图 + 预测结果
        overlay_pred = np.array(image * 255).astype(np.uint8)
        mask_pred_colored = np.zeros_like(overlay_pred)
        mask_pred_colored[mask_pred == 1] = [255, 0, 0]  # 红色表示预测可驾驶
        overlay_pred = cv2.addWeighted(overlay_pred, 0.7, mask_pred_colored, 0.3, 0)
        axes[1, 1].imshow(overlay_pred)
        axes[1, 1].set_title('原图 + 预测结果', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # 差异图
        diff = np.abs(mask_true.astype(float) - mask_pred.astype(float))
        im3 = axes[1, 2].imshow(diff, cmap='Reds', vmin=0, vmax=1)
        axes[1, 2].set_title('预测误差', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        # 添加颜色条
        plt.colorbar(im1, ax=axes[0, 1], shrink=0.6)
        plt.colorbar(im2, ax=axes[0, 2], shrink=0.6)
        plt.colorbar(im3, ax=axes[1, 2], shrink=0.6)
        
    else:
        # 简化版（不使用opencv）
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 原始图像
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('原始图像', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 真实标签
        im1 = axes[0, 1].imshow(mask_true, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[0, 1].set_title('真实标签', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # 预测结果
        im2 = axes[1, 0].imshow(mask_pred, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[1, 0].set_title('预测结果', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 差异图
        diff = np.abs(mask_true.astype(float) - mask_pred.astype(float))
        im3 = axes[1, 1].imshow(diff, cmap='Reds', vmin=0, vmax=1)
        axes[1, 1].set_title('预测误差', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(color='red', label='可驾驶区域'),
        mpatches.Patch(color='blue', label='不可驾驶区域')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
              bbox_to_anchor=(0.5, 0.95), fontsize=12)
    
    # 添加性能指标文字
    if metrics:
        metrics_text = f"""性能指标:
像素准确率: {metrics['pixel_accuracy']:.3f}
平均IoU: {metrics['mean_iou']:.3f}
不可驾驶IoU: {metrics['class_ious'][0]:.3f}
可驾驶IoU: {metrics['class_ious'][1]:.3f}"""
        
        plt.figtext(0.02, 0.02, metrics_text, fontsize=11, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)
    
    save_file = os.path.join(save_path, f'{filename}_comparison.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 可视化对比图已保存: {save_file}")
    return save_file

def create_visualization_grid(model, dataset, device, save_path, num_samples=6):
    """创建多个样本的可视化网格"""
    fig, axes = plt.subplots(4, num_samples, figsize=(num_samples * 4, 16))
    
    row_titles = ['原始图像', '真实标签', '预测结果', '预测误差']
    
    for i in range(num_samples):
        if i >= len(dataset):
            break
            
        # 获取数据
        image_tensor, mask_true = dataset[i]
        
        # 转换为可显示格式
        image_np = image_tensor.permute(1, 2, 0).numpy()
        
        # 预测
        image_tensor_batch = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor_batch)
            if isinstance(outputs, (list, tuple)):
                output = outputs[0]
            else:
                output = outputs
            
            pred = F.softmax(output, dim=1)
            pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
        
        # 计算差异
        diff = np.abs(mask_true.numpy().astype(float) - pred.astype(float))
        
        # 绘制
        axes[0, i].imshow(image_np)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel(row_titles[0], fontsize=12, fontweight='bold')
        
        axes[1, i].imshow(mask_true.numpy(), cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel(row_titles[1], fontsize=12, fontweight='bold')
        
        axes[2, i].imshow(pred, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel(row_titles[2], fontsize=12, fontweight='bold')
        
        axes[3, i].imshow(diff, cmap='Reds', vmin=0, vmax=1)
        axes[3, i].axis('off')
        if i == 0:
            axes[3, i].set_ylabel(row_titles[3], fontsize=12, fontweight='bold')
        
        # 计算并显示指标
        metrics = calculate_metrics(mask_true.numpy(), pred)
        axes[3, i].set_title(f'mIoU: {metrics["mean_iou"]:.3f}', fontsize=10)
    
    plt.tight_layout()
    grid_file = os.path.join(save_path, 'comparison_grid.png')
    plt.savefig(grid_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 对比网格图已保存: {grid_file}")
    return grid_file

def main():
    # 配置
    model_path = 'weights/custom_finetune/best_model.pth'
    data_folder = 'data/custom'
    output_folder = 'custom_finetune_results'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🎨 生成自定义数据集微调模型可视化效果对比")
    print("=" * 60)
    print(f"📱 使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)
    
    # 检查必要文件
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    if not os.path.exists(data_folder):
        print(f"❌ 数据目录不存在: {data_folder}")
        return
    
    # 加载模型
    model = load_model(model_path, device)
    
    # 加载数据集
    print(f"📂 加载数据集: {data_folder}")
    try:
        dataset = CustomDataset(
            root=data_folder,
            split='val',
            mode='val',
            original_size=True
        )
        print(f"📊 数据集大小: {len(dataset)}")
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return
    
    if len(dataset) == 0:
        print("❌ 数据集为空")
        return
    
    # 生成对比网格图
    print("\n📊 生成多样本对比网格...")
    try:
        num_samples = min(6, len(dataset))
        grid_file = create_visualization_grid(model, dataset, device, output_folder, num_samples)
    except Exception as e:
        print(f"❌ 生成网格图失败: {e}")
    
    # 生成详细的单样本对比图
    print("\n🔍 生成详细单样本对比图...")
    sample_indices = [0]
    if len(dataset) > 5:
        sample_indices.append(5)
    if len(dataset) > 10:
        sample_indices.append(10)
    
    for idx in sample_indices:
        if idx >= len(dataset):
            continue
            
        try:
            # 获取数据
            image_tensor, mask_true = dataset[idx]
            
            # 转换图像格式
            image_np = image_tensor.permute(1, 2, 0).numpy()
            
            # 预测
            image_tensor_batch = image_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(image_tensor_batch)
                if isinstance(outputs, (list, tuple)):
                    output = outputs[0]
                else:
                    output = outputs
                
                pred = F.softmax(output, dim=1)
                pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
            
            # 计算指标
            metrics = calculate_metrics(mask_true.numpy(), pred)
            
            # 生成可视化
            visualize_comparison(image_np, mask_true.numpy(), pred, 
                               output_folder, f"sample_{idx:02d}", metrics)
            
        except Exception as e:
            print(f"❌ 处理样本 {idx} 失败: {e}")
    
    print("\n✅ 可视化生成完成!")
    print(f"📁 结果保存在: {output_folder}")
    print("\n生成的文件:")
    print(f"  - comparison_grid.png: 多样本对比网格")
    for idx in sample_indices:
        if idx < len(dataset):
            print(f"  - sample_{idx:02d}_comparison.png: 详细对比图")

if __name__ == '__main__':
    main()
