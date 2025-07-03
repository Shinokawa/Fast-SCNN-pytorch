#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BDD100K可驾驶区域分割推理测试脚本
测试训练好的模型在测试集上的效果
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from models.fast_scnn import FastSCNN
from data_loader import get_segmentation_dataset

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_bdd100k_inference():
    """BDD100K推理测试"""
    print("🚗 BDD100K可驾驶区域分割推理测试")
    print("=" * 50)
    
    # 设置参数
    model_path = './weights/fast_scnn_bdd100k_best_model.pth'
    output_dir = './bdd100k_inference_results'
    test_samples = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🔧 使用设备: {device}")
    print(f"📂 模型路径: {model_path}")
    print(f"📂 输出目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请确保训练完成并保存了模型权重")
        return
    
    # 加载模型
    print(f"\n📥 加载模型...")
    model = FastSCNN(num_classes=2, aux=False)  # 二分类
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    
    # 加载测试数据集
    print(f"\n📂 加载测试数据集...")
    try:
        test_dataset = get_segmentation_dataset(
            'bdd100k', split='val', mode='val',  # 使用验证集作为测试
            subset='100k', label_type='binary',
            keep_original_size=True,  # 保持原尺寸
            max_samples=test_samples
        )
        print(f"✅ 加载了 {len(test_dataset)} 个测试样本")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return
    
    # 开始推理测试
    print(f"\n🔍 开始推理测试...")
    
    total_inference_time = 0
    results = []
    
    for i in range(min(test_samples, len(test_dataset))):
        print(f"\n📷 处理样本 {i+1}/{test_samples}")
        
        # 获取测试样本
        img_data, gt_mask = test_dataset[i]
        
        # 处理图像数据
        if isinstance(img_data, np.ndarray):
            # 如果是numpy数组，转换为PIL图像
            if len(img_data.shape) == 3:
                if img_data.shape[0] == 3:  # CHW格式
                    img_array = img_data.transpose(1, 2, 0)
                else:  # HWC格式
                    img_array = img_data
                # 确保数据范围正确
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                original_img = Image.fromarray(img_array.astype(np.uint8))
            else:
                print(f"   ⚠️  不支持的图像格式: {img_data.shape}")
                continue
        else:
            original_img = img_data
        
        # 预处理图像
        img_tensor = transform(original_img).unsqueeze(0).to(device)
        
        # 推理
        with torch.no_grad():
            start_time = time.time()
            outputs = model(img_tensor)
            inference_time = time.time() - start_time
            
            # 获取预测结果
            if isinstance(outputs, tuple):
                pred = outputs[0]  # 主输出
            else:
                pred = outputs
            
            # 转换为概率和预测类别
            prob = F.softmax(pred, dim=1)
            pred_class = torch.argmax(pred, dim=1)
            
            # 转换为numpy
            prob_np = prob.squeeze().cpu().numpy()
            pred_np = pred_class.squeeze().cpu().numpy()
        
        total_inference_time += inference_time
        
        # 计算准确率 (与真实标签对比)
        if isinstance(gt_mask, torch.Tensor):
            gt_mask_np = gt_mask.numpy()
        else:
            gt_mask_np = np.array(gt_mask)
        
        # 计算指标
        correct_pixels = np.sum(pred_np == gt_mask_np)
        total_pixels = pred_np.size
        pixel_acc = correct_pixels / total_pixels
        
        # 计算IoU
        intersection = np.sum((pred_np == 1) & (gt_mask_np == 1))
        union = np.sum((pred_np == 1) | (gt_mask_np == 1))
        iou = intersection / union if union > 0 else 0
        
        print(f"   推理时间: {inference_time*1000:.1f}ms")
        print(f"   像素准确率: {pixel_acc*100:.2f}%")
        print(f"   IoU: {iou*100:.2f}%")
        
        results.append({
            'pixel_acc': pixel_acc,
            'iou': iou,
            'inference_time': inference_time
        })
        
        # 可视化结果
        try:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # 原始图像
            axes[0].imshow(original_img)
            axes[0].set_title('原始图像', fontsize=14)
            axes[0].axis('off')
            
            # 真实标签
            axes[1].imshow(gt_mask_np, cmap='gray')
            axes[1].set_title('真实标签\n(白色=可驾驶)', fontsize=14)
            axes[1].axis('off')
            
            # 预测结果
            axes[2].imshow(pred_np, cmap='gray')
            axes[2].set_title(f'预测结果\nIoU: {iou*100:.1f}%', fontsize=14)
            axes[2].axis('off')
            
            # 可驾驶区域概率热力图
            if len(prob_np.shape) > 2:  # 多类别
                drivable_prob = prob_np[1]  # 类别1的概率
            else:
                drivable_prob = prob_np
            im = axes[3].imshow(drivable_prob, cmap='hot', vmin=0, vmax=1)
            axes[3].set_title(f'概率热力图\n推理: {inference_time*1000:.1f}ms', fontsize=14)
            axes[3].axis('off')
            plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            # 保存结果
            save_path = os.path.join(output_dir, f'inference_result_{i+1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"   💾 结果已保存: {save_path}")
            
        except Exception as e:
            print(f"   ⚠️  可视化失败: {e}")
    
    # 统计结果
    print(f"\n📊 推理测试结果统计:")
    print("=" * 50)
    
    if results:
        avg_pixel_acc = np.mean([r['pixel_acc'] for r in results])
        avg_iou = np.mean([r['iou'] for r in results])
        avg_inference_time = np.mean([r['inference_time'] for r in results])
        fps = 1.0 / avg_inference_time
        
        print(f"平均像素准确率: {avg_pixel_acc*100:.2f}%")
        print(f"平均IoU: {avg_iou*100:.2f}%")
        print(f"平均推理时间: {avg_inference_time*1000:.1f}ms")
        print(f"理论FPS: {fps:.1f}")
        print(f"总推理时间: {total_inference_time:.2f}s")
        
        # 性能评估
        print(f"\n🎯 性能评估:")
        if avg_iou > 0.75:
            print("✅ IoU优秀 (>75%)")
        elif avg_iou > 0.60:
            print("🟡 IoU良好 (>60%)")
        else:
            print("⚠️  IoU需要改进 (<60%)")
        
        if fps > 30:
            print("⚡ 推理速度优秀 (>30 FPS)")
        elif fps > 15:
            print("🟡 推理速度良好 (>15 FPS)")
        else:
            print("⚠️  推理速度较慢 (<15 FPS)")
        
        # 智能小车适用性评估
        print(f"\n🚗 智能小车适用性:")
        if avg_iou > 0.7 and fps > 10:
            print("✅ 完全适用于智能小车!")
            print("   - 精度足够安全驾驶")
            print("   - 速度满足实时要求")
        elif avg_iou > 0.6:
            print("🟡 基本适用于智能小车")
            print("   - 精度可接受，建议低速行驶")
        else:
            print("⚠️  需要进一步优化")
    
    print(f"\n💾 推理结果已保存到: {output_dir}")
    print("🎉 推理测试完成!")

if __name__ == '__main__':
    test_bdd100k_inference()
