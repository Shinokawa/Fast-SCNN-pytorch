#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试不同数据加载模式对信息保留的影响
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from data_loader import get_segmentation_dataset

def test_data_loading_modes():
    """测试不同的数据加载模式"""
    
    # 创建不同的数据加载器配置
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    
    configs = [
        {
            'name': '标准模式 (裁剪到768x768)',
            'kwargs': {
                'transform': input_transform,
                'base_size': 1024,
                'crop_size': 768,
                'sample_ratio': 0.001,  # 只取很少样本用于测试
                'max_samples': 3
            }
        },
        {
            'name': '原尺寸保持模式',
            'kwargs': {
                'transform': None,  # 不使用transform以便查看原始尺寸
                'keep_original_size': True,
                'sample_ratio': 0.001,
                'max_samples': 3
            }
        },
        {
            'name': '多尺度模式 (0.8-1.2x)',
            'kwargs': {
                'transform': None,
                'multi_scale': True,
                'min_scale': 0.8,
                'max_scale': 1.2,
                'sample_ratio': 0.001,
                'max_samples': 3
            }
        }
    ]
    
    print("=== 测试不同数据加载模式 ===\n")
    
    for config in configs:
        print(f"📊 {config['name']}")
        try:
            dataset = get_segmentation_dataset('bdd100k', split='train', mode='train',
                                             subset='100k', label_type='binary',
                                             **config['kwargs'])
            print(f"   数据集大小: {len(dataset)}")
            
            if len(dataset) > 0:
                img, mask = dataset[0]
                if hasattr(img, 'shape'):
                    print(f"   图像形状: {img.shape}")
                else:
                    print(f"   图像尺寸: {img.size}")
                
                if hasattr(mask, 'shape'):
                    print(f"   掩码形状: {mask.shape}")
                else:
                    print(f"   掩码尺寸: {mask.size}")
                
                # 计算有效像素比例
                if hasattr(mask, 'numpy'):
                    mask_np = mask.numpy()
                elif hasattr(mask, 'shape'):
                    mask_np = mask
                else:
                    mask_np = np.array(mask)
                
                total_pixels = mask_np.size
                drivable_pixels = np.sum(mask_np == 1)
                drivable_ratio = drivable_pixels / total_pixels * 100
                print(f"   可驾驶区域比例: {drivable_ratio:.2f}%")
                
        except Exception as e:
            print(f"   ❌ 错误: {str(e)}")
        
        print()

def visualize_size_comparison():
    """可视化不同模式下的尺寸对比"""
    
    configs = [
        ('标准裁剪', {'base_size': 1024, 'crop_size': 768}),
        ('原尺寸', {'keep_original_size': True}),
        ('多尺度', {'multi_scale': True, 'min_scale': 0.9, 'max_scale': 1.1})
    ]
    
    plt.figure(figsize=(15, 5))
    
    for idx, (name, kwargs) in enumerate(configs):
        try:
            dataset = get_segmentation_dataset('bdd100k', split='train', mode='train',
                                             subset='100k', label_type='binary',
                                             sample_ratio=0.0005, max_samples=1, **kwargs)
            
            if len(dataset) > 0:
                img, mask = dataset[0]
                
                # 转换为可显示格式
                if hasattr(img, 'numpy'):
                    if img.shape[0] == 3:  # CHW format
                        img_display = img.numpy().transpose(1, 2, 0)
                    else:
                        img_display = img.numpy()
                else:
                    img_display = np.array(img)
                
                if hasattr(mask, 'numpy'):
                    mask_display = mask.numpy()
                else:
                    mask_display = np.array(mask)
                
                plt.subplot(2, 3, idx + 1)
                plt.imshow(img_display)
                plt.title(f'{name}\n图像: {img_display.shape}')
                plt.axis('off')
                
                plt.subplot(2, 3, idx + 4)
                plt.imshow(mask_display, cmap='gray')
                plt.title(f'掩码: {mask_display.shape}')
                plt.axis('off')
        
        except Exception as e:
            print(f"配置 {name} 出错: {e}")
    
    plt.tight_layout()
    plt.savefig('data_loading_comparison.png', dpi=150, bbox_inches='tight')
    print("可视化结果保存到 'data_loading_comparison.png'")

if __name__ == '__main__':
    print("🔍 开始测试数据加载模式...")
    test_data_loading_modes()
    
    print("🎨 生成可视化对比...")
    try:
        visualize_size_comparison()
    except Exception as e:
        print(f"可视化生成失败: {e}")
    
    print("\n💡 建议:")
    print("1. 对于完整场景理解: 使用 --keep-original-size")
    print("2. 对于多尺度鲁棒性: 使用 --multi-scale")
    print("3. 对于快速训练: 使用标准裁剪模式")
    print("4. Fast-SCNN支持任意尺寸，可以混合使用不同尺寸的数据")
