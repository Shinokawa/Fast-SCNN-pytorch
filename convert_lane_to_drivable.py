#!/usr/bin/env python3
"""
将车道线mask转换为可驾驶区域mask
从两条白线标注转换为两线之间的填充区域标注
适用于小车可驾驶区域分割训练
"""

import os
import cv2
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

def detect_lane_lines(mask):
    """
    检测车道线并返回左右边界
    """
    # 确保mask是二值图像
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # 二值化处理
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 形态学操作去噪
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary

def find_lane_boundaries(binary_mask):
    """
    寻找车道线边界，返回左右边界坐标
    """
    height, width = binary_mask.shape
    left_boundary = []
    right_boundary = []
    
    # 逐行扫描寻找车道线像素
    for y in range(height):
        row = binary_mask[y, :]
        white_pixels = np.where(row > 127)[0]
        
        if len(white_pixels) >= 2:
            # 如果有多个白色像素点，取最左和最右作为边界
            left_x = white_pixels[0]
            right_x = white_pixels[-1]
            left_boundary.append((left_x, y))
            right_boundary.append((right_x, y))
        elif len(white_pixels) == 1:
            # 如果只有一个点，可能是车道线中心，需要估算宽度
            center_x = white_pixels[0]
            # 使用固定宽度估算（可根据实际情况调整）
            lane_width = max(20, width // 8)  # 最小20像素，或图像宽度的1/8
            left_x = max(0, center_x - lane_width // 2)
            right_x = min(width - 1, center_x + lane_width // 2)
            left_boundary.append((left_x, y))
            right_boundary.append((right_x, y))
    
    return left_boundary, right_boundary

def create_drivable_mask(mask_shape, left_boundary, right_boundary):
    """
    根据车道线边界创建可驾驶区域mask
    """
    height, width = mask_shape
    drivable_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 如果边界点太少，返回空mask
    if len(left_boundary) < 10 or len(right_boundary) < 10:
        return drivable_mask
    
    # 为每一行填充左右边界之间的区域
    for i, ((left_x, y), (right_x, _)) in enumerate(zip(left_boundary, right_boundary)):
        if left_x < right_x:
            drivable_mask[y, left_x:right_x+1] = 255
    
    # 形态学操作平滑结果
    kernel = np.ones((5, 5), np.uint8)
    drivable_mask = cv2.morphologyEx(drivable_mask, cv2.MORPH_CLOSE, kernel)
    
    return drivable_mask

def convert_single_mask(input_path, output_path, visualize=False):
    """
    转换单个mask文件
    """
    try:
        # 读取原始mask
        mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Cannot read {input_path}")
            return False
        
        # 检测车道线
        binary_mask = detect_lane_lines(mask)
        
        # 寻找边界
        left_boundary, right_boundary = find_lane_boundaries(binary_mask)
        
        # 创建可驾驶区域mask
        drivable_mask = create_drivable_mask(mask.shape, left_boundary, right_boundary)
        
        # 保存结果
        cv2.imwrite(output_path, drivable_mask)
        
        # 可视化（可选）
        if visualize:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(mask, cmap='gray')
            plt.title('Original Lane Mask')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(binary_mask, cmap='gray')
            plt.title('Processed Lane Lines')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(drivable_mask, cmap='gray')
            plt.title('Drivable Area Mask')
            plt.axis('off')
            
            vis_path = output_path.replace('.png', '_visualization.png')
            plt.tight_layout()
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def convert_all_masks(input_dir, output_dir, visualize_samples=5):
    """
    批量转换所有mask文件
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有mask文件
    mask_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    
    if not mask_files:
        print(f"No PNG files found in {input_dir}")
        return
    
    print(f"Found {len(mask_files)} mask files to convert")
    
    success_count = 0
    
    # 处理每个mask文件
    for i, filename in enumerate(tqdm(mask_files, desc="Converting masks")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # 是否可视化（只对前几个样本）
        visualize = i < visualize_samples
        
        if convert_single_mask(input_path, output_path, visualize):
            success_count += 1
    
    print(f"\nConversion completed!")
    print(f"Successfully converted: {success_count}/{len(mask_files)} files")
    print(f"Output directory: {output_dir}")
    
    if visualize_samples > 0:
        print(f"Visualization samples saved for first {min(visualize_samples, len(mask_files))} files")

def main():
    parser = argparse.ArgumentParser(description='Convert lane line masks to drivable area masks')
    parser.add_argument('--input_dir', type=str, default='data/custom/masks',
                        help='Input directory containing lane line masks')
    parser.add_argument('--output_dir', type=str, default='data/custom/drivable_masks',
                        help='Output directory for drivable area masks')
    parser.add_argument('--visualize_samples', type=int, default=5,
                        help='Number of samples to visualize (0 to disable)')
    
    args = parser.parse_args()
    
    print("Lane Line to Drivable Area Mask Converter")
    print("=" * 50)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Visualization samples: {args.visualize_samples}")
    print()
    
    convert_all_masks(args.input_dir, args.output_dir, args.visualize_samples)

if __name__ == "__main__":
    main()
