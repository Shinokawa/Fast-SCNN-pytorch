"""
简单检查生成的mask图像
"""
import os
import numpy as np
from PIL import Image

def check_mask_images():
    """检查生成的mask图像"""
    
    folders = ['test_samples_train', 'test_samples_val', 'test_samples_test']
    
    for folder in folders:
        if os.path.exists(folder):
            print(f"\n=== 检查 {folder} ===")
            
            # 列出所有processed mask文件
            mask_files = [f for f in os.listdir(folder) if f.endswith('_mask_processed.png')]
            
            for mask_file in mask_files:
                mask_path = os.path.join(folder, mask_file)
                original_mask_path = mask_path.replace('_processed.png', '_original.png')
                
                # 加载处理后的mask
                processed_mask = Image.open(mask_path)
                processed_array = np.array(processed_mask)
                
                # 加载原始mask
                if os.path.exists(original_mask_path):
                    original_mask = Image.open(original_mask_path)
                    original_array = np.array(original_mask)
                    
                    print(f"\n文件: {mask_file}")
                    print(f"  原始mask形状: {original_array.shape}")
                    print(f"  原始mask唯一值: {np.unique(original_array)}")
                    print(f"  处理后mask形状: {processed_array.shape}")
                    print(f"  处理后mask唯一值: {np.unique(processed_array)}")
                    print(f"  处理后mask模式: {processed_mask.mode}")
                    
                    # 计算车道线像素比例
                    if len(processed_array.shape) == 2:
                        lane_pixels = np.sum(processed_array > 127)
                        total_pixels = processed_array.size
                        lane_ratio = lane_pixels / total_pixels * 100
                        print(f"  车道线像素比例: {lane_ratio:.2f}%")
                    else:
                        print("  Warning: 处理后的mask不是灰度图像")

if __name__ == '__main__':
    check_mask_images()
