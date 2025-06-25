#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick Test Inference Script
快速测试推理脚本，使用训练好的模型在测试图像上进行推理
"""

import os
import sys
import subprocess
from pathlib import Path

def find_test_images():
    """查找测试图像"""
    test_root = Path("manideep1108/tusimple/versions/5/TUSimple/test_set/clips")
    test_images = []
    
    if test_root.exists():
        print(f"Searching for test images in: {test_root}")
        
        # 遍历日期文件夹
        for date_dir in ["0530", "0531", "0601"]:
            date_path = test_root / date_dir
            if date_path.exists():
                print(f"  Checking date: {date_dir}")
                
                # 遍历clip文件夹
                clip_dirs = [d for d in date_path.iterdir() if d.is_dir()]
                for clip_dir in clip_dirs[:2]:  # 每个日期取前2个clip
                    # 查找jpg图像
                    jpg_files = list(clip_dir.glob("*.jpg"))
                    for jpg_file in jpg_files[:3]:  # 每个clip取前3张图
                        test_images.append(str(jpg_file))
                        if len(test_images) >= 6:  # 总共6张图像
                            return test_images
    
    return test_images

def run_demo():
    """运行演示"""
    print("=" * 60)
    print("🚗 TUSimple Lane Detection Demo")
    print("=" * 60)
    
    # 检查模型文件
    model_files = [
        "weights/fast_scnn_tusimple_best_model.pth",
        "weights/fast_scnn_tusimple.pth",
        "weights/best_model_tusimple.pth"
    ]
    
    model_path = None
    for model_file in model_files:
        if os.path.exists(model_file):
            model_path = model_file
            break
    
    if not model_path:
        print("❌ No trained model found!")
        print("Expected one of:")
        for model_file in model_files:
            print(f"  - {model_file}")
        return False
    
    print(f"✓ Found model: {model_path}")
    
    # 查找测试图像
    test_images = find_test_images()
    if not test_images:
        print("❌ No test images found!")
        print("Please check the TUSimple dataset path.")
        return False
    
    print(f"✓ Found {len(test_images)} test images")
    
    # 创建输出目录
    output_dir = "demo_output_original_size"
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    print("\n🔄 Running inference...")
    
    # 对每张图像运行推理
    success_count = 0
    for i, image_path in enumerate(test_images):
        print(f"\n📸 Processing image {i+1}/{len(test_images)}: {os.path.basename(image_path)}")
        
        try:
            # 运行demo脚本
            cmd = [
                sys.executable, "demo_tusimple.py",
                "--model-path", model_path,
                "--image-path", image_path,
                "--output-dir", output_dir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✓ Success")
                success_count += 1
            else:
                print(f"❌ Error: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            print("❌ Timeout")
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    print(f"\n📊 Summary:")
    print(f"✓ Successfully processed: {success_count}/{len(test_images)} images")
    print(f"📁 Results saved to: {output_dir}")
    
    if success_count > 0:
        print("\n🎉 Demo completed successfully!")
        print(f"Check the '{output_dir}' folder for:")
        print("  - *_original.jpg: Original images")
        print("  - *_mask.png: Lane detection masks")  
        print("  - *_overlay.jpg: Lane overlay visualizations")
        return True
    else:
        print("\n❌ Demo failed - no images were processed successfully")
        return False

if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)
