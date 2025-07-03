#!/usr/bin/env python3
"""
透视变换应用工具

功能特性：
- 加载相机标定参数
- 将输入图像转换为鸟瞰图
- 支持批量处理
- 可调整输出参数

使用方法：
python perspective_transform.py --input image.jpg --calibration calibration.json --output bird_eye.jpg
python perspective_transform.py --input_dir images/ --calibration calibration.json --output_dir bird_eye_images/

作者：用于自动驾驶感知模块的透视变换
"""

import cv2
import numpy as np
import json
import argparse
import os
from pathlib import Path
import glob

class PerspectiveTransformer:
    def __init__(self, calibration_path):
        """
        初始化透视变换器
        
        参数：
            calibration_path: 标定文件路径
        """
        self.calibration_path = calibration_path
        self.load_calibration()
        
    def load_calibration(self):
        """加载标定数据"""
        if not os.path.exists(self.calibration_path):
            raise FileNotFoundError(f"标定文件不存在: {self.calibration_path}")
        
        with open(self.calibration_path, 'r', encoding='utf-8') as f:
            self.calibration_data = json.load(f)
        
        # 提取关键参数
        self.transform_matrix = np.array(self.calibration_data['transform_matrix'], dtype=np.float32)
        self.inverse_transform_matrix = np.array(self.calibration_data['inverse_transform_matrix'], dtype=np.float32)
        self.image_points = self.calibration_data['image_points']
        self.world_points = self.calibration_data['world_points']
        self.original_image_size = self.calibration_data['image_size']
        
        print(f"✅ 标定数据已加载: {self.calibration_path}")
        print(f"📏 原始图像尺寸: {self.original_image_size[0]} × {self.original_image_size[1]}")
        print(f"📍 标定点数: {len(self.image_points)}")
    
    def calculate_bird_eye_params(self, pixels_per_unit=20, margin_ratio=0.2):
        """
        计算鸟瞰图参数
        
        参数：
            pixels_per_unit: 每单位的像素数
            margin_ratio: 边距比例
        
        返回：
            (output_width, output_height, combined_transform, view_bounds)
        """
        # 计算世界坐标范围
        world_points_array = np.array(self.world_points)
        min_x, min_y = world_points_array.min(axis=0)
        max_x, max_y = world_points_array.max(axis=0)
        
        # 添加边距
        range_x = max_x - min_x
        range_y = max_y - min_y
        margin = max(range_x, range_y) * margin_ratio
        
        min_x -= margin
        min_y -= margin
        max_x += margin
        max_y += margin
        
        # 计算输出图像尺寸
        output_width = int((max_x - min_x) * pixels_per_unit)
        output_height = int((max_y - min_y) * pixels_per_unit)
        
        # 创建世界坐标到像素坐标的变换矩阵
        world_to_pixel = np.array([
            [pixels_per_unit, 0, -min_x * pixels_per_unit],
            [0, pixels_per_unit, -min_y * pixels_per_unit],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 组合变换矩阵：图像坐标 → 世界坐标 → 像素坐标
        combined_transform = world_to_pixel @ self.transform_matrix
        
        view_bounds = (min_x, min_y, max_x, max_y)
        
        return output_width, output_height, combined_transform, view_bounds
    
    def transform_image(self, image, pixels_per_unit=20, margin_ratio=0.2):
        """
        将图像转换为鸟瞰图
        
        参数：
            image: 输入图像 (BGR格式)
            pixels_per_unit: 每单位的像素数
            margin_ratio: 边距比例
        
        返回：
            bird_eye_image: 鸟瞰图
            view_params: 视图参数字典
        """
        # 计算鸟瞰图参数
        output_width, output_height, combined_transform, view_bounds = \
            self.calculate_bird_eye_params(pixels_per_unit, margin_ratio)
        
        # 如果输入图像尺寸与标定时不同，需要调整变换矩阵
        input_height, input_width = image.shape[:2]
        orig_width, orig_height = self.original_image_size
        
        if input_width != orig_width or input_height != orig_height:
            print(f"⚠️ 图像尺寸不匹配: {input_width}×{input_height} vs {orig_width}×{orig_height}")
            print("🔄 自动调整变换矩阵...")
            
            # 计算缩放因子
            scale_x = input_width / orig_width
            scale_y = input_height / orig_height
            
            # 创建缩放矩阵
            scale_matrix = np.array([
                [scale_x, 0, 0],
                [0, scale_y, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # 调整变换矩阵
            adjusted_transform = combined_transform @ np.linalg.inv(scale_matrix)
            combined_transform = adjusted_transform
        
        # 执行透视变换
        bird_eye_image = cv2.warpPerspective(
            image, combined_transform, 
            (output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # 准备视图参数
        view_params = {
            'output_size': (output_width, output_height),
            'view_bounds': view_bounds,
            'pixels_per_unit': pixels_per_unit,
            'margin_ratio': margin_ratio,
            'transform_matrix': combined_transform.tolist()
        }
        
        return bird_eye_image, view_params
    
    def transform_point_to_bird_eye(self, image_x, image_y, pixels_per_unit=20, margin_ratio=0.2):
        """
        将图像坐标点转换为鸟瞰图坐标
        
        参数：
            image_x, image_y: 图像坐标
            pixels_per_unit: 每单位的像素数
            margin_ratio: 边距比例
        
        返回：
            (bird_x, bird_y): 鸟瞰图像素坐标
        """
        # 计算鸟瞰图参数
        _, _, combined_transform, _ = self.calculate_bird_eye_params(pixels_per_unit, margin_ratio)
        
        # 将点转换为齐次坐标
        point_homo = np.array([[image_x, image_y, 1]], dtype=np.float32).T
        
        # 应用变换
        transformed_homo = combined_transform @ point_homo
        
        # 转换回笛卡尔坐标
        bird_x = int(transformed_homo[0, 0] / transformed_homo[2, 0])
        bird_y = int(transformed_homo[1, 0] / transformed_homo[2, 0])
        
        return bird_x, bird_y
    
    def add_grid_and_labels(self, bird_eye_image, view_params):
        """
        在鸟瞰图上添加网格和标签
        
        参数：
            bird_eye_image: 鸟瞰图
            view_params: 视图参数
        
        返回：
            带网格的鸟瞰图
        """
        annotated_image = bird_eye_image.copy()
        
        min_x, min_y, max_x, max_y = view_params['view_bounds']
        pixels_per_unit = view_params['pixels_per_unit']
        output_width, output_height = view_params['output_size']
        
        # 绘制网格
        grid_interval = 10  # 网格间隔（单位）
        grid_color = (128, 128, 128)  # 灰色
        
        # 垂直线
        x = min_x
        while x <= max_x:
            if x % grid_interval == 0:
                pixel_x = int((x - min_x) * pixels_per_unit)
                if 0 <= pixel_x < output_width:
                    cv2.line(annotated_image, (pixel_x, 0), (pixel_x, output_height-1), grid_color, 1)
                    
                    # 添加X坐标标签
                    if x != 0:  # 避免在原点重复标注
                        label = f"{int(x)}"
                        cv2.putText(annotated_image, label, (pixel_x + 2, 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_color, 1)
            x += grid_interval / 2
        
        # 水平线
        y = min_y
        while y <= max_y:
            if y % grid_interval == 0:
                pixel_y = int((y - min_y) * pixels_per_unit)
                if 0 <= pixel_y < output_height:
                    cv2.line(annotated_image, (0, pixel_y), (output_width-1, pixel_y), grid_color, 1)
                    
                    # 添加Y坐标标签
                    if y != 0:  # 避免在原点重复标注
                        label = f"{int(y)}"
                        cv2.putText(annotated_image, label, (5, pixel_y - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_color, 1)
            y += grid_interval / 2
        
        # 绘制原点
        origin_x = int((0 - min_x) * pixels_per_unit)
        origin_y = int((0 - min_y) * pixels_per_unit)
        
        if 0 <= origin_x < output_width and 0 <= origin_y < output_height:
            cv2.circle(annotated_image, (origin_x, origin_y), 5, (0, 0, 255), -1)
            cv2.putText(annotated_image, "O(0,0)", (origin_x + 8, origin_y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 标记标定点
        for i, (world_x, world_y) in enumerate(self.world_points):
            pixel_x = int((world_x - min_x) * pixels_per_unit)
            pixel_y = int((world_y - min_y) * pixels_per_unit)
            
            if 0 <= pixel_x < output_width and 0 <= pixel_y < output_height:
                cv2.circle(annotated_image, (pixel_x, pixel_y), 3, (0, 255, 255), -1)
                cv2.putText(annotated_image, f"P{i+1}", (pixel_x + 5, pixel_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        return annotated_image

def process_single_image(image_path, transformer, output_path=None, pixels_per_unit=20, 
                        margin_ratio=0.2, add_grid=False):
    """
    处理单张图像
    
    参数：
        image_path: 输入图像路径
        transformer: 透视变换器对象
        output_path: 输出路径（可选）
        pixels_per_unit: 每单位像素数
        margin_ratio: 边距比例
        add_grid: 是否添加网格
    
    返回：
        输出文件路径
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    print(f"🔄 处理图像: {image_path}")
    
    # 执行透视变换
    bird_eye_image, view_params = transformer.transform_image(
        image, pixels_per_unit, margin_ratio)
    
    # 添加网格（可选）
    if add_grid:
        bird_eye_image = transformer.add_grid_and_labels(bird_eye_image, view_params)
    
    # 确定输出路径
    if output_path is None:
        output_path = image_path.replace('.jpg', '_bird_eye.jpg').replace('.png', '_bird_eye.png')
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存结果
    cv2.imwrite(output_path, bird_eye_image)
    
    # 打印信息
    input_size = f"{image.shape[1]}×{image.shape[0]}"
    output_size = f"{view_params['output_size'][0]}×{view_params['output_size'][1]}"
    bounds = view_params['view_bounds']
    
    print(f"  📏 输入尺寸: {input_size}")
    print(f"  📏 输出尺寸: {output_size}")
    print(f"  📐 世界范围: X({bounds[0]:.1f}~{bounds[2]:.1f}), Y({bounds[1]:.1f}~{bounds[3]:.1f})")
    print(f"  💾 已保存: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="透视变换应用工具")
    parser.add_argument("--calibration", "-c", required=True, help="标定文件路径")
    parser.add_argument("--input", "-i", help="输入图像路径")
    parser.add_argument("--output", "-o", help="输出图像路径")
    parser.add_argument("--input_dir", help="输入目录路径（批量处理）")
    parser.add_argument("--output_dir", help="输出目录路径（批量处理）")
    parser.add_argument("--pixels_per_unit", type=int, default=20, help="每单位像素数 (默认: 20)")
    parser.add_argument("--margin_ratio", type=float, default=0.2, help="边距比例 (默认: 0.2)")
    parser.add_argument("--add_grid", action="store_true", help="添加网格和标签")
    parser.add_argument("--preview", action="store_true", help="显示预览窗口")
    
    args = parser.parse_args()
    
    try:
        # 创建透视变换器
        print("🧮 加载透视变换器...")
        transformer = PerspectiveTransformer(args.calibration)
        
        if args.input:
            # 单图像处理
            output_path = process_single_image(
                args.input, transformer, args.output,
                args.pixels_per_unit, args.margin_ratio, args.add_grid
            )
            
            # 预览（可选）
            if args.preview:
                print("👁️ 显示预览...")
                original = cv2.imread(args.input)
                bird_eye = cv2.imread(output_path)
                
                # 缩放以适应屏幕
                max_width = 800
                if original.shape[1] > max_width:
                    scale = max_width / original.shape[1]
                    new_width = int(original.shape[1] * scale)
                    new_height = int(original.shape[0] * scale)
                    original = cv2.resize(original, (new_width, new_height))
                
                if bird_eye.shape[1] > max_width:
                    scale = max_width / bird_eye.shape[1]
                    new_width = int(bird_eye.shape[1] * scale)
                    new_height = int(bird_eye.shape[0] * scale)
                    bird_eye = cv2.resize(bird_eye, (new_width, new_height))
                
                cv2.imshow("Original Image", original)
                cv2.imshow("Bird's Eye View", bird_eye)
                print("按任意键关闭预览...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            print(f"✅ 单图像处理完成: {output_path}")
            
        elif args.input_dir:
            # 批量处理
            if not args.output_dir:
                args.output_dir = args.input_dir + "_bird_eye"
            
            print(f"📁 批量处理: {args.input_dir} → {args.output_dir}")
            
            # 查找图像文件
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
                image_files.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))
            
            if not image_files:
                print(f"❌ 在目录中未找到图像文件: {args.input_dir}")
                return
            
            print(f"📊 找到 {len(image_files)} 个图像文件")
            
            # 确保输出目录存在
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            
            # 处理每个图像
            for i, image_path in enumerate(image_files, 1):
                print(f"\n[{i}/{len(image_files)}]", end=" ")
                
                # 确定输出文件名
                input_name = os.path.basename(image_path)
                name, ext = os.path.splitext(input_name)
                output_name = f"{name}_bird_eye{ext}"
                output_path = os.path.join(args.output_dir, output_name)
                
                try:
                    process_single_image(
                        image_path, transformer, output_path,
                        args.pixels_per_unit, args.margin_ratio, args.add_grid
                    )
                except Exception as e:
                    print(f"  ❌ 处理失败: {e}")
            
            print(f"\n✅ 批量处理完成，输出目录: {args.output_dir}")
            
        else:
            print("❌ 请指定输入图像 (--input) 或输入目录 (--input_dir)")
            
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
