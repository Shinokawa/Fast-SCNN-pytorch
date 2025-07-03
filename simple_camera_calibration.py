#!/usr/bin/env python3
"""
简化的相机透视变换标定工具 (命令行版本)

功能特性：
- 手动输入图像坐标点
- 手动输入真实世界对应点坐标
- 计算透视变换矩阵
- 生成俯视图预览
- 保存标定参数

使用方法：
python simple_camera_calibration.py --input kuruma/raw_collected_data/raw_images/raw_20250704_013409_372.jpg

操作说明：
1. 根据提示输入图像中4个点的坐标（像素坐标）
2. 输入每个点在真实世界中的坐标（单位：厘米）
3. 生成并保存标定参数

注意：建议选择地面上的矩形标记物（如A4纸）的四个角点进行标定
"""

import cv2
import numpy as np
import json
import argparse
import os
from pathlib import Path
import time

class SimpleCameraCalibrator:
    def __init__(self, image_path):
        """
        初始化简化相机标定工具
        
        参数：
            image_path: 输入图像路径
        """
        self.image_path = image_path
        self.image = None
        self.image_points = []  # 图像中的点
        self.world_points = []  # 真实世界中的点
        self.transform_matrix = None
        
        # 加载图像
        self.load_image()
        
    def load_image(self):
        """加载并准备图像"""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"图像文件不存在: {self.image_path}")
        
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"无法读取图像: {self.image_path}")
        
        # 获取图像尺寸
        self.height, self.width = self.image.shape[:2]
        print(f"📏 图像尺寸: {self.width} × {self.height}")
        
        # 保存一个带标记的图像供参考
        self.save_reference_image()
    
    def save_reference_image(self):
        """保存参考图像供查看"""
        reference_path = self.image_path.replace('.jpg', '_reference_for_calibration.jpg')
        reference_img = self.image.copy()
        
        # 添加一些辅助线和标记
        # 添加中心十字线
        center_x, center_y = self.width // 2, self.height // 2
        cv2.line(reference_img, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 1)
        cv2.line(reference_img, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 1)
        
        # 添加坐标标记
        for y in range(0, self.height, 50):
            for x in range(0, self.width, 50):
                cv2.circle(reference_img, (x, y), 2, (128, 128, 128), -1)
        
        # 添加坐标轴标记
        for x in range(0, self.width, 100):
            cv2.putText(reference_img, str(x), (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        for y in range(0, self.height, 100):
            cv2.putText(reference_img, str(y), (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imwrite(reference_path, reference_img)
        print(f"📸 参考图像已保存: {reference_path}")
        print("💡 请打开此图像查看坐标系，以便准确输入坐标")
    
    def input_image_coordinates(self):
        """输入图像坐标"""
        print("\n" + "="*60)
        print("📍 请输入图像坐标")
        print("="*60)
        print("📝 请按照顺序输入4个点在图像中的像素坐标")
        print("📐 坐标系：左上角为(0,0)，X轴向右，Y轴向下")
        print("📋 建议顺序：左上角 → 右上角 → 右下角 → 左下角")
        print("💡 请参考生成的reference图像来确定准确坐标")
        print("-"*60)
        
        point_names = ["左上角", "右上角", "右下角", "左下角"]
        
        for i in range(4):
            print(f"\n📍 点 {i+1} ({point_names[i]})")
            
            while True:
                try:
                    x_input = input(f"  请输入 X 坐标 (0-{self.width-1}): ")
                    y_input = input(f"  请输入 Y 坐标 (0-{self.height-1}): ")
                    
                    x_img = int(x_input)
                    y_img = int(y_input)
                    
                    # 验证坐标范围
                    if 0 <= x_img < self.width and 0 <= y_img < self.height:
                        self.image_points.append((x_img, y_img))
                        print(f"  ✅ 图像坐标: ({x_img}, {y_img})")
                        break
                    else:
                        print(f"  ❌ 坐标超出范围，请输入有效坐标")
                        
                except ValueError:
                    print("  ❌ 请输入有效的整数")
    
    def input_world_coordinates(self):
        """输入真实世界坐标"""
        print("\n" + "="*60)
        print("🌍 请输入真实世界坐标")
        print("="*60)
        print("📝 请按照图像点的顺序输入每个点在真实世界中的坐标")
        print("📏 单位：厘米 (cm)")
        print("📐 建议：以A4纸的一个角为原点(0,0)建立坐标系")
        print("📏 A4纸尺寸：21.0cm × 29.7cm")
        print("💡 示例坐标系（A4纸左上角为原点）：")
        print("    左上角: (0, 0)")
        print("    右上角: (21, 0)")
        print("    右下角: (21, 29.7)")
        print("    左下角: (0, 29.7)")
        print("-"*60)
        
        point_names = ["左上角", "右上角", "右下角", "左下角"]
        
        for i in range(4):
            print(f"\n📍 点 {i+1} ({point_names[i]}) - 图像坐标: {self.image_points[i]}")
            
            while True:
                try:
                    x_input = input(f"  请输入真实世界 X 坐标 (cm): ")
                    y_input = input(f"  请输入真实世界 Y 坐标 (cm): ")
                    
                    x_world = float(x_input)
                    y_world = float(y_input)
                    
                    self.world_points.append((x_world, y_world))
                    print(f"  ✅ 真实世界坐标: ({x_world}, {y_world}) cm")
                    break
                    
                except ValueError:
                    print("  ❌ 请输入有效的数字")
        
        print("\n📊 标定点对应关系:")
        print("-"*40)
        for i in range(4):
            img_pt = self.image_points[i]
            world_pt = self.world_points[i]
            print(f"{point_names[i]:8}: {img_pt} → {world_pt}")
    
    def calculate_transform_matrix(self):
        """计算透视变换矩阵"""
        print("\n🧮 计算透视变换矩阵...")
        
        # 转换为numpy数组
        src_points = np.float32(self.image_points)
        dst_points = np.float32(self.world_points)
        
        # 计算透视变换矩阵
        self.transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        print("✅ 透视变换矩阵计算完成!")
        print("\n📐 透视变换矩阵:")
        print(self.transform_matrix)
        
        # 计算逆变换矩阵（从世界坐标回到图像坐标）
        self.inverse_transform_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
        
        return self.transform_matrix
    
    def generate_bird_eye_preview(self):
        """生成鸟瞰图预览"""
        if self.transform_matrix is None:
            print("❌ 请先计算变换矩阵")
            return
        
        print("\n🦅 生成鸟瞰图预览...")
        
        # 计算输出图像尺寸
        world_points_array = np.array(self.world_points)
        min_x, min_y = world_points_array.min(axis=0)
        max_x, max_y = world_points_array.max(axis=0)
        
        # 添加边距
        margin = max(max_x - min_x, max_y - min_y) * 0.3
        min_x -= margin
        min_y -= margin
        max_x += margin
        max_y += margin
        
        # 计算像素/单位比例
        pixels_per_unit = 20  # 每厘米20像素
        
        output_width = int((max_x - min_x) * pixels_per_unit)
        output_height = int((max_y - min_y) * pixels_per_unit)
        
        print(f"📏 鸟瞰图尺寸: {output_width} × {output_height}")
        print(f"📐 范围: X({min_x:.1f} ~ {max_x:.1f}) cm, Y({min_y:.1f} ~ {max_y:.1f}) cm")
        
        # 创建变换矩阵（从图像到鸟瞰图像素坐标）
        world_to_pixel = np.array([
            [pixels_per_unit, 0, -min_x * pixels_per_unit],
            [0, pixels_per_unit, -min_y * pixels_per_unit],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 组合变换矩阵
        combined_transform = world_to_pixel @ self.transform_matrix
        
        # 执行透视变换
        bird_eye_view = cv2.warpPerspective(self.image, combined_transform, 
                                          (output_width, output_height))
        
        # 在鸟瞰图上标记标定点
        for i, (x_world, y_world) in enumerate(self.world_points):
            pixel_x = int((x_world - min_x) * pixels_per_unit)
            pixel_y = int((y_world - min_y) * pixels_per_unit)
            
            cv2.circle(bird_eye_view, (pixel_x, pixel_y), 8, (0, 255, 255), -1)
            cv2.putText(bird_eye_view, str(i+1), (pixel_x + 12, pixel_y - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 添加网格
        grid_interval = 10  # 每10cm一个网格
        grid_color = (128, 128, 128)
        
        # 垂直线
        x = min_x
        while x <= max_x:
            if abs(x % grid_interval) < 0.1:
                pixel_x = int((x - min_x) * pixels_per_unit)
                if 0 <= pixel_x < output_width:
                    cv2.line(bird_eye_view, (pixel_x, 0), (pixel_x, output_height-1), grid_color, 1)
            x += grid_interval
        
        # 水平线
        y = min_y
        while y <= max_y:
            if abs(y % grid_interval) < 0.1:
                pixel_y = int((y - min_y) * pixels_per_unit)
                if 0 <= pixel_y < output_height:
                    cv2.line(bird_eye_view, (0, pixel_y), (output_width-1, pixel_y), grid_color, 1)
            y += grid_interval
        
        # 保存预览图
        preview_path = self.image_path.replace('.jpg', '_bird_eye_preview.jpg')
        cv2.imwrite(preview_path, bird_eye_view)
        print(f"💾 鸟瞰图预览已保存: {preview_path}")
        
        return bird_eye_view, (min_x, min_y, max_x, max_y, pixels_per_unit)
    
    def save_calibration_data(self, output_path=None):
        """保存标定数据"""
        if output_path is None:
            output_path = self.image_path.replace('.jpg', '_calibration.json')
        
        calibration_data = {
            'image_path': self.image_path,
            'image_size': [self.width, self.height],
            'image_points': self.image_points,
            'world_points': self.world_points,
            'transform_matrix': self.transform_matrix.tolist(),
            'inverse_transform_matrix': self.inverse_transform_matrix.tolist(),
            'calibration_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'description': '相机透视变换标定数据 (简化版)',
            'units': 'centimeters'
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 标定数据已保存: {output_path}")
        return output_path
    
    def run_calibration(self):
        """运行完整的标定流程"""
        print("🎯 简化相机透视变换标定工具")
        print("="*50)
        print("📋 标定流程:")
        print("1. 输入图像中4个点的像素坐标")
        print("2. 输入对应的真实世界坐标（建议使用A4纸）")
        print("3. 计算透视变换矩阵")
        print("4. 生成鸟瞰图预览")
        print("5. 保存标定参数")
        print("="*50)
        
        try:
            # 输入图像坐标
            self.input_image_coordinates()
            
            # 输入真实世界坐标
            self.input_world_coordinates()
            
            # 计算变换矩阵
            self.calculate_transform_matrix()
            
            # 生成鸟瞰图预览
            bird_eye_view, view_params = self.generate_bird_eye_preview()
            
            # 询问是否保存
            while True:
                save_choice = input("\n💾 是否保存标定数据？ (y/n): ").lower()
                if save_choice in ['y', 'yes', '是']:
                    calibration_file = self.save_calibration_data()
                    break
                elif save_choice in ['n', 'no', '否']:
                    print("❌ 标定数据未保存")
                    calibration_file = None
                    break
                else:
                    print("请输入 y 或 n")
            
            # 显示最终结果
            print("\n" + "="*60)
            print("🎉 标定完成!")
            print("="*60)
            print("📊 标定结果摘要:")
            print(f"🖼️  输入图像: {self.image_path}")
            print(f"📏 图像尺寸: {self.width} × {self.height}")
            print(f"📍 标定点数: {len(self.image_points)}")
            
            if calibration_file:
                print(f"💾 标定文件: {calibration_file}")
                
                print("\n🔧 使用示例:")
                print("# 使用标定数据进行推理")
                print(f'python onnx_bird_eye_inference.py --input "你的图像.jpg" --calibration "{calibration_file}" --bird_eye --save_control_map')
            
            return {
                'transform_matrix': self.transform_matrix,
                'inverse_transform_matrix': self.inverse_transform_matrix,
                'image_points': self.image_points,
                'world_points': self.world_points,
                'calibration_file': calibration_file,
                'view_params': view_params
            }
            
        except KeyboardInterrupt:
            print("\n❌ 用户中断标定")
            return None
        except Exception as e:
            print(f"\n❌ 标定过程出错: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    parser = argparse.ArgumentParser(description="简化相机透视变换标定工具")
    parser.add_argument("--input", "-i", required=True, help="输入图像路径")
    parser.add_argument("--output", "-o", help="输出标定文件路径（可选）")
    
    args = parser.parse_args()
    
    try:
        # 创建标定工具
        calibrator = SimpleCameraCalibrator(args.input)
        
        # 运行标定
        result = calibrator.run_calibration()
        
        if result:
            print("\n✅ 标定成功完成!")
            
            # 如果指定了输出路径，复制文件
            if args.output and result['calibration_file']:
                import shutil
                shutil.copy2(result['calibration_file'], args.output)
                print(f"📄 标定文件已复制到: {args.output}")
        else:
            print("❌ 标定被取消或失败")
            
    except Exception as e:
        print(f"❌ 标定过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
