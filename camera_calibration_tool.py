#!/usr/bin/env python3
"""
相机透视变换标定工具

功能特性：
- 交互式选择图像中的4个点
- 手动输入真实世界对应点坐标
- 计算透视变换矩阵
- 生成俯视图预览
- 保存标定参数

使用方法：
python camera_calibration_tool.py --input kuruma/raw_collected_data/raw_images/raw_20250704_013409_372.jpg

操作说明：
1. 在图像上点击4个点（按顺序：左上、右上、右下、左下）
2. 输入每个点在真实世界中的坐标（单位：厘米或米）
3. 预览透视变换结果
4. 保存标定参数到文件

注意：建议选择地面上的矩形标记物（如A4纸）的四个角点进行标定
"""

import cv2
import numpy as np
import json
import argparse
import os
from pathlib import Path
import time

class CameraCalibrationTool:
    def __init__(self, image_path):
        """
        初始化相机标定工具
        
        参数：
            image_path: 输入图像路径
        """
        self.image_path = image_path
        self.image = None
        self.display_image = None
        self.image_points = []  # 图像中的点
        self.world_points = []  # 真实世界中的点
        self.transform_matrix = None
        self.window_name = "Camera Calibration Tool"
        
        # 加载图像
        self.load_image()
        
        # 设置显示参数
        self.point_color = (0, 0, 255)  # 红色
        self.line_color = (0, 255, 0)   # 绿色
        self.point_radius = 8
        self.line_thickness = 2
        
    def load_image(self):
        """加载并准备图像"""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"图像文件不存在: {self.image_path}")
        
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"无法读取图像: {self.image_path}")
        
        # 创建显示图像的副本
        self.display_image = self.image.copy()
        
        # 获取图像尺寸
        self.height, self.width = self.image.shape[:2]
        print(f"📏 图像尺寸: {self.width} × {self.height}")
        
        # 如果图像太大，缩放以适应屏幕
        max_display_width = 1200
        max_display_height = 800
        
        if self.width > max_display_width or self.height > max_display_height:
            scale_w = max_display_width / self.width
            scale_h = max_display_height / self.height
            self.display_scale = min(scale_w, scale_h)
            
            display_width = int(self.width * self.display_scale)
            display_height = int(self.height * self.display_scale)
            
            self.display_image = cv2.resize(self.display_image, (display_width, display_height))
            print(f"📺 显示尺寸: {display_width} × {display_height} (缩放比例: {self.display_scale:.2f})")
        else:
            self.display_scale = 1.0
            print("📺 显示尺寸: 原始尺寸")
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标点击回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.image_points) < 4:
                # 将显示坐标转换回原始图像坐标
                actual_x = int(x / self.display_scale)
                actual_y = int(y / self.display_scale)
                
                self.image_points.append((actual_x, actual_y))
                
                # 在显示图像上绘制点
                display_x = int(actual_x * self.display_scale)
                display_y = int(actual_y * self.display_scale)
                
                cv2.circle(self.display_image, (display_x, display_y), 
                          self.point_radius, self.point_color, -1)
                
                # 添加点序号标签
                point_num = len(self.image_points)
                cv2.putText(self.display_image, str(point_num), 
                           (display_x + 15, display_y - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.point_color, 2)
                
                print(f"✅ 已选择点 {point_num}: ({actual_x}, {actual_y})")
                
                # 如果选择了4个点，绘制连线
                if len(self.image_points) == 4:
                    self.draw_polygon()
                    print("🎯 已选择4个点，请按任意键继续输入真实世界坐标...")
                
                cv2.imshow(self.window_name, self.display_image)
    
    def draw_polygon(self):
        """绘制选中的4个点构成的多边形"""
        if len(self.image_points) >= 4:
            # 转换为显示坐标
            display_points = []
            for x, y in self.image_points[:4]:
                display_x = int(x * self.display_scale)
                display_y = int(y * self.display_scale)
                display_points.append((display_x, display_y))
            
            # 绘制多边形
            pts = np.array(display_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(self.display_image, [pts], True, self.line_color, self.line_thickness)
    
    def input_world_coordinates(self):
        """输入真实世界坐标"""
        print("\n" + "="*60)
        print("🌍 请输入真实世界坐标")
        print("="*60)
        print("📝 请按照选择点的顺序输入每个点在真实世界中的坐标")
        print("📏 单位可以是厘米(cm)或米(m)，请保持一致")
        print("📐 建议使用矩形标记物（如A4纸：21cm × 29.7cm）")
        print("-"*60)
        
        point_names = ["左上角", "右上角", "右下角", "左下角"]
        
        for i in range(4):
            print(f"\n📍 点 {i+1} ({point_names[i]}) - 图像坐标: {self.image_points[i]}")
            
            while True:
                try:
                    x_input = input(f"  请输入 X 坐标 (横向): ")
                    y_input = input(f"  请输入 Y 坐标 (纵向): ")
                    
                    x_world = float(x_input)
                    y_world = float(y_input)
                    
                    self.world_points.append((x_world, y_world))
                    print(f"  ✅ 真实世界坐标: ({x_world}, {y_world})")
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
    
    def preview_bird_eye_view(self):
        """预览鸟瞰图效果"""
        if self.transform_matrix is None:
            print("❌ 请先计算变换矩阵")
            return
        
        print("\n🦅 生成鸟瞰图预览...")
        
        # 计算输出图像尺寸
        world_points_array = np.array(self.world_points)
        min_x, min_y = world_points_array.min(axis=0)
        max_x, max_y = world_points_array.max(axis=0)
        
        # 添加边距
        margin = max(max_x - min_x, max_y - min_y) * 0.2
        min_x -= margin
        min_y -= margin
        max_x += margin
        max_y += margin
        
        # 计算像素/单位比例
        pixels_per_unit = 20  # 每单位20像素
        
        output_width = int((max_x - min_x) * pixels_per_unit)
        output_height = int((max_y - min_y) * pixels_per_unit)
        
        print(f"📏 鸟瞰图尺寸: {output_width} × {output_height}")
        print(f"📐 范围: X({min_x:.1f} ~ {max_x:.1f}), Y({min_y:.1f} ~ {max_y:.1f})")
        
        # 创建变换矩阵（从图像到鸟瞰图像素坐标）
        # 需要先变换到世界坐标，再变换到像素坐标
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
            
            cv2.circle(bird_eye_view, (pixel_x, pixel_y), 5, (0, 255, 255), -1)
            cv2.putText(bird_eye_view, str(i+1), (pixel_x + 10, pixel_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 显示鸟瞰图
        cv2.imshow("Bird's Eye View Preview", bird_eye_view)
        
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
            'description': '相机透视变换标定数据'
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 标定数据已保存: {output_path}")
        return output_path
    
    def run_calibration(self):
        """运行完整的标定流程"""
        print("🎯 相机透视变换标定工具")
        print("="*50)
        print("📋 操作说明:")
        print("1. 在图像上依次点击4个点（建议选择矩形标记物的四个角）")
        print("2. 点击顺序：左上 → 右上 → 右下 → 左下")
        print("3. 输入每个点在真实世界中的坐标")
        print("4. 预览鸟瞰图效果")
        print("5. 保存标定参数")
        print("-"*50)
        print("💡 提示：建议使用A4纸等矩形标记物进行标定")
        print("💡 A4纸尺寸：21.0cm × 29.7cm")
        print("💡 按ESC键退出，按R键重新开始")
        print("="*50)
        
        # 设置鼠标回调
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # 显示图像
        cv2.imshow(self.window_name, self.display_image)
        
        # 等待用户选择4个点
        while len(self.image_points) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键
                print("❌ 用户取消标定")
                cv2.destroyAllWindows()
                return None
            elif key == ord('r') or key == ord('R'):  # R键重新开始
                print("🔄 重新开始标定...")
                self.image_points = []
                self.world_points = []
                self.display_image = self.image.copy()
                if self.display_scale != 1.0:
                    display_width = int(self.width * self.display_scale)
                    display_height = int(self.height * self.display_scale)
                    self.display_image = cv2.resize(self.display_image, (display_width, display_height))
                cv2.imshow(self.window_name, self.display_image)
        
        # 等待用户按键继续
        cv2.waitKey(0)
        cv2.destroyWindow(self.window_name)
        
        # 输入真实世界坐标
        self.input_world_coordinates()
        
        # 计算变换矩阵
        self.calculate_transform_matrix()
        
        # 预览鸟瞰图
        bird_eye_view, view_params = self.preview_bird_eye_view()
        
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
        
        print("\n🔧 用法示例:")
        print("# 加载标定数据")
        print("import json")
        print("import cv2")
        print("import numpy as np")
        print(f"with open('{calibration_file}', 'r') as f:")
        print("    calib = json.load(f)")
        print("transform_matrix = np.array(calib['transform_matrix'])")
        print("")
        print("# 将图像转换为鸟瞰图")
        print("bird_eye = cv2.warpPerspective(image, combined_transform, (width, height))")
        
        # 等待用户查看预览
        print("\n按任意键关闭预览窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return {
            'transform_matrix': self.transform_matrix,
            'inverse_transform_matrix': self.inverse_transform_matrix,
            'image_points': self.image_points,
            'world_points': self.world_points,
            'calibration_file': calibration_file,
            'view_params': view_params
        }

def main():
    parser = argparse.ArgumentParser(description="相机透视变换标定工具")
    parser.add_argument("--input", "-i", required=True, help="输入图像路径")
    parser.add_argument("--output", "-o", help="输出标定文件路径（可选）")
    
    args = parser.parse_args()
    
    try:
        # 创建标定工具
        calibrator = CameraCalibrationTool(args.input)
        
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
