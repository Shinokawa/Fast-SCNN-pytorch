#!/usr/bin/env python3
"""
透视变换模块 - 鸟瞰图生成和透视变换

包含：
- PerspectiveTransformer: 透视变换器类，用于生成鸟瞰图
- 透视变换参数计算和图像变换功能
"""

import numpy as np
import cv2
import logging
from core.logging_config import get_module_logger

# 导入标定模块
from core.calibration import get_corrected_calibration, get_builtin_calibration

# ---------------------------------------------------------------------------------
# --- 🦅 透视变换模块 (鸟瞰图生成) ---
# ---------------------------------------------------------------------------------

logger = get_module_logger(__name__)

class PerspectiveTransformer:
    """透视变换器，用于生成鸟瞰图"""
    
    def __init__(self, calibration_data=None, use_corrected=True):
        """
        初始化透视变换器
        
        参数：
            calibration_data: 标定数据字典，如果为None则使用内置标定
            use_corrected: 是否使用校正后的标定（确保矩形鸟瞰图）
        """
        if calibration_data is None:
            if use_corrected:
                calibration_data = get_corrected_calibration()
                logger.info("✅ 使用校正后的标定参数（确保矩形鸟瞰图）")
            else:
                calibration_data = get_builtin_calibration()
                logger.warning("⚠️ 使用原始标定参数（可能产生梯形）")
        
        self.calibration_data = calibration_data
        self.transform_matrix = np.array(calibration_data['transform_matrix'], dtype=np.float32)
        self.inverse_transform_matrix = np.array(calibration_data['inverse_transform_matrix'], dtype=np.float32)
        self.image_points = calibration_data['image_points']
        self.world_points = calibration_data['world_points']
        self.original_image_size = calibration_data['image_size']
        
        logger.info("✅ 透视变换器已初始化")
        logger.info(f"📏 标定图像尺寸: {self.original_image_size[0]} × {self.original_image_size[1]}")
    
    def calculate_bird_eye_params(self, pixels_per_unit=20, margin_ratio=0.1, full_image=True):
        """
        计算鸟瞰图参数
        
        参数：
            pixels_per_unit: 每单位的像素数
            margin_ratio: 边距比例
            full_image: 是否显示完整图像的鸟瞰图
        """
        if full_image:
            # 将整个图像的四个角投影到世界坐标
            img_width, img_height = self.original_image_size
            
            # 图像的四个角点
            image_corners = np.array([
                [0, 0, 1],           # 左上角
                [img_width-1, 0, 1], # 右上角
                [img_width-1, img_height-1, 1], # 右下角
                [0, img_height-1, 1] # 左下角
            ], dtype=np.float32)
            
            # 投影到世界坐标
            world_corners = []
            for corner in image_corners:
                # 应用透视变换
                world_pt = self.transform_matrix @ corner
                world_x = world_pt[0] / world_pt[2]
                world_y = world_pt[1] / world_pt[2]
                world_corners.append([world_x, world_y])
            
            world_corners = np.array(world_corners)
            
            # 计算世界坐标范围
            min_x, min_y = world_corners.min(axis=0)
            max_x, max_y = world_corners.max(axis=0)
            
            # 添加边距
            range_x = max_x - min_x
            range_y = max_y - min_y
            margin_x = range_x * margin_ratio
            margin_y = range_y * margin_ratio
            
            min_x -= margin_x
            min_y -= margin_y
            max_x += margin_x
            max_y += margin_y
            
        else:
            # 原来的方法：只基于标定点范围
            world_points_array = np.array(self.world_points)
            min_x, min_y = world_points_array.min(axis=0)
            max_x, max_y = world_points_array.max(axis=0)
            
            # 添加边距，扩展视野
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
    
    def transform_image_and_mask(self, image, mask, pixels_per_unit=20, margin_ratio=0.1, full_image=True):
        """
        将图像和分割掩码都转换为鸟瞰图
        
        参数：
            image: 输入图像 (BGR格式)
            mask: 分割掩码 (0/255)
            pixels_per_unit: 每单位的像素数
            margin_ratio: 边距比例
            full_image: 是否显示完整图像的鸟瞰图
        
        返回：
            bird_eye_image: 鸟瞰图
            bird_eye_mask: 鸟瞰图分割掩码
            view_params: 视图参数字典
        """
        # 计算鸟瞰图参数（显示完整图像）
        output_width, output_height, combined_transform, view_bounds = \
            self.calculate_bird_eye_params(pixels_per_unit, margin_ratio, full_image)
        
        # 如果输入图像尺寸与标定时不同，需要调整变换矩阵
        input_height, input_width = image.shape[:2]
        orig_width, orig_height = self.original_image_size
        
        if input_width != orig_width or input_height != orig_height:
            logger.warning(f"⚠️ 图像尺寸不匹配: {input_width}×{input_height} vs {orig_width}×{orig_height}")
            logger.info("🔄 自动调整变换矩阵...")
            
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
        
        # 执行透视变换 - 图像
        bird_eye_image = cv2.warpPerspective(
            image, combined_transform, 
            (output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # 执行透视变换 - 分割掩码
        bird_eye_mask = cv2.warpPerspective(
            mask, combined_transform, 
            (output_width, output_height),
            flags=cv2.INTER_NEAREST,  # 使用最近邻插值保持掩码的二值性
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # 准备视图参数
        view_params = {
            'output_size': (output_width, output_height),
            'view_bounds': view_bounds,
            'pixels_per_unit': pixels_per_unit,
            'margin_ratio': margin_ratio,
            'transform_matrix': combined_transform.tolist(),
            'image_to_world_matrix': self.transform_matrix.tolist()  # 存储正确的"图像->世界"矩阵
        }
        
        return bird_eye_image, bird_eye_mask, view_params 