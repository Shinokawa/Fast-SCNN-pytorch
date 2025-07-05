#!/usr/bin/env python3
"""
标定模块 - 透视变换标定参数管理

基于用户提供的A4纸标定点进行透视变换参数计算。
包含内置标定参数和校正后的标定参数。
"""

import numpy as np
import cv2

# ---------------------------------------------------------------------------------
# --- 🔧 内置标定参数 (基于用户提供的标定点) ---
# ---------------------------------------------------------------------------------

def get_builtin_calibration():
    """
    获取内置的标定参数 (基于用户标定的A4纸)
    
    标定信息：
    - 图像尺寸: 640×360
    - 标记物: A4纸 (21.0cm × 29.7cm)
    - 图像点: [(260, 87), (378, 87), (410, 217), (231, 221)]
    - 世界点: [(0, 0), (21, 0), (21, 29.7), (0, 29.7)]  # A4纸四个角
    """
    # 图像中的4个点 (像素坐标)
    image_points = [(260, 87), (378, 87), (410, 217), (231, 221)]
    
    # 真实世界中的对应点 (厘米) - A4纸的四个角
    world_points = [(0, 0), (21, 0), (21, 29.7), (0, 29.7)]
    
    # 计算透视变换矩阵
    src_points = np.float32(image_points)
    dst_points = np.float32(world_points)
    
    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    inverse_transform_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
    
    calibration_data = {
        'image_size': [640, 360],
        'image_points': image_points,
        'world_points': world_points,
        'transform_matrix': transform_matrix.tolist(),
        'inverse_transform_matrix': inverse_transform_matrix.tolist(),
        'description': '基于A4纸标定的透视变换参数',
        'units': 'centimeters'
    }
    
    return calibration_data

def get_corrected_calibration():
    """
    获取校正后的标定参数，确保640×360原始图像在鸟瞰图中上下边界平行
    
    核心思路：
    1. 使用A4纸的4个标定点作为参考
    2. 计算整个640×360图像的4个角点在世界坐标中的位置
    3. 强制图像的上边界和下边界在世界坐标中Y值相等（平行）
    4. 重新计算校正后的透视变换矩阵
    """
    # 获取原始标定
    original_cal = get_builtin_calibration()
    original_transform = np.array(original_cal['transform_matrix'], dtype=np.float32)
    
    # 640×360图像的四个角点（像素坐标）
    img_corners = np.array([
        [0, 0, 1],           # 左上角
        [639, 0, 1],         # 右上角  
        [639, 359, 1],       # 右下角
        [0, 359, 1]          # 左下角
    ], dtype=np.float32)
    
    # 使用原始变换将图像角点投影到世界坐标
    world_corners = []
    for corner in img_corners:
        world_pt = original_transform @ corner
        world_x = world_pt[0] / world_pt[2]
        world_y = world_pt[1] / world_pt[2]
        world_corners.append([world_x, world_y])
    
    world_corners = np.array(world_corners)
    
    # 校正世界坐标：强制上下边界平行
    # 上边界：左上角和右上角的Y坐标取平均值
    # 下边界：左下角和右下角的Y坐标取平均值
    top_y = (world_corners[0][1] + world_corners[1][1]) / 2  # 上边界Y
    bottom_y = (world_corners[2][1] + world_corners[3][1]) / 2  # 下边界Y
    
    # 保持X坐标不变，只校正Y坐标
    corrected_world_corners = [
        [world_corners[0][0], top_y],     # 左上角
        [world_corners[1][0], top_y],     # 右上角 - Y与左上角相同
        [world_corners[2][0], bottom_y],  # 右下角
        [world_corners[3][0], bottom_y]   # 左下角 - Y与右下角相同
    ]
    
    # 重新计算透视变换矩阵
    # 从640×360图像角点到校正后的世界坐标
    src_points = np.float32([[0, 0], [639, 0], [639, 359], [0, 359]])
    dst_points = np.float32(corrected_world_corners);
    
    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    inverse_transform_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
    
    # 保持原始A4纸标定点用于显示
    corrected_calibration = {
        'image_size': [640, 360],
        'image_points': original_cal['image_points'],  # 保持A4纸标定点
        'world_points': original_cal['world_points'],  # 保持A4纸世界坐标
        'transform_matrix': transform_matrix.tolist(),
        'inverse_transform_matrix': inverse_transform_matrix.tolist(),
        'corrected_world_corners': corrected_world_corners,
        'original_world_corners': world_corners.tolist(),
        'description': '校正后的透视变换参数（确保640×360图像上下边界平行）',
        'units': 'centimeters'
    }
    
    print(f"🔧 透视校正完成:")
    print(f"   - 原始上边界Y: {world_corners[0][1]:.2f} ~ {world_corners[1][1]:.2f} cm")
    print(f"   - 校正上边界Y: {top_y:.2f} cm (平行)")
    print(f"   - 原始下边界Y: {world_corners[2][1]:.2f} ~ {world_corners[3][1]:.2f} cm") 
    print(f"   - 校正下边界Y: {bottom_y:.2f} cm (平行)")
    
    return corrected_calibration 