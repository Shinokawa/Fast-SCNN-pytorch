#!/usr/bin/env python3
"""
集成感知与环境建模的车道线分割推理脚本 - Atlas版本

功能特性：
- 使用Atlas NPU (华为昇腾) 进行车道线分割推理
- 支持透视变换，生成鸟瞰图
- 可视化可驾驶区域的鸟瞰图
- 为路径规划提供2D地图数据
- 🚗 集成视觉横向误差的比例-速度自适应差速控制算法
- 内置标定参数，即开即用
- 🚀 实时处理性能优化，支持日志输出

使用方法：
# 基础推理（仅分割）
python kuruma_control_dashboard.py --input image.jpg --output result.jpg

# 添加透视变换生成鸟瞰图
python kuruma_control_dashboard.py --input image.jpg --output result.jpg --bird_eye

# 生成控制用的鸟瞰图和路径规划
python kuruma_control_dashboard.py --input image.jpg --bird_eye --save_control_map

# 启用完整的视觉控制算法
python kuruma_control_dashboard.py --input image.jpg --bird_eye --save_control_map --enable_control

# 实时摄像头模式（推荐）
python kuruma_control_dashboard.py --realtime --log_file realtime_control.log

作者：基于原版扩展，集成Atlas NPU推理和实时控制
"""

import os
import sys
import cv2
import time
import numpy as np
import argparse
import json
import logging
from pathlib import Path
from threading import Thread, Lock
import queue
import base64
import io
from datetime import datetime

# Web界面相关导入
try:
    from flask import Flask, render_template_string, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️ Flask未安装，Web界面功能不可用")

# 导入scipy用于路径平滑
try:
    from scipy.optimize import curve_fit
    from scipy.interpolate import UnivariateSpline
    SCIPY_AVAILABLE = True
except ImportError:
    print("⚠️ 警告：未找到scipy库，路径平滑功能将不可用")
    print("安装命令: pip install scipy")
    SCIPY_AVAILABLE = False

# 导入Atlas推理库
try:
    from ais_bench.infer.interface import InferSession
    ATLAS_AVAILABLE = True
    print("✅ Atlas推理库加载成功")
except ImportError:
    print("❌ 错误：未找到ais_bench库，请安装Atlas推理环境")
    print("安装命令: pip install ais_bench")
    sys.exit(1)

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

# ---------------------------------------------------------------------------------
# --- 🚀🚀🚀 完美匹配的预处理 (640×360 = 640×360，与Atlas完全一致) 🚀🚀🚀 ---
# ---------------------------------------------------------------------------------

def preprocess_matched_resolution(img_bgr, target_width=640, target_height=360, dtype=np.float32):
    """
    图片预处理，与atlas_single_image_inference.py和lane_dashboard_e2e.py完全一致
    
    输入：BGR图像 (任意尺寸)
    输出：Float32/Float16 NCHW张量 (1, 3, 360, 640)
    
    处理流程：
    1. 如果输入尺寸不是640×360，先resize到640×360
    2. BGR → RGB
    3. uint8 → float32/float16 (保持[0-255]范围)
    4. HWC → CHW，添加batch维度
    """
    # 1. 调整尺寸到模型输入要求
    height, width = img_bgr.shape[:2]
    if width != target_width or height != target_height:
        print(f"📏 Resize: {width}×{height} → {target_width}×{target_height}")
        img_bgr = cv2.resize(img_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    else:
        print(f"🎯 完美匹配: {width}×{height} = {target_width}×{target_height}，无需resize!")
    
    # 2. 转换颜色通道 (BGR → RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 3. 转换数据类型 (uint8 → float16，保持[0-255]范围)
    img_typed = img_rgb.astype(dtype)
    
    # 4. 转换为CHW格式并添加batch维度 (H,W,C) → (1,C,H,W)
    img_transposed = np.transpose(img_typed, (2, 0, 1))
    return np.ascontiguousarray(img_transposed[np.newaxis, :, :, :])

# ---------------------------------------------------------------------------------
# --- 🚀🚀🚀 极简后处理 (尺寸完美匹配，与Atlas完全一致) 🚀🚀🚀 ---
# ---------------------------------------------------------------------------------

def postprocess_matched_resolution(output_tensor, original_width, original_height):
    """
    后处理，与atlas_single_image_inference.py和lane_dashboard_e2e.py完全一致
    
    输入：模型输出张量 (1, num_classes, 360, 640)
    输出：分割掩码 (original_height, original_width)
    
    处理流程：
    1. Argmax获取分割掩码
    2. 转换为可视化格式
    3. 如需要，resize回原始尺寸
    """
    # 1. Argmax获取分割掩码 (1, num_classes, H, W) → (H, W)
    pred_mask = np.argmax(output_tensor, axis=1).squeeze()
    
    # 2. 转换为可视化格式 (0/1 → 0/255)
    vis_mask = (pred_mask * 255).astype(np.uint8)
    
    # 3. 如果需要，resize回原始尺寸
    model_height, model_width = vis_mask.shape
    if original_width != model_width or original_height != model_height:
        print(f"📐 Resize back: {model_width}×{model_height} → {original_width}×{original_height}")
        vis_mask = cv2.resize(vis_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    else:
        print(f"🎯 输出尺寸匹配: {model_width}×{model_height} = {original_width}×{original_height}")
    
    return vis_mask

# ---------------------------------------------------------------------------------
# --- 🎨 可视化生成 (与Atlas完全一致) ---
# ---------------------------------------------------------------------------------

def create_visualization(original_img, mask, alpha=0.5):
    """
    创建车道线分割可视化图像，与atlas_single_image_inference.py完全一致
    
    参数：
        original_img: 原始BGR图像
        mask: 分割掩码 (0/255)
        alpha: 透明度
    
    返回：
        可视化图像 (BGR格式)
    """
    # 创建绿色覆盖层
    green_overlay = np.zeros_like(original_img, dtype=np.uint8)
    green_overlay[mask > 0] = [0, 255, 0]  # BGR格式的绿色
    
    # 融合原图和覆盖层
    vis_img = cv2.addWeighted(original_img, 1.0, green_overlay, alpha, 0)
    
    return vis_img

# ---------------------------------------------------------------------------------
# --- 🧠 Atlas NPU推理会话 ---
# ---------------------------------------------------------------------------------

class AtlasInferSession:
    """Atlas NPU推理会话，完全兼容原有接口"""
    
    def __init__(self, device_id, model_path):
        """
        初始化Atlas推理会话
        
        参数：
            device_id: NPU设备ID (通常为0)
            model_path: OM模型路径
        """
        self.device_id = device_id
        self.model_path = model_path
        
        print(f"🧠 使用Atlas NPU设备: {device_id}")
        print(f"📊 加载OM模型: {model_path}")
        
        # 创建推理会话
        self.session = InferSession(device_id, model_path)
        
        print(f"✅ Atlas推理会话初始化完成")
    
    def infer(self, inputs):
        """
        执行推理，与原有ONNX接口完全一致
        
        参数：
            inputs: 输入张量列表
        
        返回：
            outputs: 输出张量列表
        """
        input_tensor = inputs[0]
        
        # 执行Atlas推理
        outputs = self.session.infer([input_tensor])
        
        return outputs

# ---------------------------------------------------------------------------------
# --- 📊 性能分析 (与Atlas完全一致) ---
# ---------------------------------------------------------------------------------

def print_performance_analysis(times_dict, input_tensor, model_path, device_info):
    """打印详细的性能分析报告"""
    total_time = sum(times_dict.values())
    
    print("\n" + "="*60)
    print("🧠 Atlas NPU + 透视变换 性能分析")
    print("="*60)
    print(f"🧠 模型: {Path(model_path).name}")
    print(f"⚡ 推理设备: {device_info}")
    print(f"📏 输入尺寸: {input_tensor.shape[3]}×{input_tensor.shape[2]} (W×H)")
    print(f"🎯 数据类型: {str(input_tensor.dtype).upper()}")
    print("-"*60)
    
    for stage, time_ms in times_dict.items():
        percentage = (time_ms / total_time) * 100
        print(f"⏱️  {stage:15}: {time_ms:6.1f}ms ({percentage:5.1f}%)")
    
    print("-"*60)
    print(f"🏁 总耗时: {total_time:.1f}ms")
    print(f"⚡ 理论FPS: {1000/total_time:.1f}")
    print("="*60)

# ---------------------------------------------------------------------------------
# --- 📱 主推理函数 (与Atlas流程完全一致) ---
# ---------------------------------------------------------------------------------

def inference_single_image(image_path, model_path, device_id=0, 
                          save_visualization=True, save_mask=False, 
                          bird_eye=False, save_control_map=False,
                          pixels_per_unit=20, margin_ratio=0.1, full_image_bird_eye=False,
                          path_smooth_method='polynomial', path_degree=3, 
                          num_waypoints=20, min_road_width=10, edge_computing=False,
                          force_bottom_center=True, enable_control=False, 
                          steering_gain=1.0, base_speed=10.0, curvature_damping=0.1, 
                          preview_distance=30.0, max_speed=20.0, min_speed=5.0):
    """
    集成车道线分割推理和透视变换的完整感知管道 - Atlas版本
    
    参数：
        image_path: 输入图片路径
        model_path: OM模型路径
        device_id: Atlas NPU设备ID
        save_visualization: 是否保存普通可视化结果
        save_mask: 是否保存分割掩码
        bird_eye: 是否生成鸟瞰图
        save_control_map: 是否保存控制地图
        pixels_per_unit: 每单位像素数
        margin_ratio: 边距比例
        full_image_bird_eye: 是否生成完整原图的鸟瞰图（否则仅A4纸区域）
        path_smooth_method: 路径平滑方法
        path_degree: 路径拟合阶数
        num_waypoints: 路径点数量
        min_road_width: 最小可行驶宽度
        edge_computing: 边缘计算模式（极致性能优化）
        force_bottom_center: 强制拟合曲线过底边中点
    
    返回：
        dict: 包含结果路径和性能数据
    """
    # 验证输入文件
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"输入图片不存在: {image_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"🖼️  加载图片: {image_path}")
    
    # 1. 加载图片
    load_start = time.time()
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    original_height, original_width = img_bgr.shape[:2]
    load_time = (time.time() - load_start) * 1000
    
    print(f"📏 原始尺寸: {original_width}×{original_height}")
    
    # 2. 加载Atlas模型
    print(f"🧠 加载Atlas OM模型: {model_path}")
    model_start = time.time()
    model = AtlasInferSession(device_id, model_path)
    model_load_time = (time.time() - model_start) * 1000
    print(f"✅ 模型加载完成 ({model_load_time:.1f}ms)")
    
    # 3. 预处理（使用float16以匹配Atlas模型）
    print("🔄 开始预处理...")
    preprocess_start = time.time()
    input_data = preprocess_matched_resolution(img_bgr, dtype=np.float16)
    preprocess_time = (time.time() - preprocess_start) * 1000
    
    print(f"📊 输入张量形状: {input_data.shape}")
    print(f"📊 数据类型: {input_data.dtype}")
    
    # 4. Atlas NPU推理
    print("🚀 开始Atlas NPU推理...")
    inference_start = time.time()
    outputs = model.infer([input_data])
    inference_time = (time.time() - inference_start) * 1000
    
    print(f"📊 输出张量形状: {outputs[0].shape}")
    
    # 5. 后处理
    print("🔄 开始后处理...")
    postprocess_start = time.time()
    lane_mask = postprocess_matched_resolution(outputs[0], original_width, original_height)
    postprocess_time = (time.time() - postprocess_start) * 1000
    
    # 6. 透视变换（可选）
    transform_time = 0
    path_planning_time = 0
    bird_eye_image = None
    bird_eye_mask = None
    control_map = None
    view_params = None
    
    if bird_eye:
        print("🦅 开始透视变换...")
        transform_start = time.time()
        
        # 边缘计算优化：大幅降低像素密度
        if edge_computing:
            if full_image_bird_eye:
                # 边缘计算+完整图像：超低像素密度
                adjusted_pixels_per_unit = 1  # 固定1像素/单位，减少400倍计算量
                print(f"⚡ 边缘计算极致优化：像素密度 {pixels_per_unit} → {adjusted_pixels_per_unit} 像素/单位")
            else:
                # 边缘计算+A4区域：低像素密度
                adjusted_pixels_per_unit = 2  # 固定2像素/单位
                print(f"⚡ 边缘计算优化：像素密度 {pixels_per_unit} → {adjusted_pixels_per_unit} 像素/单位")
        else:
            if full_image_bird_eye:
                # 完整图像模式：极低像素密度（边缘计算友好）
                adjusted_pixels_per_unit = max(1, pixels_per_unit // 20)  # 最低1像素/单位，减少400倍计算量
                print(f"🚀 边缘计算优化：像素密度 {pixels_per_unit} → {adjusted_pixels_per_unit} 像素/单位（减少{pixels_per_unit//adjusted_pixels_per_unit}倍计算量）")
            else:
                # A4纸区域模式：中等优化
                adjusted_pixels_per_unit = max(2, pixels_per_unit // 4)  # 最低2像素/单位
                print(f"🔧 性能优化：像素密度 {pixels_per_unit} → {adjusted_pixels_per_unit} 像素/单位")
        
        transformer = PerspectiveTransformer()  # 使用内置标定参数
        bird_eye_image, bird_eye_mask, view_params = transformer.transform_image_and_mask(
            img_bgr, lane_mask, adjusted_pixels_per_unit, margin_ratio, full_image=full_image_bird_eye)
        
        transform_time = (time.time() - transform_start) * 1000
        
        # 6.5. 路径规划（可选）
        path_planning_time = 0
        if save_control_map:
            print("🛣️ 开始路径规划...")
            path_start = time.time()
            control_map, path_data = create_control_map(
                bird_eye_mask, view_params, add_grid=True, add_path=True,
                path_smooth_method=path_smooth_method,
                path_degree=path_degree,
                num_waypoints=num_waypoints,
                min_road_width=min_road_width,
                edge_computing=edge_computing,
                force_bottom_center=force_bottom_center
            )
            path_planning_time = (time.time() - path_start) * 1000
        else:
            path_data = None
        
        print(f"📐 鸟瞰图尺寸: {view_params['output_size'][0]}×{view_params['output_size'][1]}")
        bounds = view_params['view_bounds']
        print(f"📐 世界坐标范围: X({bounds[0]:.1f}~{bounds[2]:.1f}), Y({bounds[1]:.1f}~{bounds[3]:.1f}) cm")
    
    # 7. 保存结果
    save_start = time.time()
    results = {}
    
    # 确保output目录存在
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成基础文件名（不含路径和扩展名）
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 保存普通分割掩码
    if save_mask:
        mask_path = os.path.join(output_dir, f"{base_name}_onnx_mask.jpg")
        cv2.imwrite(mask_path, lane_mask)
        results['mask_path'] = mask_path
        print(f"💾 分割掩码已保存: {mask_path}")
    
    # 保存普通可视化结果
    if save_visualization:
        vis_img = create_visualization(img_bgr, lane_mask)
        vis_path = os.path.join(output_dir, f"{base_name}_onnx_result.jpg")
        cv2.imwrite(vis_path, vis_img)
        results['visualization_path'] = vis_path
        print(f"💾 可视化结果已保存: {vis_path}")
    
    # 保存鸟瞰图
    if bird_eye and bird_eye_image is not None:
        bird_eye_path = os.path.join(output_dir, f"{base_name}_bird_eye.jpg")
        cv2.imwrite(bird_eye_path, bird_eye_image)
        results['bird_eye_path'] = bird_eye_path
        print(f"💾 鸟瞰图已保存: {bird_eye_path}")
        
        # 保存带分割结果的鸟瞰图
        bird_eye_vis = create_visualization(bird_eye_image, bird_eye_mask)
        bird_eye_vis_path = os.path.join(output_dir, f"{base_name}_bird_eye_segmented.jpg")
        cv2.imwrite(bird_eye_vis_path, bird_eye_vis)
        results['bird_eye_vis_path'] = bird_eye_vis_path
        print(f"💾 鸟瞰图分割可视化已保存: {bird_eye_vis_path}")
    
    # 保存控制地图
    if save_control_map and control_map is not None:
        control_map_path = os.path.join(output_dir, f"{base_name}_control_map.jpg")
        cv2.imwrite(control_map_path, control_map)
        results['control_map_path'] = control_map_path
        print(f"💾 控制地图已保存: {control_map_path}")
        
        # 保存路径数据为JSON
        if path_data is not None:
            path_json_path = os.path.join(output_dir, f"{base_name}_path_data.json")
            save_path_data_json(path_data, path_json_path)
            results['path_json_path'] = path_json_path
            print(f"💾 路径数据已保存: {path_json_path}")
    
    # 7.5. 视觉控制算法（可选）
    control_result = None
    if enable_control and path_data is not None and view_params is not None:
        print("🚗 启动视觉横向误差控制算法...")
        control_start = time.time()
        
        # 初始化控制器
        controller = VisualLateralErrorController(
            steering_gain=steering_gain,
            base_pwm=base_speed,  # 重命名参数映射
            curvature_damping=curvature_damping,
            preview_distance=preview_distance,
            max_pwm=max_speed,    # 重命名参数映射
            min_pwm=min_speed     # 重命名参数映射
        )
        
        # 计算控制指令
        control_result = controller.compute_wheel_pwm(path_data, view_params)
        
        # 生成控制可视化地图
        if control_map is not None:
            control_vis_map = controller.generate_control_visualization(
                control_map, control_result, view_params)
            control_vis_path = os.path.join(output_dir, f"{base_name}_control_visualization.jpg")
            cv2.imwrite(control_vis_path, control_vis_map)
            results['control_vis_path'] = control_vis_path
            print(f"💾 控制可视化地图已保存: {control_vis_path}")
        
        # 保存控制数据
        control_json_path = os.path.join(output_dir, f"{base_name}_control_data.json")
        controller.save_control_data(control_result, control_json_path)
        results['control_json_path'] = control_json_path
        print(f"💾 控制数据已保存: {control_json_path}")
        
        # 打印控制分析
        controller.print_control_analysis(control_result)
        
        control_time = (time.time() - control_start) * 1000
    else:
        control_time = 0
    
    save_time = (time.time() - save_start) * 1000
    
    # 8. 性能分析
    times_dict = {
        "图片加载": load_time,
        "模型加载": model_load_time,
        "CPU预处理": preprocess_time,
        "Atlas推理": inference_time,
        "CPU后处理": postprocess_time,
        "透视变换": transform_time,
        "路径规划": path_planning_time,
        "控制计算": control_time,
        "结果保存": save_time
    }
    
    print_performance_analysis(times_dict, input_data, model_path, f"Atlas NPU {device_id}")
    
    # 9. 统计车道线像素
    lane_pixels = np.sum(lane_mask > 0)
    total_pixels = lane_mask.shape[0] * lane_mask.shape[1]
    lane_ratio = (lane_pixels / total_pixels) * 100
    
    print(f"\n📈 检测结果统计:")
    print(f"🛣️  车道线像素: {lane_pixels:,} / {total_pixels:,} ({lane_ratio:.2f}%)")
    
    if bird_eye_mask is not None:
        bird_lane_pixels = np.sum(bird_eye_mask > 0)
        bird_total_pixels = bird_eye_mask.shape[0] * bird_eye_mask.shape[1]
        bird_lane_ratio = (bird_lane_pixels / bird_total_pixels) * 100
        print(f"🦅 鸟瞰图车道线像素: {bird_lane_pixels:,} / {bird_total_pixels:,} ({bird_lane_ratio:.2f}%)")
    
    results.update({
        'lane_pixels': lane_pixels,
        'total_pixels': total_pixels,
        'lane_ratio': lane_ratio,
        'performance': times_dict,
        'device': f"Atlas NPU {device_id}",
        'view_params': view_params,
        'control_result': control_result
    })
    
    return results

# ---------------------------------------------------------------------------------
# --- 🦅 透视变换模块 (鸟瞰图生成) ---
# ---------------------------------------------------------------------------------

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
                print("✅ 使用校正后的标定参数（确保矩形鸟瞰图）")
            else:
                calibration_data = get_builtin_calibration()
                print("⚠️ 使用原始标定参数（可能产生梯形）")
        
        self.calibration_data = calibration_data
        self.transform_matrix = np.array(calibration_data['transform_matrix'], dtype=np.float32)
        self.inverse_transform_matrix = np.array(calibration_data['inverse_transform_matrix'], dtype=np.float32)
        self.image_points = calibration_data['image_points']
        self.world_points = calibration_data['world_points']
        self.original_image_size = calibration_data['image_size']
        
        print(f"✅ 透视变换器已初始化")
        print(f"📏 标定图像尺寸: {self.original_image_size[0]} × {self.original_image_size[1]}")
    
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

# ---------------------------------------------------------------------------------
# --- 🗺️ 控制地图生成模块 ---
# ---------------------------------------------------------------------------------

def create_control_map(bird_eye_mask, view_params, add_grid=True, add_path=True,
                      path_smooth_method='polynomial', path_degree=3, 
                      num_waypoints=20, min_road_width=10, edge_computing=False,
                      force_bottom_center=True):
    """
    创建用于路径规划的控制地图
    
    参数：
        bird_eye_mask: 鸟瞰图分割掩码
        view_params: 视图参数
        add_grid: 是否添加网格
        add_path: 是否添加路径规划
        path_smooth_method: 路径平滑方法
        path_degree: 路径拟合阶数
        num_waypoints: 路径点数量
        min_road_width: 最小可行驶宽度
        edge_computing: 边缘计算模式
        force_bottom_center: 强制拟合曲线过底边中点
    
    返回：
        control_map: 控制地图 (三通道BGR图像)
        path_data: 路径规划数据（如果add_path=True）
    """
    # 创建控制地图
    control_map = np.zeros((bird_eye_mask.shape[0], bird_eye_mask.shape[1], 3), dtype=np.uint8)
    
    # 可驾驶区域 - 绿色
    control_map[bird_eye_mask > 0] = [0, 255, 0]  # BGR绿色
    
    # 不可驾驶区域 - 保持黑色
    # control_map[bird_eye_mask == 0] = [0, 0, 0]  # 已经是黑色
    
    # 路径规划
    path_data = None
    if add_path:
        try:
            planner = PathPlanner(view_params)
            path_data = planner.plan_complete_path(
                bird_eye_mask, 
                smooth_method=path_smooth_method,
                degree=path_degree,
                num_waypoints=num_waypoints,
                min_width=min_road_width,
                fast_mode=edge_computing,
                force_bottom_center=force_bottom_center
            )
            
            # 在控制地图上可视化路径
            control_map = visualize_path_on_control_map(control_map, path_data, view_params)
            
            print(f"🛣️ 路径规划完成:")
            print(f"   - 中心线点数: {path_data['num_centerline_points']}")
            print(f"   - 路径点数: {path_data['num_waypoints']}")
            print(f"   - 路径长度: {path_data['path_length']:.1f} cm")
            
        except Exception as e:
            print(f"⚠️ 路径规划失败: {e}")
            path_data = None
    
    if add_grid:
        control_map = add_grid_to_control_map(control_map, view_params)
    
    return control_map, path_data

def add_grid_to_control_map(control_map, view_params):
    """
    在控制地图上添加网格和坐标标签
    
    参数：
        control_map: 控制地图
        view_params: 视图参数
    
    返回：
        带网格的控制地图
    """
    annotated_map = control_map.copy()
    
    min_x, min_y, max_x, max_y = view_params['view_bounds']
    pixels_per_unit = view_params['pixels_per_unit']
    output_width, output_height = view_params['output_size']
    
    # 绘制网格
    grid_interval = 10  # 网格间隔（单位：cm）
    grid_color = (128, 128, 128)  # 灰色
    origin_color = (0, 0, 255)    # 红色原点
    
    # 垂直线
    x = min_x
    while x <= max_x:
        if abs(x % grid_interval) < 0.1:  # 处理浮点数精度问题
            pixel_x = int((x - min_x) * pixels_per_unit)
            if 0 <= pixel_x < output_width:
                cv2.line(annotated_map, (pixel_x, 0), (pixel_x, output_height-1), grid_color, 1)
                
                # 添加X坐标标签
                if abs(x) > 0.1:  # 避免在原点重复标注
                    label = f"{int(x)}"
                    cv2.putText(annotated_map, label, (pixel_x + 2, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_color, 1)
        x += grid_interval / 2
    
    # 水平线
    y = min_y
    while y <= max_y:
        if abs(y % grid_interval) < 0.1:  # 处理浮点数精度问题
            pixel_y = int((y - min_y) * pixels_per_unit)
            if 0 <= pixel_y < output_height:
                cv2.line(annotated_map, (0, pixel_y), (output_width-1, pixel_y), grid_color, 1)
                
                # 添加Y坐标标签
                if abs(y) > 0.1:  # 避免在原点重复标注
                    label = f"{int(y)}"
                    cv2.putText(annotated_map, label, (5, pixel_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_color, 1)
        y += grid_interval / 2
    
    # 绘制原点
    origin_x = int((0 - min_x) * pixels_per_unit)
    origin_y = int((0 - min_y) * pixels_per_unit)
    
    if 0 <= origin_x < output_width and 0 <= origin_y < output_height:
        cv2.circle(annotated_map, (origin_x, origin_y), 5, origin_color, -1)
        cv2.putText(annotated_map, "O(0,0)", (origin_x + 8, origin_y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, origin_color, 1)
    
    # 标记A4纸的四个角
    for i, (world_x, world_y) in enumerate([(0, 0), (21, 0), (21, 29.7), (0, 29.7)]):
        pixel_x = int((world_x - min_x) * pixels_per_unit)
        pixel_y = int((world_y - min_y) * pixels_per_unit)
        
        if 0 <= pixel_x < output_width and 0 <= pixel_y < output_height:
            cv2.circle(annotated_map, (pixel_x, pixel_y), 3, (0, 255, 255), -1)
            cv2.putText(annotated_map, f"A4-{i+1}", (pixel_x + 5, pixel_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    
    return annotated_map

# ---------------------------------------------------------------------------------
# --- 🛣️ 路径规划模块 ---
# ---------------------------------------------------------------------------------

class PathPlanner:
    """从鸟瞰图分割掩码中提取和规划可行驶路径"""
    
    def __init__(self, view_params):
        """
        初始化路径规划器
        
        参数：
            view_params: 鸟瞰图视图参数
        """
        self.view_params = view_params
        self.pixels_per_unit = view_params['pixels_per_unit']
        self.view_bounds = view_params['view_bounds']
        
    def extract_centerline(self, bird_eye_mask, scan_from_bottom=True, min_width=10):
        """
        从鸟瞰图分割掩码中提取中心线
        
        参数：
            bird_eye_mask: 鸟瞰图分割掩码
            scan_from_bottom: 是否从图像底部开始扫描
            min_width: 最小可行驶宽度（像素）
        
        返回：
            centerline_points: 中心线点列表 [(x, y), ...]（像素坐标）
            centerline_world: 中心线点列表（世界坐标厘米）
        """
        height, width = bird_eye_mask.shape
        centerline_points = []
        
        # 确定扫描方向
        rows = range(height-1, -1, -1) if scan_from_bottom else range(height)
        
        for y in rows:
            row = bird_eye_mask[y, :]
            
            # 找到该行所有可行驶区域的连续段
            segments = self._find_drivable_segments(row, min_width)
            
            if segments:
                # 选择最大的连续段（通常是主路）
                largest_segment = max(segments, key=lambda s: s[1] - s[0])
                
                # 计算该段的中心点
                center_x = (largest_segment[0] + largest_segment[1]) // 2
                centerline_points.append((center_x, y))
        
        # 转换为世界坐标
        centerline_world = self._pixels_to_world(centerline_points)
        
        return centerline_points, centerline_world
    
    def extract_centerline_fast(self, bird_eye_mask, scan_from_bottom=True, min_width=5, skip_rows=5):
        """
        快速中心线提取（边缘计算优化版本）
        
        参数：
            bird_eye_mask: 鸟瞰图分割掩码
            scan_from_bottom: 是否从图像底部开始扫描
            min_width: 最小可行驶宽度（像素）
            skip_rows: 跳过行数（减少计算量）
        
        返回：
            centerline_points: 中心线点列表 [(x, y), ...]（像素坐标）
            centerline_world: 中心线点列表（世界坐标厘米）
        """
        height, width = bird_eye_mask.shape
        centerline_points = []
        
        # 确定扫描方向，跳行扫描以提高速度
        if scan_from_bottom:
            rows = range(height-1, -1, -skip_rows)
        else:
            rows = range(0, height, skip_rows)
        
        for y in rows:
            row = bird_eye_mask[y, :]
            
            # 快速找到中心点：使用重心法
            drivable_indices = np.where(row > 0)[0]
            
            if len(drivable_indices) >= min_width:
                # 计算重心作为中心点
                center_x = int(np.mean(drivable_indices))
                centerline_points.append((center_x, y))
        
        # 转换为世界坐标
        centerline_world = self._pixels_to_world(centerline_points)
        
        return centerline_points, centerline_world
    
    def _find_drivable_segments(self, row, min_width):
        """
        在一行中找到所有可行驶区域的连续段
        
        参数：
            row: 图像行数据
            min_width: 最小宽度
        
        返回：
            segments: 连续段列表 [(start, end), ...]
        """
        segments = []
        start = None
        
        for i, pixel in enumerate(row):
            if pixel > 0:  # 可行驶区域
                if start is None:
                    start = i
            else:  # 不可行驶区域
                if start is not None:
                    if i - start >= min_width:  # 满足最小宽度要求
                        segments.append((start, i))
                    start = None
        
        # 处理行末尾的情况
        if start is not None and len(row) - start >= min_width:
            segments.append((start, len(row)))
        
        return segments
    
    def _pixels_to_world(self, pixel_points):
        """
        将像素坐标转换为世界坐标
        
        参数：
            pixel_points: 像素坐标点列表 [(x, y), ...]
        
        返回：
            world_points: 世界坐标点列表 [(x, y), ...]（单位：厘米）
        """
        min_x, min_y, max_x, max_y = self.view_bounds
        world_points = []
        
        for px, py in pixel_points:
            # 像素坐标转世界坐标
            world_x = min_x + (px / self.pixels_per_unit)
            world_y = min_y + (py / self.pixels_per_unit)
            world_points.append((world_x, world_y))
        
        return world_points
    
    def smooth_path(self, centerline_world, method='polynomial', degree=3, force_bottom_center=True):
        """
        对中心线路径进行平滑处理 (已修正为拟合 x=f(y) 并使用权重)
        
        参数：
            centerline_world: 世界坐标中心线点列表
            method: 平滑方法 ('polynomial', 'spline')
            degree: 多项式阶数或样条阶数
            force_bottom_center: 是否强制曲线过底边中点
        
        返回：
            smooth_path_func: 平滑路径函数 x = f(y)
            fit_params: 拟合参数
        """
        if not centerline_world or not SCIPY_AVAILABLE:
            return None, None
        
        points = np.array(centerline_world)
        # 核心修正1: 我们将Y作为自变量，X作为因变量
        y_coords = points[:, 1]  # 前进方向
        x_coords = points[:, 0]  # 左右偏移

        # 按Y坐标（前进方向）排序
        sorted_indices = np.argsort(y_coords)
        y_sorted = y_coords[sorted_indices]
        x_sorted = x_coords[sorted_indices]
        
        # 用于存储最终拟合点
        final_y = y_sorted
        final_x = x_sorted
        weights = np.ones_like(final_y) # 默认权重为1
        
        # 如果需要强制过底边中点
        if force_bottom_center:
            bottom_center = self._get_bottom_center_world_coord()
            
            if bottom_center is not None:
                # 将底边中点添加到拟合点中
                # 注意：bottom_center是 (x, y) 格式
                final_y = np.append(final_y, bottom_center[1])
                final_x = np.append(final_x, bottom_center[0])
                
                # 核心修正2: 为这个点设置一个极大的权重
                weights = np.append(weights, 1e6) # 给新点一个巨大的权重
                
                # 重新排序
                sorted_indices = np.argsort(final_y)
                final_y = final_y[sorted_indices]
                final_x = final_x[sorted_indices]
                weights = weights[sorted_indices]
                
                print(f"🎯 强制拟合曲线过底边中点: ({bottom_center[0]:.1f}, {bottom_center[1]:.1f}) cm，权重: {1e6}")

        # 确保点数足够拟合
        if len(final_y) <= degree:
            print(f"⚠️ 拟合点数 ({len(final_y)}) 不足，无法进行 {degree} 阶拟合。")
            return None, None

        if method == 'polynomial':
            # 核心修正3: 拟合 x = f(y)，并传入权重
            fit_params = np.polyfit(final_y, final_x, degree, w=weights)
            smooth_path_func = np.poly1d(fit_params)
            
        elif method == 'spline':
            # 样条插值默认会穿过所有点，但这里为了统一，也使用多项式
            # 如果需要样条，也需要拟合 x=f(y)
            print("⚠️ 样条方法暂不支持权重，强制使用多项式拟合以确保过中点。")
            fit_params = np.polyfit(final_y, final_x, degree, w=weights)
            smooth_path_func = np.poly1d(fit_params)
        
        return smooth_path_func, fit_params
    
    def _get_bottom_center_world_coord(self):
        """
        获取图像底边中点的世界坐标
        
        返回：
            bottom_center: (x, y) 底边中点的世界坐标，单位厘米
        """
        try:
            # 使用正确的"图像坐标->世界坐标"变换矩阵
            if 'image_to_world_matrix' in self.view_params:
                transform_matrix = np.array(self.view_params['image_to_world_matrix'], dtype=np.float32)
            else:
                # 如果没有，作为回退，从校正配置中获取
                print("⚠️ 在view_params中未找到image_to_world_matrix，尝试从内置校准获取。")
                transform_matrix = get_corrected_calibration()
            
            # 640×360图像底边中点的像素坐标
            img_bottom_center = np.array([320, 359, 1], dtype=np.float32)  # (320, 359) 是底边中点
            
            # 投影到世界坐标
            world_pt = transform_matrix @ img_bottom_center
            world_x = world_pt[0] / world_pt[2]
            world_y = world_pt[1] / world_pt[2]
            
            return (world_x, world_y)
            
        except Exception as e:
            print(f"⚠️ 无法计算底边中点世界坐标: {e}")
            return None
    
    def generate_waypoints(self, smooth_path_func, num_points=20, y_range=None):
        """
        从平滑路径生成路径点 (已修正为基于 y 轴生成)
        
        参数：
            smooth_path_func: 平滑路径函数 x = f(y)
            num_points: 路径点数量
            y_range: Y坐标范围 (min_y, max_y)，如果为None则使用视图边界
        
        返回：
            waypoints: 路径点列表 [(x, y), ...]（世界坐标，厘米）
        """
        if smooth_path_func is None:
            return []
        
        # 核心修正: 我们应该在y轴（前进方向）上取点
        if y_range is None:
            min_x, min_y, max_x, max_y = self.view_bounds
        else:
            min_y, max_y = y_range

        # 生成均匀分布的y坐标
        y_waypoints = np.linspace(min_y, max_y, num_points)
        
        # 计算对应的x坐标
        x_waypoints = smooth_path_func(y_waypoints)
        
        # 组合成路径点 (x, y)
        waypoints = list(zip(x_waypoints, y_waypoints))
        
        return waypoints
    
    def plan_complete_path(self, bird_eye_mask, smooth_method='polynomial', degree=3, 
                          num_waypoints=20, min_width=10, fast_mode=True, force_bottom_center=True):
        """
        完整的路径规划流程
        
        参数：
            bird_eye_mask: 鸟瞰图分割掩码
            smooth_method: 平滑方法
            degree: 拟合阶数
            num_waypoints: 路径点数量
            min_width: 最小可行驶宽度
            fast_mode: 是否使用快速模式（边缘计算优化）
            force_bottom_center: 是否强制曲线过底边中点
        
        返回：
            path_data: 包含所有路径信息的字典
        """
        # 第一步：提取中心线（选择快速或精确模式）
        if fast_mode:
            centerline_pixels, centerline_world = self.extract_centerline_fast(
                bird_eye_mask, min_width=min_width//2, skip_rows=3)  # 降低要求，跳行扫描
        else:
            centerline_pixels, centerline_world = self.extract_centerline(
                bird_eye_mask, min_width=min_width)
        
        if not centerline_world:
            return {
                'centerline_pixels': [],
                'centerline_world': [],
                'smooth_path_func': None,
                'fit_params': None,
                'waypoints': [],
                'path_length': 0
            }
        
        # 第二步：路径平滑（边缘计算模式下降低阶数，强制过底边中点）
        if fast_mode:
            smooth_degree = min(2, degree)  # 最高2阶，减少计算量
        else:
            smooth_degree = degree
            
        smooth_path_func, fit_params = self.smooth_path(
            centerline_world, method=smooth_method, degree=smooth_degree, 
            force_bottom_center=force_bottom_center)
        
        # 第三步：生成路径点
        waypoints = self.generate_waypoints(smooth_path_func, num_waypoints)
        
        # 计算路径长度
        path_length = self._calculate_path_length(waypoints) if waypoints else 0
        
        return {
            'centerline_pixels': centerline_pixels,
            'centerline_world': centerline_world,
            'smooth_path_func': smooth_path_func,
            'fit_params': fit_params,
            'waypoints': waypoints,
            'path_length': path_length,
            'num_centerline_points': len(centerline_world),
            'num_waypoints': len(waypoints),
            'fast_mode': fast_mode,
            'force_bottom_center': force_bottom_center
        }
    
    def _calculate_path_length(self, waypoints):
        """计算路径总长度"""
        if len(waypoints) < 2:
            return 0
        
        total_length = 0
        for i in range(1, len(waypoints)):
            dx = waypoints[i][0] - waypoints[i-1][0]
            dy = waypoints[i][1] - waypoints[i-1][1]
            total_length += np.sqrt(dx*dx + dy*dy)
        
        return total_length

# ---------------------------------------------------------------------------------
# --- 🚗 基于视觉横向误差的比例-速度自适应差速控制算法 ---
# ---------------------------------------------------------------------------------

class VisualLateralErrorController:
    """
    基于视觉横向误差的比例-速度自适应差速控制算法
    (Proportional-Speed-Adaptive Differential Drive Control based on Visual Lateral Error)
    
    算法概述：
    本算法是一种专为纯视觉、差速转向机器人设计的开环路径跟踪控制器。
    通过鸟瞰图实时计算机器人与规划路径之间的横向偏差，利用比例控制器
    将此偏差直接转换为左右驱动轮的速度差，同时引入速度自适应机制。
    """
    
    def __init__(self, steering_gain=50.0, base_pwm=300, curvature_damping=0.1, 
                 preview_distance=30.0, max_pwm=800, min_pwm=100):
        """
        初始化控制器参数
        
        参数：
            steering_gain: 转向增益 Kp（比例控制器增益）
            base_pwm: 基础PWM值（-1000到+1000范围）
            curvature_damping: 曲率阻尼系数（速度自适应参数）
            preview_distance: 预瞄距离（cm，控制点距离机器人的距离）
            max_pwm: 最大PWM值（-1000到+1000范围）
            min_pwm: 最小PWM值（-1000到+1000范围，用于前进时的最低速度）
        """
        self.steering_gain = steering_gain
        self.base_pwm = base_pwm
        self.curvature_damping = curvature_damping
        self.preview_distance = preview_distance
        self.max_pwm = max_pwm
        self.min_pwm = min_pwm
        
        # 性能统计
        self.control_history = []
        
        print(f"🚗 视觉横向误差控制器已初始化:")
        print(f"   📐 转向增益: {steering_gain}")
        print(f"   🏃 基础PWM: {base_pwm} (-1000~+1000)")
        print(f"   🌊 曲率阻尼: {curvature_damping}")
        print(f"   👁️ 预瞄距离: {preview_distance} cm")
        print(f"   ⚡ PWM范围: {min_pwm} ~ {max_pwm} (支持双向旋转)")
    
    def calculate_lateral_error(self, path_data, view_params):
        """
        模块一：视觉误差感知 (Visual Error Perception)
        
        从路径数据中计算横向误差
        
        参数：
            path_data: 路径规划数据
            view_params: 视图参数
            
        返回：
            lateral_error: 横向误差（cm）
            car_position: 机器人当前位置（世界坐标）
            control_point: 控制点位置（世界坐标）
        """
        # 1. 定义机器人当前位置（图像底部中心点）
        car_position = self._get_car_position_world(view_params)
        
        # 2. 在路径上找到预瞄控制点
        control_point = self._find_preview_point(path_data, car_position)
        
        if control_point is None:
            return 0.0, car_position, None
        
        # 3. 计算横向误差（控制点X坐标 - 机器人X坐标）
        lateral_error = control_point[0] - car_position[0]
        
        return lateral_error, car_position, control_point
    
    def calculate_steering_adjustment(self, lateral_error):
        """
        模块二：比例转向控制 (Proportional Steering Control)
        
        参数：
            lateral_error: 横向误差（cm）
            
        返回：
            steering_adjustment: 转向调整量（PWM单位）
        """
        # 比例控制律: Steering_Adjustment = STEERING_GAIN * Lateral_Error
        steering_adjustment = self.steering_gain * lateral_error
        
        return steering_adjustment
    
    def calculate_dynamic_pwm(self, lateral_error):
        """
        模块三：动态速度自适应 (Dynamic Speed Adaptation)
        
        参数：
            lateral_error: 横向误差（cm）
            
        返回：
            dynamic_pwm: 自适应调整后的PWM值（0-1000）
        """
        # 动态PWM控制律: Dynamic_PWM = BASE_PWM / (1 + CURVATURE_DAMPING * |Lateral_Error|)
        dynamic_pwm = self.base_pwm / (1 + self.curvature_damping * abs(lateral_error))
        
        # 限制在允许的PWM范围内
        dynamic_pwm = np.clip(dynamic_pwm, self.min_pwm, self.max_pwm)
        
        return dynamic_pwm
    
    def compute_wheel_pwm(self, path_data, view_params):
        """
        完整的控制计算流程 - 输出PWM控制值
        
        参数：
            path_data: 路径规划数据
            view_params: 视图参数
            
        返回：
            control_result: 控制结果字典
        """
        # 模块一：计算横向误差
        lateral_error, car_position, control_point = self.calculate_lateral_error(path_data, view_params)
        
        # 模块二：计算转向调整
        steering_adjustment = self.calculate_steering_adjustment(lateral_error)
        
        # 模块三：计算动态PWM
        dynamic_pwm = self.calculate_dynamic_pwm(lateral_error)
        
        # 最终指令合成 - 修正差速转向逻辑
        # 当lateral_error < 0时需要左转，应该右轮快左轮慢
        # 当lateral_error > 0时需要右转，应该左轮快右轮慢
        pwm_right = dynamic_pwm - steering_adjustment  # 右轮PWM
        pwm_left = dynamic_pwm + steering_adjustment   # 左轮PWM
        
        # 限制PWM值在-1000到+1000范围内（支持双向旋转）
        pwm_right = np.clip(pwm_right, -1000, 1000)
        pwm_left = np.clip(pwm_left, -1000, 1000)
        
        # 构建控制结果
        control_result = {
            'lateral_error': lateral_error,
            'car_position': car_position,
            'control_point': control_point,
            'steering_adjustment': steering_adjustment,
            'dynamic_pwm': dynamic_pwm,
            'pwm_right': pwm_right,
            'pwm_left': pwm_left,
            'turn_direction': 'left' if lateral_error < 0 else 'right' if lateral_error > 0 else 'straight',
            'curvature_level': abs(lateral_error) / self.preview_distance,  # 曲率水平指示
            'pwm_reduction_factor': self.base_pwm / dynamic_pwm if dynamic_pwm > 0 else 1.0,
            # 兼容性字段（保持原有接口）
            'dynamic_speed': dynamic_pwm,  # 映射到PWM
            'speed_right': pwm_right,      # 映射到PWM
            'speed_left': pwm_left,        # 映射到PWM
            'speed_reduction_factor': self.base_pwm / dynamic_pwm if dynamic_pwm > 0 else 1.0
        }
        
        # 记录控制历史
        self.control_history.append(control_result.copy())
        
        return control_result
    
    def _get_car_position_world(self, view_params):
        """
        获取机器人在世界坐标系中的当前位置（图像底部中心点）
        
        参数：
            view_params: 视图参数
            
        返回：
            car_position: (x, y) 机器人位置的世界坐标（cm）
        """
        try:
            # 使用透视变换矩阵将图像底部中心点转换为世界坐标
            if 'image_to_world_matrix' in view_params:
                transform_matrix = np.array(view_params['image_to_world_matrix'], dtype=np.float32)
            else:
                # 回退到内置校准
                transform_matrix = get_corrected_calibration()
            
            # 640×360图像底部中心点的像素坐标
            img_bottom_center = np.array([320, 359, 1], dtype=np.float32)
            
            # 投影到世界坐标
            world_pt = transform_matrix @ img_bottom_center
            world_x = world_pt[0] / world_pt[2]
            world_y = world_pt[1] / world_pt[2]
            
            return (world_x, world_y)
            
        except Exception as e:
            print(f"⚠️ 无法获取机器人世界坐标: {e}")
            # 使用视图边界的底部中心作为回退
            min_x, min_y, max_x, max_y = view_params['view_bounds']
            return ((min_x + max_x) / 2, max_y)
    
    def _find_preview_point(self, path_data, car_position):
        """
        在路径上找到预瞄控制点
        
        参数：
            path_data: 路径数据
            car_position: 机器人当前位置
            
        返回：
            control_point: 控制点坐标 (x, y)，如果找不到则返回None
        """
        waypoints = path_data.get('waypoints', [])
        if not waypoints:
            return None
        
        car_x, car_y = car_position
        
        # 找到距离机器人预瞄距离最近的路径点
        best_point = None
        min_distance_diff = float('inf')
        
        for waypoint in waypoints:
            wx, wy = waypoint
            
            # 计算该点到机器人的距离
            distance = np.sqrt((wx - car_x)**2 + (wy - car_y)**2)
            
            # 找到最接近预瞄距离的点（优先选择前方的点）
            if wy < car_y:  # 只考虑前方的点（Y值更小）
                distance_diff = abs(distance - self.preview_distance)
                if distance_diff < min_distance_diff:
                    min_distance_diff = distance_diff
                    best_point = waypoint
        
        # 如果没找到前方的点，选择最前方的点
        if best_point is None and waypoints:
            best_point = min(waypoints, key=lambda p: p[1])  # Y值最小的点
        
        return best_point
    
    def generate_control_visualization(self, control_map, control_result, view_params):
        """
        在控制地图上可视化控制算法的分析结果
        
        参数：
            control_map: 原始控制地图
            control_result: 控制计算结果
            view_params: 视图参数
            
        返回：
            annotated_map: 带控制信息标注的地图
        """
        annotated_map = control_map.copy()
        
        if control_result['car_position'] is None:
            return annotated_map
        
        # 转换世界坐标到像素坐标
        car_pos_pixel = self._world_to_pixel(control_result['car_position'], view_params)
        
        # 绘制机器人位置（绿色圆圈）
        cv2.circle(annotated_map, (int(car_pos_pixel[0]), int(car_pos_pixel[1])), 
                  8, (0, 255, 0), 3)
        cv2.putText(annotated_map, "CAR", 
                   (int(car_pos_pixel[0]) + 10, int(car_pos_pixel[1]) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制控制点（紫色圆圈）
        if control_result['control_point'] is not None:
            control_pos_pixel = self._world_to_pixel(control_result['control_point'], view_params)
            cv2.circle(annotated_map, (int(control_pos_pixel[0]), int(control_pos_pixel[1])), 
                      6, (255, 0, 255), 3)
            cv2.putText(annotated_map, "TARGET", 
                       (int(control_pos_pixel[0]) + 10, int(control_pos_pixel[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
            # 绘制横向误差线（红色虚线）
            cv2.line(annotated_map, 
                    (int(car_pos_pixel[0]), int(car_pos_pixel[1])),
                    (int(control_pos_pixel[0]), int(car_pos_pixel[1])),  # 水平线显示横向误差
                    (0, 0, 255), 2)
        
        # 添加控制信息文本
        self._add_control_info_text(annotated_map, control_result)
        
        return annotated_map
    
    def _world_to_pixel(self, world_point, view_params):
        """将世界坐标转换为像素坐标"""
        min_x, min_y, max_x, max_y = view_params['view_bounds']
        pixels_per_unit = view_params['pixels_per_unit']
        
        pixel_x = (world_point[0] - min_x) * pixels_per_unit
        pixel_y = (world_point[1] - min_y) * pixels_per_unit
        
        return (pixel_x, pixel_y)
    
    def _add_control_info_text(self, image, control_result):
        """在图像上添加控制信息文本"""
        text_lines = [
            f"Lateral Error: {control_result['lateral_error']:.1f} cm",
            f"Direction: {control_result['turn_direction'].upper()}",
            f"Dynamic PWM: {control_result['dynamic_pwm']:.0f}",
            f"Left PWM: {control_result['pwm_left']:.0f}",
            f"Right PWM: {control_result['pwm_right']:.0f}",
            f"Curvature: {control_result['curvature_level']:.3f}",
            f"PWM Reduction: {control_result['pwm_reduction_factor']:.2f}x"
        ]
        
        # 在图像左上角添加控制信息
        y_offset = 20
        for line in text_lines:
            cv2.putText(image, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 18
    
    def print_control_analysis(self, control_result):
        """打印详细的控制分析结果"""
        print("\n" + "="*60)
        print("🚗 基于视觉横向误差的差速控制分析")
        print("="*60)
        
        # 基础信息
        print(f"📍 机器人位置: ({control_result['car_position'][0]:.1f}, {control_result['car_position'][1]:.1f}) cm")
        if control_result['control_point']:
            print(f"🎯 控制点位置: ({control_result['control_point'][0]:.1f}, {control_result['control_point'][1]:.1f}) cm")
        
        # 模块一：视觉误差感知
        print(f"\n📱 模块一：视觉误差感知")
        print(f"   横向误差: {control_result['lateral_error']:+.1f} cm")
        print(f"   转向方向: {control_result['turn_direction'].upper()}")
        print(f"   误差强度: {'高' if abs(control_result['lateral_error']) > 10 else '中' if abs(control_result['lateral_error']) > 5 else '低'}")
        
        # 模块二：比例转向控制
        print(f"\n🎮 模块二：比例转向控制")
        print(f"   转向调整: {control_result['steering_adjustment']:+.0f} PWM")
        print(f"   控制增益: {self.steering_gain}")
        
        # 模块三：动态PWM自适应
        print(f"\n⚡ 模块三：动态PWM自适应")
        print(f"   基础PWM: {self.base_pwm:.0f}")
        print(f"   动态PWM: {control_result['dynamic_pwm']:.0f}")
        print(f"   PWM衰减: {control_result['pwm_reduction_factor']:.2f}x")
        print(f"   曲率水平: {control_result['curvature_level']:.3f}")
        
        # 最终控制指令
        print(f"\n🛞 最终差速PWM控制指令")
        print(f"   左轮PWM: {control_result['pwm_left']:+.0f}")
        print(f"   右轮PWM: {control_result['pwm_right']:+.0f}")
        print(f"   PWM差值: {abs(control_result['pwm_right'] - control_result['pwm_left']):.0f}")
        print(f"   可直接发送给底层驱动！")
        
        # 性能建议
        self._print_performance_recommendations(control_result)
    
    def _print_performance_recommendations(self, control_result):
        """打印性能建议"""
        print(f"\n💡 性能分析与建议")
        
        error_abs = abs(control_result['lateral_error'])
        if error_abs > 15:
            print("   ⚠️ 横向误差较大，建议检查路径规划质量")
        elif error_abs < 2:
            print("   ✅ 横向误差很小，路径跟踪良好")
        else:
            print("   👍 横向误差在合理范围内")
        
        if control_result['curvature_level'] > 0.3:
            print("   🌊 进入高曲率路段，自动减速生效")
        elif control_result['curvature_level'] < 0.1:
            print("   🛣️ 直线路段，保持较高速度")
        
        speed_diff = abs(control_result['speed_right'] - control_result['speed_left'])
        if speed_diff > 10:
            print("   🔄 大幅转向指令，注意机器人稳定性")
        elif speed_diff < 2:
            print("   ➡️ 直行为主，转向调整轻微")

    def save_control_data(self, control_result, json_path):
        """
        保存控制数据到JSON文件
        
        参数：
            control_result: 控制计算结果
            json_path: JSON文件路径
        """
        # 递归转换所有numpy类型
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # 准备可序列化的数据
        serializable_control_result = convert_to_serializable(control_result)
        serializable_history = convert_to_serializable(
            self.control_history[-10:] if len(self.control_history) > 10 else self.control_history
        )
        
        control_data = {
            'algorithm_name': '基于视觉横向误差的比例-速度自适应差速控制算法',
            'algorithm_description': 'Proportional-Speed-Adaptive Differential Drive Control based on Visual Lateral Error',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {
                'steering_gain': float(self.steering_gain),
                'base_pwm': float(self.base_pwm),
                'curvature_damping': float(self.curvature_damping),
                'preview_distance': float(self.preview_distance),
                'max_pwm': float(self.max_pwm),
                'min_pwm': float(self.min_pwm)
            },
            'current_control': serializable_control_result,
            'control_history': serializable_history,
            'units': {
                'position': 'cm',
                'pwm': '-1000~+1000 (bidirectional)',
                'error': 'cm'
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(control_data, f, indent=2, ensure_ascii=False)

def visualize_path_on_control_map(control_map, path_data, view_params):
    """
    在控制地图上可视化路径规划结果
    
    参数：
        control_map: 控制地图
        path_data: 路径数据
        view_params: 视图参数
    
    返回：
        annotated_map: 带路径标注的控制地图
    """
    annotated_map = control_map.copy()
    
    if not path_data['centerline_pixels']:
        return annotated_map
    
    # 绘制原始中心线点（红色小圆点）
    for px, py in path_data['centerline_pixels']:
        cv2.circle(annotated_map, (int(px), int(py)), 2, (0, 0, 255), -1)
    
    # 绘制平滑路径（蓝色线条）
    if path_data['smooth_path_func'] is not None and path_data['waypoints']:
        waypoints_pixels = world_to_pixels(path_data['waypoints'], view_params)
        
        for i in range(len(waypoints_pixels) - 1):
            pt1 = (int(waypoints_pixels[i][0]), int(waypoints_pixels[i][1]))
            pt2 = (int(waypoints_pixels[i+1][0]), int(waypoints_pixels[i+1][1]))
            cv2.line(annotated_map, pt1, pt2, (255, 0, 0), 3)  # 蓝色粗线
    
    # 绘制路径点（黄色方块）
    if path_data['waypoints']:
        waypoints_pixels = world_to_pixels(path_data['waypoints'], view_params)
        for i, (px, py) in enumerate(waypoints_pixels):
            cv2.rectangle(annotated_map, 
                         (int(px-3), int(py-3)), (int(px+3), int(py+3)), 
                         (0, 255, 255), -1)  # 黄色方块
            
            # 标注路径点编号
            if i % 3 == 0:  # 每3个点标注一次，避免过于密集
                cv2.putText(annotated_map, f"{i}", (int(px+5), int(py-5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    
    return annotated_map

def world_to_pixels(world_points, view_params):
    """
    将世界坐标转换为像素坐标
    
    参数：
        world_points: 世界坐标点列表
        view_params: 视图参数
    
    返回：
        pixel_points: 像素坐标点列表
    """
    min_x, min_y, max_x, max_y = view_params['view_bounds']
    pixels_per_unit = view_params['pixels_per_unit']
    
    pixel_points = []
    for world_x, world_y in world_points:
        pixel_x = (world_x - min_x) * pixels_per_unit
        pixel_y = (world_y - min_y) * pixels_per_unit
        pixel_points.append((pixel_x, pixel_y))
    
    return pixel_points


# ---------------------------------------------------------------------------------
# --- 🚀 实时推理模块 (摄像头模式) ---
# ---------------------------------------------------------------------------------

def setup_logging(log_file=None):
    """配置日志系统"""
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

def realtime_inference(model_path, device_id=0, camera_index=0, 
                      camera_width=640, camera_height=360,
                      log_file=None, enable_control=True,
                      steering_gain=1.0, base_speed=10.0, 
                      curvature_damping=0.1, preview_distance=30.0,
                      max_speed=20.0, min_speed=5.0,
                      enable_web=False, no_gui=False, full_image_bird_eye=True):
    """
    实时摄像头推理模式
    
    参数：
        model_path: OM模型路径
        device_id: Atlas NPU设备ID
        camera_index: 摄像头索引
        camera_width: 摄像头宽度
        camera_height: 摄像头高度
        log_file: 日志文件路径
        enable_control: 是否启用控制算法
        enable_web: 是否启用Web界面数据更新
        no_gui: 是否禁用GUI显示
        其他: 控制参数
    """
    # 配置日志
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 启动实时推理系统")
    logger.info(f"📱 模型: {model_path}")
    logger.info(f"💾 日志文件: {log_file}")
    
    # 加载Atlas模型
    logger.info("🧠 加载Atlas NPU模型...")
    model = AtlasInferSession(device_id, model_path)
    logger.info("✅ 模型加载完成")
    
    # 初始化摄像头
    logger.info(f"📷 打开摄像头 {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"❌ 无法打开摄像头 {camera_index}")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # 确认实际参数
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"📷 摄像头参数: {actual_w}x{actual_h} @ {actual_fps}fps")
    
    # 初始化控制器
    if enable_control:
        # 使用当前文件中的控制器类
        controller = VisualLateralErrorController(
            steering_gain=steering_gain,
            base_pwm=base_speed,
            curvature_damping=curvature_damping,
            preview_distance=preview_distance,
            max_pwm=max_speed,
            min_pwm=min_speed
        )
        logger.info("🚗 控制器初始化完成")
    else:
        controller = None
    
    # 透视变换器
    transformer = PerspectiveTransformer()
    logger.info("🦅 透视变换器初始化完成")
    
    # 初始化Web界面数据
    if enable_web:
        with web_data_lock:
            web_data['is_running'] = True
            web_data['start_time'] = time.time()
            web_data['frame_count'] = 0
        logger.info("🌐 Web界面数据初始化完成")
    
    frame_count = 0
    start_time = time.time()
    total_times = {"preprocess": 0, "inference": 0, "postprocess": 0, "transform": 0, "control": 0}
    
    logger.info("🎬 开始实时推理循环...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠️ 无法读取摄像头帧")
                time.sleep(0.1)
                continue
            
            loop_start = time.time()
            
            # 1. 预处理
            preprocess_start = time.time()
            input_data = preprocess_matched_resolution(frame, dtype=np.float16)
            preprocess_time = (time.time() - preprocess_start) * 1000
            
            # 2. Atlas推理
            inference_start = time.time()
            outputs = model.infer([input_data])
            inference_time = (time.time() - inference_start) * 1000
            
            # 3. 后处理
            postprocess_start = time.time()
            lane_mask = postprocess_matched_resolution(outputs[0], actual_w, actual_h)
            postprocess_time = (time.time() - postprocess_start) * 1000
            
            # 4. 透视变换和路径规划
            transform_start = time.time()
            
            # 使用与单文件推理相同的逻辑，应用边缘计算优化
            if full_image_bird_eye:
                # 完整图像模式：应用与单文件推理相同的边缘计算优化
                adjusted_pixels_per_unit = 1  # 与单文件推理相同：边缘计算极致优化
                print(f"⚡ 边缘计算极致优化：像素密度 = {adjusted_pixels_per_unit} 像素/单位")
            else:
                # A4纸区域模式：使用较高像素密度
                adjusted_pixels_per_unit = 20
                print(f"🔧 A4纸区域鸟瞰图模式：像素密度 = {adjusted_pixels_per_unit} 像素/单位")
            
            transformer = PerspectiveTransformer()
            bird_eye_image, bird_eye_mask, view_params = transformer.transform_image_and_mask(
                frame, lane_mask, pixels_per_unit=adjusted_pixels_per_unit, margin_ratio=0.1, full_image=full_image_bird_eye)
            
            # 路径规划 - 使用与单文件推理相同的参数
            control_map, path_data = create_control_map(
                bird_eye_mask, view_params, 
                add_grid=True, add_path=True,
                path_smooth_method='polynomial',
                path_degree=3,
                num_waypoints=20,
                min_road_width=10,
                edge_computing=True, 
                force_bottom_center=True
            )
            transform_time = (time.time() - transform_start) * 1000
            
            # 检查Web界面参数更新
            if enable_web and enable_control and controller:
                with web_data_lock:
                    if web_data.get('params_updated', False):
                        # 应用新参数到控制器
                        new_params = web_data['control_params']
                        controller.steering_gain = new_params['steering_gain']
                        controller.base_pwm = new_params['base_speed']
                        controller.preview_distance = new_params['preview_distance']
                        
                        web_data['params_updated'] = False  # 重置标志
                        print(f"🎛️ 控制参数已更新: 转向增益={controller.steering_gain}, "
                              f"基础PWM={controller.base_pwm}, 预瞄距离={controller.preview_distance}cm")
            
            # 5. 控制计算
            control_time = 0
            control_result = None
            if enable_control and path_data is not None:
                control_start = time.time()
                control_result = controller.compute_wheel_pwm(path_data, view_params)
                control_time = (time.time() - control_start) * 1000
            
            # 性能统计
            frame_count += 1
            total_times["preprocess"] += preprocess_time
            total_times["inference"] += inference_time
            total_times["postprocess"] += postprocess_time
            total_times["transform"] += transform_time
            total_times["control"] += control_time
            
            pipeline_latency = (time.time() - loop_start) * 1000
            
            # 每20帧输出一次详细统计
            if frame_count % 20 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed
                
                avg_preprocess = total_times["preprocess"] / frame_count
                avg_inference = total_times["inference"] / frame_count
                avg_postprocess = total_times["postprocess"] / frame_count
                avg_transform = total_times["transform"] / frame_count
                avg_control = total_times["control"] / frame_count
                avg_total = sum(total_times.values()) / frame_count
                
                logger.info(f"📊 第{frame_count}帧性能分析:")
                logger.info(f"   预处理: {preprocess_time:.1f}ms (平均: {avg_preprocess:.1f}ms)")
                logger.info(f"   Atlas推理: {inference_time:.1f}ms (平均: {avg_inference:.1f}ms)")
                logger.info(f"   后处理: {postprocess_time:.1f}ms (平均: {avg_postprocess:.1f}ms)")
                logger.info(f"   透视变换: {transform_time:.1f}ms (平均: {avg_transform:.1f}ms)")
                if enable_control:
                    logger.info(f"   控制计算: {control_time:.1f}ms (平均: {avg_control:.1f}ms)")
                logger.info(f"   总延迟: {pipeline_latency:.1f}ms (平均: {avg_total:.1f}ms)")
                logger.info(f"   实际FPS: {avg_fps:.1f}, 理论FPS: {1000/avg_total:.1f}")
                
                # 控制信息
                if control_result:
                    logger.info(f"🚗 控制指令: 左轮={control_result['pwm_left']:.0f}, 右轮={control_result['pwm_right']:.0f}")
                    logger.info(f"   横向误差: {control_result['lateral_error']:.2f}cm, 曲率: {control_result.get('curvature_level', 0):.4f}")
            
            # 检测车道线
            lane_pixels = np.sum(lane_mask > 0)
            total_pixels = lane_mask.shape[0] * lane_mask.shape[1]
            lane_ratio = (lane_pixels / total_pixels) * 100
            
            # 每帧简要日志
            if control_result:
                logger.info(f"帧{frame_count}: 延迟{pipeline_latency:.1f}ms, 车道线{lane_ratio:.1f}%, "
                          f"控制[L:{control_result['pwm_left']:.0f}, R:{control_result['pwm_right']:.0f}]")
                # 每帧详细控制信息
                logger.info(f"   🚗 横向误差: {control_result['lateral_error']:.2f}cm, "
                          f"曲率: {control_result.get('curvature_level', 0):.4f}, "
                          f"转向: {control_result.get('turn_direction', 'unknown')}")
            else:
                logger.info(f"帧{frame_count}: 延迟{pipeline_latency:.1f}ms, 车道线{lane_ratio:.1f}%")
            
            # 更新Web界面数据
            if enable_web:
                with web_data_lock:
                    web_data['frame_count'] = frame_count
                    
                    # 调试信息
                    if control_map is not None:
                        print(f"🖼️ 生成控制地图: {control_map.shape}, 数据类型: {control_map.dtype}")
                        web_data['latest_control_map'] = control_map.copy()
                    else:
                        print("⚠️ 控制地图为None")
                        web_data['latest_control_map'] = None
                        
                    web_data['latest_stats'] = {
                        'latency': pipeline_latency,
                        'lane_ratio': lane_ratio,
                        'left_pwm': control_result['pwm_left'] if control_result else 0,
                        'right_pwm': control_result['pwm_right'] if control_result else 0,
                        'lateral_error': control_result['lateral_error'] if control_result else 0,
                        'path_curvature': control_result.get('curvature_level', 0) if control_result else 0
                    }
            
            # 检测退出条件（仅在有GUI时检查按键）
            if not no_gui:
                try:
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC键
                        logger.info("🛑 用户按ESC键退出")
                        break
                except cv2.error:
                    # 如果OpenCV GUI不可用，忽略错误
                    logger.warning("⚠️ OpenCV GUI不可用，无法检测按键")
                    no_gui = True  # 自动切换到无GUI模式
            else:
                # 无GUI模式下可以通过其他方式退出，例如文件标志
                time.sleep(0.001)  # 短暂休眠避免过度占用CPU
                
    except KeyboardInterrupt:
        logger.info("🛑 用户中断 (Ctrl+C)")
    except Exception as e:
        logger.error(f"❌ 实时推理错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # 更新Web界面状态
        if enable_web:
            with web_data_lock:
                web_data['is_running'] = False
                
        cap.release()
        
        # 仅在非无GUI模式下调用OpenCV GUI函数
        if not no_gui:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                # 忽略OpenCV GUI相关错误
                pass
                
        logger.info("🔚 实时推理系统已关闭")

# ---------------------------------------------------------------------------------
# --- 🌐 Web界面模块 ---
# ---------------------------------------------------------------------------------

# Web界面相关全局变量
web_data = {
    'latest_frame': None,
    'latest_control_map': None,
    'latest_stats': {},
    'is_running': False,
    'frame_count': 0,
    'start_time': None,
    'control_params': {
        'steering_gain': 10.0,
        'base_speed': 500.0,
        'preview_distance': 30.0
    },
    'params_updated': False
}
web_data_lock = Lock()

# HTML模板
WEB_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>实时车道线分割控制台</title>
    <meta charset="utf-8">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            background: #1a1a1a; 
            color: #fff; 
            margin: 0; 
            padding: 20px; 
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
        }
        .header { 
            text-align: center; 
            margin-bottom: 30px; 
            padding: 20px; 
            background: #2d2d2d; 
            border-radius: 10px; 
        }
        .stats-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: #2d2d2d;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        .stat-label {
            font-size: 14px;
            color: #ccc;
        }
        .image-panel {
            background: #2d2d2d;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .control-map {
            max-width: 100%;
            border: 2px solid #4CAF50;
            border-radius: 8px;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-running { background: #4CAF50; }
        .status-stopped { background: #f44336; }
        .log-panel {
            background: #2d2d2d;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            max-height: 300px;
            overflow-y: auto;
        }
        .log-entry {
            margin-bottom: 5px;
            font-family: monospace;
            font-size: 12px;
        }
        .param-panel {
            background: #2d2d2d;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .param-control {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
            gap: 15px;
        }
        .param-label {
            min-width: 120px;
            font-weight: bold;
            color: #4CAF50;
        }
        .param-slider {
            flex: 1;
            height: 6px;
            border-radius: 3px;
            background: #444;
            outline: none;
            -webkit-appearance: none;
        }
        .param-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #4CAF50;
            cursor: pointer;
        }
        .param-value {
            min-width: 80px;
            text-align: center;
            font-weight: bold;
            color: #fff;
        }
        .param-apply {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }
        .param-apply:hover {
            background: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚗 实时车道线分割控制台</h1>
            <p>
                <span id="status-indicator" class="status-indicator status-stopped"></span>
                <span id="status-text">系统停止</span>
            </p>
        </div>
        
        <div class="stats-panel">
            <div class="stat-card">
                <div class="stat-value" id="frame-count">0</div>
                <div class="stat-label">帧数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="fps">0.0</div>
                <div class="stat-label">FPS</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="latency">0</div>
                <div class="stat-label">延迟(ms)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="lane-ratio">0.0</div>
                <div class="stat-label">车道线覆盖率(%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="left-pwm">0</div>
                <div class="stat-label">左轮PWM</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="right-pwm">0</div>
                <div class="stat-label">右轮PWM</div>
            </div>
        </div>
        
        <div class="param-panel">
            <h3>🎛️ 控制参数实时调整</h3>
            <div class="param-control">
                <span class="param-label">转向增益</span>
                <input type="range" class="param-slider" id="steering-gain-slider" 
                       min="1" max="50" step="0.5" value="10">
                <span class="param-value" id="steering-gain-value">10.0</span>
            </div>
            <div class="param-control">
                <span class="param-label">基础PWM</span>
                <input type="range" class="param-slider" id="base-speed-slider" 
                       min="100" max="1000" step="10" value="500">
                <span class="param-value" id="base-speed-value">500</span>
            </div>
            <div class="param-control">
                <span class="param-label">预瞄距离(cm)</span>
                <input type="range" class="param-slider" id="preview-distance-slider" 
                       min="10" max="100" step="1" value="30">
                <span class="param-value" id="preview-distance-value">30</span>
            </div>
            <div style="text-align: center; margin-top: 15px;">
                <button class="param-apply" onclick="applyParameters()">应用参数</button>
            </div>
        </div>
        
        <div class="image-panel">
            <h3>🗺️ 实时控制地图</h3>
            <img id="control-map" class="control-map" src="/api/control_map" alt="控制地图加载中...">
        </div>
        
        <div class="log-panel">
            <h3>📋 系统日志</h3>
            <div id="log-content"></div>
        </div>
    </div>
    
    <script>
        let logEntries = [];
        const maxLogEntries = 50;
        
        function updateStats() {
            fetch('/api/stats')
            .then(response => response.json())
            .then(data => {
                document.getElementById('frame-count').textContent = data.frame_count || 0;
                document.getElementById('fps').textContent = (data.fps || 0).toFixed(1);
                document.getElementById('latency').textContent = Math.round(data.latency || 0);
                document.getElementById('lane-ratio').textContent = (data.lane_ratio || 0).toFixed(1);
                document.getElementById('left-pwm').textContent = Math.round(data.left_pwm || 0);
                document.getElementById('right-pwm').textContent = Math.round(data.right_pwm || 0);
                
                // 更新状态指示器
                const statusIndicator = document.getElementById('status-indicator');
                const statusText = document.getElementById('status-text');
                if (data.is_running) {
                    statusIndicator.className = 'status-indicator status-running';
                    statusText.textContent = '系统运行中';
                } else {
                    statusIndicator.className = 'status-indicator status-stopped';
                    statusText.textContent = '系统停止';
                }
                
                // 添加新日志条目
                if (data.latest_log) {
                    addLogEntry(data.latest_log);
                }
            })
            .catch(error => console.error('获取状态失败:', error));
        }
        
        function addLogEntry(logText) {
            const timestamp = new Date().toLocaleTimeString();
            logEntries.push(`[${timestamp}] ${logText}`);
            if (logEntries.length > maxLogEntries) {
                logEntries.shift();
            }
            
            const logContent = document.getElementById('log-content');
            logContent.innerHTML = logEntries.map(entry => 
                `<div class="log-entry">${entry}</div>`
            ).join('');
            logContent.scrollTop = logContent.scrollHeight;
        }
        
        // 定期更新控制地图
        function updateControlMap() {
            const img = document.getElementById('control-map');
            img.src = '/api/control_map?' + new Date().getTime();
        }
        
        // 参数滑块更新显示值
        function updateSliderValues() {
            const steeringGain = document.getElementById('steering-gain-slider');
            const steeringValue = document.getElementById('steering-gain-value');
            steeringValue.textContent = parseFloat(steeringGain.value).toFixed(1);
            
            const baseSpeed = document.getElementById('base-speed-slider');
            const baseValue = document.getElementById('base-speed-value');
            baseValue.textContent = baseSpeed.value;
            
            const previewDistance = document.getElementById('preview-distance-slider');
            const previewValue = document.getElementById('preview-distance-value');
            previewValue.textContent = previewDistance.value;
        }
        
        // 应用参数到系统
        function applyParameters() {
            const steeringGain = document.getElementById('steering-gain-slider').value;
            const baseSpeed = document.getElementById('base-speed-slider').value;
            const previewDistance = document.getElementById('preview-distance-slider').value;
            
            const params = {
                steering_gain: parseFloat(steeringGain),
                base_speed: parseFloat(baseSpeed),
                preview_distance: parseFloat(previewDistance)
            };
            
            fetch('/api/update_params', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(params)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    addLogEntry(`参数更新成功: 转向增益=${steeringGain}, 基础PWM=${baseSpeed}, 预瞄距离=${previewDistance}cm`);
                } else {
                    addLogEntry(`参数更新失败: ${data.error}`);
                }
            })
            .catch(error => {
                addLogEntry(`参数更新错误: ${error}`);
                console.error('参数更新失败:', error);
            });
        }
        
        // 绑定滑块事件
        document.getElementById('steering-gain-slider').addEventListener('input', updateSliderValues);
        document.getElementById('base-speed-slider').addEventListener('input', updateSliderValues);
        document.getElementById('preview-distance-slider').addEventListener('input', updateSliderValues);
        
        // 启动定时更新
        setInterval(updateStats, 1000);  // 每秒更新状态
        setInterval(updateControlMap, 2000);  // 每2秒更新控制地图
        
        // 初始加载
        updateStats();
        updateSliderValues();
    </script>
</body>
</html>
"""

def create_web_app():
    """创建Flask Web应用"""
    if not FLASK_AVAILABLE:
        return None
    
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return render_template_string(WEB_TEMPLATE)
    
    @app.route('/api/stats')
    def get_stats():
        with web_data_lock:
            stats = web_data['latest_stats'].copy()
            stats['is_running'] = web_data['is_running']
            stats['frame_count'] = web_data['frame_count']
            
            # 计算FPS
            if web_data['start_time'] and web_data['frame_count'] > 0:
                elapsed = time.time() - web_data['start_time']
                stats['fps'] = web_data['frame_count'] / elapsed if elapsed > 0 else 0
            else:
                stats['fps'] = 0
        
        return jsonify(stats)
    
    @app.route('/api/update_params', methods=['POST'])
    def update_params():
        try:
            params = request.get_json()
            
            # 验证参数
            if not params:
                return jsonify({'success': False, 'error': '无效的参数数据'})
            
            # 更新全局控制参数
            with web_data_lock:
                if 'control_params' not in web_data:
                    web_data['control_params'] = {}
                
                if 'steering_gain' in params:
                    web_data['control_params']['steering_gain'] = float(params['steering_gain'])
                if 'base_speed' in params:
                    web_data['control_params']['base_speed'] = float(params['base_speed'])
                if 'preview_distance' in params:
                    web_data['control_params']['preview_distance'] = float(params['preview_distance'])
                
                # 设置更新标志
                web_data['params_updated'] = True
            
            print(f"🎛️ Web参数更新: {params}")
            return jsonify({'success': True, 'message': '参数更新成功'})
            
        except Exception as e:
            print(f"❌ 参数更新错误: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/control_map')
    def get_control_map():
        with web_data_lock:
            if web_data['latest_control_map'] is not None:
                try:
                    # 确保图像格式正确
                    control_map = web_data['latest_control_map']
                    
                    # 调试信息
                    print(f"🖼️ Web请求控制地图: {control_map.shape}, 类型: {control_map.dtype}")
                    print(f"🖼️ 数据范围: {control_map.min()} ~ {control_map.max()}")
                    
                    # 如果是单通道图像，转换为3通道
                    if len(control_map.shape) == 2:
                        control_map = cv2.cvtColor(control_map, cv2.COLOR_GRAY2BGR)
                        print("🔄 单通道转换为3通道")
                    elif control_map.shape[2] == 1:
                        control_map = cv2.cvtColor(control_map, cv2.COLOR_GRAY2BGR)
                        print("🔄 单通道转换为3通道")
                    
                    # 确保数据类型为uint8
                    if control_map.dtype != np.uint8:
                        if control_map.max() <= 1.0:
                            control_map = (control_map * 255).astype(np.uint8)
                            print("🔄 归一化数据转换为uint8")
                        else:
                            control_map = control_map.astype(np.uint8)
                            print("🔄 数据类型转换为uint8")
                    
                    # 将OpenCV图像转换为PNG格式
                    success, buffer = cv2.imencode('.png', control_map)
                    if not success:
                        raise Exception("图像编码失败")
                        
                    print(f"✅ 控制地图编码成功，buffer长度: {len(buffer)}")
                    
                    # 返回二进制图像数据
                    from flask import Response
                    return Response(
                        buffer.tobytes(),
                        mimetype='image/png',
                        headers={'Cache-Control': 'no-cache, no-store, must-revalidate'}
                    )
                    
                except Exception as e:
                    print(f"❌ 控制地图编码错误: {e}")
                    # 返回错误提示图片
                    empty_img = np.zeros((300, 400, 3), dtype=np.uint8)
                    cv2.putText(empty_img, f"Error: {str(e)[:20]}", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    _, buffer = cv2.imencode('.png', empty_img)
                    from flask import Response
                    return Response(
                        buffer.tobytes(),
                        mimetype='image/png'
                    )
            else:
                print("⚠️ 没有可用的控制地图数据")
                # 返回空图片占位符
                empty_img = np.zeros((300, 400, 3), dtype=np.uint8)
                cv2.putText(empty_img, "No Control Map", (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.png', empty_img)
                from flask import Response
                return Response(
                    buffer.tobytes(),
                    mimetype='image/png'
                )
    
    return app

def start_web_server(port=5000):
    """启动Web服务器"""
    if not FLASK_AVAILABLE:
        print("❌ Flask未安装，无法启动Web服务器")
        return None
    
    app = create_web_app()
    if app is None:
        return None
    
    def run_server():
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
    
    server_thread = Thread(target=run_server, daemon=True)
    server_thread.start()
    
    print(f"🌐 Web界面已启动: http://localhost:{port}")
    print(f"🌐 外部访问: http://0.0.0.0:{port}")
    
    return server_thread

# ---------------------------------------------------------------------------------
# --- 📱 命令行接口 ---
# ---------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="集成感知与环境建模的车道线分割推理工具 - Atlas版本")
    
    # 主要模式选择
    parser.add_argument("--realtime", action="store_true", help="实时摄像头推理模式")
    parser.add_argument("--input", "-i", help="输入图片路径 (单张图片模式)")
    parser.add_argument("--output", "-o", help="输出可视化图片路径（可选）")
    parser.add_argument("--save_mask", help="保存分割掩码路径（可选）")
    
    # 模型和设备参数
    parser.add_argument("--model", "-m", 
                       default="./weights/fast_scnn_custom_e2e_360x640_fp16_fixed_simp.om",
                       help="OM模型路径")
    parser.add_argument("--device_id", type=int, default=0, help="Atlas NPU设备ID")
    
    # 摄像头参数
    parser.add_argument("--camera_index", type=int, default=0, help="摄像头索引")
    parser.add_argument("--camera_width", type=int, default=640, help="摄像头宽度")
    parser.add_argument("--camera_height", type=int, default=360, help="摄像头高度")
    
    # 日志参数
    parser.add_argument("--log_file", help="日志文件路径 (实时模式)")
    
    # 图像处理参数
    parser.add_argument("--pixels_per_unit", type=int, default=20, help="每单位像素数 (默认: 20)")
    parser.add_argument("--margin_ratio", type=float, default=0.1, help="边距比例 (默认: 0.1)")
    parser.add_argument("--no_vis", action="store_true", help="不保存可视化结果，仅推理")
    parser.add_argument("--bird_eye", action="store_true", help="生成鸟瞰图（使用内置A4纸标定）")
    parser.add_argument("--no_full_image_bird_eye", action="store_true", help="仅生成A4纸区域鸟瞰图（默认生成完整原图）")
    parser.add_argument("--save_control_map", action="store_true", help="保存控制地图并进行路径规划")
    parser.add_argument("--path_smooth_method", default="polynomial", choices=["polynomial", "spline"], help="路径平滑方法")
    parser.add_argument("--path_degree", type=int, default=3, help="路径拟合阶数")
    parser.add_argument("--num_waypoints", type=int, default=20, help="路径点数量")
    parser.add_argument("--min_road_width", type=int, default=10, help="最小可行驶宽度（像素）")
    parser.add_argument("--force_bottom_center", action="store_true", default=True, help="强制拟合曲线过底边中点")
    parser.add_argument("--edge_computing", action="store_true", help="边缘计算模式（极致性能优化）")
    parser.add_argument("--preview", action="store_true", help="显示预览窗口")
    
    # 控制算法参数
    parser.add_argument("--enable_control", action="store_true", help="启用视觉横向误差控制算法")
    parser.add_argument("--steering_gain", type=float, default=10.0, help="转向增益Kp (默认: 10.0)")
    parser.add_argument("--base_speed", type=float, default=500.0, help="基础PWM值 -1000~+1000 (默认: 500)")
    parser.add_argument("--curvature_damping", type=float, default=0.1, help="曲率阻尼系数 (默认: 0.1)")
    parser.add_argument("--preview_distance", type=float, default=30.0, help="预瞄距离 cm (默认: 30.0)")
    parser.add_argument("--max_speed", type=float, default=800.0, help="最大PWM值 -1000~+1000 (默认: 800)")
    parser.add_argument("--min_speed", type=float, default=100.0, help="最小PWM值，前进时最低速度 (默认: 100)")
    
    # Web界面和GUI选项
    parser.add_argument("--web", action="store_true", help="启用Web界面")
    parser.add_argument("--web_port", type=int, default=5000, help="Web界面端口 (默认: 5000)")
    parser.add_argument("--no_gui", action="store_true", help="无GUI模式（不显示OpenCV窗口，仅输出结果）")
    
    args = parser.parse_args()
    
    try:
        print("🧠 集成感知与环境建模的车道线分割推理工具 - Atlas版本")
        print("=" * 60)
        print("📝 功能: 车道线分割 + 透视变换 + 控制地图生成")
        print("📏 内置标定: 基于A4纸的透视变换参数")
        print("🚀 推理设备: Atlas NPU")
        print("=" * 60)
        
        # 实时模式
        if args.realtime:
            print("🎬 启动实时摄像头推理模式")
            
            # 启动Web服务器（如果需要）
            web_server = None
            if args.web:
                print("🌐 启动Web界面...")
                web_server = start_web_server(args.web_port)
                time.sleep(2)  # 等待服务器启动
                if args.no_gui:
                    print("💡 提示：无GUI模式下，请通过Web界面查看实时状态")
            
            realtime_inference(
                model_path=args.model,
                device_id=args.device_id,
                camera_index=args.camera_index,
                camera_width=args.camera_width,
                camera_height=args.camera_height,
                log_file=args.log_file,
                enable_control=args.enable_control,
                steering_gain=args.steering_gain,
                base_speed=args.base_speed,
                curvature_damping=args.curvature_damping,
                preview_distance=args.preview_distance,
                max_speed=args.max_speed,
                min_speed=args.min_speed,
                enable_web=args.web,
                no_gui=args.no_gui,
                full_image_bird_eye=not args.no_full_image_bird_eye  # 反转逻辑
            )
            return
        
        # 单张图片模式
        if not args.input:
            print("❌ 错误：请指定 --input 输入图片路径或使用 --realtime 进入实时模式")
            parser.print_help()
            sys.exit(1)
        
        # 自动确定输出路径
        save_visualization = not args.no_vis
        save_mask = bool(args.save_mask)
        
        # 验证输入文件存在
        if not os.path.exists(args.input):
            print(f"❌ 错误：输入图片不存在: {args.input}")
            sys.exit(1)
        
        # 执行推理
        results = inference_single_image(
            image_path=args.input,
            model_path=args.model,
            device_id=args.device_id,
            save_visualization=save_visualization,
            save_mask=save_mask,
            bird_eye=args.bird_eye,
            save_control_map=args.save_control_map,
            pixels_per_unit=args.pixels_per_unit,
            margin_ratio=args.margin_ratio,
            full_image_bird_eye=args.full_image_bird_eye,
            path_smooth_method=args.path_smooth_method,
            path_degree=args.path_degree,
            num_waypoints=args.num_waypoints,
            min_road_width=args.min_road_width,
            edge_computing=args.edge_computing,
            force_bottom_center=args.force_bottom_center,
            enable_control=args.enable_control,
            steering_gain=args.steering_gain,
            base_speed=args.base_speed,
            curvature_damping=args.curvature_damping,
            preview_distance=args.preview_distance,
            max_speed=args.max_speed,
            min_speed=args.min_speed
        )
        
        # 处理输出路径重命名
        if args.output and 'visualization_path' in results:
            import shutil
            # 确保输出目录存在
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 移动可视化结果到指定路径
            shutil.move(results['visualization_path'], args.output)
            results['visualization_path'] = args.output
            print(f"💾 可视化结果已移动到指定路径: {args.output}")
        
        # 处理掩码路径重命名
        if args.save_mask and 'mask_path' in results:
            import shutil
            # 确保输出目录存在
            mask_dir = os.path.dirname(args.save_mask)
            if mask_dir and not os.path.exists(mask_dir):
                os.makedirs(mask_dir)
            
            # 移动分割掩码到指定路径
            shutil.move(results['mask_path'], args.save_mask)
            results['mask_path'] = args.save_mask
            print(f"💾 分割掩码已移动到指定路径: {args.save_mask}")
        
        # 预览结果（可选）
        if args.preview:
            print("👁️ 显示预览...")
            original = cv2.imread(args.input)
            
            if 'visualization_path' in results:
                vis_result = cv2.imread(results['visualization_path'])
                cv2.imshow("Original Image", original)
                cv2.imshow("Segmentation Result", vis_result)
            
            if 'bird_eye_vis_path' in results:
                bird_eye_vis = cv2.imread(results['bird_eye_vis_path'])
                cv2.imshow("Bird's Eye View Segmentation", bird_eye_vis)
            
            if 'control_map_path' in results:
                control_map = cv2.imread(results['control_map_path'])
                cv2.imshow("Control Map", control_map)
            
            print("按任意键关闭预览...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print("\n✅ Atlas推理与环境建模完成！")
        print("🔧 此结果可用于后续的路径规划和车辆控制")
        
        if 'visualization_path' in results:
            print(f"🎨 分割可视化: {results['visualization_path']}")
        if 'mask_path' in results:
            print(f"🎭 分割掩码: {results['mask_path']}")
        if 'bird_eye_vis_path' in results:
            print(f"🦅 鸟瞰图分割: {results['bird_eye_vis_path']}")
        if 'control_map_path' in results:
            print(f"🗺️ 控制地图: {results['control_map_path']}")
        if 'path_json_path' in results:
            print(f"🛣️ 路径数据: {results['path_json_path']}")
        if 'control_json_path' in results:
            print(f"🚗 控制数据: {results['control_json_path']}")
        if 'control_vis_path' in results:
            print(f"🎮 控制可视化: {results['control_vis_path']}")
            
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def save_path_data_json(path_data, json_path):
    """
    将路径数据保存为JSON文件
    
    参数：
        path_data: 路径数据字典
        json_path: JSON文件路径
    """
    # 准备可序列化的数据
    json_data = {
        'centerline_world': path_data['centerline_world'],
        'waypoints': path_data['waypoints'],
        'path_length': path_data['path_length'],
        'num_centerline_points': path_data['num_centerline_points'],
        'num_waypoints': path_data['num_waypoints'],
        'fit_params': path_data['fit_params'].tolist() if path_data['fit_params'] is not None else None,
        'description': '车道中心线和路径点数据（世界坐标，单位：厘米）',
        'coordinate_system': 'world coordinates (cm)',
        'waypoints_description': '路径点，可直接用于车辆控制'
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
