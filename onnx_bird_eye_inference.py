#!/usr/bin/env python3
"""
集成感知与环境建模的车道线分割推理脚本

功能特性：
- 使用ONNX Runtime进行车道线分割推理
- 支持透视变换，生成鸟瞰图
- 可视化可驾驶区域的鸟瞰图
- 为路径规划提供2D地图数据
- 支持实时处理和批量处理

使用方法：
# 基础推理（仅分割）
python onnx_bird_eye_inference.py --input image.jpg --output result.jpg

# 添加透视变换（需要标定文件）
python onnx_bird_eye_inference.py --input image.jpg --output result.jpg --calibration calibration.json --bird_eye

# 生成控制用的鸟瞰图
python onnx_bird_eye_inference.py --input image.jpg --calibration calibration.json --bird_eye --save_control_map

作者：基于onnx_single_image_inference.py扩展，集成透视变换功能
"""

import os
import sys
import cv2
import time
import numpy as np
import argparse
import json
from pathlib import Path

# 导入ONNX Runtime
try:
    import onnxruntime as ort
except ImportError:
    print("❌ 错误：未找到onnxruntime库，请安装")
    print("CPU版本: pip install onnxruntime")
    print("GPU版本: pip install onnxruntime-gpu")
    sys.exit(1)

# ---------------------------------------------------------------------------------
# --- 🚀🚀🚀 预处理模块 (与Atlas完全一致) 🚀🚀🚀 ---
# ---------------------------------------------------------------------------------

def preprocess_matched_resolution(img_bgr, target_width=640, target_height=360, dtype=np.float32):
    """
    图片预处理，与atlas_single_image_inference.py完全一致
    
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
# --- 🚀🚀🚀 后处理模块 (与Atlas完全一致) 🚀🚀🚀 ---
# ---------------------------------------------------------------------------------

def postprocess_matched_resolution(output_tensor, original_width, original_height):
    """
    后处理，与atlas_single_image_inference.py完全一致
    
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
# --- 🎨 可视化模块 ---
# ---------------------------------------------------------------------------------

def create_visualization(original_img, mask, alpha=0.5):
    """
    创建车道线分割可视化图像
    
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
# --- 🧠 ONNX Runtime推理会话 ---
# ---------------------------------------------------------------------------------

class ONNXInferSession:
    """ONNX Runtime推理会话，模拟Atlas InferSession接口"""
    
    def __init__(self, model_path, provider='CPUExecutionProvider'):
        """
        初始化ONNX推理会话
        
        参数：
            model_path: ONNX模型路径
            provider: 执行提供者 ('CPUExecutionProvider', 'CUDAExecutionProvider')
        """
        self.model_path = model_path
        self.provider = provider
        
        # 设置执行提供者
        available_providers = ort.get_available_providers()
        if provider not in available_providers:
            print(f"⚠️ 警告: {provider} 不可用，可用提供者: {available_providers}")
            provider = 'CPUExecutionProvider'
        
        print(f"🧠 使用执行提供者: {provider}")
        
        # 创建推理会话
        self.session = ort.InferenceSession(model_path, providers=[provider])
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"📊 输入节点: {self.input_name}")
        print(f"📊 输出节点: {self.output_name}")
    
    def infer(self, inputs):
        """
        执行推理，与Atlas InferSession.infer接口一致
        
        参数：
            inputs: 输入张量列表
        
        返回：
            outputs: 输出张量列表
        """
        input_tensor = inputs[0]
        
        # 执行推理
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        
        return outputs

# ---------------------------------------------------------------------------------
# --- 🦅 透视变换模块 (鸟瞰图生成) ---
# ---------------------------------------------------------------------------------

class PerspectiveTransformer:
    """透视变换器，用于生成鸟瞰图"""
    
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
    
    def calculate_bird_eye_params(self, pixels_per_unit=20, margin_ratio=0.2):
        """计算鸟瞰图参数"""
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
    
    def transform_image_and_mask(self, image, mask, pixels_per_unit=20, margin_ratio=0.2):
        """
        将图像和分割掩码都转换为鸟瞰图
        
        参数：
            image: 输入图像 (BGR格式)
            mask: 分割掩码 (0/255)
            pixels_per_unit: 每单位的像素数
            margin_ratio: 边距比例
        
        返回：
            bird_eye_image: 鸟瞰图
            bird_eye_mask: 鸟瞰图分割掩码
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
            'transform_matrix': combined_transform.tolist()
        }
        
        return bird_eye_image, bird_eye_mask, view_params

# ---------------------------------------------------------------------------------
# --- 🗺️ 控制地图生成模块 ---
# ---------------------------------------------------------------------------------

def create_control_map(bird_eye_mask, view_params, add_grid=True):
    """
    创建用于路径规划的控制地图
    
    参数：
        bird_eye_mask: 鸟瞰图分割掩码
        view_params: 视图参数
        add_grid: 是否添加网格
    
    返回：
        control_map: 控制地图 (三通道BGR图像)
    """
    # 创建控制地图
    control_map = np.zeros((bird_eye_mask.shape[0], bird_eye_mask.shape[1], 3), dtype=np.uint8)
    
    # 可驾驶区域 - 绿色
    control_map[bird_eye_mask > 0] = [0, 255, 0]  # BGR绿色
    
    # 不可驾驶区域 - 保持黑色
    # control_map[bird_eye_mask == 0] = [0, 0, 0]  # 已经是黑色
    
    if add_grid:
        control_map = add_grid_to_control_map(control_map, view_params)
    
    return control_map

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
    grid_interval = 10  # 网格间隔（单位：cm或指定单位）
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
    
    return annotated_map

# ---------------------------------------------------------------------------------
# --- 📊 性能分析 ---
# ---------------------------------------------------------------------------------

def print_performance_analysis(times_dict, input_tensor, model_path, provider):
    """打印详细的性能分析报告"""
    total_time = sum(times_dict.values())
    
    print("\n" + "="*60)
    print("🧠 ONNX Runtime + 透视变换 性能分析")
    print("="*60)
    print(f"🧠 模型: {Path(model_path).name}")
    print(f"⚡ 执行提供者: {provider}")
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
# --- 📱 主推理函数 (集成感知与环境建模) ---
# ---------------------------------------------------------------------------------

def inference_with_bird_eye_view(image_path, model_path, calibration_path=None,
                                provider='CPUExecutionProvider', 
                                pixels_per_unit=20, margin_ratio=0.2,
                                save_visualization=True, save_mask=False,
                                save_bird_eye=False, save_control_map=False):
    """
    集成车道线分割推理和透视变换的完整感知管道
    
    参数：
        image_path: 输入图片路径
        model_path: ONNX模型路径
        calibration_path: 相机标定文件路径（可选）
        provider: ONNX执行提供者
        pixels_per_unit: 每单位像素数
        margin_ratio: 边距比例
        save_visualization: 是否保存普通可视化结果
        save_mask: 是否保存分割掩码
        save_bird_eye: 是否保存鸟瞰图
        save_control_map: 是否保存控制地图
    
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
    
    # 2. 加载模型
    print(f"🧠 加载ONNX模型: {model_path}")
    model_start = time.time()
    model = ONNXInferSession(model_path, provider)
    model_load_time = (time.time() - model_start) * 1000
    print(f"✅ 模型加载完成 ({model_load_time:.1f}ms)")
    
    # 3. 预处理
    print("🔄 开始预处理...")
    preprocess_start = time.time()
    input_data = preprocess_matched_resolution(img_bgr, dtype=np.float32)
    preprocess_time = (time.time() - preprocess_start) * 1000
    
    print(f"📊 输入张量形状: {input_data.shape}")
    print(f"📊 数据类型: {input_data.dtype}")
    
    # 4. ONNX推理
    print("🚀 开始ONNX推理...")
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
    bird_eye_image = None
    bird_eye_mask = None
    control_map = None
    view_params = None
    
    if calibration_path and os.path.exists(calibration_path):
        print("🦅 开始透视变换...")
        transform_start = time.time()
        
        transformer = PerspectiveTransformer(calibration_path)
        bird_eye_image, bird_eye_mask, view_params = transformer.transform_image_and_mask(
            img_bgr, lane_mask, pixels_per_unit, margin_ratio)
        
        # 生成控制地图
        control_map = create_control_map(bird_eye_mask, view_params, add_grid=True)
        
        transform_time = (time.time() - transform_start) * 1000
        
        print(f"📐 鸟瞰图尺寸: {view_params['output_size'][0]}×{view_params['output_size'][1]}")
        bounds = view_params['view_bounds']
        print(f"📐 世界坐标范围: X({bounds[0]:.1f}~{bounds[2]:.1f}), Y({bounds[1]:.1f}~{bounds[3]:.1f})")
    
    # 7. 保存结果
    save_start = time.time()
    results = {}
    
    # 保存普通分割掩码
    if save_mask:
        mask_path = image_path.replace('.', '_onnx_mask.')
        cv2.imwrite(mask_path, lane_mask)
        results['mask_path'] = mask_path
        print(f"💾 分割掩码已保存: {mask_path}")
    
    # 保存普通可视化结果
    if save_visualization:
        vis_img = create_visualization(img_bgr, lane_mask)
        vis_path = image_path.replace('.', '_onnx_result.')
        cv2.imwrite(vis_path, vis_img)
        results['visualization_path'] = vis_path
        print(f"💾 可视化结果已保存: {vis_path}")
    
    # 保存鸟瞰图
    if save_bird_eye and bird_eye_image is not None:
        bird_eye_path = image_path.replace('.', '_bird_eye.')
        cv2.imwrite(bird_eye_path, bird_eye_image)
        results['bird_eye_path'] = bird_eye_path
        print(f"💾 鸟瞰图已保存: {bird_eye_path}")
        
        # 保存带分割结果的鸟瞰图
        bird_eye_vis = create_visualization(bird_eye_image, bird_eye_mask)
        bird_eye_vis_path = image_path.replace('.', '_bird_eye_segmented.')
        cv2.imwrite(bird_eye_vis_path, bird_eye_vis)
        results['bird_eye_vis_path'] = bird_eye_vis_path
        print(f"💾 鸟瞰图分割可视化已保存: {bird_eye_vis_path}")
    
    # 保存控制地图
    if save_control_map and control_map is not None:
        control_map_path = image_path.replace('.', '_control_map.')
        cv2.imwrite(control_map_path, control_map)
        results['control_map_path'] = control_map_path
        print(f"💾 控制地图已保存: {control_map_path}")
    
    save_time = (time.time() - save_start) * 1000
    
    # 8. 性能分析
    times_dict = {
        "图片加载": load_time,
        "模型加载": model_load_time,
        "CPU预处理": preprocess_time,
        "ONNX推理": inference_time,
        "CPU后处理": postprocess_time,
        "透视变换": transform_time,
        "结果保存": save_time
    }
    
    print_performance_analysis(times_dict, input_data, model_path, provider)
    
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
        'provider': provider,
        'view_params': view_params
    })
    
    return results

# ---------------------------------------------------------------------------------
# --- 📱 命令行接口 ---
# ---------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="集成感知与环境建模的车道线分割推理工具")
    parser.add_argument("--input", "-i", required=True, help="输入图片路径")
    parser.add_argument("--output", "-o", help="输出可视化图片路径（可选）")
    parser.add_argument("--save_mask", help="保存分割掩码路径（可选）")
    parser.add_argument("--model", "-m", 
                       default="./weights/fast_scnn_custom_e2e_640x360_fixed_simplified.onnx",
                       help="ONNX模型路径")
    parser.add_argument("--calibration", "-c", help="相机标定文件路径")
    parser.add_argument("--provider", "-p", 
                       default="CPUExecutionProvider",
                       choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
                       help="ONNX执行提供者")
    parser.add_argument("--pixels_per_unit", type=int, default=20, help="每单位像素数 (默认: 20)")
    parser.add_argument("--margin_ratio", type=float, default=0.2, help="边距比例 (默认: 0.2)")
    parser.add_argument("--no_vis", action="store_true", help="不保存可视化结果，仅推理")
    parser.add_argument("--bird_eye", action="store_true", help="生成鸟瞰图（需要标定文件）")
    parser.add_argument("--save_control_map", action="store_true", help="保存控制地图")
    parser.add_argument("--preview", action="store_true", help="显示预览窗口")
    
    args = parser.parse_args()
    
    try:
        print("🧠 集成感知与环境建模的车道线分割推理工具")
        print("=" * 60)
        print("📝 功能: 车道线分割 + 透视变换 + 控制地图生成")
        print("=" * 60)
        
        # 检查可用的执行提供者
        available_providers = ort.get_available_providers()
        print(f"🔧 可用执行提供者: {available_providers}")
        
        # 验证输入文件存在
        if not os.path.exists(args.input):
            print(f"❌ 错误：输入图片不存在: {args.input}")
            sys.exit(1)
        
        # 验证标定文件（如果需要透视变换）
        if (args.bird_eye or args.save_control_map) and not args.calibration:
            print("❌ 错误：生成鸟瞰图或控制地图需要指定标定文件 (--calibration)")
            sys.exit(1)
        
        # 自动确定输出路径
        save_visualization = not args.no_vis
        save_mask = bool(args.save_mask)
        save_bird_eye = args.bird_eye
        save_control_map = args.save_control_map
        
        # 执行推理
        results = inference_with_bird_eye_view(
            image_path=args.input,
            model_path=args.model,
            calibration_path=args.calibration,
            provider=args.provider,
            pixels_per_unit=args.pixels_per_unit,
            margin_ratio=args.margin_ratio,
            save_visualization=save_visualization,
            save_mask=save_mask,
            save_bird_eye=save_bird_eye,
            save_control_map=save_control_map
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
        
        print("\n✅ 感知与环境建模完成！")
        print("🔧 此结果可用于后续的路径规划和车辆控制")
        
        if 'visualization_path' in results:
            print(f"🎨 分割可视化: {results['visualization_path']}")
        if 'mask_path' in results:
            print(f"🎭 分割掩码: {results['mask_path']}")
        if 'bird_eye_vis_path' in results:
            print(f"🦅 鸟瞰图分割: {results['bird_eye_vis_path']}")
        if 'control_map_path' in results:
            print(f"🗺️ 控制地图: {results['control_map_path']}")
            
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
