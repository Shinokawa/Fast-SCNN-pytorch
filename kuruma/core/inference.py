#!/usr/bin/env python3
"""
推理模块 - Atlas NPU推理会话和单图推理管道

包含：
- AtlasInferSession: Atlas NPU推理会话管理
- print_performance_analysis: 性能分析报告
- inference_single_image: 完整的单图推理管道
"""

import os
import time
import numpy as np
import cv2
from pathlib import Path

# 导入AIS-Bench接口
from ais_bench.infer.interface import InferSession

# 导入其他核心模块
from core.calibration import get_corrected_calibration, get_builtin_calibration
from core.preprocessing import preprocess_matched_resolution, postprocess_matched_resolution, create_visualization

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
                          preview_distance=30.0, max_speed=1000.0, min_speed=5.0,
                          ema_alpha=0.5, enable_smoothing=True):
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
        
        # 需要导入PerspectiveTransformer类
        from vision.transform import PerspectiveTransformer
        
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
            
            # 需要导入create_control_map函数
            from vision.path_planning import create_control_map
            
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
            # 需要导入save_path_data_json函数
            from vision.path_planning import save_path_data_json
            
            path_json_path = os.path.join(output_dir, f"{base_name}_path_data.json")
            save_path_data_json(path_data, path_json_path)
            results['path_json_path'] = path_json_path
            print(f"💾 路径数据已保存: {path_json_path}")
    
    # 7.5. 视觉控制算法（可选）
    control_result = None
    if enable_control and path_data is not None and view_params is not None:
        print("🚗 启动视觉横向误差控制算法...")
        control_start = time.time()
        
        # 需要导入VisualLateralErrorController类
        from control.visual_controller import VisualLateralErrorController
        
        # 初始化控制器
        controller = VisualLateralErrorController(
            steering_gain=steering_gain,
            base_pwm=int(base_speed),  # 重命名参数映射，转换为int
            curvature_damping=curvature_damping,
            preview_distance=preview_distance,
            max_pwm=int(max_speed),    # 重命名参数映射，转换为int
            min_pwm=int(min_speed),    # 重命名参数映射，转换为int
            ema_alpha=ema_alpha,       # EMA平滑系数
            enable_smoothing=enable_smoothing  # 是否启用平滑
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