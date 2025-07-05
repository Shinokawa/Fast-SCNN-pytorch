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

# 导入小车控制模块
try:
    from car_controller_simple import SimpleCarController
    CAR_CONTROLLER_AVAILABLE = True
    print("✅ 小车控制模块加载成功")
except ImportError:
    print("⚠️ 警告：未找到car_controller_simple模块，串口控制功能不可用")
    CAR_CONTROLLER_AVAILABLE = False

# 导入核心模块
from core.calibration import get_builtin_calibration, get_corrected_calibration
from core.preprocessing import preprocess_matched_resolution, postprocess_matched_resolution, create_visualization
from core.inference import AtlasInferSession, print_performance_analysis, inference_single_image

# 导入视觉处理模块
from vision.transform import PerspectiveTransformer
from vision.path_planning import PathPlanner, create_control_map, add_grid_to_control_map, visualize_path_on_control_map, world_to_pixels, save_path_data_json

# 导入控制模块
from control.visual_controller import VisualLateralErrorController

# 导入接口模块
from interfaces.realtime import setup_logging, realtime_inference
from interfaces.web_interface import create_web_app, start_web_server, web_data, web_data_lock





# ---------------------------------------------------------------------------------
# --- 🧠 Atlas NPU推理会话 ---
# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# --- 📱 主推理函数 (与Atlas流程完全一致) ---
# ---------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# --- 控制模块已迁移至 control/visual_controller.py ---
# ---------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------
# --- 🚀 实时推理模块已迁移到interfaces/realtime.py ---
# ---------------------------------------------------------------------------------

# setup_logging和realtime_inference函数已迁移到interfaces/realtime.py

# ---------------------------------------------------------------------------------
# --- 🌐 Web界面模块已迁移到interfaces/web_interface.py ---
# ---------------------------------------------------------------------------------

# 全局车辆控制器
car_controller = None
control_thread = None
control_enabled = False

# WEB_TEMPLATE和Web函数已迁移到interfaces/web_interface.py

# ---------------------------------------------------------------------------------
# --- 📱 命令行接口 ---
# ---------------------------------------------------------------------------------

def main():
    global car_controller
    
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
    parser.add_argument("--max_speed", type=float, default=1000.0, help="最大PWM值 -1000~+1000 (默认: 1000)")
    parser.add_argument("--min_speed", type=float, default=100.0, help="最小PWM值，前进时最低速度 (默认: 100)")
    
    # EMA时间平滑参数
    parser.add_argument("--ema_alpha", type=float, default=0.5, help="EMA平滑系数 (0.1-1.0, 默认: 0.5)")
    parser.add_argument("--enable_smoothing", action="store_true", default=True, help="启用控制指令EMA平滑 (默认: 启用)")
    parser.add_argument("--disable_smoothing", action="store_true", help="禁用控制指令EMA平滑")
    
    # Web界面和GUI选项
    parser.add_argument("--web", action="store_true", help="启用Web界面")
    parser.add_argument("--web_port", type=int, default=5000, help="Web界面端口 (默认: 5000)")
    parser.add_argument("--no_gui", action="store_true", help="无GUI模式（不显示OpenCV窗口，仅输出结果）")
    
    # 串口控制选项
    parser.add_argument("--enable_serial", action="store_true", help="启用串口控制功能")
    parser.add_argument("--serial_port", default="/dev/ttyAMA0", help="串口设备路径 (默认: /dev/ttyAMA0)")
    parser.add_argument("--auto_connect_serial", action="store_true", help="启动时自动连接串口")
    parser.add_argument("--auto_start_driving", action="store_true", help="连接串口后自动开始行驶（谨慎使用）")
    
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
                # 初始化Web数据中的串口配置
                with web_data_lock:
                    web_data['serial_port'] = args.serial_port
                    web_data['serial_enabled'] = args.enable_serial
                
                web_server = start_web_server(args.web_port)
                time.sleep(2)  # 等待服务器启动
                if args.no_gui:
                    print("💡 提示：无GUI模式下，请通过Web界面查看实时状态")
            
            # 串口控制初始化
            if args.enable_serial and CAR_CONTROLLER_AVAILABLE:
                print("🚗 串口控制功能已启用")
                if args.auto_connect_serial:
                    print(f"🔌 自动连接串口: {args.serial_port}")
                    try:
                        car_controller = SimpleCarController(port=args.serial_port)
                        if car_controller.connect():
                            print("✅ 串口连接成功")
                            with web_data_lock:
                                web_data['serial_connected'] = True
                                if args.auto_start_driving:
                                    web_data['car_driving'] = True
                                    web_data['control_enabled'] = True
                                    print("🚀 自动启动行驶模式")
                        else:
                            print("❌ 串口连接失败")
                    except Exception as e:
                        print(f"❌ 串口初始化错误: {e}")
                else:
                    print("💡 提示：请通过Web界面连接串口或添加 --auto_connect_serial 参数")
            elif args.enable_serial and not CAR_CONTROLLER_AVAILABLE:
                print("⚠️ 警告：串口控制功能已启用，但car_controller_simple模块不可用")
            else:
                print("⚠️ 串口控制功能未启用，如需使用请添加 --enable_serial 参数")
            
            # 处理EMA平滑参数
            enable_smoothing = args.enable_smoothing and not args.disable_smoothing
            ema_alpha = max(0.1, min(1.0, args.ema_alpha))  # 限制在0.1-1.0范围
            
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
                full_image_bird_eye=not args.no_full_image_bird_eye,  # 反转逻辑
                edge_computing=args.edge_computing,
                pixels_per_unit=args.pixels_per_unit,
                margin_ratio=args.margin_ratio,
                ema_alpha=ema_alpha,
                enable_smoothing=enable_smoothing
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
        # 处理EMA平滑参数
        enable_smoothing = args.enable_control and args.enable_smoothing and not args.disable_smoothing
        ema_alpha = max(0.1, min(1.0, args.ema_alpha))  # 限制在0.1-1.0范围
        
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
            full_image_bird_eye=not args.no_full_image_bird_eye,
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
            min_speed=args.min_speed,
            ema_alpha=ema_alpha,
            enable_smoothing=enable_smoothing
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

# save_path_data_json函数已迁移到vision/path_planning.py

if __name__ == "__main__":
    main()
    