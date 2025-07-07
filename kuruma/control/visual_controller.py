#!/usr/bin/env python3
"""
视觉控制器模块 - 基于视觉的横向误差控制算法

包含：
- VisualLateralErrorController: 基于视觉横向误差的比例-速度自适应差速控制算法
- 完整的PWM控制流程和可视化功能
"""

import numpy as np
import cv2
import time
import json
import logging
from core.logging_config import get_module_logger

# 导入标定模块
from core.calibration import get_corrected_calibration

# ---------------------------------------------------------------------------------
# --- 🚗 基于视觉横向误差的比例-速度自适应差速控制算法 ---
# ---------------------------------------------------------------------------------

logger = get_module_logger(__name__)

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
                 preview_distance=30.0, max_pwm=1000, min_pwm=100, 
                 ema_alpha=0.5, enable_smoothing=True):
        """
        初始化控制器参数
        
        参数：
            steering_gain: 转向增益 Kp（比例控制器增益）
            base_pwm: 基础PWM值（-1000到+1000范围）
            curvature_damping: 曲率阻尼系数（速度自适应参数）
            preview_distance: 预瞄距离（cm，控制点距离机器人的距离）
            max_pwm: 最大PWM值（-1000到+1000范围）
            min_pwm: 最小PWM值（-1000到+1000范围，用于前进时的最低速度）
            ema_alpha: EMA平滑系数（0-1，越大越灵敏，越小越平滑）
            enable_smoothing: 是否启用控制指令平滑
        """
        self.steering_gain = steering_gain
        self.base_pwm = base_pwm
        self.curvature_damping = curvature_damping
        self.preview_distance = preview_distance
        self.max_pwm = max_pwm
        self.min_pwm = min_pwm
        self.ema_alpha = ema_alpha
        self.enable_smoothing = enable_smoothing
        
        # EMA时间平滑状态 - 优化版本：只对输入信号进行平滑
        self.ema_lateral_error = None  # 对横向误差进行平滑（噪声源头）
        
        # 性能统计
        self.control_history = []
        
        logger.info(f"🚗 视觉横向误差控制器已初始化:")
        logger.info(f"   📐 转向增益: {steering_gain}")
        logger.info(f"   🏃 基础PWM: {base_pwm} (-1000~+1000)")
        logger.info(f"   🌊 曲率阻尼: {curvature_damping}")
        logger.info(f"   👁️ 预瞄距离: {preview_distance} cm")
        logger.info(f"   ⚡ PWM范围: {min_pwm} ~ {max_pwm} (支持双向旋转)")
        logger.info(f"   🔄 EMA平滑: {'启用' if enable_smoothing else '禁用'} (α={ema_alpha}) - 优化版本：输入信号平滑")
    
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
        完整的控制计算流程 - 输出PWM控制值 (优化版本：对输入信号进行平滑)
        
        参数：
            path_data: 路径规划数据
            view_params: 视图参数
            
        返回：
            control_result: 控制结果字典
        """
        # 模块一：计算原始横向误差
        raw_lateral_error, car_position, control_point = self.calculate_lateral_error(path_data, view_params)
        
        # EMA平滑优化：对输入信号（横向误差）进行平滑，而非输出PWM
        # 原因：lateral_error是噪声源头，先平滑它可以让后续所有计算都基于稳定输入
        if self.enable_smoothing:
            if self.ema_lateral_error is None:
                # 首次调用，直接使用原始值初始化
                self.ema_lateral_error = raw_lateral_error
                lateral_error = raw_lateral_error
            else:
                # 应用EMA平滑：S_t = α * Y_t + (1 - α) * S_{t-1}
                self.ema_lateral_error = (self.ema_alpha * raw_lateral_error + 
                                         (1 - self.ema_alpha) * self.ema_lateral_error)
                lateral_error = self.ema_lateral_error
        else:
            lateral_error = raw_lateral_error
        
        # 模块二：基于平滑后的lateral_error计算转向调整
        steering_adjustment = self.calculate_steering_adjustment(lateral_error)
        
        # 模块三：基于平滑后的lateral_error计算动态PWM
        dynamic_pwm = self.calculate_dynamic_pwm(lateral_error)
        
        # 最终指令合成 - 修正差速转向逻辑
        # 当lateral_error < 0时需要左转，应该右轮快左轮慢
        # 当lateral_error > 0时需要右转，应该左轮快右轮慢
        pwm_right = dynamic_pwm - steering_adjustment  # 右轮PWM
        pwm_left = dynamic_pwm + steering_adjustment   # 左轮PWM
        
        # 限制PWM值在-1000到+1000范围内（支持双向旋转）
        pwm_right = np.clip(pwm_right, -1000, 1000)
        pwm_left = np.clip(pwm_left, -1000, 1000)
        
        # 构建控制结果（基于平滑后的lateral_error计算得出）
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
            'speed_reduction_factor': self.base_pwm / dynamic_pwm if dynamic_pwm > 0 else 1.0,
            # EMA平滑状态信息（优化版本）
            'smoothing_enabled': self.enable_smoothing,
            'ema_alpha': self.ema_alpha,
            'raw_lateral_error': raw_lateral_error,  # 原始横向误差（用于对比分析）
            'smoothed_lateral_error': lateral_error,  # 平滑后横向误差
            'smoothing_effect': abs(raw_lateral_error - lateral_error) if self.enable_smoothing else 0.0  # 平滑效果量化
        }
        
        # 记录控制历史
        self.control_history.append(control_result.copy())
        
        return control_result
    
    def reset_ema_state(self):
        """
        重置EMA平滑状态（用于重新开始控制或紧急停车后恢复）
        优化版本：只重置横向误差的EMA状态
        """
        self.ema_lateral_error = None
        logger.info("🔄 EMA平滑状态已重置（优化版本：仅重置lateral_error平滑器）")
    
    def update_smoothing_params(self, ema_alpha=None, enable_smoothing=None):
        """
        动态更新EMA平滑参数（支持热更新）
        
        参数：
            ema_alpha: 新的EMA平滑系数
            enable_smoothing: 是否启用平滑
        """
        if ema_alpha is not None:
            self.ema_alpha = max(0.1, min(1.0, ema_alpha))  # 限制在0.1-1.0范围
            logger.info(f"🔄 EMA平滑系数已更新: α={self.ema_alpha}")
        
        if enable_smoothing is not None:
            old_state = self.enable_smoothing
            self.enable_smoothing = enable_smoothing
            if not enable_smoothing and old_state:
                self.reset_ema_state()  # 禁用平滑时重置状态
            logger.info(f"🔄 EMA平滑{'启用' if enable_smoothing else '禁用'}（优化版本：输入信号平滑）")

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
            logger.warning(f"⚠️ 无法获取机器人世界坐标: {e}")
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
        logger.info("\n" + "="*60)
        logger.info("🚗 基于视觉横向误差的差速控制分析")
        logger.info("="*60)
        
        # 基础信息
        logger.info(f"📍 机器人位置: ({control_result['car_position'][0]:.1f}, {control_result['car_position'][1]:.1f}) cm")
        if control_result['control_point']:
            logger.info(f"🎯 控制点位置: ({control_result['control_point'][0]:.1f}, {control_result['control_point'][1]:.1f}) cm")
        
        # 模块一：视觉误差感知
        logger.info(f"\n📱 模块一：视觉误差感知")
        logger.info(f"   横向误差: {control_result['lateral_error']:+.1f} cm")
        logger.info(f"   转向方向: {control_result['turn_direction'].upper()}")
        logger.info(f"   误差强度: {'高' if abs(control_result['lateral_error']) > 10 else '中' if abs(control_result['lateral_error']) > 5 else '低'}")
        
        # 模块二：比例转向控制
        logger.info(f"\n🎮 模块二：比例转向控制")
        logger.info(f"   转向调整: {control_result['steering_adjustment']:+.0f} PWM")
        logger.info(f"   控制增益: {self.steering_gain}")
        
        # 模块三：动态PWM自适应
        logger.info(f"\n⚡ 模块三：动态PWM自适应")
        logger.info(f"   基础PWM: {self.base_pwm:.0f}")
        logger.info(f"   动态PWM: {control_result['dynamic_pwm']:.0f}")
        logger.info(f"   PWM衰减: {control_result['pwm_reduction_factor']:.2f}x")
        logger.info(f"   曲率水平: {control_result['curvature_level']:.3f}")
        
        # 最终控制指令
        logger.info(f"\n🛞 最终差速PWM控制指令")
        logger.info(f"   左轮PWM: {control_result['pwm_left']:+.0f}")
        logger.info(f"   右轮PWM: {control_result['pwm_right']:+.0f}")
        logger.info(f"   PWM差值: {abs(control_result['pwm_right'] - control_result['pwm_left']):.0f}")
        logger.info(f"   可直接发送给底层驱动！")
        
        # 性能建议
        self._print_performance_recommendations(control_result)
    
    def _print_performance_recommendations(self, control_result):
        """打印性能建议"""
        logger.info(f"\n💡 性能分析与建议")
        
        error_abs = abs(control_result['lateral_error'])
        if error_abs > 15:
            logger.warning("   ⚠️ 横向误差较大，建议检查路径规划质量")
        elif error_abs < 2:
            logger.info("   ✅ 横向误差很小，路径跟踪良好")
        else:
            logger.info("   👍 横向误差在合理范围内")
        
        if control_result['curvature_level'] > 0.3:
            logger.warning("   🌊 进入高曲率路段，自动减速生效")
        elif control_result['curvature_level'] < 0.1:
            logger.info("   🛣️ 直线路段，保持较高速度")
        
        speed_diff = abs(control_result['speed_right'] - control_result['speed_left'])
        if speed_diff > 10:
            logger.warning("   🔄 大幅转向指令，注意机器人稳定性")
        elif speed_diff < 2:
            logger.info("   ➡️ 直行为主，转向调整轻微")

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