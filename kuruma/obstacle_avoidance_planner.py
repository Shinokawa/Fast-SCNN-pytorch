#!/usr/bin/env python3
"""
基于传统CV的BEV障碍物检测与动态避障路径规划系统
简化版本：原地转向避障，移除样条插值
"""

import numpy as np
import cv2
import json
import time
from typing import List, Tuple, Optional

class ObstacleAvoidancePlanner:
    def __init__(self, config):
        """
        config = {
            "safety_margin": 40, # 绕行时离障碍物的安全距离（像素）
            "path_resolution": 10, # 重规划路径点的密度
            "lookahead_distance": 100, # 向前看多远以寻找汇合点
            "early_turn_distance": 120, # 提前多少距离开始转向（像素），增大
            "smoothness_factor": 0.3, # 路径平滑度因子 (0-1)，降低以增加平滑度
            "max_steering_angle": 15, # 最大转向角度（度），大幅降低
            "detection_radius": 30, # 障碍物检测半径（像素），预测性检测
            "avoidance_arc_length": 80 # 避障弧形长度（像素）
        }
        """
        self.config = config

    def replan_path(self, original_path_world, obstacles_contours, view_params):
        """
        输入：
            original_path_world: 原始路径点（世界坐标）
            obstacles_contours: 障碍物轮廓（像素坐标）
            view_params: 鸟瞰图参数
        输出：
            new_path_world: 绕行障碍物的新路径（世界坐标）
        """
        if not obstacles_contours or not original_path_world:
            return original_path_world
        
        # 1. 坐标转换
        original_path_pixels = self._world_to_pixels(original_path_world, view_params)
        
        # 2. 预测性碰撞检测（不是精确重叠，而是前方区域扫描）
        threat_idx = self._find_threat_point(original_path_pixels, obstacles_contours)
        if threat_idx is None:
            return original_path_world
        
        # 3. 提前转向点计算（大幅提前）
        early_turn_distance = self.config.get('early_turn_distance', 200)  # 增大提前距离
        detection_radius = self.config.get('detection_radius', 50)         # 增大检测半径
        
        # 基于实际路径点间距计算
        if len(original_path_pixels) > 1:
            dists = [np.linalg.norm(np.array(original_path_pixels[i]) - np.array(original_path_pixels[i-1])) 
                    for i in range(1, len(original_path_pixels))]
            avg_dist = max(np.mean(dists), 1)
        else:
            avg_dist = 1
        
        # 提前转向点：在威胁点前很远就开始转向
        n_points_back = max(5, int((early_turn_distance + detection_radius) / avg_dist))
        early_turn_idx = max(0, threat_idx - n_points_back)
        
        print(f"[Avoidance] 威胁点idx={threat_idx}, 提前转向idx={early_turn_idx}, 提前距离={n_points_back*avg_dist:.1f}px")
        
        # 4. 生成平滑弧形避障路径
        smooth_path = self._generate_smooth_arc_path(
            original_path_pixels, obstacles_contours, threat_idx, early_turn_idx, view_params)
        
        if not smooth_path:
            print("[Avoidance] 避障路径生成失败")
            return None
        
        # 5. 转换回世界坐标
        new_path_world = self._pixels_to_world(smooth_path, view_params)
        print(f"[Avoidance] 避障路径生成成功，点数: {len(new_path_world)}")
        return new_path_world

    def _find_threat_point(self, path_pixels, obstacles_contours):
        """预测性威胁检测：提前检测，立即反应"""
        detection_radius = self.config.get('detection_radius', 50)  # 增大检测半径
        
        for idx, pt in enumerate(path_pixels):
            pt_int = (int(pt[0]), int(pt[1]))
            
            # 对每个障碍物检查是否在威胁范围内
            for contour in obstacles_contours:
                # 计算路径点到障碍物轮廓的最短距离
                dist = abs(cv2.pointPolygonTest(contour, pt_int, True))
                
                # 如果距离小于检测半径，认为是威胁点（更早检测）
                if dist < detection_radius:
                    print(f"[Avoidance] 提前检测到威胁: idx={idx}, 距离障碍物={dist:.1f}px, 立即避障")
                    return idx
        
        return None

    def _generate_smooth_arc_path(self, path_pixels, obstacles, threat_idx, early_turn_idx, view_params):
        """生成原地转向避障路径（简化版本，无需样条插值）"""
        # 1. 确定绕行方向
        avoidance_direction = self._determine_avoidance_direction(path_pixels[threat_idx], obstacles)
        print(f"[Avoidance] 绕行方向: {avoidance_direction}")
        
        # 2. 生成分段直线控制点
        control_points = self._generate_arc_control_points(
            path_pixels, obstacles, threat_idx, early_turn_idx, avoidance_direction, view_params)
        
        if len(control_points) < 4:
            print(f"[Avoidance] 控制点不足: {len(control_points)}")
            return None
        
        # 3. 直接返回控制点，无需样条插值
        print(f"[Avoidance] 原地转向路径生成完成，总点数: {len(control_points)}")
        return control_points

    def _generate_arc_control_points(self, path_pixels, obstacles, threat_idx, early_turn_idx, direction, view_params):
        """生成原地转向避障路径（停车→转向→前进→转回）"""
        safety_margin = self.config.get('safety_margin', 40)
        
        # 确保索引安全
        if early_turn_idx >= len(path_pixels) or threat_idx >= len(path_pixels):
            return path_pixels
        
        print(f"[Avoidance] 生成原地转向避障路径: early_idx={early_turn_idx}, threat_idx={threat_idx}")
        
        # 1. 从view_params获取鸟瞰图的实际尺寸
        min_x, min_y, max_x, max_y = view_params['view_bounds']
        pixels_per_unit = view_params['pixels_per_unit']
        
        # 计算鸟瞰图像素尺寸
        width = int((max_x - min_x) * pixels_per_unit)
        height = int((max_y - min_y) * pixels_per_unit)
        
        # 车辆位置：鸟瞰图底部中央
        vehicle_start_pixel = np.array([width // 2, height - 1])
        vehicle_direction = np.array([0, -1])  # 垂直向上
        
        print(f"[Avoidance] 鸟瞰图尺寸: {width}x{height}, 车辆起点: {vehicle_start_pixel}")
        print(f"[Avoidance] 原地转向避障方向: {direction}")
        
        control_points = []
        
        # 2. 添加到停车点的直线段
        if early_turn_idx < len(path_pixels):
            stop_point = np.array(path_pixels[early_turn_idx])
        else:
            # 计算停车点位置
            stop_distance = early_turn_idx * 8
            stop_point = vehicle_start_pixel + vehicle_direction * stop_distance
        
        # 从车辆起点到停车点的直线
        stop_distance = np.linalg.norm(stop_point - vehicle_start_pixel)
        num_approach_points = max(3, int(stop_distance / 15))
        
        for i in range(num_approach_points + 1):
            progress = i / num_approach_points
            approach_point = vehicle_start_pixel + (stop_point - vehicle_start_pixel) * progress
            control_points.append(approach_point.tolist())
        
        print(f"[Avoidance] 接近段点数: {num_approach_points + 1}")
        
        # 3. 原地转向阶段（固定左转90度）
        turn_angle_deg = 90  # 固定左转90度，转向更充分
        turn_angle_rad = np.radians(turn_angle_deg)
        
        # 转向后的前进方向（左转90度）
        cos_angle = np.cos(turn_angle_rad)
        sin_angle = np.sin(turn_angle_rad)
        
        # 原方向是 [0, -1]（向上），左转90度后变为 [-1, 0]（向左）
        new_direction = np.array([
            vehicle_direction[0] * cos_angle - vehicle_direction[1] * sin_angle,
            vehicle_direction[0] * sin_angle + vehicle_direction[1] * cos_angle
        ])
        
        print(f"[Avoidance] 原地左转90度, 新方向: {new_direction}")
        
        # 4. 转向后向左前进阶段
        avoidance_distance = safety_margin * 4  # 增大避障距离
        num_forward_points = max(10, int(avoidance_distance / 8))
        
        for i in range(1, num_forward_points + 1):
            forward_distance = (avoidance_distance / num_forward_points) * i
            forward_point = stop_point + new_direction * forward_distance
            control_points.append(forward_point.tolist())
        
        print(f"[Avoidance] 向左前进段: {num_forward_points}点, 距离{avoidance_distance}像素")
        
        # 5. 原地转回（右转90度回到原方向）
        last_forward_point = np.array(control_points[-1])
        
        # 右转90度回到原方向
        return_turn_angle = -turn_angle_rad  # 右转90度
        return_cos = np.cos(return_turn_angle)
        return_sin = np.sin(return_turn_angle)
        
        # 转回后的方向：[-1, 0] 右转90度 → [0, -1]（向上）
        final_direction = np.array([
            new_direction[0] * return_cos - new_direction[1] * return_sin,
            new_direction[0] * return_sin + new_direction[1] * return_cos
        ])
        
        print(f"[Avoidance] 原地右转90度回到原方向, 最终方向: {final_direction}")
        
        # 6. 转回后继续向前
        continue_distance = 200  # 增大继续前进的距离
        num_continue_points = int(continue_distance / 10)
        
        for i in range(1, num_continue_points + 1):
            continue_dist = (continue_distance / num_continue_points) * i
            continue_point = last_forward_point + final_direction * continue_dist
            control_points.append(continue_point.tolist())
        
        print(f"[Avoidance] 继续向前段: {num_continue_points}点, 距离{continue_distance}像素")
        print(f"[Avoidance] 原地转向路径总点数: {len(control_points)}")
        
        return control_points

    def _determine_avoidance_direction(self, threat_point, obstacles):
        """确定绕行方向 - 固定左转策略"""
        print(f"[Avoidance] 威胁点位置: {threat_point}")
        
        # 固定策略：始终左转避障
        # 简单、可预测、稳定
        direction = "left"
        
        print(f"[Avoidance] 固定左转避障策略")
        return direction

    def _world_to_pixels(self, world_points, view_params):
        min_x, min_y, _, _ = view_params['view_bounds']
        pixels_per_unit = view_params['pixels_per_unit']
        pixel_points = []
        for wx, wy in world_points:
            px = (wx - min_x) * pixels_per_unit
            py = (wy - min_y) * pixels_per_unit
            pixel_points.append([px, py])
        return pixel_points

    def _pixels_to_world(self, pixel_points, view_params):
        min_x, min_y, _, _ = view_params['view_bounds']
        pixels_per_unit = view_params['pixels_per_unit']
        world_points = []
        for px, py in pixel_points:
            wx = min_x + px / pixels_per_unit
            wy = min_y + py / pixels_per_unit
            world_points.append((wx, wy))
        return world_points

    def _find_collision_point(self, path_pixels, obstacles_contours):
        """保留原方法以兼容（但推荐使用_find_threat_point）"""
        for idx, pt in enumerate(path_pixels):
            pt_int = (int(pt[0]), int(pt[1]))
            for contour in obstacles_contours:
                if cv2.pointPolygonTest(contour, pt_int, False) >= 0:
                    return idx
        return None

    def _find_rejoin_point(self, path_pixels, obstacles_contours, threat_idx):
        """寻找汇合点：距离障碍物足够远的路径点"""
        safety_distance = self.config.get('safety_margin', 40) + 20  # 额外安全距离
        
        start_search = min(threat_idx + 5, len(path_pixels))  # 从威胁点后开始搜索
        
        for idx in range(start_search, len(path_pixels)):
            pt = path_pixels[idx]
            pt_int = (int(pt[0]), int(pt[1]))
            
            # 检查是否远离所有障碍物
            safe = True
            for contour in obstacles_contours:
                dist = abs(cv2.pointPolygonTest(contour, pt_int, True))
                if dist < safety_distance:
                    safe = False
                    break
            
            if safe:
                return idx
        
        return None

class AvoidanceController:
    """专门用于避障的控制器，支持原地转向和高增益控制"""
    
    def __init__(self):
        # 避障专用控制参数
        self.turn_pwm = 800        # 原地转向PWM（高增益）
        self.forward_pwm = 600     # 避障前进PWM
        self.stop_pwm = 0          # 停车PWM
        
    def compute_avoidance_control(self, path_data, current_waypoint_idx=0):
        """
        计算避障控制信号
        返回格式与标准控制器一致：
        {'pwm_left': int, 'pwm_right': int, 'lateral_error': float, 
         'car_position': tuple, 'control_point': tuple, 'action': str}
        """
        waypoints = path_data.get('waypoints', [])
        if not waypoints or current_waypoint_idx >= len(waypoints):
            return {
                'pwm_left': 0, 
                'pwm_right': 0, 
                'lateral_error': 0.0,
                'car_position': None,
                'control_point': None,
                'action': 'stop'
            }
        
        # 根据路径位置判断避障阶段
        total_waypoints = len(waypoints)
        progress = current_waypoint_idx / total_waypoints
        
        print(f"[AvoidanceControl] 避障进度: {current_waypoint_idx}/{total_waypoints} ({progress:.2%})")
        
        # 当前目标点
        current_waypoint = waypoints[min(current_waypoint_idx, len(waypoints)-1)]
        
        # 阶段1：接近停车点（0-20%）
        if progress < 0.2:
            return {
                'pwm_left': 300,
                'pwm_right': 300,
                'lateral_error': 0.0,
                'car_position': (0, 0),  # 车辆起点
                'control_point': current_waypoint,
                'action': 'approach_stop'
            }
        
        # 阶段2：原地左转90度（20-25%）
        elif progress < 0.25:
            return {
                'pwm_left': -self.turn_pwm,   # 左轮反转
                'pwm_right': self.turn_pwm,   # 右轮正转
                'lateral_error': 0.0,
                'car_position': current_waypoint,
                'control_point': current_waypoint,
                'action': 'turn_left_90'
            }
        
        # 阶段3：向左前进（25-75%）
        elif progress < 0.75:
            return {
                'pwm_left': self.forward_pwm,
                'pwm_right': self.forward_pwm,
                'lateral_error': 0.0,
                'car_position': current_waypoint,
                'control_point': current_waypoint,
                'action': 'move_left'
            }
        
        # 阶段4：原地右转90度回到原方向（75-80%）
        elif progress < 0.8:
            return {
                'pwm_left': self.turn_pwm,    # 左轮正转
                'pwm_right': -self.turn_pwm,  # 右轮反转
                'lateral_error': 0.0,
                'car_position': current_waypoint,
                'control_point': current_waypoint,
                'action': 'turn_right_90'
            }
        
        # 阶段5：继续向前（80-100%）
        else:
            return {
                'pwm_left': self.forward_pwm,
                'pwm_right': self.forward_pwm,
                'lateral_error': 0.0,
                'car_position': current_waypoint,
                'control_point': current_waypoint,
                'action': 'continue_forward'
            } 