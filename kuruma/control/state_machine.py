#!/usr/bin/env python3
"""
状态机模块 - 管理实时推理中的巡线模式和避障模式

包含：
- CarState: 车辆状态枚举
- ObstacleAvoidanceStateMachine: 障碍物避障状态机
- 状态切换逻辑和避障动作执行
"""

import time
import threading
from enum import Enum
from typing import Dict, Optional, Callable, Any
import logging

# 导入统一日志配置
from core.logging_config import get_module_logger, log_system_initialization

class CarState(Enum):
    """车辆状态枚举"""
    LANE_FOLLOWING = "lane_following"  # 巡线模式
    OBSTACLE_DETECTED = "obstacle_detected"  # 检测到障碍物
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"  # 避障模式
    AVOIDANCE_REVERSE = "avoidance_reverse"  # 避障反向状态
    RETURNING_TO_LANE = "returning_to_lane"  # 返回巡线模式

class ObstacleAvoidanceStateMachine:
    """
    障碍物避障状态机
    
    管理车辆在不同状态间的切换：
    - 巡线模式：正常跟随车道线
    - 避障模式：执行固定的避障动作
    - 状态切换和时序控制
    """
    
    def __init__(self, 
                 avoidance_left_speed=400,
                 avoidance_right_speed=700,
                 avoidance_duration=2.0,
                 reverse_duration=2.0,
                 obstacle_detection_interval=10):
        """
        初始化状态机
        
        参数:
            avoidance_left_speed: 避障时左轮速度
            avoidance_right_speed: 避障时右轮速度
            avoidance_duration: 避障动作持续时间（秒）
            reverse_duration: 反向动作持续时间（秒）
            obstacle_detection_interval: 障碍物检测间隔（帧数）
        """
        self.current_state = CarState.LANE_FOLLOWING
        self.previous_state = None
        
        # 避障参数
        self.avoidance_left_speed = avoidance_left_speed
        self.avoidance_right_speed = avoidance_right_speed
        self.avoidance_duration = avoidance_duration
        self.reverse_duration = reverse_duration
        
        # 障碍物检测参数
        self.obstacle_detection_interval = obstacle_detection_interval
        self.frame_count = 0
        
        # 状态机计时器
        self.state_start_time = time.time()
        self.state_lock = threading.Lock()
        
        # 回调函数
        self.on_state_change = None
        self.car_controller = None
        
        # 日志
        self.logger = get_module_logger(__name__)
        
        # 记录初始化配置到日志
        config = {
            "避障左轮速度": avoidance_left_speed,
            "避障右轮速度": avoidance_right_speed,
            "避障持续时间": f"{avoidance_duration}s",
            "反向持续时间": f"{reverse_duration}s",
            "障碍物检测间隔": f"{obstacle_detection_interval}帧"
        }
        log_system_initialization("障碍物避障状态机", config)
    
    def set_car_controller(self, car_controller):
        """设置小车控制器"""
        self.car_controller = car_controller
        self.logger.info("🚗 小车控制器已设置")
    
    def set_state_change_callback(self, callback: Callable):
        """设置状态变化回调函数"""
        self.on_state_change = callback
        self.logger.info("📞 状态变化回调函数已设置")
    
    def get_current_state(self) -> CarState:
        """获取当前状态"""
        with self.state_lock:
            return self.current_state
    
    def is_obstacle_detection_frame(self) -> bool:
        """检查是否为障碍物检测帧"""
        return self.frame_count % self.obstacle_detection_interval == 0
    
    def update_frame_count(self):
        """更新帧计数"""
        self.frame_count += 1
    
    def _change_state(self, new_state: CarState):
        """内部状态切换方法"""
        with self.state_lock:
            if self.current_state != new_state:
                self.previous_state = self.current_state
                self.current_state = new_state
                self.state_start_time = time.time()
                
                self.logger.info(f"🔄 状态切换: {self.previous_state.value} → {new_state.value}")
                
                # 触发回调函数
                if self.on_state_change:
                    self.on_state_change(self.previous_state, new_state)
    
    def process_frame(self, obstacle_detected: bool = False, obstacle_result: Optional[Dict] = None) -> Dict:
        """
        处理每一帧的状态逻辑
        
        参数:
            obstacle_detected: 是否检测到障碍物
            obstacle_result: 障碍物检测结果
            
        返回:
            control_decision: 控制决策字典
        """
        current_time = time.time()
        time_in_state = current_time - self.state_start_time
        
        current_state = self.get_current_state()
        
        # 状态机主要逻辑
        if current_state == CarState.LANE_FOLLOWING:
            # 巡线模式：检查是否有障碍物
            if obstacle_detected and self.is_obstacle_detection_frame():
                num_obstacles = obstacle_result['num_obstacles'] if obstacle_result else 0
                self.logger.info(f"🚧 检测到障碍物: {num_obstacles}个")
                self._change_state(CarState.OBSTACLE_DETECTED)
                return self._get_control_decision()
            else:
                # 正常巡线，返回控制权给巡线算法
                return {
                    'mode': 'lane_following',
                    'override_control': False,
                    'left_speed': None,
                    'right_speed': None,
                    'message': '正常巡线模式'
                }
        
        elif current_state == CarState.OBSTACLE_DETECTED:
            # 障碍物检测状态：立即进入避障模式
            self._change_state(CarState.OBSTACLE_AVOIDANCE)
            self.logger.info("🚗 开始避障动作")
            return self._get_control_decision()
            
        elif current_state == CarState.OBSTACLE_AVOIDANCE:
            # 避障模式：执行固定避障动作
            if time_in_state >= self.avoidance_duration:
                self._change_state(CarState.AVOIDANCE_REVERSE)
                self.logger.info("🔄 避障完成，开始反向动作")
            return self._get_control_decision()
            
        elif current_state == CarState.AVOIDANCE_REVERSE:
            # 反向动作：执行反向动作
            if time_in_state >= self.reverse_duration:
                self._change_state(CarState.RETURNING_TO_LANE)
                self.logger.info("🔄 反向动作完成，返回巡线模式")
            return self._get_control_decision()
            
        elif current_state == CarState.RETURNING_TO_LANE:
            # 返回巡线：立即切换到巡线模式
            self._change_state(CarState.LANE_FOLLOWING)
            self.logger.info("✅ 已返回巡线模式")
            return self._get_control_decision()
        
        # 默认返回巡线模式
        return {
            'mode': 'lane_following',
            'override_control': False,
            'left_speed': None,
            'right_speed': None,
            'message': '默认巡线模式'
        }
    
    def _get_control_decision(self) -> Dict:
        """获取当前状态的控制决策"""
        current_state = self.get_current_state()
        
        if current_state == CarState.LANE_FOLLOWING:
            return {
                'mode': 'lane_following',
                'override_control': False,
                'left_speed': None,
                'right_speed': None,
                'message': '巡线模式'
            }
            
        elif current_state == CarState.OBSTACLE_DETECTED:
            return {
                'mode': 'obstacle_detected',
                'override_control': True,
                'left_speed': 0,
                'right_speed': 0,
                'message': '检测到障碍物，准备避障'
            }
            
        elif current_state == CarState.OBSTACLE_AVOIDANCE:
            return {
                'mode': 'obstacle_avoidance',
                'override_control': True,
                'left_speed': self.avoidance_left_speed,
                'right_speed': self.avoidance_right_speed,
                'message': f'避障模式 - 左轮{self.avoidance_left_speed}, 右轮{self.avoidance_right_speed}'
            }
            
        elif current_state == CarState.AVOIDANCE_REVERSE:
            return {
                'mode': 'avoidance_reverse',
                'override_control': True,
                'left_speed': -self.avoidance_right_speed,  # 反向：左右轮速度交换并取负
                'right_speed': -self.avoidance_left_speed,
                'message': f'反向避障模式 - 左轮{-self.avoidance_right_speed}, 右轮{-self.avoidance_left_speed}'
            }
            
        elif current_state == CarState.RETURNING_TO_LANE:
            return {
                'mode': 'returning_to_lane',
                'override_control': False,
                'left_speed': None,
                'right_speed': None,
                'message': '返回巡线模式'
            }
        
        # 默认情况
        return {
            'mode': 'unknown',
            'override_control': False,
            'left_speed': None,
            'right_speed': None,
            'message': '未知状态'
        }
    
    def force_stop(self):
        """强制停止并返回巡线模式"""
        with self.state_lock:
            self.logger.info("🛑 强制停止状态机")
            self._change_state(CarState.LANE_FOLLOWING)
            
            # 如果有小车控制器，发送停止指令
            if self.car_controller and hasattr(self.car_controller, 'stop'):
                try:
                    self.car_controller.stop()
                    self.logger.info("🚗 小车已停止")
                except Exception as e:
                    self.logger.error(f"❌ 停止小车时出错: {e}")
    
    def get_state_info(self) -> Dict:
        """获取状态机信息"""
        current_time = time.time()
        time_in_state = current_time - self.state_start_time
        
        return {
            'current_state': self.current_state.value,
            'previous_state': self.previous_state.value if self.previous_state else None,
            'time_in_state': time_in_state,
            'frame_count': self.frame_count,
            'next_obstacle_detection_frame': self.obstacle_detection_interval - (self.frame_count % self.obstacle_detection_interval)
        }
    
    def print_state_info(self):
        """记录状态机信息到日志"""
        info = self.get_state_info()
        self.logger.info(f"🎯 状态机信息:")
        self.logger.info(f"   当前状态: {info['current_state']}")
        self.logger.info(f"   上一状态: {info['previous_state']}")
        self.logger.info(f"   状态持续时间: {info['time_in_state']:.2f}秒")
        self.logger.info(f"   帧计数: {info['frame_count']}")
        self.logger.info(f"   下次障碍物检测: {info['next_obstacle_detection_frame']}帧后") 