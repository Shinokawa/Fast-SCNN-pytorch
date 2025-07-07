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
    OBSTACLE_AVOIDANCE_STAGE1 = "obstacle_avoidance_stage1"  # 避障第一段
    OBSTACLE_AVOIDANCE_STAGE2 = "obstacle_avoidance_stage2"  # 避障第二段
    OBSTACLE_AVOIDANCE_STAGE3 = "obstacle_avoidance_stage3"  # 避障第三段
    RETURNING_TO_LANE = "returning_to_lane"  # 返回巡线模式
    RED_LIGHT_WAITING = "red_light_waiting"  # 红灯等待状态

class ObstacleAvoidanceStateMachine:
    """
    障碍物避障状态机
    
    管理车辆在不同状态间的切换：
    - 巡线模式：正常跟随车道线
    - 三段避障模式：执行三段不同的避障动作
    - 状态切换和时序控制
    """
    
    def __init__(self, 
                 # 三段避障参数：每段有左轮PWM、右轮PWM、持续时间
                 stage1_left_speed=400,
                 stage1_right_speed=700,
                 stage1_duration=1.5,
                 stage2_left_speed=-300,
                 stage2_right_speed=600,
                 stage2_duration=2.0,
                 stage3_left_speed=500,
                 stage3_right_speed=200,
                 stage3_duration=1.5,
                 obstacle_detection_interval=10,
                 traffic_light_detection_interval=10):
        """
        初始化状态机
        
        参数:
            stage1_left_speed: 第一段左轮速度
            stage1_right_speed: 第一段右轮速度 
            stage1_duration: 第一段持续时间（秒）
            stage2_left_speed: 第二段左轮速度
            stage2_right_speed: 第二段右轮速度
            stage2_duration: 第二段持续时间（秒）
            stage3_left_speed: 第三段左轮速度
            stage3_right_speed: 第三段右轮速度
            stage3_duration: 第三段持续时间（秒）
            obstacle_detection_interval: 障碍物检测间隔（帧数）
            traffic_light_detection_interval: 交通灯检测间隔（帧数）
        """
        self.current_state = CarState.LANE_FOLLOWING
        self.previous_state = None
        
        # 三段避障参数
        self.stage1_left_speed = stage1_left_speed
        self.stage1_right_speed = stage1_right_speed
        self.stage1_duration = stage1_duration
        
        self.stage2_left_speed = stage2_left_speed
        self.stage2_right_speed = stage2_right_speed
        self.stage2_duration = stage2_duration
        
        self.stage3_left_speed = stage3_left_speed
        self.stage3_right_speed = stage3_right_speed
        self.stage3_duration = stage3_duration
        
        # 障碍物检测参数
        self.obstacle_detection_interval = obstacle_detection_interval
        self.frame_count = 0
        
        # 交通灯检测参数
        self.traffic_light_detection_interval = traffic_light_detection_interval
        
        # 状态机计时器
        self.state_start_time = time.time()
        self.state_lock = threading.Lock()
        
        # 🚀 防死循环保护机制
        self.last_avoidance_time = 0  # 上次避障完成的时间
        self.avoidance_cooldown = 5.0  # 避障冷却时间（秒）
        self.consecutive_avoidance_count = 0
        self.max_consecutive_avoidances = 3  # 最大连续避障次数
        self.avoidance_reset_time = 10.0  # 避障计数重置时间（秒）
        
        # 回调函数
        self.on_state_change = None
        self.car_controller = None
        
        # 日志
        self.logger = get_module_logger(__name__)
        
        # 记录初始化配置到日志
        config = {
            "第一段": f"左轮{stage1_left_speed}, 右轮{stage1_right_speed}, 时长{stage1_duration}s",
            "第二段": f"左轮{stage2_left_speed}, 右轮{stage2_right_speed}, 时长{stage2_duration}s", 
            "第三段": f"左轮{stage3_left_speed}, 右轮{stage3_right_speed}, 时长{stage3_duration}s",
            "障碍物检测间隔": f"{obstacle_detection_interval}帧",
            "交通灯检测间隔": f"{traffic_light_detection_interval}帧",
            "避障冷却时间": f"{self.avoidance_cooldown}s",
            "最大连续避障次数": self.max_consecutive_avoidances,
            "避障计数重置时间": f"{self.avoidance_reset_time}s"
        }
        log_system_initialization("三段避障状态机", config)
    
    def update_avoidance_params(self, **kwargs):
        """
        动态更新避障参数（用于Web界面实时调整）
        
        参数:
            stage1_left_speed, stage1_right_speed, stage1_duration
            stage2_left_speed, stage2_right_speed, stage2_duration
            stage3_left_speed, stage3_right_speed, stage3_duration
        """
        updated_params = []
        
        # 第一段参数
        if 'stage1_left_speed' in kwargs:
            self.stage1_left_speed = kwargs['stage1_left_speed']
            updated_params.append(f"第一段左轮={self.stage1_left_speed}")
        if 'stage1_right_speed' in kwargs:
            self.stage1_right_speed = kwargs['stage1_right_speed']
            updated_params.append(f"第一段右轮={self.stage1_right_speed}")
        if 'stage1_duration' in kwargs:
            self.stage1_duration = kwargs['stage1_duration']
            updated_params.append(f"第一段时长={self.stage1_duration}s")
            
        # 第二段参数
        if 'stage2_left_speed' in kwargs:
            self.stage2_left_speed = kwargs['stage2_left_speed']
            updated_params.append(f"第二段左轮={self.stage2_left_speed}")
        if 'stage2_right_speed' in kwargs:
            self.stage2_right_speed = kwargs['stage2_right_speed']
            updated_params.append(f"第二段右轮={self.stage2_right_speed}")
        if 'stage2_duration' in kwargs:
            self.stage2_duration = kwargs['stage2_duration']
            updated_params.append(f"第二段时长={self.stage2_duration}s")
            
        # 第三段参数
        if 'stage3_left_speed' in kwargs:
            self.stage3_left_speed = kwargs['stage3_left_speed']
            updated_params.append(f"第三段左轮={self.stage3_left_speed}")
        if 'stage3_right_speed' in kwargs:
            self.stage3_right_speed = kwargs['stage3_right_speed']
            updated_params.append(f"第三段右轮={self.stage3_right_speed}")
        if 'stage3_duration' in kwargs:
            self.stage3_duration = kwargs['stage3_duration']
            updated_params.append(f"第三段时长={self.stage3_duration}s")
        
        if updated_params:
            self.logger.info(f"🔧 避障参数已更新: {', '.join(updated_params)}")
    
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
    
    def is_traffic_light_detection_frame(self) -> bool:
        """检查是否为交通灯检测帧"""
        return self.frame_count % self.traffic_light_detection_interval == 0
    
    def update_frame_count(self):
        """更新帧计数"""
        self.frame_count += 1
    
    def is_avoidance_allowed(self) -> bool:
        """
        检查是否允许进行避障（防死循环保护）
        
        返回:
            bool: True表示允许避障，False表示被保护机制阻止
        """
        current_time = time.time()
        
        # 检查避障冷却时间
        time_since_last_avoidance = current_time - self.last_avoidance_time
        if time_since_last_avoidance < self.avoidance_cooldown:
            self.logger.warning(f"🛡️ 避障冷却中，剩余{self.avoidance_cooldown - time_since_last_avoidance:.1f}秒")
            return False
        
        # 检查连续避障次数限制
        if self.consecutive_avoidance_count >= self.max_consecutive_avoidances:
            self.logger.warning(f"🛡️ 连续避障次数达到上限({self.max_consecutive_avoidances})，拒绝避障")
            return False
        
        return True
    
    def update_avoidance_tracking(self):
        """更新避障跟踪信息（记录避障开始）"""
        # 记录避障开始
        self.consecutive_avoidance_count += 1
        self.logger.info(f"🚧 开始第{self.consecutive_avoidance_count}次三段避障")
    
    def _check_and_reset_avoidance_count(self):
        """检查并重置避障计数（如果需要）"""
        current_time = time.time()
        
        # 如果距离上次避障时间超过重置时间，重置计数
        if (self.last_avoidance_time > 0 and 
            current_time - self.last_avoidance_time > self.avoidance_reset_time and
            self.consecutive_avoidance_count > 0):
            self.logger.info(f"🔄 避障计数重置：{self.consecutive_avoidance_count} → 0")
            self.consecutive_avoidance_count = 0
    
    def complete_avoidance_cycle(self):
        """完成一次避障循环"""
        self.last_avoidance_time = time.time()
        self.logger.info(f"✅ 三段避障循环完成，进入{self.avoidance_cooldown}秒冷却期")
    
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
    
    def process_frame(self, obstacle_detected: bool = False, obstacle_result: Optional[Dict] = None, 
                     traffic_light_detected: bool = False, traffic_light_status: str = "unknown") -> Dict:
        """
        处理每一帧的状态逻辑
        
        参数:
            obstacle_detected: 是否检测到障碍物
            obstacle_result: 障碍物检测结果
            traffic_light_detected: 是否检测到交通灯
            traffic_light_status: 交通灯状态 ("red", "green", "unknown")
            
        返回:
            control_decision: 控制决策字典
        """
        current_time = time.time()
        time_in_state = current_time - self.state_start_time
        
        current_state = self.get_current_state()
        
        # 状态机主要逻辑
        if current_state == CarState.LANE_FOLLOWING:
            # 巡线模式：优先检查红灯，然后检查障碍物
            
            # 🚦 检查红灯（优先级最高）
            if (traffic_light_detected and traffic_light_status == "red" and 
                self.is_traffic_light_detection_frame()):
                self.logger.info("🚦 检测到红灯，进入等待状态")
                self._change_state(CarState.RED_LIGHT_WAITING)
                return self._get_control_decision()
            
            # 🚧 检查障碍物
            elif obstacle_detected and self.is_obstacle_detection_frame():
                num_obstacles = obstacle_result['num_obstacles'] if obstacle_result else 0
                self.logger.info(f"🚧 检测到障碍物: {num_obstacles}个")
                
                # 🛡️ 应用防死循环保护机制
                if self.is_avoidance_allowed():
                    self.update_avoidance_tracking()
                    self._change_state(CarState.OBSTACLE_DETECTED)
                    return self._get_control_decision()
                else:
                    # 被保护机制阻止，继续巡线但记录警告
                    self.logger.warning("🛡️ 避障被保护机制阻止，继续巡线模式")
                    return {
                        'mode': 'lane_following_protected',
                        'override_control': False,
                        'left_speed': None,
                        'right_speed': None,
                        'message': '巡线模式(避障保护激活)'
                    }
            else:
                # 检查并重置避障计数（如果需要）
                self._check_and_reset_avoidance_count()
                
                # 正常巡线，返回控制权给巡线算法
                return {
                    'mode': 'lane_following',
                    'override_control': False,
                    'left_speed': None,
                    'right_speed': None,
                    'message': '正常巡线模式'
                }
        
        elif current_state == CarState.OBSTACLE_DETECTED:
            # 障碍物检测状态：立即进入第一段避障
            self._change_state(CarState.OBSTACLE_AVOIDANCE_STAGE1)
            self.logger.info("🚗 开始第一段避障动作")
            return self._get_control_decision()
            
        elif current_state == CarState.OBSTACLE_AVOIDANCE_STAGE1:
            # 第一段避障：执行第一段避障动作
            if time_in_state >= self.stage1_duration:
                self._change_state(CarState.OBSTACLE_AVOIDANCE_STAGE2)
                self.logger.info("🔄 第一段避障完成，开始第二段避障动作")
            return self._get_control_decision()
            
        elif current_state == CarState.OBSTACLE_AVOIDANCE_STAGE2:
            # 第二段避障：执行第二段避障动作
            if time_in_state >= self.stage2_duration:
                self._change_state(CarState.OBSTACLE_AVOIDANCE_STAGE3)
                self.logger.info("🔄 第二段避障完成，开始第三段避障动作")
            return self._get_control_decision()
            
        elif current_state == CarState.OBSTACLE_AVOIDANCE_STAGE3:
            # 第三段避障：执行第三段避障动作
            if time_in_state >= self.stage3_duration:
                self._change_state(CarState.RETURNING_TO_LANE)
                self.logger.info("🔄 第三段避障完成，返回巡线模式")
            return self._get_control_decision()
            
        elif current_state == CarState.RED_LIGHT_WAITING:
            # 红灯等待状态：检查是否转绿灯
            if (traffic_light_detected and traffic_light_status == "green" and 
                self.is_traffic_light_detection_frame()):
                self.logger.info("🚦 检测到绿灯，返回巡线模式")
                self._change_state(CarState.LANE_FOLLOWING)
                return self._get_control_decision()
            else:
                # 继续等待红灯，保持停车状态
                return self._get_control_decision()
                
        elif current_state == CarState.RETURNING_TO_LANE:
            # 返回巡线：立即切换到巡线模式
            self.complete_avoidance_cycle()  # 🛡️ 标记避障循环完成
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
                'message': '检测到障碍物，准备三段避障'
            }
            
        elif current_state == CarState.OBSTACLE_AVOIDANCE_STAGE1:
            return {
                'mode': 'obstacle_avoidance_stage1',
                'override_control': True,
                'left_speed': self.stage1_left_speed,
                'right_speed': self.stage1_right_speed,
                'message': f'第一段避障 - 左轮{self.stage1_left_speed}, 右轮{self.stage1_right_speed}'
            }
            
        elif current_state == CarState.OBSTACLE_AVOIDANCE_STAGE2:
            return {
                'mode': 'obstacle_avoidance_stage2',
                'override_control': True,
                'left_speed': self.stage2_left_speed,
                'right_speed': self.stage2_right_speed,
                'message': f'第二段避障 - 左轮{self.stage2_left_speed}, 右轮{self.stage2_right_speed}'
            }
            
        elif current_state == CarState.OBSTACLE_AVOIDANCE_STAGE3:
            return {
                'mode': 'obstacle_avoidance_stage3',
                'override_control': True,
                'left_speed': self.stage3_left_speed,
                'right_speed': self.stage3_right_speed,
                'message': f'第三段避障 - 左轮{self.stage3_left_speed}, 右轮{self.stage3_right_speed}'
            }
            
        elif current_state == CarState.RED_LIGHT_WAITING:
            return {
                'mode': 'red_light_waiting',
                'override_control': True,
                'left_speed': 0,
                'right_speed': 0,
                'message': '红灯等待中'
            }
            
        elif current_state == CarState.RETURNING_TO_LANE:
            return {
                'mode': 'returning_to_lane',
                'override_control': False,
                'left_speed': None,
                'right_speed': None,
                'message': '返回巡线模式'
            }
        
        # 默认返回停车状态
        return {
            'mode': 'unknown',
            'override_control': True,
            'left_speed': 0,
            'right_speed': 0,
            'message': '未知状态，停车'
        }
    
    def get_state_info(self) -> Dict:
        """获取状态机信息"""
        current_time = time.time()
        time_in_state = current_time - self.state_start_time
        time_since_last_avoidance = current_time - self.last_avoidance_time if self.last_avoidance_time > 0 else float('inf')
        
        return {
            'current_state': self.current_state.value,
            'previous_state': self.previous_state.value if self.previous_state else None,
            'time_in_state': time_in_state,
            'frame_count': self.frame_count,
            'next_obstacle_detection_frame': self.obstacle_detection_interval - (self.frame_count % self.obstacle_detection_interval),
            'next_traffic_light_detection_frame': self.traffic_light_detection_interval - (self.frame_count % self.traffic_light_detection_interval),
            # 🛡️ 防死循环保护状态
            'consecutive_avoidance_count': self.consecutive_avoidance_count,
            'max_consecutive_avoidances': self.max_consecutive_avoidances,
            'time_since_last_avoidance': time_since_last_avoidance,
            'avoidance_cooldown_remaining': max(0, self.avoidance_cooldown - time_since_last_avoidance),
            'avoidance_protection_active': not self.is_avoidance_allowed(),
            # 三段避障参数信息
            'avoidance_params': {
                'stage1': {
                    'left_speed': self.stage1_left_speed,
                    'right_speed': self.stage1_right_speed,
                    'duration': self.stage1_duration
                },
                'stage2': {
                    'left_speed': self.stage2_left_speed,
                    'right_speed': self.stage2_right_speed,
                    'duration': self.stage2_duration
                },
                'stage3': {
                    'left_speed': self.stage3_left_speed,
                    'right_speed': self.stage3_right_speed,
                    'duration': self.stage3_duration
                }
            }
        }
    
    def print_state_info(self):
        """记录状态机信息到日志"""
        info = self.get_state_info()
        self.logger.info(f"🎯 三段避障状态机信息:")
        self.logger.info(f"   当前状态: {info['current_state']}")
        self.logger.info(f"   上一状态: {info['previous_state']}")
        self.logger.info(f"   状态持续时间: {info['time_in_state']:.2f}秒")
        self.logger.info(f"   帧计数: {info['frame_count']}")
        self.logger.info(f"   下次障碍物检测: {info['next_obstacle_detection_frame']}帧后")
        self.logger.info(f"   下次交通灯检测: {info['next_traffic_light_detection_frame']}帧后")
        
        # 🛡️ 防死循环保护信息
        self.logger.info(f"   连续避障次数: {info['consecutive_avoidance_count']}/{info['max_consecutive_avoidances']}")
        self.logger.info(f"   距离上次避障: {info['time_since_last_avoidance']:.1f}秒")
        self.logger.info(f"   避障冷却剩余: {info['avoidance_cooldown_remaining']:.1f}秒")
        self.logger.info(f"   避障保护激活: {'是' if info['avoidance_protection_active'] else '否'}")
        
        # 三段避障参数信息
        params = info['avoidance_params']
        self.logger.info(f"   第一段: 左{params['stage1']['left_speed']}, 右{params['stage1']['right_speed']}, {params['stage1']['duration']}s")
        self.logger.info(f"   第二段: 左{params['stage2']['left_speed']}, 右{params['stage2']['right_speed']}, {params['stage2']['duration']}s")
        self.logger.info(f"   第三段: 左{params['stage3']['left_speed']}, 右{params['stage3']['right_speed']}, {params['stage3']['duration']}s")
    
    def force_stop(self):
        """强制停止状态机（紧急情况）"""
        with self.state_lock:
            self.current_state = CarState.LANE_FOLLOWING
            self.logger.warning("🛑 状态机已强制停止，回到巡线模式") 