#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化小车控制模块
只控制两组轮子的速度，转向逻辑在上层处理
"""

import time
import threading
import struct
from periphery import Serial
from typing import Optional, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleCarController:
    """
    简化小车控制模块
    只控制两组轮子的速度，转向逻辑在上层处理
    """
    
    # 通信协议定义
    PROTOCOL_HEADER = 0xAA  # 协议头
    PROTOCOL_TAIL = 0x55    # 协议尾
    
    def __init__(self, port="/dev/ttyAMA0", baudrate=115200, timeout=1.0):
        """
        初始化小车控制器
        
        Args:
            port: 串口设备名
            baudrate: 波特率
            timeout: 超时时间
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self.is_connected = False
        self.lock = threading.Lock()
        
        # 当前轮子速度
        self.left_wheel_speed = 0   # 左轮组速度 (-1000 到 +1000)
        self.right_wheel_speed = 0  # 右轮组速度 (-1000 到 +1000)
        
        # 配置参数
        self.max_speed = 1000       # 最大速度
        self.min_speed = -1000      # 最小速度（后退）
        
        # 状态监控
        self.last_command_time = 0
        self.command_timeout = 0.5  # 命令超时时间
        
    def connect(self) -> bool:
        """
        连接串口
        
        Returns:
            bool: 连接是否成功
        """
        try:
            with self.lock:
                if self.serial is None:
                    self.serial = Serial(self.port, self.baudrate)
                    self.serial.timeout = self.timeout
                    self.is_connected = True
                    logger.info(f"成功连接到串口 {self.port}")
                    
                    # 发送停止命令确保小车处于安全状态
                    self.stop()
                    return True
                else:
                    logger.warning("串口已经连接")
                    return True
        except Exception as e:
            logger.error(f"连接串口失败: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """断开串口连接"""
        with self.lock:
            if self.serial:
                try:
                    self.serial.close()
                    logger.info("串口连接已断开")
                except Exception as e:
                    logger.error(f"断开串口连接时出错: {e}")
                finally:
                    self.serial = None
                    self.is_connected = False
    
    def set_wheel_speeds(self, left_speed: int, right_speed: int) -> bool:
        """
        设置两组轮子的速度
        
        Args:
            left_speed: 左轮组速度 (-1000 到 +1000)
            right_speed: 右轮组速度 (-1000 到 +1000)
            
        Returns:
            bool: 设置是否成功
        """
        # 限制速度范围
        left_speed = max(self.min_speed, min(self.max_speed, left_speed))
        right_speed = max(self.min_speed, min(self.max_speed, right_speed))
        
        # 检查速度是否有变化
        if left_speed == self.left_wheel_speed and right_speed == self.right_wheel_speed:
            return True  # 速度没有变化，不需要发送命令
        
        try:
            # 发送速度命令
            success = self._send_speed_command(left_speed, right_speed)
            
            if success:
                self.left_wheel_speed = left_speed
                self.right_wheel_speed = right_speed
                self.last_command_time = time.time()
                logger.info(f"轮子速度设置为: 左轮={left_speed}, 右轮={right_speed}")
            
            return success
        except Exception as e:
            logger.error(f"设置轮子速度失败: {e}")
            return False
    
    def set_motion(self, speed: float, steering: float) -> bool:
        """
        设置运动（速度和转向）
        这是上层接口，将速度和转向转换为轮子速度
        
        Args:
            speed: 速度值 (0.0 - 1.0)
            steering: 转向值 (-1.0 - 1.0)
                      -1.0: 最大左转
                       0.0: 直行
                       1.0: 最大右转
            
        Returns:
            bool: 设置是否成功
        """
        # 限制输入范围
        speed = max(0.0, min(1.0, speed))
        steering = max(-1.0, min(1.0, steering))
        
        # 计算轮子速度
        base_speed = int(speed * self.max_speed)
        
        if abs(steering) < 0.01:
            # 直行
            left_speed = base_speed
            right_speed = base_speed
        else:
            # 转向：通过左右轮速度差实现
            # steering = (left_speed - right_speed) / (left_speed + right_speed)
            # 当steering=1时，右轮快，左轮慢
            # 当steering=-1时，左轮快，右轮慢
            
            # 计算速度差
            speed_diff = int(base_speed * steering * 0.8)  # 0.8是转向强度系数
            
            left_speed = base_speed - speed_diff
            right_speed = base_speed + speed_diff
            
            # 确保速度在合理范围内
            left_speed = max(self.min_speed, min(self.max_speed, left_speed))
            right_speed = max(self.min_speed, min(self.max_speed, right_speed))
        
        # 设置轮子速度
        return self.set_wheel_speeds(left_speed, right_speed)
    
    def set_speed(self, speed: float) -> bool:
        """
        设置前进速度（直行）
        
        Args:
            speed: 速度值 (0.0 - 1.0)
            
        Returns:
            bool: 设置是否成功
        """
        return self.set_motion(speed, 0.0)
    
    def set_steering(self, steering: float) -> bool:
        """
        设置转向（保持当前速度）
        
        Args:
            steering: 转向值 (-1.0 - 1.0)
            
        Returns:
            bool: 设置是否成功
        """
        # 获取当前速度
        current_speed = max(abs(self.left_wheel_speed), abs(self.right_wheel_speed)) / self.max_speed
        return self.set_motion(current_speed, steering)
    
    def stop(self) -> bool:
        """
        停止小车
        
        Returns:
            bool: 停止是否成功
        """
        return self.set_wheel_speeds(0, 0)
    
    def forward(self, speed: float) -> bool:
        """
        前进
        
        Args:
            speed: 速度值 (0.0 - 1.0)
            
        Returns:
            bool: 设置是否成功
        """
        return self.set_speed(speed)
    
    def backward(self, speed: float) -> bool:
        """
        后退
        
        Args:
            speed: 速度值 (0.0 - 1.0)
            
        Returns:
            bool: 设置是否成功
        """
        speed = max(0.0, min(1.0, speed))
        base_speed = int(speed * self.max_speed)
        return self.set_wheel_speeds(-base_speed, -base_speed)
    
    def turn_left(self, speed: float, turn_intensity: float = 0.5) -> bool:
        """
        左转
        
        Args:
            speed: 速度值 (0.0 - 1.0)
            turn_intensity: 转向强度 (0.0 - 1.0)
            
        Returns:
            bool: 设置是否成功
        """
        return self.set_motion(speed, -turn_intensity)
    
    def turn_right(self, speed: float, turn_intensity: float = 0.5) -> bool:
        """
        右转
        
        Args:
            speed: 速度值 (0.0 - 1.0)
            turn_intensity: 转向强度 (0.0 - 1.0)
            
        Returns:
            bool: 设置是否成功
        """
        return self.set_motion(speed, turn_intensity)
    
    def spin_left(self, speed: float) -> bool:
        """
        原地左转
        
        Args:
            speed: 速度值 (0.0 - 1.0)
            
        Returns:
            bool: 设置是否成功
        """
        speed = max(0.0, min(1.0, speed))
        base_speed = int(speed * self.max_speed)
        return self.set_wheel_speeds(-base_speed, base_speed)
    
    def spin_right(self, speed: float) -> bool:
        """
        原地右转
        
        Args:
            speed: 速度值 (0.0 - 1.0)
            
        Returns:
            bool: 设置是否成功
        """
        speed = max(0.0, min(1.0, speed))
        base_speed = int(speed * self.max_speed)
        return self.set_wheel_speeds(base_speed, -base_speed)
    
    def _send_speed_command(self, left_speed: int, right_speed: int) -> bool:
        """
        发送速度命令
        
        Args:
            left_speed: 左轮速度
            right_speed: 右轮速度
            
        Returns:
            bool: 发送是否成功
        """
        if not self.is_connected or self.serial is None:
            logger.error("串口未连接")
            return False
        
        try:
            with self.lock:
                # 构建命令包
                packet = struct.pack('<B', self.PROTOCOL_HEADER)
                packet += struct.pack('<hh', left_speed, right_speed)  # 2个16位有符号整数
                
                # 计算校验和
                checksum = sum(packet[1:]) & 0xFF
                packet += struct.pack('<B', checksum)
                packet += struct.pack('<B', self.PROTOCOL_TAIL)
                
                # 发送命令
                self.serial.write(packet)
                logger.debug(f"发送速度命令: 左轮={left_speed}, 右轮={right_speed}, 包={packet.hex()}")
                return True
                
        except Exception as e:
            logger.error(f"发送速度命令失败: {e}")
            return False
    
    def get_current_speeds(self) -> Tuple[int, int]:
        """
        获取当前轮子速度
        
        Returns:
            Tuple[int, int]: (左轮速度, 右轮速度)
        """
        return self.left_wheel_speed, self.right_wheel_speed
    
    def get_current_state(self) -> dict:
        """
        获取当前状态
        
        Returns:
            dict: 当前状态信息
        """
        return {
            'left_wheel_speed': self.left_wheel_speed,
            'right_wheel_speed': self.right_wheel_speed,
            'connected': self.is_connected,
            'last_command_time': self.last_command_time,
            'command_timeout': self.is_command_timeout()
        }
    
    def is_command_timeout(self) -> bool:
        """
        检查命令是否超时
        
        Returns:
            bool: 是否超时
        """
        return time.time() - self.last_command_time > self.command_timeout
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()
        self.disconnect()


# 使用示例
def main():
    """使用示例"""
    controller = SimpleCarController()
    
    try:
        # 连接小车
        if not controller.connect():
            print("连接失败")
            return
        
        print("简化小车控制模块初始化完成")
        print("可用命令:")
        print("  set_wheel_speeds(left, right) - 直接设置轮子速度")
        print("  set_motion(speed, steering) - 设置运动和转向")
        print("  forward(speed) - 前进")
        print("  backward(speed) - 后退")
        print("  turn_left(speed, intensity) - 左转")
        print("  turn_right(speed, intensity) - 右转")
        print("  spin_left(speed) - 原地左转")
        print("  spin_right(speed) - 原地右转")
        print("  stop() - 停止")
        
        # 测试基本功能
        print("\n开始测试...")
        
        # 前进
        print("1. 前进测试")
        controller.forward(0.5)
        time.sleep(2)
        
        # 左转
        print("2. 左转测试")
        controller.turn_left(0.3, 0.6)
        time.sleep(2)
        
        # 右转
        print("3. 右转测试")
        controller.turn_right(0.3, 0.6)
        time.sleep(2)
        
        # 原地旋转
        print("4. 原地旋转测试")
        controller.spin_left(0.4)
        time.sleep(2)
        
        # 后退
        print("5. 后退测试")
        controller.backward(0.3)
        time.sleep(2)
        
        # 停止
        print("6. 停止测试")
        controller.stop()
        time.sleep(1)
        
        # 获取状态
        print("7. 状态查询")
        state = controller.get_current_state()
        print(f"当前状态: {state}")
        
        print("测试完成")
        
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        controller.stop()
        controller.disconnect()


if __name__ == "__main__":
    main() 