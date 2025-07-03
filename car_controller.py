#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小车控制模块
提供转向强度和前进速度的接口，通过串口与STM32通信
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

class CarController:
    """
    小车控制模块
    提供转向强度和前进速度的接口
    """
    
    # 通信协议定义
    PROTOCOL_HEADER = 0xAA  # 协议头
    PROTOCOL_TAIL = 0x55    # 协议尾
    
    # 命令类型
    CMD_SET_SPEED = 0x01    # 设置速度
    CMD_SET_STEERING = 0x02 # 设置转向
    CMD_SET_MOTION = 0x03   # 设置运动（速度+转向）
    CMD_EMERGENCY_STOP = 0x04 # 紧急停止
    CMD_GET_STATUS = 0x05   # 获取状态
    CMD_ACK = 0x06          # 确认响应
    
    # 运动模式
    MODE_STOP = 0x00        # 停止
    MODE_FORWARD = 0x01     # 前进
    MODE_BACKWARD = 0x02    # 后退
    MODE_LEFT = 0x03        # 左转
    MODE_RIGHT = 0x04       # 右转
    MODE_DIFFERENTIAL = 0x05 # 差速转向
    
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
        
        # 当前状态
        self.current_speed = 0.0      # 当前速度 (0.0 - 1.0)
        self.current_steering = 0.0   # 当前转向 (-1.0 - 1.0)
        self.current_mode = self.MODE_STOP
        
        # 配置参数
        self.max_speed = 1.0          # 最大速度
        self.min_speed = 0.0          # 最小速度
        self.max_steering = 1.0       # 最大转向角度
        self.min_steering = -1.0      # 最小转向角度
        
        # 差速控制参数
        self.wheel_base = 0.2         # 轮距 (米)
        self.wheel_radius = 0.05      # 轮子半径 (米)
        self.max_wheel_speed = 1000   # 最大轮子速度 (PWM值)
        
        # 状态监控
        self.last_command_time = 0
        self.command_timeout = 0.5    # 命令超时时间
        
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
                    
                    # 发送初始化命令
                    self._send_init_command()
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
    
    def _send_init_command(self):
        """发送初始化命令"""
        try:
            # 发送停止命令确保小车处于安全状态
            self._send_command(self.CMD_EMERGENCY_STOP, [])
            time.sleep(0.1)
            logger.info("初始化命令已发送")
        except Exception as e:
            logger.error(f"发送初始化命令失败: {e}")
    
    def set_speed(self, speed: float) -> bool:
        """
        设置前进速度
        
        Args:
            speed: 速度值 (0.0 - 1.0)
            
        Returns:
            bool: 设置是否成功
        """
        speed = max(self.min_speed, min(self.max_speed, speed))
        
        if abs(speed - self.current_speed) < 0.01:
            return True  # 速度没有变化，不需要发送命令
        
        try:
            # 将速度转换为PWM值 (0-1000)
            pwm_speed = int(speed * self.max_wheel_speed)
            
            # 发送速度命令
            data = struct.pack('<H', pwm_speed)  # 2字节无符号整数
            success = self._send_command(self.CMD_SET_SPEED, data)
            
            if success:
                self.current_speed = speed
                self.last_command_time = time.time()
                logger.info(f"速度设置为: {speed:.2f} (PWM: {pwm_speed})")
            
            return success
        except Exception as e:
            logger.error(f"设置速度失败: {e}")
            return False
    
    def set_steering(self, steering: float) -> bool:
        """
        设置转向强度
        
        Args:
            steering: 转向值 (-1.0 - 1.0)
                      -1.0: 最大左转
                       0.0: 直行
                       1.0: 最大右转
            
        Returns:
            bool: 设置是否成功
        """
        steering = max(self.min_steering, min(self.max_steering, steering))
        
        if abs(steering - self.current_steering) < 0.01:
            return True  # 转向没有变化，不需要发送命令
        
        try:
            # 将转向值转换为差速控制参数
            # steering = (left_speed - right_speed) / (left_speed + right_speed)
            # 当steering=0时，左右轮速度相等
            # 当steering=1时，右轮速度最大，左轮速度最小
            # 当steering=-1时，左轮速度最大，右轮速度最小
            
            # 计算左右轮速度比例
            if abs(steering) < 0.01:
                # 直行
                left_ratio = 1.0
                right_ratio = 1.0
            else:
                # 转向
                left_ratio = 1.0 - steering * 0.5
                right_ratio = 1.0 + steering * 0.5
                
                # 确保比例在合理范围内
                left_ratio = max(0.3, min(1.0, left_ratio))
                right_ratio = max(0.3, min(1.0, right_ratio))
            
            # 发送转向命令
            data = struct.pack('<ff', left_ratio, right_ratio)  # 2个4字节浮点数
            success = self._send_command(self.CMD_SET_STEERING, data)
            
            if success:
                self.current_steering = steering
                self.last_command_time = time.time()
                logger.info(f"转向设置为: {steering:.2f} (左轮比例: {left_ratio:.2f}, 右轮比例: {right_ratio:.2f})")
            
            return success
        except Exception as e:
            logger.error(f"设置转向失败: {e}")
            return False
    
    def set_motion(self, speed: float, steering: float) -> bool:
        """
        同时设置速度和转向
        
        Args:
            speed: 速度值 (0.0 - 1.0)
            steering: 转向值 (-1.0 - 1.0)
            
        Returns:
            bool: 设置是否成功
        """
        speed = max(self.min_speed, min(self.max_speed, speed))
        steering = max(self.min_steering, min(self.max_steering, steering))
        
        try:
            # 将速度和转向转换为PWM值
            pwm_speed = int(speed * self.max_wheel_speed)
            
            # 计算左右轮速度比例
            if abs(steering) < 0.01:
                left_ratio = 1.0
                right_ratio = 1.0
            else:
                left_ratio = 1.0 - steering * 0.5
                right_ratio = 1.0 + steering * 0.5
                left_ratio = max(0.3, min(1.0, left_ratio))
                right_ratio = max(0.3, min(1.0, right_ratio))
            
            # 计算实际PWM值
            left_pwm = int(pwm_speed * left_ratio)
            right_pwm = int(pwm_speed * right_ratio)
            
            # 发送运动命令
            data = struct.pack('<HHH', pwm_speed, left_pwm, right_pwm)  # 3个2字节无符号整数
            success = self._send_command(self.CMD_SET_MOTION, data)
            
            if success:
                self.current_speed = speed
                self.current_steering = steering
                self.last_command_time = time.time()
                logger.info(f"运动设置为: 速度={speed:.2f}, 转向={steering:.2f} (左轮PWM: {left_pwm}, 右轮PWM: {right_pwm})")
            
            return success
        except Exception as e:
            logger.error(f"设置运动失败: {e}")
            return False
    
    def stop(self) -> bool:
        """
        停止小车
        
        Returns:
            bool: 停止是否成功
        """
        try:
            success = self._send_command(self.CMD_EMERGENCY_STOP, [])
            if success:
                self.current_speed = 0.0
                self.current_steering = 0.0
                self.current_mode = self.MODE_STOP
                self.last_command_time = time.time()
                logger.info("小车已停止")
            return success
        except Exception as e:
            logger.error(f"停止小车失败: {e}")
            return False
    
    def get_status(self) -> Optional[dict]:
        """
        获取小车状态
        
        Returns:
            dict: 状态信息，如果失败返回None
        """
        try:
            # 发送状态查询命令
            success = self._send_command(self.CMD_GET_STATUS, [])
            if not success:
                return None
            
            # 读取状态响应
            response = self._read_response()
            if response and len(response) >= 8:
                # 解析状态数据
                status = struct.unpack('<HHHH', response[:8])
                return {
                    'left_front_speed': status[0],
                    'left_rear_speed': status[1],
                    'right_front_speed': status[2],
                    'right_rear_speed': status[3],
                    'current_speed': self.current_speed,
                    'current_steering': self.current_steering,
                    'mode': self.current_mode
                }
            return None
        except Exception as e:
            logger.error(f"获取状态失败: {e}")
            return None
    
    def _send_command(self, cmd_type: int, data: bytes) -> bool:
        """
        发送命令
        
        Args:
            cmd_type: 命令类型
            data: 命令数据
            
        Returns:
            bool: 发送是否成功
        """
        if not self.is_connected or self.serial is None:
            logger.error("串口未连接")
            return False
        
        try:
            with self.lock:
                # 构建命令包
                cmd_len = len(data)
                packet = struct.pack('<BB', self.PROTOCOL_HEADER, cmd_type)
                packet += struct.pack('<B', cmd_len)
                packet += data
                
                # 计算校验和
                checksum = sum(packet[1:]) & 0xFF
                packet += struct.pack('<B', checksum)
                packet += struct.pack('<B', self.PROTOCOL_TAIL)
                
                # 发送命令
                self.serial.write(packet)
                logger.debug(f"发送命令: {packet.hex()}")
                return True
                
        except Exception as e:
            logger.error(f"发送命令失败: {e}")
            return False
    
    def _read_response(self, timeout: float = 0.1) -> Optional[bytes]:
        """
        读取响应
        
        Args:
            timeout: 超时时间
            
        Returns:
            bytes: 响应数据，如果失败返回None
        """
        if not self.is_connected or self.serial is None:
            return None
        
        try:
            with self.lock:
                start_time = time.time()
                response = b''
                
                while time.time() - start_time < timeout:
                    if self.serial.in_waiting > 0:
                        data = self.serial.read(self.serial.in_waiting)
                        response += data
                        
                        # 检查是否收到完整的响应包
                        if len(response) >= 4:
                            if response[0] == self.PROTOCOL_HEADER and response[-1] == self.PROTOCOL_TAIL:
                                # 验证校验和
                                checksum = sum(response[1:-2]) & 0xFF
                                if checksum == response[-2]:
                                    logger.debug(f"收到响应: {response.hex()}")
                                    return response[3:-2]  # 返回数据部分
                    
                    time.sleep(0.001)  # 短暂延时
                
                logger.warning("读取响应超时")
                return None
                
        except Exception as e:
            logger.error(f"读取响应失败: {e}")
            return None
    
    def is_command_timeout(self) -> bool:
        """
        检查命令是否超时
        
        Returns:
            bool: 是否超时
        """
        return time.time() - self.last_command_time > self.command_timeout
    
    def get_current_state(self) -> dict:
        """
        获取当前状态
        
        Returns:
            dict: 当前状态信息
        """
        return {
            'speed': self.current_speed,
            'steering': self.current_steering,
            'mode': self.current_mode,
            'connected': self.is_connected,
            'last_command_time': self.last_command_time,
            'command_timeout': self.is_command_timeout()
        }
    
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
    controller = CarController()
    
    try:
        # 连接小车
        if not controller.connect():
            print("连接失败")
            return
        
        print("小车控制模块初始化完成")
        print("可用命令:")
        print("  set_speed(speed) - 设置速度 (0.0-1.0)")
        print("  set_steering(steering) - 设置转向 (-1.0-1.0)")
        print("  set_motion(speed, steering) - 同时设置速度和转向")
        print("  stop() - 停止小车")
        print("  get_status() - 获取状态")
        
        # 测试基本功能
        print("\n开始测试...")
        
        # 前进
        print("1. 前进测试")
        controller.set_motion(0.5, 0.0)
        time.sleep(2)
        
        # 左转
        print("2. 左转测试")
        controller.set_motion(0.3, -0.5)
        time.sleep(2)
        
        # 右转
        print("3. 右转测试")
        controller.set_motion(0.3, 0.5)
        time.sleep(2)
        
        # 停止
        print("4. 停止测试")
        controller.stop()
        time.sleep(1)
        
        # 获取状态
        print("5. 状态查询")
        status = controller.get_status()
        if status:
            print(f"当前状态: {status}")
        
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