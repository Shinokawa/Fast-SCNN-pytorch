#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小车控制模块使用示例
展示如何在其他程序中使用小车控制模块
"""

import time
import math
from car_controller import CarController

class CarControlExample:
    """小车控制使用示例类"""
    
    def __init__(self):
        self.controller = CarController()
        self.is_running = False
    
    def start(self):
        """启动控制模块"""
        if self.controller.connect():
            print("✅ 小车控制模块启动成功")
            self.is_running = True
            return True
        else:
            print("❌ 小车控制模块启动失败")
            return False
    
    def stop(self):
        """停止控制模块"""
        if self.is_running:
            self.controller.stop()
            self.controller.disconnect()
            self.is_running = False
            print("🛑 小车控制模块已停止")
    
    def demo_basic_movement(self):
        """演示基本运动"""
        print("\n=== 基本运动演示 ===")
        
        # 前进
        print("🚗 前进...")
        self.controller.set_motion(0.4, 0.0)
        time.sleep(3)
        
        # 停止
        print("🛑 停止...")
        self.controller.stop()
        time.sleep(1)
        
        # 左转
        print("⬅️  左转...")
        self.controller.set_motion(0.3, -0.5)
        time.sleep(2)
        
        # 右转
        print("➡️  右转...")
        self.controller.set_motion(0.3, 0.5)
        time.sleep(2)
        
        # 停止
        self.controller.stop()
        print("✅ 基本运动演示完成")
    
    def demo_speed_ramp(self):
        """演示速度渐变"""
        print("\n=== 速度渐变演示 ===")
        
        # 渐进加速
        print("🚀 渐进加速...")
        for speed in range(0, 11):
            speed_val = speed / 10.0
            self.controller.set_speed(speed_val)
            print(f"   速度: {speed_val:.1f}")
            time.sleep(0.5)
        
        # 渐进减速
        print("🛑 渐进减速...")
        for speed in range(10, -1, -1):
            speed_val = speed / 10.0
            self.controller.set_speed(speed_val)
            print(f"   速度: {speed_val:.1f}")
            time.sleep(0.5)
        
        self.controller.stop()
        print("✅ 速度渐变演示完成")
    
    def demo_steering_control(self):
        """演示转向控制"""
        print("\n=== 转向控制演示 ===")
        
        # 设置基础速度
        self.controller.set_speed(0.3)
        time.sleep(0.5)
        
        # 蛇形运动
        print("🐍 蛇形运动...")
        for i in range(10):
            steering = math.sin(i * 0.5) * 0.6  # 正弦波转向
            self.controller.set_steering(steering)
            print(f"   转向: {steering:+.2f}")
            time.sleep(0.8)
        
        # 停止
        self.controller.stop()
        print("✅ 转向控制演示完成")
    
    def demo_figure_eight(self):
        """演示8字形运动"""
        print("\n=== 8字形运动演示 ===")
        
        # 8字形运动参数
        radius = 0.8  # 转向强度
        speed = 0.25  # 基础速度
        duration = 15  # 运动时间
        
        print(f"🔄 8字形运动 ({duration}秒)...")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # 计算8字形的转向
            t = (time.time() - start_time) * 0.5  # 时间参数
            steering = radius * math.sin(t) * math.cos(t)
            
            self.controller.set_motion(speed, steering)
            time.sleep(0.1)
        
        self.controller.stop()
        print("✅ 8字形运动演示完成")
    
    def demo_autonomous_behavior(self):
        """演示自主行为"""
        print("\n=== 自主行为演示 ===")
        
        # 模拟避障行为
        print("🤖 模拟避障行为...")
        
        # 前进直到检测到障碍
        print("   前进中...")
        self.controller.set_motion(0.3, 0.0)
        time.sleep(2)
        
        # 检测到障碍，左转避障
        print("   ⚠️  检测到障碍，左转避障...")
        self.controller.set_motion(0.2, -0.7)
        time.sleep(1.5)
        
        # 继续前进
        print("   继续前进...")
        self.controller.set_motion(0.3, 0.0)
        time.sleep(2)
        
        # 右转回到原路径
        print("   ➡️  回到原路径...")
        self.controller.set_motion(0.2, 0.7)
        time.sleep(1.5)
        
        # 停止
        self.controller.stop()
        print("✅ 自主行为演示完成")
    
    def demo_status_monitoring(self):
        """演示状态监控"""
        print("\n=== 状态监控演示 ===")
        
        # 查询初始状态
        print("📊 查询初始状态...")
        status = self.controller.get_status()
        if status:
            print(f"   当前状态: {status}")
        
        # 设置运动并监控
        print("📈 运动状态监控...")
        self.controller.set_motion(0.4, 0.3)
        
        for i in range(5):
            time.sleep(1)
            status = self.controller.get_status()
            if status:
                print(f"   第{i+1}秒状态: 左轮={status['left_front_speed']}, 右轮={status['right_front_speed']}")
        
        self.controller.stop()
        print("✅ 状态监控演示完成")
    
    def demo_emergency_stop(self):
        """演示紧急停止"""
        print("\n=== 紧急停止演示 ===")
        
        # 高速运动
        print("🚗 高速运动...")
        self.controller.set_motion(0.8, 0.0)
        time.sleep(1)
        
        # 紧急停止
        print("🛑 紧急停止!")
        self.controller.stop()
        time.sleep(1)
        
        # 验证停止状态
        status = self.controller.get_status()
        if status:
            print(f"   停止后状态: 左轮={status['left_front_speed']}, 右轮={status['right_front_speed']}")
        
        print("✅ 紧急停止演示完成")
    
    def run_all_demos(self):
        """运行所有演示"""
        print("🎬 开始运行所有演示...")
        
        demos = [
            ("基本运动", self.demo_basic_movement),
            ("速度渐变", self.demo_speed_ramp),
            ("转向控制", self.demo_steering_control),
            ("8字形运动", self.demo_figure_eight),
            ("自主行为", self.demo_autonomous_behavior),
            ("状态监控", self.demo_status_monitoring),
            ("紧急停止", self.demo_emergency_stop),
        ]
        
        for demo_name, demo_func in demos:
            try:
                print(f"\n🎯 开始演示: {demo_name}")
                demo_func()
                print(f"✅ {demo_name} 演示完成")
            except Exception as e:
                print(f"❌ {demo_name} 演示失败: {e}")
            
            # 演示间隔
            time.sleep(2)
        
        print("\n🎉 所有演示完成!")

def main():
    """主函数"""
    print("🚗 小车控制模块使用示例")
    print("=" * 50)
    
    # 创建示例对象
    example = CarControlExample()
    
    try:
        # 启动控制模块
        if not example.start():
            return
        
        # 运行所有演示
        example.run_all_demos()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
    finally:
        # 确保安全停止
        example.stop()

if __name__ == "__main__":
    main() 