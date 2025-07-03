#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化小车控制模块测试脚本
"""

import time
import sys
from car_controller_simple import SimpleCarController

def test_basic_control():
    """测试基本控制功能"""
    print("=== 基本控制功能测试 ===")
    
    with SimpleCarController() as controller:
        if not controller.is_connected:
            print("❌ 连接失败")
            return False
        
        print("✅ 连接成功")
        
        # 测试停止
        print("1. 测试停止...")
        if controller.stop():
            print("✅ 停止命令发送成功")
        else:
            print("❌ 停止命令发送失败")
            return False
        
        time.sleep(1)
        
        # 测试前进
        print("2. 测试前进...")
        if controller.forward(0.5):
            print("✅ 前进命令发送成功")
        else:
            print("❌ 前进命令发送失败")
            return False
        
        time.sleep(2)
        
        # 测试后退
        print("3. 测试后退...")
        if controller.backward(0.3):
            print("✅ 后退命令发送成功")
        else:
            print("❌ 后退命令发送失败")
            return False
        
        time.sleep(2)
        
        # 测试停止
        print("4. 测试停止...")
        if controller.stop():
            print("✅ 停止命令发送成功")
        else:
            print("❌ 停止命令发送失败")
            return False
        
        time.sleep(1)
        
        print("✅ 基本控制功能测试通过")
        return True

def test_steering_control():
    """测试转向控制功能"""
    print("\n=== 转向控制功能测试 ===")
    
    with SimpleCarController() as controller:
        if not controller.is_connected:
            print("❌ 连接失败")
            return False
        
        # 测试左转
        print("1. 测试左转...")
        if controller.turn_left(0.4, 0.6):
            print("✅ 左转命令发送成功")
        else:
            print("❌ 左转命令发送失败")
            return False
        
        time.sleep(2)
        
        # 测试右转
        print("2. 测试右转...")
        if controller.turn_right(0.4, 0.6):
            print("✅ 右转命令发送成功")
        else:
            print("❌ 右转命令发送失败")
            return False
        
        time.sleep(2)
        
        # 测试停止
        controller.stop()
        print("✅ 转向控制功能测试通过")
        return True

def test_spin_control():
    """测试原地旋转功能"""
    print("\n=== 原地旋转功能测试 ===")
    
    with SimpleCarController() as controller:
        if not controller.is_connected:
            print("❌ 连接失败")
            return False
        
        # 测试原地左转
        print("1. 测试原地左转...")
        if controller.spin_left(0.3):
            print("✅ 原地左转命令发送成功")
        else:
            print("❌ 原地左转命令发送失败")
            return False
        
        time.sleep(2)
        
        # 测试原地右转
        print("2. 测试原地右转...")
        if controller.spin_right(0.3):
            print("✅ 原地右转命令发送成功")
        else:
            print("❌ 原地右转命令发送失败")
            return False
        
        time.sleep(2)
        
        # 测试停止
        controller.stop()
        print("✅ 原地旋转功能测试通过")
        return True

def test_direct_wheel_control():
    """测试直接轮子控制功能"""
    print("\n=== 直接轮子控制功能测试 ===")
    
    with SimpleCarController() as controller:
        if not controller.is_connected:
            print("❌ 连接失败")
            return False
        
        # 测试直接设置轮子速度
        print("1. 测试直接设置轮子速度...")
        if controller.set_wheel_speeds(300, 500):
            print("✅ 直接轮子速度设置成功")
        else:
            print("❌ 直接轮子速度设置失败")
            return False
        
        time.sleep(2)
        
        # 测试不同速度组合
        print("2. 测试不同速度组合...")
        test_speeds = [
            (500, 500),   # 直行
            (300, 700),   # 右转
            (700, 300),   # 左转
            (-300, 300),  # 原地左转
            (300, -300),  # 原地右转
        ]
        
        for i, (left, right) in enumerate(test_speeds, 1):
            print(f"   测试组合 {i}: 左轮={left}, 右轮={right}")
            if controller.set_wheel_speeds(left, right):
                time.sleep(1)
            else:
                print(f"❌ 组合 {i} 设置失败")
                return False
        
        # 测试停止
        controller.stop()
        print("✅ 直接轮子控制功能测试通过")
        return True

def test_motion_control():
    """测试运动控制功能"""
    print("\n=== 运动控制功能测试 ===")
    
    with SimpleCarController() as controller:
        if not controller.is_connected:
            print("❌ 连接失败")
            return False
        
        # 测试不同的运动和转向组合
        test_motions = [
            (0.5, 0.0),   # 直行
            (0.4, -0.3),  # 左转
            (0.4, 0.3),   # 右转
            (0.6, -0.5),  # 快速左转
            (0.6, 0.5),   # 快速右转
            (0.2, -0.8),  # 慢速大角度左转
            (0.2, 0.8),   # 慢速大角度右转
        ]
        
        for i, (speed, steering) in enumerate(test_motions, 1):
            print(f"{i}. 测试运动: 速度={speed:.1f}, 转向={steering:+.1f}")
            if controller.set_motion(speed, steering):
                print(f"✅ 运动设置成功")
                time.sleep(1.5)
            else:
                print(f"❌ 运动设置失败")
                return False
        
        # 测试停止
        controller.stop()
        print("✅ 运动控制功能测试通过")
        return True

def test_speed_ramp():
    """测试速度渐变功能"""
    print("\n=== 速度渐变功能测试 ===")
    
    with SimpleCarController() as controller:
        if not controller.is_connected:
            print("❌ 连接失败")
            return False
        
        # 渐进加速
        print("1. 渐进加速...")
        for speed in range(0, 11):
            speed_val = speed / 10.0
            if controller.forward(speed_val):
                print(f"   速度: {speed_val:.1f}")
                time.sleep(0.3)
            else:
                print(f"❌ 速度 {speed_val:.1f} 设置失败")
                return False
        
        # 渐进减速
        print("2. 渐进减速...")
        for speed in range(10, -1, -1):
            speed_val = speed / 10.0
            if controller.forward(speed_val):
                print(f"   速度: {speed_val:.1f}")
                time.sleep(0.3)
            else:
                print(f"❌ 速度 {speed_val:.1f} 设置失败")
                return False
        
        # 测试停止
        controller.stop()
        print("✅ 速度渐变功能测试通过")
        return True

def test_status_monitoring():
    """测试状态监控功能"""
    print("\n=== 状态监控功能测试 ===")
    
    with SimpleCarController() as controller:
        if not controller.is_connected:
            print("❌ 连接失败")
            return False
        
        # 查询初始状态
        print("1. 查询初始状态...")
        state = controller.get_current_state()
        speeds = controller.get_current_speeds()
        print(f"   当前状态: {state}")
        print(f"   轮子速度: 左轮={speeds[0]}, 右轮={speeds[1]}")
        
        # 设置运动并监控
        print("2. 运动状态监控...")
        controller.set_motion(0.4, 0.3)
        
        for i in range(5):
            time.sleep(1)
            speeds = controller.get_current_speeds()
            print(f"   第{i+1}秒速度: 左轮={speeds[0]}, 右轮={speeds[1]}")
        
        # 测试停止
        controller.stop()
        print("✅ 状态监控功能测试通过")
        return True

def test_error_handling():
    """测试错误处理功能"""
    print("\n=== 错误处理功能测试 ===")
    
    with SimpleCarController() as controller:
        if not controller.is_connected:
            print("❌ 连接失败")
            return False
        
        # 测试超出范围的速度值
        print("1. 测试超出范围的速度值...")
        if controller.set_wheel_speeds(1500, 500):  # 超出最大速度
            print("❌ 应该拒绝超出范围的速度值")
            return False
        else:
            print("✅ 正确拒绝超出范围的速度值")
        
        print("2. 测试负速度值...")
        if controller.set_wheel_speeds(-1500, 500):  # 超出最小速度
            print("❌ 应该拒绝超出范围的速度值")
            return False
        else:
            print("✅ 正确拒绝超出范围的速度值")
        
        print("✅ 错误处理功能测试通过")
        return True

def main():
    """主测试函数"""
    print("🚗 简化小车控制模块测试开始")
    print("=" * 50)
    
    tests = [
        ("基本控制功能", test_basic_control),
        ("转向控制功能", test_steering_control),
        ("原地旋转功能", test_spin_control),
        ("直接轮子控制", test_direct_wheel_control),
        ("运动控制功能", test_motion_control),
        ("速度渐变功能", test_speed_ramp),
        ("状态监控功能", test_status_monitoring),
        ("错误处理功能", test_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
        
        print("-" * 50)
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！简化小车控制模块工作正常。")
        return 0
    else:
        print("⚠️  部分测试失败，请检查硬件连接和固件。")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        sys.exit(1) 