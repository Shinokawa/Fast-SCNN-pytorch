#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小车控制模块测试脚本
"""

import time
import sys
from car_controller import CarController

def test_basic_control():
    """测试基本控制功能"""
    print("=== 基本控制功能测试 ===")
    
    with CarController() as controller:
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
        if controller.set_motion(0.3, 0.0):
            print("✅ 前进命令发送成功")
        else:
            print("❌ 前进命令发送失败")
            return False
        
        time.sleep(2)
        
        # 测试左转
        print("3. 测试左转...")
        if controller.set_motion(0.2, -0.5):
            print("✅ 左转命令发送成功")
        else:
            print("❌ 左转命令发送失败")
            return False
        
        time.sleep(2)
        
        # 测试右转
        print("4. 测试右转...")
        if controller.set_motion(0.2, 0.5):
            print("✅ 右转命令发送成功")
        else:
            print("❌ 右转命令发送失败")
            return False
        
        time.sleep(2)
        
        # 测试停止
        print("5. 测试停止...")
        if controller.stop():
            print("✅ 停止命令发送成功")
        else:
            print("❌ 停止命令发送失败")
            return False
        
        time.sleep(1)
        
        print("✅ 基本控制功能测试通过")
        return True

def test_speed_control():
    """测试速度控制功能"""
    print("\n=== 速度控制功能测试 ===")
    
    with CarController() as controller:
        if not controller.is_connected:
            print("❌ 连接失败")
            return False
        
        # 测试不同速度
        speeds = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for i, speed in enumerate(speeds, 1):
            print(f"{i}. 测试速度 {speed:.1f}...")
            if controller.set_speed(speed):
                print(f"✅ 速度 {speed:.1f} 设置成功")
                time.sleep(1)
            else:
                print(f"❌ 速度 {speed:.1f} 设置失败")
                return False
        
        # 停止
        controller.stop()
        print("✅ 速度控制功能测试通过")
        return True

def test_steering_control():
    """测试转向控制功能"""
    print("\n=== 转向控制功能测试 ===")
    
    with CarController() as controller:
        if not controller.is_connected:
            print("❌ 连接失败")
            return False
        
        # 设置基础速度
        controller.set_speed(0.2)
        time.sleep(0.5)
        
        # 测试不同转向角度
        steering_values = [-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8]
        
        for i, steering in enumerate(steering_values, 1):
            print(f"{i}. 测试转向 {steering:+.1f}...")
            if controller.set_steering(steering):
                print(f"✅ 转向 {steering:+.1f} 设置成功")
                time.sleep(1)
            else:
                print(f"❌ 转向 {steering:+.1f} 设置失败")
                return False
        
        # 停止
        controller.stop()
        print("✅ 转向控制功能测试通过")
        return True

def test_combined_control():
    """测试组合控制功能"""
    print("\n=== 组合控制功能测试 ===")
    
    with CarController() as controller:
        if not controller.is_connected:
            print("❌ 连接失败")
            return False
        
        # 测试不同的速度和转向组合
        test_cases = [
            (0.3, 0.0),   # 直行
            (0.4, -0.3),  # 左转
            (0.4, 0.3),   # 右转
            (0.6, -0.5),  # 快速左转
            (0.6, 0.5),   # 快速右转
            (0.2, -0.8),  # 慢速大角度左转
            (0.2, 0.8),   # 慢速大角度右转
        ]
        
        for i, (speed, steering) in enumerate(test_cases, 1):
            print(f"{i}. 测试组合: 速度={speed:.1f}, 转向={steering:+.1f}...")
            if controller.set_motion(speed, steering):
                print(f"✅ 组合控制设置成功")
                time.sleep(1.5)
            else:
                print(f"❌ 组合控制设置失败")
                return False
        
        # 停止
        controller.stop()
        print("✅ 组合控制功能测试通过")
        return True

def test_status_query():
    """测试状态查询功能"""
    print("\n=== 状态查询功能测试 ===")
    
    with CarController() as controller:
        if not controller.is_connected:
            print("❌ 连接失败")
            return False
        
        # 查询当前状态
        print("1. 查询当前状态...")
        status = controller.get_status()
        if status:
            print("✅ 状态查询成功")
            print(f"   当前状态: {status}")
        else:
            print("❌ 状态查询失败")
            return False
        
        # 设置运动后再次查询
        print("2. 设置运动后查询状态...")
        controller.set_motion(0.4, 0.2)
        time.sleep(0.5)
        
        status = controller.get_status()
        if status:
            print("✅ 运动状态查询成功")
            print(f"   运动状态: {status}")
        else:
            print("❌ 运动状态查询失败")
            return False
        
        # 停止
        controller.stop()
        print("✅ 状态查询功能测试通过")
        return True

def test_error_handling():
    """测试错误处理功能"""
    print("\n=== 错误处理功能测试 ===")
    
    with CarController() as controller:
        if not controller.is_connected:
            print("❌ 连接失败")
            return False
        
        # 测试超出范围的值
        print("1. 测试超出范围的速度值...")
        if controller.set_speed(1.5):  # 超出最大速度
            print("❌ 应该拒绝超出范围的速度值")
            return False
        else:
            print("✅ 正确拒绝超出范围的速度值")
        
        print("2. 测试超出范围的转向值...")
        if controller.set_steering(1.5):  # 超出最大转向
            print("❌ 应该拒绝超出范围的转向值")
            return False
        else:
            print("✅ 正确拒绝超出范围的转向值")
        
        # 测试负速度
        print("3. 测试负速度值...")
        if controller.set_speed(-0.5):  # 负速度
            print("❌ 应该拒绝负速度值")
            return False
        else:
            print("✅ 正确拒绝负速度值")
        
        print("✅ 错误处理功能测试通过")
        return True

def main():
    """主测试函数"""
    print("🚗 小车控制模块测试开始")
    print("=" * 50)
    
    tests = [
        ("基本控制功能", test_basic_control),
        ("速度控制功能", test_speed_control),
        ("转向控制功能", test_steering_control),
        ("组合控制功能", test_combined_control),
        ("状态查询功能", test_status_query),
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
        print("🎉 所有测试通过！小车控制模块工作正常。")
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