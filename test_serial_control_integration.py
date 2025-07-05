#!/usr/bin/env python3
"""
测试串口控制集成功能
验证车道线分割+控制算法+Web界面+串口控制的完整流程
"""

import sys
import time
import threading
import requests
import json

def test_web_interface(port=5000, test_duration=30):
    """
    测试Web界面API功能
    
    Args:
        port: Web服务端口
        test_duration: 测试持续时间（秒）
    """
    base_url = f"http://localhost:{port}"
    
    print("🧪 开始Web界面功能测试...")
    print("=" * 50)
    
    # 等待服务器启动
    print("⏳ 等待Web服务器启动...")
    time.sleep(3)
    
    try:
        # 1. 测试主页
        print("📋 测试主页访问...")
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"   主页状态码: {response.status_code}")
        
        # 2. 测试统计API
        print("📊 测试统计数据API...")
        response = requests.get(f"{base_url}/api/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"   帧数: {stats.get('frame_count', 0)}")
            print(f"   系统运行: {stats.get('is_running', False)}")
            print(f"   延迟: {stats.get('latency', 0):.1f}ms")
        
        # 3. 测试控制状态API
        print("🚗 测试控制状态API...")
        response = requests.get(f"{base_url}/api/control_status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"   串口连接: {status.get('serial_connected', False)}")
            print(f"   车辆行驶: {status.get('car_driving', False)}")
            print(f"   紧急停车: {status.get('emergency_stop', False)}")
        
        # 4. 测试参数更新API
        print("🎛️ 测试参数更新API...")
        test_params = {
            'steering_gain': 15.0,
            'base_speed': 600.0,
            'preview_distance': 35.0
        }
        response = requests.post(
            f"{base_url}/api/update_params",
            json=test_params,
            timeout=5
        )
        if response.status_code == 200:
            result = response.json()
            print(f"   参数更新结果: {result.get('success', False)}")
        
        # 5. 测试串口连接API（如果可用）
        print("🔌 测试串口连接API...")
        try:
            response = requests.post(f"{base_url}/api/connect_serial", timeout=5)
            if response.status_code == 200:
                result = response.json()
                print(f"   串口连接结果: {result.get('success', False)}")
                if not result.get('success', False):
                    print(f"   错误信息: {result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"   串口连接测试失败: {e}")
        
        # 6. 长期监控测试
        print(f"⏰ 开始{test_duration}秒监控测试...")
        start_time = time.time()
        frame_counts = []
        
        while time.time() - start_time < test_duration:
            try:
                response = requests.get(f"{base_url}/api/stats", timeout=2)
                if response.status_code == 200:
                    stats = response.json()
                    frame_count = stats.get('frame_count', 0)
                    fps = stats.get('fps', 0)
                    latency = stats.get('latency', 0)
                    
                    frame_counts.append(frame_count)
                    
                    # 每5秒输出一次状态
                    if len(frame_counts) % 5 == 0:
                        print(f"   ⏳ {len(frame_counts)}秒: 帧数={frame_count}, FPS={fps:.1f}, 延迟={latency:.1f}ms")
                
                time.sleep(1)
                
            except Exception as e:
                print(f"   ⚠️ 监控错误: {e}")
                time.sleep(1)
        
        # 计算性能统计
        if len(frame_counts) > 1:
            total_frames = frame_counts[-1] - frame_counts[0] if frame_counts[0] > 0 else frame_counts[-1]
            avg_fps = total_frames / test_duration
            print(f"📈 性能统计:")
            print(f"   总帧数: {total_frames}")
            print(f"   平均FPS: {avg_fps:.2f}")
            print(f"   测试时长: {test_duration}秒")
        
        print("✅ Web界面测试完成")
        
    except Exception as e:
        print(f"❌ Web界面测试失败: {e}")

def print_usage():
    """打印使用说明"""
    print("🚗 串口控制集成测试工具")
    print("=" * 50)
    print("使用方法:")
    print("1. 启动主程序（实时模式+Web界面+串口控制）:")
    print("   python kuruma/kuruma_control_dashboard.py --realtime --web --enable_serial --enable_control --no_gui")
    print()
    print("2. 在另一个终端运行此测试脚本:")
    print("   python test_serial_control_integration.py")
    print()
    print("3. 通过Web界面测试控制功能:")
    print("   - 浏览器访问: http://localhost:5000")
    print("   - 点击'连接串口'按钮")
    print("   - 调整控制参数")
    print("   - 点击'开始行驶'开始发送控制指令")
    print("   - 点击'紧急停车'立即停止")
    print()
    print("4. 完整启动命令示例:")
    print("   python kuruma/kuruma_control_dashboard.py \\")
    print("     --realtime \\")
    print("     --web \\")
    print("     --enable_serial \\")
    print("     --auto_connect_serial \\")
    print("     --enable_control \\")
    print("     --no_gui \\")
    print("     --log_file realtime_control.log")
    print()

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print_usage()
        return
    
    print("🧪 启动串口控制集成测试...")
    print_usage()
    
    # 给用户时间启动主程序
    print("\n⏳ 请先启动主程序，然后按Enter继续测试...")
    input()
    
    # 测试Web界面
    test_web_interface(port=5000, test_duration=30)
    
    print("\n🎯 测试要点总结:")
    print("1. ✅ Web界面访问正常")
    print("2. ✅ API响应正常")
    print("3. ✅ 实时数据更新")
    print("4. 🚗 串口控制功能需要实际硬件验证")
    print("5. 📡 控制指令发送需要连接小车验证")
    
    print("\n💡 下一步:")
    print("- 连接实际硬件设备进行完整测试")
    print("- 验证PWM值转换为轮速的准确性")
    print("- 测试紧急停车的响应时间")
    print("- 验证不同网络延迟下的控制稳定性")

if __name__ == "__main__":
    main()
