#!/usr/bin/env python3
"""
8FPS控制性能实时监控工具
监控串口控制指令的发送频率和延迟
"""

import requests
import time
import json
from datetime import datetime

def monitor_8fps_performance(duration=60, target_fps=8):
    """
    监控8FPS控制性能
    
    Args:
        duration: 监控时长（秒）
        target_fps: 目标FPS
    """
    print("🚗 8FPS控制性能实时监控")
    print("=" * 50)
    print(f"目标FPS: {target_fps}")
    print(f"目标控制间隔: {1000/target_fps:.1f}ms")
    print(f"监控时长: {duration}秒")
    print()
    
    base_url = "http://localhost:5000"
    
    # 性能数据
    fps_samples = []
    latency_samples = []
    control_intervals = []
    last_control_time = None
    
    print("⏳ 开始监控...")
    start_time = time.time()
    
    while time.time() - start_time < duration:
        try:
            # 获取系统状态
            response = requests.get(f"{base_url}/api/stats", timeout=2)
            if response.status_code == 200:
                stats = response.json()
                
                current_time = time.time()
                current_fps = stats.get('fps', 0)
                current_latency = stats.get('latency', 0)
                
                fps_samples.append(current_fps)
                latency_samples.append(current_latency)
                
                # 计算控制间隔
                if last_control_time is not None:
                    interval = (current_time - last_control_time) * 1000  # ms
                    control_intervals.append(interval)
                last_control_time = current_time
                
                # 每5秒输出一次状态
                elapsed = time.time() - start_time
                if int(elapsed) % 5 == 0 and elapsed > 0:
                    avg_fps = sum(fps_samples[-5:]) / min(5, len(fps_samples))
                    avg_latency = sum(latency_samples[-5:]) / min(5, len(latency_samples))
                    
                    print(f"⏰ {elapsed:.0f}s: FPS={avg_fps:.1f}, 延迟={avg_latency:.1f}ms, "
                          f"左轮PWM={stats.get('left_pwm', 0):.0f}, 右轮PWM={stats.get('right_pwm', 0):.0f}")
            
            # 获取控制状态
            response = requests.get(f"{base_url}/api/control_status", timeout=2)
            if response.status_code == 200:
                control_status = response.json()
                
                # 检查串口状态
                if not control_status.get('serial_connected', False):
                    print("⚠️ 警告: 串口未连接")
                
                if not control_status.get('car_driving', False):
                    print("⚠️ 注意: 车辆未处于行驶状态")
            
            time.sleep(1)  # 每秒检查一次
            
        except Exception as e:
            print(f"❌ 监控错误: {e}")
            time.sleep(1)
    
    # 性能分析
    print("\n📊 性能分析结果:")
    print("=" * 50)
    
    if fps_samples:
        avg_fps = sum(fps_samples) / len(fps_samples)
        min_fps = min(fps_samples)
        max_fps = max(fps_samples)
        
        print(f"FPS统计:")
        print(f"   平均FPS: {avg_fps:.2f}")
        print(f"   最小FPS: {min_fps:.2f}")
        print(f"   最大FPS: {max_fps:.2f}")
        print(f"   FPS稳定性: {'良好' if max_fps - min_fps < 2 else '一般' if max_fps - min_fps < 4 else '较差'}")
    
    if latency_samples:
        avg_latency = sum(latency_samples) / len(latency_samples)
        min_latency = min(latency_samples)
        max_latency = max(latency_samples)
        
        print(f"\n延迟统计:")
        print(f"   平均延迟: {avg_latency:.1f}ms")
        print(f"   最小延迟: {min_latency:.1f}ms")
        print(f"   最大延迟: {max_latency:.1f}ms")
        print(f"   延迟稳定性: {'优秀' if max_latency - min_latency < 20 else '良好' if max_latency - min_latency < 50 else '一般'}")
    
    if control_intervals:
        avg_interval = sum(control_intervals) / len(control_intervals)
        target_interval = 1000 / target_fps
        
        print(f"\n控制更新间隔:")
        print(f"   平均间隔: {avg_interval:.1f}ms")
        print(f"   目标间隔: {target_interval:.1f}ms")
        print(f"   间隔偏差: {abs(avg_interval - target_interval):.1f}ms")
        
        # 8FPS达成率
        within_target = sum(1 for x in control_intervals if x <= target_interval * 1.1)
        achievement_rate = (within_target / len(control_intervals)) * 100
        
        print(f"   8FPS达成率: {achievement_rate:.1f}%")
        print(f"   性能评级: {'优秀' if achievement_rate > 90 else '良好' if achievement_rate > 80 else '需要优化'}")
    
    # 总结建议
    print(f"\n💡 性能总结:")
    if avg_fps >= 7.5 and avg_latency <= 130:
        print("✅ 系统性能优秀，完全满足8FPS控制更新要求")
        print("🚀 可以安全地进行高精度车道线跟踪控制")
    elif avg_fps >= 6.5 and avg_latency <= 150:
        print("⚡ 系统性能良好，基本满足控制要求")
        print("🔧 建议启用--edge_computing优化性能")
    else:
        print("⚠️ 系统性能需要优化")
        print("🔧 建议:")
        print("   - 降低摄像头分辨率")
        print("   - 启用边缘计算模式")
        print("   - 检查系统负载")
        print("   - 优化控制参数")

def main():
    print("🚗 8FPS控制性能监控工具")
    print("使用前请确保系统已启动并运行在:")
    print("python kuruma/kuruma_control_dashboard.py --realtime --web --enable_serial --enable_control")
    print()
    
    try:
        # 检查系统是否运行
        response = requests.get("http://localhost:5000/api/stats", timeout=5)
        if response.status_code == 200:
            print("✅ 检测到系统正在运行")
            monitor_8fps_performance(duration=30)  # 30秒监控
        else:
            print("❌ 无法连接到系统，请先启动主程序")
    except Exception as e:
        print(f"❌ 连接错误: {e}")
        print("请确保系统已启动并监听5000端口")

if __name__ == "__main__":
    main()
