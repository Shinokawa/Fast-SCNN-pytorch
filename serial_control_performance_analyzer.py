#!/usr/bin/env python3
"""
串口控制性能分析工具
分析8fps下的控制信息更新性能
"""

import time
import threading
import struct
from typing import List, Dict
import statistics
import numpy as np

class SerialControlPerformanceAnalyzer:
    """串口控制性能分析器"""
    
    def __init__(self):
        self.control_timestamps = []
        self.serial_send_times = []
        self.frame_processing_times = []
        self.control_intervals = []
        self.serial_latencies = []
        self.lock = threading.Lock()
        
        # 性能统计
        self.total_frames = 0
        self.successful_sends = 0
        self.failed_sends = 0
        self.max_interval = 0
        self.min_interval = float('inf')
        
    def record_frame_start(self):
        """记录帧处理开始时间"""
        return time.time()
    
    def record_control_calculation(self, start_time):
        """记录控制计算完成时间"""
        calculation_time = (time.time() - start_time) * 1000
        with self.lock:
            self.frame_processing_times.append(calculation_time)
        return calculation_time
    
    def record_serial_send(self, success: bool):
        """记录串口发送结果"""
        current_time = time.time()
        
        with self.lock:
            self.control_timestamps.append(current_time)
            
            if success:
                self.successful_sends += 1
                # 计算发送间隔
                if len(self.control_timestamps) > 1:
                    interval = current_time - self.control_timestamps[-2]
                    self.control_intervals.append(interval)
                    self.max_interval = max(self.max_interval, interval)
                    self.min_interval = min(self.min_interval, interval)
            else:
                self.failed_sends += 1
            
            self.total_frames += 1
    
    def record_serial_latency(self, latency_ms: float):
        """记录串口发送延迟"""
        with self.lock:
            self.serial_latencies.append(latency_ms)
    
    def get_performance_analysis(self) -> Dict:
        """获取性能分析结果"""
        with self.lock:
            if not self.control_timestamps:
                return {"error": "没有数据"}
            
            # 计算基本统计
            total_time = self.control_timestamps[-1] - self.control_timestamps[0] if len(self.control_timestamps) > 1 else 0
            avg_fps = len(self.control_timestamps) / total_time if total_time > 0 else 0
            success_rate = self.successful_sends / self.total_frames if self.total_frames > 0 else 0
            
            analysis = {
                "basic_stats": {
                    "total_frames": self.total_frames,
                    "successful_sends": self.successful_sends,
                    "failed_sends": self.failed_sends,
                    "success_rate": success_rate * 100,
                    "average_fps": avg_fps,
                    "total_time_seconds": total_time
                }
            }
            
            # 控制间隔分析
            if self.control_intervals:
                analysis["control_intervals"] = {
                    "count": len(self.control_intervals),
                    "min_interval_ms": self.min_interval * 1000,
                    "max_interval_ms": self.max_interval * 1000,
                    "avg_interval_ms": statistics.mean(self.control_intervals) * 1000,
                    "std_interval_ms": statistics.stdev(self.control_intervals) * 1000 if len(self.control_intervals) > 1 else 0,
                    "median_interval_ms": statistics.median(self.control_intervals) * 1000
                }
                
                # 8fps目标分析
                target_interval_ms = 1000 / 8  # 125ms
                intervals_ms = [x * 1000 for x in self.control_intervals]
                within_target = sum(1 for x in intervals_ms if x <= target_interval_ms * 1.1)  # 允许10%误差
                
                analysis["fps_target_analysis"] = {
                    "target_fps": 8,
                    "target_interval_ms": target_interval_ms,
                    "within_target_count": within_target,
                    "within_target_percentage": (within_target / len(intervals_ms)) * 100,
                    "can_achieve_8fps": within_target / len(intervals_ms) > 0.9  # 90%的帧能达到目标
                }
            
            # 帧处理时间分析
            if self.frame_processing_times:
                analysis["frame_processing"] = {
                    "count": len(self.frame_processing_times),
                    "min_time_ms": min(self.frame_processing_times),
                    "max_time_ms": max(self.frame_processing_times),
                    "avg_time_ms": statistics.mean(self.frame_processing_times),
                    "std_time_ms": statistics.stdev(self.frame_processing_times) if len(self.frame_processing_times) > 1 else 0
                }
            
            # 串口延迟分析
            if self.serial_latencies:
                analysis["serial_latency"] = {
                    "count": len(self.serial_latencies),
                    "min_latency_ms": min(self.serial_latencies),
                    "max_latency_ms": max(self.serial_latencies),
                    "avg_latency_ms": statistics.mean(self.serial_latencies),
                    "std_latency_ms": statistics.stdev(self.serial_latencies) if len(self.serial_latencies) > 1 else 0
                }
            
            return analysis
    
    def print_performance_report(self):
        """打印性能报告"""
        analysis = self.get_performance_analysis()
        
        if "error" in analysis:
            print(f"❌ {analysis['error']}")
            return
        
        print("🚗 串口控制性能分析报告")
        print("=" * 60)
        
        # 基本统计
        basic = analysis["basic_stats"]
        print(f"📊 基本统计:")
        print(f"   总帧数: {basic['total_frames']}")
        print(f"   成功发送: {basic['successful_sends']}")
        print(f"   失败发送: {basic['failed_sends']}")
        print(f"   成功率: {basic['success_rate']:.1f}%")
        print(f"   平均FPS: {basic['average_fps']:.2f}")
        print(f"   运行时长: {basic['total_time_seconds']:.1f}秒")
        
        # 控制间隔分析
        if "control_intervals" in analysis:
            intervals = analysis["control_intervals"]
            print(f"\n⏱️ 控制更新间隔:")
            print(f"   最小间隔: {intervals['min_interval_ms']:.1f}ms")
            print(f"   最大间隔: {intervals['max_interval_ms']:.1f}ms")
            print(f"   平均间隔: {intervals['avg_interval_ms']:.1f}ms")
            print(f"   标准差: {intervals['std_interval_ms']:.1f}ms")
            print(f"   中位数: {intervals['median_interval_ms']:.1f}ms")
        
        # 8fps目标分析
        if "fps_target_analysis" in analysis:
            target = analysis["fps_target_analysis"]
            print(f"\n🎯 8FPS目标分析:")
            print(f"   目标间隔: {target['target_interval_ms']:.1f}ms")
            print(f"   达标次数: {target['within_target_count']}/{intervals['count']}")
            print(f"   达标率: {target['within_target_percentage']:.1f}%")
            print(f"   能否达到8FPS: {'✅ 是' if target['can_achieve_8fps'] else '❌ 否'}")
            
            if target['can_achieve_8fps']:
                print(f"   💡 评估: 系统可以稳定实现每秒8次控制更新")
            else:
                print(f"   ⚠️ 评估: 系统无法稳定达到8FPS控制更新")
        
        # 帧处理时间
        if "frame_processing" in analysis:
            processing = analysis["frame_processing"]
            print(f"\n🔄 帧处理性能:")
            print(f"   最快处理: {processing['min_time_ms']:.1f}ms")
            print(f"   最慢处理: {processing['max_time_ms']:.1f}ms")
            print(f"   平均处理: {processing['avg_time_ms']:.1f}ms")
            print(f"   处理稳定性: {processing['std_time_ms']:.1f}ms (标准差)")
        
        # 串口延迟
        if "serial_latency" in analysis:
            latency = analysis["serial_latency"]
            print(f"\n📡 串口通信延迟:")
            print(f"   最低延迟: {latency['min_latency_ms']:.1f}ms")
            print(f"   最高延迟: {latency['max_latency_ms']:.1f}ms")
            print(f"   平均延迟: {latency['avg_latency_ms']:.1f}ms")
            print(f"   延迟稳定性: {latency['std_latency_ms']:.1f}ms (标准差)")
        
        # 性能建议
        self._print_performance_recommendations(analysis)
    
    def _print_performance_recommendations(self, analysis):
        """打印性能建议"""
        print(f"\n💡 性能优化建议:")
        
        if "fps_target_analysis" in analysis:
            target = analysis["fps_target_analysis"]
            if target['can_achieve_8fps']:
                print("   ✅ 当前性能已满足8FPS控制更新要求")
                print("   🚀 可以考虑进一步优化以提高响应精度")
            else:
                print("   ⚠️ 当前性能无法稳定达到8FPS控制更新")
                
                # 分析瓶颈
                if "frame_processing" in analysis:
                    processing = analysis["frame_processing"]
                    if processing['avg_time_ms'] > 100:  # 超过100ms
                        print("   📊 瓶颈：帧处理时间过长，建议优化推理算法")
                
                if "serial_latency" in analysis:
                    latency = analysis["serial_latency"]
                    if latency['avg_latency_ms'] > 20:  # 超过20ms
                        print("   📡 瓶颈：串口通信延迟过高，检查波特率和硬件")
                
                if "control_intervals" in analysis:
                    intervals = analysis["control_intervals"]
                    if intervals['std_interval_ms'] > 50:  # 标准差超过50ms
                        print("   ⏱️ 瓶颈：控制间隔不稳定，系统负载可能过高")
        
        # 通用建议
        print("   🔧 优化建议:")
        print("     - 使用边缘计算模式减少图像处理负载")
        print("     - 提高串口波特率(如921600)")
        print("     - 减少日志输出频率")
        print("     - 使用专用线程处理串口通信")
        print("     - 考虑异步发送控制指令")

def simulate_8fps_control_loop():
    """模拟8FPS控制循环测试"""
    print("🧪 模拟8FPS控制循环性能测试")
    print("=" * 50)
    
    analyzer = SerialControlPerformanceAnalyzer()
    target_fps = 8
    target_interval = 1.0 / target_fps  # 0.125秒
    test_duration = 10  # 10秒测试
    
    print(f"📋 测试参数:")
    print(f"   目标FPS: {target_fps}")
    print(f"   目标间隔: {target_interval*1000:.1f}ms")
    print(f"   测试时长: {test_duration}秒")
    print(f"   预期帧数: {target_fps * test_duration}")
    
    print(f"\n🏃 开始模拟测试...")
    
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < test_duration:
        frame_start = analyzer.record_frame_start()
        
        # 模拟控制计算（包括推理、透视变换、路径规划）
        # 根据实际情况，这部分通常需要80-120ms
        simulated_processing_time = 0.095 + np.random.normal(0, 0.015)  # 95ms ± 15ms
        time.sleep(max(0, simulated_processing_time))
        
        processing_time = analyzer.record_control_calculation(frame_start)
        
        # 模拟串口发送
        serial_start = time.time()
        # 串口发送通常很快，1-5ms
        simulated_serial_time = 0.002 + np.random.normal(0, 0.001)  # 2ms ± 1ms
        time.sleep(max(0, simulated_serial_time))
        serial_latency = (time.time() - serial_start) * 1000
        
        # 模拟发送成功（95%成功率）
        success = np.random.random() > 0.05
        analyzer.record_serial_send(success)
        analyzer.record_serial_latency(serial_latency)
        
        frame_count += 1
        
        # 保持目标帧率（如果可能）
        elapsed = time.time() - frame_start
        if elapsed < target_interval:
            time.sleep(target_interval - elapsed)
    
    print(f"✅ 模拟测试完成，实际执行了 {frame_count} 帧")
    print()
    analyzer.print_performance_report()

def main():
    """主函数"""
    print("🚗 串口控制性能分析工具")
    print("=" * 60)
    print("此工具用于分析8FPS下串口控制信息更新的性能表现")
    print()
    
    print("选择测试模式:")
    print("1. 模拟测试 - 模拟8FPS控制循环")
    print("2. 实际测试说明 - 如何在真实系统中测试")
    print()
    
    choice = input("请选择 (1/2): ").strip()
    
    if choice == "1":
        simulate_8fps_control_loop()
    elif choice == "2":
        print_real_test_instructions()
    else:
        print("❌ 无效选择")

def print_real_test_instructions():
    """打印实际测试说明"""
    print("📋 实际系统性能测试说明")
    print("=" * 50)
    print()
    print("🔧 在真实系统中测试8FPS控制性能:")
    print()
    print("1. 启动系统并启用性能监控:")
    print("   python kuruma/kuruma_control_dashboard.py \\")
    print("     --realtime \\")
    print("     --web \\")
    print("     --enable_serial \\")
    print("     --enable_control \\")
    print("     --no_gui \\")
    print("     --log_file performance_test.log")
    print()
    print("2. 通过Web界面监控:")
    print("   - 访问 http://localhost:5000")
    print("   - 观察FPS显示")
    print("   - 检查延迟数据")
    print("   - 观察控制指令发送频率")
    print()
    print("3. 性能指标判断:")
    print("   ✅ 8FPS可达成条件:")
    print("      - 平均延迟 < 125ms")
    print("      - FPS > 7.5")
    print("      - 串口发送成功率 > 95%")
    print("      - 控制更新间隔稳定")
    print()
    print("   ⚠️ 可能的性能瓶颈:")
    print("      - Atlas推理延迟过高")
    print("      - 透视变换计算耗时")
    print("      - 串口通信阻塞")
    print("      - 系统负载过高")
    print()
    print("4. 优化措施:")
    print("   🔧 软件优化:")
    print("      - 启用边缘计算模式 (--edge_computing)")
    print("      - 降低图像分辨率")
    print("      - 减少日志输出")
    print("      - 优化路径规划参数")
    print()
    print("   🔧 硬件优化:")
    print("      - 提高串口波特率")
    print("      - 使用更快的存储设备")
    print("      - 确保Atlas NPU散热良好")
    print("      - 减少系统后台进程")
    print()
    print("💡 结论:")
    print("根据理论分析，8FPS下每秒8次控制更新是完全可行的。")
    print("关键在于优化每个组件的处理时间，确保总延迟控制在125ms以内。")

if __name__ == "__main__":
    main()
