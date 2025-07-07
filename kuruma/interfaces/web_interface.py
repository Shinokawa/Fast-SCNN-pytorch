#!/usr/bin/env python3
"""
Web界面模块 - 实时控制台和参数调整界面

包含：
- WEB_TEMPLATE: 完整的HTML模板
- web_data: 全局数据存储
- create_web_app: Flask应用创建
- start_web_server: Web服务器启动
- 所有Flask路由函数
"""

import time
import json
import cv2
import numpy as np
from threading import Thread, Lock

# 尝试导入Flask
try:
    from flask import Flask, render_template_string, jsonify, request, Response
    FLASK_AVAILABLE = True
except ImportError:
    print("⚠️ 警告：Flask未安装，Web界面功能不可用")
    FLASK_AVAILABLE = False

# 尝试导入小车控制器
try:
    from car_controller_simple import SimpleCarController
    CAR_CONTROLLER_AVAILABLE = True
    print("✅ 小车控制模块加载成功")
except ImportError:
    print("⚠️ 警告：未找到car_controller_simple模块，串口控制功能不可用")
    CAR_CONTROLLER_AVAILABLE = False

# ---------------------------------------------------------------------------------
# --- 🌐 Web界面模块 ---
# ---------------------------------------------------------------------------------

# Web界面相关全局变量
web_data = {
    'latest_frame': None,
    'latest_control_map': None,
    'latest_stats': {},
    'is_running': False,
    'frame_count': 0,
    'start_time': None,
    'control_params': {
        'steering_gain': 10.0,
        'base_speed': 500.0,
        'preview_distance': 30.0,
        'curvature_damping': 0.1
    },
    'params_updated': False,
    # 串口控制相关状态
    'serial_enabled': False,      # 串口功能是否启用
    'serial_connected': False,    # 串口是否连接
    'car_driving': False,         # 小车是否正在行驶
    'emergency_stop': False,      # 紧急停车状态
    'last_control_command': None, # 最后发送的控制指令
    'serial_port': '/dev/ttyAMA0', # 串口设备
    'control_enabled': False,     # 控制算法是否激活
    # 避障相关状态
    'obstacle_detection_enabled': False,  # 障碍物检测是否启用
    'obstacle_detected': False,           # 是否检测到障碍物
    'obstacle_count': 0,                  # 检测到的障碍物数量
    'avoidance_active': False,            # 避障是否激活
    'state_machine_state': 'lane_following', # 状态机当前状态
    'state_time': 0.0,                    # 当前状态持续时间
    'control_source': 'none',             # 控制来源(lane_following/obstacle_avoidance)
    'avoidance_command': None,            # 避障指令
    'avoidance_config': {                 # 三段避障配置
        'stage1_left_speed': 400,
        'stage1_right_speed': 700,
        'stage1_duration': 1.5,
        'stage2_left_speed': -300,
        'stage2_right_speed': 600,
        'stage2_duration': 2.0,
        'stage3_left_speed': 500,
        'stage3_right_speed': 200,
        'stage3_duration': 1.5
    }
}
web_data_lock = Lock()

# 🔧 修复：声明为模块级变量，但不初始化，让主文件管理
car_controller = None
control_thread = None
control_enabled = False

# HTML模板
WEB_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>实时车道线分割控制台</title>
    <meta charset="utf-8">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            background: #1a1a1a; 
            color: #fff; 
            margin: 0; 
            padding: 20px; 
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
        }
        .header { 
            text-align: center; 
            margin-bottom: 30px; 
            padding: 20px; 
            background: #2d2d2d; 
            border-radius: 10px; 
        }
        .stats-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: #2d2d2d;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        .stat-label {
            font-size: 14px;
            color: #ccc;
        }
        .image-panel {
            background: #2d2d2d;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .control-map {
            max-width: 100%;
            border: 2px solid #4CAF50;
            border-radius: 8px;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-running { background: #4CAF50; }
        .status-stopped { background: #f44336; }
        .traffic-light-red { background: #FF4444; color: white; }
        .traffic-light-green { background: #4CAF50; color: white; }
        .traffic-light-unknown { background: #FFA500; color: white; }
        .log-panel {
            background: #2d2d2d;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            max-height: 300px;
            overflow-y: auto;
        }
        .log-entry {
            margin-bottom: 5px;
            font-family: monospace;
            font-size: 12px;
        }
        .param-panel {
            background: #2d2d2d;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .param-control {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
            gap: 15px;
        }
        .param-label {
            min-width: 120px;
            font-weight: bold;
            color: #4CAF50;
        }
        .param-slider {
            flex: 1;
            height: 6px;
            border-radius: 3px;
            background: #444;
            outline: none;
            -webkit-appearance: none;
        }
        .param-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #4CAF50;
            cursor: pointer;
        }
        .param-value {
            min-width: 80px;
            text-align: center;
            font-weight: bold;
            color: #fff;
        }
        .param-apply {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }
        .param-apply:hover {
            background: #45a049;
        }
        
        /* 车辆控制面板样式 */
        .control-panel {
            background: #2d2d2d;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 2px solid #FF9800;
        }
        .control-status {
            margin-bottom: 20px;
        }
        .status-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        .status-label {
            font-weight: bold;
            color: #FF9800;
        }
        .status-value {
            font-weight: bold;
            padding: 2px 8px;
            border-radius: 4px;
        }
        .status-connected {
            background: #4CAF50;
            color: white;
        }
        .status-disconnected {
            background: #f44336;
            color: white;
        }
        .status-driving {
            background: #2196F3;
            color: white;
        }
        .status-stopped {
            background: #757575;
            color: white;
        }
        .control-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .control-btn {
            flex: 1;
            min-width: 120px;
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .start-btn {
            background: #4CAF50;
            color: white;
        }
        .start-btn:hover {
            background: #45a049;
            transform: translateY(-2px);
        }
        .stop-btn {
            background: #f44336;
            color: white;
        }
        .stop-btn:hover {
            background: #da190b;
            transform: translateY(-2px);
        }
        .connect-btn {
            background: #2196F3;
            color: white;
        }
        .connect-btn:hover {
            background: #1976D2;
            transform: translateY(-2px);
        }
        .control-btn:disabled {
            background: #666;
            cursor: not-allowed;
            transform: none;
        }
        
        /* 开关按钮样式 */
        .switch {
            position: relative;
            display: inline-block;
            width: 60px;
            height: 34px;
        }
        .switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 34px;
        }
        .slider:before {
            position: absolute;
            content: "";
            height: 26px;
            width: 26px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }
        input:checked + .slider {
            background-color: #4CAF50;
        }
        input:checked + .slider:before {
            transform: translateX(26px);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚗 实时车道线分割控制台</h1>
            <p>
                <span id="status-indicator" class="status-indicator status-stopped"></span>
                <span id="status-text">系统停止</span>
            </p>
        </div>
        
        <div class="stats-panel">
            <div class="stat-card">
                <div class="stat-value" id="frame-count">0</div>
                <div class="stat-label">帧数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="fps">0.0</div>
                <div class="stat-label">FPS</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="latency">0</div>
                <div class="stat-label">延迟(ms)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="lane-ratio">0.0</div>
                <div class="stat-label">车道线覆盖率(%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="left-pwm">0</div>
                <div class="stat-label">左轮PWM</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="right-pwm">0</div>
                <div class="stat-label">右轮PWM</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="lateral-error">0.0</div>
                <div class="stat-label">横向误差(cm)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="path-curvature">0.0</div>
                <div class="stat-label">路径曲率</div>
            </div>
        </div>
        
        <div class="param-panel">
            <h3>🎛️ 控制参数实时调整</h3>
            <div class="param-control">
                <span class="param-label">转向增益</span>
                <input type="range" class="param-slider" id="steering-gain-slider" 
                       min="1" max="50" step="0.5" value="10">
                <span class="param-value" id="steering-gain-value">10.0</span>
            </div>
            <div class="param-control">
                <span class="param-label">基础PWM</span>
                <input type="range" class="param-slider" id="base-speed-slider" 
                       min="100" max="1000" step="10" value="500">
                <span class="param-value" id="base-speed-value">500</span>
            </div>
            <div class="param-control">
                <span class="param-label">预瞄距离(cm)</span>
                <input type="range" class="param-slider" id="preview-distance-slider" 
                       min="10" max="100" step="1" value="30">
                <span class="param-value" id="preview-distance-value">30</span>
            </div>
            <div class="param-control">
                <span class="param-label">阻尼系数</span>
                <input type="range" class="param-slider" id="curvature-damping-slider" 
                       min="0.01" max="1.0" step="0.01" value="0.1">
                <span class="param-value" id="curvature-damping-value">0.1</span>
            </div>
            <div class="param-control">
                <span class="param-label">EMA平滑系数</span>
                <input type="range" class="param-slider" id="ema-alpha-slider" 
                       min="0.1" max="1.0" step="0.01" value="0.5">
                <span class="param-value" id="ema-alpha-value">0.5</span>
            </div>
            <div class="param-control">
                <span class="param-label">启用平滑</span>
                <label class="switch">
                    <input type="checkbox" id="enable-smoothing-checkbox" checked>
                    <span class="slider"></span>
                </label>
                <span class="param-value" id="enable-smoothing-value">启用</span>
            </div>
            <div style="text-align: center; margin-top: 15px;">
                <button class="param-apply" onclick="applyParameters()">应用参数</button>
            </div>
        </div>
        
        <div class="control-panel">
            <h3>🚗 车辆控制</h3>
            <div class="control-status">
                <div class="status-item">
                    <span class="status-label">串口状态:</span>
                    <span id="serial-status" class="status-value status-disconnected">未连接</span>
                </div>
                <div class="status-item">
                    <span class="status-label">行驶状态:</span>
                    <span id="driving-status" class="status-value status-stopped">停止</span>
                </div>
            </div>
            <div class="control-buttons">
                <button id="start-driving-btn" class="control-btn start-btn" onclick="startDriving()">
                    🚀 开始行驶
                </button>
                <button id="emergency-stop-btn" class="control-btn stop-btn" onclick="emergencyStop()">
                    🛑 紧急停车
                </button>
                <button id="connect-serial-btn" class="control-btn connect-btn" onclick="connectSerial()">
                    🔌 连接串口
                </button>
            </div>
        </div>
        
        <div class="control-panel">
            <h3>🚧 避障状态</h3>
            <div class="control-status">
                <div class="status-item">
                    <span class="status-label">障碍物检测:</span>
                    <span id="obstacle-detection-status" class="status-value status-disconnected">未启用</span>
                </div>
                <div class="status-item">
                    <span class="status-label">检测到障碍物:</span>
                    <span id="obstacle-detected-status" class="status-value status-stopped">无</span>
                </div>
                <div class="status-item">
                    <span class="status-label">障碍物数量:</span>
                    <span id="obstacle-count" class="status-value">0</span>
                </div>
                <div class="status-item">
                    <span class="status-label">避障状态:</span>
                    <span id="avoidance-status" class="status-value status-stopped">未激活</span>
                </div>
                <div class="status-item">
                    <span class="status-label">状态机模式:</span>
                    <span id="state-machine-mode" class="status-value">lane_following</span>
                </div>
                <div class="status-item">
                    <span class="status-label">控制来源:</span>
                    <span id="control-source" class="status-value">none</span>
                </div>
                <div class="status-item">
                    <span class="status-label">状态持续时间:</span>
                    <span id="state-time" class="status-value">0.0s</span>
                </div>
                <div class="status-item">
                    <span class="status-label">连续避障次数:</span>
                    <span id="consecutive-avoidance-count" class="status-value">0</span>
                </div>
                <div class="status-item">
                    <span class="status-label">避障保护:</span>
                    <span id="avoidance-protection" class="status-value status-stopped">未激活</span>
                </div>
                <div class="status-item">
                    <span class="status-label">冷却剩余时间:</span>
                    <span id="avoidance-cooldown" class="status-value">0.0s</span>
                </div>
            </div>
        </div>
        
        <div class="control-panel">
            <h3>🚦 交通灯状态</h3>
            <div class="control-status">
                <div class="status-item">
                    <span class="status-label">交通灯检测:</span>
                    <span id="traffic-light-detection-status" class="status-value status-disconnected">未启用</span>
                </div>
                <div class="status-item">
                    <span class="status-label">检测到交通灯:</span>
                    <span id="traffic-light-detected-status" class="status-value status-stopped">无</span>
                </div>
                <div class="status-item">
                    <span class="status-label">交通灯状态:</span>
                    <span id="traffic-light-status" class="status-value">unknown</span>
                </div>
                <div class="status-item">
                    <span class="status-label">检测置信度:</span>
                    <span id="traffic-light-confidence" class="status-value">0.0</span>
                </div>
                <div class="status-item">
                    <span class="status-label">下次检测倒计时:</span>
                    <span id="traffic-light-countdown" class="status-value">0帧</span>
                </div>
            </div>
        </div>
        
        <div class="param-panel">
            <h3>🛠️ 三段避障参数实时调整</h3>
            
            <h4 style="color: #FF9800; margin: 20px 0 10px 0;">第一段避障</h4>
            <div class="param-control">
                <span class="param-label">左轮PWM</span>
                <input type="range" class="param-slider" id="stage1-left-speed-slider" 
                       min="-1000" max="1000" step="10" value="400">
                <span class="param-value" id="stage1-left-speed-value">400</span>
            </div>
            <div class="param-control">
                <span class="param-label">右轮PWM</span>
                <input type="range" class="param-slider" id="stage1-right-speed-slider" 
                       min="-1000" max="1000" step="10" value="700">
                <span class="param-value" id="stage1-right-speed-value">700</span>
            </div>
            <div class="param-control">
                <span class="param-label">持续时间(s)</span>
                <input type="range" class="param-slider" id="stage1-duration-slider" 
                       min="0.1" max="5.0" step="0.1" value="1.5">
                <span class="param-value" id="stage1-duration-value">1.5</span>
            </div>
            
            <h4 style="color: #FF9800; margin: 20px 0 10px 0;">第二段避障</h4>
            <div class="param-control">
                <span class="param-label">左轮PWM</span>
                <input type="range" class="param-slider" id="stage2-left-speed-slider" 
                       min="-1000" max="1000" step="10" value="-300">
                <span class="param-value" id="stage2-left-speed-value">-300</span>
            </div>
            <div class="param-control">
                <span class="param-label">右轮PWM</span>
                <input type="range" class="param-slider" id="stage2-right-speed-slider" 
                       min="-1000" max="1000" step="10" value="600">
                <span class="param-value" id="stage2-right-speed-value">600</span>
            </div>
            <div class="param-control">
                <span class="param-label">持续时间(s)</span>
                <input type="range" class="param-slider" id="stage2-duration-slider" 
                       min="0.1" max="5.0" step="0.1" value="2.0">
                <span class="param-value" id="stage2-duration-value">2.0</span>
            </div>
            
            <h4 style="color: #FF9800; margin: 20px 0 10px 0;">第三段避障</h4>
            <div class="param-control">
                <span class="param-label">左轮PWM</span>
                <input type="range" class="param-slider" id="stage3-left-speed-slider" 
                       min="-1000" max="1000" step="10" value="500">
                <span class="param-value" id="stage3-left-speed-value">500</span>
            </div>
            <div class="param-control">
                <span class="param-label">右轮PWM</span>
                <input type="range" class="param-slider" id="stage3-right-speed-slider" 
                       min="-1000" max="1000" step="10" value="200">
                <span class="param-value" id="stage3-right-speed-value">200</span>
            </div>
            <div class="param-control">
                <span class="param-label">持续时间(s)</span>
                <input type="range" class="param-slider" id="stage3-duration-slider" 
                       min="0.1" max="5.0" step="0.1" value="1.5">
                <span class="param-value" id="stage3-duration-value">1.5</span>
            </div>
            
            <div style="text-align: center; margin-top: 20px;">
                <button class="param-apply" onclick="applyAvoidanceParameters()">应用避障参数</button>
            </div>
        </div>
        
        <div class="image-panel">
            <h3>🗺️ 实时控制地图</h3>
            <img id="control-map" class="control-map" src="/api/control_map" alt="控制地图加载中...">
        </div>
        
        <div class="log-panel">
            <h3>📋 系统日志</h3>
            <div id="log-content"></div>
        </div>
    </div>
    
    <script>
        let logEntries = [];
        const maxLogEntries = 50;
        
        function updateStats() {
            fetch('/api/stats')
            .then(response => response.json())
            .then(data => {
                document.getElementById('frame-count').textContent = data.frame_count || 0;
                document.getElementById('fps').textContent = (data.fps || 0).toFixed(1);
                document.getElementById('latency').textContent = Math.round(data.latency || 0);
                document.getElementById('lane-ratio').textContent = (data.lane_ratio || 0).toFixed(1);
                document.getElementById('left-pwm').textContent = Math.round(data.left_pwm || 0);
                document.getElementById('right-pwm').textContent = Math.round(data.right_pwm || 0);
                document.getElementById('lateral-error').textContent = (data.lateral_error || 0).toFixed(1);
                document.getElementById('path-curvature').textContent = (data.path_curvature || 0).toFixed(3);
                
                // 更新状态指示器
                const statusIndicator = document.getElementById('status-indicator');
                const statusText = document.getElementById('status-text');
                if (data.is_running) {
                    statusIndicator.className = 'status-indicator status-running';
                    statusText.textContent = '系统运行中';
                } else {
                    statusIndicator.className = 'status-indicator status-stopped';
                    statusText.textContent = '系统停止';
                }
                
                // 更新避障状态
                updateAvoidanceStatus(data);
                
                // 更新交通灯状态
                updateTrafficLightStatus(data);
                
                // 添加新日志条目
                if (data.latest_log) {
                    addLogEntry(data.latest_log);
                }
            })
            .catch(error => console.error('获取状态失败:', error));
        }
        
        function updateAvoidanceStatus(data) {
            // 更新障碍物检测状态
            const obstacleDetectionStatus = document.getElementById('obstacle-detection-status');
            if (data.obstacle_detection_enabled) {
                obstacleDetectionStatus.textContent = '已启用';
                obstacleDetectionStatus.className = 'status-value status-connected';
            } else {
                obstacleDetectionStatus.textContent = '未启用';
                obstacleDetectionStatus.className = 'status-value status-disconnected';
            }
            
            // 更新障碍物检测结果
            const obstacleDetectedStatus = document.getElementById('obstacle-detected-status');
            if (data.obstacle_detected) {
                obstacleDetectedStatus.textContent = '检测到';
                obstacleDetectedStatus.className = 'status-value status-driving';
            } else {
                obstacleDetectedStatus.textContent = '无';
                obstacleDetectedStatus.className = 'status-value status-stopped';
            }
            
            // 更新障碍物数量
            document.getElementById('obstacle-count').textContent = data.obstacle_count || 0;
            
            // 更新避障状态
            const avoidanceStatus = document.getElementById('avoidance-status');
            if (data.avoidance_active) {
                avoidanceStatus.textContent = '激活中';
                avoidanceStatus.className = 'status-value status-driving';
            } else {
                avoidanceStatus.textContent = '未激活';
                avoidanceStatus.className = 'status-value status-stopped';
            }
            
            // 更新状态机模式
            const stateMachineMode = document.getElementById('state-machine-mode');
            stateMachineMode.textContent = data.state_machine_state || 'unknown';
            
            // 根据状态机模式设置颜色
            if (data.state_machine_state === 'lane_following') {
                stateMachineMode.className = 'status-value status-connected';
            } else if (data.state_machine_state === 'obstacle_avoidance') {
                stateMachineMode.className = 'status-value status-driving';
            } else if (data.state_machine_state === 'emergency_stop') {
                stateMachineMode.className = 'status-value status-disconnected';
            } else {
                stateMachineMode.className = 'status-value status-stopped';
            }
            
            // 更新控制来源
            const controlSource = document.getElementById('control-source');
            controlSource.textContent = data.control_source || 'none';
            
            // 根据控制来源设置颜色
            if (data.control_source === 'lane_following') {
                controlSource.className = 'status-value status-connected';
            } else if (data.control_source === 'obstacle_avoidance') {
                controlSource.className = 'status-value status-driving';
            } else {
                controlSource.className = 'status-value status-stopped';
            }
            
            // 更新状态持续时间
            document.getElementById('state-time').textContent = (data.state_time || 0).toFixed(1) + 's';
            
            // 更新防死循环保护状态
            document.getElementById('consecutive-avoidance-count').textContent = data.consecutive_avoidance_count || 0;
            
            // 更新避障保护状态
            const avoidanceProtection = document.getElementById('avoidance-protection');
            if (data.avoidance_protection_active) {
                avoidanceProtection.textContent = '激活中';
                avoidanceProtection.className = 'status-value status-driving';
            } else {
                avoidanceProtection.textContent = '未激活';
                avoidanceProtection.className = 'status-value status-stopped';
            }
            
            // 更新冷却剩余时间
            const cooldownTime = data.avoidance_cooldown_remaining || 0;
            document.getElementById('avoidance-cooldown').textContent = cooldownTime.toFixed(1) + 's';
        }
        
        function updateTrafficLightStatus(data) {
            // 更新交通灯检测状态
            const trafficLightDetectionStatus = document.getElementById('traffic-light-detection-status');
            if (data.traffic_light_detection_enabled) {
                trafficLightDetectionStatus.textContent = '已启用';
                trafficLightDetectionStatus.className = 'status-value status-connected';
            } else {
                trafficLightDetectionStatus.textContent = '未启用';
                trafficLightDetectionStatus.className = 'status-value status-disconnected';
            }
            
            // 更新交通灯检测结果
            const trafficLightDetectedStatus = document.getElementById('traffic-light-detected-status');
            if (data.traffic_light_detected) {
                trafficLightDetectedStatus.textContent = '检测到';
                trafficLightDetectedStatus.className = 'status-value status-driving';
            } else {
                trafficLightDetectedStatus.textContent = '无';
                trafficLightDetectedStatus.className = 'status-value status-stopped';
            }
            
            // 更新交通灯状态
            const trafficLightStatus = document.getElementById('traffic-light-status');
            const status = data.traffic_light_status || 'unknown';
            trafficLightStatus.textContent = status;
            
            // 根据交通灯状态设置颜色
            if (status === 'red') {
                trafficLightStatus.className = 'status-value traffic-light-red';
            } else if (status === 'green') {
                trafficLightStatus.className = 'status-value traffic-light-green';
            } else {
                trafficLightStatus.className = 'status-value traffic-light-unknown';
            }
            
            // 更新交通灯检测置信度
            const confidence = data.traffic_light_confidence || 0.0;
            document.getElementById('traffic-light-confidence').textContent = confidence.toFixed(2);
            
            // 更新下次检测倒计时
            const countdown = data.traffic_light_countdown || 0;
            document.getElementById('traffic-light-countdown').textContent = countdown + '帧';
        }
        
        function addLogEntry(logText) {
            const timestamp = new Date().toLocaleTimeString();
            logEntries.push(`[${timestamp}] ${logText}`);
            if (logEntries.length > maxLogEntries) {
                logEntries.shift();
            }
            
            const logContent = document.getElementById('log-content');
            logContent.innerHTML = logEntries.map(entry => 
                `<div class="log-entry">${entry}</div>`
            ).join('');
            logContent.scrollTop = logContent.scrollHeight;
        }
        
        // 定期更新控制地图
        function updateControlMap() {
            const img = document.getElementById('control-map');
            img.src = '/api/control_map?' + new Date().getTime();
        }
        
        // 参数滑块更新显示值
        function updateSliderValues() {
            const steeringGain = document.getElementById('steering-gain-slider');
            const steeringValue = document.getElementById('steering-gain-value');
            steeringValue.textContent = parseFloat(steeringGain.value).toFixed(1);
            
            const baseSpeed = document.getElementById('base-speed-slider');
            const baseValue = document.getElementById('base-speed-value');
            baseValue.textContent = baseSpeed.value;
            
            const previewDistance = document.getElementById('preview-distance-slider');
            const previewValue = document.getElementById('preview-distance-value');
            previewValue.textContent = previewDistance.value;
            
            const curvatureDamping = document.getElementById('curvature-damping-slider');
            const dampingValue = document.getElementById('curvature-damping-value');
            dampingValue.textContent = parseFloat(curvatureDamping.value).toFixed(2);
            
            const emaAlpha = document.getElementById('ema-alpha-slider');
            const emaValue = document.getElementById('ema-alpha-value');
            emaValue.textContent = parseFloat(emaAlpha.value).toFixed(2);
            
            const enableSmoothing = document.getElementById('enable-smoothing-checkbox');
            const smoothingValue = document.getElementById('enable-smoothing-value');
            smoothingValue.textContent = enableSmoothing.checked ? '启用' : '禁用';
        }
        
        // 应用参数到系统
        function applyParameters() {
            const steeringGain = document.getElementById('steering-gain-slider').value;
            const baseSpeed = document.getElementById('base-speed-slider').value;
            const previewDistance = document.getElementById('preview-distance-slider').value;
            const curvatureDamping = document.getElementById('curvature-damping-slider').value;
            const emaAlpha = document.getElementById('ema-alpha-slider').value;
            const enableSmoothing = document.getElementById('enable-smoothing-checkbox').checked;
            
            const params = {
                steering_gain: parseFloat(steeringGain),
                base_speed: parseFloat(baseSpeed),
                preview_distance: parseFloat(previewDistance),
                curvature_damping: parseFloat(curvatureDamping),
                ema_alpha: parseFloat(emaAlpha),
                enable_smoothing: enableSmoothing
            };
            
            fetch('/api/update_params', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(params)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    addLogEntry(`参数更新成功: 转向增益=${steeringGain}, 基础PWM=${baseSpeed}, 预瞄距离=${previewDistance}cm, 阻尼系数=${curvatureDamping}, EMA平滑=${enableSmoothing ? '启用' : '禁用'}(α=${emaAlpha})`);
                } else {
                    addLogEntry(`参数更新失败: ${data.error}`);
                }
            })
            .catch(error => {
                addLogEntry(`参数更新错误: ${error}`);
                console.error('参数更新失败:', error);
            });
        }
        
        // 车辆控制函数
        function connectSerial() {
            fetch('/api/connect_serial', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    addLogEntry('串口连接成功');
                    updateControlStatus();
                } else {
                    addLogEntry(`串口连接失败: ${data.error}`);
                }
            })
            .catch(error => {
                addLogEntry(`串口连接错误: ${error}`);
                console.error('串口连接失败:', error);
            });
        }
        
        function startDriving() {
            fetch('/api/start_driving', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    addLogEntry('🚀 开始行驶模式');
                    updateControlStatus();
                } else {
                    addLogEntry(`启动行驶失败: ${data.error}`);
                }
            })
            .catch(error => {
                addLogEntry(`启动行驶错误: ${error}`);
                console.error('启动行驶失败:', error);
            });
        }
        
        function emergencyStop() {
            fetch('/api/emergency_stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    addLogEntry('🛑 紧急停车执行成功');
                    updateControlStatus();
                } else {
                    addLogEntry(`紧急停车失败: ${data.error}`);
                }
            })
            .catch(error => {
                addLogEntry(`紧急停车错误: ${error}`);
                console.error('紧急停车失败:', error);
            });
        }
        
        function updateControlStatus() {
            fetch('/api/control_status')
            .then(response => response.json())
            .then(data => {
                // 更新串口状态
                const serialStatus = document.getElementById('serial-status');
                if (data.serial_connected) {
                    serialStatus.textContent = '已连接';
                    serialStatus.className = 'status-value status-connected';
                } else {
                    serialStatus.textContent = '未连接';
                    serialStatus.className = 'status-value status-disconnected';
                }
                
                // 更新行驶状态
                const drivingStatus = document.getElementById('driving-status');
                if (data.car_driving) {
                    drivingStatus.textContent = '行驶中';
                    drivingStatus.className = 'status-value status-driving';
                } else {
                    drivingStatus.textContent = '停止';
                    drivingStatus.className = 'status-value status-stopped';
                }
                
                // 更新按钮状态
                const startBtn = document.getElementById('start-driving-btn');
                const stopBtn = document.getElementById('emergency-stop-btn');
                const connectBtn = document.getElementById('connect-serial-btn');
                
                if (data.serial_connected) {
                    connectBtn.textContent = '🔌 串口已连接';
                    connectBtn.disabled = true;
                    startBtn.disabled = false;
                    stopBtn.disabled = false;
                } else {
                    connectBtn.textContent = '🔌 连接串口';
                    connectBtn.disabled = false;
                    startBtn.disabled = true;
                    stopBtn.disabled = true;
                }
                
                if (data.car_driving) {
                    startBtn.disabled = true;
                    startBtn.textContent = '🚗 行驶中';
                } else {
                    if (data.serial_connected) {
                        startBtn.disabled = false;
                    }
                    startBtn.textContent = '🚀 开始行驶';
                }
            })
            .catch(error => {
                console.error('获取控制状态失败:', error);
            });
        }
        
        // 避障参数滑块更新显示值
        function updateAvoidanceSliderValues() {
            // 第一段避障参数
            const stage1LeftSpeed = document.getElementById('stage1-left-speed-slider');
            const stage1LeftValue = document.getElementById('stage1-left-speed-value');
            stage1LeftValue.textContent = stage1LeftSpeed.value;
            
            const stage1RightSpeed = document.getElementById('stage1-right-speed-slider');
            const stage1RightValue = document.getElementById('stage1-right-speed-value');
            stage1RightValue.textContent = stage1RightSpeed.value;
            
            const stage1Duration = document.getElementById('stage1-duration-slider');
            const stage1DurationValue = document.getElementById('stage1-duration-value');
            stage1DurationValue.textContent = parseFloat(stage1Duration.value).toFixed(1);
            
            // 第二段避障参数
            const stage2LeftSpeed = document.getElementById('stage2-left-speed-slider');
            const stage2LeftValue = document.getElementById('stage2-left-speed-value');
            stage2LeftValue.textContent = stage2LeftSpeed.value;
            
            const stage2RightSpeed = document.getElementById('stage2-right-speed-slider');
            const stage2RightValue = document.getElementById('stage2-right-speed-value');
            stage2RightValue.textContent = stage2RightSpeed.value;
            
            const stage2Duration = document.getElementById('stage2-duration-slider');
            const stage2DurationValue = document.getElementById('stage2-duration-value');
            stage2DurationValue.textContent = parseFloat(stage2Duration.value).toFixed(1);
            
            // 第三段避障参数
            const stage3LeftSpeed = document.getElementById('stage3-left-speed-slider');
            const stage3LeftValue = document.getElementById('stage3-left-speed-value');
            stage3LeftValue.textContent = stage3LeftSpeed.value;
            
            const stage3RightSpeed = document.getElementById('stage3-right-speed-slider');
            const stage3RightValue = document.getElementById('stage3-right-speed-value');
            stage3RightValue.textContent = stage3RightSpeed.value;
            
            const stage3Duration = document.getElementById('stage3-duration-slider');
            const stage3DurationValue = document.getElementById('stage3-duration-value');
            stage3DurationValue.textContent = parseFloat(stage3Duration.value).toFixed(1);
        }
        
        // 应用三段避障参数到系统
        function applyAvoidanceParameters() {
            const stage1LeftSpeed = document.getElementById('stage1-left-speed-slider').value;
            const stage1RightSpeed = document.getElementById('stage1-right-speed-slider').value;
            const stage1Duration = document.getElementById('stage1-duration-slider').value;
            
            const stage2LeftSpeed = document.getElementById('stage2-left-speed-slider').value;
            const stage2RightSpeed = document.getElementById('stage2-right-speed-slider').value;
            const stage2Duration = document.getElementById('stage2-duration-slider').value;
            
            const stage3LeftSpeed = document.getElementById('stage3-left-speed-slider').value;
            const stage3RightSpeed = document.getElementById('stage3-right-speed-slider').value;
            const stage3Duration = document.getElementById('stage3-duration-slider').value;
            
            const avoidanceParams = {
                stage1_left_speed: parseInt(stage1LeftSpeed),
                stage1_right_speed: parseInt(stage1RightSpeed),
                stage1_duration: parseFloat(stage1Duration),
                stage2_left_speed: parseInt(stage2LeftSpeed),
                stage2_right_speed: parseInt(stage2RightSpeed),
                stage2_duration: parseFloat(stage2Duration),
                stage3_left_speed: parseInt(stage3LeftSpeed),
                stage3_right_speed: parseInt(stage3RightSpeed),
                stage3_duration: parseFloat(stage3Duration)
            };
            
            fetch('/api/update_avoidance_params', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(avoidanceParams)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    addLogEntry(`三段避障参数更新成功: 第一段[${stage1LeftSpeed},${stage1RightSpeed},${stage1Duration}s] 第二段[${stage2LeftSpeed},${stage2RightSpeed},${stage2Duration}s] 第三段[${stage3LeftSpeed},${stage3RightSpeed},${stage3Duration}s]`);
                } else {
                    addLogEntry(`避障参数更新失败: ${data.error}`);
                }
            })
            .catch(error => {
                addLogEntry(`避障参数更新错误: ${error}`);
                console.error('避障参数更新失败:', error);
            });
        }
        
        // 绑定滑块事件
        document.getElementById('steering-gain-slider').addEventListener('input', updateSliderValues);
        document.getElementById('base-speed-slider').addEventListener('input', updateSliderValues);
        document.getElementById('preview-distance-slider').addEventListener('input', updateSliderValues);
        document.getElementById('curvature-damping-slider').addEventListener('input', updateSliderValues);
        document.getElementById('ema-alpha-slider').addEventListener('input', updateSliderValues);
        document.getElementById('enable-smoothing-checkbox').addEventListener('change', updateSliderValues);
        
        // 绑定三段避障参数滑块事件
        document.getElementById('stage1-left-speed-slider').addEventListener('input', updateAvoidanceSliderValues);
        document.getElementById('stage1-right-speed-slider').addEventListener('input', updateAvoidanceSliderValues);
        document.getElementById('stage1-duration-slider').addEventListener('input', updateAvoidanceSliderValues);
        
        document.getElementById('stage2-left-speed-slider').addEventListener('input', updateAvoidanceSliderValues);
        document.getElementById('stage2-right-speed-slider').addEventListener('input', updateAvoidanceSliderValues);
        document.getElementById('stage2-duration-slider').addEventListener('input', updateAvoidanceSliderValues);
        
        document.getElementById('stage3-left-speed-slider').addEventListener('input', updateAvoidanceSliderValues);
        document.getElementById('stage3-right-speed-slider').addEventListener('input', updateAvoidanceSliderValues);
        document.getElementById('stage3-duration-slider').addEventListener('input', updateAvoidanceSliderValues);
        
        // 启动定时更新
        setInterval(updateStats, 1000);  // 每秒更新状态
        setInterval(updateControlMap, 2000);  // 每2秒更新控制地图
        setInterval(updateControlStatus, 1000);  // 每秒更新控制状态
        
        // 初始加载
        updateStats();
        updateSliderValues();
        updateAvoidanceSliderValues();
        updateControlStatus();
    </script>
</body>
</html>
"""

def create_web_app():
    """创建Flask Web应用"""
    if not FLASK_AVAILABLE:
        return None
    
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return render_template_string(WEB_TEMPLATE)
    
    @app.route('/api/stats')
    def get_stats():
        with web_data_lock:
            stats = web_data['latest_stats'].copy()
            stats['is_running'] = web_data['is_running']
            stats['frame_count'] = web_data['frame_count']
            
            # 计算FPS
            if web_data['start_time'] and web_data['frame_count'] > 0:
                elapsed = time.time() - web_data['start_time']
                stats['fps'] = web_data['frame_count'] / elapsed if elapsed > 0 else 0
            else:
                stats['fps'] = 0
            
            # 添加避障相关状态
            stats['obstacle_detection_enabled'] = web_data.get('obstacle_detection_enabled', False)
            stats['obstacle_detected'] = web_data.get('obstacle_detected', False)
            stats['obstacle_count'] = web_data.get('obstacle_count', 0)
            stats['avoidance_active'] = web_data.get('avoidance_active', False)
            stats['state_machine_state'] = web_data.get('state_machine_state', 'unknown')
            stats['state_time'] = web_data.get('state_time', 0.0)
            stats['control_source'] = web_data.get('control_source', 'none')
            
            # 添加交通灯检测相关状态
            stats['traffic_light_detection_enabled'] = web_data.get('traffic_light_detection_enabled', False)
            stats['traffic_light_detected'] = web_data.get('traffic_light_detected', False)
            stats['traffic_light_status'] = web_data.get('traffic_light_status', 'unknown')
            stats['traffic_light_confidence'] = web_data.get('traffic_light_confidence', 0.0)
            stats['traffic_light_countdown'] = web_data.get('traffic_light_countdown', 0)
            
            # 添加防死循环保护相关状态
            stats['consecutive_avoidance_count'] = web_data.get('consecutive_avoidance_count', 0)
            stats['avoidance_protection_active'] = web_data.get('avoidance_protection_active', False)
            stats['avoidance_cooldown_remaining'] = web_data.get('avoidance_cooldown_remaining', 0.0)
        
        return jsonify(stats)
    
    @app.route('/api/update_params', methods=['POST'])
    def update_params():
        try:
            params = request.get_json()
            
            # 验证参数
            if not params:
                return jsonify({'success': False, 'error': '无效的参数数据'})
            
            # 更新全局控制参数
            with web_data_lock:
                if 'control_params' not in web_data:
                    web_data['control_params'] = {}
                
                if 'steering_gain' in params:
                    web_data['control_params']['steering_gain'] = float(params['steering_gain'])
                if 'base_speed' in params:
                    web_data['control_params']['base_speed'] = float(params['base_speed'])
                if 'preview_distance' in params:
                    web_data['control_params']['preview_distance'] = float(params['preview_distance'])
                if 'curvature_damping' in params:
                    web_data['control_params']['curvature_damping'] = float(params['curvature_damping'])
                if 'ema_alpha' in params:
                    web_data['control_params']['ema_alpha'] = max(0.1, min(1.0, float(params['ema_alpha'])))
                if 'enable_smoothing' in params:
                    web_data['control_params']['enable_smoothing'] = bool(params['enable_smoothing'])
                
                # 设置更新标志
                web_data['params_updated'] = True
            
            print(f"🎛️ Web参数更新: {params}")
            return jsonify({'success': True, 'message': '参数更新成功'})
            
        except Exception as e:
            print(f"❌ 参数更新错误: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/update_avoidance_params', methods=['POST'])
    def update_avoidance_params():
        """更新三段避障参数"""
        try:
            params = request.get_json()
            
            # 验证参数
            if not params:
                return jsonify({'success': False, 'error': '无效的避障参数数据'})
            
            updated_params = {}
            
            # 更新三段避障参数
            with web_data_lock:
                if 'avoidance_config' not in web_data:
                    web_data['avoidance_config'] = {}
                
                # 第一段避障参数
                if 'stage1_left_speed' in params:
                    web_data['avoidance_config']['stage1_left_speed'] = int(params['stage1_left_speed'])
                    updated_params['stage1_left_speed'] = web_data['avoidance_config']['stage1_left_speed']
                if 'stage1_right_speed' in params:
                    web_data['avoidance_config']['stage1_right_speed'] = int(params['stage1_right_speed'])
                    updated_params['stage1_right_speed'] = web_data['avoidance_config']['stage1_right_speed']
                if 'stage1_duration' in params:
                    web_data['avoidance_config']['stage1_duration'] = float(params['stage1_duration'])
                    updated_params['stage1_duration'] = web_data['avoidance_config']['stage1_duration']
                
                # 第二段避障参数
                if 'stage2_left_speed' in params:
                    web_data['avoidance_config']['stage2_left_speed'] = int(params['stage2_left_speed'])
                    updated_params['stage2_left_speed'] = web_data['avoidance_config']['stage2_left_speed']
                if 'stage2_right_speed' in params:
                    web_data['avoidance_config']['stage2_right_speed'] = int(params['stage2_right_speed'])
                    updated_params['stage2_right_speed'] = web_data['avoidance_config']['stage2_right_speed']
                if 'stage2_duration' in params:
                    web_data['avoidance_config']['stage2_duration'] = float(params['stage2_duration'])
                    updated_params['stage2_duration'] = web_data['avoidance_config']['stage2_duration']
                
                # 第三段避障参数
                if 'stage3_left_speed' in params:
                    web_data['avoidance_config']['stage3_left_speed'] = int(params['stage3_left_speed'])
                    updated_params['stage3_left_speed'] = web_data['avoidance_config']['stage3_left_speed']
                if 'stage3_right_speed' in params:
                    web_data['avoidance_config']['stage3_right_speed'] = int(params['stage3_right_speed'])
                    updated_params['stage3_right_speed'] = web_data['avoidance_config']['stage3_right_speed']
                if 'stage3_duration' in params:
                    web_data['avoidance_config']['stage3_duration'] = float(params['stage3_duration'])
                    updated_params['stage3_duration'] = web_data['avoidance_config']['stage3_duration']
                
                # 设置避障参数更新标志
                web_data['avoidance_params_updated'] = True
            
            print(f"🎛️ 三段避障参数更新: {updated_params}")
            return jsonify({'success': True, 'updated_params': updated_params, 'message': '避障参数更新成功'})
            
        except Exception as e:
            print(f"❌ 避障参数更新错误: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/control_map')
    def get_control_map():
        with web_data_lock:
            if web_data['latest_control_map'] is not None:
                try:
                    # 确保图像格式正确
                    control_map = web_data['latest_control_map']
                    
                    # 调试信息
                    print(f"🖼️ Web请求控制地图: {control_map.shape}, 类型: {control_map.dtype}")
                    print(f"🖼️ 数据范围: {control_map.min()} ~ {control_map.max()}")
                    
                    # 如果是单通道图像，转换为3通道
                    if len(control_map.shape) == 2:
                        control_map = cv2.cvtColor(control_map, cv2.COLOR_GRAY2BGR)
                        print("🔄 单通道转换为3通道")
                    elif control_map.shape[2] == 1:
                        control_map = cv2.cvtColor(control_map, cv2.COLOR_GRAY2BGR)
                        print("🔄 单通道转换为3通道")
                    
                    # 确保数据类型为uint8
                    if control_map.dtype != np.uint8:
                        if control_map.max() <= 1.0:
                            control_map = (control_map * 255).astype(np.uint8)
                            print("🔄 归一化数据转换为uint8")
                        else:
                            control_map = control_map.astype(np.uint8)
                            print("🔄 数据类型转换为uint8")
                    
                    # 将OpenCV图像转换为PNG格式
                    success, buffer = cv2.imencode('.png', control_map)
                    if not success:
                        raise Exception("图像编码失败")
                        
                    print(f"✅ 控制地图编码成功，buffer长度: {len(buffer)}")
                    
                    # 返回二进制图像数据
                    return Response(
                        buffer.tobytes(),
                        mimetype='image/png',
                        headers={'Cache-Control': 'no-cache, no-store, must-revalidate'}
                    )
                    
                except Exception as e:
                    print(f"❌ 控制地图编码错误: {e}")
                    # 返回错误提示图片
                    empty_img = np.zeros((300, 400, 3), dtype=np.uint8)
                    cv2.putText(empty_img, f"Error: {str(e)[:20]}", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    _, buffer = cv2.imencode('.png', empty_img)
                    return Response(
                        buffer.tobytes(),
                        mimetype='image/png'
                    )
            else:
                print("⚠️ 没有可用的控制地图数据")
                # 返回空图片占位符
                empty_img = np.zeros((300, 400, 3), dtype=np.uint8)
                cv2.putText(empty_img, "No Control Map", (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.png', empty_img)
                return Response(
                    buffer.tobytes(),
                    mimetype='image/png'
                )
    
    # 串口控制相关API
    @app.route('/api/connect_serial', methods=['POST'])
    def connect_serial():
        global car_controller
        try:
            if not CAR_CONTROLLER_AVAILABLE:
                return jsonify({'success': False, 'error': '小车控制模块不可用'})
            
            # 🔧 修复：如果car_controller已存在且已连接，直接返回成功
            if car_controller is not None and car_controller.is_connected:
                with web_data_lock:
                    web_data['serial_connected'] = True
                    web_data['serial_enabled'] = True
                print("✅ 串口已经连接，无需重复连接")
                return jsonify({'success': True, 'message': '串口已连接'})
            
            # 初始化车辆控制器
            if car_controller is None:
                with web_data_lock:
                    port = web_data.get('serial_port', '/dev/ttyAMA0')
                car_controller = SimpleCarController(port=port)
            
            # 连接串口
            if car_controller.connect():
                with web_data_lock:
                    web_data['serial_connected'] = True
                    web_data['serial_enabled'] = True
                
                print("✅ 串口连接成功")
                return jsonify({'success': True, 'message': '串口连接成功'})
            else:
                print("❌ 串口连接失败")
                return jsonify({'success': False, 'error': '串口连接失败'})
                
        except Exception as e:
            print(f"❌ 串口连接错误: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/start_driving', methods=['POST'])
    def start_driving():
        global car_controller
        try:
            if car_controller is None or not car_controller.is_connected:
                return jsonify({'success': False, 'error': '串口未连接'})
            
            with web_data_lock:
                web_data['car_driving'] = True
                web_data['emergency_stop'] = False
                web_data['control_enabled'] = True
            
            print("🚀 开始行驶模式")
            return jsonify({'success': True, 'message': '开始行驶模式已启动'})
            
        except Exception as e:
            print(f"❌ 启动行驶错误: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/emergency_stop', methods=['POST'])
    def emergency_stop():
        global car_controller
        try:
            if car_controller is not None and car_controller.is_connected:
                # 立即发送停止指令
                car_controller.stop()
                print("🛑 紧急停车指令已发送")
            
            with web_data_lock:
                web_data['car_driving'] = False
                web_data['emergency_stop'] = True
                web_data['control_enabled'] = False
                web_data['last_control_command'] = {'left_speed': 0, 'right_speed': 0}
            
            print("🛑 紧急停车模式激活")
            return jsonify({'success': True, 'message': '紧急停车已执行'})
            
        except Exception as e:
            print(f"❌ 紧急停车错误: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/control_status')
    def get_control_status():
        global car_controller
        
        with web_data_lock:
            status = {
                'serial_connected': web_data.get('serial_connected', False),
                'car_driving': web_data.get('car_driving', False),
                'emergency_stop': web_data.get('emergency_stop', False),
                'control_enabled': web_data.get('control_enabled', False),
                'last_control_command': web_data.get('last_control_command', None)
            }
        
        # 检查实际串口状态
        if car_controller is not None:
            status['actual_serial_connected'] = car_controller.is_connected
            status['current_speeds'] = car_controller.get_current_speeds()
        
        return jsonify(status)
    
    return app

def start_web_server(port=5000):
    """启动Web服务器"""
    if not FLASK_AVAILABLE:
        print("❌ Flask未安装，无法启动Web服务器")
        return None
    
    app = create_web_app()
    if app is None:
        return None
    
    def run_server():
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
    
    server_thread = Thread(target=run_server, daemon=True)
    server_thread.start()
    
    print(f"🌐 Web界面已启动: http://localhost:{port}")
    print(f"🌐 外部访问: http://0.0.0.0:{port}")
    
    return server_thread 