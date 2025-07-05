import os
import sys
import threading
from flask import Flask, render_template_string, request, redirect, url_for, flash

# 确保可以导入car_controller_simple.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kuruma.car_controller_simple import SimpleCarController

app = Flask(__name__)
app.secret_key = 'car_controller_secret_key'

# 控制器单例
controller = SimpleCarController()
controller_lock = threading.Lock()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>小车轮子速度控制</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            background: #f6f8fa;
            font-family: 'Segoe UI', 'PingFang SC', Arial, sans-serif;
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 420px;
            margin: 40px auto;
            background: #fff;
            border-radius: 18px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.10);
            padding: 32px 28px 24px 28px;
        }
        h2 {
            text-align: center;
            color: #222;
            margin-bottom: 28px;
            font-weight: 600;
            letter-spacing: 1px;
        }
        label {
            display: block;
            margin-top: 18px;
            color: #444;
            font-size: 16px;
        }
        input[type=number] {
            width: 120px;
            padding: 8px 10px;
            border: 1.5px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            margin-top: 8px;
            background: #f9f9f9;
            transition: border 0.2s;
        }
        input[type=number]:focus {
            border: 1.5px solid #1976d2;
            outline: none;
            background: #fff;
        }
        .btn-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 28px;
        }
        button {
            padding: 10px 28px;
            border: none;
            border-radius: 8px;
            background: #1976d2;
            color: #fff;
            font-size: 16px;
            font-weight: 500;
            box-shadow: 0 2px 8px rgba(25,118,210,0.08);
            cursor: pointer;
            transition: background 0.2s, box-shadow 0.2s;
        }
        button:hover {
            background: #1565c0;
            box-shadow: 0 4px 16px rgba(25,118,210,0.13);
        }
        .stop-btn {
            background: #e53935;
            margin-left: 0;
            box-shadow: 0 2px 8px rgba(229,57,53,0.08);
        }
        .stop-btn:hover {
            background: #b71c1c;
            box-shadow: 0 4px 16px rgba(229,57,53,0.13);
        }
        .wasd-hint {
            margin-top: 32px;
            color: #555;
            font-size: 15px;
            background: #f1f8fe;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 1px 4px rgba(25,118,210,0.04);
        }
        .status {
            margin-top: 28px;
            color: #1976d2;
            font-size: 16px;
            background: #e3f2fd;
            border-radius: 8px;
            padding: 12px 16px;
            box-shadow: 0 1px 4px rgba(25,118,210,0.04);
        }
        .error {
            color: #e53935;
            background: #ffebee;
            border-radius: 8px;
            padding: 10px 14px;
            margin-top: 18px;
            font-size: 15px;
        }
        .status, .error {
            margin-bottom: 0;
        }
        @media (max-width: 600px) {
            .container { padding: 18px 4vw 12px 4vw; }
            h2 { font-size: 20px; }
            input[type=number] { width: 90px; font-size: 15px; }
            button { font-size: 15px; padding: 8px 12px; }
        }
    </style>
    <script>
    let speed = 900;
    const minSpeed = 900;
    const maxSpeed = 1000;
    const speedStep = 10;
    document.addEventListener('DOMContentLoaded', function() {
        document.body.addEventListener('keydown', function(e) {
            let cmd = null;
            if (e.key === 'w' || e.key === 'W') cmd = 'w';
            if (e.key === 's' || e.key === 'S') cmd = 's';
            if (e.key === 'a' || e.key === 'A') cmd = 'a';
            if (e.key === 'd' || e.key === 'D') cmd = 'd';
            if (e.key === 'h' || e.key === 'H') cmd = 'h';
            if (e.key === 'l' || e.key === 'L') cmd = 'l';
            if (cmd) {
                fetch('/wasd', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ cmd: cmd })
                }).then(() => window.location.reload());
            }
        });
    });
    </script>
</head>
<body>
<div class="container">
    <h2>小车轮子速度控制</h2>
    <form method="post">
        <label>左轮速度 (-1000 ~ 1000):
            <input type="number" name="left_speed" min="-1000" max="1000" required value="{{ left }}">
        </label>
        <label>右轮速度 (-1000 ~ 1000):
            <input type="number" name="right_speed" min="-1000" max="1000" required value="{{ right }}">
        </label>
        <div class="btn-row">
            <button type="submit">发送速度</button>
            <button type="submit" name="stop" value="1" class="stop-btn">紧急停止</button>
        </div>
    </form>
    <div class="wasd-hint">
        <b>键盘控制：</b><br>
        W-前进，S-后退，A-左转，D-右转，H-加速，L-减速<br>
        当前WASD速度区间：900-1000，步进10，初始900
    </div>
    {% with messages = get_flashed_messages(with_categories=true) %}
      {% if messages %}
        {% for category, message in messages %}
          <div class="{{ category }}">{{ message }}</div>
        {% endfor %}
      {% endif %}
    {% endwith %}
    <div class="status">
        当前速度：左轮 {{ left }}，右轮 {{ right }}<br>
        串口连接状态：{{ '已连接' if connected else '未连接' }}
    </div>
</div>
</body>
</html>
'''

# WASD控制状态
wasd_state = {
    'speed': 0,
    'direction': 'stop'  # 可选: stop, forward, backward, left, right
}

@app.route('/', methods=['GET', 'POST'])
def index():
    left = controller.left_wheel_speed
    right = controller.right_wheel_speed
    connected = controller.is_connected
    if request.method == 'POST':
        try:
            if 'stop' in request.form:
                # 紧急停止
                with controller_lock:
                    if not controller.is_connected:
                        controller.connect()
                    success = controller.set_wheel_speeds(0, 0)
                if success:
                    flash('已发送紧急停止命令', 'status')
                else:
                    flash('紧急停止失败', 'error')
                left = 0
                right = 0
                connected = controller.is_connected
            else:
                left_speed = int(request.form['left_speed'])
                right_speed = int(request.form['right_speed'])
                with controller_lock:
                    if not controller.is_connected:
                        controller.connect()
                    success = controller.set_wheel_speeds(left_speed, right_speed)
                if success:
                    flash('速度设置成功', 'status')
                else:
                    flash('速度设置失败', 'error')
                left = left_speed
                right = right_speed
                connected = controller.is_connected
        except Exception as e:
            flash(f'发生错误: {e}', 'error')
    return render_template_string(HTML_TEMPLATE, left=left, right=right, connected=connected)

@app.route('/wasd', methods=['POST'])
def wasd_control():
    import json
    data = request.get_json()
    cmd = data.get('cmd')
    global wasd_state
    speed = wasd_state['speed']
    direction = wasd_state['direction']
    min_speed = 0
    max_speed = 1000
    step = 10
    if cmd == 'h':
        if direction == 'forward':
            speed = min(max_speed, speed + step)
        elif direction == 'backward':
            speed = max(-max_speed, speed - step)
    elif cmd == 'l':
        if direction == 'forward':
            speed = max(min_speed, speed - step)
        elif direction == 'backward':
            speed = min(-min_speed, speed + step)
    elif cmd == 'w':
        direction = 'forward'
        speed = min_speed
    elif cmd == 's':
        direction = 'backward'
        speed = -min_speed
    elif cmd == 'a':
        direction = 'left'
    elif cmd == 'd':
        direction = 'right'
    # 控制逻辑
    left, right = 0, 0
    if direction == 'forward':
        left = right = speed
    elif direction == 'backward':
        left = right = speed
    elif direction == 'left':
        left = speed - step * 2 if speed > 0 else speed + step * 2
        right = speed
    elif direction == 'right':
        left = speed
        right = speed - step * 2 if speed > 0 else speed + step * 2
    elif direction == 'stop':
        left = right = 0
    with controller_lock:
        if not controller.is_connected:
            controller.connect()
        controller.set_wheel_speeds(left, right)
    wasd_state['speed'] = speed
    wasd_state['direction'] = direction
    return ('', 204)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 