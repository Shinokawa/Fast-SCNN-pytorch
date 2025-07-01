import os
import sys
import threading
from flask import Flask, render_template_string, request, redirect, url_for, flash

# 确保可以导入car_controller_simple.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from car_controller_simple import SimpleCarController

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
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 400px; margin: auto; }
        label { display: block; margin-top: 15px; }
        input[type=number] { width: 100px; }
        .status { margin-top: 20px; color: green; }
        .error { color: red; }
        button { margin-top: 20px; padding: 8px 20px; }
    </style>
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
        <button type="submit">发送速度</button>
    </form>
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

@app.route('/', methods=['GET', 'POST'])
def index():
    left = controller.left_wheel_speed
    right = controller.right_wheel_speed
    connected = controller.is_connected
    if request.method == 'POST':
        try:
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 