# 小车控制模块

这是一个高级小车控制模块，提供转向强度和前进速度的接口，通过串口与STM32通信，实现精确的差速控制。

## 功能特性

- 🚗 **精确控制**: 支持0.0-1.0的速度控制和-1.0到1.0的转向控制
- ⚡ **差速转向**: 通过左右轮速度差实现平滑转向
- 🔒 **安全保护**: 内置紧急停止和超时保护机制
- 📊 **状态监控**: 实时查询小车运动状态
- 🛡️ **错误处理**: 完善的错误检测和处理机制
- 🔧 **易于集成**: 简洁的API接口，便于其他程序调用

## 硬件要求

- STM32F103系列微控制器
- 4个直流电机（左前、左后、右前、右后）
- 电机驱动模块（支持PWM控制）
- 串口通信模块

## 通信协议

### 协议格式
```
[头部][命令][长度][数据][校验和][尾部]
```

- **头部**: 0xAA
- **尾部**: 0x55
- **校验和**: 除头部外所有字节的累加和

### 命令类型

| 命令 | 值 | 描述 | 数据格式 |
|------|----|----|----------|
| CMD_SET_SPEED | 0x01 | 设置速度 | uint16_t (PWM值) |
| CMD_SET_STEERING | 0x02 | 设置转向 | float left_ratio, float right_ratio |
| CMD_SET_MOTION | 0x03 | 设置运动 | uint16_t base_speed, uint16_t left_pwm, uint16_t right_pwm |
| CMD_EMERGENCY_STOP | 0x04 | 紧急停止 | 无 |
| CMD_GET_STATUS | 0x05 | 获取状态 | 无 |
| CMD_ACK | 0x06 | 确认响应 | uint8_t cmd_type |

## 安装和使用

### 1. 安装依赖

```bash
pip install periphery
```

### 2. 基本使用

```python
from car_controller import CarController

# 创建控制器实例
controller = CarController(port="/dev/ttyAMA0", baudrate=115200)

# 连接小车
if controller.connect():
    # 设置前进速度
    controller.set_speed(0.5)
    
    # 设置转向
    controller.set_steering(-0.3)  # 左转
    
    # 同时设置速度和转向
    controller.set_motion(0.6, 0.2)  # 速度0.6，右转0.2
    
    # 停止
    controller.stop()
    
    # 断开连接
    controller.disconnect()
```

### 3. 使用上下文管理器

```python
with CarController() as controller:
    # 自动连接和断开
    controller.set_motion(0.4, 0.0)  # 前进
    time.sleep(2)
    controller.set_motion(0.3, -0.5)  # 左转
    time.sleep(2)
    # 自动停止和断开
```

## API 参考

### CarController 类

#### 初始化参数
- `port`: 串口设备名 (默认: "/dev/ttyAMA0")
- `baudrate`: 波特率 (默认: 115200)
- `timeout`: 超时时间 (默认: 1.0秒)

#### 主要方法

##### `connect() -> bool`
连接串口设备
- **返回**: 连接是否成功

##### `disconnect()`
断开串口连接

##### `set_speed(speed: float) -> bool`
设置前进速度
- **参数**: speed (0.0-1.0)
- **返回**: 设置是否成功

##### `set_steering(steering: float) -> bool`
设置转向强度
- **参数**: steering (-1.0到1.0，负值左转，正值右转)
- **返回**: 设置是否成功

##### `set_motion(speed: float, steering: float) -> bool`
同时设置速度和转向
- **参数**: 
  - speed (0.0-1.0)
  - steering (-1.0到1.0)
- **返回**: 设置是否成功

##### `stop() -> bool`
紧急停止小车
- **返回**: 停止是否成功

##### `get_status() -> Optional[dict]`
获取小车状态
- **返回**: 状态字典或None

##### `get_current_state() -> dict`
获取当前控制状态
- **返回**: 当前状态信息

## 差速控制原理

小车通过左右轮速度差实现转向：

```
左轮速度 = 基础速度 × 左轮比例
右轮速度 = 基础速度 × 右轮比例
```

转向计算：
```
steering = (左轮速度 - 右轮速度) / (左轮速度 + 右轮速度)
```

- steering = 0: 直行
- steering > 0: 右转
- steering < 0: 左转

## STM32固件

### 编译要求
- STM32F10x标准外设库
- 支持浮点运算

### 主要文件
- `car_controller_stm32.c`: 主控制固件
- `motor.c/h`: 电机控制模块
- `usart.c/h`: 串口通信模块

### 编译和烧录
1. 将固件代码添加到STM32项目中
2. 配置串口为115200波特率
3. 编译并烧录到STM32

## 测试

### 运行测试
```bash
python test_car_controller.py
```

### 运行演示
```bash
python car_control_example.py
```

## 集成示例

### 与车道检测系统集成

```python
from car_controller import CarController
import cv2
import numpy as np

class LaneFollowingCar:
    def __init__(self):
        self.controller = CarController()
        self.controller.connect()
    
    def process_lane_detection(self, image):
        # 车道检测算法
        # ... 检测车道线 ...
        
        # 计算转向角度
        steering_angle = self.calculate_steering(lane_center)
        
        # 控制小车
        self.controller.set_motion(0.4, steering_angle)
    
    def calculate_steering(self, lane_center):
        # 将车道中心位置转换为转向角度
        # 假设图像中心为0，左右边界为±1
        return (lane_center - 0.5) * 2
```

### 与避障系统集成

```python
class ObstacleAvoidanceCar:
    def __init__(self):
        self.controller = CarController()
        self.controller.connect()
    
    def avoid_obstacle(self, distance, angle):
        if distance < 0.5:  # 距离小于50cm
            if angle > 0:  # 障碍物在右侧
                self.controller.set_motion(0.2, -0.8)  # 左转避障
            else:  # 障碍物在左侧
                self.controller.set_motion(0.2, 0.8)   # 右转避障
        else:
            self.controller.set_motion(0.4, 0.0)  # 正常前进
```

## 故障排除

### 常见问题

1. **连接失败**
   - 检查串口设备名是否正确
   - 确认STM32已正确烧录固件
   - 检查波特率设置

2. **命令无响应**
   - 检查串口连接
   - 确认固件正在运行
   - 检查通信协议是否匹配

3. **运动异常**
   - 检查电机连接
   - 确认PWM配置正确
   - 检查电源供应

### 调试模式

启用调试日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 安全注意事项

1. **紧急停止**: 始终确保紧急停止功能可用
2. **速度限制**: 在测试时使用较低速度
3. **环境安全**: 在安全的环境中测试
4. **电源管理**: 确保电源稳定供应

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 更新日志

### v1.0.0
- 初始版本发布
- 支持基本的速度和转向控制
- 实现差速控制算法
- 添加状态监控功能
- 完善错误处理机制 