import serial
import time
import sys
# 推荐使用 getch 来获取实时按键，无需回车
# 如果你安装的是 py-getch, 就 from py_getch import getch
# 如果是 getch-msvc, 就 from getch import getch (可能需要调整)
# 这里我们假设一个通用的 getch 实现
try:
    from getch import getch as _getch
    def getch():
        c = _getch()
        return c.decode() if isinstance(c, bytes) else c
except ImportError:
    # 如果getch库不存在，提供一个简单的基于input的备用方案
    print("警告: 'getch' 库未找到。将使用基于 input() 的慢速模式。")
    print("请在每次输入后按 Enter 键。")
    def getch():
        return input()

# --- 配置区 ---
# !!! 重要：把这里替换成你上一步找到的实际设备路径 !!!
SERIAL_PORT = '/dev/cu.HC-05' 
BAUD_RATE = 9600  # 必须和你的STM32队友设置的波特率完全一致

# --- 指令定义 ---
CMD_FORWARD = 'w'
CMD_BACKWARD = 's'
CMD_LEFT = 'a'
CMD_RIGHT = 'd'
CMD_STOP = 'p'
CMD_SPEED_UP = 'h'   # 提速
CMD_SPEED_DOWN = 'l' # 降速

def main():
    ser = None  # 初始化ser变量
    print("--- STM32 蓝牙遥控脚本 ---")
    print(f"尝试连接到: {SERIAL_PORT}")

    try:
        # 连接到串口
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("✅ 连接成功！小车已准备就绪。")
        print("\n--- 操作指南 ---")
        print("   W: 前进")
        print("   S: 后退")
        print("   A: 左转")
        print("   D: 右转")
        print(" Space: 停止")
        print(" E: 提速")
        print(" Z: 降速")
        print(" 0-9: 设置速度")
        print("   Q: 退出程序")
        print("------------------")

        while True:
            # 获取键盘单个字符输入
            char = getch()

            # 将输入转换为小写，方便比较
            key = char.lower()

            if key == 'w':
                print("发送指令: 前进 (w)")
                ser.write(CMD_FORWARD.encode('utf-8'))
                data = ser.readline()
                print(data)
            elif key == 's':
                print("发送指令: 后退 (s)")
                ser.write(CMD_BACKWARD.encode('utf-8'))
            elif key == 'a':
                print("发送指令: 左转 (a)")
                ser.write(CMD_LEFT.encode('utf-8'))
            elif key == 'd':
                print("发送指令: 右转 (d)")
                ser.write(CMD_RIGHT.encode('utf-8'))
            elif char == ' ':  # 注意：空格键不转换大小写
                print("发送指令: 停止 (p)")
                ser.write(CMD_STOP.encode('utf-8'))
            elif key == 'e':
                print("发送指令: 提速 (h)")
                ser.write(CMD_SPEED_UP.encode('utf-8'))
            elif key == 'z':
                print("发送指令: 降速 (l)")
                ser.write(CMD_SPEED_DOWN.encode('utf-8'))
            elif '0' <= char <= '9':
                print(f"发送指令: 设置速度等级 ({char})")
                ser.write(char.encode('utf-8'))
            elif key == 'q':
                print("程序退出。")
                break
            else:
                # 忽略其他按键
                pass

            # 新增：读取并打印小车返回的信息
            if ser.in_waiting:
                try:
                    response = ser.readline().decode(errors='ignore').strip()
                    if response:
                        print(f"小车返回: {response}")
                except Exception as e:
                    print(f"读取串口返回信息出错: {e}")

    except serial.SerialException as e:
        print(f"❌ 错误: 无法连接到 {SERIAL_PORT}。")
        print("请检查：")
        print("1. 小车是否已开机？")
        print("2. 蓝牙是否已与Mac成功配对？")
        print("3. 脚本中的 SERIAL_PORT 名称是否正确？")
        print(f"系统提示: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("串口连接已关闭。")

if __name__ == '__main__':
    main()