import cv2
import numpy as np
import os

# --- 配置 ---
# 把你的实拍图放在和这个脚本相同的目录下，并命名为 calibration_image.jpg
CALIBRATION_IMAGE_PATH = "calibration_image.jpg" 
# 确保这个尺寸和你的摄像头/主程序中使用的尺寸完全一致
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 640

# 全局变量
points_list = []
window_name = "Calibration - Click 4 Points"

def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数，用于记录点击的坐标点"""
    global points_list, image_display

    # 当鼠标左键按下时
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_list) < 4:
            # 将点击的坐标 (x, y) 添加到列表中
            points_list.append([x, y])
            print(f"已选择第 {len(points_list)} 个点: ({x}, {y})")

            # 在图像上画一个圆圈来标记点击的位置
            cv2.circle(image_display, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, image_display)
            
            # 如果已经选择了4个点，就处理并打印结果
            if len(points_list) == 4:
                print("\n--- 4个点已选择完毕！ ---")
                print("点击顺序: 左下 -> 右下 -> 右上 -> 左上")
                
                # 绘制梯形，方便确认
                pts = np.array(points_list, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(image_display, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
                cv2.imshow(window_name, image_display)
                
                # 以可以直接复制的格式打印 NumPy 数组
                print("\n请将以下代码复制到你的主程序中，替换 'src' 变量：\n")
                np_array_str = f"src = np.float32({points_list})"
                print(np_array_str)
                print("\n现在可以按 'q' 键退出标定程序。")

        else:
            print("已经选择了4个点。请按 'c' 清除后重选，或按 'q' 退出。")


def main():
    global points_list, image_display
    
    # 检查标定图像是否存在
    if not os.path.exists(CALIBRATION_IMAGE_PATH):
        print(f"错误: 标定图像 '{CALIBRATION_IMAGE_PATH}' 未找到！")
        print("请将你的实拍图放在此目录下并重命名。")
        return

    # 加载并调整图像尺寸
    image_original = cv2.imread(CALIBRATION_IMAGE_PATH)
    image_resized = cv2.resize(image_original, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    image_display = image_resized.copy() # 创建一个副本用于绘制

    # 创建窗口并设置鼠标回调
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("=== 欢迎使用透视变换标定工具 ===")
    print(f"图像尺寸: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")
    print("\n请按照以下顺序点击图像中的4个点，构成车道线所在的梯形区域：")
    print("  1. 左下角 (Left Bottom)")
    print("  2. 右下角 (Right Bottom)")
    print("  3. 右上角 (Right Top)")
    print("  4. 左上角 (Left Top)")
    print("\n操作指南:")
    print(" - 按 'c' 键可以清除所有点，重新开始选择。")
    print(" - 按 'q' 键退出程序。")

    while True:
        cv2.imshow(window_name, image_display)
        key = cv2.waitKey(1) & 0xFF

        # 按 'q' 退出
        if key == ord('q'):
            break
        # 按 'c' 清除
        elif key == ord('c'):
            print("\n已清除所有点，请重新选择。")
            points_list = []
            image_display = image_resized.copy() # 重置显示的图像
            
    cv2.destroyAllWindows()
    print("\n标定程序结束。")

if __name__ == "__main__":
    main()