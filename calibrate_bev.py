import cv2
import numpy as np

# ==============================================================================
# 1. 复制并粘贴我们之前优化的 PerspectiveTransformer 类
#    (这里我只包含必要的部分，以保持脚本简洁)
# ==============================================================================
class PerspectiveTransformer:
    def __init__(self, paper_dist_cm=20.0):
        # 固定参数
        self.src_points = np.float32([[260, 87], [378, 87], [410, 217], [231, 221]])
        A4_WIDTH_CM, A4_HEIGHT_CM = 21.0, 29.7
        pixels_per_cm = 10.0  # 使用稍低的分辨率以加快实时刷新
        bev_width_cm, bev_height_cm = 150.0, 150.0

        # 计算BEV尺寸
        self.bev_pixel_width = int(bev_width_cm * pixels_per_cm)
        self.bev_pixel_height = int(bev_height_cm * pixels_per_cm)
        
        # 计算A4纸在BEV中的像素尺寸
        paper_bev_width = A4_WIDTH_CM * pixels_per_cm
        paper_bev_height = A4_HEIGHT_CM * pixels_per_cm
        margin_x = (self.bev_pixel_width - paper_bev_width) / 2
        
        # --- 关键部分：这里的 paper_dist_cm 是可变的 ---
        paper_bottom_y = self.bev_pixel_height - (paper_dist_cm * pixels_per_cm)
        paper_top_y = paper_bottom_y - paper_bev_height
        
        self.dst_points = np.float32([
            [margin_x, paper_top_y],
            [margin_x + paper_bev_width, paper_top_y],
            [margin_x + paper_bev_width, paper_bottom_y],
            [margin_x, paper_bottom_y]
        ])
        
        self.transform_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def transform_image(self, image):
        return cv2.warpPerspective(
            image, self.transform_matrix,
            (self.bev_pixel_width, self.bev_pixel_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

# ==============================================================================
# 2. 交互式校准的主程序
# ==============================================================================

# --- 全局变量 ---
# !!! 修改为你自己的图片路径，这张图片最好有清晰的平行线 !!!
IMAGE_PATH = "data/custom/images/raw_20250702_024235_522.jpg" 
WINDOW_NAME = "Interactive BEV Calibration"
TRACKBAR_NAME = "Paper Distance (cm)"

# 加载图像
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"错误: 无法加载图片 '{IMAGE_PATH}'")
    exit()

# 统一调整图像尺寸到标定时的尺寸 (640x360)
image = cv2.resize(image, (640, 360))


def on_trackbar_change(val):
    """滑动条的回调函数"""
    # 1. 从滑动条获取当前的距离值 (OpenCV滑动条是整数，我们除以10来获得小数)
    distance_cm = val / 10.0
    
    # 2. 使用当前距离值创建一个新的变换器实例
    transformer = PerspectiveTransformer(paper_dist_cm=distance_cm)
    
    # 3. 生成新的鸟瞰图
    bev_image = transformer.transform_image(image)
    
    # 4. 在鸟瞰图上画几条水平参考线，帮助判断是否平行
    h, w = bev_image.shape[:2]
    for i in range(1, 4):
        y = int(h * i / 4)
        cv2.line(bev_image, (0, y), (w, y), (0, 255, 255), 1) # 黄色参考线
        
    # 5. 显示更新后的鸟瞰图
    cv2.imshow(WINDOW_NAME, bev_image)
    
    # 打印当前值
    print(f"当前尝试的距离: {distance_cm:.1f} cm")


# --- 创建窗口和滑动条 ---
cv2.namedWindow(WINDOW_NAME)

# 创建滑动条。范围是 0 到 1000，对应 0.0cm 到 100.0cm
# 初始值设为 200，对应 20.0cm
cv2.createTrackbar(TRACKBAR_NAME, WINDOW_NAME, 200, 1000, on_trackbar_change)

# --- 首次调用以显示初始图像 ---
print("=== 交互式鸟瞰图校准工具 ===")
print("操作说明:")
print("1. 拖动滑动条来调整A4纸的虚拟距离。")
print("2. 观察右侧窗口中的平行线（如车道线）。")
print("3. 目标：让这些线在鸟瞰图中也保持平行。")
print("4. 找到最佳值后，按 'q' 键退出，并将该值记录下来。")
print("==============================")
on_trackbar_change(200) # 初始调用

# 等待用户操作，按'q'退出
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()