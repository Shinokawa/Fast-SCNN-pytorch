import os
import cv2
import time
import numpy as np
import math # ⭐ 引入math库用于计算
from ais_bench.infer.interface import InferSession

# --------------------------------------------------------------------------
# --- ⚙️ 1. 配置参数 ---
# --------------------------------------------------------------------------
DEVICE_ID = 0
MODEL_PATH = "./weights/fast_scnn_tusimple_bs1.om"
INPUT_IMAGE_PATH = "raw_20250628_015947_359.jpg"
OUTPUT_IMAGE_PATH = "result_with_control_signal.jpg" # 修改了输出文件名
OUTPUT_BEV_IMAGE_PATH = "result_bev_visualization.jpg"

# 图像和模型尺寸
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 640
MODEL_WIDTH, MODEL_HEIGHT = 640, 640  # ⭐ 添加模型输入尺寸定义

SRC_POINTS = np.float32([[47, 590], [629, 572], [458, 421], [212, 434]])
DST_POINTS = np.float32([
    [IMAGE_WIDTH * 0.15, IMAGE_HEIGHT], [IMAGE_WIDTH * 0.85, IMAGE_HEIGHT],
    [IMAGE_WIDTH * 0.85, 0], [IMAGE_WIDTH * 0.15, 0]
])

YM_PER_PIX = 30 / IMAGE_HEIGHT
XM_PER_PIX = 3.7 / (DST_POINTS[1][0] - DST_POINTS[0][0])

# ⭐ --- 新增：控制相关参数 ---
class PIDController:
    """一个简单的PID控制器实现"""
    def __init__(self, Kp, Ki, Kd, set_point=0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.set_point = set_point
        self.prev_error, self.integral = 0, 0
        self.last_time = time.time()
        
    def update(self, current_value):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt == 0: return 0

        error = self.set_point - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        self.prev_error = error
        self.last_time = current_time
        return output

# 车辆物理参数 (用于前馈控制) - 这是一个估计值，需要根据实际车辆调整
WHEELBASE = 2.5 # 车辆轴距，单位：米

# --------------------------------------------------------------------------
# --- 🧠 2. 核心处理函数 ---
# --------------------------------------------------------------------------
# (preprocess, postprocess, find_lane_pixels_and_fit, calculate_curvature_and_offset 等函数与之前完全相同)
# ... (此处省略，以保持简洁) ...
def preprocess(img_bgr, model_w, model_h):
    blob = cv2.dnn.blobFromImage(img_bgr, scalefactor=1.0/255.0, size=(MODEL_WIDTH, MODEL_HEIGHT), swapRB=True, crop=False)
    blob -= np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    blob /= np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    return blob.astype(np.float16)
def postprocess(output_tensor, w, h):
    pred_mask = np.argmax(output_tensor, axis=1).squeeze()
    return (pred_mask * 255).astype(np.uint8)
def find_lane_pixels_and_fit(warped_mask):
    histogram = np.sum(warped_mask[warped_mask.shape[0]//2:, :], axis=0)
    midpoint = np.int32(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows=9; margin=100; minpix=50
    window_height = np.int32(warped_mask.shape[0]//nwindows)
    nonzero = warped_mask.nonzero(); nonzeroy = np.array(nonzero[0]); nonzerox = np.array(nonzero[1])
    leftx_current, rightx_current = leftx_base, rightx_base
    left_lane_inds, right_lane_inds = [], []
    for window in range(nwindows):
        win_y_low = warped_mask.shape[0] - (window+1)*window_height
        win_y_high = warped_mask.shape[0] - window*window_height
        win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
        win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds); right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix: leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix: rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
    left_lane_inds = np.concatenate(left_lane_inds); right_lane_inds = np.concatenate(right_lane_inds)
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
    left_fit, right_fit = None, None
    if len(leftx) > 0: left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) > 0: right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit, (leftx, lefty), (rightx, righty)
def calculate_curvature_and_offset(left_fit, right_fit, lane_pixels, img_shape):
    (leftx, lefty), (rightx, righty) = lane_pixels
    h, w = img_shape
    left_fit_cr, right_fit_cr = None, None
    if left_fit is not None and len(lefty)>0: left_fit_cr = np.polyfit(lefty*YM_PER_PIX, leftx*XM_PER_PIX, 2)
    if right_fit is not None and len(righty)>0: right_fit_cr = np.polyfit(righty*YM_PER_PIX, rightx*XM_PER_PIX, 2)
    y_eval = h-1; y_eval_m = y_eval*YM_PER_PIX
    left_curverad, right_curverad = 0, 0
    if left_fit_cr is not None: left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_m + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    if right_fit_cr is not None: right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_m + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    curvature = 0
    if left_curverad > 0 and right_curverad > 0: curvature = (left_curverad + right_curverad) / 2
    elif left_curverad > 0: curvature = left_curverad
    else: curvature = right_curverad
    offset_m = 0
    if left_fit is not None and right_fit is not None:
        lane_center_x_px = (left_fit[0]*y_eval**2+left_fit[1]*y_eval+left_fit[2] + right_fit[0]*y_eval**2+right_fit[1]*y_eval+right_fit[2])/2
        car_center_x_px = w / 2
        offset_m = (car_center_x_px - lane_center_x_px) * XM_PER_PIX
    return curvature, offset_m

# ⭐ --- 新增：控制信号生成函数 ---
def generate_control_signal(pid_controller, offset_m, curvature):
    """
    根据偏移量和曲率生成模拟的转向角信号。
    """
    # 1. 反馈部分 (PID): 纠正横向偏移
    # 注意：输入到PID的是负的偏移量。
    # 因为如果车在车道右侧（offset > 0），我们需要一个负的（向左）控制信号来纠正。
    pid_steering = pid_controller.update(-offset_m)

    # 2. 前馈部分: 适应道路曲率
    # 使用阿克曼转向几何的简化公式: steer = atan(L/R)
    # L是轴距, R是转弯半径 (即曲率的倒数)
    # 当曲率很大（直线）时，R趋于无穷大，前馈角趋于0
    feed_forward_steering = 0
    if curvature > 0: # 避免除以零
        # 为了防止曲率过大导致半径过小，可以设置一个最小半径阈值
        radius = max(curvature, 1.0) # 假设最小转弯半径为1米
        feed_forward_steering = math.atan(WHEELBASE / radius)
    
    # 3. 组合并转换为角度
    # 将弧度转换为角度
    feed_forward_steering_deg = math.degrees(feed_forward_steering)

    # 最终转向角是两部分之和。注意符号！
    # 如果偏移和曲率方向一致（例如，右转弯且车偏在弯道内侧），两部分会叠加
    # 这里我们简单相加，但实际PID的输出符号已经包含了方向
    total_steering_angle = pid_steering + feed_forward_steering_deg

    # 4. 限制转向角范围 (例如，限制在-25到+25度之间)
    max_steer_angle = 25.0
    total_steering_angle = np.clip(total_steering_angle, -max_steer_angle, max_steer_angle)

    return total_steering_angle


# ⭐ --- 修改：可视化函数，增加显示控制信号 ---
def draw_visualization(original_image, left_fit, right_fit, Minv, curvature, offset, steering_angle):
    """将所有信息（车道，数据，控制信号）都绘制到图像上"""
    vis_image = draw_lane_on_image(original_image, left_fit, right_fit, Minv, curvature, offset)

    # --- 绘制控制信号文本 ---
    cv2.putText(vis_image, f'Steering Angle: {steering_angle:.2f} deg', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # --- 绘制一个简单的方向盘/转向指示器 ---
    h, w = vis_image.shape[:2]
    center_x, center_y = w // 2, h - 50 # 指示器中心位置
    
    # 画一个背景圆
    cv2.circle(vis_image, (center_x, center_y), 30, (50, 50, 50), -1)
    cv2.circle(vis_image, (center_x, center_y), 30, (255, 255, 255), 2)
    
    # 计算指针的终点
    angle_rad = -math.radians(steering_angle) # cv2中角度是逆时针，但我们习惯右转为正
    pointer_length = 28
    end_x = int(center_x + pointer_length * math.sin(angle_rad))
    end_y = int(center_y - pointer_length * math.cos(angle_rad))
    
    # 画出指针
    cv2.line(vis_image, (center_x, center_y), (end_x, end_y), (0, 255, 255), 2)
    
    return vis_image
# (create_bev_visualization 函数与之前完全相同)
def create_bev_visualization(warped_mask, left_fit, right_fit, lane_pixels):
    (leftx, lefty), (rightx, righty) = lane_pixels; h, w = warped_mask.shape[:2]
    bev_viz = np.dstack((warped_mask, warped_mask, warped_mask))
    bev_viz[lefty, leftx] = [0, 0, 255]; bev_viz[righty, rightx] = [255, 0, 0]
    ploty = np.linspace(0, h-1, h)
    if left_fit is not None:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        cv2.polylines(bev_viz, [np.array([np.transpose(np.vstack([left_fitx, ploty]))]).astype(np.int32)], isClosed=False, color=(0,255,255), thickness=4)
    if right_fit is not None:
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        cv2.polylines(bev_viz, [np.array([np.transpose(np.vstack([right_fitx, ploty]))]).astype(np.int32)], isClosed=False, color=(0,255,255), thickness=4)
    return bev_viz
def draw_lane_on_image(original_image, left_fit, right_fit, Minv, curvature, offset):
    h, w = original_image.shape[:2]; vis_image = original_image.copy()
    color_warp = np.zeros((h, w, 3), dtype='uint8')
    if left_fit is not None and right_fit is not None:
        ploty = np.linspace(0, h-1, h); left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]; right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))]); pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right)); cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))
        newwarp = cv2.warpPerspective(color_warp, Minv, (w,h)); vis_image = cv2.addWeighted(vis_image, 1, newwarp, 0.3, 0)
    cv2.putText(vis_image, f'Curvature: {curvature:.0f} m', (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(vis_image, f'Offset: {offset:.2f} m', (30,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return vis_image

# --------------------------------------------------------------------------
# --- 🚀 3. 主执行流程 ---
# --------------------------------------------------------------------------
if __name__ == '__main__':
    print("--- 开始处理静态图片车道线检测并生成控制信号 ---")

    # --- 1. 初始化和加载 ---
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"❌ 错误: 输入图片 '{INPUT_IMAGE_PATH}' 不存在！")
        exit()

    model = InferSession(DEVICE_ID, MODEL_PATH)
    original_image = cv2.imread(INPUT_IMAGE_PATH)
    original_image = cv2.resize(original_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    
    # ⭐ 实例化PID控制器 (重要：这些Kp, Ki, Kd值是示例，需要大量调试！)
    # 先从一个较小的Kp开始，Ki和Kd设为0
    pid_steer_controller = PIDController(Kp=0.8, Ki=0.01, Kd=0.1)
    print(f"✅ PID控制器已初始化 (Kp={pid_steer_controller.Kp}, Ki={pid_steer_controller.Ki}, Kd={pid_steer_controller.Kd})")

    # --- 2. 推理、分割、变换、拟合 ---
    input_data = preprocess(original_image, MODEL_WIDTH, MODEL_HEIGHT)
    outputs = model.infer([input_data])
    lane_mask = postprocess(outputs[0], IMAGE_WIDTH, IMAGE_HEIGHT)
    
    M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
    Minv = cv2.getPerspectiveTransform(DST_POINTS, SRC_POINTS)
    warped_mask = cv2.warpPerspective(lane_mask, M, (IMAGE_WIDTH, IMAGE_HEIGHT), flags=cv2.INTER_NEAREST)
    
    left_fit, right_fit, left_pixels, right_pixels = find_lane_pixels_and_fit(warped_mask)
    curvature, offset = calculate_curvature_and_offset(left_fit, right_fit, (left_pixels, right_pixels), (IMAGE_HEIGHT, IMAGE_WIDTH))

    # ⭐ --- 3. 生成控制信号 ---
    steering_angle = generate_control_signal(pid_steer_controller, offset, curvature)
    print("\n--- 🎮 控制信号 ---")
    print(f"  - 模拟转向角: {steering_angle:.2f} 度")
    print("-" * 20)

    # --- 4. 可视化与保存 ---
    final_result_image = draw_visualization(original_image, left_fit, right_fit, Minv, curvature, offset, steering_angle)
    bev_visualization_image = create_bev_visualization(warped_mask, left_fit, right_fit, (left_pixels, right_pixels))
    
    cv2.imwrite(OUTPUT_IMAGE_PATH, final_result_image)
    print(f"✔️ 最终结果图已保存至 '{OUTPUT_IMAGE_PATH}'")
    cv2.imwrite(OUTPUT_BEV_IMAGE_PATH, bev_visualization_image)
    print(f"✔️ 鸟瞰图可视化已保存至 '{OUTPUT_BEV_IMAGE_PATH}'")

    print("\n--- 执行完毕 ---")