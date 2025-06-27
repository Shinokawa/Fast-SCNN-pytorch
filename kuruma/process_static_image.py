import os
import cv2
import time
import numpy as np
from ais_bench.infer.interface import InferSession

# --------------------------------------------------------------------------
# --- ⚙️ 1. 配置参数 ---
# --------------------------------------------------------------------------
# --- 模型和设备配置 ---
DEVICE_ID = 0
MODEL_PATH = "./weights/fast_scnn_tusimple_bs1.om" # 确保这是你的模型路径
INPUT_IMAGE_PATH = "raw_20250628_015947_359.jpg" # 你的测试图片
OUTPUT_IMAGE_PATH = "result_image.jpg" # 处理结果保存路径
OUTPUT_BEV_IMAGE_PATH = "result_bev_visualization.jpg" # 鸟瞰图可视化结果的保存路径

# --- 图像和模型尺寸 ---
MODEL_WIDTH = 640
MODEL_HEIGHT = 640
IMAGE_WIDTH = 640  # 假设测试图和模型输入尺寸一致
IMAGE_HEIGHT = 640

# --- ⭐ 透视变换标定参数 (使用你校正后的坐标) ---
SRC_POINTS = np.float32([[47, 590], [629, 572], [458, 421], [212, 434]])
DST_POINTS = np.float32([
    [IMAGE_WIDTH * 0.15, IMAGE_HEIGHT], # 左下
    [IMAGE_WIDTH * 0.85, IMAGE_HEIGHT], # 右下
    [IMAGE_WIDTH * 0.85, 0],            # 右上
    [IMAGE_WIDTH * 0.15, 0]             # 左上
])

# --- 真实世界转换系数 (重要！需要根据你的摄像头和场景标定) ---
YM_PER_PIX = 30 / IMAGE_HEIGHT  # 假设鸟瞰图垂直方向代表30米
XM_PER_PIX = 3.7 / (DST_POINTS[1][0] - DST_POINTS[0][0]) # 假设车道宽3.7米


# --------------------------------------------------------------------------
# --- 🧠 2. 核心处理函数 ---
# --------------------------------------------------------------------------

# --- 预处理 ---
BLOB_MEAN_FOR_SUBTRACTION = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
BLOB_STD_FOR_DIVISION = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

def preprocess(img_bgr):
    """使用cv2.dnn.blobFromImage进行高效预处理"""
    blob = cv2.dnn.blobFromImage(img_bgr, scalefactor=1.0/255.0, size=(MODEL_WIDTH, MODEL_HEIGHT), swapRB=True, crop=False)
    blob -= BLOB_MEAN_FOR_SUBTRACTION
    blob /= BLOB_STD_FOR_DIVISION
    return blob.astype(np.float16)

# --- 后处理 ---
def postprocess(output_tensor, original_width, original_height):
    """从模型输出中提取语义分割掩码"""
    pred_mask = np.argmax(output_tensor, axis=1).squeeze()
    vis_mask = (pred_mask * 255).astype(np.uint8)
    if (original_width, original_height) != (MODEL_WIDTH, MODEL_HEIGHT):
        vis_mask_resized = cv2.resize(vis_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        return vis_mask_resized
    return vis_mask

# --- 车道线拟合 ---
def find_lane_pixels_and_fit(warped_mask):
    """在鸟瞰图上使用滑动窗口法寻找车道线像素并进行多项式拟合"""
    histogram = np.sum(warped_mask[warped_mask.shape[0]//2:, :], axis=0)
    midpoint = np.int32(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    margin = 100
    minpix = 50
    window_height = np.int32(warped_mask.shape[0]//nwindows)

    nonzero = warped_mask.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current, rightx_current = leftx_base, rightx_base
    left_lane_inds, right_lane_inds = [], []

    for window in range(nwindows):
        win_y_low = warped_mask.shape[0] - (window+1)*window_height
        win_y_high = warped_mask.shape[0] - window*window_height
        win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
        win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix: leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix: rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    left_fit, right_fit = None, None
    if len(leftx) > 0: left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) > 0: right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, right_fit, (leftx, lefty), (rightx, righty)


# --- 曲率和偏移计算 ---
def calculate_curvature_and_offset(left_fit, right_fit, lane_pixels, img_shape):
    """根据拟合的曲线计算曲率和车辆中心偏移（单位：米）"""
    (leftx, lefty), (rightx, righty) = lane_pixels
    h, w = img_shape
    
    left_fit_cr, right_fit_cr = None, None
    if left_fit is not None and len(lefty) > 0:
        left_fit_cr = np.polyfit(lefty * YM_PER_PIX, leftx * XM_PER_PIX, 2)
    if right_fit is not None and len(righty) > 0:
        right_fit_cr = np.polyfit(righty * YM_PER_PIX, rightx * XM_PER_PIX, 2)

    y_eval = h - 1
    y_eval_m = y_eval * YM_PER_PIX # 在真实世界尺度下计算
    
    # 曲率
    left_curverad, right_curverad = 0, 0
    if left_fit_cr is not None:
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_m + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    if right_fit_cr is not None:
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_m + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    curvature = 0
    if left_curverad > 0 and right_curverad > 0: curvature = (left_curverad + right_curverad) / 2
    elif left_curverad > 0: curvature = left_curverad
    else: curvature = right_curverad

    # 偏移
    offset_m = 0
    if left_fit is not None and right_fit is not None:
        lane_center_x_px = (left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2] + right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]) / 2
        car_center_x_px = w / 2
        offset_m = (car_center_x_px - lane_center_x_px) * XM_PER_PIX

    return curvature, offset_m


# --- 可视化函数1: 绘制最终结果图 ---
def draw_lane_on_image(original_image, left_fit, right_fit, Minv, curvature, offset):
    """将检测到的车道区域叠加回原始图像"""
    h, w = original_image.shape[:2]
    vis_image = original_image.copy()
    
    color_warp = np.zeros((h, w, 3), dtype='uint8')

    if left_fit is not None and right_fit is not None:
        ploty = np.linspace(0, h - 1, h)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
        vis_image = cv2.addWeighted(vis_image, 1, newwarp, 0.3, 0)
    
    cv2.putText(vis_image, f'Curvature: {curvature:.0f} m', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis_image, f'Offset from Center: {offset:.2f} m', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return vis_image


# --- 可视化函数2: 创建鸟瞰图 ---
def create_bev_visualization(warped_mask, left_fit, right_fit, lane_pixels):
    """创建一个彩色的鸟瞰图，用于调试和可视化"""
    (leftx, lefty), (rightx, righty) = lane_pixels
    h, w = warped_mask.shape[:2]
    
    bev_viz = np.dstack((warped_mask, warped_mask, warped_mask))
    
    bev_viz[lefty, leftx] = [0, 0, 255]   # 左车道像素: 红色
    bev_viz[righty, rightx] = [255, 0, 0]  # 右车道像素: 蓝色

    ploty = np.linspace(0, h - 1, h)
    
    if left_fit is not None:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        cv2.polylines(bev_viz, [pts_left.astype(np.int32)], isClosed=False, color=(0, 255, 255), thickness=4) # 拟合线: 黄色

    if right_fit is not None:
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
        cv2.polylines(bev_viz, [pts_right.astype(np.int32)], isClosed=False, color=(0, 255, 255), thickness=4)

    return bev_viz


# --------------------------------------------------------------------------
# --- 🚀 3. 主执行流程 ---
# --------------------------------------------------------------------------
if __name__ == '__main__':
    print("--- 开始处理静态图片车道线检测 ---")

    # --- 1. 初始化和加载 ---
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"❌ 错误: 输入图片 '{INPUT_IMAGE_PATH}' 不存在！")
        exit()

    print(f"🧠 正在加载模型: {MODEL_PATH}")
    model = InferSession(DEVICE_ID, MODEL_PATH)
    
    print(f"🖼️ 正在加载图像: {INPUT_IMAGE_PATH}")
    original_image = cv2.imread(INPUT_IMAGE_PATH)
    original_image = cv2.resize(original_image, (IMAGE_WIDTH, IMAGE_HEIGHT))

    start_time = time.time()

    # --- 2. 推理和分割 ---
    input_data = preprocess(original_image)
    outputs = model.infer([input_data])
    lane_mask = postprocess(outputs[0], IMAGE_WIDTH, IMAGE_HEIGHT)
    print("✅ 模型推理和后处理完成。")

    # --- 3. 透视变换 ---
    M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
    Minv = cv2.getPerspectiveTransform(DST_POINTS, SRC_POINTS)
    warped_mask = cv2.warpPerspective(lane_mask, M, (IMAGE_WIDTH, IMAGE_HEIGHT), flags=cv2.INTER_NEAREST)
    print("✅ 透视变换完成，已生成鸟瞰图掩码。")

    # --- 4. 路径拟合与计算 ---
    left_fit, right_fit, left_pixels, right_pixels = find_lane_pixels_and_fit(warped_mask)
    if left_fit is None or right_fit is None:
        print("⚠️ 警告: 未能同时拟合左右车道线，部分计算可能不准确。")
    else:
        print("✅ 左右车道线拟合成功。")
    
    print("\n--- 🛣️ 路线方程 (x = Ay^2 + By + C) ---")
    if left_fit is not None:
        print(f"  - 左车道线方程: x = {left_fit[0]:.6f}*y^2 + {left_fit[1]:.4f}*y + {left_fit[2]:.2f}")
    else:
        print("  - 左车道线方程: 未检测到")
    if right_fit is not None:
        print(f"  - 右车道线方程: x = {right_fit[0]:.6f}*y^2 + {right_fit[1]:.4f}*y + {right_fit[2]:.2f}")
    else:
        print("  - 右车道线方程: 未检测到")
    print("-" * 40)
        
    curvature, offset = calculate_curvature_and_offset(left_fit, right_fit, (left_pixels, right_pixels), (IMAGE_HEIGHT, IMAGE_WIDTH))
    print(f"📊 计算结果: 曲率 = {curvature:.2f} m, 中心偏移 = {offset:.2f} m")

    # --- 5. 可视化与保存 ---
    final_result_image = draw_lane_on_image(original_image, left_fit, right_fit, Minv, curvature, offset)
    bev_visualization_image = create_bev_visualization(warped_mask, left_fit, right_fit, (left_pixels, right_pixels))
    
    total_time = time.time() - start_time
    print(f"⏱️ 总处理耗时: {total_time * 1000:.1f} ms")

    # 在最终结果图上画出标定框，方便调试
    cv2.polylines(final_result_image, [SRC_POINTS.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

    # 保存两个结果文件
    cv2.imwrite(OUTPUT_IMAGE_PATH, final_result_image)
    print(f"✔️ 最终结果图已保存至 '{OUTPUT_IMAGE_PATH}'")
    cv2.imwrite(OUTPUT_BEV_IMAGE_PATH, bev_visualization_image)
    print(f"✔️ 鸟瞰图可视化已保存至 '{OUTPUT_BEV_IMAGE_PATH}'")

    print("\n--- 执行完毕 ---")