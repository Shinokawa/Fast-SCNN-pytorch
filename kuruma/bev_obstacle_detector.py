import cv2
import numpy as np

class BevObstacleDetector:
    def __init__(self, config):
        """
        使用配置字典初始化检测器。
        config = {
            "safe_zone_ratio": (0.4, 0.6, 0.8, 1.0), # (x_start, x_end, y_start, y_end) 比例
            "hsv_k": 2.5, # HSV通道标准差倍数
            "min_obstacle_area": 100, # 最小障碍物面积（像素）
            "morph_kernel_size": 5, # 形态学操作核大小
            "max_path_dist": 50 # 路径点最大距离筛选
        }
        """
        self.config = config
        print("🚧 [BEV Obstacle Detector] Initialized with traditional CV methods.")

    def detect(self, bird_eye_image, planned_path_pixels=None):
        """
        主检测函数
        :param bird_eye_image: 原始的鸟瞰图 (BGR格式)
        :param planned_path_pixels: 规划的路径点（像素坐标），用于聚焦检测
        :return: (obstacle_mask, obstacle_contours)
                 - obstacle_mask: 标记了障碍物的二值图像
                 - obstacle_contours: 障碍物的轮廓列表
        """
        h, w = bird_eye_image.shape[:2]
        # 1. 获取安全区
        x0, x1, y0, y1 = self.config.get("safe_zone_ratio", (0.4, 0.6, 0.8, 1.0))
        sx0, sx1 = int(x0 * w), int(x1 * w)
        sy0, sy1 = int(y0 * h), int(y1 * h)
        safe_zone = bird_eye_image[sy0:sy1, sx0:sx1]

        # 2. 动态学习路面颜色模型
        hsv = cv2.cvtColor(safe_zone, cv2.COLOR_BGR2HSV)
        h_mean, s_mean, v_mean = hsv[...,0].mean(), hsv[...,1].mean(), hsv[...,2].mean()
        h_std, s_std, v_std = hsv[...,0].std(), hsv[...,1].std(), hsv[...,2].std()
        k = self.config.get("hsv_k", 2.5)
        lower = np.array([
            max(0, h_mean - k*h_std),
            max(0, s_mean - k*s_std),
            max(0, v_mean - k*v_std)
        ], dtype=np.uint8)
        upper = np.array([
            min(180, h_mean + k*h_std),
            min(255, s_mean + k*s_std),
            min(255, v_mean + k*v_std)
        ], dtype=np.uint8)

        # 3. 路面分割
        hsv_full = cv2.cvtColor(bird_eye_image, cv2.COLOR_BGR2HSV)
        road_mask = cv2.inRange(hsv_full, lower, upper)
        potential_obstacle_mask = cv2.bitwise_not(road_mask)

        # 4. 形态学处理
        kernel_size = self.config.get("morph_kernel_size", 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned = cv2.morphologyEx(potential_obstacle_mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        # 5. 轮廓提取与筛选
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.config.get("min_obstacle_area", 100)
        filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # 可选：与路径点距离筛选
        if planned_path_pixels is not None and len(planned_path_pixels) > 0:
            filtered2 = []
            for cnt in filtered:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    dists = [np.linalg.norm(np.array([cx, cy]) - np.array(pt)) for pt in planned_path_pixels]
                    if min(dists) < self.config.get("max_path_dist", 50):
                        filtered2.append(cnt)
            filtered = filtered2

        return cleaned, filtered 