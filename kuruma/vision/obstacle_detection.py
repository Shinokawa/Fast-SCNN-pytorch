#!/usr/bin/env python3
"""
障碍物检测模块 - 基于传统OpenCV方法

针对简单环境：
- 两条白线作为车道
- 灰色地面
- 白线中的小方盒子是障碍物

使用颜色分割和形状检测来识别障碍物
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import json

class ObstacleDetector:
    """
    障碍物检测器
    
    使用传统OpenCV方法检测白线中的小方盒子障碍物
    """
    
    def __init__(self, 
                 white_lower=(200, 200, 200),
                 white_upper=(255, 255, 255),
                 gray_lower=(50, 50, 50),    # 调整为更宽的灰色范围
                 gray_upper=(200, 200, 200), # 调整为更宽的灰色范围
                 min_area=50,                # 降低最小面积阈值
                 max_area=10000,
                 min_aspect_ratio=0.2,       # 降低最小宽高比
                 max_aspect_ratio=5.0,       # 增大最大宽高比
                 erosion_kernel_size=2,      # 减小腐蚀核大小
                 dilation_kernel_size=3,     # 减小膨胀核大小
                 shrink_factor=0.7,          # 软间隔收缩因子
                 roi_top=0.2,                # 检测区域顶部比例
                 roi_bottom=0.9):            # 检测区域底部比例
        """
        初始化障碍物检测器
        
        参数:
            white_lower: 白线颜色下限 (B, G, R)
            white_upper: 白线颜色上限 (B, G, R)
            gray_lower: 灰色地面下限 (B, G, R)
            gray_upper: 灰色地面上限 (B, G, R)
            min_area: 最小障碍物面积
            max_area: 最大障碍物面积
            min_aspect_ratio: 最小宽高比
            max_aspect_ratio: 最大宽高比
            erosion_kernel_size: 腐蚀核大小
            dilation_kernel_size: 膨胀核大小
            shrink_factor: 软间隔收缩因子 (0.5-1.0)
            roi_top: 检测区域顶部比例 (0.0-1.0)
            roi_bottom: 检测区域底部比例 (0.0-1.0)
        """
        # 颜色范围
        self.white_lower = np.array(white_lower, dtype=np.uint8)
        self.white_upper = np.array(white_upper, dtype=np.uint8)
        self.gray_lower = np.array(gray_lower, dtype=np.uint8)
        self.gray_upper = np.array(gray_upper, dtype=np.uint8)
        
        # 形状过滤参数
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        
        # 软间隔参数
        self.shrink_factor = max(0.5, min(1.0, shrink_factor))
        
        # ROI参数
        self.roi_top = max(0.0, min(1.0, roi_top))
        self.roi_bottom = max(0.0, min(1.0, roi_bottom))
        
        # 形态学操作核
        self.erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
        self.dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        
        print("✅ 障碍物检测器初始化完成")
        print(f"📊 白线颜色范围: {tuple(self.white_lower)} - {tuple(self.white_upper)}")
        print(f"📊 灰色地面范围: {tuple(self.gray_lower)} - {tuple(self.gray_upper)}")
        print(f"📊 面积范围: {self.min_area} - {self.max_area}")
        print(f"📊 宽高比范围: {self.min_aspect_ratio:.2f} - {self.max_aspect_ratio:.2f}")
        print(f"🎯 软间隔收缩因子: {self.shrink_factor:.2f}")
        print(f"📐 ROI检测区域: {self.roi_top:.1%} - {self.roi_bottom:.1%}")
    
    def _create_conservative_lane_mask(self, lane_mask, width, roi_top, roi_bottom):
        """
        创建保守的车道掩码（使用图像中心区域）
        
        参数:
            lane_mask: 要填充的车道掩码
            width: 图像宽度
            roi_top: ROI顶部
            roi_bottom: ROI底部
        """
        print("⚠️ 使用保守的中心车道估计策略")
        
        # 使用图像中心的一定比例作为车道区域
        lane_width_ratio = 0.6  # 车道宽度占图像宽度的60%
        lane_width = int(width * lane_width_ratio)
        lane_center = width // 2
        
        left_bound = lane_center - lane_width // 2
        right_bound = lane_center + lane_width // 2
        
        # 确保边界在图像范围内
        left_bound = max(0, left_bound)
        right_bound = min(width, right_bound)
        
        # 填充车道掩码
        for y in range(roi_top, roi_bottom):
            lane_mask[y, left_bound:right_bound] = 255
        
        print(f"✅ 保守策略：ROI区域 Y({roi_top}-{roi_bottom}), 车道宽度 {lane_width}px")
    
    def detect_obstacles(self, image: np.ndarray, segmentation_mask: Optional[np.ndarray] = None) -> Dict:
        """
        检测障碍物 - 结合颜色、位置和形状的"减法策略"
        
        策略:
        1. 在精准的车道掩码内，提取所有白色/高亮物体（车道线+障碍物）。
        2. 利用位置和形状特征，专门识别出车道线。
        3. 从所有白色物体中减去车道线，剩下的就是障碍物候选。
        4. 对候选物体进行最终的形状验证。
        
        参数:
            image: 输入图像 (BGR格式)
            segmentation_mask: 分割掩码，必须提供以确保准确性
            
        返回:
            detection_result: 包含障碍物信息的字典
        """
        height, width = image.shape[:2]
        
        # 1. 基于深度学习分割结果创建精确的车道区域
        if segmentation_mask is not None:
            print("🧠 使用深度学习分割结果创建车道区域")
            lane_mask = self._create_lane_mask_from_segmentation(segmentation_mask, height, width)
        else:
            print("⚠️ 警告：未提供分割掩码，使用保守的中心区域估计")
            # 如果没有分割掩码，使用保守策略
            lane_mask = np.zeros((height, width), dtype=np.uint8)
            roi_top = int(height * self.roi_top)
            roi_bottom = int(height * self.roi_bottom)
            self._create_conservative_lane_mask(lane_mask, width, roi_top, roi_bottom)
        
        print("🎯 开始减法策略：提取白色物体 → 识别车道线 → 执行减法 → 验证障碍物")
        
        # ===== 🎯 步骤一：在车道内提取所有白色/高亮区域 =====
        print("📋 步骤1: 提取所有白色/高亮区域")
        
        # 转换为HSV色彩空间，对光照变化更鲁棒
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 定义一个宽松的白色/高亮颜色范围
        # S(饱和度)较低，V(亮度)较高
        lower_bright = np.array([0, 0, 180])   # 亮度下限调高
        upper_bright = np.array([180, 50, 255]) # 饱和度上限放宽
        
        bright_mask = cv2.inRange(hsv, lower_bright, upper_bright)
        
        # 只保留车道内的白色区域
        all_bright_objects_mask = cv2.bitwise_and(bright_mask, lane_mask)

        # 形态学操作清理噪点
        kernel = np.ones((3,3), np.uint8)
        all_bright_objects_mask = cv2.morphologyEx(all_bright_objects_mask, cv2.MORPH_CLOSE, kernel)
        
        print(f"   发现白色/高亮区域像素数: {np.sum(all_bright_objects_mask > 0)}")
        
        # ===== 🔍 步骤二：识别并隔离车道线 =====
        print("📋 步骤2: 识别车道线")
        
        # 🎯 正确的车道线边界区域：只有左右两侧，不包括上下
        
        # 创建左右边界掩码
        left_right_border_mask = np.zeros_like(lane_mask)
        
        # 找到车道掩码的轮廓
        lane_contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if lane_contours:
            # 找到最大的车道轮廓（主车道）
            main_lane_contour = max(lane_contours, key=cv2.contourArea)
            
            # 对每一行，找到车道的左右边界
            roi_top = int(height * self.roi_top)
            roi_bottom = int(height * self.roi_bottom)
            
            border_width = 15  # 边界宽度，可调参数
            
            for y in range(roi_top, roi_bottom):
                # 找到这一行中车道掩码的最左和最右像素
                row_pixels = np.where(lane_mask[y, :] > 0)[0]
                
                if len(row_pixels) > 0:
                    left_x = row_pixels[0]
                    right_x = row_pixels[-1]
                    
                    # 创建左边界区域
                    left_start = max(0, left_x - border_width // 2)
                    left_end = min(width, left_x + border_width // 2)
                    left_right_border_mask[y, left_start:left_end] = 255
                    
                    # 创建右边界区域  
                    right_start = max(0, right_x - border_width // 2)
                    right_end = min(width, right_x + border_width // 2)
                    left_right_border_mask[y, right_start:right_end] = 255
        
        # 为了对比，也保留原来的全周边界方法
        erosion_kernel = np.ones((8, 8), np.uint8)
        eroded_lane_mask = cv2.erode(lane_mask, erosion_kernel, iterations=1)
        full_border_mask = cv2.subtract(lane_mask, eroded_lane_mask)
        
        print(f"   创建边界区域: 左右边界像素数={np.sum(left_right_border_mask > 0)}, "
              f"全周边界像素数={np.sum(full_border_mask > 0)}")
        
        # 使用正确的左右边界作为主要判断依据
        lane_line_border_mask = left_right_border_mask
        combined_border_mask = left_right_border_mask

        # 找到所有亮区中的轮廓
        contours, _ = cv2.findContours(all_bright_objects_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lane_lines_only_mask = np.zeros_like(lane_mask)
        lane_line_count = 0
        
        print(f"   分析 {len(contours)} 个白色轮廓，识别车道线...")
        
        for cnt in contours:
            # 判断轮廓是否为车道线
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            
            # 跳过过小的轮廓
            if area < 20:
                continue
            
            # 🎯 条件1: 轮廓与边界区域有重叠（使用合并的边界区域）
            contour_mask = np.zeros_like(lane_mask)
            cv2.drawContours(contour_mask, [cnt], -1, (255,255,255), -1)
            
            # 检查与左右边界的重叠
            border_overlap = cv2.bitwise_and(contour_mask, lane_line_border_mask)
            combined_overlap = cv2.bitwise_and(contour_mask, combined_border_mask)
            
            border_overlap_ratio = np.sum(border_overlap > 0) / np.sum(contour_mask > 0) if np.sum(contour_mask > 0) > 0 else 0
            combined_overlap_ratio = np.sum(combined_overlap > 0) / np.sum(contour_mask > 0) if np.sum(contour_mask > 0) > 0 else 0
            
            # 🎯 计算线条的真实宽高比
            try:
                # 使用最小外接矩形获得精确的线条特征
                min_rect = cv2.minAreaRect(cnt)
                (rect_center_x, rect_center_y), (rect_width, rect_height), angle = min_rect
                
                # 确保width是线的宽度（较小维度），height是线的长度（较大维度）
                if rect_width > rect_height:
                    line_width, line_length = rect_height, rect_width
                else:
                    line_width, line_length = rect_width, rect_height
                
                # 线条的真实宽高比 = 宽度/长度（对于车道线应该很小，如0.01-0.05）
                line_aspect_ratio = line_width / line_length if line_length > 0 else 0
                
                # 传统的边界矩形宽高比（用于显示对比）
                bbox_aspect_ratio = w / h if h > 0 else 0
                
            except:
                # 如果计算失败，使用简化方法
                line_aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
                bbox_aspect_ratio = w / h if h > 0 else 0
            
            # 条件2: 形状特征（严格的线条特征）
            is_true_line = line_aspect_ratio < 0.12  # 宽度/长度 < 12%，真正的线条
            is_medium_line = line_aspect_ratio < 0.15  # 宽度/长度 < 15%，可能的线条
            is_long_and_thin = bbox_aspect_ratio < 0.5  # 传统判断：高度比宽度大很多
            
            # 条件3: 位置特征（靠近图像边缘，真正的车道线通常在边缘）
            is_near_edge = (x < width * 0.25) or (x + w > width * 0.75)  # 放宽边缘范围
            
            # 条件4: 延伸特征（垂直延伸较长）
            is_vertical_extension = h > height * 0.25  # 稍微降低延伸要求
            
            # 🎯 基于正确左右边界的车道线判断逻辑
            is_lane_line = (
                # 高置信度：真正的线条 + 左右边界重叠
                (is_true_line and border_overlap_ratio > 0.1) or
                # 中等置信度：较好线条 + 强左右边界重叠  
                (is_medium_line and border_overlap_ratio > 0.2) or
                # 位置置信度：靠近边缘 + 线条特征 + 一些边界重叠
                (is_near_edge and is_true_line and border_overlap_ratio > 0.05) or
                # 传统置信度：细长形状 + 边缘位置 + 垂直延伸
                (is_long_and_thin and is_near_edge and is_vertical_extension)
            )
            
            if is_lane_line:
                cv2.drawContours(lane_lines_only_mask, [cnt], -1, (255,255,255), -1)
                lane_line_count += 1
                print(f"     识别车道线: 面积={area:.0f}, 边界宽高比={bbox_aspect_ratio:.1f}, "
                      f"线条宽度/长度={line_aspect_ratio:.3f}, 左右边界重叠={border_overlap_ratio:.2f}")

        print(f"   共识别出 {lane_line_count} 个车道线区域")
        
        # ===== ✂️ 步骤三：执行"减法"操作 =====
        print("📋 步骤3: 执行减法操作")
        
        obstacles_candidate_mask = cv2.subtract(all_bright_objects_mask, lane_lines_only_mask)

        # 对候选掩码进行一些清理，去除减法后残留的小噪点
        obstacles_candidate_mask = cv2.morphologyEx(obstacles_candidate_mask, cv2.MORPH_OPEN, kernel)
        
        candidate_pixels = np.sum(obstacles_candidate_mask > 0)
        print(f"   减法后剩余像素数: {candidate_pixels}")
        
        # ===== ✅ 步骤四：分析剩余轮廓，确认障碍物 =====
        print("📋 步骤4: 分析障碍物候选")
        
        obstacle_contours, _ = cv2.findContours(obstacles_candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        print(f"   发现 {len(obstacle_contours)} 个障碍物候选轮廓")

        # ROI区域限制
        roi_top = int(height * self.roi_top)
        roi_bottom = int(height * self.roi_bottom)
        center_x = width // 2
        
        for i, contour in enumerate(obstacle_contours):
            area = cv2.contourArea(contour)
            
            # 面积过滤：调整为盒子的大致面积范围
            if area < 100 or area > 8000:  # 提高上限到8000
                print(f"     候选{i+1}: 面积{area:.0f} - 被面积过滤器排除")
                continue

            x, y, w, h = cv2.boundingRect(contour)
            
            # ROI位置过滤
            if y < roi_top or y + h > roi_bottom:
                print(f"     候选{i+1}: 不在ROI区域内 - 被位置过滤器排除")
                continue
            
            # 中心位置优先
            center_contour_x = x + w // 2
            center_contour_y = y + h // 2
            distance_to_center = abs(center_contour_x - center_x)
            
            # 🎯 专门针对线条的宽高比计算
            try:
                # 方法1：最小外接矩形（对斜线更准确）
                min_rect = cv2.minAreaRect(contour)
                (rect_center_x, rect_center_y), (rect_width, rect_height), angle = min_rect
                
                # 确保width是线的宽度（较小维度），height是线的长度（较大维度）
                if rect_width > rect_height:
                    line_width, line_length = rect_height, rect_width
                else:
                    line_width, line_length = rect_width, rect_height
                
                # 🎯 线条的真实宽高比 = 宽度/长度（应该很小）
                line_aspect_ratio = line_width / line_length if line_length > 0 else 0
                
                # 边界矩形的传统宽高比（用于对比）
                bbox_aspect_ratio = w / h if h > 0 else 0
                
                # 🎯 计算填充度：边界矩形中被轮廓填满的比例
                bbox_area = w * h
                fill_ratio = area / bbox_area if bbox_area > 0 else 0
                
                # 🎯 线条特征判断
                # 对于车道线：line_aspect_ratio应该很小（< 0.05）
                # 对于障碍物：line_aspect_ratio应该相对较大（> 0.2）
                is_line_like = line_aspect_ratio < 0.08  # 宽度/长度 < 8%，典型线条特征
                
                print(f"     候选{i+1}几何分析: 面积={area:.0f}, 边界矩形宽高比={bbox_aspect_ratio:.2f}, "
                      f"线条宽度/长度={line_aspect_ratio:.3f}, 填充度={fill_ratio:.2f}")
                
            except:
                # 如果最小外接矩形计算失败，使用简化计算
                bbox_aspect_ratio = w / h if h > 0 else 0
                line_aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
                is_line_like = line_aspect_ratio < 0.08
                fill_ratio = area / (w * h) if w * h > 0 else 0
                print(f"     候选{i+1}使用简化计算: 边界宽高比={bbox_aspect_ratio:.2f}, "
                      f"线条比例={line_aspect_ratio:.3f}, 填充度={fill_ratio:.2f}")
            
            # 🎯 线条过滤：排除典型的线条特征
            if is_line_like:
                print(f"     候选{i+1}: 线条宽度/长度比={line_aspect_ratio:.3f}过小 - 典型线条特征，被排除")
                continue
            
            # 🎯 不再使用边界矩形宽高比进行极端过滤，因为我们已经有了更准确的线条比例
            
            # 条件3：填充度过滤 - 矩形应该大部分被填满
            if fill_ratio < 0.3:  # 填充度至少30%
                print(f"     候选{i+1}: 填充度{fill_ratio:.2f}过低 - 被填充度过滤器排除")
                continue

            # 坚实度过滤，实心物体坚实度接近1
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < 0.6:  # 降低坚实度要求到0.6
                print(f"     候选{i+1}: 坚实度{solidity:.2f} - 被实心度过滤器排除")
                continue
            
            # 计算紧致度
            perimeter = cv2.arcLength(contour, True)
            compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # 位置得分：越靠近中心得分越高
            position_score = 1.0 - (distance_to_center / (width * 0.5))
            position_score = max(0.0, min(1.0, position_score))
            
            # 🎯 综合置信度计算（加入填充度和几何特征）
            geometry_score = (line_aspect_ratio * 2 + fill_ratio) / 3  # 几何得分
            confidence = (solidity * 0.3 + compactness * 0.2 + position_score * 0.2 + geometry_score * 0.3)
            
            print(f"     ✅ 候选{i+1}确认为障碍物: 面积={area:.0f}, 填充度={fill_ratio:.2f}, "
                  f"几何得分={geometry_score:.2f}, 置信度={confidence:.2f}")
            
            obstacle = {
                'bbox': (x, y, w, h),
                'center': (center_contour_x, center_contour_y),
                'area': area,
                'aspect_ratio': bbox_aspect_ratio,  # 边界矩形宽高比
                'line_aspect_ratio': line_aspect_ratio,  # 线条宽度/长度比
                'fill_ratio': fill_ratio,  # 新增：填充度
                'solidity': solidity,
                'compactness': compactness,
                'position_score': position_score,
                'geometry_score': geometry_score,  # 新增：几何得分
                'distance_to_center': distance_to_center,
                'contour': contour.tolist(),
                'confidence': confidence
            }
            obstacles.append(obstacle)

        # 按置信度排序
        obstacles.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 为了调试，准备最终障碍物掩码
        final_obstacle_mask = np.zeros_like(lane_mask)
        if obstacles:
            for obstacle in obstacles:
                contour_array = np.array(obstacle['contour'], dtype=np.int32)
                cv2.fillPoly(final_obstacle_mask, [contour_array], (255,))

        # 返回检测结果，包含调试信息
        result = {
            'obstacles': obstacles,
            'num_obstacles': len(obstacles),
            'lane_mask': lane_mask,  # 精准ROI
            'obstacle_mask': final_obstacle_mask,  # 最终确认的障碍物
            'debug_masks': {  # 新增，用于调试
                '1_all_bright': all_bright_objects_mask,
                '2_lane_border': lane_line_border_mask,
                '3_lane_lines_only': lane_lines_only_mask,
                '4_candidates': obstacles_candidate_mask
            },
            'detection_params': {
                'detection_method': 'subtraction_based',  # 新的减法策略标识
                'min_area': self.min_area,
                'max_area': self.max_area,
                'min_aspect_ratio': self.min_aspect_ratio,
                'max_aspect_ratio': self.max_aspect_ratio,
                'roi_top': self.roi_top,
                'roi_bottom': self.roi_bottom,
                'bright_color_range': (tuple(lower_bright), tuple(upper_bright)),
                'border_width': 15,  # 替换原来的erosion_kernel_size
                'lane_lines_found': lane_line_count,
                'candidates_pixels': candidate_pixels
            }
        }
        
        print(f"✅ 减法策略完成: 发现 {len(obstacles)} 个障碍物")
        
        return result
    
    def _create_lane_mask_from_segmentation(self, segmentation_mask: np.ndarray, height: int, width: int) -> np.ndarray:
        """
        基于深度学习分割结果创建车道掩码
        
        参数:
            segmentation_mask: 分割掩码 (0表示背景，255表示可驾驶区域)
            height: 图像高度
            width: 图像宽度
            
        返回:
            lane_mask: 车道掩码
        """
        # 🎯 ROI优化：只在指定区域内进行车道检测，避免边缘噪声
        roi_top = int(height * self.roi_top)
        roi_bottom = int(height * self.roi_bottom)
        
        # 确保分割掩码是二值图像
        if len(segmentation_mask.shape) == 3:
            segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2GRAY)
        
        # 二值化处理
        _, binary_mask = cv2.threshold(segmentation_mask, 127, 255, cv2.THRESH_BINARY)
        
        # 🎯 关键优化：只在ROI区域内提取轮廓，避免边缘噪声影响
        roi_mask = np.zeros_like(binary_mask)
        roi_mask[roi_top:roi_bottom, :] = binary_mask[roi_top:roi_bottom, :]
        
        # 查找ROI区域内的可驾驶区域轮廓
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("⚠️ 未在ROI区域内找到可驾驶区域")
            # 创建默认的空掩码
            lane_mask = np.zeros((height, width), dtype=np.uint8)
            self._create_conservative_lane_mask(lane_mask, width, roi_top, roi_bottom)
            return lane_mask
        
        # 找到最大的连通区域（应该是主要的可驾驶区域）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 🎯 SVM软间隔式优化：智能凸包收缩
        hull = cv2.convexHull(largest_contour)
        
        # 计算凸包的面积和周长
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, True)
        
        # 如果凸包过于"松散"，进行收缩优化
        compactness = 4 * np.pi * hull_area / (hull_perimeter ** 2) if hull_perimeter > 0 else 0
        
        if compactness < 0.3:  # 形状不够紧致
            print("🎯 应用软间隔优化：收缩凸包以聚焦核心区域")
            
            # 方法1：基于距离变换的收缩
            temp_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(temp_mask, [hull], (255,))
            
            # 计算距离变换
            dist_transform = cv2.distanceTransform(temp_mask, cv2.DIST_L2, 5)
            
            # 找到距离变换的最大值点（形状中心）
            max_dist = np.max(dist_transform)
            
            # 设置收缩阈值（类似软间隔的松弛变量）
            shrink_factor = self.shrink_factor  # 使用实例变量
            threshold = max_dist * shrink_factor
            
            # 创建收缩后的掩码
            _, shrunk_mask = cv2.threshold((dist_transform > threshold).astype(np.uint8) * 255, 127, 255, cv2.THRESH_BINARY)
            
            # 重新计算轮廓
            shrunk_contours, _ = cv2.findContours(shrunk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if shrunk_contours:
                # 选择最大的收缩轮廓
                largest_shrunk = max(shrunk_contours, key=cv2.contourArea)
                hull = cv2.convexHull(largest_shrunk)
                print(f"✅ 凸包收缩完成：面积 {hull_area:.0f} → {cv2.contourArea(hull):.0f}px²")
            else:
                print("⚠️ 收缩失败，使用原始凸包")
        
        # 创建车道掩码
        lane_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(lane_mask, [hull], (255,))
        
        # 🎯 严格限制在ROI区域内
        lane_mask[:roi_top, :] = 0      # 清除ROI上方区域
        lane_mask[roi_bottom:, :] = 0   # 清除ROI下方区域
        
        # 🎯 进一步优化：移除小的孤立区域
        # 使用形态学操作连接相近的区域
        kernel = np.ones((5, 5), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
        
        # 移除小的连通区域
        contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 只保留最大的连通区域
            largest_final = max(contours, key=cv2.contourArea)
            lane_mask.fill(0)
            cv2.fillPoly(lane_mask, [largest_final], (255,))
        
        print(f"✅ 基于深度学习分割创建车道掩码: ROI({roi_top}-{roi_bottom}), 面积={cv2.contourArea(hull):.0f}px²")
        
        return lane_mask
    
    def visualize_obstacles(self, image: np.ndarray, detection_result: Dict) -> np.ndarray:
        """
        可视化障碍物检测结果
        
        参数:
            image: 原始图像
            detection_result: 检测结果
            
        返回:
            visualization: 可视化图像
        """
        vis_image = image.copy()
        obstacles = detection_result['obstacles']
        
        # 绘制障碍物
        for i, obstacle in enumerate(obstacles):
            x, y, w, h = obstacle['bbox']
            center_x, center_y = obstacle['center']
            confidence = obstacle['confidence']
            
            # 绘制边界框
            color = (0, 0, 255)  # 红色
            thickness = 2
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, thickness)
            
            # 绘制中心点
            cv2.circle(vis_image, (center_x, center_y), 3, color, -1)
            
            # 绘制标签
            label = f"Obstacle {i+1}"
            confidence_text = f"{confidence:.2f}"
            
            # 文本背景
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(vis_image, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # 文本
            cv2.putText(vis_image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 置信度文本
            cv2.putText(vis_image, confidence_text, (x, y + h + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 在图像上添加检测统计信息
        stats_text = f"Obstacles: {len(obstacles)}"
        cv2.putText(vis_image, stats_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return vis_image
    
    def create_debug_visualization(self, image: np.ndarray, detection_result: Dict) -> np.ndarray:
        """
        创建调试可视化图像，显示减法策略的各个步骤
        
        参数:
            image: 原始图像
            detection_result: 检测结果
            
        返回:
            debug_vis: 调试可视化图像
        """
        h, w = image.shape[:2]
        
        # 创建2x4布局的调试图像
        debug_vis = np.zeros((h * 2, w * 4, 3), dtype=np.uint8)

        # 获取调试掩码
        debug_masks = detection_result.get('debug_masks', {})
        lane_mask = detection_result.get('lane_mask', np.zeros((h, w), dtype=np.uint8))
        obstacle_mask = detection_result.get('obstacle_mask', np.zeros((h, w), dtype=np.uint8))
        
        # 第一行：原始图像 → 车道掩码 → 所有亮物体 → 车道线边界
        # 左上：原始图像
        debug_vis[0:h, 0:w] = image
        cv2.putText(debug_vis, "Original Image", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 中上1：车道掩码（精准ROI）
        lane_mask_colored = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
        lane_mask_colored[lane_mask > 0] = [0, 255, 0]  # 绿色
        debug_vis[0:h, w:w*2] = lane_mask_colored
        cv2.putText(debug_vis, "Lane Mask (Precise ROI)", (w + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 中上2：所有高亮物体（步骤1结果）
        mask1 = debug_masks.get('1_all_bright', np.zeros((h,w), dtype=np.uint8))
        mask1_colored = cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR)
        mask1_colored[mask1 > 0] = [255, 255, 255]  # 白色
        debug_vis[0:h, w*2:w*3] = mask1_colored
        cv2.putText(debug_vis, "1. All Bright Objects", (w*2 + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 右上：车道线边界区域
        mask2 = debug_masks.get('2_lane_border', np.zeros((h,w), dtype=np.uint8))
        mask2_colored = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
        mask2_colored[mask2 > 0] = [0, 255, 255]  # 黄色
        debug_vis[0:h, w*3:w*4] = mask2_colored
        cv2.putText(debug_vis, "2. Lane Border Region", (w*3 + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 第二行：识别的车道线 → 减法候选区域 → 最终障碍物 → 叠加结果
        # 左下：仅车道线（步骤2结果）
        mask3 = debug_masks.get('3_lane_lines_only', np.zeros((h,w), dtype=np.uint8))
        mask3_colored = cv2.cvtColor(mask3, cv2.COLOR_GRAY2BGR)
        mask3_colored[mask3 > 0] = [255, 0, 0]  # 蓝色
        debug_vis[h:h*2, 0:w] = mask3_colored
        cv2.putText(debug_vis, "3. Lane Lines Only", (5, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 中下1：减法候选区域（步骤3结果）
        mask4 = debug_masks.get('4_candidates', np.zeros((h,w), dtype=np.uint8))
        mask4_colored = cv2.cvtColor(mask4, cv2.COLOR_GRAY2BGR)
        mask4_colored[mask4 > 0] = [0, 165, 255]  # 橙色
        debug_vis[h:h*2, w:w*2] = mask4_colored
        cv2.putText(debug_vis, "4. Obstacle Candidates", (w + 5, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_vis, "(After Subtraction)", (w + 5, h+35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 中下2：最终确认的障碍物
        obstacle_mask_colored = cv2.cvtColor(obstacle_mask, cv2.COLOR_GRAY2BGR)
        obstacle_mask_colored[obstacle_mask > 0] = [0, 0, 255]  # 红色
        debug_vis[h:h*2, w*2:w*3] = obstacle_mask_colored
        cv2.putText(debug_vis, "5. Final Obstacles", (w*2 + 5, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 右下：最终检测结果叠加在原图上
        final_result = self.visualize_obstacles(image.copy(), detection_result)
        debug_vis[h:h*2, w*3:w*4] = final_result
        cv2.putText(debug_vis, "Final Detection", (w*3 + 5, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 添加策略流程说明
        detection_params = detection_result.get('detection_params', {})
        detection_method = detection_params.get('detection_method', 'unknown')
        lane_lines_found = detection_params.get('lane_lines_found', 0)
        candidates_pixels = detection_params.get('candidates_pixels', 0)
        
        # 底部添加统计信息
        info_y = h*2 - 30
        cv2.putText(debug_vis, f"Method: {detection_method}", (5, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(debug_vis, f"Lane lines found: {lane_lines_found}", (w + 5, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(debug_vis, f"Candidate pixels: {candidates_pixels}", (w*2 + 5, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        num_obstacles = detection_result.get('num_obstacles', 0)
        cv2.putText(debug_vis, f"Final obstacles: {num_obstacles}", (w*3 + 5, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # 在顶部添加策略流程箭头
        arrow_y = 5
        cv2.putText(debug_vis, "Subtraction Strategy Flow: Extract Bright → Identify Lanes → Subtract → Validate", 
                   (w//2 - 200, arrow_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return debug_vis


def save_obstacle_data(obstacles: List[Dict], output_path: str) -> None:
    """
    保存障碍物检测数据到JSON文件
    
    参数:
        obstacles: 障碍物列表
        output_path: 输出文件路径
    """
    # 处理不可序列化的numpy数组
    serializable_obstacles = []
    for obstacle in obstacles:
        serializable_obstacle = obstacle.copy()
        # contour已经转换为列表，无需额外处理
        serializable_obstacles.append(serializable_obstacle)
    
    data = {
        'obstacles': serializable_obstacles,
        'num_obstacles': len(obstacles),
        'timestamp': str(np.datetime64('now'))
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 障碍物数据已保存: {output_path}")


def create_obstacle_detector(config: Optional[Dict] = None) -> ObstacleDetector:
    """
    创建障碍物检测器的工厂函数
    
    参数:
        config: 配置字典，可选
        
    返回:
        ObstacleDetector: 障碍物检测器实例
    """
    if config is None:
        config = {}
    
    return ObstacleDetector(
        white_lower=config.get('white_lower', (200, 200, 200)),
        white_upper=config.get('white_upper', (255, 255, 255)),
        gray_lower=config.get('gray_lower', (50, 50, 50)),
        gray_upper=config.get('gray_upper', (200, 200, 200)),
        min_area=config.get('min_area', 50),
        max_area=config.get('max_area', 10000),
        min_aspect_ratio=config.get('min_aspect_ratio', 0.2),
        max_aspect_ratio=config.get('max_aspect_ratio', 5.0),
        erosion_kernel_size=config.get('erosion_kernel_size', 2),
        dilation_kernel_size=config.get('dilation_kernel_size', 3),
        shrink_factor=config.get('shrink_factor', 0.7),
        roi_top=config.get('roi_top', 0.2),
        roi_bottom=config.get('roi_bottom', 0.9)
    ) 