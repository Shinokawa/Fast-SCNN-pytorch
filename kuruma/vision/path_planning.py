#!/usr/bin/env python3
"""
路径规划模块 - 中心线提取、路径平滑和控制地图生成

包含：
- PathPlanner: 路径规划器类，用于从鸟瞰图中提取和规划可行驶路径
- create_control_map: 创建控制地图函数
- add_grid_to_control_map: 在控制地图上添加网格
- visualize_path_on_control_map: 在控制地图上可视化路径
- world_to_pixels: 世界坐标转换为像素坐标
"""

import numpy as np
import cv2

# 检查scipy是否可用
try:
    from scipy.interpolate import interp1d, splprep, splev
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
    print("✅ SciPy已加载，支持高级路径平滑")
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy未安装，将使用numpy基础拟合")

# 导入标定模块
from core.calibration import get_corrected_calibration

# ---------------------------------------------------------------------------------
# --- 🗺️ 控制地图生成模块 ---
# ---------------------------------------------------------------------------------

def create_control_map(bird_eye_mask, view_params, add_grid=True, add_path=True,
                      path_smooth_method='polynomial', path_degree=3, 
                      num_waypoints=20, min_road_width=10, edge_computing=False,
                      force_bottom_center=True):
    """
    创建用于路径规划的控制地图
    
    参数：
        bird_eye_mask: 鸟瞰图分割掩码
        view_params: 视图参数
        add_grid: 是否添加网格
        add_path: 是否添加路径规划
        path_smooth_method: 路径平滑方法
        path_degree: 路径拟合阶数
        num_waypoints: 路径点数量
        min_road_width: 最小可行驶宽度
        edge_computing: 边缘计算模式
        force_bottom_center: 强制拟合曲线过底边中点
    
    返回：
        control_map: 控制地图 (三通道BGR图像)
        path_data: 路径规划数据（如果add_path=True）
    """
    # 创建控制地图
    control_map = np.zeros((bird_eye_mask.shape[0], bird_eye_mask.shape[1], 3), dtype=np.uint8)
    
    # 可驾驶区域 - 绿色
    control_map[bird_eye_mask > 0] = [0, 255, 0]  # BGR绿色
    
    # 不可驾驶区域 - 保持黑色
    # control_map[bird_eye_mask == 0] = [0, 0, 0]  # 已经是黑色
    
    # 路径规划
    path_data = None
    if add_path:
        try:
            planner = PathPlanner(view_params)
            path_data = planner.plan_complete_path(
                bird_eye_mask, 
                smooth_method=path_smooth_method,
                degree=path_degree,
                num_waypoints=num_waypoints,
                min_width=min_road_width,
                fast_mode=edge_computing,
                force_bottom_center=force_bottom_center
            )
            
            # 在控制地图上可视化路径
            control_map = visualize_path_on_control_map(control_map, path_data, view_params)
            
            print(f"🛣️ 路径规划完成:")
            print(f"   - 中心线点数: {path_data['num_centerline_points']}")
            print(f"   - 路径点数: {path_data['num_waypoints']}")
            print(f"   - 路径长度: {path_data['path_length']:.1f} cm")
            
        except Exception as e:
            print(f"⚠️ 路径规划失败: {e}")
            path_data = None
    
    if add_grid:
        control_map = add_grid_to_control_map(control_map, view_params)
    
    return control_map, path_data

def add_grid_to_control_map(control_map, view_params):
    """
    在控制地图上添加网格和坐标标签
    
    参数：
        control_map: 控制地图
        view_params: 视图参数
    
    返回：
        带网格的控制地图
    """
    annotated_map = control_map.copy()
    
    min_x, min_y, max_x, max_y = view_params['view_bounds']
    pixels_per_unit = view_params['pixels_per_unit']
    output_width, output_height = view_params['output_size']
    
    # 绘制网格
    grid_interval = 10  # 网格间隔（单位：cm）
    grid_color = (128, 128, 128)  # 灰色
    origin_color = (0, 0, 255)    # 红色原点
    
    # 垂直线
    x = min_x
    while x <= max_x:
        if abs(x % grid_interval) < 0.1:  # 处理浮点数精度问题
            pixel_x = int((x - min_x) * pixels_per_unit)
            if 0 <= pixel_x < output_width:
                cv2.line(annotated_map, (pixel_x, 0), (pixel_x, output_height-1), grid_color, 1)
                
                # 添加X坐标标签
                if abs(x) > 0.1:  # 避免在原点重复标注
                    label = f"{int(x)}"
                    cv2.putText(annotated_map, label, (pixel_x + 2, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_color, 1)
        x += grid_interval / 2
    
    # 水平线
    y = min_y
    while y <= max_y:
        if abs(y % grid_interval) < 0.1:  # 处理浮点数精度问题
            pixel_y = int((y - min_y) * pixels_per_unit)
            if 0 <= pixel_y < output_height:
                cv2.line(annotated_map, (0, pixel_y), (output_width-1, pixel_y), grid_color, 1)
                
                # 添加Y坐标标签
                if abs(y) > 0.1:  # 避免在原点重复标注
                    label = f"{int(y)}"
                    cv2.putText(annotated_map, label, (5, pixel_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_color, 1)
        y += grid_interval / 2
    
    # 绘制原点
    origin_x = int((0 - min_x) * pixels_per_unit)
    origin_y = int((0 - min_y) * pixels_per_unit)
    
    if 0 <= origin_x < output_width and 0 <= origin_y < output_height:
        cv2.circle(annotated_map, (origin_x, origin_y), 5, origin_color, -1)
        cv2.putText(annotated_map, "O(0,0)", (origin_x + 8, origin_y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, origin_color, 1)
    
    # 标记A4纸的四个角
    for i, (world_x, world_y) in enumerate([(0, 0), (21, 0), (21, 29.7), (0, 29.7)]):
        pixel_x = int((world_x - min_x) * pixels_per_unit)
        pixel_y = int((world_y - min_y) * pixels_per_unit)
        
        if 0 <= pixel_x < output_width and 0 <= pixel_y < output_height:
            cv2.circle(annotated_map, (pixel_x, pixel_y), 3, (0, 255, 255), -1)
            cv2.putText(annotated_map, f"A4-{i+1}", (pixel_x + 5, pixel_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    
    return annotated_map

# ---------------------------------------------------------------------------------
# --- 🛣️ 路径规划模块 ---
# ---------------------------------------------------------------------------------

class PathPlanner:
    """从鸟瞰图分割掩码中提取和规划可行驶路径"""
    
    def __init__(self, view_params):
        """
        初始化路径规划器
        
        参数：
            view_params: 鸟瞰图视图参数
        """
        self.view_params = view_params
        self.pixels_per_unit = view_params['pixels_per_unit']
        self.view_bounds = view_params['view_bounds']
        
    def extract_centerline(self, bird_eye_mask, scan_from_bottom=True, min_width=10):
        """
        从鸟瞰图分割掩码中提取中心线
        
        参数：
            bird_eye_mask: 鸟瞰图分割掩码
            scan_from_bottom: 是否从图像底部开始扫描
            min_width: 最小可行驶宽度（像素）
        
        返回：
            centerline_points: 中心线点列表 [(x, y), ...]（像素坐标）
            centerline_world: 中心线点列表（世界坐标厘米）
        """
        height, width = bird_eye_mask.shape
        centerline_points = []
        
        # 确定扫描方向
        rows = range(height-1, -1, -1) if scan_from_bottom else range(height)
        
        for y in rows:
            row = bird_eye_mask[y, :]
            
            # 找到该行所有可行驶区域的连续段
            segments = self._find_drivable_segments(row, min_width)
            
            if segments:
                # 选择最大的连续段（通常是主路）
                largest_segment = max(segments, key=lambda s: s[1] - s[0])
                
                # 计算该段的中心点
                center_x = (largest_segment[0] + largest_segment[1]) // 2
                centerline_points.append((center_x, y))
        
        # 转换为世界坐标
        centerline_world = self._pixels_to_world(centerline_points)
        
        return centerline_points, centerline_world
    
    def extract_centerline_fast(self, bird_eye_mask, scan_from_bottom=True, min_width=5, skip_rows=5):
        """
        快速中心线提取（边缘计算优化版本）
        
        参数：
            bird_eye_mask: 鸟瞰图分割掩码
            scan_from_bottom: 是否从图像底部开始扫描
            min_width: 最小可行驶宽度（像素）
            skip_rows: 跳过行数（减少计算量）
        
        返回：
            centerline_points: 中心线点列表 [(x, y), ...]（像素坐标）
            centerline_world: 中心线点列表（世界坐标厘米）
        """
        height, width = bird_eye_mask.shape
        centerline_points = []
        
        # 确定扫描方向，跳行扫描以提高速度
        if scan_from_bottom:
            rows = range(height-1, -1, -skip_rows)
        else:
            rows = range(0, height, skip_rows)
        
        for y in rows:
            row = bird_eye_mask[y, :]
            
            # 快速找到中心点：使用重心法
            drivable_indices = np.where(row > 0)[0]
            
            if len(drivable_indices) >= min_width:
                # 计算重心作为中心点
                center_x = int(np.mean(drivable_indices))
                centerline_points.append((center_x, y))
        
        # 转换为世界坐标
        centerline_world = self._pixels_to_world(centerline_points)
        
        return centerline_points, centerline_world
    
    def _find_drivable_segments(self, row, min_width):
        """
        在一行中找到所有可行驶区域的连续段
        
        参数：
            row: 图像行数据
            min_width: 最小宽度
        
        返回：
            segments: 连续段列表 [(start, end), ...]
        """
        segments = []
        start = None
        
        for i, pixel in enumerate(row):
            if pixel > 0:  # 可行驶区域
                if start is None:
                    start = i
            else:  # 不可行驶区域
                if start is not None:
                    if i - start >= min_width:  # 满足最小宽度要求
                        segments.append((start, i))
                    start = None
        
        # 处理行末尾的情况
        if start is not None and len(row) - start >= min_width:
            segments.append((start, len(row)))
        
        return segments
    
    def _pixels_to_world(self, pixel_points):
        """
        将像素坐标转换为世界坐标
        
        参数：
            pixel_points: 像素坐标点列表 [(x, y), ...]
        
        返回：
            world_points: 世界坐标点列表 [(x, y), ...]（单位：厘米）
        """
        min_x, min_y, max_x, max_y = self.view_bounds
        world_points = []
        
        for px, py in pixel_points:
            # 像素坐标转世界坐标
            world_x = min_x + (px / self.pixels_per_unit)
            world_y = min_y + (py / self.pixels_per_unit)
            world_points.append((world_x, world_y))
        
        return world_points
    
    def smooth_path(self, centerline_world, method='polynomial', degree=3, force_bottom_center=True):
        """
        对中心线路径进行平滑处理 (已修正为拟合 x=f(y) 并使用权重)
        
        参数：
            centerline_world: 世界坐标中心线点列表
            method: 平滑方法 ('polynomial', 'spline')
            degree: 多项式阶数或样条阶数
            force_bottom_center: 是否强制曲线过底边中点
        
        返回：
            smooth_path_func: 平滑路径函数 x = f(y)
            fit_params: 拟合参数
        """
        if not centerline_world or not SCIPY_AVAILABLE:
            return None, None
        
        points = np.array(centerline_world)
        # 核心修正1: 我们将Y作为自变量，X作为因变量
        y_coords = points[:, 1]  # 前进方向
        x_coords = points[:, 0]  # 左右偏移

        # 按Y坐标（前进方向）排序
        sorted_indices = np.argsort(y_coords)
        y_sorted = y_coords[sorted_indices]
        x_sorted = x_coords[sorted_indices]
        
        # 用于存储最终拟合点
        final_y = y_sorted
        final_x = x_sorted
        weights = np.ones_like(final_y) # 默认权重为1
        
        # 如果需要强制过底边中点
        if force_bottom_center:
            bottom_center = self._get_bottom_center_world_coord()
            
            if bottom_center is not None:
                # 将底边中点添加到拟合点中
                # 注意：bottom_center是 (x, y) 格式
                final_y = np.append(final_y, bottom_center[1])
                final_x = np.append(final_x, bottom_center[0])
                
                # 核心修正2: 为这个点设置一个极大的权重
                weights = np.append(weights, 1e6) # 给新点一个巨大的权重
                
                # 重新排序
                sorted_indices = np.argsort(final_y)
                final_y = final_y[sorted_indices]
                final_x = final_x[sorted_indices]
                weights = weights[sorted_indices]
                
                print(f"🎯 强制拟合曲线过底边中点: ({bottom_center[0]:.1f}, {bottom_center[1]:.1f}) cm，权重: {1e6}")

        # 确保点数足够拟合
        if len(final_y) <= degree:
            print(f"⚠️ 拟合点数 ({len(final_y)}) 不足，无法进行 {degree} 阶拟合。")
            return None, None

        if method == 'polynomial':
            # 核心修正3: 拟合 x = f(y)，并传入权重
            fit_params = np.polyfit(final_y, final_x, degree, w=weights)
            smooth_path_func = np.poly1d(fit_params)
            
        elif method == 'spline':
            # 样条插值默认会穿过所有点，但这里为了统一，也使用多项式
            # 如果需要样条，也需要拟合 x=f(y)
            print("⚠️ 样条方法暂不支持权重，强制使用多项式拟合以确保过中点。")
            fit_params = np.polyfit(final_y, final_x, degree, w=weights)
            smooth_path_func = np.poly1d(fit_params)
        
        return smooth_path_func, fit_params
    
    def _get_bottom_center_world_coord(self):
        """
        获取图像底边中点的世界坐标
        
        返回：
            bottom_center: (x, y) 底边中点的世界坐标，单位厘米
        """
        try:
            # 使用正确的"图像坐标->世界坐标"变换矩阵
            if 'image_to_world_matrix' in self.view_params:
                transform_matrix = np.array(self.view_params['image_to_world_matrix'], dtype=np.float32)
            else:
                # 如果没有，作为回退，从校正配置中获取
                print("⚠️ 在view_params中未找到image_to_world_matrix，尝试从内置校准获取。")
                transform_matrix = get_corrected_calibration()
            
            # 640×360图像底边中点的像素坐标
            img_bottom_center = np.array([320, 359, 1], dtype=np.float32)  # (320, 359) 是底边中点
            
            # 投影到世界坐标
            world_pt = transform_matrix @ img_bottom_center
            world_x = world_pt[0] / world_pt[2]
            world_y = world_pt[1] / world_pt[2]
            
            return (world_x, world_y)
            
        except Exception as e:
            print(f"⚠️ 无法计算底边中点世界坐标: {e}")
            return None
    
    def generate_waypoints(self, smooth_path_func, num_points=20, y_range=None):
        """
        从平滑路径生成路径点 (已修正为基于 y 轴生成)
        
        参数：
            smooth_path_func: 平滑路径函数 x = f(y)
            num_points: 路径点数量
            y_range: Y坐标范围 (min_y, max_y)，如果为None则使用视图边界
        
        返回：
            waypoints: 路径点列表 [(x, y), ...]（世界坐标，厘米）
        """
        if smooth_path_func is None:
            return []
        
        # 核心修正: 我们应该在y轴（前进方向）上取点
        if y_range is None:
            min_x, min_y, max_x, max_y = self.view_bounds
        else:
            min_y, max_y = y_range

        # 生成均匀分布的y坐标
        y_waypoints = np.linspace(min_y, max_y, num_points)
        
        # 计算对应的x坐标
        x_waypoints = smooth_path_func(y_waypoints)
        
        # 组合成路径点 (x, y)
        waypoints = list(zip(x_waypoints, y_waypoints))
        
        return waypoints
    
    def plan_complete_path(self, bird_eye_mask, smooth_method='polynomial', degree=3, 
                          num_waypoints=20, min_width=10, fast_mode=True, force_bottom_center=True):
        """
        完整的路径规划流程
        
        参数：
            bird_eye_mask: 鸟瞰图分割掩码
            smooth_method: 平滑方法
            degree: 拟合阶数
            num_waypoints: 路径点数量
            min_width: 最小可行驶宽度
            fast_mode: 是否使用快速模式（边缘计算优化）
            force_bottom_center: 是否强制曲线过底边中点
        
        返回：
            path_data: 包含所有路径信息的字典
        """
        # 第一步：提取中心线（选择快速或精确模式）
        if fast_mode:
            centerline_pixels, centerline_world = self.extract_centerline_fast(
                bird_eye_mask, min_width=min_width//2, skip_rows=3)  # 降低要求，跳行扫描
        else:
            centerline_pixels, centerline_world = self.extract_centerline(
                bird_eye_mask, min_width=min_width)
        
        if not centerline_world:
            return {
                'centerline_pixels': [],
                'centerline_world': [],
                'smooth_path_func': None,
                'fit_params': None,
                'waypoints': [],
                'path_length': 0
            }
        
        # 第二步：路径平滑（边缘计算模式下降低阶数，强制过底边中点）
        if fast_mode:
            smooth_degree = min(2, degree)  # 最高2阶，减少计算量
        else:
            smooth_degree = degree
            
        smooth_path_func, fit_params = self.smooth_path(
            centerline_world, method=smooth_method, degree=smooth_degree, 
            force_bottom_center=force_bottom_center)
        
        # 第三步：生成路径点
        waypoints = self.generate_waypoints(smooth_path_func, num_waypoints)
        
        # 计算路径长度
        path_length = self._calculate_path_length(waypoints) if waypoints else 0
        
        return {
            'centerline_pixels': centerline_pixels,
            'centerline_world': centerline_world,
            'smooth_path_func': smooth_path_func,
            'fit_params': fit_params,
            'waypoints': waypoints,
            'path_length': path_length,
            'num_centerline_points': len(centerline_world),
            'num_waypoints': len(waypoints),
            'fast_mode': fast_mode,
            'force_bottom_center': force_bottom_center
        }
    
    def _calculate_path_length(self, waypoints):
        """计算路径总长度"""
        if len(waypoints) < 2:
            return 0
        
        total_length = 0
        for i in range(1, len(waypoints)):
            dx = waypoints[i][0] - waypoints[i-1][0]
            dy = waypoints[i][1] - waypoints[i-1][1]
            total_length += np.sqrt(dx*dx + dy*dy)
        
        return total_length

# ---------------------------------------------------------------------------------
# --- 🎨 路径可视化模块 ---
# ---------------------------------------------------------------------------------

def visualize_path_on_control_map(control_map, path_data, view_params):
    """
    在控制地图上可视化路径规划结果
    
    参数：
        control_map: 控制地图
        path_data: 路径数据
        view_params: 视图参数
    
    返回：
        annotated_map: 带路径标注的控制地图
    """
    annotated_map = control_map.copy()
    
    if not path_data['centerline_pixels']:
        return annotated_map
    
    # 绘制原始中心线点（红色小圆点）
    for px, py in path_data['centerline_pixels']:
        cv2.circle(annotated_map, (int(px), int(py)), 2, (0, 0, 255), -1)
    
    # 绘制平滑路径（蓝色线条）
    if path_data['smooth_path_func'] is not None and path_data['waypoints']:
        waypoints_pixels = world_to_pixels(path_data['waypoints'], view_params)
        
        for i in range(len(waypoints_pixels) - 1):
            pt1 = (int(waypoints_pixels[i][0]), int(waypoints_pixels[i][1]))
            pt2 = (int(waypoints_pixels[i+1][0]), int(waypoints_pixels[i+1][1]))
            cv2.line(annotated_map, pt1, pt2, (255, 0, 0), 3)  # 蓝色粗线
    
    # 绘制路径点（黄色方块）
    if path_data['waypoints']:
        waypoints_pixels = world_to_pixels(path_data['waypoints'], view_params)
        for i, (px, py) in enumerate(waypoints_pixels):
            cv2.rectangle(annotated_map, 
                         (int(px-3), int(py-3)), (int(px+3), int(py+3)), 
                         (0, 255, 255), -1)  # 黄色方块
            
            # 标注路径点编号
            if i % 3 == 0:  # 每3个点标注一次，避免过于密集
                cv2.putText(annotated_map, f"{i}", (int(px+5), int(py-5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    
    return annotated_map

def world_to_pixels(world_points, view_params):
    """
    将世界坐标转换为像素坐标
    
    参数：
        world_points: 世界坐标点列表
        view_params: 视图参数
    
    返回：
        pixel_points: 像素坐标点列表
    """
    min_x, min_y, max_x, max_y = view_params['view_bounds']
    pixels_per_unit = view_params['pixels_per_unit']
    
    pixel_points = []
    for world_x, world_y in world_points:
        pixel_x = (world_x - min_x) * pixels_per_unit
        pixel_y = (world_y - min_y) * pixels_per_unit
        pixel_points.append((pixel_x, pixel_y))
    
    return pixel_points

def save_path_data_json(path_data, json_path):
    """
    将路径数据保存为JSON文件
    
    参数：
        path_data: 路径数据字典
        json_path: JSON文件路径
    """
    import json
    
    # 准备可序列化的数据
    json_data = {
        'centerline_world': path_data['centerline_world'],
        'waypoints': path_data['waypoints'],
        'path_length': path_data['path_length'],
        'num_centerline_points': path_data['num_centerline_points'],
        'num_waypoints': path_data['num_waypoints'],
        'fit_params': path_data['fit_params'].tolist() if path_data['fit_params'] is not None else None,
        'description': '车道中心线和路径点数据（世界坐标，单位：厘米）',
        'coordinate_system': 'world coordinates (cm)',
        'waypoints_description': '路径点，可直接用于车辆控制'
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False) 