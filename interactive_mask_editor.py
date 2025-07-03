#!/usr/bin/env python3
"""
交互式Mask编辑器
支持手动选择和填色两条白线之间的区域，提升mask标注精度

功能：
1. 加载图片和对应的mask
2. 多种交互模式：点击填充、矩形框选、多边形绘制、画笔涂抹
3. 实时预览编辑效果
4. 支持撤销/重做操作
5. 保存编辑后的mask

使用方法：
python interactive_mask_editor.py --image_dir data/custom/images --mask_dir data/custom/masks

快捷键：
- 鼠标左键：根据当前模式进行操作
- 鼠标右键：删除/擦除
- 'f': 切换到填充模式（点击自动填充连通区域）
- 'r': 切换到矩形框选模式
- 'p': 切换到多边形绘制模式
- 'b': 切换到画笔涂抹模式
- 'u': 撤销上一步操作
- 'Ctrl+z': 撤销
- 'Ctrl+y': 重做
- 's': 保存当前mask
- 'n': 下一张图片
- 'prev': 上一张图片
- 'q': 退出程序
- '+/-': 调整画笔大小
"""

import cv2
import numpy as np
import os
import argparse
import glob
from typing import List, Tuple, Optional
import json
from datetime import datetime

class InteractiveMaskEditor:
    def __init__(self, image_dir: str, mask_dir: str):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        # 获取图片列表
        self.image_files = self._get_image_files()
        self.current_index = 0
        
        # 编辑模式
        self.FILL_MODE = 0      # 填充模式
        self.RECT_MODE = 1      # 矩形框选模式
        self.POLYGON_MODE = 2   # 多边形绘制模式
        self.BRUSH_MODE = 3     # 画笔涂抹模式
        
        self.current_mode = self.FILL_MODE
        self.mode_names = ["填充", "矩形", "多边形", "画笔"]
        
        # 编辑状态
        self.original_image = None
        self.original_mask = None
        self.current_mask = None
        self.display_image = None
        
        # 操作历史（用于撤销/重做）
        self.history = []
        self.history_index = -1
        self.max_history = 20
        
        # 交互状态
        self.drawing = False
        self.start_point = None
        self.polygon_points = []
        self.brush_size = 10
        
        # 窗口设置
        self.window_name = "交互式Mask编辑器"
        self.info_height = 100
        
    def _get_image_files(self) -> List[str]:
        """获取图片文件列表"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(self.image_dir, ext)))
        # 去重并排序
        return sorted(list(set(files)))
    
    def _get_mask_path(self, image_path: str) -> str:
        """根据图片路径获取对应的mask路径"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(self.mask_dir, f"{base_name}.png")
        return mask_path
    
    def _load_current_image_and_mask(self):
        """加载当前图片和mask"""
        if not self.image_files:
            print("没有找到图片文件！")
            return False
            
        image_path = self.image_files[self.current_index]
        mask_path = self._get_mask_path(image_path)
        
        # 加载图片
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            print(f"无法加载图片: {image_path}")
            return False
        
        # 加载mask
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path)
            if mask is not None:
                # 如果是彩色mask，转换为单通道
                if len(mask.shape) == 3:
                    # 检测红色像素(BGR格式中的[0,0,255])作为可驾驶区域
                    red_pixels = (mask[:,:,2] > 128) & (mask[:,:,1] < 128) & (mask[:,:,0] < 128)
                    self.original_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
                    self.original_mask[red_pixels] = 255
                else:
                    self.original_mask = mask.copy()
            else:
                # 创建空白mask
                h, w = self.original_image.shape[:2]
                self.original_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            # 创建空白mask
            h, w = self.original_image.shape[:2]
            self.original_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 初始化当前mask
        self.current_mask = self.original_mask.copy()
        
        # 清空历史
        self.history = [self.current_mask.copy()]
        self.history_index = 0
        
        self._update_display()
        return True
    
    def _save_to_history(self):
        """保存当前状态到历史"""
        # 移除history_index之后的所有历史
        self.history = self.history[:self.history_index + 1]
        
        # 添加新状态
        self.history.append(self.current_mask.copy())
        self.history_index += 1
        
        # 限制历史长度
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.history_index -= 1
    
    def _undo(self):
        """撤销操作"""
        if self.history_index > 0:
            self.history_index -= 1
            self.current_mask = self.history[self.history_index].copy()
            self._update_display()
            print("撤销成功")
    
    def _redo(self):
        """重做操作"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.current_mask = self.history[self.history_index].copy()
            self._update_display()
            print("重做成功")
    
    def _update_display(self):
        """更新显示图像"""
        # 创建彩色mask覆盖图
        h, w = self.original_image.shape[:2]
        
        # 创建彩色mask
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        colored_mask[self.current_mask == 255] = [0, 0, 255]  # 红色表示可驾驶区域
        
        # 将mask叠加到原图上
        alpha = 0.4
        self.display_image = cv2.addWeighted(self.original_image, 1-alpha, colored_mask, alpha, 0)
        
        # 添加信息面板
        info_panel = np.zeros((self.info_height, w, 3), dtype=np.uint8)
        
        # 显示当前信息
        image_name = os.path.basename(self.image_files[self.current_index])
        info_text = [
            f"图片: {image_name} ({self.current_index + 1}/{len(self.image_files)})",
            f"模式: {self.mode_names[self.current_mode]}",
            f"画笔大小: {self.brush_size}",
            "操作: 左键填色, 右键擦除, f/r/p/b切换模式, s保存, n下一张, q退出"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(info_panel, text, (10, 20 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 合并显示图像和信息面板
        full_display = np.vstack([self.display_image, info_panel])
        cv2.imshow(self.window_name, full_display)
    
    def _flood_fill(self, x: int, y: int, fill_value: int = 255):
        """泛洪填充"""
        if x < 0 or y < 0 or x >= self.current_mask.shape[1] or y >= self.current_mask.shape[0]:
            return
        
        # 使用OpenCV的floodFill
        mask = np.zeros((self.current_mask.shape[0] + 2, self.current_mask.shape[1] + 2), np.uint8)
        cv2.floodFill(self.current_mask, mask, (x, y), fill_value)
    
    def _draw_rectangle(self, pt1: Tuple[int, int], pt2: Tuple[int, int], fill_value: int = 255):
        """绘制矩形"""
        cv2.rectangle(self.current_mask, pt1, pt2, fill_value, -1)
    
    def _draw_polygon(self, points: List[Tuple[int, int]], fill_value: int = 255):
        """绘制多边形"""
        if len(points) >= 3:
            points_array = np.array(points, np.int32)
            cv2.fillPoly(self.current_mask, [points_array], fill_value)
    
    def _draw_brush(self, x: int, y: int, fill_value: int = 255):
        """画笔绘制"""
        cv2.circle(self.current_mask, (x, y), self.brush_size, fill_value, -1)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        # 调整坐标（考虑信息面板）
        if y >= self.display_image.shape[0]:
            return
        
        fill_value = 255 if event != cv2.EVENT_RBUTTONDOWN and event != cv2.EVENT_RBUTTONUP else 0
        
        if self.current_mode == self.FILL_MODE:
            # 填充模式
            if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
                self._save_to_history()
                self._flood_fill(x, y, fill_value)
                self._update_display()
                
        elif self.current_mode == self.RECT_MODE:
            # 矩形模式
            if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
                self.drawing = True
                self.start_point = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                # 实时预览矩形
                temp_display = self.display_image.copy()
                color = (0, 255, 0) if fill_value == 255 else (0, 0, 255)
                cv2.rectangle(temp_display, self.start_point, (x, y), color, 2)
                
                # 添加信息面板
                h, w = self.original_image.shape[:2]
                info_panel = np.zeros((self.info_height, w, 3), dtype=np.uint8)
                cv2.putText(info_panel, f"矩形选择: ({self.start_point[0]},{self.start_point[1]}) -> ({x},{y})", 
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                full_display = np.vstack([temp_display, info_panel])
                cv2.imshow(self.window_name, full_display)
                
            elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
                if self.drawing:
                    self._save_to_history()
                    self._draw_rectangle(self.start_point, (x, y), fill_value)
                    self._update_display()
                    self.drawing = False
                    
        elif self.current_mode == self.POLYGON_MODE:
            # 多边形模式
            if event == cv2.EVENT_LBUTTONDOWN:
                self.polygon_points.append((x, y))
                print(f"添加点: ({x}, {y}), 当前点数: {len(self.polygon_points)}")
                
                # 显示当前多边形
                temp_display = self.display_image.copy()
                if len(self.polygon_points) > 1:
                    for i in range(len(self.polygon_points) - 1):
                        cv2.line(temp_display, self.polygon_points[i], self.polygon_points[i+1], (0, 255, 0), 2)
                for point in self.polygon_points:
                    cv2.circle(temp_display, point, 3, (0, 255, 0), -1)
                    
                # 添加信息面板
                h, w = self.original_image.shape[:2]
                info_panel = np.zeros((self.info_height, w, 3), dtype=np.uint8)
                cv2.putText(info_panel, f"多边形绘制: {len(self.polygon_points)}个点, 右键完成", 
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                full_display = np.vstack([temp_display, info_panel])
                cv2.imshow(self.window_name, full_display)
                
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(self.polygon_points) >= 3:
                    self._save_to_history()
                    self._draw_polygon(self.polygon_points, fill_value)
                    self._update_display()
                    print(f"完成多边形绘制，共{len(self.polygon_points)}个点")
                self.polygon_points = []
                
        elif self.current_mode == self.BRUSH_MODE:
            # 画笔模式
            if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
                self.drawing = True
                self._save_to_history()
                self._draw_brush(x, y, fill_value)
                self._update_display()
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self._draw_brush(x, y, fill_value)
                self._update_display()
            elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
                self.drawing = False
    
    def _save_current_mask(self):
        """保存当前mask"""
        if self.current_mask is None:
            return
            
        image_path = self.image_files[self.current_index]
        mask_path = self._get_mask_path(image_path)
        
        # 确保mask目录存在
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        
        # 创建标准的二分类mask：可驾驶区域为红色(255,0,0)，不可驾驶为黑色(0,0,0)
        color_mask = np.zeros((self.current_mask.shape[0], self.current_mask.shape[1], 3), dtype=np.uint8)
        color_mask[self.current_mask == 255] = [0, 0, 255]  # BGR格式：红色
        
        # 保存彩色mask
        cv2.imwrite(mask_path, color_mask)
        print(f"Mask已保存到: {mask_path}")
        
        # 保存编辑日志
        log_path = os.path.join(self.mask_dir, "edit_log.json")
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "image_file": os.path.basename(image_path),
            "mask_file": os.path.basename(mask_path),
            "action": "manual_edit"
        }
        
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
        
        print(f"已保存mask: {mask_path}")
    
    def _next_image(self):
        """下一张图片"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self._load_current_image_and_mask()
        else:
            print("已经是最后一张图片了")
    
    def _prev_image(self):
        """上一张图片"""
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current_image_and_mask()
        else:
            print("已经是第一张图片了")
    
    def run(self):
        """运行编辑器"""
        if not self.image_files:
            print("没有找到图片文件！")
            return
        
        print(f"找到 {len(self.image_files)} 张图片")
        print("交互式Mask编辑器启动中...")
        print("\n操作说明:")
        print("- 鼠标左键: 填色")
        print("- 鼠标右键: 擦除")
        print("- 'f': 填充模式（点击自动填充连通区域）")
        print("- 'r': 矩形框选模式")
        print("- 'p': 多边形绘制模式")
        print("- 'b': 画笔涂抹模式")
        print("- 'u' 或 Ctrl+Z: 撤销")
        print("- Ctrl+Y: 重做")
        print("- 's': 保存当前mask")
        print("- 'n': 下一张图片")
        print("- ',' 或 'prev': 上一张图片")
        print("- '+'/'-': 调整画笔大小")
        print("- 'q' 或 ESC: 退出程序")
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # 加载第一张图片
        if not self._load_current_image_and_mask():
            return
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # q或ESC退出
                break
            elif key == ord('f'):  # 填充模式
                self.current_mode = self.FILL_MODE
                self.polygon_points = []
                print("切换到填充模式")
                self._update_display()
            elif key == ord('r'):  # 矩形模式
                self.current_mode = self.RECT_MODE
                self.polygon_points = []
                print("切换到矩形模式")
                self._update_display()
            elif key == ord('p'):  # 多边形模式
                self.current_mode = self.POLYGON_MODE
                print("切换到多边形模式")
                self._update_display()
            elif key == ord('b'):  # 画笔模式
                self.current_mode = self.BRUSH_MODE
                self.polygon_points = []
                print("切换到画笔模式")
                self._update_display()
            elif key == ord('u') or key == 26:  # u或Ctrl+Z撤销
                self._undo()
            elif key == 25:  # Ctrl+Y重做
                self._redo()
            elif key == ord('s'):  # 保存
                self._save_current_mask()
            elif key == ord('n'):  # 下一张
                self._next_image()
            elif key == ord(','):  # 上一张
                self._prev_image()
            elif key == ord('+') or key == ord('='):  # 增大画笔
                self.brush_size = min(50, self.brush_size + 2)
                print(f"画笔大小: {self.brush_size}")
                self._update_display()
            elif key == ord('-'):  # 减小画笔
                self.brush_size = max(1, self.brush_size - 2)
                print(f"画笔大小: {self.brush_size}")
                self._update_display()
        
        cv2.destroyAllWindows()
        print("编辑器已退出")

def main():
    parser = argparse.ArgumentParser(description='交互式Mask编辑器')
    parser.add_argument('--image_dir', type=str, default='data/custom/images',
                        help='图片目录路径')
    parser.add_argument('--mask_dir', type=str, default='data/custom/masks',
                        help='Mask目录路径')
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.image_dir):
        print(f"图片目录不存在: {args.image_dir}")
        return
    
    # 创建mask目录（如果不存在）
    os.makedirs(args.mask_dir, exist_ok=True)
    
    # 启动编辑器
    editor = InteractiveMaskEditor(args.image_dir, args.mask_dir)
    editor.run()

if __name__ == "__main__":
    main()
