#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交通灯检测测试脚本
基于Atlas昇腾推理best.om模型，测试红绿灯检测功能
"""

import cv2
import numpy as np
import torch
import os
import random
import sys
import time
from datetime import datetime

# 添加atlasyolo路径到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'atlasyolo'))

try:
    from mindx.sdk import Tensor
    from mindx.sdk import base
    from my_utils import get_labels_from_txt, letterbox, scale_coords, nms, draw_bbox
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在昇腾环境中运行此脚本")
    sys.exit(1)

class TrafficLightDetector:
    """交通灯检测器"""
    
    def __init__(self, model_path, labels_path, device_id=0):
        """
        初始化检测器
        
        Args:
            model_path: 模型文件路径
            labels_path: 标签文件路径 
            device_id: 设备ID
        """
        self.device_id = device_id
        self.model_path = model_path
        self.labels_path = labels_path
        
        # 初始化昇腾资源
        base.mx_init()
        self.model = base.model(modelPath=model_path, deviceId=device_id)
        self.labels_dict = get_labels_from_txt(labels_path)
        
        print(f"模型加载成功: {model_path}")
        print(f"检测类别: {self.labels_dict}")
        
    def detect_image(self, image_path):
        """
        检测单张图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            tuple: (原图, 检测结果图, 检测数据, 交通灯状态)
        """
        # 读取图片
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"无法读取图片: {image_path}")
            return None, None, None, None
            
        original_img = img_bgr.copy()
        
        # 数据预处理
        img, scale_ratio, pad_size = letterbox(img_bgr, new_shape=[640, 640])
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.expand_dims(img, 0).astype(np.float32)  # (1, 3, 640, 640)
        img = np.ascontiguousarray(img) / 255.0  # 归一化
        img_tensor = Tensor(img)
        
        # 模型推理
        start_time = time.time()
        output = self.model.infer([img_tensor])[0]
        inference_time = time.time() - start_time
        
        # 后处理
        output.to_host()
        output = np.array(output)
        boxout = nms(torch.tensor(output), conf_thres=0.4, iou_thres=0.5)
        
        if len(boxout[0]) == 0:
            print(f"未检测到任何目标: {os.path.basename(image_path)}")
            return original_img, img_bgr, [], "unknown"
            
        pred_all = boxout[0].numpy()
        scale_coords([640, 640], pred_all[:, :4], img_bgr.shape, ratio_pad=(scale_ratio, pad_size))
        
        # 绘制检测框
        result_img = draw_bbox(pred_all, img_bgr, (0, 255, 0), 2, self.labels_dict)
        
        # 分析交通灯状态
        traffic_light_status = self.analyze_traffic_light(pred_all)
        
        detection_info = {
            'image_path': image_path,
            'inference_time': inference_time,
            'detections': pred_all,
            'traffic_light_status': traffic_light_status,
            'detection_count': len(pred_all)
        }
        
        return original_img, result_img, detection_info, traffic_light_status
        
    def analyze_traffic_light(self, detections):
        """
        分析检测结果中的交通灯状态
        
        Args:
            detections: 检测结果数组
            
        Returns:
            str: 交通灯状态 ('red', 'green', 'unknown')
        """
        red_confidence = 0
        green_confidence = 0
        
        for detection in detections:
            class_id = int(detection[5])
            confidence = detection[4]
            
            if class_id == 1:  # red类别
                red_confidence = max(red_confidence, confidence)
            elif class_id == 2:  # green类别
                green_confidence = max(green_confidence, confidence)
                
        # 判断交通灯状态
        if red_confidence > 0.5:
            return "red"
        elif green_confidence > 0.5:
            return "green"
        else:
            return "unknown"
            
    def get_driving_decision(self, traffic_light_status):
        """
        根据交通灯状态做出驾驶决策
        
        Args:
            traffic_light_status: 交通灯状态
            
        Returns:
            str: 驾驶决策
        """
        if traffic_light_status == "red":
            return "STOP - 红灯停车"
        elif traffic_light_status == "green":
            return "GO - 绿灯通行"
        else:
            return "CAUTION - 未检测到明确交通灯信号"

def select_random_images(data_dir, num_images=5):
    """
    从数据目录随机选择图片
    
    Args:
        data_dir: 数据目录路径
        num_images: 选择的图片数量
        
    Returns:
        list: 选中的图片路径列表
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_images = []
    
    for file in os.listdir(data_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            all_images.append(os.path.join(data_dir, file))
    
    if len(all_images) < num_images:
        print(f"数据目录中只有 {len(all_images)} 张图片，少于请求的 {num_images} 张")
        return all_images
    
    selected_images = random.sample(all_images, num_images)
    return selected_images

def main():
    """主函数"""
    print("=" * 60)
    print("Atlas昇腾交通灯检测测试")
    print("=" * 60)
    
    # 配置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    atlasyolo_dir = os.path.join(project_root, 'car')
    
    model_path = os.path.join(atlasyolo_dir, 'best.om')
    labels_path = os.path.join(atlasyolo_dir, 'labels.txt')
    data_dir = os.path.join(atlasyolo_dir, 'data')
    output_dir = os.path.join(current_dir, 'test', 'output')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return
    if not os.path.exists(labels_path):
        print(f"错误: 标签文件不存在 {labels_path}")
        return
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在 {data_dir}")
        return
    
    print(f"模型路径: {model_path}")
    print(f"标签路径: {labels_path}")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    
    # 随机选择5张图片
    print("\n正在随机选择测试图片...")
    selected_images = select_random_images(data_dir, 5)
    print(f"选中 {len(selected_images)} 张图片进行测试:")
    for i, img_path in enumerate(selected_images, 1):
        print(f"  {i}. {os.path.basename(img_path)}")
    
    # 初始化检测器
    print("\n初始化交通灯检测器...")
    try:
        detector = TrafficLightDetector(model_path, labels_path)
    except Exception as e:
        print(f"初始化检测器失败: {e}")
        return
    
    # 开始检测
    print("\n开始检测...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    
    for i, image_path in enumerate(selected_images, 1):
        print(f"\n--- 检测第 {i} 张图片 ---")
        print(f"图片: {os.path.basename(image_path)}")
        
        # 执行检测
        original_img, result_img, detection_info, traffic_light_status = detector.detect_image(image_path)
        
        if detection_info is None:
            continue
            
        # 获取驾驶决策
        driving_decision = detector.get_driving_decision(traffic_light_status)
        
        print(f"推理时间: {detection_info['inference_time']:.4f}秒")
        print(f"检测到目标数量: {detection_info['detection_count']}")
        print(f"交通灯状态: {traffic_light_status}")
        print(f"驾驶决策: {driving_decision}")
        
        # 保存结果
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{timestamp}_{base_name}_traffic_light_result.jpg")
        cv2.imwrite(output_path, result_img)
        print(f"结果已保存: {os.path.basename(output_path)}")
        
        # 在图片上添加交通灯状态文字
        status_img = result_img.copy()
        status_text = f"Status: {traffic_light_status.upper()}"
        decision_text = driving_decision
        
        # 添加状态文字背景
        (text_width, text_height), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(status_img, (10, 10), (10 + text_width + 10, 10 + text_height + 40), (0, 0, 0), -1)
        
        # 根据交通灯状态选择颜色
        if traffic_light_status == "red":
            color = (0, 0, 255)  # 红色
        elif traffic_light_status == "green":
            color = (0, 255, 0)  # 绿色
        else:
            color = (0, 255, 255)  # 黄色
            
        cv2.putText(status_img, status_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(status_img, decision_text, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 保存带状态的图片
        status_output_path = os.path.join(output_dir, f"{timestamp}_{base_name}_with_status.jpg")
        cv2.imwrite(status_output_path, status_img)
        
        results.append({
            'image': os.path.basename(image_path),
            'status': traffic_light_status,
            'decision': driving_decision,
            'inference_time': detection_info['inference_time'],
            'detection_count': detection_info['detection_count']
        })
    
    # 输出测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    red_count = sum(1 for r in results if r['status'] == 'red')
    green_count = sum(1 for r in results if r['status'] == 'green')
    unknown_count = sum(1 for r in results if r['status'] == 'unknown')
    avg_inference_time = sum(r['inference_time'] for r in results) / len(results) if results else 0
    
    print(f"总测试图片数: {len(results)}")
    print(f"检测到红灯: {red_count} 张")
    print(f"检测到绿灯: {green_count} 张")
    print(f"未明确识别: {unknown_count} 张")
    print(f"平均推理时间: {avg_inference_time:.4f}秒")
    print(f"结果保存目录: {output_dir}")
    
    # 详细结果
    print("\n详细结果:")
    for i, result in enumerate(results, 1):
        print(f"{i:2d}. {result['image']:<25} | {result['status']:<7} | {result['decision']:<20} | {result['inference_time']:.4f}s")
    
    print("\n测试完成！")

if __name__ == "__main__":
    main() 