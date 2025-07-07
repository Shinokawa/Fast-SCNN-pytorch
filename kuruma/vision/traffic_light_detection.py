#!/usr/bin/env python3
"""
交通灯检测模块 - 基于Atlas昇腾推理的红绿灯检测

功能特性：
- 使用Atlas NPU (华为昇腾) 进行交通灯检测推理
- 支持红灯、绿灯检测
- 为状态机提供交通灯状态信息
- 集成best.om模型推理逻辑

使用方法：
detector = create_traffic_light_detector(model_path, labels_path, device_id)
result = detector.detect_traffic_light(image)
"""

import cv2
import numpy as np
import torch
import os
import sys
import time
import logging
from typing import Dict, Optional, Tuple

# 导入统一日志配置
from core.logging_config import get_module_logger

# 🔧 修复：先定义logger
logger = get_module_logger(__name__)

# 导入Atlas会话管理器
from core.atlas_session_manager import get_atlas_session, create_tensor

class TrafficLightDetector:
    """
    交通灯检测器
    
    基于Atlas昇腾推理的红绿灯检测，集成了YOLOv5模型推理逻辑
    """
    
    def __init__(self, model_path: str, labels_path: str, device_id: int = 0):
        """
        初始化交通灯检测器
        
        Args:
            model_path: 模型文件路径 (best.om)
            labels_path: 标签文件路径 (labels.txt)
            device_id: Atlas设备ID
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.device_id = device_id
        
        # 设置日志
        self.logger = get_module_logger(__name__)
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"标签文件不存在: {labels_path}")
        
        # 获取Atlas会话管理器
        self.session_manager = get_atlas_session()
        
        # 初始化Atlas环境
        if not self.session_manager.initialize_atlas(device_id):
            raise RuntimeError("Atlas环境初始化失败")
        
        # 生成模型名称
        model_name = f"traffic_light_detector_{device_id}"
        
        # 加载交通灯检测模型
        self.model = self.session_manager.load_model(model_name, model_path, device_id)
        if self.model is None:
            raise RuntimeError(f"交通灯检测模型加载失败: {model_path}")
        self.labels_dict = self._load_labels(labels_path)
        
        # 导入工具函数
        self._import_utils()
        
        self.logger.info(f"✅ 交通灯检测器初始化成功")
        self.logger.info(f"   模型: {model_path}")
        self.logger.info(f"   标签: {self.labels_dict}")
        self.logger.info(f"   设备: {device_id}")
    
    def _load_labels(self, labels_path: str) -> Dict[int, str]:
        """加载标签文件"""
        labels_dict = {}
        with open(labels_path, 'r', encoding='utf-8') as f:
            for cat_id, label in enumerate(f.readlines()):
                labels_dict[cat_id] = label.strip()
        return labels_dict
    
    def _import_utils(self):
        """导入工具函数"""
        try:
            # 导入本地工具函数
            from .traffic_light_utils import letterbox, scale_coords, nms
            self.letterbox = letterbox
            self.scale_coords = scale_coords
            self.nms = nms
            self.logger.info("✅ 交通灯检测工具函数导入成功")
            
        except ImportError as e:
            self.logger.error(f"❌ 导入交通灯检测工具函数失败: {e}")
            raise
    
    def detect_traffic_light(self, image: np.ndarray, conf_thres: float = 0.4, iou_thres: float = 0.5) -> Dict:
        """
        检测图片中的交通灯
        
        Args:
            image: 输入图片 (BGR格式)
            conf_thres: 置信度阈值
            iou_thres: IoU阈值
            
        Returns:
            Dict: 检测结果
            {
                'detected': bool,           # 是否检测到交通灯
                'status': str,              # 交通灯状态 ('red', 'green', 'unknown')
                'confidence': float,        # 最高置信度
                'detections': List[Dict],   # 所有检测结果
                'inference_time': float     # 推理时间
            }
        """
        start_time = time.time()
        
        try:
            # 数据预处理
            img, scale_ratio, pad_size = self.letterbox(image, new_shape=[640, 640])
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
            img = np.expand_dims(img, 0).astype(np.float32)  # (1, 3, 640, 640)
            img = np.ascontiguousarray(img) / 255.0  # 归一化
            img_tensor = create_tensor(img)
            if img_tensor is None:
                raise RuntimeError("创建输入Tensor失败")
            
            # 模型推理
            inference_start = time.time()
            outputs = self.model.infer([img_tensor])
            output = outputs[0]  # 已经是numpy数组
            inference_time = time.time() - inference_start
            
            # 后处理 - 直接使用numpy数组
            boxout = self.nms(torch.tensor(output), conf_thres=conf_thres, iou_thres=iou_thres)
            
            if len(boxout[0]) == 0:
                # 未检测到任何目标
                return {
                    'detected': False,
                    'status': 'unknown',
                    'confidence': 0.0,
                    'detections': [],
                    'inference_time': inference_time,
                    'total_time': time.time() - start_time
                }
            
            pred_all = boxout[0].numpy()
            self.scale_coords([640, 640], pred_all[:, :4], image.shape, ratio_pad=(scale_ratio, pad_size))
            
            # 分析检测结果
            detections = []
            red_confidence = 0.0
            green_confidence = 0.0
            
            for detection in pred_all:
                class_id = int(detection[5])
                confidence = detection[4]
                bbox = detection[:4]
                
                detection_info = {
                    'class_id': class_id,
                    'class_name': self.labels_dict.get(class_id, 'unknown'),
                    'confidence': confidence,
                    'bbox': bbox.tolist()
                }
                detections.append(detection_info)
                
                # 更新红绿灯置信度
                if class_id == 1:  # red类别
                    red_confidence = max(red_confidence, confidence)
                elif class_id == 2:  # green类别
                    green_confidence = max(green_confidence, confidence)
            
            # 判断交通灯状态
            if red_confidence > 0.5:
                status = "red"
                max_confidence = red_confidence
            elif green_confidence > 0.5:
                status = "green"
                max_confidence = green_confidence
            else:
                status = "unknown"
                max_confidence = max(red_confidence, green_confidence)
            
            total_time = time.time() - start_time
            
            return {
                'detected': len(detections) > 0,
                'status': status,
                'confidence': max_confidence,
                'red_confidence': red_confidence,
                'green_confidence': green_confidence,
                'detections': detections,
                'inference_time': inference_time,
                'total_time': total_time
            }
            
        except Exception as e:
            self.logger.error(f"❌ 交通灯检测错误: {e}")
            return {
                'detected': False,
                'status': 'unknown',
                'confidence': 0.0,
                'detections': [],
                'inference_time': 0.0,
                'total_time': time.time() - start_time,
                'error': str(e)
            }
    
    def get_driving_decision(self, traffic_light_status: str) -> str:
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

def create_traffic_light_detector(model_path: str = None, labels_path: str = None, device_id: int = 0) -> Optional[TrafficLightDetector]:
    """
    创建交通灯检测器工厂函数
    
    Args:
        model_path: 模型文件路径，如果为None则使用默认路径
        labels_path: 标签文件路径，如果为None则使用默认路径
        device_id: Atlas设备ID
        
    Returns:
        TrafficLightDetector实例或None（如果创建失败）
    """
    logger = get_module_logger(__name__)
    
    try:
        # 确定默认路径
        if model_path is None or labels_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            
            # 首先尝试car目录（用户当前配置）
            car_dir = os.path.join(project_root, 'car')
            atlasyolo_dir = os.path.join(project_root, 'atlasyolo')
            
            if model_path is None:
                # 优先检查car目录，然后是atlasyolo目录
                if os.path.exists(os.path.join(car_dir, 'best.om')):
                    model_path = os.path.join(car_dir, 'best.om')
                else:
                    model_path = os.path.join(atlasyolo_dir, 'best.om')
            
            if labels_path is None:
                # 优先检查car目录，然后是atlasyolo目录
                if os.path.exists(os.path.join(car_dir, 'labels.txt')):
                    labels_path = os.path.join(car_dir, 'labels.txt')
                else:
                    labels_path = os.path.join(atlasyolo_dir, 'labels.txt')
        
        # 创建检测器
        detector = TrafficLightDetector(model_path, labels_path, device_id)
        logger.info("🚦 交通灯检测器创建成功")
        return detector
        
    except Exception as e:
        logger.error(f"❌ 创建交通灯检测器失败: {e}")
        return None

def test_traffic_light_detector():
    """测试交通灯检测器"""
    logger = get_module_logger(__name__)
    
    # 创建检测器
    detector = create_traffic_light_detector()
    if detector is None:
        logger.error("❌ 无法创建交通灯检测器")
        return
    
    # 测试图片路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    atlasyolo_dir = os.path.join(project_root, 'atlasyolo')
    test_image_path = os.path.join(atlasyolo_dir, 'data', 'green00004.jpg')
    
    if not os.path.exists(test_image_path):
        logger.warning(f"⚠️ 测试图片不存在: {test_image_path}")
        return
    
    # 读取测试图片
    image = cv2.imread(test_image_path)
    if image is None:
        logger.error(f"❌ 无法读取测试图片: {test_image_path}")
        return
    
    # 执行检测
    logger.info(f"🔍 开始检测测试图片: {os.path.basename(test_image_path)}")
    result = detector.detect_traffic_light(image)
    
    # 输出结果
    logger.info(f"检测结果:")
    logger.info(f"  检测到交通灯: {result['detected']}")
    logger.info(f"  交通灯状态: {result['status']}")
    logger.info(f"  置信度: {result['confidence']:.3f}")
    logger.info(f"  推理时间: {result['inference_time']:.3f}秒")
    logger.info(f"  总时间: {result['total_time']:.3f}秒")
    
    if result['detections']:
        logger.info(f"  检测详情:")
        for i, det in enumerate(result['detections']):
            logger.info(f"    {i+1}. {det['class_name']}: {det['confidence']:.3f}")

if __name__ == "__main__":
    test_traffic_light_detector() 