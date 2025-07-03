#!/usr/bin/env python3
"""
验证模型预测结果与真实标签的对比
"""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from models.fast_scnn import FastSCNN

def load_model(model_path, device):
    """加载训练好的模型"""
    model = FastSCNN(num_classes=2, aux=True)  # 必须与训练时保持一致
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, base_size=1024):
    """预处理图像 - 与训练时保持完全一致"""
    image = Image.open(image_path).convert('RGB')
    
    # 与训练时一致：调整到base_size尺寸
    image = image.resize((base_size, base_size), Image.BILINEAR)
    
    # 转换为tensor，与custom.py中_val_sync_transform保持一致
    img_array = np.array(image).astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(img_array.transpose((2, 0, 1))).unsqueeze(0)
    
    return input_tensor, image

def load_and_process_mask(mask_path, base_size=1024):
    """加载并处理mask - 与训练时保持完全一致"""
    mask = Image.open(mask_path)
    
    # 转换为二分类标签，与custom.py保持一致
    mask_array = np.array(mask)
    if len(mask_array.shape) == 3:
        # 彩色mask，转换为灰度
        mask_array = mask_array[:, :, 0]  # 取第一个通道
    
    # 黑白mask: 白色(255)为可驾驶区域(1)，黑色(0)为不可驾驶区域(0)
    # 使用阈值128来处理可能的抗锯齿边缘
    binary_mask = (mask_array > 128).astype(np.uint8)
    
    # 调整到与训练时相同的尺寸
    mask_pil = Image.fromarray(binary_mask)
    mask_pil = mask_pil.resize((base_size, base_size), Image.NEAREST)
    binary_mask = np.array(mask_pil)
    
    return binary_mask

def compare_prediction_with_label(image_path, mask_path, model, device, base_size=1024):
    """比较预测结果与真实标签"""
    # 预处理图像 - 与训练时完全一致
    input_tensor, original_image = preprocess_image(image_path, base_size)
    input_tensor = input_tensor.to(device)
    
    # 模型预测
    with torch.no_grad():
        outputs = model(input_tensor)
        if isinstance(outputs, tuple):
            prediction = outputs[0]  # 主输出
        else:
            prediction = outputs
        
        # 获取预测结果
        prediction = F.softmax(prediction, dim=1)
        predicted_mask = torch.argmax(prediction, dim=1)
        predicted_mask = predicted_mask.cpu().numpy()[0]
    
    # 加载并处理真实mask - 与训练时完全一致
    binary_true_mask = load_and_process_mask(mask_path, base_size)
    
    # 现在两个mask都是相同尺寸的binary格式
    
    # 计算指标
    intersection = np.logical_and(predicted_mask, binary_true_mask)
    union = np.logical_or(predicted_mask, binary_true_mask)
    
    # IoU计算
    iou = np.sum(intersection) / (np.sum(union) + 1e-8)
    
    # 像素准确率
    correct_pixels = np.sum(predicted_mask == binary_true_mask)
    total_pixels = predicted_mask.shape[0] * predicted_mask.shape[1]
    pixel_accuracy = correct_pixels / total_pixels
    
    # 可驾驶区域比例
    pred_drivable_ratio = np.sum(predicted_mask) / total_pixels
    true_drivable_ratio = np.sum(binary_true_mask) / total_pixels
    
    return {
        'predicted_mask': predicted_mask,
        'true_mask': binary_true_mask,
        'iou': iou,
        'pixel_accuracy': pixel_accuracy,
        'pred_drivable_ratio': pred_drivable_ratio,
        'true_drivable_ratio': true_drivable_ratio,
        'original_image': original_image
    }

def visualize_comparison(result, save_path):
    """可视化对比结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始图像
    axes[0, 0].imshow(result['original_image'])
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 真实标签
    axes[0, 1].imshow(result['true_mask'], cmap='gray')
    axes[0, 1].set_title(f'True Label (Drivable: {result["true_drivable_ratio"]*100:.1f}%)')
    axes[0, 1].axis('off')
    
    # 预测结果
    axes[1, 0].imshow(result['predicted_mask'], cmap='gray')
    axes[1, 0].set_title(f'Prediction (Drivable: {result["pred_drivable_ratio"]*100:.1f}%)')
    axes[1, 0].axis('off')
    
    # 差异图
    diff = np.abs(result['predicted_mask'].astype(float) - result['true_mask'].astype(float))
    axes[1, 1].imshow(diff, cmap='Reds')
    axes[1, 1].set_title(f'Difference (IoU: {result["iou"]:.3f}, Acc: {result["pixel_accuracy"]:.3f})')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './weights/custom_scratch/fast_scnn_custom.pth'
    
    print("=" * 60)
    print("验证训练模型的预测效果")
    print("=" * 60)
    
    # 加载模型
    model = load_model(model_path, device)
    print(f"模型加载成功: {model_path}")
    
    # 测试几个样本
    data_dir = 'data/custom'
    images_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'masks')
    
    # 获取所有样本进行测试
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    print(f"找到 {len(image_files)} 个图像文件")
    
    results_dir = 'model_validation_results'
    os.makedirs(results_dir, exist_ok=True)
    
    total_iou = 0
    total_acc = 0
    count = 0
    
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        mask_file = os.path.splitext(image_file)[0] + '.png'
        mask_path = os.path.join(masks_dir, mask_file)
        
        if not os.path.exists(mask_path):
            print(f"跳过 {image_file}: 找不到对应的mask")
            continue
        
        print(f"\n处理: {image_file}")
        
        # 比较预测结果
        result = compare_prediction_with_label(image_path, mask_path, model, device, base_size=1024)
        
        print(f"  IoU: {result['iou']:.3f}")
        print(f"  像素准确率: {result['pixel_accuracy']:.3f}")
        print(f"  真实可驾驶区域: {result['true_drivable_ratio']*100:.1f}%")
        print(f"  预测可驾驶区域: {result['pred_drivable_ratio']*100:.1f}%")
        
        # 保存可视化结果
        save_path = os.path.join(results_dir, f"{os.path.splitext(image_file)[0]}_comparison.png")
        visualize_comparison(result, save_path)
        print(f"  可视化保存到: {save_path}")
        
        total_iou += result['iou']
        total_acc += result['pixel_accuracy']
        count += 1
    
    if count > 0:
        avg_iou = total_iou / count
        avg_acc = total_acc / count
        
        print(f"\n" + "=" * 60)
        print(f"平均结果 (测试了 {count} 个样本):")
        print(f"  平均IoU: {avg_iou:.3f}")
        print(f"  平均像素准确率: {avg_acc:.3f}")
        print(f"结果保存在: {results_dir}")
        print("=" * 60)

if __name__ == "__main__":
    main()
