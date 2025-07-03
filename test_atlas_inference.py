#!/usr/bin/env python3
"""
单张图片推理测试脚本 (模拟Atlas NPU推理流程)

此脚本用于在开发环境中测试推理流程，无需真实的Atlas硬件。
模拟了atlas_single_image_inference.py的完整流程。
"""

import os
import sys
import cv2
import time
import numpy as np
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------------
# --- 🚀🚀🚀 完美匹配的预处理 (与Atlas脚本完全一致) 🚀🚀🚀 ---
# ---------------------------------------------------------------------------------

def preprocess_matched_resolution(img_bgr, target_width=640, target_height=360, dtype=np.float16):
    """
    图片预处理，与atlas_single_image_inference.py完全一致
    """
    # 1. 调整尺寸到模型输入要求
    height, width = img_bgr.shape[:2]
    if width != target_width or height != target_height:
        print(f"📏 Resize: {width}×{height} → {target_width}×{target_height}")
        img_bgr = cv2.resize(img_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    else:
        print(f"🎯 完美匹配: {width}×{height} = {target_width}×{target_height}，无需resize!")
    
    # 2. 转换颜色通道 (BGR → RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 3. 转换数据类型 (uint8 → float16，保持[0-255]范围)
    img_typed = img_rgb.astype(dtype)
    
    # 4. 转换为CHW格式并添加batch维度 (H,W,C) → (1,C,H,W)
    img_transposed = np.transpose(img_typed, (2, 0, 1))
    return np.ascontiguousarray(img_transposed[np.newaxis, :, :, :])

# ---------------------------------------------------------------------------------
# --- 🚀🚀🚀 极简后处理 (与Atlas脚本完全一致) 🚀🚀🚀 ---
# ---------------------------------------------------------------------------------

def postprocess_matched_resolution(output_tensor, original_width, original_height):
    """
    后处理，与atlas_single_image_inference.py完全一致
    """
    # 1. Argmax获取分割掩码 (1, num_classes, H, W) → (H, W)
    pred_mask = np.argmax(output_tensor, axis=1).squeeze()
    
    # 2. 转换为可视化格式 (0/1 → 0/255)
    vis_mask = (pred_mask * 255).astype(np.uint8)
    
    # 3. 如果需要，resize回原始尺寸
    model_height, model_width = vis_mask.shape
    if original_width != model_width or original_height != model_height:
        print(f"📐 Resize back: {model_width}×{model_height} → {original_width}×{original_height}")
        vis_mask = cv2.resize(vis_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    else:
        print(f"🎯 输出尺寸匹配: {model_width}×{model_height} = {original_width}×{original_height}")
    
    return vis_mask

# ---------------------------------------------------------------------------------
# --- 🎨 可视化生成 (与Atlas脚本完全一致) ---
# ---------------------------------------------------------------------------------

def create_visualization(original_img, mask, alpha=0.5):
    """创建车道线分割可视化图像"""
    # 创建绿色覆盖层
    green_overlay = np.zeros_like(original_img, dtype=np.uint8)
    green_overlay[mask > 0] = [0, 255, 0]  # BGR格式的绿色
    
    # 融合原图和覆盖层
    vis_img = cv2.addWeighted(original_img, 1.0, green_overlay, alpha, 0)
    
    return vis_img

# ---------------------------------------------------------------------------------
# --- 🤖 模拟Atlas NPU推理 ---
# ---------------------------------------------------------------------------------

class MockInferSession:
    """模拟Atlas InferSession，用于测试"""
    
    def __init__(self, device_id, model_path):
        self.device_id = device_id
        self.model_path = model_path
        print(f"🤖 模拟加载模型: {model_path} (设备: {device_id})")
        time.sleep(0.1)  # 模拟加载时间
    
    def infer(self, inputs):
        """模拟推理，生成随机的车道线分割结果"""
        input_tensor = inputs[0]
        batch, channels, height, width = input_tensor.shape
        
        # 模拟NPU推理时间
        time.sleep(0.001)  # 1ms模拟推理时间
        
        # 生成模拟的分割结果 (2个类别: 背景和车道线)
        # 创建一些简单的车道线模式
        output = np.zeros((batch, 2, height, width), dtype=np.float32)
        
        # 背景类别 (索引0)
        output[:, 0, :, :] = 0.8
        
        # 车道线类别 (索引1) - 创建两条垂直的车道线
        lane_width = 20
        left_lane_center = width // 3
        right_lane_center = 2 * width // 3
        
        # 左车道线
        output[:, 1, :, left_lane_center-lane_width//2:left_lane_center+lane_width//2] = 0.9
        # 右车道线
        output[:, 1, :, right_lane_center-lane_width//2:right_lane_center+lane_width//2] = 0.9
        
        # 背景区域对应降低
        output[:, 0, :, left_lane_center-lane_width//2:left_lane_center+lane_width//2] = 0.1
        output[:, 0, :, right_lane_center-lane_width//2:right_lane_center+lane_width//2] = 0.1
        
        return [output]

# ---------------------------------------------------------------------------------
# --- 📊 性能分析 (与Atlas脚本完全一致) ---
# ---------------------------------------------------------------------------------

def print_performance_analysis(times_dict, input_shape, model_path):
    """打印详细的性能分析报告"""
    total_time = sum(times_dict.values())
    
    print("\n" + "="*60)
    print("🤖 模拟Atlas NPU 单张图片推理性能分析")
    print("="*60)
    print(f"🧠 模型: {Path(model_path).name}")
    print(f"📏 输入尺寸: {input_shape[3]}×{input_shape[2]} (W×H)")
    print(f"🎯 数据类型: {str(input_shape).split('.')[-1].upper()}")
    print(f"⚠️  注意: 这是模拟推理，非真实Atlas性能")
    print("-"*60)
    
    for stage, time_ms in times_dict.items():
        percentage = (time_ms / total_time) * 100
        print(f"⏱️  {stage:12}: {time_ms:6.1f}ms ({percentage:5.1f}%)")
    
    print("-"*60)
    print(f"🏁 总耗时: {total_time:.1f}ms")
    print(f"⚡ 理论FPS: {1000/total_time:.1f}")
    print("="*60)

# ---------------------------------------------------------------------------------
# --- 📱 主推理函数 (与Atlas脚本流程完全一致) ---
# ---------------------------------------------------------------------------------

def test_inference_single_image(image_path, model_path="mock_model.om", device_id=0, save_visualization=True, save_mask=False):
    """
    测试单张图片推理流程（模拟Atlas）
    """
    # 验证输入文件
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"输入图片不存在: {image_path}")
    
    print(f"🖼️  加载图片: {image_path}")
    
    # 1. 加载图片
    load_start = time.time()
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    original_height, original_width = img_bgr.shape[:2]
    load_time = (time.time() - load_start) * 1000
    
    print(f"📏 原始尺寸: {original_width}×{original_height}")
    
    # 2. 加载模型 (模拟)
    print(f"🧠 模拟加载模型: {model_path}")
    model_start = time.time()
    model = MockInferSession(device_id, model_path)
    model_load_time = (time.time() - model_start) * 1000
    print(f"✅ 模型加载完成 ({model_load_time:.1f}ms)")
    
    # 3. 预处理
    print("🔄 开始预处理...")
    preprocess_start = time.time()
    input_data = preprocess_matched_resolution(img_bgr, dtype=np.float16)
    preprocess_time = (time.time() - preprocess_start) * 1000
    
    print(f"📊 输入张量形状: {input_data.shape}")
    print(f"📊 数据类型: {input_data.dtype}")
    
    # 4. NPU推理 (模拟)
    print("🤖 开始模拟NPU推理...")
    inference_start = time.time()
    outputs = model.infer([input_data])
    inference_time = (time.time() - inference_start) * 1000
    
    print(f"📊 输出张量形状: {outputs[0].shape}")
    
    # 5. 后处理
    print("🔄 开始后处理...")
    postprocess_start = time.time()
    lane_mask = postprocess_matched_resolution(outputs[0], original_width, original_height)
    postprocess_time = (time.time() - postprocess_start) * 1000
    
    # 6. 保存结果
    save_start = time.time()
    results = {}
    
    if save_mask:
        mask_path = image_path.replace('.', '_test_mask.')
        cv2.imwrite(mask_path, lane_mask)
        results['mask_path'] = mask_path
        print(f"💾 分割掩码已保存: {mask_path}")
    
    if save_visualization:
        vis_img = create_visualization(img_bgr, lane_mask)
        vis_path = image_path.replace('.', '_test_result.')
        cv2.imwrite(vis_path, vis_img)
        results['visualization_path'] = vis_path
        print(f"💾 可视化结果已保存: {vis_path}")
    
    save_time = (time.time() - save_start) * 1000
    
    # 7. 性能分析
    times_dict = {
        "图片加载": load_time,
        "模型加载": model_load_time,
        "CPU预处理": preprocess_time,
        "模拟推理": inference_time,
        "CPU后处理": postprocess_time,
        "结果保存": save_time
    }
    
    print_performance_analysis(times_dict, input_data.shape, model_path)
    
    # 8. 统计车道线像素
    lane_pixels = np.sum(lane_mask > 0)
    total_pixels = lane_mask.shape[0] * lane_mask.shape[1]
    lane_ratio = (lane_pixels / total_pixels) * 100
    
    print(f"\n📈 检测结果统计:")
    print(f"🛣️  车道线像素: {lane_pixels:,} / {total_pixels:,} ({lane_ratio:.2f}%)")
    
    results.update({
        'lane_pixels': lane_pixels,
        'total_pixels': total_pixels,
        'lane_ratio': lane_ratio,
        'performance': times_dict
    })
    
    return results

# ---------------------------------------------------------------------------------
# --- 📱 命令行接口 ---
# ---------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="测试Atlas NPU单张图片推理流程 (模拟)")
    parser.add_argument("--input", "-i", required=True, help="输入图片路径")
    parser.add_argument("--output", "-o", help="输出可视化图片路径（可选）")
    parser.add_argument("--save_mask", help="保存分割掩码路径（可选）")
    parser.add_argument("--model", "-m", 
                       default="mock_fast_scnn_model.om",
                       help="模拟模型路径")
    parser.add_argument("--device", "-d", type=int, default=0, help="模拟NPU设备ID")
    parser.add_argument("--no_vis", action="store_true", help="不保存可视化结果，仅推理")
    
    args = parser.parse_args()
    
    try:
        print("🤖 模拟Atlas NPU 单张图片车道线分割推理测试")
        print("=" * 50)
        print("⚠️  注意: 这是测试脚本，使用模拟推理结果")
        print("📝 用于验证预处理/后处理流程与真实Atlas脚本的一致性")
        print("=" * 50)
        
        # 自动确定输出路径
        save_visualization = not args.no_vis
        save_mask = bool(args.save_mask)
        
        if args.output and save_visualization:
            # 如果指定了输出路径，重命名原图以使用指定路径
            import shutil
            target_input = args.output.replace('_test_result.', '.')
            if target_input != args.input:
                shutil.copy2(args.input, target_input)
                args.input = target_input
        
        # 执行测试推理
        results = test_inference_single_image(
            image_path=args.input,
            model_path=args.model,
            device_id=args.device,
            save_visualization=save_visualization,
            save_mask=save_mask
        )
        
        # 如果指定了掩码保存路径，重命名
        if args.save_mask and 'mask_path' in results:
            import shutil
            shutil.move(results['mask_path'], args.save_mask)
            results['mask_path'] = args.save_mask
            print(f"💾 分割掩码已保存到指定路径: {args.save_mask}")
        
        # 如果指定了输出路径，重命名可视化结果
        if args.output and 'visualization_path' in results:
            import shutil
            shutil.move(results['visualization_path'], args.output)
            results['visualization_path'] = args.output
            print(f"💾 可视化结果已保存到指定路径: {args.output}")
        
        print("\n✅ 测试推理完成！")
        print("🔧 现在可以将相同的流程部署到真实的Atlas环境中")
        
        if 'visualization_path' in results:
            print(f"🎨 可视化结果: {results['visualization_path']}")
        if 'mask_path' in results:
            print(f"🎭 分割掩码: {results['mask_path']}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
