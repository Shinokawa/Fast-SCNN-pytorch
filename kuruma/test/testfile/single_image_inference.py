import os
import cv2
import time
import numpy as np
import argparse
from pathlib import Path

from ais_bench.infer.interface import InferSession

# --- 配置参数 ---
DEVICE_ID = 0
# 使用您的模型路径
MODEL_PATH = "./weights/fast_scnn_custom_e2e_360x640_fp16_fixed_simp.om"
# 模型输入尺寸：高×宽 = 360×640
MODEL_WIDTH = 640
MODEL_HEIGHT = 360

# ---------------------------------------------------------------------------------
# --- 🚀🚀🚀 完美匹配的预处理 (640×360) 🚀🚀🚀 ---
# ---------------------------------------------------------------------------------

def preprocess_image(img_bgr, target_width=MODEL_WIDTH, target_height=MODEL_HEIGHT, dtype=np.float16):
    """
    预处理输入图片到模型所需的尺寸和格式
    
    Args:
        img_bgr: BGR格式的输入图片
        target_width: 目标宽度
        target_height: 目标高度
        dtype: 数据类型
    
    Returns:
        processed_img: 预处理后的图片 (NCHW格式)
        original_shape: 原始图片形状 (height, width)
    """
    original_height, original_width = img_bgr.shape[:2]
    
    # 1. 如果尺寸不匹配，进行resize
    if original_width != target_width or original_height != target_height:
        img_resized = cv2.resize(img_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        print(f"📐 图片尺寸调整: {original_width}×{original_height} -> {target_width}×{target_height}")
    else:
        img_resized = img_bgr
        print(f"🎯 图片尺寸完美匹配: {original_width}×{original_height}")
    
    # 2. 转换颜色通道 (BGR -> RGB)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # 3. 转换数据类型 (uint8 -> float16，保持[0-255]范围)
    img_typed = img_rgb.astype(dtype)
    
    # 4. 转换为CHW格式并添加batch维度
    img_transposed = np.transpose(img_typed, (2, 0, 1))
    processed_img = np.ascontiguousarray(img_transposed[np.newaxis, :, :, :])
    
    return processed_img, (original_height, original_width)

# ---------------------------------------------------------------------------------
# --- 🚀🚀🚀 后处理函数 🚀🚀🚀 ---
# ---------------------------------------------------------------------------------

def postprocess_output(output_tensor, original_shape):
    """
    后处理模型输出，生成车道线掩码
    
    Args:
        output_tensor: 模型输出张量
        original_shape: 原始图片形状 (height, width)
    
    Returns:
        lane_mask: 车道线掩码 (原始图片尺寸)
    """
    original_height, original_width = original_shape
    
    # 1. Argmax获取分割掩码
    pred_mask = np.argmax(output_tensor, axis=1).squeeze()
    
    # 2. 转换为可视化格式
    vis_mask = (pred_mask * 255).astype(np.uint8)
    
    # 3. 如果需要，resize回原始尺寸
    if vis_mask.shape != (original_height, original_width):
        lane_mask = cv2.resize(vis_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        print(f"📐 掩码尺寸调整: {vis_mask.shape} -> {original_height}×{original_width}")
    else:
        lane_mask = vis_mask
        print(f"🎯 掩码尺寸完美匹配: {original_height}×{original_width}")
    
    return lane_mask

def create_visualization(original_img, lane_mask, alpha=0.5):
    """
    创建车道线检测可视化结果
    
    Args:
        original_img: 原始BGR图片
        lane_mask: 车道线掩码
        alpha: 叠加透明度
    
    Returns:
        vis_img: 可视化结果图片
    """
    # 创建绿色叠加层
    green_overlay = np.zeros_like(original_img, dtype=np.uint8)
    green_overlay[lane_mask > 0] = [0, 255, 0]  # BGR格式的绿色
    
    # 叠加原图和掩码
    vis_img = cv2.addWeighted(original_img, 1.0, green_overlay, alpha, 0)
    
    return vis_img

def add_info_text(img, inference_time, original_shape, model_input_shape):
    """
    在图片上添加推理信息文本
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)  # 绿色
    thickness = 2
    
    # 添加推理时间
    time_text = f"Inference: {inference_time:.1f}ms"
    cv2.putText(img, time_text, (15, 30), font, font_scale, color, thickness)
    
    # 添加图片尺寸信息
    size_text = f"Size: {original_shape[1]}x{original_shape[0]} -> {model_input_shape[1]}x{model_input_shape[0]}"
    cv2.putText(img, size_text, (15, 65), font, font_scale, color, thickness)
    
    # 添加模型信息
    model_text = f"Model: Fast-SCNN (FP16)"
    cv2.putText(img, model_text, (15, 100), font, font_scale, color, thickness)
    
    return img

def inference_single_image(image_path, output_dir=None, show_result=True, save_result=True):
    """
    对单张图片进行车道线检测推理
    
    Args:
        image_path: 输入图片路径
        output_dir: 输出目录
        show_result: 是否显示结果
        save_result: 是否保存结果
    """
    # 检查输入图片是否存在
    if not os.path.exists(image_path):
        print(f"❌ 错误: 图片文件不存在 - {image_path}")
        return
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 模型文件不存在 - {MODEL_PATH}")
        return
    
    print("=" * 80)
    print("🚀 Fast-SCNN 车道线检测 - 单图推理")
    print("=" * 80)
    print(f"📁 输入图片: {image_path}")
    print(f"🧠 模型路径: {MODEL_PATH}")
    print(f"🎯 模型输入: {MODEL_WIDTH}×{MODEL_HEIGHT} (FP16)")
    
    # 加载模型
    print("\n⏳ 正在加载模型...")
    start_time = time.time()
    model = InferSession(DEVICE_ID, MODEL_PATH)
    load_time = (time.time() - start_time) * 1000
    print(f"✅ 模型加载完成 ({load_time:.1f}ms)")
    
    # 读取图片
    print("\n📖 正在读取图片...")
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"❌ 错误: 无法读取图片 - {image_path}")
        return
    
    original_shape = original_img.shape[:2]
    print(f"✅ 图片读取成功，尺寸: {original_shape[1]}×{original_shape[0]}")
    
    # 预处理
    print("\n⚙️ 正在预处理...")
    preprocess_start = time.time()
    input_data, original_shape = preprocess_image(original_img)
    preprocess_time = (time.time() - preprocess_start) * 1000
    print(f"✅ 预处理完成 ({preprocess_time:.1f}ms)")
    print(f"   输入张量形状: {input_data.shape}")
    print(f"   数据类型: {input_data.dtype}")
    
    # NPU推理
    print("\n🚀 正在进行NPU推理...")
    inference_start = time.time()
    outputs = model.infer([input_data])
    inference_time = (time.time() - inference_start) * 1000
    print(f"✅ NPU推理完成 ({inference_time:.1f}ms)")
    print(f"   输出张量形状: {outputs[0].shape}")
    
    # 后处理
    print("\n🎨 正在后处理...")
    postprocess_start = time.time()
    lane_mask = postprocess_output(outputs[0], original_shape)
    postprocess_time = (time.time() - postprocess_start) * 1000
    print(f"✅ 后处理完成 ({postprocess_time:.1f}ms)")
    
    # 创建可视化结果
    print("\n🖼️ 正在生成可视化结果...")
    vis_img = create_visualization(original_img, lane_mask)
    vis_img = add_info_text(vis_img, inference_time, original_shape, (MODEL_HEIGHT, MODEL_WIDTH))
    
    # 统计信息
    total_time = preprocess_time + inference_time + postprocess_time
    print("\n📊 性能统计:")
    print(f"   预处理时间: {preprocess_time:.1f}ms ({preprocess_time/total_time*100:.1f}%)")
    print(f"   NPU推理时间: {inference_time:.1f}ms ({inference_time/total_time*100:.1f}%)")
    print(f"   后处理时间: {postprocess_time:.1f}ms ({postprocess_time/total_time*100:.1f}%)")
    print(f"   总处理时间: {total_time:.1f}ms")
    print(f"   等效FPS: {1000/total_time:.1f}")
    
    # 保存结果
    if save_result:
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        input_name = Path(image_path).stem
        
        # 保存原始掩码
        mask_path = os.path.join(output_dir, f"{input_name}_mask.png")
        cv2.imwrite(mask_path, lane_mask)
        
        # 保存可视化结果
        vis_path = os.path.join(output_dir, f"{input_name}_result.jpg")
        cv2.imwrite(vis_path, vis_img)
        
        print(f"\n💾 结果已保存:")
        print(f"   掩码文件: {mask_path}")
        print(f"   可视化文件: {vis_path}")
    
    # 显示结果
    if show_result:
        print("\n👁️ 正在显示结果...")
        print("   按任意键关闭窗口")
        
        # 创建窗口并显示
        window_name = f"Fast-SCNN Lane Detection - {Path(image_path).name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\n✅ 推理完成!")
    print("=" * 80)

def main():
    global MODEL_PATH, DEVICE_ID
    parser = argparse.ArgumentParser(description="Fast-SCNN 车道线检测 - 单图推理")
    parser.add_argument("--input", "-i", required=True, help="输入图片路径")
    parser.add_argument("--output", "-o", default=None, help="输出目录 (默认为输入图片同目录)")
    parser.add_argument("--model", "-m", default=MODEL_PATH, help=f"模型文件路径 (默认: {MODEL_PATH})")
    parser.add_argument("--device", "-d", type=int, default=DEVICE_ID, help=f"设备ID (默认: {DEVICE_ID})")
    parser.add_argument("--no-show", action="store_true", help="不显示结果窗口")
    parser.add_argument("--no-save", action="store_true", help="不保存结果文件")
    
    args = parser.parse_args()
    
    # 更新全局配置
    
    MODEL_PATH = args.model
    DEVICE_ID = args.device
    
    # 执行推理
    inference_single_image(
        image_path=args.input,
        output_dir=args.output,
        show_result=not args.no_show,
        save_result=not args.no_save
    )

if __name__ == "__main__":
    # 如果没有命令行参数，使用默认测试
    import sys
    if len(sys.argv) == 1:
        print("=" * 80)
        print("🔧 测试模式 - 请提供输入图片路径")
        print("=" * 80)
        print("使用方法:")
        print("  python single_image_inference.py --input <image_path>")
        print("\n完整参数:")
        print("  --input  | -i  : 输入图片路径 (必需)")
        print("  --output | -o  : 输出目录 (可选)")
        print("  --model  | -m  : 模型文件路径")
        print("  --device | -d  : NPU设备ID")
        print("  --no-show      : 不显示结果窗口")
        print("  --no-save      : 不保存结果文件")
        print("\n示例:")
        print("  python single_image_inference.py -i test_image.jpg")
        print("  python single_image_inference.py -i test_image.jpg -o results/")
        print("  python single_image_inference.py -i test_image.jpg --no-show")
        print("=" * 80)
    else:
        main()
