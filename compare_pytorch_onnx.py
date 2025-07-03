import os
import cv2
import torch
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import argparse
import random
from tqdm import tqdm

from models.fast_scnn import FastSCNN

# --- PyTorch Model and Preprocessing ---

def get_pytorch_model(model_path, device, num_classes=2):
    """加载PyTorch模型"""
    model = FastSCNN(num_classes=num_classes, aux=True) # <--- 启用辅助头以匹配模型权重
    state_dict = torch.load(model_path, map_location=device)
    # 处理在DataParallel中训练的模型
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def get_pytorch_transform(input_size=(360, 640), base_size=1024):
    """获取与validate_model_predictions.py一致的预处理"""
    # 参照 validate_model_predictions.py 的预处理流程
    # 1. Resize到base_size（与训练时一致）
    # 2. ToTensor (将 HWC, [0, 255] 转为 CHW, [0.0, 1.0])
    # 3. Normalize (这里我们自定义的训练没有使用ImageNet的均值和方差)
    return transforms.Compose([
        transforms.Resize((base_size, base_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

def run_pytorch_inference(model, image_path, device, input_size=(360, 640), base_size=1024):
    """运行PyTorch模型推理 - 模拟ONNX的端到端处理方式"""
    # 使用与ONNX完全相同的图像加载方式
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 步骤1: resize到640×360（与ONNX输入一致）
    h, w = input_size  # (360, 640)
    image_resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 步骤2: 手动模拟ONNX内部的处理 - resize到1024×1024
    image_base = cv2.resize(image_resized, (base_size, base_size), interpolation=cv2.INTER_LINEAR)
    
    # 步骤3: 转换为tensor并归一化（模拟ONNX内部的预处理）
    image_tensor = torch.from_numpy(image_base.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        if isinstance(output, tuple):
            output = output[0]
    
    # 获取分割图 (C, H, W) -> (H, W)
    pred_mask_base = torch.argmax(output.squeeze(), dim=0).cpu().numpy().astype(np.uint8)
    
    # 步骤4: 手动模拟ONNX内部的后处理 - resize回640×360
    pred_mask = cv2.resize(pred_mask_base, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return pred_mask

# --- ONNX Model Inference ---

def get_onnx_session(onnx_path):
    """加载ONNX推理会话"""
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
        return session
    except Exception as e:
        print(f"Error loading ONNX session: {e}")
        print("Attempting to load with CPUExecutionProvider only.")
        try:
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            return session
        except Exception as e2:
            print(f"Failed to load ONNX model with CPU as well: {e2}")
            return None


def run_onnx_inference(session, image_path, input_size=(360, 640)):
    """运行端到端ONNX模型推理 - 接受原始图像数据，无需复杂预处理"""
    # 使用OpenCV加载原始图像（这是最常见的部署场景）
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 直接resize到ONNX模型期望的输入尺寸
    h, w = input_size  # (360, 640)
    image_resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

    # HWC -> CHW, and add batch dimension，保持原始数据范围[0-255]
    input_tensor = np.transpose(image_resized, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
    
    # 不做任何归一化！ONNX模型是端到端的，内部会处理一切

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    result = session.run([output_name], {input_name: input_tensor})[0]
    
    # (1, C, H, W) -> (H, W)
    pred_mask = np.argmax(result.squeeze(), axis=0).astype(np.uint8)
    
    return pred_mask

# --- Comparison Logic ---

def get_color_mask(mask, num_classes=2):
    """为分割图上色"""
    colors = np.array([
        [0, 0, 0],       # Class 0: background (black)
        [255, 0, 0],     # Class 1: lane/drivable (red)
    ], dtype=np.uint8)
    
    if num_classes > len(colors):
        # Generate random colors for additional classes
        new_colors = np.random.randint(0, 255, size=(num_classes - len(colors), 3), dtype=np.uint8)
        colors = np.vstack([colors, new_colors])
        
    color_mask = colors[mask]
    return color_mask

def compare_masks(mask1, mask2):
    """比较两个mask，返回差异像素数和差异图"""
    diff = np.abs(mask1.astype(int) - mask2.astype(int))
    diff_pixels = np.sum(diff > 0)
    diff_mask = (diff > 0).astype(np.uint8) * 255 # 差异处显示为白色
    return diff_pixels, diff_mask

def visualize_comparison(original_img_path, pytorch_mask, onnx_mask, diff_mask, output_path):
    """生成并保存对比图"""
    # 加载原始图像并调整大小以匹配输出
    original_img = cv2.imread(original_img_path)
    original_img_resized = cv2.resize(original_img, (pytorch_mask.shape[1], pytorch_mask.shape[0]))

    # 为mask上色
    pytorch_color = get_color_mask(pytorch_mask)
    onnx_color = get_color_mask(onnx_mask)
    
    # 将差异图转为3通道
    diff_viz = cv2.cvtColor(diff_mask, cv2.COLOR_GRAY2BGR)

    # 叠加mask到原图
    pytorch_overlay = cv2.addWeighted(original_img_resized, 0.6, pytorch_color, 0.4, 0)
    onnx_overlay = cv2.addWeighted(original_img_resized, 0.6, onnx_color, 0.4, 0)

    # 创建标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    def add_title(img, text):
        cv2.putText(img, text, (10, 25), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    pytorch_viz = add_title(pytorch_overlay, "PyTorch")
    onnx_viz = add_title(onnx_overlay, "ONNX")
    diff_viz_titled = add_title(diff_viz, "Difference (White)")

    # 拼接图像
    # 确保所有图像都是相同的高度
    h, w, _ = original_img_resized.shape
    top_row = np.hstack((original_img_resized, pytorch_viz))
    bottom_row = np.hstack((onnx_viz, diff_viz_titled))
    
    # 如果宽度不匹配，需要调整
    if top_row.shape[1] != bottom_row.shape[1]:
        # 这是不应该发生的情况，但作为保险
        min_w = min(top_row.shape[1], bottom_row.shape[1])
        top_row = top_row[:, :min_w]
        bottom_row = bottom_row[:, :min_w]

    comparison_img = np.vstack((top_row, bottom_row))
    
    cv2.imwrite(output_path, comparison_img)

def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX Fast-SCNN model predictions.")
    parser.add_argument('--pytorch-model', type=str, default='./weights/custom_scratch/fast_scnn_custom.pth',
                        help='Path to the PyTorch model weights.')
    parser.add_argument('--onnx-model', type=str, default='./weights/fast_scnn_custom_e2e_640x360_fixed_simplified.onnx',
                        help='Path to the simplified ONNX model.')
    parser.add_argument('--image-dir', type=str, default='./data/custom/images',
                        help='Directory containing images for comparison.')
    parser.add_argument('--image-path', type=str, default=None,
                        help='Path to a specific image to compare. Overrides --image-dir.')
    parser.add_argument('--output-dir', type=str, default='./model_validation_results/pytorch_vs_onnx',
                        help='Directory to save comparison results.')
    parser.add_argument('--num-images', type=int, default=20,
                        help='Number of random images to test.')
    parser.add_argument('--input-size', type=str, default='360,640',
                        help='Input size as height,width.')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of segmentation classes.')
    
    args = parser.parse_args()

    # 解析输入尺寸
    try:
        h, w = map(int, args.input_size.split(','))
        input_size = (h, w)
    except ValueError:
        print("Error: --input-size format must be 'height,width'")
        return

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print("Loading PyTorch model...")
    pytorch_model = get_pytorch_model(args.pytorch_model, device, args.num_classes)
    print("Loading ONNX model...")
    onnx_session = get_onnx_session(args.onnx_model)
    if onnx_session is None:
        return
    
    # 不再需要预定义的transform，因为我们将在推理函数内部处理所有步骤
    base_size = 1024  # 与validate_model_predictions.py保持一致

    # 获取图像列表
    selected_files = []
    if args.image_path:
        # 如果指定了单个图像路径，直接使用
        if os.path.isfile(args.image_path):
            selected_files = [args.image_path]
            print(f"Using specified image: {args.image_path}")
        else:
            print(f"Error: Specified image path does not exist: {args.image_path}")
            return
    else:
        # 否则，从目录中随机选择图像
        try:
            image_files = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                print(f"Error: No images found in '{args.image_dir}'")
                return
            if len(image_files) < args.num_images:
                print(f"Warning: Found only {len(image_files)} images, will test on all of them.")
                selected_files = image_files
            else:
                selected_files = random.sample(image_files, args.num_images)
        except FileNotFoundError:
            print(f"Error: Image directory not found at '{args.image_dir}'")
            return

    total_diff_pixels = 0
    total_pixels = 0

    print(f"Starting comparison for {len(selected_files)} images...")
    print(f"Both models now use identical processing: cv2 load -> 640×360 -> 1024×1024 -> inference -> 640×360")
    
    for image_path in tqdm(selected_files, desc="Comparing models"):
        # PyTorch 推理（现在使用与ONNX完全相同的处理方式）
        pytorch_mask = run_pytorch_inference(pytorch_model, image_path, device, input_size, base_size)
        
        # ONNX 推理（ONNX模型内部已包含所有处理：640×360->1024×1024->推理->640×360）
        onnx_mask = run_onnx_inference(onnx_session, image_path, input_size)

        # 比较
        diff_pixels, diff_mask = compare_masks(pytorch_mask, onnx_mask)
        
        total_diff_pixels += diff_pixels
        total_pixels += np.prod(pytorch_mask.shape)

        # 可视化
        img_name = os.path.basename(image_path)
        output_path = os.path.join(args.output_dir, f"comparison_{img_name}")
        visualize_comparison(image_path, pytorch_mask, onnx_mask, diff_mask, output_path)

    # 打印总结
    print('\n--- Comparison Summary ---')
    if total_pixels > 0:
        avg_diff_percentage = (total_diff_pixels / total_pixels) * 100
        print(f"Compared {len(selected_files)} images.")
        print(f"Total different pixels: {total_diff_pixels}")
        print(f"Average difference: {avg_diff_percentage:.6f}%")
    else:
        print("No images were processed.")
    
    print(f"Visual comparison results saved in: {args.output_dir}")

if __name__ == '__main__':
    main()
