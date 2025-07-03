"""
测试custom数据集端到端ONNX模型
验证ONNX模型的正确性，与PyTorch模型对比，支持可视化结果
"""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import time
from models.fast_scnn import FastSCNN

def load_pytorch_model(model_path, device):
    """加载PyTorch模型用于对比"""
    model = FastSCNN(num_classes=2, aux=True)
    state_dict = torch.load(model_path, map_location='cpu')
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def preprocess_image_pytorch(image_path, target_size=(1024, 1024)):
    """PyTorch模型预处理 (与validate_model_predictions.py一致)"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size, Image.BILINEAR)
    
    img_array = np.array(image).astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(img_array.transpose((2, 0, 1))).unsqueeze(0)
    
    return input_tensor, image

def preprocess_image_onnx(image_path, target_size=(360, 640)):
    """ONNX模型预处理 (端到端，任意尺寸输入)"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 端到端ONNX模型可以接受任意尺寸，内部会自动resize
    # 但为了最佳性能，建议输入目标尺寸
    image_resized = cv2.resize(image, (target_size[1], target_size[0]))  # W, H
    
    # 转换为NCHW格式，保持0-255范围
    input_data = image_resized.transpose(2, 0, 1)[None].astype(np.float32)
    
    return input_data, image, image_resized

def compare_models(pytorch_model, onnx_session, image_path, device, args):
    """对比PyTorch和ONNX模型输出"""
    print(f"\n🔍 对比测试: {os.path.basename(image_path)}")
    
    # PyTorch推理
    print("  🐍 PyTorch推理...")
    pytorch_input, original_image = preprocess_image_pytorch(image_path, (1024, 1024))
    pytorch_input = pytorch_input.to(device)
    
    with torch.no_grad():
        start_time = time.time()
        pytorch_outputs = pytorch_model(pytorch_input)
        pytorch_time = time.time() - start_time
        
        if isinstance(pytorch_outputs, tuple):
            pytorch_output = pytorch_outputs[0]
        else:
            pytorch_output = pytorch_outputs
        
        pytorch_probs = F.softmax(pytorch_output, dim=1)
        pytorch_mask = torch.argmax(pytorch_probs, dim=1).cpu().numpy()[0]
    
    # ONNX推理
    print("  🔧 ONNX推理...")
    onnx_input, _, onnx_image = preprocess_image_onnx(image_path, (args.input_height, args.input_width))
    
    start_time = time.time()
    onnx_outputs = onnx_session.run(None, {'input': onnx_input})
    onnx_time = time.time() - start_time
    
    onnx_output = onnx_outputs[0][0]  # Remove batch dimension
    onnx_mask = np.argmax(onnx_output, axis=0)
    
    # 结果统计
    pytorch_drivable = np.sum(pytorch_mask) / pytorch_mask.size
    onnx_drivable = np.sum(onnx_mask) / onnx_mask.size
    
    print(f"  ⚡ PyTorch时间: {pytorch_time*1000:.2f}ms")
    print(f"  ⚡ ONNX时间: {onnx_time*1000:.2f}ms")
    print(f"  🚀 ONNX加速比: {pytorch_time/onnx_time:.2f}x")
    print(f"  🎯 PyTorch可驾驶区域: {pytorch_drivable*100:.1f}%")
    print(f"  🎯 ONNX可驾驶区域: {onnx_drivable*100:.1f}%")
    
    # 可视化对比
    if args.save_comparison:
        save_comparison(original_image, onnx_image, pytorch_mask, onnx_mask, 
                       image_path, args.output_dir)
    
    return {
        'pytorch_time': pytorch_time,
        'onnx_time': onnx_time,
        'speedup': pytorch_time / onnx_time,
        'pytorch_drivable': pytorch_drivable,
        'onnx_drivable': onnx_drivable,
        'pytorch_mask': pytorch_mask,
        'onnx_mask': onnx_mask
    }

def save_comparison(original_image, onnx_image, pytorch_mask, onnx_mask, image_path, output_dir):
    """保存对比可视化结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始图像
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image (1024x1024)')
    axes[0, 0].axis('off')
    
    # ONNX输入图像
    axes[0, 1].imshow(onnx_image)
    axes[0, 1].set_title('ONNX Input (640x360)')
    axes[0, 1].axis('off')
    
    # 空白
    axes[0, 2].axis('off')
    
    # PyTorch结果
    axes[1, 0].imshow(pytorch_mask, cmap='gray')
    axes[1, 0].set_title(f'PyTorch Result\nDrivable: {np.sum(pytorch_mask)/pytorch_mask.size*100:.1f}%')
    axes[1, 0].axis('off')
    
    # ONNX结果
    axes[1, 1].imshow(onnx_mask, cmap='gray')
    axes[1, 1].set_title(f'ONNX Result\nDrivable: {np.sum(onnx_mask)/onnx_mask.size*100:.1f}%')
    axes[1, 1].axis('off')
    
    # 差异图
    if pytorch_mask.shape == onnx_mask.shape:
        diff = np.abs(pytorch_mask.astype(float) - onnx_mask.astype(float))
        diff_ratio = np.sum(diff) / diff.size
        axes[1, 2].imshow(diff, cmap='Reds')
        axes[1, 2].set_title(f'Difference\nDiff: {diff_ratio*100:.1f}%')
    else:
        # 尺寸不同时，resize后对比
        pytorch_resized = cv2.resize(pytorch_mask.astype(np.uint8), 
                                   (onnx_mask.shape[1], onnx_mask.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
        diff = np.abs(pytorch_resized.astype(float) - onnx_mask.astype(float))
        diff_ratio = np.sum(diff) / diff.size
        axes[1, 2].imshow(diff, cmap='Reds')
        axes[1, 2].set_title(f'Difference (Resized)\nDiff: {diff_ratio*100:.1f}%')
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    filename = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f"{filename}_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 对比图保存: {save_path}")

def test_onnx_model(onnx_path, test_images, args):
    """测试ONNX模型"""
    try:
        import onnxruntime as ort
        
        print(f"🧪 测试ONNX模型: {onnx_path}")
        
        # 创建推理会话
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"  🖥️  执行提供商: {session.get_providers()}")
        
        # 加载PyTorch模型用于对比
        pytorch_model = None
        if args.compare_pytorch and args.pytorch_model:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pytorch_model = load_pytorch_model(args.pytorch_model, device)
            print(f"  🐍 PyTorch模型加载成功")
        
        results = []
        
        for image_path in test_images:
            if not os.path.exists(image_path):
                print(f"  ⚠️  图像不存在: {image_path}")
                continue
            
            if pytorch_model is not None:
                # 对比测试
                result = compare_models(pytorch_model, session, image_path, device, args)
                results.append(result)
            else:
                # 仅ONNX测试
                print(f"\n🧪 ONNX测试: {os.path.basename(image_path)}")
                
                onnx_input, _, onnx_image = preprocess_image_onnx(image_path, (args.input_height, args.input_width))
                
                start_time = time.time()
                onnx_outputs = session.run(None, {'input': onnx_input})
                onnx_time = time.time() - start_time
                
                onnx_output = onnx_outputs[0][0]
                onnx_mask = np.argmax(onnx_output, axis=0)
                onnx_drivable = np.sum(onnx_mask) / onnx_mask.size
                
                print(f"  ⚡ ONNX时间: {onnx_time*1000:.2f}ms")
                print(f"  🚀 理论FPS: {1/onnx_time:.1f}")
                print(f"  🎯 可驾驶区域: {onnx_drivable*100:.1f}%")
                print(f"  📊 输出尺寸: {onnx_mask.shape}")
                
                results.append({
                    'onnx_time': onnx_time,
                    'onnx_drivable': onnx_drivable
                })
        
        # 汇总统计
        if results:
            avg_onnx_time = np.mean([r.get('onnx_time', 0) for r in results])
            avg_onnx_fps = 1 / avg_onnx_time if avg_onnx_time > 0 else 0
            
            print(f"\n📊 测试汇总 (共{len(results)}张图像):")
            print(f"  ⚡ 平均ONNX推理时间: {avg_onnx_time*1000:.2f}ms")
            print(f"  🚀 平均理论FPS: {avg_onnx_fps:.1f}")
            
            if pytorch_model is not None:
                avg_pytorch_time = np.mean([r.get('pytorch_time', 0) for r in results])
                avg_speedup = np.mean([r.get('speedup', 0) for r in results])
                
                print(f"  ⚡ 平均PyTorch推理时间: {avg_pytorch_time*1000:.2f}ms")
                print(f"  🚀 平均加速比: {avg_speedup:.2f}x")
                
                # 可驾驶区域一致性
                pytorch_drivables = [r.get('pytorch_drivable', 0) for r in results]
                onnx_drivables = [r.get('onnx_drivable', 0) for r in results]
                consistency = np.mean([1 - abs(p - o) for p, o in zip(pytorch_drivables, onnx_drivables)])
                
                print(f"  🎯 结果一致性: {consistency*100:.1f}%")
        
        return True
        
    except ImportError:
        print("❌ onnxruntime未安装")
        print("   安装命令: pip install onnxruntime-gpu  # GPU版本")
        print("   安装命令: pip install onnxruntime      # CPU版本")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def parse_args():
    parser = argparse.ArgumentParser(description='Test Custom Dataset End-to-End ONNX Model')
    parser.add_argument('--onnx-model', type=str, required=True,
                        help='path to ONNX model')
    parser.add_argument('--pytorch-model', type=str, 
                        default='./weights/custom_scratch/fast_scnn_custom.pth',
                        help='path to PyTorch model for comparison')
    parser.add_argument('--test-images', type=str, nargs='+',
                        help='test image paths')
    parser.add_argument('--test-dir', type=str,
                        help='directory containing test images')
    parser.add_argument('--input-size', type=str, default='360,640',
                        help='ONNX model input size as height,width')
    parser.add_argument('--compare-pytorch', action='store_true', default=True,
                        help='compare with PyTorch model')
    parser.add_argument('--save-comparison', action='store_true', default=True,
                        help='save comparison visualizations')
    parser.add_argument('--output-dir', type=str, default='onnx_test_results',
                        help='output directory for results')
    args = parser.parse_args()
    
    # Parse input size
    h, w = map(int, args.input_size.split(','))
    args.input_height = h
    args.input_width = w
    
    return args

def main():
    args = parse_args()
    
    print("🧪 Custom数据集端到端ONNX模型测试")
    print("=" * 50)
    
    # 收集测试图像
    test_images = []
    
    if args.test_images:
        test_images.extend(args.test_images)
    
    if args.test_dir and os.path.isdir(args.test_dir):
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            import glob
            test_images.extend(glob.glob(os.path.join(args.test_dir, ext)))
    
    # 如果没有指定测试图像，尝试使用data/custom/images中的图像
    if not test_images:
        custom_images_dir = 'data/custom/images'
        if os.path.isdir(custom_images_dir):
            test_images = [os.path.join(custom_images_dir, f) 
                          for f in os.listdir(custom_images_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"🔍 自动发现测试图像: {len(test_images)}张")
    
    if not test_images:
        print("❌ 未找到测试图像！")
        print("请使用 --test-images 或 --test-dir 指定测试图像")
        return
    
    print(f"📁 测试图像: {len(test_images)}张")
    for img in test_images[:5]:  # 只显示前5张
        print(f"   {os.path.basename(img)}")
    if len(test_images) > 5:
        print(f"   ... 还有{len(test_images)-5}张")
    
    # 测试ONNX模型
    success = test_onnx_model(args.onnx_model, test_images, args)
    
    if success:
        print(f"\n✅ 测试完成!")
        if args.save_comparison:
            print(f"📁 结果保存在: {args.output_dir}")
    else:
        print(f"\n❌ 测试失败!")

if __name__ == '__main__':
    main() 