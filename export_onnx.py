"""
Export Fast-SCNN model to ONNX format
支持动态输入尺寸和固定尺寸两种导出方式
"""
import os
import torch
import torch.onnx
import numpy as np
from models.fast_scnn import get_fast_scnn
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Export Fast-SCNN to ONNX')
    parser.add_argument('--model-path', type=str, 
                        default='./weights/fast_scnn_tusimple_best_model.pth',
                        help='path to trained model')
    parser.add_argument('--output-path', type=str, 
                        default='./weights/fast_scnn_tusimple.onnx',
                        help='output ONNX model path')
    parser.add_argument('--input-size', type=str, default='768,1024',
                        help='input size as height,width (default: 768,1024 - same as training)')
    parser.add_argument('--dynamic', action='store_true', default=False,
                        help='export with dynamic input shapes (Note: Fast-SCNN has adaptive pooling which may not work with dynamic shapes)')
    parser.add_argument('--opset-version', type=int, default=11,
                        help='ONNX opset version (default: 11)')
    parser.add_argument('--simplify', action='store_true', default=False,
                        help='simplify ONNX model using onnx-simplifier')
    args = parser.parse_args()
    
    # Parse input size
    h, w = map(int, args.input_size.split(','))
    args.input_height = h
    args.input_width = w
    
    return args

def load_model(model_path, device):
    """加载训练好的模型"""
    print(f"Loading model from {model_path}...")
    
    # 创建模型
    model = get_fast_scnn(dataset='tusimple', aux=False)
    
    # 加载权重
    if os.path.isfile(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        # Handle DataParallel models
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print("Model loaded successfully!")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.to(device)
    model.eval()
    return model

def export_onnx(model, args):
    """导出ONNX模型"""
    device = next(model.parameters()).device
    
    # 创建示例输入 (使用训练时的crop_size作为参考尺寸)
    dummy_input = torch.randn(1, 3, args.input_height, args.input_width).to(device)
    
    print(f"\nExporting ONNX model...")
    print(f"Reference input size: {args.input_height}x{args.input_width} (training crop_size)")
    print(f"Dynamic shapes: {args.dynamic}")
    print(f"ONNX opset version: {args.opset_version}")
    
    if args.dynamic:
        print("💡 Fast-SCNN supports arbitrary input sizes due to its fully convolutional architecture!")
    
    # 设置动态轴
    if args.dynamic:
        dynamic_axes = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
        print("Using dynamic input/output shapes - model can handle any resolution")
    else:
        dynamic_axes = None
        print(f"Using fixed input/output shapes: {args.input_height}x{args.input_width}")
    
    # 导出ONNX
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                args.output_path,
                export_params=True,
                opset_version=args.opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
        print(f"✅ ONNX model exported successfully to: {args.output_path}")
        
        # 检查导出的模型
        import onnx
        onnx_model = onnx.load(args.output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model validation passed!")
        
        # 打印模型信息
        print(f"\nModel Information:")
        print(f"  Input shape: {[d.dim_value if d.dim_value > 0 else 'dynamic' for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]}")
        print(f"  Output shape: {[d.dim_value if d.dim_value > 0 else 'dynamic' for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]}")
        print(f"  Model size: {os.path.getsize(args.output_path) / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error during ONNX export: {str(e)}")
        return False
    
    return True

def simplify_onnx(model_path):
    """简化ONNX模型"""
    try:
        import onnxsim
        print(f"\nSimplifying ONNX model...")
        
        # 加载模型
        import onnx
        model = onnx.load(model_path)
        
        # 简化
        model_simplified, check = onnxsim.simplify(model)
        
        if check:
            # 保存简化后的模型
            simplified_path = model_path.replace('.onnx', '_simplified.onnx')
            onnx.save(model_simplified, simplified_path)
            
            original_size = os.path.getsize(model_path) / 1024 / 1024
            simplified_size = os.path.getsize(simplified_path) / 1024 / 1024
            
            print(f"✅ Model simplified successfully!")
            print(f"  Original size: {original_size:.2f} MB")
            print(f"  Simplified size: {simplified_size:.2f} MB")
            print(f"  Size reduction: {(1 - simplified_size/original_size)*100:.1f}%")
            print(f"  Simplified model saved to: {simplified_path}")
        else:
            print("❌ Model simplification failed!")
            
    except ImportError:
        print("⚠️  onnx-simplifier not installed. Install with: pip install onnx-simplifier")
    except Exception as e:
        print(f"❌ Error during simplification: {str(e)}")

def test_onnx_inference(onnx_path, input_shape):
    """测试ONNX模型推理"""
    try:
        import onnxruntime as ort
        print(f"\nTesting ONNX inference...")
        
        # 创建推理会话
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        print(f"  Execution providers: {session.get_providers()}")
        
        # 创建测试输入
        dummy_input = np.random.randn(1, 3, input_shape[0], input_shape[1]).astype(np.float32)
        
        # 推理
        import time
        start_time = time.time()
        outputs = session.run(None, {'input': dummy_input})
        inference_time = time.time() - start_time
        
        print(f"✅ ONNX inference successful!")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {outputs[0].shape}")
        print(f"  Inference time: {inference_time*1000:.2f}ms")
        
        return True
        
    except ImportError:
        print("⚠️  onnxruntime not installed. Install with: pip install onnxruntime-gpu")
        return False
    except Exception as e:
        print(f"❌ ONNX inference test failed: {str(e)}")
        return False

def main():
    args = parse_args()
    
    print("🚀 Fast-SCNN ONNX Export Tool")
    print("=" * 50)
    
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    model = load_model(args.model_path, device)
    
    # 导出ONNX
    success = export_onnx(model, args)
    
    if success:
        # 简化模型（可选）
        if args.simplify:
            simplify_onnx(args.output_path)
        
        # 测试推理
        test_onnx_inference(args.output_path, (args.input_height, args.input_width))
        
        print(f"\n🎉 Export completed successfully!")
        print(f"ONNX model saved to: {args.output_path}")
        
        # 使用建议
        print(f"\n💡 Usage Tips:")
        print(f"1. Recommended: Dynamic input size (supports any resolution):")
        print(f"   python export_onnx.py --dynamic")
        print(f"2. For fixed input size inference (training size {args.input_height}x{args.input_width}):")
        print(f"   python export_onnx.py --input-size {args.input_height},{args.input_width}")
        print(f"3. For custom input size with dynamic support:")
        print(f"   python export_onnx.py --dynamic --input-size HEIGHT,WIDTH")
        print(f"4. To simplify the model:")
        print(f"   python export_onnx.py --dynamic --simplify")
        print(f"\n🔍 Note: Fast-SCNN supports arbitrary input sizes thanks to:")
        print(f"   - Fully convolutional architecture")
        print(f"   - Adaptive feature fusion with interpolation")
        print(f"   - No fixed-size fully connected layers")
        
    else:
        print("❌ Export failed!")

if __name__ == '__main__':
    main()
