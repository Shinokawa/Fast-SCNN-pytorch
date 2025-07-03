"""
Export Fast-SCNN model to ONNX format (End-to-End version for Atlas NPU optimization)
集成预处理操作，实现端到端推理，充分利用Atlas开发板NPU性能
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import traceback
import numpy as np

# Add imports for onnx, onnxsim, and onnxruntime, handling potential import errors
try:
    import onnx
except ImportError:
    print("Warning: onnx is not installed. Please install it via 'pip install onnx'")
    onnx = None

try:
    import onnxsim
except ImportError:
    print("Warning: onnx-simplifier is not installed. Please install it via 'pip install onnx-simplifier'")
    onnxsim = None

try:
    import onnxruntime as ort
except ImportError:
    print("Warning: onnxruntime is not installed. Please install it via 'pip install onnxruntime'")
    ort = None

from models.fast_scnn import FastSCNN, get_fast_scnn

class EndToEndFastSCNN(nn.Module):
    """端到端Fast-SCNN模型，集成了预处理和可选的后处理"""
    def __init__(self, backbone_model, input_size=(640, 360), base_size=1024, mean=None, std=None, apply_softmax=True):
        super(EndToEndFastSCNN, self).__init__()
        self.backbone = backbone_model
        self.preprocessor = EndToEndPreprocessing(input_size, base_size, mean, std)
        self.apply_softmax = apply_softmax
        self.input_size = input_size  # 最终输出尺寸

    def forward(self, x):
        # 预处理：640×360 -> 1024×1024
        x = self.preprocessor(x)
        
        # 模型推理：1024×1024 -> 1024×1024
        x = self.backbone(x)
        
        if isinstance(x, tuple):
            x = x[0] # 获取主输出
        
        # 后处理：将结果从 1024×1024 resize 回 640×360
        if x.shape[2] != self.input_size[1] or x.shape[3] != self.input_size[0]:  # 注意：input_size是(W,H)
            x = F.interpolate(x, size=(self.input_size[1], self.input_size[0]), mode='bilinear', align_corners=False)
        
        if self.apply_softmax:
            x = F.softmax(x, dim=1)
            
        return x

class EndToEndPreprocessing(nn.Module):
    """端到端预处理模块，集成所有预处理操作到NPU中"""
    
    def __init__(self, input_size=(640, 360), base_size=1024, mean=None, std=None):
        super(EndToEndPreprocessing, self).__init__()
        self.input_size = input_size
        self.base_size = base_size  # 内部处理尺寸，与validate_model_predictions.py一致
        
        # 仅在提供了mean和std时才注册为buffer
        if mean is not None and std is not None:
            self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        else:
            self.mean = None
            self.std = None
        
    def forward(self, x):
        """
        输入: 原始图像 (B, 3, H, W) 范围 [0, 255] uint8 或 float32
        输出: 预处理后的图像 (B, 3, base_size, base_size)
        """
        # 1. 转换数据类型
        if x.dtype == torch.uint8:
            x = x.float()
        
        # 2. 尺寸调整到base_size（与validate_model_predictions.py一致）
        if x.shape[2] != self.base_size or x.shape[3] != self.base_size:
            x = F.interpolate(x, size=(self.base_size, self.base_size), mode='bilinear', align_corners=False)
        
        # 3. 归一化到[0,1]
        x = x / 255.0
        
        # 4. ImageNet标准化 (可选)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        
        return x

class OnnxCompatiblePyramidPooling(nn.Module):
    """ONNX兼容的PyramidPooling模块，适配1024x1024输入 (特征图 32x32)"""
    
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(OnnxCompatiblePyramidPooling, self).__init__()
        # 目标池化尺寸: 1x1, 2x2, 4x4, 8x8, 对应输入特征图 32x32
        self.pool1 = nn.AvgPool2d(kernel_size=(32, 32))   # 32x32 -> 1x1
        self.pool2 = nn.AvgPool2d(kernel_size=(16, 16))   # 32x32 -> 2x2  
        self.pool3 = nn.AvgPool2d(kernel_size=(8, 8))     # 32x32 -> 4x4
        self.pool4 = nn.AvgPool2d(kernel_size=(4, 4))     # 32x32 -> 8x8
        
        inter_channels = in_channels // 4
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True))
        self.out = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
                                 norm_layer(in_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[2:] # 获取动态的特征图尺寸
        feat1 = F.interpolate(self.conv1(self.pool1(x)), size=(h, w), mode='bilinear', align_corners=False)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), size=(h, w), mode='bilinear', align_corners=False)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), size=(h, w), mode='bilinear', align_corners=False)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), size=(h, w), mode='bilinear', align_corners=False)
        
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x

def replace_pyramid_pooling(model):
    """替换模型中的PyramidPooling为ONNX兼容版本"""
    # 查找并替换PyramidPooling模块
    for name, module in model.named_modules():
        if hasattr(module, 'ppm') and hasattr(module.ppm, 'pool'):
            # 获取原有参数
            in_channels = module.ppm.conv1.conv[0].in_channels
            norm_layer = type(module.ppm.conv1.conv[1])
            
            # 创建新的兼容模块
            new_ppm = OnnxCompatiblePyramidPooling(in_channels, norm_layer, {})
            
            # 复制权重
            new_ppm.conv1.load_state_dict(module.ppm.conv1.conv.state_dict())
            new_ppm.conv2.load_state_dict(module.ppm.conv2.conv.state_dict())
            new_ppm.conv3.load_state_dict(module.ppm.conv3.conv.state_dict())
            new_ppm.conv4.load_state_dict(module.ppm.conv4.conv.state_dict())
            new_ppm.out.load_state_dict(module.ppm.out.conv.state_dict())
            
            # 替换模块
            module.ppm = new_ppm
            print("✅ Replaced PyramidPooling with ONNX-Compatible version for 1024x1024 input (32x32 feature map)")
            break
    
    return model

def parse_args():
    parser = argparse.ArgumentParser(description='Export Fast-SCNN to ONNX (End-to-End version for Atlas NPU)')
    parser.add_argument('--model-path', type=str, 
                        default='./weights/custom_scratch/fast_scnn_custom.pth',
                        help='path to trained model')
    parser.add_argument('--output-path', type=str, 
                        default='./weights/fast_scnn_custom_e2e_640x360.onnx',
                        help='output ONNX model path')
    parser.add_argument('--input-size', type=str, default='640,360',
                        help='input size as width,height (default: 640,360)')
    parser.add_argument('--opset-version', type=int, default=11,
                        help='ONNX opset version (default: 11)')
    parser.add_argument('--simplify', action='store_true', default=True,
                        help='simplify ONNX model using onnx-simplifier')
    parser.add_argument('--end-to-end', action='store_true', default=True,
                        help='export end-to-end model with preprocessing (default: True)')
    parser.add_argument('--include-postprocess', action='store_true', default=False,
                        help='include post-processing in the model')
    parser.add_argument('--mean', type=str, default=None,
                        help='normalization mean values (default: None, as per custom training)')
    parser.add_argument('--std', type=str, default=None,
                        help='normalization std values (default: None, as per custom training)')
    parser.add_argument('--input-range', type=str, default='0-255',
                        help='input pixel value range: 0-255 or 0-1 (default: 0-255)')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='number of classes for the segmentation model (default: 2 for custom dataset)')
    args = parser.parse_args()
    
    # Parse input size
    w, h = map(int, args.input_size.split(','))
    args.input_width = w
    args.input_height = h
    
    # Parse normalization parameters
    if args.mean:
        args.mean = list(map(float, args.mean.split(',')))
    if args.std:
        args.std = list(map(float, args.std.split(',')))
    
    return args

def load_model(model_path, device, args):
    """加载训练好的模型并创建端到端包装器"""
    print(f"Loading model from {model_path}...")
    
    # 创建基础模型
    # The --aux flag was used in training, so we should keep it here
    backbone_model = FastSCNN(num_classes=args.num_classes, aux=True)
    
    # 加载权重
    if os.path.isfile(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        # Handle DataParallel models
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        backbone_model.load_state_dict(state_dict)
        print("✅ Backbone model loaded successfully!")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # 替换PyramidPooling为ONNX兼容版本
    backbone_model = replace_pyramid_pooling(backbone_model)
    
    if args.end_to_end:
        # 创建端到端模型
        print("🔧 Creating end-to-end model with integrated preprocessing...")
        base_size = 1024  # 与validate_model_predictions.py保持一致
        model = EndToEndFastSCNN(
            backbone_model=backbone_model,
            input_size=(args.input_width, args.input_height),  # 外部输入尺寸：640×360
            base_size=base_size,  # 内部处理尺寸：1024×1024
            mean=args.mean,
            std=args.std,
            apply_softmax=True
        )
        
        if args.include_postprocess:
            print("🔧 Adding post-processing module...")
            # 如果需要后处理，可以进一步包装
            # 这里暂时保持简单，只做softmax
            pass
            
        print("✅ End-to-end model created successfully!")
        print(f"   📊 Input: Raw image [0-255] -> Shape: (B, 3, {args.input_height}, {args.input_width})")
        print(f"   � Internal processing: Resize to {base_size}×{base_size} for optimal accuracy")
        print(f"   �📊 Output: Segmentation probabilities -> Shape: (B, {args.num_classes}, {args.input_height}, {args.input_width})")
    else:
        # 使用原始模型
        model = backbone_model
        print("✅ Using original model (without preprocessing integration)")
    
    model.to(device)
    model.eval()
    return model

def export_onnx(model, args):
    """导出端到端ONNX模型"""
    if not onnx:
        print("❌ ONNX library not found. Skipping export.")
        return False
        
    device = next(model.parameters()).device
    
    # 创建示例输入（原始图像格式：640×360）
    if args.input_range == '0-255':
        # 模拟摄像头输入：uint8格式 [0, 255] -> but use float32 for export
        dummy_input = torch.randint(0, 256, (1, 3, args.input_height, args.input_width), dtype=torch.float32).to(device)
        input_desc = "Raw image [0-255]"
    else:
        # 模拟已归一化输入：float32格式 [0, 1]
        dummy_input = torch.rand(1, 3, args.input_height, args.input_width).to(device)
        input_desc = "Normalized image [0-1]"
    
    print(f"\n🚀 Exporting End-to-End ONNX model...")
    print(f"📊 Input format: {input_desc}")
    print(f"📊 Input size: {args.input_height}x{args.input_width}")
    print(f"📊 ONNX opset version: {args.opset_version}")
    
    if args.end_to_end:
        print("✨ Integrated preprocessing: ✅")
        print(f"   - Input: {args.input_width}×{args.input_height} -> Resize to 1024×1024")
        print("   - Pixel value normalization [0,255] -> [0,1]")
        if args.mean and args.std:
            print(f"   - ImageNet standardization (mean={args.mean}, std={args.std})")
        else:
            print("   - ImageNet standardization: ❌ (Not used for this model)")
        print(f"   - Model inference at 1024×1024 resolution")
        print(f"   - Output resize: 1024×1024 -> {args.input_width}×{args.input_height}")
        print("   - Softmax activation for probability output")
    else:
        print("⚠️  Preprocessing integration: ❌ (traditional mode)")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # 导出ONNX
    try:
        with torch.no_grad():
            # 测试前向传播
            print("🔍 Testing forward pass...")
            test_output = model(dummy_input)
            print(f"✅ Forward pass successful! Output shape: {test_output.shape}")
            
            torch.onnx.export(
                model,
                dummy_input,
                args.output_path,
                export_params=True,
                opset_version=args.opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                verbose=False,
            )
        print(f"✅ ONNX model exported successfully to: {args.output_path}")
        
        # 检查导出的模型
        onnx_model = onnx.load(args.output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model validation passed!")
        
        # 打印模型信息
        print(f"\n📋 Model Information:")
        print(f"  📊 Input shape: {[d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]}")
        print(f"  📊 Output shape: {[d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]}")
        print(f"  💾 Model size: {os.path.getsize(args.output_path) / 1024 / 1024:.2f} MB")
        
        if args.end_to_end:
            print(f"\n🎯 Atlas NPU Optimization Benefits:")
            print(f"  🚀 CPU load reduced: Preprocessing moved to NPU")
            print(f"  ⚡ Memory efficiency: No intermediate CPU-NPU transfers")
            print(f"  🎯 Pipeline simplification: Single inference call")
        
    except Exception as e:
        print(f"❌ Error during ONNX export: {str(e)}")
        traceback.print_exc()
        return False
    
    return True

def simplify_onnx(model_path):
    """简化ONNX模型"""
    if not onnx or not onnxsim:
        print("⚠️ ONNX or onnx-simplifier not installed. Skipping simplification.")
        return model_path
        
    try:
        print(f"\n✨ Simplifying ONNX model...")
        
        # 加载模型
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
            print(f"  Saved to: {simplified_path}")
            print(f"  Original size: {original_size:.2f} MB")
            print(f"  Simplified size: {simplified_size:.2f} MB")
            print(f"  Size reduction: {(1 - simplified_size/original_size)*100:.1f}%")
            return simplified_path
        else:
            print("❌ ONNX simplification check failed.")
            return model_path
            
    except Exception as e:
        print(f"❌ Error during simplification: {str(e)}")
        return model_path

def test_onnx_inference(onnx_path, input_shape, args):
    """测试端到端ONNX模型推理"""
    if not ort:
        print("⚠️ onnxruntime not installed. Skipping inference test.")
        return

    try:
        print(f"\n🧪 Testing End-to-End ONNX inference...")
        
        # 创建推理会话
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"✅ Inference session created for {onnx_path} on {session.get_providers()[0]}")

        # 创建随机输入图像
        height, width = input_shape
        if args.input_range == '0-255':
            dummy_input = np.random.randint(0, 256, (1, 3, height, width)).astype(np.float32)
            input_desc = "Raw image [0-255]"
        else:
            dummy_input = np.random.rand(1, 3, height, width).astype(np.float32)
            input_desc = "Normalized image [0-1]"
        
        print(f"   - Input shape: {dummy_input.shape}")
        print(f"   - Input type: {input_desc}")

        # 执行推理
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print("   - Running inference...")
        result = session.run([output_name], {input_name: dummy_input})[0]
        
        print(f"✅ Inference successful!")
        print(f"   - Output shape: {result.shape}")
        print(f"   - Output data type: {result.dtype}")

    except Exception as e:
        print(f"❌ Error during ONNX inference test: {str(e)}")
        traceback.print_exc()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载并准备模型
    model = load_model(args.model_path, device, args)

    # 导出ONNX
    success = export_onnx(model, args)

    # 简化ONNX
    if success and args.simplify:
        simplified_path = simplify_onnx(args.output_path)
        # 测试简化后的模型
        test_onnx_inference(simplified_path, (args.input_height, args.input_width), args)
    elif success:
        # 测试未简化的模型
        test_onnx_inference(args.output_path, (args.input_height, args.input_width), args)

if __name__ == '__main__':
    main()
